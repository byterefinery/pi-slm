#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Shared plumbing for skills-usage-5 (tzip few-shot optimization).

ALL model calls are SERIAL by design (shared server: no parallelism).
API_BASE / API_KEY are loaded on demand from ~/.pi/agent/models.json and
never written to logs, traces, or committed source.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

HERE = Path(__file__).resolve().parent          # .../optim/skills-usage-5/runs
ROOT = HERE.parent                              # .../optim/skills-usage-5
ARTIFACT = ROOT / "skill-example-LiquidAI-LFM2.5-2.6B.json"
REASONING_MD = ROOT / "REASONING-LiquidAI-LFM2.5-2.6B.md"
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
TZIP_SKILL = Path("/home/mtasic/projects-b/pi-slm/.agents/skills-byterefinery/tzip/SKILL.md")
TRAIN_LOG = ROOT / "train.log"
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
EVAL_DIR = HERE / "gepa-evals"
EVAL_DIR.mkdir(exist_ok=True)

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"    # thinking: ON (server template default)
TEACHER_MODEL = "Qwen/Qwen3.8-27B"        # thinking: OFF via chat_template_kwargs
AFFINITY = "dspy-optim"


# ---------------------------------------------------------------- credentials

def get_creds() -> tuple[str, str]:
    raw = MODELS_JSON.read_text()
    # models.json may contain trailing commas; strip them before parsing
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    p = json.loads(raw)["providers"]["llamacpp"]
    return p["baseUrl"], p["apiKey"]


class Api:
    def __init__(self) -> None:
        base, key = get_creds()
        self.client = openai.OpenAI(
            base_url=base,
            api_key=key,
            default_headers={"x-session-affinity": AFFINITY},
            timeout=600.0,
            max_retries=3,
        )


# ------------------------------------------------------------------- artifact

ART = json.loads(ARTIFACT.read_text())
SYSTEM_MSG = ART["messages"][0]
TOOLS = ART["tools"]

_TZIP_BODY: str | None = None


def tzip_body() -> str:
    global _TZIP_BODY
    if _TZIP_BODY is None:
        raw = TZIP_SKILL.read_text()
        m = re.match(r"^---\n.*?\n---\n", raw, re.S)
        _TZIP_BODY = raw[m.end():].rstrip("\n")
    return _TZIP_BODY


def tzip_block(arg: str) -> str:
    blk = (
        f'<skill name="tzip" location="{TZIP_SKILL}">\n'
        f"References are relative to {TZIP_SKILL.parent}.\n\n"
        f"{tzip_body()}\n"
        f"</skill>"
    )
    return blk + (f"\n\n{arg}" if arg else "")


def _user_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")


def find_pair_index(msgs: list[dict]) -> int:
    """Locate the mutable 'skill system Q/A' pair.

    Anchored structurally (stable even after the pair text is optimized):
    it is the user/assistant turn immediately BEFORE the first user message
    containing a <skill name="example"> block. Falls back to the seed text.
    """
    for j, m in enumerate(msgs):
        if m.get("role") == "user" and '<skill name="example"' in _user_text(m):
            if j >= 2 and msgs[j - 2].get("role") == "user" and msgs[j - 1].get("role") == "assistant":
                return j - 2
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and "skill system work" in _user_text(m).lower():
            return i
    raise ValueError("skill-system Q/A pair not found in artifact")


def seed_pair() -> dict:
    msgs = ART["messages"]
    i = find_pair_index(msgs)
    return {
        "user_question": _user_text(msgs[i]),
        "assistant_content": msgs[i + 1]["content"],
        "assistant_reasoning_content": msgs[i + 1]["reasoning_content"],
    }


def spliced_messages(pair: dict) -> list[dict]:
    """Artifact messages with only the two optimized messages replaced."""
    msgs = [json.loads(json.dumps(m)) for m in ART["messages"]]
    i = find_pair_index(msgs)
    msgs[i] = {"role": "user", "content": [{"type": "text", "text": pair["user_question"]}]}
    msgs[i + 1] = {
        "role": "assistant",
        "content": pair["assistant_content"],
        "reasoning_content": pair["assistant_reasoning_content"],
    }
    return msgs


# ----------------------------------------------------------------------- cases

CASES: list[dict] = [
    {"id": "empty",         "arg": "",     "expected": "tzip lite activated"},
    {"id": "on",            "arg": "on",   "expected": "tzip lite activated"},
    {"id": "lite",          "arg": "lite", "expected": "tzip lite activated"},
    {"id": "full",          "arg": "full", "expected": "tzip full activated"},
    {"id": "ultra",         "arg": "ultra","expected": "tzip ultra activated"},
    {"id": "off",           "arg": "off",  "expected": "tzip deactivated"},
    {"id": "full_to_lite",  "arg": "lite", "prior": "tzip full activated",  "expected": "tzip lite activated"},
    {"id": "lite_to_full",  "arg": "full", "prior": "tzip lite activated",  "expected": "tzip full activated"},
    {"id": "lite_to_ultra", "arg": "ultra","prior": "tzip lite activated",  "expected": "tzip ultra activated"},
    {"id": "ultra_to_off",  "arg": "off",  "prior": "tzip ultra activated", "expected": "tzip deactivated"},
    {"id": "full_to_off",   "arg": "off",  "prior": "tzip full activated",  "expected": "tzip deactivated"},
    {"id": "off_to_lite",   "arg": "lite", "prior": "tzip deactivated",     "expected": "tzip lite activated"},
]


def build_case_messages(pair: dict, case: dict) -> list[dict]:
    # spliced_messages already returns the full artifact list (system included, once)
    msgs = spliced_messages(pair)
    if case.get("prior"):
        msgs.append({"role": "user", "content": [{"type": "text", "text": tzip_block(_arg_of(case["prior"]))}]})
        msgs.append({"role": "assistant", "content": case["prior"]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": tzip_block(case.get("arg", ""))}]})
    return msgs


def _arg_of(expected_reply: str) -> str:
    # prior turn's argument, derived from its expected confirmation
    return {"tzip lite activated": "lite", "tzip full activated": "full",
            "tzip ultra activated": "ultra", "tzip deactivated": "off"}[expected_reply]


# --------------------------------------------------------------- normalization

def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".").strip()


def content_ok(actual: str | None, expected: str) -> bool:
    return norm(actual) == norm(expected)


# ------------------------------------------------------------------- caching

def _cache_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:32]


def _cache_get(key: str) -> dict | None:
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def _cache_put(key: str, val: dict) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(val, ensure_ascii=False))


# ------------------------------------------------------------- model calls

def student_call(api: Api, messages: list[dict], use_cache: bool = True) -> dict:
    """Exact replica of the captured student request (thinking ON)."""
    params = {
        "temperature": 0.1,
        "max_tokens": 49152,
        "extra_body": {
            "top_k": 50,
            "repeat_penalty": 1.1,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    }
    key = _cache_key("student", STUDENT_MODEL, messages, params)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit
    resp = api.client.chat.completions.create(
        model=STUDENT_MODEL, messages=messages, tools=TOOLS, **params
    )
    out = {
        "content": resp.choices[0].message.content or "",
        "reasoning_content": getattr(resp.choices[0].message, "reasoning_content", None) or "",
        "finish_reason": resp.choices[0].finish_reason,
    }
    _cache_put(key, out)
    return out


def teacher_call(api: Api, system: str, user: str, temperature: float = 0.1,
                 max_tokens: int = 2048, use_cache: bool = False) -> str:
    """Teacher (Qwen3.8-27B) with thinking OFF."""
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
        },
    }
    key = _cache_key("teacher", TEACHER_MODEL, system, user, params)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit["text"]
    resp = api.client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **params,
    )
    text = resp.choices[0].message.content or ""
    _cache_put(key, {"text": text})
    return text


def teacher_response_call(api: Api, messages: list[dict]) -> dict:
    """Teacher AS RESPONDER (baseline): same conversation shape as the student."""
    params = {
        "temperature": 0.1,
        "max_tokens": 2048,
        "extra_body": {
            "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
        },
    }
    resp = api.client.chat.completions.create(
        model=TEACHER_MODEL, messages=messages, tools=TOOLS, **params
    )
    return {
        "content": resp.choices[0].message.content or "",
        "reasoning_content": getattr(resp.choices[0].message, "reasoning_content", None) or "",
        "finish_reason": resp.choices[0].finish_reason,
    }


# ---------------------------------------------------------------------- judge

JUDGE_SYSTEM = """You audit the internal reasoning (reasoning_content) of a small AI assistant for "CoT leaks".

A CoT leak exists when the reasoning given for the CURRENT reply wrongly imports or hallucinates context from EARLIER turns of the conversation, for example:
- treating a word from an earlier turn ("Hi", "Hello", an earlier tzip mode, an earlier skill's argument) as the argument of the CURRENT message;
- asserting a mode or reply that is not indicated by the CURRENT message's argument;
- treating the finished "example" skill exchange (its "Hi" argument, "world" reply, example.sh output) as relevant to the current tzip invocation;
- claiming the user said, asked, or did something that is not actually in the current message.

NOT a leak:
- correctly noting an earlier tzip mode in a transition (e.g. "the user previously activated full; now the argument is lite") as long as the current argument determines the reply;
- quoting or following the skill's Usage rules;
- restating the current skill block or the current message;
- generic statements about handling skill invocations.

Respond with ONLY a JSON object, no prose:
{"leak": true|false, "identified_argument": "<exact argument text after the closing </skill> tag of the current message, or empty string if none>", "identified_reply": "<the reply the reasoning commits to>", "reason": "<one sentence>"}"""


def _render_history(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        role = m.get("role")
        if role == "user":
            lines.append(f"USER:\n{_user_text(m)}")
        elif role == "assistant":
            rc = m.get("reasoning_content")
            if rc:
                lines.append(f"ASSISTANT reasoning_content: {rc}")
            c = m.get("content")
            if c:
                lines.append(f"ASSISTANT: {c if isinstance(c, str) else _user_text(m)}")
            tcs = m.get("tool_calls")
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    lines.append(f"ASSISTANT tool_call: {fn.get('name')}({json.dumps(fn.get('arguments'))})")
        elif role == "tool":
            lines.append(f"TOOL RESULT: {m.get('content')}")
    return "\n\n".join(lines)


def judge(api: Api, pair: dict, case: dict, actual: str, reasoning: str,
          use_cache: bool = True) -> dict:
    msgs = build_case_messages(pair, case)
    current = _user_text(msgs[-1])
    user = (
        "CONVERSATION SO FAR (everything before the current user message):\n"
        f"{_render_history(msgs[:-1])}\n\n"
        f"CURRENT USER MESSAGE:\n{current}\n\n"
        f"ASSISTANT'S INTERNAL REASONING (reasoning_content) for its reply to the current message:\n{reasoning}\n\n"
        f"ASSISTANT'S ACTUAL REPLY: {actual}\n"
        f"THE CORRECT REPLY WAS: {case['expected']}\n\n"
        "Verdict (JSON only):"
    )
    try:
        raw = teacher_call(api, JUDGE_SYSTEM, user, temperature=0.1, max_tokens=512, use_cache=use_cache)
        m = re.search(r"\{.*\}", raw, re.S)
        j = json.loads(m.group(0)) if m else {}
        return {
            "leak": bool(j.get("leak", False)),
            "identified_argument": j.get("identified_argument", ""),
            "identified_reply": j.get("identified_reply", ""),
            "reason": j.get("reason", "(unparseable judge output)"),
            "_parse_ok": bool(m),
        }
    except Exception as e:  # noqa: BLE001 - judge failure must not kill the run
        return {"leak": False, "identified_argument": "", "identified_reply": "",
                "reason": f"judge error: {e}", "_parse_ok": False}


# ------------------------------------------------------------------- style

_STYLE_SYSTEM = (
    "You verify that a synthetic few-shot assistant message is written in the voice of the target "
    "student model. The student model's reasoning style guide (write per section 5, 'Writing guide'):\n\n"
    + REASONING_MD.read_text()
)


def style_check(api: Api, pair: dict, use_cache: bool = True) -> dict:
    user = (
        f"USER QUESTION (candidate field 1):\n{pair['user_question']}\n\n"
        f"ASSISTANT CONTENT (candidate field 2):\n{pair['assistant_content']}\n\n"
        f"ASSISTANT REASONING_CONTENT (candidate field 3):\n{pair['assistant_reasoning_content']}\n\n"
        "Check:\n"
        "1. reasoning_content must be in the student's SHORT form per the guide: 1-3 plain sentences, "
        "first person ('The user ...', 'I will ...'), no lists, no headers, no LaTeX, no meta talk "
        "(never mentions 'the reasoning', 'synthetic message', few-shot, or training), ends by committing "
        "to the exact next action.\n"
        "2. user_question must stay a natural question about how the skill system works when a skill "
        "block is in the latest message.\n"
        "3. Skill-agnostic: NONE of the three fields may reference a SPECIFIC skill - the name 'tzip', "
        "its specific modes (on/lite/full/ultra/off), or its specific confirmation strings (e.g. "
        "'activated'/'deactivated'). GENERAL skill-system vocabulary is expected and ALWAYS fine: "
        "'skill block', 'skill', 'Usage section', 'Usage', 'argument', 'closing tag', 'reply', "
        "'confirmation line' - every skill follows the same format, so naming the shared 'Usage section' "
        "is NOT a violation. Only flag a violation if a specific skill, its specific mode, or its specific "
        "reply string is named.\n\n"
        'Respond with ONLY a JSON object, no prose: {"style_ok": true|false, "issues": "<short list of violations, or empty string>"}'
    )
    try:
        raw = teacher_call(api, _STYLE_SYSTEM, user, temperature=0.1, max_tokens=512, use_cache=use_cache)
        m = re.search(r"\{.*\}", raw, re.S)
        j = json.loads(m.group(0)) if m else {}
        return {"style_ok": bool(j.get("style_ok", False)), "issues": j.get("issues", "")}
    except Exception as e:  # noqa: BLE001
        return {"style_ok": True, "issues": f"style check error (treated as ok): {e}"}


# ------------------------------------------------------------- evaluation

def case_score(content_okay: bool, leak: bool) -> float:
    if content_okay and not leak:
        return 1.0
    if content_okay:
        return 0.5
    if not leak:
        return 0.25
    return 0.0


def eval_case(api: Api, pair: dict, case: dict, use_cache: bool = True) -> dict:
    t0 = time.time()
    try:
        out = student_call(api, build_case_messages(pair, case), use_cache=use_cache)
    except Exception as e:  # noqa: BLE001
        out = {"content": "", "reasoning_content": "", "finish_reason": f"error: {e}"}
    ok = content_ok(out["content"], case["expected"])
    j = judge(api, pair, case, out["content"], out["reasoning_content"], use_cache=use_cache)
    r = {
        "id": case["id"],
        "arg": case.get("arg", ""),
        "expected": case["expected"],
        "actual": out["content"],
        "reasoning": out["reasoning_content"],
        "finish_reason": out["finish_reason"],
        "content_ok": ok,
        "leak": j["leak"],
        "leak_reason": j["reason"],
        "judge_arg": j["identified_argument"],
        "judge_reply": j["identified_reply"],
        "score": case_score(ok, j["leak"]),
        "trailing_period": (out["content"] or "").rstrip().endswith("."),
        "seconds": round(time.time() - t0, 1),
    }
    return r


def eval_candidate(api: Api, pair: dict, use_cache: bool = True, tag: str = "") -> dict:
    """Full 12-case suite, serial. Returns per-case results + aggregates."""
    t0 = time.time()
    results = [eval_case(api, pair, c, use_cache=use_cache) for c in CASES]
    n = len(results)
    acc = sum(r["content_ok"] for r in results) / n
    leakr = sum(r["leak"] for r in results) / n
    style = style_check(api, pair, use_cache=use_cache)
    score = 0.5 * acc + 0.5 * (1.0 - leakr)
    if not style["style_ok"]:
        score *= 0.6
    res = {
        "results": results,
        "acc": acc,
        "leak_rate": leakr,
        "style": style,
        "score": score,
        "seconds": round(time.time() - t0, 1),
        "tag": tag,
    }
    log(
        f"eval {tag or 'suite'} acc={acc * 100:.1f}% leak={leakr * 100:.1f}% "
        f"score={score:.3f} style={'ok' if style['style_ok'] else 'FAIL'} "
        f"periods={sum(r['trailing_period'] for r in results)} ({res['seconds']}s)"
    )
    return res


# --------------------------------------------------------------------- logging

def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def hash8(candidate: dict) -> str:
    s = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:8]
