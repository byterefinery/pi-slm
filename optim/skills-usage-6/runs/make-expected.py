#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson"]
# ///

"""Build expected-responses/: the PERFECT student response for every tzip and
example skill message, as separate JSON files (user message + assistant
response chain only — no system prompt, no other messages).

- tzip cases (no tools): simulated with the TEACHER model
  (Qwen/Qwen3.8-27B, thinking ON — allowed only for this generation task),
  which writes the visible reply + a reasoning_content in the student
  model's voice per REASONING-LiquidAI-LFM2.5-2.6B.md.
- example cases: REAL pi coding-agent runs (isolated random temp dirs,
  --mode json, teacher model, thinking high) to observe the actual
  tool call/response chain; retried until the chain is perfect (no tool
  errors, correct final reply, required tool call present).

Never prints API keys or base URLs."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import orjson

HERE = Path(__file__).resolve().parent.parent
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
FROZEN = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"
REASONING_MD = HERE / "REASONING-LiquidAI-LFM2.5-2.6B.md"
OUT_DIR = HERE / "expected-responses"

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
PROVIDER = "llamacpp"
TZIP_SKILL_DIR = Path("/home/mtasic/projects-b/pi-slm/.agents/skills-byterefinery/tzip")
EXAMPLE_SKILL_DIR = Path("/home/mtasic/projects-b/pi-slm/.agents/skills-byterefinery/example")
PI_TIMEOUT = 240

frozen = json.loads(FROZEN.read_text())
EXAMPLE_BLOCK = frozen["messages"][7]["content"][0]["text"]  # canonical location
TZIP_LOCATION = str(TZIP_SKILL_DIR)
EXAMPLE_LOCATION = str(EXAMPLE_SKILL_DIR)

raw_cfg = MODELS_JSON.read_text()
CFG = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw_cfg))
PROV = CFG["providers"][PROVIDER]
BASE, KEY = PROV["baseUrl"], PROV["apiKey"]
TEACHER_SAMPLING = {}
for m in PROV.get("models", []):
    if m.get("id") == TEACHER_MODEL:
        TEACHER_SAMPLING = {k: v for k, v in m.get("samplingParams", {}).items()
                            if v is not None and k in ("temperature", "top_k", "min_p",
                                                       "presence_penalty", "repeat_penalty")}
        break


def new_sid() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")


def http_chat(model: str, messages: list, *, max_tokens: int, temperature: float,
              extra_body: dict, tools=None, retries: int = 3):
    """POST /chat/completions with a FRESH per-request session affinity."""
    aff_prefix = "dspy-optim-lfm2.5-2.6b" if model == STUDENT_MODEL else "dspy-optim-qwen3.8-27b"
    last = None
    for _ in range(retries):
        body = {"model": model, "messages": messages, "stream": False,
                "max_tokens": max_tokens, "temperature": temperature, **extra_body}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            BASE.rstrip("/") + "/chat/completions",
            data=orjson.dumps(body),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
                     "x-session-affinity": f"{aff_prefix}-{new_sid()}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = orjson.loads(r.read())
            served = d.get("model")
            if served is not None and served != model:
                raise RuntimeError(f"routing drift: served {served!r}")
            return d
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


THINKING_ON = {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}

# ---------------------------------------------------------------------------
# tzip cases — teacher-simulated perfect student responses
# ---------------------------------------------------------------------------

def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        return text.split("---", 2)[2].lstrip("\n")
    return text


TZIP_SKILL_MD_TEXT = (TZIP_SKILL_DIR / "SKILL.md").read_text()


def build_tzip_block_text(arg: str) -> str:
    body = strip_frontmatter(TZIP_SKILL_MD_TEXT)
    block = (f'<skill name="tzip" location="{TZIP_LOCATION}/SKILL.md">\n'
             f"References are relative to {TZIP_LOCATION}.\n\n{body}\n</skill>")
    if arg:
        block += f"\n\n{arg}"
    return block

TZIP_EXPECTED = {
    "": "tzip lite activated",
    "on": "tzip lite activated",
    "lite": "tzip lite activated",
    "full": "tzip full activated",
    "ultra": "tzip ultra activated",
    "off": "tzip deactivated",
    "OFF": "tzip deactivated",
}

GUIDE = REASONING_MD.read_text().split("## 5. Writing guide", 1)[1]
GUIDE = "## 5. Writing guide" + GUIDE  # keep the heading


def norm(s: str) -> str:
    return " ".join((s or "").strip().casefold().split())


def teacher_tzip_case(arg: str, max_attempts: int = 5) -> dict:
    expected = TZIP_EXPECTED[arg]
    user_msg = build_tzip_block_text(arg)
    prompt = f"""You are writing the PERFECT response of a small student coding agent
(model {STUDENT_MODEL}, thinking ON) in a pi coding-agent session. The student receives exactly one
user message, shown between the markers at the end. It contains a full skill block and, after the
block's closing tag, a bare argument (possibly absent). The student must follow the skill block's
instructions EXACTLY and reply with exactly what the skill requires — nothing added, nothing omitted.

Write the student's response as a STRICT JSON object with exactly two keys:
- "reasoning_content": the student's hidden thinking trace, in the student model's own natural voice.
  Writing guide (follow it exactly):
{GUIDE}
  For this simple skill invocation use the short form: 1-3 plain sentences, no lists, no headers,
  no LaTeX. Open by naming the situation in the first person ("The user ..."), state the decision
  with "I will ..." / "Let me ...", and END by committing to the exact reply the visible answer gives.
- "content": the student's visible reply — the exact line the skill's instructions require for this
  argument, verbatim. No code fences, no commentary, no extra text.

USER MESSAGE:
<<<USER_MESSAGE
{user_msg}
USER_MESSAGE>>>

Respond with ONLY the JSON object — no prose outside it."""
    for attempt in range(1, max_attempts + 1):
        d = http_chat(TEACHER_MODEL, [{"role": "user", "content": prompt}],
                      max_tokens=2048,
                      temperature=TEACHER_SAMPLING.get("temperature", 1.0),
                      extra_body={k: v for k, v in TEACHER_SAMPLING.items() if k != "temperature"} | THINKING_ON)
        content_raw = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", content_raw, re.DOTALL)
        if not m:
            print(f"  tzip {arg!r}: attempt {attempt}: no JSON in teacher reply")
            continue
        try:
            gen = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            print(f"  tzip {arg!r}: attempt {attempt}: bad JSON: {e}")
            continue
        content = (gen.get("content") or "").strip()
        reasoning = (gen.get("reasoning_content") or "").strip()
        problems = []
        if norm(content) != norm(expected):
            problems.append(f"content {content!r} != expected {expected!r}")
        if not (30 <= len(reasoning) <= 500):
            problems.append(f"reasoning length {len(reasoning)}")
        if reasoning and not re.search(r"\bI \w+|Let me", reasoning):
            problems.append("reasoning lacks first-person voice")
        if problems:
            print(f"  tzip {arg!r}: attempt {attempt}: {problems}")
            continue
        return {"content": content, "reasoning_content": reasoning, "attempts": attempt}
    raise RuntimeError(f"tzip {arg!r}: no perfect generation in {max_attempts} attempts")


# ---------------------------------------------------------------------------
# example cases — real pi runs (isolated temp dirs, JSON mode), retried
# ---------------------------------------------------------------------------

CASE_SENSITIVITY_NOTE = (
    "Skill-block arguments are matched CASE-SENSITIVELY against the skill's own Usage list: "
    "the example skill's 'Hello' case requires the capital-H spelling exactly; any other text "
    "(including case variants like 'hello') is 'any other text' and must go to scripts/example.sh."
)

EXAMPLE_CASES = [
    # (case_id, argument, kind, append-for-retry)
    # The three usages of the example skill in skill-example-LiquidAI-LFM2.5-2.6B.json
    # (messages [7]-[16]): no argument, "Hello", "Hi" — plus the training-suite
    # generalization variants ("Bye", "hello", "call script").
    ("example-empty", "", "no-tool", ""),
    ("example-hello", "Hello", "read-hello", ""),
    ("example-hi", "Hi", "script", ""),
    ("example-by", "Bye", "script", ""),
    ("example-hello-lower", "hello", "script", CASE_SENSITIVITY_NOTE),
    ("example-call-script", "call script", "script", ""),
]


def run_pi_example(arg: str, append: str = "") -> dict:
    prompt = "/skill:example" + (f" {arg}" if arg else "")
    tmp = Path(tempfile.mkdtemp(prefix="example-perfect-"))
    skill_dest = tmp / "skills" / "example"
    skill_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(EXAMPLE_SKILL_DIR, skill_dest)
    try:
        cmd = [
            "pi", "--offline", "--no-session", "--mode", "json",
            "--no-context-files", "--no-extensions", "--no-prompt-templates", "--no-skills",
            "--skill", str(skill_dest),
            "--provider", PROVIDER,
            "--model", TEACHER_MODEL,
            "--thinking", "high",
        ]
        if append:
            cmd += ["--append-system-prompt", append]
        cmd += ["-p", prompt]
        env = {**os.environ, "PI_SKIP_VERSION_CHECK": "1"}
        proc = subprocess.run(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=PI_TIMEOUT, text=True)
        events_path = tmp / "events.jsonl"
        events_path.write_text(proc.stdout)
        (tmp / "stderr.log").write_text(proc.stderr)
        if proc.returncode != 0:
            return {"ok": False, "error": f"pi rc={proc.returncode}: {proc.stderr[-200:]}", "tmp": tmp}
        events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
        end = [e for e in events if e.get("type") == "agent_end"]
        if not end:
            return {"ok": False, "error": "no agent_end event", "tmp": tmp}
        return {"ok": True, "tmp": tmp, "messages": end[0].get("messages", [])}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "tmp": tmp}


def tool_error(msg: dict) -> bool:
    if msg.get("isError"):
        return True
    c = msg.get("content")
    if isinstance(c, str):
        head = c.strip()[:80].casefold()
        return head.startswith("error") or "enoent" in head or "no such file" in head
    return False


def to_openai_shape(msgs: list) -> tuple[dict | None, list]:
    """Convert pi agent_end messages (native parts shape: text/thinking/toolCall
    parts, role 'toolResult') to (user_message, assistant_chain) in the
    frozen-transcript (OpenAI-like) shape."""
    user = None
    chain = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            if user is not None:
                break  # only the first user message is the skill request
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            user = {"role": "user", "content": [{"type": "text", "text": c}]}
        elif role == "assistant":
            out = {"role": "assistant", "content": None}
            c = m.get("content")
            if isinstance(c, str):
                out["content"] = c
            elif isinstance(c, list):
                text_bits, tool_calls, thinking = [], [], []
                for p in c:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "toolCall":
                        tool_calls.append({
                            "id": p.get("id") or f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {"name": p.get("name"), "arguments": p.get("arguments")},
                        })
                    elif p.get("type") == "text" and p.get("text"):
                        text_bits.append(p["text"])
                    elif p.get("type") == "thinking" and p.get("thinking"):
                        thinking.append(p["thinking"])
                if text_bits:
                    out["content"] = "".join(text_bits)
                if tool_calls:
                    out["tool_calls"] = tool_calls
                if thinking:
                    out["reasoning_content"] = "".join(thinking)
            if m.get("tool_calls") and not out.get("tool_calls"):
                out["tool_calls"] = m["tool_calls"]
            if m.get("reasoning_content") and not out.get("reasoning_content"):
                out["reasoning_content"] = m["reasoning_content"]
            chain.append(out)
        elif role in ("tool", "toolResult"):
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            out = {"role": "tool", "content": c}
            tcid = m.get("tool_call_id") or m.get("toolCallId")
            if tcid:
                out["tool_call_id"] = tcid
            if m.get("isError"):
                out["isError"] = True
            chain.append(out)
    return user, chain


def check_example(case_id: str, kind: str, chain: list) -> list:
    """Return a list of problems (empty == perfect chain)."""
    problems = []
    tool_msgs = [m for m in chain if m["role"] == "tool"]
    asst_msgs = [m for m in chain if m["role"] == "assistant"]
    for tm in tool_msgs:
        if tool_error(tm):
            problems.append(f"tool error: {str(tm.get('content'))[:80]!r}")
    final = None
    for m in reversed(asst_msgs):
        if m.get("content"):
            final = m["content"]
            break
    if final is None:
        problems.append("no final assistant text")
        return problems
    if kind == "no-tool":
        if asst_msgs and any(m.get("tool_calls") for m in asst_msgs):
            problems.append("unexpected tool calls")
        if norm(final) != norm("This is an example skill."):
            problems.append(f"final {final[:60]!r}")
    elif kind == "read-hello":
        read_calls = [tc for m in asst_msgs for tc in m.get("tool_calls", [])
                      if tc["function"]["name"] == "read"
                      and str(tc["function"].get("arguments", {}).get("path", "")).endswith("03-hello.md")]
        if len(read_calls) != 1:
            problems.append(f"expected 1 read of 03-hello.md, got {len(read_calls)}")
        if norm(final) != "world":
            problems.append(f"final {final[:60]!r} != 'world'")
    elif kind == "script":
        bash_calls = [tc for m in asst_msgs for tc in m.get("tool_calls", [])
                      if tc["function"]["name"] == "bash"
                      and "example.sh" in str(tc["function"].get("arguments", {}).get("command", ""))]
        if len(bash_calls) != 1:
            problems.append(f"expected 1 bash example.sh call, got {len(bash_calls)}")
        if "this is example.sh output" not in norm(final):
            problems.append(f"final {final[:60]!r}")
        if len(final) > 200:  # ideal = concise report of the output, no commentary
            problems.append(f"verbose final reply ({len(final)} chars)")
    return problems


def example_case(case_id: str, arg: str, kind: str, max_attempts: int = 8,
                 append: str = "", max_append_attempts: int = 0) -> dict:
    """Run the case up to max_attempts times (plain). If that fails and
    max_append_attempts > 0, retry with `append` added to the system prompt
    (used to disambiguate the skill's case-sensitivity, matching the project's
    ground truth); the append is recorded in generated_by."""
    for round_ in range(2):
        use_append = append if (round_ == 1 and max_append_attempts) else ""
        limit = max_append_attempts if round_ == 1 else max_attempts
        for attempt in range(1, limit + 1):
            res = run_pi_example(arg, append=use_append)
            if not res["ok"]:
                print(f"  {case_id}: attempt {attempt}: {res['error']}")
                if res.get("tmp"):
                    shutil.rmtree(res["tmp"], ignore_errors=True)
                continue
            user, chain = to_openai_shape(res["messages"])
            tmpdir = str(res["tmp"])
            shutil.rmtree(res["tmp"], ignore_errors=True)
            problems = check_example(case_id, kind, chain)
            if problems:
                print(f"  {case_id}: attempt {attempt}: {problems}")
                continue
            # rewrite the temp-dir location to the canonical one, in the user
            # block AND in tool-call arguments (paths/commands)
            user_text = user["content"][0]["text"]
            user_text = user_text.replace(f"{tmpdir}/skills/example", EXAMPLE_LOCATION)
            chain = json.loads(json.dumps(chain).replace(f"{tmpdir}/skills/example", EXAMPLE_LOCATION))
            extra = " + case-sensitivity note appended to system prompt" if use_append else ""
            return {
                "case": case_id, "skill": "example", "argument": arg,
                "student_model": STUDENT_MODEL,
                "generated_by": f"pi coding agent, isolated temp dir, --mode json, {TEACHER_MODEL} thinking high, "
                                f"{attempt} attempt{'s' if attempt != 1 else ''}{extra}",
                "user": {"role": "user", "content": [{"type": "text", "text": user_text}]},
                "assistant": chain,
            }
    raise RuntimeError(f"{case_id}: no perfect chain in {max_attempts}+{max_append_attempts} attempts")


# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for arg in ("", "on", "lite", "full", "ultra", "off", "OFF"):
        case_id = f"tzip-{arg or 'empty'}"
        print(f"== {case_id} (teacher simulation, thinking ON)")
        gen = teacher_tzip_case(arg)
        rec = {
            "case": case_id, "skill": "tzip", "argument": arg,
            "student_model": STUDENT_MODEL,
            "generated_by": f"{TEACHER_MODEL} (thinking ON) via HTTP, "
                            f"reasoning in student voice per REASONING-LiquidAI-LFM2.5-2.6B.md "
                            f"({gen['attempts']} attempt{'s' if gen['attempts'] != 1 else ''})",
            "user": {"role": "user", "content": [{"type": "text", "text": build_tzip_block_text(arg)}]},
            "assistant": [{"role": "assistant", "content": gen["content"],
                           "reasoning_content": gen["reasoning_content"]}],
        }
        out = OUT_DIR / f"{case_id}.json"
        out.write_text(json.dumps(rec, indent=2))
        summary.append((case_id, gen["content"]))
        print(f"  -> {out.name}: {gen['content']!r}")
    for case_id, arg, kind, append in EXAMPLE_CASES:
        print(f"== {case_id} (real pi, isolated temp dir)")
        rec = example_case(case_id, arg, kind, append=append, max_append_attempts=4)
        out = OUT_DIR / f"{case_id}.json"
        out.write_text(json.dumps(rec, indent=2))
        final = [m for m in rec["assistant"] if m["role"] == "assistant" and m.get("content")][-1]["content"]
        n_tools = sum(1 for m in rec["assistant"] if m.get("tool_calls"))
        summary.append((case_id, f"{final!r} ({n_tools} tool call(s))"))
        print(f"  -> {out.name}: {final!r}")
    print("\n=== expected-responses/ complete ===")
    for c, r in summary:
        print(f"  {c:22s} {r}")


if __name__ == "__main__":
    main()
