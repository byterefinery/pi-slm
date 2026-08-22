#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "httpx", "orjson", "jinja2"]
# ///

'''
NOTE: this is how you instantiate dspy.LM - keep this string/comment
IMPORTANT: Radon values for API_BASE and API_KEY on demand from `~/.pi/agent/models.json`. Never leak them in text, logs or source code.

```python
lm = dspy.LM(
    f"openai/{model}", # used model
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=..., # read for used model
    max_tokens=16384,
    extra_headers={"x-session-affinity": f"dspy-optim-lfm2.5-2.6b-{session_id}"},  # per model + per REQUEST; see session_affinity() below
    extra_body={
        "top_k": ..., # read for used model, or omit if missing
        "min_p": ..., # read for used model, or omit if missing
        "presence_penalty": ..., # read for used model, or omit if missing
        "repeat_penalty": ..., # read for used model, or omit if missing
    },
)
```
'''

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"  # thinking: ON
TEACHER_MODEL = "Qwen/Qwen3.8-27B"  # thinking: OFF
PROVIDER = "llamacpp"
# Every request must carry the x-session-affinity header. A FRESH session id
# is minted per request so the proxy distributes requests evenly across the
# backend workers (an unchanging global value pins the whole run to one
# worker — that's how the 20260821 runs silently trained on the wrong model).
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # process start; logs only


def new_session_id() -> str:
    """Fresh per-request session id (UTC, microsecond precision => unique per request)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")


def session_affinity(model: str, sid: str | None = None) -> str:
    """Affinity value for one request: dspy-optim-<model-slug>-<fresh session id>."""
    sid = sid or new_session_id()
    if model == STUDENT_MODEL:
        return f"dspy-optim-lfm2.5-2.6b-{sid}"
    if model == TEACHER_MODEL:
        return f"dspy-optim-qwen3.8-27b-{sid}"
    return f"dspy-optim-{sid}"


def session_header(model: str, sid: str | None = None) -> dict:
    return {"x-session-affinity": session_affinity(model, sid)}
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
ROLLOUTS_DIR = RUNS_DIR / "rollouts"
LOG_FILE = HERE / "train.log"
OUTPUT_JSON = RUNS_DIR / "teacher-tzip-responses.json"

TZIP_SKILL_DIR = Path("/home/mtasic/projects-b/pi-slm/.agents/skills-byterefinery/tzip")
TZIP_SKILL_MD = TZIP_SKILL_DIR / "SKILL.md"
TZIP_REAL_LOCATION = str(TZIP_SKILL_DIR)

TRANSCRIPT_JSON = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"
PAIR_KEYS = ("user_content", "assistant_content", "reasoning_content")
MAX_TOOL_STEPS = 4
STUDENT_MAX_TOKENS = 65536
TEACHER_MAX_TOKENS = 16384

TEACHER_THINKING = "off"
MAX_ATTEMPTS = 4
PI_TIMEOUT = 240  # seconds per pi run

N_DRAWS = 10  # sampling draws per case; score = fraction of perfect draws (reliability)

EVAL_LOG = RUNS_DIR / "eval-log.jsonl"
GEPA_RUN_DIR = RUNS_DIR / "gepa-run"
BEST_PAIR_JSON = RUNS_DIR / "best-pair.json"
BASELINE_JSON = RUNS_DIR / "baseline.json"
FINAL_VERIFY_JSON = RUNS_DIR / "final-verification.json"
OPTIMIZED_TRANSCRIPT = RUNS_DIR / "skill-example-LiquidAI-LFM2.5-2.6B-optimized.json"
E2E_JSON = RUNS_DIR / "e2e-verification.json"

_eval_lock = threading.Lock()

# Student call parameters — frozen from the recorded request body.
STUDENT_PARAMS = {
    "temperature": 0.1,
    "max_tokens": STUDENT_MAX_TOKENS,
    "extra_body": {
        "top_k": 50,
        "repeat_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    },
}
# Teacher sampling — read from models.json at runtime (see load_models).

# All possible tzip invocations: bare argument following the skill block.
# "" means the user message is the skill block alone (empty user message).
TZIP_INPUTS = ["", "on", "lite", "full", "ultra", "off"]

# Canonical reply per the tzip skill (v1):
# "Communication pattern ... `<thing> [action] [reason]` ... Reply with mode name
# (e.g., "tzip lite activated", "tzip deactivated")."  Default mode is `lite`.
EXPECTED_REPLY = {
    "": "tzip lite activated",
    "on": "tzip lite activated",
    "lite": "tzip lite activated",
    "full": "tzip full activated",
    "ultra": "tzip ultra activated",
    "off": "tzip deactivated",
}

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_reply(text: str) -> str:
    return " ".join((text or "").strip().casefold().split()).rstrip(".")


def parse_events(path: Path) -> dict:
    """Parse pi --mode json event stream; return user message, messages, tool calls."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    end = [e for e in events if e.get("type") == "agent_end"]
    if not end:
        return {"ok": False, "error": "no agent_end event", "events": events}
    messages = end[0].get("messages", [])
    user_message = None
    tool_calls = []
    for m in messages:
        role = m.get("role")
        if role == "user" and user_message is None:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            user_message = c
        elif role == "assistant":
            parts = m.get("content", [])
            if isinstance(parts, str):
                parts = [{"type": "text", "text": parts}]
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "toolCall":
                    tool_calls.append({"name": p.get("name"), "arguments": p.get("arguments")})
    # final assistant text = text of last assistant message that has text
    final_text = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            parts = m.get("content", [])
            if isinstance(parts, str):
                parts = [{"type": "text", "text": parts}]
            txt = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
            if txt.strip():
                final_text = txt.strip()
                break
    return {
        "ok": True,
        "user_message": user_message,
        "messages": messages,
        "tool_calls": tool_calls,
        "final_text": final_text,
    }


def run_pi_once(arg: str) -> dict:
    """Run the teacher model in pi inside a fresh isolated temp dir for one tzip input."""
    prompt = "/skill:tzip" + (f" {arg}" if arg else "")
    tmp = Path(tempfile.mkdtemp(prefix="tzip-teacher-"))
    skill_dest = tmp / "skills" / "tzip"
    skill_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TZIP_SKILL_DIR, skill_dest)
    try:
        cmd = [
            "pi",
            "--offline",
            "--no-session",
            "--mode", "json",
            "--no-context-files",
            "--no-extensions",
            "--no-prompt-templates",
            "--no-skills",
            "--skill", str(skill_dest),
            "--provider", PROVIDER,
            "--model", TEACHER_MODEL,
            "--thinking", TEACHER_THINKING,
            "-p", prompt,
        ]
        env = {**os.environ, "PI_SKIP_VERSION_CHECK": "1"}
        log.debug("pi run: arg=%r cwd=%s", arg, tmp)
        proc = subprocess.run(
            cmd, cwd=tmp, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=PI_TIMEOUT, text=True,
        )
        events_path = tmp / "events.jsonl"
        events_path.write_text(proc.stdout)
        (tmp / "stderr.log").write_text(proc.stderr)
        if proc.returncode != 0:
            log.warning("pi exited rc=%s stderr_tail=%r", proc.returncode, proc.stderr[-300:])
            return {"ok": False, "error": f"pi rc={proc.returncode}", "tmp": tmp}
        parsed = parse_events(events_path)
        if not parsed["ok"]:
            return {"ok": False, "error": parsed.get("error", "parse failed"), "tmp": tmp}
        return {"ok": True, "tmp": tmp, **{k: parsed[k] for k in ("user_message", "messages", "tool_calls", "final_text")}}
    except subprocess.TimeoutExpired:
        log.warning("pi run timed out after %ss (arg=%r)", PI_TIMEOUT, arg)
        return {"ok": False, "error": "timeout", "tmp": tmp}


def rollout_case(arg: str) -> dict:
    """Run a tzip input with retries until the canonical reply is produced."""
    expected = EXPECTED_REPLY[arg]
    label = arg if arg else "empty"
    log.info("=== case %r: expecting %r ===", label, expected)
    attempts = []
    accepted = None
    case_dir = ROLLOUTS_DIR / label
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = run_pi_once(arg)
        record = {"attempt": attempt, "ok": res["ok"]}
        if not res["ok"]:
            record["error"] = res["error"]
            log.warning("case %r attempt %d: %s", label, attempt, res["error"])
            if "tmp" in res:
                shutil.rmtree(res["tmp"], ignore_errors=True)
            attempts.append(record)
            continue
        reply = res.get("final_text") or ""
        verified = normalize_reply(reply) == normalize_reply(expected)
        record.update({
            "reply": reply,
            "verified": verified,
            "tool_calls": res.get("tool_calls", []),
            "user_message": res.get("user_message"),
        })
        log.info(
            "case %r attempt %d: verified=%s tool_calls=%d reply=%r",
            label, attempt, verified, len(res.get("tool_calls", [])), reply[:120],
        )
        attempts.append(record)
        if verified:
            accepted = res
            break
    trace = None
    user_message = None
    reply = None
    if accepted is not None:
        tmp = accepted.pop("tmp")
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(case_dir / "accepted"))
        user_message = accepted["user_message"]
        reply = accepted["final_text"]
        trace = accepted["messages"]
    return {
        "mode": label,
        "arg": arg or None,
        "expected_reply": expected,
        "user_message": user_message,
        "assistant_reply": reply,
        "verified": accepted is not None,
        "attempts": attempts,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Student / teacher HTTP layer (creds read on demand, never logged)
# ---------------------------------------------------------------------------

def load_models() -> dict:
    raw = MODELS_JSON.read_text()
    data = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
    return data["providers"][PROVIDER]


def make_client(models: dict):
    """Plain httpx client. DO NOT use the openai SDK here: the proxy routes
    openai-SDK-shaped traffic (its body key order / client signature) to the
    Qwen workers — observed 320/320 Qwen vs ~75/75 LFM for identical logical
    requests sent with raw httpx/urllib (see runs/route-*.py probes)."""
    import httpx

    return httpx.Client(base_url=models["baseUrl"], timeout=300,
                        headers={"Authorization": f"Bearer {models['apiKey']}"})


# ---- duck-typed response (subset of the openai SDK object shape) ----------

class _Fn:
    __slots__ = ("name", "arguments")

    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _TC:
    __slots__ = ("id", "type", "function")

    def __init__(self, d):
        self.id = d.get("id")
        self.type = d.get("type", "function")
        self.function = _Fn(d["function"]["name"], d["function"]["arguments"])


class _Msg:
    __slots__ = ("content", "reasoning_content", "tool_calls")

    def __init__(self, d):
        self.content = d.get("content")
        self.reasoning_content = d.get("reasoning_content")
        self.tool_calls = [_TC(t) for t in (d.get("tool_calls") or [])]


class _Choice:
    __slots__ = ("message", "finish_reason")

    def __init__(self, d):
        self.message = _Msg(d.get("message") or {})
        self.finish_reason = d.get("finish_reason")


class _Resp:
    __slots__ = ("model", "choices")

    def __init__(self, d):
        self.model = d.get("model")
        self.choices = [_Choice(c) for c in (d.get("choices") or [])]


ROUTE_WAIT_MAX = 600  # max seconds to wait for the correct worker to come back


_route_salt = [0]


def next_salt() -> str:
    """Unique per-call salt sent as the OpenAI-standard `user` body field.

    The proxy distributes requests across backend workers by hashing the
    request; the `user` field changes the hash, so each retry is a fresh
    draw of the worker pool instead of re-hitting the same (wrong) worker."""
    _route_salt[0] += 1
    return f"dspy-optim-{RUN_ID}-{_route_salt[0]}"


def raw_chat(client, model, messages, *, max_tokens, temperature, extra_body, tools=None, affinity, user):
    """Single chat-completion POST with the MINIMAL working header set
    (content-type, authorization, x-session-affinity — no SDK signature)."""
    import orjson

    body = {"model": model, "messages": messages, "stream": False,
            "max_tokens": max_tokens, "temperature": temperature, **extra_body,
            "user": user}
    if tools:
        body["tools"] = tools
    r = client.post("/chat/completions", content=orjson.dumps(body),
                    headers={"content-type": "application/json", "x-session-affinity": affinity})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return _Resp(orjson.loads(r.content))


def call_lm(client, model: str, messages: list, *, max_tokens: int, temperature: float, extra_body: dict, tools: list | None = None):
    """Chat completion with model-identity enforcement.

    Every POST mints a FRESH per-request session id for x-session-affinity so
    the proxy spreads requests across the backend worker pool; some workers
    serve LFM, some serve Qwen, and the `model` field does NOT override the
    route. We NEVER accept a response from the wrong model: on drift we retry
    with a fresh affinity + fresh `user` salt (new draw) in 20s cycles until
    the requested model serves the call, up to ROUTE_WAIT_MAX, then abort."""
    affinity = session_affinity(model)
    user = next_salt()
    last_err = None
    for attempt in range(3):  # budget for transient server errors
        try:
            r = raw_chat(client, model, messages, max_tokens=max_tokens, temperature=temperature,
                         extra_body=extra_body, tools=tools, affinity=affinity, user=user)
            served = r.model
            waited = 0
            while served is not None and served != model and waited < ROUTE_WAIT_MAX:
                log.warning("routing drift: requested=%s served=%s — new affinity+salt, retry in 20s "
                            "(waited %ds of %ds)", model, served, waited, ROUTE_WAIT_MAX)
                time.sleep(20)
                waited += 20
                affinity = session_affinity(model)  # fresh per-request draw
                user = next_salt()
                r = raw_chat(client, model, messages, max_tokens=max_tokens, temperature=temperature,
                             extra_body=extra_body, tools=tools, affinity=affinity, user=user)
                served = r.model
            if served is not None and served != model:
                log.critical("routing drift persisted %ds for %s: served=%s (affinity=%s) — aborting",
                             waited, model, served, affinity)
                raise RuntimeError(f"model served {served!r} != requested {model!r} after {waited}s wait")
            return r
        except RuntimeError:
            raise
        except Exception as e:  # transient server errors; retry, then raise
            last_err = e
            log.warning("LM call failed (%s): %s", model, type(e).__name__)
            time.sleep(2)
    raise last_err


ROUTE_PROBE_MAX = 36  # session-id attempts before giving up (6 fast + 30 at 60s ≈ 30min)


def check_served(r, requested: str, context: str) -> str:
    """Fail hard if the proxy served a different model than requested —
    silently training/evaluating against the wrong model poisons the metric.
    (Affinities are per-request and ephemeral; the drift detail is logged by
    call_lm, which mints them.)"""
    served = getattr(r, "model", None)
    if served is not None and served != requested:
        log.critical("routing drift in %s: requested=%s served=%s — aborting run",
                     context, requested, served)
        raise RuntimeError(f"model served {served!r} != requested {requested!r} in {context}")
    return served or ""


def verify_routing(client) -> None:
    """Confirm the live proxy is actually serving each model. Affinities are
    FRESH per request (per-request session ids), so a successful probe proves
    the worker pool for that model is reachable right now, not a specific pin.
    Retry until the student probe is served by the LFM worker AND the teacher
    probe by Qwen; slow (60s-spaced) attempts also cover a downed LFM worker.
    Fail hard otherwise — never train against the wrong model."""
    frozen = json.loads(TRANSCRIPT_JSON.read_text())
    probe_msgs = frozen["messages"][:12]
    tools = frozen.get("tools") or []
    n_fast = 6
    for attempt in range(1, ROUTE_PROBE_MAX + 1):
        if attempt > n_fast + 1:
            log.info("routing probe: LFM worker not confirmed yet — waiting 60s before attempt %d/%d",
                     attempt, ROUTE_PROBE_MAX)
            time.sleep(60)
        s_aff = session_affinity(STUDENT_MODEL)  # fresh per-request session id
        t_aff = session_affinity(TEACHER_MODEL)
        try:
            s_r = raw_chat(client, STUDENT_MODEL, probe_msgs,
                          max_tokens=STUDENT_PARAMS["max_tokens"], temperature=STUDENT_PARAMS["temperature"],
                          extra_body=STUDENT_PARAMS["extra_body"], tools=tools or None,
                          affinity=s_aff, user=next_salt())
            s_served = s_r.model
        except Exception as e:
            log.warning("routing probe (student) attempt %d failed: %s: %s", attempt, type(e).__name__, str(e)[:120])
            continue
        try:
            t_r = raw_chat(client, TEACHER_MODEL,
                          [{"role": "user", "content": "Reply with the single word: pong"}],
                          max_tokens=64, temperature=0.0,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                          affinity=t_aff, user=next_salt())
            t_served = t_r.model
        except Exception as e:
            log.warning("routing probe (teacher) attempt %d failed: %s: %s", attempt, type(e).__name__, str(e)[:120])
            continue
        log.info("routing probe attempt %d (student sid=%s | teacher sid=%s): student served=%s | teacher served=%s",
                 attempt, s_aff.rsplit("-", 1)[-1], t_aff.rsplit("-", 1)[-1], s_served, t_served)
        if s_served == STUDENT_MODEL and t_served == TEACHER_MODEL:
            log.info("routing VERIFIED (per-request affinities): student worker and teacher worker both reachable")
            return
    raise RuntimeError(
        f"could not confirm routing after {ROUTE_PROBE_MAX} attempts — the student worker is not reachable under "
        f"fresh per-request affinities; aborting rather than training against the wrong model")


def teacher_lm(client, models: dict):
    """Build the reflection-LM callable (LanguageModel protocol: str -> str)."""
    sp = models.get("teacher_sampling", {})

    def _lm(prompt):
        r = call_lm(
            client,
            TEACHER_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=TEACHER_MAX_TOKENS,
            temperature=sp.get("temperature", 1.0),
            extra_body={
                "top_k": sp.get("top_k"),
                "min_p": sp.get("min_p"),
                "presence_penalty": sp.get("presence_penalty"),
                "repeat_penalty": sp.get("repeat_penalty"),
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        check_served(r, TEACHER_MODEL, "teacher reflection")
        return r.choices[0].message.content or ""

    return _lm


# ---------------------------------------------------------------------------
# Frozen transcript + test cases
# ---------------------------------------------------------------------------

def strip_frontmatter(text: str) -> str:
    return text.split("---", 2)[2].strip()


def build_tzip_block_text(arg: str) -> str:
    body = strip_frontmatter(TZIP_SKILL_MD.read_text())
    block = (
        f'<skill name="tzip" location="{TZIP_REAL_LOCATION}/SKILL.md">\n'
        f"References are relative to {TZIP_REAL_LOCATION}.\n\n{body}\n</skill>"
    )
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
}


def make_frozen(frozen: dict):
    """Pre-extracted frozen pieces shared by all cases."""
    prefix = [copy.deepcopy(m) for m in frozen["messages"][:5]]
    suffix = [copy.deepcopy(m) for m in frozen["messages"][7:]]
    example_block = frozen["messages"][7]["content"][0]["text"]
    tools = frozen["tools"]
    hello_ref = frozen["messages"][11]["content"]
    tzip_skill_text = TZIP_SKILL_MD.read_text()
    cases = []
    for arg in ("", "on", "lite", "full", "ultra", "off"):
        cases.append({
            "id": f"tzip-{arg or 'empty'}",
            "skill": "tzip",
            "arg": arg,
            "check": "tzip",
            "expected": TZIP_EXPECTED[arg],
            "user_message": [{"type": "text", "text": build_tzip_block_text(arg)}],
        })
    # example skill: rules from its Usage (no-arg fixed reply, "Hello" ->
    # references/03-hello.md -> "world", any other text -> scripts/example.sh).
    cases.append({
        "id": "example-hello",
        "skill": "example",
        "arg": "Hello",
        "check": "example-hello",
        "expected": "world",
        "user_message": [{"type": "text", "text": example_block + "\n\nHello"}],
    })
    # unknown argument (never seen in the frozen history) — generalization test
    cases.append({
        "id": "example-by",
        "skill": "example",
        "arg": "Bye",
        "check": "example-script",
        "expected": "This is example.sh output.",
        "user_message": [{"type": "text", "text": example_block + "\n\nBye"}],
    })
    cases.append({
        "id": "example-empty",
        "skill": "example",
        "arg": "",
        "check": "example-empty",
        "expected": "This is an example skill.",
        "user_message": [{"type": "text", "text": example_block}],
    })
    # robustness variants (ground truth straight from the skills' own Usage text):
    # uppercase mode argument — tzip 'off' case, typed uppercase
    cases.append({
        "id": "tzip-OFF",
        "skill": "tzip",
        "arg": "OFF",
        "check": "tzip",
        "expected": "tzip deactivated",
        "user_message": [{"type": "text", "text": build_tzip_block_text("OFF")}],
    })
    # lowercase 'hello' is NOT 'Hello' per the skill's exact wording -> any other text -> script
    cases.append({
        "id": "example-hello-lower",
        "skill": "example",
        "arg": "hello",
        "check": "example-script",
        "expected": "This is example.sh output.",
        "user_message": [{"type": "text", "text": example_block + "\n\nhello"}],
    })
    # skill explicitly lists: asked to "call script" -> run the script
    cases.append({
        "id": "example-call-script",
        "skill": "example",
        "arg": "call script",
        "check": "example-script",
        "expected": "This is example.sh output.",
        "user_message": [{"type": "text", "text": example_block + "\n\ncall script"}],
    })
    canned = {"hello_ref": hello_ref, "script_out": "This is example.sh output.\n", "tzip_skill": tzip_skill_text}
    # Holdout: a skill never present in the conversation — final verification only,
    # NEVER in the GEPA dataset (tests generalization to unknown skills).
    echo_body = (
        "# echo\n\n"
        "Minimal echo skill.\n\n## Usage\n\n"
        "- Invoked with any text — reply exactly `echoed: <text>`, where `<text>` is the "
        "argument. Nothing else.\n"
        "- Invoked with no extra text — reply exactly `echoed:`."
    )
    echo_block = (
        '<skill name="echo" location="/tmp/skills/echo/SKILL.md">\n'
        "References are relative to /tmp/skills/echo.\n\n" + echo_body + "\n</skill>"
    )
    holdout = [{
        "id": "echo-holdout",
        "skill": "echo",
        "arg": "ping",
        "check": "echo",
        "expected": "echoed: ping",
        "user_message": [{"type": "text", "text": echo_block + "\n\nping"}],
    }]
    return {"prefix": prefix, "suffix": suffix, "tools": tools, "cases": cases,
            "canned": canned, "holdout": holdout}


def canned_tool_result(name: str, args: dict, canned: dict):
    """Deterministic simulated tool results (no external reads, no execution)."""
    if name == "read":
        p = str(args.get("path", ""))
        if p.endswith("03-hello.md"):
            return canned["hello_ref"], False
        if "tzip" in p and "SKILL.md" in p:
            return canned["tzip_skill"], False
        return f"(simulated) file not available: {p}", True
    if name == "bash":
        c = str(args.get("command", ""))
        if "example.sh" in c:
            return canned["script_out"], False
        return "(simulated) bash command not available in this evaluation", True
    return f"(simulated) tool {name} not available in this evaluation", True


def run_student_case(client, pair: dict, case: dict, fz: dict) -> dict:
    """Run the student on frozen context + pair + appended skill usage."""
    messages = list(fz["prefix"])
    tools = fz["tools"]
    messages.append({"role": "user", "content": [{"type": "text", "text": pair["user_content"]}]})
    messages.append({
        "role": "assistant",
        "content": pair["assistant_content"],
        "reasoning_content": pair["reasoning_content"],
    })
    messages.extend(fz["suffix"])
    messages.append({"role": "user", "content": case["user_message"]})
    steps = []
    r = None
    for _ in range(MAX_TOOL_STEPS + 1):
        r = call_lm(client, STUDENT_MODEL, messages,
                    max_tokens=STUDENT_PARAMS["max_tokens"],
                    temperature=STUDENT_PARAMS["temperature"],
                    extra_body=STUDENT_PARAMS["extra_body"],
                    tools=tools)
        served = check_served(r, STUDENT_MODEL, "run_student_case")
        choice = r.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
                "reasoning_content": getattr(msg, "reasoning_content", None) or "",
            })
            for tc in tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                out, is_err = canned_tool_result(tc.function.name, args, fz["canned"])
                messages.append({"role": "tool", "content": out, "tool_call_id": tc.id})
                steps.append({"tool": tc.function.name, "args": args,
                              "result": out[:200], "error": is_err})
            continue
        return {
            "content": msg.content or "",
            "reasoning": getattr(msg, "reasoning_content", None) or "",
            "tool_steps": steps,
            "finish": choice.finish_reason,
            "model_served": served,
        }
    return {"content": "", "reasoning": getattr(r.choices[0].message, "reasoning_content", None) or "",
            "tool_steps": steps, "finish": "max_tool_steps",
            "model_served": served}


# ---------------------------------------------------------------------------
# Scoring + GEPA evaluator
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    return " ".join((s or "").strip().casefold().split()).rstrip(".")


def score_case(case: dict, res: dict) -> float:
    c, e = norm(res["content"]), norm(case["expected"])
    kind = case["check"]
    if kind == "tzip":
        if c == e:
            return 1.0
        if e and e in c:
            return 0.6
        word = e.rsplit(" ", 1)[-1]
        if word in ("activated", "deactivated"):
            word = e.rsplit(" ", 2)[-2]
        if word and word in c:
            return 0.3
        return 0.0
    if kind == "example-hello":
        if c == "world":
            return 1.0
        if "world" in c.split():
            return 0.6
        return 0.0
    if kind == "example-script":
        if "this is example.sh output" in c:
            return 1.0
        if "example.sh" in c:
            return 0.3
        return 0.0
    if kind == "example-empty":
        if c == e:
            return 1.0
        if "example skill" in c:
            return 0.6
        return 0.0
    if kind == "echo":
        if c == e:
            return 1.0
        if e in c:
            return 0.6
        if "echoed" in c:
            return 0.3
        return 0.0
    return 0.0


def eval_case(client, candidate: dict, case: dict, fz: dict, phase: str, n_draws: int = N_DRAWS) -> tuple[float, dict]:
    """Run a case over n_draws sampling draws; score = fraction of perfect draws.

    Returns (mean_score, side_info) — side_info carries per-draw replies so the
    reflection LM can see the variance, anchored on the worst draw.
    """
    t0 = time.time()
    draws = []
    try:
        for i in range(n_draws):
            res = run_student_case(client, candidate, case, fz)
            s = score_case(case, res)
            draws.append((s, res))
            log_eval(phase, f"{case['id']}#d{i + 1}", cand_hash(candidate), s, res, time.time() - t0, model=STUDENT_MODEL)
    except Exception as e:
        return 0.0, {"Case": case["id"], "Error": f"{type(e).__name__}: {e}"}
    mean = sum(s for s, _ in draws) / len(draws)
    worst = min(draws, key=lambda d: d[0])
    wscore, wres = worst
    side_info = {
        "Case": f"{case['id']} (skill={case['skill']}, argument={case['arg'] or '(none)'})",
        "Expected reply (exact)": case["expected"],
        "Score (mean of %d draws)" % n_draws: f"{mean:.2f}",
        "Draws": " | ".join(
            f"d{i + 1}: score {s:.2f} reply {r['content'][:160]!r}"
            for i, (s, r) in enumerate(draws)
        ),
        "Worst draw reasoning trace": wres["reasoning"] or "(none)",
        "Worst draw tool steps": json.dumps(wres["tool_steps"]) if wres["tool_steps"] else "(none)",
        "Constraint": (
            "The optimized pair must stay a GENERAL skill-system lesson: it must not mention "
            "tzip, example, or any specific skill, and must not teach skill-specific steps. It must "
            "work for ANY skill block. reasoning_content must be 1-3 short first-person sentences in "
            "the student model's voice (style: 'The user ...', 'I will ...'), ending with the exact "
            "action the visible reply performs. The lesson must cover: a skill block is a fresh, "
            "independent invocation; the argument is the bare text immediately after the block's "
            "closing tag; when there is no argument, use the default the skill itself specifies; "
            "follow the skill's instructions exactly (including any tool steps the skill defines); "
            "reply with exactly what the skill asks for, verbatim, nothing added."
        ),
    }
    log.info("[eval %s] %s cand=%s model=%s mean=%.2f draws=[%s]",
             phase, case["id"], cand_hash(candidate), STUDENT_MODEL, mean,
             ",".join(f"{s:.0f}" for s, _ in draws))
    return mean, side_info


def log_eval(phase: str, case_id: str, cand_hash: str, score: float, res: dict, latency: float, model: str = STUDENT_MODEL):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase, "case": case_id, "cand": cand_hash, "model": model,
        "score": score, "reply": res["content"],
        "reasoning_chars": len(res["reasoning"]),
        "tool_steps": res["tool_steps"], "finish": res["finish"],
        "model_served": res.get("model_served"),
        "latency_s": round(latency, 2),
    }
    with _eval_lock:
        with open(EVAL_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


def cand_hash(candidate: dict) -> str:
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()[:16]


def make_evaluator(client, fz: dict, phase: str, n_draws: int = N_DRAWS):
    def evaluate(candidate, example=None):
        return eval_case(client, candidate, example, fz, phase, n_draws)

    return evaluate


def cmd_rollout() -> None:
    ROLLOUTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("### teacher rollout start: %s (thinking=%s) skill=%s",
             TEACHER_MODEL, TEACHER_THINKING, TZIP_SKILL_DIR)
    cases = []
    for arg in TZIP_INPUTS:
        cases.append(rollout_case(arg))
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher_model": TEACHER_MODEL,
        "provider": PROVIDER,
        "teacher_thinking": TEACHER_THINKING,
        "skill": {"name": "tzip", "source": str(TZIP_SKILL_DIR)},
        "note": (
            "Gold teacher responses for tzip skill usage. user_message is the exact "
            "skill-block + argument message pi generated (location path points at a "
            "random temp dir); rewrite location when embedding in training transcripts."
        ),
        "cases": cases,
    }
    OUTPUT_JSON.write_text(json.dumps(out, indent=2))
    n_ok = sum(1 for c in cases if c["verified"])
    log.info("### teacher rollout done: %d/%d verified -> %s", n_ok, len(cases), OUTPUT_JSON)


# ---------------------------------------------------------------------------
# GEPA training of the message pair
# ---------------------------------------------------------------------------

TZIP_BODY = strip_frontmatter(TZIP_SKILL_MD.read_text())

OBJECTIVE = (
    "Make the student model (LiquidAI/LFM2.5-2.6B, thinking ON, a pi coding agent) handle skill "
    "invocations perfectly. "
    "The only modifiable part is a 3-component synthetic message pair (user_content, assistant_content, "
    "reasoning_content) inserted at a fixed position in a frozen conversation. After the frozen conversation "
    "ends, a final user message containing a full skill block plus a bare argument is appended, and the "
    "student must respond exactly as that skill's instructions require. Primary goal (7 tzip cases, each "
    "must reach 1.0): the tzip skill demands a single-line reply in the pattern '<thing> [action] [reason]': "
    "no argument / 'on' / 'lite' -> 'tzip lite activated'; 'full' -> 'tzip full activated'; "
    "'ultra' -> 'tzip ultra activated'; 'off' and uppercase 'OFF' -> 'tzip deactivated'. Secondary goal "
    "(5 example-skill cases, must not regress): 'Hello' -> read references/03-hello.md, then reply exactly "
    "'world'; unknown text 'Bye' -> run scripts/example.sh with that text and report the output line "
    "'This is example.sh output.'; no argument -> reply exactly 'This is an example skill.'; lowercase "
    "'hello' is NOT 'Hello' so per the skill's Usage it is any other text -> run scripts/example.sh and "
    "report 'This is example.sh output.'; 'call script' -> run scripts/example.sh and report the output. "
    "Each case is sampled %d times (temperature 0.1); its score is the fraction of perfect draws, so the "
    "pair must make the behavior RELIABLE, not just occasionally correct. The pair must be fully general: it "
    "teaches how to handle ANY skill block and must not name or special-case any specific skill." % N_DRAWS
)

BACKGROUND = f"""Frozen conversation structure (pi coding-agent session, student model LiquidAI/LFM2.5-2.6B, thinking on):
- system prompt: pi coding-agent instructions; available skills listed (tzip among them); tools: read, write, edit, bash.
- User: "What are available skills?" -> assistant lists skills.
- User: "What are available tools?" -> assistant lists tools.
- [PAIR UNDER OPTIMIZATION: user_content + assistant_content + reasoning_content]
- User: skill block of the 'example' skill + no argument -> assistant replies exactly "This is an example skill."
- User: example skill block + "Hello" -> assistant reads references/03-hello.md (which says: reply exactly 'world', nothing else) -> replies "world".
- User: example skill block + "Hi" -> assistant runs bash scripts/example.sh Hi -> script prints 'This is example.sh output.' -> assistant reports it in a code block.
- [APPENDED TEST: a full skill block + bare argument, per the cases in the evaluation data.]

The full tzip SKILL.md content (this is what arrives in the appended tzip test messages):
```
{TZIP_BODY}
```

Student model behavior (observed): the visible answer follows the last committed action in reasoning_content. "
"Its reasoning voice is short first-person narration ('The user ...', 'I will ...', 'Let me ...'), 1-3 sentences "
"for simple actions, ending with the exact next action. It can drift: if its reasoning commits to describing or "
"explaining instead of replying, the answer drifts too.

Tool simulation in the evaluation: read of '.../example/references/03-hello.md' returns the file that says to "
"reply exactly 'world'; bash commands containing 'example.sh' print 'This is example.sh output.'; other tool "
"calls return simulated errors. The tzip reply must be plain text (no tools needed)."""

_OBJECTIVE_HEAD = (
    "Overall objective: " + OBJECTIVE
)

_CONTEXT_TAIL = (
    "\n\nAnalyze the evaluation data below. The key diagnostic is the student's reasoning trace: the visible "
    "answer follows the last action the reasoning commits to. Identify why the student deviates from the "
    "expected reply on each failing case, and change the component you are editing so the student is pushed "
    "toward the exact required behavior for ALL cases (both skills). Respect every hard constraint."
    "\n\nRespond with ONLY one fenced code block containing the complete new value — no other code blocks, "
    "no prose outside the block."
)

def _build_templates(background: str) -> dict:
    return {
        "user_content": (
            "You are optimizing the USER message of a synthetic 2-message teaching pair in the middle of a "
            "frozen conversation with a small student LLM.\n" + _OBJECTIVE_HEAD + "\n"
            "This component is the user's question that sets up the teaching turn about how the skill system "
            "works and what to do when a skill block appears in the user's message.\n"
            "Hard constraints: natural user question, 1-3 sentences, no markdown; must not name tzip or any "
            "specific skill; must stay about the skill system in general (skill blocks, their argument text, "
            "following the skill's instructions exactly, replying exactly as required, default when no argument).\n"
            + background
            + "\n\nCurrent value of user_content:\n```\n<curr_param>\n```\n\nEvaluation data:\n```\n<side_info>\n```"
            + _CONTEXT_TAIL
        ),
        "assistant_content": (
            "You are optimizing the ASSISTANT's visible reply of a synthetic 2-message teaching pair in the "
            "middle of a frozen conversation with a small student LLM.\n" + _OBJECTIVE_HEAD + "\n"
            "This component is the assistant's short visible answer to the user's question about the skill "
            "system — the lesson that makes the student handle any later skill block correctly.\n"
            "Hard constraints: 1-3 sentences, plain text, coding-assistant voice; must not name tzip or any "
            "specific skill, no skill-specific steps; must convey (in general terms): a skill block is a fresh, "
            "independent invocation; the argument is the bare text immediately after the block's closing tag; "
            "with no argument, use the default the skill itself specifies; follow the skill's instructions "
            "exactly, including any tool steps the skill defines; reply with exactly what the skill asks for, "
            "verbatim, nothing added.\n"
            + background
            + "\n\nCurrent value of assistant_content:\n```\n<curr_param>\n```\n\nEvaluation data:\n```\n<side_info>\n```"
            + _CONTEXT_TAIL
        ),
        "reasoning_content": (
            "You are optimizing the ASSISTANT's hidden reasoning trace (reasoning_content) of a synthetic "
            "2-message teaching pair in the middle of a frozen conversation with a small student LLM.\n" + _OBJECTIVE_HEAD + "\n"
            "This component is written in the student model's own voice and shows HOW it should think about the "
            "skill-system question; the student's later visible answer follows the last action this trace "
            "commits to, so it must end by committing to the exact general procedure.\n"
            "Hard constraints: 1-3 short plain sentences; first-person student voice ('The user ...', 'I will ...', "
            "'Let me ...'); no markdown, no lists, no meta talk about 'reasoning' or 'synthetic'; must not name "
            "tzip or any specific skill; must end with the commitment to follow the skill's instructions exactly "
            "and reply with exactly the response the skill requires.\n"
            + background
            + "\n\nCurrent value of reasoning_content:\n```\n<curr_param>\n```\n\nEvaluation data:\n```\n<side_info>\n```"
            + _CONTEXT_TAIL
        ),
    }


TEMPLATES = _build_templates(BACKGROUND)

# ---------------------------------------------------------------------------
# Round 2: warm-start from the round-1 best pair, bigger budget, and a
# BACKGROUND extension that hands the reflection LM the exact failure modes
# of the round-1 pair (tzip-full ignored, tzip-off cross-skill contamination,
# example-empty default). This text is seen ONLY by the teacher reflection
# model — the student never sees the objective, so naming cases here is fine.
# ---------------------------------------------------------------------------

GEPA_RUN_DIR2 = RUNS_DIR / "gepa-run-2"
BEST_PAIR_JSON2 = RUNS_DIR / "best-pair-r2.json"
FINAL_VERIFY_JSON2 = RUNS_DIR / "final-verification-r2.json"
OPTIMIZED_TRANSCRIPT2 = RUNS_DIR / "skill-example-LiquidAI-LFM2.5-2.6B-optimized-r2.json"
MAX_METRIC_CALLS2 = 300

BACKGROUND_R2 = BACKGROUND + """

KNOWN FAILURE MODES of the current best pair (observed in the previous round; the new pair must fix these):
1. ARGUMENT IGNORED -> DEFAULT FALLBACK: on the tzip 'full' case the student replied with the skill's
   DEFAULT-mode line instead of the 'full' line (score 0.00/10 draws). It reuses the most familiar
   previous reply pattern instead of mapping the argument to the mode listed in the skill's own Usage.
2. CROSS-SKILL CONTAMINATION: on the tzip 'off' case the student replied with a line belonging to the
   EARLIER example-skill exchange ("This is an example skill.") instead of following the current
   tzip block's deactivation instruction (score 0.18). And the example-skill NO-ARGUMENT case scored
   0.00 — replies or habits from one skill's block are leaking into a different skill's block.
3. The two 'OFF' uppercase and 'off' lowercase deactivations both failed (0.21 / 0.18) even though the
   skill's Usage lists both spellings — the student must treat every argument spelling the skill's own
   Usage lists as valid, and must not second-guess or normalize arguments on its own.
Required properties of the fixed pair: the student must (a) re-read the CURRENT skill block's
instructions on EVERY invocation and map the bare argument to that skill's own accepted-argument list
or default — never carry over a reply pattern from an earlier, different skill; (b) use the skill's
own default when there is no argument. The pair must remain a GENERAL skill-system lesson: it must not
name tzip, example, or any specific skill, and must not teach skill-specific steps."""

TEMPLATES_R2 = _build_templates(BACKGROUND_R2)


# ---------------------------------------------------------------------------
# Round 3: warm-start from the round-2 best pair with a full two-round
# post-mortem. Key new insight: the round-2 USER question ("restate the
# rule...") primed the student to RESTATE RULES on every later skill block
# (example-hello 0.70 -> 0.12). The user question must ask what REPLY to
# give, never to restate/explain a rule. Seen ONLY by the teacher.
# ---------------------------------------------------------------------------

GEPA_RUN_DIR3 = RUNS_DIR / "gepa-run-3"
BEST_PAIR_JSON3 = RUNS_DIR / "best-pair-r3.json"
FINAL_VERIFY_JSON3 = RUNS_DIR / "final-verification-r3.json"
OPTIMIZED_TRANSCRIPT3 = RUNS_DIR / "skill-example-LiquidAI-LFM2.5-2.6B-optimized-r3.json"
MAX_METRIC_CALLS3 = 250

BACKGROUND_R3 = BACKGROUND + """

TWO PREVIOUS ROUNDS WERE RUN. Post-mortem (what failed, and why):

1. TZIP 'FULL' (0/20 across both rounds): the student's reasoning trace shows it CAN read the
   Usage list — it quotes '`tzip full` -> Drop articles...' — and STILL replies with the
   default-mode line. The default entry is too salient. The lesson must make the student find
   the Usage entry whose text contains the bare argument and reply exactly as THAT entry
   requires, and must state that the skill's default applies ONLY when no argument text
   follows the closing tag.
2. TZIP 'OFF' (0.34 best): the student's trace shows it parses the message as "invoked with
   no extra text (just the skill name)" — it MISSES the one-word argument that follows the
   closing tag, and it also reaches for the EARLIER example-skill's no-argument rule ("This is
   an example skill."). The lesson must make the bare text after the closing tag salient as
   THE argument (a single word counts), and must state that one skill's rules never apply to
   another skill's block.
3. USER-QUESTION PRIMING (the round-2 pair's biggest bug): the round-2 USER message asked the
   student to "restate the rule...". The student then treated EVERY LATER SKILL BLOCK as a
   rule-restating question: on the example-skill 'Hello' case it replied with explanations
   ("The example skill's Usage section defines these rules...") instead of following the
   skill's instruction to read its reference file and reply 'world' (0.70 -> 0.12). The user
   message must ask what REPLY to give / what to DO (round-1 style: "When I send you a message
   with a skill block and some text, how do I know exactly what reply to give?"), and must
   NEVER ask to restate, describe, or explain a rule — the student copies the question's
   intent onto later turns. The assistant's answer must commit to EXECUTING the skill's
   procedure (follow its steps, run its tool steps, reply exactly), not to explaining it.
4. TZIP 'OFF'/'OFF' deactivation (0.21-0.34): the student invents a mode line ('tzip off
   activated') instead of the skill's own deactivation line. General lesson: the argument
   selects the skill's OWN entry for that text — including entries that are not mode lines
   (e.g. a deactivation or a default)."""

TEMPLATES_R3 = _build_templates(BACKGROUND_R3)


def run_with_messages(client, messages: list, tools: list, model: str, params: dict, canned: dict) -> dict:
    """Tool loop over a ready-made message list with explicit request params."""
    steps = []
    msgs = [copy.deepcopy(m) for m in messages]
    r = None
    for _ in range(MAX_TOOL_STEPS + 1):
        r = call_lm(client, model, msgs,
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    extra_body=params["extra_body"],
                    tools=tools)
        served = check_served(r, model, "run_with_messages")
        choice = r.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
                "reasoning_content": getattr(msg, "reasoning_content", None) or "",
            })
            for tc in tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                out, is_err = canned_tool_result(tc.function.name, args, canned)
                msgs.append({"role": "tool", "content": out, "tool_call_id": tc.id})
                steps.append({"tool": tc.function.name, "args": args, "result": out[:200], "error": is_err})
            continue
        return {"content": msg.content or "", "reasoning": getattr(msg, "reasoning_content", None) or "",
                "tool_steps": steps, "finish": choice.finish_reason,
            "model_served": served}
    return {"content": "", "reasoning": "", "tool_steps": steps, "finish": "max_tool_steps",
            "model_served": served}


def cmd_e2e(file: str | None = None) -> None:
    """End-to-end: run the WHOLE saved optimized transcript (all messages, the
    file's own model/request params) with the optimized pair in place, appending
    each tzip usage at the end. Verifies on the student model (thinking on).
    Optionally takes a transcript path (default: round-1 optimized transcript)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    transcript = Path(file) if file else OPTIMIZED_TRANSCRIPT
    if not transcript.exists():
        log.error("%s not found — run 'train' first", transcript)
        return
    doc = json.loads(transcript.read_text())
    fz = make_frozen(json.loads(TRANSCRIPT_JSON.read_text()))  # canned tool results
    models = load_models()
    client = make_client(models)
    verify_routing(client)  # abort unless the pinned affinity really reaches both workers
    params = {
        "temperature": doc.get("temperature"),
        "max_tokens": doc.get("max_tokens", STUDENT_MAX_TOKENS),
        "extra_body": {
            k: doc[k] for k in ("top_k", "repeat_penalty", "chat_template_kwargs") if k in doc
        },
    }
    tools = doc.get("tools", [])
    model = doc.get("model", STUDENT_MODEL)
    e2e_out = RUNS_DIR / (f"e2e-{transcript.stem.replace('skill-example-LiquidAI-LFM2.5-2.6B-', '').replace('.json', '')}.json"
                          if transcript.stem != OPTIMIZED_TRANSCRIPT.stem else "e2e-verification.json")
    log.info("### e2e start: file=%s model=%s thinking=%s temp=%s top_k=%s repeat_penalty=%s msgs=%d",
             transcript.name, model, params["extra_body"].get("chat_template_kwargs"),
             params["temperature"], params["extra_body"].get("top_k"),
             params["extra_body"].get("repeat_penalty"), len(doc["messages"]))
    out = {}
    n_ok = 0
    for arg in ("", "on", "lite", "full", "ultra", "off"):
        msgs = doc["messages"] + [{"role": "user", "content": [{"type": "text", "text": build_tzip_block_text(arg)}]}]
        res = run_with_messages(client, msgs, tools, model, params, fz["canned"])
        expected = TZIP_EXPECTED[arg]
        ok = norm(res["content"]) == norm(expected)
        n_ok += int(ok)
        out[arg or "empty"] = {
            "expected": expected, "reply": res["content"], "reasoning": res["reasoning"],
            "tool_steps": res["tool_steps"], "finish": res["finish"],
            "ok": ok, "score": score_case({"check": "tzip", "expected": expected}, res),
        }
        log.info("e2e %-6s ok=%s model=%s expected=%r reply=%r", arg or "empty", ok, model, expected, res["content"][:100])
    e2e_out.write_text(json.dumps({"model": model, "file": transcript.name, "cases": out}, indent=2))
    log.info("### e2e done: %d/6 exact matches -> %s", n_ok, e2e_out)


def cmd_train() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(TRANSCRIPT_JSON.read_text())
    fz = make_frozen(frozen)
    models = load_models()
    # teacher sampling params from models.json (values used, never logged)
    for m in models.get("models", []):
        if m.get("id") == TEACHER_MODEL:
            models["teacher_sampling"] = {k: v for k, v in m.get("samplingParams", {}).items()
                                          if v is not None and k in ("temperature", "top_k", "min_p", "presence_penalty", "repeat_penalty")}
            break
    client = make_client(models)
    verify_routing(client)  # abort unless the pinned affinity really reaches both workers

    seed = {
        "user_content": frozen["messages"][5]["content"][0]["text"],
        "assistant_content": frozen["messages"][6]["content"],
        "reasoning_content": frozen["messages"][6]["reasoning_content"],
    }
    log.info("### train start: seed pair from frozen transcript; %d test cases", len(fz["cases"]))
    log.info("student model (evaluated, thinking ON): %s | teacher model (reflection, thinking OFF): %s",
             STUDENT_MODEL, TEACHER_MODEL)

    # --- baseline of the seed pair ---------------------------------------
    baseline = {}
    for case in fz["cases"]:
        score, _ = eval_case(client, seed, case, fz, "baseline")
        baseline[case["id"]] = round(score, 3)
    BASELINE_JSON.write_text(json.dumps(baseline, indent=2))
    mean_base = sum(baseline.values()) / len(baseline)
    log.info("baseline (mean over %d draws each): %s (mean %.3f)", N_DRAWS, baseline, mean_base)

    # --- GEPA -------------------------------------------------------------
    from gepa import NoImprovementStopper, ScoreThresholdStopper
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    reflect = teacher_lm(client, models)
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(GEPA_RUN_DIR),
            seed=7,
            max_metric_calls=150,
            parallel=False,  # sequential evaluation, per run policy
            cache_evaluation=True,
            raise_on_exception=False,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflect,
            reflection_minibatch_size=3,
            module_selector="round_robin",
            skip_perfect_score=True,
            perfect_score=1.0,
            reflection_prompt_template=TEMPLATES,
        ),
        stop_callbacks=[
            ScoreThresholdStopper(threshold=1.0),
            NoImprovementStopper(max_iterations_without_improvement=12),
        ],
    )
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=make_evaluator(client, fz, "gepa"),
        dataset=fz["cases"],
        valset=None,  # multi-task mode: valset defaults to dataset (we WANT this overfit)
        config=config,  # objective/background embedded in the custom reflection templates
    )
    best = result.best_candidate
    result_info = {
        "best_idx": getattr(result, "best_idx", None),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "num_candidates": len(getattr(result, "candidates", []) or []),
        "val_aggregate_scores": [round(s, 3) for s in (getattr(result, "val_aggregate_scores", []) or [])],
    }
    log.info("GEPA done: %s", result_info)

    # --- final verification: 9 train cases + 1 unknown-skill holdout ------
    final = {}
    all_cases = fz["cases"] + fz["holdout"]
    for case in all_cases:
        draws = []
        for i in range(N_DRAWS):
            t0 = time.time()
            res = run_student_case(client, best, case, fz)
            s = score_case(case, res)
            log_eval("final", f"{case['id']}#d{i + 1}", cand_hash(best), s, res, time.time() - t0)
            draws.append({
                "score": round(s, 3),
                "reply": res["content"],
                "reasoning": res["reasoning"],
                "tool_steps": res["tool_steps"],
                "finish": res["finish"],
            })
        final[case["id"]] = {"mean": round(sum(d["score"] for d in draws) / len(draws), 3),
                              "expected": case["expected"], "draws": draws}
    train_mean = sum(v["mean"] for k, v in final.items() if k != "echo-holdout") / len(fz["cases"])
    FINAL_VERIFY_JSON.write_text(json.dumps({"model": STUDENT_MODEL, "thinking": True,
                                             "n_draws": N_DRAWS, "cases": final}, indent=2))
    log.info("final verification train-mean=%.3f: %s", train_mean,
             {k: v["mean"] for k, v in final.items() if k != "echo-holdout"})
    hold = final.get("echo-holdout")
    log.info("holdout unknown-skill (echo) mean=%.3f draws=%s",
             hold["mean"], [d["score"] for d in hold["draws"]])
    # baseline holdout for comparison (same seed pair, same unknown skill)
    base_hold = {}
    for i in range(N_DRAWS):
        res = run_student_case(client, seed, fz["holdout"][0], fz)
        base_hold[i + 1] = round(score_case(fz["holdout"][0], res), 3)
    log.info("seed baseline holdout (echo) draws: %s", base_hold)

    # --- deliverables ------------------------------------------------------
    BEST_PAIR_JSON.write_text(json.dumps({"student_model": STUDENT_MODEL, "teacher_model": TEACHER_MODEL,
                                          "pair": best, "seed": seed, "result": result_info,
                                          "baseline": baseline, "final_train_mean": train_mean,
                                          "final_holdout": final.get("echo-holdout", {}).get("mean"),
                                          "seed_holdout_draws": base_hold}, indent=2))
    out = copy.deepcopy(frozen)
    out["messages"][5] = {"role": "user", "content": [{"type": "text", "text": best["user_content"]}]}
    out["messages"][6] = {"role": "assistant", "content": best["assistant_content"],
                          "reasoning_content": best["reasoning_content"]}
    OPTIMIZED_TRANSCRIPT.write_text(json.dumps(out, indent=2))
    log.info("### train done: pair -> %s; transcript -> %s", BEST_PAIR_JSON, OPTIMIZED_TRANSCRIPT)
    log.info("best pair:\n  user: %s\n  assistant: %s\n  reasoning: %s",
             best["user_content"], best["assistant_content"], best["reasoning_content"])


def cmd_train2() -> None:
    """Round 2: warm-start GEPA from the round-1 best pair (runs/best-pair.json),
    300 metric-call budget, failure-mode-annotated reflection templates."""
    frozen = json.loads(TRANSCRIPT_JSON.read_text())
    fz = make_frozen(frozen)
    models = load_models()
    for m in models.get("models", []):
        if m.get("id") == TEACHER_MODEL:
            models["teacher_sampling"] = {k: v for k, v in m.get("samplingParams", {}).items()
                                          if v is not None and k in ("temperature", "top_k", "min_p", "presence_penalty", "repeat_penalty")}
            break
    client = make_client(models)
    verify_routing(client)  # abort unless the pinned affinity really reaches both workers

    if not BEST_PAIR_JSON.exists():
        log.error("%s not found — run 'train' (round 1) first", BEST_PAIR_JSON)
        raise SystemExit(1)
    round1 = json.loads(BEST_PAIR_JSON.read_text())
    seed = round1["pair"]
    log.info("### train2 start: warm-start from round-1 best pair (train mean %.3f)", round1.get("final_train_mean", float("nan")))
    log.info("round-1 best pair: user=%r | assistant=%s...", seed["user_content"][:80], seed["assistant_content"][:80])
    log.info("student model (evaluated, thinking ON): %s | teacher model (reflection, thinking OFF): %s",
             STUDENT_MODEL, TEACHER_MODEL)

    from gepa import NoImprovementStopper, ScoreThresholdStopper
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    reflect = teacher_lm(client, models)
    GEPA_RUN_DIR2.mkdir(parents=True, exist_ok=True)
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(GEPA_RUN_DIR2),
            seed=2026,
            max_metric_calls=MAX_METRIC_CALLS2,
            parallel=False,  # sequential evaluation, per run policy
            cache_evaluation=True,
            raise_on_exception=False,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflect,
            reflection_minibatch_size=3,
            module_selector="round_robin",
            skip_perfect_score=True,
            perfect_score=1.0,
            reflection_prompt_template=TEMPLATES_R2,
        ),
        stop_callbacks=[
            ScoreThresholdStopper(threshold=1.0),
            NoImprovementStopper(max_iterations_without_improvement=12),
        ],
    )
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=make_evaluator(client, fz, "gepa2"),
        dataset=fz["cases"],
        valset=None,  # multi-task mode: valset defaults to dataset (we WANT this overfit)
        config=config,  # objective/background/failure-modes embedded in the custom templates
    )
    best = result.best_candidate
    result_info = {
        "best_idx": getattr(result, "best_idx", None),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "num_candidates": len(getattr(result, "candidates", []) or []),
        "val_aggregate_scores": [round(s, 3) for s in (getattr(result, "val_aggregate_scores", []) or [])],
    }
    log.info("GEPA round 2 done: %s", result_info)

    # --- final verification (same protocol as round 1: 12 train cases + holdout) ---
    final = {}
    all_cases = fz["cases"] + fz["holdout"]
    for case in all_cases:
        draws = []
        for i in range(N_DRAWS):
            t0 = time.time()
            res = run_student_case(client, best, case, fz)
            s = score_case(case, res)
            log_eval("final2", f"{case['id']}#d{i + 1}", cand_hash(best), s, res, time.time() - t0)
            draws.append({
                "score": round(s, 3),
                "reply": res["content"],
                "reasoning": res["reasoning"],
                "tool_steps": res["tool_steps"],
                "finish": res["finish"],
            })
        final[case["id"]] = {"mean": round(sum(d["score"] for d in draws) / len(draws), 3),
                              "expected": case["expected"], "draws": draws}
    train_mean = sum(v["mean"] for k, v in final.items() if k != "echo-holdout") / len(fz["cases"])
    FINAL_VERIFY_JSON2.write_text(json.dumps({"model": STUDENT_MODEL, "thinking": True,
                                              "n_draws": N_DRAWS, "cases": final}, indent=2))
    log.info("final2 verification train-mean=%.3f: %s", train_mean,
             {k: v["mean"] for k, v in final.items() if k != "echo-holdout"})
    hold = final.get("echo-holdout")
    log.info("holdout unknown-skill (echo) mean=%.3f draws=%s",
             hold["mean"], [d["score"] for d in hold["draws"]])

    # --- deliverables (round-1 artifacts untouched) ----------------------------
    BEST_PAIR_JSON2.write_text(json.dumps({"student_model": STUDENT_MODEL, "teacher_model": TEACHER_MODEL,
                                           "pair": best, "seed": seed,
                                           "warm_started_from": "runs/best-pair.json (round 1 best)",
                                           "result": result_info,
                                           "final_train_mean": train_mean,
                                           "final_holdout": final.get("echo-holdout", {}).get("mean")}, indent=2))
    out = copy.deepcopy(frozen)
    out["messages"][5] = {"role": "user", "content": [{"type": "text", "text": best["user_content"]}]}
    out["messages"][6] = {"role": "assistant", "content": best["assistant_content"],
                          "reasoning_content": best["reasoning_content"]}
    OPTIMIZED_TRANSCRIPT2.write_text(json.dumps(out, indent=2))
    log.info("### train2 done: pair -> %s; transcript -> %s", BEST_PAIR_JSON2, OPTIMIZED_TRANSCRIPT2)
    log.info("best pair:\n  user: %s\n  assistant: %s\n  reasoning: %s",
             best["user_content"], best["assistant_content"], best["reasoning_content"])


def cmd_train3() -> None:
    """Round 3: warm-start from the round-2 best pair with the full two-round
    post-mortem in the reflection background."""
    frozen = json.loads(TRANSCRIPT_JSON.read_text())
    fz = make_frozen(frozen)
    models = load_models()
    for m in models.get("models", []):
        if m.get("id") == TEACHER_MODEL:
            models["teacher_sampling"] = {k: v for k, v in m.get("samplingParams", {}).items()
                                          if v is not None and k in ("temperature", "top_k", "min_p", "presence_penalty", "repeat_penalty")}
            break
    client = make_client(models)
    verify_routing(client)

    if not BEST_PAIR_JSON2.exists():
        log.error("%s not found — run 'train2' first", BEST_PAIR_JSON2)
        raise SystemExit(1)
    round2 = json.loads(BEST_PAIR_JSON2.read_text())
    seed = round2["pair"]
    log.info("### train3 start: warm-start from round-2 best pair (train mean %.3f)", round2.get("final_train_mean", float("nan")))
    log.info("student model (evaluated, thinking ON): %s | teacher model (reflection, thinking OFF): %s",
             STUDENT_MODEL, TEACHER_MODEL)

    from gepa import NoImprovementStopper, ScoreThresholdStopper
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    reflect = teacher_lm(client, models)
    GEPA_RUN_DIR3.mkdir(parents=True, exist_ok=True)
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(GEPA_RUN_DIR3),
            seed=1337,
            max_metric_calls=MAX_METRIC_CALLS3,
            parallel=False,  # sequential evaluation, per run policy
            cache_evaluation=True,
            raise_on_exception=False,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflect,
            reflection_minibatch_size=3,
            module_selector="round_robin",
            skip_perfect_score=True,
            perfect_score=1.0,
            reflection_prompt_template=TEMPLATES_R3,
        ),
        stop_callbacks=[
            ScoreThresholdStopper(threshold=1.0),
            NoImprovementStopper(max_iterations_without_improvement=12),
        ],
    )
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=make_evaluator(client, fz, "gepa3"),
        dataset=fz["cases"],
        valset=None,  # multi-task mode: valset defaults to dataset (we WANT this overfit)
        config=config,  # objective/background/post-mortem embedded in the custom templates
    )
    best = result.best_candidate
    result_info = {
        "best_idx": getattr(result, "best_idx", None),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "num_candidates": len(getattr(result, "candidates", []) or []),
        "val_aggregate_scores": [round(s, 3) for s in (getattr(result, "val_aggregate_scores", []) or [])],
    }
    log.info("GEPA round 3 done: %s", result_info)

    final = {}
    all_cases = fz["cases"] + fz["holdout"]
    for case in all_cases:
        draws = []
        for i in range(N_DRAWS):
            t0 = time.time()
            res = run_student_case(client, best, case, fz)
            s = score_case(case, res)
            log_eval("final3", f"{case['id']}#d{i + 1}", cand_hash(best), s, res, time.time() - t0)
            draws.append({
                "score": round(s, 3),
                "reply": res["content"],
                "reasoning": res["reasoning"],
                "tool_steps": res["tool_steps"],
                "finish": res["finish"],
            })
        final[case["id"]] = {"mean": round(sum(d["score"] for d in draws) / len(draws), 3),
                              "expected": case["expected"], "draws": draws}
    train_mean = sum(v["mean"] for k, v in final.items() if k != "echo-holdout") / len(fz["cases"])
    FINAL_VERIFY_JSON3.write_text(json.dumps({"model": STUDENT_MODEL, "thinking": True,
                                              "n_draws": N_DRAWS, "cases": final}, indent=2))
    log.info("final3 verification train-mean=%.3f: %s", train_mean,
             {k: v["mean"] for k, v in final.items() if k != "echo-holdout"})
    hold = final.get("echo-holdout")
    log.info("holdout unknown-skill (echo) mean=%.3f draws=%s",
             hold["mean"], [d["score"] for d in hold["draws"]])

    BEST_PAIR_JSON3.write_text(json.dumps({"student_model": STUDENT_MODEL, "teacher_model": TEACHER_MODEL,
                                           "pair": best, "seed": seed,
                                           "warm_started_from": "runs/best-pair-r2.json (round 2 best)",
                                           "result": result_info,
                                           "final_train_mean": train_mean,
                                           "final_holdout": final.get("echo-holdout", {}).get("mean")}, indent=2))
    out = copy.deepcopy(frozen)
    out["messages"][5] = {"role": "user", "content": [{"type": "text", "text": best["user_content"]}]}
    out["messages"][6] = {"role": "assistant", "content": best["assistant_content"],
                          "reasoning_content": best["reasoning_content"]}
    OPTIMIZED_TRANSCRIPT3.write_text(json.dumps(out, indent=2))
    log.info("### train3 done: pair -> %s; transcript -> %s", BEST_PAIR_JSON3, OPTIMIZED_TRANSCRIPT3)
    log.info("best pair:\n  user: %s\n  assistant: %s\n  reasoning: %s",
             best["user_content"], best["assistant_content"], best["reasoning_content"])


def main() -> None:
    setup_logging()
    log.info("run id: %s", RUN_ID)
    log.info("affinity: fresh per-request session ids — student=dspy-optim-lfm2.5-2.6b-<sid> | teacher=dspy-optim-qwen3.8-27b-<sid>")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rollout", help="teacher pi rollouts for all tzip inputs")
    sub.add_parser("train", help="GEPA-optimize the synthetic message pair")
    sub.add_parser("train2", help="round 2: warm-start GEPA from the round-1 best pair")
    sub.add_parser("train3", help="round 3: warm-start from round-2 best with two-round post-mortem")
    e = sub.add_parser("e2e", help="whole optimized transcript + tzip test at the end, on the student")
    e.add_argument("file", nargs="?", default=None, help="optimized transcript to run (default: round-1)")
    args = p.parse_args()
    if args.cmd == "rollout":
        cmd_rollout()
    elif args.cmd == "train":
        cmd_train()
    elif args.cmd == "train2":
        cmd_train2()
    elif args.cmd == "train3":
        cmd_train3()
    elif args.cmd == "e2e":
        cmd_e2e(args.file)


if __name__ == "__main__":
    main()
