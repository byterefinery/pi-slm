#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
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
    extra_headers={"x-session-affinity": "dspy-optim"},
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
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"  # thinking: ON
TEACHER_MODEL = "Qwen/Qwen3.8-27B"  # thinking: OFF
PROVIDER = "llamacpp"
SESSION_HEADER = {"x-session-affinity": "dspy-optim"}
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
ROLLOUTS_DIR = RUNS_DIR / "rollouts"
LOG_FILE = HERE / "train.log"
OUTPUT_JSON = RUNS_DIR / "teacher-tzip-responses.json"

TZIP_SKILL_DIR = Path("/home/mtasic/projects-b/pi-slm/.agents/skills-byterefinery/tzip")

TEACHER_THINKING = "off"
MAX_ATTEMPTS = 4
PI_TIMEOUT = 240  # seconds per pi run

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


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rollout", help="teacher pi rollouts for all tzip inputs")
    args = p.parse_args()
    if args.cmd == "rollout":
        cmd_rollout()


if __name__ == "__main__":
    main()
