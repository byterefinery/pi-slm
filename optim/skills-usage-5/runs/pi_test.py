#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Real-pi test: preload the few-shot (with candidate pair) into crafted pi v3
session files and run the 12 tzip cases through the actual `pi` CLI in random
temp dirs. Student (thinking ON) and teacher (thinking OFF). All serial.

Usage: uv run --script pi_test.py [seed|best]
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    Api, ART, CASES, TZIP_SKILL, content_ok, find_pair_index, judge, log,
    seed_pair, tzip_block,
)

HERE = Path(__file__).resolve().parent
PI_BIN = shutil.which("pi") or "pi"
BASE_SYSTEM_PROMPT = (
    "You are an expert coding assistant operating inside pi, a coding agent harness. "
    "You help users by reading files, executing commands, editing code, and writing new files."
)
STUDENT_SPEC = ("LiquidAI/LFM2.5-2.6B", "high")
TEACHER_SPEC = ("Qwen/Qwen3.8-27B", "off")
ZERO_USAGE = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
              "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}
BASE_TS = datetime(2026, 8, 21, 10, 0, 0, tzinfo=timezone.utc)


def load_pair(which: str) -> dict:
    if which == "seed":
        return seed_pair()
    f = HERE / "best_candidate.json"
    return json.loads(f.read_text())


def _user_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")


def _assistant_blocks(m: dict) -> list[dict]:
    blocks = []
    rc = m.get("reasoning_content")
    if rc:
        blocks.append({"type": "thinking", "thinking": rc})
    c = m.get("content")
    if c:
        blocks.append({"type": "text", "text": c if isinstance(c, str) else _user_text(m)})
    for tc in m.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        blocks.append({"type": "toolCall", "id": tc.get("id", uuid.uuid4().hex[:8]),
                       "name": fn.get("name"), "arguments": args or {}})
    return blocks


def build_session_file(path: Path, pair: dict, case: dict, model_id: str, cwd: Path) -> None:
    """Session = system-less few-shot (with spliced pair) + prior tzip turns (transition).
    The CURRENT invocation is passed as the pi prompt, not preloaded."""
    msgs = [json.loads(json.dumps(m)) for m in ART["messages"]]
    i = find_pair_index(msgs)
    msgs[i] = {"role": "user", "content": [{"type": "text", "text": pair["user_question"]}]}
    msgs[i + 1] = {"role": "assistant", "content": pair["assistant_content"],
                   "reasoning_content": pair["assistant_reasoning_content"]}
    body = msgs[1:]  # drop pi-system message
    if case.get("prior"):
        from common import _arg_of
        body.append({"role": "user", "content": [{"type": "text", "text": tzip_block(_arg_of(case["prior"]))}]})
        body.append({"role": "assistant", "content": case["prior"]})

    lines = [json.dumps({"type": "session", "version": 3, "id": str(uuid.uuid4()),
                         "timestamp": BASE_TS.isoformat().replace("+00:00", ".000Z"),
                         "cwd": str(cwd)})]
    prev_id = None
    ts = BASE_TS
    for m in body:
        ts = ts + timedelta(milliseconds=1000)
        eid = uuid.uuid4().hex[:8]
        role = m.get("role")
        if role == "user":
            msg = {"role": "user", "content": _user_text(m)}
        elif role == "assistant":
            blocks = _assistant_blocks(m)
            msg = {"role": "assistant", "content": blocks, "provider": "llamacpp",
                   "model": model_id, "usage": ZERO_USAGE,
                   "stopReason": "toolUse" if any(b["type"] == "toolCall" for b in blocks) else "stop"}
        elif role == "tool":
            msg = {"role": "toolResult", "toolCallId": m.get("tool_call_id"),
                   "toolName": _tool_name_for(body, m.get("tool_call_id")),
                   "content": [{"type": "text", "text": m.get("content", "")}], "isError": False}
        else:
            continue
        lines.append(json.dumps({"type": "message", "id": eid, "parentId": prev_id,
                                 "timestamp": ts.isoformat().replace("+00:00", ".000Z"),
                                 "message": msg}))
        prev_id = eid
    path.write_text("\n".join(lines) + "\n")


def _tool_name_for(body: list[dict], tool_call_id: str) -> str:
    for m in body:
        for tc in m.get("tool_calls", []):
            if tc.get("id") == tool_call_id:
                return tc.get("function", {}).get("name", "unknown")
    return "unknown"


def run_pi(model_id: str, thinking: str, cwd: Path, session: Path, prompt: str,
           timeout: int = 300, extra_args: list[str] | None = None) -> dict:
    cmd = [
        PI_BIN,
        "--model", f"llamacpp/{model_id}",
        "--thinking", thinking,
        "--no-tools", "--tools", "read,write,edit,bash",
        "--system-prompt", BASE_SYSTEM_PROMPT,
        "--no-context-files",
        "--skill", str(TZIP_SKILL),
        "--session", str(session),
        "-p", prompt,
    ]
    if extra_args:
        cmd = [cmd[0]] + extra_args + cmd[1:]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    # read the final assistant entry from the (updated) session file
    entries = [json.loads(l) for l in session.read_text().splitlines() if l.strip()]
    asst = [e for e in entries if e.get("type") == "message"
            and e.get("message", {}).get("role") == "assistant"]
    text, thinking_txt = "", ""
    if asst:
        for b in asst[-1]["message"].get("content", []):
            if b.get("type") == "text":
                text = b.get("text", "")
            elif b.get("type") == "thinking":
                thinking_txt = b.get("thinking", "")
    return {
        "exit_code": p.returncode,
        "stdout": p.stdout[-4000:],
        "stderr": p.stderr[-2000:],
        "content": text,
        "reasoning_content": thinking_txt,
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "best"
    pair = load_pair(which)
    api = Api()
    root = HERE / "pisessions"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    out = {"which": which, "models": {}}
    for label, (model_id, thinking) in [("student", STUDENT_SPEC), ("teacher", TEACHER_SPEC)]:
        results = []
        for case in CASES:  # strictly serial
            cwd = Path(tempfile.mkdtemp(prefix=f"pitzip-{label}-"))
            session = root / f"{label}-{case['id']}.jsonl"
            build_session_file(session, pair, case, model_id, cwd)
            prompt = tzip_block(case.get("arg", ""))
            try:
                r = run_pi(model_id, thinking, cwd, session, prompt)
            except subprocess.TimeoutExpired:
                r = {"exit_code": -1, "stdout": "", "stderr": "timeout", "content": "", "reasoning_content": ""}
            ok = content_ok(r["content"], case["expected"])
            j = judge(api, pair, case, r["content"], r["reasoning_content"], use_cache=False)
            results.append({
                "id": case["id"], "expected": case["expected"], "actual": r["content"],
                "reasoning": r["reasoning_content"], "content_ok": ok, "leak": j["leak"],
                "leak_reason": j["reason"], "exit_code": r["exit_code"],
                "stderr_tail": r["stderr"][-300:],
            })
            log(f"pi-{label} {case['id']}: ok={ok} leak={j['leak']} exit={r['exit_code']} "
                f"actual={r['content']!r} thinking_len={len(r['reasoning_content'])}")
        n = len(results)
        acc = sum(r["content_ok"] for r in results) / n
        leakr = sum(r["leak"] for r in results) / n
        out["models"][label] = {"acc": acc, "leak_rate": leakr, "results": results}
        log(f"pi-{label} DONE acc={acc * 100:.1f}% leak={leakr * 100:.1f}%")

    (HERE / "pi_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
