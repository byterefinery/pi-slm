#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Isolation test for case 'full' (3 variants, serial) to explain the pi gap:
  A: JSON system prompt + history reasoning_content  (faithful; expect OK)
  B: pi   system prompt + history reasoning_content  (system-prompt effect)
  C: JSON system prompt, history reasoning STRIPPED  (stripped-CoT effect)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, CASES, build_case_messages, content_ok, student_call

HERE = Path(__file__).resolve().parent
pair = json.loads((HERE / "best_candidate.json").read_text())
case = next(c for c in CASES if c["id"] == "full")

PI_SYSTEM = ("You are an expert coding assistant operating inside pi, a coding agent harness. "
             "You help users by reading files, executing commands, editing code, and writing new files.")


def strip_history_reasoning(msgs):
    out = [json.loads(json.dumps(m)) for m in msgs]
    for m in out:
        if m.get("role") == "assistant":
            m.pop("reasoning_content", None)
    return out


api = Api()
base = build_case_messages(pair, case)

variants = {
    "A_json_sys_with_rc": base,
    "B_pi_sys_with_rc": None,
    "C_json_sys_no_rc": None,
}
b = json.loads(json.dumps(base))
b[0] = {"role": "system", "content": PI_SYSTEM}
variants["B_pi_sys_with_rc"] = b
variants["C_json_sys_no_rc"] = strip_history_reasoning(base)

res = {}
for name, msgs in variants.items():
    out = student_call(api, msgs, use_cache=False)
    ok = content_ok(out["content"], case["expected"])
    res[name] = {"ok": ok, "content": out["content"][:200], "reasoning": out["reasoning_content"][:200]}
    print(f"\n=== {name} === ok={ok}")
    print("content:", repr(out["content"][:150]))
    print("reasoning:", repr(out["reasoning_content"][:150]))
(HERE / "isolation_full.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
