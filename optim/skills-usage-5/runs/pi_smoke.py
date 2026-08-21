#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""pi smoke: ONE teacher case (lite) to validate --session/-p mechanics."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CASES, content_ok, log
from pi_test import STUDENT_SPEC, TEACHER_SPEC, build_session_file, run_pi

HERE = Path(__file__).resolve().parent
pair = json.loads((HERE / "best_candidate.json").read_text())
case = next(c for c in CASES if c["id"] == "lite")
model_id, thinking = TEACHER_SPEC

cwd = Path(tempfile.mkdtemp(prefix="pitzip-smoke-"))
session = HERE / "pisessions-smoke.jsonl"
build_session_file(session, pair, case, model_id, cwd)
r = run_pi(model_id, thinking, cwd, session, __import__("common").tzip_block(case["arg"]))
print("exit:", r["exit_code"])
print("content:", repr(r["content"]))
print("thinking_len:", len(r["reasoning_content"]))
print("stdout tail:", r["stdout"][-500:])
print("stderr tail:", r["stderr"][-500:])
log(f"pi-smoke teacher/lite: exit={r['exit_code']} content={r['content']!r} ok={content_ok(r['content'], case['expected'])} thinking_len={len(r['reasoning_content'])}")
