#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Real-HOME pi run WITH --no-extensions (same as the failing 16.7% runs
otherwise: real HOME, best pair, student thinking ON, direct to server).
Tests whether global pi state (extensions/settings) degrades the student.
Serial.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, CASES, content_ok, judge, log, tzip_block
from pi_test import STUDENT_SPEC, run_pi

HERE = Path(__file__).resolve().parent
model_id, thinking = STUDENT_SPEC
pair = json.loads((HERE / "best_candidate.json").read_text())
api = Api()
results = []
for case in CASES:  # strictly serial
    cwd = Path(tempfile.mkdtemp(prefix="pinoext-"))
    s = HERE / f"pisessions-noext-{case['id']}.jsonl"
    # build session (real-HOME run: same builder, session file in runs/)
    from pi_test import build_session_file
    build_session_file(s, pair, case, model_id, cwd)
    r = run_pi(model_id, thinking, cwd, s, tzip_block(case.get("arg", "")),
               extra_args=["--no-extensions"])
    ok = content_ok(r["content"], case["expected"])
    j = judge(api, pair, case, r["content"], r["reasoning_content"], use_cache=False)
    results.append({"id": case["id"], "ok": ok, "leak": j["leak"],
                    "content": r["content"][:200], "reasoning": r["reasoning_content"][:300],
                    "exit_code": r["exit_code"], "stderr_tail": r["stderr"][-200:]})
    log(f"pi-noext {case['id']}: ok={ok} leak={j['leak']} exit={r['exit_code']} actual={r['content'][:50]!r}")
n = len(results)
acc = sum(r["ok"] for r in results) / n
leak = sum(r["leak"] for r in results) / n
(HERE / "pi_noext.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
log(f"pi-noext DONE acc={acc * 100:.1f}% leak={leak * 100:.1f}%")
