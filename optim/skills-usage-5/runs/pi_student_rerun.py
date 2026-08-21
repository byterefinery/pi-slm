#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Student-only actual-pi re-run (best pair), to check 16.7% reproducibility. Serial."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, CASES, content_ok, judge, log, tzip_block
from pi_test import STUDENT_SPEC, build_session_file, run_pi

HERE = Path(__file__).resolve().parent
model_id, thinking = STUDENT_SPEC
pair = json.loads((HERE / "best_candidate.json").read_text())
api = Api()
results = []
for case in CASES:  # strictly serial
    cwd = Path(tempfile.mkdtemp(prefix="pirr-"))
    s = HERE / f"pisessions-rr-{case['id']}.jsonl"
    build_session_file(s, pair, case, model_id, cwd)
    r = run_pi(model_id, thinking, cwd, s, tzip_block(case.get("arg", "")))
    ok = content_ok(r["content"], case["expected"])
    j = judge(api, pair, case, r["content"], r["reasoning_content"], use_cache=False)
    results.append({"id": case["id"], "ok": ok, "leak": j["leak"],
                    "content": r["content"][:200], "reasoning": r["reasoning_content"][:300],
                    "exit_code": r["exit_code"]})
    log(f"pi-rerun {case['id']}: ok={ok} leak={j['leak']} actual={r['content'][:50]!r}")
n = len(results)
acc = sum(r["ok"] for r in results) / n
leak = sum(r["leak"] for r in results) / n
(HERE / "pi_student_rerun.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
log(f"pi-rerun DONE acc={acc * 100:.1f}% leak={leak * 100:.1f}%")
