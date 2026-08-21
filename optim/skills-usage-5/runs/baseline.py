#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Teacher baseline: how reliably can the teacher (thinking OFF) use tzip
under the UNOPTIMIZED (seed) few-shot? All calls serial."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    Api, CASES, build_case_messages, content_ok, judge, log, seed_pair,
    teacher_response_call,
)


def main() -> None:
    api = Api()
    pair = seed_pair()
    results = []
    for case in CASES:  # strictly serial
        msgs = build_case_messages(pair, case)
        out = teacher_response_call(api, msgs)
        ok = content_ok(out["content"], case["expected"])
        j = judge(api, pair, case, out["content"], out["reasoning_content"], use_cache=True)
        results.append({
            "id": case["id"], "expected": case["expected"], "actual": out["content"],
            "reasoning": out["reasoning_content"], "content_ok": ok, "leak": j["leak"],
            "leak_reason": j["reason"], "finish_reason": out["finish_reason"],
        })
        log(f"baseline-teacher {case['id']}: ok={ok} leak={j['leak']} actual={out['content']!r}")
    n = len(results)
    acc = sum(r["content_ok"] for r in results) / n
    leakr = sum(r["leak"] for r in results) / n
    out_f = Path(__file__).resolve().parent / "baseline_teacher.json"
    out_f.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log(f"baseline-teacher DONE acc={acc * 100:.1f}% leak={leakr * 100:.1f}% n={n}")


if __name__ == "__main__":
    main()
