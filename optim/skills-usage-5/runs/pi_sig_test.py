#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Decisive test: does adding thinkingSignature='reasoning_content' to the
session's thinking blocks make pi emit reasoning_content in the request, and
does that restore the student's JSON-regime behaviour?

Runs student on case 'full' (failed before) with and without the signature.
Serial.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CASES, content_ok, log, tzip_block
import pi_test

HERE = Path(__file__).resolve().parent
pair = json.loads((HERE / "best_candidate.json").read_text())
case = next(c for c in CASES if c["id"] == "full")
model_id, thinking = pi_test.STUDENT_SPEC


def build_with_sig(path, with_sig: bool):
    # replicate pi_test.build_session_file but optionally add thinkingSignature
    from common import ART, find_pair_index, _user_text
    msgs = [json.loads(json.dumps(m)) for m in ART["messages"]]
    i = find_pair_index(msgs)
    msgs[i] = {"role": "user", "content": [{"type": "text", "text": pair["user_question"]}]}
    msgs[i + 1] = {"role": "assistant", "content": pair["assistant_content"],
                   "reasoning_content": pair["assistant_reasoning_content"]}
    body = msgs[1:]
    if case.get("prior"):
        from common import _arg_of
        body.append({"role": "user", "content": [{"type": "text", "text": tzip_block(_arg_of(case["prior"]))}]})
        body.append({"role": "assistant", "content": case["prior"]})
    # build via pi_test but patch _assistant_blocks
    import pi_test as pt
    orig = pt._assistant_blocks
    def patched(m):
        blocks = orig(m)
        if with_sig:
            for b in blocks:
                if b.get("type") == "thinking":
                    b["thinkingSignature"] = "reasoning_content"
        return blocks
    pt._assistant_blocks = patched
    try:
        pt.build_session_file(path, pair, case, model_id, cwd)
    finally:
        pt._assistant_blocks = orig


for with_sig in (False, True):
    cwd = Path(tempfile.mkdtemp(prefix=f"pitzip-sig{str(with_sig).lower()}-"))
    s = HERE / f"pisessions-sig{str(with_sig).lower()}.jsonl"
    build_with_sig(s, with_sig)
    r = pi_test.run_pi(model_id, thinking, cwd, s, tzip_block(case["arg"]))
    ok = content_ok(r["content"], case["expected"])
    print(f"\n=== with_sig={with_sig} ===")
    print(f"content: {r['content']!r}")
    print(f"ok={ok} expected={case['expected']!r}")
    log(f"pi-sig-test student/full with_sig={with_sig}: ok={ok} content={r['content'][:80]!r}")
