#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Plumbing smoke: 1 student call (case 'lite', seed pair) + judge + style check."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, eval_case, log, seed_pair, style_check

CASE = {"id": "lite", "arg": "lite", "expected": "tzip lite activated"}

api = Api()
pair = seed_pair()
r = eval_case(api, pair, CASE, use_cache=True)
style = style_check(api, pair, use_cache=True)
out = {"case": r, "style": style}
Path(__file__).resolve().parent.joinpath("smoke.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
log(f"smoke student content={r['actual']!r} ok={r['content_ok']} finish={r['finish_reason']}")
log(f"smoke reasoning={r['reasoning'][:200]!r}")
log(f"smoke judge leak={r['leak']} arg={r['judge_arg']!r} reason={r['leak_reason']!r}")
log(f"smoke style_ok={style['style_ok']} issues={style['issues']!r}")
