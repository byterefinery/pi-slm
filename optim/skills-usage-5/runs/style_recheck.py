#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Style re-check of the best candidate under the corrected rubric (3 fresh, serial)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, log, style_check

HERE = Path(__file__).resolve().parent
best = json.loads((HERE / "best_candidate.json").read_text())
api = Api()
verdicts = [style_check(api, best, use_cache=False) for _ in range(3)]
for i, v in enumerate(verdicts, 1):
    log(f"style-fresh-{i}: ok={v['style_ok']} issues={v['issues']!r}")
(HERE / "style_recheck.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
