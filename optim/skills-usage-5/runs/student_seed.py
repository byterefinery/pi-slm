#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Student seed baseline (BEFORE optimization), all calls serial."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, eval_candidate, log, seed_pair


def main() -> None:
    api = Api()
    pair = seed_pair()
    res = eval_candidate(api, pair, use_cache=True, tag="student-seed")
    out_f = Path(__file__).resolve().parent / "student_seed.json"
    out_f.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for r in res["results"]:
        log(f"student-seed {r['id']}: ok={r['content_ok']} leak={r['leak']} actual={r['actual']!r}")


if __name__ == "__main__":
    main()
