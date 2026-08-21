#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Variance probe: 3 FRESH (uncached) full-suite runs of the given candidate.
Usage: uv run --script variance.py [seed|best]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, eval_candidate, log, seed_pair

HERE = Path(__file__).resolve().parent


def load_best() -> dict:
    f = HERE / "best_candidate.json"
    if not f.exists():
        raise SystemExit("best_candidate.json not found - run optimize.py first")
    return json.loads(f.read_text())


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "seed"
    pair = load_best() if which == "best" else seed_pair()
    api = Api()
    runs = []
    for i in range(3):
        r = eval_candidate(api, pair, use_cache=False, tag=f"variance-{which}-{i + 1}")
        r["per_case"] = {x["id"]: (x["content_ok"], x["leak"]) for x in r["results"]}
        del r["results"]
        runs.append(r)
    accs = [r["acc"] for r in runs]
    leaks = [r["leak_rate"] for r in runs]
    log(
        f"variance-{which} min-of-3 acc={min(accs) * 100:.1f}% leak={max(leaks) * 100:.1f}% "
        f"(accs={accs}, leaks={leaks})"
    )
    (HERE / f"variance_{which}.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
