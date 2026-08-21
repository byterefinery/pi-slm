#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Final evaluation of the GEPA-best candidate + byte-faithful artifact write-back.

1. deterministic eval (cached student/judge calls) -> acc, leak
2. 3 fresh (uncached) runs -> min-of-3 acc, max leak
3. splice the best pair into skill-example-LiquidAI-LFM2.5-2.6B.json, asserting
   that ONLY the two optimized messages change.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ARTIFACT, Api, eval_candidate, find_pair_index, log,
)

HERE = Path(__file__).resolve().parent


def load_best() -> dict:
    f = HERE / "best_candidate.json"
    if not f.exists():
        raise SystemExit("best_candidate.json not found - run optimize.py first")
    return json.loads(f.read_text())


def write_back(pair: dict) -> None:
    orig_text = ARTIFACT.read_text()
    orig = json.loads(orig_text)
    art = json.loads(orig_text)
    i = find_pair_index(art["messages"])
    art["messages"][i] = {"role": "user", "content": [{"type": "text", "text": pair["user_question"]}]}
    art["messages"][i + 1] = {
        "role": "assistant",
        "content": pair["assistant_content"],
        "reasoning_content": pair["assistant_reasoning_content"],
    }
    # assert: nothing else changed
    assert art["model"] == orig["model"]
    assert art["tools"] == orig["tools"]
    assert art["max_tokens"] == orig["max_tokens"]
    assert art["temperature"] == orig["temperature"]
    assert art["top_k"] == orig["top_k"]
    assert art["repeat_penalty"] == orig["repeat_penalty"]
    assert art["chat_template_kwargs"] == orig["chat_template_kwargs"]
    assert art["stream"] == orig["stream"]
    assert len(art["messages"]) == len(orig["messages"])
    for k in range(len(art["messages"])):
        if k in (i, i + 1):
            continue
        assert art["messages"][k] == orig["messages"][k], f"unexpected change at message {k}"
    ARTIFACT.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n")
    log(f"write-back OK: only messages[{i}] and messages[{i + 1}] changed")


def main() -> None:
    best = load_best()
    api = Api()

    det = eval_candidate(api, best, use_cache=True, tag="final-deterministic")
    (HERE / "final_deterministic.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))

    fresh = []
    for i in range(3):
        r = eval_candidate(api, best, use_cache=False, tag=f"final-fresh-{i + 1}")
        r["per_case"] = {x["id"]: (x["content_ok"], x["leak"]) for x in r["results"]}
        del r["results"]
        fresh.append(r)
    (HERE / "final_fresh.json").write_text(json.dumps(fresh, indent=2, ensure_ascii=False))

    accs = [r["acc"] for r in fresh]
    leaks = [r["leak_rate"] for r in fresh]
    log(
        f"FINAL best: deterministic acc={det['acc'] * 100:.1f}% leak={det['leak_rate'] * 100:.1f}% "
        f"style={'ok' if det['style']['style_ok'] else 'FAIL'} | "
        f"min-of-3-fresh acc={min(accs) * 100:.1f}% leak={max(leaks) * 100:.1f}%"
    )
    write_back(best)
    log("FINAL done")


if __name__ == "__main__":
    main()
