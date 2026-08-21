#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///

'''
NOTE: this is how you instantiate dspy.LM - keep this string/comment
IMPORTANT: Radon values for API_BASE and API_KEY on demand from `~/.pi/agent/models.json`. Never leak them in text, logs or source code.

```python
lm = dspy.LM(
    f"openai/{model}", # used model
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=..., # read for used model
    max_tokens=16384,
    extra_headers={"x-session-affinity": "dspy-optim"},
    extra_body={
        "top_k": ..., # read for used model, or omit if missing
        "min_p": ..., # read for used model, or omit if missing
        "presence_penalty": ..., # read for used model, or omit if missing
        "repeat_penalty": ..., # read for used model, or omit if missing
    },
)
```

skills-usage-5 orchestrator. This run uses raw OpenAI-client calls (runs/common.py)
instead of dspy.LM so the request body replicates the captured
skill-example-*.json exactly (content arrays, reasoning_content history,
chat_template_kwargs, tools). Same endpoint, headers, and sampling params.

Stages (all model calls SERIAL - shared server):
  baseline  - teacher (thinking OFF) as responder on the 12 tzip cases (seed pair)
  seed      - student (thinking ON) on the 12 tzip cases (seed pair) = BEFORE numbers
  optimize  - GEPA (gepa.optimize_anything) optimizes ONLY the skill-system Q/A pair
  final     - deterministic + 3 fresh evals of the best pair, then write-back into
              skill-example-LiquidAI-LFM2.5-2.6B.json (only the pair changes)
  pi        - real `pi` CLI test in random temp dirs (student thinking ON, teacher
              thinking OFF), 12 cases each, preloaded session files

Usage: uv run train.py [baseline|seed|optimize|final|pi]   (default: all)
Progress is appended to train.log.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "runs"))
from common import get_creds, log  # noqa: E402

API_BASE, API_KEY = get_creds()  # never logged


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["baseline", "seed", "optimize", "final", "pi"]
    stages = order if stage == "all" else [stage]
    for s in stages:
        if s == "baseline":
            import baseline
            baseline.main()
        elif s == "seed":
            import student_seed
            student_seed.main()
        elif s == "optimize":
            import optimize
            optimize.main()
        elif s == "final":
            import final_eval
            final_eval.main()
        elif s == "pi":
            import pi_test
            pi_test.main()
        else:
            raise SystemExit(f"unknown stage: {s}")
    log("orchestrator done")


if __name__ == "__main__":
    main()
