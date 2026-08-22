#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Systematic SDK bisection: minimal call (the shape a user would run
'openai directly') with the ORIGINAL affinity 'dspy-optim', then add one
component at a time: tools, extra_body fields, full prefix. Identify which
component flips routing from LFM to Qwen."""

import json
import re
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
TRANSCRIPT = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"

raw = MODELS_JSON.read_text()
cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
p = cfg["providers"]["llamacpp"]
base, key = p["baseUrl"], p["apiKey"]
frozen = json.loads(TRANSCRIPT.read_text())
prefix = frozen["messages"][:12]
SHORT = [{"role": "user", "content": "Say: world"}]


def probe(tag, messages, aff, tools=None, extra_body=None, max_tokens=65536, model="LiquidAI/LFM2.5-2.6B"):
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key, timeout=600)
    kw = dict(model=model, messages=messages, stream=False, max_tokens=max_tokens,
              temperature=0.1, extra_headers={"x-session-affinity": aff})
    if tools is not None:
        kw["tools"] = tools
    if extra_body is not None:
        kw["extra_body"] = extra_body
    t0 = time.time()
    try:
        r = client.chat.completions.create(**kw)
        print(f"  {tag:42s} -> served={r.model}  {time.time()-t0:.1f}s  content={str(r.choices[0].message.content)[:30]!r}")
    except Exception as e:
        print(f"  {tag:42s} -> ERROR {type(e).__name__}: {str(e)[:200]}")
    finally:
        client.close()


CTK = {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}

def main():
    print(f"({datetime.now()}) SDK bisection, affinity variants")
    probe("S1 short, aff 'dspy-optim'", SHORT, "dspy-optim")
    probe("S2 short, aff lfm static", SHORT, "dspy-optim-lfm2.5-2.6b")
    probe("S3 short, aff lfm ts", SHORT, "dspy-optim-lfm2.5-2.6b-20260822-003000")
    probe("S4 short + extra_body CTK only", SHORT, "dspy-optim", extra_body=dict(CTK))
    probe("S5 short + top_k/repeat only", SHORT, "dspy-optim", extra_body={"top_k": 50, "repeat_penalty": 1.1})
    probe("S6 short + full extra_body", SHORT, "dspy-optim",
          extra_body={"top_k": 50, "repeat_penalty": 1.1, **CTK})
    probe("S7 short + tools", SHORT, "dspy-optim", tools=frozen.get("tools"))
    probe("S8 full prefix, aff dspy-optim", prefix, "dspy-optim")
    probe("S9 full prefix + tools + eb", prefix, "dspy-optim", tools=frozen.get("tools"),
          extra_body={"top_k": 50, "repeat_penalty": 1.1, **CTK})
    probe("S10 full prefix + tools + eb, lfm ts aff", prefix, "dspy-optim-lfm2.5-2.6b-20260822-003001",
          tools=frozen.get("tools"), extra_body={"top_k": 50, "repeat_penalty": 1.1, **CTK})


if __name__ == "__main__":
    main()
