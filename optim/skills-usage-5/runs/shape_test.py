#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Template-shape test on the captured pi request (case 'full'), 3x each:
  V1: verbatim (history user=str, last user=list)   [what pi sends]
  V2: last user content -> plain string
  V3: ALL user content -> plain strings
Serial. 9 student calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import content_ok, get_creds
import openai

HERE = Path(__file__).resolve().parent
base, key = get_creds()
client = openai.OpenAI(base_url=base, api_key=key,
                       default_headers={"x-session-affinity": "dspy-optim"}, timeout=600)
cap = json.loads((HERE / "proxy-captures" / "proxy-req-1.json").read_text())
EXPECTED = "tzip full activated"


def variant(kind):
    msgs = json.loads(json.dumps(cap["messages"]))
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            text = "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
            if kind == "V2_last_str" and i == len(msgs) - 1:
                m["content"] = text
            elif kind == "V3_all_str":
                m["content"] = text
    return msgs


tally = {}
for kind in ("V1_verbatim", "V2_last_str", "V3_all_str"):
    for rep in range(3):
        resp = client.chat.completions.create(
            model="LiquidAI/LFM2.5-2.6B", messages=variant(kind), temperature=0.1,
            max_tokens=49152,
            extra_body={"top_k": 50, "repeat_penalty": 1.1, "reasoning_effort": "high"},
        )
        text = resp.choices[0].message.content or ""
        ok = content_ok(text, EXPECTED)
        tally.setdefault(kind, []).append(ok)
        print(f"{kind} rep{rep + 1}: ok={ok} {text[:70]!r}", flush=True)
print()
for k, v in tally.items():
    print(f"{k}: {sum(v)}/{len(v)} correct")
