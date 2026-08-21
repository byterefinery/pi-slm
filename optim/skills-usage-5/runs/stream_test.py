#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Streaming isolation on case 'full' (captured pi request, seed pair):
  S1: stream=False            (control)
  S2: stream=True             (pi streams)
  S3: stream=True + BEST pair spliced (actual pi test used best pair)
Serial.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_case_messages, content_ok, find_pair_index, get_creds
import openai

HERE = Path(__file__).resolve().parent
base, key = get_creds()
client = openai.OpenAI(base_url=base, api_key=key,
                       default_headers={"x-session-affinity": "dspy-optim"}, timeout=600)

cap = json.loads((HERE / "proxy-captures" / "proxy-req-1.json").read_text())
best = json.loads((HERE / "best_candidate.json").read_text())
EXPECTED = "tzip full activated"

# splice best pair into a pi-regime message list
faithful = build_case_messages(best, {"id": "full", "arg": "full", "expected": EXPECTED})
pi_best = json.loads(json.dumps(cap["messages"]))
i = find_pair_index(pi_best)
pi_best[i] = {"role": "user", "content": [{"type": "text", "text": best["user_question"]}]}
pi_best[i + 1] = {"role": "assistant", "content": best["assistant_content"]}
# strip history reasoning (pi regime)
for m in pi_best:
    if m.get("role") == "assistant":
        m.pop("reasoning_content", None)


def run(name, messages, stream):
    resp = client.chat.completions.create(
        model="LiquidAI/LFM2.5-2.6B", messages=messages, temperature=0.1, max_tokens=49152,
        extra_body={"top_k": 50, "repeat_penalty": 1.1, "reasoning_effort": "high"},
        stream=stream,
    )
    if not stream:
        m = resp.choices[0].message
        return m.content or ""
    content, reasoning = [], []
    for chunk in resp:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d.content:
            content.append(d.content)
        rc = getattr(d, "reasoning_content", None)
        if rc:
            reasoning.append(rc)
    text = "".join(content)
    print(f"  (reasoning streamed: {len(''.join(reasoning))} chars)")
    return text


for name, msgs, stream in [
    ("S1_nostream_seed", cap["messages"], False),
    ("S2_stream_seed", cap["messages"], True),
    ("S3_stream_best", pi_best, True),
]:
    text = run(name, msgs, stream)
    print(f"\n=== {name} === ok={content_ok(text, EXPECTED)}")
    print("content:", repr(text[:140]))
