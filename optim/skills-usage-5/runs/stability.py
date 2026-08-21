#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Stability head-to-head on case 'full' (5x each, fresh, serial):
  FAITHFUL: deployment JSON format (history reasoning_content + chat_template_kwargs)
  PI-CAP:   exact captured standard-pi request (reasoning_effort, no history rc,
            history user=str, last user=list)
Which format is the student reliable on?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_case_messages, content_ok, get_creds
import openai

HERE = Path(__file__).resolve().parent
base, key = get_creds()
client = openai.OpenAI(base_url=base, api_key=key,
                       default_headers={"x-session-affinity": "dspy-optim"}, timeout=600)
pair = json.loads((HERE / "best_candidate.json").read_text())
case = {"id": "full", "arg": "full", "expected": "tzip full activated"}
EXPECTED = "tzip full activated"
cap = json.loads((HERE / "proxy-captures" / "proxy-req-1.json").read_text())

faithful_msgs = build_case_messages(pair, case)
pi_msgs = cap["messages"]  # verbatim captured (seed pair inside, but same shape)


def run(name, msgs, n=5):
    oks = []
    for i in range(n):
        resp = client.chat.completions.create(
            model="LiquidAI/LFM2.5-2.6B", messages=msgs, temperature=0.1, max_tokens=49152,
            extra_body={"top_k": 50, "repeat_penalty": 1.1,
                        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
            if name == "FAITHFUL" else
            {"top_k": 50, "repeat_penalty": 1.1, "reasoning_effort": "high"},
        )
        text = resp.choices[0].message.content or ""
        ok = content_ok(text, EXPECTED)
        oks.append(ok)
        print(f"  {name} {i + 1}/5 ok={ok} {text[:50]!r}", flush=True)
    return oks


print("FAITHFUL:")
f = run("FAITHFUL", faithful_msgs)
print("PI-CAP:")
p = run("PI-CAP", pi_msgs)
print(f"\nFAITHFUL {sum(f)}/5   PI-CAP {sum(p)}/5")
