#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Replay the captured pi request (case 'full') with controlled modifications:
  R0: captured pi request verbatim                 (expect FAIL like pi)
  R1: R0 + chat_template_kwargs{enable_thinking,preserve_thinking}, no reasoning_effort
  R2: R1 + history reasoning_content restored
Which flips the student to 'tzip full activated' localizes the cause. Serial.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, build_case_messages, content_ok, get_creds

HERE = Path(__file__).resolve().parent
base, key = get_creds()
import openai
client = openai.OpenAI(base_url=base, api_key=key, default_headers={"x-session-affinity": "dspy-optim"}, timeout=600)

pair = json.loads((HERE / "best_candidate.json").read_text())
case = {"id": "full", "arg": "full", "expected": "tzip full activated"}
EXPECTED = "tzip full activated"

cap = json.loads((HERE / "proxy-captures" / "proxy-req-1.json").read_text())
r0 = cap  # verbatim

r1 = json.loads(json.dumps(cap))
r1.pop("reasoning_effort", None)
r1["chat_template_kwargs"] = {"enable_thinking": True, "preserve_thinking": True}

# restore history reasoning_content (from the faithful builder) into r2
faithful = build_case_messages(pair, case)
rc_map = []  # align: faithful assistant order
fi = 0
r2 = json.loads(json.dumps(r1))
for m in r2["messages"]:
    if m.get("role") == "assistant":
        # find matching faithful assistant (by content) and copy reasoning_content
        fm = next((x for x in faithful[fi:] if x.get("role") == "assistant"), None)
        if fm is not None and fm.get("reasoning_content"):
            m["reasoning_content"] = fm["reasoning_content"]
        fi += 1

for name, req in [("R0_verbatim_pi", r0), ("R1_ctk", r1), ("R2_ctk+rc", r2)]:
    req = json.loads(json.dumps(req))
    req["stream"] = False
    extra = {}
    for k in ("top_k", "repeat_penalty", "chat_template_kwargs", "reasoning_effort",
              "top_p", "min_p", "presence_penalty", "max_tokens", "temperature"):
        if k in req:
            extra[k] = req.pop(k)
    # standard OpenAI params stay as kwargs; llama.cpp-specific go to extra_body
    std = {k: extra.pop(k) for k in ("temperature", "max_tokens") if k in extra}
    resp = client.chat.completions.create(model=req.pop("model"), extra_body=extra, **std, **req)
    msg = resp.choices[0].message
    content = msg.content or ""
    ok = content_ok(content, EXPECTED)
    print(f"\n=== {name} === ok={ok}")
    print("content:", repr(content[:120]))
