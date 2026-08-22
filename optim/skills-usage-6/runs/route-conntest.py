#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson", "httpx"]
# ///

"""Test connection-level route pinning:
A) ONE persistent httpx connection, 6 calls (varying affinity + user salt)
B) SIX fresh httpx connections, 6 calls
If (A) sticks to one worker and (B) varies, the proxy pins per TCP connection."""

import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import httpx
import orjson

HERE = Path(__file__).resolve().parent.parent
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
TRANSCRIPT = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"

raw = MODELS_JSON.read_text()
cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
p = cfg["providers"]["llamacpp"]
base, key = p["baseUrl"], p["apiKey"]
frozen = json.loads(TRANSCRIPT.read_text())
BODY = {
    "model": "LiquidAI/LFM2.5-2.6B",
    "messages": frozen["messages"][:12],
    "max_tokens": 65536,
    "temperature": 0.1,
    "stream": False,
    "top_k": 50,
    "repeat_penalty": 1.1,
    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    "tools": frozen.get("tools"),
}
URL = base.rstrip("/") + "/chat/completions"
ORJ = orjson.dumps(BODY)


def one_call(client, i, aff_suffix):
    h = {
        "content-type": "application/json",
        "Authorization": f"Bearer {key}",
        "x-session-affinity": f"dspy-optim-lfm2.5-2.6b-conn-{aff_suffix}",
    }
    b = dict(BODY)
    b["user"] = f"conn-test-{aff_suffix}-{i}"
    t0 = time.time()
    r = client.post(URL, content=orjson.dumps(b), headers=h)
    d = orjson.loads(r.content)
    print(f"    call {i}: aff=...{aff_suffix}-{i} -> served={d.get('model')} lat={time.time()-t0:.1f}s")
    return d.get("model")


def main():
    print(f"({datetime.now()})")
    print("A) ONE persistent connection (keep-alive), 6 calls, varying affinity+user:")
    with httpx.Client(timeout=600) as c:
        for i in range(1, 7):
            one_call(c, i, "persist")
    print("B) SIX fresh connections, 6 calls, varying affinity+user:")
    for i in range(1, 7):
        with httpx.Client(timeout=600) as c:
            one_call(c, i, "fresh")
    print("C) ONE persistent connection, 4 calls, SAME affinity+user (pure stickiness check):")
    with httpx.Client(timeout=600) as c:
        for i in range(1, 5):
            one_call(c, i, "stick")


if __name__ == "__main__":
    main()
