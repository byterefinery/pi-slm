#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson"]
# ///

"""Bisect the SDK's exact captured request (headers + orjson body, replayed
over urllib) by dropping header groups one at a time until routing flips
from Qwen to LFM. Never prints API key/URL."""

import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

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
AFF = "dspy-optim-lfm2.5-2.6b-hdrbisect-1"
URL = base.rstrip("/") + "/chat/completions"
ORJ = orjson.dumps(BODY)

STAINLESS = {
    "x-stainless-lang": "python",
    "x-stainless-package-version": "3.3.1",
    "x-stainless-os": "Linux",
    "x-stainless-arch": "x64",
    "x-stainless-runtime": "CPython",
    "x-stainless-runtime-version": "3.14.3",
    "x-stainless-async": "false",
    "x-stainless-retry-count": "0",
    "x-stainless-read-timeout": "600",
}
SDK_FULL = {
    "accept-encoding": "gzip, deflate",
    "connection": "keep-alive",
    "accept": "application/json",
    "user-agent": "OpenAI/Python 3.3.1",
    **STAINLESS,
}
MINIMAL = {"content-type": "application/json"}


def send(headers, tag):
    h = {**headers, "content-type": "application/json", "Authorization": f"Bearer {key}",
         "x-session-affinity": AFF}
    t0 = time.time()
    req = urllib.request.Request(URL, data=ORJ, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            import gzip
            rawb = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                rawb = gzip.decompress(rawb)
            d = json.loads(rawb.decode())
        print(f"  {tag:48s} -> served={d.get('model')}  lat={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  {tag:48s} -> ERROR {type(e).__name__}: {str(e)[:140]}")


def main():
    print(f"({datetime.now()}) affinity={AFF} body=orjson({len(ORJ)}b)")
    send({**SDK_FULL, "user-agent": "OpenAI/Python 3.3.1"}, "R0 full SDK headers (repro bad)")
    send({"accept-encoding": "gzip, deflate", "accept": "application/json",
          "user-agent": "OpenAI/Python 3.3.1"}, "R1 no x-stainless-*")
    send(STAINLESS, "R2 stainless only, no accept-encoding/UA")
    send({"accept-encoding": "gzip, deflate"}, "R3 accept-encoding only")
    send({"accept-encoding": "identity"}, "R4 accept-encoding: identity")
    send({}, "R5 minimal (control, expect LFM)")
    # repeat the two most informative once more
    print("  -- repeat --")
    send({**SDK_FULL, "user-agent": "OpenAI/Python 3.3.1"}, "R0b full SDK headers")
    send({}, "R5b minimal")


if __name__ == "__main__":
    main()
