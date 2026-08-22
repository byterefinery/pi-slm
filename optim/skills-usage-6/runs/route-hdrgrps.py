#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson", "httpx"]
# ///

"""Which SDK header triggers Qwen routing? Start from raw-httpx minimal
(LFM baseline) and add SDK header groups one at a time."""

import json
import re
import time
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
AFF = "dspy-optim-lfm2.5-2.6b-hdrgrps"
URL = base.rstrip("/") + "/chat/completions"
ORJ_B = orjson.dumps(BODY)

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


def call(extra_headers, tag):
    h = {"content-type": "application/json", "Authorization": f"Bearer {key}",
         "x-session-affinity": AFF}
    h.update(extra_headers)
    t0 = time.time()
    try:
        with httpx.Client(timeout=600) as c:
            r = c.post(URL, content=ORJ_B, headers=h)
        d = orjson.loads(r.content)
        print(f"  {tag:38s} -> served={d.get('model')}  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  {tag:38s} -> ERROR {type(e).__name__}: {str(e)[:150]}")


def main():
    print(f"({datetime.now()}) header-group bisection on raw httpx (orjson body)")
    call({}, "H0 baseline minimal")
    call({"user-agent": "OpenAI/Python 3.3.1"}, "H1 + user-agent OpenAI/Python")
    call(STAINLESS, "H2 + all x-stainless-*")
    call({"accept": "application/json"}, "H3 + accept application/json")
    call({"accept-encoding": "gzip, deflate"}, "H4 + accept-encoding gzip,deflate")
    call({"user-agent": "OpenAI/Python 3.3.1", **STAINLESS}, "H5 + UA + all stainless")
    call({"accept": "application/json", "accept-encoding": "gzip, deflate",
          "user-agent": "OpenAI/Python 3.3.1", **STAINLESS}, "H6 + full SDK set")
    print("  -- repeats of the two most informative --")
    call(STAINLESS, "H2b + all x-stainless-*")
    call({}, "H0b baseline minimal")


if __name__ == "__main__":
    main()
