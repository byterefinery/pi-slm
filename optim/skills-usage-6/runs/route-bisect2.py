#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "orjson"]
# ///

"""Routing bisection: does the proxy key on request BYTES (body serialization)
or on the HTTP client/transport? Same affinity, same logical body; vary
serialization (json.dumps vs orjson) and transport (urllib vs httpx)."""

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
AFF = "dspy-optim-lfm2.5-2.6b-bisect-2"


def do(tag, url_body_bytes, headers, use_httpx=False):
    t0 = time.time()
    try:
        if use_httpx:
            import httpx
            with httpx.Client(timeout=600) as c:
                r = c.post(base.rstrip("/") + "/chat/completions",
                           content=url_body_bytes, headers=headers)
            d = orjson.loads(r.content)
        else:
            req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                         data=url_body_bytes, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode())
        served = d.get("model")
        content = str(d["choices"][0]["message"].get("content"))[:40]
        print(f"  {tag:52s} -> served={served}  lat={time.time()-t0:.1f}s  content={content!r}")
    except Exception as e:
        print(f"  {tag:52s} -> ERROR {type(e).__name__}: {str(e)[:150]}")


def main():
    hdr = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "x-session-affinity": AFF,
    }
    json_bytes = json.dumps(BODY).encode()
    orj_bytes = orjson.dumps(BODY)
    print(f"body bytes: json.dumps={len(json_bytes)} orjson={len(orj_bytes)}")
    print(f"({datetime.now()}) affinity={AFF}")
    do("A urllib + json.dumps bytes (control)", json_bytes, hdr)
    do("B urllib + orjson bytes", orj_bytes, hdr)
    do("C httpx  + json.dumps bytes, minimal headers", json_bytes, hdr, use_httpx=True)
    do("D httpx  + orjson bytes, minimal headers", orj_bytes, hdr, use_httpx=True)


if __name__ == "__main__":
    main()
