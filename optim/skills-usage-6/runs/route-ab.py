#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai", "httpx", "orjson"]
# ///

"""Controlled A/B: in the SAME minute, SAME affinity, SAME body, interleave
openai-SDK / raw-httpx / urllib calls. Also dump the SDK's exact header list
for this run. If SDK stays Qwen while others hit LFM, it's a client effect."""

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
AFF = "dspy-optim-lfm2.5-2.6b-ab-1"
JSON_B = json.dumps(BODY).encode()
ORJ_B = orjson.dumps(BODY)
URL = base.rstrip("/") + "/chat/completions"


class CaptureTransport(httpx.BaseTransport):
    def __init__(self):
        self._real = httpx.HTTPTransport()
        self.last = None

    def handle_request(self, request):
        self.last = (dict(request.headers.items()), request.content)
        return self._real.handle_request(request)


def tag(res):
    return res if res else "ERROR"


def main():
    from openai import OpenAI
    cap = CaptureTransport()
    sdk_client = OpenAI(base_url=base, api_key=key, timeout=600,
                        http_client=httpx.Client(transport=cap))
    raw_client = httpx.Client(timeout=600)
    print(f"({datetime.now()}) A/B: 3 rounds x [SDK, raw-httpx, urllib], same affinity {AFF}")
    for rnd in range(3):
        # 1) SDK
        t0 = time.time()
        try:
            r = sdk_client.chat.completions.create(
                model=BODY["model"], messages=BODY["messages"], stream=False,
                max_tokens=BODY["max_tokens"], temperature=BODY["temperature"],
                extra_body={"top_k": BODY["top_k"], "repeat_penalty": BODY["repeat_penalty"],
                            "chat_template_kwargs": BODY["chat_template_kwargs"]},
                tools=BODY["tools"],
                extra_headers={"x-session-affinity": AFF})
            print(f"  r{rnd+1} SDK       -> served={r.model}  {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  r{rnd+1} SDK       -> ERROR {type(e).__name__}: {str(e)[:150]}")
        # 2) raw httpx
        t0 = time.time()
        try:
            r = raw_client.post(URL, content=ORJ_B,
                                headers={"content-type": "application/json",
                                         "Authorization": f"Bearer {key}",
                                         "x-session-affinity": AFF})
            d = orjson.loads(r.content)
            print(f"  r{rnd+1} raw-httpx -> served={d.get('model')}  {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  r{rnd+1} raw-httpx -> ERROR {type(e).__name__}: {str(e)[:150]}")
        # 3) urllib
        t0 = time.time()
        try:
            req = urllib.request.Request(URL, data=JSON_B,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {key}",
                                                  "x-session-affinity": AFF})
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read().decode())
            print(f"  r{rnd+1} urllib    -> served={d.get('model')}  {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  r{rnd+1} urllib    -> ERROR {type(e).__name__}: {str(e)[:150]}")
    print("\nSDK exact headers (last call):")
    if cap.last:
        for k, v in cap.last[0].items():
            print(f"  {k}: {str(v)[:90]}")
    sdk_client.close()
    raw_client.close()


if __name__ == "__main__":
    main()
