#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai", "httpx", "orjson"]
# ///

"""Capture the EXACT request (headers + body bytes) the openai SDK sends,
compare against a urllib control, then replay the SDK's exact bytes over
urllib to separate body vs header causality. Never prints API key/URL."""

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
AFF = "dspy-optim-lfm2.5-2.6b-capture-1"
URL = base.rstrip("/") + "/chat/completions"
OUT = Path(__file__).resolve().parent / "captured-sdk-request.json"


class CaptureTransport(httpx.BaseTransport):
    def __init__(self):
        self._real = httpx.HTTPTransport()
        self.captured = None

    def handle_request(self, request):
        self.captured = {
            "method": request.method,
            "path": request.url.raw_path.decode(),
            "headers": {k: (v if "auth" not in k.lower() else "<redacted>") for k, v in request.headers.multi_items()},
            "body": orjson.loads(request.content) if request.content else None,
            "body_len": len(request.content or b""),
        }
        return self._real.handle_request(request)

    def close(self):
        self._real.close()


def urllib_send(body_bytes, headers, tag):
    t0 = time.time()
    req = urllib.request.Request(URL, data=body_bytes, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
        print(f"  {tag:56s} -> served={d.get('model')}  lat={time.time()-t0:.1f}s")
        return d.get("model")
    except Exception as e:
        print(f"  {tag:56s} -> ERROR {type(e).__name__}: {str(e)[:150]}")
        return None


def main():
    from openai import OpenAI
    transport = CaptureTransport()
    client = OpenAI(base_url=base, api_key=key, timeout=600,
                    http_client=httpx.Client(transport=transport))
    t0 = time.time()
    r = client.chat.completions.create(
        model=BODY["model"], messages=BODY["messages"], stream=False,
        max_tokens=BODY["max_tokens"], temperature=BODY["temperature"],
        extra_body={"top_k": BODY["top_k"], "repeat_penalty": BODY["repeat_penalty"],
                    "chat_template_kwargs": BODY["chat_template_kwargs"]},
        tools=BODY["tools"],
        extra_headers={"x-session-affinity": AFF},
    )
    sdk_served = r.model
    cap = transport.captured
    OUT.write_text(json.dumps(cap, indent=2))
    sdk_body = cap["body"]
    print(f"openai SDK sent: served={sdk_served} lat={time.time()-t0:.1f}s (captured -> {OUT.name})")
    print("\nSDK request headers:")
    for k, v in cap["headers"].items():
        print(f"  {k}: {str(v)[:100]}")

    # body diff: logical fields in SDK body vs my BODY
    extra = {k: v for k, v in sdk_body.items() if k not in BODY}
    missing = [k for k in BODY if k not in sdk_body]
    diff = {k: (BODY[k], sdk_body[k]) for k in BODY if k in sdk_body and BODY[k] != sdk_body[k]}
    print(f"\nbody fields extra in SDK: {list(extra)}  missing: {missing}  differing: {list(diff)}")
    for k in extra:
        print(f"  extra {k} = {str(extra[k])[:120]}")
    for k in diff:
        print(f"  diff  {k}: mine={str(BODY[k])[:80]} | sdk={str(diff[k][1])[:80]}")

    print("\n== replay over urllib ==")
    sdk_bytes = orjson.dumps(sdk_body)
    hdr_urllib = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                  "x-session-affinity": AFF}
    urllib_send(json.dumps(BODY).encode(), hdr_urllib, "control: urllib body + urllib headers")
    urllib_send(sdk_bytes, hdr_urllib, "SDK body bytes + urllib headers")
    sdk_hdr = {k: v for k, v in cap["headers"].items()}
    if "auth" in str(sdk_hdr).lower():
        pass
    sdk_hdr = {k: (v if "auth" not in k.lower() else f"Bearer {key}") for k, v in sdk_hdr.items()}
    urllib_send(sdk_bytes, sdk_hdr, "SDK body bytes + SDK headers")


if __name__ == "__main__":
    main()
