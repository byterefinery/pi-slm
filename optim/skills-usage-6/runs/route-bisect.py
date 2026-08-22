#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Head-to-head routing bisect: identical body + identical affinity value,
sent via urllib and via the openai SDK. Logs which client the proxy
routes to the LFM worker and which to Qwen. Never prints API key/URL."""

import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

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


def urllib_probe(aff: str, extra_headers: dict | None = None, body: dict | None = None):
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "x-session-affinity": aff}
    if extra_headers:
        h.update(extra_headers)
    b = body if body is not None else BODY
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(b).encode(), headers=h)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
            print(f"  urllib  aff={aff} lat={time.time()-t0:.1f}s -> served={d.get('model')} "
                  f"content={str(d['choices'][0]['message'].get('content'))[:40]!r}")
    except Exception as e:
        print(f"  urllib  aff={aff}: ERROR {type(e).__name__}: {str(e)[:150]}")


def openai_probe(aff: str, default_headers: dict | None = None, extra: dict | None = None):
    from openai import OpenAI
    kw = {"base_url": base, "api_key": key, "timeout": 600}
    if default_headers:
        kw["default_headers"] = default_headers
    client = OpenAI(**kw)
    params = dict(
        model=BODY["model"], messages=BODY["messages"], stream=False,
        max_tokens=BODY["max_tokens"], temperature=BODY["temperature"],
        extra_body={
            "top_k": BODY["top_k"],
            "repeat_penalty": BODY["repeat_penalty"],
            "chat_template_kwargs": BODY["chat_template_kwargs"],
        },
        tools=BODY["tools"],
        extra_headers={"x-session-affinity": aff},
    )
    if extra:
        params.update(extra)
    t0 = time.time()
    try:
        r = client.chat.completions.create(**params)
        print(f"  openai  aff={aff} lat={time.time()-t0:.1f}s -> served={r.model} "
              f"content={str(r.choices[0].message.content)[:40]!r}")
    except Exception as e:
        print(f"  openai  aff={aff}: ERROR {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    import sys
    V = sys.argv[1] if len(sys.argv) > 1 else "dspy-optim-lfm2.5-2.6b-bisect-1"
    print(f"== HEAD-TO-HEAD, identical body + affinity {V} == ({datetime.now()})")
    urllib_probe(V)
    openai_probe(V)
    print()
    print("== repeat ==")
    urllib_probe(V)
    openai_probe(V)
    print()
    print("== header bisection (urllib, adding openai-SDK-style headers) ==")
    urllib_probe(V, {"User-Agent": "OpenAI/Python 2.32.0"})
    urllib_probe(V, {"User-Agent": "OpenAI/Python 2.32.0",
                     "x-stainless-lang": "python",
                     "x-stainless-package-version": "2.32.0",
                     "x-stainless-os": "Linux",
                     "x-stainless-arch": "x64",
                     "x-stainless-runtime": "CPython",
                     "x-stainless-runtime-version": "3.14.0",
                     "x-stainless-async": "false",
                     "X-Stainless-Retry-Count": "0"})
    print()
    print("== openai SDK with urllib-style User-Agent override ==")
    openai_probe(V, default_headers={"User-Agent": "Python-urllib/3.14"})
