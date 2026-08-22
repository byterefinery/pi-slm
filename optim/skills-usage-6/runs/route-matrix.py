#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson"]
# ///

"""Which body BYTES route to LFM? Test stdlib-json variants vs orjson, and
whether affinity still matters per byte-string."""

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
URL = base.rstrip("/") + "/chat/completions"


def send(body_bytes, aff, tag):
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "x-session-affinity": aff}
    t0 = time.time()
    req = urllib.request.Request(URL, data=body_bytes, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
        print(f"  {tag:58s} aff={aff[-18:]:18s} -> served={d.get('model')}  lat={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  {tag:58s} aff={aff[-18:]:18s} -> ERROR {type(e).__name__}: {str(e)[:120]}")


def main():
    variants = {
        "json.dumps default            ": json.dumps(BODY).encode(),
        "json.dumps compact separators ": json.dumps(BODY, separators=(",", ":")).encode(),
        "json.dumps ensure_ascii=False ": json.dumps(BODY, ensure_ascii=False).encode(),
        "orjson                        ": orjson.dumps(BODY),
        "orjson OPT_SORT_KEYS          ": orjson.dumps(BODY, option=orjson.OPT_SORT_KEYS),
    }
    for k, v in variants.items():
        print(f"{k} ({len(v)} bytes)")
    print(f"({datetime.now()})")
    A = "dspy-optim-lfm2.5-2.6b-matrix-1"
    B = "dspy-optim-lfm2.5-2.6b-matrix-2"
    for tag, b in variants.items():
        send(b, A, tag + " aff1")
    print()
    # cross: json-default bytes and orjson bytes with the OTHER affinity
    send(variants["json.dumps default            "], B, "json-default")
    send(variants["orjson                        "], B, "orjson        ")


if __name__ == "__main__":
    main()
