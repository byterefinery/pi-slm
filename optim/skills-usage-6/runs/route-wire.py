#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai", "httpx", "orjson"]
# ///

"""Capture the SDK's EXACT wire request (header order/case + body bytes),
replay it over a raw TCP/TLS socket (fresh connection, like urllib), then
drop headers one at a time until routing flips Qwen -> LFM."""

import json
import re
import socket
import ssl
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
host, port = "", 443
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
AFF = "dspy-optim-lfm2.5-2.6b-wire-1"


class CaptureTransport(httpx.BaseTransport):
    def __init__(self):
        self._real = httpx.HTTPTransport()
        self.request = None

    def handle_request(self, request):
        self.request = request
        return self._real.handle_request(request)


def parse_served(resp_bytes):
    try:
        d = json.loads(resp_bytes)
        return d.get("model"), str(d["choices"][0]["message"].get("content"))[:40]
    except Exception:
        return None, None


def raw_send(request_bytes):
    """Send exact bytes over a fresh TCP/TLS connection; return (served, content)."""
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=600) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as tls:
            tls.sendall(request_bytes)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = tls.recv(65536)
                if not chunk:
                    break
                buf += chunk
            head, _, rest = buf.partition(b"\r\n\r\n")
            head_l = head.lower().decode("latin1")
            m = re.search(r"content-length: (\d+)", head_l)
            if m:
                total = int(m.group(1))
                while len(rest) < total:
                    rest += tls.recv(65536)
                body = rest[:total]
            elif "transfer-encoding: chunked" in head_l:
                body = b""
                while True:
                    while b"\r\n" not in rest:
                        rest += tls.recv(65536)
                    size_line, _, rest = rest.partition(b"\r\n")
                    size = int(size_line.split(b";")[0].strip(), 16)
                    if size == 0:
                        while b"\r\n" not in rest:
                            rest += tls.recv(65536)
                        rest, _, _ = rest.partition(b"\r\n")
                        break
                    while len(rest) < size + 2:
                        rest += tls.recv(65536)
                    body += rest[:size]
                    rest = rest[size + 2:]
            else:  # read until close
                while True:
                    chunk = tls.recv(65536)
                    if not chunk:
                        break
                    rest += chunk
                body = rest
    import gzip
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return parse_served(body)


def build_wire(headers_items, body_bytes):
    out = b"POST /v1/chat/completions HTTP/1.1\r\n"
    for k, v in headers_items:
        out += f"{k}: {v}\r\n".encode()
    out += b"\r\n" + body_bytes
    return out


def main():
    from openai import OpenAI
    transport = CaptureTransport()
    client = OpenAI(base_url=f"https://{host}/v1", api_key=key, timeout=600,
                    http_client=httpx.Client(transport=transport))
    r = client.chat.completions.create(
        model=BODY["model"], messages=BODY["messages"], stream=False,
        max_tokens=BODY["max_tokens"], temperature=BODY["temperature"],
        extra_body={"top_k": BODY["top_k"], "repeat_penalty": BODY["repeat_penalty"],
                    "chat_template_kwargs": BODY["chat_template_kwargs"]},
        tools=BODY["tools"],
        extra_headers={"x-session-affinity": AFF},
        user="wire-test-1",
    )
    req = transport.request
    hdrs = req.headers.items()  # exact order + case as httpx would send
    body = req.content
    print(f"SDK call served={r.model}; captured {len(hdrs)} headers, {len(body)} body bytes")

    print("\n== W1: exact SDK wire bytes over fresh raw TLS connection ==")
    t0 = time.time()
    print("  raw-replica          ->", raw_send(build_wire(hdrs, body)), f"({time.time()-t0:.1f}s)")

    # bisection: drop groups
    def filtered(exclude_prefixes=(), exclude_exact=()):
        return [(k, v) for k, v in hdrs
                if not any(k.lower().startswith(x) for x in exclude_prefixes)
                and k.lower() not in exclude_exact]

    print("\n== header bisection (raw replay, fresh connections) ==")
    for tag, ex in [
        ("no x-stainless-*   ", ("x-stainless-", ())),
        ("no accept-encoding ", ((), "accept-encoding")),
        ("no user-agent      ", ((), "user-agent")),
        ("no accept          ", ((), "accept")),
        ("no connection      ", ((), "connection")),
        ("no x-session-aff   ", ((), "x-session-affinity")),
        ("only required 3    ", None),
    ]:
        if ex is None:
            keep = [(k, v) for k, v in hdrs if k.lower() in ("host", "authorization", "content-type", "content-length")]
        else:
            keep = filtered(*ex)
        # recompute content-length for safety
        keep = [(k, v) for k, v in keep if k.lower() != "content-length"] + [("content-length", str(len(body)))]
        t0 = time.time()
        res = raw_send(build_wire(keep, body))
        print(f"  {tag} ({len(keep)} hdrs)   -> {res[0]} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
