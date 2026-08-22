#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson"]
# ///

"""Clean 2D matrix over RAW sockets (identical transport, fresh connection per
call): body serialization (json.dumps vs orjson) x affinity value, plus
header-set variants. Prints HTTP status + raw snippet on parse failure."""

import json
import re
import socket
import ssl
import time
from datetime import datetime
from pathlib import Path

import orjson

HERE = Path(__file__).resolve().parent.parent
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
TRANSCRIPT = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"

raw = MODELS_JSON.read_text()
cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
p = cfg["providers"]["llamacpp"]
key = p["apiKey"]
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


def raw_send(hdr_items, body_bytes):
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=600) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as tls:
            out = b"POST /v1/chat/completions HTTP/1.1\r\n"
            for k, v in hdr_items:
                out += f"{k}: {v}\r\n".encode()
            out += b"\r\n" + body_bytes
            tls.sendall(out)
            buf = b""
            while b"\r\n\r\n" not in buf:
                c = tls.recv(65536)
                if not c:
                    break
                buf += c
            head, _, rest = buf.partition(b"\r\n\r\n")
            head_l = head.lower().decode("latin1")
            status = head.split(b"\r\n")[0].decode("latin1")
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
                        break
                    while len(rest) < size + 2:
                        rest += tls.recv(65536)
                    body += rest[:size]
                    rest = rest[size + 2:]
            else:
                while True:
                    c = tls.recv(65536)
                    if not c:
                        break
                    rest += c
                body = rest
    import gzip
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    try:
        d = json.loads(body)
        return d.get("model"), status
    except Exception:
        return f"UNPARSED:{status}:{body[:120]!r}", status


def main():
    affA = "dspy-optim-lfm2.5-2.6b-2d-aa"
    affB = "dspy-optim-lfm2.5-2.6b-2d-bb"
    json_b = json.dumps(BODY).encode()
    orj_b = orjson.dumps(BODY)
    orj_user = orjson.dumps({**BODY, "user": "wire-test-2"})

    def minimal(aff):
        return [("host", host), ("authorization", f"Bearer {key}"),
                ("content-type", "application/json"), ("x-session-affinity", aff),
                ("content-length", None)]

    print(f"({datetime.now()}) 2D matrix: body x affinity, minimal headers, raw sockets")
    cases = [
        ("json.dumps, affA", minimal(affA), json_b),
        ("json.dumps, affB", minimal(affB), json_b),
        ("orjson,     affA", minimal(affA), orj_b),
        ("orjson,     affB", minimal(affB), orj_b),
        ("orjson+user,affA", minimal(affA), orj_user),
        ("orjson+user,affB", minimal(affB), orj_user),
    ]
    for tag, hdr, b in cases:
        h = [(k, str(len(b)) if v is None else v) for k, v in hdr]
        t0 = time.time()
        served, status = raw_send(h, b)
        print(f"  {tag:16s} -> {served}  [{status}]  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
