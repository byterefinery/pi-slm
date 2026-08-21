#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Offline check of the pi session builder (no model calls)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi_test import build_session_file
from common import CASES, seed_pair

case = next(c for c in CASES if c["id"] == "full_to_lite")
cwd = Path(tempfile.mkdtemp(prefix="pitzip-offline-"))
s = Path(__file__).resolve().parent / "pisessions-offline.jsonl"
build_session_file(s, seed_pair(), case, "LiquidAI/LFM2.5-2.6B", cwd)
lines = s.read_text().splitlines()
print("entries:", len(lines))
for l in lines:
    e = json.loads(l)
    if e["type"] == "session":
        print("SESSION version", e["version"], "cwd", e["cwd"])
    else:
        m = e["message"]
        r = m.get("role")
        kinds = [b.get("type") for b in m.get("content", [])] if r == "assistant" else []
        c = m.get("content")
        txt = (c if isinstance(c, str) else "")[:60]
        print(f"  id={e['id']} parent={e['parentId']} role={r} kinds={kinds} "
              f"toolName={m.get('toolName', '')} txt={txt!r}")
