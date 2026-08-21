#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Structural verification of build_case_messages (no model calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CASES, build_case_messages, seed_pair

pair = seed_pair()
for cid in ("empty", "full_to_lite"):
    case = next(c for c in CASES if c["id"] == cid)
    msgs = build_case_messages(pair, case)
    roles = [m.get("role") for m in msgs]
    n_sys = sum(1 for m in msgs if m.get("role") == "system")
    print(f"{cid}: total={len(msgs)} system_count={n_sys}")
    print("   roles:", roles)
    last = msgs[-1]
    print("   last is user tzip block:",
          last.get("role") == "user" and '<skill name="tzip"' in last["content"][0]["text"])
    print("   tzip arg in last msg:", repr(last["content"][0]["text"].split("</skill>")[-1]))
    assert n_sys == 1, "system message duplicated!"
