#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson"]
# ///
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mkexp", HERE / "runs" / "make-expected.py")
mk = importlib.util.module_from_spec(spec)
sys.argv = ["make-expected"]
spec.loader.exec_module(mk)
print("== smoke: tzip-full (teacher)")
gen = mk.teacher_tzip_case("full")
print("content:", repr(gen["content"]))
print("reasoning:", repr(gen["reasoning_content"][:300]))
print("== smoke: example-hello (pi)")
rec = mk.example_case("example-hello", "Hello", "read-hello")
print("user tail:", repr(rec["user"]["content"][0]["text"][-80:]))
for m in rec["assistant"]:
    if m["role"] == "assistant":
        tcs = m.get("tool_calls")
        print("assistant:", repr(str(m.get("content"))[:60]), "tool_calls:", [t["function"]["name"] for t in (tcs or [])])
        if m.get("reasoning_content"):
            print("  reasoning:", repr(m["reasoning_content"][:200]))
    else:
        print("tool:", repr(str(m.get("content"))[:80]))
