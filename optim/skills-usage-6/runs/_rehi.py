#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["orjson"]
# ///
import importlib.util, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mkexp", HERE / "runs" / "make-expected.py")
mk = importlib.util.module_from_spec(spec)
sys.argv = ["make-expected"]
spec.loader.exec_module(mk)

base = mk.check_example
def strict(case_id, kind, chain):
    probs = base(case_id, kind, chain)
    if kind == "script":
        finals = [m for m in chain if m["role"] == "assistant" and m.get("content")]
        if finals and len(finals[-1]["content"]) > 120:
            probs.append(f"verbose final ({len(finals[-1]['content'])} chars)")
    return probs
mk.check_example = strict
rec = mk.example_case("example-hi", "Hi", "script", max_attempts=10)
out = mk.OUT_DIR / "example-hi.json"
out.write_text(json.dumps(rec, indent=2))
final = [m for m in rec["assistant"] if m["role"] == "assistant" and m.get("content")][-1]["content"]
print("WROTE", out.name)
print("final:", repr(final))
print("generated_by:", rec["generated_by"])
