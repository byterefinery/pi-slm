#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["gepa[full]"]
# ///
import inspect
from gepa.optimize_anything import GEPAConfig
print("GEPAConfig fields:")
for name, p in inspect.signature(GEPAConfig).parameters.items():
    print(f"  {name}: {str(p.default)[:100]}")
import gepa, os
root = os.path.dirname(gepa.__file__)
import subprocess
out = subprocess.run(["grep", "-rn", "resume", root, "--include=*.py", "-l"], capture_output=True, text=True)
print("\nfiles mentioning resume:", out.stdout.strip())
