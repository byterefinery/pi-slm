#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Write the GEPA-best pair back into the artifact (only the pair changes)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import load_best, write_back

write_back(load_best())
