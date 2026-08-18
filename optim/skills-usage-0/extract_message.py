#!/usr/bin/env python3
"""Extract usageThinking() + SKILL_USAGE_EXAMPLE from src/slm.ts into a
candidate JSON ({"content": ..., "reasoning_content": ...}) for replay3.py."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "src" / "slm.ts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "current.json"

src = SRC.read_text()

m = re.search(r"function usageThinking\(\): string \{\n\treturn `(.*?)`;", src, re.S)
thinking = m.group(1).replace("\\`", "`")

m = re.search(r"const SKILL_USAGE_EXAMPLE = \[(.*?)\]\.join\(\"\\n\"\);", src, re.S)
# finditer (findall would report non-participating groups as '')
pattern = re.compile(r'(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')')
content = "\n".join(
    (mm.group(1) if mm.group(2) is None else mm.group(2))
    .replace('\\"', '"')
    .replace("\\\\", "\\")
    for mm in pattern.finditer(m.group(1))
)

out = {"role": "assistant", "content": content, "reasoning_content": thinking}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {OUT}  (content {len(content)} chars, reasoning {len(thinking)} chars)")
