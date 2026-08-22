#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Validate the salted-retry mechanism: repeated openai-SDK calls with fresh
`user` salts until one is served by the requested model (LFM for student)."""

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("train", HERE / "train.py")
tr = importlib.util.module_from_spec(spec)
sys.argv = ["train.py"]
spec.loader.exec_module(tr)

models = tr.load_models()
client = tr.make_client(models)
frozen = json.loads(tr.TRANSCRIPT_JSON.read_text())
msgs = frozen["messages"][:12]
tools = frozen.get("tools")

hits = 0
draws = 0
for i in range(12):
    salt = tr.next_salt()
    t0 = time.time()
    r = client.chat.completions.create(
        model=tr.STUDENT_MODEL, messages=msgs, stream=False,
        max_tokens=tr.STUDENT_PARAMS["max_tokens"], temperature=tr.STUDENT_PARAMS["temperature"],
        extra_body=tr.STUDENT_PARAMS["extra_body"], tools=tools,
        extra_headers=tr.session_header(tr.STUDENT_MODEL), user=salt)
    draws += 1
    served = r.model
    hits += int(served == tr.STUDENT_MODEL)
    print(f"draw {draws:2d}: salt={salt[-8:]:8s} served={served} lat={time.time()-t0:.1f}s "
          f"content={str(r.choices[0].message.content)[:30]!r}")
    if served == tr.STUDENT_MODEL:
        break
    time.sleep(3)
print(f"student: {hits}/{draws} draws served by LFM")
