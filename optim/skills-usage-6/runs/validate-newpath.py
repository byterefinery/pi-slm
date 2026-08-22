#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "orjson"]
# ///
import importlib.util, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train", HERE / "train.py")
tr = importlib.util.module_from_spec(spec)
sys.argv = ["train.py"]
spec.loader.exec_module(tr)
models = tr.load_models()
client = tr.make_client(models)
sid = tr.verify_routing(client)
print(f"verify_routing OK (session {sid})")
frozen = json.loads(tr.TRANSCRIPT_JSON.read_text())
msgs = frozen["messages"][:12]
for i in range(3):
    t0 = time.time()
    r = tr.call_lm(client, tr.STUDENT_MODEL, msgs,
                   max_tokens=tr.STUDENT_PARAMS["max_tokens"],
                   temperature=tr.STUDENT_PARAMS["temperature"],
                   extra_body=tr.STUDENT_PARAMS["extra_body"],
                   tools=frozen.get("tools"))
    print(f"student draw {i+1}: served={r.model} content={str(r.choices[0].message.content)[:30]!r} {time.time()-t0:.1f}s")
r = tr.call_lm(client, tr.TEACHER_MODEL, [{"role": "user", "content": "Reply with the single word: pong"}],
               max_tokens=64, temperature=0.0,
               extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print(f"teacher draw: served={r.model} content={str(r.choices[0].message.content)[:30]!r}")
