"""Preflight: run seed baseline over all 9 cases (imported from train.py)."""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("train", "train.py")
tr = importlib.util.module_from_spec(spec)
sys.argv = ["train.py"]
spec.loader.exec_module(tr)

frozen = json.loads(tr.TRANSCRIPT_JSON.read_text())
fz = tr.make_frozen(frozen)
models = tr.load_models()
for m in models.get("models", []):
    if m.get("id") == tr.TEACHER_MODEL:
        models["teacher_sampling"] = {
            k: v for k, v in m.get("samplingParams", {}).items()
            if v is not None and k in ("temperature", "top_k", "min_p", "presence_penalty", "repeat_penalty")
        }
client = tr.make_client(models)
tr.verify_routing(client)  # abort unless the pinned affinity really reaches both workers
seed = {
    "user_content": frozen["messages"][5]["content"][0]["text"],
    "assistant_content": frozen["messages"][6]["content"],
    "reasoning_content": frozen["messages"][6]["reasoning_content"],
}
for case in fz["cases"]:
    res = tr.run_student_case(client, seed, case, fz)
    s = tr.score_case(case, res)
    print(f"{case['id']:16s} score={s:.2f} reply={res['content']!r} steps={len(res['tool_steps'])} finish={res['finish']}")
    for st in res["tool_steps"]:
        print("     step:", st["tool"], json.dumps(st["args"])[:100], "err=", st["error"])
    if s < 1.0:
        print("     reasoning:", res["reasoning"][:250])
