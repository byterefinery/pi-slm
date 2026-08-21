#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Capture the EXACT request standard pi sends (via logging proxy) for one case,
and diff it against the faithful JSON request shape. Serial."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
from common import CASES, MODELS_JSON, TZIP_SKILL, seed_pair, tzip_block

PI_BIN = shutil.which("pi") or "pi"
BASE_SYSTEM_PROMPT = ("You are an expert coding assistant operating inside pi, a coding agent harness. "
                     "You help users by reading files, executing commands, editing code, and writing new files.")
PORT = 8899
LOGDIR = HERE / "proxy-captures"
if LOGDIR.exists():
    shutil.rmtree(LOGDIR)
LOGDIR.mkdir()

# 1. temp HOME with models.json pointed at the proxy
home = Path(tempfile.mkdtemp(prefix="pihome-"))
agentdir = home / ".pi" / "agent"
agentdir.mkdir(parents=True)
raw = re.sub(r",(\s*[}\]])", r"\1", MODELS_JSON.read_text())
models = json.loads(raw)
models["providers"]["llamacpp"]["baseUrl"] = f"http://127.0.0.1:{PORT}/v1"
(agentdir / "models.json").write_text(json.dumps(models, indent=2))

# 2. start proxy
proxy = subprocess.Popen(
    [sys.executable, str(HERE / "proxy.py"), str(MODELS_JSON), str(PORT)],
    cwd=LOGDIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
import time
time.sleep(1.5)

try:
    # 3. build session + run ONE student case through pi
    from pi_test import STUDENT_SPEC, build_session_file, run_pi
    pair = seed_pair()
    case = next(c for c in CASES if c["id"] == "full")
    model_id, thinking = STUDENT_SPEC
    cwd = Path(tempfile.mkdtemp(prefix="pictl-"))
    session = HERE / "pisessions-capture.jsonl"
    build_session_file(session, pair, case, model_id, cwd)
    env = dict(os.environ, HOME=str(home))
    cmd = [
        PI_BIN, "--model", f"llamacpp/{model_id}", "--thinking", thinking,
        "--no-tools", "--tools", "read,write,edit,bash",
        "--system-prompt", BASE_SYSTEM_PROMPT, "--no-context-files",
        "--skill", str(TZIP_SKILL), "--session", str(session),
        "--offline", "-p", tzip_block(case["arg"]),
    ]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300, env=env)
    print("pi exit:", p.returncode)
    print("content:", repr((p.stdout or "")[:300]))
finally:
    proxy.terminate()

# 4. dump the captured request(s)
caps = sorted(LOGDIR.glob("proxy-req-*.json"))
print(f"\ncaptured {len(caps)} request(s)")
if caps:
    req = json.loads(caps[0].read_text())
    (LOGDIR / "req0-pretty.json").write_text(json.dumps(req, indent=2, ensure_ascii=False))
    print("model:", req.get("model"))
    print("top-level keys:", sorted(req.keys()))
    print("stream:", req.get("stream"))
    print("temperature:", req.get("temperature"), "top_k:", req.get("top_k"), "repeat_penalty:", req.get("repeat_penalty"))
    print("reasoning_effort:", req.get("reasoning_effort"))
    print("chat_template_kwargs:", req.get("chat_template_kwargs"))
    print("n_messages:", len(req.get("messages", [])))
    for m in req.get("messages", []):
        role = m.get("role")
        has_rc = "reasoning_content" in m
        c = m.get("content")
        ct = "str" if isinstance(c, str) else ("list" if isinstance(c, list) else repr(c))
        extra = " [reasoning_content]" if has_rc else ""
        tail = ""
        if isinstance(c, str) and role == "user" and "<skill name=\"tzip\"" in c:
            tail = " tail=" + repr(c[-40:])
        print(f"  {role}: content={ct}{extra}{tail}")
