#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Full 12-case pi test (student, best pair) through the logging proxy with a
CLEAN temp HOME (no global pi state). Captures every request for analysis.
Serial. Writes runs/pi_test_cleanhome.json + train.log lines."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, CASES, MODELS_JSON, TZIP_SKILL, content_ok, judge, log

HERE = Path(__file__).resolve().parent
PI_BIN = shutil.which("pi") or "pi"
BASE_SYSTEM_PROMPT = ("You are an expert coding assistant operating inside pi, a coding agent harness. "
                     "You help users by reading files, executing commands, editing code, and writing new files.")
PORT = 8899
CAPDIR = HERE / "proxy-captures-clean"
if CAPDIR.exists():
    shutil.rmtree(CAPDIR)
CAPDIR.mkdir()

from pi_test import STUDENT_SPEC, build_session_file, run_pi

model_id, thinking = STUDENT_SPEC
pair = json.loads((HERE / "best_candidate.json").read_text())
api = Api()

home = Path(tempfile.mkdtemp(prefix="pihome-clean-"))
agentdir = home / ".pi" / "agent"
agentdir.mkdir(parents=True)
raw = re.sub(r",(\s*[}\]])", r"\1", MODELS_JSON.read_text())
models = json.loads(raw)
models["providers"]["llamacpp"]["baseUrl"] = f"http://127.0.0.1:{PORT}/v1"
(agentdir / "models.json").write_text(json.dumps(models, indent=2))

proxy = subprocess.Popen(
    [sys.executable, str(HERE / "proxy.py"), str(MODELS_JSON), str(PORT)],
    cwd=CAPDIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
time.sleep(1.5)

env = dict(os.environ, HOME=str(home))
results = []
try:
    for case in CASES:  # strictly serial
        cwd = Path(tempfile.mkdtemp(prefix="pictc-"))
        session = HERE / f"pisessions-clean-{case['id']}.jsonl"
        build_session_file(session, pair, case, model_id, cwd)
        cmd = [
            PI_BIN, "--model", f"llamacpp/{model_id}", "--thinking", thinking,
            "--no-tools", "--tools", "read,write,edit,bash",
            "--system-prompt", BASE_SYSTEM_PROMPT, "--no-context-files",
            "--skill", str(TZIP_SKILL), "--session", str(session),
            "-p", __import__("common").tzip_block(case.get("arg", "")),
        ]
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300, env=env)
        entries = [json.loads(l) for l in session.read_text().splitlines() if l.strip()]
        asst = [e for e in entries if e.get("type") == "message"
                and e.get("message", {}).get("role") == "assistant"]
        text, thinking_txt = "", ""
        if asst:
            for b in asst[-1]["message"].get("content", []):
                if b.get("type") == "text":
                    text = b.get("text", "")
                elif b.get("type") == "thinking":
                    thinking_txt = b.get("thinking", "")
        ok = content_ok(text, case["expected"])
        j = judge(api, pair, case, text, thinking_txt, use_cache=False)
        results.append({"id": case["id"], "expected": case["expected"], "actual": text,
                        "reasoning": thinking_txt, "content_ok": ok, "leak": j["leak"],
                        "leak_reason": j["reason"], "exit_code": p.returncode})
        log(f"pi-cleanhome {case['id']}: ok={ok} leak={j['leak']} exit={p.returncode} "
            f"actual={text[:60]!r} thinking_len={len(thinking_txt)}")
finally:
    proxy.terminate()

n = len(results)
acc = sum(r["content_ok"] for r in results) / n
leakr = sum(r["leak"] for r in results) / n
(HERE / "pi_test_cleanhome.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
log(f"pi-cleanhome DONE acc={acc * 100:.1f}% leak={leakr * 100:.1f}%")
