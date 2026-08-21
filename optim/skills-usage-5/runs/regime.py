#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Controlled regime experiment (best pair, 12 cases, fresh, serial).
  F = deployment/faithful: chat_template_kwargs{enable_thinking,preserve_thinking}
      + history reasoning_content present (the captured JSON request).
  P = standard-pi: reasoning_effort=high, no chat_template_kwargs,
      history reasoning_content stripped, pi-style system prompt.
Reports per-regime accuracy and CoT-leak (teacher judge).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Api, CASES, build_case_messages, content_ok, get_creds,
                    judge, log)
import openai

HERE = Path(__file__).resolve().parent
base, key = get_creds()
client = openai.OpenAI(base_url=base, api_key=key,
                       default_headers={"x-session-affinity": "dspy-optim"}, timeout=600)
pair = json.loads((HERE / "best_candidate.json").read_text())
api = Api()

PI_SYSTEM = ("You are an expert coding assistant operating inside pi, a coding agent harness. "
             "You help users by reading files, executing commands, editing code, and writing new files.")


def call(model, messages, regime):
    msgs = json.loads(json.dumps(messages))
    if regime == "P":
        # pi-style: pi system prompt, strip history reasoning, reasoning_effort, no ctk
        msgs[0] = {"role": "system", "content": PI_SYSTEM}
        for m in msgs:
            if m.get("role") == "assistant":
                m.pop("reasoning_content", None)
        extra = {"reasoning_effort": "high"}
    else:  # F faithful
        extra = {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=0.1, max_tokens=49152,
        extra_body={"top_k": 50, "repeat_penalty": 1.1, **extra},
    )
    msg = resp.choices[0].message
    return (msg.content or ""), (getattr(msg, "reasoning_content", None) or "")


def run_regime(regime):
    out = []
    for case in CASES:
        msgs = build_case_messages(pair, case)
        content, reasoning = call("LiquidAI/LFM2.5-2.6B", msgs, regime)
        ok = content_ok(content, case["expected"])
        j = judge(api, pair, case, content, reasoning, use_cache=False)
        out.append({"id": case["id"], "ok": ok, "leak": j["leak"], "content": content[:120],
                    "reasoning": reasoning[:120]})
    n = len(out)
    acc = sum(r["ok"] for r in out) / n
    leak = sum(r["leak"] for r in out) / n
    return acc, leak, out


res = {}
for regime in ("F", "P"):
    acc, leak, out = run_regime(regime)
    res[regime] = {"acc": acc, "leak": leak, "cases": out}
    log(f"regime-{regime} acc={acc * 100:.1f}% leak={leak * 100:.1f}%")
    for r in out:
        log(f"  {r['id']}: ok={r['ok']} leak={r['leak']} content={r['content'][:60]!r}")
(HERE / "regime_experiment.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("done")
