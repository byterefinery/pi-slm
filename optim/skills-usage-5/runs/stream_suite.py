#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Streaming hypothesis: full 12-case suite (deployment regime messages,
chat_template_kwargs{enable_thinking,preserve_thinking} + history CoT) but
stream=True, as pi (and the captured deployment) do. Serial.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, CASES, build_case_messages, content_ok, get_creds, judge
import openai

HERE = Path(__file__).resolve().parent
base, key = get_creds()
client = openai.OpenAI(base_url=base, api_key=key,
                       default_headers={"x-session-affinity": "dspy-optim"}, timeout=600)
pair = json.loads((HERE / "best_candidate.json").read_text())
api = Api()
results = []
for case in CASES:  # strictly serial
    msgs = build_case_messages(pair, case)
    resp = client.chat.completions.create(
        model="LiquidAI/LFM2.5-2.6B", messages=msgs, temperature=0.1, max_tokens=49152,
        extra_body={"top_k": 50, "repeat_penalty": 1.1,
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
        stream=True,
    )
    content, reasoning = [], []
    for chunk in resp:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d.content:
            content.append(d.content)
        rc = getattr(d, "reasoning_content", None)
        if rc:
            reasoning.append(rc)
    text = "".join(content)
    ok = content_ok(text, case["expected"])
    j = judge(api, pair, case, text, "".join(reasoning), use_cache=False)
    results.append({"id": case["id"], "ok": ok, "leak": j["leak"], "content": text[:150],
                    "reasoning": "".join(reasoning)[:200]})
    print(f"{case['id']}: ok={ok} leak={j['leak']} {text[:60]!r}", flush=True)
n = len(results)
acc = sum(r["ok"] for r in results) / n
leak = sum(r["leak"] for r in results) / n
(HERE / "stream_suite.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nSTREAM SUITE acc={acc * 100:.1f}% leak={leak * 100:.1f}%")
