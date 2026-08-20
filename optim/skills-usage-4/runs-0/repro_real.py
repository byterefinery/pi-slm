# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai>=2.9.0",
# ]
# ///
"""
repro_real.py — Reconstruct the EXACT wire context of the real pi session
(01a01ee9, the one where the student answered 'tzip lite activated' to a
tzip `full` invocation and its reasoning leaked example.sh), verify the
reconstruction against the session's reported input token count (5579), and
re-run the failing case N times to measure the failure rate.

Context source of truth: the pi session log itself (every message the model
saw), plus the system prompt reconstructed from the base file with the
real session's 5-skill set (the real session's available-skills YAML lists
git, skman, webfetch, websearch, tzip — not the file's 7).

Usage: uv run --script repro_real.py [N]
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
SESSION_LOG = Path("/home/mtasic/projects-b/pi-slm/01a01ee9-65fa-7a0e-b5f4-89fe8bc1ea24.jsonl")
BASE_FILE = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"
TZIP_FILE = HERE / "skill-example-tzip-LiquidAI-LFM2.5-2.6B.json"  # carries the same tools list
EXPECTED_INPUT_TOKENS = 5579  # usage.input of the real session's tzip reply
REPRO_DIR = HERE / "repro-real"

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

REPRO_DIR.mkdir(exist_ok=True)
LOG_FILE = REPRO_DIR / "repro-real.log"
_log_fh = open(LOG_FILE, "w")
_log_fh.write(f"=== repro-real run started {datetime.now().isoformat(timespec='seconds')} ===\n")
_log_fh.flush()


def say(text: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


# ---------------------------------------------------------------------------
# model config (same provider block the rest of the project uses)
# ---------------------------------------------------------------------------

def load_prov():
    raw = MODELS_JSON.read_text()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)  # models.json allows trailing commas
    cfg = json.loads(raw)
    prov = cfg["providers"]["llamacpp"]
    model = next(m for m in prov["models"] if m["id"] == "LiquidAI/LFM2.5-2.6B")
    return prov, model


def build_context():
    """Reconstruct the real pi session's wire context.

    Returns (system_prompt, ctx_dialogue, tzip_user_text, client, extra_body, temp).
    Verified against the real session's usage.input (5579, within ~40 tokens of
    transcription slack in the pi template sections).
    """
    prov, model_cfg = load_prov()
    client = OpenAI(base_url=prov["baseUrl"], api_key=prov["apiKey"])
    sp = model_cfg.get("samplingParams", {})
    extra_body = {
        **{k: sp[k] for k in ("top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty") if k in sp},
        # Real session: no preserve_thinking -> the LFM2.5 template strips
        # all historical  blocks (keep_thinking=false before the last user
        # message); enable_thinking still on so the model thinks now.
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
    }
    temp = sp.get("temperature", 0.1)

    # ------------------------------------------------------------------
    # 1) system prompt: the FULL pi buildSystemPrompt2 output for the real
    #    session (tools list + guidelines + pi docs + project context +
    #    5 skills + cwd), reconstructed and verified by token count.
    # ------------------------------------------------------------------
    system_prompt = (HERE / "real-system-prompt-full.txt").read_text()
    say(f"System prompt (full pi build, 5 skills + project context): {len(system_prompt)} chars")

    # ------------------------------------------------------------------
    # 2) dialogue: parse the session log into OpenAI wire messages
    # ------------------------------------------------------------------
    lines = SESSION_LOG.read_text().splitlines()
    msgs: list[dict] = []
    tzip_user_text: str | None = None
    for ln in lines:
        r = json.loads(ln)
        if r.get("type") == "custom_message":
            content = r.get("content")
            msgs.append({"role": "user", "content": content if isinstance(content, str) else json.dumps(content)})
        elif r.get("type") == "message":
            m = r["message"]
            role = m.get("role")
            if role == "assistant":
                thinking, text, tool_calls = [], [], []
                for b in m.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "thinking":
                        thinking.append(b.get("thinking", ""))
                    elif b.get("type") == "text" and b.get("text"):
                        text.append(b["text"])
                    elif b.get("type") == "toolCall":
                        tool_calls.append(
                            {"id": b.get("id") or m.get("toolCallId"), "type": "function",
                             "function": {"name": b["name"], "arguments": b["arguments"]}}
                        )
                msg = {"role": "assistant", "content": "\n".join(text) if text else None}
                if thinking:
                    msg["reasoning_content"] = "\n".join(thinking)
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                msgs.append(msg)
            elif role == "toolResult":
                msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId"),
                             "content": "".join(b.get("text", "") for b in m.get("content") or [] if isinstance(b, dict))})
            elif role == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))
                msgs.append({"role": "user", "content": text})
                if text.startswith('<skill name="tzip"'):
                    tzip_user_text = text

    assert tzip_user_text, "tzip user message not found in session log"
    ctx_dialogue = list(msgs[:-1])  # everything up to (not incl.) the tzip call
    say(f"Parsed {len(ctx_dialogue)} context messages + tzip user message ({len(tzip_user_text)} chars, tail {tzip_user_text[-20:]!r})")

    # The log ends with [user tzip 'full'] [assistant reply]: the real context
    # at the moment of the tzip call contains NEITHER — drop both.
    assert msgs[-1].get("role") == "assistant", "expected the session's tzip reply last"
    assert "tzip" in (msgs[-2].get("content") or ""), "expected the tzip user message second-to-last"
    return system_prompt, msgs[:-2], tzip_user_text, client, extra_body, temp


def verify_tokens(client, system_prompt, ctx_dialogue, tzip_user_text, extra_body, temp):
    # ------------------------------------------------------------------
    # verify the reconstruction: input token count must match the
    # real session's usage.input (5579)
    # ------------------------------------------------------------------
    probe = client.chat.completions.create(
        model="LiquidAI/LFM2.5-2.6B",
        messages=[{"role": "system", "content": system_prompt}, *ctx_dialogue,
                  {"role": "user", "content": tzip_user_text}],
        temperature=temp, max_tokens=1,
        extra_headers={"x-session-affinity": "repro-real"}, extra_body=extra_body,
    )
    used = getattr(probe.usage, "prompt_tokens", None) if probe.usage else None
    say(f"Token check: reconstructed context input tokens = {used} (real session reported {EXPECTED_INPUT_TOKENS}) "
        f"{'MATCH' if used == EXPECTED_INPUT_TOKENS else 'CLOSE/MISMATCH — context differs slightly'}")
    return used


def main() -> None:
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    system_prompt, ctx_dialogue, tzip_user_text, client, extra_body, temp = build_context()
    verify_tokens(client, system_prompt, ctx_dialogue, tzip_user_text, extra_body, temp)

    # ------------------------------------------------------------------
    # replay the failing case N times
    # ------------------------------------------------------------------
    say(f"\nReplaying the real failure case {n_runs}x: tzip `full` after the full base dialogue.")
    n_ok = n_bad = 0
    leaks = 0
    for k in range(n_runs):
        resp = client.chat.completions.create(
            model="LiquidAI/LFM2.5-2.6B",
            messages=[{"role": "system", "content": system_prompt}, *ctx_dialogue,
                      {"role": "user", "content": tzip_user_text}],
            temperature=temp, max_tokens=1024,
            extra_headers={"x-session-affinity": "repro-real"}, extra_body=extra_body,
        )
        msg = resp.choices[0].message
        reply = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None) or ""
        ok = reply == "tzip full activated"
        n_ok += ok
        n_bad += (not ok)
        leaked = "example" in reasoning.lower() or "Hi" in reasoning
        leaks += leaked
        tag = "OK  " if ok else "FAIL"
        say(f"  [{tag}] run {k + 1:2d}: {reply!r} | example-leak-in-reasoning: {leaked}")
        if not ok:
            say(f"           reasoning: {reasoning[:700]!r}")
        time.sleep(0.2)

    say(f"\nTOTAL: {n_ok} correct ('tzip full activated') / {n_bad} wrong ({n_runs} runs); "
        f"{leaks} runs with example-skill content in reasoning.")
    if n_bad:
        say("=> REAL-SESSION FAILURE REPRODUCED under the reconstructed context.")
    else:
        say("=> Not reproduced this run (failure is stochastic; increase N).")


if __name__ == "__main__":
    main()
