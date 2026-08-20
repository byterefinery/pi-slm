# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=2.9.0"]
# ///
"""Hard cases: a tzip invocation IMMEDIATELY after a completed example-skill
invocation (same context shape as the user's live failure, where the previous
skill's argument/rules bled into the tzip answer)."""
import json, sys, time
from datetime import datetime
from pathlib import Path
import repro_real

def say(t):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {t}", flush=True)

def main():
    system_prompt, ctx, tzip_user_text, client, extra_body, temp = repro_real.build_context()
    block_end = tzip_user_text.rfind("\n\n")
    tzip_block = tzip_user_text[:block_end]
    # the example block from the real context (the plain one-shot user message)
    ex_block = next(m["content"] for m in ctx if m.get("role") == "user" and m["content"].startswith('<skill name="example"'))
    ex_answer_plain = "This is an example skill."

    cases = [
        # (name, extra setup [(user_text, assistant_reply)], tzip arg, expected)
        ("H0 tzip 'full' alone (user's live case)", [], "full", "tzip full activated"),
        ("H1 example('full') then tzip 'full'", [(ex_block + "\n\nfull", "example.sh output:\n\n```\nThis is example.sh output.\n```")], "full", "tzip full activated"),
        ("H2 example('on') then tzip 'on'", [(ex_block + "\n\non", "example.sh output:\n\n```\nThis is example.sh output.\n```")], "on", "tzip lite activated"),
        ("H3 example('off') then tzip 'full'", [(ex_block + "\n\noff", "example.sh output:\n\n```\nThis is example.sh output.\n```")], "full", "tzip full activated"),
        ("H4 example('full') then tzip 'lite'", [(ex_block + "\n\nfull", "example.sh output:\n\n```\nThis is example.sh output.\n```")], "lite", "tzip lite activated"),
    ]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for name, setup, arg, expected in cases:
        n_ok = n_leak = 0
        fails = []
        for k in range(n):
            ctxm = [{"role": "system", "content": system_prompt}, *ctx]
            for u, a in setup:
                ctxm.append({"role": "user", "content": u})
                ctxm.append({"role": "assistant", "content": a})
            ctxm.append({"role": "user", "content": tzip_block + "\n\n" + arg})
            resp = client.chat.completions.create(model="LiquidAI/LFM2.5-2.6B", messages=ctxm,
                temperature=temp, max_tokens=512, extra_headers={"x-session-affinity": "hardcases"}, extra_body=extra_body)
            m = resp.choices[0].message
            reply = (m.content or "").strip()
            reasoning = getattr(m, "reasoning_content", None) or ""
            ok = reply == expected
            leak = "example" in reasoning.lower()
            n_ok += ok; n_leak += leak
            if not ok:
                fails.append((k + 1, reply, reasoning[:300]))
            time.sleep(0.15)
        say(f"[{ 'PASS' if n_ok == n else f'FAIL {n - n_ok}/{n}' }] {name}: ok {n_ok}/{n}, leaks {n_leak}/{n}")
        for k, reply, snip in fails:
            say(f"    !! run {k}: {reply!r}")
            say(f"       reasoning: {snip!r}")

if __name__ == "__main__":
    main()
