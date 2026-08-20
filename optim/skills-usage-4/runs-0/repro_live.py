# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=2.9.0"]
# ///
"""Replay the user's EXACT live pattern: two consecutive tzip invocations.

The real session: /tzip (bare block) -> 'tzip lite activated' -> /tzip full
-> the model repeated 'tzip lite activated' (stale mode). Token-verified:
base + [block] + ['tzip lite activated'] + [block+full] = 5581 ≈ real 5579.

Usage: uv run --script repro_live.py [N]
"""
import sys, time
from datetime import datetime
import repro_real

def say(t):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {t}", flush=True)

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    sp, ctx, tzip_user, client, body, temp = repro_real.build_context()
    block = tzip_user[:tzip_user.rfind("\n\n")]
    say("Token verify: base + [block] + ['tzip lite activated'] + [block+full]")
    r = client.chat.completions.create(model="LiquidAI/LFM2.5-2.6B",
        messages=[{"role": "system", "content": sp}, *ctx,
                  {"role": "user", "content": block},
                  {"role": "assistant", "content": "tzip lite activated"},
                  {"role": "user", "content": tzip_user}],
        temperature=temp, max_tokens=1, extra_headers={"x-session-affinity": "live"}, extra_body=body)
    say(f"  input tokens = {r.usage.prompt_tokens} (real session: 5579)")

    cases = [
        # (name, first-arg (None = bare block), first-reply, second-arg, expected)
        ("R1: /tzip -> /tzip full          (THE user's live failure)", None, "tzip lite activated", "full", "tzip full activated"),
        ("R2: /tzip -> /tzip lite          (repeat same mode)",        None, "tzip lite activated", "lite", "tzip lite activated"),
        ("R3: /tzip full -> /tzip lite     (user's 'starts with full')", "full", "tzip full activated", "lite", "tzip lite activated"),
        ("R4: /tzip full -> /tzip ultra",  "full", "tzip full activated", "ultra", "tzip ultra activated"),
        ("R5: /tzip -> /tzip off",         None, "tzip lite activated", "off", "tzip deactivated"),
        ("R6: /tzip ultra -> /tzip full",  "ultra", "tzip ultra activated", "full", "tzip full activated"),
    ]
    for name, a1, r1, a2, expected in cases:
        n_ok = n_leak = 0
        fails = []
        for k in range(n):
            ctxm = [{"role": "system", "content": sp}, *ctx]
            first = block if a1 is None else block + "\n\n" + a1
            ctxm += [{"role": "user", "content": first}, {"role": "assistant", "content": r1}]
            ctxm += [{"role": "user", "content": block + "\n\n" + a2}]
            resp = client.chat.completions.create(model="LiquidAI/LFM2.5-2.6B", messages=ctxm,
                temperature=temp, max_tokens=512, extra_headers={"x-session-affinity": "live"}, extra_body=body)
            m = resp.choices[0].message
            reply = (m.content or "").strip()
            reasoning = getattr(m, "reasoning_content", None) or ""
            ok = reply == expected
            leak = "example" in reasoning.lower()
            n_ok += ok; n_leak += leak
            if not ok:
                fails.append((k + 1, reply, reasoning[:300]))
            time.sleep(0.15)
        tag = "PASS" if n_ok == n else f"FAIL {n - n_ok}/{n}"
        say(f"[{tag}] {name}: ok {n_ok}/{n} leaks {n_leak}/{n} (expected {expected!r})")
        for k, reply, snip in fails[:3]:
            say(f"    !! run {k}: {reply!r}")
            say(f"       reasoning: {snip!r}")

if __name__ == "__main__":
    main()
