# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=2.9.0"]
# ///
"""Focused stress for candidate step-1 variants: the 11 activations + the 6 live
consecutive-invocation patterns + the 3 leak-hotspot transitions, x10 each."""
import json, sys, time
from datetime import datetime
from pathlib import Path
import repro_real

CONFIRM = {"on": "tzip lite activated", "lite": "tzip lite activated", "full": "tzip full activated",
           "ultra": "tzip ultra activated", "off": "tzip deactivated"}
MODES = ["on", "lite", "full", "ultra", "off"]

def say(t):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {t}", flush=True)

def main():
    variant = sys.argv[1]
    pair = json.loads(Path(sys.argv[2]).read_text())
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    sp, ctx, tzip_user, client, body, temp = repro_real.build_context()
    i1 = next(i for i, m in enumerate(ctx) if m.get("role") == "user" and "skill system work" in m["content"])
    ctx[i1 + 1] = {"role": "assistant", "content": pair["step1_assistant"],
                   "reasoning_content": pair.get("step1_reasoning", "")}
    block = tzip_user[:tzip_user.rfind("\n\n")]

    cases = []
    for arg in ("tzip", "tzip on", "on", "tzip lite", "lite", "tzip full", "full", "tzip ultra", "ultra", "tzip off", "off"):
        cases.append((f"A {arg!r}", [], arg))
    for a1, r1, a2 in [("", "tzip lite activated", "full"), ("", "tzip lite activated", "lite"),
                       ("full", "tzip full activated", "lite"), ("full", "tzip full activated", "ultra"),
                       ("", "tzip lite activated", "off"), ("ultra", "tzip ultra activated", "full")]:
        cases.append((f"live {a1 or '(bare)'} -> {a2}", [(a1, r1)], a2))
    for a, b in [("ultra", "off"), ("lite", "off"), ("full", "off")]:
        cases.append((f"B {a} -> {b}", [(f"tzip {a}", CONFIRM[a])], f"tzip {b}"))

    tot_ok = tot_n = tot_leak = 0
    for name, setup, arg in cases:
        key = arg.strip().split()[-1]
        expected = CONFIRM.get(key, CONFIRM["on"])
        n_ok = n_leak = 0
        fails = []
        for k in range(n):
            ctxm = [{"role": "system", "content": sp}, *ctx]
            for a, r in setup:
                ctxm.append({"role": "user", "content": block if not a else block + "\n\n" + a})
                ctxm.append({"role": "assistant", "content": r})
            ctxm.append({"role": "user", "content": block + "\n\n" + arg})
            resp = client.chat.completions.create(model="LiquidAI/LFM2.5-2.6B", messages=ctxm, temperature=temp,
                max_tokens=512, extra_headers={"x-session-affinity": "variant"}, extra_body=body)
            m = resp.choices[0].message
            reply = (m.content or "").strip()
            reasoning = getattr(m, "reasoning_content", None) or ""
            ok = reply == expected
            n_ok += ok; tot_ok += ok
            leak = "example" in reasoning.lower()
            n_leak += leak; tot_leak += leak
            if not ok:
                fails.append((k + 1, reply))
            tot_n += n
            time.sleep(0.12)
        tag = "PASS" if n_ok == n else f"FAIL {n - n_ok}/{n}"
        say(f"[{tag}] {name:<22} ok {n_ok}/{n} leaks {n_leak}/{n}")
        for k, reply in fails[:3]:
            say(f"    !! run {k}: {reply!r}")
    say(f"=== {variant}: {tot_ok}/{tot_n} correct, {tot_leak}/{tot_n} leaks ===")

if __name__ == "__main__":
    main()
