# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai>=2.9.0",
# ]
# ///
"""
stress_real.py — Full activation/transition stress under the REAL pi-session
context (reconstructed from the 01a01ee9 session log; see repro_real.py).

Covers what the user asked for: each start state individually, transitions
between all states, and the high-risk full->X transitions — measured for
both wrong replies and cross-skill leakage (example-skill content in the
student's reasoning, which must stay isolated per invocation).

Usage: uv run --script stress_real.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import repro_real

HERE = Path(__file__).resolve().parent
OUT = HERE / "stress-real"
OUT.mkdir(exist_ok=True)
LOG = OUT / "stress-real.log"
_fh = open(LOG, "w")


def say(t: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {t}"
    print(line, flush=True)
    _fh.write(line + "\n")
    _fh.flush()


CONFIRM = {
    "on": "tzip lite activated",
    "lite": "tzip lite activated",
    "full": "tzip full activated",
    "ultra": "tzip ultra activated",
    "off": "tzip deactivated",
}
MODES = ["on", "lite", "full", "ultra", "off"]


def main() -> None:
    system_prompt, ctx, tzip_user_text, client, extra_body, temp = repro_real.build_context()

    # Optionally swap in the optimized step-1 answer (from the GEPA pipeline).
    pair_file = sys.argv[1] if len(sys.argv) > 1 else "step1-pair-optimized.json"
    pair_path = Path(pair_file)
    if pair_path.exists():
        pair = json.loads(pair_path.read_text())
        i1 = next(i for i, m in enumerate(ctx) if m.get("role") == "user" and "skill system work" in (m.get("content") or ""))
        ctx[i1 + 1] = {
            "role": "assistant",
            "content": pair["step1_assistant"],
            "reasoning_content": pair.get("step1_reasoning", ""),
        }
        say(f"Using OPTIMIZED step-1 pair from {pair_path.name}")
    else:
        say(f"No {pair_path.name} — stress runs on the SEED (unoptimized) pair")

    used = repro_real.verify_tokens(client, system_prompt, ctx, tzip_user_text, extra_body, temp)

    # tzip skill block (the part before the trailing argument) of the real call
    block_end = tzip_user_text.rfind("\n\n")
    tzip_block = tzip_user_text[:block_end]
    assert tzip_block.startswith('<skill name="tzip"')

    results = []  # (name, expected, reply, ok, leaked, reasoning_snippet)

    def run_case(name, setup, arg, n=10):
        """setup: list of (arg, reply) already in context; arg: current tzip argument."""
        key = arg.strip().split()[-1] if arg.strip() else ""
        expected = CONFIRM.get(key, CONFIRM["on"])  # '' / 'tzip' -> default lite
        n_ok = n_leak = 0
        failures = []
        for k in range(n):
            ctxm = [{"role": "system", "content": system_prompt}, *ctx]
            for a, r in setup:
                ctxm.append({"role": "user", "content": tzip_block + "\n\n" + a})
                ctxm.append({"role": "assistant", "content": r})
            ctxm.append({"role": "user", "content": tzip_block + "\n\n" + arg})
            resp = client.chat.completions.create(
                model="LiquidAI/LFM2.5-2.6B", messages=ctxm, temperature=temp, max_tokens=512,
                extra_headers={"x-session-affinity": "stress-real"}, extra_body=extra_body)
            m = resp.choices[0].message
            reply = (m.content or "").strip()
            reasoning = getattr(m, "reasoning_content", None) or ""
            ok = reply == expected
            leaked = "example" in reasoning.lower() or "example.sh" in reasoning
            n_ok += ok
            n_leak += leaked
            if not ok:
                failures.append((k + 1, reply, reasoning[:400]))
            time.sleep(0.15)
        results.append((name, expected, n_ok, n, n_leak, failures))
        status = "PASS" if n_ok == n else f"FAIL {n - n_ok}/{n}"
        say(f"  [{status}] {name:<22} expected: {expected!r} | ok {n_ok}/{n} | reasoning-leaks: {n_leak}/{n}")
        for k, reply, snippet in failures:
            say(f"      !! run {k}: student {reply!r}")
            say(f"         reasoning: {snippet!r}")

    say(f"\nTOKEN CHECK: {used} (real session: {repro_real.EXPECTED_INPUT_TOKENS})")

    say("\n=== PHASE A — single activations, fresh start, x10 each ===")
    for arg in ("tzip", "tzip on", "on", "tzip lite", "lite", "tzip full", "full",
                "tzip ultra", "ultra", "tzip off", "off"):
        run_case(f"A {arg!r}", [], arg)

    say("\n=== PHASE B — all 20 transitions, canonical setup, x10 each ===")
    for a in MODES:
        for b in MODES:
            if a == b:
                continue
            setup = [(f"tzip {a}", CONFIRM[a])]
            run_case(f"B {a} -> {b}", setup, f"tzip {b}")

    n_tests = sum(r[3] for r in results)
    n_ok = sum(r[2] for r in results)
    n_leak = sum(r[4] for r in results)
    say(f"\nTOTAL: {n_ok}/{n_tests} correct; {n_tests - n_ok} wrong replies; {n_leak}/{n_tests} runs with example-skill content in reasoning.")
    for name, expected, ok, n, leaks, failures in results:
        if ok < n:
            say(f"  !! {name}: {ok}/{n} (expected {expected!r})")


if __name__ == "__main__":
    main()
