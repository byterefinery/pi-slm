#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///

'''
Mode-by-mode tzip test suite with a TEACHER JUDGE.

Student: LiquidAI/LFM2.5-2.6B (thinking ON) on the frozen production prefix
(system + skills/tools Q&A + optimized step-1 pair + 3 example one-shots).
Teacher: Qwen/Qwen3.8-27B (thinking OFF) judges EVERY student answer against
the tzip Usage rules (1.0 exact / 0.5 right-mode-wrong-wording / 0.0 wrong) and
gives a one-line reason. The expected confirmation is also computed locally for
display, but the verdict comes from the teacher.

Phases:
  A. SINGLE-MODE ACTIVATIONS — every mode (on, lite, full, ultra, off), both
     phrasings ("tzip <mode>" as the skill's Usage specifies, and bare "<mode>"),
     plus bare "tzip" (default). Fresh prefix, one tzip message, student answers.
  B. ALL TRANSITIONS — every ordered pair of different modes (20): the start
     state is established canonically ("tzip <start>" -> canonical
     confirmation), then the student answers the transition message
     ("tzip <target>") and the teacher judges.
  C. RANDOM WALKS — for all 5 starting states, a walk of 3 random steps
     (seed=7); the start is canonical, then each step is answered by the student
     ON ITS OWN PREVIOUS ANSWERS (chained, like real use) and judged.

Log: test-modes.log (stdout).
'''

import json
import random
import re
from pathlib import Path

import train
from train import (
    EXAMPLE_FILE,
    PAIR_FILE,
    HERE,
    PI_ROOT,
    TZIP_SKILL_DIR,
    load_model_configs,
    build_lms,
    make_user,
    user_text_of,
    tzip_skill_block,
    rollout,
    norm,
    say,
)

MODES = ["on", "lite", "full", "ultra", "off"]
CONFIRM = {
    "on": "tzip lite activated",
    "lite": "tzip lite activated",
    "full": "tzip full activated",
    "ultra": "tzip ultra activated",
    "off": "tzip deactivated",
}

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def load_base_ctx():
    """Frozen production prefix with the optimized step-1 pair."""
    doc = json.loads(EXAMPLE_FILE.read_text())
    msgs, tools = doc["messages"], doc["tools"]
    pair = json.loads(PAIR_FILE.read_text())
    i1 = next(idx for idx, m in enumerate(msgs) if user_text_of(m).startswith("How does skill system work?"))
    msgs[i1] = make_user(pair["step1_user"])
    msgs[i1 + 1] = {"role": "assistant", "content": pair["step1_assistant"], "reasoning_content": pair["step1_reasoning"]}
    return msgs, tools


def tzip_usage_rules():
    """The tzip Usage section (what the judge checks against)."""
    body = (TZIP_SKILL_DIR / "SKILL.md").read_text()
    body = re.sub(r"\A---\n.*?\n---\n", "", body)
    m = re.search(r"## Usage\n(.*?)(?=\n## )", body, re.S)
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# teacher judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are judging a small assistant model's reply in a pi conversation, against the rules of the `tzip` skill.

The tzip Usage rules (from the skill block the model saw):
{rules}

Conversation so far (tzip invocations, in order — replies are the model's own):
{history}

The current tzip invocation argument is: {current}

The model's reply to this invocation:
{reply}
The model's internal reasoning for this reply:
{reasoning}

Judge ONLY this one reply, strictly against the rules:
- the argument selects the mode: '' / 'tzip' / 'tzip on' / 'tzip lite' / 'on' / 'lite' -> lite; 'tzip full' / 'full' -> full; 'tzip ultra' / 'ultra' -> ultra; 'tzip off' / 'off' -> deactivated
- the reply must be EXACTLY the mode confirmation and nothing else: 'tzip lite activated', 'tzip full activated', 'tzip ultra activated', or 'tzip deactivated'
- extra words, explanations, or a wrong mode are errors

Output exactly three lines, nothing else:
VERDICT: PASS|PARTIAL|FAIL
SCORE: 1.0|0.5|0.0
REASON: <one short sentence>"""


def judge(teacher, history, current, reply, reasoning):
    hist = "\n".join(f'  {a!r} -> {r!r}' for a, r in history) or "  (none — first invocation)"
    prompt = JUDGE_PROMPT.format(
        rules=tzip_usage_rules(),
        history=hist,
        current=current,
        reply=reply or "(no text reply)",
        reasoning=(reasoning or "(none)")[:800],
    )
    resp = teacher.forward(messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    vs = re.search(r"VERDICT:\s*(PASS|PARTIAL|FAIL)", text)
    ss = re.search(r"SCORE:\s*([0-9.]+)", text)
    rs = re.search(r"REASON:\s*(.+)", text)
    verdict = vs.group(1) if vs else ("PASS" if (ss and float(ss.group(1)) >= 0.999) else "FAIL")
    score = float(ss.group(1)) if ss else (1.0 if verdict == "PASS" else 0.0)
    reason = rs.group(1).strip() if rs else "(no reason parsed)"
    return verdict, score, reason


# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------


def tzip_user(arg):
    block = tzip_skill_block()
    return make_user(block + (f"\n\n{arg}" if arg else ""))


def expected_for(arg):
    a = (arg or "").strip()
    if a in ("", "tzip"):
        return CONFIRM["on"]  # default -> lite
    if a in CONFIRM:
        return CONFIRM[a]
    parts = a.split()
    if len(parts) == 2 and parts[0] == "tzip" and parts[1] in CONFIRM:
        return CONFIRM[parts[1]]
    return "(see rules)"


def turns_to_msgs(turns):
    out = []
    for arg, reply in turns:
        out += [tzip_user(arg), {"role": "assistant", "content": reply}]
    return out


def run_one(student, tools, base_ctx, turns, current_arg, tag):
    """Append turns + current tzip user message, student rolls out, teacher judges."""
    ctx = base_ctx + turns_to_msgs(turns) + [tzip_user(current_arg)]
    transcript, final = rollout(student, ctx, tools, max_turns=4)
    reply = (final.get("content") or "").strip()
    history = [(a, r) for a, r in turns]
    verdict, score, reason = judge(teacher_global, history, current_arg, reply, final.get("reasoning_content"))
    n_tc = sum(len(m.get("tool_calls") or []) for m in transcript)
    say(f"[{tag}] {current_arg!r} (after: {[a for a, _ in turns] or 'fresh start'})")
    say(f"    expected: {expected_for(current_arg)!r}")
    say(f"    student:  {reply or '(no text reply)'} (tool calls: {n_tc})")
    say(f"    judge:    {verdict} {score:.1f} — {reason}")
    return verdict, score, final


teacher_global = None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    global teacher_global
    API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG = load_model_configs()
    train.API_BASE, train.API_KEY = API_BASE, API_KEY  # for say() masking
    say(f"tzip mode suite — student thinking ON, teacher JUDGE thinking OFF. API base: {API_BASE[:8]}*** (masked)")
    student, teacher = build_lms(API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG)
    teacher_global = teacher
    base_ctx, TOOLS = load_base_ctx()
    say(f"Prefix: {len(base_ctx)} messages (frozen) + step-1 pair from {PAIR_FILE.name}.")

    results = []

    # ---- Phase A: single-mode activations, both phrasings ----
    say(f"\n{'=' * 74}\nPHASE A — single-mode activations (fresh start, one tzip message)\n{'=' * 74}")
    phrasings = [("tzip", "(default)")]
    for mode in MODES:
        phrasings.append((f"tzip {mode}", f"'tzip {mode}'"))
        phrasings.append((mode, f"bare '{mode}'"))
    for arg, label in phrasings:
        v, s, _ = run_one(student, TOOLS, base_ctx, [], arg, f"A {label}")
        results.append(("A", arg, v, s))

    # ---- Phase B: all ordered transitions (start state canonical) ----
    say(f"\n{'=' * 74}\nPHASE B — all mode-to-mode transitions (start canonical, student answers the switch)\n{'=' * 74}")
    for a in MODES:
        for b in MODES:
            if a == b:
                continue
            turns = [(f"tzip {a}", CONFIRM[a])]
            v, s, _ = run_one(student, TOOLS, base_ctx, turns, f"tzip {b}", f"B {a} -> {b}")
            results.append(("B", f"{a} -> {b}", v, s))

    # ---- Phase C: random walks, all starting states, chained on student answers ----
    say(f"\n{'=' * 74}\nPHASE C — random walks (3 steps, start canonical, chained on student's own answers, seed=7)\n{'=' * 74}")
    rng = random.Random(7)
    for start in MODES:
        seq = [start]
        cur = start
        for _ in range(3):
            cur = rng.choice([m for m in MODES if m != cur])
            seq.append(cur)
        say(f"\n  walk starting from {start!r}: {seq}")
        turns = [(f"tzip {start}", CONFIRM[start])]
        for i, step in enumerate(seq[1:], start=2):
            v, s, final = run_one(student, TOOLS, base_ctx, turns, f"tzip {step}", f"C {start}-walk step{i}")
            results.append(("C", f"{start}-walk step{i} ({seq[i - 2]} -> {step})", v, s))
            # chain on the STUDENT's own answer, like in real use
            turns.append((f"tzip {step}", (final.get("content") or "").strip()))

    # ---- summary ----
    say(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    for phase in ("A", "B", "C"):
        pr = [r for r in results if r[0] == phase]
        np_ = sum(1 for r in pr if r[2] == "PASS")
        npt = sum(1 for r in pr if r[2] == "PARTIAL")
        nf = sum(1 for r in pr if r[2] == "FAIL")
        say(f"  Phase {phase}: {np_} PASS / {npt} PARTIAL / {nf} FAIL  ({np_ + npt + nf} tests)")
        for r in pr:
            if r[2] != "PASS":
                say(f"    !! [{r[0]}] {r[1]}: {r[2]} {r[3]:.1f}")
    np_ = sum(1 for r in results if r[2] == "PASS")
    npt = sum(1 for r in results if r[2] == "PARTIAL")
    nf = sum(1 for r in results if r[2] == "FAIL")
    say(f"  TOTAL: {np_} PASS / {npt} PARTIAL / {nf} FAIL  ({len(results)} tests)")


if __name__ == "__main__":
    main()
