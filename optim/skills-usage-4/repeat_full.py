#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///
"""Repeat the full-start scenarios to expose intermittent failures.
- 'tzip full' fresh start x10, bare 'full' x10
- full -> on/lite/ultra/off x5 each
Student generates, teacher judges (same judge as test-modes.py)."""
import re
import train
from train import (load_model_configs, build_lms, make_user, user_text_of,
                   tzip_skill_block, rollout, say, EXAMPLE_FILE, PAIR_FILE, TZIP_SKILL_DIR)

CONFIRM = {"on": "tzip lite activated", "lite": "tzip lite activated",
           "full": "tzip full activated", "ultra": "tzip ultra activated",
           "off": "tzip deactivated"}


def tzip_usage_rules():
    body = (TZIP_SKILL_DIR / "SKILL.md").read_text()
    body = re.sub(r"\A---\n.*?\n---\n", "", body)
    m = re.search(r"## Usage\n(.*?)(?=\n## )", body, re.S)
    return m.group(1).strip()


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
    hist = "\n".join(f"  {a!r} -> {r!r}" for a, r in history) or "  (none — first invocation)"
    prompt = JUDGE_PROMPT.format(rules=tzip_usage_rules(), history=hist, current=current,
                                 reply=reply or "(no text reply)", reasoning=(reasoning or "(none)")[:800])
    resp = teacher.forward(messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    vs = re.search(r"VERDICT:\s*(PASS|PARTIAL|FAIL)", text)
    ss = re.search(r"SCORE:\s*([0-9.]+)", text)
    rs = re.search(r"REASON:\s*(.+)", text)
    verdict = vs.group(1) if vs else "FAIL"
    score = float(ss.group(1)) if ss else 0.0
    reason = rs.group(1).strip() if rs else "(no reason parsed)"
    return verdict, score, reason


def main():
    import train as t
    API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG = load_model_configs()
    t.API_BASE, t.API_KEY = API_BASE, API_KEY
    student, teacher = build_lms(API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG)

    doc = train.json.loads(EXAMPLE_FILE.read_text())
    msgs, tools = doc["messages"], doc["tools"]
    pair = train.json.loads(PAIR_FILE.read_text())
    i1 = next(idx for idx, m in enumerate(msgs) if user_text_of(m).startswith("How does skill system work?"))
    msgs[i1] = make_user(pair["step1_user"])
    msgs[i1 + 1] = {"role": "assistant", "content": pair["step1_assistant"], "reasoning_content": pair["step1_reasoning"]}
    base_ctx = msgs
    block = tzip_skill_block()

    def tuser(arg):
        return make_user(block + (f"\n\n{arg}" if arg else ""))

    results = []
    say("REPEAT FULL-START STRESS (teacher-judged)")

    # fresh-start repeats
    for arg, n in (("tzip full", 10), ("full", 10)):
        for k in range(n):
            ctx = base_ctx + [tuser(arg)]
            _, final = rollout(student, ctx, tools, max_turns=4)
            reply = (final.get("content") or "").strip()
            v, s, r = judge(teacher, [], arg, reply, final.get("reasoning_content"))
            results.append((f"A-repeat '{arg}' #{k + 1}", v, s, reply))
            if v != "PASS":
                say(f"  !! [{v} {s:.1f}] '{arg}' #{k + 1}: {reply!r} — {r}")

    # full -> target repeats (x10 each)
    for target in ("on", "lite", "ultra", "off"):
        turns = [("tzip full", CONFIRM["full"])]
        for k in range(10):
            ctx = base_ctx
            for a, rep in turns:
                ctx += [tuser(a), {"role": "assistant", "content": rep}]
            ctx += [tuser(f"tzip {target}")]
            _, final = rollout(student, ctx, tools, max_turns=4)
            reply = (final.get("content") or "").strip()
            v, s, r = judge(teacher, turns, f"tzip {target}", reply, final.get("reasoning_content"))
            results.append((f"B-repeat full -> {target} #{k + 1}", v, s, reply))
            if v != "PASS":
                say(f"  !! [{v} {s:.1f}] full -> {target} #{k + 1}: {reply!r} — {r}")

    # X -> full repeats (transitions INTO full, x10 each)
    for src in ("on", "lite", "ultra"):
        turns = [(f"tzip {src}", CONFIRM[src])]
        for k in range(10):
            ctx = base_ctx
            for a, rep in turns:
                ctx += [tuser(a), {"role": "assistant", "content": rep}]
            ctx += [tuser("tzip full")]
            _, final = rollout(student, ctx, tools, max_turns=4)
            reply = (final.get("content") or "").strip()
            v, s, r = judge(teacher, turns, "tzip full", reply, final.get("reasoning_content"))
            results.append((f"B-repeat {src} -> full #{k + 1}", v, s, reply))
            if v != "PASS":
                say(f"  !! [{v} {s:.1f}] {src} -> full #{k + 1}: {reply!r} — {r}")

    n_pass = sum(1 for r in results if r[1] == "PASS")
    say(f"TOTAL: {n_pass} PASS / {len(results) - n_pass} non-PASS ({len(results)} tests)")
    for name, v, s, reply in results:
        mark = "ok " if v == "PASS" else "!! "
        say(f"  [{mark}] {name}: {reply!r} ({v} {s:.1f})")


if __name__ == "__main__":
    main()
