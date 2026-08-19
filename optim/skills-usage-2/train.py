#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai"]
# ///

'''
Teach the student LM (LiquidAI/LFM2.5-2.6B) to handle pi skill invocations like
the teacher LM (Qwen/Qwen3.8-27B).

The only lever: ONE synthetic user/assistant pair (content + reasoning_content)
inserted into the student conversation just AFTER the existing synthetic
skills/tools Q&A pairs and BEFORE the real skill-invocation user message.

Pipeline:
  1. Baseline - student answers the recorded skill invocation WITHOUT the pair.
  2. GEPA     - a "coach" dspy.ChainOfThought (run by the TEACHER) writes the
                synthetic pair from a fixed SPEC. GEPA optimizes the coach's
                INSTRUCTION: each candidate is scored by rolling the STUDENT out
                on the full conversation with the candidate pair and grading the
                student's content against the TEACHER's content. The metric
                returns dspy.Prediction(score=..., feedback=...) so GEPA's
                reflection LM (the teacher) can rewrite the instruction.
  3. Final    - the best coach writes the pair once; it is saved to
                synthetic-pair.json and the student is re-run on all examples.

Fixed / untouchable: the system prompt and the two existing synthetic Q&A pairs.

NOTE: this is how you instantiate dspy.LM - keep this string/comment
lm = dspy.LM(
    f"openai/{model}", # used model
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=..., # read for used model
    max_tokens=16384,
    extra_headers={"x-session-affinity": "dspy-optim"},
    extra_body={
        "top_k": ..., # read for used model, or omit if missing
        "min_p": ..., # read for used model, or omit if missing
        "presence_penalty": ..., # read for used model, or omit if missing
        "repeat_penalty": ..., # read for used model, or omit if missing
    },
)

Run (all modes log live progress to stdout; redirect to a file):
    uv run --script train.py             # baseline + GEPA + best-of-N pair selection + final eval
    uv run --script train.py smoke       # tiny budget, validates the pipeline only
    uv run --script train.py --select-only   # re-pick the pair from the saved program (DRAFTS env var, default 5)
    uv run --script train.py --verify    # evaluate the saved synthetic-pair.json only, no drafting

Env vars: BUDGET (GEPA metric calls, default 180), DRAFTS (best-of-N drafts, default 5),
          WARM_START=1 (seed GEPA from coach-instruction.txt, the previous best).

Artifacts:
    coach-optimized.json    state-only program (SkillCoach().load(...))
    coach-instruction.txt   the winning coach instruction (GEPA's output text)
    synthetic-pair.json     the final synthetic user/assistant pair (the deliverable)
    gepa_log/               per-candidate instructions, best outputs, GEPA state
'''

import difflib
import json
import logging
import os
import re
import sys
import time

import dspy

# ----------------------------------------------------------------------------- setup

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")


def load_api() -> tuple[str, str]:
    """Read (api_base, api_key) from pi's models.json - the first provider with both."""
    text = open(MODELS_JSON).read()
    cfg = json.loads(re.sub(r",\s*([}\]])", r"\1", text))  # pi's file may have trailing commas
    for prov in cfg.get("providers", {}).values():
        if prov.get("baseUrl") and prov.get("apiKey"):
            return prov["baseUrl"], prov["apiKey"]
    raise SystemExit(f"no provider with baseUrl+apiKey in {MODELS_JSON}")


API_BASE, API_KEY = load_api()

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"
TEACHER_MODEL = "Qwen/Qwen3.8-27B"

TEACHER_FILE = os.path.join(HERE, "tzip-full-Qwen-Qwen3.8-27B.json")
STUDENT_FILE = os.path.join(HERE, "tzip-full-LiquidAI-LFM2.5-2.6B.json")

# Sampling params from ~/.pi/agent/models.json for each model.
STUDENT = dspy.LM(
    f"openai/{STUDENT_MODEL}",
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=0.1,
    max_tokens=16384,
    timeout=300,      # fail fast instead of clogging the shared server queue
    extra_headers={"x-session-affinity": "dspy-optim"},
    extra_body={"top_k": 50, "repeat_penalty": 1.1, "reasoning_effort": "high"},
    cache=True,  # same conversation -> same reply; different pair -> fresh call
    num_retries=5,
)

TEACHER = dspy.LM(
    f"openai/{TEACHER_MODEL}",
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=1.0,
    max_tokens=16384,
    extra_headers={"x-session-affinity": "dspy-optim"},
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        # teacher runs with thinking OFF: reasoning takes too long; drafting and
        # reflection are faster and just as effective without it (student stays on)
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    },
    cache=False,  # coach must get fresh pair drafts for every candidate
    timeout=300,
    num_retries=5,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warmup(lm: dspy.LM, name: str, tries: int = 40, sleep_s: float = 15.0) -> None:
    """Wait out server-side model loading (cold start) before spending budget on it."""
    for i in range(1, tries + 1):
        try:
            lm(messages=[{"role": "user", "content": "Reply: ok"}], max_tokens=64)
            log(f"warmup {name}: ready")
            return
        except Exception as e:
            log(f"warmup {name}: not ready yet ({type(e).__name__}), retry {i}/{tries} in {sleep_s:.0f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"{name} never became ready")


def text_of(msg: dict) -> str:
    """Flatten a message content that may be a string or a list of parts."""
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


# ----------------------------------------------------------------------------- data

def load_cases() -> list[dict]:
    """Build eval cases: the full tzip matrix (default/lite/full/ultra/off + mode switches).

    Canonical required behaviors (checked every round, reported per case):
      /skill:tzip full  -> "tzip full activated"
      /skill:tzip       -> "tzip lite activated"   (no argument: the skill's default)
      The skill's default is Lite for 'tzip', 'tzip on' and 'tzip lite' alike.
    """
    t_msgs = json.load(open(TEACHER_FILE))["messages"]
    s_msgs = json.load(open(STUDENT_FILE))["messages"]

    # student conversation layout (indices):
    # 0 system | 1 user(what skills) | 2 assistant | 3 user(what tools) | 4 assistant
    # 5 user(skill invocation) | 6 assistant(failed answer)
    assert [m["role"] for m in s_msgs[:6]] == ["system", "user", "assistant", "user", "assistant", "user"]
    invoke = text_of(s_msgs[5])
    assert "</skill>" in invoke, "expected a <skill> block in the last user message"
    skill_block, arg = invoke.rsplit("</skill>", 1)
    real_reasoning = t_msgs[6].get("reasoning_content", "")

    def block_invoke(arg: str) -> str:
        return f"{skill_block}</skill>" + (f"\n\n{arg}" if arg else "")

    def simple(name: str, arg: str, target: str, is_real: bool = False) -> dict:
        return {"name": name, "arg": arg,
                "turns": [{"invoke": block_invoke(arg), "reply": None}],
                "target": target, "is_real": is_real}

    def switch(name: str, arg1: str, reply1: str, reasoning1: str, arg2: str, target: str) -> dict:
        """Mode already active, user re-invokes the skill with a different argument."""
        return {"name": name, "arg": arg2,
                "turns": [{"invoke": block_invoke(arg1), "reply": reply1, "reply_reasoning": reasoning1},
                          {"invoke": block_invoke(arg2), "reply": None}],
                "target": target, "is_real": False}

    def sw(name: str, mode1: str, mode2: str) -> dict:
        """Mode switch from mode1 (already active, confirmed) to mode2 (the argument of the latest block)."""
        reply1 = "tzip deactivated" if mode1 == "off" else f"tzip {mode1} activated"
        reasoning1 = (f"The user invoked the tzip skill with {mode1} mode. Per the skill, "
                      "I will reply with the short mode confirmation.")
        target = "tzip deactivated" if mode2 == "off" else f"tzip {mode2} activated"
        return switch(name, mode1, reply1, reasoning1, mode2, target)

    # Full tzip matrix: default (no arg), lite, full, ultra, off
    # - plus ALL 12 mode switches (a later skill message with a new argument wins).
    return [
        simple("tzip-default", "", "tzip lite activated"),
        simple("tzip-on", "on", "tzip lite activated"),
        simple("tzip-lite", "lite", "tzip lite activated"),
        simple("tzip-full", "full", "tzip full activated"),
        simple("tzip-ultra", "ultra", "tzip ultra activated"),
        simple("tzip-off", "off", "tzip deactivated"),
        # lite <-> ultra
        sw("tzip-lite-to-ultra", "lite", "ultra"),
        sw("tzip-ultra-to-lite", "ultra", "lite"),
        # lite <-> full (existing recorded-context variants)
        switch("tzip-lite-to-full", "", "tzip lite activated",
               "The user invoked the tzip skill with no mode - the default is lite. Per the skill: reply with the mode name.",
               "full", "tzip full activated"),
        switch("tzip-full-to-lite", "full", "tzip full activated.", real_reasoning,
               "lite", "tzip lite activated"),
        # full <-> ultra
        sw("tzip-full-to-ultra", "full", "ultra"),
        sw("tzip-ultra-to-full", "ultra", "full"),
        # lite <-> off
        sw("tzip-lite-to-off", "lite", "off"),
        sw("tzip-off-to-lite", "off", "lite"),
        # full <-> off
        sw("tzip-full-to-off", "full", "off"),
        sw("tzip-off-to-full", "off", "full"),
        # ultra <-> off
        sw("tzip-ultra-to-off", "ultra", "off"),
        sw("tzip-off-to-ultra", "off", "ultra"),
    ]


def build_examples(cases: list[dict]) -> list:
    """Real teacher pair counts triple so it dominates GEPA's objective."""
    exs = []
    for c in cases:
        for _ in range(3 if c["is_real"] else 1):
            exs.append(dspy.Example(
                spec=SPEC, turns=c["turns"], target=c["target"],
                name=c["name"], arg=c["arg"],
            ).with_inputs("spec"))
    return exs


# ----------------------------------------------------------------------------- student rollout

PREFIX: list[dict] = []  # filled in main(): system + the two existing synthetic pairs
TOOLS: list[dict] = []   # filled in main(): the pi tools array (fidelity to deployment)


def student_rollout(turns: list[dict], pair: dict | None) -> dict:
    """Student completion: fixed prefix (+ optional synthetic pair) + skill turn(s).

    Each turn: {invoke: user text, reply: known assistant reply (None = model generates)}.
    """
    messages = list(PREFIX)
    if pair:
        messages.append({"role": "user", "content": pair["synth_user"]})
        messages.append({"role": "assistant", "content": pair["synth_content"],
                         "reasoning_content": pair["synth_reasoning"]})
    for t in turns:
        messages.append({"role": "user", "content": t["invoke"]})
        if t.get("reply"):  # intermediate turn of a multi-turn (mode switch) case
            msg = {"role": "assistant", "content": t["reply"]}
            if t.get("reply_reasoning"):
                msg["reasoning_content"] = t["reply_reasoning"]
            messages.append(msg)
    out = STUDENT(messages=messages, max_tokens=4096, tools=TOOLS)
    last = out[-1] if out else {}
    if isinstance(last, str):
        last = {"text": last}
    return {
        "text": (last.get("text") or last.get("content") or "").strip(),
        "reasoning": (last.get("reasoning_content") or "").strip(),
        "tool_calls": last.get("tool_calls") or [],
    }


# ----------------------------------------------------------------------------- scoring

BANNED_WORDS = {"tzip", "git", "skman", "webfetch", "websearch", "find-skills"}


def genericity(cap_report: list[str]) -> float:
    """The pair must be skill-agnostic: cap score if it names a specific skill.

    (Pattern words like 'activated' are allowed: teaching the generic confirmation
    pattern is legitimate; hardcoding a skill's name is overfitting.)"""
    text = norm(" ".join(str(v) for v in cap_report))
    hits = sorted(w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", text))
    if hits:
        return 0.8
    if len(text) > 1200:
        return 0.85
    return 1.0


def grade(target: str, got: dict) -> tuple[float, str]:
    """Score one student reply against the teacher target. Returns (score, problem)."""
    if got["tool_calls"]:
        names = [tc.get("name") or (tc.get("function") or {}).get("name") for tc in got["tool_calls"]]
        return 0.0, f"student called tool(s) {names} instead of replying in text"
    t, g = norm(target), norm(got["text"])
    if not g:
        return 0.0, "student produced no visible text (still reasoning / truncated?)"
    if t == g or t.rstrip(".!") == g.rstrip(".!"):
        return 1.0, "exact match"
    terms = [w.strip(".,!?") for w in t.split() if len(w.strip(".,!?")) > 2]
    coverage = sum(w in g for w in terms) / len(terms) if terms else 0.0
    fuzzy = difflib.SequenceMatcher(None, t, g).ratio()
    score = round(0.55 * coverage + 0.45 * fuzzy, 4)
    # Diagnose the dominant failure mode for the reflection LM.
    if "available" in g and ("tool" in g or "skill" in g):
        problem = "student re-answered an EARLIER question (skills/tools list) instead of " \
                  "following the <skill> block in the latest user message"
    elif coverage < 0.6:
        problem = "reply is off-topic vs the target"
    else:
        problem = "reply is close but not exact (check skill name, argument, wording, format)"
    return score, problem


def coach_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric: roll the STUDENT out with the candidate pair, grade vs TEACHER.

    trace=None  -> scoring path (float score is what counts)
    trace!=None -> bootstrap/feedback path (feedback text drives the reflection LM)
    Both paths return dspy.Prediction(score=..., feedback=...) for GEPA.
    """
    pair = {k: str(getattr(pred, k, "") or "").strip()
            for k in ("synth_user", "synth_content", "synth_reasoning")}
    if not all(pair.values()):
        return dspy.Prediction(score=0.0,
                               feedback="Coach produced an empty field in the pair. "
                                        "All three fields (synth_user, synth_content, synth_reasoning) are required.")
    try:
        got = student_rollout(example.turns, pair)
    except Exception as e:  # provider hiccup -> treat as failure, do not crash GEPA
        return dspy.Prediction(score=0.0, feedback=f"Student rollout failed: {type(e).__name__}: {e}")
    score, problem = grade(example.target, got)
    score = min(score, genericity([pair["synth_user"], pair["synth_content"], pair["synth_reasoning"]]))
    feedback = (
        f"STUDENT REPLY:   {got['text'][:400]!r}\n"
        f"STUDENT THINKING: {got['reasoning'][:300]!r}\n"
        f"TEACHER TARGET:  {example.target!r}\n"
        f"USER ARGUMENT AFTER </skill> IN THE LAST USER MESSAGE: {example.arg!r} - the skill "
        f"instructions in the <skill> block define the required reply, which must use this argument.\n"
        f"SYNTHETIC PAIR TRIED:\n  user:      {pair['synth_user'][:300]!r}\n"
        f"  assistant: {pair['synth_content'][:300]!r}\n"
        f"  reasoning: {pair['synth_reasoning'][:200]!r}\n"
        f"PROBLEM: {problem}.\n"
        f"REQUIREMENT: after the pair, the student must treat a <skill> block in the LATEST user "
        f"message as an active instruction, follow the skill and the user argument, and reply in "
        f"text exactly as the teacher does - no re-answering older questions, no tool calls. "
        f"The pair must stay generic (no specific skill names) and terse."
    )
    log(f"metric {example.name:<12} score={score:.3f} student={got['text'][:70]!r}")
    return dspy.Prediction(score=score, feedback=feedback)


# ----------------------------------------------------------------------------- coach program

REASONING_DOC = open(os.path.join(HERE, "REASONING-LFM2.5-2.6B.md")).read().strip()

SPEC = """CONTEXT
You design ONE synthetic user message and ONE synthetic assistant message (content + reasoning_content)
for a small 2.6B reasoning LLM ("student") running inside a coding agent called pi.

CONVERSATION LAYOUT (fixed - never change it)
1. system prompt: pi coding assistant; lists available skills (name, description, location).
2. synthetic user: "What are available skills?" -> synthetic assistant: the skill list, ending "These are skills, not tools."
3. synthetic user: "What are available tools?" -> synthetic assistant: the tool list, ending "These are tools, not skills."
4. [THE PAIR YOU WRITE GOES HERE]
5. real user message: a skill invocation. It contains the full skill instructions inside a block
   <skill name="NAME" location="PATH"> ... </skill>
   optionally followed by the user's argument (a few words) after </skill>.

HOW THE SKILL SYSTEM WORKS (Agent Skills, github.com/agentskills/agentskills)
- Skills are folders with a SKILL.md; at startup the agent knows only each skill's name and description (discovery).
- When a task matches, the harness ACTIVATES the skill by injecting the full SKILL.md text into the
  user message inside the <skill name=... location=...> block.
- On activation the agent must treat the skill as active right now: follow its instructions, honor
  any user argument after the block, and reply in the format the skill demands.
- Relative paths inside a skill resolve against the skill's own directory.
- A skill can stay active across following replies until the user deactivates it.

OBSERVED FAILURES (why your pair is needed)
1. Step 5 invoked the "tzip" skill with argument "full" (tzip is a token-pruning mode skill whose
   instructions say to reply with the mode name, e.g. "tzip full activated").
   Teacher (Qwen3.8-27B) replied exactly: tzip full activated.
   Student (LFM2.5-2.6B) sometimes re-answers the earlier "What are available tools?" question and
   prints the tools list again - it ignores the <skill> block in the latest user message.
2. The student sometimes gets the shape right but copies the skill's EXAMPLE word: for argument
   "full" it replies "tzip lite activated" (the skill's first example) instead of building the
   reply from the user's actual argument. A good pair teaches the student to substitute the user's
   argument into the skill's reply pattern, and to never copy example words from inside the skill.

GOAL
After your pair, on step 5 (and any future skill invocation) the student must:
- treat the <skill> block in the latest user message as an active instruction,
- follow the skill's instructions and the user argument,
- reply in the exact format the skill asks for (here: match the teacher's reply),
- if a later message re-invokes the skill with a different argument, the new argument wins
  (a mode switch) and the student confirms the new state in the same format,
- never re-answer an older question, never call tools for a pure skill activation.

STYLE RULES FOR YOUR PAIR
- Plain, short text in the same terse register as the existing synthetic pairs.
- GENERIC: never mention a specific skill name (no "tzip", "git", "skman", "webfetch",
  "websearch", "find-skills"). The pair is reused for every skill that will ever be invoked.
  (Generic pattern words are fine, e.g. teaching that a confirmation is built from the skill name,
  the user's argument, and the skill's own wording.)
- The assistant message must include reasoning_content written in the STUDENT's native reasoning
  voice (see the STUDENT REASONING STYLE section below): open by naming the situation,
  commit to the action with "I will ..." / "Let me ...", short form (1-3 sentences) for this
  simple rule, and END by committing to the exact next action - the student's answer follows
  whatever action its reasoning last commits to.
- No markdown headers, no code fences, at most a couple of short bullets, <= ~300 chars per field.
- NEVER embed a <skill> block inside a synthetic message: the student treats any <skill> block it
  sees as a real activation and gets derailed (it starts answering that block instead of the real one).
  Demonstrate the rule with plain words only (e.g. 'the skill name, then my argument word').

""" + "\nSTUDENT REASONING STYLE (write the synthetic reasoning_content following this guide):\n\n" + REASONING_DOC


class WriteSkillCoachPair(dspy.Signature):
    """Write the synthetic user/assistant pair (content + reasoning) inserted into the
    student's conversation just before the real skill-invocation user message, so the
    student reliably follows the invoked skill and replies like the teacher."""

    spec = dspy.InputField(desc="Full context: layout, skill-system rules, failure, style rules")
    synth_user = dspy.OutputField(desc="Synthetic user message text (short, plain, one line-ish)")
    synth_content = dspy.OutputField(desc="Synthetic assistant reply text (1-3 short sentences, plain)")
    synth_reasoning = dspy.OutputField(desc="Synthetic assistant reasoning_content (1-2 terse sentences)")


SEED_INSTRUCTION = (
    "Write the pair described in the spec. Its job: make the student treat a <skill> block in the "
    "LATEST user message as an active instruction to follow right now (skill instructions + user "
    "argument after the block), and reply only to that message - no re-answering older questions, "
    "no tool calls for a pure skill activation. Teach the student to build any short confirmation "
    "from the skill's reply pattern with the user's actual argument substituted in, never copying "
    "example words from inside the skill. Keep the user message a short question like the earlier "
    "synthetic pairs; keep the assistant answer 1-3 sentences; write reasoning_content in the "
    "student's native voice per the style guide in the spec (situation, 'I will/let me', end by "
    "committing to the exact action). The pair must be generic: no specific skill names."
)


def seed_instruction() -> str:
    """Warm-start from a previous best instruction when WARM_START=1 and it exists."""
    path = os.path.join(HERE, "coach-instruction.txt")
    if os.environ.get("WARM_START") and os.path.exists(path):
        text = open(path).read().strip()
        if text:
            return text
    return SEED_INSTRUCTION


class SkillCoach(dspy.Module):
    """Teacher LM writes the synthetic pair; GEPA optimizes the coach's instruction."""

    def __init__(self):
        super().__init__()
        self.coach = dspy.ChainOfThought(WriteSkillCoachPair)
        self.coach.predict.signature = self.coach.predict.signature.with_instructions(seed_instruction())

    def forward(self, spec: str):
        return self.coach(spec=spec)


# ----------------------------------------------------------------------------- run

def run_eval(cases: list[dict], tag: str, pair: dict | None = None):
    log(f"--- {tag} ---")
    for c in cases:
        got = student_rollout(c["turns"], pair)
        score, problem = grade(c["target"], got)
        log(f"{tag} {c['name']:<12} score={score:.3f}")
        log(f"  teacher: {c['target']!r}")
        log(f"  student: {got['text'][:160]!r}" + (f"  [tools: {got['tool_calls']}]" if got["tool_calls"] else ""))
        if score < 1.0:
            log(f"  problem: {problem}")


def draft_pair(best: SkillCoach) -> dict:
    out = best(spec=SPEC)
    return {k: str(getattr(out, k, "") or "").strip()
            for k in ("synth_user", "synth_content", "synth_reasoning")}


def score_pair(cases: list[dict], pair: dict) -> float:
    """Weighted mean student grade for one draft (real teacher pair counts triple)."""
    total, weight = 0.0, 0
    for c in cases:
        w = 3 if c["is_real"] else 1
        got = student_rollout(c["turns"], pair)
        s, _ = grade(c["target"], got)
        s = min(s, genericity([pair["synth_user"], pair["synth_content"], pair["synth_reasoning"]]))
        total, weight = total + s * w, weight + w
    return total / weight


def pick_best_pair(best: SkillCoach, cases: list[dict], n: int) -> dict:
    """The coach (teacher) is stochastic: draft N pairs, keep the best student score."""
    best_pair, best_avg = None, -1.0
    for i in range(1, n + 1):
        pair = draft_pair(best)
        avg = score_pair(cases, pair)
        log(f"draft {i}/{n} avg={avg:.3f}")
        log(f"  user:      {pair['synth_user'][:150]!r}")
        log(f"  assistant: {pair['synth_content'][:150]!r}")
        log(f"  reasoning: {pair['synth_reasoning'][:120]!r}")
        if avg > best_avg:
            best_avg, best_pair = avg, pair
    return best_pair


def main():
    global PREFIX, TOOLS
    logging.getLogger("dspy").setLevel(logging.INFO)
    cases = load_cases()
    s_msgs = json.load(open(STUDENT_FILE))
    PREFIX = s_msgs["messages"][:5]          # system + existing synthetic pairs (untouched)
    TOOLS = s_msgs["tools"]                  # tools are present in real deployment

    warmup(STUDENT, f"student {STUDENT_MODEL}")
    warmup(TEACHER, f"teacher {TEACHER_MODEL}")

    dspy.configure(lm=TEACHER)               # coach predictor uses the teacher

    if "--verify" in sys.argv:                # evaluate the saved pair only, no drafting
        pair = json.load(open(os.path.join(HERE, "synthetic-pair.json")))
        run_eval(cases, "VERIFY (saved pair)", pair=pair)
        return

    if "--select-only" in sys.argv:          # re-pick the pair from a saved program
        best = SkillCoach()
        saved = os.path.join(HERE, "coach-optimized.json")
        if os.path.exists(saved):
            best.load(saved)
        elif os.environ.get("WARM_START"):
            log("select-only: no saved program, using warm-start instruction")
        else:
            raise SystemExit("--select-only needs coach-optimized.json or WARM_START=1")
    else:
        examples = build_examples(cases)
        run_eval(cases, "BASELINE (no pair)")
        budget = 14 if "smoke" in sys.argv else int(os.environ.get("BUDGET", "180"))
        log(f"GEPA: budget={budget} metric calls, valset={len(examples)} examples")
        optimizer = dspy.GEPA(
            metric=coach_metric,
            max_metric_calls=budget,
            reflection_lm=TEACHER,           # teacher reflects on failures, rewrites instruction
            num_threads=1,  # shared server: one in-flight request at a time
            reflection_minibatch_size=3,
            log_dir=os.path.join(HERE, "gepa_log"),
            seed=42,
            gepa_kwargs={"use_cloudpickle": True},  # state-save must pickle the dynamic dspy signature
            track_stats=True,
            track_best_outputs=True,
        )
        best = optimizer.compile(SkillCoach(), trainset=examples)
        # Persist the state-only program (re-instantiate with SkillCoach().load(...))
        # and the winning coach instruction.
        best.save(os.path.join(HERE, "coach-optimized.json"))
        _, coach_pred = list(best.named_predictors())[0]  # the module has one predictor
        log(f"coach instruction:\n{coach_pred.signature.instructions}\n")
        open(os.path.join(HERE, "coach-instruction.txt"), "w").write(coach_pred.signature.instructions)

    n_drafts = 2 if "smoke" in sys.argv else int(os.environ.get("DRAFTS", "5"))
    pair = pick_best_pair(best, cases, n_drafts)
    log(f"SYNTHETIC PAIR (best of {n_drafts} drafts):")
    json.dump({"synth_user": pair["synth_user"],
               "synth_content": pair["synth_content"],
               "synth_reasoning": pair["synth_reasoning"]},
              open(os.path.join(HERE, "synthetic-pair.json"), "w"), indent=2)

    run_eval(cases, "FINAL (with pair)", pair=pair)


if __name__ == "__main__":
    main()
