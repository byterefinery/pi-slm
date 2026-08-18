#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy"]
# ///
"""
Teach the STUDENT model to reliably INVOKE skills, using the TEACHER model.

The only thing we may change between two model calls is ONE synthetic user turn
and ONE synthetic assistant turn (content + reasoning) inserted right AFTER the
existing synthetic skills/tools pair and BEFORE the user's skill-invoke message.
Everything else -- the system message, the earlier synthetic turns, the skill
invoke message, and the sampling params -- stays exactly as-is.

Design (GEPA optimizer)
-----------------------
We build a thin DSPy module, `SkillResponder`, whose predictor instruction IS
the synthetic assistant message. GEPA (`dspy.GEPA`) evolves that instruction:

  * generator LM  = the STUDENT (the model being taught) -- it runs the faithful
                    request (originals + the evolved synthetic pair) and replies;
  * reflection LM = the TEACHER -- it reads the student's failing trace, the
                    skill context, and the teacher's correct answer (via the
                    metric's textual feedback) and proposes a better rule;
  * metric        = teacher-as-judge, returns Prediction(score, feedback).

So GEPA's reflective instruction rewriting searches for the synthetic assistant
rule that makes the *student* execute the invoked skill correctly (apply the
user's argument, emit the skill's required reply), in a general, non-leaky way.

Run:      uv run --script train.py     (progress -> train.log and terminal)
Outputs:  synthetic.json (the evolved synthetic user+assistant pair)
          gepa-run/      (GEPA state dir: candidates, scores, reflections)
          train.log      (full progress log)
"""

import difflib
import json
import os
import re
import sys
import time
import traceback

import dspy

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"
TEACHER_MODEL = "Qwen/Qwen3.8-27B"

TEACHER_FILE = os.path.join(HERE, "tzip-full-Qwen-Qwen3.8-27B.json")
STUDENT_FILE = os.path.join(HERE, "tzip-full-LiquidAI-LFM2.5-2.6B.json")
LOG_FILE = os.path.join(HERE, "train.log")
OUT_SYNTH = os.path.join(HERE, "synthetic.json")
GEPA_RUN_DIR = os.path.join(HERE, "gepa-run")

MAX_TOKENS = 2048       # the skill reply is short; keep student calls cheap
TEACHER_TOKENS = 4096   # room for the reflection LM to propose a new rule
EVALS_PER_CANDIDATE = 2  # (used for the final best-of recheck; GEPA does its own search)

# Fixed synthetic USER turn (the assistant turn + its reasoning are what GEPA evolves).
SYNTHETIC_USER = "When a skill is invoked in this conversation, how should I respond?"

# Condensed Agent Skills specification (from https://agentskills.io). Used to build
# a solid GENERIC seed for the synthetic rule; GEPA refines it from the traces.
_SPEC = (
    "Agent Skills: a skill is a folder with a SKILL.md (YAML frontmatter with name and "
    "description, then a Markdown body of instructions), optionally bundling scripts/, "
    "references/, assets/. Loading is progressive: (1) Discovery - the system prompt "
    "lists each skill's name/description; (2) Activation - when a request matches, the "
    "full SKILL.md body is loaded; in this harness an activation arrives as a user "
    "message containing a <skill name='...' location='...'> block with the loaded "
    "SKILL.md body, followed by the user's argument text (e.g. a mode, target, or "
    "question); (3) Execution - the agent follows the body: does what it instructs, in "
    "the exact reply format it specifies, applying the trailing argument (do not "
    "substitute a default when the argument selects an option), and runs bundled "
    "scripts as needed. An activation is a fresh instruction for the current turn: do "
    "not re-answer earlier, unrelated questions. Resolve relative paths against the "
    "skill's location."
)

SEED_RULE = (
    "A user message that contains a <skill name=... location=...> block means that "
    "skill is now activated: its SKILL.md body is the instruction for THIS turn, and "
    "the text after the block is the skill's argument. EXECUTE the skill for that "
    "argument - apply the argument (if it selects a specific option, use THAT option, "
    "not a default) and reply ONLY with the exact response the SKILL.md body requires "
    "for it (the skill's own format, e.g. its activation/confirmation line). Do not "
    "explain, describe, or summarize the skill, and do not re-answer earlier questions."
)


# --------------------------------------------------------------------------- #
# Logging (progress to both terminal and train.log)
# --------------------------------------------------------------------------- #
_log_fh = open(LOG_FILE, "a")
_log_fh.write(f"\n========== train.py start {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
_log_fh.flush()


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


# --------------------------------------------------------------------------- #
# Config: read the llamacpp (OpenAI-compatible) provider settings
# --------------------------------------------------------------------------- #
def _provider():
    """models.json may have trailing commas; strip them so json can parse it."""
    raw = open(MODELS_JSON).read()
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))["providers"]["llamacpp"]


def _load_provider():
    p = _provider()
    return p["baseUrl"], p["apiKey"]


def _model_cfg(provider, model_id):
    for m in provider["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(model_id)


def _sampling(model):
    """Map pi samplingParams -> llama.cpp-compatible request params.
    `presence_penalty`/`frequency_penalty` are OpenAI-only (rejected by
    llama.cpp), so they are omitted. `repeat_penalty`/`top_k`/`min_p` go in the
    body so llama.cpp applies them; `temperature` is a native dspy.LM arg."""
    sp = model.get("samplingParams", {})
    body = {k: sp[k] for k in ("top_k", "min_p", "repeat_penalty") if k in sp}
    return sp.get("temperature"), body


def make_lm(model_id, api_base, api_key, provider, max_tokens=MAX_TOKENS):
    """Build a dspy.LM for a model id using its sampling params from models.json."""
    temperature, body = _sampling(_model_cfg(provider, model_id))
    kwargs = dict(
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={"x-session-affinity": "dspy-optim"},
    )
    if body:
        kwargs["extra_body"] = body  # merged into the request payload
    return dspy.LM(f"openai/{model_id}", **kwargs)


# --------------------------------------------------------------------------- #
# Data: load the ORIGINAL messages (never mutated) + the two answers
# --------------------------------------------------------------------------- #
def load_data():
    """Return (prefix_messages, invoke_message, teacher_answer, student_answer).

    prefix = the original messages EXCLUDING the final user+assistant pair
             (system + the earlier synthetic skills/tools turns)
    invoke = the final user message (the skill invocation)
    """
    t = json.load(open(TEACHER_FILE))["messages"]
    s = json.load(open(STUDENT_FILE))["messages"]

    if t[-1]["role"] != "assistant" or s[-1]["role"] != "assistant":
        raise SystemExit("expected the last message to be the assistant reply in both dumps")
    if t[-2]["role"] != "user" or s[-2]["role"] != "user":
        raise SystemExit("expected the second-to-last message to be the user (skill invoke)")

    def _flatten(c):
        if isinstance(c, list):
            return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""

    prefix = [{"role": m["role"], "content": _flatten(m.get("content"))} for m in t[:-2]]
    return prefix, _flatten(t[-2]["content"]), t[-1]["content"], s[-1]["content"]


def _completion_text(raw):
    """Extract the assistant text from a dspy LM call result."""
    if isinstance(raw, list):
        for part in raw:
            if isinstance(part, dict):
                return (part.get("text") or part.get("content") or "").strip()
        return str(raw).strip()
    if isinstance(raw, dict):
        return (raw.get("text") or raw.get("content") or "").strip()
    return str(raw).strip()


def _parse_json(text):
    """Robustly extract a JSON object from a possibly-fenced/truncated LLM reply."""
    cands = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        cands.append(fence.group(1))
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# Adapter: send the ORIGINAL messages, inserting ONE synthetic user+assistant
# pair right before the skill-invoke user message. The predictor's instruction
# (evolved by GEPA) becomes the synthetic assistant content; the predictor's
# `reasoning` output field becomes the assistant reasoning_content.
# --------------------------------------------------------------------------- #
class SkillInsertAdapter(dspy.ChatAdapter):
    def __init__(self, prefix, invoke):
        super().__init__()
        self.prefix = prefix
        self.invoke = invoke

    def format(self, signature, demos, inputs):
        rule = signature.instructions            # the synthetic ASSISTANT message
        reasoning = inputs.get("reasoning")      # the synthetic ASSISTANT reasoning
        msgs = list(self.prefix)
        msgs.append({"role": "user", "content": SYNTHETIC_USER})
        assistant = {"role": "assistant", "content": rule}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        msgs.append(assistant)
        msgs.append({"role": "user", "content": self.invoke})
        return msgs

    def parse(self, signature, completion):
        """The raw completion IS the reply (no ChatAdapter section markers)."""
        return {
            "reasoning": "",
            "response": (completion or "").strip(),
        }


# --------------------------------------------------------------------------- #
# Metric: teacher-as-judge, returns score + textual feedback (for GEPA)
# --------------------------------------------------------------------------- #
def judge(judge_lm, student_text, teacher_answer, invoke):
    """LLM-as-judge (teacher): does the student reply execute the skill for the
    requested argument and match the teacher? Returns (score, feedback)."""
    prompt = (
        "You are judging whether a weak model handled a skill invocation correctly.\n"
        "A coding agent was given this skill-invocation message (a <skill ...> block "
        "with the loaded SKILL.md body, followed by the user's argument text):\n"
        f"---\n{invoke}\n---\n\n"
        f"The CORRECT reply (from a strong model) is:\n---\n{teacher_answer}\n---\n\n"
        f"The weak model's reply is:\n---\n{student_text}\n---\n\n"
        "Judge the weak model's reply: does it EXECUTE the skill for the user's "
        "argument (the option the argument selects, not a default) in the reply "
        "format the skill specifies, matching the correct reply?\n"
        "Respond with ONLY a JSON object: {\"score\": <0.0 to 1.0>, "
        "\"feedback\": <2-4 sentences: what the correct reply does, what the weak "
        "reply did wrong (wrong option / explained the skill / re-answered earlier "
        "questions / wrong format), and the general principle that would fix it>}."
    )
    raw = judge_lm(prompt, temperature=0.0)
    data = _parse_json(_completion_text(raw))
    if data is None or "score" not in data:
        return 0.0, "Judge could not be parsed; treating the reply as a failure."
    try:
        s = float(data["score"])
    except Exception:
        s = 0.0
    return max(0.0, min(1.0, s)), str(data.get("feedback", "")).strip()


# --------------------------------------------------------------------------- #
# The DSPy program: a single predictor whose instruction is the synthetic rule.
# GEPA evolves `respond.signature.instructions`.
# --------------------------------------------------------------------------- #
class _ResponderSig(dspy.Signature):
    """The predictor's instruction (set per-instance) is the synthetic assistant
    message. `reasoning` is the assistant's reasoning_content; `response` is the
    reply to the skill invoke."""
    reasoning: dspy.Reasoning = dspy.InputField()
    response: str = dspy.OutputField()


class SkillResponder(dspy.Module):
    def __init__(self, rule=SEED_RULE):
        super().__init__()
        sig = _ResponderSig.with_instructions(rule)
        self.respond = dspy.Predict(sig)

    def forward(self):
        # The adapter reads everything it needs (instruction + the empty `reasoning`
        # input) and ignores the rest; the reply is the raw completion text.
        return self.respond(reasoning="")


# --------------------------------------------------------------------------- #
# Run a program once (student as generator) and judge the reply
# --------------------------------------------------------------------------- #
def run_and_judge(program, student, judge_lm, prefix, invoke, teacher_answer,
                  rollout_id=None):
    kwargs = {}
    if rollout_id is not None:
        kwargs = {"rollout_id": rollout_id}
    with dspy.context(lm=student, adapter=SkillInsertAdapter(prefix, invoke),
                      **kwargs):
        pred = program()
    text = pred.response if hasattr(pred, "response") else str(pred)
    sc, feedback = judge(judge_lm, text, teacher_answer, invoke)
    return text, sc, feedback


# --------------------------------------------------------------------------- #
# GEPA metric: runs the student, returns Prediction(score, feedback)
# --------------------------------------------------------------------------- #
def make_gepa_metric(student, judge_lm, prefix, invoke, teacher_answer):
    # GEPA's metric must accept (gold, pred, trace, pred_name, pred_trace).
    def metric(gold, pred, trace, pred_name, pred_trace):
        # `pred` is the candidate program (with GEPA's evolved instruction).
        text, sc, feedback = run_and_judge(
            pred, student, judge_lm, prefix, invoke, teacher_answer)
        return dspy.Prediction(
            score=sc,
            feedback=(
                f"Skill invocation: the user invoked a skill; the correct reply is "
                f"{teacher_answer!r}. Student replied {text!r}. Judge: {feedback}"
            ),
        )
    return metric


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    api_base, api_key = _load_provider()
    provider = _provider()

    log(f"API base : {api_base}")
    log(f"student  : {STUDENT_MODEL}")
    log(f"teacher  : {TEACHER_MODEL}")

    prefix, invoke, teacher_answer, student_answer = load_data()
    log("\n--- data ---")
    log(f"teacher answer: {teacher_answer!r}")
    log(f"student answer: {student_answer!r}")
    log(f"invoke (tail): ...{invoke[-60:]!r}")
    log(f"prefix messages: {len(prefix)}  (system + {len(prefix) - 1} turns)")
    log(f"synthetic user (fixed): {SYNTHETIC_USER!r}")

    student = make_lm(STUDENT_MODEL, api_base, api_key, provider)
    teacher = make_lm(TEACHER_MODEL, api_base, api_key, provider, max_tokens=TEACHER_TOKENS)

    example = dspy.Example(skill=invoke, teacher=teacher_answer).with_inputs("skill")
    trainset = [example]

    # Baseline: student with the seed (generic) rule.
    base_text, base_score, base_fb = run_and_judge(
        SkillResponder(), student, teacher, prefix, invoke, teacher_answer)
    log(f"\nbaseline (seed rule): score={base_score:.3f}  reply: {base_text!r}")
    log(f"  feedback: {base_fb}")

    metric = make_gepa_metric(student, teacher, prefix, invoke, teacher_answer)

    log(f"\n--- running GEPA optimizer "
        f"(generator={STUDENT_MODEL}, reflection={TEACHER_MODEL}) ---")
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=teacher,
        max_metric_calls=25,
        reflection_minibatch_size=1,
        num_threads=1,
        use_merge=False,
        warn_on_score_mismatch=False,
    )
    optimized = optimizer.compile(
        SkillResponder(rule=SEED_RULE), trainset=trainset, valset=trainset)

    evolved_rule = optimized.respond.signature.instructions
    log("\n--- GEPA done ---")
    log(f"evolved synthetic assistant rule:\n{evolved_rule}")

    # Final recheck of the evolved rule (best of N) with a clear before/after.
    log(f"\n--- final recheck (best-of-{EVALS_PER_CANDIDATE}) ---")
    runs = [run_and_judge(optimized, student, teacher, prefix, invoke,
                          teacher_answer, rollout_id=k)
            for k in range(EVALS_PER_CANDIDATE)]
    k = max(range(len(runs)), key=lambda j: runs[j][1])
    best_text, best_score, best_fb = runs[k]
    for i, (t, s, fb) in enumerate(runs):
        log(f"  run {i}: score={s:.3f}  reply: {t!r}")
    log(f"\nbest reply : {best_text!r}")
    log(f"best score : {best_score:.3f}  (baseline {base_score:.3f})")
    log(f"feedback   : {best_fb}")

    # Assemble the deliverable: the synthetic pair + the full request.
    final_msgs = list(prefix)
    final_msgs.append({"role": "user", "content": SYNTHETIC_USER})
    # The synthetic assistant's reasoning is the student's (best) internal
    # reasoning when it produced the reply under this rule.
    assistant = {"role": "assistant", "content": evolved_rule}
    if runs[k][0]:
        assistant["reasoning_content"] = _best_reasoning(optimized, student,
                                                         prefix, invoke, teacher_answer)
    final_msgs.append(assistant)
    final_msgs.append({"role": "user", "content": invoke})

    with open(OUT_SYNTH, "w") as f:
        json.dump({
            "synthetic_user": SYNTHETIC_USER,
            "synthetic_assistant": evolved_rule,
            "final_request": final_msgs,
            "baseline_score": round(base_score, 3),
            "best_score": round(best_score, 3),
        }, f, indent=2)
    log(f"\nsaved -> {OUT_SYNTH}")

    log("\n--- final request (originals + inserted synthetic) ---")
    for m in final_msgs:
        body = m["content"]
        tag = " [reasoning]" if m.get("reasoning_content") else ""
        log(f"  [{m['role']}]{tag} {body[:120].replace(chr(10), ' ')}"
            + ("..." if len(body) > 120 else ""))

    log("\n--- diff (student before -> after) ---")
    diff = difflib.unified_diff(
        base_text.splitlines(), best_text.splitlines(),
        fromfile="student (before)", tofile="student (after)", lineterm="")
    log("\n".join(diff) if list(diff) else "(no difference)")

    log("\ntrain.py done")


def _best_reasoning(program, student, prefix, invoke, teacher_answer):
    """Capture the student's internal reasoning (reasoning_content) when it answers
    under the evolved rule (for the synthetic assistant's reasoning field)."""
    with dspy.context(lm=student, adapter=SkillInsertAdapter(prefix, invoke)):
        raw = student(messages=SkillInsertAdapter(prefix, invoke).format(
            program.respond.signature, [], {}))
    r = raw[0].get("reasoning_content", "") if isinstance(raw, list) and raw else ""
    return r or ""


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n!! train.py failed !!")
        traceback.print_exc()
        _log_fh.flush()
        sys.exit(1)
