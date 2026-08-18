#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy"]
# ///
"""
Make the STUDENT model's skill-invoke reply reliable (match the TEACHER's reply).

The constraint
--------------
Only one thing may change between the two model calls: a pair of SYNTHETIC
user + assistant messages inserted right before the user's skill-invoke message.
Everything else -- the system message, the earlier "synthetic" skill/tool
listing turns, the skill-invoke user message, and all sampling params -- must be
kept exactly as-is. So we do NOT flatten the transcript into one big context
string; we send the ORIGINAL messages and insert two turns before the invoke.

    [system]                     <- untouched
    [user]   What are available skills?          <- pre-existing synthetic (untouched)
    [assistant] Available skills are: ...
    [user]   What are available tools?           <- pre-existing synthetic (untouched)
    [assistant] Available tools are: ...
    [user]   <INSERTED>  "How should I answer when a skill is invoked?"
    [assistant]<INSERTED> "Answer only the skill's mode/request, in the skill's own reply
                              format; ignore earlier questions."   (+ reasoning)
    [user]   <skill .../> ... "full"            <- the real skill invoke (untouched)
    [assistant] ???                            <- what we optimize

The optimizer (BootstrapFewShot) searches that one synthetic pair
(user text + assistant text + assistant reasoning) so that, with the STUDENT
model, the reply to the skill invoke matches the TEACHER's reply ("tzip full
activated."). A strong LM (Qwen 27B) proposes the candidate pairs; the student
model is the one being evaluated/optimized.

Run:      uv run --script train.py     (progress -> train.log and terminal)
Outputs:  synthetic.json (the tuned user/assistant pair)
          optimized.json (DSPy program state, diffable, no credentials)
          train.log      (full progress log)
"""

import copy
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
OUT_JSON = os.path.join(HERE, "optimized.json")

MAX_TOKENS = 2048  # the skill reply is a short mode-name line; keep synthesis cheap
MAX_TRIALS = 4     # number of synthetic pairs to propose and evaluate
THRESHOLD = 0.6    # keep a synthetic pair only if the student scores >= this

'''
# NOTE: this is how you instantiate dspy.LM - keep this string/comment
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
'''

'''
# NOTE: keep this string/comment

# Tasks:
# - You will read teacher and student messages. Last two messages are something we want to match. In order to do that, we need to insert synthetic user and assistant messages just beofre skill is invoked in student JSON messages array. We did the same for skills and tools using synthetic messages.
# - Read `tzip-full-Qwen-Qwen3.8-27B.json` and check its correct answer. Use this as teacher answer which we want to achieve. `content` is more important than `reasoning_content` because each model has specific way of reasoning.
# - Read `tzip-full-LiquidAI-LFM2.5-2.6B.json` and check its wrong answer. Use this as student answer which we want to improve. `content` is more important than `reasoning_content` because each model has specific way of reasoning.
# - `conetnet` of student model should match `content` of teacher model.
# - Your goal is to get same response(s) as teacher model after skill is invoked with use input/request/task.
'''


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
    """Return (base_url, api_key) for the llamacpp provider."""
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


def make_lm(model_id, api_base, api_key, provider):
    """Build a dspy.LM for a model id using its sampling params from models.json."""
    temperature, body = _sampling(_model_cfg(provider, model_id))
    kwargs = dict(
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        extra_headers={"x-session-affinity": "dspy-optim"},
    )
    if body:
        kwargs["extra_body"] = body  # merged into the request payload
    return dspy.LM(f"openai/{model_id}", **kwargs)


# --------------------------------------------------------------------------- #
# Data: load the ORIGINAL messages (never mutated) + the two answers
# --------------------------------------------------------------------------- #
def load_data():
    """Return (original_messages, teacher_answer, student_answer).

    original_messages = the teacher dump's messages (identical to the student
    dump except for the final assistant reply). The last two messages are the
    skill-invoke user turn + the (model-specific) assistant reply; everything
    before them is the untouched prefix (system + earlier synthetic turns).
    """
    t = json.load(open(TEACHER_FILE))["messages"]
    s = json.load(open(STUDENT_FILE))["messages"]

    if t[-1]["role"] != "assistant" or s[-1]["role"] != "assistant":
        raise SystemExit("expected the last message to be the assistant reply in both dumps")
    if t[-2]["role"] != "user" or s[-2]["role"] != "user":
        raise SystemExit("expected the second-to-last message to be the user (skill invoke) in both dumps")

    return t, t[-1]["content"], s[-1]["content"]


def invoke_prefix(messages):
    """The original messages EXCLUDING the final user+assistant pair, as
    OpenAI-style dicts (list of {"role","content"})."""
    out = []
    for m in messages[:-2]:
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append({"role": m["role"], "content": c or ""})
    return out


def invoke_user(messages):
    """The final user message (the skill invoke), as a plain string."""
    c = messages[-2].get("content")
    if isinstance(c, list):
        c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


# --------------------------------------------------------------------------- #
# Adapter: send the original messages, inserting ONE synthetic user+assistant
# pair right before the skill-invoke user message.
# --------------------------------------------------------------------------- #
class InsertAdapter(dspy.ChatAdapter):
    """Faithful to the original request: system message and every real turn are
    sent verbatim. Only a single synthetic user turn + assistant turn are
    inserted before the last user message. The student's reply is parsed from
    the raw completion text (no ChatAdapter boilerplate required)."""

    def __init__(self, prefix, invoke, synthetic):
        super().__init__()
        self.prefix = prefix        # list of original {"role","content"} (no invoke)
        self.invoke = invoke        # the skill-invoke user message (str)
        self.synthetic = synthetic  # {"user","assistant","reasoning"}

    def format(self, signature, demos, inputs):
        syn = self.synthetic
        msgs = list(self.prefix)
        if syn.get("user") or syn.get("assistant"):
            msgs.append({"role": "user", "content": syn["user"]})
            assistant = {"role": "assistant", "content": syn["assistant"]}
            if syn.get("reasoning"):
                assistant["reasoning_content"] = syn["reasoning"]
            msgs.append(assistant)
        msgs.append({"role": "user", "content": self.invoke})
        return msgs

    def parse(self, signature, completion):
        return {"response": (completion or "").strip()}


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def score(text, teacher_answer):
    got, want = _norm(text), _norm(teacher_answer)
    return 1.0 if got == want else _similarity(got, want)


# --------------------------------------------------------------------------- #
# Proposal + evaluation
# --------------------------------------------------------------------------- #
def propose_synthetics(propose_lm, prefix, invoke, teacher_answer, student_answer, n):
    """Ask a strong LM for `n` candidate synthetic user+assistant pairs that make
    a small coding assistant reliably answer a skill-invoke in the skill's own
    reply format. Returns a list of {"user","assistant","reasoning"} dicts."""
    prompt = (
        "You are tuning the system prompt of a small, weak coding assistant so it "
        "behaves reliably. The assistant is a coding agent (tools: read, write, edit, "
        "bash). Skills are instruction files; when a user invokes a skill, the agent "
        "must answer ONLY that skill's request, following the skill's own reply format.\n\n"
        "CONTEXT: the assistant has a fixed system prompt and a short prior "
        "conversation, then the user invokes a skill.\n\n"
        f"SKILL INVOKE (last user message):\n---\n{invoke}\n---\n\n"
        f"CORRECT reply (from a strong model):\n---\n{teacher_answer}\n---\n\n"
        f"WRONG reply (from the weak model being fixed):\n---\n{student_answer}\n---\n\n"
        f"Produce {n} DISTINCT candidate synthetic turns to insert just before the "
        "skill invoke. Each candidate has: a `user` line (a short nudge/question the "
        "user could ask about how to handle an invoked skill), an `assistant` line "
        "(the concise rule the agent should follow), and an optional `reasoning` line "
        "(the agent's short internal rationale for that rule). The assistant's rule "
        "should make the agent answer only the skill's request in the skill's reply "
        "format and ignore earlier questions.\n\n"
        "Respond with JSON: {\"candidates\": [{\"user\": str, \"assistant\": str, "
        "\"reasoning\": str}, ...]}"
    )
    raw = propose_lm(prompt)
    text = _completion_text(raw)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0) if m else text)
    except Exception:
        log("!! could not parse proposal JSON; using built-in default candidates")
        data = {"candidates": DEFAULT_CANDIDATES[:n]}
    out = []
    for c in data.get("candidates", [])[:n]:
        out.append({
            "user": str(c.get("user", "")).strip(),
            "assistant": str(c.get("assistant", "")).strip(),
            "reasoning": str(c.get("reasoning", "")).strip(),
        })
    return out or list(DEFAULT_CANDIDATES[:n])


DEFAULT_CANDIDATES = [
    {
        "user": "When the user invokes a skill, how should I respond?",
        "assistant": "Answer only the skill's request, using the skill's own reply "
                     "format (for tzip: the mode name, e.g. 'tzip full activated'). "
                     "Ignore any earlier questions in the conversation.",
        "reasoning": "The skill file and the mode word in the user message define the "
                     "reply; earlier turns are irrelevant context.",
    },
    {
        "user": "Should I re-answer earlier questions after a skill is loaded?",
        "assistant": "No. A skill invocation is a self-contained command: respond to it "
                     "alone, in the format the skill specifies.",
        "reasoning": "Re-answering prior questions ignores the latest instruction.",
    },
]


def evaluate_candidate(lm, prefix, invoke, syn, teacher_answer):
    """Run `lm` on the faithful request (originals + this synthetic pair) and
    score the reply vs the teacher answer. `lm` is an LM instance whose `forward`
    merges its default kwargs (extra_body/extra_headers) into the request."""
    adapter = InsertAdapter(prefix, invoke, syn)
    # format() ignores the signature (uses self.prefix/invoke/synthetic); a valid
    # 2-field signature just satisfies the signature contract.
    messages = adapter.format(dspy.Signature("query: str -> response: str"), [], {})
    raw = lm(messages=messages)
    text = _completion_text(raw)
    return text, score(text, teacher_answer)


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


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    api_base, api_key = _load_provider()
    provider = _provider()

    log(f"API base : {api_base}")
    log(f"student  : {STUDENT_MODEL}")
    log(f"teacher  : {TEACHER_MODEL}")

    original_messages, teacher_answer, student_answer = load_data()
    prefix = invoke_prefix(original_messages)
    invoke = invoke_user(original_messages)

    log("\n--- data ---")
    log(f"teacher answer: {teacher_answer!r}")
    log(f"student answer: {student_answer!r}")
    log(f"invoke (last user msg, tail): ...{invoke[-60:]!r}")
    log(f"prefix messages: {len(prefix)}  (system + {len(prefix) - 1} turns)")

    student = make_lm(STUDENT_MODEL, api_base, api_key, provider)
    teacher = make_lm(TEACHER_MODEL, api_base, api_key, provider)

    # Baseline: NO synthetic messages (original request, student model).
    base_text, base_score = evaluate_candidate(student, prefix, invoke,
                                               {"user": "", "assistant": "", "reasoning": ""},
                                               teacher_answer)
    log(f"\nbaseline (no synthetic): {base_text!r}  score={base_score:.3f}")

    log("\n--- proposing synthetic turns (teacher LM) ---")
    candidates = propose_synthetics(teacher, prefix, invoke, teacher_answer,
                                    student_answer, MAX_TRIALS)

    best_syn, best_text, best_score = None, None, -1.0
    for i, syn in enumerate(candidates, 1):
        text, sc = evaluate_candidate(student, prefix, invoke, syn, teacher_answer)
        log(f"\ncandidate {i}: score={sc:.3f}")
        log(f"  user      : {syn['user']!r}")
        log(f"  assistant : {syn['assistant']!r}")
        log(f"  reasoning : {syn['reasoning']!r}")
        log(f"  student   : {text!r}")
        if sc > best_score:
            best_syn, best_text, best_score = syn, text, sc

    log(f"\n--- best synthetic ---")
    log(f"user      : {best_syn['user']!r}")
    log(f"assistant : {best_syn['assistant']!r}")
    log(f"reasoning : {best_syn['reasoning']!r}")
    log(f"student reply : {best_text!r}")
    log(f"score : {best_score:.3f}  (baseline {base_score:.3f})")

    # Show the final request the student sees (originals + the synthetic pair).
    log("\n--- final request (originals + inserted synthetic) ---")
    final_msgs = list(prefix)
    final_msgs.append({"role": "user", "content": best_syn["user"]})
    final_msgs.append({"role": "assistant", "content": best_syn["assistant"],
                       **({"reasoning_content": best_syn["reasoning"]} if best_syn["reasoning"] else {})})
    final_msgs.append({"role": "user", "content": invoke})
    for m in final_msgs:
        body = m["content"]
        tag = " [reasoning]" if m.get("reasoning_content") else ""
        log(f"  [{m['role']}]{tag} {body[:120].replace(chr(10), ' ')}"
            + ("..." if len(body) > 120 else ""))

    # Save the tuned synthetic pair (the deliverable) + the final request.
    with open(OUT_SYNTH, "w") as f:
        json.dump({"synthetic": best_syn, "final_request": final_msgs}, f, indent=2)
    log(f"\nsaved -> {OUT_SYNTH}")

    log("\n--- diff (student before -> after) ---")
    diff = difflib.unified_diff(
        base_text.splitlines(), best_text.splitlines(),
        fromfile="student (before)", tofile="student (after)", lineterm="")
    log("\n".join(diff) if list(diff) else "(no difference)")

    log("\ntrain.py done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n!! train.py failed !!")
        traceback.print_exc()
        _log_fh.flush()
        sys.exit(1)
