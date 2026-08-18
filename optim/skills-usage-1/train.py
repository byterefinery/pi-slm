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
MAX_TRIALS = 16    # number of synthetic pairs to propose and evaluate
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
def propose_synthetics(propose_lm, n):
    """Ask a strong LM for `n` candidate synthetic user+assistant pairs that teach
    a small coding assistant the GENERAL rules for using the agent-skills system
    (so whatever skill is invoked it handles it correctly). The candidates must NOT
    mention the specific skill/answer being tested -- they must stay generic.
    Returns a list of {"user","assistant","reasoning"} dicts."""
    prompt = (
        "A small coding assistant (tools: read, write, edit, bash) is unreliable at "
        "using the Agent Skills system. I will insert ONE synthetic user turn and ONE "
        "synthetic assistant turn into its conversation, right before a skill is "
        "invoked, to teach it the GENERAL rules for using skills. After that it must "
        "correctly handle ANY skill, whatever skill is invoked.\n\n"
        "Facts about the Agent Skills system (from https://agentskills.io):\n"
        "- A skill is a folder with a SKILL.md file (YAML frontmatter with `name` and "
        "`description`, then a Markdown body of instructions). A skill may bundle "
        "scripts/, references/, assets/.\n"
        "- The system prompt lists available skills as name/description/location.\n"
        "- An INVOCATION arrives as a user message whose content is a `<skill name=... "
        "location=...>` block containing the loaded SKILL.md body, followed by the "
        "user's argument text (e.g. a mode, a target, a question). That message means "
        "the skill is now activated and the trailing text is its argument.\n"
        "- On activation the agent EXECUTES the skill for THIS turn: it takes the "
        "trailing user text and USES IT as the skill's argument (do not ignore it, do "
        "not substitute a default), then PRODUCES the reply the skill's body specifies "
        "for THAT argument (e.g. the exact activation/confirmation line the skill tells "
        "it to say, with the argument filled in). It does NOT explain, describe, or "
        "summarize the skill -- it does the action and emits the skill's own required "
        "output. It runs any bundled scripts the skill says to run.\n"
        "- The skill file is a fresh instruction for THIS turn. It overrides/answers "
        "the current request; the agent must NOT re-answer earlier unrelated questions "
        "from the conversation.\n"
        "- Relative paths inside a skill resolve against the skill's directory (the "
        "location given in the message).\n\n"
        f"Produce {n} DISTINCT candidate synthetic turns. Each has:\n"
        "- `user`: a short question the user could ask about how to use a skill.\n"
        "- `assistant`: a concise GENERAL, EXECUTION-ORIENTED rule set for when a skill "
        "is invoked. It must stress: recognize the <skill> block as an activation; the "
        "trailing text is the argument; DO the skill's action and EMIT the exact reply "
        "the skill's body requires for that argument (the skill's own format, e.g. its "
        "activation line) -- NOT an explanation or summary of the skill; and do not "
        "re-answer earlier questions. Keep it short and directive (imperative).\n"
        "- `reasoning`: the agent's short internal rationale for that rule.\n\n"
        "STRICT: do NOT name the specific skill being tested (no 'tzip'), do NOT give "
        "any specific answer such as a mode name like 'tzip full activated', and do NOT "
        "hard-code the tested output. Keep it fully generic so it works for any skill.\n"
        "Keep each field SHORT (one or two sentences each).\n\n"
        "Respond with ONLY a JSON object: {\"candidates\": [{\"user\": str, "
        "\"assistant\": str, \"reasoning\": str}, ...]}"
    )
    raw = propose_lm(prompt, temperature=1.0)  # fresh sampling -> diverse candidates
    text = _completion_text(raw)
    data = _parse_json(text)
    if data is None:
        log("!! could not parse proposal JSON; using built-in default candidates")
        data = {"candidates": DEFAULT_CANDIDATES[:n]}
    out = []
    for c in data.get("candidates", [])[:n]:
        u = str(c.get("user", "")).strip()
        a = str(c.get("assistant", "")).strip()
        r = str(c.get("reasoning", "")).strip()
        if not a:
            continue
        # Hard guard: never leak the specific skill or its answer into the rule.
        if "tzip" in (u + " " + a + " " + r).lower():
            continue
        out.append({"user": u, "assistant": a, "reasoning": r})
    return out or list(DEFAULT_CANDIDATES[:n])


# Fallback candidates: generic agent-skills rules (used if the LM proposal fails).
DEFAULT_CANDIDATES = [
    {
        "user": "How do I handle a skill when it is invoked in the conversation?",
        "assistant": "A user message that contains a <skill name=... location=...> "
                     "block means that skill's SKILL.md has just been loaded and its "
                     "body is the instruction for THIS turn. Follow the body exactly: "
                     "do what it says, reply in the exact format it specifies, and treat "
                     "any text after the skill block as the skill's argument. Do not "
                     "re-answer earlier questions in the conversation.",
        "reasoning": "The loaded SKILL.md is the authoritative instruction for the "
                     "current request; the trailing text is its argument, so the reply "
                     "follows the skill's own format, not prior turns.",
    },
    {
        "user": "What should I do when the skill's instructions differ from an earlier reply?",
        "assistant": "The invoked skill's SKILL.md body wins for the current turn. If it "
                     "says to run a bundled script (scripts/...), run it with the bash tool; "
                     "if it references files, resolve relative paths against the skill's "
                     "location. Answer only the current skill request in the skill's own "
                     "reply format.",
        "reasoning": "Skills use progressive disclosure: the full SKILL.md is loaded on "
                     "activation and is the instruction to execute now.",
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


def _parse_json(text):
    """Robustly extract a JSON object from a possibly-fenced/truncated LLM reply."""
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        candidates.append(fence.group(1))
    # largest brace-delimited substring
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        candidates.append(text[i:j + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


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
    # The teacher only proposes the synthetic text; give it a larger budget.
    teacher = make_lm(TEACHER_MODEL, api_base, api_key, provider, max_tokens=6144)

    # Baseline: NO synthetic messages (original request, student model).
    base_text, base_score = evaluate_candidate(student, prefix, invoke,
                                               {"user": "", "assistant": "", "reasoning": ""},
                                               teacher_answer)
    log(f"\nbaseline (no synthetic): {base_text!r}  score={base_score:.3f}")

    log("\n--- proposing synthetic turns (teacher LM, generic skill-usage rules) ---")
    candidates = propose_synthetics(teacher, MAX_TRIALS)

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
