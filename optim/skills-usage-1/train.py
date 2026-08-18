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

# Condensed Agent Skills specification (from https://agentskills.io, spec section).
# The teacher LM reads this to SELF-DERIVE the general skill-usage rules, keeping
# the synthetic teaching grounded in the real spec and fully generic.
_SPEC = """
Agent Skills -- a standardized, open format for extending AI agent capabilities.

A skill is a folder containing a SKILL.md file (required). SKILL.md has YAML
frontmatter (`name`, `description`, and optional fields) followed by a Markdown
body of instructions. A skill may bundle scripts/ (executable code), references/
(documents), and assets/ (templates/resources).

Progressive disclosure (how agents load skills):
1. Discovery: at startup the agent loads only each skill's `name` and `description`
   (from the system prompt's available-skills list).
2. Activation: when a task/request matches a skill, the agent loads the FULL
   SKILL.md body into context. In this harness an ACTIVATION arrives as a user
   message whose content is a `<skill name="..." location="...">` block holding the
   loaded SKILL.md body, followed by the user's argument text (e.g. a mode, target,
   or question).
3. Execution: the agent FOLLOWS the loaded SKILL.md body -- it does what the body
   instructs, in the exact reply format the body specifies, optionally running
   bundled scripts (scripts/...) or loading referenced files as needed.

Execution rules:
- On activation, the SKILL.md body is the authoritative instruction for THIS turn.
- The text after the <skill> block is the skill's ARGUMENT; apply it (e.g. select
  the requested mode/target). Do not ignore it and do not substitute a default when
  the argument explicitly selects an option.
- Produce the reply the skill's body requires for that argument (e.g. the exact
  activation/confirmation line the skill tells the agent to say) -- do not explain,
  summarize, or describe the skill.
- An activation is a fresh instruction for the current turn; do not re-answer
  earlier, unrelated questions from the conversation.
- Resolve relative paths inside a skill against the skill's directory (its location).
"""
MAX_TRIALS = 10      # distinct candidate rules drafted per proposal round
CANDIDATE_ROUNDS = 6 # proposal rounds (fresh sampling) to build the candidate pool
EVALS_PER_CANDIDATE = 2  # evaluate each candidate N times, keep the best (reliability)
THRESHOLD = 0.6    # (informational) keep a synthetic pair only if the student scores >= this

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


def judge(judge_lm, student_text, teacher_answer, invoke):
    """LLM-as-judge (teacher model): does the STUDENT reply match the TEACHER reply
    and correctly execute the invoked skill? Purely LLM-based -- no edit-distance.
    Returns (score 0..1, verdict)."""
    prompt = (
        "You are judging whether a weak model handled a skill invocation correctly.\n"
        "\nA coding agent was given this skill-invocation message (a <skill ...> block "
        "with the loaded SKILL.md body, followed by the user's argument text):\n"
        f"---\n{invoke}\n---\n\n"
        f"The CORRECT reply (from a strong model) is:\n---\n{teacher_answer}\n---\n\n"
        f"The weak model's reply is:\n---\n{student_text}\n---\n\n"
        "Judge the weak model's reply:\n"
        "- Does it correctly EXECUTE the skill for the user's argument (the option "
        "selected in the argument, not a default), in the reply format the skill "
        "specifies?\n"
        "- Does it match what the correct reply does?\n\n"
        "Respond with ONLY a JSON object: {\"score\": <0.0 to 1.0>, \"verdict\": "
        "<one short sentence>}. score=1.0 means it matches the correct reply and "
        "executes the skill for the right argument; lower scores for wrong option, "
        "an explanation of the skill, or re-answering earlier questions."
    )
    raw = judge_lm(prompt, temperature=0.0)
    data = _parse_json(_completion_text(raw))
    if data is None or "score" not in data:
        return 0.0, "judge parse failed"
    try:
        s = float(data["score"])
    except Exception:
        s = 0.0
    return max(0.0, min(1.0, s)), str(data.get("verdict", "")).strip()


# --------------------------------------------------------------------------- #
# Proposal + evaluation
# --------------------------------------------------------------------------- #
def _derive_skill_rules(propose_lm):
    """Stage 1: have the strong LM READ the agent-skills spec and SELF-DERIVE the
    general, execution-oriented rules for using a skill. This keeps the rules
    grounded in the real spec and generic (no hard-coding, no leak)."""
    prompt = (
        "Read this Agent Skills specification, then derive the GENERAL rules a coding "
        "agent must follow to correctly use and EXECUTE a skill once it is invoked. "
        "The rules must be fully generic (apply to ANY skill), and must be phrased as "
        "short imperative instructions a weak model can follow.\n\n"
        f"SPECIFICATION:\n---\n{_SPEC}\n---\n\n"
        "Pay special attention to: how a skill is discovered, activated, and executed; "
        "what an invocation message looks like (a <skill ...> block with the loaded "
        "SKILL.md body, followed by the user's argument text); that on activation the "
        "agent must EXECUTE the skill for that argument and produce the exact reply the "
        "SKILL.md body specifies (not an explanation/summary of the skill); that the "
        "invocation is a fresh instruction for the current turn (do not re-answer "
        "earlier questions); and how bundled scripts/references are used.\n\n"
        "Respond with ONLY a JSON object: {\"rules\": [str, ...]} where each str is one "
        "short, general, imperative rule. 5-8 rules. No specific skill names or "
        "specific answer examples."
    )
    raw = propose_lm(prompt, temperature=0.7)
    data = _parse_json(_completion_text(raw))
    rules = [str(r).strip() for r in (data or {}).get("rules", []) if str(r).strip()]
    return rules or list(FALLBACK_RULES)


def propose_synthetics(propose_lm, n, rules):
    """Stage 2: from the general `rules`, draft `n` DISTINCT synthetic user+assistant
    pairs (phrased many different ways) that teach a small coding assistant those
    rules, so whatever skill is invoked it handles it. Candidates must stay generic.
    Returns a list of {"user","assistant","reasoning"} dicts."""
    rule_block = "\n".join(f"- {r}" for r in rules)
    prompt = (
        "A small, weak coding assistant (tools: read, write, edit, bash) is unreliable "
        "at using skills. I will insert ONE synthetic user turn and ONE synthetic "
        "assistant turn into its conversation, right before a skill is invoked, to "
        "teach it the following GENERAL rules (derived from the Agent Skills spec):\n"
        f"{rule_block}\n\n"
        f"Write {n} DISTINCT candidate synthetic turns that teach these rules. Vary the "
        "phrasing, the angle, and the emphasis across candidates. Each has:\n"
        "- `user`: a short question the user could ask about using a skill.\n"
        "- `assistant`: the concise GENERAL rule the agent must follow on skill "
        "activation. It MUST be an imperative EXECUTION mandate, in this spirit: 'Treat "
        "the <skill> block as an activation. Take the text after it as the skill's "
        "argument. EXECUTE the skill for that argument -- fill in the argument and emit "
        "the exact reply the skill's body requires (the skill's own format/activation "
        "line). Do NOT explain, describe, or summarize the skill. If the argument selects "
        "a specific option, use THAT option, not a default. Do not re-answer earlier "
        "questions.' Vary the wording/angle per candidate but keep that core mandate.\n"
        "- `reasoning`: the agent's short internal rationale for the rule.\n\n"
        "STRICT: do NOT name the specific skill being tested, do NOT give any specific "
        "mode/answer example, do NOT hard-code the tested output. Fully generic.\n"
        "Keep each field short.\n\n"
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


# Generic, spec-grounded rules used if the stage-1 self-derivation fails.
FALLBACK_RULES = [
    "Recognize a user message containing a <skill name=... location=...> block as a skill ACTIVATION.",
    "The text after the <skill> block is the skill's ARGUMENT for this turn.",
    "EXECUTE the skill: apply the argument and produce the exact reply the SKILL.md body specifies for it.",
    "Emit the skill's own required output (e.g. its activation/confirmation line); do not explain, summarize, or describe the skill.",
    "When the argument selects a specific option, use that option -- do not fall back to a default.",
    "A skill activation is a fresh instruction for the current turn; do not re-answer earlier, unrelated questions.",
    "Run any bundled scripts the skill says to run, resolving relative paths against the skill's location.",
]


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


def evaluate_candidate(lm, judge_lm, prefix, invoke, syn, teacher_answer):
    """Run `lm` on the faithful request (originals + this synthetic pair), then let
    `judge_lm` (teacher, LLM-as-judge) score the reply against the teacher answer.
    `lm` is an LM instance whose `forward` merges its default kwargs (extra_body / "
    extra_headers) into the request."""
    adapter = InsertAdapter(prefix, invoke, syn)
    # format() ignores the signature (uses self.prefix/invoke/synthetic); a valid
    # 2-field signature just satisfies the signature contract.
    messages = adapter.format(dspy.Signature("query: str -> response: str"), [], {})
    raw = lm(messages=messages)
    text = _completion_text(raw)
    sc, verdict = judge(judge_lm, text, teacher_answer, invoke)
    return text, sc, verdict


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
    base_text, base_score, base_verdict = evaluate_candidate(
        student, teacher, prefix, invoke,
        {"user": "", "assistant": "", "reasoning": ""}, teacher_answer)
    log(f"\nbaseline (no synthetic): score={base_score:.3f}  verdict: {base_verdict}")
    log(f"  reply: {base_text!r}")

    log("\n--- stage 1: teacher self-derives general skill rules from the spec ---")
    rules = _derive_skill_rules(teacher)
    for r in rules:
        log(f"  - {r}")

    # Search: draft several distinct candidate rules; evaluate each one more than
    # once (best-of-N) so we keep a rule the student follows RELIABLY, not a lucky
    # one-shot hit. The student's temperature is low, so repeated evals are near-
    # deterministic and cheap to reason about.
    log(f"\n--- stage 2: drafting + evaluating {CANDIDATE_ROUNDS} rounds of "
        f"{MAX_TRIALS} candidates (best-of-{EVALS_PER_CANDIDATE}) ---")
    seen, candidates = set(), []
    for _ in range(CANDIDATE_ROUNDS):
        for syn in propose_synthetics(teacher, MAX_TRIALS, rules):
            key = _norm(syn["assistant"])
            if key and key not in seen:
                seen.add(key)
                candidates.append(syn)
        if len(candidates) >= MAX_TRIALS * 2:
            break
    log(f"unique candidates to evaluate: {len(candidates)}")

    def best_of(syn):
        """Score a synthetic rule best-of-N (keep the max over N student runs,
        judged by the teacher LLM)."""
        runs = [evaluate_candidate(student, teacher, prefix, invoke, syn, teacher_answer)
                for _ in range(EVALS_PER_CANDIDATE)]
        k = max(range(len(runs)), key=lambda j: runs[j][1])
        text, sc, verdict = runs[k]
        return sc, text, verdict

    best_syn, best_text, best_score, best_verdict = None, None, -1.0, ""
    for i, syn in enumerate(candidates, 1):
        sc, text, verdict = best_of(syn)
        log(f"\ncandidate {i}: best score={sc:.3f}  verdict: {verdict}")
        log(f"  assistant : {syn['assistant']!r}")
        log(f"  student   : {text!r}")
        if sc > best_score:
            best_syn, best_text, best_score, best_verdict = syn, text, sc, verdict

    log(f"\n--- best synthetic (rule the student follows most reliably) ---")
    log(f"user      : {best_syn['user']!r}")
    log(f"assistant : {best_syn['assistant']!r}")
    log(f"reasoning : {best_syn['reasoning']!r}")
    log(f"student reply : {best_text!r}")
    log(f"score : {best_score:.3f}  (baseline {base_score:.3f})  verdict: {best_verdict}")

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
