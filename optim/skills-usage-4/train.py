#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///

'''
Train the student LM (LiquidAI/LFM2.5-2.6B, thinking ON) with the teacher LM
(Qwen/Qwen3.8-27B, thinking OFF) on the pi skill-system step-1 pair, against the
live failing context. Only the step-1 teaching pair may change:
    user:        "How does skill system work? When a skill block is in my latest message, what do I do?"
    reasoning:   (assistant reasoning_content)
    content:     (assistant content)
Everything else (system prompt, skills/tools Q&A, the three `example` one-shots,
the tzip <skill> block) is frozen. Skill invocations are placed AFTER the
example few-shots, exactly as in the live session.

The two defects being fixed (both reported by the user, both stochastic):
1. ACTIVATION FAILURE — when a skill is invoked directly in a mode (especially
   the bare argument `full`), the student usually needs many attempts from
   scratch to get it right; mode-to-mode transitions also fail (the previous
   mode is repeated, the previous call is answered, or the reply is empty).
   -> the metric scores each test on TWO independent student rollouts and takes
   the MIN, so a stochastic failure on either sample fails the item.
2. CROSS-SKILL LEAK — context from previous skill invocations (the frozen
   `example` one-shots) leaks into a fresh, independent skill invocation: the
   student's reasoning drags in the example skill's name/argument/rules and the
   final reply can be wrong.
   -> every sample's reasoning is leak-checked; a correct reply whose reasoning
   mixes in the previous skill is capped at 0.5.

Invocation form (user-specified): the skill block is followed by a BARE
argument — `full`, `lite`, `ultra`, `off`, `on`, or nothing (default). NEVER
"tzip full"-style prefixed calls. This applies to direct activations AND to
transition setups.

Teacher (Qwen/Qwen3.8-27B, thinking OFF, DEFAULT sampling params from
models.json) is used for all of: judging each student reply against the tzip
Usage rules, generating the ground-truth tzip replies, and generating the
tzip few-shot `reasoning_content` entries in the student model's own voice,
following REASONING-*.md.

Test matrix (37 items, all bare arguments):
- 11 DIRECT activations from a fresh start: '' x2, 'on', 'lite', 'full' x3
  (the most frequent live failure), 'ultra' x2, 'off' x2
- 20 TRANSITIONS: every mode -> every other mode (setup = bare-argument
  invocation of the start mode + its canonical confirmation)
- 6 LIVE consecutive-invocation patterns, incl. the exact live failure
  (bare block -> 'tzip lite activated', then bare block + 'full')

Flow (all logged to stdout -> train.log):
1.  connectivity probes (teacher + student)
2.  teacher ground-truth tzip sequence (replies + LFM-voice reasoning)
3.  write skill-example-tzip-*.json with the SEED (quoted) pair — inspectable
4.  BASELINE: quoted seed / original seed / V2 (slm.ts) on the matrix (N=1)
5.  GEPA: reflection LM = teacher, multi-sample min-of-2 metric, budget 500
6.  FINAL: N=3 multi-sample validation — best vs quoted vs original + leak rates
7.  random walks (chained on the student's own replies) + robustness probes
8.  write the final artifact (best pair) + step1-pair-optimized.json (meta)

HuggingFace chat-template playground compatibility: LFM2.5-2.6B's
chat_template.jinja raises "Tool call arguments must be a mapping" when tool
arguments are JSON-encoded strings, so after every write we render the whole
document with the real template (runs-0/lfm25-chat-template.jinja) — a
successful render proves the file loads in the playground.

API credentials: read at runtime from ~/.pi/agent/models.json (provider
`llamacpp`), masked everywhere; never written to any file, log, or state.
'''

import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import dspy
from gepa import NoImprovementStopper, ScoreThresholdStopper, TimeoutStopCondition
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

HERE = Path(__file__).resolve().parent
PI_ROOT = HERE.parents[1]  # /home/mtasic/projects-b/pi-slm
RUNS0 = HERE / "runs-0"  # previous-run artifacts (real context, chat template, seed pairs)

MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
EXAMPLE_FILE = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"  # tools list source (OpenAI format)
TZIP_FILE = HERE / "skill-example-tzip-LiquidAI-LFM2.5-2.6B.json"
REAL_CTX_FILE = RUNS0 / "real-base-context.json"  # the REAL pi-session wire context (token-verified)
PAIR_FILE = HERE / "step1-pair-optimized.json"
TEMPLATE = RUNS0 / "lfm25-chat-template.jinja"  # LFM2.5-2.6B chat template (HF playground compat check)
REASONING_MD = HERE / "REASONING-LiquidAI-LFM2.5-2.6B.md"
RUN_DIR = HERE / "gepa_runs" / "skills-usage-4"

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"  # thinking: ON
TEACHER_MODEL = "Qwen/Qwen3.8-27B"  # thinking: OFF
PROVIDER = "llamacpp"
SESSION_HEADER = {"x-session-affinity": "dspy-optim"}

MAX_METRIC_CALLS = 500
GEPA_TIMEOUT_S = 9000
MINI_BATCH = 10
PARALLEL = 2
N_BASELINE = 1  # orientation only
N_GEPA = 2  # min-of-N metric: a stochastic failure on either sample fails the item
N_FINAL = 3  # go/no-go evidence
STUDENT_MAX_TOKENS = 16384
TEACHER_MAX_TOKENS = 2048

API_BASE = None
API_KEY = None


def say(*parts):
    """Print with a timestamp, API_BASE/API_KEY masked (never leak them)."""
    s = " ".join(str(p) for p in parts)
    for secret in (API_BASE, API_KEY):
        if secret:
            s = s.replace(secret, "***")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)


# ---------------------------------------------------------------------------
# the step-1 pair: the ONLY editable region of the prompt
# ---------------------------------------------------------------------------

# SEED — the pair quoted by the user as "this part of the prompt" (the one that
# currently ships in skill-example-tzip-*.json). GEPA optimizes from here.
SEED_PAIR = {
    "step1_user": "How does skill system work? When a skill block is in my latest message, what do I do?",
    "step1_reasoning": (
        "I treat each skill block as a fresh, self-contained call: I read only this block's name, Usage rules, and "
        "the argument text after its closing tag, and ignore everything before it. If a different mode is already "
        "active, I still derive the reply only from this argument \u2014 I never repeat the old confirmation. If the "
        "previous message ended with a short word from another finished skill, that word is not this argument; the "
        "argument is exactly the text after this block's closing tag. I never import the previous skill's rules, "
        "outputs, or state into this reply. My reply is always the single short confirmation line this skill's Usage "
        "requires \u2014 never empty, never extra."
    ),
    "step1_assistant": (
        "Each <skill name=...>SKILL BODY</skill> block is a self-contained, one-shot invocation that resets my "
        "state. I derive this reply using ONLY the current block: the skill name from the current opening tag, the "
        "Usage rules from the current SKILL BODY, and the argument from the text immediately after the current "
        "closing tag. I ignore earlier skill blocks, their arguments, their Usage rules, and any previous "
        "confirmation lines as finished history; they do not apply. I do not carry over an old argument into a new "
        "block or fall back to a default because of an earlier short word. I select the mode strictly from the "
        "current argument (using the current block's default only when there is genuinely no argument after the "
        "current closing tag), and I reply with exactly the short confirmation line that the current mode requires "
        "\u2014 never empty, never a stale or previous-mode confirmation."
    ),
}

# V2 — the pair currently shipped in src/slm.ts (SKILLSYS_EXPLAIN / SKILLSYS_THINKING).
V2_PAIR = {
    "step1_user": "How does skill system work? When a skill block is in my latest message, what do I do?",
    "step1_reasoning": (
        "The user wants the rule for a skill block in my latest message. I will explain the `<skill> SKILL BODY "
        "</skill> [USER MESSAGE]` shape: the block is the active instruction, the text after the block is the "
        "current argument, and the reply is the exact confirmation the skill requires."
    ),
    "step1_assistant": (
        "A skill invocation is a `<skill> SKILL BODY </skill>` block with the user message after it: the block "
        "carries the skill's instructions, and the text after the closing tag is the argument for this invocation. "
        "The skill block in my latest message starts a fresh invocation: its name, its Usage rules, and its "
        "argument \u2014 the text right after its own closing tag \u2014 come only from that block. I will use the "
        "argument after the block as the current argument, not an example from inside the skill. If there is no "
        "argument, I will use the skill's required default wording. I will follow the skill's Usage rules for the "
        "current argument and reply text only with the exact required confirmation: skill name, current argument or "
        "default wording, and required wording."
    ),
}
# ORIGINAL SEED — the pair from the real session / real-base-context.json (loaded at runtime).

CONSTRAINT = (
    "Only the step-1 teaching pair may change: step1_user, step1_assistant, step1_reasoning. The rule must stay "
    "GENERAL for ANY <skill> SKILL BODY </skill> [ARGUMENT] block (no tzip, no example, no mode names, no specific "
    "confirmation strings), SHORT, and in first person ('I will ...'). Invocations use BARE arguments after the "
    "block ('full', never 'tzip full'). Prefer short, positive, concrete binding statements ('each block starts a "
    "fresh invocation; the argument is exactly the text after its own closing tag') over emphatic negative commands "
    "\u2014 heavy 'ignore earlier skills / never repeat the old confirmation' wording measurably INCREASES "
    "cross-skill leaks and causes empty replies on this 2.6B student. The required reply is the skill's short "
    "confirmation line \u2014 never empty."
)

OBJECTIVE = (
    "Maximize the student model's (LiquidAI/LFM2.5-2.6B) accuracy on mode activations of a skill it has never "
    "seen, in the EXACT live conversation it will run in. Each test: after a frozen prefix (pi system prompt, "
    "skills/tools Q&A, the step-1 teaching pair, and three completed one-shot invocations of a different 'example' "
    "skill), the user sends a <skill>SKILL BODY</skill> block followed by a BARE argument \u2014 the mode word "
    "alone ('full', 'lite', 'ultra', 'off', 'on', or nothing; never a 'tzip full'-style prefix). The block's "
    "Usage rules map the argument to a mode and require replying with only the exact short mode-confirmation "
    "line. Two failure classes must be eliminated: (1) ACTIVATION FAILURE \u2014 direct activations from a fresh "
    "start (the bare argument 'full' fails most often) and mode-to-mode transitions, where the student repeats "
    "the PREVIOUS confirmation, answers the PREVIOUS call instead of the current one, or replies empty; (2) "
    "CROSS-SKILL LEAK \u2014 the student's reasoning drags in the earlier 'example' skill's invocation state "
    "(its name, its short-word argument, its Usage rules or tool outputs), which corrupts the reply. Scoring: "
    f"each test gets {N_GEPA} independent student rollouts; the item score is the MIN of the two teacher-judged "
    "scores (1.0 exact confirmation for the mode the current argument selects, 0.5 right mode wrong wording OR a "
    "correct reply whose reasoning leaks the previous skill, 0.0 wrong mode or empty reply), so a stochastic "
    "failure on EITHER sample fails the item \u2014 the goal is reliability, not a lucky sample. The only text "
    "you may improve is the step-1 teaching pair (step1_user, step1_assistant, step1_reasoning). The rule must "
    "stay GENERAL \u2014 it must work for ANY <skill> block: no tzip, no example, no mode names, no specific "
    "confirmation strings. Keep it SHORT and first person. Prefer short, positive, concrete binding statements "
    "over emphatic negative commands \u2014 on this 2.6B student, heavy 'ignore earlier skills / never repeat "
    "the old confirmation' wording measurably INCREASES cross-skill leaks and causes empty replies. The reply "
    "must never be empty."
)

BACKGROUND = (
    "Test matrix (37 items, all with BARE arguments after the skill block, never 'tzip X' prefixes): 11 DIRECT "
    "mode activations from a fresh start ('' twice \u2192 default lite, 'on', 'lite', 'full' three times \u2014 "
    "bare 'full' is the user's most frequent live failure \u2014 'ultra' twice, 'off' twice), 20 mode-to-mode "
    "TRANSITIONS (the start mode is established by a bare-argument invocation with its canonical confirmation, "
    "then the student answers the switch with a bare argument \u2014 a transition failure repeats the start "
    "mode's confirmation, answers the start call, or goes empty), and 6 LIVE consecutive-invocation patterns "
    "(back-to-back bare-block invocations, including the exact live failure: bare block \u2192 'tzip lite "
    "activated', then bare block + 'full'). 'on' maps to the 'lite' mode. Each item is scored on TWO "
    "independent student rollouts (score = min of the two teacher-judged scores), so a reliable pair must "
    "produce the exact confirmation on BOTH samples. The judge's REASON in each side-info dict names the defect "
    "(stale-mode confirmation, previous-call answer, wrong mode, empty reply, cross-skill leak); the Samples "
    "line shows, per rollout, the student reply, the verdict, and the leak flag. Scoring: 1.0 exact "
    "confirmation, 0.5 right mode wrong wording or correct reply with a cross-skill leak, 0.0 otherwise."
)


# ---------------------------------------------------------------------------
# models.json
# ---------------------------------------------------------------------------

def load_model_configs():
    raw = MODELS_JSON.read_text()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)  # models.json allows trailing commas
    cfg = json.loads(raw)
    prov = cfg["providers"][PROVIDER]
    models = {m["id"]: m for m in prov["models"]}
    return prov["baseUrl"], prov["apiKey"], models[STUDENT_MODEL], models[TEACHER_MODEL]


def sampling_body(model_cfg, thinking_kwargs):
    """extra_body for dspy.LM from models.json samplingParams + thinking switch."""
    sp = model_cfg.get("samplingParams", {})
    body = {
        k: sp[k]
        for k in ("top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty")
        if k in sp
    }
    if thinking_kwargs:
        body["chat_template_kwargs"] = thinking_kwargs
    return body, sp.get("temperature")


def build_lms(api_base, api_key, student_cfg, teacher_cfg):
    """Student (thinking ON, cache OFF \u2014 multi-sampling needs fresh calls) +
    teacher (thinking OFF, DEFAULT sampling params from models.json), used for
    judging, ground-truth replies, and LFM-voice reasoning generation."""
    s_body, s_temp = sampling_body(student_cfg, {"enable_thinking": True, "preserve_thinking": True})
    student = dspy.LM(
        f"openai/{STUDENT_MODEL}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=s_temp,
        max_tokens=STUDENT_MAX_TOKENS,
        cache=False,
        extra_headers=dict(SESSION_HEADER),
        extra_body=s_body,
    )
    t_body, t_temp = sampling_body(teacher_cfg, {"enable_thinking": False})
    teacher = dspy.LM(
        f"openai/{TEACHER_MODEL}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=t_temp,
        max_tokens=TEACHER_MAX_TOKENS,
        extra_headers=dict(SESSION_HEADER),
        extra_body=t_body,
    )
    return student, teacher


# ---------------------------------------------------------------------------
# chat / tools / rollouts
# ---------------------------------------------------------------------------

def chat(lm, messages, tools=None):
    """One assistant turn. Returns the assistant message in the recorded file shape."""
    kwargs = {"tools": tools} if tools else {}
    resp = lm.forward(messages=messages, **kwargs)
    m = resp.choices[0].message
    out = {"role": "assistant", "content": m.content}
    if getattr(m, "reasoning_content", None):
        out["reasoning_content"] = m.reasoning_content
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": arguments_obj(tc.function.arguments)},
            }
            for tc in m.tool_calls
        ]
    return out


def arguments_obj(args):
    """Tool arguments must be OBJECTS in the recorded file (LFM2.5's chat template raises
    on JSON-encoded strings), so parse string arguments into dicts."""
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}
    return args if isinstance(args, dict) else {}


def last_skill_dir(messages):
    dirs = re.findall(r'<skill name="[^"]*" location="([^"]+)"', json.dumps(messages))
    if not dirs:
        return None
    p = Path(dirs[-1]).parent
    if p.is_dir():
        return str(p)
    alt = p.parent / "skills-byterefinery" / p.name  # relocated skill dirs
    return str(alt) if alt.is_dir() else str(p)


def exec_read(path, cwd, skill_dir):
    p = Path(path)
    if not p.is_absolute():
        cands = [Path(d) / p for d in (skill_dir, cwd) if d]
        p = next((c for c in cands if c.is_file()), Path(cwd) / p)
    try:
        return p.read_text()
    except Exception as e:  # noqa: BLE001
        return f"Error reading {path}: {e}"


def exec_bash(command, cwd, skill_dir):
    run_cwd = Path(cwd)
    if skill_dir and re.search(r"(?<!/)\b(scripts|references)/", command):
        run_cwd = Path(skill_dir)  # skill rule: relative paths resolve against the skill dir
    try:
        r = subprocess.run(command, shell=True, cwd=str(run_cwd), capture_output=True, text=True, timeout=30)
        out = (r.stdout + (r.stderr if r.stderr else "")).strip()
        return out or "(no output)"
    except Exception as e:  # noqa: BLE001
        return f"Error running command: {e}"


def rollout(lm, messages, tools, max_turns=4):
    """Multi-turn rollout: execute tool calls until the model replies with content."""
    msgs = list(messages)
    skill_dir = last_skill_dir(msgs)
    transcript = []
    for _ in range(max_turns):
        m = chat(lm, msgs, tools)
        msgs.append(m)
        transcript.append(m)
        if not m.get("tool_calls"):
            return transcript, m
        for tc in m["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"] or {}
            if name == "read":
                result = exec_read(args.get("path", ""), PI_ROOT, skill_dir)
            elif name == "bash":
                result = exec_bash(args.get("command", ""), PI_ROOT, skill_dir)
            else:
                result = f"Error: unknown tool {name}"
            tool_msg = {"role": "tool", "content": str(result), "tool_call_id": tc["id"]}
            msgs.append(tool_msg)
            transcript.append(tool_msg)
    return transcript, transcript[-1]


def user_text_of(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if p.get("type") == "text")
    return c or ""


def make_user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def norm(s):
    s = re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".")
    return s


# ---------------------------------------------------------------------------
# tzip block / ground-truth sequence (teacher, thinking off)
# ---------------------------------------------------------------------------

_TZIP_BLOCK: str | None = None


def tzip_skill_block():
    """The EXACT tzip <skill> block from the real pi session (byte-identical wire text)."""
    global _TZIP_BLOCK
    if _TZIP_BLOCK is None:
        _TZIP_BLOCK = json.loads(REAL_CTX_FILE.read_text())["tzip_block"]
    return _TZIP_BLOCK


def tzip_usage_rules():
    """The tzip Usage section, parsed from the block itself (source of truth)."""
    m = re.search(r"## Usage\n(.*?)(?=\n## )", tzip_skill_block(), re.S)
    return m.group(1).strip()


MODES = ["on", "lite", "full", "ultra", "off"]
CONFIRM = {
    "on": "tzip lite activated",
    "lite": "tzip lite activated",
    "full": "tzip full activated",
    "ultra": "tzip ultra activated",
    "off": "tzip deactivated",
}


def expected_for(arg):
    """Canonical confirmation for a BARE argument ('' = default -> lite)."""
    a = (arg or "").strip().lower()
    if a.startswith("tzip "):  # defensive only; tests use bare arguments
        a = a[len("tzip "):].strip()
    if a in ("", "tzip", "on", "lite"):
        return CONFIRM["on"]
    if a in CONFIRM:
        return CONFIRM[a]
    return "(see rules)"


def tzip_user(arg):
    """One tzip invocation: the block + a BARE argument (empty = bare block)."""
    block = tzip_skill_block()
    return make_user(block + (f"\n\n{arg}" if arg else ""))


# SEQ: (bare argument, previous mode) — 11 invocations for the artifact file.
SEQ = [
    ("full", "inactive"),    # 1  DIRECT activation: full
    ("ultra", "full"),       # 2  transition up: full -> ultra
    ("off", "ultra"),        # 3  deactivation
    ("lite", "inactive"),    # 4  DIRECT activation: lite
    ("full", "lite"),        # 5  transition up: lite -> full
    ("off", "full"),         # 6  deactivation
    ("ultra", "inactive"),   # 7  DIRECT activation: ultra
    ("lite", "ultra"),       # 8  transition down (jump): ultra -> lite
    ("off", "lite"),         # 9  deactivation
    ("on", "inactive"),      # 10 DIRECT activation: on (= lite)
    ("off", "on"),           # 11 deactivation
]


def teacher_reply(teacher, ctx):
    """One teacher turn (thinking off, default sampling params, no tools). Retries once on empty content."""
    for _ in range(2):
        m = teacher.forward(messages=ctx).choices[0].message
        c = (m.content or "").strip()
        if c:
            return c
    raise RuntimeError("teacher returned empty content twice")


def lfm_reasoning(teacher, arg, prev_mode, reply):
    """Teacher (thinking off, default sampling params) simulates the student's
    reasoning_content in the LFM2.5-2.6B voice, following REASONING-*.md."""
    arg_part = f' with the BARE argument "{arg}"' if arg else " with no extra text after the block (default mode)"
    prev_part = "inactive (not activated yet)" if prev_mode == "inactive" else f"active in {prev_mode} mode"
    prompt = f"""You are writing the `reasoning_content` (the hidden chain-of-thought) of a small 2.6B assistant model for one turn of a conversation.

How this 2.6B model actually reasons — style guide, follow it strictly:
{REASONING_MD.read_text()}

Situation: in the user's latest message the `tzip` skill block (a token-pruning mode skill, whose Usage rule maps the BARE argument after the block to a mode and requires replying with the mode name plus "activated"/"deactivated") is invoked{arg_part}. Before this message, tzip was {prev_part}. The model's reply for this turn is exactly:
{reply}

Write the reasoning_content this 2.6B model would generate right before that reply.
- 1 to 3 short first-person sentences, plain prose (no lists, no headers, no markdown)
- the first sentence names the situation ("The user ..."), then commit to the exact reply
- no meta talk about "reasoning", "synthetic", or "the model"
Output ONLY the reasoning text, nothing else."""
    resp = teacher.forward(messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip().strip('"')


def generate_tzip_sequence(teacher, base_msgs):
    """Teacher generates the tzip sequence: ground-truth replies + LFM-voice reasoning."""
    seq_msgs, items = [], []
    for i, (arg, prev_mode) in enumerate(SEQ):
        user_msg = tzip_user(arg)
        canonical = CONFIRM["on" if arg in ("", "on") else (arg or "full")]
        reply = teacher_reply(teacher, base_msgs + seq_msgs + [user_msg])
        if norm(reply) != norm(canonical):
            say(f"  WARN seq[{i + 1}]: teacher reply {reply!r} != canonical {canonical!r} (keeping teacher's)")
        reason = lfm_reasoning(teacher, arg, prev_mode, reply)
        asst = {"role": "assistant", "content": reply, "reasoning_content": reason}
        seq_msgs += [user_msg, asst]
        items.append(
            {
                "name": f"tzip {arg or '(default)'} #{i + 1}",
                "user_text": user_text_of(user_msg),
                "expected": reply,
            }
        )
        say(f"  seq[{i + 1:2d}/{len(SEQ)}] tzip {arg or '(none)':<6} (was: {prev_mode:>8}) -> {reply!r}")
    return seq_msgs, items


# ---------------------------------------------------------------------------
# document assembly + HF chat-template-playground compatibility check
# ---------------------------------------------------------------------------

def make_doc(messages, model_id, model_cfg, tools):
    sp = model_cfg.get("samplingParams", {})
    doc = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": model_cfg.get("maxTokens", STUDENT_MAX_TOKENS),
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    }
    for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"):
        if k in sp:
            doc[k] = sp[k]
    return doc


def normalize_tool_arguments(messages):
    """Ensure every tool_call's arguments is an OBJECT (LFM2.5 template raises on strings)."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = arguments_obj(tc["function"]["arguments"])


def validate_playground_compat(doc):
    """Render the document with LFM2.5-2.6B's real chat template — the same template the
    HuggingFace chat-template playground applies (transformers engine). The template raises
    'Tool call arguments must be a mapping' on JSON-string arguments, so a successful
    render proves the file is playground-compatible. The {% generation %}/{% endgeneration %}
    tags are transformers engine markers that are no-ops for rendering, so they are stripped
    before plain-jinja2 compilation."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    src = re.sub(r"\{%-?\s*generation\s*-?%\}|\{%-?\s*endgeneration\s*-?%\}", "", TEMPLATE.read_text())
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **kw: json.dumps(
        x, **{k: v for k, v in kw.items() if k in ("ensure_ascii", "indent", "separators", "sort_keys")}
    )
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(src)
    rendered = tpl.render(
        messages=doc["messages"],
        tools=doc["tools"],
        preserve_thinking=True,
        add_generation_prompt=True,
    )
    n_tc = sum(len(m.get("tool_calls") or []) for m in doc["messages"])
    n_rendered = rendered.count("<|tool_call_start|>")
    assert n_rendered == n_tc, f"tool calls rendered {n_rendered} != expected {n_tc}"
    return rendered


def write_tzip_file(base_msgs, tools, tzip_msgs, pair, model_cfg, tag):
    """Write the artifact document with the given step-1 pair swapped into the real context."""
    i = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    msgs = list(base_msgs)
    msgs[i] = make_user(pair["step1_user"])
    msgs[i + 1] = {"role": "assistant", "content": pair["step1_assistant"], "reasoning_content": pair["step1_reasoning"]}
    normalize_tool_arguments(msgs)
    doc = make_doc(msgs + tzip_msgs, STUDENT_MODEL, model_cfg, tools)
    rendered = validate_playground_compat(doc)  # raises on JSON-string tool arguments
    TZIP_FILE.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    say(
        f"Wrote {TZIP_FILE.name} ({len(doc['messages'])} messages, {tag}). "
        f"Playground check: rendered {len(rendered)} chars with the real LFM2.5 template, "
        f"{rendered.count('<|tool_call_start|>')} tool calls OK, "
        f"{rendered.count(chr(60) + chr(124) + 'im_start' + chr(124) + chr(62))} im_start tags, "
        f"{rendered.count(chr(60) + 'think' + chr(62))}/{rendered.count(chr(60) + '/think' + chr(62))} think open/close."
    )


# ---------------------------------------------------------------------------
# teacher judge + multi-sample min-of-N metric
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are judging a small assistant model's reply in a pi conversation, against the rules of the `tzip` skill.

The tzip Usage rules (from the skill block the model saw):
{rules}

Conversation so far (tzip invocations, in order — each invocation is the <skill> block plus a BARE argument; replies are the model's own):
{history}

The current tzip invocation: <skill> block + BARE argument {current!r} (empty means no argument -> default mode)

The model's reply to this invocation:
{reply}
The model's internal reasoning for this reply:
{reasoning}

Judge ONLY this one reply, strictly against the rules:
- the BARE argument selects the mode: '' / 'on' / 'lite' -> lite; 'full' -> full; 'ultra' -> ultra; 'off' -> deactivated
- the reply must be EXACTLY the mode confirmation and nothing else: 'tzip lite activated', 'tzip full activated', 'tzip ultra activated', or 'tzip deactivated'
- extra words, explanations, a wrong mode, or an empty reply are errors

Output exactly three lines, nothing else:
VERDICT: PASS|PARTIAL|FAIL
SCORE: 1.0|0.5|0.0
REASON: <one short sentence>"""


def judge_reply(teacher, history, current, reply, reasoning):
    """Teacher (thinking off, default sampling params) judges one student reply. Returns (verdict, score, reason)."""
    hist = "\n".join(f"  bare argument {a!r} -> {r!r}" for a, r in history) or "  (none \u2014 first invocation)"
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
    verdict = vs.group(1) if vs else "FAIL"
    score = float(ss.group(1)) if ss else 0.0
    reason = rs.group(1).strip() if rs else "(no reason parsed)"
    return verdict, score, reason


def reasoning_leaks_skill_mixup(reasoning):
    """True when the student's reasoning drags in the OTHER (example) skill's
    invocation state — the cross-skill leak the isolation rule must prevent."""
    r = (reasoning or "").lower()
    return ("example" in r) or ("03-hello" in r) or ("'hi'" in r and ("tzip" in r or "mode" in r))


def build_judge_items():
    """The 36-item live matrix, ALL with bare arguments (never 'tzip X' prefixes):
    A. 10 direct activations from a fresh start
    B. 20 mode-to-mode transitions (bare-argument setup)
    C. 6 live consecutive-invocation patterns (incl. the exact live failure)."""
    items = []
    direct = [
        ("", "activate (empty, default)"),
        ("", "activate (empty, default)"),
        ("on", "activate 'on'"),
        ("lite", "activate 'lite'"),
        ("full", "activate 'full'"),
        ("full", "activate 'full'"),
        ("full", "activate 'full'"),
        ("ultra", "activate 'ultra'"),
        ("ultra", "activate 'ultra'"),
        ("off", "activate 'off'"),
        ("off", "activate 'off'"),
    ]
    for arg, label in direct:
        items.append({"name": f"A {label}", "setup": [], "current": arg})
    for a in MODES:
        for b in MODES:
            if a != b:
                items.append(
                    {"name": f"B {a} -> {b}", "setup": [(a, CONFIRM[a])], "current": b}
                )
    live = [
        ("", "tzip lite activated", "full"),        # THE live failure: bare block -> bare 'full'
        ("", "tzip lite activated", "lite"),        # repeat same mode
        ("full", "tzip full activated", "lite"),    # starts with full
        ("full", "tzip full activated", "ultra"),
        ("", "tzip lite activated", "off"),
        ("ultra", "tzip ultra activated", "full"),
    ]
    for a1, r1, a2 in live:
        items.append({"name": f"C live {a1 or '(empty)'} -> {a2}", "setup": [(a1, r1)], "current": a2})
    return items


def make_judge_evaluator(student, teacher, tools, base_msgs, n_samples):
    """Student rolls out n_samples times on (candidate pair + frozen prefix + item setup);
    the TEACHER judges each sample; item score = MIN of sample scores.
    A correct reply whose reasoning leaks the previous skill is capped at 0.5 per sample."""
    i = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    before, after = base_msgs[:i], base_msgs[i + 2 :]

    def evaluate(candidate, example=None):
        if example is None:
            return 0.0, {"error": "evaluator called without an example"}
        base_ctx = before + [
            make_user(candidate["step1_user"]),
            {
                "role": "assistant",
                "content": candidate["step1_assistant"],
                "reasoning_content": candidate["step1_reasoning"],
            },
        ] + after
        scores, lines, n_leaks = [], [], 0
        for s in range(n_samples):
            ctx = list(base_ctx)
            for arg, reply in example["setup"]:
                ctx += [tzip_user(arg), {"role": "assistant", "content": reply}]
            ctx += [tzip_user(example["current"])]
            try:
                transcript, final = rollout(student, ctx, tools, max_turns=2)
            except Exception as exc:  # noqa: BLE001
                return 0.0, {
                    "Task": example["name"],
                    "Status": "ERROR",
                    "Error": f"{type(exc).__name__}: {exc}",
                    "Constraint": CONSTRAINT,
                }
            reply = (final.get("content") or "").strip()
            reasoning = final.get("reasoning_content") or ""
            verdict, score, reason = judge_reply(teacher, example["setup"], example["current"], reply, reasoning)
            leak = reasoning_leaks_skill_mixup(reasoning)
            if leak and score == 1.0:
                score = 0.5
                reason = reason + " | NOTE: reply correct, but reasoning mixes in the previous skill's invocation state (isolation violation)"
            if leak:
                n_leaks += 1
            scores.append(score)
            lines.append(
                f"[{s + 1}/{n_samples}] reply={reply or '(empty)'} judge={verdict} {score:.1f} "
                f"leak={'YES' if leak else 'no'} :: {reason}"
            )
        item_score = min(scores)
        status = "PASS" if item_score >= 0.999 else ("PARTIAL" if item_score > 0 else "FAIL")
        side = {
            "Task": example["name"],
            "Status": status,
            "Score (min of %d samples)" % n_samples: item_score,
            "Expected reply": expected_for(example["current"]),
            "Leaks": f"{n_leaks}/{n_samples}",
            "Samples": " | ".join(lines),
            "Constraint": CONSTRAINT,
        }
        return item_score, side

    return evaluate


def evaluate_all(evaluate, candidate, items, label, n_samples):
    say(f"\n{'=' * 74}\n{label}  (n_samples={n_samples})\n{'=' * 74}")
    total, n_leaks, n_samples_total, fails = 0.0, 0, 0, []
    for it in items:
        s, side = evaluate(candidate, it)
        total += s
        n_leaks += int(side.get("Leaks", "0/0").split("/")[0])
        n_samples_total += int(side.get("Leaks", "0/0").split("/")[1])
        mark = "PASS " if side["Status"] == "PASS" else ("PART " if side["Status"] == "PARTIAL" else "FAIL")
        say(f"  [{mark}] {it['name']:<28} {s:.1f} | expected: {expected_for(it['current'])!r}")
        if side["Status"] != "PASS":
            fails.append(it["name"])
            for ln in side.get("Samples", "").split(" | "):
                say(f"  {'':15} {ln}")
    agg = total / len(items)
    leak_rate = n_leaks / n_samples_total if n_samples_total else 0.0
    say(f"  aggregate: {agg:.3f} | leaks: {n_leaks}/{n_samples_total} ({leak_rate:.0%}) | failing items: {len(fails)}")
    return agg, leak_rate, fails


# ---------------------------------------------------------------------------
# random walks + robustness probes (log only)
# ---------------------------------------------------------------------------

def run_random_walks(student, teacher, tools, base_msgs, candidate, label, steps=3, seed=7):
    """Random mode walks chained on the STUDENT'S OWN replies (its errors propagate,
    exactly like a live session)."""
    i = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    before, after = base_msgs[:i], base_msgs[i + 2 :]
    base_ctx = before + [
        make_user(candidate["step1_user"]),
        {"role": "assistant", "content": candidate["step1_assistant"], "reasoning_content": candidate["step1_reasoning"]},
    ] + after
    rng = random.Random(seed)
    say(f"\n{'=' * 74}\n{label}: random walks ({steps} steps per start, chained on student's own replies)\n{'=' * 74}")
    n_pass = n_total = n_leaks = 0
    for start in MODES:
        seq = [start]
        cur = start
        for _ in range(steps):
            cur = rng.choice([m for m in MODES if m != cur])
            seq.append(cur)
        say(f"\n  walk starting from {start!r}: {seq}")
        history = [(start, CONFIRM[start])]
        for n, step in enumerate(seq[1:], start=2):
            ctx = list(base_ctx)
            for a, r in history:
                ctx += [tzip_user(a), {"role": "assistant", "content": r}]
            ctx += [tzip_user(step)]
            _, final = rollout(student, ctx, tools, max_turns=2)
            reply = (final.get("content") or "").strip()
            reasoning = final.get("reasoning_content") or ""
            verdict, score, reason = judge_reply(teacher, history, step, reply, reasoning)
            leak = reasoning_leaks_skill_mixup(reasoning)
            if leak:
                n_leaks += 1
            n_pass += int(score >= 0.999)
            n_total += 1
            say(f"  [C{start}] step {n} {step!r}: {verdict} {score:.1f} leak={'YES' if leak else 'no'}")
            say(f"      expected: {expected_for(step)!r} | student: {reply or '(empty)'}")
            history.append((step, reply))
    say(f"\n  {label} walks total: {n_pass}/{n_total} PASS, {n_leaks}/{n_total} samples leaked")


def run_robustness_probes(student, base_msgs, tools, tzip_msgs, candidate):
    """Diagnostic probes beyond the matrix: verbose arguments, repeated modes, and
    example-skill behavior after a tzip sequence. Logged only."""
    say(f"\n{'=' * 74}\nROBUSTNESS PROBES (log only, not part of the file)\n{'=' * 74}")
    i1 = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    base_ctx = base_msgs[:i1] + [
        make_user(candidate["step1_user"]),
        {"role": "assistant", "content": candidate["step1_assistant"], "reasoning_content": candidate["step1_reasoning"]},
    ] + base_msgs[i1 + 2 :]
    block = tzip_skill_block()
    ex_block = next(
        user_text_of(m) for m in base_msgs if m.get("role") == "user" and user_text_of(m).startswith('<skill name="example"')
    )
    probes = [
        ("verbose 'ultra please' (fresh start)", [], block + "\n\nultra please", "tzip ultra activated"),
        ("verbose 'off now' (fresh start)", [], block + "\n\noff now", "tzip deactivated"),
        ("case variant 'TZIP FULL' (fresh start)", [], block + "\n\nTZIP FULL", "tzip full activated"),
        # prefix = real tzip sequence MESSAGES (user 'full' + its confirmation), not (arg, reply) setup tuples
        ("repeat 'full' after 'full' (same mode)", tzip_msgs[:2], block + "\n\nfull", "tzip full activated"),
        # full rollout: the skill requires reading references/03-hello.md before replying 'world'
        ("example 'Hello' after tzip sequence (full rollout)", tzip_msgs[:4], ex_block + "\n\nHello", "world"),
        ("example (no text) while tzip active (was full)", tzip_msgs[:2], ex_block, "This is an example skill."),
    ]
    for name, prefix, user_text, expect in probes:
        ctx = base_ctx + prefix + [make_user(user_text)]
        transcript, final = rollout(student, ctx, tools, max_turns=3)
        content = (final.get("content") or "").strip()
        status = "OK  " if norm(content) == norm(expect) else "DIFF"
        n_tc = sum(len(m.get("tool_calls") or []) for m in transcript)
        say(f"  [{status}] {name}\n  {'':4} expected: {expect!r}\n  {'':4} student:  {content!r} (tool calls: {n_tc})")
    # tzip persistence: ACTIVE EVERY RESPONSE until 'tzip off' — inspection only (style, no exact answer)
    plain_q = "What are the capital cities of France and Germany?"
    for name, prefix in [
        ("tzip ACTIVE (full mode): plain question", tzip_msgs[:2]),
        ("tzip OFF (after full tzip sequence): plain question", tzip_msgs[:6]),
    ]:
        ctx = base_ctx + prefix + [make_user(plain_q)]
        _, final = rollout(student, ctx, tools, max_turns=3)
        say(f"  [INFO] {name}\n  {'':4} question: {plain_q!r}\n  {'':4} student:  {(final.get('content') or '')!r}\n  {'':4} reasoning:  {(final.get('reasoning_content') or '')[:200]!r}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global API_BASE, API_KEY
    API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG = load_model_configs()
    # NOTE: API_BASE / API_KEY stay in memory only — never printed, never written to any file.
    say(f"Loaded provider '{PROVIDER}' config for {STUDENT_MODEL} (thinking on) and {TEACHER_MODEL} (thinking off, default sampling params).")
    assert REAL_CTX_FILE.exists(), f"{REAL_CTX_FILE} missing"
    assert TEMPLATE.exists(), f"{TEMPLATE} missing"
    assert REASONING_MD.exists(), f"{REASONING_MD} missing"

    student, teacher = build_lms(API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG)

    # sanity check: one cheap student + teacher call (no URLs or keys printed)
    sp = STUDENT_CFG.get("samplingParams", {})
    say(f"Student LM under evaluation: openai/{STUDENT_MODEL} \u2014 thinking ON (enable_thinking=True, "
        f"preserve_thinking=True), cache=False, temp={sp.get('temperature')}, top_k={sp.get('top_k')}, "
        f"repeat_penalty={sp.get('repeat_penalty')}")
    say(f"Teacher LM (judge / ground-truth / LFM-voice reasoning / GEPA reflection): "
        f"openai/{TEACHER_MODEL} \u2014 thinking OFF, default sampling params")
    t_probe = chat(teacher, [{"role": "user", "content": "Reply with the single word: pong"}])
    say(f"Teacher probe: {t_probe['content']!r}")
    s_probe = chat(student, [{"role": "user", "content": "Reply with the single word: pong"}])
    s_reason = (s_probe.get("reasoning_content") or "").strip()
    if not s_reason:
        raise RuntimeError("student returned no reasoning_content \u2014 thinking is NOT on; aborting before evaluation")
    say(f"Student probe: content={s_probe['content']!r} reasoning={s_reason[:60]!r}... [thinking ON verified]")

    # ---- frozen base conversation: the REAL pi-session wire context ----
    real_ctx = json.loads(REAL_CTX_FILE.read_text())
    base_msgs = [{"role": "system", "content": real_ctx["system_prompt"]}, *real_ctx["messages"]]
    TOOLS = json.loads(EXAMPLE_FILE.read_text())["tools"]  # session tools (OpenAI format)
    normalize_tool_arguments(base_msgs)
    i1 = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    original_seed = {
        "step1_user": real_ctx["step1_user"],
        "step1_assistant": real_ctx["step1_assistant"],
        "step1_reasoning": real_ctx["step1_reasoning"],
    }
    assert user_text_of(base_msgs[i1]) == original_seed["step1_user"]
    assert base_msgs[i1 + 1]["content"] == original_seed["step1_assistant"]
    assert norm(SEED_PAIR["step1_assistant"]) != norm(original_seed["step1_assistant"]), "SEED_PAIR should be the quoted pair, not the original seed"
    candidates = {
        "quoted (current artifact)": SEED_PAIR,
        "original (real session)": original_seed,
        "V2 (slm.ts)": V2_PAIR,
    }
    say(f"Base conversation: REAL pi-session context, {len(base_msgs)} messages (incl. system), {len(TOOLS)} tools; "
        f"only the step-1 pair at index {i1}/{i1 + 1} is optimized. Skill invocations are appended after the "
        f"example few-shots (message {len(base_msgs)}+), all with BARE arguments.")

    # ---- tzip ground truth (teacher, thinking off) + LFM-voice reasoning ----
    say(f"Generating {len(SEQ)} tzip invocations with the teacher model (thinking off) ...")
    tzip_msgs, _ = generate_tzip_sequence(teacher, base_msgs)

    # ---- initial file with the SEED (quoted) pair (inspectable) ----
    write_tzip_file(base_msgs, TOOLS, tzip_msgs, SEED_PAIR, STUDENT_CFG, tag="seed (quoted) pair")

    # ---- stage 1: baselines (N=1, orientation) ----
    items = build_judge_items()
    baseline_eval = make_judge_evaluator(student, teacher, TOOLS, base_msgs, n_samples=N_BASELINE)
    baselines = {}
    for name, cand in candidates.items():
        agg, leaks, _ = evaluate_all(baseline_eval, cand, items, f"BASELINE (N={N_BASELINE}) — {name}", N_BASELINE)
        baselines[name] = (agg, leaks)

    # ---- stage 2: GEPA from the quoted seed, multi-sample min-of-2 metric ----
    say(f"\nGEPA — {len(items)} validation items, dataset={len(items) + 4} (weighted), "
        f"minibatch={MINI_BATCH}, n_samples={N_GEPA} (min), max_metric_calls={MAX_METRIC_CALLS}, "
        f"reflection LM = {TEACHER_MODEL} (thinking off), timeout={GEPA_TIMEOUT_S}s.")
    dataset = [{**it, "id": f"d-{n + 1}"} for n, it in enumerate(items)]
    # weighted duplicates so a random minibatch reliably contains the high-risk cases
    extra = [
        {"name": "A activate 'full' (dup)", "setup": [], "current": "full"},
        {"name": "A activate 'full' (dup)", "setup": [], "current": "full"},
        {"name": "A activate (empty, default) (dup)", "setup": [], "current": ""},
        {"name": "C live (empty) -> full (dup)", "setup": [("", "tzip lite activated")], "current": "full"},
    ]
    dataset += [{**it, "id": f"dx-{n + 1}"} for n, it in enumerate(extra)]
    valset = [{**it, "id": f"v-{n + 1}"} for n, it in enumerate(items)]
    gepa_eval = make_judge_evaluator(student, teacher, TOOLS, base_msgs, n_samples=N_GEPA)
    result = optimize_anything(
        seed_candidate=dict(SEED_PAIR),
        evaluator=gepa_eval,
        dataset=dataset,
        valset=valset,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(RUN_DIR),
                seed=7,
                max_metric_calls=MAX_METRIC_CALLS,
                parallel=PARALLEL,
                display_progress_bar=True,
            ),
            reflection=ReflectionConfig(reflection_lm=lambda p: teacher.forward(messages=[{"role": "user", "content": p}]).choices[0].message.content or "", reflection_minibatch_size=MINI_BATCH),
            stop_callbacks=[
                # With min-of-2 scoring, 0.99 on 37 items means EVERY item passed on BOTH samples.
                ScoreThresholdStopper(threshold=0.99),
                NoImprovementStopper(max_iterations_without_improvement=8),
                TimeoutStopCondition(timeout_seconds=GEPA_TIMEOUT_S),
            ],
        ),
    )
    best = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx] if result.val_aggregate_scores else float("nan")
    say(f"GEPA done: best val score = {best_score:.3f} (candidate index {result.best_idx} of {len(result.candidates)}), "
        f"metric calls used: {result.total_metric_calls}.")
    say("\nImprovement trajectory (per-candidate validation aggregate score):")
    scores = result.val_aggregate_scores or []
    for idx, cand in enumerate(result.candidates):
        sc = scores[idx] if idx < len(scores) else None
        sc_s = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(sc)
        mark = "  <-- best" if idx == result.best_idx else ""
        say(f"  candidate {idx:2d}: {sc_s}{mark}")
    run_log = RUN_DIR / "run_log.txt"
    if run_log.exists():
        say("\nGEPA run_log digest:")
        for line in run_log.read_text().splitlines():
            if "Base program full valset" in line or "Found a better program" in line:
                say(f"  {line}")
    changed = any(best.get(k) != SEED_PAIR[k] for k in ("step1_user", "step1_assistant", "step1_reasoning"))
    say(f"Pair changed by GEPA: {changed}")

    # ---- stage 3: FINAL multi-sample validation (N=3): best vs quoted vs original ----
    final_eval = make_judge_evaluator(student, teacher, TOOLS, base_msgs, n_samples=N_FINAL)
    finals = {}
    for name, cand in (("GEPA best", best), ("quoted (current artifact)", SEED_PAIR), ("original (real session)", original_seed)):
        agg, leaks, fails = evaluate_all(final_eval, cand, items, f"FINAL (N={N_FINAL}, min-of-3) — {name}", N_FINAL)
        finals[name] = (agg, leaks, fails)

    # ---- stage 4: random walks + robustness probes on the best pair ----
    run_random_walks(student, teacher, TOOLS, base_msgs, best, "FINAL")
    run_robustness_probes(student, base_msgs, TOOLS, tzip_msgs, best)

    # ---- write the final artifact + standalone pair JSON ----
    write_tzip_file(base_msgs, TOOLS, tzip_msgs, best, STUDENT_CFG, tag="optimized pair")
    PAIR_FILE.write_text(
        json.dumps(
            {
                **best,
                "meta": {
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "student": f"{STUDENT_MODEL} (thinking on)",
                    "teacher": f"{TEACHER_MODEL} (thinking off, default sampling params from models.json)",
                    "seed": "quoted pair (current skill-example-tzip artifact)",
                    "metric": f"teacher-judged, {N_GEPA} independent student samples per item, score = MIN; "
                              "cross-skill leak in reasoning caps a correct reply at 0.5; bare arguments only",
                    "baseline_N1": {k: {"aggregate": round(v[0], 4), "leaks": round(v[1], 4)} for k, v in baselines.items()},
                    "gepa_best_valscore": round(best_score, 4) if best_score == best_score else None,
                    "final_N3": {k: {"aggregate": round(v[0], 4), "leaks": round(v[1], 4), "failing_items": v[2]} for k, v in finals.items()},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    say(f"Saved optimized step-1 pair to {PAIR_FILE.name} and into {TZIP_FILE.name}.")
    say("\nOptimized step-1 pair:")
    say(f"  user: {best['step1_user']}")
    say(f"  assistant: {best['step1_assistant']}")
    say(f"  reasoning: {best['step1_reasoning']}")
    say(
        f"\nSummary: baseline(N=1) "
        + ", ".join(f"{k}={v[0]:.3f} (leaks {v[1]:.0%})" for k, v in baselines.items())
        + f"; GEPA best valscore={best_score:.3f}; final(N=3) "
        + ", ".join(f"{k}={v[0]:.3f} (leaks {v[1]:.0%})" for k, v in finals.items())
        + "."
    )


if __name__ == "__main__":
    main()
