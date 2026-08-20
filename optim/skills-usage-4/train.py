#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///

'''
Re-optimize the step-1 skill-teaching pair for the student LM (LiquidAI/LFM2.5-2.6B,
thinking ON) with dspy + GEPA, against an UNSEEN skill: `tzip`.

The problem: the step-1 pair (user: "How does skill system work? ..." + the assistant rule)
was originally tuned in a context where only `tzip` invocations appeared and the `example`
skill few-shot was NOT part of the prefix. After the `example` few-shot (3 one-shots) was
added to the conversation prefix, the pair had to make the student generalize the rule to a
skill it has never seen while the example one-shots sit in context. This run re-optimizes
ONLY that pair:

- Frozen prefix (never modified): system message, the two skills/tools Q&A pairs, and the
  three `example` one-shots (no text -> exact line; "Hello" -> read reference file;
  "Hi" -> run script with the text as CLI parameters).
- Optimized: the step-1 pair only (user text + assistant content + assistant reasoning).
- Test invocations appended after the prefix: 11 `tzip` messages covering DIRECT mode
  activations from the inactive state (full, lite, ultra, on — no prior mode to
  transition from, each after a `tzip off` or at conversation start) and mode
  TRANSITIONS (full->ultra up, lite->full up, ultra->lite down-jump) plus
  deactivations. `tzip` never appears in the prefix: this is the unseen-skill test.
- Robustness probes (log only, not in the file): verbose arguments ("tzip on",
  "ultra please", "off now", "TZIP FULL"), example-skill leak checks around the tzip
  sequence (full rollout with tool execution), and tzip persistence checks (plain
  questions while tzip is active vs. after "tzip off").
- Ground-truth replies: teacher Qwen/Qwen3.8-27B (thinking OFF, no tools).
- reasoning_content of the tzip assistant messages: simulated by the teacher in
  LFM2.5-2.6B voice, following REASONING-LiquidAI-LFM2.5-2.6B.md.

Flow (all logged to train.log via stdout):
1. probes (teacher/student connectivity)
2. teacher ground truth for the 11 tzip invocations
3. write skill-example-tzip-LiquidAI-LFM2.5-2.6B.json with the SEED pair (inspectable)
4. BASELINE: evaluate the unoptimized pair on all 11 items  <- performance without optimization
5. GEPA (reflection LM = teacher, thinking off)            <- improvement progress
6. FINAL: re-evaluate the best pair on all 11 items, rewrite the file with the optimized pair
7. ROBUSTNESS PROBES: verbose-argument and example-leak checks (log only)

HuggingFace chat-template playground compatibility: LFM2.5-2.6B's chat_template.jinja
RAISES "Tool call arguments must be a mapping" when tool_call arguments are JSON-encoded
strings, so after every write we render the whole document with the real template
(lfm25-chat-template.jinja) — a successful render proves the file loads in the playground
(tool arguments are objects, content shapes are accepted).

API credentials: read at runtime from ~/.pi/agent/models.json (provider `llamacpp`),
masked everywhere; never written to any file, log, or state.
'''

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import dspy
import gepa
from gepa import NoImprovementStopper, ScoreThresholdStopper, TimeoutStopCondition
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

HERE = Path(__file__).resolve().parent
PI_ROOT = HERE.parents[1]  # /home/mtasic/projects-b/pi-slm

MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
EXAMPLE_FILE = HERE / "skill-example-LiquidAI-LFM2.5-2.6B.json"  # frozen base conversation
TZIP_FILE = HERE / "skill-example-tzip-LiquidAI-LFM2.5-2.6B.json"
PAIR_FILE = HERE / "step1-pair-optimized.json"
TEMPLATE = HERE / "lfm25-chat-template.jinja"  # LFM2.5-2.6B chat template (HF playground compat check)
REASONING_MD = HERE / "REASONING-LiquidAI-LFM2.5-2.6B.md"
RUN_DIR = HERE / "gepa_runs" / "skills-usage-4"

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"  # thinking: on
TEACHER_MODEL = "Qwen/Qwen3.8-27B"  # thinking: off
PROVIDER = "llamacpp"
SESSION_HEADER = {"x-session-affinity": "dspy-optim"}

TZIP_SKILL_DIR = PI_ROOT / ".agents" / "skills-byterefinery" / "tzip"
TZIP_LOCATION = str(TZIP_SKILL_DIR / "SKILL.md")

MAX_METRIC_CALLS = 200
STUDENT_MAX_TOKENS = 16384

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
    s_body, s_temp = sampling_body(student_cfg, {"enable_thinking": True, "preserve_thinking": True})
    student = dspy.LM(
        f"openai/{STUDENT_MODEL}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=s_temp,
        max_tokens=STUDENT_MAX_TOKENS,
        extra_headers=dict(SESSION_HEADER),
        extra_body=s_body,
    )
    t_body, t_temp = sampling_body(teacher_cfg, {"enable_thinking": False})  # thinking off, so it works fast
    teacher = dspy.LM(
        f"openai/{TEACHER_MODEL}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=t_temp,
        max_tokens=STUDENT_MAX_TOKENS,
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
    return str(Path(dirs[-1]).parent) if dirs else None


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
# tzip test sequence (tzip is UNSEEN in the prefix): DIRECT mode activations from
# the inactive state (no prior mode to transition from) + mode transitions +
# deactivations. (argument, previous mode, canonical reply)
# ---------------------------------------------------------------------------

SEQ = [
    ("full", "inactive", "tzip full activated"),    # 1  DIRECT activation: full
    ("ultra", "full", "tzip ultra activated"),      # 2  transition up: full -> ultra
    ("off", "ultra", "tzip deactivated"),            # 3  deactivation
    ("lite", "inactive", "tzip lite activated"),    # 4  DIRECT activation: lite
    ("full", "lite", "tzip full activated"),         # 5  transition up: lite -> full
    ("off", "full", "tzip deactivated"),             # 6  deactivation
    ("ultra", "inactive", "tzip ultra activated"),  # 7  DIRECT activation: ultra
    ("lite", "ultra", "tzip lite activated"),        # 8  transition down (jump): ultra -> lite
    ("off", "lite", "tzip deactivated"),             # 9  deactivation
    ("on", "inactive", "tzip lite activated"),       # 10 DIRECT activation: on (= lite)
    ("off", "on", "tzip deactivated"),               # 11 deactivation
]


def tzip_skill_block():
    body = (TZIP_SKILL_DIR / "SKILL.md").read_text()
    body = re.sub(r"\A---\n.*?\n---\n", "", body).strip("\n")
    return (
        f'<skill name="tzip" location="{TZIP_LOCATION}">\n'
        f"References are relative to {TZIP_SKILL_DIR}.\n\n{body}\n</skill>"
    )


# ---------------------------------------------------------------------------
# teacher ground truth: replies + LFM-voice reasoning (per REASONING-...md)
# ---------------------------------------------------------------------------

def teacher_reply(teacher, ctx):
    """One teacher turn (thinking off, no tools). Retries once on empty content."""
    for _ in range(2):
        resp = teacher.forward(messages=ctx)
        m = resp.choices[0].message
        c = (m.content or "").strip()
        if c:
            return c
        time.sleep(1)
    raise RuntimeError("teacher returned empty content twice")


def lfm_reasoning(teacher, arg, prev_mode, reply):
    """Teacher simulates the student's reasoning_content in LFM2.5-2.6B voice."""
    arg_part = f' with the argument "{arg}"' if arg else " with no extra text (default mode)"
    prev_part = "inactive (not activated yet)" if prev_mode == "inactive" else f"active in {prev_mode} mode"
    prompt = f"""You are writing the `reasoning_content` (the hidden chain-of-thought) of a small 2.6B assistant model for one turn of a conversation.

How this 2.6B model actually reasons — style guide, follow it strictly:
{REASONING_MD.read_text()}

Situation: in the user's latest message the `tzip` skill block (a token-pruning mode skill, whose Usage rule maps the argument to a mode and requires replying with the mode name plus "activated"/"deactivated") is invoked{arg_part}. Before this message, tzip was {prev_part}. The model's reply for this turn is exactly:
{reply}

Write the reasoning_content this 2.6B model would generate right before that reply.
- 1 to 3 short first-person sentences, plain prose (no lists, no headers, no markdown)
- the first sentence names the situation ("The user ..."), then commit to the exact reply
- no meta talk about "reasoning", "synthetic", or "the model"
Output ONLY the reasoning text, nothing else."""
    resp = teacher.forward(messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip().strip('"')


def generate_tzip_sequence(teacher, base_msgs):
    """Teacher (thinking off) generates the tzip sequence: replies + LFM-voice reasoning."""
    block = tzip_skill_block()
    seq_msgs, items = [], []
    for i, (arg, prev_mode, canonical) in enumerate(SEQ):
        text = block + (f"\n\n{arg}" if arg else "")
        user_msg = make_user(text)
        prefix = seq_msgs[:]
        reply = teacher_reply(teacher, base_msgs + prefix + [user_msg])
        if norm(reply) != norm(canonical):
            say(f"  WARN seq[{i + 1}]: teacher reply {reply!r} != canonical {canonical!r} (keeping teacher's)")
        reason = lfm_reasoning(teacher, arg, prev_mode, reply)
        asst = {"role": "assistant", "content": reply, "reasoning_content": reason}
        seq_msgs += [user_msg, asst]
        items.append(
            {
                "name": f"tzip {arg or '(default)'} #{i + 1}",
                "user_text": text,
                "expected": reply,
                "prefix": prefix,
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
    tags are transformers engine markers that are no-ops for rendering (they only track
    assistant-token indices), so they are stripped before plain-jinja2 compilation."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    # transformers' engine: sandboxed env, trim/lstrip blocks, non-HTML-escaping tojson,
    # raise_exception global; generation/endgeneration tags are no-ops for rendering
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


def write_tzip_file(base_msgs, tools, tzip_msgs, step1, model_cfg, tag):
    step1_user, step1_content, step1_reasoning = step1
    i = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    msgs = list(base_msgs)
    msgs[i] = make_user(step1_user)
    msgs[i + 1] = {"role": "assistant", "content": step1_content, "reasoning_content": step1_reasoning}
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
# Goal — GEPA optimization of the step-1 pair
# ---------------------------------------------------------------------------

CONSTRAINT = (
    "The step-1 pair must stay a GENERAL rule for any skill block of the form "
    "`<skill> SKILL BODY </skill> [USER MESSAGE]`; it must not mention tzip, example, "
    "or any specific reply text. The student is a 2.6B model: keep the pair short, "
    "concrete, and in first person ('I will ...')."
)

OBJECTIVE = (
    "Maximize the student model's (LiquidAI/LFM2.5-2.6B) accuracy on pi skill invocations of a skill it has "
    "NEVER seen before. The student sees a fixed conversation prefix: the pi system prompt, two synthetic Q&A "
    "pairs listing available skills and tools, the step-1 teaching pair (the ONLY text you may improve: "
    "step1_user, step1_assistant, step1_reasoning), and three frozen one-shot demonstrations of the 'example' "
    "skill: invoked with no text -> reply exactly the required line; invoked with 'Hello' -> read the required "
    "reference file, then reply its required word; invoked with 'Hi' -> run the skill script with the user's "
    "text as CLI parameters, then report the output. The problem being fixed: with these example one-shots in "
    "context, the student must still follow a FRESH skill block — the example demonstrations must not leak into "
    "how it answers a different skill. After the prefix, the user sends 11 invocations of the 'tzip' skill (a "
    "token-pruning mode; UNSEEN in the prefix): each message is a <skill name=tzip>SKILL BODY</skill> block "
    "optionally followed by an argument. The tzip Usage rule maps the argument to a mode (''/'on'/'lite' -> "
    "lite, 'full' -> full, 'ultra' -> ultra, 'off' -> deactivate) and requires replying with only the mode "
    "confirmation, e.g. 'tzip lite activated', 'tzip full activated', 'tzip ultra activated', 'tzip "
    "deactivated'. The 11 messages alternate DIRECT activations (from the inactive state, each after a "
    "deactivation or at conversation start: full, lite, ultra, on) and mode TRANSITIONS (full->ultra up, "
    "lite->full up, ultra->lite down-jump), ending with deactivations. The student must treat the tzip block as the "
    "active instruction, read the argument after the closing tag as the current argument, and reply with only "
    "the exact confirmation the tzip block requires — never replying like the example skill."
)

BACKGROUND = (
    "The step-1 teaching pair explains the <skill> SKILL BODY </skill> [USER MESSAGE] shape: the block carries "
    "the skill's instructions, the text after the block is the current argument, and the reply is the exact "
    "confirmation the skill requires. Rules for improving it: it must stay GENERAL (no tzip, no example, no "
    "specific reply text), keep teaching the block shape and the argument rule, be written in the student "
    "model's first-person voice, and stay short. The frozen parts (system message, the two skills/tools Q&A "
    "pairs, the three example one-shots) are never optimized. The 11 test messages are tzip DIRECT mode "
    "activations (from the inactive state: full, lite, ultra, on) and tzip mode transitions; each item's "
    "'Expected reply' is the exact confirmation string. Scoring: exact normalized "
    "reply match = 1.0, containment = 0.5, otherwise 0.0."
)


def score_item(example, final_msg, transcript):
    actual = final_msg.get("content") or ""
    e, a = norm(example["expected"]), norm(actual)
    if e and a and e == a:
        score = 1.0
    elif e and a and (e in a or a in e):
        score = 0.5
    else:
        score = 0.0
    tool_calls = [tc for m in transcript if m.get("tool_calls") for tc in m["tool_calls"]]
    status = "PASS" if score >= 0.999 else ("PARTIAL" if score > 0 else "FAIL")
    side = {
        "Task": example["name"],
        "Status": status,
        "Expected reply": example["expected"],
        "Student reply": actual or "(empty)",
        "Student tool calls": "; ".join(
            f'{tc["function"]["name"]}({json.dumps(tc["function"]["arguments"])})' for tc in tool_calls
        )
        or "none",
        "Student reasoning_content": (final_msg.get("reasoning_content") or "none")[:600],
        "Constraint": CONSTRAINT,
    }
    return score, side


def make_evaluator(student, tools, base_msgs):
    i = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    before, after = base_msgs[:i], base_msgs[i + 2 :]  # step-1 pair is the only variable part

    def evaluate(candidate, example=None):
        if example is None:
            return 0.0, {"error": "evaluator called without an example"}
        ctx = before + [make_user(candidate["step1_user"])] + [
            {
                "role": "assistant",
                "content": candidate["step1_assistant"],
                "reasoning_content": candidate["step1_reasoning"],
            }
        ] + after + list(example["prefix"]) + [make_user(example["user_text"])]
        try:
            transcript, final = rollout(student, ctx, tools, max_turns=4)
        except Exception as exc:  # noqa: BLE001
            return 0.0, {"Task": example["name"], "Status": "ERROR", "Error": f"{type(exc).__name__}: {exc}", "Constraint": CONSTRAINT}
        return score_item(example, final, transcript)

    return evaluate


def make_reflection_lm(teacher):
    def reflection_lm(prompt):
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        resp = teacher.forward(messages=msgs)
        return resp.choices[0].message.content or ""

    return reflection_lm


def evaluate_all(evaluate, candidate, items, label):
    say(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    total = 0.0
    for it in items:
        s, side = evaluate(candidate, it)
        total += s
        say(f"  [{side['Status']:7s}] {it['name']:<22} {s:.1f} | expected: {it['expected']!r}")
        say(f"  {'':15} student: {side.get('Student reply', '')!r}")
    say(f"  aggregate: {total / len(items):.3f}")
    return total / len(items)


def run_robustness_probes(student, base_msgs, tools, tzip_msgs, candidate):
    """Diagnostic probes beyond the file/valset: verbose arguments and example-skill
    leak checks around the tzip sequence. Logged only."""
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
        ("tzip verbose 'tzip on' (first message)", [], block + "\n\ntzip on", "tzip lite activated"),
        ("tzip verbose 'ultra please' (was full)", tzip_msgs[:2], block + "\n\nultra please", "tzip ultra activated"),
        ("tzip verbose 'off now' (was ultra)", tzip_msgs[:4], block + "\n\noff now", "tzip deactivated"),
        ("tzip case variant 'TZIP FULL' (first message)", [], block + "\n\nTZIP FULL", "tzip full activated"),
        # full rollout: the skill requires reading references/03-hello.md before replying 'world'
        ("example 'Hello' after full tzip sequence (full rollout)", tzip_msgs, ex_block + "\n\nHello", "world"),
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
    say(f"Loaded provider '{PROVIDER}' config for {STUDENT_MODEL} (thinking on) and {TEACHER_MODEL} (thinking off).")
    say(f"API base: {API_BASE[:8]}*** (masked)")

    student, teacher = build_lms(API_BASE, API_KEY, STUDENT_CFG, TEACHER_CFG)

    # sanity check: one cheap student + teacher call
    t_probe = chat(teacher, [{"role": "user", "content": "Reply with the single word: pong"}])
    say(f"Teacher probe: {t_probe['content']!r}")
    s_probe = chat(student, [{"role": "user", "content": "Reply with the single word: pong"}])
    say(f"Student probe: content={s_probe['content']!r} reasoning={(s_probe.get('reasoning_content') or '')[:60]!r}...")

    # ---- frozen base conversation (system + Q&A + step-1 seed + 3 example one-shots) ----
    assert EXAMPLE_FILE.exists() and EXAMPLE_FILE.stat().st_size > 0, f"{EXAMPLE_FILE} missing or empty"
    base_doc = json.loads(EXAMPLE_FILE.read_text())
    base_msgs, TOOLS = base_doc["messages"], base_doc["tools"]
    normalize_tool_arguments(base_msgs)
    i1 = next(idx for idx, m in enumerate(base_msgs) if user_text_of(m).startswith("How does skill system work?"))
    seed_user, seed_content = user_text_of(base_msgs[i1]), base_msgs[i1 + 1]["content"]
    seed_reasoning = base_msgs[i1 + 1]["reasoning_content"]
    say(f"Base conversation: {len(base_msgs)} messages, {len(TOOLS)} tools (frozen; only the step-1 pair at index {i1}-{i1 + 1} is optimized).")

    # ---- tzip ground truth (teacher, thinking off) + LFM-voice reasoning ----
    say(f"Generating {len(SEQ)} tzip invocations (unseen skill) with the teacher model (thinking off)...")
    tzip_msgs, items = generate_tzip_sequence(teacher, base_msgs)

    # ---- initial file with the SEED pair (inspectable before optimization) ----
    write_tzip_file(base_msgs, TOOLS, tzip_msgs, (seed_user, seed_content, seed_reasoning), STUDENT_CFG, tag="seed pair")

    # ---- evaluate: baseline (unoptimized) -> GEPA -> final (optimized) ----
    evaluate = make_evaluator(student, TOOLS, base_msgs)
    seed_candidate = {"step1_user": seed_user, "step1_assistant": seed_content, "step1_reasoning": seed_reasoning}

    baseline_agg = evaluate_all(evaluate, seed_candidate, items, "BASELINE — unoptimized (seed) step-1 pair, no GEPA yet")

    say(f"\nGEPA optimization — {len(items)} validation items, max_metric_calls={MAX_METRIC_CALLS}, reflection LM = {TEACHER_MODEL} (thinking off).")
    dataset = [{**it, "id": f"d-{n + 1}"} for n, it in enumerate(items)]
    valset = [{**it, "id": f"v-{n + 1}"} for n, it in enumerate(items)]
    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluate,
        dataset=dataset,
        valset=valset,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(RUN_DIR),
                seed=7,
                max_metric_calls=MAX_METRIC_CALLS,
                parallel=2,
                display_progress_bar=True,
            ),
            reflection=ReflectionConfig(reflection_lm=make_reflection_lm(teacher), reflection_minibatch_size=3),
            stop_callbacks=[
                ScoreThresholdStopper(threshold=0.95),
                NoImprovementStopper(max_iterations_without_improvement=10),
                TimeoutStopCondition(timeout_seconds=5400),
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

    # ---- final validation of the optimized pair ----
    final_agg = evaluate_all(evaluate, best, items, f"FINAL — optimized step-1 pair (baseline {baseline_agg:.3f} -> {best_score:.3f})")

    # ---- robustness probes (log only) ----
    run_robustness_probes(student, base_msgs, TOOLS, tzip_msgs, best)

    # ---- write the optimized pair into the file + standalone JSON ----
    write_tzip_file(base_msgs, TOOLS, tzip_msgs, (best["step1_user"], best["step1_assistant"], best["step1_reasoning"]), STUDENT_CFG, tag="optimized pair")
    PAIR_FILE.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n")
    say(f"Saved optimized step-1 pair to {PAIR_FILE.name} and into {TZIP_FILE.name}.")
    say("\nOptimized step-1 pair:")
    say(f"  user: {best['step1_user']}")
    say(f"  assistant: {best['step1_assistant']}")
    say(f"  reasoning: {best['step1_reasoning']}")
    say(f"\nSummary: baseline (unoptimized) = {baseline_agg:.3f}, GEPA best val = {best_score:.3f}, final re-check = {final_agg:.3f}.")


if __name__ == "__main__":
    main()
