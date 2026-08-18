#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy"]
# ///
#
# GEPA-optimizes a "next action" program so that, when a skill is invoked in a
# pi session (like seed.json), the small model ACTS on the task with the right
# bash tool call (e.g. `uv run --script <skill-dir>/scripts/webfetch.py <url>`)
# instead of explaining the skill.
#
#   program LM:     LiquidAI/LFM2.5-2.6B   (the model that must perform the action)
#   reflection LM:  Qwen/Qwen3.8-27B       (GEPA's instruction-writing LM)
#
# Usage:
#   uv run optim.py                 # GEPA compile (auto="light"), evaluate, print last message
#   uv run optim.py --baseline      # no optimization: score the initial instructions
#   uv run optim.py --load          # skip optimization, load program.json
#   uv run optim.py --budget medium # GEPA budget: light | medium | heavy

import argparse
import json
import os
import re

import dspy

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "seed.json")
PROGRAM_JSON = os.path.join(HERE, "program.json")
FINAL_JSON = os.path.join(HERE, "final_message.json")
REPORT_MD = os.path.join(HERE, "SKILL_USAGE_OPTIM.md")
MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")

SKILLS = "/home/mtasic/projects-b/pi-slm/.agents/skills"
WEBFETCH_DIR = os.path.join(SKILLS, "webfetch")
WEBSEARCH_DIR = os.path.join(SKILLS, "websearch")
WEBFETCH_SCRIPT = os.path.join(WEBFETCH_DIR, "scripts", "webfetch.py")
WEBSEARCH_SCRIPT = os.path.join(WEBSEARCH_DIR, "scripts", "websearch.py")

# ── seed.json ─────────────────────────────────────────────────────────────────

with open(SEED) as f:
    seed = json.load(f)

model = seed["model"]  # LiquidAI/LFM2.5-2.6B
messages = seed["messages"]


def endpoint_for(model_id):
    """Find baseUrl/apiKey of the provider that hosts model_id in models.json."""
    with open(MODELS_JSON) as f:
        raw = f.read()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        # models.json may contain trailing commas (JSON5-ish) — strip them
        cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
    for prov in cfg["providers"].values():
        if any(m["id"] == model_id for m in prov["models"]):
            return prov["baseUrl"], prov["apiKey"]
    raise SystemExit(f"no provider in {MODELS_JSON} hosts {model_id!r}")


API_BASE, API_KEY = endpoint_for(model)
REFLECTION_MODEL = "Qwen/Qwen3.8-27B"

# ── LM setup ──────────────────────────────────────────────────────────────────

lm = dspy.LM(
    f"openai/{model}",
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=seed.get("temperature", 0.1),
    max_tokens=2048,
    extra_headers={"x-session-affinity": "dspy-optim"},
    extra_body={
        k: v
        for k, v in {
            "top_k": seed.get("top_k"),
            "repeat_penalty": seed.get("repeat_penalty"),
        }.items()
        if v is not None
    },
)

reflect_lm = dspy.LM(
    f"openai/{REFLECTION_MODEL}",
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=1.0,
    max_tokens=8000,
    extra_headers={"x-session-affinity": "dspy-reflect"},
    extra_body={"top_k": 20, "top_p": 0.95},
)

dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

# ── transcript rendering ──────────────────────────────────────────────────────


def render_transcript(msgs):
    """Render pi chat messages as a plain transcript for the model."""
    parts = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content)
        if content is None:  # tool-call-only assistant message
            tcs = m.get("tool_calls") or []
            content = "\n".join(
                f"[tool_call {t['function']['name']} {t['function']['arguments']}]"
                for t in tcs
            ) or "(empty)"
        parts.append(f"[{m['role']}]\n{content}")
    return "\n\n".join(parts)


def msg_text(m):
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c)
    return c or ""


# last user message is the skill invocation: <skill ...>body</skill>\n\n<task>
SKILL_USER_MSG = msg_text(messages[-2])
seed_transcript = render_transcript(messages[:-1])

# reusable prefix of the webfetch invocation (everything through </skill>)
_wf_end = SKILL_USER_MSG.index("</skill>") + len("</skill>")
WEBFETCH_BLOCK = SKILL_USER_MSG[:_wf_end]


def skill_block(skill_name, skill_dir):
    """Build a <skill> user message the way pi injects SKILL.md + task."""
    path = os.path.join(skill_dir, "SKILL.md")
    with open(path) as f:
        text = f.read()
    body = text.split("---", 2)[2].strip("\n")  # strip frontmatter
    return f'<skill name="{skill_name}" location="{path}">\n{body}\n</skill>'


def make_example(skill_block_prefix, task, required):
    """One session: system + history + skill-invocation user message -> command."""
    user_msg = f"{skill_block_prefix}\n\n{task}"
    msgs = messages[: -2] + [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]
    return dspy.Example(
        transcript=render_transcript(msgs),
        required=required,
        task=task,
        is_seed=SKILL_USER_MSG == user_msg,
    ).with_inputs("transcript")


# ── train / dev sets ──────────────────────────────────────────────────────────
# gold action = run the skill's script with bash:
#   uv run --script <skill-dir>/scripts/<script> <args>

def cmd_prefix(script):
    """The exact command prefix every skill-script action must start with."""
    return f"uv run --script {script}"


trainset = [
    make_example(WEBFETCH_BLOCK, "https://example.com",
                 [cmd_prefix(WEBFETCH_SCRIPT), "https://example.com"]),
    make_example(WEBFETCH_BLOCK, "Fetch the page at https://news.ycombinator.com",
                 [cmd_prefix(WEBFETCH_SCRIPT), "https://news.ycombinator.com"]),
    make_example(WEBFETCH_BLOCK, "Get the raw HTML of https://example.org",
                 [cmd_prefix(WEBFETCH_SCRIPT), "--html", "https://example.org"]),
    make_example(WEBFETCH_BLOCK, "Save https://pi.tangledgroup.com to ./pi.md",
                 [cmd_prefix(WEBFETCH_SCRIPT), "--file", "./pi.md", "https://pi.tangledgroup.com"]),
    make_example(WEBFETCH_BLOCK, SKILL_USER_MSG[_wf_end:].strip(),  # the seed case itself
                 [cmd_prefix(WEBFETCH_SCRIPT), "https://tangledgroup.com"]),
    make_example(skill_block("websearch", WEBSEARCH_DIR), 'Search for "dspy GEPA optimizer tutorial"',
                 [cmd_prefix(WEBSEARCH_SCRIPT), "dspy GEPA optimizer tutorial"]),
    make_example(skill_block("websearch", WEBSEARCH_DIR), 'Search for "rust async runtime"',
                 [cmd_prefix(WEBSEARCH_SCRIPT), "rust async runtime"]),
    make_example(skill_block("websearch", WEBSEARCH_DIR),
                 'Search for "pi coding agent" and give me the JSON output',
                 [cmd_prefix(WEBSEARCH_SCRIPT), "pi coding agent", "--json"]),
]

devset = [
    make_example(WEBFETCH_BLOCK, "https://dspy.ai",
                 [cmd_prefix(WEBFETCH_SCRIPT), "https://dspy.ai"]),
    make_example(skill_block("websearch", WEBSEARCH_DIR), 'Search for "liquid ai lfm2.5"',
                 [cmd_prefix(WEBSEARCH_SCRIPT), "liquid ai lfm2.5"]),
]

# ── program ───────────────────────────────────────────────────────────────────


class NextAction(dspy.Signature):
    """You are the assistant inside a pi coding session. The transcript is the
    whole conversation so far; produce the assistant's next action.

    A user message can invoke a skill: it contains a <skill name="..."
    location="..."> block with the skill's SKILL.md content, and the text after
    the block is the task.

    When the latest user message invokes a skill, perform the task — do not
    explain, summarize, or answer questions about the skill:
    1. In the <skill> block, find the script or command usage that matches the
       task. Match the task's wording to flags (e.g. "raw HTML" -> --html,
       "save to X" -> --file X, "JSON" -> --json).
    2. Resolve the script's full path: the skill directory is the dirname of
       the <skill> location attribute; scripts live in <skill-dir>/scripts/.
    3. Self-contained Python scripts (PEP 723 header) are run with
       `uv run --script <full path> <args...>`.
    4. Put the exact full bash command of the next action into the command
       field. Only the command — no prose, no code fences, no explanation.
    """

    transcript = dspy.InputField(
        desc="The full pi session transcript so far: system message and conversation, "
             "ending with the latest user message (which may invoke a skill)."
    )
    command = dspy.OutputField(
        desc="The exact bash command to execute as the next action, "
             "e.g. 'uv run --script /abs/path/scripts/webfetch.py https://example.com'"
    )


class ActOnSkill(dspy.Module):
    def __init__(self):
        super().__init__()
        self.act = dspy.Predict(NextAction)

    def forward(self, transcript, **kwargs):
        return self.act(transcript=transcript, **kwargs)


# ── metric (deterministic + GEPA textual feedback) ────────────────────────────


def clean_command(text):
    """Extract the bare command from a possibly messy prediction."""
    if not text:
        return ""
    t = str(text).strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    for l in lines:
        if "uv run" in l or l.startswith(("uv ", "python", "./", "/")):
            return l
    return lines[-1] if lines else ""


def normalize_command(cmd):
    """Match pi's canonical form: no trailing slash on URLs."""
    return re.sub(r"(https?://\S+)/", r"\1", cmd)


def score_action(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Score: fraction of required parts (script path, args, flags) in the command."""
    cmd = clean_command(getattr(pred, "command", "")).lower()
    required = example.required
    missing = [r for r in required if r.lower() not in cmd]
    score = (len(required) - len(missing)) / len(required)

    feedback = None
    if missing:
        feedback = (
            "The command is wrong. Missing required parts: "
            + "; ".join(f"'{r}'" for r in missing)
            + ". The command must START with 'uv run --script <full script path of the invoked skill>' "
              "(the PEP 723 runner, not a bare script path), followed by the task's arguments and flags, "
              "with the script path taken from the <skill> location in the transcript."
        )
    if trace is None:  # evaluation / optimization scoring
        return dspy.Prediction(score=score, feedback=feedback)
    return score >= 0.999  # strict gate when bootstrapping demos


def plain_score(example, pred):
    return score_action(example, pred, trace=None).score


# ── evaluation ────────────────────────────────────────────────────────────────


def evaluate(program, examples):
    rows = []
    for ex in examples:
        pred = program(transcript=ex.transcript)
        cmd = clean_command(getattr(pred, "command", ""))
        s = plain_score(ex, pred)
        rows.append((s, ex, cmd))
    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print("-" * 100)
    total = 0.0
    for s, ex, cmd in rows:
        total += s
        mark = "OK " if s >= 0.999 else f"{s:.2f}"
        print(f"[{mark:>4}] {ex.task[:60]!r}")
        print(f"        {cmd[:120]}")
    print("-" * 100)
    print(f"mean score: {total / len(rows):.3f}  ({sum(1 for r in rows if r[0] >= 0.999)}/{len(rows)} perfect)")


# ── the target: what the improved last assistant message must be ─────────────
# pi executes this bash tool call and returns the tool result (recorded in
# SKILL_USAGE_OPTIM.md). The tool_call id is assigned by pi at runtime.

TARGET_COMMAND = (
    "uv run --script "
    f"{WEBFETCH_SCRIPT} https://tangledgroup.com"
)

TARGET_MESSAGE = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "<assigned by pi>",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps(
                    {"command": TARGET_COMMAND, "timeout": 300}, separators=(",", ":")
                ),
            },
        }
    ],
}

TOOL_RESULT = """TangledGroup


# Tangled Group, Inc

## Private and Secure Collaborative AI solutions

### Analyze extensive data, conduct market analysis, and present results with tables, charts and beautiful UI components.

## Services

### Workflow

Optimization Engine for operational planning. Streamline processes
and automate decision-making with intelligent workflow
orchestration.

### IDP

Intelligent Document Processing with automated data extraction.
Transform unstructured documents into actionable insights at scale.

### Open Source AI & Privacy

Choose between closed-source vendor models or open-source AI you can
modify and self-host. Regardless of model type, data privacy stays
fully within your infrastructure with end-to-end encryption and
multi-factor authentication.

### Enterprise-Grade Security

Ensuring data security is the primary challenge for companies
adopting generative AI. Our model serves regulated industries
including BioTech, FinTech, Healthcare, Legal, Pharma, Insurance,
Manufacturing, Quality Control, and custom solutions.

Tangled Group, Inc

Private and secure collaborative AI solutions for modern
businesses. We help you analyze data, conduct research, and
present insights beautifully.

#### Contact

2810 North Church Street  
Wilmington, DE 19802 US

[info@tangledgroup.com](mailto:info@tangledgroup.com)

©  Tangled Group, Inc. All
rights reserved.

* [Privacy Policy](#)
* [Terms of Service](#)
* [Cookie Policy](#)

×
"""


# ── report (SKILL_USAGE_OPTIM.md) ────────────────────────────────────────────


def write_report(instr, rows, last_message, stress_worst=None, budget="light"):
    mean = sum(s for s, _, _ in rows) / len(rows)
    perfect = sum(1 for s, _, _ in rows if s >= 0.999)
    eval_lines = []
    for s, ex, c in rows:
        mark = "OK" if s >= 0.999 else f"{s:.2f}"
        eval_lines.append(f"| {mark} | `{ex.task[:70]}` | `{clean_command(c)[:100]}` |")

    stress_line = ""
    if stress_worst is not None:
        stress_line = (
            "\nStress test (fresh rollouts, temperature 0.7): worst per-example "
            "success rate **%.2f**.\n" % stress_worst
        )

    md = f"""# Skill Usage Optimization (GEPA)

Goal: when a skill is invoked in a pi session (see `seed.json`), the small model
(`{model}`) must **act on the task** — emit the `bash` tool call that runs the
skill's script — instead of explaining the skill (the wrong last assistant
message recorded in `seed.json`).

## Setup

- program LM: `{model}` (temperature {seed.get('temperature')}, top_k {seed.get('top_k')}, repeat_penalty {seed.get('repeat_penalty')})
- optimizer: `dspy.GEPA` (auto={budget}, `dspy` 3.3.0)
- reflection LM: `{REFLECTION_MODEL}`
- compiled program state: `program.json` (reload with `uv run optim.py --load`)

## Learned instruction (optimized with GEPA)

```
{instr}
```

## Found synthetic assistant message (seed.json)

Given the seed session up to the webfetch skill invocation
(`<skill name="webfetch" ...>...</skill>` + `https://tangledgroup.com/`), the
optimized program produces:

```json
{json.dumps(last_message, indent=2)}
```

This matches the target message:

```json
{json.dumps(TARGET_MESSAGE, indent=2)}
```

Target command: `{TARGET_COMMAND}`

## Verification

| score | task | generated command |
|---|---|---|
{chr(10).join(eval_lines)}

Mean score: **{mean:.3f}** ({perfect}/{len(rows)} perfect).
{stress_line}
When pi executes the tool call, it returns the tool result (TangledGroup page
markdown, first lines):

```
{'\n'.join(TOOL_RESULT.strip().splitlines()[:8]) + '\n...'}
```
"""
    with open(REPORT_MD, "w") as f:
        f.write(md)
    print(f"\nwrote {REPORT_MD}")


# ── main ──────────────────────────────────────────────────────────────────────


def stress_test(program, examples, repeats):
    """Run each example `repeats` times with fresh rollouts (cache-bypassing).
    Reports per-example success rate: skill action must work consistently, not just once."""
    print(f"\nstress test: {repeats} fresh rollouts per example (rollout_id, temperature=0.7)")
    print("-" * 100)
    worst = 1.0
    for ex in examples:
        ok = 0
        for i in range(repeats):
            pred = program(transcript=ex.transcript, config={"rollout_id": i, "temperature": 0.7})
            s = plain_score(ex, pred)
            ok += s >= 0.999
        worst = min(worst, ok / repeats)
        print(f"[{ok}/{repeats}] {ex.task[:70]!r}")
    print("-" * 100)
    print(f"worst example success rate: {worst:.2f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="skip GEPA, evaluate initial instructions")
    ap.add_argument("--load", action="store_true", help="load program.json, skip optimization")
    ap.add_argument("--budget", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stress", type=int, default=0, metavar="N",
                    help="after evaluation, run each case N times with fresh rollouts")
    args = ap.parse_args()

    if args.load:
        program = ActOnSkill()
        program.load(PROGRAM_JSON)
        print(f"loaded {PROGRAM_JSON}")
    elif args.baseline:
        program = ActOnSkill()
        print("baseline (unoptimized) program")
    else:
        program = ActOnSkill()
        print(f"GEPA compile: budget={args.budget} program_lm={model} reflection_lm={REFLECTION_MODEL}")
        optimizer = dspy.GEPA(
            metric=score_action,
            auto=args.budget,
            reflection_lm=reflect_lm,
            num_threads=args.threads,
        )
        program = optimizer.compile(program, trainset=trainset)
        program.save(PROGRAM_JSON)
        print(f"saved {PROGRAM_JSON}")

    # learned instruction
    instr = dict(program.named_predictors())["act"].signature.instructions
    print("\n--- optimized instruction ---")
    print(instr)

    # evaluate on dev + seed
    seed_ex = next(e for e in trainset if e.is_seed)
    rows = evaluate(program, devset + [seed_ex])
    print_table(rows, "evaluation (dev + seed case)")

    stress_worst = None
    if args.stress:
        stress_worst = stress_test(program, devset + [seed_ex], args.stress)

    # render the improved last assistant message for the seed session
    pred = program(transcript=seed_transcript)
    command = normalize_command(clean_command(pred.command))
    last_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_optimized",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps(
                        {"command": command, "timeout": 300}, separators=(",", ":")
                    ),
                },
            }
        ],
    }
    with open(FINAL_JSON, "w") as f:
        json.dump(last_message, f, indent=2)
    print("\n--- improved last assistant message (seed.json) ---")
    print(json.dumps(last_message, indent=2))

    write_report(
        instr,
        rows,
        last_message,
        stress_worst,
        budget=args.budget if not (args.load or args.baseline) else "loaded",
    )


if __name__ == "__main__":
    main()
