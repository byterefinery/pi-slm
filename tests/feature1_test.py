#!/usr/bin/env python3
"""
Feature 1 tests for the slm extension (src/slm.ts).

The extension simulates a short user/assistant dialogue at the start of a
new session, so small language models get the available skills/tools
reminder in the conversation context (not only in the system prompt):

    1. system message                       (pi default)
    2. user:      "Available skills"         (simulated)
    3. assistant: synthetic thinking + skills as YAML
    4. user:      "Available tools"          (simulated)
    5. assistant: synthetic thinking + tools as YAML
    6. user:      the first real user request

Runs `pi` in random temp dirs under /tmp with an isolated agent dir
(temp HOME + PI_CODING_AGENT_SESSION_DIR), model
LiquidAI/LFM2.5-2.6B, and validates BOTH the session JSONL and the actual
provider payload (captured by tests/payload-logger.ts via the
before_provider_request event, file in $SLM_PAYLOAD_LOG):

  S1  skills listed correctly          (default run; skills+tools present)
  S2  tools listed correctly           (default 4-tool set, exact signatures)
  S3  restricted tools                 (-t read,grep -> listing follows active set)
  S4  no skills                        (--no-skills -> skills: [])
  S5  no tools                         (-nt -> tools: [])
  S6  repeated prompts (continued)     (single injection; restored dialogue
                                          at the head of the wire context)
  S7  non-reasoning model              (no synthetic thinking blocks)
  S8  JSON mode                        (ask messages emitted as display:true
                                          message events -> visible in TUI)
  S9  user message verbatim + tools    (first user request text unchanged;
                                          top-level 'tools' field identical
                                          with/without the extension -> pi core)

Wire reasoning: for the openai-completions API (llama.cpp) each synthetic
assistant message carries a `reasoning_content` field with the short
synthetic reasoning (content stays the pure YAML document).

Ordering invariant: [bookkeeping entries] -> ask "Available skills"
(custom_message slm-skills) -> assistant skills YAML (message entry) ->
ask "Available tools" (custom_message slm-tools) -> assistant tools YAML
-> first real user message. The wire context starts with system, then the
same dialogue, then the user request.

Exit code: 0 = all passed, 1 = at least one failure.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "src" / "slm.ts"
LOGGER_EXT = REPO / "tests" / "payload-logger.ts"
REAL_AGENT = Path.home() / ".pi" / "agent"
MODEL = "LiquidAI/LFM2.5-2.6B"

SKILL_ALPHA_DESC = (
    'Alpha test skill: handles "quotes", colons: and #hashes '
    "for YAML escaping checks."
)
SKILL_BETA_DESC = "Beta test skill without references or scripts."

ASK_SKILLS = "Available skills"
ASK_TOOLS = "Available tools"

# Must mirror the thinking lines in src/slm.ts.
def exp_skills_thinking(n: int) -> str:
    return (f"scanned loaded skills - {n} found. I will check whether the "
            f"task matches a description, and if so read that skill's "
            f"SKILL.md and the reference files listed below.")


def exp_tools_thinking(n: int) -> str:
    return (f"scanned active tools - {n} found. I will pick the narrowest "
            f"tool that fits the task.")

# pi 0.84.2 system-prompt snippets for the built-in tools.
TOOL_SNIPPETS = {
    "read": "Read file contents",
    "bash": "Execute bash commands (ls, grep, find, etc.)",
    "edit": (
        "Make precise file edits with exact text replacement, "
        "including multiple disjoint edits in one call"
    ),
    "write": "Create or overwrite files",
    "grep": "Search file contents for patterns (respects .gitignore)",
}

failures: list[str] = []
passes = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passes
    if cond:
        passes += 1
        print(f"  [PASS] {name}")
    else:
        failures.append(name)
        print(f"  [FAIL] {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# pi runner / session helpers
# --------------------------------------------------------------------------

def run_pi(home: Path, session_dir: Path, cwd: Path,
           extra_args: list[str], prompt: str,
           continue_session: bool = False, mode: str = "text",
           payload_log: Path | None = None,
           with_ext: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # fully isolated environment: temp HOME (no global skills/settings/
    # ~/.agents/skills), sessions in the temp session dir
    env["HOME"] = str(home)
    env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
    env.pop("PI_CODING_AGENT_DIR", None)
    if payload_log is not None:
        env["SLM_PAYLOAD_LOG"] = str(payload_log)
    args = ["pi", "--offline", "-a"]
    if with_ext:
        args += ["-e", str(EXT)]
    args += ["-e", str(LOGGER_EXT), "--model", MODEL]
    if continue_session:
        args.append("-c")
    if mode != "text":
        args += ["--mode", mode]
    args += extra_args
    if mode == "text":
        args += ["-p", prompt]
    else:
        args.append(prompt)
    return subprocess.run(
        args, cwd=str(cwd), env=env,
        capture_output=True, text=True, timeout=300,
    )


def setup_home(base: Path, reasoning: bool = True) -> Path:
    """Temp HOME with an agent dir holding a minimal strict models.json:
    the llamacpp provider (copied verbatim from the user's config, so the
    API key stays out of this file) plus only the LFM test model with the
    requested reasoning flag. Returns the home dir."""
    home = base / "home"
    agent = home / ".pi" / "agent"
    agent.mkdir(parents=True, exist_ok=True)
    raw = (REAL_AGENT / "models.json").read_text()
    # the user's file is lenient JSON (trailing commas); make it strict
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    models = json.loads(raw)
    prov = models["providers"]["llamacpp"]
    lfm = next(m for m in prov["models"] if m["id"] == MODEL)
    lfm["reasoning"] = reasoning
    minimal = {"providers": {"llamacpp": {**prov, "models": [lfm]}}}
    (agent / "models.json").write_text(json.dumps(minimal, indent=2))
    return home


def write_fixture_skills(cwd: Path) -> dict:
    """Create fixture skills; returns {name: {description, references, scripts}}
    for the skills that SHOULD be listed (hidden skill excluded)."""
    alpha = cwd / ".pi" / "skills" / "slm-alpha"
    (alpha / "references" / "deep").mkdir(parents=True)
    (alpha / "scripts" / "inner").mkdir(parents=True)
    (alpha / "SKILL.md").write_text(
        "---\n"
        "name: slm-alpha\n"
        f'description: "{SKILL_ALPHA_DESC.replace(chr(34), chr(92) + chr(34))}"\n'
        "---\n# SLM Alpha\nBody.\n"
    )
    (alpha / "references" / "api.md").write_text("api docs\n")
    (alpha / "references" / "deep" / "guide.md").write_text("deep docs\n")
    (alpha / "scripts" / "run.sh").write_text("#!/bin/sh\necho run\n")
    (alpha / "scripts" / "inner" / "helper.py").write_text("def h(): pass\n")

    beta = cwd / ".pi" / "skills" / "slm-beta"
    beta.mkdir(parents=True)
    (beta / "SKILL.md").write_text(
        "---\nname: slm-beta\n"
        f"description: {SKILL_BETA_DESC}\n---\n# SLM Beta\n"
    )

    hidden = cwd / ".pi" / "skills" / "slm-hidden"
    hidden.mkdir(parents=True)
    (hidden / "SKILL.md").write_text(
        "---\nname: slm-hidden\n"
        "description: Hidden skill that must NOT be listed.\n"
        "disable-model-invocation: true\n---\n# SLM Hidden\n"
    )

    return {
        "slm-alpha": {
            "description": SKILL_ALPHA_DESC,
            "references": [
                str(alpha / "references" / "api.md"),
                str(alpha / "references" / "deep" / "guide.md"),
            ],
            "scripts": [
                str(alpha / "scripts" / "inner" / "helper.py"),
                str(alpha / "scripts" / "run.sh"),
            ],
        },
        "slm-beta": {
            "description": SKILL_BETA_DESC,
            "references": [],
            "scripts": [],
        },
    }


def read_session(session_dir: Path, cwd: Path):
    """Return (entries, session_file) of the most recent session for cwd.

    With PI_CODING_AGENT_SESSION_DIR the files sit directly in the session
    dir; with the default layout they are grouped in a --<cwd>-- subdir.
    """
    safe = "--" + str(cwd).lstrip("/").replace("/", "-") + "--"
    files = list(session_dir.glob("*.jsonl")) or list(
        (session_dir / safe).glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"no session file found in {session_dir}")
    f = max(files, key=lambda p: p.stat().st_mtime)
    # sanity: the session must belong to this cwd
    lines = [line for line in f.read_text().splitlines() if line.strip()]
    header = json.loads(lines[0])
    assert header.get("cwd") == str(cwd), f"session {f} is for {header.get('cwd')}"
    # skip the session header (type "session"); it is not a tree entry
    entries = [json.loads(line) for line in lines[1:]]
    return entries, f


def read_payloads(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def wire_text(content) -> str:
    """Plain text of a wire message content (str or content-block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


# --------------------------------------------------------------------------
# session-dialogue checks
# --------------------------------------------------------------------------

def find_ask(entries, custom_type: str) -> list[dict]:
    return [e for e in entries
            if e.get("type") == "custom_message"
            and e.get("customType") == custom_type]


def assistant_after(entries, idx: int) -> dict | None:
    """The message entry directly after entries[idx], if it is an
    assistant message; else None."""
    j = idx + 1
    if j >= len(entries):
        return None
    e = entries[j]
    if e.get("type") != "message" or e.get("message", {}).get("role") != "assistant":
        return None
    return e


def check_dialogue(entries, scenario: str, expect_user_count: int = 1):
    """Validate the simulated dialogue in the session:
       ask_skills -> assistant -> ask_tools -> assistant -> user(s)."""
    asks_s = find_ask(entries, "slm-skills")
    asks_t = find_ask(entries, "slm-tools")
    users = [i for i, e in enumerate(entries)
             if e.get("type") == "message"
             and e.get("message", {}).get("role") == "user"]
    check(f"{scenario}: exactly one 'Available skills' ask", len(asks_s) == 1,
          f"found {len(asks_s)}")
    check(f"{scenario}: exactly one 'Available tools' ask", len(asks_t) == 1,
          f"found {len(asks_t)}")
    check(f"{scenario}: {expect_user_count} real user message(s)",
          len(users) == expect_user_count, f"found {len(users)}")
    if len(asks_s) != 1 or len(asks_t) != 1 or len(users) < expect_user_count:
        return None

    i_s, i_t = entries.index(asks_s[0]), entries.index(asks_t[0])
    a1, a2 = assistant_after(entries, i_s), assistant_after(entries, i_t)
    check(f"{scenario}: assistant message directly after skills ask",
          a1 is not None, f"next entry: {entries[i_s + 1].get('type') if i_s + 1 < len(entries) else 'EOF'}")
    check(f"{scenario}: assistant message directly after tools ask",
          a2 is not None, f"next entry: {entries[i_t + 1].get('type') if i_t + 1 < len(entries) else 'EOF'}")
    check(f"{scenario}: order ask-skills -> asst -> ask-tools -> asst -> user",
          i_s < i_s + 1 < i_t < i_t + 1 < users[0],
          f"askS@{i_s} askT@{i_t} user@{users[0]}")
    preceding = entries[:i_s]
    check(f"{scenario}: only bookkeeping entries precede the dialogue",
          len(preceding) > 0 and all(
              e.get("type") in ("model_change", "thinking_level_change")
              for e in preceding),
          f"preceding: {[e.get('type') for e in preceding]}")

    if a1 is None or a2 is None:
        return None

    check(f"{scenario}: skills ask content is exactly {ASK_SKILLS!r}",
          asks_s[0].get("content") == ASK_SKILLS, repr(asks_s[0].get("content")))
    check(f"{scenario}: tools ask content is exactly {ASK_TOOLS!r}",
          asks_t[0].get("content") == ASK_TOOLS, repr(asks_t[0].get("content")))
    check(f"{scenario}: asks are displayed (visible in TUI)",
          asks_s[0].get("display") is True and asks_t[0].get("display") is True,
          f"display: {asks_s[0].get('display')}, {asks_t[0].get('display')}")
    return {"ask_s": asks_s[0], "ask_t": asks_t[0], "a1": a1, "a2": a2,
            "users": users}


def check_assistant_entry(e: dict, scenario: str, label: str,
                          expect_thinking: bool, thinking_prefix: str):
    """Validate one synthetic assistant message entry; returns its text."""
    m = e["message"]
    check(f"{scenario}: {label} entry role is assistant", m.get("role") == "assistant")
    check(f"{scenario}: {label} stopReason is 'stop'", m.get("stopReason") == "stop",
          repr(m.get("stopReason")))
    u = m.get("usage")
    check(f"{scenario}: {label} usage is zero (synthetic)",
          isinstance(u, dict) and u.get("input") == 0 and u.get("output") == 0
          and u.get("totalTokens") == 0, repr(u)[:160])
    check(f"{scenario}: {label} carries model metadata",
          bool(m.get("provider")) and bool(m.get("model")) and bool(m.get("api")),
          f"provider={m.get('provider')!r} model={m.get('model')!r} api={m.get('api')!r}")
    content = m.get("content")
    check(f"{scenario}: {label} content is a block list",
          isinstance(content, list) and content, repr(content)[:160])
    if not isinstance(content, list):
        return ""
    thinking = [b for b in content if b.get("type") == "thinking"]
    text_blocks = [b for b in content if b.get("type") == "text"]
    if expect_thinking:
        check(f"{scenario}: {label} has exactly one thinking block",
              len(thinking) == 1, f"found {len(thinking)}")
        if thinking:
            t = thinking[0].get("thinking", "")
            check(f"{scenario}: {label} thinking starts with {thinking_prefix!r}",
                  t.startswith(thinking_prefix), repr(t)[:160])
            check(f"{scenario}: {label} thinking is short (< 300 chars)",
                  0 < len(t) < 300, f"len={len(t)}")
            check(f"{scenario}: {label} thinking carries reasoning_content "
                  f"signature (openai-completions wire replay)",
                  thinking[0].get("thinkingSignature") == "reasoning_content",
                  repr(thinking[0].get("thinkingSignature")))
    else:
        check(f"{scenario}: {label} has no thinking block", len(thinking) == 0,
              f"found {len(thinking)}")
    check(f"{scenario}: {label} has exactly one text block", len(text_blocks) == 1,
          f"found {len(text_blocks)}")
    return text_blocks[0].get("text", "") if text_blocks else ""


def check_skills_yaml(text: str, expected: dict, scenario: str) -> None:
    try:
        doc = yaml.safe_load(text)
        check(f"{scenario}: skills YAML parses", True)
    except yaml.YAMLError as e:
        check(f"{scenario}: skills YAML parses", False, str(e))
        return
    check(f"{scenario}: skills YAML top level is a mapping",
          isinstance(doc, dict), repr(doc)[:120])
    skills = doc.get("skills") if isinstance(doc, dict) else None
    if not isinstance(skills, list):
        check(f"{scenario}: skills YAML has 'skills' list", False, repr(doc)[:200])
        return
    check(f"{scenario}: skills YAML has 'skills' list", True)
    names = [s.get("name") for s in skills]
    check(f"{scenario}: listed skill names", sorted(names) == sorted(expected),
          f"got {names}")
    # token saving: skill names are plain (unquoted) scalars
    for name in expected:
        check(f"{scenario}: skill name '{name}' is a plain (unquoted) scalar",
              re.search(rf"^\s*- name: {re.escape(name)}\n", text, flags=re.M) is not None,
              repr(text))
    # the alpha description contains ': ' -> must fall back to double quotes
    if "slm-alpha" in expected:
        check(f"{scenario}: description containing ': ' is double-quoted",
              'description: "Alpha test skill: handles' in text, repr(text)[:300])
    for name, exp in expected.items():
        entry = next((s for s in skills if s.get("name") == name), None)
        if entry is None:
            continue
        check(f"{scenario}: skill '{name}' has single-line description",
              isinstance(entry.get("description"), str)
              and "\n" not in entry["description"]
              and entry["description"] == exp["description"],
              repr(entry.get("description")))
        refs = entry.get("references")
        scripts = entry.get("scripts")
        check(f"{scenario}: skill '{name}' references are absolute paths",
              isinstance(refs, list)
              and sorted(refs) == sorted(exp["references"])
              and all(str(r).startswith("/") and Path(r).is_file() for r in refs),
              f"got {refs}, expected {exp['references']}")
        check(f"{scenario}: skill '{name}' scripts are absolute paths",
              isinstance(scripts, list)
              and sorted(scripts) == sorted(exp["scripts"])
              and all(str(s).startswith("/") and Path(s).is_file() for s in scripts),
              f"got {scripts}, expected {exp['scripts']}")
    check(f"{scenario}: hidden skill not listed", "slm-hidden" not in names,
          f"got {names}")


def check_tools_yaml(text: str, expected: list[str], scenario: str) -> None:
    try:
        doc = yaml.safe_load(text)
        check(f"{scenario}: tools YAML parses", True)
    except yaml.YAMLError as e:
        check(f"{scenario}: tools YAML parses", False, str(e))
        return
    check(f"{scenario}: tools YAML top level is a mapping",
          isinstance(doc, dict), repr(doc)[:120])
    tools = doc.get("tools") if isinstance(doc, dict) else None
    if not isinstance(tools, list):
        check(f"{scenario}: tools YAML has 'tools' list", False, repr(doc)[:200])
        return
    check(f"{scenario}: tools YAML has 'tools' list", True)
    names = [t.get("name") for t in tools]
    check(f"{scenario}: listed tool names (order preserved)",
          names == expected, f"got {names}, expected {expected}")
    for t in tools:
        name = t.get("name")
        desc = t.get("description")
        check(f"{scenario}: tool '{name}' has single-line description",
              isinstance(desc, str) and desc != "" and "\n" not in desc,
              repr(desc))
        if name in TOOL_SNIPPETS:
            check(f"{scenario}: tool '{name}' description matches pi snippet",
                  desc == TOOL_SNIPPETS[name],
                  f"got {desc!r}, expected {TOOL_SNIPPETS[name]!r}")
            # token saving: snippet descriptions are plain (unquoted) scalars
            check(f"{scenario}: tool '{name}' description is a plain scalar",
                  re.search(rf"^\s+description: {re.escape(TOOL_SNIPPETS[name])}\n",
                            text, flags=re.M) is not None,
                  repr(text))
        check_signature(t, scenario)


def check_signature(t: dict, scenario: str) -> None:
    """Whole function signature: parameters must be the JSON schema as YAML."""
    name = t.get("name")
    params = t.get("parameters")
    check(f"{scenario}: tool '{name}' has 'parameters' signature",
          isinstance(params, dict), repr(params)[:200])
    if not isinstance(params, dict):
        return
    check(f"{scenario}: tool '{name}' parameters.type is object",
          params.get("type") == "object", repr(params)[:200])
    props = params.get("properties")
    check(f"{scenario}: tool '{name}' parameters.properties is a mapping",
          isinstance(props, dict) and props, repr(params)[:200])
    if not isinstance(props, dict):
        return
    for pname, pschema in props.items():
        check(f"{scenario}: tool '{name}' param '{pname}' has a type",
              isinstance(pschema, dict) and isinstance(pschema.get("type"), str),
              repr(pschema)[:200])
    if name == "read":
        check(f"{scenario}: read signature exact",
              params.get("required") == ["path"]
              and props["path"]["type"] == "string"
              and props["offset"]["type"] == "number"
              and props["limit"]["type"] == "number",
              repr(params)[:400])
    if name == "bash":
        check(f"{scenario}: bash signature exact",
              params.get("required") == ["command"]
              and props["command"]["type"] == "string"
              and props["timeout"]["type"] == "number",
              repr(params)[:400])
    if name == "edit":
        check(f"{scenario}: edit signature exact (nested array+items)",
              params.get("required") == ["path", "edits"]
              and props["edits"]["type"] == "array"
              and props["edits"]["items"]["type"] == "object"
              and props["edits"]["items"].get("required") == ["oldText", "newText"],
              repr(params)[:600])


# --------------------------------------------------------------------------
# wire (provider payload) checks
# --------------------------------------------------------------------------

def check_wire(payloads: list[dict], scenario: str, prompt: str,
               skills_yaml: str, tools_yaml: str,
               prompt_index: int = 5,
               expect_history: list[tuple[str, str]] | None = None,
               exp_skills_thinking: str | None = None,
               exp_tools_thinking: str | None = None) -> None:
    """The first provider payload must carry:
       system, user 'Available skills', assistant <skills yaml>,
       user 'Available tools', assistant <tools yaml>, then (for
       continued sessions) the restored history, then user <prompt>.
       expect_history describes the messages between index 5 and
       prompt_index (text None = any text). For reasoning models the
       synthetic assistant messages must carry a `reasoning_content`
       field with the expected thinking text; for non-reasoning models
       no reasoning field may be present."""
    msgs = next((p.get("messages") for p in payloads
                 if isinstance(p, dict) and isinstance(p.get("messages"), list)),
                None)
    if msgs is None:
        check(f"{scenario}: wire payload with messages captured", False,
              f"{len(payloads)} payloads logged")
        return
    check(f"{scenario}: wire payload with messages captured", True)
    need = [
        ("system", msgs[0].get("role"), None),
        ("user asks 'Available skills'",
         msgs[1].get("role") == "user" and wire_text(msgs[1].get("content")) == ASK_SKILLS,
         f"role={msgs[1].get('role')} text={wire_text(msgs[1].get('content'))[:60]!r}"),
        ("assistant replies with skills YAML",
         msgs[2].get("role") == "assistant" and wire_text(msgs[2].get("content")) == skills_yaml,
         f"role={msgs[2].get('role')} text={wire_text(msgs[2].get('content'))[:80]!r}"),
        ("user asks 'Available tools'",
         msgs[3].get("role") == "user" and wire_text(msgs[3].get("content")) == ASK_TOOLS,
         f"role={msgs[3].get('role')} text={wire_text(msgs[3].get('content'))[:60]!r}"),
        ("assistant replies with tools YAML",
         msgs[4].get("role") == "assistant" and wire_text(msgs[4].get("content")) == tools_yaml,
         f"role={msgs[4].get('role')} text={wire_text(msgs[4].get('content'))[:80]!r}"),
        ("current user request at the expected position",
         msgs[prompt_index].get("role") == "user"
         and wire_text(msgs[prompt_index].get("content")) == prompt,
         f"idx={prompt_index} role={msgs[prompt_index].get('role')} text={wire_text(msgs[prompt_index].get('content'))[:60]!r}"),
    ]
    if len(msgs) < max(6, prompt_index + 1):
        check(f"{scenario}: wire has >= {max(6, prompt_index + 1)} messages",
              False, f"got {len(msgs)}")
        return
    for name, cond, detail in need:
        check(f"{scenario}: wire: {name}", cond, detail)
    # the assistant reply content on the wire is a plain string (the
    # OpenAI-completions serializer flattens text; the synthetic thinking
    # block is not sent for this provider)
    check(f"{scenario}: wire assistant content is a plain string (pure YAML)",
          isinstance(msgs[2].get("content"), str)
          and isinstance(msgs[4].get("content"), str),
          f"skills: {type(msgs[2].get('content')).__name__}, tools: {type(msgs[4].get('content')).__name__}")
    for idx, expected, label in (
        (2, exp_skills_thinking, "skills assistant"),
        (4, exp_tools_thinking, "tools assistant"),
    ):
        if expected is None:
            check(f"{scenario}: wire {label} has no reasoning fields",
                  not ({"thinking", "reasoning", "reasoning_content"}
                       & set(msgs[idx])),
                  f"keys: {sorted(set(msgs[idx]))}")
        else:
            check(f"{scenario}: wire {label} carries reasoning_content",
                  msgs[idx].get("reasoning_content") == expected,
                  f"got {msgs[idx].get('reasoning_content')!r}")
    if expect_history:
        got = [(m.get("role"), wire_text(m.get("content")))
               for m in msgs[5:prompt_index]]
        check(f"{scenario}: wire history between dialogue and request",
              len(got) == len(expect_history)
              and all(g[0] == e[0] and (e[1] is None or g[1] == e[1])
                      for g, e in zip(got, expect_history)),
              f"got {[(g[0], g[1][:40]) for g in got]}")


# --------------------------------------------------------------------------
# scenario runner
# --------------------------------------------------------------------------

PROMPT = "Reply with the single word: done"


def new_workspace(tag: str, reasoning: bool = True):
    """Random temp cwd + isolated HOME/session dirs."""
    root = Path(tempfile.mkdtemp(prefix=f"slm-f1-{tag}-", dir="/tmp"))
    home = setup_home(root, reasoning=reasoning)
    session_dir = root / "sessions"
    cwd = root / "work"
    cwd.mkdir()
    expected = write_fixture_skills(cwd)
    return root, home, session_dir, cwd, expected


def run_and_check_dialogue(tag: str, skills_label: str, tools_label: str,
                           extra_args: list[str], reasoning: bool,
                           prompt: str = PROMPT,
                           expected_tools: list[str] | None = None):
    """One fresh-session run: session dialogue + wire context checks.
    Returns (skills_yaml, tools_yaml) texts (or ("", "") on failure)."""
    expected_tools = expected_tools or ["read", "bash", "edit", "write"]
    root, home, session_dir, cwd, expected_skills = new_workspace(tag,
                                                                  reasoning=reasoning)
    WORKSPACES.append(root)
    payload_log = root / "payloads.jsonl"
    proc = run_pi(home, session_dir, cwd, extra_args, prompt,
                  payload_log=payload_log)
    check(f"{skills_label}: pi exited 0", proc.returncode == 0,
          proc.stdout[-400:] + proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    dlg = check_dialogue(entries, skills_label)
    payloads = read_payloads(payload_log)
    check(f"{skills_label}: at least one provider payload captured",
          len(payloads) >= 1, f"{len(payloads)} payloads")
    if dlg is None:
        return "", ""
    skills_yaml = check_assistant_entry(
        dlg["a1"], skills_label, "skills assistant", expect_thinking=reasoning,
        thinking_prefix="scanned loaded skills -")
    tools_yaml = check_assistant_entry(
        dlg["a2"], tools_label, "tools assistant", expect_thinking=reasoning,
        thinking_prefix="scanned active tools -")
    check_skills_yaml(skills_yaml, expected_skills, skills_label)
    check_tools_yaml(tools_yaml, expected_tools, tools_label)
    check_wire(payloads, skills_label, prompt, skills_yaml, tools_yaml,
               exp_skills_thinking=(exp_skills_thinking(len(expected_skills))
                                    if reasoning else None),
               exp_tools_thinking=(exp_tools_thinking(len(expected_tools))
                                   if reasoning else None))
    return skills_yaml, tools_yaml


WORKSPACES: list[Path] = []


def main() -> int:
    # ------------------------------------------------------------------ S1+S2
    print("S1: skills listed correctly (default: skills + tools)")
    print("S2: tools listed correctly (shares the S1 run)")
    skills_yaml, tools_yaml = run_and_check_dialogue(
        "s1", "S1", "S2", [], reasoning=True)
    check("S2: tools YAML lists the default 4 tools in order",
          [t.get("name") for t in yaml.safe_load(tools_yaml)["tools"]]
          == ["read", "bash", "edit", "write"] if tools_yaml else False)

    # ------------------------------------------------------------------ S3
    print("S3: restricted tools (-t read,grep)")
    run_and_check_dialogue(
        "s3", "S3", "S3", ["-t", "read,grep"], reasoning=True,
        expected_tools=["read", "grep"])

    # ------------------------------------------------------------------ S4
    print("S4: no skills (--no-skills)")
    root, home, session_dir, cwd, _ = new_workspace("s4")
    WORKSPACES.append(root)
    payload_log = root / "payloads.jsonl"
    proc = run_pi(home, session_dir, cwd, ["--no-skills"], PROMPT,
                  payload_log=payload_log)
    check("S4: pi exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    dlg = check_dialogue(entries, "S4")
    if dlg:
        skills_yaml = check_assistant_entry(
            dlg["a1"], "S4", "skills assistant", expect_thinking=True,
            thinking_prefix="scanned loaded skills -")
        tools_yaml = check_assistant_entry(
            dlg["a2"], "S4", "tools assistant", expect_thinking=True,
            thinking_prefix="scanned active tools -")
        check("S4: skills YAML is an empty list",
              yaml.safe_load(skills_yaml) == {"skills": []}
              and skills_yaml.strip() == "skills: []", repr(skills_yaml)[:120])
        t0 = dlg["a1"]["message"]["content"][0]
        check("S4: skills thinking says 0 found",
              t0.get("type") == "thinking" and "0 found" in t0.get("thinking", ""),
              repr(t0)[:200])
        check_tools_yaml(tools_yaml, ["read", "bash", "edit", "write"], "S4")
        check_wire(read_payloads(payload_log), "S4", PROMPT, skills_yaml,
                   tools_yaml,
                   exp_skills_thinking=exp_skills_thinking(0),
                   exp_tools_thinking=exp_tools_thinking(4))

    # ------------------------------------------------------------------ S5
    print("S5: no tools (-nt)")
    root, home, session_dir, cwd, expected_skills = new_workspace("s5")
    WORKSPACES.append(root)
    payload_log = root / "payloads.jsonl"
    proc = run_pi(home, session_dir, cwd, ["-nt"], PROMPT,
                  payload_log=payload_log)
    check("S5: pi exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    dlg = check_dialogue(entries, "S5")
    if dlg:
        skills_yaml = check_assistant_entry(
            dlg["a1"], "S5", "skills assistant", expect_thinking=True,
            thinking_prefix="scanned loaded skills -")
        tools_yaml = check_assistant_entry(
            dlg["a2"], "S5", "tools assistant", expect_thinking=True,
            thinking_prefix="scanned active tools -")
        check_skills_yaml(skills_yaml, expected_skills, "S5")
        check("S5: tools YAML is an empty list",
              yaml.safe_load(tools_yaml) == {"tools": []}
              and tools_yaml.strip() == "tools: []", repr(tools_yaml)[:120])
        check_wire(read_payloads(payload_log), "S5", PROMPT, skills_yaml,
                   tools_yaml,
                   exp_skills_thinking=exp_skills_thinking(len(expected_skills)),
                   exp_tools_thinking=exp_tools_thinking(0))

    # ------------------------------------------------------------------ S6
    print("S6: repeated prompts (session continued; dialogue restored)")
    root, home, session_dir, cwd, expected_skills = new_workspace("s6")
    WORKSPACES.append(root)
    log1 = root / "payloads1.jsonl"
    log2 = root / "payloads2.jsonl"
    skills_yaml = tools_yaml = ""
    proc = run_pi(home, session_dir, cwd, [], PROMPT, payload_log=log1)
    check("S6: first run exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    dlg = check_dialogue(entries, "S6")
    if dlg:
        skills_yaml = check_assistant_entry(
            dlg["a1"], "S6", "skills assistant", expect_thinking=True,
            thinking_prefix="scanned loaded skills -")
        tools_yaml = check_assistant_entry(
            dlg["a2"], "S6", "tools assistant", expect_thinking=True,
            thinking_prefix="scanned active tools -")
    if skills_yaml:
        check_wire(read_payloads(log1), "S6-run1", PROMPT, skills_yaml,
                   tools_yaml,
                   exp_skills_thinking=exp_skills_thinking(len(expected_skills)),
                   exp_tools_thinking=exp_tools_thinking(4))

    proc = run_pi(home, session_dir, cwd, [], "Reply with the single word: again",
                  continue_session=True, payload_log=log2)
    check("S6: continued run exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    dlg2 = check_dialogue(entries, "S6-run2", expect_user_count=2)
    if dlg2 is not None:
        i_s = entries.index(dlg2["ask_s"])
        i_t = entries.index(dlg2["ask_t"])
        i_u1, i_u2 = dlg2["users"]
        after_first_prompt = [e for e in entries[i_u1:]
                              if e.get("type") == "custom_message"]
        check("S6-run2: no re-injection after the first prompt",
              not any(e.get("customType") in ("slm-skills", "slm-tools")
                      for e in after_first_prompt),
              f"got {[e.get('customType') for e in after_first_prompt]}")
        check("S6-run2: dialogue sits before both user messages",
              i_s < i_t < i_u1 < i_u2,
              f"askS@{i_s} askT@{i_t} users@{dlg2['users']}")
        # the restored run must carry the SAME assistant texts (from the
        # session file) and the wire context starts with the full dialogue
        a1b = assistant_after(entries, i_s)
        a2b = assistant_after(entries, i_t)
        same = (a1b is not None and a2b is not None
                and a1b["message"]["content"][-1]["text"] == skills_yaml
                and a2b["message"]["content"][-1]["text"] == tools_yaml)
        check("S6-run2: restored assistant texts identical to run 1", same)
        check_wire(
            read_payloads(log2), "S6-run2", "Reply with the single word: again",
            skills_yaml, tools_yaml,
            prompt_index=7,
            expect_history=[
                ("user", PROMPT),          # the first request
                ("assistant", None),       # real reply to it
            ],
            exp_skills_thinking=exp_skills_thinking(len(expected_skills)),
            exp_tools_thinking=exp_tools_thinking(4))

    # ------------------------------------------------------------------ S7
    print("S7: non-reasoning model (no synthetic thinking blocks)")
    run_and_check_dialogue("s7", "S7", "S7", [], reasoning=False)

    # ------------------------------------------------------------------ S8
    print("S8: JSON mode — ask messages emitted as display events")
    root, home, session_dir, cwd, expected_skills = new_workspace("s8")
    WORKSPACES.append(root)
    payload_log = root / "payloads.jsonl"
    s8_skills = s8_tools = ""
    proc = run_pi(home, session_dir, cwd, [], PROMPT, mode="json",
                  payload_log=payload_log)
    check("S8: pi exited 0", proc.returncode == 0,
          proc.stdout[-400:] + proc.stderr[-400:])
    events = [json.loads(line) for line in proc.stdout.splitlines()
              if line.strip()]
    msg_starts = [e for e in events
                  if e.get("type") == "message_start"
                  and isinstance(e.get("message"), dict)]
    asks = [e["message"] for e in msg_starts
            if e["message"].get("role") == "custom"
            and e["message"].get("customType") in ("slm-skills", "slm-tools")]
    check("S8: message_start events for both asks emitted",
          {m.get("customType") for m in asks} == {"slm-skills", "slm-tools"},
          f"got {[m.get('customType') for m in asks]}")
    for m in asks:
        check(f"S8: ask {m.get('customType')} is display:true with plain text",
              m.get("display") is True
              and m.get("content") in (ASK_SKILLS, ASK_TOOLS),
              f"display={m.get('display')!r} content={m.get('content')!r}")
    # events must arrive in dialogue order: ask -> (assistant is not a
    # live event) -> ask -> real user
    live_order = [
        m.get("customType") if m.get("role") == "custom" else m.get("role")
        for m in (e["message"] for e in msg_starts)
    ]
    first_live = live_order[:3]
    check("S8: live event order starts ask-skills, ask-tools, user",
          first_live == ["slm-skills", "slm-tools", "user"],
          f"got {first_live}")
    entries, _ = read_session(session_dir, cwd)
    dlg = check_dialogue(entries, "S8")
    if dlg:
        s8_skills = check_assistant_entry(
            dlg["a1"], "S8", "skills assistant", expect_thinking=True,
            thinking_prefix="scanned loaded skills -")
        s8_tools = check_assistant_entry(
            dlg["a2"], "S8", "tools assistant", expect_thinking=True,
            thinking_prefix="scanned active tools -")
        check_wire(read_payloads(payload_log), "S8", PROMPT, s8_skills,
                   s8_tools,
                   exp_skills_thinking=exp_skills_thinking(len(expected_skills)),
                   exp_tools_thinking=exp_tools_thinking(4))

    # ------------------------------------------------------------------ S9
    print("S9: first user message verbatim; 'tools' field comes from pi core")
    s9_prompt = "list tools"
    root, home, session_dir, cwd, _ = new_workspace("s9")
    WORKSPACES.append(root)
    log_ext = root / "payloads_ext.jsonl"
    proc = run_pi(home, session_dir, cwd, [], s9_prompt, payload_log=log_ext)
    check("S9: pi (with extension) exited 0", proc.returncode == 0,
          proc.stdout[-400:] + proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    dlg = check_dialogue(entries, "S9")
    p_ext = read_payloads(log_ext)
    msgs = next((p.get("messages") for p in p_ext
                 if isinstance(p, dict) and isinstance(p.get("messages"), list)),
                None) if p_ext else None
    if msgs:
        check("S9: first user message sent verbatim (extension run)",
              msgs[-1].get("role") == "user"
              and wire_text(msgs[-1].get("content")) == s9_prompt,
              f"last msg: {json.dumps(msgs[-1])[:200]}")
    # baseline: same run WITHOUT the extension
    root2, home2, session_dir2, cwd2, _ = new_workspace("s9b")
    WORKSPACES.append(root2)
    log_base = root2 / "payloads_base.jsonl"
    proc = run_pi(home2, session_dir2, cwd2, [], s9_prompt,
                  payload_log=log_base, with_ext=False)
    check("S9: pi (without extension) exited 0", proc.returncode == 0,
          proc.stderr[-400:])
    p_base = read_payloads(log_base)
    msgs_base = next((p.get("messages") for p in p_base
                      if isinstance(p, dict)
                      and isinstance(p.get("messages"), list)), None) \
        if p_base else None
    if msgs_base:
        check("S9: first user message sent verbatim (baseline run)",
              msgs_base[-1].get("role") == "user"
              and wire_text(msgs_base[-1].get("content")) == s9_prompt,
              f"last msg: {json.dumps(msgs_base[-1])[:200]}")
    tools_ext = next((p.get("tools") for p in p_ext
                      if isinstance(p, dict) and p.get("tools")), None)
    tools_base = next((p.get("tools") for p in p_base
                       if isinstance(p, dict) and p.get("tools")), None)
    check("S9: top-level 'tools' field present (pi core function defs)",
          isinstance(tools_ext, list) and len(tools_ext) == 4,
          f"got {tools_ext!r}" if tools_ext is not None else "missing")
    if isinstance(tools_ext, list):
        check("S9: tools are the 4 active built-in tools",
              sorted(t.get("function", {}).get("name") for t in tools_ext)
              == sorted(["read", "bash", "edit", "write"]),
              f"got {[t.get('function', {}).get('name') for t in tools_ext]}")
        check("S9: each tool def has name + JSON-schema parameters",
              all(isinstance(t.get("function", {}).get("name"), str)
                  and isinstance(t.get("function", {}).get("parameters"), dict)
                  for t in tools_ext), repr(tools_ext)[:300])
    check("S9: 'tools' field identical with and without the extension "
          "(extension never touches it)",
          tools_ext == tools_base
          and tools_ext is not None,
          f"ext: {[t.get('function', {}).get('name') for t in (tools_ext or [])]}, "
          f"base: {[t.get('function', {}).get('name') for t in (tools_base or [])]}")

    # ------------------------------------------------------------------ done
    print(f"\n{passes} checks passed, {len(failures)} failed")
    if failures:
        print("Failed checks:")
        for f in failures:
            print(f"  - {f}")
        for w in WORKSPACES:
            shutil.rmtree(w, ignore_errors=True)
        return 1
    for w in WORKSPACES:
        shutil.rmtree(w, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
