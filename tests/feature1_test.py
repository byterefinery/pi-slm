#!/usr/bin/env python3
"""
Feature 1 tests for the slm extension (src/slm.ts).

Runs `pi` in random temp dirs under /tmp with an isolated agent dir
(PI_CODING_AGENT_DIR) and session dir (PI_CODING_AGENT_SESSION_DIR), model
LiquidAI/LFM2.5-2.6B (reasoning: true), and validates the session JSONL:

  S1  skills listed correctly          (default run; skills+tools both present)
  S2  tools listed correctly           (default 4-tool set, exact descriptions)
  S3  restricted tools                 (-t read,grep -> listing follows active set)
  S4  no skills                        (--no-skills -> skills: [])
  S5  no tools                         (-nt -> tools: [])
  S6  mixed order, repeated prompts    (continue session; single injection only)
  S7  non-reasoning model              (no '# thinking:' lines)

Ordering invariant in every scenario: the synthetic slm-skills message comes
first, then the slm-tools message, then the user message — and only
non-message startup entries (model_change, thinking_level_change) may
precede them.

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
REAL_AGENT = Path.home() / ".pi" / "agent"
MODEL = "LiquidAI/LFM2.5-2.6B"

SKILL_ALPHA_DESC = (
    'Alpha test skill: handles "quotes", colons: and #hashes '
    "for YAML escaping checks."
)
SKILL_BETA_DESC = "Beta test skill without references or scripts."

# pi 0.84.2 system-prompt snippets for the default built-in tools.
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
           continue_session: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # fully isolated environment: temp HOME (no global skills/settings/
    # ~/.agents/skills), sessions in the temp session dir
    env["HOME"] = str(home)
    env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
    env.pop("PI_CODING_AGENT_DIR", None)
    args = [
        "pi", "--offline", "-a",
        "-e", str(EXT),
        "--model", MODEL,
    ]
    if continue_session:
        args.append("-c")
    args += extra_args + ["-p", prompt]
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


def synth_entries(entries):
    """Return (skills_entry, tools_entry, user_msg_idx)."""
    skills = [e for e in entries
              if e.get("type") == "custom_message" and e.get("customType") == "slm-skills"]
    tools = [e for e in entries
             if e.get("type") == "custom_message" and e.get("customType") == "slm-tools"]
    users = [i for i, e in enumerate(entries)
             if e.get("type") == "message" and e.get("message", {}).get("role") == "user"]
    return skills, tools, users


def check_ordering(entries, scenario: str, expect_user_count: int = 1) -> None:
    skills, tools, users = synth_entries(entries)
    check(f"{scenario}: exactly one slm-skills message", len(skills) == 1,
          f"found {len(skills)}")
    check(f"{scenario}: exactly one slm-tools message", len(tools) == 1,
          f"found {len(tools)}")
    check(f"{scenario}: {expect_user_count} user message(s)", len(users) == expect_user_count,
          f"found {len(users)}")
    if len(skills) == 1 and len(tools) == 1 and users:
        i_s, i_t, i_u = entries.index(skills[0]), entries.index(tools[0]), users[0]
        check(f"{scenario}: order skills -> tools -> user",
              i_s < i_t < i_u,
              f"skills@{i_s} tools@{i_t} user@{i_u}")
        # only non-message startup entries may precede the skills message
        preceding = entries[:i_s]
        bad = [e for e in preceding if e.get("type") in ("message", "custom_message")]
        check(f"{scenario}: no conversation entries before synthetic messages",
              not bad, f"preceding: {[e.get('type') for e in preceding]}")
        check(f"{scenario}: startup entries are bookkeeping only",
              all(e.get("type") in ("model_change", "thinking_level_change")
                  for e in preceding),
              f"preceding: {[e.get('type') for e in preceding]}")
    return skills, tools, users


def check_skills_content(content: str, expected: dict, scenario: str,
                         expect_thinking: bool) -> None:
    check(f"{scenario}: skills message starts with 'Available skills:'",
          content.startswith("Available skills:\n"))
    # token saving: skill names are plain (unquoted) scalars
    for name in expected:
        check(f"{scenario}: skill name '{name}' is a plain (unquoted) scalar",
              re.search(rf"^\s*- name: {re.escape(name)}\n", content,
                        flags=re.M) is not None,
              repr(content))
    # the alpha description contains ': ' -> must fall back to double quotes
    check("note: description containing ': ' is quoted (YAML-safe fallback)",
          'description: "Alpha test skill: handles' in content)

    if expect_thinking:
        m = re.search(r"^# thinking: .+$", content, flags=re.M)
        check(f"{scenario}: skills message has short synthetic reasoning",
              bool(m) and len(m.group(0)) < 300,
              repr(content.splitlines()[1:3]))
    else:
        check(f"{scenario}: skills message has no synthetic reasoning",
              not re.search(r"^# thinking:", content, flags=re.M))

    try:
        doc = yaml.safe_load(content)
        check(f"{scenario}: skills message is valid YAML", True)
    except yaml.YAMLError as e:
        check(f"{scenario}: skills message is valid YAML", False, str(e))
        return
    check(f"{scenario}: skills YAML top level is a mapping",
          isinstance(doc, dict), repr(doc)[:120])
    skills = doc.get("skills")
    if not isinstance(skills, list):
        check(f"{scenario}: skills YAML has 'skills' list", False, repr(doc)[:200])
        return
    check(f"{scenario}: skills YAML has 'skills' list", True)

    names = [s.get("name") for s in skills]
    check(f"{scenario}: listed skill names",
          sorted(names) == sorted(expected),
          f"got {names}")
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
    check(f"{scenario}: hidden skill not listed",
          "slm-hidden" not in names, f"got {names}")


def check_tools_content(content: str, expected: list[str], scenario: str,
                        expect_thinking: bool) -> None:
    check(f"{scenario}: tools message starts with 'Available tools:'",
          content.startswith("Available tools:\n"))
    if expect_thinking:
        m = re.search(r"^# thinking: .+$", content, flags=re.M)
        check(f"{scenario}: tools message has short synthetic reasoning",
              bool(m) and len(m.group(0)) < 300,
              repr(content.splitlines()[1:3]))
    else:
        check(f"{scenario}: tools message has no synthetic reasoning",
              not re.search(r"^# thinking:", content, flags=re.M))

    try:
        doc = yaml.safe_load(content)
        check(f"{scenario}: tools message is valid YAML", True)
    except yaml.YAMLError as e:
        check(f"{scenario}: tools message is valid YAML", False, str(e))
        return
    check(f"{scenario}: tools YAML top level is a mapping",
          isinstance(doc, dict), repr(doc)[:120])
    tools = doc.get("tools")
    if not isinstance(tools, list):
        check(f"{scenario}: tools YAML has 'tools' list", False, repr(doc)[:200])
        return
    names = [t.get("name") for t in tools]
    check(f"{scenario}: listed tool names (order preserved)",
          names == expected, f"got {names}, expected {expected}")
    for t in tools:
        desc = t.get("description")
        check(f"{scenario}: tool '{t.get('name')}' has single-line description",
              isinstance(desc, str) and desc != "" and "\n" not in desc,
              repr(desc))
        if t.get("name") in TOOL_SNIPPETS:
            check(f"{scenario}: tool '{t.get('name')}' description matches pi snippet",
                  desc == TOOL_SNIPPETS[t["name"]],
                  f"got {desc!r}, expected {TOOL_SNIPPETS[t['name']]!r}")
            # token saving: snippet descriptions are plain (unquoted) scalars
            check(f"{scenario}: tool '{t.get('name')}' description is a plain scalar",
                  re.search(rf"^\s+description: {re.escape(TOOL_SNIPPETS[t['name']])}\n",
                            content, flags=re.M) is not None,
                  repr(content))
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


def new_workspace(tag: str, reasoning: bool = True):
    """Random temp cwd + isolated HOME/session dirs."""
    root = Path(tempfile.mkdtemp(prefix=f"slm-f1-{tag}-", dir="/tmp"))
    home = setup_home(root, reasoning=reasoning)
    session_dir = root / "sessions"
    cwd = root / "work"
    cwd.mkdir()
    expected = write_fixture_skills(cwd)
    return root, home, session_dir, cwd, expected


def main() -> int:
    prompt = "Reply with the single word: done"
    workspaces: list[Path] = []

    # ------------------------------------------------------------------ S1+S2
    print("S1: skills listed correctly (default: skills + tools)")
    root, home, session_dir, cwd, expected = new_workspace("s1")
    workspaces.append(root)
    proc = run_pi(home, session_dir, cwd, [], prompt)
    check("S1: pi exited 0", proc.returncode == 0,
          proc.stdout[-400:] + proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    skills, tools, _ = check_ordering(entries, "S1")
    check_skills_content(skills[0]["content"], expected, "S1", expect_thinking=True)

    print("S2: tools listed correctly (default 4-tool set)")
    check_tools_content(tools[0]["content"],
                        ["read", "bash", "edit", "write"], "S2",
                        expect_thinking=True)

    # ------------------------------------------------------------------ S3
    print("S3: restricted tools (-t read,grep)")
    root, home, session_dir, cwd, expected = new_workspace("s3")
    workspaces.append(root)
    proc = run_pi(home, session_dir, cwd, ["-t", "read,grep"], prompt)
    check("S3: pi exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    skills, tools, _ = check_ordering(entries, "S3")
    check_tools_content(tools[0]["content"], ["read", "grep"], "S3",
                        expect_thinking=True)
    doc = yaml.safe_load(skills[0]["content"])
    check("S3: skills still listed with tools restricted",
          sorted(s.get("name") for s in doc["skills"]) == sorted(expected),
          f"got {[s.get('name') for s in doc['skills']]}")

    # ------------------------------------------------------------------ S4
    print("S4: no skills (--no-skills)")
    root, home, session_dir, cwd, _ = new_workspace("s4")
    workspaces.append(root)
    proc = run_pi(home, session_dir, cwd, ["--no-skills"], prompt)
    check("S4: pi exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    skills, tools, _ = check_ordering(entries, "S4")
    doc = yaml.safe_load(skills[0]["content"])
    check("S4: skills message lists empty list", doc["skills"] == [],
          repr(doc))
    check("S4: skills message reasoning says 0 found",
          re.search(r"# thinking: scanned loaded skills - 0 found\. .+",
                    skills[0]["content"]) is not None,
          repr(skills[0]["content"]))
    check_tools_content(tools[0]["content"],
                        ["read", "bash", "edit", "write"], "S4",
                        expect_thinking=True)

    # ------------------------------------------------------------------ S5
    print("S5: no tools (-nt)")
    root, home, session_dir, cwd, expected = new_workspace("s5")
    workspaces.append(root)
    proc = run_pi(home, session_dir, cwd, ["-nt"], prompt)
    check("S5: pi exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    skills, tools, _ = check_ordering(entries, "S5")
    doc = yaml.safe_load(tools[0]["content"])
    check("S5: tools message lists empty list", doc["tools"] == [], repr(doc))
    check_skills_content(skills[0]["content"], expected, "S5",
                         expect_thinking=True)

    # ------------------------------------------------------------------ S6
    print("S6: mixed order, repeated prompts (session continued)")
    root, home, session_dir, cwd, expected = new_workspace("s6")
    workspaces.append(root)
    proc = run_pi(home, session_dir, cwd, [], prompt)
    check("S6: first run exited 0", proc.returncode == 0, proc.stderr[-400:])
    proc = run_pi(home, session_dir, cwd, [],
                  "Reply with the single word: again",
                  continue_session=True)
    check("S6: continued run exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    skills, tools, users = check_ordering(entries, "S6", expect_user_count=2)
    if len(skills) == 1 and len(tools) == 1 and len(users) == 2:
        # the single synthetic pair must sit before the first user message;
        # nothing synthetic may appear around the second prompt
        i_s, i_t = entries.index(skills[0]), entries.index(tools[0])
        after_first_prompt = [e for e in entries[users[0]:]
                              if e.get("type") == "custom_message"]
        check("S6: no re-injection after the first prompt",
              not any(e.get("customType") in ("slm-skills", "slm-tools")
                      for e in after_first_prompt),
              f"got {[e.get('customType') for e in after_first_prompt]}")
        check("S6: synthetic pair sits before both user messages",
              i_s < i_t < users[0] < users[1],
              f"skills@{i_s} tools@{i_t} users@{users}")
    check_skills_content(skills[0]["content"], expected, "S6",
                         expect_thinking=True)
    check_tools_content(tools[0]["content"],
                        ["read", "bash", "edit", "write"], "S6",
                        expect_thinking=True)

    # ------------------------------------------------------------------ S7
    print("S7: non-reasoning model (no synthetic reasoning lines)")
    root, home, session_dir, cwd, expected = new_workspace("s7",
                                                           reasoning=False)
    workspaces.append(root)
    proc = run_pi(home, session_dir, cwd, [], prompt)
    check("S7: pi exited 0", proc.returncode == 0, proc.stderr[-400:])
    entries, _ = read_session(session_dir, cwd)
    skills, tools, _ = check_ordering(entries, "S7")
    check_skills_content(skills[0]["content"], expected, "S7",
                         expect_thinking=False)
    check_tools_content(tools[0]["content"],
                        ["read", "bash", "edit", "write"], "S7",
                        expect_thinking=False)

    # ------------------------------------------------------------------ done
    print(f"\n{passes} checks passed, {len(failures)} failed")
    if failures:
        print("Failed checks:")
        for f in failures:
            print(f"  - {f}")
        for w in workspaces:
            shutil.rmtree(w, ignore_errors=True)
        return 1
    for w in workspaces:
        shutil.rmtree(w, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
