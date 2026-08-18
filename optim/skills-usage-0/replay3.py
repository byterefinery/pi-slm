#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
#
# Replay the seed session with the current teaching message (from src/slm.ts
# via --candidate, default candidate.json) for the three real-skill
# invocations and score the bash tool call each produces:
#
#   E1  webfetch verb:  <skill webfetch> + "fetch https://tangledgroup.com/ and summarize it"
#   E2  websearch look: <skill websearch> + "look for tangled group repos"
#   E3  webfetch bare:  <skill webfetch> + "https://tangledgroup.com/"
#
# Usage:
#   uv run --script replay3.py --candidate ../src/slm-extract.json
#   uv run --script replay3.py                     # default: candidate.json
#   uv run --script replay3.py --n 5 --only E2

import argparse
import json
import os
import re
import sys

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "seed.json")
REPO = "/home/mtasic/projects-b/pi-slm"
SKILLS = f"{REPO}/.agents/skills"
MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")

with open(SEED) as f:
    seed = json.load(f)

model = seed["model"]
tools = seed["tools"]


def endpoint_for(model_id):
    with open(MODELS_JSON) as f:
        raw = f.read()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
    for prov in cfg["providers"].values():
        if any(m["id"] == model_id for m in prov["models"]):
            return prov["baseUrl"], prov["apiKey"]
    raise SystemExit(f"no provider in {MODELS_JSON} hosts {model_id!r}")


API_BASE, API_KEY = endpoint_for(model)
client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=300)

PREAMBLE = seed["messages"][:6]  # system + skills Q&A + tools Q&A + skill-usage ask


def m7_text(m):
    c = m["content"]
    return c if isinstance(c, str) else "".join(
        b.get("text", "") for b in c if isinstance(b, dict))


def make_skill_block(skill: str, task: str) -> str:
    """Replicate pi's /skill: expansion for <skill> (repo skills)."""
    path = f"{SKILLS}/{skill}/SKILL.md"
    raw = open(path).read()
    body = re.sub(r"^---\n.*?\n---\n", "", raw, count=1, flags=re.S).strip()
    return (
        f'<skill name="{skill}" location="{path}">\n'
        f"References are relative to {SKILLS}/{skill}.\n"
        f"\n{body}\n"
        f"</skill>\n"
        f"\n{task}"
    )


def user_msg(text: str):
    return {"role": "user", "content": text}


E3_TEXT = m7_text(seed["messages"][7])  # original bare-URL invocation
EXAMPLES = {
    "E1": {
        "task": "fetch https://tangledgroup.com/ and summarize it",
        "msg": user_msg(E3_TEXT.split("</skill>\n\n")[0] + "</skill>\n\n"
                       "fetch https://tangledgroup.com/ and summarize it"),
        "required": [f"uv run --script {SKILLS}/webfetch/scripts/webfetch.py",
                     "https://tangledgroup.com"],
        "invented": ["--html", "--file", "--impersonate", "--no-ai-targeted",
                     "--tool"],
    },
    "E2": {
        "task": "look for tangled group repos",
        "msg": user_msg(make_skill_block("websearch", "look for tangled group repos")),
        "required": [f"uv run --script {SKILLS}/websearch/scripts/websearch.py",
                     '"tangled group repos"'],
        "invented": [],
    },
    "E3": {
        "task": "https://tangledgroup.com/",
        "msg": user_msg(E3_TEXT),
        "required": [f"uv run --script {SKILLS}/webfetch/scripts/webfetch.py",
                     "https://tangledgroup.com"],
        "invented": ["--html", "--file", "--impersonate", "--no-ai-targeted",
                     "--tool"],
    },
}


def score(cmd: str, spec) -> tuple[float, str]:
    missing = []
    for r in spec["required"]:
        if r in cmd:
            continue
        # accept single or double quoting for arguments with spaces
        if r.startswith('"'):
            inner = r[1:-1]
            if f'"{inner}"' in cmd or f"'{inner}'" in cmd:
                continue
        missing.append(r)
    if missing:
        return 0.0, f"missing {missing} in {cmd!r}"
    bad = [f for f in spec["invented"] if f in cmd]
    if bad:
        return 0.0, f"invented flags {bad} in {cmd!r}"
    return 1.0, cmd


def load_candidate(path):
    with open(path) as f:
        c = json.load(f)
    return c["content"], c.get("reasoning_content", "")


def run(tag: str, example: dict, content: str, reasoning: str, n: int) -> int:
    messages = PREAMBLE + [
        {"role": "assistant",
         **({"content": content, "reasoning_content": reasoning} if reasoning
            else {"content": content})},
        example["msg"],
    ]
    ok = 0
    for i in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=seed.get("temperature", 0.1),
            max_tokens=8192,
            extra_headers={"x-session-affinity": "replay3"},
            extra_body={
                "top_k": seed.get("top_k", 50),
                "repeat_penalty": seed.get("repeat_penalty", 1.1),
                "reasoning_effort": seed.get("reasoning_effort", "high"),
            },
        )
        m = resp.choices[0].message
        s, info = 0.0, "no tool call"
        tcs = m.tool_calls or []
        if len(tcs) == 1 and tcs[0].function.name == "bash":
            try:
                args = json.loads(tcs[0].function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            s, info = score((args.get("command") or "").strip(), example)
        elif len(tcs) > 1:
            info = f"{len(tcs)} tool calls"
        ok += s >= 1.0
        mark = "PASS" if s == 1.0 else "FAIL"
        extra = ""
        if m.content and s < 1.0:
            extra = f"  content: {m.content[:90].strip()!r}"
        print(f"  [{i + 1}/{n}] {mark}  {info}{extra}")
    print(f"  => {ok}/{n}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=os.path.join(HERE, "candidate.json"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    content, reasoning = load_candidate(args.candidate)
    print(f"candidate: {os.path.basename(args.candidate)} "
          f"(reasoning={'on' if reasoning else 'off'}, n={args.n})")
    totals = 0
    count = 0
    for tag, spec in EXAMPLES.items():
        if args.only and tag not in args.only:
            continue
        print(f"{tag}: {spec['task']!r}")
        totals += run(tag, spec, content, reasoning, args.n)
        count += args.n
    print(f"TOTAL: {totals}/{count}")
    sys.exit(0 if totals == count else 1)


if __name__ == "__main__":
    main()
