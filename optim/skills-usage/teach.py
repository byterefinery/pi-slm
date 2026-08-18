#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
#
# Synthesize/verify the teaching assistant message (text content + reasoning
# content) that, placed in the seed session right after "How can a skill be
# used?", makes LiquidAI/LFM2.5-2.6B emit the PERFECT bash tool call when the
# webfetch skill is invoked (no GEPA instructions, only the real pi system
# prompt + conversation).
#
# The winning message is saved as assistant_message.json:
#   {"role": "assistant", "content": "...", "reasoning_content": "..."}
#
# Usage:
#   uv run --script teach.py --original        # baseline: seed's existing answer
#   uv run --script teach.py                   # test candidate.json (n=5 fresh calls)
#   uv run --script teach.py --n 5 --save      # verify + save assistant_message.json
#   uv run --script teach.py --no-reasoning    # omit reasoning_content in the request

import argparse
import json
import os
import re

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "seed.json")
CANDIDATE = os.path.join(HERE, "candidate.json")
ASSISTANT_MSG = os.path.join(HERE, "assistant_message.json")
MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")
WEBFETCH_SCRIPT = (
    "/home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/scripts/webfetch.py"
)

with open(SEED) as f:
    raw = f.read()
try:
    seed = json.loads(raw)
except json.JSONDecodeError:
    seed = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))

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

# messages[0:6] = system + skills Q&A + tools Q&A + "How can a skill be used?"
PREAMBLE = seed["messages"][:6]
SKILL_INVOCATION = seed["messages"][7:8]  # <skill webfetch> + https://tangledgroup.com/


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_messages(candidate, include_reasoning=True):
    """Seed session with the candidate assistant answer inserted at [6]."""
    msg = {"role": "assistant", "content": candidate["content"]}
    if include_reasoning and candidate.get("reasoning_content"):
        msg["reasoning_content"] = candidate["reasoning_content"]
    return PREAMBLE + [msg] + SKILL_INVOCATION


def score_response(resp_msg):
    """1.0 iff resp_msg is the perfect bash tool call for the seed invocation."""
    tcs = resp_msg.tool_calls or []
    if len(tcs) != 1:
        return 0.0, f"expected exactly 1 tool call, got {len(tcs)}"
    tc = tcs[0]
    if tc.function.name != "bash":
        return 0.0, f"tool {tc.function.name!r}, wanted 'bash'"
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        return 0.0, f"arguments are not JSON: {tc.function.arguments!r}"
    cmd = (args.get("command") or "").strip()
    required = [f"uv run --script {WEBFETCH_SCRIPT}", "https://tangledgroup.com"]
    missing = [r for r in required if r not in cmd]
    if missing:
        return 0.0, f"command missing {missing}: {cmd!r}"
    return 1.0, cmd


def run(candidate, n, include_reasoning=True, verbose=True):
    messages = build_messages(candidate, include_reasoning)
    ok = 0
    for i in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=seed.get("temperature", 0.1),
            max_tokens=8192,
            extra_headers={"x-session-affinity": "teach-test"},
            extra_body={
                "top_k": seed.get("top_k", 50),
                "repeat_penalty": seed.get("repeat_penalty", 1.1),
                "reasoning_effort": seed.get("reasoning_effort", "high"),
            },
        )
        resp_msg = resp.choices[0].message
        s, info = score_response(resp_msg)
        ok += s >= 1.0
        if verbose:
            thinking = (resp_msg.reasoning_content or "")[:120].replace("\n", " ")
            print(f"  [{i + 1}/{n}] {'PASS' if s == 1.0 else 'FAIL'}  {info}")
            if thinking:
                print(f"         thinking: {thinking}...")
            if resp_msg.content:
                print(f"         content: {resp_msg.content[:120].strip()!r}")
    print(f"  => {ok}/{n} perfect tool calls")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=CANDIDATE, help="candidate message JSON file")
    ap.add_argument("--original", action="store_true",
                    help="test the seed's own 'How can a skill be used?' answer instead")
    ap.add_argument("--n", type=int, default=5, help="fresh calls per test")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="omit reasoning_content from the request")
    ap.add_argument("--save", action="store_true",
                    help="if all pass, write the candidate to assistant_message.json")
    args = ap.parse_args()

    if args.original:
        candidate = {
            "role": "assistant",
            "content": seed["messages"][6]["content"],
            "reasoning_content": seed["messages"][6].get("reasoning_content", ""),
        }
        label = "original seed answer"
    else:
        candidate = load_json(args.candidate)
        label = os.path.basename(args.candidate)

    print(f"candidate: {label}  (n={args.n}, reasoning in request: "
          f"{not args.no_reasoning})")
    ok = run(candidate, args.n, include_reasoning=not args.no_reasoning)

    if args.save:
        if ok == args.n:
            out = {
                "role": "assistant",
                "content": candidate["content"],
                "reasoning_content": candidate.get("reasoning_content", ""),
            }
            with open(ASSISTANT_MSG, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\nsaved {ASSISTANT_MSG}")
        else:
            print(f"\nnot saved: only {ok}/{args.n} passed")


if __name__ == "__main__":
    main()
