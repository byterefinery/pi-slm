#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
"""Replay the exact tzip wire context (payloads.jsonl of a tzip run) for all
4 levels and score: activation reply must state the mode name, no tool calls."""
import json, os, re, sys, requests, uuid

MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")
raw = open(MODELS_JSON).read()
cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
prov = next(p for p in cfg["providers"].values()
            if any(m["id"] == "LiquidAI/LFM2.5-2.6B" for m in p["models"]))

ROOT = sys.argv[1]            # tzip run root (has payloads.jsonl)
CAND = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3

payload = json.loads(open(ROOT + "/payloads.jsonl").readline())
cand = json.load(open(CAND))
msgs = [dict(x) for x in payload["messages"]]
msgs[6] = {"role": "assistant", "content": cand["content"],
           "reasoning_content": cand["reasoning_content"]}
body_base = {
    "model": payload["model"],
    "tools": payload.get("tools"),
    "temperature": payload.get("temperature", 0.1),
    "max_tokens": payload.get("max_tokens", 8192),
    "top_k": payload.get("top_k"),
    "repeat_penalty": payload.get("repeat_penalty"),
    "reasoning_effort": payload.get("reasoning_effort"),
    "stream": False,
}

# build the real tzip invocation message (pi expansion)
tzip_dir = "/tmp/slm-f7w-tzip-50iArt/work/.agents/skills/tzip"
SKILL_MD = None
import glob
m = glob.glob("/tmp/slm-f7-logs/tzip-lite.root")
if m:
    r7 = open(m[0]).read().strip()
    SKILL_MD = r7 + "/work/.agents/skills/tzip/SKILL.md"
raw_md = open(SKILL_MD).read()
body_md = re.sub(r"^---\n.*?\n---\n", "", raw_md, count=1, flags=re.S).strip()
tzip_dir = os.path.dirname(SKILL_MD)

def invocation(level):
    return (f'<skill name="tzip" location="{SKILL_MD}">\n'
            f"References are relative to {tzip_dir}.\n"
            f"\n{body_md}\n"
            f"</skill>\n"
            f"\n{level}")

EXPECT = {
    "lite": ["tzip lite activated"],
    "full": ["tzip full activated"],
    "ultra": ["tzip ultra activated"],
    "off": ["tzip deactivated"],
}

for level in ("lite", "full", "ultra", "off"):
    ok_act = ok_tool = 0
    for i in range(N):
        body = dict(body_base)
        body["messages"] = msgs[:6] + [msgs[6], {"role": "user",
                                                 "content": invocation(level)}]
        r = requests.post(prov["baseUrl"] + "/chat/completions",
                          headers={"Authorization": f"Bearer {prov['apiKey']}",
                                   "Content-Type": "application/json",
                                   "x-session-affinity": "tzip-" + uuid.uuid4().hex[:8]},
                          json=body, timeout=300)
        m = r.json()["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        content = (m.get("content") or "").strip()
        no_tools = len(tcs) == 0
        act = any(e in content for e in EXPECT[level])
        ok_act += act
        ok_tool += no_tools
        first = content.split("\n")[0][:100] if content else "(empty)"
        print(f"  [{level}] {i + 1}/{N} "
              f"{'ACT-OK' if act else 'ACT-FAIL'} "
              f"{'no-tools' if no_tools else f'tools={len(tcs)}'}  {first!r}")
    print(f"== {level}: activation {ok_act}/{N}, no-tools {ok_tool}/{N}")
