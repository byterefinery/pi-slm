#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
#
# Replay the EXACT wire payload of a finished pi run (payloads.jsonl, first
# payload), swapping the skill-usage assistant message (index 6) for a
# candidate, and score the model's next action. Tests the candidate in the
# precise context where a real run failed/succeeded.
#
# Usage:
#   uv run --script replay_payload.py --payload <run>/payloads.jsonl \
#       --candidate candidate.json --n 5
#       [--expect-sub "uv run --script" --expect-sub "tangledgroup"]

import argparse
import json
import os
import re

from openai import OpenAI

MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--expect-sub", action="append", default=[])
    ap.add_argument("--no-tool-ok", action="store_true",
                    help="expect a text answer (no tool call); score on substrings")
    ap.add_argument("--stream", action="store_true",
                    help="send stream=true (match a live pi run)")
    args = ap.parse_args()

    cand = json.load(open(args.candidate))
    payload = json.loads(open(args.payload).readline())
    model = payload["model"]
    base_url, api_key = endpoint_for(model)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)

    msgs = payload["messages"]
    assert msgs[5]["role"] == "user" and "skill be used" in json.dumps(msgs[5]), \
        "payload msg[5] is not the skill-usage ask"
    assert msgs[6]["role"] == "assistant", "payload msg[6] is not assistant"

    def attempt():
        cand_msg = {"role": "assistant", "content": cand["content"]}
        if cand.get("reasoning_content"):
            cand_msg["reasoning_content"] = cand["reasoning_content"]
        r = client.chat.completions.create(
            model=model,
            messages=msgs[:6] + [cand_msg] + msgs[7:],
            tools=payload.get("tools"),
            temperature=payload.get("temperature", 0.1),
            top_p=payload.get("top_p"),
            max_tokens=payload.get("max_tokens", 8192),
            stream=args.stream,
            extra_headers={"x-session-affinity": "replay-payload"},
            extra_body={k: payload[k] for k in
                        ("top_k", "repeat_penalty", "reasoning_effort")
                        if k in payload},
        )
        if not args.stream:
            return r.choices[0].message
        # aggregate the stream into a message-like object
        content, reasoning, tcs = [], [], []
        for chunk in r:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d.content:
                content.append(d.content)
            if getattr(d, "reasoning_content", None):
                reasoning.append(d.reasoning_content)
            tc = d.tool_calls
            if tc:
                for t in tc:
                    while len(tcs) <= t.index:
                        tcs.append({"index": t.index, "id": "",
                                    "type": "function",
                                    "function": {"name": "",
                                                 "arguments": ""}})
                    slot = tcs[t.index]
                    if t.id:
                        slot["id"] = t.id
                    if t.function:
                        if t.function.name:
                            slot["function"]["name"] += t.function.name
                        if t.function.arguments:
                            slot["function"]["arguments"] += \
                                t.function.arguments
        class M:
            pass
        m = M()
        m.content = "".join(content) or None
        m.reasoning_content = "".join(reasoning) or None
        m.tool_calls = tcs or None
        return m

    def tc_field(tc, *path):
        cur = tc
        for p in path:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                cur = getattr(cur, p, None)
        return cur

    ok = 0
    for i in range(args.n):
        m = attempt()
        info = ""
        tcs = m.tool_calls or []
        if args.no_tool_ok:
            cmd = (m.content or "")
            ok_here = all(s in cmd for s in args.expect_sub)
            info = f"answer {len(cmd)} chars" + ("" if ok_here else f" missing {args.expect_sub}")
        elif len(tcs) == 1 and tc_field(tcs[0], "function", "name") == "bash":
            try:
                cmd = (json.loads(tc_field(tcs[0], "function", "arguments") or "{}")
                       .get("command") or "")
            except json.JSONDecodeError:
                cmd = ""
            ok_here = all(s in cmd for s in args.expect_sub)
            info = cmd
            if not ok_here:
                info = f"missing {args.expect_sub} in {cmd!r}"
        else:
            ok_here = False
            info = (f"{len(tcs)} tool calls; "
                    f"content: {(m.content or '')[:80]!r}")
        ok += ok_here
        print(f"  [{i + 1}/{args.n}] {'PASS' if ok_here else 'FAIL'}  {info}")
    print(f"  => {ok}/{args.n}")


if __name__ == "__main__":
    main()
