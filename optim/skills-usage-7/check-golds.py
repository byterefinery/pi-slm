#!/usr/bin/env python3
"""Verify expected-responses golds against fresh teacher runs.

For every single-turn gold case: run the teacher LM (Qwen3.8-27B, thinking off)
in a real tool loop (bash/read executed locally, like the student's pi session)
with the case prompt (live skill block + user text), then score the teacher's
transcript with the SAME score_case() used for students. If the teacher — the
reference for correct behavior under the current skill docs — scores < 1.0, the
gold (or the matcher) is stale and must be updated.
"""
import importlib.util
import json
import subprocess
import sys

import httpx

HERE = __import__("pathlib").Path(__file__).parent


def main() -> None:
    spec = importlib.util.spec_from_file_location("train", str(HERE / "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    api_base, api_key = train._endpoints()
    s = train._sampling(train.TEACHER_MODEL)
    sampling = {
        k: s[k] for k in ("top_k", "min_p", "presence_penalty", "repeat_penalty", "top_p") if k in s
    }
    sampling["temperature"] = s.get("temperature", 1.0)
    sampling["max_tokens"] = 16384

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string", "description": "Bash command to execute"},
                        "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the contents of a file. Supports text files and images.",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write content to a file (not available in this check; will be refused).",
                "parameters": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    ]

    SYSTEM = (
        "You are an expert coding assistant. Use the available tools when needed. "
        "Follow the skill instructions in the user message exactly."
    )

    def parse_args(tc: dict) -> dict:
        raw = tc["function"].get("arguments") or {}
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def run_tool(tc: dict) -> str:
        name = tc["function"]["name"]
        args = parse_args(tc)
        if name == "bash":
            cmd = str(args.get("command", ""))
            to = float(args.get("timeout") or 60)
            try:
                p = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=to, cwd=str(HERE)
                )
                out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
                return (out or "(no output)")[:20000]
            except subprocess.TimeoutExpired:
                return f"error: command timed out after {to:.0f}s"
        if name == "read":
            path = args.get("path", "")
            try:
                return __import__("pathlib").Path(path).read_text()[:20000]
            except Exception as e:  # noqa: BLE001
                return f"error: {e}"
        if name == "write":
            return "error: write tool is not available in this environment"
        return f"error: unknown tool {name}"

    def teacher_case(prompt: str, label: str, max_turns: int = 6) -> tuple[list[dict], str | None]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        students: list[dict] = []
        final: str | None = None
        for _ in range(max_turns):
            resp = httpx.post(
                api_base.rstrip("/") + "/chat/completions",
                # the llamacpp router requires x-session-affinity or it
                # returns an empty 200 body
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "x-session-affinity": f"goldcheck-{label}",
                },
                json={
                    "model": train.TEACHER_MODEL,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    **sampling,
                },
                timeout=300,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = msg.get("content") or ""
                break
            students.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": [
                        {
                            "name": tc["function"]["name"],
                            "args": parse_args(tc),
                        }
                        for tc in tcs
                    ],
                }
            )
            messages.append(msg)
            for tc in tcs:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": run_tool(tc),
                    }
                )
        return students, final

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    cases = train.load_cases()
    print(f"{'case':24s} {'score':>6s}  gold-tools -> teacher-tools / finals")
    print("-" * 100)
    for case in cases:
        if only and case["case"] not in only:
            continue
        try:
            students, final = teacher_case(case["prompt"], case["case"])
        except Exception as e:  # noqa: BLE001
            print(f"{case['case']:24s} ERROR {e}")
            continue
        score, detail = train.score_case(case, students, final)
        def gold_args(tc: dict) -> dict:
            raw = tc["function"].get("arguments") or {}
            return raw if isinstance(raw, dict) else json.loads(raw or "{}")

        gold_steps = [
            {
                "name": tc["function"]["name"],
                "args": gold_args(tc),
            }
            for step in case["expected"]
            for tc in step.get("tool_calls") or []
        ]
        gold_final = ""
        for step in reversed(case["expected"]):
            if step.get("content") and not step.get("tool_calls"):
                gold_final = step["content"]
                break
        print(f"{case['case']:24s} {score:6.2f}  tool={detail['tool'][:60]}")
        for gt in gold_steps:
            print(f"    gold : {gt['name']} {json.dumps(gt['args'])[:80]}")
        for tt in [tc for st in students for tc in st["tool_calls"]]:
            print(f"    teach: {tt['name']} {json.dumps(tt['args'])[:80]}")
        print(f"    gold final : {gold_final[:100]!r}")
        print(f"    teach final: {(final or '')[:100]!r}")
        print(f"    final check: {detail['final'][:80]}")


if __name__ == "__main__":
    main()
