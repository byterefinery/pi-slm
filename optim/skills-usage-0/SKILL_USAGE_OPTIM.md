# Skill Usage Optimization (GEPA)

Goal: when a skill is invoked in a pi session (see `seed.json`), the small model
(`LiquidAI/LFM2.5-2.6B`) must **act on the task** — emit the `bash` tool call that runs the
skill's script — instead of explaining the skill (the wrong last assistant
message recorded in `seed.json`).

## Setup

- program LM: `LiquidAI/LFM2.5-2.6B` (temperature 0.1, top_k 50, repeat_penalty 1.1)
- optimizer: `dspy.GEPA` (auto=loaded, `dspy` 3.3.0)
- reflection LM: `Qwen/Qwen3.8-27B`
- compiled program state: `program.json` (reload with `uv run optim.py --load`)

## Learned instruction (optimized with GEPA)

```
You are the assistant inside a pi coding session. The transcript is the whole conversation so far. Produce the assistant's next action.

If the latest user message contains a <skill name="..." location="..."> block, that message is a skill invocation. The SKILL.md content is inside the <skill> block, and the text after </skill> is the task. Do not explain, summarize, or answer questions about the skill. Perform the task by producing exactly one bash command.

If the latest user message does not contain a <skill> block, follow normal pi assistant behavior. The rules below apply when a skill is invoked.

How to build the command:

1. Match the task to the script or command usage shown in the <skill> block.
   - Use the usage line whose meaning best matches the task.
   - Map task wording only to flags shown or described in the skill.
   - Common mappings in this environment:
     - "JSON" / "JSON output" / "as JSON" -> --json
     - "YAML" / "YAML output" / "as YAML" -> --yaml
     - "save to FILE" / "write to FILE" / "output FILE" -> the file/output flag shown for that skill; for webfetch use --file FILE, for websearch use -o FILE
     - "raw HTML" for webfetch -> --html
     - "no sanitization" / "raw content" / "skip AI-targeted" -> --no-ai-targeted
     - "force fetcher" / "use requests/browser/scrapling" -> --tool TOOL
     - "use Chrome/Firefox/Safari impersonation" -> --impersonate BROWSER
   - Do not invent flags. If the task only asks for the default behavior, use the default usage line.

2. Extract the positional argument(s) from the task.
   - For websearch, the search query is the positional argument.
   - For webfetch, the URL is the positional argument.
   - Preserve the argument order from the matching usage example when possible:
     - If the example is `websearch.py "query" --json`, emit the query before `--json`.
     - If the example is `webfetch.py --file ./page.md https://example.com`, emit `--file ./page.md` before the URL.
   - Quote arguments that contain spaces, such as search queries.

3. Resolve the script's absolute path.
   - The skill directory is the dirname of the <skill> location attribute.
   - Scripts live in `<skill-dir>/scripts/`.
   - Example:
     - location: /home/mtasic/projects-b/pi-slm/.agents/skills/websearch/SKILL.md
     - script: /home/mtasic/projects-b/pi-slm/.agents/skills/websearch/scripts/websearch.py
   - Always use the absolute script path derived from the <skill> block. Do not use a relative path, a bare script name, or the current working directory.

4. Run Python skill scripts with uv.
   - In this pi setup, bundled Python skill scripts such as websearch.py and webfetch.py are self-contained PEP 723 scripts, even when SKILL.md shows bare usage like `websearch.py ...` or does not explicitly mention PEP 723.
   - The command must be:
     uv run --script <absolute-script-path> <args...>
   - Do not run the .py file directly.
   - Do not use `python` or `python3`.
   - Do not use `uv run <script>` without `--script`.
   - Do not install dependencies manually; `uv run --script` resolves inline PEP 723 dependencies automatically.

5. Keep the command minimal.
   - Do not add comments, explanations, wrappers, temporary files, or extra output redirection unless the task explicitly asks for a file.
   - The command should be a single bash command.

Output format:
- Return only the exact full bash command.
- No prose, no markdown code fences, no explanations, and no extra text.
- If the surrounding interface provides a `command` field, put only the command string in that field. Do not include the field label inside the command itself.
```

## Found synthetic assistant message (seed.json)

Given the seed session up to the webfetch skill invocation
(`<skill name="webfetch" ...>...</skill>` + `https://tangledgroup.com/`), the
optimized program produces:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_optimized",
      "type": "function",
      "function": {
        "name": "bash",
        "arguments": "{\"command\":\"uv run --script /home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/scripts/webfetch.py https://tangledgroup.com\",\"timeout\":300}"
      }
    }
  ]
}
```

This matches the target message:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "<assigned by pi>",
      "type": "function",
      "function": {
        "name": "bash",
        "arguments": "{\"command\":\"uv run --script /home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/scripts/webfetch.py https://tangledgroup.com\",\"timeout\":300}"
      }
    }
  ]
}
```

Target command: `uv run --script /home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/scripts/webfetch.py https://tangledgroup.com`

## Verification

| score | task | generated command |
|---|---|---|
| OK | `https://dspy.ai` | `uv run --script /home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/scripts/webfetch.py https://d` |
| OK | `Search for "liquid ai lfm2.5"` | `uv run --script /home/mtasic/projects-b/pi-slm/.agents/skills/websearch/scripts/websearch.py "liquid` |
| OK | `https://tangledgroup.com/` | `uv run --script /home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/scripts/webfetch.py https://t` |

Mean score: **1.000** (3/3 perfect).

Stress test (fresh rollouts, temperature 0.7): worst per-example success rate **1.00**.

When pi executes the tool call, it returns the tool result (TangledGroup page
markdown, first lines):

```
TangledGroup


# Tangled Group, Inc

## Private and Secure Collaborative AI solutions

### Analyze extensive data, conduct market analysis, and present results with tables, charts and beautiful UI components.
...
```

## Synthetic teaching message (assistant_message.json)

A synthetic assistant message (text `content` + `reasoning_content`) answering
"How can a skill be used?" — crafted from the GEPA-learned rules above, in
first person — that, when placed in the seed session right after that question,
makes `LiquidAI/LFM2.5-2.6B` emit the perfect bash tool call for the webfetch
invocation (real pi system prompt, four tools, no GEPA instructions).

Saved as `assistant_message.json`:

- `content` — the pattern: `/skill:<name>` → `<skill name=... location=...>`
  block + task; act, don't explain; match usage line + flags; positional arg
  from the task; absolute script path = dirname(location) + `/scripts/`;
  run PEP 723 scripts via `uv run --script <abs path> <args>`.
- `reasoning_content` — the matching chain of thought ("I perform the task,
  I run the matching script via `uv run --script ...`").

Verification against `LiquidAI/LFM2.5-2.6B` (`uv run --script teach.py`):

| request variant | perfect tool calls |
|---|---|
| seed's original answer (baseline) | 0/5 |
| synthetic message, with `reasoning_content` | **10/10** (two runs of 5) |
| synthetic message, text only (no reasoning) | 0/3 — model explains the skill again |

The `reasoning_content` is load-bearing: without it the small model falls back
to explaining the skill. Keep both fields when replaying/using the message.
