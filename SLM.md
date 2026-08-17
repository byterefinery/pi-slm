# SLM Extension Specification

Reliability extension for Small Language Models.

## Feature 1:
  - On new session, insert first two synthetic (compact in terms of tokens) messages.
  - First message is with `Available skills:` as valid YAML, and should have name (single line text), description (single line text), list of reference paths for reference files as absolute paths, and list of scripts with  script files as absolute paths. If model supports reasoning insert both short synthetic reasoning. You can find available skills from `pi` internal API.
  - Second message is `Available tools:` as valid YAML. You can find available tools from `pi` internal API.
  - After these two synthetic messages are inserted insert user message/request.
  - Reason why we do this is that small language models forget from system message mentioned skills and tools, so we remind them what they can use.
  - Clarifications (agreed during implementation):
    - tools entries carry the whole function signature: the tool's `parameters` JSON schema (all params and types) converted to YAML.
    - YAML strings are emitted as plain (unquoted) scalars wherever YAML-safe, to save tokens; double quotes only as a correctness fallback.

---

## Feature 1: Available skills and tools

### Problem

Small language models reliably use skills and tools only while their names are
fresh in context. Pi announces them once, in the system prompt: skills as an
`<available_skills>` XML block (name/description/location) and tools as
one-line snippets (see `system-prompt.ts` / `formatSkillsForPrompt`). SLMs
frequently forget that content by the first real request — they hallucinate
capability names, skip available skills, or fumble tool parameter formats.

### Solution

`src/slm.ts` hooks `before_agent_start` (fires after the user submits a
prompt, before the agent loop; the user message is not persisted yet). When
the session branch contains no user message yet — i.e. this is the first
request of a new session (resumed/forked sessions and all later prompts are
skipped) — the extension injects two synthetic messages via
`pi.sendMessage()` (no `triggerTurn`), which appends each message
synchronously to both the agent state and the session tree. Final order in
the session and in the LLM context:

```
[custom_message slm-skills] → [custom_message slm-tools] → [user message]
```

(Only non-message bookkeeping entries such as `model_change` /
`thinking_level_change` may precede them.) Both messages are `display: false`
(hidden from the TUI) and participate in LLM context, so the reminder also
persists for the rest of the session.

Message 1 — `Available skills:` — valid YAML. Skill set = the same loaded
skills the system prompt is built from (`event.systemPromptOptions.skills`),
minus skills with `disable-model-invocation` (mirrors
`formatSkillsForPrompt`). Per skill: `name`, single-line `description`,
`references` (absolute paths of all files under the skill dir's
`references/`, recursive, sorted), `scripts` (same for `scripts/`).

Message 2 — `Available tools:` — valid YAML. Tool set = the active tools
(`selectedTools`, fallback `pi.getActiveTools()`). Per tool: `name`,
single-line `description` (the one-line system-prompt snippet from
`toolSnippets`, fallback flattened `ToolInfo.description`), and the whole
function signature — the tool's `parameters` JSON schema from
`pi.getAllTools()` converted to YAML (all params and types; `null` when the
tool has no parameters).

If the active model supports reasoning (`ctx.model.reasoning === true`, e.g.
`LiquidAI/LFM2.5-2.6B`), each message carries one short synthetic reasoning
line — a YAML comment (`# thinking: ...`) right under the header, so the
document stays valid YAML:

```
Available skills:
# thinking: scanned loaded skills - 2 found. I will check whether the task matches a description, and if so read that skill's SKILL.md and the reference files listed below.
skills:
  - name: slm-alpha
    description: Alpha test skill: handles "quotes", colons: and #hashes for YAML escaping checks.
    references:
      - /tmp/.../slm-alpha/references/api.md
      - /tmp/.../slm-alpha/references/deep/guide.md
    scripts:
      - /tmp/.../slm-alpha/scripts/inner/helper.py
      - /tmp/.../slm-alpha/scripts/run.sh
  - name: slm-beta
    description: Beta test skill without references or scripts.
    references: []
    scripts: []
```

```
Available tools:
# thinking: scanned active tools - 4 found. I will pick the narrowest tool that fits the task.
tools:
  - name: read
    description: Read file contents
    parameters:
      type: object
      required: [path]
      properties:
        path:
          type: string
          description: Path to the file to read (relative or absolute)
        offset:
          type: number
          description: Line number to start reading from (1-indexed)
        limit:
          type: number
          description: Maximum number of lines to read
  - name: bash
    description: Execute bash commands (ls, grep, find, etc.)
    parameters:
      type: object
      required: [command]
      properties:
        command:
          type: string
          description: Bash command to execute
        timeout:
          type: number
          description: Timeout in seconds (optional, no default timeout)
  - name: edit
    description: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
    parameters:
      type: object
      required: [path, edits]
      properties:
        path:
          type: string
          description: Path to the file to edit (relative or absolute)
        edits:
          type: array
          items:
            type: object
            required: [oldText, newText]
            properties:
              oldText:
                type: string
                description: Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in one call.
              newText:
                type: string
                description: Replacement text for this targeted edit.
          description: One or more targeted replacements. Each edit is matched against the original file, not incrementally.
  - name: write
    description: Create or overwrite files
    parameters:
      type: object
      required: [path, content]
      properties:
        path:
          type: string
          description: Path to the file to write (relative or absolute)
        content:
          type: string
          description: Content to write to the file
```

Token economy: strings are emitted as plain (unquoted) YAML scalars whenever
that is safe (fallback to double-quoted + escaped only when the content would
otherwise change meaning: `: `, ` #`, leading indicator chars,
leading/trailing whitespace, newlines, or values that YAML would re-type as
number/bool/null/date).

Empty sets are listed explicitly (`skills: []` / `tools: []`) — both
messages are always inserted.

### Validation

Test suite: `tests/feature1_test.py` (Python 3 + PyYAML only; no other
deps). Run: `python3 tests/feature1_test.py`.

Each scenario runs the real `pi` (0.84.2) with `pi -e src/slm.ts --model
LiquidAI/LFM2.5-2.6B -p ...` in a random temp dir under `/tmp`, fully
isolated via a temp `HOME` (no global skills/settings/`~/.agents/skills`
interference) and a temp session dir. Fixture skills are planted in
`<tmp>/.pi/skills/`: `slm-alpha` (nested `references/` + `scripts/`,
description full of YAML-hostile chars), `slm-beta` (bare), `slm-hidden`
(`disable-model-invocation: true`). The session JSONL is then parsed and
checked (structure, ordering, YAML validity, content, quoting style).

Scenarios:

| # | Scenario | Checks |
|---|----------|--------|
| S1 | skills listed correctly (default run) | exactly one `slm-skills` + one `slm-tools` custom message; order `skills → tools → user`; only bookkeeping entries precede them; YAML parses; listed skills == `{slm-alpha, slm-beta}` (hidden excluded); descriptions single-line and exact; reference/script paths absolute, correct, existing; names are plain (unquoted) scalars |
| S2 | tools listed correctly | `tools` == `[read, bash, edit, write]` in pi's default order; descriptions == pi's one-line snippets and emitted unquoted; each tool has the full `parameters` JSON-schema-as-YAML signature (`type`, `required`, per-param `type`; nested `array`/`items` for `edit` verified exactly) |
| S3 | mixed: restricted tools (`-t read,grep`) | tools listing follows the active set (`[read, grep]` with signatures); skills listing unaffected |
| S4 | mixed: no skills (`--no-skills`) | `skills: []` (with `0 found` reasoning line); full tools listing still correct |
| S5 | mixed: no tools (`-nt`) | `tools: []`; full skills listing still correct |
| S6 | mixed order, repeated prompts (continued session) | second prompt via `pi -c`: synthetic pair injected exactly once, sits before both user messages, no re-injection around the second prompt |
| S7 | non-reasoning model (models.json with `reasoning: false`) | both messages present and correct, but no `# thinking:` lines |

Result (2026-08-17, pi 0.84.2, model `LiquidAI/LFM2.5-2.6B`):
**305 checks passed, 0 failed.**

### Notes

- New-session detection uses "no user message in the session branch" at
  `before_agent_start` time — not entry count, because startup
  bookkeeping entries (`model_change`, `thinking_level_change`) already
  exist when `--model` is passed.
- The synthetic messages are persisted `custom_message` entries
  (`customType: slm-skills` / `slm-tools`), so they stay in context for all
  later turns of the session; they are captured once, at the first prompt.
  Mid-session tool/skill changes (`/tools`, `/reload`) do not rewrite them.
- `references`/`scripts` follow the Agent Skills standard layout
  (`<skill dir>/references/`, `<skill dir>/scripts/`), scanned recursively;
  other file kinds are not listed.
- Skills with `disable-model-invocation: true` are excluded, matching what
  the system prompt shows the model.
- The JSON→YAML emitter is dependency-free (single-file constraint): block
  style, flow style for scalar arrays, plain scalars when safe.
- Works in all run modes (validated via `-p`; TUI/RPC/JSON share the same
  hook path).

---

...
