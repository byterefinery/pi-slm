# SLM Extension Specification

Reliability extension for Small Language Models.

## Feature 1:
  - On new session, simulate a short user/assistant dialogue before the first user request:
    1. system message (pi default, untouched)
    2. user: `Available skills` (simulated)
    3. assistant: short synthetic thinking/reasoning, then available skills as YAML in the main content
    4. user: `Available tools` (simulated)
    5. assistant: short synthetic thinking/reasoning, then available tools as YAML in the main content
    6. the first-ever user request
  - Skills YAML: name (single line text), description (single line text), reference file paths as absolute paths, script file paths as absolute paths. If the model supports reasoning, insert short synthetic reasoning (on the wire it must be visible on the assistant message, not dropped). Available skills come from `pi` internal API.
  - The first real user request is sent exactly as typed; the top-level `tools` field of the provider request is pi core's native function-calling definitions (active tools), not something the extension adds.
  - Tools YAML: name (single line), description (single line), whole function signature (`parameters` JSON schema as YAML, all params and types). Available tools come from `pi` internal API.
  - Reason: small language models forget skills and tools mentioned in the system message, so we remind them what they can use — inside the conversation itself, in a natural Q/A shape.
  - Clarifications (agreed during implementation):
    - tools entries carry the whole function signature: the tool's `parameters` JSON schema (all params and types) converted to YAML.
    - YAML strings are emitted as plain (unquoted) scalars wherever YAML-safe, to save tokens; double quotes only as a correctness fallback.
    - redesign (2026-08-17): the reminder is a simulated user→assistant dialogue instead of two custom user messages — the assistant side is a real assistant message (thinking block + YAML), not a user-role custom message.

---

## Feature 1: Available skills and tools (simulated dialogue)

### Problem

Small language models reliably use skills and tools only while their names
are fresh in context. Pi announces them once, in the system prompt: skills
as an `<available_skills>` XML block (name/description/location) and tools
as one-line snippets (see `system-prompt.ts` / `formatSkillsForPrompt`).
SLMs frequently forget that content by the first real request — they
hallucinate capability names, skip available skills, or fumble tool
parameter formats.

### Solution

`src/slm.ts` hooks `before_agent_start` (fires after the user submits a
prompt, before the agent loop; the user message is not persisted yet). When
the session branch contains no user message yet — i.e. this is the first
request of a new session (resumed/forked sessions and all later prompts are
skipped) — the extension persists a simulated four-message dialogue at the
start of the session, right after the startup bookkeeping entries:

```
[model_change / thinking_level_change]
[custom_message slm-skills]  "Available skills"   (simulated user, display: true)
[message assistant]          thinking + skills YAML
[custom_message slm-tools]   "Available tools"    (simulated user, display: true)
[message assistant]          thinking + tools YAML
[user message]               the first real request
```

The resulting context sent to the provider is exactly:

```
1. system message            (pi default, untouched)
2. user:      "Available skills"
3. assistant: <skills YAML>  (thinking block kept in pi, see below)
4. user:      "Available tools"
5. assistant: <tools YAML>
6. user:      first real user request
```

**Mechanics — three parts:**

1. *Detection* — at `before_agent_start` time the session branch contains
   no `message` entry with role `user` (startup bookkeeping entries such as
   `model_change` / `thinking_level_change` already exist, so entry count
   is not a reliable signal).
2. *Persistence* — the simulated user messages go in with
   `pi.sendMessage()` (custom messages, `display: true`, no `triggerTurn`;
   the non-streaming path appends synchronously to the session tree and to
   the agent state, and emits `message_start`/`message_end` so the TUI
   shows them live). The synthetic assistant messages are persisted with
   `SessionManager.appendMessage()` — the runtime object behind
   `ctx.sessionManager` is the full `SessionManager` instance (the public
   context type only exposes its read-only pick), so this is the same
   public method the core uses, one cast away.
3. *Live LLM context* — persisted entries alone are not enough for the
   current run: the provider is called with the agent's in-memory state,
   which `pi.sendMessage()` populates with the two asks but nothing adds
   assistant messages to (there is no official API for that). So the
   extension also subscribes to the official `context` event
   (`transformContext`), which fires on every provider call with the full
   `AgentMessage[]`: when the session is the one this process injected
   into, it re-inserts the two synthetic assistant messages right after
   each simulated user message if they are missing from the live state.
   For resumed/continued sessions the state is restored from the session
   file and already contains the full dialogue, the per-session check
   no-ops, and nothing is duplicated.

**Skills YAML** — skill set = the same loaded skills the system prompt is
built from (`event.systemPromptOptions.skills`), minus skills with
`disable-model-invocation` (mirrors `formatSkillsForPrompt`). Per skill:
`name`, single-line `description`, `references` (absolute paths of all
files under the skill dir's `references/`, recursive, sorted), `scripts`
(same for `scripts/`).

**Tools YAML** — tool set = the active tools (`selectedTools`, fallback
`pi.getActiveTools()`). Per tool: `name`, single-line `description` (the
one-line system-prompt snippet from `toolSnippets`, fallback flattened
`ToolInfo.description`), and the whole function signature — the tool's
`parameters` JSON schema from `pi.getAllTools()` converted to YAML (all
params and types; `null` when the tool has no parameters).

**Synthetic reasoning** — if the active model supports reasoning
(`ctx.model.reasoning === true`, e.g. `LiquidAI/LFM2.5-2.6B`), each
synthetic assistant message carries one short synthetic reasoning line as a
real `thinking` content block before the YAML text block:

- skills: `scanned loaded skills - <n> found. I will check whether the task
  matches a description, and if so read that skill's SKILL.md and the
  reference files listed below.`
- tools: `scanned active tools - <n> found. I will pick the narrowest tool
  that fits the task.`

For the OpenAI Completions API (the llama.cpp server path) the block
additionally carries `thinkingSignature: "reasoning_content"`. Pi's
serializer (`openai-completions.js`) then emits the thinking text as a
`reasoning_content` field on the wire message — the standard
OpenAI-compatible convention for carrying reasoning in chat history (the
replay path pi documents for llama.cpp server) — while `content` stays the
pure YAML document: `reasoning_content` on the wire is guaranteed, not
dropped. For other APIs no signature is set: the block stays in the
session/TUI and is replayed or dropped by the provider's serializer as
usual (it is never sent with a signature the provider would reject).

Sample (session JSONL, abridged — the skills assistant entry):

```json
{
  "role": "assistant",
  "content": [
    { "type": "thinking",
      "thinking": "scanned loaded skills - 1 found. I will check whether the task matches a description, and if so read that skill's SKILL.md and the reference files listed below.",
      "thinkingSignature": "reasoning_content" },
    { "type": "text",
      "text": "skills:\n  - name: demo-skill\n    description: A demo skill for sampling.\n    references:\n      - /tmp/.../demo-skill/references/ref.md\n    scripts: []" }
  ],
  "api": "openai-completions",
  "provider": "llamacpp",
  "model": "LiquidAI/LFM2.5-2.6B",
  "usage": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
             "totalTokens": 0, "cost": { ...zeros... } },
  "stopReason": "stop"
}
```

Skills YAML sample (the `text` block):

```
skills:
  - name: slm-alpha
    description: "Alpha test skill: handles \"quotes\", colons: and #hashes for YAML escaping checks."
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

Tools YAML sample (abridged to two tools):

```
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
                description: Exact text for one targeted replacement. ...
              newText:
                type: string
                description: Replacement text for this targeted edit.
          description: One or more targeted replacements. ...
```

**Wire behavior (what the provider actually receives)** — validated against
the real provider payload (`before_provider_request`). Net wire order for
the first request (llamacpp/LFM, reasoning on):

```
system  → user "Available skills" → assistant <skills YAML> + reasoning_content
       → user "Available tools"  → assistant <tools YAML> + reasoning_content
       → user <first real request>
```

The OpenAI Completions serializer sends the assistant reply `content` as a
plain string (the pure YAML document); the synthetic reasoning rides in the
`reasoning_content` field (see above). The first real user request is sent
exactly as typed (OpenAI content-block form `[{type:"text",text:...}]`,
built by pi core). The request's top-level `tools` field is pi core's
native function-calling definitions for the active tools (`context.tools` →
`llmContext.tools` in the agent loop) — the extension never touches it;
it is sent on every request while tools are active, independently of the
synthetic tools YAML (validated in S9: identical with and without the
extension).

**Display** — the simulated user messages are shown in the TUI live
(custom-message styling, `display: true`). The synthetic assistant messages
have no live streaming events in the run they were injected in, so they
render when the session is loaded/reopened (they are regular assistant
message entries in the session file, with expandable thinking blocks).

**Token economy** — strings are emitted as plain (unquoted) YAML scalars
whenever that is safe (fallback to double-quoted + escaped only when the
content would otherwise change meaning: `: `, ` #`, leading indicator
chars, leading/trailing whitespace, newlines, or values that YAML would
re-type as number/bool/null/date). Empty sets are listed explicitly
(`skills: []` / `tools: []`) — the dialogue is always complete.

### Validation

Test suite: `tests/feature1_test.py` (Python 3 + PyYAML only; no other
deps). Run: `python3 tests/feature1_test.py`.

Each scenario runs the real `pi` (0.84.2) with `pi -e src/slm.ts --model
LiquidAI/LFM2.5-2.6B` in a random temp dir under `/tmp`, fully isolated via
a temp `HOME` (no global skills/settings/`~/.agents/skills` interference)
and a temp session dir. A second helper extension
(`tests/payload-logger.ts`, env-gated) logs every provider request payload
via the `before_provider_request` event, so the suite validates **both**
the session JSONL (dialogue structure, ordering, assistant message shape,
YAML validity/content/quoting) **and** the actual wire context (message
roles, contents, order — including the restored dialogue on continued
sessions). Fixture skills are planted in `<tmp>/.pi/skills/`: `slm-alpha`
(nested `references/` + `scripts/`, description full of YAML-hostile
chars), `slm-beta` (bare), `slm-hidden` (`disable-model-invocation: true`).

Scenarios:

| # | Scenario | Checks |
|---|----------|--------|
| S1 | skills listed correctly (default run) | exactly one ask + one assistant per side; order `askS → asst → askT → asst → user` with only bookkeeping entries before; assistant entry shape (role, `stopReason: stop`, zero usage, model metadata, one thinking + one text block, thinking carries the `reasoning_content` signature); skills YAML parses; listed skills == `{slm-alpha, slm-beta}` (hidden excluded); descriptions single-line and exact; reference/script paths absolute, correct, existing; names/descriptions are plain (unquoted) scalars; wire: `system → ask → skills YAML → ask → tools YAML → user`, assistant wire content a plain string, wire `reasoning_content` equals the session thinking text |
| S2 | tools listed correctly | same run: `tools` == `[read, bash, edit, write]` in pi's default order; descriptions == pi's one-line snippets, emitted unquoted; each tool carries the full `parameters` JSON-schema-as-YAML signature (`type`, `required`, per-param `type`; nested `array`/`items` for `edit` verified exactly) |
| S3 | restricted tools (`-t read,grep`) | tools listing follows the active set (`[read, grep]` with signatures); skills listing unaffected; wire consistent |
| S4 | no skills (`--no-skills`) | assistant skills text is exactly `skills: []`; thinking says `0 found`; full tools listing still correct |
| S5 | no tools (`-nt`) | assistant tools text is exactly `tools: []`; full skills listing still correct |
| S6 | repeated prompts (`pi -c` continued) | dialogue injected exactly once and sits before both user messages; no re-injection around the second prompt; the continued run's wire context starts with the restored dialogue, then the restored history (first request + real reply), then the new prompt; restored assistant texts identical to run 1 |
| S7 | non-reasoning model (`reasoning: false`) | both assistant messages present and correct but with no thinking block; no `reasoning_content` fields on the wire; everything else identical |
| S8 | JSON mode (`--mode json`) | `message_start` events for both asks are emitted `display: true` with the exact ask texts, in dialogue order (ask → ask → user) — i.e. the asks are visible in the TUI; wire consistent |
| S9 | user message verbatim + tools field from pi core | first user request (`list tools`) sent verbatim in both an extension run and a baseline run without the extension; the request's top-level `tools` field (4 built-in function definitions with name + JSON-schema parameters) is identical in both runs — the extension never touches it |

Result (2026-08-17, pi 0.84.2, model `LiquidAI/LFM2.5-2.6B`):
**538 checks passed, 0 failed.**

### Notes

- New-session detection uses "no user message in the session branch" at
  `before_agent_start` time — not entry count, because startup
  bookkeeping entries (`model_change`, `thinking_level_change`) already
  exist when `--model` is passed.
- The dialogue is persisted once per session (two `custom_message` entries
  + two assistant `message` entries) and stays at the head of the context
  for all later turns. Resumed/continued/forked sessions restore it from
  the session file — the `context` handler then no-ops (per-session
  guard), so nothing is duplicated. Mid-session tool/skill changes
  (`/tools`, `/reload`) do not rewrite it.
- If a compaction drops the dialogue entries from the kept context, the
  `context` handler no-ops (the asks are gone from the live state) and the
  reminder is gone too — same semantics as any compacted history.
- The `context` event is the official context-rewrite hook
  (`transformContext` in pi-agent-core, applied on every provider call);
  the extension uses it because there is no official way to push
  assistant-role messages into the agent's live state — only
  `pi.sendMessage()` (custom/user-role) writes there directly.
- `ctx.sessionManager` is typed as a read-only `Pick` of `SessionManager`,
  but the runtime object is the full instance; `appendMessage()` is the
  same public class method the core uses to persist messages.
- The request's top-level `tools` field (native function-calling
  definitions) and the OpenAI content-block form of user messages are pi
  core behavior — the extension only rewrites the `messages` array via the
  `context` event (S9 validates the `tools` field is byte-identical with
  and without the extension).
- If a session is switched to a different model mid-session, pi's
  cross-model message transform degrades the synthetic thinking block to
  plain text (standard pi behavior for foreign-model thinking); the YAML
  content is unaffected.
- `references`/`scripts` follow the Agent Skills standard layout
  (`<skill dir>/references/`, `<skill dir>/scripts/`), scanned recursively;
  other file kinds are not listed.
- Skills with `disable-model-invocation: true` are excluded, matching what
  the system prompt shows the model.
- The JSON→YAML emitter is dependency-free (single-file constraint): block
  style, flow style for scalar arrays, plain scalars when safe.
- Works in all run modes (validated via `-p` and `--mode json`; TUI/RPC
  share the same hook path).

---
