# pi-slm Extension Specification

Reliability extension for small language models (<14B params). Blocks destructive actions, detects loops, corrects hallucinations, and helps models recover from errors.

Features are ordered by the request lifecycle: setup → input → tool call → tool result → output → cleanup.

---

## Feature 1: Listing Guard (Skills + Tools)

**Event:** `before_agent_start` (capture), `message_end` (intercept)

Replace hallucinated skills and tools listings with factual ones. Single mechanism, two detection paths. Captured data feeds Feature 5 (Tool Hallucination Guard) and Feature 7 (Skill Invocation).

### Capture

On `before_agent_start`:
- Skills: capture `event.systemPromptOptions.skills` into `knownSkills` (fields: `name`, `description`, `filePath`).
- Tools: capture `event.systemPromptOptions.selectedTools` + `toolSnippets` into `knownTools`.

### Intercept

On `message_end` for assistant messages:
- **Skills listing**: detect via header (`available skill`, `installed skill`, `list of skill`, etc.) + list structure. Replace with factual data as a simple markdown list.
- **Tools listing**: detect via header (`available tool`, `i have access to`, `tools available`, etc.) + list/table structure. Replace with factual data in the model's native format (the same format shown in the system prompt).
- Preserve any trailing non-listing content (follow-up questions, etc.).

### Skills format

Simple markdown list:

```
Available skills:

- git: Git version control
- websearch: Search the web
```

If empty: "No skills are currently loaded."

### Tools format

Native model format — the compact list as the model sees it in its system prompt:

```
Available tools:

- read: Read file contents
- write: Create or overwrite files
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- bash: Execute bash commands (ls, grep, find, etc.)
```

If empty: "No tools are currently active."

---

## Feature 2: Skill Invocation

**Event:** `input`

Handle `/skill:<NAME>` directives in user input.

### Syntax

```
/skill:SKILL_NAME [user_message]
```

### Mechanism

On `input`:
- Match `/skill:(\S+)(?:\s+(.+))?` at start of `event.text`.
- Resolve skills from `knownSkills`; if empty, discover from filesystem (`~/.pi/agent/skills/`, `~/.agents/skills/`, `.pi/skills/`, `.agents/skills/`).
- If not found, return `{ action: "transform", text: "Skill not found: ..." }`.
- Load SKILL.md from `skill.filePath`.
- Build transformed prompt: skill content + task + instruction to load references via `read` tool.
- Return `{ action: "transform", text: prompt }`.

### Notes

- Fires at step 2 of input processing, before pi's built-in skill expansion (step 3). Returning `transform` prevents built-in expansion from seeing `/skill:`.
- References in `skill/references/` are NOT auto-loaded. Model uses `read` tool on demand.

---

## Feature 3: Tool Hallucination Guard

**Event:** `tool_call` (fires first, before Features 4 and 5)

Block tool calls to non-existent tools and calls with missing required parameters.

### Validation

On `tool_call`:
- Check `event.toolName` against `knownTools`.
- If not found, return `{ block: true, reason: "Unknown tool. Available: <names>." }`.
- For known tools, validate required params against real native schemas:
  - `read`: `path` (non-empty string)
  - `write`: `path` (non-empty string) + `content` (non-empty string)
  - `edit`: `path` (non-empty string) + `edits` (non-empty array)
  - `bash`: `command` (non-empty string)
- If missing, return `{ block: true, reason: "Missing required parameter: ..." }`.

### Notes

- Skip tool names starting with `$` or `_` (internal conventions).

---

## Feature 4: Write Guard

**Event:** `tool_call` (fires second, after Feature 3)

Block `write` on existing files and bash patterns that bypass it.

### Write tool interception

On `tool_call` for `write`:
- Resolve `event.input.path` relative to `ctx.cwd`.
- Check existence via `fs.access(path, F_OK)`.
- If exists, return `{ block: true, reason: "File already exists. Use edit instead." }`.
- If not, allow through.

### Bash bypass interception

On `tool_call` for `bash`, detect and block file-write patterns:
- Redirect writes: `> path` or `>> path` anywhere in command (catches `echo ... > f`, `printf ... > f`, `cat > f`)
- `tee` / `tee -a` writing to files
- `sed -i` in-place edits
- `rm -rf` targeting `.git`, `.pi`, `.agents`, `node_modules`
- `truncate`, `shred`, `dd of=...` targeting files

Return `{ block: true, reason: "Blocked: <pattern>. Use edit tool." }`.

---

## Feature 5: Loop Detection

**Event:** `tool_call` (fires third, after Features 3 and 4), `message_end`, `session_start`, `turn_end`

Abort when the model repeats the same tool call N consecutive times. Extends beyond simple tool-call repetition to detect reads of the same file that produce identical thinking and output across turns.

### Problem

The original sliding window catches consecutive identical tool calls within a single turn. However, small models exhibit two additional loop patterns:

1. **Interleaved reads** — `read A, bash X, read A, bash Y, read A` — the standard window sees mixed calls and never triggers, even though the same file is being re-read repeatedly.
2. **Cross-turn read loops** — the model reads a file, produces thinking and output, then in the next turn reads the same file again with the same thinking and output. The per-turn window resets at `turn_end`, so this pattern is invisible.

### Solution

Three complementary detection layers, all subject to `PI_LOOP_THRESHOLD`:

| Layer | What it catches | Mechanism | Scope |
|-------|----------------|-----------|-------|
| **callWindow** | Consecutive identical tool calls | Sliding window of `(toolName, normalizedInput)` signatures | Same turn, all tools |
| **pendingReads** | Same file re-read within a turn, even interleaved with other tools | Counts prior reads of same `(path, offset, limit)` in current turn | Same turn, `read` only |
| **readLoopWindow** | Repeated reads across turns with identical thinking + output | Signatures of `read:path:offset:limit:hash(thinking+output)`, persists across turns | Cross-turn, `read` only |

### Configuration

- Default threshold: 3. Tunable via `PI_LOOP_THRESHOLD` (min 2).
- Read at `session_start`, cached.

### Mechanism

#### callWindow (standard)

On `tool_call` for any tool:
- Build signature: `(toolName, normalizedInput)`. Normalization:
  - `bash`: strip/normalize whitespace in command
  - `read`/`write`/`edit`: `toolName:path:offset:limit`
  - Other: `toolName:JSON(input)`
- Push onto sliding window, trim to threshold length.
- If all entries identical, return `{ block: true, reason: "...", terminate: true }`.
- Clear on `turn_end`.

#### pendingReads (same-turn read duplicates)

On `tool_call` for `read`:
- Count entries in `pendingReads` matching `(path, offset, limit)`.
- If `count + 1 >= threshold`, return `{ block: true, reason: "...", terminate: true }`.
- Push current read params onto `pendingReads`.
- Clear `pendingReads` on `turn_end` and on any non-`read` tool call.

#### readLoopWindow (cross-turn read loops)

On `message_end` for assistant messages:
- If `pendingReads` is non-empty, extract all text content (thinking + output).
- Normalize text: strip `<thinking>` tags, collapse whitespace.
- Compute hash (djb2 → base36) of normalized text.
- For each pending read, build signature: `read:path:offset:limit:hash`.
- Push signature onto `readLoopWindow`, trim to threshold length.

On `tool_call` for `read`:
- If `readLoopWindow` has >= threshold entries and all are identical, return `{ block: true, reason: "...", terminate: true }`.

`readLoopWindow` persists across turns (not cleared at `turn_end`). Cleared on any non-`read` tool call (pattern break).

---

## Feature 6: EISDIR to Directory Listing

**Event:** `tool_result` (fires first on result)

Convert `read` errors on directories into directory listings.

On `tool_result` for `read` with `EISDIR` in error:
- Extract path from `event.input.path`.
- Run `readdir(resolvedPath, { withFileTypes: true })`.
- Return `{ content: [{ type: "text", text: listing }], isError: false }`.
- Format: `d dirname` / `- filename` / `l symlinkname`, sorted alphabetically, one per line.
- If `readdir` fails, keep original error.

---

## Feature 7: Tool Schema Hint

**Event:** `tool_result` (detection), `context` (injection)

Inject plain-language hints when tool calls fail parameter validation.

### Detection

On `tool_result` with `isError`:
- Match schema-error patterns: `invalid argument`, `missing required parameter`, `expected type`, `unknown parameter`, `validation error`, `schema error`.
- Add tool name to `failedToolSchemas` set.

### Injection

On `context`:
- If `failedToolSchemas` is non-empty, append a hint to the last user message:
  `"Note: The following tool(s) had parameter validation errors. Check tool definitions for correct parameter names and types: <names>."`
- Clear the set after injection (one-time per failure).

### Safety

- Loop detection (Feature 5) still applies.
- Only inject for schema errors, not file-not-found or permission errors.

---

## Event Handler Order

### `tool_call` (sequential)

1. Feature 3 — Tool Hallucination Guard (block unknown tools / missing params first)
2. Feature 4 — Write Guard (block write on existing files, bash bypasses)
3. Feature 5 — Loop Detection (track and abort repeats)

### Full event mapping

| Event | Features |
|-------|----------|
| `session_start` | Feature 5 (read threshold) |
| `before_agent_start` | Feature 1 (capture skills + tools) |
| `input` | Feature 2 (skill invocation) |
| `tool_call` | Features 3, 4, 5 (in order) |
| `tool_result` | Features 6, 7 |
| `context` | Feature 7 (inject hints) |
| `turn_end` | Feature 5 (reset callWindow + pendingReads) |
| `message_end` | Feature 1 (listing guard), Feature 5 (build readLoopWindow signatures) |

---

## Module Structure

Single file: `src/pi-slm.ts`. No external dependencies beyond `@earendil-works/pi-coding-agent`, `node:fs`, `node:path`.

### State

- `loopThreshold: number` — from env or default 3
- `callWindow: Array<{ toolName: string; signature: string }>` — standard loop detection window (cleared at turn_end)
- `pendingReads: Array<{ path: string; offset: string; limit: string }>` — read calls in current turn (cleared at turn_end)
- `readLoopWindow: string[]` — cross-turn read loop signatures with thinking/output hash (persists across turns)
- `knownSkills: Array<{ name: string; description: string; filePath?: string }>` — captured skills
- `knownTools: Array<{ name: string; description: string }>` — captured tools
- `failedToolSchemas: Set<string>` — tools with schema errors

### Helpers

- `buildSignature(toolName, input)` — normalize tool call for loop detection
- `hashText(text)` — strip thinking tags, normalize whitespace, compute djb2 hash (base36) for compact signature
- `detectBashBypass(command)` — check bypass patterns
- `isSkillsListing(text)` — detect skills listing messages
- `isToolsListing(text)` — detect tools listing messages
- `buildSkillsListing()` — build factual skills replacement text (simple list)
- `buildToolsListing()` — build factual tools replacement text (native format)
- `extractNonListingContent(text)` — preserve trailing content
- `discoverSkills(cwd)` — filesystem skill discovery
