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

**Event:** `tool_call` (fires third, after Features 3 and 4), `session_start`, `turn_end`

Detects repeated tool calls. Two behaviors depending on tool type:

- **`read`**: warns with a random nudge message, does **not** terminate — lets the model recover
- **All other tools**: blocks with `terminate: true` — aborts the turn

### Problem

Small models repeatedly read the same file with identical arguments, getting stuck in doom loops. The original sliding window only catches consecutive identical calls within a single turn — it misses cross-turn repeats (window resets at `turn_end`) and interleaved reads (`read A, bash X, read A`).

### Solution

A persistent sliding window (`readLoopWindow`) tracks `read` call signatures across turns. On threshold breach, returns a randomly picked warning from a pool of 10 messages — no termination, so the model can pivot. Non-read tools use the original `callWindow` with `terminate: true`.

| Tool | Window | Scope | On breach |
|------|--------|-------|-----------|
| `read` | `readLoopWindow` (persists across turns) | Same args, any turn | Warn (random message), continue |
| other | `callWindow` (reset at `turn_end`) | Consecutive identical calls | Block + terminate |

### Configuration

- Default threshold: 3. Tunable via `PI_LOOP_THRESHOLD` (min 2).
- Read at `session_start`, cached.

### Mechanism

#### read tool

On `tool_call` for `read`:
- Build signature: `read:path:offset:limit`.
- Push onto `readLoopWindow`, trim to threshold length.
- If all entries identical, return `{ block: true, reason: pickRandom(doomLoopWarnings) }` — **no terminate**.
- `readLoopWindow` persists across turns. Cleared on any non-`read` tool call.

#### other tools

On `tool_call` for non-read tools:
- Build signature: `(toolName, normalizedInput)`. Normalization:
  - `bash`: strip/normalize whitespace in command
  - `write`/`edit`: `toolName:path:offset:limit`
  - Other: `toolName:JSON(input)`
- Push onto `callWindow`, trim to threshold length.
- If all entries identical, return `{ block: true, reason: "...", terminate: true }`.
- Clear `callWindow` on `turn_end`.

### Warning messages

10 varied messages, randomly picked on each breach:

```
Warning: doom looping detected, let's try another approach.
Heads up: you're stuck in a read loop. Try something different.
Notice: repeated reads of the same file. Pivot to a new strategy.
Alert: circular file reading detected. Change direction.
Caution: you keep reading the same file. Explore a different path.
Tip: spinning in circles on this file. Try a fresh angle.
Reminder: repeated file access. Shift your approach.
Watch out: loop detected on this read. Move on.
Hint: you've already read this. Consider the next step.
Pause: looping on the same read. Let's try another route.
```

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
| `turn_end` | Feature 5 (reset callWindow) |
| `message_end` | Feature 1 (listing guard) |

---

## Module Structure

Single file: `src/pi-slm.ts`. No external dependencies beyond `@earendil-works/pi-coding-agent`, `node:fs`, `node:path`.

### State

- `loopThreshold: number` — from env or default 3
- `callWindow: Array<{ toolName: string; signature: string }>` — loop detection window for non-read tools (cleared at turn_end)
- `readLoopWindow: string[]` — persistent window for read loop signatures across turns (cleared on non-read tool call)
- `doomLoopWarnings: string[]` — pool of 10 varied warning messages for read loop breaches
- `knownSkills: Array<{ name: string; description: string; filePath?: string }>` — captured skills
- `knownTools: Array<{ name: string; description: string }>` — captured tools
- `failedToolSchemas: Set<string>` — tools with schema errors

### Helpers

- `buildSignature(toolName, input)` — normalize tool call for loop detection
- `pickRandom<T>(arr: T[]): T` — pick a random element from an array
- `detectBashBypass(command)` — check bypass patterns
- `isSkillsListing(text)` — detect skills listing messages
- `isToolsListing(text)` — detect tools listing messages
- `buildSkillsListing()` — build factual skills replacement text (simple list)
- `buildToolsListing()` — build factual tools replacement text (native format)
- `extractNonListingContent(text)` — preserve trailing content
- `discoverSkills(cwd)` — filesystem skill discovery
