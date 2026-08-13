# pi-slm Extension Specification

Small Language Model reliability extension for pi. Targets models under 14B parameters (LFM2, RWKV7 family, similar).

## Feature 1: Write Guard

Block the `write` tool on files that already exist. Redirect the model to use `edit` instead. Also block common bash bypasses.

### Mechanism

Use the `tool_call` event to intercept `write` and `bash` calls.

### Write tool interception

On `tool_call` for tool `write`:
- Resolve the target path relative to `ctx.cwd`.
- Check if the file exists via `fs.access`.
- If it exists, return `{ block: true, reason: ... }` with a message telling the model to use `edit` instead.
- If it does not exist, allow the call through.

### Bash bypass interception

On `tool_call` for tool `bash`, detect these patterns in `event.input.command`:
- `rm <path> && ...` or `rm <path>; ...` followed by a write-like action (redirect `>`, `tee`, `cat >`, `echo >`, `printf >`, `cp ... <path>`).
- `rm -rf <dir>` targeting a protected directory (`node_modules`, `.git`, `.pi`, `.agents`).
- Truncation via `> <path>` or `: > <path>` on an existing file.
- `truncate`, `shred`, or `dd` targeting an existing file.

For each detected pattern, return `{ block: true, reason: ... }`.

### Notes

- Use `fs.promises.access` with `fs.constants.F_OK` for existence checks.
- Resolve relative paths before checking.
- Do not block `write` to paths inside `node_modules/` or `.git/` here. That is a separate concern. The write guard only cares about existing-vs-new.

## Feature 2: Loop Detection

Detect when the model issues the same tool call with the same or near-identical inputs N consecutive times. Abort with a structured reason.

### Configuration

- Default threshold: 3 consecutive repeats.
- Tunable via environment variable `PI_LOOP_THRESHOLD` (integer, minimum 2).
- Read the env var at session start. Cache the value.

### Mechanism

Use the `tool_call` event to track consecutive calls.

Maintain a sliding window of the last N tool call signatures. A signature is the tuple `(toolName, normalizedInput)`. Normalization means:
- For `bash`, strip leading/trailing whitespace and normalize internal whitespace in the command string.
- For `read`, `write`, `edit`, use the resolved path and stringified params (offset, limit).
- For custom tools, stringify the full input object.

On each `tool_call`:
- Push the signature onto the window. Trim to length N.
- If all N signatures in the window are identical, return `{ block: true, reason: ..., terminate: true }`.
- The reason message includes the tool name, the repeated input, and the count.

### Reset conditions

- On `turn_end`, clear the window. This means the count resets each turn.
- On `agent_end`, clear the window.

### Notes

- Consecutive means within the same turn, not across turns. The model can retry on the next turn.
- Use `terminate: true` so the agent stops early rather than continuing to call the same tool.

## Feature 3: EISDIR to Directory Listing

When `read` is called on a directory path, the raw error is `EISDIR: illegal operation on a directory, read <path>`. Convert this into a directory listing so the model gets the answer it wanted in the same turn.

### Mechanism

Use the `tool_result` event.

On `tool_result` for tool `read`:
- Check if `event.isError` is true and the error content contains `EISDIR`.
- Extract the path from `event.input.path`.
- Run `fs.promises.readdir` with `withFileTypes: true` on the resolved path.
- Format the output as a clean listing: one entry per line, showing type indicator (d for directory, - for file, l for symlink), name, and nothing else. Sort alphabetically.
- Return `{ content: [...], isError: false }` to replace the error with the listing.

### Error handling

If `readdir` itself fails (permission denied, path gone), keep the original error result unchanged.

### Notes

- Use `ctx.signal` if available for abort support on the `readdir` call.
- The listing format should be minimal and LLM-friendly. No sizes, no timestamps.

## Feature 4: Skills Listing Guard

When the model generates text listing available skills, intercept and replace with the actual skills from the system prompt. Prevents hallucinated skill names.

### Mechanism

Two-part approach: capture the real list, then intercept hallucinated responses.

#### Capture phase

On `before_agent_start`, read `event.systemPromptOptions.skills` to get the actual loaded skills. Store them in module-level state. Extract skill names and brief descriptions.

#### Intercept phase

On `message_end` for assistant messages:
- Check if the message content contains a skills listing. Detection via regex patterns like:
  - `available\s+skill` (case-insensitive)
  - `installed\s+skill` (case-insensitive)
  - `list\s+of\s+skill` (case-insensitive)
  - Bullet or numbered lists containing skill-like names after a skills header.
- If a skills listing is detected, replace the full message text with a factual listing built from the captured skills data.
- The replacement text lists each skill name and its one-line description. If no skills are loaded, state that clearly.

### Notes

- The detection regex should be broad enough to catch variations but not trigger on unrelated text that happens to contain the word "skill".
- Require a header or introductory phrase about skills/tools before the list to avoid false positives.

## Feature 5: Tools Listing Guard

Same as Feature 4 but for tools. When the model generates text listing available tools, intercept and replace with the actual tools from the system prompt.

### Mechanism

#### Capture phase

On `before_agent_start`, read `event.systemPromptOptions.selectedTools` and `event.systemPromptOptions.toolSnippets` to get the actual active tools and their descriptions. Store in module-level state.

#### Intercept phase

On `message_end` for assistant messages:
- Check if the message content contains a tools listing. Detection via regex patterns like:
  - `available\s+tool` (case-insensitive)
  - `list\s+of\s+tool` (case-insensitive)
  - `i can use` or `i have access to` followed by tool-like names.
- If a tools listing is detected, replace the full message text with a factual listing built from the captured tools data.
- The replacement text lists each tool name and its one-line description. If no tools are active, state that clearly.

### Notes

- Same false-positive avoidance as Feature 4. Require a header or introductory phrase.
- `selectedTools` may be an array of tool names or tool objects. Handle both shapes.
- `toolSnippets` maps tool names to one-line descriptions.

## Module Structure

Single file: `pi-slm.ts`. No submodules. No external dependencies beyond `@earendil-works/pi-coding-agent` and `node:fs` / `node:path`.

### State

Module-level variables:
- `loopThreshold: number` -- read from `PI_LOOP_THRESHOLD` or default 3.
- `callWindow: Array<{ toolName: string; signature: string }>` -- sliding window for loop detection.
- `knownSkills: Array<{ name: string; description: string }>` -- captured from `before_agent_start`.
- `knownTools: Array<{ name: string; description: string }>` -- captured from `before_agent_start`.

### Event handlers

- `session_start` -- read `PI_LOOP_THRESHOLD`, initialize state.
- `before_agent_start` -- capture skills and tools from `systemPromptOptions`.
- `tool_call` -- write guard (Feature 1), loop detection (Feature 2).
- `tool_result` -- EISDIR conversion (Feature 3).
- `turn_end` -- reset call window.
- `message_end` -- skills guard (Feature 4), tools guard (Feature 5).

## Edge Cases

- Write guard: symlinks. Resolve the real path before checking existence. Do not follow symlinks for the existence check itself. If the symlink target exists, block.
- Loop detection: different tool call IDs but same tool and input. The signature does not include `toolCallId`.
- EISDIR: nested directory paths. `readdir` works on any valid directory path.
- Skills/tools guard: messages that mention skills or tools in passing (not as a listing). The regex should require a list structure (bullets, numbers, or a header line) to trigger.
- Skills/tools guard: the model asks a follow-up question in the same message as the listing. Replace only the listing portion, keep the question. This is handled by replacing the full message text with the corrected listing plus any non-listing content preserved.
