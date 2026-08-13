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

## Feature 6: Explicit Skill Invocation

Handle `/skill:<SKILL_NAME>` directives in user input. Load the skill's SKILL.md and execute the task per its instructions.

### Syntax

```
/skill:<SKILL_NAME> [user_message]
```

- `SKILL_NAME` -- required. Must match the skill's `name` field in frontmatter exactly.
- `user_message` -- optional. The task or question to pass to the skill. If omitted, use the skill for the current context.

### Mechanism

Use the `input` event to detect and handle `/skill:` directives.

On `input`:
- Check if `event.text` starts with `/skill:`.
- Extract `SKILL_NAME` (everything between `/skill:` and the first space or end of string).
- Extract `user_message` (everything after the first space following `SKILL_NAME`, trimmed).
- Find the matching skill in `knownSkills` by name (case-sensitive exact match).
- If no match, return `{ action: "transform", text: "Skill not found: " + SKILL_NAME + ". Available skills: " + knownSkills.map(s => s.name).join(", ") }`.
- If match found, load the skill's SKILL.md file content.
- Build a transformed prompt that includes:
  - The skill's full SKILL.md body.
  - The user_message appended at the end (if provided).
  - Instructions to load reference files on demand via the `read` tool.
- Return `{ action: "transform", text: <transformed prompt> }`.

### Reference file handling

Reference files are not loaded automatically. The transformed prompt instructs the model to use the `read` tool to load only the references needed for the task. References live in `<skill>/references/NN-topic.md` and are linked from the `## References` section of SKILL.md.

### Notes

- Skills are resolved from `knownSkills` captured in `before_agent_start`.
- The skill's `location` field from `systemPromptOptions.skills` gives the full path to SKILL.md.
- Do not create, edit, or delete skills. Use them as-is.
- If the skill file cannot be read, return an error message to the user.

## Feature 7: Tool Definition Retry

When a tool call fails, inject the tool's definition back into the context so the SLM can see the correct signature. This helps small models recover from schema errors without wasting turns.

### Mechanism

Use the `tool_result` event to detect failed tool calls and the `context` event to inject tool definitions.

#### Failure detection

On `tool_result`:
- Check if `event.isError` is true.
- Check if the error is schema-related (parameter validation error, missing required field, wrong type, etc.).
- If schema-related, record the tool name in a `failedToolSchemas` set.

#### Definition injection

On `context` (before each LLM call):
- If `failedToolSchemas` is not empty, build a tool definition reminder block.
- For each tool name in the set, look up its schema from `pi.getAllTools()` or from the captured `knownTools` data.
- Append a structured reminder to the last user message or inject as a custom message.
- The reminder format: `Tool definition reminder for <tool_name>: <schema description>`.
- Clear the `failedToolSchemas` set after injection (one-time reminder).

### Loop safety

- The loop detection (Feature 2) still applies. Injecting tool definitions does not bypass loop detection.
- If the same tool fails N times with schema errors, loop detection will abort.
- The tool definition is only injected once per failure, not on every turn.

### Error patterns detected

- `Invalid tool arguments` or `Tool call validation failed`
- `Missing required parameter`
- `Expected type` mismatches
- `Unknown parameter`
- Any error message containing `schema`, `parameter`, `argument`, or `validation`

### Notes

- Use `pi.getAllTools()` to get the full tool schema for the reminder.
- The reminder should be concise. Include tool name, required params, and param descriptions.
- Do not inject definitions for non-schema errors (file not found, permission denied, etc.).

## Feature 8: Tool Call Hallucination Guard

Block tool calls that reference tools which do not exist in the active tool set. SLMs frequently invent tool names (`search_file`, `run_command`, `file_write`, `execute`) or use wrong parameter shapes for real tools.

### Mechanism

Use the `tool_call` event to validate every tool call against the known tool registry.

#### Validation

On `tool_call`:
- Check if `event.toolName` exists in `knownTools` (captured in `before_agent_start`).
- If the tool name is not found in `knownTools`, return `{ block: true, reason: ... }`.
- The reason message lists the actual available tool names so the model can self-correct.

#### Parameter shape validation

For known tools, validate that required parameters are present and non-empty:
- `read` requires `path` (non-empty string).
- `write` requires `path` and `content` (non-empty strings).
- `edit` requires `path` and `edits` (non-empty array).
- `bash` requires `command` (non-empty string).
- For extension tools, check that the input object has at least one non-empty string value (heuristic — extension tool schemas vary).

If a required parameter is missing or empty, return `{ block: true, reason: ... }` with the correct parameter name.

#### Fuzzy match suggestion

When a tool name is not found, check for a close match using Levenshtein distance or simple prefix/suffix similarity against `knownTools`. If a close match exists (distance <= 2 or shares a prefix of length >= 3), include the suggestion in the reason message:
- `Unknown tool "search_file". Did you mean "read"?` → no, too different
- `Unknown tool "filewrite". Did you mean "write"?` → yes, close enough

For the suggestion, compare the unknown name against each known tool name using:
- Case-insensitive exact match (catch `Write` vs `write`).
- Case-insensitive contains (catch `file_write` contains `write`).
- Character-level edit distance <= 2.

If multiple matches exist, pick the one with the smallest distance. If tied, pick the shortest name.

### Notes

- Built-in tools (`read`, `write`, `edit`, `bash`) are always in `knownTools` when not excluded via `--exclude-tools`. The guard handles excluded tools correctly because `knownTools` reflects what's actually active.
- Extension tools from `pi.getAllTools()` or `systemPromptOptions.selectedTools` are included.
- Do not block on tool names that start with `$` or `_` — these may be internal pi conventions.
- The guard fires before write guard (Feature 1) and loop detection (Feature 2) in the `tool_call` handler.


## Feature 9: Skill Reference Hallucination Guard

Intercept assistant messages that reference skills by name, capability, or command when those skills do not exist. SLMs invent skill names (`code-review`, `debug-helper`, `test-runner`) or attribute capabilities to real skills that they don't have.

### Mechanism

Use the `message_end` event to scan assistant messages for skill references.

#### Detection

On `message_end` for assistant messages, scan the text for skill references using these patterns:
- Named reference: backtick-wrapped or bold skill names that look like skill identifiers (`code-review`, `debug-helper`, `my-skill`). Detection via regex: `\b[a-z][a-z0-9_-]{2,}[a-z0-9]\b` appearing in a context that suggests a skill reference (near words like "skill", "use", "invoke", "load", "run", or in a list).
- Capability claim: phrases like "I can use the X skill to", "The X skill will", "Using skill X", "Skill X provides".
- Command reference: phrases like "run npx skills add X", "install skill X", "load skill X" where X is not a known skill.
- Invocation suggestion: the model suggests using `/skill:X` where X is not a known skill.

For each detected reference, check if the skill name exists in `knownSkills`.

#### Replacement

If hallucinated skill references are found:
- Replace the referenced skill name(s) with the closest matching real skill name (if any), or remove the reference.
- If the model claims a capability that no loaded skill provides, replace the claim with a factual statement about what skills are actually available.
- If the model suggests installing a skill via `npx skills add`, allow it (installation is a valid action). Only block references to skills presented as already available.
- Preserve the surrounding message text. Only replace the hallucinated portions.

#### Replacement strategy

Build the replacement by:
1. Find each hallucinated skill reference in the text.
2. For each, check if a close match exists in `knownSkills` (same fuzzy matching as Feature 8).
3. If a close match exists, replace the hallucinated name with the real name.
4. If no close match exists, replace the reference with a note: `[no such skill loaded]`.
5. If the entire message is a skill recommendation or capability claim based on hallucinated skills, replace the full message with a factual statement about available skills (reuse `buildSkillsListing()` from Feature 4).

### Distinction from Feature 4

Feature 4 (Skills Listing Guard) replaces full listing messages. Feature 9 catches inline skill references throughout any message — a single mention of a non-existent skill in the middle of a paragraph, a wrong skill name in a code example, or a fabricated skill capability claim.

### Notes

- The reference detection regex must not trigger on generic words that happen to match the skill name pattern. Require proximity to skill-related keywords or structural context (backticks, bold, list items, `/skill:` prefix).
- Do not block mentions of skills in the context of installation or discovery ("you could install a skill for X"). Only block references presented as already available.
- The `find-skills` skill itself is a real skill for discovery. References to it should not be flagged.
- If `knownSkills` is empty (e.g., `input` event before `before_agent_start`), skip the guard — there's nothing to validate against.

## Module Structure

Single file: `pi-slm.ts`. No submodules. No external dependencies beyond `@earendil-works/pi-coding-agent` and `node:fs` / `node:path`.

### State

Module-level variables:
- `loopThreshold: number` -- read from `PI_LOOP_THRESHOLD` or default 3.
- `callWindow: Array<{ toolName: string; signature: string }>` -- sliding window for loop detection.
- `knownSkills: Array<{ name: string; description: string; location?: string }>` -- captured from `before_agent_start`.
- `knownTools: Array<{ name: string; description: string }>` -- captured from `before_agent_start`.
- `failedToolSchemas: Set<string>` -- tool names that had schema errors, for Feature 7.

### Event handlers

- `session_start` -- read `PI_LOOP_THRESHOLD`, initialize state.
- `before_agent_start` -- capture skills and tools from `systemPromptOptions`.
- `input` -- explicit skill invocation (Feature 6).
- `tool_call` -- tool hallucination guard (Feature 8), write guard (Feature 1), loop detection (Feature 2).
- `tool_result` -- EISDIR conversion (Feature 3), schema failure detection (Feature 7).
- `context` -- tool definition injection (Feature 7).
- `turn_end` -- reset call window.
- `message_end` -- skills listing guard (Feature 4), tools listing guard (Feature 5), skill reference guard (Feature 9).

### Helper functions

- `buildSignature(toolName, input)` -- normalize tool call input for loop detection.
- `detectBashBypass(command)` -- check for write-guard bypass patterns.
- `isSkillsListing(text)` / `isToolsListing(text)` -- detect listing messages.
- `buildSkillsListing()` / `buildToolsListing()` -- build factual replacement text.
- `extractNonListingContent(text)` -- preserve trailing non-listing content.
- `discoverSkills(cwd)` -- scan filesystem for skills when `knownSkills` is empty.
- `findClosestTool(name, knownTools)` -- fuzzy match unknown tool name against known set.
- `findClosestSkill(name, knownSkills)` -- fuzzy match unknown skill name against known set.
- `detectSkillReferences(text)` -- find inline skill references in message text.

## Edge Cases

- Write guard: symlinks. Resolve the real path before checking existence. Do not follow symlinks for the existence check itself. If the symlink target exists, block.
- Loop detection: different tool call IDs but same tool and input. The signature does not include `toolCallId`.
- EISDIR: nested directory paths. `readdir` works on any valid directory path.
- Skills/tools guard: messages that mention skills or tools in passing (not as a listing). The regex should require a list structure (bullets, numbers, or a header line) to trigger.
- Skills/tools guard: the model asks a follow-up question in the same message as the listing. Replace only the listing portion, keep the question. This is handled by replacing the full message text with the corrected listing plus any non-listing content preserved.
- Skill invocation: skill name not found. Return a helpful error with the list of available skills.
- Skill invocation: skill file unreadable. Return an error message, do not crash.
- Skill invocation: `/skill:` with no name after colon. Treat as invalid, return error.
- Tool retry: non-schema errors (file not found, permission denied). Do not inject tool definitions for these.
- Tool retry: loop detection still applies. Schema reminder does not bypass loop threshold.
- Tool retry: definition injected only once per failure. Clear the set after injection.
- Tool hallucination: the model uses a tool name that is a close variant of a real tool (e.g., `file_read` vs `read`). The fuzzy matcher suggests the correct name.
- Tool hallucination: the model uses a real tool name with completely wrong parameters (e.g., `read` with `query` instead of `path`). Parameter shape validation catches this.
- Tool hallucination: extension tools with dynamic names. The guard validates against `knownTools` which reflects what's actually active.
- Skill reference: the model mentions a skill name that matches a real npm package or GitHub repo but isn't loaded. The guard flags it as not loaded (not hallucinated — it exists but isn't installed).
- Skill reference: the model suggests installing a skill. Installation suggestions are allowed; only references to skills presented as already available are flagged.
- Skill reference: skill names that are common English words (e.g., a skill named `search`). The detection requires structural context (backticks, bold, `/skill:` prefix, or proximity to skill-related keywords) to avoid false positives.
- Skill reference: the model correctly uses a skill but misstates its capability. The guard replaces the capability claim with the factual description from `knownSkills`.
- Fuzzy matching: multiple equally-close matches. Pick the shortest name. If still tied, pick the first alphabetically.
