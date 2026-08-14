# SLM Extension Specification

Reliability extension for small language models (<14B params).

Intercepts tool calls, corrects hallucinations, detects loops, deduplicates redundant reads, and steers the model away from degenerate behavior patterns observed in session analysis.

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `PI_EDIT_THRESHOLD` | `3` (min 2) | Max retries for failed `edit` tool calls before reporting failure and changing strategy |
| `PI_LOOP_THRESHOLD` | `3` (min 2) | Number of repeated patterns before loop intervention triggers |
| `PI_READ_DEDUP_WINDOW` | `5` | Lookback window (turns) for detecting duplicate `read` calls on the same file |
| `PI_SLM_ENABLED` | `true` | Master toggle to enable/disable the extension |

---

## Feature 1: Blocks Destructive Actions

### Problem

Small models have poor understanding of tool semantics. They frequently misuse `write` (overwriting entire files instead of targeted edits) and attempt to bypass `edit` by writing raw content. The `write` tool is destructive — it replaces the entire file. For SLMs, this means losing all existing code in a file with a partial or incorrect replacement. The analysis shows a 50% edit failure rate; when edits fail, models fall back to `write` as a crutch.

Additionally, models sometimes try to bypass the `edit` tool entirely by using `bash` to write files (e.g., `echo "content" > file.py` or `cat > file.py << 'EOF'`).

### Solution

- **`write` tool interception:** On `tool_call` for `write`, block the call and return a reason telling the model to use `edit` instead. Provide the correct `edit` schema format in the response so the model understands the expected structure.
- **`bash` bypass detection:** On `tool_call` for `bash`, detect redirection patterns (`>`, `>>`, `cat >`, `tee`, heredocs writing to files, `printf >`, `echo >`). Block destructive bash file writes and redirect to `edit`/`write` tools.
- **Force `edit` correctness:** When blocking, include the exact `edit` tool parameter format: `path`, `edits[].oldText` (exact text to find), `edits[].newText` (replacement text).

### Validation

- `write` tool calls are blocked with a clear reason message
- `bash` calls with `> file` or `cat > file` are blocked
- `bash` calls with `echo "x" >> file` are blocked
- Normal `bash` commands pass through untouched
- `read`, `edit`, `grep`, `find`, `ls` pass through untouched

### Notes

- Allow `bash` writes to temp files or files under `/tmp/`
- Allow `bash` writes when the target is a new file (not existing) — though `write` is already blocked, so this is for consistency
- The block reason should be instructive, not just "blocked" — tell the model exactly what to do instead
- Consider a whitelist mode where certain paths are allowed for `write` (e.g., test outputs)

### Actions

- Implement `tool_call` handler for `write` — always block, return reason with `edit` usage instructions
- Implement `tool_call` handler for `bash` — parse command for file-write patterns, block if detected
- Add regex patterns for bash write detection: `>\s*\S+`, `>>\s*\S+`, `cat\s*>\s*\S+`, `tee\s+\S+`, heredoc `<<` patterns
- Add `isDestructiveBash(command: string): boolean` helper
- Add status indicator in UI showing blocked actions count

---

## Feature 2: Edit Failure Retry with Threshold

### Problem

The analysis shows a 50% edit failure rate with small models. The failure is almost always in `edits[].oldText` — the model provides text that doesn't match exactly (whitespace differences, wrong line numbers, stale content from memory). When edits fail, the model re-reads the file, tries again with the same incorrect `oldText`, and loops.

Without intervention, the model wastes turns on the same failed edit, consuming context and making no progress.

### Solution

- Track consecutive `edit` failures per file path (or per `oldText` hash).
- On each `edit` failure (tool returns "Could not find the exact text"), increment a retry counter.
- On retry (same file, failed edit), inject a synthetic user message or modify the tool result to:
  1. Show the model the actual current content of the file (re-read it)
  2. Give precise guidance on what `oldText` must match exactly
  3. Suggest using `read` first to get current content
- After `PI_EDIT_THRESHOLD` consecutive failures on the same file, report the failure and instruct the model to change strategy (e.g., use `read` to inspect, then provide a smaller/more precise `oldText`, or rewrite the section entirely with a fresh `write` to a temp file).
- Reset the counter on a successful `edit`.

### Validation

- After 2 consecutive failed edits on the same file, the model receives guidance
- After `PI_EDIT_THRESHOLD` failures, strategy change is enforced
- Successful edit resets the counter
- Different files have independent counters

### Notes

- The retry guidance should vary (random messages) to avoid the model memorizing a single response
- Read the actual file content on failure to provide accurate guidance
- Consider hashing `oldText` to detect if the model is retrying with identical text
- The error message from the `edit` tool already says "Could not find the exact text" — augment it with the actual file content snippet

### Actions

- Add `Map<string, number>` to track edit failure counts per file path
- Implement `tool_result` handler for `edit` — detect failure, increment counter, return augmented content
- Implement `tool_call` handler for `edit` — check if this is a retry of a previously failed `oldText` (hash comparison)
- Add `getEditRetryMessage(file: string, attempt: number): string` helper with varied messages
- On threshold breach, inject steering message to change strategy
- Reset counter on successful `edit` (check `event.isError === false`)

---

## Feature 3: Skills Listing Hallucination Correction

### Problem

Small models hallucinate available skills. When asked (explicitly or implicitly) what skills are available, they invent skill names, descriptions, and capabilities that don't exist. This leads to failed skill invocations and wasted turns.

The model may list skills in response to prompts like "what skills do you have", "list available skills", or even implicitly when planning a task.

### Solution

- Detect skill listing attempts in assistant text content via pattern matching:
  - Text triggers: "available skill", "installed skill", "list of skill", "skills available", "i have access to.*skill"
  - Structural triggers: markdown lists or tables following a skills-related header
- When detected, replace the hallucinated list with the factual list of installed skills from the resource loader.
- Format as a simple markdown list with name and description.
- For non-English prompts, use a lightweight heuristic (not LLM-based) to detect skill listing intent.

### Validation

- When model outputs "Available skills:" followed by a list, the list is replaced with factual data
- Non-English skill queries are detected (heuristic-based)
- Normal text containing the word "skill" in other contexts is not affected

### Notes

- Detection should be done on `message_end` for assistant messages, not streaming (too expensive)
- The factual skill list comes from `ctx.getSystemPromptOptions().skills` or the resource loader
- Only replace when the model is clearly listing/enumerating skills, not when discussing a skill conceptually
- Consider using `context` event to filter/replace rather than `message_end` (which may not allow content changes)

### Actions

- Implement `message_end` or `context` handler to detect skill listing patterns
- Add `detectSkillListing(text: string): boolean` helper with regex patterns
- Add `getFactualSkillsList(ctx: ExtensionContext): string` helper that formats installed skills
- Add `replaceSkillListing(message: AgentMessage, factualList: string): AgentMessage` helper
- Handle non-English detection via keyword patterns (not translation)

---

## Feature 4: Tools Listing Hallucination Correction

### Problem

Same as skills — small models hallucinate available tools. They invent tool names, parameters, and capabilities. When planning or responding to "what tools do you have", the model lists tools that don't exist or describes existing tools incorrectly.

### Solution

- Detect tool listing attempts in assistant text content:
  - Text triggers: "available tool", "i have access to", "tools available", "my tools", "tool.*list"
  - Structural triggers: markdown lists or tables following a tools-related header
- Replace hallucinated list with factual tool list from `pi.getActiveTools()` or `event.systemPromptOptions.selectedTools`.
- Format as a simple markdown list with tool name and one-line description.

### Validation

- When model outputs "Available tools:" followed by a list, the list is replaced
- Normal text mentioning "tool" in other contexts is not affected
- Tool list matches actually registered tools

### Notes

- Same detection approach as skills listing
- Tool list is more stable (built-in tools don't change as often as skills)
- Can cache the factual tool list since it doesn't change mid-session often

### Actions

- Implement `message_end` or `context` handler to detect tool listing patterns
- Add `detectToolListing(text: string): boolean` helper with regex patterns
- Add `getFactualToolsList(ctx: ExtensionContext): string` helper
- Add `replaceToolListing(message: AgentMessage, factualList: string): AgentMessage` helper
- Cache tool list to avoid repeated lookups

---

## Feature 5: Skill Invocation Handling

### Problem

When the model attempts to invoke a skill (e.g., "let me use the git skill" or `/skill:git`), it may:
- Invent skill names that don't exist
- Use incorrect invocation syntax
- Not understand how skills work in the pi ecosystem

### Solution

- Detect skill invocation attempts in assistant text or tool calls.
- Validate the skill name against the actual installed skills list.
- If the skill exists, provide correct invocation syntax or auto-invoke.
- If the skill doesn't exist, correct the model with the list of available skills.

### Validation

- Attempted invocation of non-existent skill is corrected
- Correct skill names are allowed to proceed
- Auto-invocation works for known skills

### Notes

- Skill invocation in pi is typically via `/skill:name` in user input, not assistant output
- The model may describe using a skill without actually invoking it — this is just text, not an action
- Focus on cases where the model is clearly trying to invoke a skill as an action

### Actions

- Implement `message_end` handler to detect skill invocation patterns
- Add `detectSkillInvocation(text: string): { name: string; matched: boolean } | null` helper
- Cross-reference with installed skills list
- Return corrective guidance if skill not found

---

## Feature 6: Skill Loading — References and Scripts

### Problem

Skills have reference files (e.g., `references/01-rpc-mode.md`) and scripts. The paths in SKILL.md are relative to the skill directory. Small models may:
- Misinterpret relative paths
- Try to read references with wrong paths
- Execute scripts from wrong working directories

### Solution

- When a skill is loaded, resolve all reference paths to absolute paths.
- On `tool_call` for `read`, if the path looks like a skill reference (contains skill name or reference pattern), resolve it to the absolute path.
- On `tool_call` for `bash`, if the command executes a skill script, ensure it runs relative to the skill's base directory.
- Convert relative paths in skill documentation to absolute paths before they reach the model.

### Validation

- Reading a skill reference with relative path resolves correctly
- Executing a skill script runs from the correct directory
- Non-skill paths pass through unchanged

### Notes

- Skill base directory is available from the skill metadata (`skill.baseDir`)
- Reference paths in SKILL.md use relative syntax like `references/file.md`
- Script paths may be shell scripts, Python, or TypeScript

### Actions

- Build a `Map<string, string>` of relative → absolute skill reference paths at `session_start`
- Implement `tool_call` handler for `read` — resolve skill reference paths
- Implement `tool_call` handler for `bash` — detect skill script execution, adjust cwd
- Add `resolveSkillPath(path: string, skills: Skill[]): string | null` helper

---

## Feature 7: Tool Invocation — Plain-Language Hints on Failures

### Problem

When tool calls fail parameter validation, the error messages are technical and don't help small models understand what went wrong. The model sees "Could not find the exact text" but doesn't understand that `oldText` must match byte-for-byte, including whitespace.

### Solution

- On `tool_result` for any tool with `isError: true`, inject plain-language hints explaining:
  - What parameter was wrong
  - What the correct format should be
  - An example of correct usage
- For `edit` failures: explain that `oldText` must match exactly, show the actual file content around the expected location
- For `read` failures on directories: convert to directory listing (see Feature 11)
- For `bash` failures: explain common shell errors in plain language

### Validation

- Failed `edit` calls get augmented error messages with file content
- Failed `read` on directories returns listing
- Normal successful tool calls pass through unchanged

### Notes

- Augment, don't replace — keep the original error, add guidance after it
- Vary the guidance messages to avoid model memorization
- Keep hints concise — small models have limited context

### Actions

- Implement `tool_result` handler for all tools — detect `isError`, augment content
- Add `getEditFailureHint(file: string, oldText: string, actualContent: string): string` helper
- Add `getReadFailureHint(path: string, error: string): string` helper
- Add `getBashFailureHint(command: string, error: string): string` helper

---

## Feature 8: Loop Detection

### Problem

The analysis shows massive repetitive behavior:
- Thinking blocks with paragraphs repeated 10-24 times
- "Code looks correct" said 24 times
- Same file read 30 times
- Same series of messages repeating in loops

Small models get stuck in reasoning loops where they re-analyze, doubt, re-read, and repeat the same cycle without making progress.

### Solution

- Track recent tool calls and assistant messages for repetition patterns.
- Detect three types of loops:
  1. **Tool call loops:** Same tool + same parameters (or very similar) repeated `PI_LOOP_THRESHOLD` times
  2. **Message content loops:** Assistant text content that is identical or near-identical to a previous message
  3. **Sequence loops:** A series of N messages that repeats (e.g., read → edit fail → read → edit fail)
- On loop detection, inject a synthetic steering message with a randomized warning from the predefined list.
- Remove duplicate/repeated messages from context to free tokens.

### Validation

- 3+ identical `read` calls on the same file with same params triggers intervention
- 3+ identical assistant messages triggers intervention
- Sequence patterns (read → edit → read → edit) are detected
- Random warning messages are used (not the same one each time)

### Notes

- Loop detection messages (varied, randomly picked):
  ```
  Warning: Doom looping detected, let's try another approach.
  Warning: Heads up, you're stuck in a doom loop. Try something different.
  Warning: Repeated doom loop. Pivot to a new strategy.
  Warning: You keep doom looping. Explore a different path.
  Warning: Spinning in circles. Try a fresh angle.
  Warning: Repeated doom looping. Shift your approach.
  Warning: Loop detected. Move on.
  Warning: You've looped and repeated messages. Consider the next step.
  Warning: Looping on the same messages. Let's try another route.
  ```
- Use `context` event to remove repeated messages from the context window
- Use `turn_end` to track patterns across turns
- Hash tool call inputs for comparison (JSON.stringify of params)
- For near-identical text, use a simple similarity heuristic (not fuzzy matching — just exact or whitespace-normalized comparison)

### Actions

- Add `LoopDetector` class with methods:
  - `recordToolCall(toolName: string, paramsHash: string)`
  - `recordMessage(text: string)`
  - `detectToolLoop(): LoopInfo | null`
  - `detectMessageLoop(): LoopInfo | null`
  - `detectSequenceLoop(): LoopInfo | null`
  - `getRandomWarning(): string`
- Implement `turn_end` handler — record tool calls and messages
- Implement `context` handler — remove detected loop messages
- Implement `tool_call` handler — inject warning on loop detection
- Add sliding window of recent calls/messages (configurable size)

---

## Feature 9: Hallucination Correction

### Problem

Small models hallucinate:
- File contents that don't match reality
- Code that "looks correct" but is broken
- Tool results that never happened
- Task completion when work is unfinished

The analysis shows the model concluding "No modifications are required" despite the code still being broken, and saying "code looks correct" 24 times while producing 8 unfixed tracebacks.

### Solution

- On `message_end` for assistant messages, detect hallucination patterns:
  - "code looks correct" / "code is correct" without a successful test run
  - "no modifications required" / "everything works" when recent tool results show errors
  - Claims of task completion when pending work exists
- When detected, inject a corrective steering message that:
  - Points out the discrepancy ("the last test produced a traceback")
  - Asks the model to verify by actually running the code
  - Provides the actual recent tool results as evidence
- On `context` event, remove hallucinated tool results from context (tool results the model claims happened but aren't in the session log).

### Validation

- "Code looks correct" without recent successful test is corrected
- False completion messages are intercepted
- Hallucinated tool results are removed from context
- Legitimate success messages pass through

### Notes

- Check recent tool results for errors before allowing "success" claims
- Cross-reference model claims with actual session history
- Don't over-correct — some legitimate "looks correct" messages are fine after real verification

### Actions

- Implement `message_end` handler for assistant messages — detect hallucination patterns
- Add `detectHallucination(text: string, recentResults: ToolResult[]): HallucinationInfo | null` helper
- Add `getCorrectionMessage(hallucination: HallucinationInfo, evidence: string[]): string` helper
- Implement `context` handler — remove hallucinated tool result entries
- Track "last verified success" timestamp to gate success claims

---

## Feature 10: Error Recovery Assistance

### Problem

When the model encounters errors (tracebacks, test failures, tool errors), it often:
- Ignores the error and moves on
- Tries the same thing again without understanding the error
- Gets stuck in a read → try → fail → read loop

The analysis shows 8 Python tracebacks that were never successfully resolved. The model reads the error, reads the code, says "looks correct", tries again, gets the same error.

### Solution

- On `tool_result` for `bash` with non-zero exit code or traceback output:
  - Parse the error/traceback
  - Inject a plain-language explanation of what went wrong
  - Suggest specific debugging steps
- On `turn_end`, if the last tool result was an error and the model's next action is to re-read the same file, inject guidance to analyze the error instead.
- Provide error categorization: syntax error, runtime error, logic error, tool error.

### Validation

- Python tracebacks get plain-language explanations injected
- Model is steered toward error analysis instead of re-reading
- Error categories are correctly identified

### Notes

- Parse common error formats: Python tracebacks, shell errors, TypeScript errors
- Keep explanations concise — small models can't handle long error analysis
- Focus on the most recent/last error line, not the full stack trace

### Actions

- Implement `tool_result` handler for `bash` — detect errors, parse tracebacks
- Add `parseTraceback(output: string): { type: string; file: string; line: number; message: string } | null` helper
- Add `getErrorExplanation(error: ParsedError): string` helper
- Implement `tool_call` handler — detect re-read after error, inject guidance
- Add error history tracking

---

## Feature 11: Directory Read Errors → Directory Listing

### Problem

When the `read` tool is called on a directory path, it returns an `EISDIR` error. Small models don't understand this error and may retry or get confused. The useful response is a directory listing.

### Solution

- On `tool_result` for `read` with `EISDIR` error:
  - Read the directory contents using `readdir`
  - Replace the error result with a formatted directory listing
  - Format as a simple text listing with file types (file/directory indicators)

### Validation

- `read` on a directory returns a formatted listing instead of EISDIR error
- `read` on a file passes through normally
- `read` on non-existent path still returns error

### Notes

- Use `fs/promises` `readdir` with `{ withFileTypes: true }` for type info
- Format: `- filename` for files, `d directory/` for directories
- Include the path being listed in the output header

### Actions

- Implement `tool_result` handler for `read` — detect `EISDIR` error
- Add `directoryListing(path: string): string` helper using `readdir`
- Replace error content with formatted listing
- Set `isError: false` on the result so the model doesn't see it as a failure

---

## Feature 12: Duplicate Read Deduplication

### Problem

The analysis shows a file was read 30 times out of 69 total tool calls (43%). Often the same offset/limit, with no file modifications between reads. The model can't retain file content in context (because massive thinking consumes tokens) and keeps re-reading to "verify" edits.

This wastes tool calls, consumes context, and makes no progress.

### Solution

- Track `read` tool calls with their file path, offset, and limit.
- Track file modifications (via `edit`, `write` tool results).
- When a `read` call is detected that matches a previous read of the same file (same path, same or overlapping offset/limit) with no modifications in between:
  - Block the duplicate `read` call
  - Return the cached result from the first read
  - Inject a synthetic steering message: "You already read this file recently and it hasn't changed. Instead of re-reading, try [specific alternative action]."
- In `context` event, remove duplicate `read` toolResult messages from context (keep only the first one).

### Validation

- Second read of the same file without changes is blocked/cached
- Cached result matches the original read content
- Read after a file modification is allowed (cache invalidated)
- Different files have independent caches

### Notes

- Cache key: `path + offset + limit` (normalize path)
- Invalidate cache on `edit` or `write` success for that path
- The synthetic steering message should suggest a specific alternative (e.g., "try running the test", "try a different fix", "move on to the next task")
- Don't block reads with different offsets/limits (the model might be reading different sections)

### Actions

- Add `Map<string, ReadCacheEntry>` for read cache: `{ content: string; timestamp: number; offset?: number; limit?: number }`
- Add `getReadCacheKey(path: string, offset?: number, limit?: number): string` helper
- Implement `tool_call` handler for `read` — check cache, return cached or allow
- Implement `tool_result` handler for `edit`/`write` — invalidate cache on success
- Implement `context` handler — remove duplicate toolResult entries
- Add `getSteeringMessage(path: string, lastReadTime: number): string` helper with varied messages

---

## Feature 13: Message Sequence Loop Detection and Removal

### Problem

Beyond individual tool call loops, the model gets stuck in sequences of messages that repeat:
- read file → edit fails → read file → edit fails (same pattern)
- read file → "code looks correct" → run test → error → read file (same pattern)

The analysis shows this pattern repeating multiple times. Each cycle consumes context without progress.

### Solution

- Track message sequences (tool calls + assistant text) as a rolling window.
- Detect when a sequence of N messages (configurable, default 3-5) repeats.
- On detection:
  - In `context` event, remove the later occurrences of the repeated sequence (keep the first)
  - Inject synthetic reasoning content to steer the model in a new direction
  - The synthetic content should suggest: trying a different approach, reading documentation, simplifying the problem, or asking for clarification

### Validation

- Repeated 3-message sequences are detected and deduplicated
- Only later occurrences are removed (first is kept)
- Synthetic steering content is injected
- Normal non-repeating sequences pass through

### Notes

- Sequence comparison: normalize tool call params (ignore timestamps, toolCallId)
- For text messages, compare normalized content (whitespace-collapsed)
- Sequence length threshold: 3 messages minimum
- The synthetic content should be varied and context-aware

### Actions

- Add `SequenceDetector` class:
  - `recordMessage(entry: SessionEntry)`
  - `detectRepeatedSequence(): { start: number; end: number; pattern: Entry[] } | null`
- Implement `turn_end` handler — record entries
- Implement `context` handler — remove repeated sequence entries
- Add `getSyntheticSteering(context: string): string` helper with varied messages
- Add sequence window size config

---

## Implementation Architecture

### Extension Structure

```typescript
export default function (pi: ExtensionAPI) {
  // Configuration
  const config = loadConfig();

  // State
  const loopDetector = new LoopDetector(config.loopThreshold);
  const sequenceDetector = new SequenceDetector();
  const readCache = new Map<string, ReadCacheEntry>();
  const editFailureCounts = new Map<string, number>();
  const skillPaths = new Map<string, string>();

  // --- Tool Call Interception ---
  pi.on("tool_call", async (event) => {
    // Feature 1: Block write
    if (event.toolName === "write") return blockWrite(event);
    // Feature 1: Block destructive bash
    if (event.toolName === "bash") return blockDestructiveBash(event);
    // Feature 6: Resolve skill paths
    if (event.toolName === "read") return resolveSkillReadPath(event);
    // Feature 12: Deduplicate reads
    if (event.toolName === "read") return deduplicateRead(event);
    // Feature 8: Loop detection on tool calls
    return detectToolLoop(event);
  });

  // --- Tool Result Modification ---
  pi.on("tool_result", async (event) => {
    // Feature 2: Edit failure tracking
    if (event.toolName === "edit") return handleEditFailure(event);
    // Feature 7: Plain-language hints
    return addFailureHints(event);
    // Feature 11: Directory listing
    if (event.toolName === "read") return handleDirectoryRead(event);
    // Feature 10: Error recovery
    if (event.toolName === "bash") return addErrorRecovery(event);
  });

  // --- Message Correction ---
  pi.on("message_end", async (event) => {
    // Feature 3: Skills listing
    // Feature 4: Tools listing
    // Feature 5: Skill invocation
    // Feature 9: Hallucination correction
    return correctHallucinations(event);
  });

  // --- Context Modification ---
  pi.on("context", async (event) => {
    // Feature 8: Remove loop messages
    // Feature 12: Remove duplicate reads
    // Feature 13: Remove repeated sequences
    return pruneContext(event);
  });

  // --- Turn Tracking ---
  pi.on("turn_end", async (event) => {
    // Feature 8: Record for loop detection
    // Feature 13: Record for sequence detection
    recordTurn(event);
  });

  // --- Session Init ---
  pi.on("session_start", async (_event, ctx) => {
    // Feature 6: Build skill path map
    buildSkillPathMap(ctx);
  });
}
```

### Helper Modules

| Module | Purpose |
|---|---|
| `loop-detector.ts` | LoopDetector class — tool/message/sequence loop detection |
| `sequence-detector.ts` | SequenceDetector class — message sequence pattern matching |
| `cache.ts` | ReadCache class — file read caching and invalidation |
| `patterns.ts` | Regex patterns for detection (bash writes, skill listings, hallucinations) |
| `messages.ts` | Varied warning/steering message generators |
| `error-parser.ts` | Traceback and error parsing utilities |
| `config.ts` | Environment variable loading and validation |

### Key Design Decisions

1. **Non-blocking by default:** Most features augment or steer, not block. Only destructive actions (Feature 1) and hard loops (Feature 8) are blocked.
2. **Varied messages:** All injected messages use randomized selection to prevent model memorization.
3. **Context-aware:** Steering messages reference actual file names, error messages, and recent actions.
4. **Configurable thresholds:** All thresholds are env-variable configurable with sensible defaults.
5. **Independent features:** Each feature can work standalone; no feature depends on another.
6. **Minimal overhead:** Detection runs on `turn_end`/`message_end`, not on every token.

---

## Doom Loop Warning Messages

Varied messages, randomly picked on each breach:

```
Warning: Doom looping detected, let's try another approach.
Warning: Heads up, you're stuck in a doom loop. Try something different.
Warning: Repeated doom loop. Pivot to a new strategy.
Warning: You keep doom looping. Explore a different path.
Warning: Spinning in circles. Try a fresh angle.
Warning: Repeated doom looping. Shift your approach.
Warning: Loop detected. Move on.
Warning: You've looped and repeated messages. Consider the next step.
Warning: Looping on the same messages. Let's try another route.
```
