# SLM Extension Specification

Known issues of reliability of small language models (SLMs), and the feature list
of the pi extension (`src/slm.ts`, loaded with `pi -e src/slm.ts`) that mitigates
them. Target harness: pi 0.84.2.

- Applies in all modes: interactive TUI, `-p`, `--mode json`, `--mode rpc`.
- Primary target/test model: local `LiquidAI/LFM2.5-2.6B` (an SLM).
- Validation runs an isolated `pi` instance in a random temp dir under `/tmp`.
  The extension keeps per-process (per-session) in-memory state only.
- The extension relies on pi's internal TypeScript API (extension events,
  session manager, built-in tool factories, resource/model state). It never
  re-parses CLI arguments, never re-implements resource discovery, and adds
  no dependencies (single `.ts` file).

## Agreed design decisions

- Destructive `write` = a `write` that overwrites an existing file completely.
  When intercepted, a small change is converted into a deterministic `edit`
  tool call; a huge change falls back to plain `write`.
- `bash` bypass interception (redirects / `tee` / `sed -i` workarounds) is
  **not** implemented: `write` is no longer "blocked" (it is converted or
  allowed), and bypass attempts are rare edge cases.
- `edits[].oldText` values that don't exist in the file are dropped from the
  call. If none survive, report that an empty edit was sent, remove the
  assistant message that caused the call from the active session log, and
  replace it with a hint message.
- The two startup synthetic messages are custom messages rendered as **user
  role** in the LLM context; they are injected before the first user request
  is processed and are visible in the TUI log and the session JSONL.
- **No synthetic reasoning content is generated at all.** User-role messages
  cannot carry reasoning: the LFM2.5-2.6B and Qwen3.8-27B chat templates
  render thinking only inside assistant turns, and pi can only inject
  user-role (custom) messages. (Decision: skip.)
- Skill YAML (startup message) and skill hints (use / error) use **absolute
  paths**; `path` is a required field. SLMs handle absolute paths better.
- Tool/skill error hints are appended to the failing tool result (kept
  attached in log, JSONL, and context), not sent as separate messages.
- `bash` always carries `timeout`: 60 s base, +60 s step after each timeout
  failure, capped at 300 s. Built-in `bash` tool only.

## Feature List

1. **Destructive-write protection** — intercept `write` on existing files;
   convert small changes into a deterministic `edit` tool call (exact `path`,
   `edits[].oldText`, `edits[].newText`); fall back to `write` when the change
   is huge or `edit` cannot satisfy it.
2. **`edit` oldText validation** — keep only `edits[]` whose `oldText` exists
   in the file; if none exist, report that an empty edit was sent and remove
   the assistant message that caused the call, replacing it with a hint.
3. **Startup context messages** — on a new session, insert the first two
   synthetic (token-compact) messages `Available skills:` and
   `Available tools:` as YAML (name, description, absolute paths; tool
   parameter schemas with `type:` only where it differs from `string`,
   `strict: false` by default).
4. **Skill structure hint** — when a skill is used (first read inside its
   directory) and was not used before, append a YAML hint of that skill
   (name, description, absolute `path`, absolute `references:` / `scripts:`,
   sections omitted when absent). Reading skill files is a regular read.
5. **Mandatory `bash` timeout** — every built-in `bash` call carries
   `timeout` (60 s base; +60 s step after each timeout failure; 300 s cap).
6. **Tool invocation error hint** — on every tool failure, append the failed
   tool's YAML definition so the model can recall the tool contract and retry;
   for an unknown tool name, append the list of available tools.
7. **Skill invocation error hint** — on a failed read that maps to a known
   skill, append that skill's YAML hint; for an unknown skill name, append the
   available-skills list.
8. **`read` on directory → listing** — convert `EISDIR` into a directory
   listing (`ls -F` of the path) instead of an error.

Example of the available-skills YAML (startup message):

```
Available skills:
- name: git
  description: Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.
  path: /abs/path/to/.agents/skills/git/SKILL.md
- name: webfetch
  description: Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.
  path: /abs/path/to/.agents/skills/webfetch/SKILL.md
...
```

Example of the available-tools YAML (startup message):

```
Available tools:
- name: read
  description: Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.
  parameters:
    required:
      - path
    properties:
      path:
        description: Path to the file to read (relative or absolute)
      offset:
        type: number
        description: Line number to start reading from (1-indexed)
      limit:
        type: number
        description: Maximum number of lines to read
  strict: false
...
```

Default tool property type is `type: string`, so only use `type: ...` where it
differs (e.g. `type: number`). Default tool is `strict: false`, so only include
`strict: true` where otherwise stated.

Example of a skill structure hint (features 4/7):

```
- name: git
  description: Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.
  path: /abs/path/to/.agents/skills/git/SKILL.md
  references:
    - /abs/path/to/.agents/skills/git/references/01-command-reference.md
    - /abs/path/to/.agents/skills/git/references/02-worktrees.md
    - ...
  scripts:
    - /abs/path/to/.agents/skills/git/scripts/script1.sh
    - ...
```

If no references are used in the skill do not output `references:` in the YAML.
If no scripts are used in the skill do not output `scripts:` in the YAML.

---

## 1. Destructive-write protection (write → edit conversion)

### Problem
SLMs call `write` on existing files with a full new content, silently
destroying file parts they did not intend to change. `write` "creates the
file if it doesn't exist, overwrites if it does" — a small model cannot
reliably express a small change with a full-content tool.

### Solution
Override the built-in `write` tool: register a same-named tool (identical
name, description, parameter schema) whose `execute` delegates to the
original implementation (exported `createWriteTool()`):

- Target file does not exist → delegate to the original `write` (new-file
  creation stays normal).
- Target file exists:
  - proposed content == existing content → no-op success result;
  - otherwise compute a simple line diff (common prefix/suffix lines, LCS
    over the middle with a size guard). If the change is **small**, build a
    deterministic `edit` call (`{path, edits[]}`, each hunk's `oldText` is
    the exact existing text, `newText` the replacement) and execute it via
    the original implementation (exported `createEditTool()`);
  - otherwise (huge change) → delegate to the original `write`.

"Small" (tunable constants): at most 5 changed hunks, at most 60 total
changed lines (added + removed), at most 4000 lines on each side, and every
hunk's `oldText` occurs exactly once in the file. Any violation — including a
non-unique `oldText`, i.e. `edit` cannot satisfy the requirement — falls back
to plain `write` (a real full overwrite is allowed).

`bash` bypass interception is intentionally not implemented (see design
decisions): `write` is converted or allowed, never "blocked", and bypass
attempts are rare edge cases.

### Validation
In a temp dir with a multi-section file: prompt the local LFM model to change
one line. Inspect the session JSONL: the change was applied through the edit
path (exact hunk, `oldText` matches the file byte-for-byte), unrelated file
content unchanged. Then prompt a complete rewrite of the file: the write path
is used. Unit tests for the pure diff→edits conversion: identical content,
single-line change, multi-hunk change, non-unique hunk text, huge diff,
file-creation (no existing file).

### Notes
- The conversion is deterministic: the extension, not the model, builds the
  `edit` call (correct `path`, `edits[].oldText`, `edits[].newText`).
- The model still sees and calls `write`; the safe `edit` mechanism is applied
  underneath, and the model is steered toward `edit` in follow-up calls.
- Delegation to the original tool implementations preserves pi's behavior
  (file-mutation queueing, renderers, truncation).

---

## 2. `edit` oldText validation + empty-edit recovery

### Problem
SLMs frequently emit `edits[].oldText` values that don't exist in the file
(whitespace drift, stale memory of the content). After such a failure they
get stuck repeating the same bad call.

### Solution
`tool_call` hook on `edit`:

- Read the target file (path resolved against the session cwd). If the file
  cannot be read, let the original execution happen — its error triggers the
  tool-error hint (feature 6).
- Keep only edits whose `oldText` exists in the file content; mutate
  `event.input.edits` in place (pi supports in-place argument mutation before
  execution).
- Some edits survive → the reduced call executes as-is.
- No edits survive → block the call with the reason "empty edit: none of the
  `edits[].oldText` values matched `<path>`; re-read the file and retry with
  exact text", and at `turn_end` remove the assistant message that caused the
  call from the active session log: sessions are append-only, so use
  `sessionManager.branch()` back to the entry preceding that assistant
  message (the failed call and its result leave the active branch, mirroring
  `/tree` semantics), then append a replacement hint message (custom message,
  displayed) telling the model to `read <path>` and retry `edit` with exact
  text.

### Validation
Temp dir, file with known content; engineer a call whose `oldText` is stale
(e.g. prompt the model to edit from memory, or mutate the file between
`read` and `edit`). Partial match → only the surviving edits are applied.
Zero match → the "empty edit" reason appears, the bad assistant message is off
the active branch (visible in `/tree`), the hint message is present, and the
model re-reads the file and retries.

### Notes
- "Exists" = at least one occurrence. Non-unique `oldText` is deliberately
  kept: the built-in `edit` tool reports the uniqueness error, which triggers
  the tool-error hint (feature 6) and teaches the model to add context.
- The zero-survivor case is rare. Branch-based removal never rewrites the
  JSONL file.
- Fallback if mid-run branching proves to desync the agent loop (to be
  verified at runtime): block-with-reason only (no removal), same hint text.

---

## 3. Startup context messages (available skills / available tools)

### Problem
SLMs under-use skills and misuse tools: the default system prompt lists
skills without their file layout or absolute paths, and tool parameter
schemas live far from the conversation.

### Solution
On `session_start` of a fresh session (no session entries yet — covers
startup and `/new`; skipped on resume/fork/reload), inject two synthetic
messages via `pi.sendMessage()` (custom messages, `display: true`):

1. `Available skills:` — YAML of the active skills (from
   `ctx.getSystemPromptOptions().skills`): `name`, `description`, `path`
   (absolute path to SKILL.md).
2. `Available tools:` — YAML of the active tools (intersection of
   `pi.getActiveTools()` and `pi.getAllTools()`): `name`, `description`,
   `parameters` (`required`, `properties` with `description`, `type:` emitted
   only where it differs from `string`), `strict: false` (`strict: true` only
   if the provider payload says so — captured once from
   `before_provider_request`).

The YAML is hand-rolled (no dependency): compact, one entry per skill/tool.

`session_start` fires before the first user prompt is processed, so both
messages precede the user request in context and log. They persist as custom
message entries (visible in the TUI and the JSONL) and render as **user role**
in the LLM context (`convertToLlm` maps custom → user). No synthetic reasoning
is attached (see design decisions).

### Validation
Start `pi -e src/slm.ts` in a fresh temp dir against the local LFM model.
The first two session entries are the two custom messages. The skills YAML
lists every discovered skill with an absolute `path`. The tools YAML matches
the active tool set and their schemas (spot-check `edit`'s nested `edits`
array schema and `read`'s numeric `offset`/`limit`). Resume an existing
session → no re-injection. Same behavior in `-p` mode.

### Notes
- `path` is absolute by design (SLMs handle absolute paths better).
- pi's default system prompt keeps its own skills section; these messages are
  additive in-conversation context.
- Injection gate is "session has no entries", which is robust across
  startup/new/reload/resume reasons.

---

## 4. Skill structure hint on first use

### Problem
When a skill is loaded, the model sees only the SKILL.md content; it doesn't
know the skill's file layout (references/scripts) and must guess paths.

### Solution
Track the active skills (name → baseDir) from
`ctx.getSystemPromptOptions().skills` at session start. Reading skill files is
a regular read — nothing special about the read itself (references/scripts are
loaded on demand by the model). In the `tool_result` hook on `read`: if the
read path is inside a known skill's `baseDir` and that skill was not hinted
before in this session, append a YAML hint to the tool result: `name`,
`description`, `path` (absolute), `references:` (absolute paths of the files
under `references/`, omitted if the directory is absent/empty), `scripts:`
(same for `scripts/`). Mark the skill as hinted (per-session in-memory set).
The skill load (SKILL.md content) happens with the read itself; the hint
arrives with the result, so by the next message the model knows the full
structure.

### Validation
Temp dir with a skill that has `references/` and a skill with `scripts/`
(e.g. `git` and `webfetch`). Have the model read `<skill>/SKILL.md`. The read
result contains the YAML hint with absolute `references:`/`scripts:` entries.
A second skill-file read in the same session produces no second hint. A skill
without `references/`/`scripts/` yields a hint without those sections.

### Notes
- Trigger: first read of any file inside the skill directory (covers
  SKILL.md and direct on-demand reference reads).
- `references:`/`scripts:` are discovered by listing the subdirectories at
  hint time; entries are absolute paths.

---

## 5. Mandatory `bash` timeout with escalation

### Problem
SLMs often omit the `bash` `timeout` argument (the schema marks it optional);
an unbounded command can hang the session.

### Solution
`tool_call` hook on `bash`: if `timeout` is missing or invalid, set it in
place. Base 60 s. When a bash result reports a timeout failure (error text
`timeout:<n>`), increase the next default by 60 s: 60 → 120 → 180 → 240 →
300 (capped at 300 s and held). State is per-session in-memory. Applies to
the built-in `bash` tool only.

### Validation
Temp dir. Prompt a command that sleeps longer than 60 s. The first call
executes with `timeout: 60` and fails with a timeout error; the next `bash`
call in the session executes with `timeout: 120` (visible in the session
JSONL / tool call args). A model-supplied explicit timeout is respected as-is.

### Notes
- Argument mutation happens before execution, so the executed call always
  carries a timeout. Whether the persisted assistant entry reflects the
  mutation depends on pi's message-persistence timing (verify at runtime;
  cosmetic only).
- The 300 s cap matches the usual 60 s step ladder (60…300).

---

## 6. Tool invocation error → tool definition hint

### Problem
When a tool call fails (bad arguments, unknown tool, missing file), SLMs
often don't recall the tool's definition and repeat the same mistake.

### Solution
`tool_result` hook: if the result is an error, find the tool in the set of
available tools (`pi.getAllTools()`):

- known tool → append to the result content the tool's YAML definition (same
  compact format as feature 3: `name`, `description`, `parameters`,
  `strict`) so the model can recall the contract and retry;
- unknown tool name → append "unknown tool `<name>`; available tools:" with
  the compact list of active tools (name + one-line description).

Every tool failure gets the hint — including "file not found" reads (the file
may legitimately have disappeared in the background; the reminder is still
useful). The hint is appended to the tool result, not sent as a separate
message, so it stays attached in the log, the JSONL, and the context.

### Validation
Temp dir. Cause each kind of failure and check the result content:
call a nonexistent tool name; call `read` with a malformed argument shape;
`read` a file that was deleted between `ls` and `read`. The result contains
the tool YAML (known tools) or the available-tools list (unknown tool).

### Notes
- Repeats on every failure of the same tool (no de-duplication) — repetition
  is beneficial for SLMs.
- Hints also fire for errors produced by the blocked empty-edit call
  (feature 2) and by write/edit conversion fallbacks — intended: the model is
  reminded of the tool contract.

---

## 7. Skill invocation error → skill usage hint

### Problem
SLMs get skill file paths wrong (relative instead of absolute, wrong skill
name) and fail to load the skill.

### Solution
`tool_result` hook on failed `read`: if the failed path maps to a known
skill (the absolute path starts with the skill's `baseDir`, or the path
contains `skills/<name>` matching a known skill name), append that skill's
YAML hint (same shape as feature 4: `name`, `description`, absolute `path`,
absolute `references:`/`scripts:` omitted when absent) so the model can
correct the path and retry. If the path references an unknown skill name under
a skills directory, append the available-skills YAML (`name`, `description`,
absolute `path`) instead.

### Validation
Temp dir with skills. Have the model read a wrong relative path
(`skills/git/references/01.md`) and a nonexistent skill
(`<skills dir>/nonexistent/SKILL.md`). The first result contains the `git`
skill hint with absolute paths; the second contains the available-skills
list.

### Notes
- Skill reads remain regular reads (feature 4); this feature only reacts to
  failures.
- The hint and the feature-4 hint share the same YAML renderer and the same
  per-session "hinted" set (a failed load that produced a hint also counts
  as "used", preventing hint loops).

---

## 8. `read` on directory → directory listing

### Problem
`read` on a directory fails with `EISDIR`; the model must discover on its
own that `ls` is the right tool.

### Solution
`tool_result` hook on `read`: if the result is an error, `stat` the requested
path (resolved against the session cwd). If it is a directory, replace the
result with the directory listing produced by `pi.exec("ls", ["-F", <abs
path>])` and mark the result as non-error — the read is effectively converted
into `ls` of the path.

### Validation
Temp dir with files and subdirectories. `read` a directory. The result is a
listing (subdirectories marked with `/`), `isError` is false, and the model
continues by reading files. If `ls` itself fails, the original error result
is kept.

### Notes
- `ls -F` marks directories (`/`) so the model can distinguish file entries.
- The hook keys off `stat` on the requested path, not on the error text, so
  it is robust to wording changes in the read tool.

---

## Testing

- **Unit** — the pure logic (line diff → `edit` call, YAML emitter,
  `oldText` filtering, skill-path matching) lives in `src/slm.ts` as pure
  functions and is tested directly with Node (no pi, no network).
- **E2E** — `tests/e2e.mjs` (self-contained, run `node tests/e2e.mjs`,
  ~5-10 min): spawns isolated `pi -p` runs in a random temp dir
  (`pi --no-extensions -a -e src/slm.ts --model LiquidAI/LFM2.5-2.6B:high`),
  one fresh session per scenario, and asserts on the persisted session
  JSONL plus on-disk side effects:
  - `t1a/t1b/t1c` — F1: write on new file (plain write), small change
    (converted to the built-in edit, "Successfully replaced"), identical
    content (no-op message, file untouched).
  - `t2` — F2: fabricated `oldText` → blocked "Empty edit"; asserts the
    removed-pair/grafted-response/hint branch shape (and accepts the guard
    path where the model recovered, keeping the history).
  - `t4/t7a/t7b` — F4/F7: skill-structure hint on first read inside a known
    skill dir; skill hint on failed read in a known dir; available-skills
    list for an unknown skill name. (A read-only planted skill: a wandering
    model once rewrote SKILL.md and corrupted the frontmatter.)
  - `t5a/t5b` — F5: no-timeout `sleep 75` killed at the base 60 s; after
    that failure a retry completes at the escalated 120 s default.
  - `t6` — F6: tool failure gets the tool's YAML definition appended.
  - `t8` — F8: `read` on a directory → non-error `ls -F` listing.
  - Every session is also checked for F3's two startup custom messages
    (skills/tools YAML, absolute paths, before the first message).
  The test uses an isolated `PI_CODING_AGENT_DIR` inside the temp dir
  (models.json copied from the user's config), so nothing is written
  outside the temp dir. LLM-dependent scenarios are retried per attempt
  (fresh session each) and print a JSONL diagnostic dump on failure.
- No writes outside the temp dir; `.pi/`, `misc/`, and gitignored paths are
  not touched.

## Status

Spec finalized (2026-08-16). `src/slm.ts` implemented; e2e-validated with
`node tests/e2e.mjs` (11/11 scenarios) against local `LiquidAI/LFM2.5-2.6B`.
Known environment caveat: the LLM endpoint intermittently stalls for minutes;
retried attempts absorb that, but a bad day can make the suite slow.
