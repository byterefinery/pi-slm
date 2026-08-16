# Sessions

Sessions are stored as JSONL files with a tree structure. Each entry has an `id` and `parentId`, enabling in-place branching without creating new files.

## Storage & CLI

Sessions auto-save to `~/.pi/agent/sessions/`, organized by working directory.

```bash
pi -c                  # Continue most recent session (for this cwd)
pi -r                  # Browse and select from past sessions
pi --no-session        # Ephemeral mode (don't save)
pi --name "my task"    # Set session display name at startup
pi --session <path|id> # Use specific session file or partial UUID
pi --fork <path|id>    # Fork a session file or partial UUID into a new session
pi --session-dir <dir> # Custom session storage directory
```

Use `/session` in interactive mode to see the current session file and ID before reusing it with `--session <id>` or `--fork <id>`.

## The Session Picker

`/resume` and `pi -r` open the same interactive picker for the current project. Type to search; toggle path display with Ctrl+P, sort mode with Ctrl+S, named-only filter with Ctrl+N; rename with Ctrl+R; delete with Ctrl+D (uses the `trash` CLI when available).

## Naming

```text
/name Refactor auth module
```

or at startup: `pi --name "CI audit" -p "Review this build failure"`. Named sessions are easier to find in the picker.

## Branching

### `/tree`

Navigate the session tree in-place: select any previous point, continue from there, and switch between branches. All history stays in one file.

- Search by typing; fold/unfold and jump between branches with Ctrl+←/Ctrl+→ (Alt+←/Alt+→); page with ←/→
- Filter modes (Ctrl+O): default → no-tools → user-only → labeled-only → all
- Ctrl+X copies the selected message
- Shift+L labels entries as bookmarks; Shift+T toggles label timestamps

### `/fork`

Create a new session file from a previous user message on the active branch. Opens a selector, copies the active path up to that point, and places the selected prompt in the editor for modification.

### `/clone`

Duplicate the current active branch into a new session file at the current position. Keeps the full active-path history, opens with an empty editor.

### `--fork <path|id>` (CLI)

Fork an existing session file or partial session UUID directly from the CLI into a new session in the current project.

## Compaction

Long sessions can exhaust the context window. Compaction summarizes older messages while keeping recent ones. It is **lossy** — the full history remains in the JSONL file; use `/tree` to revisit.

- **Manual:** `/compact` or `/compact <custom instructions>`
- **Automatic:** enabled by default. Triggers when `contextTokens > contextWindow - reserveTokens` (default `reserveTokens` 16384) — recovers and retries on context overflow, or proactively when approaching the limit.

How it works: walk backwards from the newest message until `keepRecentTokens` (default 20000) is reached, summarize the older span with a structured LLM summary (iterating over the previous summary on repeated compactions), append a compaction entry, and rebuild context from the summary plus messages after the kept boundary. Cut points never land on tool results (they must stay with their tool call).

Settings: `compaction.enabled`, `compaction.reserveTokens`, `compaction.keepRecentTokens` in settings.json.

## Export & Share

- `/export [file]` — session to HTML (or JSONL)
- `/share` — upload as a private GitHub gist with a shareable HTML link
- `/import <file>` — import and resume a session from JSONL
- `pi --export <in> [out]` — export from the CLI
