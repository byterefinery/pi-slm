# Interactive Mode

The interactive TUI has four areas: a startup header (shortcuts, loaded context files, skills, extensions), the message stream, the editor (border color indicates thinking level), and a footer (cwd, session name, token/cache usage, cost, context usage, current model).

## Editor

| Feature | How |
|---------|-----|
| File reference | Type `@` to fuzzy-search project files |
| Path completion | Tab |
| Multi-line | Shift+Enter (Ctrl+Enter on Windows Terminal) |
| External editor | Ctrl+G opens `externalEditor`, `$VISUAL`, `$EDITOR`, Notepad on Windows, or `nano` |
| Clipboard | Ctrl+V pastes an image or text (Alt+V on Windows); drag images onto the terminal |
| Bash command | `!command` runs and sends output to the model |
| Hidden bash | `!!command` runs without sending output |

## Slash Commands

Type `/` in the editor. Extensions register custom commands; skills appear as `/skill:name`; prompt templates expand via `/templatename`.

| Command | Description |
|---------|-------------|
| `/login`, `/logout` | Manage provider credentials |
| `/llama` | Download, load, unload llama.cpp router models |
| `/model` | Switch models |
| `/scoped-models` | Enable/disable models for Ctrl+P cycling |
| `/settings` | Thinking level, theme, message delivery, transport |
| `/resume` | Pick from previous sessions |
| `/new` | Start a new session |
| `/name <name>` | Set session display name |
| `/session` | Show session file, ID, messages, tokens, cost |
| `/tree` | Jump to any point in the session and continue from there |
| `/trust` | Save project trust decision (restart required) |
| `/fork` | Create a new session from a previous user message |
| `/clone` | Duplicate the current active branch into a new session |
| `/compact [prompt]` | Manually compact context, optional custom instructions |
| `/copy` | Copy last assistant message to clipboard |
| `/export [file]` | Export session to HTML or JSONL |
| `/import <file>` | Import and resume a session from JSONL |
| `/share` | Upload as private GitHub gist with shareable HTML link |
| `/reload` | Reload keybindings, extensions, skills, prompts, themes, context files |
| `/hotkeys` | Show all keyboard shortcuts |
| `/changelog` | Display version history |
| `/quit` | Quit pi |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+C | Clear editor |
| Ctrl+C twice | Quit |
| Escape | Cancel/abort |
| Escape twice | Open `/tree` |
| Ctrl+L | Open model selector |
| Ctrl+P / Shift+Ctrl+P | Cycle scoped models forward/backward |
| Shift+Tab | Cycle thinking level |
| Ctrl+O | Collapse/expand tool output |
| Ctrl+T | Collapse/expand thinking blocks |
| Ctrl+X | Copy the last assistant message |

All bindings are customizable in `~/.pi/agent/keybindings.json` (namespaced ids like `tui.input.newLine`, `tui.editor.deleteWordBackward`, format `modifier+key`); run `/reload` after editing.

## Message Queue

Submit messages while the agent is working:

- **Enter** queues a *steering* message — delivered after the current assistant turn finishes its tool calls
- **Alt+Enter** queues a *follow-up* message — delivered only after the agent finishes all work
- **Escape** aborts and restores queued messages to the editor
- **Alt+Up** retrieves queued messages back to the editor

Delivery modes are configurable via `steeringMode`/`followUpMode` in settings: `"one-at-a-time"` (default) or `"all"`. On Windows Terminal, `Alt+Enter` is fullscreen by default — remap it so pi receives the shortcut.

## TUI Modes

`regular` (default) uses the main screen with terminal-owned scrollback. Experimental `fullscreen` (`--tui-mode fullscreen`) scrolls the transcript inside the terminal viewport while the editor/status/footer stay fixed; mouse/trackpad scrolls the region under the pointer, `Ctrl+Shift+F` searches the transcript, `Home`/`End` jump to the top/bottom.
