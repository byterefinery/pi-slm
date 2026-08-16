---
name: pi-0-84-2
description: Pi coding agent CLI (v0.84.2) — minimal, extensible terminal coding agent. Use when running the `pi` CLI — interactive TUI sessions, one-shot `pi -p` prompts, JSON/RPC modes, continuing or forking sessions, choosing models/providers/thinking levels, tool allowlists, installing pi packages, or configuring settings, context files, and environment variables. Covers install, authentication, and the full CLI flag reference.
license: MIT
compatibility: Requires pi 0.84.2+ installed globally (Node.js >= 22.19 or Bun); a provider API key or subscription login
metadata:
  tags:
    - cli
    - coding-agent
    - llm
    - terminal
---

# pi 0.84.2

## Overview

Pi is a minimal terminal coding harness. It adapts to your workflow via TypeScript extensions, skills, prompt templates, themes, and pi packages — without forking the core. It runs in four modes:

- **Interactive** (default) — TUI with editor, slash commands, and session tree
- **Print** (`-p`, `--print`) — one-shot prompt, prints the response and exits
- **JSON** (`--mode json`) — all session events as JSON lines for tooling
- **RPC** (`--mode rpc`) — JSONL command protocol over stdin/stdout

By default pi gives the model four tools: `read`, `write`, `edit`, `bash`. Additional built-in read-only tools (`grep`, `find`, `ls`) are available through tool options. Pi intentionally ships **no** MCP, sub-agents, permission popups, plan mode, to-dos, or background bash — build or install those workflows as extensions/packages, or use tmux and containers.

This skill documents pi 0.84.2.

## Install & Authenticate

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
# or
curl -fsSL https://pi.dev/install.sh | sh
```

Authenticate with an API key environment variable or `/login` inside interactive mode:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, GEMINI_API_KEY, ...
pi
```

Or use a subscription: start `pi`, run `/login`, pick a provider (Claude Pro/Max, ChatGPT Plus/Pro (Codex), GitHub Copilot, xAI, OpenRouter, Radius). Credentials store in `~/.pi/agent/auth.json` (0600 perms) and take priority over env vars.

## Usage

```bash
pi [options] [@files...] [messages...]
```

### Modes

| Flag | Description |
|------|-------------|
| (default) | Interactive TUI |
| `-p`, `--print` | Print response and exit (also merges piped stdin into the prompt) |
| `--mode json` | Output all session events as JSON lines |
| `--mode rpc` | JSONL command protocol over stdin/stdout |
| `--export <in> [out]` | Export a session to HTML |

```bash
pi -p "Summarize this codebase"
cat README.md | pi -p "Summarize this text"
```

### Model Options

| Option | Description |
|--------|-------------|
| `--provider <name>` | Provider (`anthropic`, `openai`, `google`, ...) |
| `--model <pattern>` | Model pattern or ID; supports `provider/id` and optional `:<thinking>` shorthand |
| `--api-key <key>` | API key (overrides environment variables) |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `--models <patterns>` | Comma-separated patterns for Ctrl+P cycling |
| `--list-models [search]` | List available models |

### Session Options

| Option | Description |
|--------|-------------|
| `-c`, `--continue` | Continue the most recent session |
| `-r`, `--resume` | Browse and select a past session |
| `--session <path\|id>` | Use a specific session file or partial UUID |
| `--fork <path\|id>` | Fork a session file or partial UUID into a new session |
| `--session-dir <dir>` | Custom session storage directory |
| `--no-session` | Ephemeral mode (don't save) |
| `--name <name>`, `-n <name>` | Set session display name at startup |

### Tool Options

| Option | Description |
|--------|-------------|
| `--tools <list>`, `-t <list>` | Allowlist specific tools (built-in, extension, custom) |
| `--exclude-tools <list>`, `-xt <list>` | Disable specific tools |
| `--no-builtin-tools`, `-nbt` | Disable built-in tools, keep extension/custom tools |
| `--no-tools`, `-nt` | Disable all tools |

Built-in tools: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`.

### Resource Options

| Option | Description |
|--------|-------------|
| `-e`, `--extension <source>` | Load extension from path, npm, or git (repeatable) |
| `--no-extensions` | Disable extension discovery |
| `--skill <path>` | Load a skill (repeatable) |
| `--no-skills` | Disable skill discovery |
| `--prompt-template <path>` | Load a prompt template (repeatable) |
| `--no-prompt-templates` | Disable prompt template discovery |
| `--theme <path>` | Load a theme (repeatable) |
| `--no-themes` | Disable theme discovery |
| `--no-context-files`, `-nc` | Disable `AGENTS.md`/`CLAUDE.md` discovery |

Combine `--no-*` with explicit flags to load exactly what you need: `pi --no-extensions -e ./my-extension.ts`.

### Other Options

| Option | Description |
|--------|-------------|
| `--system-prompt <text>` | Replace the default prompt (context files and skills still appended) |
| `--append-system-prompt <text>` | Append to the system prompt |
| `--tui-mode <mode>` | `regular` (default) or experimental `fullscreen` |
| `--use-theme <name>` | Set the initial theme for this run without changing settings |
| `--offline` | Disable all startup network operations |
| `--verbose` | Force verbose startup |
| `-a`, `--approve` | Trust project-local files for this run |
| `-na`, `--no-approve` | Ignore project-local files for this run |
| `-h`, `--help` | Show help |
| `-v`, `--version` | Show version |

### File Arguments

Prefix files with `@` to include them in the message:

```bash
pi @prompt.md "Answer this"
pi -p @screenshot.png "What's in this image?"
```

### Package Commands

```bash
pi install <source> [-l]     # npm: / git: / https:// / ssh:// sources; -l = project-local
pi remove <source> [-l]      # pi uninstall is an alias
pi update [--all|--extensions|--models|--self [ --force ]]
pi list                      # List installed packages
pi config                    # Enable/disable extensions, skills, prompts, themes
```

Packages install to `~/.pi/agent/git|npm/` globally, or `.pi/git|npm/` with `-l`. Pinned git `@ref` packages are skipped by `pi update` — reinstall at the new ref to move them.

### Auth Commands

```bash
pi auth check --provider <p> [--model <m>] [--json] [--credentials] [--no-refresh]
pi auth print-api-key --provider <p> [--model <m>]
pi auth print-bearer-token --provider <p> [--model <m>] [--min-expiry <duration>]
```

Requires at least one of `--provider` or `--model`. `auth check` refreshes expired OAuth credentials by default; `--no-refresh` prevents that; `--credentials` emits the credential itself.

### Examples

```bash
pi "List all .ts files in src/"                 # interactive with initial prompt
pi -p "Summarize this codebase"                 # one-shot
pi --name "release audit" -p "Audit this repo"  # named one-shot
pi --model openai/gpt-4o "Help me refactor"     # provider-prefixed model
pi --model sonnet:high "Solve this"             # thinking-level shorthand
pi --models "claude-*,gpt-4o"                   # limit Ctrl+P cycling
pi -t read,grep,find,ls -p "Review the code"    # read-only mode
pi -xt ask_question                             # disable one tool, keep the rest
pi -c                                           # continue most recent session
```

## Gotchas

- **Non-interactive modes never show a trust prompt.** In `-p`, `--mode json`, and `--mode rpc`, project trust falls back to `defaultProjectTrust` in global settings (`ask`/`never` ignore project-local resources; `always` trusts them). Override per run with `-a`/`-na`. `pi update` never prompts.
- **Install with `--ignore-scripts`** — pi needs no npm lifecycle scripts; this keeps installs safe and fast.
- **`-p` merges piped stdin into the prompt** — `cat file | pi -p "..."` appends file content to the message rather than treating it separately.
- **Sessions are per working directory.** They auto-save under `~/.pi/agent/sessions/` organized by cwd, so `pi -c` resumes the most recent session *for that directory*. Use `/session` to print the current ID before reusing it with `--session <id>`.
- **Compaction is lossy.** Auto-compaction summarizes older messages to free context; full history stays in the session JSONL — use `/tree` to revisit earlier points.
- **Context files concatenate.** `AGENTS.md`/`CLAUDE.md` load from `~/.pi/agent/`, every parent directory, and cwd. `AGENTS.override.md` replaces both in its directory. After edits, run `/reload` or restart.
- **No built-in sandbox.** Pi runs with the permissions of the user that launched it. For stronger boundaries, containerize it (Docker, Gondolin micro-VM, or OpenShell patterns).
- **Child-process markers.** The CLI sets `AI_AGENT=pi` and `PI_CODING_AGENT=true`; bash-tool commands additionally get `PI_SESSION_ID`, `PI_SESSION_FILE`, `PI_PROVIDER`, `PI_MODEL`, `PI_REASONING_LEVEL`. Detect pi via these variables, not the system prompt.
- **Startup network calls.** Pi checks `pi.dev` for updates and may send an anonymous install/update ping. `PI_OFFLINE=1` or `--offline` disables everything; `PI_SKIP_VERSION_CHECK=1` disables only the version check.
- **`!cmd` vs `!!cmd` (interactive):** `!` sends command output to the model, `!!` does not. Neither runs through the bash tool, so no `PI_*` session variables are injected.

## References

- [01-interactive-mode](references/01-interactive-mode.md) — editor features, slash commands, keyboard shortcuts, message queue
- [02-sessions](references/02-sessions.md) — session storage, branching with `/tree`, `/fork`/`/clone`, compaction
- [03-providers-models](references/03-providers-models.md) — authentication, API keys, subscriptions, custom providers
- [04-settings-environment](references/04-settings-environment.md) — settings.json options, environment variables, project trust
- [05-programmatic-modes](references/05-programmatic-modes.md) — JSON event stream, RPC protocol, SDK
