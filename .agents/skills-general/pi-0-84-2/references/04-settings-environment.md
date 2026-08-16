# Settings & Environment

## Settings Files

JSON settings; project overrides global. Edit directly or use `/settings`.

| Location | Scope |
|----------|-------|
| `~/.pi/agent/settings.json` | Global (all projects) |
| `.pi/settings.json` | Project (overrides global) |

## Key Settings

### Model & Thinking

| Setting | Default | Description |
|---------|---------|-------------|
| `defaultProvider` | - | e.g. `"anthropic"`, `"openai"` |
| `defaultModel` | - | Default model ID |
| `defaultThinkingLevel` | - | `off`...`max` |
| `hideThinkingBlock` | `false` | Hide thinking blocks |
| `thinkingBudgets` | - | Custom token budgets per thinking level |

### UI & Display

| Setting | Default | Description |
|---------|---------|-------------|
| `theme` | `"dark"` | `"dark"`, `"light"`, or custom |
| `externalEditor` | `$VISUAL`/`$EDITOR`/Notepad/`nano` | Ctrl+G editor; use `"code --wait"` for VS Code |
| `quietStartup` | `false` | Hide startup header |
| `defaultProjectTrust` | `"ask"` | Fallback trust: `ask`, `always`, `never` (global only) |
| `doubleEscapeAction` | `"tree"` | Double-Escape: `tree`, `fork`, `none` |
| `tuiMode` | `"regular"` | `regular` or experimental `fullscreen` |
| `enableInstallTelemetry` | `true` | Anonymous install/update ping (does not control update checks) |
| `steeringMode` / `followUpMode` | `"one-at-a-time"` | Queued-message delivery: `one-at-a-time` or `all` |
| `transport` | - | Provider transport preference: `sse`, `websocket`, `auto` |

### Compaction & Network

| Setting | Default | Description |
|---------|---------|-------------|
| `compaction.enabled` | `true` | Auto-compaction |
| `compaction.reserveTokens` | `16384` | Tokens reserved for the response |
| `compaction.keepRecentTokens` | `20000` | Recent tokens kept unsummarized |
| `httpProxy` | - | Applied as `HTTP_PROXY`/`HTTPS_PROXY` (global only) |

Keybindings live in `~/.pi/agent/keybindings.json`.

## Project Trust

On interactive startup, pi asks before trusting a project folder that contains project-local settings/resources or `.agents/skills` and has no saved decision (checked in `~/.pi/agent/trust.json` for the folder or a parent). Trusting allows loading `.pi/settings.json` and `.pi` resources, installing missing project packages, and executing project extensions.

- **Before** the decision, only context files, user/global extensions, and CLI `-e` extensions load.
- **Non-interactive modes** (`-p`, `--mode json`, `--mode rpc`) never prompt; they use `defaultProjectTrust` (`ask`/`never` ignore project resources, `always` trusts). Override one run with `--approve`/`-a` or `--no-approve`/`-na`.
- `/trust` saves a decision for future sessions (writes `trust.json` only; restart pi to apply).
- `pi config` and package commands use the same flow except `pi update`, which never prompts.

## Environment Variables

### Process configuration (read by pi)

| Variable | Description |
|----------|-------------|
| `PI_CODING_AGENT_DIR` | Override config dir (default `~/.pi/agent`) |
| `PI_CODING_AGENT_SESSION_DIR` | Override session storage (overridden by `--session-dir`) |
| `PI_PACKAGE_DIR` | Override package dir (useful for Nix/Guix store paths) |
| `PI_OFFLINE` | Disable all startup network operations |
| `PI_SKIP_VERSION_CHECK` | Disable the `pi.dev` latest-version request |
| `PI_TELEMETRY` | Override install/update telemetry + provider attribution headers (`1`/`0`) |
| `PI_CACHE_RETENTION` | `long` for extended prompt caching (Anthropic 1h, OpenAI 24h) |
| `PI_SHARE_VIEWER_URL` | Override the base URL used by `/share` |
| `PI_HARDWARE_CURSOR` | `1` to show the hardware cursor |
| `PI_TUI_ESC_TIMEOUT` | ms to wait after a lone ESC before treating it as Escape (default 100 over SSH, 10 otherwise) |
| `HTTP_PROXY`, `HTTPS_PROXY` | Proxy outbound HTTP |
| `VISUAL`, `EDITOR` | External editor fallback for Ctrl+G |

### Process markers (set by the CLI)

| Variable | Value |
|----------|-------|
| `AI_AGENT` | `pi` — generic agent marker for tooling |
| `PI_CODING_AGENT` | `true` — Pi-specific child-process marker |

Set by CLI and RPC entry points; not set when pi is embedded via the SDK.

### Bash-tool session environment

Commands run by the LLM-callable bash tool receive (resolved at each command start; not injected into user `!`/`!!` commands):

| Variable | Description |
|----------|-------------|
| `PI_SESSION_ID` | Current session ID |
| `PI_SESSION_FILE` | Absolute session JSONL path; unset for ephemeral sessions |
| `PI_PROVIDER` | Selected model provider |
| `PI_MODEL` | Selected model ID |
| `PI_REASONING_LEVEL` | Effective reasoning level |

To report which model runs: `printf '%s/%s\n' "$PI_PROVIDER" "$PI_MODEL"` — do not infer from the system prompt.
