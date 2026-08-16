# Programmatic Modes

Pi exposes three programmatic surfaces: `--mode json` (event stream), `--mode rpc` (command protocol), and the TypeScript SDK.

## JSON Event Stream Mode

```bash
pi --mode json "Your prompt"
```

Outputs all session events as JSON lines to stdout. The first line is the session header:

```json
{"type":"session","version":3,"id":"uuid","timestamp":"...","cwd":"/path"}
```

Followed by lifecycle events as they occur:

- **Agent:** `agent_start`, `agent_end`
- **Turn:** `turn_start`, `turn_end`
- **Message:** `message_start`, `message_update`, `message_end`
- **Tool:** `tool_execution_start`, `tool_execution_update`, `tool_execution_end`
- Plus `queue_update` (pending steering/follow-up queues) and `compaction_start`/`compaction_end`.

`message_update` records are delta-only (they omit the cumulative `message` field and `partial` snapshots to keep the stream linear); `message_end` carries the final authoritative message. Filter with jq:

```bash
pi --mode json "List files" 2>/dev/null | jq -c 'select(.type == "message_end")'
```

## RPC Mode

Headless operation over stdin/stdout:

```bash
pi --mode rpc [options]
```

- **Commands:** JSON objects sent to stdin, one per line
- **Responses:** `{"type":"response","command":...,"success":...}` per command
- **Events:** agent events streamed to stdout as JSON lines
- Commands may carry an optional `id` for request/response correlation

**Framing is strict JSONL with LF as the only record delimiter.** Split on `\n` only; strip an optional trailing `\r`; do **not** use generic line readers — Node `readline` also splits on U+2028/U+2029, which are valid inside JSON strings, and is not protocol-compliant.

Command groups (see upstream `docs/rpc.md` for full payloads):

| Group | Commands |
|-------|----------|
| Prompting | `prompt` (with optional `images`, `streamingBehavior: steer\|followUp` while streaming), `steer`, `follow_up`, `abort` |
| Session lifecycle | `new_session`, `switch_session`, `fork`, `clone`, `set_session_name` |
| State | `get_state`, `get_messages`, `get_entries`, `get_tree`, `get_session_stats`, `get_last_assistant_text` |
| Model | `set_model`, `cycle_model`, `get_available_models` |
| Thinking | `set_thinking_level`, `cycle_thinking_level`, `get_available_thinking_levels` |
| Queue modes | `set_steering_mode`, `set_follow_up_mode` |
| Compaction / retry | `compact`, `set_auto_compaction`, `set_auto_retry`, `abort_retry` |
| Bash | `bash`, `abort_bash` |
| Output | `export_html`, `get_commands` |

If the agent is streaming and you send `prompt` without `streamingBehavior`, the command errors — queue with `steer`/`follow_up` instead.

## SDK

For Node.js/TypeScript, import the package instead of spawning a subprocess:

```typescript
import { createAgentSession, ModelRuntime, SessionManager } from "@earendil-works/pi-coding-agent";

const modelRuntime = await ModelRuntime.create();
const { session } = await createAgentSession({
  sessionManager: SessionManager.inMemory(),
  modelRuntime,
});

await session.prompt("What files are in the current directory?");
```

For advanced multi-session runtimes use `createAgentSessionRuntime()` / `AgentSessionRuntime`. See upstream `docs/sdk.md` and `examples/sdk/`.
