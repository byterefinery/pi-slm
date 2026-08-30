/**
 * pi-slm.ts — pi extension: SLM simulated message injection.
 *
 * At the start of every NEW session, inserts an arbitrary list of simulated
 * (slm) messages from `pi-slm.json` as REGULAR session messages — in the
 * exact order they appear in the file — so that:
 *
 *   1. they are plain {"type":"message"} entries (user / assistant /
 *      toolResult) in the session JSONL: visible in the TUI transcript as
 *      soon as the session starts and in /export <f>.jsonl. No custom entry
 *      types are written to the session.
 *   2. they sit at the beginning of the conversation, so the user can then
 *      ask questions and the LLM sees them in chronological order:
 *        [slm messages…] → [user question] → [assistant answer]
 *   3. resumed / continued sessions (pi -c, /resume, /fork, /import) are
 *      left untouched — they already contain the injected messages, and pi
 *      loads them into the agent state automatically at startup.
 *
 * Implementation notes: pi snapshots the agent's in-memory message state
 * from the session BEFORE the session_start event fires, so in a brand-new
 * session the injected messages are missing from the agent state for the
 * rest of that session's lifetime (the session file, however, is correct).
 * The `context` hook compensates: it re-prepends the persisted slm prefix
 * to the LLM context on every call where it is missing. Resumed sessions
 * already have the prefix in their state, so nothing is injected there.
 *
 * Supported roles (OpenAI chat format):
 *
 *   { "role": "user",      "content": "..." }
 *   { "role": "assistant", "content": "...", "reasoning_content": "...",
 *     "tool_calls": [{ "id": "...", "function": { "name": "...", "arguments": "..." } }] }
 *   { "role": "tool",      "tool_call_id": "...", "name": "...", "content": "...",
 *     "is_error": false }
 *
 * The file is read once at session start. Lookup order:
 *   1. <cwd>/pi-slm.json
 *   2. <extension directory>/pi-slm.json
 *
 * pi-slm.json format:
 * { "messages": [ ...any number of the messages above, in order... ] }
 *
 * A debug line is printed to stderr on injection (suppress with
 * PI_SLM_QUIET=1).
 *
 * Usage:  pi -e /path/to/pi-slm.ts
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const JSON_NAME = "pi-slm.json";

interface SlmJsonToolCall {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string | Record<string, unknown> };
}

interface SlmJsonMessage {
  role: string;
  content?: string | Array<{ type?: string; text?: string }> | null;
  reasoning_content?: string;
  tool_calls?: SlmJsonToolCall[];
  tool_call_id?: string;
  name?: string;
  is_error?: boolean;
}

interface SlmPayload {
  messages: unknown[];
  source: string;
}

/** Directory containing this extension file (jiti provides __dirname). */
function extensionDir(): string {
  try {
    return typeof __dirname !== "undefined" ? __dirname : process.cwd();
  } catch {
    return process.cwd();
  }
}

/** Accepts a plain string or an array of content blocks (joins text blocks). */
function textOf(content: SlmJsonMessage["content"] | unknown): string {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content
      .filter((b: any) => b && b.type === "text")
      .map((b: any) => (typeof b.text === "string" ? b.text : ""))
      .join("")
      .trim();
  }
  return "";
}

/** Loads and parses pi-slm.json from the first existing candidate path. */
function loadPayload(extDir: string): SlmPayload | null {
  const candidates = [resolve(process.cwd(), JSON_NAME), resolve(extDir, JSON_NAME)];
  for (const file of candidates) {
    if (!existsSync(file)) continue;
    try {
      const parsed = JSON.parse(readFileSync(file, "utf8")) as { messages?: SlmJsonMessage[] };
      if (!Array.isArray(parsed.messages) || parsed.messages.length === 0) {
        console.error(`[pi-slm] ${file}: "messages" must be a non-empty array`);
        return null;
      }
      return { messages: parsed.messages, source: file };
    } catch (err: any) {
      console.error(`[pi-slm] ${file}: failed to parse: ${err?.message ?? err}`);
      return null;
    }
  }
  console.error(`[pi-slm] ${JSON_NAME} not found in ${process.cwd()} or ${extDir} — nothing injected`);
  return null;
}

const ZERO_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

type ModelRef = { api?: string; provider?: string; id?: string };

/**
 * Converts one OpenAI-style slm message into a pi session message.
 * Returns null (with a logged reason) if the message is invalid.
 */
function convertMessage(
  m: SlmJsonMessage,
  idx: number,
  model: ModelRef | undefined,
  now: number,
): { msg: any; toolCallIds: string[] } | null {
  switch (m?.role) {
    case "user": {
      const text = textOf(m.content);
      if (!text) {
        console.error(`[pi-slm] messages[${idx}]: user message has no text content — skipping injection`);
        return null;
      }
      return { msg: { role: "user", content: text, timestamp: now }, toolCallIds: [] };
    }

    case "assistant": {
      const content: any[] = [];
      if (m.reasoning_content && m.reasoning_content.trim()) {
        content.push({ type: "thinking", thinking: m.reasoning_content.trim() });
      }
      const text = textOf(m.content);
      if (text) {
        content.push({ type: "text", text });
      }

      const toolCallIds: string[] = [];
      const toolCalls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
      for (let i = 0; i < toolCalls.length; i++) {
        const tc = toolCalls[i];
        const name = tc?.function?.name;
        if (!name) {
          console.error(`[pi-slm] messages[${idx}].tool_calls[${i}]: missing function.name — skipping injection`);
          return null;
        }
        let args: Record<string, unknown>;
        const rawArgs = tc.function?.arguments;
        if (rawArgs === undefined || rawArgs === null) {
          args = {};
        } else if (typeof rawArgs === "string") {
          try {
            const parsed = JSON.parse(rawArgs);
            args = parsed && typeof parsed === "object" ? parsed : { value: parsed };
          } catch {
            console.error(`[pi-slm] messages[${idx}].tool_calls[${i}]: arguments is not valid JSON — using {}`);
            args = {};
          }
        } else {
          args = rawArgs;
        }
        const callId = tc.id ?? `slm_call_${idx}_${i}`;
        toolCallIds.push(callId);
        content.push({ type: "toolCall", id: callId, name, arguments: args });
      }

      if (content.length === 0) {
        console.error(`[pi-slm] messages[${idx}]: assistant message has no content, reasoning_content or tool_calls — skipping injection`);
        return null;
      }
      return {
        msg: {
          role: "assistant",
          content,
          api: model?.api ?? "openai-completions",
          provider: model?.provider ?? "unknown",
          model: model?.id ?? "unknown",
          usage: ZERO_USAGE,
          stopReason: toolCalls.length > 0 ? "toolUse" : "stop",
          timestamp: now,
        },
        toolCallIds,
      };
    }

    case "tool": {
      const toolCallId = m.tool_call_id;
      if (!toolCallId) {
        console.error(`[pi-slm] messages[${idx}]: tool result missing tool_call_id — skipping injection`);
        return null;
      }
      const text = textOf(m.content);
      if (!text) {
        console.error(`[pi-slm] messages[${idx}]: tool result has no content — skipping injection`);
        return null;
      }
      return {
        msg: {
          role: "toolResult",
          toolCallId,
          toolName: m.name ?? "unknown",
          content: [{ type: "text", text }],
          isError: m.is_error === true,
          timestamp: now,
        },
        toolCallIds: [],
      };
    }

    default:
      console.error(`[pi-slm] messages[${idx}]: unknown role "${m?.role}" (user | assistant | tool) — skipping injection`);
      return null;
  }
}

/**
 * Converts a whole payload, in order. All-or-nothing: returns null (with a
 * logged reason) if any message is invalid, so a partial conversation is
 * never written to the session or injected.
 */
function convertPayload(
  payload: SlmPayload,
  model: ModelRef | undefined,
  now: number,
): any[] | null {
  const converted: any[] = [];
  const declaredToolCallIds = new Set<string>();
  for (let i = 0; i < payload.messages.length; i++) {
    const out = convertMessage(payload.messages[i] as SlmJsonMessage, i, model, now);
    if (!out) return null;
    for (const id of out.toolCallIds) declaredToolCallIds.add(id);
    converted.push(out.msg);
  }
  for (const msg of converted) {
    if (msg.role === "toolResult" && !declaredToolCallIds.has(msg.toolCallId)) {
      console.error(`[pi-slm] tool result for unknown tool_call_id "${msg.toolCallId}" — skipping injection`);
      return null;
    }
  }
  return converted;
}

/** True if the LLM context already starts with the injected slm prefix. */
function hasSlmPrefix(messages: any[], slmPrefix: any[]): boolean {
  if (slmPrefix.length === 0) return true;
  const first = slmPrefix[0];
  const m = messages[0];
  if (!m || m.role !== first.role) return false;
  const a = textOf(m.content);
  const b = textOf(first.content);
  return b !== "" && a === b;
}

export default function (pi: ExtensionAPI) {
  const extDir = extensionDir();

  // The slm prefix persisted into THIS session at startup (null for resumed
  // sessions, which already contain it in the session file).
  let slmPrefix: any[] | null = null;

  // Fired for every session runtime, before the user's first message exists
  // (verified for interactive and print/JSON/RPC modes alike).
  pi.on("session_start", async (_event, ctx) => {
    slmPrefix = null;

    // Only message-less (fresh) sessions get the payload. Resumed, continued
    // and forked sessions already contain the injected messages.
    const branch = ctx.sessionManager.getBranch() as Array<{ type: string }>;
    if (branch.some((e) => e.type === "message")) {
      return;
    }

    const payload = loadPayload(extDir);
    if (!payload) return;

    const converted = convertPayload(payload, ctx.model as ModelRef | undefined, Date.now());
    if (!converted) return;

    // Append as regular session messages, in file order. They land at the
    // start of the conversation; the user's own messages follow after them.
    for (const msg of converted) {
      ctx.sessionManager.appendMessage(msg);
    }
    slmPrefix = converted;

    if (process.env.PI_SLM_QUIET !== "1") {
      console.error(
        `[pi-slm] injected ${converted.length} message(s) from ${payload.source} at session start — user can now ask questions`,
      );
    }
  });

  // The agent's in-memory state for a brand-new session was snapshotted
  // before the session_start append, so for the lifetime of that session the
  // slm prefix is missing from the LLM context. Re-prepend it on every call
  // where it is absent. Resumed sessions (and this session after compaction
  // or /tree navigation, which rebuild state from the session file) already
  // have the prefix, so nothing is added.
  pi.on("context", (event) => {
    if (!slmPrefix || slmPrefix.length === 0) return;
    const messages = event.messages;
    if (!messages.some((m) => m.role === "user")) return; // nothing to anchor on
    if (hasSlmPrefix(messages, slmPrefix)) return;

    if (process.env.PI_SLM_QUIET !== "1") {
      console.error(
        `[pi-slm] prepended ${slmPrefix.length} message(s) to LLM context ` +
          `(${messages.length} -> ${messages.length + slmPrefix.length} context messages)`,
      );
    }
    return { messages: [...slmPrefix, ...messages] };
  });
}
