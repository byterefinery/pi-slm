/**
 * pi-slm.ts — pi extension: SLM simulated message injection.
 *
 * At the start of every NEW (message-less) session, appends the simulated
 * (slm) messages from pi-slm.json as REGULAR session messages, in file
 * order, so the transcript starts with:
 *   [slm messages…] → [user question] → [assistant answer]
 * They are plain {"type":"message"} entries (no custom types): visible in
 * the TUI transcript and in /export <f>.jsonl. Resumed / continued / forked
 * sessions (pi -c, /resume, /fork, /import) already contain the messages
 * and are left untouched.
 *
 * A "system" message (any position) never becomes a conversation message.
 * While present in the file, its content acts as pi's OVERRIDE system
 * prompt (like --system-prompt / SYSTEM.md) for every prompt — new and
 * resumed alike, since the file is re-read on each prompt (live edits
 * apply; removing it restores pi's own prompt). Like pi's own override it
 * replaces the default prompt body while keeping pi's standard tail: the
 * append-system-prompt text, the <project_context> context files, the
 * skills section, and the current working directory line.
 *
 * Implementation: pi snapshots the in-memory message state of a new
 * session BEFORE session_start fires, so the injected prefix is missing
 * from the agent state for the session's lifetime (the session file is
 * correct). The `context` hook re-prepends the persisted prefix to the LLM
 * context on every call where it is missing; resumed sessions (and states
 * rebuilt after compaction or /tree) already have it. The system override
 * is applied via `before_agent_start` by reassembling the prompt exactly
 * as pi's buildSystemPrompt does for a custom prompt (that function is not
 * exported; mirrored for pi 0.84.3).
 *
 * pi-slm.json format (OpenAI chat format, in order):
 * { "messages": [
 *     { "role": "system",    "content": "..." },              // optional
 *     { "role": "user",      "content": "..." },
 *     { "role": "assistant", "content": "...", "reasoning_content": "...",
 *       "tool_calls": [{ "id": "...", "function": { "name": "...", "arguments": "..." } }] },
 *     { "role": "tool", "tool_call_id": "...", "name": "...", "content": "...", "is_error": false }
 * ] }
 *
 * Lookup order: 1. <cwd>/pi-slm.json  2. <extension dir>/pi-slm.json
 * Conversation messages are read once at session start; "system" is re-read
 * on every prompt. Conversion is all-or-nothing: any invalid message skips
 * the whole injection (reason logged). Errors are always reported on
 * stderr; with PI_SLM_DEBUG=1 status lines are printed too. Default: silent.
 *
 * Usage:  pi -e /path/to/pi-slm.ts
 */
import type { BuildSystemPromptOptions, ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { formatSkillsForPrompt } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const JSON_NAME = "pi-slm.json";
const log = (msg: string) => console.error(`[pi-slm] ${msg}`);

interface SlmJsonToolCall {
  id?: string;
  function?: { name?: string; arguments?: string | Record<string, unknown> };
}

interface SlmJsonMessage {
  role: string;
  content?: unknown;
  reasoning_content?: string;
  tool_calls?: SlmJsonToolCall[];
  tool_call_id?: string;
  name?: string;
  is_error?: boolean;
}

/** Injected message shapes (AgentMessage, see docs/session-format.md). */
type SlmSessionMessage =
  | { role: "user"; content: string; timestamp: number }
  | {
      role: "assistant";
      content: any[];
      api: string;
      provider: string;
      model: string;
      usage: typeof ZERO_USAGE;
      stopReason: "stop" | "toolUse";
      timestamp: number;
    }
  | { role: "toolResult"; toolCallId: string; toolName: string; content: any[]; isError: boolean; timestamp: number };

interface SlmPayload {
  /** Conversation messages (user/assistant/tool), in file order. */
  messages: SlmJsonMessage[];
  /** Content of the first "system" message, or null when absent. */
  system: string | null;
  source: string;
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

/** Accepts a plain string or a content-block array (joins text blocks). */
function textOf(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    const blocks = content as Array<{ type?: string; text?: string }>;
    return blocks
      .map((b) => (b.type === "text" && typeof b.text === "string" ? b.text : ""))
      .join("")
      .trim();
  }
  return "";
}

/** Loads and parses pi-slm.json from cwd, then from the extension dir. */
function loadPayload(extDir: string, quiet = false): SlmPayload | null {
  const file = [resolve(process.cwd(), JSON_NAME), resolve(extDir, JSON_NAME)].find(existsSync);
  if (!file) {
    if (!quiet) log(`${JSON_NAME} not found in ${process.cwd()} or ${extDir} — nothing injected`);
    return null;
  }
  let parsed: { messages?: SlmJsonMessage[] };
  try {
    parsed = JSON.parse(readFileSync(file, "utf8"));
  } catch (err: any) {
    log(`${file}: failed to parse: ${err?.message ?? err}`);
    return null;
  }
  if (!Array.isArray(parsed.messages)) {
    log(`${file}: "messages" must be an array`);
    return null;
  }
  let system: string | null = null;
  const messages: SlmJsonMessage[] = [];
  for (const [i, m] of parsed.messages.entries()) {
    if (m?.role === "system") {
      const text = textOf(m.content);
      if (!text) log(`messages[${i}]: "system" message has no text content — ignoring`);
      else if (system === null) system = text;
      else log(`messages[${i}]: multiple "system" messages — using the first one`);
      continue;
    }
    messages.push(m);
  }
  if (messages.length === 0 && system === null) {
    log(`${file}: "messages" must contain at least one conversation message or a "system" message`);
    return null;
  }
  return { messages, system, source: file };
}

/** Tool-call arguments as an object; null when a JSON string fails to parse. */
function toolCallArgs(raw: string | Record<string, unknown> | null | undefined): Record<string, unknown> | null {
  if (raw == null) return {};
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : { value: parsed };
    } catch {
      return null;
    }
  }
  return raw;
}

/** Converts one OpenAI-style slm message to a pi session message, or null (with a logged reason) if invalid. */
function convertMessage(
  m: SlmJsonMessage,
  idx: number,
  model: ModelRef | undefined,
  now: number,
): { msg: SlmSessionMessage; toolCallIds: string[] } | null {
  switch (m?.role) {
    case "user": {
      const text = textOf(m.content);
      if (!text) {
        log(`messages[${idx}]: user message has no text content — skipping injection`);
        return null;
      }
      return { msg: { role: "user", content: text, timestamp: now }, toolCallIds: [] };
    }

    case "assistant": {
      const content: any[] = [];
      if (m.reasoning_content?.trim()) content.push({ type: "thinking", thinking: m.reasoning_content.trim() });
      const text = textOf(m.content);
      if (text) content.push({ type: "text", text });

      const toolCallIds: string[] = [];
      const toolCalls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
      for (const [i, tc] of toolCalls.entries()) {
        const name = tc?.function?.name;
        if (!name) {
          log(`messages[${idx}].tool_calls[${i}]: missing function.name — skipping injection`);
          return null;
        }
        let args = toolCallArgs(tc.function?.arguments);
        if (args === null) {
          log(`messages[${idx}].tool_calls[${i}]: arguments is not valid JSON — using {}`);
          args = {};
        }
        const callId = tc.id ?? `slm_call_${idx}_${i}`;
        toolCallIds.push(callId);
        content.push({ type: "toolCall", id: callId, name, arguments: args });
      }

      if (content.length === 0) {
        log(`messages[${idx}]: assistant message has no content, reasoning_content or tool_calls — skipping injection`);
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
          stopReason: toolCallIds.length > 0 ? "toolUse" : "stop",
          timestamp: now,
        },
        toolCallIds,
      };
    }

    case "tool": {
      if (!m.tool_call_id) {
        log(`messages[${idx}]: tool result missing tool_call_id — skipping injection`);
        return null;
      }
      const text = textOf(m.content);
      if (!text) {
        log(`messages[${idx}]: tool result has no content — skipping injection`);
        return null;
      }
      return {
        msg: {
          role: "toolResult",
          toolCallId: m.tool_call_id,
          toolName: m.name ?? "unknown",
          content: [{ type: "text", text }],
          isError: m.is_error === true,
          timestamp: now,
        },
        toolCallIds: [],
      };
    }

    default:
      log(`messages[${idx}]: unknown role "${m?.role}" (user | assistant | tool) — skipping injection`);
      return null;
  }
}

/** Converts a whole payload, in order. All-or-nothing: null = logged failure, nothing injected. */
function convertPayload(payload: SlmPayload, model: ModelRef | undefined, now: number): SlmSessionMessage[] | null {
  const converted: SlmSessionMessage[] = [];
  const toolCallIds = new Set<string>();
  for (const [i, m] of payload.messages.entries()) {
    const out = convertMessage(m, i, model, now);
    if (!out) return null;
    out.toolCallIds.forEach((id) => toolCallIds.add(id));
    converted.push(out.msg);
  }
  const orphan = converted.find((m) => m.role === "toolResult" && !toolCallIds.has(m.toolCallId));
  if (orphan) {
    log(`tool result for unknown tool_call_id "${orphan.toolCallId}" — skipping injection`);
    return null;
  }
  return converted;
}

/**
 * Builds the system prompt exactly as pi does for an OVERRIDE prompt
 * (mirrors the customPrompt branch of buildSystemPrompt, pi 0.84.3 — the
 * function is not exported by the package): the prompt replaces the default
 * body while pi's standard tail is kept.
 */
function buildOverrideSystemPrompt(options: BuildSystemPromptOptions, customPrompt: string): string {
  const promptCwd = options.cwd.replace(/\\/g, "/");
  const appendSection = options.appendSystemPrompt ? `\n\n${options.appendSystemPrompt}` : "";
  const contextFiles = options.contextFiles ?? [];
  const skills = options.skills ?? [];

  let prompt = customPrompt;
  if (appendSection) prompt += appendSection;
  if (contextFiles.length > 0) {
    prompt += "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n";
    for (const { path: filePath, content } of contextFiles) {
      prompt += `<project_instructions path="${filePath}">\n${content}\n</project_instructions>\n\n`;
    }
    prompt += "</project_context>\n";
  }
  if ((!options.selectedTools || options.selectedTools.includes("read")) && skills.length > 0) {
    prompt += formatSkillsForPrompt(skills);
  }
  prompt += `\nCurrent working directory: ${promptCwd}\n`;
  return prompt;
}

/** True if the LLM context already starts with the injected slm prefix. */
function hasSlmPrefix(messages: any[], slmPrefix: SlmSessionMessage[]): boolean {
  return slmPrefix.every((m, i) => messages[i]?.role === m.role && textOf(messages[i].content) === textOf(m.content));
}

export default function (pi: ExtensionAPI) {
  const extDir = typeof __dirname !== "undefined" ? __dirname : process.cwd();

  // The slm prefix persisted into THIS session at startup (null for resumed
  // sessions, which already contain it in the session file).
  let slmPrefix: SlmSessionMessage[] | null = null;

  // While pi-slm.json contains a "system" message, its content acts as pi's
  // override system prompt for every agent run (new and resumed alike — the
  // file is re-read on each prompt, so live edits apply).
  pi.on("before_agent_start", async (event) => {
    const payload = loadPayload(extDir, true);
    if (!payload?.system) return;

    const options = event.systemPromptOptions;
    // Assemble exactly like pi does for an override prompt, so context
    // files, skills and the cwd line are preserved (raw override as
    // fallback if the structured options are ever missing).
    const systemPrompt = options?.cwd ? buildOverrideSystemPrompt(options, payload.system) : payload.system;

    if (process.env.PI_SLM_DEBUG === "1")
      log(`overriding system prompt with "system" message from ${payload.source} (${payload.system.length} chars)`);
    return { systemPrompt };
  });

  // Fires for every session runtime, before the user's first message exists
  // (verified for interactive and print/JSON/RPC modes alike).
  pi.on("session_start", async (_event, ctx) => {
    slmPrefix = null;

    // Only message-less (fresh) sessions get the payload. Resumed, continued
    // and forked sessions already contain the injected messages.
    if ((ctx.sessionManager.getBranch() as Array<{ type: string }>).some((e) => e.type === "message")) return;

    const payload = loadPayload(extDir);
    if (!payload || payload.messages.length === 0) return; // system-only: applied via before_agent_start

    const converted = convertPayload(payload, ctx.model as ModelRef | undefined, Date.now());
    if (!converted) return;

    // Append as regular session messages, in file order. They land at the
    // start of the conversation; the user's own messages follow after them.
    for (const msg of converted) ctx.sessionManager.appendMessage(msg);
    slmPrefix = converted;

    if (process.env.PI_SLM_DEBUG === "1")
      log(`injected ${converted.length} message(s) from ${payload.source} at session start — user can now ask questions`);
  });

  // The in-memory state of a brand-new session was snapshotted before the
  // session_start append, so the slm prefix is missing from the LLM context
  // for the session's lifetime. Re-prepend it on every call where it is
  // absent (resumed sessions and states rebuilt from the session file
  // already have the prefix, so nothing is added there).
  pi.on("context", (event) => {
    if (!slmPrefix) return;
    const messages = event.messages;
    if (!messages.some((m) => m.role === "user")) return; // nothing to anchor on
    if (hasSlmPrefix(messages, slmPrefix)) return;

    if (process.env.PI_SLM_DEBUG === "1")
      log(`prepended ${slmPrefix.length} message(s) to LLM context (${messages.length} -> ${messages.length + slmPrefix.length} context messages)`);
    return { messages: [...slmPrefix, ...messages] };
  });
}
