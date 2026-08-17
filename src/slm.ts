/**
 * slm — Small Language Model reliability extension for pi (0.84.2).
 *
 * Implements the features specified in SLM_KNOWN_ISSUES.md:
 *
 *  1. Destructive-write protection — a `write` on an existing file whose
 *     change is small is converted into a deterministic `edit` call (exact
 *     path / edits[].oldText / edits[].newText); huge changes or non-unique
 *     hunks fall back to the built-in `write`.
 *  2. `edit` oldText validation — edits whose oldText does not exist in the
 *     file are dropped; if none survive, the call is blocked ("empty edit"),
 *     the assistant message that caused it is moved off the active session
 *     branch, and a replacement hint message is appended.
 *  3. Startup context messages — on a new session, two custom messages are
 *     injected before the first user request is processed: the active skills
 *     and the active tools, as compact YAML (absolute paths, parameter
 *     schemas, strict flags).
 *  4. Skill structure hint — the first successful read inside a known skill
 *     directory gets a YAML hint (absolute path / references / scripts)
 *     appended to the result.
 *  5. Mandatory bash timeout — every built-in `bash` call carries `timeout`
 *     (60 s base, +60 s after each timeout failure, capped at 300 s).
 *  6. Tool error hint — every tool failure gets the tool's YAML definition
 *     appended (unknown tool names get the list of available tools).
 *  7. Skill error hint — a failed read that maps to a known skill gets that
 *     skill's YAML hint appended; unknown skill names get the available
 *     skills list.
 *  8. `read` on a directory — an EISDIR error is replaced with an
 *     `ls -F` listing of the path (non-error).
 *
 * Load with:  pi -e src/slm.ts
 *
 * Single file, no dependencies — only pi's internal TypeScript API is used
 * (extension events, session manager, built-in tool factories, model/tool
 * state). All state is per-session and in-memory. Works in all modes
 * (interactive TUI, -p, --mode json, --mode rpc).
 */

import {
  createEditToolDefinition,
  createWriteToolDefinition,
  isToolCallEventType,
} from "@earendil-works/pi-coding-agent";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import type {
  AgentToolResult,
  AgentToolUpdateCallback,
  BeforeAgentStartEvent,
  BeforeProviderRequestEvent,
  EditToolDetails,
  ExtensionAPI,
  ExtensionContext,
  SessionStartEvent,
  SessionManager,
  Skill,
  ToolCallEventResult,
  ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve as resolvePath, sep } from "node:path";

// ============================================================================
// Constants
// ============================================================================

const CT_SKILLS = "slm_available_skills";
const CT_TOOLS = "slm_available_tools";
const CT_HINT = "slm_hint";

const BASH_TIMEOUT_BASE = 60;
const BASH_TIMEOUT_STEP = 60;
const BASH_TIMEOUT_MAX = 300;

/** write → edit conversion thresholds (see SLM_KNOWN_ISSUES.md, feature 1). */
const MAX_HUNKS = 5;
const MAX_CHANGED_LINES = 60;
const MAX_SIDE_LINES = 4000;
/** Size guard for the LCS over the changed middle region (cells in the DP). */
const MAX_LCS_CELLS = 2_000_000;

// ============================================================================
// Pure helpers (unit-testable, no pi imports at runtime)
// ============================================================================

type SlmEdit = { oldText: string; newText: string };
type TextBlock = { type: "text"; text: string };

/** Quote a YAML scalar only when plain style is not safe. */
function yq(value: string): string {
  if (value === "") return '""';
  const plain =
    /^[A-Za-z][A-Za-z0-9 .,_()!?/'-]*$/.test(value) &&
    !value.includes(":") &&
    !value.includes(" #") &&
    !value.includes("\n");
  return plain ? value : JSON.stringify(value);
}

/** Render one tool as the compact YAML used by the startup message / hints. */
function renderToolYaml(tool: { name: string; description: string; parameters: unknown }, strict: boolean): string {
  const L: string[] = [];
  L.push(`- name: ${tool.name}`);
  L.push(`  description: ${yq(tool.description)}`);
  const params = tool.parameters as Record<string, unknown> | null;
  if (params && typeof params === "object") {
    L.push("  parameters:");
    renderObjectSchema(params, "    ", L, true, 0);
  } else {
    L.push("  parameters: {}");
  }
  L.push(`  strict: ${strict ? "true" : "false"}`);
  return L.join("\n");
}

function renderObjectSchema(
  schema: Record<string, unknown>,
  indent: string,
  L: string[],
  omitTypeLine: boolean,
  depth: number,
): void {
  if (depth > 4) return;
  if (!omitTypeLine) L.push(`${indent}type: object`);
  const required = schema.required;
  if (Array.isArray(required) && required.length > 0) {
    L.push(`${indent}required:`);
    for (const r of required) {
      if (typeof r === "string") L.push(`${indent}  - ${r}`);
    }
  }
  const props = schema.properties as Record<string, unknown> | undefined;
  if (props && typeof props === "object") {
    L.push(`${indent}properties:`);
    for (const [key, value] of Object.entries(props)) {
      renderProperty(key, value as Record<string, unknown>, `${indent}  `, L, depth + 1);
    }
  }
}

function renderProperty(key: string, schema: Record<string, unknown>, indent: string, L: string[], depth: number): void {
  const type = typeof schema.type === "string" ? schema.type : undefined;
  L.push(`${indent}${key}:`);
  if (type && type !== "string") L.push(`${indent}  type: ${type}`);
  if (typeof schema.description === "string") L.push(`${indent}  description: ${yq(schema.description)}`);
  if (type === "array" && schema.items && typeof schema.items === "object") {
    L.push(`${indent}  items:`);
    renderObjectSchema(schema.items as Record<string, unknown>, `${indent}    `, L, false, depth + 1);
    return;
  }
  if (schema.properties && typeof schema.properties === "object") {
    L.push(`${indent}  properties:`);
    for (const [subKey, subValue] of Object.entries(schema.properties as Record<string, unknown>)) {
      renderProperty(subKey, subValue as Record<string, unknown>, `${indent}    `, L, depth + 1);
    }
  }
}

/** Render the "Available skills:" body (one entry per skill). */
function renderSkillsYaml(skills: Array<{ name: string; description: string; path: string }>): string {
  if (skills.length === 0) return "(none discovered)";
  const L: string[] = [];
  for (const s of skills) {
    L.push(`- name: ${s.name}`);
    L.push(`  description: ${yq(s.description)}`);
    L.push(`  path: ${s.path}`);
  }
  return L.join("\n");
}

/** Render a skill structure hint (features 4/7). references/scripts omitted when absent. */
function renderSkillHintYaml(
  skill: { name: string; description: string; path: string },
  references: string[],
  scripts: string[],
): string {
  const L: string[] = [];
  L.push(`- name: ${skill.name}`);
  L.push(`  description: ${yq(skill.description)}`);
  L.push(`  path: ${skill.path}`);
  if (references.length > 0) {
    L.push("  references:");
    for (const r of references) L.push(`    - ${r}`);
  }
  if (scripts.length > 0) {
    L.push("  scripts:");
    for (const s of scripts) L.push(`    - ${s}`);
  }
  return L.join("\n");
}

function countOccurrences(haystack: string, needle: string): number {
  if (needle === "") return 0;
  let count = 0;
  let i = haystack.indexOf(needle);
  while (i !== -1) {
    count += 1;
    i = haystack.indexOf(needle, i + needle.length);
  }
  return count;
}

/** Keep only edits whose oldText exists in the file content (feature 2). */
function filterSurvivingEdits(content: string, edits: SlmEdit[]): SlmEdit[] {
  return edits.filter((e) => typeof e.oldText === "string" && e.oldText.length > 0 && content.includes(e.oldText));
}

interface DiffHunk {
  a0: number;
  a1: number;
  b0: number;
  b1: number;
}

/** LCS over two line arrays; returns maximal contiguous changed runs. */
function lcsHunks(a: string[], b: string[]): DiffHunk[] | null {
  const m = a.length;
  const n = b.length;
  if (m * n > MAX_LCS_CELLS) return null;
  const w = n + 1;
  const dp = new Uint32Array((m + 1) * (n + 1));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      dp[i * w + j] = a[i] === b[j]
        ? dp[(i + 1) * w + j + 1] + 1
        : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1]);
    }
  }
  type Op = { op: "eq" | "del" | "ins"; ai: number; bi: number };
  const ops: Op[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (a[i] === b[j]) {
      ops.push({ op: "eq", ai: i, bi: j });
      i += 1;
      j += 1;
    } else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) {
      ops.push({ op: "del", ai: i, bi: j });
      i += 1;
    } else {
      ops.push({ op: "ins", ai: i, bi: j });
      j += 1;
    }
  }
  while (i < m) {
    ops.push({ op: "del", ai: i, bi: j });
    i += 1;
  }
  while (j < n) {
    ops.push({ op: "ins", ai: i, bi: j });
    j += 1;
  }
  const hunks: DiffHunk[] = [];
  let k = 0;
  while (k < ops.length) {
    if (ops[k].op === "eq") {
      k += 1;
      continue;
    }
    const a0 = ops[k].ai;
    const b0 = ops[k].bi;
    let a1 = a0;
    let b1 = b0;
    while (k < ops.length && ops[k].op !== "eq") {
      if (ops[k].op === "del") a1 = ops[k].ai + 1;
      else b1 = ops[k].bi + 1;
      k += 1;
    }
    hunks.push({ a0, a1, b0, b1 });
  }
  return hunks;
}

function lineStarts(file: string, n: number): number[] {
  const starts = new Array<number>(n + 1);
  starts[0] = 0;
  // starts[i] = (index of the newline terminating line i-1) + 1. Search from
  // starts[i-1] itself: when line i-1 is empty, its terminating newline sits
  // exactly at starts[i-1].
  for (let i = 1; i < n; i++) starts[i] = file.indexOf("\n", starts[i - 1]) + 1;
  starts[n] = file.length;
  return starts;
}

/**
 * Convert a small content change into deterministic `edit` hunks.
 * Common prefix/suffix lines are trimmed, the changed middle is aligned with
 * LCS (size-guarded). oldText/newText are exact file substrings (line
 * boundaries included), so applying the edits string-replaces the old content
 * with the new one exactly. Pure insertions are anchored on an adjacent old
 * line (the edit tool expresses changes as replacements); returns null when
 * the change is too big or cannot be expressed.
 */
export function diffToEdits(oldText: string, newText: string): SlmEdit[] | null {
  const a = oldText.split("\n");
  const b = newText.split("\n");
  if (a.length > MAX_SIDE_LINES || b.length > MAX_SIDE_LINES) return null;
  let s = 0;
  while (s < a.length && s < b.length && a[s] === b[s]) s += 1;
  let ea = a.length;
  let eb = b.length;
  while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) {
    ea -= 1;
    eb -= 1;
  }
  const ma = a.slice(s, ea);
  const mb = b.slice(s, eb);
  if (ma.length === 0 && mb.length === 0) return [];
  let hunks: DiffHunk[] | null;
  if (ma.length * mb.length <= MAX_LCS_CELLS) {
    hunks = lcsHunks(ma, mb);
  } else {
    // Guard exceeded: only convert when the whole middle is small anyway.
    hunks =
      ma.length + mb.length <= MAX_CHANGED_LINES
        ? [{ a0: 0, a1: ma.length, b0: 0, b1: mb.length }]
        : null;
  }
  if (!hunks || hunks.length === 0 || hunks.length > MAX_HUNKS) return null;

  const oldStarts = lineStarts(oldText, a.length);
  const newStarts = lineStarts(newText, b.length);
  const rangeText = (starts: number[], file: string, i0: number, i1: number): string =>
    file.slice(starts[i0], i1 >= starts.length ? file.length : starts[i1]);

  let changed = 0;
  const edits: SlmEdit[] = [];
  for (const h of hunks) {
    let a0 = h.a0;
    let a1 = h.a1;
    let b0 = h.b0;
    let b1 = h.b1;
    if (a0 === a1) {
      // Pure insertion: anchor on a neighbouring old line so the edit tool
      // can express it as a replacement.
      if (s + a0 > 0) {
        // Anchor on the old line immediately before the insertion point.
        a0 -= 1;
        b0 -= 1;
      } else if (ma.length > 0) {
        // Insertion at the very start of the file: anchor on the first line.
        a1 = a0 + 1;
        b1 += 1;
      } else if (ea < a.length) {
        // Middle is entirely new: anchor on the first suffix line.
        a0 = ea - s;
        a1 = a0 + 1;
        b1 = eb - s + 1;
      } else {
        return null; // no old line to anchor the insertion on
      }
    }
    const oldSeg = rangeText(oldStarts, oldText, s + a0, s + a1);
    const newSeg = rangeText(newStarts, newText, s + b0, s + b1);
    if (oldSeg === "") {
      // Only possible for a hunk that covers just the trailing empty line,
      // i.e. the model dropped the file's final newline.
      if (s + a0 === a.length - 1 && s + a1 === a.length && newSeg === "") continue;
      return null; // not expressible as an edit
    }
    changed += a1 - a0 + (b1 - b0);
    if (changed > MAX_CHANGED_LINES) return null;
    edits.push({ oldText: oldSeg, newText: newSeg });
  }
  return edits;
}

/** Extract the concatenated text of a tool result content block list. */
function extractText(content: Array<{ type: string; text?: string }>): string {
  return content
    .filter((c) => c.type === "text" && typeof c.text === "string")
    .map((c) => c.text as string)
    .join("\n");
}

/** Capture provider-declared `strict` flags for tools (feature 3). */
function collectStrict(payload: unknown, out: Map<string, boolean>): void {
  const tools = (payload as { tools?: unknown } | null)?.tools;
  if (!Array.isArray(tools)) return;
  for (const t of tools as Array<Record<string, unknown>>) {
    if (!t || typeof t !== "object") continue;
    const fn = t.function as Record<string, unknown> | undefined;
    const name = typeof fn?.name === "string" ? fn.name : typeof t.name === "string" ? t.name : null;
    if (!name) continue;
    const strict = fn ? fn.strict : t.strict;
    if (typeof strict === "boolean") out.set(name, strict);
  }
}

// ============================================================================
// Per-session state
// ============================================================================

interface PendingEmptyEdit {
  toolCallId: string;
  assistantEntryId: string;
  assistantParentId: string | null;
  path: string;
}

interface SLMState {
  /** Inject the two startup messages on the next before_agent_start. */
  pendingStartup: boolean;
  /** Active skills captured from before_agent_start systemPromptOptions. */
  skills: Skill[];
  /** Skill names that already received a structure hint this session. */
  hintedSkills: Set<string>;
  /** Current default bash timeout (base + escalation steps, capped). */
  bashTimeout: number;
  /** Provider-declared strict flags captured from request payloads. */
  strictByTool: Map<string, boolean>;
  /** Zero-survivor edit call awaiting branch removal (feature 2). */
  pendingEmptyEdit: PendingEmptyEdit | null;
}

function freshState(): SLMState {
  return {
    pendingStartup: false,
    skills: [],
    hintedSkills: new Set(),
    bashTimeout: BASH_TIMEOUT_BASE,
    strictByTool: new Map(),
    pendingEmptyEdit: null,
  };
}

// ============================================================================
// Extension
// ============================================================================

export default function slmExtension(pi: ExtensionAPI): void {
  const states = new Map<string, SLMState>();

  const getState = (ctx: ExtensionContext): SLMState => {
    const id = ctx.sessionManager.getSessionId();
    let st = states.get(id);
    if (!st) {
      st = freshState();
      states.set(id, st);
    }
    return st;
  };

  const writableSession = (ctx: ExtensionContext): SessionManager =>
    ctx.sessionManager as unknown as SessionManager;

  /** List files (absolute paths) under a directory, recursively, sorted. */
  const listFiles = (dir: string): string[] => {
    const out: string[] = [];
    const walk = (d: string, depth: number): void => {
      if (depth > 3 || out.length >= 100) return;
      let entries;
      try {
        entries = readdirSync(d, { withFileTypes: true });
      } catch {
        return;
      }
      for (const e of entries) {
        if (e.name.startsWith(".")) continue;
        const p = join(d, e.name);
        if (e.isDirectory()) walk(p, depth + 1);
        else out.push(p);
      }
    };
    walk(dir, 0);
    return out.sort();
  };

  const skillHintText = (skill: Skill): string => {
    const refs = listFiles(join(skill.baseDir, "references"));
    const scripts = listFiles(join(skill.baseDir, "scripts"));
    return (
      `Skill '${skill.name}' structure (absolute paths):\n` +
      renderSkillHintYaml({ name: skill.name, description: skill.description, path: skill.filePath }, refs, scripts)
    );
  };

  const availableSkillsYaml = (st: SLMState): string =>
    renderSkillsYaml(
      st.skills
        .filter((s) => !s.disableModelInvocation)
        .map((s) => ({ name: s.name, description: s.description, path: s.filePath })),
    );

  const findSkillForPath = (st: SLMState, abs: string): Skill | undefined =>
    st.skills.find((s) => abs === s.baseDir || abs.startsWith(s.baseDir + sep));

  const knownSkills = (st: SLMState): Array<{ name: string; description: string; path: string }> =>
    st.skills
      .filter((s) => !s.disableModelInvocation)
      .map((s) => ({ name: s.name, description: s.description, path: s.filePath }));

  // --------------------------------------------------------------------------
  // Feature 3 (setup): decide whether this session needs the startup messages
  // --------------------------------------------------------------------------
  pi.on("session_start", (event: SessionStartEvent, ctx: ExtensionContext) => {
    const st = getState(ctx);
    st.pendingEmptyEdit = null;
    const entries = ctx.sessionManager.getEntries();
    const hasOurs = entries.some(
      (e) => e.type === "custom_message" && (e.customType === CT_SKILLS || e.customType === CT_TOOLS),
    );
    const hasMessages = entries.some((e) => e.type === "message" || e.type === "custom_message");
    if (hasOurs) {
      st.pendingStartup = false;
    } else if (event.reason === "startup" || event.reason === "new" || !hasMessages) {
      st.pendingStartup = true;
    } else {
      st.pendingStartup = false; // resume/fork/reload with existing history
    }
  });

  // --------------------------------------------------------------------------
  // Feature 3 (inject): send the two startup messages before the first
  // request is processed; capture skills/tools for the rest of the session
  // --------------------------------------------------------------------------
  pi.on("before_agent_start", (event: BeforeAgentStartEvent, ctx: ExtensionContext) => {
    const st = getState(ctx);
    st.skills = event.systemPromptOptions?.skills ?? [];
    if (!st.pendingStartup) return;
    st.pendingStartup = false;

    const skillsYaml = availableSkillsYaml(st);
    const active = new Set(pi.getActiveTools());
    const tools = pi.getAllTools().filter((t) => active.has(t.name));
    const toolsYaml = tools.map((t) => renderToolYaml(t, st.strictByTool.get(t.name) ?? false)).join("\n");

    pi.sendMessage(
      { customType: CT_SKILLS, content: `Available skills:\n${skillsYaml}`, display: true },
      { triggerTurn: false },
    );
    pi.sendMessage(
      { customType: CT_TOOLS, content: `Available tools:\n${toolsYaml}`, display: true },
      { triggerTurn: false },
    );
  });

  // --------------------------------------------------------------------------
  // Feature 3 (strict flags): capture provider-declared strict per tool
  // --------------------------------------------------------------------------
  pi.on("before_provider_request", (event: BeforeProviderRequestEvent, ctx: ExtensionContext) => {
    collectStrict(event.payload, getState(ctx).strictByTool);
  });

  // --------------------------------------------------------------------------
  // Feature 5: mandatory bash timeout; feature 2: edit oldText validation
  // --------------------------------------------------------------------------
  pi.on("tool_call", (event, ctx: ExtensionContext): ToolCallEventResult | void => {
    const st = getState(ctx);

    if (isToolCallEventType("bash", event)) {
      const t = event.input.timeout;
      if (typeof t !== "number" || !Number.isFinite(t) || t <= 0) {
        event.input.timeout = st.bashTimeout;
      }
      return;
    }

    if (!isToolCallEventType("edit", event)) return;

    const edits = event.input.edits;
    if (!Array.isArray(edits) || edits.length === 0) return;
    const abs = resolvePath(ctx.cwd, event.input.path);
    let content: string | null = null;
    try {
      content = readFileSync(abs, "utf8");
    } catch {
      content = null; // unreadable: let the built-in tool report the error
    }
    if (content === null) return;

    const survivors = filterSurvivingEdits(content, edits);
    if (survivors.length === edits.length) return;
    if (survivors.length > 0) {
      // In-place mutation: the executed call carries only surviving edits.
      event.input.edits = survivors;
      return;
    }

    // Zero survivors: block, and remember the assistant entry so it can be
    // moved off the active branch once the blocked result is persisted.
    const leaf = ctx.sessionManager.getLeafEntry();
    if (leaf?.type === "message") {
      const msg = leaf.message as { role?: string };
      if (msg.role === "assistant") {
        st.pendingEmptyEdit = {
          toolCallId: event.toolCallId,
          assistantEntryId: leaf.id,
          assistantParentId: leaf.parentId,
          path: event.input.path,
        };
      }
    }
    return {
      block: true,
      reason:
        `Empty edit: none of the edits[].oldText values existed in ${event.input.path}. ` +
        `The edit was not applied. Re-read the file with the read tool, then retry edit with ` +
        `oldText copied exactly from the file content.`,
    };
  });

  // --------------------------------------------------------------------------
  // Feature 2 (removal): when the run settles, move the assistant message of
  // a blocked empty edit (and its blocked result) off the active branch and
  // append the replacement hint message. Only done when the model made no
  // other tool calls after the block — otherwise the history shows a
  // legitimate recovery and is kept as-is (block reason remains in log).
  // --------------------------------------------------------------------------
  pi.on("agent_settled", (_event, ctx: ExtensionContext) => {
    const st = getState(ctx);
    const pending = st.pendingEmptyEdit;
    if (!pending) return;
    st.pendingEmptyEdit = null;
    if (pending.assistantParentId === null) return; // assistant was first entry: never delete

    const sm = writableSession(ctx);
    const hint = (): void => {
      const text =
        `Note: the edit call to ${pending.path} was an empty edit (none of its edits[].oldText ` +
        `values existed in the file) and was removed from this session log. Read the file with ` +
        `the read tool, then retry the edit with oldText copied exactly from the file content.`;
      if (ctx.mode === "print") {
        // Log-only: keep the model's response the last agent-state message so
        // print mode still prints it (the process exits right after anyway).
        sm.appendCustomMessageEntry(CT_HINT, text, true);
      } else {
        pi.sendMessage({ customType: CT_HINT, content: text, display: true }, { triggerTurn: false });
      }
    };

    const leaf = sm.getLeafEntry();
    const leafMsg = leaf?.type === "message" ? (leaf.message as { role?: string; toolCallId?: string }) : null;

    // Case A: the run terminated right after the blocked call (no final
    // assistant message): the leaf is the blocked tool result itself.
    if (leafMsg?.role === "toolResult" && leafMsg.toolCallId === pending.toolCallId) {
      sm.branch(pending.assistantParentId);
      hint();
      return;
    }

    // Case B: the model answered with a final assistant message (no tool
    // calls) directly after the block. Re-append that response and drop the
    // bad pair:  [.., user, finalText', hint].
    if (leaf?.type !== "message" || leafMsg?.role !== "assistant") return;
    const content = (leaf.message as { content?: unknown[] }).content;
    if (Array.isArray(content) && content.some((b) => (b as { type?: string })?.type === "toolCall")) return;
    if (leaf.parentId === null) return;
    const blockedResult = sm.getEntry(leaf.parentId);
    if (blockedResult?.type !== "message" || blockedResult.parentId === null) return;
    const blockedMsg = blockedResult.message as { role?: string; toolCallId?: string };
    if (blockedMsg.role !== "toolResult" || blockedMsg.toolCallId !== pending.toolCallId) return;
    const assistantEntry = sm.getEntry(blockedResult.parentId);
    if (!assistantEntry || assistantEntry.id !== pending.assistantEntryId) return;

    sm.branch(pending.assistantParentId);
    sm.appendMessage(leaf.message as unknown as AssistantMessage); // keep the model's response (new entry)
    hint();
  });

  // --------------------------------------------------------------------------
  // Features 4/5/6/7/8: react to tool results
  // --------------------------------------------------------------------------
  pi.on("tool_result", async (event: ToolResultEvent, ctx: ExtensionContext) => {
    const st = getState(ctx);
    const appended: TextBlock[] = [];

    // Feature 8: read on a directory → replace EISDIR error with a listing.
    if (event.toolName === "read" && event.isError) {
      const p = (event.input as { path?: unknown }).path;
      if (typeof p === "string" && p.length > 0) {
        const abs = resolvePath(ctx.cwd, p);
        let isDir = false;
        try {
          isDir = statSync(abs).isDirectory();
        } catch {
          isDir = false;
        }
        if (isDir) {
          try {
            const r = await pi.exec("ls", ["-F", abs], { cwd: ctx.cwd, timeout: 10_000 });
            if (r.code === 0) {
              return {
                content: [{ type: "text", text: `Directory listing of ${abs} (read converted to ls):\n${r.stdout}` }],
                details: event.details,
                isError: false,
              };
            }
          } catch {
            // keep the original error below
          }
        }
      }
    }

    // Feature 5: escalate the bash timeout default after a timeout failure.
    if (event.toolName === "bash" && event.isError) {
      if (/Command timed out after \d+ seconds/.test(extractText(event.content))) {
        st.bashTimeout = Math.min(BASH_TIMEOUT_MAX, st.bashTimeout + BASH_TIMEOUT_STEP);
      }
    }

    // Feature 4: first successful read inside a known skill directory.
    if (event.toolName === "read" && !event.isError) {
      const p = (event.input as { path?: unknown }).path;
      if (typeof p === "string" && p.length > 0) {
        const abs = resolvePath(ctx.cwd, p);
        const skill = findSkillForPath(st, abs);
        if (skill && !st.hintedSkills.has(skill.name)) {
          st.hintedSkills.add(skill.name);
          appended.push({ type: "text", text: skillHintText(skill) });
        }
      }
    }

    if (event.isError) {
      // Feature 6: tool definition hint (or available-tools list).
      const tool = pi.getAllTools().find((t) => t.name === event.toolName);
      if (tool) {
        appended.push({
          type: "text",
          text: `Tool '${event.toolName}' failed. Recall its definition:\n${renderToolYaml(tool, st.strictByTool.get(tool.name) ?? false)}`,
        });
      } else {
        const active = new Set(pi.getActiveTools());
        const list = pi
          .getAllTools()
          .filter((t) => active.has(t.name))
          .map((t) => `- ${t.name}: ${t.description}`)
          .join("\n");
        appended.push({ type: "text", text: `Unknown tool '${event.toolName}'. Available tools:\n${list}` });
      }

      // Feature 7: skill path hints on failed reads.
      if (event.toolName === "read") {
        const p = (event.input as { path?: unknown }).path;
        if (typeof p === "string" && p.length > 0) {
          const abs = resolvePath(ctx.cwd, p);
          const skill = findSkillForPath(st, abs);
          if (skill) {
            st.hintedSkills.add(skill.name); // count as "used": no hint loop
            appended.push({ type: "text", text: skillHintText(skill) });
          } else {
            const parts = abs.split(sep);
            const i = parts.lastIndexOf("skills");
            const name = i >= 0 && i < parts.length - 1 ? parts[i + 1] : null;
            if (name && !st.skills.some((s) => s.name === name)) {
              appended.push({
                type: "text",
                text: `Unknown skill '${name}'. Available skills:\n${renderSkillsYaml(knownSkills(st))}`,
              });
            }
          }
        }
      }
    }

    if (appended.length === 0) return;
    return {
      content: [...event.content, ...appended],
      details: event.details,
      isError: event.isError,
    };
  });

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------
  pi.on("session_shutdown", (_event, ctx: ExtensionContext) => {
    states.delete(ctx.sessionManager.getSessionId());
  });

  // --------------------------------------------------------------------------
  // Feature 1: destructive-write protection (write → deterministic edit)
  // --------------------------------------------------------------------------
  const template = createWriteToolDefinition(process.cwd());
  pi.registerTool({
    ...template,
    execute: async (toolCallId, params, signal, onUpdate, ctx) => {
      const cwd = ctx.cwd;
      const delegateWrite = () => createWriteToolDefinition(cwd).execute(toolCallId, params, signal, onUpdate, ctx);
      let existing: string | null = null;
      try {
        existing = readFileSync(resolvePath(cwd, params.path), "utf8");
      } catch {
        existing = null; // new file (or unreadable): plain write
      }
      if (existing === null) return delegateWrite();
      if (params.content === existing) {
        return {
          content: [{ type: "text", text: "No changes: the file already contains exactly the proposed content (write was a no-op)." }],
          details: undefined,
        };
      }
      const edits = diffToEdits(existing, params.content);
      if (
        edits &&
        edits.length > 0 &&
        edits.every((e) => countOccurrences(existing, e.oldText) === 1)
      ) {
        // Small, unique change: apply it through the built-in edit tool.
        const res = await createEditToolDefinition(cwd).execute(
          toolCallId,
          { path: params.path, edits },
          signal,
          onUpdate as AgentToolUpdateCallback<EditToolDetails | undefined> | undefined,
          ctx,
        );
        return res as unknown as AgentToolResult<undefined>;
      }
      // Huge change (or edit cannot satisfy it): real full overwrite.
      return delegateWrite();
    },
  });
}

/** Re-exported for unit tests (pure logic only). */
export const __slmTest = {
  yq,
  renderToolYaml,
  renderSkillsYaml,
  renderSkillHintYaml,
  countOccurrences,
  filterSurvivingEdits,
  diffToEdits,
  collectStrict,
  extractText,
};
export type { SlmEdit };
