/**
 * slm.ts — Small Language Model reliability extension for pi.
 *
 * Load with:  pi -e ./slm.ts
 *
 * Two features:
 *
 * 1. Minimal system prompt: `before_agent_start` replaces pi's full system
 *    prompt with a single short system message. Tools are NOT rendered into
 *    the prompt text: pi keeps passing them to the LLM as the structured
 *    `tools` array in the provider payload, so the model's chat template
 *    renders them.
 *
 * 2. Simulated dialogue at the start of every new session (ported from
 *    src/slm.ts — only the three listing/usage pairs, without the skill
 *    few-shot invocations). On the first prompt of a new session, three
 *    simulated user/assistant pairs are injected at the start:
 *
 *      1.  system message                 (the single sentence above)
 *      2.  user:      "What are available skills?"   (simulated)
 *      3.  assistant: skills YAML (loaded skills, single-line descriptions,
 *                      absolute paths of references/ and scripts/ files)
 *      4.  user:      "What are available tools?"    (simulated)
 *      5.  assistant: tools YAML (active tools, single-line descriptions,
 *                      full parameters schemas)
 *      6.  user:      "How does skill system work? ..." (simulated)
 *      7.  assistant: the <skill> block contract
 *      8.  user:      the first real user request
 *
 * If the active model supports reasoning (`model.reasoning === true`), each
 * synthetic assistant message carries one short synthetic reasoning line as
 * a real `thinking` content block. For the OpenAI Completions API (the
 * llama.cpp server path) that block is sent on the wire as a
 * `reasoning_content` field of the assistant message, while `content` stays
 * the pure answer (the YAML document or the contract).
 *
 * Mechanics (same as src/slm.ts):
 *  - `before_agent_start` fires after the user submits a prompt but before
 *    the agent loop. At that point the user message is not persisted yet,
 *    hence the new-session check: the session branch contains no user
 *    message.
 *  - The simulated user messages are injected with `pi.sendMessage()`
 *    (custom messages; no triggerTurn). They are persisted as
 *    custom_message entries and also enter the agent's in-memory state.
 *  - The synthetic assistant replies are persisted with
 *    `SessionManager.appendMessage()`, but they do not enter the agent's
 *    in-memory state (only pi.sendMessage() does). So the extension also
 *    subscribes to the `context` event, which fires on every LLM call with
 *    the full message array: it re-inserts each missing synthetic reply
 *    right after its simulated user message when it is not already there.
 *    On resumed/continued sessions the dialogue is restored from the
 *    session file into the agent state, the detection no-ops, and nothing
 *    is duplicated.
 *
 * Token economy: YAML strings are emitted as plain (unquoted) scalars
 * whenever YAML-safe; double quotes are used only as a correctness
 * fallback.
 */

import type {
	AssistantMessage,
	Model,
	TextContent,
	ThinkingContent,
	Usage,
} from "@earendil-works/pi-ai";
import type {
	BuildSystemPromptOptions,
	ExtensionAPI,
	SessionManager,
	Skill,
	ToolInfo,
} from "@earendil-works/pi-coding-agent";
import { readdirSync } from "node:fs";
import { join, resolve } from "node:path";

/** The entire system message: this one sentence, nothing else. */
const SYSTEM_MESSAGE =
	"You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.";

/** Custom message types for the three simulated user asks. */
const SKILLS_CUSTOM_TYPE = "available-skills";
const TOOLS_CUSTOM_TYPE = "available-tools";
const SKILLSYS_CUSTOM_TYPE = "skill-system";

const SKILLS_ASK = "What are available skills?";
const TOOLS_ASK = "What are available tools?";
const SKILLSYS_ASK =
	"How does skill system work? When a skill block is in my latest message, what do I do?";

/** Synthetic reply to the skill-system question (the <skill> block contract). */
const SKILLSYS_EXPLAIN =
	"I treat every <skill> block as a fresh, independent invocation. I identify the mode by reading the bare argument in the text immediately following the current block's closing tag. I reply with the exact short confirmation line required for that specific mode.";
const SKILLSYS_THINKING =
	"I treat each skill block as a fresh, self-contained call: I read only this block's name, Usage rules, and the argument text after its closing tag, and ignore everything before it. If a different mode is already active, I still derive the reply only from this argument — I never repeat the old confirmation. If the previous message ended with a short word from another finished skill, that word is not this argument; the argument is exactly the text after this block's closing tag. I never import the previous skill's rules, outputs, or state into this reply. My reply is always the single short confirmation line this skill's Usage requires — never empty, never extra.";

/** Collapse all whitespace runs to single spaces and trim. */
function oneLine(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

/**
 * Render a value as a YAML double-quoted scalar. Always valid YAML:
 * backslashes, quotes, and line breaks are escaped. Used only as a
 * fallback when a string is not safe as a plain (unquoted) scalar.
 */
function yamlScalar(value: string): string {
	return (
		'"' +
		value
			.replace(/\\/g, "\\\\")
			.replace(/"/g, "\\\"")
			.replace(/\n/g, "\\n")
			.replace(/\r/g, "\\r")
			.replace(/\t/g, "\\t") +
		'"'
	);
}

/**
 * True when the string can be emitted as a YAML plain (unquoted) scalar
 * in block context without changing its meaning. Quoting is the fallback
 * when any of these hold: empty, leading/trailing whitespace, control
 * chars, starts with an indicator char, contains ": " or ends with ":",
 * contains " #" (comment), or looks like a number/bool/null/date.
 */
function isPlainSafe(s: string): boolean {
	if (s === "") {
		return false;
	}
	if (/^[\s\u0000-\u001f]|\s$/.test(s)) {
		return false;
	}
	if (/[\n\r\t]/.test(s)) {
		return false;
	}
	if (/^[-?:&*!|>%@`"'#,[\]{}]/.test(s)) {
		return false;
	}
	if (/: /.test(s) || s.endsWith(":")) {
		return false;
	}
	if (/ #/.test(s)) {
		return false;
	}
	if (/^(null|~|true|false|yes|no|on|off)$/i.test(s)) {
		return false;
	}
	if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s)) {
		return false;
	}
	if (/^0[xob][0-9a-f]+$/i.test(s)) {
		return false;
	}
	if (/^\d{4}-\d{1,2}-\d{1,2}([Tt ].*)?$/.test(s)) {
		return false;
	}
	return true;
}

/** Token-saving string scalar: plain when safe, double-quoted otherwise. */
function yamlStr(value: string): string {
	return isPlainSafe(value) ? value : yamlScalar(value);
}

/** Single-line string scalar (spec: name/description fields). */
function yamlField(value: string): string {
	return yamlStr(oneLine(value));
}

/** Keys safe to emit bare (JSON-schema property names are identifiers). */
const SAFE_KEY = /^[A-Za-z_][A-Za-z0-9_./-]*$/;

function isSimple(value: unknown): boolean {
	return (
		value === null ||
		typeof value === "string" ||
		typeof value === "number" ||
		typeof value === "boolean"
	);
}

/**
 * Convert a JSON value to YAML (block style; flow style for scalar
 * arrays). Dependency-free: pi's tool `parameters` schemas are plain
 * JSON (TypeBox symbols are not enumerable string keys, so they never
 * appear here). Strings are emitted as plain scalars when YAML-safe
 * (token saving), double-quoted otherwise, so the output is valid YAML
 * for any input.
 */
function jsonToYaml(value: unknown, indent: number): string {
	if (value === null || value === undefined) {
		return "null";
	}
	const t = typeof value;
	if (t === "string") {
		return yamlStr(value as string);
	}
	if (t === "number" || t === "boolean") {
		return String(value);
	}
	if (Array.isArray(value)) {
		if (value.length === 0) {
			return "[]";
		}
		if (value.every(isSimple)) {
			return `[${value.map((v) => jsonToYaml(v, indent)).join(", ")}]`;
		}
		const pad = " ".repeat(indent);
		return value
			.map((v) => {
				const body = jsonToYaml(v, indent + 2);
				const nl = body.indexOf("\n");
				const head = nl === -1 ? body : body.slice(0, nl);
				const rest = nl === -1 ? "" : "\n" + body.slice(nl + 1);
				return pad + "- " + head.slice(indent + 2) + rest;
			})
			.join("\n");
	}
	if (t === "object") {
		const obj = value as Record<string, unknown>;
		const keys = Object.keys(obj);
		if (keys.length === 0) {
			return "{}";
		}
		const pad = " ".repeat(indent);
		return keys
			.map((k) => {
				const key = SAFE_KEY.test(k) ? k : yamlScalar(k);
				const body = jsonToYaml(obj[k], indent + 2);
				return body.includes("\n")
					? `${pad}${key}:\n${body}`
					: `${pad}${key}: ${body}`;
			})
			.join("\n");
	}
	return "null";
}

/**
 * List all files under dir as absolute paths (sorted, recursive).
 * Returns [] when dir does not exist.
 */
function listFiles(dir: string): string[] {
	const out: string[] = [];
	const walk = (d: string): void => {
		let entries;
		try {
			entries = readdirSync(d, { withFileTypes: true });
		} catch {
			return;
		}
		for (const entry of entries) {
			const full = join(d, entry.name);
			if (entry.isDirectory()) {
				walk(full);
			} else if (entry.isFile()) {
				out.push(resolve(full));
			}
		}
	};
	walk(dir);
	out.sort();
	return out;
}

/** One short synthetic reasoning line for the skills assistant message. */
function skillsThinking(count: number): string {
	return `I found ${count} skills. I will check whether the task matches a description, and if so read that skill's SKILL.md and the reference files listed below.`;
}

/** One short synthetic reasoning line for the tools assistant message. */
function toolsThinking(count: number): string {
	return `I found ${count} tools. I will pick the narrowest tool that fits the task.`;
}

/**
 * Build the available-skills YAML document (main content of the synthetic
 * assistant message answering "Available skills").
 *
 * Each skill entry: name, description (single line), references (absolute
 * paths of files under <skill dir>/references/), scripts (absolute paths
 * of files under <skill dir>/scripts/). Empty lists are `[]`.
 */
function buildSkillsYaml(skills: Skill[]): string {
	if (skills.length === 0) {
		return "No available skills: []";
	}
	const lines: string[] = ["Available skills are:"];
	for (const skill of skills) {
		lines.push(`  - name: ${yamlField(skill.name)}`);
		lines.push(`    description: ${yamlField(skill.description)}`);
		const references = listFiles(join(skill.baseDir, "references"));
		lines.push(`    references: ${references.length > 0 ? "" : "[]"}`.trimEnd());
		for (const p of references) {
			lines.push(`      - ${yamlStr(p)}`);
		}
		const scripts = listFiles(join(skill.baseDir, "scripts"));
		lines.push(`    scripts: ${scripts.length > 0 ? "" : "[]"}`.trimEnd());
		for (const p of scripts) {
			lines.push(`      - ${yamlStr(p)}`);
		}
	}
	lines.push("These are skills, not tools.");
	return lines.join("\n");
}

/**
 * Build the available-tools YAML document (main content of the synthetic
 * assistant message answering "Available tools").
 *
 * Each tool entry is the whole function signature:
 *   - name: <tool name>
 *   - description: <single line; the one-line system-prompt snippet when
 *     available, else the flattened tool description>
 *   - parameters: <the tool's parameters JSON schema converted to YAML,
 *     with all params and types; null when the tool has no parameters>
 */
function buildToolsYaml(
	names: string[],
	snippets: Record<string, string> | undefined,
	allTools: ToolInfo[],
): string {
	if (names.length === 0) {
		return "No available tools: []";
	}
	const byName = new Map<string, ToolInfo>(allTools.map((t) => [t.name, t]));
	const lines: string[] = ["Available tools are:"];
	for (const name of names) {
		const tool = byName.get(name);
		const description = snippets?.[name] ?? oneLine(tool?.description ?? "");
		lines.push(`  - name: ${yamlField(name)}`);
		lines.push(`    description: ${yamlField(description)}`);
		if (tool && tool.parameters !== undefined) {
			const paramsYaml = jsonToYaml(tool.parameters, 6);
			lines.push(
				paramsYaml.includes("\n")
					? `    parameters:\n${paramsYaml}`
					: `    parameters: ${paramsYaml}`,
			);
		} else {
			lines.push("    parameters: null");
		}
	}
	lines.push("");
	lines.push("These are tools, not skills.");
	return lines.join("\n");
}

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

/**
 * Build a synthetic assistant message: a short text answer (the YAML
 * document or the contract) preceded, when the model supports reasoning,
 * by one short synthetic thinking block (it rides on the wire as
 * `reasoning_content` for the OpenAI Completions API).
 */
function makeSynthAssistant(
	text: string,
	thinking: string | undefined,
	model: Model<any>,
): AssistantMessage {
	const content: (TextContent | ThinkingContent)[] = [];
	if (thinking !== undefined) {
		const block: ThinkingContent = { type: "thinking", thinking };
		// OpenAI Completions serializers (pi's openai-completions.js) emit
		// `assistantMsg[signature] = <thinking text>` for thinking blocks
		// that carry a signature — the documented replay path for
		// llama.cpp server. Content stays the pure answer.
		if (model.api === "openai-completions") {
			block.thinkingSignature = "reasoning_content";
		}
		content.push(block);
	}
	content.push({ type: "text", text });
	return {
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: ZERO_USAGE,
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

/**
 * True when the (restored or live) message is the synthetic assistant
 * reply we injected: it shares one of our content blocks (thinking text
 * or answer text).
 */
function isSyntheticReply(m: unknown, expected: AssistantMessage): boolean {
	if (typeof m !== "object" || m === null) {
		return false;
	}
	const msg = m as { role?: string; content?: unknown };
	if (msg.role !== "assistant" || !Array.isArray(msg.content)) {
		return false;
	}
	const blocks = msg.content as Array<Record<string, unknown>>;
	for (const block of expected.content) {
		if (
			block.type === "thinking" &&
			blocks.some((b) => b.type === "thinking" && b.thinking === (block as ThinkingContent).thinking)
		) {
			return true;
		}
		if (
			block.type === "text" &&
			blocks.some((b) => b.type === "text" && b.text === (block as TextContent).text)
		) {
			return true;
		}
	}
	return false;
}

export default function slm(pi: ExtensionAPI) {
	// In-memory state for the session this extension injected into.
	const state: {
		sessionId: string | undefined;
		// The simulated user messages' custom types, in dialogue order.
		customTypes: string[];
		// Parallel to customTypes: the synthetic assistant reply that must
		// follow each simulated user message in the LLM context. Built per
		// session (carries the active model's metadata).
		replies: AssistantMessage[];
	} = {
		sessionId: undefined,
		customTypes: [],
		replies: [],
	};

	// ------------------------------------------------------------------
	// System prompt + injection: first prompt of a new session.
	// ------------------------------------------------------------------
	pi.on("before_agent_start", (event, ctx) => {
		const result: { systemPrompt: string } = { systemPrompt: SYSTEM_MESSAGE };
		if (!ctx.model) {
			return result;
		}
		// Only the first prompt of a new session gets the reminder.
		// At before_agent_start time the user message is not persisted yet
		// (startup entries such as model_change may already exist), so the
		// check is: no user message anywhere in the session branch.
		const hasUserMessage = ctx.sessionManager
			.getBranch()
			.some((entry) => entry.type === "message" && entry.message.role === "user");
		if (hasUserMessage) {
			return result;
		}

		const opts: BuildSystemPromptOptions = event.systemPromptOptions;
		const reasoning = ctx.model.reasoning === true;

		// Skills — the same loaded set the system prompt is built from,
		// minus skills hidden from the model (disableModelInvocation).
		const skills = (opts.skills ?? []).filter((s) => !s.disableModelInvocation);
		const skillsYaml = buildSkillsYaml(skills);

		// Tools — the active tool set, with one-line descriptions.
		const activeTools = opts.selectedTools ?? pi.getActiveTools();
		const toolsYaml = buildToolsYaml(activeTools, opts.toolSnippets, pi.getAllTools());

		// The three simulated user asks (shown in the TUI).
		const plan: Array<{ customType: string; userContent: string }> = [
			{ customType: SKILLS_CUSTOM_TYPE, userContent: SKILLS_ASK },
			{ customType: TOOLS_CUSTOM_TYPE, userContent: TOOLS_ASK },
			{ customType: SKILLSYS_CUSTOM_TYPE, userContent: SKILLSYS_ASK },
		];

		// The synthetic assistant reply for each simulated ask.
		state.sessionId = ctx.sessionManager.getSessionId();
		state.customTypes = plan.map((seg) => seg.customType);
		state.replies = [
			makeSynthAssistant(skillsYaml, reasoning ? skillsThinking(skills.length) : undefined, ctx.model),
			makeSynthAssistant(toolsYaml, reasoning ? toolsThinking(activeTools.length) : undefined, ctx.model),
			makeSynthAssistant(SKILLSYS_EXPLAIN, reasoning ? SKILLSYS_THINKING : undefined, ctx.model),
		];

		// Persist the simulated dialogue in order (per segment: the
		// simulated user message, then its synthetic reply).
		// pi.sendMessage() (no triggerTurn) appends the custom message to
		// the session and to the agent state synchronously (its
		// non-streaming path contains no awaits), so the following
		// appendMessage() calls continue the session tree right after it.
		// The synthetic assistant message is persisted through the
		// SessionManager itself: ctx.sessionManager is typed as a read-only
		// pick, but the runtime object is the full SessionManager instance,
		// whose appendMessage() is the same method the core uses.
		const sm = ctx.sessionManager as unknown as SessionManager;
		for (let i = 0; i < plan.length; i++) {
			pi.sendMessage({
				customType: plan[i].customType,
				content: plan[i].userContent,
				display: true,
			});
			sm.appendMessage(state.replies[i]);
		}
		return result;
	});

	// ------------------------------------------------------------------
	// LLM context: guarantee the synthetic replies are present (on every
	// provider call) right after their simulated user messages.
	//
	// Why this is needed: the persisted entries above are not in the
	// agent's in-memory state (only pi.sendMessage() adds to it), and the
	// in-memory state is what gets sent to the provider. For resumed
	// sessions the state is restored from the session file, already
	// containing the full dialogue — the per-session check below then
	// no-ops and nothing is duplicated.
	// ------------------------------------------------------------------
	pi.on("context", (event, ctx) => {
		try {
			if (
				state.sessionId === undefined ||
				state.sessionId !== ctx.sessionManager.getSessionId()
			) {
				return;
			}
			if (state.customTypes.length === 0 || state.replies.length !== state.customTypes.length) {
				return;
			}
			const msgs = event.messages;
			const expected = new Map<string, AssistantMessage>(
				state.customTypes.map((customType, i) => [customType, state.replies[i]]),
			);
			const hasAsk = msgs.some(
				(m) => m.role === "custom" && expected.has(m.customType),
			);
			if (!hasAsk) {
				return;
			}
			const out = [...msgs];
			const inserts: Array<[number, AssistantMessage]> = [];
			for (let i = 0; i < msgs.length; i++) {
				const m = msgs[i];
				if (m.role !== "custom") {
					continue;
				}
				const reply = expected.get(m.customType);
				if (!reply) {
					continue;
				}
				// Whether the expected reply already sits right after the
				// simulated user message (restored sessions have it; fresh
				// runs don't).
				const present = isSyntheticReply(msgs[i + 1], reply);
				if (!present) {
					inserts.push([i + 1, reply]);
				}
			}
			if (inserts.length > 0) {
				// splice backwards so earlier indices stay valid
				for (let k = inserts.length - 1; k >= 0; k--) {
					out.splice(inserts[k][0], 0, inserts[k][1]);
				}
				return { messages: out };
			}
		} catch {
			// transformContext contract: never throw; leave context unchanged.
		}
	});
}
