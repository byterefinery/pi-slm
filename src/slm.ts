/**
 * slm — Small Language Model reliability extension for pi.
 *
 * Load with:  pi -e src/slm.ts
 *
 * Single file, no dependencies — only pi's internal TypeScript API is used
 * (extension events, session manager, model/tool state) plus node builtins.
 * All extension state is in-memory. Works in all modes (interactive TUI,
 * -p, --mode json, --mode rpc).
 *
 * Feature 1: Available skills and tools reminder (simulated dialogue).
 *
 * Small language models frequently forget which skills/tools are available
 * when that information only lives in the system prompt. On the first prompt
 * of a new session, this extension simulates a short user/assistant dialogue
 * at the very beginning of the conversation, so the context itself reminds
 * the model:
 *
 *   1. system message                (pi default, untouched)
 *   2. user:       "Available skills"  (simulated)
 *   3. assistant:  short synthetic thinking + available skills as YAML
 *   4. user:       "Available tools"   (simulated)
 *   5. assistant:  short synthetic thinking + available tools as YAML
 *   6. user:       "How can a skill be used?" (simulated)
 *   7. assistant:  short synthetic thinking + a one-shot example of a skill
 *                  invocation (a generic "example" skill whose SKILL.md
 *                  lists example script/command usages with arguments, plus
 *                  the assistant picking the matching usage and running it
 *                  through the bash tool)
 *   8. user:       the first real user request
 *
 * The skills YAML mirrors what the system prompt exposes (see
 * system-prompt.ts / formatSkillsForPrompt): the loaded skills minus those
 * with disable-model-invocation, each with name, single-line description,
 * and absolute paths of the skill dir's references/ and scripts/ files.
 * The tools YAML lists the active tools with name, single-line description
 * (the one-line system-prompt snippet when available), and the whole
 * function signature: the tool's `parameters` JSON schema (from
 * pi.getAllTools()) converted to YAML, with all params and their types.
 *
 * If the active model supports reasoning (`model.reasoning === true`),
 * each synthetic assistant message carries one short synthetic reasoning
 * line as a real `thinking` content block. For the OpenAI Completions API
 * (the llama.cpp server path) that block is sent on the wire as a
 * `reasoning_content` field of the assistant message — the standard
 * OpenAI-compatible way to carry reasoning in chat history — while
 * `content` stays the answer (the YAML document for skills/tools, the
 * one-shot example text for skill-usage). For other APIs no signature is
 * set: the block stays in the session/TUI and is replayed or dropped by
 * the provider's serializer as usual.
 *
 * Token economy: YAML strings are emitted as plain (unquoted) scalars
 * whenever YAML-safe; double quotes are used only as a correctness
 * fallback.
 *
 * Mechanics:
 *  - `before_agent_start` fires after the user submits a prompt but before
 *    the agent loop. At that point the user message is not persisted yet
 *    (startup bookkeeping entries like model_change/thinking_level_change
 *    may exist, so entry count is not a reliable signal), hence the
 *    new-session check: the session branch contains no user message.
 *  - The simulated user messages are injected with `pi.sendMessage()`
 *    (custom messages, display: true; no triggerTurn). They are persisted
 *    as custom_message entries and also enter the agent state.
 *  - The synthetic assistant messages are persisted with
 *    `SessionManager.appendMessage()` (the runtime object behind
 *    ctx.sessionManager is the full SessionManager, although the public
 *    context type only exposes its read-only pick).
 *  - Persisted entries alone are not enough for the *current* run: the
 *    agent's in-memory state (what the LLM actually sees) is built
 *    separately, and only pi.sendMessage() adds to it. So the extension
 *    also subscribes to the official `context` event (transformContext),
 *    which fires on every LLM call with the full message array: it
 *    re-inserts the three synthetic assistant messages right after each
 *    simulated user message when they are missing from the live state.
 *    On resumed/continued sessions the dialogue is restored from the
 *    session file into the agent state, the detection below no-ops, and
 *    nothing is duplicated.
 *
 * Display: the simulated user messages are shown in the TUI (custom
 * message styling); the synthetic assistant messages are part of the
 * session file, so they render when the session is loaded/reopened (they
 * have no live streaming events in the run they were injected in).
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

const SKILLS_CUSTOM_TYPE = "available-skills";
const TOOLS_CUSTOM_TYPE = "available-tools";
const USAGE_CUSTOM_TYPE = "skill-usage";
const SKILLS_ASK = "What are available skills?";
const TOOLS_ASK = "What are available tools?";
const USAGE_ASK = "How can a skill be used?";

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
 *
 * The first line of the result starts at `indent` spaces; continuation
 * lines are indented consistently.
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

/** One short synthetic reasoning line for the skill-usage assistant message. */
function usageThinking(): string {
	return `I will show one example: the user invokes a skill with /skill:<name>, pi puts its SKILL.md in a <skill> block of the user message, and I run the matching example command with the bash tool.`;
}

/**
 * One-shot example answering "How can a skill be used?": a generic
 * "example" skill (deliberately a name that collides with no real or
 * popular skill) whose SKILL.md lists harmless example invocations — a
 * bash script with a positional arg and a python script with a positional
 * arg plus a flag — and the assistant picking the usage that matches the
 * task, resolving the script path from the skill dir, and running it with
 * the bash tool. This is the pattern the SLM must follow whenever a real
 * <skill> block arrives in a user message.
 */
const SKILL_USAGE_EXAMPLE = [
	"A skill is invoked by the user with /skill:<name> [task]",
	"",
	"/skill:example List the files in '/data'.",
	"",
	"and pi expands it into a user message with the skill's SKILL.md body in a <skill> block:",
	"```",
	'<skill name="example" location="/home/user/project/.agent/skills/example/SKILL.md">',
	"",
	"# example",
	"",
	"## Usage",
	"",
	"\`\`\`bash",
	"example.sh DIR           # list the files in DIR",
	"example.py DIR --json    # count the files in DIR, as JSON",
	"\`\`\`",
	"</skill>",
	"List the files in /data.",
	"```",
  "The text after the </skill> block is the task - a sentence or bare arguments like a URL or a search query.",
  "I perform that task, I do not explain the skill.",
	"If the <skill> block shows script or command usages, I identify the right one for the task, resolve its path from the skill dir (dirname of location), and run it with the `bash` tool:",
	"```bash",
  "/home/user/project/.agent/skills/example/scripts/example.sh '/data'",
	"```",
].join("\n");

/**
 * Build the available-skills YAML document (main content of the synthetic
 * assistant message answering "Available skills").
 *
 *   skills: []                 (or a list of skill entries)
 *
 * Each skill entry: name, description (single line), references (absolute
 * paths of files under <skill dir>/references/), scripts (absolute paths of
 * files under <skill dir>/scripts/). Empty lists are `[]`.
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
 *   tools: []                  (or a list of tool entries)
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
 * Build the synthetic assistant message answering an slm ask. Main content
 * is the YAML document; when the model supports reasoning a short synthetic
 * thinking block precedes it.
 */
function makeSynthAssistant(
	yaml: string,
	thinking: string | undefined,
	model: Model<any>,
): AssistantMessage {
	const content: (TextContent | ThinkingContent)[] = [];
	if (thinking !== undefined) {
		const block: ThinkingContent = { type: "thinking", thinking };
		// OpenAI Completions serializers (pi's openai-completions.js) emit
		// `assistantMsg[signature] = <thinking text>` for thinking blocks
		// that carry a signature — the documented replay path for
		// llama.cpp server. Content stays the pure YAML document.
		if (model.api === "openai-completions") {
			block.thinkingSignature = "reasoning_content";
		}
		content.push(block);
	}
	content.push({ type: "text", text: yaml });
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

/** True when the (restored) message is the synthetic assistant we injected. */
function isOurAssistant(
	m: unknown,
	ours: AssistantMessage,
): boolean {
	if (typeof m !== "object" || m === null) {
		return false;
	}
	const msg = m as { role?: string; content?: unknown };
	if (msg.role !== "assistant" || !Array.isArray(msg.content)) {
		return false;
	}
	const ourText = ours.content.find((b) => b.type === "text") as
		| TextContent
		| undefined;
	const ourThinking = ours.content.find((b) => b.type === "thinking") as
		| ThinkingContent
		| undefined;
	for (const block of msg.content as Array<Record<string, unknown>>) {
		if (
			ourThinking &&
			block.type === "thinking" &&
			block.thinking === ourThinking.thinking
		) {
			return true;
		}
		if (ourText && block.type === "text" && block.text === ourText.text) {
			return true;
		}
	}
	return false;
}

export default function slmExtension(pi: ExtensionAPI) {
	// In-memory state for the session this extension injected into.
	const state: {
		sessionId: string | undefined;
		skillsAssistant: AssistantMessage | undefined;
		toolsAssistant: AssistantMessage | undefined;
		usageAssistant: AssistantMessage | undefined;
	} = {
		sessionId: undefined,
		skillsAssistant: undefined,
		toolsAssistant: undefined,
		usageAssistant: undefined,
	};

	// ------------------------------------------------------------------
	// Injection: first prompt of a new session.
	// ------------------------------------------------------------------
	pi.on("before_agent_start", (event, ctx) => {
		if (!ctx.model) {
			return;
		}
		// Only the first prompt of a new session gets the reminder.
		// At before_agent_start time the user message is not persisted yet
		// (startup entries such as model_change may already exist), so the
		// check is: no user message anywhere in the session branch.
		const hasUserMessage = ctx.sessionManager
			.getBranch()
			.some((entry) => entry.type === "message" && entry.message.role === "user");
		if (hasUserMessage) {
			return;
		}

		const opts: BuildSystemPromptOptions = event.systemPromptOptions;
		const reasoning = ctx.model.reasoning === true;

		// Skills — the same loaded set the system prompt is built from,
		// minus skills hidden from the model (disableModelInvocation).
		const skills = (opts.skills ?? []).filter((s) => !s.disableModelInvocation);
		const skillsYaml = buildSkillsYaml(skills);

		// Tools — the active tool set, with one-line descriptions.
		const activeTools = opts.selectedTools ?? pi.getActiveTools();
		const toolsYaml = buildToolsYaml(
			activeTools,
			opts.toolSnippets,
			pi.getAllTools(),
		);

		state.sessionId = ctx.sessionManager.getSessionId();
		state.skillsAssistant = makeSynthAssistant(
			skillsYaml,
			reasoning ? skillsThinking(skills.length) : undefined,
			ctx.model,
		);
		state.toolsAssistant = makeSynthAssistant(
			toolsYaml,
			reasoning ? toolsThinking(activeTools.length) : undefined,
			ctx.model,
		);
		state.usageAssistant = makeSynthAssistant(
			SKILL_USAGE_EXAMPLE,
			reasoning ? usageThinking() : undefined,
			ctx.model,
		);

		// Persist the simulated dialogue in order:
		//   user "Available skills"     -> assistant skills YAML
		//   user "Available tools"      -> assistant tools YAML
		//   user "How can a skill be used?" -> assistant one-shot example
		// pi.sendMessage() (no triggerTurn) appends the custom message to
		// the session and to the agent state synchronously (its
		// non-streaming path contains no awaits), so the following
		// appendMessage() calls continue the session tree right after it.
		// The synthetic assistant messages are persisted through the
		// SessionManager itself: ctx.sessionManager is typed as a read-only
		// pick, but the runtime object is the full SessionManager instance,
		// whose appendMessage() is the same method the core uses.
		const sm = ctx.sessionManager as unknown as SessionManager;
		pi.sendMessage({
			customType: SKILLS_CUSTOM_TYPE,
			content: SKILLS_ASK,
			display: true,
		});
		sm.appendMessage(state.skillsAssistant);
		pi.sendMessage({
			customType: TOOLS_CUSTOM_TYPE,
			content: TOOLS_ASK,
			display: true,
		});
		sm.appendMessage(state.toolsAssistant);
		pi.sendMessage({
			customType: USAGE_CUSTOM_TYPE,
			content: USAGE_ASK,
			display: true,
		});
		sm.appendMessage(state.usageAssistant);
	});

	// ------------------------------------------------------------------
	// LLM context: guarantee the synthetic assistant messages are present
	// (on every provider call) right after their simulated user messages.
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
			const msgs = event.messages;
			// The simulated asks and the synthetic assistant reply each maps
			// to (skills, tools, skill-usage).
			const byType = new Map<string, AssistantMessage | undefined>([
				[SKILLS_CUSTOM_TYPE, state.skillsAssistant],
				[TOOLS_CUSTOM_TYPE, state.toolsAssistant],
				[USAGE_CUSTOM_TYPE, state.usageAssistant],
			]);
			const hasAsk = msgs.some(
				(m) => m.role === "custom" && byType.has(m.customType),
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
				const ours = byType.get(m.customType);
				if (!ours) {
					continue;
				}
				if (isOurAssistant(msgs[i + 1], ours)) {
					continue;
				}
				inserts.push([i, ours]);
			}
			if (inserts.length > 0) {
				// splice backwards so earlier indices stay valid
				for (let k = inserts.length - 1; k >= 0; k--) {
					out.splice(inserts[k][0] + 1, 0, inserts[k][1]);
				}
				return { messages: out };
			}
		} catch {
			// transformContext contract: never throw; leave context unchanged.
		}
	});
}
