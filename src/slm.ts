/**
 * slm — Small Language Model reliability extension for pi.
 *
 * Load with:  pi -e src/slm.ts
 *
 * Single file, no dependencies — only pi's internal TypeScript API is used
 * (extension events, session manager, built-in tool factories, model/tool
 * state). All state is per-session and in-memory. Works in all modes
 * (interactive TUI, -p, --mode json, --mode rpc).
 *
 * Implements the features:
 *
 * Feature 1: Available skills and tools reminder.
 *   Small language models frequently forget which skills/tools are available
 *   when that information only lives in the system prompt. On the first
 *   prompt of a new session, this extension inserts two compact synthetic
 *   messages (custom messages, hidden from the TUI) immediately before the
 *   user message, so the conversation context itself reminds the model:
 *
 *     1. "Available skills:" — valid YAML list of the loaded skills with
 *        name, single-line description, absolute reference file paths
 *        (skill dir `references/`), and absolute script file paths
 *        (skill dir `scripts/`). Skills mirror what the system prompt
 *        exposes (see system-prompt.ts / formatSkillsForPrompt): loaded
 *        skills minus those with disable-model-invocation.
 *     2. "Available tools:" — valid YAML list of the active tools with
 *        name, single-line description (the one-line system-prompt
 *        snippet), and the whole function signature: the tool's
 *        `parameters` JSON schema (from pi.getAllTools()) converted to
 *        YAML, with all params and their types.
 *
 *   If the active model supports reasoning (`model.reasoning === true`),
 *   each synthetic message carries one short synthetic reasoning line
 *   (`# thinking: ...`, a YAML comment so the document stays valid).
 *
 *   Mechanics: the hook is `before_agent_start`, which fires after the user
 *   submits a prompt but before the agent loop starts. At that point the
 *   user message is not persisted yet (startup bookkeeping entries like
 *   model_change/thinking_level_change may exist, so entry count is not a
 *   reliable signal), hence the check: the session branch contains no user
 *   message yet => this is the first request of a new session.
 *   `pi.sendMessage()` (no triggerTurn) appends each synthetic message to
 *   both the agent state and the session tree synchronously, hence the
 *   final order in the session and in the LLM context is:
 *   [skills message, tools message, user message]. Subsequent prompts (and
 *   resumed/forked sessions, which already have user messages) never get
 *   another injection.
 */

import type {
	BuildSystemPromptOptions,
	ExtensionAPI,
	Skill,
	ToolInfo,
} from "@earendil-works/pi-coding-agent";
import { readdirSync } from "node:fs";
import { join, resolve } from "node:path";

const SKILLS_CUSTOM_TYPE = "slm-skills";
const TOOLS_CUSTOM_TYPE = "slm-tools";

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

/** One short synthetic reasoning line for the skills message. */
function skillsThinking(count: number): string {
	return `thinking: scanned loaded skills - ${count} found. I will check whether the task matches a description, and if so read that skill's SKILL.md and the reference files listed below.`;
}

/** One short synthetic reasoning line for the tools message. */
function toolsThinking(count: number): string {
	return `thinking: scanned active tools - ${count} found. I will pick the narrowest tool that fits the task.`;
}

/**
 * Build the "Available skills:" message content.
 *
 * Layout (one YAML document):
 *   Available skills:
 *   # thinking: ...            (only for reasoning models)
 *   skills: []                 (or a list of skill entries)
 *
 * Each skill entry: name, description (single line), references (absolute
 * paths of files under <skill dir>/references/), scripts (absolute paths of
 * files under <skill dir>/scripts/). Empty lists are `[]`.
 */
function buildSkillsMessage(skills: Skill[], reasoning: boolean): string {
	const lines: string[] = ["Available skills:"];
	if (reasoning) {
		lines.push(`# ${skillsThinking(skills.length)}`);
	}
	if (skills.length === 0) {
		lines.push("skills: []");
		return lines.join("\n");
	}
	lines.push("skills:");
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
	return lines.join("\n");
}

/**
 * Build the "Available tools:" message content.
 *
 * Layout (one YAML document):
 *   Available tools:
 *   # thinking: ...            (only for reasoning models)
 *   tools: []                  (or a list of tool entries)
 *
 * Each tool entry is the whole function signature:
 *   - name: <tool name>
 *   - description: <single line; the one-line system-prompt snippet when
 *     available, else the flattened tool description>
 *   - parameters: <the tool's parameters JSON schema converted to YAML,
 *     with all params and types; null when the tool has no parameters>
 */
function buildToolsMessage(
	names: string[],
	snippets: Record<string, string> | undefined,
	allTools: ToolInfo[],
	reasoning: boolean,
): string {
	const byName = new Map<string, ToolInfo>(allTools.map((t) => [t.name, t]));
	const lines: string[] = ["Available tools:"];
	if (reasoning) {
		lines.push(`# ${toolsThinking(names.length)}`);
	}
	if (names.length === 0) {
		lines.push("tools: []");
		return lines.join("\n");
	}
	lines.push("tools:");
	for (const name of names) {
		const tool = byName.get(name);
		const description =
			snippets?.[name] ?? oneLine(tool?.description ?? "");
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
	return lines.join("\n");
}

export default function slmExtension(pi: ExtensionAPI) {
	pi.on("before_agent_start", (event, ctx) => {
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
		const reasoning = ctx.model?.reasoning === true;

		// 1) Skills — the same loaded set the system prompt is built from,
		//    minus skills hidden from the model (disableModelInvocation).
		const skills = (opts.skills ?? []).filter((s) => !s.disableModelInvocation);
		pi.sendMessage({
			customType: SKILLS_CUSTOM_TYPE,
			content: buildSkillsMessage(skills, reasoning),
			display: false,
		});

		// 2) Tools — the active tool set, with one-line descriptions.
		const activeTools = opts.selectedTools ?? pi.getActiveTools();
		pi.sendMessage({
			customType: TOOLS_CUSTOM_TYPE,
			content: buildToolsMessage(activeTools, opts.toolSnippets, pi.getAllTools(), reasoning),
			display: false,
		});
	});
}
