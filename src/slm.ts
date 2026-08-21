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
 * Feature: simulated dialogue at the start of every new session.
 *
 * Small language models forget which skills/tools are available and how
 * pi's skill system works when that information only lives in the system
 * prompt. On the first prompt of a new session, this extension simulates a
 * short user/assistant dialogue at the very beginning of the conversation,
 * so the context itself reminds the model:
 *
 *   1.  system message                (pi default, untouched)
 *   2.  user:       "What are available skills?"   (simulated)
 *   3.  assistant:  short synthetic thinking + available skills as YAML
 *   4.  user:       "What are available tools?"    (simulated)
 *   5.  assistant:  short synthetic thinking + available tools as YAML
 *   6.  user:       "How does skill system work? When a skill block is in
 *                   my latest message, what do I do?" (simulated)
 *   7.  assistant:  the <skill> block contract
 *   8.  user:       <skill> block, no argument          (simulated)
 *   9.  assistant:  the exact fixed reply the skill requires
 *   10. user:       <skill> block + "Hello"             (simulated)
 *   11. assistant:  read toolCall (skill reference file)
 *   12. tool result: the reference file content
 *   13. assistant:  the exact reply the reference file requires
 *   14. user:       <skill> block + "Hi"                (simulated)
 *   15. assistant:  bash toolCall (skill script)
 *   16. tool result: the script output
 *   17. assistant:  the script output report
 *   18. user:       the first real user request
 *
 * Steps 8-17 are a few-shot of three real skill invocations. The user
 * messages are the exact <skill> blocks pi expands /skill:example into
 * (pi's _expandSkillCommand shape: tag, "References are relative to" line,
 * SKILL.md body without frontmatter, the argument after the block), rooted
 * at the session's current working directory (<cwd>/.agents/skills/example/
 * ...) — pi's standard project skill location. The paths inside the few-shot
 * (the block's location attribute, the read path, the bash command) are the
 * session's real absolute paths, so the SLM derives the absolute paths of a
 * real incoming block by the same rule it just saw applied: the skill dir
 * from the location attribute, scripts and references under it. The
 * invocations cover the skill's three Usage branches: no argument (fixed
 * reply), "Hello" (read the reference file, then reply exactly what it
 * says), and any other text (run the skill's script with the text as CLI
 * parameters and report the output). The few-shot teaches the full loop:
 * <skill> block in, perform (fixed reply, read, or bash), result out.
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
 * contract/fixed replies/script report for the skill few-shot). For other
 * APIs no signature is set: the block stays in the session/TUI and is
 * replayed or dropped by the provider's serializer as usual.
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
 *    (custom messages; no triggerTurn). They are persisted as
 *    custom_message entries and also enter the agent state. The three
 *    bulky <skill> block invocations use display: false so they stay in
 *    the session file and the LLM context without flooding the TUI.
 *  - The synthetic assistant and tool-result messages are persisted with
 *    `SessionManager.appendMessage()` (the runtime object behind
 *    ctx.sessionManager is the full SessionManager, although the public
 *    context type only exposes its read-only pick).
 *  - Persisted entries alone are not enough for the *current* run: the
 *    agent's in-memory state (what the LLM actually sees) is built
 *    separately, and only pi.sendMessage() adds to it. So the extension
 *    also subscribes to the official `context` event (transformContext),
 *    which fires on every LLM call with the full message array: it
 *    re-inserts the missing synthetic replies (assistant messages and
 *    tool results) right after each simulated user message when they are
 *    not already there. On resumed/continued sessions the dialogue is
 *    restored from the session file into the agent state, the detection
 *    below no-ops, and nothing is duplicated.
 *
 * Display: the simulated user asks are shown in the TUI (custom message
 * styling); the synthetic assistant messages and tool results are part of
 * the session file, so they render when the session is loaded/reopened
 * (they have no live streaming events in the run they were injected in).
 */

import type {
	AssistantMessage,
	Model,
	TextContent,
	ThinkingContent,
	ToolCall,
	ToolResultMessage,
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
const SKILLSYS_CUSTOM_TYPE = "skill-system";
const SKILL_EXAMPLE_CUSTOM_TYPE = "skill-example-plain";
const SKILL_HELLO_CUSTOM_TYPE = "skill-example-hello";
const SKILL_SCRIPT_CUSTOM_TYPE = "skill-example-script";
const SKILLS_ASK = "What are available skills?";
const TOOLS_ASK = "What are available tools?";
const SKILLSYS_ASK =
	"How does skill system work? When a skill block is in my latest message, what do I do?";

/** SKILL.md body of the "example" skill (frontmatter stripped, trimmed). */
const EXAMPLE_SKILL_BODY = [
	"# example",
	"",
	"A minimal skill that demonstrates the shape of an agent skill without doing anything special.",
	"",
	"## Overview",
	"",
	"`example` exists to show what a skill is made of — frontmatter metadata, a SKILL.md body, a helper script in `scripts/`, and on-demand docs in `references/` — with the least possible behavior. Its only jobs are replying with a fixed message or running `example.sh`.",
	"",
	"## Usage",
	"",
	"- **Invoked with no extra text** — reply exactly `This is an example skill.` Nothing else. No explanations, no script.",
	"- **Invoked with \"Hello\"** — do not guess the response. Load [03-hello](references/03-hello.md) and follow it exactly; the instructions for this case live there and nowhere else.",
	"- **Invoked with any other text, or asked to \"call script\"** — pass the user's text (if any) as CLI parameters to `example.sh` and report the output; running with no parameters is fine:",
	"",
	"  ```bash",
	"  bash scripts/example.sh any given text",
	"  ```",
	"",
	"Script paths are relative to this skill's directory. The script echoes a fixed line regardless of the parameters; the parameters only demonstrate how the agent forwards input to a script.",
	"",
	"## References",
	"",
	"- [01-structure](references/01-structure.md) — What each part of this skill is for",
	"- [02-invocation](references/02-invocation.md) — How invocation and script calls behave",
	"- [03-hello](references/03-hello.md) — Full instructions for the \"Invoked with 'Hello'\" scenario",
].join("\n");

/**
 * The exact <skill> block pi expands /skill:example into
 * (_expandSkillCommand: tag, "References are relative to" line, body,
 * closing tag), rooted at skillDir (<cwd>/.agents/skills/example — pi's
 * standard project skill location). The argument, when present, is
 * appended after the block separated by a single newline.
 */
function buildExampleSkillBlock(skillDir: string): string {
	return [
		`<skill name="example" location="${skillDir}/SKILL.md">`,
		`References are relative to ${skillDir}.`,
		EXAMPLE_SKILL_BODY,
		"</skill>",
	].join("\n");
}

/**
 * Synthetic reply to the skill-system question (the <skill> block contract).
 * GEPA-optimized pair, trained and validated in optim/skills-usage-4 (train.py,
 * step1-pair-optimized.json): final min-of-3 accuracy 1.000 with 0% cross-skill
 * CoT leak on the 37-item bare-argument matrix, vs the previous V2 pair
 * (0.899 / 8%). The question (SKILLSYS_ASK) is unchanged.
 */
const SKILLSYS_EXPLAIN =
	"I treat every <skill> block as a fresh, independent invocation. I identify the mode by reading the bare argument in the text immediately following the current block's closing tag. I reply with the exact short confirmation line required for that specific mode.";
const SKILLSYS_THINKING =
	"I treat each skill block as a fresh, self-contained call: I read only this block's name, Usage rules, and the argument text after its closing tag, and ignore everything before it. If a different mode is already active, I still derive the reply only from this argument — I never repeat the old confirmation. If the previous message ended with a short word from another finished skill, that word is not this argument; the argument is exactly the text after this block's closing tag. I never import the previous skill's rules, outputs, or state into this reply. My reply is always the single short confirmation line this skill's Usage requires — never empty, never extra.";

/** Few-shot 1 (no argument): the skill's required fixed reply. */
const SKILL_EXAMPLE_ANSWER = "This is an example skill.";
const SKILL_EXAMPLE_THINKING =
	"The user is invoking the example skill with no extra text. The Usage rule says to reply exactly `This is an example skill.` with nothing else. I will reply with that exact line.";

/** Few-shot 2 ("Hello"): read the reference file, then reply exactly what it says. */
const READ_HELLO_TOOL_CALL_ID = "1juKPjqsGTBbNP0MR1I560XFx44dOcx7";
const SKILL_HELLO_CALL_THINKING =
	"The user invoked the example skill with \"Hello\". The Usage rule says not to guess the response: load references/03-hello.md and follow it exactly. Let me read that file.";
const HELLO_REFERENCE_TEXT = [
	"# Hello Invocation",
	"",
	"The instruction source for the \"Invoked with 'Hello'\" scenario in SKILL.md's `## Usage`. When the user invokes the skill with \"Hello\", load this file and follow it — SKILL.md deliberately holds no instructions for this case.",
	"",
	"## Response",
	"",
	"Reply exactly:",
	"",
	"> world",
	"",
	"No greeting, no punctuation, no commentary, no script. The reply is the single word above and nothing else.",
	"",
	"## Why it lives here",
	"",
	"This file demonstrates progressive disclosure. The logic for the \"Hello\" case sits in a reference file instead of in SKILL.md or `scripts/`, so the agent must open a reference to learn how to respond. The fixed \"Hello\" → \"world\" mapping is part of the demonstration.",
	"",
].join("\n");
const SKILL_HELLO_ANSWER = "world";
const SKILL_HELLO_ANSWER_THINKING =
	"The reference file says to reply exactly `world` with no greeting, punctuation, or commentary. I will reply with that single word.";

/** Few-shot 3 ("Hi"): run the skill's script with the text as CLI parameters. */
const RUN_SCRIPT_TOOL_CALL_ID = "5wi4XVTF6doSiyHCRNEnB3UKqxxxSCbg";
const SKILL_SCRIPT_CALL_THINKING =
	"The user invoked the example skill with \"Hi\". That is any other text, so I will pass it as CLI parameters to scripts/example.sh and report the output. Let me run the script.";
const SCRIPT_OUTPUT_TEXT = "This is example.sh output.\n";
const SKILL_SCRIPT_REPORT = [
	"example.sh output:",
	"",
	"```",
	"This is example.sh output.",
	"```",
].join("\n");
const SKILL_SCRIPT_REPORT_THINKING =
	"The script printed one line. The skill says to report the output. I will report it in a code block.";

/**
 * One simulated user message of the dialogue: the custom type it is
 * persisted as, its content, and whether it is shown in the TUI.
 * Built per session: the three few-shot invocations embed the <skill>
 * block rooted at the session's current working directory. The synthetic
 * replies that must follow each user message in the LLM context are built
 * alongside in before_agent_start (they carry the active model's metadata
 * and the synthetic thinking blocks).
 */
function buildDialoguePlan(exampleSkillDir: string): Array<{
	customType: string;
	userContent: string;
	display: boolean;
}> {
	const block = buildExampleSkillBlock(exampleSkillDir);
	return [
		{ customType: SKILLS_CUSTOM_TYPE, userContent: SKILLS_ASK, display: true },
		{ customType: TOOLS_CUSTOM_TYPE, userContent: TOOLS_ASK, display: true },
		{ customType: SKILLSYS_CUSTOM_TYPE, userContent: SKILLSYS_ASK, display: true },
		{ customType: SKILL_EXAMPLE_CUSTOM_TYPE, userContent: block, display: false },
		{ customType: SKILL_HELLO_CUSTOM_TYPE, userContent: `${block}\nHello`, display: false },
		{ customType: SKILL_SCRIPT_CUSTOM_TYPE, userContent: `${block}\nHi`, display: false },
	];
}

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
 * Build a synthetic assistant message. The content is a short text answer
 * (the YAML document or the few-shot reply) and/or a toolCall block; when
 * the model supports reasoning a short synthetic thinking block precedes
 * them (it rides on the wire as `reasoning_content` for the OpenAI
 * Completions API). Messages with a toolCall carry stopReason "toolUse".
 */
function makeSynthAssistant(
	text: string | undefined,
	thinking: string | undefined,
	model: Model<any>,
	toolCall?: { id: string; name: string; arguments: Record<string, unknown> },
): AssistantMessage {
	const content: (TextContent | ThinkingContent | ToolCall)[] = [];
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
	if (toolCall) {
		content.push({ type: "toolCall", id: toolCall.id, name: toolCall.name, arguments: toolCall.arguments });
	}
	if (text !== undefined) {
		content.push({ type: "text", text });
	}
	return {
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: ZERO_USAGE,
		stopReason: toolCall ? "toolUse" : "stop",
		timestamp: Date.now(),
	};
}

/** Build a synthetic tool-result message (pi's role "toolResult" shape). */
function makeSynthToolResult(
	toolCallId: string,
	toolName: string,
	text: string,
): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName,
		content: [{ type: "text", text }],
		isError: false,
		timestamp: Date.now(),
	};
}

/**
 * True when the (restored or live) message is one of the synthetic replies
 * we injected: an assistant message sharing one of our content blocks
 * (thinking text, answer text, or toolCall id+name), or a tool result with
 * the same toolCallId and payload text.
 */
function isSyntheticReply(
	m: unknown,
	expected: AssistantMessage | ToolResultMessage,
): boolean {
	if (typeof m !== "object" || m === null) {
		return false;
	}
	const msg = m as { role?: string; content?: unknown; toolCallId?: string };
	if (expected.role === "toolResult") {
		const expText = (expected.content[0] as TextContent | undefined)?.text;
		const msgBlocks = Array.isArray(msg.content)
			? (msg.content as Array<Record<string, unknown>>)
			: [];
		const msgText = (msgBlocks[0] as TextContent | undefined)?.text;
		return (
			msg.role === "toolResult" &&
			msg.toolCallId === expected.toolCallId &&
			msgText === expText
		);
	}
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
		if (
			block.type === "toolCall" &&
			blocks.some(
				(b) =>
					b.type === "toolCall" &&
					(b as ToolCall).id === (block as ToolCall).id &&
					(b as ToolCall).name === (block as ToolCall).name,
			)
		) {
			return true;
		}
	}
	return false;
}

export default function slmExtension(pi: ExtensionAPI) {
	// In-memory state for the session this extension injected into.
	const state: {
		sessionId: string | undefined;
		// The simulated user messages' custom types, in dialogue order.
		customTypes: string[];
		// Parallel to customTypes: the synthetic replies (assistant
		// messages and tool results) that must follow each simulated user
		// message in the LLM context. Built per session (model metadata +
		// synthetic thinking depend on the active model; the few-shot paths
		// depend on the session's working directory).
		replies: Array<Array<AssistantMessage | ToolResultMessage>>;
	} = {
		sessionId: undefined,
		customTypes: [],
		replies: [],
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

		// The simulated user messages: asks plus the three few-shot
		// invocations, whose <skill> block is rooted at the session's
		// current working directory (<cwd>/.agents/skills/example) — pi's
		// standard project skill location, so the few-shot's absolute paths
		// are the session's real absolute paths.
		const exampleSkillDir = join(opts.cwd, ".agents", "skills", "example");
		const plan = buildDialoguePlan(exampleSkillDir);

		// The synthetic replies for each simulated user message, in plan
		// order: skills listing, tools listing, the skill contract, and the
		// three few-shot invocations (fixed reply, read-reference,
		// run-script) with their tool results.
		state.sessionId = ctx.sessionManager.getSessionId();
		state.customTypes = plan.map((seg) => seg.customType);
		state.replies = [
			[
				makeSynthAssistant(
					skillsYaml,
					reasoning ? skillsThinking(skills.length) : undefined,
					ctx.model,
				),
			],
			[
				makeSynthAssistant(
					toolsYaml,
					reasoning ? toolsThinking(activeTools.length) : undefined,
					ctx.model,
				),
			],
			[
				makeSynthAssistant(
					SKILLSYS_EXPLAIN,
					reasoning ? SKILLSYS_THINKING : undefined,
					ctx.model,
				),
			],
			[
				makeSynthAssistant(
					SKILL_EXAMPLE_ANSWER,
					reasoning ? SKILL_EXAMPLE_THINKING : undefined,
					ctx.model,
				),
			],
			[
				makeSynthAssistant(
					undefined,
					reasoning ? SKILL_HELLO_CALL_THINKING : undefined,
					ctx.model,
					{
						id: READ_HELLO_TOOL_CALL_ID,
						name: "read",
						arguments: { path: `${exampleSkillDir}/references/03-hello.md` },
					},
				),
				makeSynthToolResult(READ_HELLO_TOOL_CALL_ID, "read", HELLO_REFERENCE_TEXT),
				makeSynthAssistant(
					SKILL_HELLO_ANSWER,
					reasoning ? SKILL_HELLO_ANSWER_THINKING : undefined,
					ctx.model,
				),
			],
			[
				makeSynthAssistant(
					undefined,
					reasoning ? SKILL_SCRIPT_CALL_THINKING : undefined,
					ctx.model,
					{
						id: RUN_SCRIPT_TOOL_CALL_ID,
						name: "bash",
						arguments: { command: `bash ${exampleSkillDir}/scripts/example.sh Hi` },
					},
				),
				makeSynthToolResult(RUN_SCRIPT_TOOL_CALL_ID, "bash", SCRIPT_OUTPUT_TEXT),
				makeSynthAssistant(
					SKILL_SCRIPT_REPORT,
					reasoning ? SKILL_SCRIPT_REPORT_THINKING : undefined,
					ctx.model,
				),
			],
		];

		// Persist the simulated dialogue in order (per segment: the
		// simulated user message, then its synthetic replies).
		// pi.sendMessage() (no triggerTurn) appends the custom message to
		// the session and to the agent state synchronously (its
		// non-streaming path contains no awaits), so the following
		// appendMessage() calls continue the session tree right after it.
		// The synthetic assistant/tool-result messages are persisted
		// through the SessionManager itself: ctx.sessionManager is typed as
		// a read-only pick, but the runtime object is the full
		// SessionManager instance, whose appendMessage() is the same method
		// the core uses.
		const sm = ctx.sessionManager as unknown as SessionManager;
		for (let i = 0; i < plan.length; i++) {
			const seg = plan[i];
			pi.sendMessage({
				customType: seg.customType,
				content: seg.userContent,
				display: seg.display,
			});
			for (const reply of state.replies[i]) {
				sm.appendMessage(reply);
			}
		}
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
			// The simulated asks and the synthetic replies each map to
			// (skills, tools, skill-system, few-shot invocations).
			const expected = new Map<string, Array<AssistantMessage | ToolResultMessage>>(
				state.customTypes.map((customType, i) => [customType, state.replies[i]]),
			);
			const hasAsk = msgs.some(
				(m) => m.role === "custom" && expected.has(m.customType),
			);
			if (!hasAsk) {
				return;
			}
			const out = [...msgs];
			const inserts: Array<[number, AssistantMessage | ToolResultMessage]> = [];
			for (let i = 0; i < msgs.length; i++) {
				const m = msgs[i];
				if (m.role !== "custom") {
					continue;
				}
				const seq = expected.get(m.customType);
				if (!seq) {
					continue;
				}
				// How much of the expected reply sequence already sits
				// right after the simulated user message (restored
				// sessions have all of it; fresh runs have none).
				let k = 0;
				while (k < seq.length && isSyntheticReply(msgs[i + 1 + k], seq[k])) {
					k++;
				}
				if (k < seq.length) {
					for (let j = k; j < seq.length; j++) {
						inserts.push([i + 1 + k, seq[j]]);
					}
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
