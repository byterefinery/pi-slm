/**
 * pair-override.ts — overrides the synthetic "skill-system" pair injected by
 * slm.ts, driven by a single JSON file (no slm.ts edits needed).
 *
 * Pair file: $SLM_SKILL_PAIR_FILE, or `skill-pair.json` in the working dir:
 *
 *   { "ask": "...", "explain": "...", "thinking": "..." }
 *
 *   - "explain"  -> the synthetic assistant reply's content
 *   - "thinking" -> that reply's reasoning (reasoning_content on the wire for
 *                   the OpenAI Completions API); "" omits it
 *   - "ask"      -> optional rewrite of the simulated user question on the wire
 *
 * Load AFTER slm.ts:   pi -e slm.ts -e pair-override.ts
 *
 * Mechanics: on every LLM call the `context` event carries the full message
 * array; this extension locates the simulated skill-system question (the
 * custom message slm.ts sends) and makes sure the assistant reply directly
 * after it is the pair from the file — replacing slm.ts's built-in reply when
 * present, inserting one when missing. slm.ts's handler may re-insert its
 * built-in reply on each call; this handler always runs after it and wins.
 */
import type {
	AssistantMessage,
	Model,
	ThinkingContent,
	Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { readFileSync } from "node:fs";
import { join } from "node:path";

/** The simulated skill-system question slm.ts sends (fixed in slm.ts). */
const SKILLSYS_ASK =
	"How does skill system work? When a skill block is in my latest message, what do I do?";

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function loadPair(): { ask: string; explain: string; thinking: string } | null {
	try {
		const path = process.env.SLM_SKILL_PAIR_FILE || join(process.cwd(), "skill-pair.json");
		const data = JSON.parse(readFileSync(path, "utf8"));
		if (typeof data.explain !== "string" || data.explain.length === 0) {
			return null;
		}
		return {
			ask: typeof data.ask === "string" && data.ask.length > 0 ? data.ask : "",
			explain: data.explain,
			thinking: typeof data.thinking === "string" ? data.thinking : "",
		};
	} catch {
		return null;
	}
}

function msgText(m: unknown): string {
	if (typeof m !== "object" || m === null) return "";
	const c = (m as { content?: unknown }).content;
	if (typeof c === "string") return c;
	if (Array.isArray(c)) {
		return c
			.filter((b) => b && typeof b === "object" && (b as { type?: string }).type === "text")
			.map((b) => (b as { text?: string }).text ?? "")
			.join("");
	}
	return "";
}

export default function pairOverride(pi: ExtensionAPI) {
	pi.on("context", (event, ctx) => {
		try {
			const pair = loadPair();
			if (!pair) return;
			const model: Model<any> | undefined = ctx.model;
			const reasoning = model?.reasoning === true;

			const msgs = event.messages as Array<Record<string, unknown>>;
			let idx = -1;
			for (let i = 0; i < msgs.length; i++) {
				const m = msgs[i];
				const role = m.role;
				if (role !== "custom" && role !== "user") continue;
				const text = msgText(m);
				if (
					(m.customType === "skill-system" ||
						text === SKILLSYS_ASK ||
						(pair.ask.length > 0 && text === pair.ask)) &&
					// the skill-system question is the only such message
					(idx === -1)
				) {
					idx = i;
					break;
				}
			}
			if (idx === -1) return;

			const out = [...msgs];

			// Optional: rewrite the simulated question on the wire.
			if (pair.ask.length > 0) {
				const m = out[idx];
				if (typeof m.content === "string") {
					out[idx] = { ...m, content: pair.ask };
				} else if (Array.isArray(m.content)) {
					out[idx] = {
						...m,
						content: m.content.map((b) =>
							b && typeof b === "object" && (b as { type?: string }).type === "text"
								? { ...(b as object), text: pair.ask }
								: b,
						),
					};
				}
			}

			// Build the override reply (same shape slm.ts's replies use).
			const content: Array<{ type: string; thinking?: string; text?: string } & Record<string, unknown>> = [];
			if (reasoning && pair.thinking.length > 0) {
				const block: Record<string, unknown> = { type: "thinking", thinking: pair.thinking };
				if (model?.api === "openai-completions") {
					// rides on the wire as `reasoning_content` (see slm.ts).
					block.thinkingSignature = "reasoning_content";
				}
				content.push(block as ThinkingContent);
			}
			content.push({ type: "text", text: pair.explain });
			const reply: AssistantMessage = {
				role: "assistant",
				content: content as AssistantMessage["content"],
				api: model?.api ?? "openai-completions",
				provider: model?.provider ?? "",
				model: model?.id ?? "",
				usage: ZERO_USAGE,
				stopReason: "stop",
				timestamp: Date.now(),
			};

			// The reply must sit directly after the question: replace whatever
			// is there (slm.ts's built-in reply or a previous override), else
			// insert.
			const next = out[idx + 1];
			if (next && next.role === "assistant") {
				out[idx + 1] = reply;
			} else {
				out.splice(idx + 1, 0, reply);
			}
			return { messages: out };
		} catch {
			// transformContext contract: never throw; leave context unchanged.
		}
	});
}
