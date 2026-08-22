/**
 * slm.ts — minimal system prompt for small language models.
 *
 * Replaces pi's full system prompt with a single short system message.
 * Tools are NOT rendered into the prompt text: pi keeps passing them to
 * the LLM as the structured `tools` array in the provider payload, so the
 * model's chat template renders them.
 *
 * Usage: pi -e ./slm.ts
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const SYSTEM_MESSAGE =
	"You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.";

export default function slm(pi: ExtensionAPI) {
	pi.on("before_agent_start", async () => {
		return { systemPrompt: SYSTEM_MESSAGE };
	});
}
