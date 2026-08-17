/**
 * Test helper for the slm suite: appends every provider request payload to
 * the file named by $SLM_PAYLOAD_LOG (one JSON document per line). Loaded
 * alongside the extension under test; inert when the env var is unset.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "node:fs";

export default function (pi: ExtensionAPI) {
	const out = process.env.SLM_PAYLOAD_LOG;
	if (!out) {
		return;
	}
	pi.on("before_provider_request", (event) => {
		appendFileSync(out, JSON.stringify(event.payload) + "\n");
	});
}
