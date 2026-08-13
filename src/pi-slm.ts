/**
 * pi-slm - Small Language Model reliability extension.
 *
 * Features:
 * 1. Write guard — blocks write on existing files, redirects to edit.
 * 2. Loop detection — aborts repeated identical tool calls.
 * 3. EISDIR handling — converts directory read errors to listings.
 * 4. Listing guard — replaces hallucinated skill/tool lists with real ones.
 * 5. Skill invocation — handles /skill:<NAME> directives.
 * 6. Tool schema hint — injects hints on parameter validation failures.
 * 7. Tool hallucination guard — blocks calls to non-existent tools.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { access, readdir, readFile } from "fs/promises";
import { constants } from "fs";
import { resolve } from "path";

// -- State --

let loopThreshold = 3;
let callWindow: Array<{ toolName: string; signature: string }> = [];
let knownSkills: Array<{ name: string; description: string; filePath?: string }> = [];
let knownTools: Array<{ name: string; description: string }> = [];
let failedToolSchemas: Set<string> = new Set();
let listingIntent: "skills" | "tools" | "both" | null = null;

// -- Helpers --

async function discoverSkills(cwd: string): Promise<Array<{ name: string; description: string; filePath?: string }>> {
    const home = process.env.HOME ?? "/";

    // Walk up from cwd to find project-local skill dirs (like pi does)
    const projectDirs: string[] = [];
    let current = resolve(cwd);
    while (true) {
        for (const sub of [".pi/skills", ".agents/skills"]) {
            projectDirs.push(resolve(current, sub));
        }
        const parent = resolve(current, "..");
        if (parent === current) break; // reached root
        current = parent;
    }

    const dirs = [
        resolve(home, ".pi/agent/skills"),
        resolve(home, ".agents/skills"),
        ...projectDirs,
    ];

    const seen = new Set<string>();
    const results: Array<{ name: string; description: string; filePath?: string }> = [];

    for (const dir of dirs) {
        try {
            const entries = await readdir(dir, { withFileTypes: true });
            for (const entry of entries) {
                if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
                if (seen.has(entry.name)) continue;

                const skillMd = resolve(dir, entry.name, "SKILL.md");
                try {
                    const content = await readFile(skillMd, "utf-8");
                    const nameMatch = content.match(/^---\s*\nname:\s*(.+?)\n/);
                    const descMatch = content.match(/^---\s*\n[^\n]*\ndescription:\s*(.+?)\n/);
                    const name = nameMatch ? nameMatch[1].trim() : entry.name;
                    const desc = descMatch ? descMatch[1].trim() : "";
                    seen.add(name);
                    results.push({ name, description: desc, filePath: skillMd });
                } catch {
                    // No SKILL.md or unreadable
                }
            }
        } catch {
            // Directory doesn't exist
        }
    }

    return results;
}

function buildSignature(toolName: string, input: Record<string, unknown>): string {
    if (toolName === "bash") {
        const cmd = (input.command as string ?? "").replace(/\s+/g, " ").trim();
        return `bash:${cmd}`;
    }
    if (toolName === "read" || toolName === "write" || toolName === "edit") {
        const path = (input.path as string ?? "").replace(/\s+/g, " ").trim();
        const offset = input.offset ?? "";
        const limit = input.limit ?? "";
        return `${toolName}:${path}:${offset}:${limit}`;
    }
    return `${toolName}:${JSON.stringify(input)}`;
}

function detectBashBypass(command: string): string | null {
    // Redirect writes: catches `echo ... > f`, `printf ... > f`, `cat > f`, `> f`
    if (/\s*>\s*\S+/.test(command) && !/\bif\b/.test(command)) return "file write via redirect (>)";
    // Append redirect
    if (/\s*>>\s*\S+/.test(command)) return "file append via redirect (>>)";
    // tee writing to files
    if (/\btee\s+(-a\s+)?\S+/.test(command)) return "file write via tee";
    // sed in-place
    if (/\bsed\s+-i\b/.test(command)) return "in-place edit via sed -i";
    // rm -rf targeting protected dirs
    if (/\brm\s+(-rf?\s+|--recursive\s+)?(?:\.git|\.pi|\.agents|node_modules)/.test(command)) return "rm targeting protected directory";
    // truncate
    if (/\btruncate\b\s+\S+/.test(command)) return "truncate command";
    // shred
    if (/\bshred\b\s+\S+/.test(command)) return "shred command";
    // dd write
    if (/\bdd\b.*\bof=\S+/.test(command)) return "dd write command";
    return null;
}

function detectListingIntent(text: string): "skills" | "tools" | "both" | null {
    const lower = text.toLowerCase();
    const hasSkill = /\bskill/i.test(text);
    const hasTool = /\btool/i.test(text);

    // Check for listing context (not just mentioning the word)
    const askingForListing =
        /\blist\b/i.test(text) ||
        /\bshow\b/i.test(text) ||
        /\bwhat\b.*\bhave\b/i.test(text) ||
        /\bavailable\b/i.test(text) ||
        /\baccess\b/i.test(text) ||
        /\bcatalog\b/i.test(text) ||
        /\boptions\b/i.test(text) ||
        /what\s+(are|can|do)/i.test(text);

    if (!askingForListing) return null;

    if (hasSkill && hasTool) return "both";
    if (hasSkill) return "skills";
    if (hasTool) return "tools";
    return null;
}

function isSkillsListing(text: string): boolean {
    // Allow optional markdown (**) and flexible spacing between keywords
    const hasHeader =
        /available\s*\*?\*?\s*skill/i.test(text) ||
        /installed\s*\*?\*?\s*skill/i.test(text) ||
        /skill\s+installed/i.test(text) ||
        /list.*skill/i.test(text) ||
        /skill.*list/i.test(text) ||
        /here are.*skill/i.test(text) ||
        /i have access to.*skill/i.test(text) ||
        /i can use.*skill/i.test(text) ||
        /skill\s*(list|catalog|options)/i.test(text) ||
        /\b\d+\s*\*?\*?\s*skill/i.test(text) ||
        /skill\s+available/i.test(text) ||
        /skill.*and.*tool/i.test(text) ||
        /tool.*and.*skill/i.test(text);

    const hasList =
        /[-*]\s+\S+/.test(text) ||
        /\d+\.\s+\S+/.test(text) ||
        /\|.*\|.*\|/.test(text) ||
        /\n\s*\S+\s*\n\s*\S+/.test(text);

    return hasHeader && hasList;
}

function isToolsListing(text: string): boolean {
    // Allow optional markdown (**) and flexible spacing between keywords
    const hasHeader =
        /available\s*\*?\*?\s*tool/i.test(text) ||
        /list.*tool/i.test(text) ||
        /tool.*list/i.test(text) ||
        /here are.*tool/i.test(text) ||
        /i have access to.*tool/i.test(text) ||
        /i can use.*tool/i.test(text) ||
        /tool\s*(list|catalog|options)/i.test(text) ||
        /function\s+tool/i.test(text) ||
        /tools?\s*i can/i.test(text) ||
        /tools?\s*available/i.test(text) ||
        /my\s+tools?/i.test(text) ||
        /callable\s+tool/i.test(text) ||
        /tool.*description/i.test(text) ||
        /skill.*and.*tool/i.test(text) ||
        /tool.*and.*skill/i.test(text);

    const hasList =
        /[-*]\s+\S+/.test(text) ||
        /\d+\.\s+\S+/.test(text) ||
        /\|.*\|.*\|/.test(text) ||
        /\n\s*\S+\s*\n\s*\S+/.test(text);

    return hasHeader && hasList;
}

function buildSkillsListing(): string {
    if (knownSkills.length === 0) return "No skills are currently loaded.";
    const lines = knownSkills.map((s) => `- ${s.name}: ${s.description}`);
    return "Available skills:\n\n" + lines.join("\n");
}

function buildToolsListing(): string {
    if (knownTools.length === 0) return "No tools are currently active.";
    const lines = knownTools.map((t) => `- ${t.name}: ${t.description}`);
    return "Available tools:\n\n" + lines.join("\n");
}



// -- Extension --

export default function (pi: ExtensionAPI) {
    // session_start: read config
    pi.on("session_start", async (_event, _ctx) => {
        const envVal = process.env.PI_LOOP_THRESHOLD;
        if (envVal !== undefined) {
            const parsed = parseInt(envVal, 10);
            if (!isNaN(parsed) && parsed >= 2) {
                loopThreshold = parsed;
            }
        }
        callWindow = [];
        failedToolSchemas = new Set();
    });

    // before_agent_start: capture skills and tools
    pi.on("before_agent_start", async (event, _ctx) => {
        const opts = event.systemPromptOptions;

        knownSkills = [];
        if (opts.skills && Array.isArray(opts.skills)) {
            for (const skill of opts.skills) {
                if (skill && typeof skill === "object") {
                    knownSkills.push({
                        name: (skill.name as string) ?? "unknown",
                        description: (skill.description as string) ?? "",
                        filePath: skill.filePath as string | undefined,
                    });
                }
            }
        }

        knownTools = [];
        if (opts.selectedTools) {
            const tools = Array.isArray(opts.selectedTools) ? opts.selectedTools : [];
            for (const tool of tools) {
                if (tool && typeof tool === "object" && tool.name) {
                    const name = tool.name as string;
                    const snippet = opts.toolSnippets
                        ? ((opts.toolSnippets as Record<string, string>)[name] ?? "")
                        : "";
                    knownTools.push({ name, description: snippet });
                } else if (typeof tool === "string") {
                    const snippet = opts.toolSnippets
                        ? ((opts.toolSnippets as Record<string, string>)[tool] ?? "")
                        : "";
                    knownTools.push({ name: tool, description: snippet });
                }
            }
        }
    });

    // input: skill invocation
    pi.on("input", async (event, ctx) => {
        const skillMatch = event.text.match(/^\/skill:(\S+)(?:\s+(.+))?$/s);
        if (!skillMatch) return;

        const skillName = skillMatch[1];
        const userMessage = skillMatch[2]?.trim();

        // Try known skills first, then discover from filesystem
        let skill = knownSkills.find((s) => s.name === skillName);
        let allSkills = [...knownSkills];

        if (!skill) {
            const discovered = await discoverSkills(ctx.cwd);
            skill = discovered.find((s) => s.name === skillName);
            const seen = new Set(knownSkills.map((s) => s.name));
            for (const s of discovered) {
                if (!seen.has(s.name)) allSkills.push(s);
            }
        }

        if (!skill) {
            const available = allSkills.map((s) => s.name).join(", ");
            return { action: "transform" as const, text: `Skill not found: "${skillName}". Available skills: ${available}` };
        }

        if (!skill.filePath) {
            return { action: "transform" as const, text: `Skill "${skillName}" has no file location. Cannot load.` };
        }

        let skillContent: string;
        try {
            skillContent = await readFile(skill.filePath, "utf-8");
        } catch (err) {
            return { action: "transform" as const, text: `Failed to read skill "${skillName}": ${err instanceof Error ? err.message : String(err)}` };
        }

        const dir = resolve(skill.filePath, "..");
        let prompt = `---\nSkill: ${skill.name}\n---\n\n${skillContent}`;
        prompt += `\n\n---\nInstructions:\n---\nFollow the skill's instructions above to complete the task.`;
        prompt += `\nReference files are NOT loaded automatically. Use the read tool to load only the references you need from: ${dir}/references/`;
        if (userMessage) {
            prompt += `\n\n---\nTask:\n---\n${userMessage}`;
        }

        return { action: "transform" as const, text: prompt };
    });

    // tool_call: Feature 7 -> Feature 1 -> Feature 2
    pi.on("tool_call", async (event, ctx) => {
        // Feature 7: Tool Hallucination Guard
        const toolName = event.toolName;
        if (!toolName.startsWith("$") && !toolName.startsWith("_")) {
            const knownToolNames = knownTools.map((t) => t.name);
            if (knownToolNames.length > 0 && !knownToolNames.includes(toolName)) {
                return {
                    block: true,
                    reason: `Unknown tool "${toolName}". Available tools: ${knownToolNames.join(", ")}.`,
                };
            }

            // Parameter shape validation for known tools
            if (toolName === "read") {
                const path = event.input.path as string;
                if (!path || typeof path !== "string" || path.trim() === "") {
                    return { block: true, reason: `Tool "read" requires a non-empty "path" parameter.` };
                }
            } else if (toolName === "write") {
                const path = event.input.path as string;
                const content = event.input.content as string;
                if (!path || typeof path !== "string" || path.trim() === "") {
                    return { block: true, reason: `Tool "write" requires a non-empty "path" parameter.` };
                }
                if (!content || typeof content !== "string" || content.trim() === "") {
                    return { block: true, reason: `Tool "write" requires a non-empty "content" parameter.` };
                }
            } else if (toolName === "edit") {
                const path = event.input.path as string;
                const edits = event.input.edits as unknown[];
                if (!path || typeof path !== "string" || path.trim() === "") {
                    return { block: true, reason: `Tool "edit" requires a non-empty "path" parameter.` };
                }
                if (!edits || !Array.isArray(edits) || edits.length === 0) {
                    return { block: true, reason: `Tool "edit" requires a non-empty "edits" array.` };
                }
            } else if (toolName === "bash") {
                const command = event.input.command as string;
                if (!command || typeof command !== "string" || command.trim() === "") {
                    return { block: true, reason: `Tool "bash" requires a non-empty "command" parameter.` };
                }
            }
        }

        // Feature 1a: Write guard — block write on existing files
        if (event.toolName === "write") {
            const inputPath = event.input.path as string;
            const targetPath = resolve(ctx.cwd, inputPath);
            try {
                await access(targetPath, constants.F_OK);
                return {
                    block: true,
                    reason: `File "${inputPath}" already exists. Use the edit tool to modify it instead of write.`,
                };
            } catch {
                // File does not exist, allow write
            }
        }

        // Feature 1b: Bash bypass interception
        if (event.toolName === "bash") {
            const command = event.input.command as string;
            const bypass = detectBashBypass(command);
            if (bypass) {
                return {
                    block: true,
                    reason: `Blocked bash command: ${bypass}. This pattern can bypass the write guard. Use the edit tool for file modifications.`,
                };
            }
        }

        // Feature 2: Loop detection
        const signature = buildSignature(event.toolName, event.input);
        callWindow.push({ toolName: event.toolName, signature });

        if (callWindow.length > loopThreshold) {
            callWindow.shift();
        }

        if (callWindow.length >= loopThreshold) {
            const first = callWindow[0].signature;
            const allSame = callWindow.every((entry) => entry.signature === first);

            if (allSame) {
                return {
                    block: true,
                    reason: `Loop detected: tool "${event.toolName}" called ${callWindow.length} consecutive times with the same input. Try a different approach.`,
                    terminate: true,
                };
            }
        }
    });

    // tool_result: Feature 3 (EISDIR) + Feature 6 (schema detection)
    pi.on("tool_result", async (event, _ctx) => {
        const contentText = event.content
            .map((c) => (typeof c === "object" && "text" in c ? (c.text as string) : ""))
            .join("");

        // Feature 3: EISDIR to directory listing
        if (event.toolName === "read" && event.isError && contentText.includes("EISDIR")) {
            const inputPath = event.input.path as string;
            const resolvedPath = resolve(_ctx.cwd, inputPath);

            try {
                const entries = await readdir(resolvedPath, { withFileTypes: true });
                const lines = entries
                    .map((e) => {
                        const type = e.isDirectory() ? "d" : e.isSymbolicLink() ? "l" : "-";
                        return `${type} ${e.name}`;
                    })
                    .sort();

                return {
                    content: [{ type: "text", text: lines.join("\n") }],
                    isError: false,
                };
            } catch {
                // readdir failed, keep original error
            }
        }

        // Feature 6: Schema failure detection
        if (event.isError) {
            const isSchemaError =
                /invalid\s+(tool\s+)?argument/i.test(contentText) ||
                /tool\s+call\s+validation/i.test(contentText) ||
                /missing\s+required\s+(parameter|field|param)/i.test(contentText) ||
                /expected\s+type/i.test(contentText) ||
                /unknown\s+(parameter|param|argument)/i.test(contentText) ||
                /tool.*schema.*error/i.test(contentText) ||
                /parameter\s+validation\s+error/i.test(contentText);

            if (isSchemaError) {
                failedToolSchemas.add(event.toolName);
            }
        }
    });

    // context: Feature 6 (inject schema hints)
    pi.on("context", async (event, _ctx) => {
        if (failedToolSchemas.size === 0) return;

        const toolNames = Array.from(failedToolSchemas);
        const hintText =
            "\n\nNote: The following tool(s) had parameter validation errors on the last call. " +
            `Check the tool definitions in your instructions for correct parameter names and types: ` +
            toolNames.join(", ") + ".";

        const userMessages = event.messages.filter((m) => m.role === "user");
        if (userMessages.length > 0) {
            const lastUserMsg = userMessages[userMessages.length - 1];
            if (Array.isArray(lastUserMsg.content)) {
                const lastContent = lastUserMsg.content[lastUserMsg.content.length - 1];
                if (lastContent && typeof lastContent === "object" && "text" in lastContent) {
                    lastContent.text = (lastContent.text as string) + hintText;
                }
            }
        }

        failedToolSchemas.clear();
    });

    // turn_end: reset loop window and listing intent
    pi.on("turn_end", async (_event, _ctx) => {
        callWindow = [];
        listingIntent = null;
    });

    // message_end: capture user intent + Feature 4 (listing guard)
    pi.on("message_end", async (event, _ctx) => {
        // Capture intent from user messages
        if (event.message.role === "user") {
            const text = event.message.content
                .map((c) => (typeof c === "object" && "text" in c ? (c.text as string) : ""))
                .join(" ");
            listingIntent = detectListingIntent(text);
            return;
        }

        if (event.message.role !== "assistant") return;

        const content = event.message.content;
        if (!content || !Array.isArray(content)) return;

        const textParts = content
            .map((c) => (typeof c === "object" && "text" in c ? (c.text as string) : ""))
            .filter(Boolean);

        if (textParts.length === 0) return;

        let replaced = false;
        const newTextParts = [];

        for (const part of textParts) {
            const hasSkills = isSkillsListing(part);
            const hasTools = isToolsListing(part);

            if (hasSkills || hasTools) {
                // Only output what the user asked for.
                // If no intent captured (e.g. mid-conversation), output what model produced.
                let replacement = "";
                const intent = listingIntent;

                if (intent === "skills") {
                    replacement = buildSkillsListing();
                } else if (intent === "tools") {
                    replacement = buildToolsListing();
                } else if (intent === "both") {
                    replacement = buildSkillsListing() + "\n\n" + buildToolsListing();
                } else {
                    // No intent captured — output what model detected
                    if (hasSkills) replacement = buildSkillsListing();
                    if (hasTools) {
                        if (hasSkills) replacement += "\n\n";
                        replacement += buildToolsListing();
                    }
                }

                newTextParts.push(replacement);
                replaced = true;
            } else {
                newTextParts.push(part);
            }
        }

        if (replaced) {
            return {
                message: {
                    ...event.message,
                    content: newTextParts.map((text) => ({ type: "text", text })),
                },
            };
        }
    });
}
