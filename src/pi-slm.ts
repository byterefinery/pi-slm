/**
 * pi-slm - Small Language Model reliability extension.
 *
 * Features:
 * 1. Write guard - blocks write on existing files, redirects to edit.
 * 2. Loop detection - aborts repeated identical tool calls.
 * 3. EISDIR handling - converts directory read errors to listings.
 * 4. Skills listing guard - replaces hallucinated skill lists with real ones.
 * 5. Tools listing guard - replaces hallucinated tool lists with real ones.
 * 6. Explicit skill invocation - handles /skill:<NAME> directives.
 * 7. Tool definition retry - injects schema on validation failure.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";
import { access, readdir, readFile } from "fs/promises";
import { constants } from "fs";
import { resolve } from "path";

// Loop detection threshold. Read from env at session start.
let loopThreshold = 3;

// Sliding window of recent tool call signatures.
let callWindow: Array<{ toolName: string; signature: string }> = [];

// Captured skills and tools from system prompt options.
let knownSkills: Array<{ name: string; description: string; location?: string }> = [];
let knownTools: Array<{ name: string; description: string }> = [];

// Tool names that had schema errors, for Feature 7.
let failedToolSchemas: Set<string> = new Set();

// Build a signature for loop detection.
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
    // For other tools, stringify the whole input.
    return `${toolName}:${JSON.stringify(input)}`;
}

// Detect bash bypass patterns.
function detectBashBypass(command: string): string | null {
    // rm followed by write-like action.
    if (/\brm\s+(-rf?\s+|--recursive\s+)?\S+\s*&&/.test(command)) return "rm followed by chained command";
    if (/\brm\s+(-rf?\s+|--recursive\s+)?\S+\s*;/.test(command)) return "rm followed by semicolon-chained command";

    // Truncation via redirect on existing file.
    if (/^\s*>\s+\S+/.test(command) || /^\s*:\s*>\s+\S+/.test(command)) {
        return "file truncation via redirect";
    }

    // truncate or shred targeting a file.
    if (/\btruncate\b\s+\S+/.test(command)) return "truncate command";
    if (/\bshred\b\s+\S+/.test(command)) return "shred command";

    // dd writing to a file.
    if (/\bdd\b.*\bof=\S+/.test(command)) return "dd write command";

    // cp overwriting a file.
    if (/\bcp\b\s+.*\s+\S+$/.test(command) && !/\bcp\b.*--backup/.test(command)) {
        // cp without backup flag, writing to a path.
        return "cp overwrite command";
    }

    return null;
}

// Check if a message contains a skills listing.
function isSkillsListing(text: string): boolean {
    const hasHeader =
        /available\s+skill/i.test(text) ||
        /installed\s+skill/i.test(text) ||
        /list\s+of\s+skill/i.test(text) ||
        /here are the.*skill/i.test(text) ||
        /i have access to.*skill/i.test(text) ||
        /i can use.*skill/i.test(text) ||
        /skill\s*(list|catalog|options)/i.test(text);

    const hasList =
        /[-*]\s+\S+/.test(text) ||
        /\d+\.\s+\S+/.test(text) ||
        /\n\s*\S+\s*\n\s*\S+/.test(text);

    return hasHeader && hasList;
}

// Check if a message contains a tools listing.
function isToolsListing(text: string): boolean {
    const hasHeader =
        /available\s+tool/i.test(text) ||
        /list\s+of\s+tool/i.test(text) ||
        /here are the.*tool/i.test(text) ||
        /i have access to.*tool/i.test(text) ||
        /i can use.*tool/i.test(text) ||
        /tool\s*(list|catalog|options)/i.test(text) ||
        /function\s+tool/i.test(text) ||
        /tools i can/i.test(text) ||
        /tools available/i.test(text) ||
        /my tools/i.test(text) ||
        /callable tool/i.test(text) ||
        /tool.*description/i.test(text);

    const hasList =
        /[-*]\s+\S+/.test(text) ||
        /\d+\.\s+\S+/.test(text) ||
        /\|.*\|.*\|/.test(text) ||
        /\n\s*\S+\s*\n\s*\S+/.test(text);

    return hasHeader && hasList;
}

// Build a factual skills listing.
function buildSkillsListing(): string {
    if (knownSkills.length === 0) {
        return "No skills are currently loaded.";
    }
    const lines = knownSkills.map((s) => `- ${s.name}: ${s.description}`);
    return "Available skills:\n\n" + lines.join("\n");
}

// Build a factual tools listing.
function buildToolsListing(): string {
    if (knownTools.length === 0) {
        return "No tools are currently active.";
    }
    const lines = knownTools.map((t) => `- ${t.name}: ${t.description}`);
    return "Available tools:\n\n" + lines.join("\n");
}

// Extract the non-listing portion of a message (follow-up questions, etc.).
function extractNonListingContent(text: string): string | null {
    // Look for content after the list that is not part of the list.
    // Split on common list patterns and check for trailing content.
    const lines = text.split("\n");
    let inList = false;
    let listEnd = -1;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (/^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line)) {
            inList = true;
        } else if (inList && line !== "") {
            listEnd = i;
            break;
        } else if (inList && line === "") {
            // Blank line after list items might end the list.
            // Check if next non-blank line is not a list item.
            let j = i + 1;
            while (j < lines.length && lines[j].trim() === "") j++;
            if (j < lines.length && !/^[-*]\s+/.test(lines[j].trim()) && !/^\d+\.\s+/.test(lines[j].trim())) {
                listEnd = j;
                break;
            }
        }
    }

    if (listEnd > 0) {
        const trailing = lines.slice(listEnd).join("\n").trim();
        if (trailing) return trailing;
    }

    return null;
}

export default function (pi: ExtensionAPI) {
    // Read loop threshold from environment.
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

    // Capture skills and tools from system prompt options.
    pi.on("before_agent_start", async (event, _ctx) => {
        const opts = event.systemPromptOptions;

        // Capture skills.
        knownSkills = [];
        if (opts.skills && Array.isArray(opts.skills)) {
            for (const skill of opts.skills) {
                if (skill && typeof skill === "object") {
                    knownSkills.push({
                        name: (skill.name as string) ?? "unknown",
                        description: (skill.description as string) ?? "",
                        location: skill.location as string | undefined,
                    });
                }
            }
        }

        // Capture tools.
        knownTools = [];
        if (opts.selectedTools) {
            const tools = Array.isArray(opts.selectedTools) ? opts.selectedTools : [];
            for (const tool of tools) {
                if (tool && typeof tool === "object" && tool.name) {
                    const name = tool.name as string;
                    const snippet = opts.toolSnippets
                        ? ((opts.toolSnippets as Record<string, string>)[name] ?? "")
                        : "";
                    knownTools.push({
                        name,
                        description: snippet,
                    });
                } else if (typeof tool === "string") {
                    const snippet = opts.toolSnippets
                        ? ((opts.toolSnippets as Record<string, string>)[tool] ?? "")
                        : "";
                    knownTools.push({
                        name: tool,
                        description: snippet,
                    });
                }
            }
        }
    });

    // Feature 6: Explicit skill invocation.
    pi.on("input", async (event, _ctx) => {
        const skillMatch = event.text.match(/^\/skill:(\S+)(?:\s+(.+))?$/s);
        if (!skillMatch) return;

        const skillName = skillMatch[1];
        const userMessage = skillMatch[2]?.trim();

        // Find matching skill.
        const skill = knownSkills.find((s) => s.name === skillName);
        if (!skill) {
            const available = knownSkills.map((s) => s.name).join(", ");
            return {
                action: "transform" as const,
                text: `Skill not found: "${skillName}". Available skills: ${available}`,
            };
        }

        // Load skill file.
        if (!skill.location) {
            return {
                action: "transform" as const,
                text: `Skill "${skillName}" has no file location. Cannot load.`,
            };
        }

        let skillContent: string;
        try {
            skillContent = await readFile(skill.location, "utf-8");
        } catch (err) {
            return {
                action: "transform" as const,
                text: `Failed to read skill "${skillName}": ${err instanceof Error ? err.message : String(err)}`,
            };
        }

        // Build transformed prompt.
        const dir = resolve(skill.location, "..");
        let prompt = `---\nSkill: ${skill.name}\n---\n\n${skillContent}`;
        prompt += `\n\n---\nInstructions:\n---\nFollow the skill's instructions above to complete the task.`;
        prompt += `\nReference files are NOT loaded automatically. Use the read tool to load only the references you need from: ${dir}/references/`;
        if (userMessage) {
            prompt += `\n\n---\nTask:\n---\n${userMessage}`;
        }

        return {
            action: "transform" as const,
            text: prompt,
        };
    });

    // Feature 1: Write guard. Feature 2: Loop detection.
    pi.on("tool_call", async (event, ctx) => {
        // Feature 1a: Block write on existing files.
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
                // File does not exist, allow write.
            }
        }

        // Feature 1b: Block bash bypasses.
        if (isToolCallEventType("bash", event)) {
            const command = event.input.command as string;
            const bypass = detectBashBypass(command);
            if (bypass) {
                return {
                    block: true,
                    reason: `Blocked bash command: ${bypass}. This pattern can bypass the write guard. Use the edit tool for file modifications.`,
                };
            }
        }

        // Feature 2: Loop detection.
        const signature = buildSignature(event.toolName, event.input);
        callWindow.push({ toolName: event.toolName, signature });

        if (callWindow.length > loopThreshold) {
            callWindow.shift();
        }

        if (callWindow.length >= loopThreshold) {
            const first = callWindow[0].signature;
            const allSame = callWindow.every((entry) => entry.signature === first);

            if (allSame) {
                const count = callWindow.length;
                return {
                    block: true,
                    reason: `Loop detected: tool "${event.toolName}" called ${count} consecutive times with the same input. This likely indicates the model is stuck. Try a different approach or rephrase the task.`,
                    terminate: true,
                };
            }
        }
    });

    // Feature 3: EISDIR to directory listing. Feature 7: Schema failure detection.
    pi.on("tool_result", async (event, _ctx) => {
        // Feature 3: EISDIR handling.
        if (event.toolName === "read" && event.isError) {
            const contentText = event.content
                .map((c) => (typeof c === "object" && "text" in c ? (c.text as string) : ""))
                .join("");

            if (contentText.includes("EISDIR")) {
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
                    // readdir failed, keep original error.
                }
            }
        }

        // Feature 7: Detect schema-related failures.
        if (event.isError) {
            const contentText = event.content
                .map((c) => (typeof c === "object" && "text" in c ? (c.text as string) : ""))
                .join("");

            const isSchemaError =
                /invalid\s+(tool\s+)?argument/i.test(contentText) ||
                /validation\s+fail/i.test(contentText) ||
                /missing\s+required\s+param/i.test(contentText) ||
                /expected\s+type/i.test(contentText) ||
                /unknown\s+param/i.test(contentText) ||
                /schema/i.test(contentText) ||
                /parameter.*error/i.test(contentText);

            if (isSchemaError) {
                failedToolSchemas.add(event.toolName);
            }
        }
    });

    // Feature 7: Inject tool definition reminders on schema failures.
    pi.on("context", async (event, _ctx) => {
        if (failedToolSchemas.size === 0) return;

        // Get all available tools for schema lookup.
        const allTools = pi.getAllTools();
        const toolMap = new Map(allTools.map((t) => [t.name, t]));

        // Build reminder text.
        const reminders = Array.from(failedToolSchemas).map((toolName) => {
            const tool = toolMap.get(toolName);
            if (!tool) return null;

            // Build a concise schema reminder.
            const lines = [`Tool definition reminder for "${toolName}":`];
            if (tool.description) {
                lines.push(`  Description: ${tool.description}`);
            }
            lines.push(`  Usage: Call "${toolName}" with the parameters defined below.`);

            return lines.join("\n");
        }).filter(Boolean);

        if (reminders.length === 0) {
            failedToolSchemas.clear();
            return;
        }

        const reminderText = "\n\n" + reminders.join("\n\n") + "\n";

        // Append reminder to the last user message.
        const userMessages = event.messages.filter((m) => m.role === "user");
        if (userMessages.length > 0) {
            const lastUserMsg = userMessages[userMessages.length - 1];
            if (Array.isArray(lastUserMsg.content)) {
                const lastContent = lastUserMsg.content[lastUserMsg.content.length - 1];
                if (lastContent && typeof lastContent === "object" && "text" in lastContent) {
                    lastContent.text = (lastContent.text as string) + reminderText;
                }
            }
        }

        // Clear after one-time injection.
        failedToolSchemas.clear();
    });

    // Reset loop window on turn end.
    pi.on("turn_end", async (_event, _ctx) => {
        callWindow = [];
        failedToolSchemas = new Set();
    });

    // Feature 4 and 5: Skills and tools listing guards.
    pi.on("message_end", async (event, _ctx) => {
        if (event.message.role !== "assistant") return;

        const content = event.message.content;
        if (!content || !Array.isArray(content)) return;

        const textParts = content
            .map((c) => (typeof c === "object" && "text" in c ? (c.text as string) : ""))
            .filter(Boolean);

        if (textParts.length === 0) return;

        const fullText = textParts.join("\n");
        let replaced = false;
        const newTextParts = [];

        for (const part of textParts) {
            if (isSkillsListing(part)) {
                const nonListing = extractNonListingContent(part);
                let replacement = buildSkillsListing();
                if (nonListing) {
                    replacement += "\n\n" + nonListing;
                }
                newTextParts.push(replacement);
                replaced = true;
            } else if (isToolsListing(part)) {
                const nonListing = extractNonListingContent(part);
                let replacement = buildToolsListing();
                if (nonListing) {
                    replacement += "\n\n" + nonListing;
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
