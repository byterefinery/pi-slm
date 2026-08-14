/**
 * slm - Small Language Model reliability extension for Pi.
 *
 * Intercepts tool calls, corrects hallucinations, detects loops,
 * deduplicates redundant reads, and steers the model away from
 * degenerate behavior patterns.
 */

import type { ExtensionAPI, ToolCallEvent, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";
import { readdir, readFile } from "fs/promises";
import { resolve, isAbsolute } from "path";

// Configuration

function loadConfig() {
  const num = (envVar: string, defaultVal: number, min: number) => {
    const v = parseInt(process.env[envVar] ?? "", 10);
    return isNaN(v) ? defaultVal : Math.max(min, v);
  };

  return {
    loopThreshold: num("PI_LOOP_THRESHOLD", 3, 2),
    editThreshold: num("PI_EDIT_THRESHOLD", 3, 2),
    readDedupWindow: num("PI_READ_DEDUP_WINDOW", 3, 2),
    enabled: process.env.PI_SLM_ENABLED !== "false",
  };
}

// Doom Loop Warning Messages

const DOOM_LOOP_WARNINGS = [
  "Warning: Doom looping detected, let's try another approach.",
  "Warning: Heads up, you're stuck in a doom loop. Try something different.",
  "Warning: Repeated doom loop. Pivot to a new strategy.",
  "Warning: You keep doom looping. Explore a different path.",
  "Warning: Spinning in circles. Try a fresh angle.",
  "Warning: Repeated doom looping. Shift your approach.",
  "Warning: Loop detected. Move on.",
  "Warning: You've looped and repeated messages. Consider the next step.",
  "Warning: Looping on the same messages. Let's try another route.",
];

function randomWarning(): string {
  return DOOM_LOOP_WARNINGS[Math.floor(Math.random() * DOOM_LOOP_WARNINGS.length)];
}

// Read Dedup Steering Messages

const READ_DEDUP_MESSAGES = [
  "You already read this file recently and it hasn't changed. Instead of re-reading, try applying an edit or running a test.",
  "This file was already read and hasn't been modified. Use the content you already have in context.",
  "Duplicate read detected — the file hasn't changed since your last read. Move forward with what you know.",
  "You've already read this file. Rather than re-reading, try making a change or testing your fix.",
  "File unchanged since last read. Please proceed with editing or testing instead of re-reading.",
];

function randomReadDedupMessage(path: string): string {
  const base = READ_DEDUP_MESSAGES[Math.floor(Math.random() * READ_DEDUP_MESSAGES.length)];
  return `${base} (File: ${path})`;
}

// Edit Retry Messages

const EDIT_RETRY_MESSAGES = [
  "Your oldText didn't match. Use read to get the exact current content, then copy-paste the exact text (including whitespace) as oldText.",
  "Edit failed — oldText must match byte-for-byte. Re-read the file and copy the exact text you want to replace.",
  "The text you're trying to replace doesn't match the file. Read the file first, then use the exact text as oldText.",
  "Edit mismatch. The file content may have changed. Re-read it and provide the exact current text as oldText.",
  "oldText must be an exact match including whitespace and indentation. Read the file to get the current content.",
];

function randomEditRetryMessage(file: string, attempt: number): string {
  const base = EDIT_RETRY_MESSAGES[Math.floor(Math.random() * EDIT_RETRY_MESSAGES.length)];
  return `Edit attempt ${attempt} on ${file} failed. ${base}`;
}

// Hallucination Correction Messages

const HALLUCINATION_CORRECTIONS = [
  "You said the code looks correct, but recent tool results show errors. Please verify by actually running the code or tests.",
  "Recent tool results contain errors. The code may not be working as you think. Check the errors before declaring success.",
  "You declared success but there are unresolved errors in recent tool results. Please fix them first.",
  "Your claim of success doesn't match the recent error output. Re-examine the errors and fix the issues.",
];

function randomHallucinationCorrection(): string {
  return HALLUCINATION_CORRECTIONS[Math.floor(Math.random() * HALLUCINATION_CORRECTIONS.length)];
}

// Sequence Steering Messages

const SEQUENCE_STEERING_MESSAGES = [
  "You seem stuck in a repeating pattern. Try a different approach — perhaps simplify the problem or read documentation.",
  "This pattern has repeated. Consider stepping back and trying a fundamentally different strategy.",
  "Repeated sequence detected. Instead of continuing this cycle, try asking for clarification or breaking the problem into smaller steps.",
  "You're repeating the same steps. Pivot: try a fresh approach, different tool, or smaller edit scope.",
];

function randomSequenceSteering(): string {
  return SEQUENCE_STEERING_MESSAGES[Math.floor(Math.random() * SEQUENCE_STEERING_MESSAGES.length)];
}

// Bash Write Detection

// Patterns that indicate file writing via bash
const BASH_WRITE_PATTERNS = [
  // echo "content" > file
  /\b(echo|printf)\s+.*?>\s*(?!\/)(?!\/dev\/)(?!\d)\S+/, 
  // echo "content" >> file
  /\b(echo|printf)\s+.*?>>\s*(?!\/)(?!\/dev\/)(?!\d)\S+/, 
  // cat > file
  /\bcat\s*>\s*(?!\/dev\/)(?!\d)\S+/, 
  // cat >> file
  /\bcat\s*>>\s*(?!\/dev\/)(?!\d)\S+/, 
  // cat > file << EOF (heredoc)
  /\bcat\s*>>?\s*\S+\s*<</, 
  // tee file
  /\btee\s+(-a\s+)?(?!\/dev\/)(?!\/tmp\/)(?!\/var\/tmp\/)\S+/, 
];

// Patterns that are NOT destructive (shell internals)
const BASH_SAFE_PATTERNS = [
  /2>&1/,           // stderr to stdout
  />&2/,            // stdout to stderr
  /\d>\d/,          // fd to fd redirect
  />\s*\/dev\//,    // redirect to /dev/null etc.
  /\|\s*\w+/,       // pipe to command
];

function isDestructiveBash(command: string): boolean {
  // Allow writes to /tmp/ or /var/tmp/ (only standalone paths, not /tmp/ in middle of path)
  // Match patterns like "> /tmp/file" or "echo x > /tmp/file"
  const tmpWritePattern = />\s*(?:\/tmp\/|\/var\/tmp\/)/;
  if (tmpWritePattern.test(command)) return false;

  // Skip safe patterns
  for (const safe of BASH_SAFE_PATTERNS) {
    if (safe.test(command)) return false;
  }

  // Check for write patterns
  for (const pattern of BASH_WRITE_PATTERNS) {
    if (pattern.test(command)) return true;
  }

  return false;
}

// Skill Listing Detection

const SKILL_LISTING_PATTERNS = [
  /(?:available|installed|list of)\s+skill/i,
  /skills?\s+available/i,
  /i?\s*have\s+access\s+to.*skill/i,
  /here\s+(?:are|is).*skill/i,
  /skill.*(?:list|summary|overview)/i,
  /(?:these|the following)\s+skills?/i,
];

function detectSkillListing(text: string): boolean {
  // Must have a skill listing trigger AND be structured as a list/table
  const hasTrigger = SKILL_LISTING_PATTERNS.some(p => p.test(text));
  const hasListStructure = /\n\s*[-*•]\s+/.test(text) || /\|.*skill/i.test(text);
  return hasTrigger && (hasListStructure || text.length < 300);
}

// Tool Listing Detection

const TOOL_LISTING_PATTERNS = [
  /(?:available|list of)\s+tool/i,
  /tools?\s+available/i,
  /i?\s*have\s+access\s+to.*tool/i,
  /my\s+tools?/i,
  /tool.*(?:list|summary|overview)/i,
  /(?:these|the following)\s+tools?/i,
  /here\s+(?:are|is).*tool/i,
];

function detectToolListing(text: string): boolean {
  const hasTrigger = TOOL_LISTING_PATTERNS.some(p => p.test(text));
  const hasListStructure = /\n\s*[-*•]\s+/.test(text) || /\|.*tool/i.test(text);
  return hasTrigger && (hasListStructure || text.length < 300);
}

// Skill Invocation Detection

const SKILL_INVOCATION_PATTERNS = [
  /(?:use|invoke|run|call|load)\s+(?:the\s+)?skill\s*(?:named\s*)?["']?(\w+)/i,
  /\/skill\s*:\s*(\w+)/i,
  /(?:let me|ill|im going to)\s+(?:use|invoke)\s+(?:the\s+)?(\w+)\s+skill/i,
  /(?:activating|loading|using)\s+(?:the\s+)?(\w+)\s+skill/i,
];

function detectSkillInvocation(text: string): { name: string; matched: boolean } | null {
  for (const pattern of SKILL_INVOCATION_PATTERNS) {
    const match = text.match(pattern);
    if (match && match[1]) {
      return { name: match[1].toLowerCase(), matched: true };
    }
  }
  return null;
}

// Hallucination Detection

const HALLUCINATION_PATTERNS = [
  // "code looks correct" — specific phrase, not just "correct"
  { pattern: /code\s+looks\s+correct/i, type: "false_correct" as const },
  // "no modifications required" — claiming task is done
  { pattern: /no\s+modifications?\s+required/i, type: "false_complete" as const },
  // "everything works fine" — claiming success
  { pattern: /everything\s+works\s+fine/i, type: "false_complete" as const },
  // "task is complete" — claiming completion
  { pattern: /task\s+is\s+(?:complete|done|finished)/i, type: "false_complete" as const },
  // "all tests pass" — claiming tests passed
  { pattern: /all\s+tests?\s+pass/i, type: "false_complete" as const },
  // "successfully fixed" — claiming fix worked
  { pattern: /successfully\s+fixed/i, type: "false_complete" as const },
];

interface HallucinationInfo {
  type: "false_correct" | "false_complete";
  text: string;
}

function detectHallucination(text: string, recentErrorCount: number): HallucinationInfo | null {
  // Only flag if there are recent errors
  if (recentErrorCount === 0) return null;

  for (const { pattern, type } of HALLUCINATION_PATTERNS) {
    if (pattern.test(text)) {
      return { type, text };
    }
  }
  return null;
}

// Traceback/Error Parsing

interface ParsedError {
  type: string;
  file?: string;
  line?: number;
  message: string;
}

function parseTraceback(output: string): ParsedError | null {
  // Python traceback
  const pythonMatch = output.match(/(\w+Error):\s*(.+?)(?:\n|$)/);
  if (pythonMatch) {
    const fileLineMatch = output.match(/File\s+"([^"]+)",\s*line\s+(\d+)/);
    return {
      type: pythonMatch[1],
      file: fileLineMatch?.[1],
      line: fileLineMatch ? parseInt(fileLineMatch[2], 10) : undefined,
      message: pythonMatch[2].trim(),
    };
  }

  // Generic error patterns
  const genericMatch = output.match(/(?:error|exception|fail(?:ed|ure))\s*[:\-]?\s*(.+)/i);
  if (genericMatch) {
    return { type: "GenericError", message: genericMatch[1].trim() };
  }

  // Shell error
  const shellMatch = output.match(/(?:bash|sh|zsh):\s*(\S+):\s*(?:command not found|No such file|Permission denied)/i);
  if (shellMatch) {
    return { type: "ShellError", message: shellMatch[0] };
  }

  return null;
}

function getErrorExplanation(error: ParsedError): string {
  switch (error.type) {
    case "NameError":
      return `NameError: ${error.message}. A variable or function name is not defined. Check for typos or missing imports.`;
    case "TypeError":
      return `TypeError: ${error.message}. Wrong type used. Check function arguments and variable types.`;
    case "AttributeError":
      return `AttributeError: ${error.message}. An object doesn't have the attribute you're accessing.`;
    case "SyntaxError":
      return `SyntaxError: ${error.message}. The code has a syntax issue. Check brackets, colons, and indentation.`;
    case "KeyError":
      return `KeyError: ${error.message}. A dictionary key doesn't exist. Check the key name or add a default.`;
    case "IndexError":
      return `IndexError: ${error.message}. List/index out of range. Check array bounds.`;
    case "ShellError":
      return `Shell error: ${error.message}. The command or file wasn't found.`;
    default:
      return `Error (${error.type}): ${error.message}. Check the error details and fix the issue.`;
  }
}

// Read Cache

interface ReadCacheEntry {
  content: string;
  timestamp: number;
  offset?: number;
  limit?: number;
}

function getReadCacheKey(path: string, offset?: number, limit?: number): string {
  return `${path}:${offset ?? 0}:${limit ?? 0}`;
}

// Loop Detector

interface LoopInfo {
  count: number;
  key: string;
}

class LoopDetector {
  private toolCallHistory: Array<{ toolName: string; paramsHash: string; timestamp: number }> = [];
  private messageHistory: Array<{ text: string; timestamp: number }> = [];
  private readonly threshold: number;
  private readonly windowSize: number;

  constructor(threshold: number, windowSize = 10) {
    this.threshold = threshold;
    this.windowSize = windowSize;
  }

  recordToolCall(toolName: string, paramsHash: string): void {
    this.toolCallHistory.push({ toolName, paramsHash, timestamp: Date.now() });
    if (this.toolCallHistory.length > this.windowSize) {
      this.toolCallHistory.shift();
    }
  }

  recordMessage(text: string): void {
    // Normalize whitespace for comparison
    const normalized = text.replace(/\s+/g, " ").trim();
    this.messageHistory.push({ text: normalized, timestamp: Date.now() });
    if (this.messageHistory.length > this.windowSize) {
      this.messageHistory.shift();
    }
  }

  detectToolLoop(): LoopInfo | null {
    if (this.toolCallHistory.length < this.threshold) return null;

    const recent = this.toolCallHistory.slice(-this.threshold);
    const first = recent[0];
    const allSame = recent.every(
      (entry) => entry.toolName === first.toolName && entry.paramsHash === first.paramsHash
    );

    if (allSame) {
      return { count: this.threshold, key: `${first.toolName}:${first.paramsHash}` };
    }

    // Also check for same tool with similar params (for read with same path)
    const sameTool = recent.every((entry) => entry.toolName === first.toolName);
    if (sameTool) {
      const samePath = new Set(recent.map((e) => {
        try {
          const params = JSON.parse(e.paramsHash);
          return params.path;
        } catch {
          return e.paramsHash;
        }
      }));
      if (samePath.size === 1) {
        return { count: this.threshold, key: `same_tool:${first.toolName}` };
      }
    }

    return null;
  }

  detectMessageLoop(): LoopInfo | null {
    if (this.messageHistory.length < this.threshold) return null;

    const recent = this.messageHistory.slice(-this.threshold);
    const first = recent[0].text;

    const allSame = recent.every((entry) => entry.text === first);
    if (allSame && first.length > 20) {
      return { count: this.threshold, key: `msg:${first.slice(0, 50)}` };
    }

    return null;
  }

  getRandomWarning(): string {
    return randomWarning();
  }
}

// Sequence Detector

interface SequenceEntry {
  type: "tool_call" | "tool_result" | "assistant_text";
  key: string;
}

class SequenceDetector {
  private entries: SequenceEntry[] = [];
  private readonly minSequenceLength = 3;
  private readonly maxWindowSize = 20;

  recordEntry(entry: SequenceEntry): void {
    this.entries.push(entry);
    if (this.entries.length > this.maxWindowSize) {
      this.entries.shift();
    }
  }

  detectRepeatedSequence(): { start: number; end: number; pattern: SequenceEntry[] } | null {
    if (this.entries.length < this.minSequenceLength * 2) return null;

    // Check for repeated sequences of length 3-5
    for (let seqLen = this.minSequenceLength; seqLen <= 5; seqLen++) {
      for (let i = 0; i <= this.entries.length - seqLen * 2; i++) {
        const first = this.entries.slice(i, i + seqLen);
        const second = this.entries.slice(i + seqLen, i + seqLen * 2);

        if (this.sequencesMatch(first, second)) {
          return {
            start: i + seqLen,
            end: i + seqLen * 2,
            pattern: first,
          };
        }
      }
    }

    return null;
  }

  private sequencesMatch(a: SequenceEntry[], b: SequenceEntry[]): boolean {
    if (a.length !== b.length) return false;
    return a.every((entry, i) => entry.type === b[i].type && entry.key === b[i].key);
  }
}

// Main Extension

export default function (pi: ExtensionAPI) {
  const config = loadConfig();

  if (!config.enabled) {
    // Extension loaded but disabled — no-op
    return;
  }

  // State

  const loopDetector = new LoopDetector(config.loopThreshold);
  const sequenceDetector = new SequenceDetector();
  const readCache = new Map<string, ReadCacheEntry>();
  const editFailureCounts = new Map<string, number>();
  const skillPaths = new Map<string, string>();
  let blockedActionsCount = 0;
  let recentErrorCount = 0;
  let lastToolError: string | null = null;
  let lastErrorFile: string | null = null;
  let lastErrorMessage: string | null = null;
  const currentBatchReads = new Set<string>(); // Track reads within a single batch
  let lastHallucinationCorrectionTime = 0; // Cooldown for hallucination corrections
  let hallucinationCorrectionCount = 0; // Track total corrections per session

  // Session Start

  pi.on("session_start", async (_event, ctx) => {
    // Reset state
    blockedActionsCount = 0;
    recentErrorCount = 0;
    lastToolError = null;
    lastErrorFile = null;
    lastErrorMessage = null;
    lastHallucinationCorrectionTime = 0;
    hallucinationCorrectionCount = 0;

    if (ctx.hasUI) {
      ctx.ui.setStatus("slm", "SLM active");
    }
  });

  // Before Agent Start — Build skill path map

  pi.on("before_agent_start", async (event, ctx) => {
    const skills = event.systemPromptOptions?.skills ?? [];

    // Clear and rebuild skill paths (may change between turns)
    skillPaths.clear();
    for (const skill of skills) {
      const baseDir = skill.sourceInfo?.baseDir;
      if (baseDir) {
        skillPaths.set(skill.name, baseDir);
        if (skill.sourceInfo?.path) {
          skillPaths.set(skill.sourceInfo.path, baseDir);
        }
      }
    }
  });

  // Tool Call Interception

  pi.on("tool_call", async (event: ToolCallEvent, ctx) => {
    if (!config.enabled) return;

    // Feature 1: Block write tool
    if (event.toolName === "write") {
      blockedActionsCount++;
      if (ctx.hasUI) ctx.ui.setStatus("slm", `SLM: ${blockedActionsCount} blocked`);
      return {
        block: true,
        reason:
          "The `write` tool is blocked for safety. Use `edit` instead to make targeted changes.\n\n" +
          "Use `edit` with this format:\n" +
          "- `path`: the file to modify\n" +
          "- `edits[].oldText`: the exact text to find (must match byte-for-byte)\n" +
          "- `edits[].newText`: the replacement text\n\n" +
          "First use `read` to get the current file content, then use `edit` with the exact text as `oldText`.",
      };
    }

    // Feature 1: Block destructive bash
    if (event.toolName === "bash") {
      if (isToolCallEventType("bash", event)) {
        const cmd = event.input.command;
        if (isDestructiveBash(cmd)) {
          blockedActionsCount++;
          if (ctx.hasUI) ctx.ui.setStatus("slm", `SLM: ${blockedActionsCount} blocked`);
          return {
            block: true,
            reason:
              `Blocked: bash command attempts to write to a file:\n  ${cmd}\n\n` +
              "Use the `edit` tool to modify files instead. For new files, `write` is also blocked — " +
              "use `edit` on an existing file or create a minimal file with `edit` first.\n\n" +
              "Edit format: `path`, `edits[].oldText` (exact text to find), `edits[].newText` (replacement).",
          };
        }
      }
    }

    // Feature 6: Resolve skill reference paths for read
    if (event.toolName === "read") {
      if (isToolCallEventType("read", event)) {
        const resolved = resolveSkillPath(event.input.path, skillPaths);
        if (resolved && resolved !== event.input.path) {
          event.input.path = resolved;
        }
      }
    }

    // Feature 12: Deduplicate reads
    if (event.toolName === "read") {
      if (isToolCallEventType("read", event)) {
        const cacheKey = getReadCacheKey(
          event.input.path,
          event.input.offset,
          event.input.limit
        );

        // Check within-batch duplicates (parallel tool calls)
        if (currentBatchReads.has(cacheKey)) {
          return {
            block: true,
            reason: randomReadDedupMessage(event.input.path),
          };
        }
        currentBatchReads.add(cacheKey);

        // Check cross-turn cache
        const cached = readCache.get(cacheKey);
        if (cached && (Date.now() - cached.timestamp) < 120000) {
          return {
            block: true,
            reason: randomReadDedupMessage(event.input.path),
          };
        }
      }
    }

    // Feature 8: Loop detection on tool calls
    if (isToolCallEventType("bash", event)) {
      loopDetector.recordToolCall("bash", JSON.stringify({ command: event.input.command }));
    } else if (isToolCallEventType("read", event)) {
      loopDetector.recordToolCall("read", JSON.stringify({ path: event.input.path }));
    } else if (event.toolName === "edit") {
      loopDetector.recordToolCall("edit", JSON.stringify(event.input));
    }

    const toolLoop = loopDetector.detectToolLoop();
    if (toolLoop && event.toolName !== "read") {
      // Inject warning via user message
      pi.sendUserMessage(randomWarning(), { deliverAs: "steer" });
    }

    // Check for re-read after error (Feature 10)
    if (
      event.toolName === "read" &&
      lastErrorFile &&
      isToolCallEventType("read", event)
    ) {
      if (event.input.path === lastErrorFile) {
        pi.sendUserMessage(
          `Instead of re-reading ${lastErrorFile}, analyze the error: ${lastErrorMessage}`,
          { deliverAs: "steer" }
        );
      }
    }
  });

  // Tool Result Modification

  pi.on("tool_result", async (event: ToolResultEvent, ctx) => {
    if (!config.enabled) return;

    // Track errors
    if (event.isError) {
      recentErrorCount = Math.min(recentErrorCount + 1, 10);
      lastToolError = event.content?.[0]?.text ?? "";
    } else {
      recentErrorCount = Math.max(recentErrorCount - 1, 0);
    }

    // Feature 2: Edit failure tracking
    if (event.toolName === "edit") {
      if (event.isError) {
        const filePath = (event.input as any)?.path ?? "unknown";
        const count = (editFailureCounts.get(filePath) ?? 0) + 1;
        editFailureCounts.set(filePath, count);

        // Read the actual file content for guidance
        let fileContent = "";
        try {
          const absPath = resolve(ctx.cwd, filePath);
          fileContent = await readFile(absPath, "utf-8");
        } catch {
          // File may not exist
        }

        let augmentedContent = event.content?.[0]?.text ?? "Edit failed.";

        if (count >= config.editThreshold) {
          // Threshold breached — change strategy
          augmentedContent += `\n\n[SLM: Edit threshold reached (${count} failures on ${filePath}). ` +
            `Strategy change required. Options:\n` +
            `1. Read the full file with \`read\` and provide a smaller, precise \`oldText\`\n` +
            `2. Break the edit into multiple smaller edits\n` +
            `3. Write the corrected content to a temp file and compare\n\n` +
            `Current file content (first 100 lines):\n\`\`\`\n${fileContent.split("\n").slice(0, 100).join("\n")}\n\`\`\`]\n`;
        } else if (count >= 2) {
          // Guidance on retry
          augmentedContent += `\n\n[SLM: ${randomEditRetryMessage(filePath, count)}\n\n` +
            `Current file content (around the edit area):\n\`\`\`\n${fileContent.split("\n").slice(0, 50).join("\n")}\n\`\`\`]\n`;
        }

        // Store error info for Feature 10
        lastErrorFile = filePath;
        lastErrorMessage = "Edit failed: text mismatch";

        return { content: [{ type: "text", text: augmentedContent }] };
      } else {
        // Success — reset counter
        const filePath = (event.input as any)?.path;
        if (filePath) {
          editFailureCounts.delete(filePath);
          // Invalidate read cache for this file
          for (const [key] of readCache) {
            if (key.startsWith(filePath + ":")) {
              readCache.delete(key);
            }
          }
        }
        lastToolError = null;
        lastErrorFile = null;
        lastErrorMessage = null;
      }
    }

    // Feature 11: Directory read errors → directory listing
    if (event.toolName === "read" && event.isError) {
      const errorText = event.content?.[0]?.text ?? "";
      if (errorText.includes("EISDIR")) {
        const dirPath = (event.input as any)?.path ?? "";
        try {
          const listing = await directoryListing(dirPath, ctx.cwd);
          return {
            content: [{ type: "text", text: listing }],
            isError: false,
          };
        } catch {
          // Return original error if listing fails
        }
      }
    }

    // Feature 12: Cache successful reads
    if (event.toolName === "read" && !event.isError) {
      const readInput = event.input as any;
      const cacheKey = getReadCacheKey(readInput.path, readInput.offset, readInput.limit);
      const content = event.content?.[0]?.text ?? "";
      readCache.set(cacheKey, {
        content,
        timestamp: Date.now(),
        offset: readInput.offset,
        limit: readInput.limit,
      });
    }

    // Invalidate cache on write success
    if ((event.toolName === "write" || event.toolName === "edit") && !event.isError) {
      const filePath = (event.input as any)?.path;
      if (filePath) {
        for (const [key] of readCache) {
          if (key.startsWith(filePath + ":")) {
            readCache.delete(key);
          }
        }
      }
    }

    // Feature 10: Error recovery for bash
    if (event.toolName === "bash" && event.isError) {
      const errorText = event.content?.[0]?.text ?? "";
      const parsedError = parseTraceback(errorText);

      if (parsedError) {
        const explanation = getErrorExplanation(parsedError);
        lastErrorFile = parsedError.file ?? null;
        lastErrorMessage = parsedError.message;
        lastToolError = errorText;

        return {
          content: [
            {
              type: "text",
              text: `${errorText}\n\n[SLM: ${explanation}]`,
            },
          ],
        };
      }

      lastToolError = errorText;
    }

    // Feature 7: Plain-language hints on other failures
    if (event.isError && event.toolName !== "edit" && event.toolName !== "bash" && event.toolName !== "read") {
      const errorText = event.content?.[0]?.text ?? "Unknown error";
      return {
        content: [
          {
            type: "text",
            text: `${errorText}\n\n[SLM: The tool call failed. Check the error message and try again with corrected parameters.]`,
          },
        ],
      };
    }
  });

  // Message Correction

  pi.on("message_end", async (event, ctx) => {
    if (!config.enabled) return;

    if (event.message.role !== "assistant") return;

    const textContent = event.message.content
      ?.filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("\n") ?? "";

    // Skip messages already corrected by SLM
    if (textContent.includes("[SLM: Here are the factual" )) return;

    // Feature 3: Skills listing hallucination
    if (detectSkillListing(textContent)) {
      const factualList = getFactualSkillsList(skillPaths);
      if (factualList) {
        const corrected = `[SLM: Here are the factual skills installed in this environment.]

${factualList}`;
        return {
          message: {
            ...event.message,
            content: event.message.content?.map((c: any) =>
              c.type === "text" ? { ...c, text: corrected } : c
            ),
          },
        };
      }
    }

    // Feature 4: Tools listing hallucination
    if (detectToolListing(textContent)) {
      const factualList = getFactualToolsList();
      const corrected = `[SLM: Here are the factual tools in this environment.]

${factualList}`;
      return {
        message: {
          ...event.message,
          content: event.message.content?.map((c: any) =>
            c.type === "text" ? { ...c, text: corrected } : c
          ),
        },
      };
    }

    // Feature 5: Skill invocation handling
    const invocation = detectSkillInvocation(textContent);
    if (invocation) {
      const availableSkills = [...skillPaths.keys()].filter(
        (k) => k.length < 20 // Only skill names, not paths
      );
      if (!availableSkills.includes(invocation.name)) {
        // Skill doesn't exist — inject correction
        pi.sendUserMessage(
          `The skill "${invocation.name}" doesn't exist. Available skills: ${availableSkills.join(", ") || "none"}.`,
          { deliverAs: "steer" }
        );
      }
    }

    // Feature 9: Hallucination correction (with cooldown to prevent loops)
    const hallucination = detectHallucination(textContent, recentErrorCount);
    if (hallucination) {
      // Cooldown: max 3 corrections per session, 60s between corrections
      const now = Date.now();
      if (hallucinationCorrectionCount >= 3) return; // Max corrections
      if (now - lastHallucinationCorrectionTime < 60000) return; // 60s cooldown

      lastHallucinationCorrectionTime = now;
      hallucinationCorrectionCount++;

      // Collect evidence from recent errors
      const evidence: string[] = [];
      if (lastToolError) {
        evidence.push(`Recent error: ${lastToolError.slice(0, 200)}`);
      }

      const correction = randomHallucinationCorrection();
      const fullMessage = evidence.length > 0
        ? `${correction}\n\n${evidence.join("\n")}`
        : correction;

      pi.sendUserMessage(fullMessage, { deliverAs: "steer" });
    }

    // Record message for loop detection
    loopDetector.recordMessage(textContent);
    const msgLoop = loopDetector.detectMessageLoop();
    if (msgLoop) {
      pi.sendUserMessage(randomWarning(), { deliverAs: "steer" });
    }

    // Record for sequence detection
    if (textContent.length > 10) {
      sequenceDetector.recordEntry({
        type: "assistant_text",
        key: textContent.replace(/\s+/g, " ").trim().slice(0, 100),
      });
    }
  });

  // Context Modification

  pi.on("context", async (event, ctx) => {
    if (!config.enabled) return;

    const messages = event.messages ?? [];
    let modified = false;

    // Feature 12: Remove duplicate read toolResults from context
    const seenReads = new Map<string, number>(); // cacheKey -> first index
    const filteredMessages = messages.filter((msg: any, idx: number) => {
      if (msg.type === "toolResult" && msg.toolName === "read") {
        const path = msg.input?.path ?? "";
        const offset = msg.input?.offset ?? 0;
        const limit = msg.input?.limit ?? 0;
        const key = `${path}:${offset}:${limit}`;

        const firstIdx = seenReads.get(key);
        if (firstIdx !== undefined) {
          // Duplicate — remove
          modified = true;
          return false;
        }
        seenReads.set(key, idx);
      }
      return true;
    });

    // Feature 13: Remove repeated sequences
    const seqLoop = sequenceDetector.detectRepeatedSequence();
    if (seqLoop) {
      // Inject synthetic steering
      pi.sendUserMessage(randomSequenceSteering(), { deliverAs: "steer" });
    }

    if (modified) {
      return { messages: filteredMessages };
    }
  });

  // Turn End

  pi.on("turn_end", async (event, ctx) => {
    if (!config.enabled) return;

    // Record tool results for sequence detection
    const toolResults = event.toolResults ?? [];
    for (const result of toolResults) {
      const toolName = result.toolName ?? "unknown";
      const isError = result.isError ?? false;
      sequenceDetector.recordEntry({
        type: "tool_result",
        key: `${toolName}:${isError ? "error" : "ok"}`,
      });
    }

    // Record tool calls for sequence detection
    const entries = ctx.sessionManager.getEntries();
    const lastAssistant = [...entries].reverse().find(e => e.type === "assistant");
    if (lastAssistant) {
      const toolCalls = (lastAssistant as any).toolCalls ?? [];
      for (const tc of toolCalls) {
        sequenceDetector.recordEntry({
          type: "tool_call",
          key: `${tc.name}:${JSON.stringify(tc.input)?.slice(0, 50)}`,
        });
      }
    }

    // Update status
    if (ctx.hasUI) {
      const status = `SLM: ${blockedActionsCount} blocked | ${recentErrorCount} errors`;
      ctx.ui.setStatus("slm", status);
    }
  });

  // Turn Start

  pi.on("turn_start", async (event, ctx) => {
    if (!config.enabled) return;
    // Clear batch tracking for new turn
    currentBatchReads.clear();
    // Clear error tracking at start of new turns (soft reset)
    // Keep recentErrorCount but clear specific error messages
    lastErrorFile = null;
    lastErrorMessage = null;
  });
}

// Helper Functions

function resolveSkillPath(path: string, skillPaths: Map<string, string>): string | null {
  if (isAbsolute(path)) return null;

  // Check if path looks like a skill reference
  for (const [skillName, baseDir] of skillPaths) {
    // Check if the path starts with a reference pattern
    if (path.startsWith("references/") || path.startsWith("scripts/")) {
      // This might be a relative skill reference — try to resolve
      const resolved = resolve(baseDir, path);
      return resolved;
    }
  }

  return null;
}

function getFactualSkillsList(skillPaths: Map<string, string>): string | null {
  const skills = [...skillPaths.entries()]
    .filter(([key]) => key.length < 20) // Only skill names, not paths
    .map(([name]) => name);

  if (skills.length === 0) return null;

  // Use a format that doesn't match the detection pattern itself
  let list = "Here are the skills installed in this environment:\n\n";
  for (const name of skills) {
    list += `- **${name}**: Installed skill\n`;
  }
  return list;
}

function getFactualToolsList(): string {
  const tools = [
    { name: "read", desc: "Read file contents (text files and images)" },
    { name: "write", desc: "Create or overwrite files" },
    { name: "edit", desc: "Make precise text replacements in files" },
    { name: "bash", desc: "Execute shell commands" },
  ];

  // Use a format that doesn't match the detection pattern itself
  let list = "Here are the tools in this environment:\n\n";
  for (const tool of tools) {
    list += `- **${tool.name}**: ${tool.desc}\n`;
  }
  return list;
}

async function directoryListing(dirPath: string, cwd: string): Promise<string> {
  const absPath = isAbsolute(dirPath) ? dirPath : resolve(cwd, dirPath);

  try {
    const entries = await readdir(absPath, { withFileTypes: true });

    let listing = `**Directory listing for:** ${absPath}\n\n`;

    // Sort: directories first, then files
    entries.sort((a, b) => {
      if (a.isDirectory() && !b.isDirectory()) return -1;
      if (!a.isDirectory() && b.isDirectory()) return 1;
      return a.name.localeCompare(b.name);
    });

    for (const entry of entries) {
      if (entry.isDirectory()) {
        listing += `d ${entry.name}/\n`;
      } else {
        listing += `- ${entry.name}\n`;
      }
    }

    return listing;
  } catch (error: any) {
    return `Error listing directory ${absPath}: ${error.message}`;
  }
}
