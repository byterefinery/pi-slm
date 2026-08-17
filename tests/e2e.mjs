#!/usr/bin/env node
/**
 * E2E tests for the slm extension (src/slm.ts).
 *
 * Spawns isolated `pi` instances in a random temp dir (one fresh session per
 * scenario) against the local `LiquidAI/LFM2.5-2.6B` model, and asserts on
 * the persisted session JSONL (startup messages, converted calls, appended
 * hints, branch structure) plus on-disk side effects.
 *
 *   node tests/e2e.mjs               # run all scenarios (~5-10 min)
 *   node tests/e2e.mjs t2 t5a        # run a subset (see SCENARIOS)
 *   PI_BIN=/path/to/pi node tests/e2e.mjs
 *   PI_MODELS_JSON=/path/models.json # models config to copy in (default: ~/.pi/agent/models.json)
 *
 * Runs fully isolated: an ephemeral PI_CODING_AGENT_DIR inside the temp dir
 * holds the session logs and the (copied) models config, so nothing outside
 * the temp dir is read or written, and the user's pi settings are untouched.
 *
 * Scenarios that rely on the model following explicit instructions are
 * retried up to `attempts` times (fresh session each attempt); failures print
 * a session diagnostic dump.
 */

import { spawnSync } from "node:child_process";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  readdirSync,
  rmSync,
  chmodSync,
  copyFileSync,
  existsSync,
} from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join, resolve } from "node:path";

// ============================================================================
// Setup
// ============================================================================

const REPO = resolve(new URL("..", import.meta.url).pathname);
const EXT = join(REPO, "src", "slm.ts");
const PI = process.env.PI_BIN || "pi";
const MODEL = "LiquidAI/LFM2.5-2.6B:high";
const SKILL = "sktest";
// Project skills are discovered under <cwd>/.agents/skills (walked to the git
// root); the temp dir is not a git repo, so only the temp dir itself counts.
const SKILL_DIR = ".agents/skills/sktest";

const TMP = mkdtempSync(join(tmpdir(), "pi-slm-e2e."));
const SKILL_PATH = join(TMP, SKILL_DIR, "SKILL.md");
const SKILL_REF_PATH = join(TMP, SKILL_DIR, "references", "01-notes.md");
const SKILL_MD = [
  "---",
  "name: sktest",
  "description: Test skill for slm e2e validation.",
  "---",
  "",
  "# sktest",
  "",
  "Do the test thing.",
  "",
].join("\n");

/**
 * (Re)create the skill and make it read-only: a wandering model may read the
 * skill file, see the structure hint appended to the result, and "helpfully"
 * rewrite the file — corrupting the frontmatter and killing discovery for
 * every later session in the run. Read-only files make that write fail
 * (and exercise feature 6 instead).
 */
function ensureSkill() {
  mkdirSync(join(TMP, SKILL_DIR, "references"), { recursive: true });
  for (const [path, content] of [[SKILL_PATH, SKILL_MD], [SKILL_REF_PATH, "# notes\n"]]) {
    try {
      chmodSync(path, 0o644);
    } catch {}
    writeFileSync(path, content);
    chmodSync(path, 0o444);
  }
}
ensureSkill();

// Fully isolated agent dir inside the temp dir: the test never touches the
// user's real ~/.pi/agent (no retry:100 amplification of provider stalls,
// no session logs outside the temp dir, no global skills).
const AGENT_DIR = join(TMP, ".pi-agent");
mkdirSync(AGENT_DIR, { recursive: true });
const GLOBAL_MODELS = join(homedir(), ".pi", "agent", "models.json");
if (!existsSync(process.env.PI_MODELS_JSON || GLOBAL_MODELS)) {
  console.error(`missing ${GLOBAL_MODELS} (or set PI_MODELS_JSON); the test needs the llamacpp provider config for LiquidAI/LFM2.5-2.6B`);
  process.exit(2);
}
copyFileSync(process.env.PI_MODELS_JSON || GLOBAL_MODELS, join(AGENT_DIR, "models.json"));

// Same encoding pi uses (session-manager: strip leading slash, / \ : → -, wrap in --…--).
const SESSIONS_DIR = join(AGENT_DIR, "sessions", `--${TMP.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`);
mkdirSync(SESSIONS_DIR, { recursive: true });

const PI_ENV = {
  ...process.env,
  PI_CODING_AGENT_DIR: AGENT_DIR,
  PI_OFFLINE: "1", // no startup network ops (version check, telemetry)
  PI_SKIP_VERSION_CHECK: "1",
  PI_TELEMETRY: "0",
};

// ============================================================================
// pi runner
// ============================================================================

function listSessions() {
  return readdirSync(SESSIONS_DIR).filter((f) => f.endsWith(".jsonl"));
}

/** Run one `pi -p` session in the temp dir; returns { code, stdout, entries, file }. */
function runPi(prompt, { timeoutMs = 240_000 } = {}) {
  const before = new Set(listSessions());
  const r = spawnSync(
    PI,
    ["--no-extensions", "-a", "-e", EXT, "-p", "--model", MODEL, prompt],
    { cwd: TMP, timeout: timeoutMs, encoding: "utf8", maxBuffer: 10 * 1024 * 1024, env: PI_ENV },
  );
  const fresh = listSessions().filter((f) => !before.has(f));
  if (fresh.length !== 1) {
    // pi exited before creating a session (startup error, provider failure...)
    return {
      code: r.status,
      stdout: r.stdout ?? "",
      stderr: r.stderr ?? `no session created (${fresh.length} new files)`,
      entries: [],
      file: null,
      noSession: true,
    };
  }
  const file = join(SESSIONS_DIR, fresh[0]);
  let entries = [];
  try {
    // On timeout the session file exists but the run never settled.
    entries = readFileSync(file, "utf8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l));
  } catch {}
  if (r.error) {
    return { code: null, stdout: r.stdout ?? "", stderr: `pi killed after ${timeoutMs / 1000}s`, entries, file, timedOut: true };
  }
  return { code: r.status, stdout: r.stdout ?? "", stderr: r.stderr ?? "", entries, file };
}

/** Match a tool-call path argument on its basename (models often use absolute paths). */
const pathIs = (name) => (args) => String(args.path ?? "").replace(/\/+$/, "").split("/").pop() === name;

/** Match a bash command that runs exactly `sleep N` (not e.g. sleep 750). */
const sleepCmd = (n) => (args) => new RegExp(`(^|\\s|;)sleep ${n}(\\s|$|;)`).test(String(args.command ?? ""));

/**
 * Find the last tool call matching `match` (and its result) so that extra
 * model calls (echo, pwd, retries) cannot skew the assertion.
 */
function resultFor(entries, toolName, match) {
  const tcs = toolCalls(entries, toolName).filter((t) => match(t.args));
  if (tcs.length === 0) return { missing: `no ${toolName} call matching ${match.toString()}` };
  const tc = tcs[tcs.length - 1];
  const tr = toolResults(entries, toolName).find((t) => t.toolCallId === tc.id);
  if (!tr) return { missing: `${toolName} call has no result` };
  return { tc, tr };
}

// ============================================================================
// Session JSONL helpers
// ============================================================================

const msg = (e) => (e.type === "message" ? e.message : null);
const text = (blocks) =>
  (blocks ?? [])
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text)
    .join("\n");

function toolCalls(entries, name) {
  const out = [];
  for (const e of entries) {
    const m = msg(e);
    if (!m || m.role !== "assistant") continue;
    for (const b of m.content ?? []) {
      if (b.type === "toolCall" && (!name || b.name === name)) {
        out.push({ entry: e, id: b.id, name: b.name, args: b.arguments ?? {} });
      }
    }
  }
  return out;
}

function toolResults(entries, name) {
  return entries
    .filter((e) => msg(e)?.role === "toolResult" && (!name || msg(e).toolName === name))
    .map((e) => ({
      id: e.id,
      parentId: e.parentId,
      toolName: msg(e).toolName,
      toolCallId: msg(e).toolCallId,
      isError: msg(e).isError === true,
      blocks: (msg(e).content ?? []).filter((b) => b.type === "text").map((b) => b.text),
      text: text(msg(e).content),
      timestamp: e.timestamp, // entry-level ISO timestamp (result persisted)
    }));
}

/** Walk the active branch from the last persisted entry back to the root. */
function leafChain(entries) {
  const byId = new Map(entries.map((e) => [e.id, e]));
  const chain = [];
  let cur = entries[entries.length - 1];
  while (cur) {
    chain.push(cur);
    cur = cur.parentId != null ? byId.get(cur.parentId) : undefined;
  }
  return chain;
}

function diag(run) {
  const { entries } = run;
  const chain = leafChain(entries)
    .map((e) =>
      e.type === "message"
        ? `message(${msg(e).role}${msg(e).toolCallId ? ":" + msg(e).toolCallId.slice(0, 6) : ""}) ${e.id}`
        : `${e.type}${e.customType ? "(" + e.customType + ")" : ""} ${e.id}`,
    )
    .join(" <- ");
  const calls = toolCalls(entries)
    .map((t) => `${t.name} ${JSON.stringify(t.args).slice(0, 200)}`)
    .join(" | ");
  const results = toolResults(entries)
    .map((t) => `${t.toolName} err=${t.isError} ${t.text.slice(0, 160).replace(/\n/g, " ")}`)
    .join(" | ");
  return [`  leaf chain: ${chain}`, `  tool calls: ${calls || "(none)"}`, `  tool results: ${results || "(none)"}`, `  stdout: ${run.stdout.slice(0, 200).replace(/\n/g, " ")}`, `  stderr: ${run.stderr.slice(0, 300).replace(/\n/g, " ")}`];
}

// ============================================================================
// Shared assertions
// ============================================================================

/** Feature 3: every new session must start with the two synthetic messages. */
function checkStartup(run, errs) {
  const { entries } = run;
  const firstMsg = entries.findIndex((e) => e.type === "message");
  const sk = entries.findIndex((e) => e.type === "custom_message" && e.customType === "slm_available_skills");
  const tl = entries.findIndex((e) => e.type === "custom_message" && e.customType === "slm_available_tools");
  if (sk < 0 || tl < 0) return errs.push("startup messages missing");
  if (firstMsg >= 0 && (sk > firstMsg || tl > firstMsg)) errs.push("startup messages not before first message");
  if (tl !== sk + 1) errs.push("startup messages not adjacent");
  const skc = entries[sk].content;
  const tlc = entries[tl].content;
  if (!skc.startsWith("Available skills:")) errs.push("skills message bad prefix");
  if (!skc.includes(`- name: ${SKILL}`)) errs.push("skills message missing sktest");
  if (!skc.includes(`  path: ${SKILL_PATH}`)) errs.push("skills message missing absolute sktest path");
  if (!tlc.startsWith("Available tools:")) errs.push("tools message bad prefix");
  for (const n of ["read", "bash", "edit", "write"]) {
    if (!tlc.includes(`- name: ${n}`)) errs.push(`tools message missing ${n}`);
  }
  const b0 = tlc.indexOf("- name: bash");
  const b1 = tlc.indexOf("- name:", b0 + 1);
  const bash = b0 >= 0 ? tlc.slice(b0, b1 > 0 ? b1 : undefined) : "";
  if (!bash.includes("timeout:") || !bash.includes("type: number")) errs.push("bash yaml missing timeout:number");
  if (!bash.includes("strict: false")) errs.push("bash yaml missing strict flag");
}

/** Assert a tool result's appended hint text is present. */
function expectHint(tr, needle, errs, label) {
  if (!tr) return errs.push(`${label}: no tool result`);
  if (!tr.blocks.some((b) => b.includes(needle))) {
    errs.push(`${label}: hint missing (${needle.slice(0, 60)}...)`);
  }
}

// ============================================================================
// Scenarios
// ============================================================================

const SCENARIOS = {
  // Feature 1a: write on a NEW file → plain write delegation.
  t1a: {
    attempts: 3,
    setup: () => {},
    prompt:
      "Use the write tool to create the file t1a.txt with content: hello. " +
      "Then reply DONE. Do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "write", pathIs("t1a.txt"));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (tr.isError) return errs.push("write result errored");
      if (!tr.text.includes("Successfully wrote")) errs.push("write result not a plain write");
      const file = join(TMP, "t1a.txt");
      const content = readFileSync(file, "utf8").trim();
      if (content !== "hello") errs.push(`t1a.txt content: ${JSON.stringify(content)}`);
    },
  },

  // Feature 1b: write on an EXISTING file with a small change → converted to a
  // deterministic edit (result text comes from the built-in edit tool).
  t1b: {
    attempts: 3,
    setup: () => writeFileSync(join(TMP, "t1b.txt"), "alpha\nbeta\ngamma\n"),
    prompt:
      "Use the write tool to overwrite t1b.txt with exactly this 3-line content:\n" +
      "alpha\nBETA\ngamma\n" +
      "Then reply DONE. Do not verify, do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "write", pathIs("t1b.txt"));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (tr.isError) return errs.push("write result errored");
      if (!/Successfully replaced \d+ block\(s\) in .*t1b\.txt/.test(tr.text)) {
        errs.push(`write not converted to edit (result: ${tr.text.slice(0, 80)})`);
      }
      const content = readFileSync(join(TMP, "t1b.txt"), "utf8").trim();
      if (content !== "alpha\nBETA\ngamma") errs.push(`t1b.txt content: ${JSON.stringify(content)}`);
    },
  },

  // Feature 1c: write with identical content → no-op message, file untouched.
  t1c: {
    attempts: 3,
    // No trailing newline: the model writes the word verbatim, which then
    // matches the file exactly (no-op path). A stray "STABLE\n" becomes a
    // legitimate 1-line edit instead.
    setup: () => writeFileSync(join(TMP, "t1c.txt"), "STABLE"),
    prompt:
      "Use the write tool. Target file: t1c.txt. The new content must be exactly one line: STABLE\n" +
      "Then reply DONE. Do not verify, do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "write", pathIs("t1c.txt"));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (tr.isError) return errs.push("write result errored");
      const file = join(TMP, "t1c.txt");
      if (!tr.text.includes("No changes:")) {
        // A tiny diff (e.g. trailing newline) may legitimately become an edit.
        if (!/Successfully replaced \d+ block\(s\) in .*t1c\.txt/.test(tr.text)) {
          errs.push(`unexpected write result: ${tr.text.slice(0, 80)}`);
        }
      } else if (readFileSync(file, "utf8") !== "STABLE") {
        errs.push("no-op write modified the file");
      }
      if (readFileSync(file, "utf8").trim() !== "STABLE") errs.push("t1c.txt content changed");
    },
  },

  // Feature 2: edit with a fabricated oldText → blocked as empty edit, blocked
  // pair moved off the active branch, replacement hint appended, and print
  // mode still prints the model's final response.
  t2: {
    attempts: 3,
    setup: () => writeFileSync(join(TMP, "t2.txt"), "keep this line\n"),
    prompt:
      "Call the edit tool exactly once with path 't2.txt' and one edit: " +
      "oldText 'zzz-not-here', newText 'new'. " +
      "After that single edit call, reply with exactly: DONE\n" +
      "Do not read the file. Do not retry the edit. Do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      // The model may retry the bad edit after the first block (it is told to
      // re-read and retry). The removal applies to the LAST blocked pair; any
      // earlier pairs stay on the branch as legitimate history.
      const badEdits = toolCalls(run.entries, "edit").filter(
        (t) => String(t.args.path).endsWith("t2.txt") && JSON.stringify(t.args.edits ?? []).includes("zzz-not-here"),
      );
      if (badEdits.length === 0) return errs.push("no bad edit call found");
      const tc = badEdits[badEdits.length - 1];
      const tr = toolResults(run.entries).find((t) => t.toolCallId === tc.id);
      if (!tr) return errs.push("no result for the last bad edit");
      if (!tr.isError || !tr.text.includes("Empty edit:") || !tr.text.includes("was not applied")) {
        return errs.push(`bad edit not blocked as empty edit (isError=${tr.isError}): ${tr.text.slice(0, 120)}`);
      }
      const chain = leafChain(run.entries);
      const hint = run.entries.find((e) => e.type === "custom_message" && e.customType === "slm_hint");
      if (hint) {
        // Shape A/B: the blocked pair was the last activity, so it was removed
        // and the hint grafted in.
        if (!hint.content.includes("removed from this session log")) {
          return errs.push("slm_hint entry has wrong text");
        }
        if (chain[0] !== hint) return errs.push("leaf is not the hint entry");
        const reparented = chain[1];
        const rm = reparented ? msg(reparented) : null;
        if (!reparented || !rm || rm.role !== "assistant" || (rm.content ?? []).some((b) => b.type === "toolCall")) {
          return errs.push("hint parent is not a plain assistant message");
        }
        const blockedAsst = run.entries.find((e) => e.id === tr.parentId);
        if (!blockedAsst || blockedAsst.type !== "message" || msg(blockedAsst)?.role !== "assistant") {
          return errs.push("blocked pair's assistant not found");
        }
        if (reparented.parentId !== blockedAsst.parentId) errs.push("re-appended response not grafted at the blocked call's parent");
        const chainIds = new Set(chain.map((e) => e.id));
        if (chainIds.has(blockedAsst.id) || chainIds.has(tr.id)) errs.push("blocked pair still on the active branch");
      } else {
        // Guard path: the model recovered after the block (e.g. re-read the
        // file as instructed) — the history is kept as-is, no removal.
        const chainIds = new Set(chain.map((e) => e.id));
        if (!chainIds.has(tr.id)) return errs.push("no hint but blocked result off-branch (corrupt shape)");
        const leafMsg = msg(chain[0]);
        if (!leafMsg || leafMsg.role !== "assistant" || (leafMsg.content ?? []).some((b) => b.type === "toolCall")) {
          return errs.push("guard path: leaf is not the final assistant text");
        }
      }
      if (readFileSync(join(TMP, "t2.txt"), "utf8") !== "keep this line\n") errs.push("t2.txt was modified");
      if (run.stdout.trim().length === 0) errs.push("print mode printed nothing");
    },
  },

  // Feature 4: first successful read inside a known skill dir → structure hint
  // appended to the (non-error) result.
  t4: {
    attempts: 3,
    setup: () => {},
    prompt: `Use the read tool to read the file ${SKILL_DIR}/SKILL.md. Then reply OK. Do not call any other tool.`,
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "read", (a) => String(a.path).includes(`${SKILL_DIR}/SKILL.md`));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (tr.isError) return errs.push("skill read errored");
      if (!tr.blocks[0]?.includes("Test skill for slm e2e validation")) errs.push("read result missing file content");
      expectHint(tr, "Skill 'sktest' structure (absolute paths):", errs, "t4");
      expectHint(tr, "references:", errs, "t4");
      expectHint(tr, join(TMP, SKILL_DIR, "references", "01-notes.md"), errs, "t4");
    },
  },

  // Feature 5a: bash call without timeout → base 60 s injected.
  t5a: {
    attempts: 3,
    timeoutMs: 360_000,
    setup: () => {},
    prompt:
      "Use the bash tool to run the command: sleep 75\n" +
      "Do not pass a timeout argument. Do not run any other command. " +
      "After it finishes or is killed, report the result in one line.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "bash", sleepCmd(75));
      if (found.missing) return errs.push(found.missing);
      const { tc, tr } = found;
      if (typeof tc.args.timeout === "number") return errs.push(`model set explicit timeout ${tc.args.timeout}s`);
      if (!tr.isError) return errs.push("sleep 75 completed (no injected timeout?)");
      const m = tr.text.match(/Command timed out after (\d+) seconds/);
      if (!m) return errs.push(`no timeout error text: ${tr.text.slice(0, 120)}`);
      if (m[1] !== "60") errs.push(`timeout was ${m[1]}s, expected base 60s`);
    },
  },

  // Feature 5b: after a timeout failure the default escalates by +60 s
  // (60 → 120): a no-timeout `sleep 75` retry must complete at the escalated
  // default (without escalation it would die at 60 s again).
  t5b: {
    attempts: 4,
    timeoutMs: 480_000,
    setup: () => {},
    prompt:
      // Escalation across turns. The first no-timeout `sleep 75` dies at the
      // base 60s; the natural model behavior is to retry the same command,
      // which then runs at the escalated 120s default and COMPLETES (75 <
      // 120). Without escalation the retry would die at 60s again.
      "Use the bash tool to run the command: sleep 75\n" +
      "Do not pass a timeout argument (the system manages timeouts). " +
      "If it times out, retry the same command exactly once, again without a timeout argument. " +
      "Do not run any other command. Then report the results of both attempts in one line.",
    assert(run, errs) {
      checkStartup(run, errs);
      const calls = toolCalls(run.entries, "bash").filter((t) => sleepCmd(75)(t.args));
      if (calls.length < 2) return errs.push(`expected 2 sleep-75 calls (initial + retry), got ${calls.length}`);
      for (const tc of calls) {
        if (typeof tc.args.timeout === "number") return errs.push(`model set an explicit timeout (${tc.args.timeout}s); re-run`);
      }
      const [f1, f2] = calls;
      if (f1.entry.id === f2.entry.id) return errs.push("model bundled both calls into one turn (re-run)");
      const tr1 = toolResults(run.entries, "bash").find((t) => t.toolCallId === f1.id);
      const tr2 = toolResults(run.entries, "bash").find((t) => t.toolCallId === f2.id);
      if (!tr1 || !tr2) return errs.push("missing result for a sleep-75 call");
      const m1 = tr1.text.match(/Command timed out after (\d+) seconds/);
      if (!tr1.isError || !m1 || m1[1] !== "60") errs.push(`first call not killed at 60s: ${tr1.text.slice(0, 100)}`);
      if (tr2.isError) return errs.push(`retry call still failed (no escalation?): ${tr2.text.slice(0, 100)}`);
      // The retry must have actually run sleep 75 to completion (~75s).
      const durMs = Date.parse(tr2.timestamp) - Date.parse(f2.entry.timestamp);
      if (!(durMs >= 65_000 && durMs <= 200_000)) errs.push(`retry duration ${durMs}ms inconsistent with sleep 75 completing`);
    },
  },

  // Feature 6: tool failure → the failed tool's YAML definition appended.
  t6: {
    attempts: 3,
    setup: () => {},
    prompt: "Use the read tool to read the file ghost-file-xyz.txt. Then tell me the error text. Do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "read", (a) => String(a.path).includes("ghost-file-xyz.txt"));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (!tr.isError) return errs.push("read unexpectedly succeeded");
      expectHint(tr, "Tool 'read' failed. Recall its definition:", errs, "t6");
      expectHint(tr, "- name: read", errs, "t6");
      expectHint(tr, "strict:", errs, "t6");
    },
  },

  // Feature 7a: failed read inside a known skill dir → that skill's hint.
  t7a: {
    attempts: 3,
    setup: () => {},
    prompt: `Use the read tool to read the file ${SKILL_DIR}/MISSING.md. Then tell me the error text. Do not call any other tool.`,
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "read", (a) => String(a.path).includes(`${SKILL_DIR}/MISSING.md`));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (!tr.isError) return errs.push("read unexpectedly succeeded");
      expectHint(tr, "Skill 'sktest' structure (absolute paths):", errs, "t7a");
      expectHint(tr, join(TMP, SKILL_DIR, "references", "01-notes.md"), errs, "t7a");
    },
  },

  // Feature 7b: failed read of an unknown skill → available-skills list.
  t7b: {
    attempts: 3,
    setup: () => {},
    prompt: "Use the read tool to read the file .pi/skills/ghostskill/SKILL.md. Then tell me the error text. Do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "read", (a) => String(a.path).includes("ghostskill"));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (!tr.isError) return errs.push("read unexpectedly succeeded");
      expectHint(tr, "Unknown skill 'ghostskill'. Available skills:", errs, "t7b");
      expectHint(tr, "- name: sktest", errs, "t7b");
    },
  },

  // Feature 8: read on a directory → EISDIR replaced by an ls -F listing.
  t8: {
    attempts: 3,
    setup: () => {
      mkdirSync(join(TMP, "adir"), { recursive: true });
      writeFileSync(join(TMP, "adir", "inner.txt"), "x\n");
    },
    prompt: "Use the read tool to read the path adir. Then say in one line what you see. Do not call any other tool.",
    assert(run, errs) {
      checkStartup(run, errs);
      const found = resultFor(run.entries, "read", pathIs("adir"));
      if (found.missing) return errs.push(found.missing);
      const { tr } = found;
      if (tr.isError) return errs.push("directory read still an error (EISDIR not converted)");
      if (!tr.text.startsWith("Directory listing of ")) errs.push(`bad listing prefix: ${tr.text.slice(0, 80)}`);
      if (!tr.text.includes("inner.txt")) errs.push("listing missing inner.txt");
    },
  },
};

// ============================================================================
// Runner
// ============================================================================

const wanted = process.argv.slice(2).length
  ? process.argv.slice(2)
  : Object.keys(SCENARIOS);
for (const w of wanted) {
  if (!SCENARIOS[w]) {
    console.error(`unknown scenario ${w}; available: ${Object.keys(SCENARIOS).join(", ")}`);
    process.exit(2);
  }
}

console.log(`slm e2e — temp dir: ${TMP}`);
console.log(`        extension:  ${EXT}`);
console.log(`        model:      ${MODEL}\n`);

const results = [];
let failed = false;

for (const name of wanted) {
  const spec = SCENARIOS[name];
  const t0 = Date.now();
  let lastErrs = null;
  let ok = false;
  for (let attempt = 1; attempt <= spec.attempts && !ok; attempt++) {
    ensureSkill();
    spec.setup();
    const run = runPi(spec.prompt, { timeoutMs: spec.timeoutMs });
    if (run.timedOut) {
      lastErrs = ["pi timed out (model likely looped)", ...diag(run).slice(1, 3)];
      continue;
    }
    if (run.noSession) {
      lastErrs = [`no session created (exit ${run.code}): ${run.stderr.slice(0, 200)}`];
      continue;
    }
    const errs = [];
    spec.assert(run, errs);
    if (errs.length === 0) {
      ok = true;
    } else {
      lastErrs = errs;
      if (attempt < spec.attempts) console.log(`  ${name} attempt ${attempt} failed, retrying...`);
    }
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  results.push([name, ok]);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  (${secs}s, ${spec.attempts} max attempts)`);
  if (!ok) {
    failed = true;
    console.log("  failures:");
    for (const e of lastErrs ?? []) console.log(`    - ${e}`);
  }
}

const passed = results.filter(([, o]) => o).length;
if (!failed) {
  rmSync(TMP, { recursive: true, force: true });
  console.log(`\n${passed}/${results.length} scenarios passed (temp dir cleaned: ${TMP})`);
} else {
  console.log(`\n${passed}/${results.length} scenarios passed (temp dir kept for inspection: ${TMP})`);
}
process.exit(failed ? 1 : 0);
