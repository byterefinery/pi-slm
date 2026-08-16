# SLM Recommendation — Improving LFM2.5-2.6B Usage in pi (webfetch task family)

**Companion to** `SLM_ANALYSIS.md` (sessions `01a00957…` and `01a00951…` vs Qwen3.8-27B baselines `01a00967…` / `01a00956…`).
**Goal:** close the first-move-reliability gap so a 2.6B SLM executes skill-driven tasks the way Qwen3.8-27B does — first action correct, no operator nudge, canonical tool invocation — *by the harness, not the model*.

The analysis found four failure clusters, all detectable by the harness:

- **A. Action bias** — the URL in the prompt was treated as documentation; the model stopped to ask "what should I fetch?" (both sessions).
- **B. Skill-script invocation** — literal copy of `webfetch.py <url>` from the skill doc → `127` ×3, `uv run` without a path, `find /`, `chmod +x` detour, one repeated identical command; `python3` instead of `uv run --script` → degraded fetcher.
- **C. Instruction fidelity** — `tzip … full` answered as "lite" twice; active pruning mode ignored.
- **D. Confabulation** — thinking stream claimed `chmod +x` fixed what the absolute path fixed.

Cluster B is the biggest single waste (7 of 13 turns in the tzip session) and is **100% derivable by the harness**: the skill directory, the script, and its shebang are all known to pi. Priority order: **P0 = do first, P1 = cheap high value, P2 = tuning, P3 = carry-over insurance.**

---

## 1. P0 — Canonical skill-invocation guard (new extension, highest impact)

### 1.1 Skill script registry (built once, from what pi already loaded)

`before_agent_start` receives `event.systemPromptOptions.skills` — the loaded skills and their locations. For each skill directory, scan for scripts (recursively, e.g. `scripts/*.py`) and read the first line:

- shebang `#!/usr/bin/env -S uv run --script` (PEP 723) → canonical runner `uv run --script <abs-path>`
- executable bit, plain shebang → `<abs-path>`
- otherwise → `python3 <abs-path>`

Store `basename → { abs, run(args) }` in a session map. In the analyzed environment this yields exactly one entry: `webfetch.py → uv run --script /home/…/.agents/skills/webfetch/scripts/webfetch.py`.

### 1.2 Rewrite or block at `tool_call` (bash)

`tool_call` fires before execution, `event.input` is **mutable in place, and mutations affect the actual tool execution** (verified against `docs/extensions.md`). Rules, in order:

1. **Bare script name** (`webfetch.py …`, possibly after `cd … &&`): rewrite `event.input.command`, replacing the bare name with the canonical invocation, preserving all arguments. No user-visible change; the model's plan is unchanged, only the invocation is made real.
2. **Wrong interpreter for a PEP 723 script** (`python3 <skill-script>`, `uv run <basename>`): same rewrite — this is what kills the environment bypass that degraded the webfetch-session output (regex `html_to_md` instead of scrapling/markdownify).
3. **Unresolvable but skill-related** (e.g. name close to a registry entry): `return { block: true, reason: "…canonical command: <exact>" }` — the block reason is the feedback the model receives, so state the command verbatim.

Expected effect on the recorded sessions:

- tzip session turns 4–11 (7 bash calls, 4 failures, ~70 s, `find /`, `chmod +x`, repeated command) collapse to **one** canonical call at turn 4; fetch by ~06:47:37 instead of 06:48:22.
- webfetch session turn 8 (`python3 scripts/webfetch.py --md … 2>&1`) becomes `uv run --script …/scripts/webfetch.py https://tangledgroup.com` → scrapling output, same turn.

### 1.3 Why rewrite instead of block-and-teach

Blocking with a reason teaches the model (good for the record), but for a 2.6B model the fastest path is the harness silently emitting the command the model *meant*. Block-and-teach is kept only for the ambiguous cases where the harness cannot safely rewrite. This mirrors the known-issues feature "Tool invocation: inject plain-language hints when tool calls fail validation", applied at pre-execution instead of post-execution.

---

## 2. P1 — Explicit task framing (fixes the action bias, clusters A + C)

`before_agent_start` can inject a message and extend the system prompt for the turn (`event.prompt` is available). Detect the pattern that failed in both sessions — a `<skill name="…">…</skill>` block whose prompt continues with a bare argument:

- **URL argument** (`https?://…`): inject a persistent message:
  > Action: fetch `<URL>` now with the `<skill>` script. Run the canonical command. Do not ask for confirmation. Do not substitute a different URL (e.g. example.com).
- **Short mode/argument word** (e.g. `full`, `off`): inject:
  > Skill activation requested with mode `<arg>`. Reply with the required confirmation that echoes the mode verbatim, in at most 2 sentences. No menus, no explanations.

Plus a one-line systemPrompt tail for SLM runs: *"A URL or mode word appearing after a skill block in a user message is the task input — act on it in your first tool call."*

Expected effect: eliminates the 1–2 operator re-prompts per session (both LFM sessions needed one; the tzip session needed three prompts for one fetch), the `example.com` drift, and most of the "both skills active — what would you like to do?" turns. The activation directive doubles as the tzip mode fix (cluster C): the required reply format names the mode word, which a 2.6B model can echo even when it cannot parse the intent.

---

## 3. P1 — Shell-error hint injection (backstop when §1 misses)

`tool_result` (bash, `isError`) can patch the result content before it reaches the model. Pattern the failure signatures seen in the data:

- `command not found` (exit 127):
  - name in the skill registry → append: *"127 means the shell could not find `<name>` in PATH. Use the canonical invocation: `<uv run --script <abs> <args>`."*
  - otherwise → *"127 = program not found in PATH. For scripts use an absolute path or `./relative/path`. Do not `chmod`. Do not search the whole filesystem (`find /`)."*
- `No such file or directory` on a read/exec path → *"The file is not at that path. List the parent directory (`ls <dir>`) to find the actual location."* (This is the one class LFM already recovers from — the hint only shortens it.)

The explicit **do-nots** matter: both observed detours (`chmod +x` on an already-755 file, `find /` at 19.6 s) were rational-sounding but wrong, and a 2.6B model will not self-exclude them.

Expected effect: even without §1, the tzip-session detour becomes ≤2 turns (127 → hint → canonical path) instead of 8.

---

## 4. P2 — Verbosity & configuration budget

| Setting | Recorded (LFM) | Recommended | Why |
|---|---|---|---|
| `max_tokens`/response | 49,152 | **8,192** | Largest response observed was 862 output tokens; 8k is 9× headroom and bounds runaways |
| Thinking level | `high` | **A/B test `low`; default `low` for skill-execution tasks** | The 832-token thinking block before the tzip fix produced *zero* correct hypotheses; on these short tasks long CoT is cost without signal. Keep `high` only if A/B shows a recovery-rate gain |
| `temperature` | 0.1 | 0.1 baseline; **0.4–0.5 burst for 1–2 requests after a repeated identical failure** (via `before_provider_request` payload override) | The byte-identical `cd && webfetch.py` ×2 is a mild attractor; a temporary stochastic bump is cheap escape insurance (carry-over from the lisp-python finding) |
| `repeat_penalty` | 1.1 | 1.1 | fine |
| Skill catalog | full project catalog | **curate to the task's skill(s)**; minimal `AGENTS.md` | Every extra skill/resource in the prompt is a lure: the redundant SKILL.md re-read and the menu-style "what mode do you want?" replies are attention scattered across options |
| Confirmation style | (none) | SystemPrompt tail: *"Confirmations and status replies: max 2 sentences. No option menus."* | 862-token confirmation to a one-word activation (Qwen: 118) |

---

## 5. P2 — Redundant-read deduplication

`tool_call` (read): if the target path's **full content is already in the current context** — inside a user message (as with the skill block that *is* SKILL.md) or as an earlier successful read result for the same path with no intervening write — block with a reason:

> "Full content of `<path>` is already in context (see turn N). No changes since. Do not re-read; act on what you have."

Observed waste this removes: the SKILL.md re-read in webfetch turn 1 (~1k tokens of context), and — with a softer variant ("if you only need to *run* the script, run it; don't read 13 KB first") — the whole-script pre-reads in both sessions (~3.6k tokens each). This generalizes the known-issues "repeated `read` dedup" rule from "same path, no changes" to "content already present in context".

---

## 6. P3 — Final-summary grounding (anti-confabulation)

The tzip session's thinking stream claimed `chmod +x` fixed the failure; the evidence says the absolute path did. For SLM runs, at `turn_end` (or via `message_end`, which can replace the finalized assistant message):

- If the turn is a **final text-only reply** for a command-execution task, extract causal claims ("after X, it worked" / "X fixed …").
- Cross-check against tool history: X must be immediately followed by a successful result for it to stand.
- On contradiction, append one line to the reply: *"Correction: the command that actually succeeded was: `<exact last successful command>`."*
- Regardless of claims found, **append the exact last successful command** to command-task summaries — it makes every summary verifiable and costs ~15 tokens.

This is the "corrects hallucinations" feature specialized to the one confabulation class observed in this family (causal misattribution in the reasoning stream, user-facing text intact).

---

## 7. P3 — Carry-over: repetition circuit breaker

Keep the loop guard specified for the lisp-python session (identical `(args, failing result)` ×3 consecutive → block with imperative recovery hint; ×5 → `terminate` + `ctx.ui.notify`). Neither webfetch-family session doom-looped (tasks too short), but the tzip session already emitted the same failing command twice (turns 7/10) — one more repetition and it would have crossed a threshold. The breaker is cheap insurance against the regime documented in `misc/lisp-python/SLM_ANALYSIS.md`, where the same model burned 1,498 identical failures.

---

## 8. Traceability: issue → fix → mechanism → effect

| # | Issue (from analysis) | Fix | pi mechanism / hook | Effect in the recorded sessions |
|---|---|---|---|---|
| 1 | URL not executed; "what should I fetch?" (both LFM sessions) | Task framing (§2) | `before_agent_start` message + systemPrompt injection | Fetch on first tool call; 1–2 re-prompts and 2–3 turns saved per session |
| 2 | `example.com` drift (webfetch T6) | Task framing (§2) | same | N/A — requested URL fetched at T2 |
| 3 | Bare-name `127` ×3 + `uv run` w/o path (tzip T4–T5, T7, T10) | Canonical invocation guard (§1) | `tool_call` in-place `event.input.command` rewrite | 7 bash calls → 1 |
| 4 | `find /` 19.6 s (tzip T6) | Guard + 127 hint bans `find /` (§1, §3) | `tool_call` / `tool_result` | N/A |
| 5 | `chmod +x` detour (tzip T9) | 127 hint: "Do not chmod" (§3) | `tool_result` content patch | N/A |
| 6 | `python3` env bypass → degraded fetcher (webfetch T8) | Canonical invocation `uv run --script` (§1) | `tool_call` rewrite | scrapling/markdownify output, same turn |
| 7 | `tzip full` → "lite" ×2 | Activation compliance directive (§2) | `before_agent_start` (required reply names the mode) | confirmation echoes `full` |
| 8 | 862-token confirmations, un-pruned style | Verbosity budget (§4) | config + systemPrompt tail | ≤2-sentence confirmations (~10× fewer tokens) |
| 9 | SKILL.md re-read (webfetch T1); 13 KB pre-reads ×2 | Redundant-read dedup (§5) | `tool_call` block with reason | ~1k + ~3.6k×2 tokens saved |
| 10 | `chmod` causal confabulation (tzip T12 thinking) | Summary grounding (§6) | `turn_end` / `message_end` check + correction append | claim contradicted and corrected in the record |
| 11 | Identical failing command ×2 (tzip T7/T10) | Circuit breaker carry-over (§7) | `tool_call` block at ×3 | stops by the 3rd repetition |
| 12 | Mild re-sampling attractor | Adaptive temperature (§4) | `before_provider_request` payload override | stochastic escape after repeated failure |

---

## 9. Concrete pi setup for LFM2.5-2.6B (starting point)

```bash
pi --model LiquidAI/LFM2.5-2.6B --thinking low \
   --skill .agents/skills/webfetch \        # curate to the task's skill(s)
   -e ./slm-skill-guard.ts                  # §1 + §2 + §3 + §5 + §6 in one extension
```

- **Provider params:** `temperature 0.1` (0.5 burst on repeated failure), `repeat_penalty 1.1`, `max_tokens 8192`.
- **Circuit breaker:** identical failing `(args, result)` ×3 → block with imperative hint; ×5 → terminate + notify (§7).
- **A/B:** `--thinking low` vs `high` on a small task suite; keep only what the recovery metrics justify.

### `slm-skill-guard.ts` sketch (illustrative; hook signatures verified against `docs/extensions.md`)

```ts
import { isToolCallEventType, isBashToolResult } from "@earendil-works/pi-coding-agent";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, basename } from "node:path";

type Entry = { name: string; abs: string; run: (args: string) => string };
const registry = new Map<string, Entry>();
let activation: { skill: string; arg: string } | null = null;

function registerSkillDir(dir: string) {
  const walk = (d: string) => {
    for (const f of readdirSync(d)) {
      const p = join(d, f);
      statSync(p).isDirectory() ? walk(p) :
      f.endsWith(".py") && !registry.has(f) && registry.set(f, makeEntry(p));
    }
  };
  try { walk(dir); } catch {}
}
function makeEntry(abs: string): Entry {
  const shebang = readFileSync(abs, "utf8").split("\n")[0] ?? "";
  const run = (args: string) =>
    shebang.includes("uv run --script")
      ? `uv run --script ${abs}${args ? " " + args : ""}`
      : statSync(abs).mode & 0o111
        ? `${abs}${args ? " " + args : ""}`
        : `python3 ${abs}${args ? " " + args : ""}`;
  return { name: basename(abs), abs, run };
}

pi.on("before_agent_start", (event, ctx) => {
  for (const s of event.systemPromptOptions.skills ?? []) registerSkillDir(s.dir ?? s.location ?? "");
  const m = /<skill name="([^"]+)">[\s\S]*?<\/skill>\s*(\S+)\s*$/.exec(event.prompt ?? "");
  const out: Record<string, unknown> = {};
  if (m) {
    activation = { skill: m[1], arg: m[2] };
    const isUrl = /^https?:\/\//.test(m[2]);
    out.message = {
      customType: "slm-skill-guard", display: false,
      content: isUrl
        ? `Action: fetch ${m[2]} now with the ${m[1]} script. Run the canonical command. Do not ask, do not substitute another URL.`
        : `Skill activation requested with mode "${m[2]}". Reply with the confirmation echoing the mode verbatim, max 2 sentences. No menus.`,
    };
    out.systemPrompt = event.systemPrompt +
      "\n\nA URL or mode word after a skill block is the task input — act on it in your first tool call. Confirmations: max 2 sentences.";
  }
  return out;
});

pi.on("tool_call", (event, ctx) => {
  if (isToolCallEventType("bash", event)) {
    const c = event.input.command;
    for (const e of registry.values()) {
      const bare = new RegExp(`(^|[;&|]\\s*|\\s)${e.name}(\\s|$)`);
      if (bare.test(c) && !c.includes(e.abs)) {
        const args = c.replace(new RegExp(`.*?${e.name}(\\s.*)?$`), "$1").trim();
        event.input.command = c.replace(e.name, e.run(args));   // in-place: affects real execution
        return;
      }
    }
  }
  if (isToolCallEventType("read", event)) {
    if (fullContentAlreadyInContext(ctx, event.input.path))
      return { block: true, reason: `Full content of ${event.input.path} is already in context. Do not re-read; act on what you have.` };
  }
});

pi.on("tool_result", (event, ctx) => {
  if (!isBashToolResult(event) || !event.isError) return;
  const text = JSON.stringify(event.content);
  if (/command not found/.test(text)) {
    const e = [...registry.values()].find(e => event.input.command.includes(e.name));
    const hint = e
      ? `Hint: 127 = not in PATH. Canonical invocation: ${e.run("<args>")}. Do not chmod. Do not use find /.`
      : `Hint: 127 = program not found in PATH. Use an absolute path or ./relative/path. Do not chmod. Do not search the whole filesystem.`;
    return { content: [ { type: "text", text: text + "\n" + hint } ] };
  }
});

pi.on("turn_end", (event, ctx) => {
  // §6: for text-only final replies, verify causal claims against tool history;
  // append "Correction: the command that actually succeeded was: <…>" on contradiction,
  // and always append the exact last successful command to command-task summaries.
});
```

(Helper functions omitted; `fullContentAlreadyInContext` checks user-message text and prior read results for the same path via `ctx.sessionManager`; the §6 claim check is a regex over the final text matched against the tool-result sequence.)

---

## 10. Expected impact

Applying P0+P1+P2 to the same task suites:

- **LFM tzip session:** 13 requests / 8 tool calls / 4 failures / 5 user messages → ~6 requests / 2–3 tool calls / 0 failures / 2 user messages; fetch completes ~45 s earlier; `tzip full` acknowledged correctly.
- **LFM webfetch session:** 9 requests → ~5; no nudge; scrapling-quality output on turn 2.
- **Output tokens:** confirmation/verbosity cuts bring LFM's 5,249-token tzip session toward ~2,000 (the 3.4–7.2× gap vs Qwen narrows to ~2×, which is plausibly the capability floor for a 2.6B model on these tasks).
- **Residual gap** (what the harness cannot fix): the model still needs one extra exploration turn for anything not derivable from the skill registry, and its summaries remain chatty. That is model-side (training/instruction-tuning territory, as in the lisp-python P5).

**Bottom line:** for short skill-execution tasks, LFM2.5-2.6B becomes competitive in *wall-clock-to-result* once the harness supplies the two things the model demonstrably lacks — a canonical invocation it never had to discover, and an explicit "act now" frame it never inferred. The loop-killer carry-over keeps it bounded when the tasks get longer.
