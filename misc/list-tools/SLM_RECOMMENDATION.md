# Recommendation: improving `LiquidAI/LFM2.5-2.6B` usage in `pi`

**Basis:** `SLM_ANALYSIS.md` (this directory). The measured failure mode is:

1. **Interpretation error** — SLM confuses *skills* (documentation in the system prompt) with
   *tools* (the actual callable functions), because the skills block is more salient.
2. **Exploration detour** — SLM answers by 6 tool calls / 5 LLM turns what a bigger model
   answers in 1 turn with 0 tool calls.
3. **No budget** — nothing stops the exploration loop; each extra turn costs a full LLM
   round-trip and confabulates when it hits gaps.

**Constraint (per review):** no new slash commands, no new LLM-callable tools, no new skills.
All changes are made through **existing pi extension event hooks** (one extension file) plus
non-extension configuration. Nothing new is added to the model's vocabulary — the extension
only *filters, rewrites, and caps* what already exists.

**Design principle:** don't ask the SLM to *reason its way* to an answer the harness already
knows. Make the right answer **deterministic** (the harness answers or rewrites the question
before the LLM sees it) or **salient** (the harness restates it where the SLM actually looks),
and cap the cost of being wrong — without new surfaces.

---

## R1 (highest value) — Deterministic handling of meta-questions (`input` hook)

A meta-question like "list available tools" should not need the LLM — it is exactly the task
where the SLM fails. The `input` event fires when user input is received, **before**
skill/template expansion and before `before_agent_start`; a handler may return
`{action: "handled"}` (skip the agent entirely, extension shows its own feedback) or
`{action: "transform", text}` (rewrite the prompt, then continue normally). This is a
transparent interception of free-form text — **not** a new command.

- **TUI/RPC mode** (`ctx.hasUI`): match meta-question patterns
  (`/^(list|show|what)\b.*available tools?/i`), excluding prompts that mention "skill",
  "command", or names of actual project tools → print the exact tool list via
  `ctx.ui.notify()` (data from `pi.getActiveTools()`) and return `{action: "handled"}`.
  Zero tokens, zero latency, zero model errors.
- **print/JSON mode** (no UI): return `{action: "transform"}` with an unambiguous prompt,
  e.g. *"Your callable tools are exactly: read, write, edit, bash. Skills (git, webfetch,
  websearch) are documentation, not tools. List only the tools."* — this is exactly the
  correction that made LFM answer correctly in run 2 (F6), now applied automatically.

Gate on "active model is small" (see R4) so flagship models keep their normal behaviour; make
it user-disableable via a setting (the extension reads its config; no new CLI flag is
required — a `registerFlag` is optional and can be omitted to stay within the constraint).

## R2 — SLM-aware system prompt reinforcement (`before_agent_start`)

For small models, patch the system prompt for every turn. `before_agent_start` receives
`event.systemPrompt` and the structured `event.systemPromptOptions`
(`.selectedTools`, `.toolSnippets`, `.skills`), so the extension can rebuild it precisely —
no new resources are involved, only a rewrite of the existing prompt.

- **Append (recency!) a disambiguation block at the end of the system prompt**, after the
  skills section — for a 2.6B model the last thing read dominates:

  ```
  ## TOOLS vs SKILLS (read this carefully)
  TOOLS are the only functions you can call right now:
  - read: …(one line each, rebuilt from systemPromptOptions.selectedTools)
  SKILLS (git, webfetch, websearch) are instruction documents. They are NOT tools and are
  never called as functions. To use a skill you read its SKILL.md with the read tool.
  If the user asks which tools exist, answer immediately from the list above.
  Never call tools to answer a question your context already answers.
  ```

- **Compress the `<available_skills>` block** for small models to name + one-line description
  (replace the section in `event.systemPrompt` with a compact rebuild from
  `systemPromptOptions.skills`). This removes the salient decoy that caused the conflation
  (F1) while keeping the existing skills discoverable.

- The last sentence of the block is the **explicit anti-detour guideline** — targets F2/F3
  directly at the source (the prompt), not at the symptom.

## R3 — Tool-turn budget guard for small models (anti-detour, `tool_call`/`turn_start`)

Even with R1–R2, the SLM may still go on exploration detours for *real* tasks. Cap it using
hooks that only **block or nudge existing tool calls** — no new tool is added:

- Track turns with `agent_start` (reset counter per user prompt) and `turn_start`.
- In `tool_call` (which **can block**), when the model is small and the current agent run has
  already used ≥ N turns of inspection (N = 3–4), block further *inspection* calls:

  ```ts
  return { block: true,
    reason: "Tool budget reached. Answer the user's question now using the information you already have. Do not call any more tools." };
  ```

  The blocked result is returned to the model as a tool error, which reliably forces a final
  answer (F3, F7). Block only pure inspection (`read`, and `bash` whose command starts with
  `ls/rg/grep/find/cat/head/tail/…`); `write`/`edit` and mutating bash stay unrestricted so
  genuine multi-step coding work is unaffected. Log every block via `pi.appendEntry()`
  (existing session-entry API) so it is reviewable in the session.
- Softer variant: on the budget turn, use the `context` event (fires before each LLM call,
  messages are modifiable) to append a nudge message instead of hard-blocking.

## R4 — Model detection (shared by R1–R3)

The extension must only alter behaviour for small models. `ctx.model` is available in
`input` / `before_agent_start` / `tool_call`. Use an explicit, overridable list plus a size
heuristic:

```ts
const SLM_PATTERNS = [/lfm/i, /-2\.6b\b/i, /-1\.5b\b/i, /-3b\b/i]; // extend as needed
function isSLM(ctx: ExtensionContext) {
  const id = ctx.model?.id ?? "";
  return SLM_PATTERNS.some(re => re.test(id));
}
```

Allow per-project override in `.pi/slm-assist.json`
(`{ "slmModels": ["LiquidAI/LFM2.5-2.6B"], "metaIntercept": true, "toolBudget": 3 }`) and a
`model_select` listener (notification only) so the user sees when the active model crosses
into/out of SLM mode.

## R5 — Non-extension configuration changes (apply independently)

Settings/CLI-level, not extension code, but they measurably reduce the failure surface for
SLMs:

1. **Don't load unneeded skills for SLM sessions.** `--no-skills` (or `pi config`) removes
   the entire decoy section from the prompt. For a 2.6B model running simple tasks, 3 skills
   + 4 tools in one prompt is more concepts than it can reliably disambiguate. Load skills
   explicitly with `--skill <path>` only when needed.
2. **Sampling:** LFM2.5's configured `temperature: 0.1` / `repeat_penalty: 1.1` are very
   conservative. Try `temperature: 0.3–0.5`, `repeat_penalty: 1.0` for this model (pi's
   llamacpp model config) — moderate temperature helps a small model escape a wrong initial
   interpretation within a single turn. Re-benchmark both ways.
3. **Keep thinking ON (`high`), do not lower it.** LFM's traces at `high` were present and
   directionally useful; F8 shows the gap vs `xhigh`-Qwen is depth of self-correction, not
   presence. A future `minimal`/`low` comparison is worth running, but don't remove thinking
   to save tokens — the output-token savings are small (reasoning tokens were a minority of
   the 1344) and it removes the only self-check the model has.

## R6 — Benchmarking hygiene (so future comparisons are meaningful)

- Run each model with a **cold server** (or fresh context) to avoid the KV-cache artifact
  seen in LFM-1 (`cacheRead: 1270` from the previous session).
- Use ≥3 task classes (meta-question, single-file code change, multi-file task), ≥3 repeats.
- Record per-turn: latency, input/output/cache tokens, tool calls, and a pass/fail rubric.
  The existing `pi-llm-debugging` req/res capture is a good mechanism; an extension can also
  log payloads via the `before_provider_request` hook automatically.

---

## Reference: single extension implementing R1–R4

```typescript
// ~/.pi/agent/extensions/slm-assist.ts
// Event hooks only: no registerTool, no registerCommand, no new skills.
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const SLM_PATTERNS = [/lfm/i, /-2\.6b\b/i];
const META_Q = /^(list|show|what)\b.*available tools?/i;
const TOOL_BUDGET = 3;
const INSPECT_RE = /^(ls|rg|grep|find|cat|head|tail|wc|pwd|tree)\b/;

const isSLM = (ctx: ExtensionContext) =>
  SLM_PATTERNS.some((re) => re.test(ctx.model?.id ?? ""));

export default function (pi: ExtensionAPI) {
  let turnsInRun = 0;

  pi.on("agent_start", () => { turnsInRun = 0; });
  pi.on("turn_start",  () => { turnsInRun += 1; });

  // R1: deterministic meta-questions (no new command — transparent input interception)
  pi.on("input", (event, ctx) => {
    if (!isSLM(ctx) || !META_Q.test(event.text) ||
        /skill|command/i.test(event.text)) return; // pass through
    if (ctx.hasUI) {
      ctx.ui.notify("Tools: " + pi.getActiveTools().join(", "), "info");
      return { action: "handled" }; // agent skipped; zero LLM cost
    }
    return { action: "transform",
      text: `Your callable tools are exactly: ${pi.getActiveTools().join(", ")}. ` +
            `Skills are instruction documents, not tools. List only the tools.` };
  });

  // R2: SLM-aware system prompt (recency: append after the skills block)
  pi.on("before_agent_start", (event, ctx) => {
    if (!isSLM(ctx)) return;
    const tools = event.systemPromptOptions.selectedTools
      .map(t => `- ${t.name}: ${t.snippet ?? ""}`.trim())
      .join("\n");
    return { systemPrompt: event.systemPrompt +
      `\n\n## TOOLS vs SKILLS (read carefully)\n` +
      `TOOLS are the only functions you can call:\n${tools}\n` +
      `SKILLS are instruction documents (read their SKILL.md with the read tool); they are NOT tools.\n` +
      `If the user asks which tools exist, answer immediately from the list above.\n` +
      `Never call tools to answer a question your context already answers.` };
  });

  // R3: tool-turn budget for SLMs (caps inspection detours, allows real work)
  pi.on("tool_call", (event, ctx) => {
    if (!isSLM(ctx) || turnsInRun < TOOL_BUDGET) return;
    const isInspection =
      event.toolName === "read" ||
      (event.toolName === "bash" && INSPECT_RE.test(String(event.input.command ?? "")));
    if (!isInspection) return; // allow write/edit and bash mutations
    return { block: true,
      reason: "Tool budget reached. Answer the user now using the information you already have; do not call more tools." };
  });
}
```

Notes:
- Only existing extension APIs are used: `input`, `before_agent_start`, `tool_call`,
  `agent_start`, `turn_start` events, `pi.getActiveTools()`, `ctx.ui.notify`, and
  (optionally) `pi.appendEntry()` to log budget blocks. Nothing is registered for the LLM to
  see.
- The R3 guard only blocks *inspection* calls (`read`, bash ls/rg/find/…) beyond the budget;
  `write`/`edit` and mutating bash commands stay unrestricted. Tune `TOOL_BUDGET` and
  `INSPECT_RE` per model.
- `input` handlers see raw text before skill/template expansion and run before
  `before_agent_start`; returning `handled` skips the agent entirely (TUI/RPC only, hence the
  `ctx.hasUI` guard; `print`/JSON modes fall back to the transform path).
- Everything is inert for non-SLM models, so the same extension is safe to keep installed
  alongside `Qwen3.8-27B` or any larger model.

## Expected impact (from the measured runs)

| Item | Before (LFM-0) | After R1, TUI | After R1 transform / R2 (no UI, or pattern not matched) |
|---|---|---|---|
| "list avilable tools" | 5 turns, 6 tool calls, 1344 out-tok, **wrong** | **0 LLM calls, exact, ~0 s** | 1 turn, 0 tool calls, correct (as in LFM-1) |
| Detour risk on real tasks | unbounded (F3) | n/a | capped at `TOOL_BUDGET` inspection turns (R3) |

The single most important change is R1: it converts the SLM's observed worst case into a
deterministic harness operation with no new command, tool, or skill in existence. R2 makes
unmatched phrasings robust by putting the disambiguation where the SLM actually looks, and R3
bounds the damage of any remaining misinterpretation.
