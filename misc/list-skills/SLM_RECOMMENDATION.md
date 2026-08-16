# Recommendation — improving `LiquidAI/LFM2.5-2.6B` usage in `pi`

Companion to `SLM_ANALYSIS.md`. The measured defect: LFM2.5-2.6B **over-verifies** —
it re-reads the filesystem to confirm what the system prompt already declares
(5 LLM calls, 7 tool calls, 15.2× tokens vs Qwen3.8-27B on "list available skills"),
and then answers with a less accurate view of what is actually *available*.
It follows explicit counter-instructions perfectly ("do not load them" → 0 tool calls),
so the fix is to move that rule from user phrasing into the harness — primarily as a
**pi extension**, plus a couple of non-extension settings.

## Strategy

One extension (e.g. `slm-guard.ts`) that activates **only when the active model is an
SLM** (config list by `provider/model`, or heuristic: declared params < ~8B). All hooks
below are real pi extension APIs (see `docs/extensions.md`).

### 1. Inject a "declared state is authoritative" rule into the system prompt  *(highest impact)*

`before_agent_start` fires per user prompt and can replace the system prompt:

```ts
const SLM_RULES = `
SLM operating rules:
- The <available_skills> section of the system prompt is the COMPLETE, authoritative
  list of skills available in this session. If the user asks which skills exist or are
  available, answer directly from that section. Do NOT read SKILL.md files and do NOT
  list skill directories unless the user explicitly asks to inspect a specific skill.
- If the answer is already in the system prompt or in a previous tool result, answer
  directly. Do not re-verify information that was already provided.
- Files on disk that are not in <available_skills> are NOT available skills.`;

pi.on("before_agent_start", async (event, ctx) => {
  if (!isSLM(ctx.model)) return;                 // e.g. ctx.model.id.startsWith("LFM")
  return { systemPrompt: event.systemPrompt + "\n" + SLM_RULES };
});
```

Why this is the core fix: LFM-0 call 1 shows the model *saw* the three declared skills
and still chose to "verify by reading the actual skill files"; run 1 proves the same
model suppresses that behaviour under an explicit instruction. Encoding the rule
permanently removes dependence on the user's phrasing. Expected effect on the measured
task: LFM-0 becomes 1 call / 0 tools / ≈ 2.4k tokens / ≈ 3 s — matching Qwen.

Use `event.systemPromptOptions.skills` (exposed by the same event) to make the rule
conditional on skills actually being loaded, and to enumerate them explicitly
("the available skills are exactly: git, webfetch, websearch") — an explicit list helps
a 2.6B model more than a general rule.

### 2. Guard `read` on skill files  *(second line of defence)*

`tool_call` can mutate input or block (the block `reason` is fed back to the model as a
tool result, which doubles as teaching):

```ts
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";

pi.on("tool_call", async (event, ctx) => {
  if (!isSLM(ctx.model)) return;
  if (isToolCallEventType("read", event)) {
    if (/\/SKILL\.md$/.test(event.input.path ?? "")) {
      // Default to frontmatter only (name + description ≈ 60–100 tokens)
      // instead of the 2.5k–4k token full body:
      if (event.input.limit === undefined) event.input.limit = 15;
    }
  }
});
```

Option A (shown): cap `limit` so a stray `read` of SKILL.md costs ≈ 100 tokens
(frontmatter) instead of ≈ 9.1k for the five full bodies. Option B: return
`{ block: true, reason: "Skill bodies load on demand only when the task matches the skill description. Re-issue with explicit offset/limit if you truly need the full file." }`
— stronger, and the reason text reinforces rule 1 in-context.

This converts LFM's instinct to inspect files from *expensive and wrong* into *cheap
and bounded*: verification stays possible (the model is not blind), but a full-body
read can no longer silently dominate the context. Keep the exception: reads of
`references/*.md` after a skill was deliberately loaded stay unrestricted.

### 3. Right-size the thinking level for the SLM

Run 0 burned ≈ 923 thinking tokens (59% of output) mostly re-planning; Qwen used 44 at
`xhigh`. `high` is the wrong default for a 2.6B model in pi:

- Recommended: run LFM2.5 at `--thinking low` (or `off` for lookup-style prompts).
- Extension support: on `model_select`, if the new model is an SLM, call
  `pi.setThinkingLevel("low")` (and `ctx.ui.notify` the change); on `thinking_level_select`
  show a status hint. This is notification/state, not enforcement — keep the user in
  control.

Expected effect: shorter reasoning, fewer "let me double-check" escalations, and
meaningfully less VRAM/prefill per call on the local llama.cpp server.

### 4. Bound tool-result size for SLMs  *(context hygiene)*

`tool_result` handlers can rewrite results. Cap results aimed at SLMs (e.g. > 2–4k
chars → keep head + tail + a pointer to fetch more via `read` offset/limit or a
narrower `bash` command). Rationale: LFM-0's call 5 prefilled a 12.9k-token prompt
built up by unbounded reads; every future call in the session would re-pay for it
(19k cached tokens in this one short session). SLMs also degrade in recall with long
contexts, so truncation protects answer quality, not just cost.

### 5. Optional: a zero-cost `list_skills` tool

`pi.registerTool("list_skills", …)` returning name + one-line description of the loaded
skills (the same data pi already has in `systemPromptOptions.skills`). It gives the SLM
a deterministic, ~0-token tool path for "what skills are there?".
**Trade-off:** every registered tool costs schema tokens in *every* request, and a 2.6B
model pays for that permanently — so prefer rules 1+2 first; add the tool only if
post-hoc session logs show enumeration questions still cause file reads.

### 6. Optional: per-turn budget telemetry

On `turn_end`/`agent_end`, accumulate per-turn tokens and tool-call counts for SLM
sessions; `ctx.ui.notify` (or a status line) when a turn exceeds a budget (e.g. > 3
tool calls on an information-only prompt, or > 8k prompt tokens). This turns the
over-verification pattern from invisible into measurable, so rules 1–2 can be tuned
against real sessions. (`ctx.compact()` exists as a heavier fallback, but compaction is
lossy — avoid leaning on it.)

## Non-extension levers (no code needed)

| Lever | Setting | Why |
|---|---|---|
| Thinking level | `pi --model lfm2.5 --thinking low` | §3; halves output tokens on this task class |
| Skills hygiene | enable only skills actually needed (`pi config`) | fewer declared skills = smaller prompt *and* fewer temptations to read files; also fixes the "5 dirs vs 3 declared" ambiguity that made LFM's answer look plausible |
| Sampling | keep temp 0.1 / top_k 50 / repeat_penalty 1.1 | already in pi's model config for LFM; do not loosen — run 0 shows no benefit from more randomness |
| Server | keep llama.cpp prefix caching warm | 19 013/31 098 prompt tokens were cache-reads; identical system prompts across sessions (this extension keeps that stable — append the rule, don't rebuild the prompt) |

## What NOT to do

- **Blanket-ban file reads for SLMs.** On genuine coding tasks LFM *must* read files;
  the defect is default policy, not capability. Rules 1+2 leave an explicit escape
  (user says "inspect skill X" → unbounded read).
- **Replace the system prompt wholesale.** pi appends context files and skills after
  `--system-prompt`; the extension must append, and should key off
  `event.systemPromptOptions` rather than re-discovering resources.
- **Auto-escalate model selection silently** (`model_select` is user-facing). Notifying
  "this task may benefit from a larger model" is fine; swapping models mid-task is not.

## Validation plan

Re-run the four measured sessions with the extension active (same prompts, same
llama.cpp router):

1. "list available skills" on LFM2.5 → expect **1 LLM call, 0 tools, ≈ 2.2–2.6k
   tokens, ≤ 5 s** (vs 5 calls / 7 tools / 32.7k / 16.4 s).
2. "just list avilable skills, do not load them" → unchanged 1 call / 0 tools.
3. Negative control: "show me the git skill's commit-message rules" → the model *should*
   read `git/SKILL.md` (frontmatter cap lifted or `references/08` read) — verify the
   guard does not block legitimate skill loading.
4. A real multi-step coding task (e.g. from `misc/list-tools-*` or a scratch repo) to
   confirm no regression on tasks that do need file access; compare turns and tokens
   against Qwen3.8-27B on the same task.

Success criterion: LFM2.5-2.6B matches Qwen3.8-27B's call/tool count on
information-only questions while keeping its ≈ 2.6× per-token speed, with thinking
level `low`, and no blocked reads on legitimate skill-triggered tasks.
