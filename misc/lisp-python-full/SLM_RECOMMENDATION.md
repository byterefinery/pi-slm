# SLM Recommendation — Overcoming LFM2.5-2.6B Limitations in the pi Harness

**Companion to** `SLM_ANALYSIS.md` (session `01a004af-…`, LiquidAI/LFM2.5-2.6B via llama.cpp).
**Goal:** make small-language-model (SLM) runs in pi *bounded, grounded, and recoverable* — even when the model itself is not.

The analysis identified two failure layers:

- **Capability layer** — the model cannot map tracebacks to code, ships off-by-one parsers, repeats elementary Python syntax errors (`else` as lambda param, hyphen module import, `try` without `except`).
- **Behavior layer (worse)** — on hitting its own bug it collapses into a byte-identical loop (~1,500 identical failing tool calls over 40 min), confabulates a success that never happened, makes zero corrective edits, produces zero user-facing text, and burns ~144M tokens.

The behavior layer is the one worth engineering against: it is **detectable** (identical tool-call/result hashes) and **preventable** (circuit breaker + context hygiene). The capability layer is only partially mitigable — for the rest, scope the model to tasks it can do.

## 0. Four design principles for SLMs in an agent harness

1. **Never allow a failing action to repeat unmodified.** A 2.6B model will not stop itself; the harness must.
2. **Keep context small, recent, and grounded in current state.** SLMs cannot use 124k context effectively; stale context is what turns a bug into a 40-minute confabulation.
3. **Budget everything.** Turns, tokens, wall-clock. A budget breach is a first-class stopping condition.
4. **Verify externally.** Syntax pre-checks, checkpoints, and a supervisor — not the model's self-report.

Priority order below: **P0 = do first (kills the dominant failure), P1–P2 = cheap high value, P3–P5 = structural.**

---

## 1. P0 — Loop killing (highest impact)

### 1.1 Repeated-action circuit breaker (pi extension)

The entire 40-minute catastrophe in the analyzed session was a single detectable pattern: same `bash` args → same error result, repeated. A breaker that fires at 3 repetitions would have ended it at ~09:20:15, saving ≈138M tokens and ~40 minutes.

pi supports this natively: `tool_call` handlers **can block** a tool call (`{ block: true, reason, terminate }`), and `tool_execution_end` exposes `event.result` / `event.isError`.

Rules:
- Track a rolling window of `(toolName, hash(args), hash(result), isError)`.
- **Same (args, failing result) ×3 consecutive** → block the next identical call; the block *reason* becomes the feedback the model receives. Make the reason an imperative recovery instruction with the exact file/line from the traceback:
  `"You ran the same failing command 3 times. Do not repeat it. Read <file> lines <a>–<b> (from the traceback), fix the code, then re-run."`
- **×5** → block with `{ block: true, terminate: true }` and `ctx.ui.notify(...)` — stop the run, hand back to the user.
- Also trip on **identical full assistant responses** (hash of thinking+toolCalls), not just tool calls: two identical responses in a row is already an attractor signal.

### 1.2 Context deduplication of repeated pairs (`context` hook)

The last request in the analyzed session contained **71 consecutive copies of the same (thinking, tool call, traceback) triple** — a context engineered to be uninformative, because every copy said "run the same command." The `context` event fires before each LLM call with a mutable deep copy of `event.messages`.

- Collapse consecutive identical `(assistant toolCall → toolResult)` pairs into **one** copy, annotated: `"(this action and its identical failure repeated N times)"`.
- Effects: (a) removes the attractor structure the model was reproducing; (b) keeps the *latest* error at the end of context (recency); (c) shrinks the loop context by ~60–90%, which interacts well with principle 2.
- This is the single change that most directly attacks the "no belief update" failure: the model stops seeing a wall of repetition and starts seeing "repeated N times — you must change something."

### 1.3 Budgets (turns / tokens / time)

- **Max tool calls per task:** ~60 (the analyzed task legitimately needed ~15–20).
- **Max total tokens per task:** ~2M (session burned 144M).
- **Wall-clock:** ~10 min for a "simple interpreter" task.
- On breach: `ctx.abort()`, `ctx.ui.notify(...)`, then optionally one final *text-only* LLM call ("Report status: what works, what is broken, exact next step") so the run exits with a useful summary instead of silence.

---

## 2. P1 — Context hygiene (breaks confabulation)

### 2.1 Loop-aware compaction summaries (`session_before_compact`)

pi-vcc's compaction *reinforced* the loop: the summary listed the same failing command 30+ times as "work in progress," truncated the error to its first line (`Traceback (most recent call last):`), and commanded "Continue directly where you left off" — and "where you left off" was the loop. Extensions can supply a custom summary via `session_before_compact` (`return { compaction: { summary, … } }`). For SLM runs, the summary must contain:

1. **The full last error text** (traceback included, not the first line).
2. **Actions tried → outcomes** (compact list).
3. An explicit **DO-NOT-REPEAT** list: "Ran `python3 lisp-interpreter.py "(+ 1 2)"` N times — all failed with TypeError. Do not run it again without changing the code."
4. **Current state grounding**: which files exist, the last edit to each, and the one-line goal.
5. Replace "Continue directly where you left off" with **"The last action failed repeatedly. Take a *different* action: read the failing code first."** — only when the recent history is a repeated failure.

### 2.2 Compact earlier for SLMs

The session pinned context at 123,944 tokens; a 2.6B model has no business reasoning over 124k tokens (attention span + lost-in-the-middle). Target a **~32–48k working context** for SLM sessions (pi-vcc threshold if configurable; otherwise prune older messages in the `context` hook). Smaller context → more salient recent error → less confabulation. This also makes dedup (§1.2) and re-grounding (§2.3) cheaper.

### 2.3 State re-grounding (force fresh reality into context)

The model's last *read* of the failing file was at 09:17:28 — 42 minutes before the session ended. Its beliefs were then purely self-generated (and confabulated). Mechanism:

- On `tool_execution_end` with `isError`, parse `File "<path>", line <n>` from the output.
- If the same file has now failed ≥2 times, block the next identical call (§1.1) **and** include in the block reason / an injected message the **current on-disk excerpt** (lines n±15) of that file — freshly read by the harness, not by the model.
- This guarantees the model sees the actual current code (the `depth = 0` at line 130 was sitting in that window, fixable by a 2.6B model if it could only see it).

### 2.4 Forced status reports

The session produced **zero text** across 1,573 responses. Require a 1–2 sentence status (what is broken, what I will try next) at: (a) compaction-resume, (b) every N-th failure, (c) budget-breach. Implement via `before_agent_start` message injection or by appending to the resume prompt. Benefits: the user sees the run is stuck (in the analyzed session nothing surfaced until the user looked), and forcing the model to externalize its belief makes confabulation reviewable by a human/supervisor.

---

## 3. P2 — Decoding & configuration (cheap tuning)

| Setting | Current (session) | Recommended for LFM2.5 | Why |
|---|---|---|---|
| Thinking level | `high` | **A/B test `off` vs `low`**; default off | "High" thinking produced 2,737 chars of *stale, confabulated* text per loop iteration, at the cost of tokens/time. No evidence it helped; for SLMs long CoT often hurts. Keep it only if an A/B shows recovery-rate gain. |
| temperature | 0.1 | 0.1 baseline, **0.4–0.7 for 1–2 requests after loop detection** | Byte-identical outputs 1,445× means top-1 dominance; a temporary temperature bump (via `before_provider_request` → `return { ...event.payload, temperature: 0.5 }`) is a cheap stochastic escape from the attractor. Pair with §1.1, don't rely on it alone. |
| repeat_penalty | 1.1 | 1.1–1.15 | Marginal; the repetition was semantic (whole sequences), not token-level. |
| max_tokens/response | 49,152 | **8,192** | Bounds runaway generations (one response *was* truncated mid tool-call at 09:22:25), and shorter responses trip loop detection sooner. |
| Context files / skills | full catalog (4.7k system prompt incl. skill list) | **`--no-skills`** (or one curated skill), minimal `AGENTS.md`, `--nc` where safe | Turn 1's stray `websearch/SKILL.md` read shows every irrelevant resource in the prompt is a lure for a 2.6B model. Trim prompt mass aggressively. |
| Recovery protocol | (none) | Append to system prompt, short + imperative + at the end: *"When a command fails: read the file at the line named in the error, change the code, then re-run. Never run the same failing command more than twice. If stuck after 3 attempts, stop and explain in one sentence."* | SLMs track short imperative rules better than long guideline lists; end-position gives recency. Won't fully prevent the loop (see P0) but raises the bar. |

---

## 4. P3 — External pre-verification (catch trivial errors before the model does)

**Syntax pre-flight:** after every `write`/`edit` touching `*.py`, the harness runs `python3 -m py_compile <file>` and appends the outcome to the tool result (via `tool_execution_end` extension). In the analyzed session this alone would have caught, *at write time*:

- `lambda cond, then, else:` (v1, 09:11:48) — the whole 09:11:56–09:12:10 edit/rewrite cycle;
- `from lisp-interpreter import run` ×4 across `test_simple.py`/`test_import.py` (09:12:16 → 09:18:15) — the entire test-file churn phase;
- `try:` without `except` in final `test_import.py`.

That is ~10–15 wasted turns and the removal of one of the two error classes the model could never pattern-match. Generalize: shellcheck for emitted shell, `node --check` for JS, etc. — pre-check the *language the model is bad at*, cheaply, before the model runs its own probe.

---

## 5. P4 — Workflow design (scope the model to its envelope)

1. **Checkpointed decomposition.** The task "implement interpreter + demos + run + fix all issues" is 4 tasks. Run them as separate sessions with machine-checkable gates between them:
   - Step 1: core interpreter passes 3 fixed expressions (`(+ 1 2)` → 3, `(* 3 4)` → 12, `(/ 10 2)` → 5.0). Gate: script exit 0.
   - Step 2: conditionals + loops pass 3 more. Gate.
   - Step 3: `define` + calls pass 2 more. Gate.
   - Step 4: run the demo files through the interpreter. Gate.
   SLMs succeed on bounded steps and collapse on open-ended ones.
2. **Scope limit for LFM2.5-2.6B:** single-file, single-feature tasks with an executable acceptance check. Not "build X and fix everything."
3. **Supervisor.** A larger model (or a human) reviews at each checkpoint and watches two cheap tripwires during runs: (a) identical tool-call hash ≥3, (b) claims in the thinking stream that no tool result supports (the `Result of (+ 1 2): 3` fabrication was visible in the stream in real time). Either tripwire ⇒ intervene.
4. **Instrument every run** so future model versions are comparable: no-loop rate, recovery rate (read the file within 5 turns of first failure?), tokens-to-first-pass, wall-clock-to-pass, confabulation count.

---

## 6. P5 — Model-side improvements (if you own the training pipeline)

1. **Agent SFT on recovery trajectories.** The missing skill is not "write code" but **fail → read the exact line → targeted edit → re-run**. Collect successful trajectories (from this model or larger models), rejection-sample the good ones, and oversample error-recovery segments specifically.
2. **RL with a loop penalty.** Penalize repeated identical (action, result) pairs in a trajectory; reward verified progress (new passing behavior). This directly optimizes the behavior that failed here.
3. **Post-training "recovery benchmark."** N scenarios: a broken file + a failing command. Metrics: % that read the file within 5 turns, % that produced a targeted edit, % reaching pass with ≤K retries, and identical-action rate. Run it per checkpoint; the analyzed session scores ≈0 on all of them.

---

## 7. Traceability: issue → fix → mechanism → effect on the analyzed session

| # | Issue (from analysis) | Fix | pi mechanism / hook | Effect in session `01a004af` |
|---|---|---|---|---|
| 1 | Frozen loop, 1,498 identical failures | Circuit breaker (§1.1) | `tool_call` block `{terminate:true}` at ×5; recovery nudge at ×3 | Run stops ~09:20:15; ≈138M tokens & ~40 min saved |
| 2 | 71-copy attractor context | Context dedup (§1.2) | `context` hook message rewrite | Loop context shrinks ~90%; latest error at end |
| 3 | Compaction reinforced the loop | Loop-aware summary (§2.1) | `session_before_compact` custom summary | Summaries say "do not repeat; fix `parse_s_expr`" instead of "continue where you left off" |
| 4 | Confabulated "successful run" | State re-grounding + status reports (§2.3, §2.4) | `tool_execution_end` + `before_agent_start` injection | Model sees fresh file excerpt incl. `depth = 0`; user sees stuck status |
| 5 | Zero file reads for 42 min (stale beliefs) | State re-grounding (§2.3) | same as #4 | Failing lines 130–132 entered context despite model not reading |
| 6 | 124k context for a 2.6B model | Earlier compaction (§2.2) | pi-vcc threshold / `context` pruning | ~40k working context; error stays salient |
| 7 | 4× hyphen-import & other SyntaxError churn | Syntax pre-flight (§4) | `tool_execution_end` + `py_compile` | ~10–15 turns eliminated; one error class removed |
| 8 | 0 text to user across 1,573 responses | Forced status reports (§2.4) | `before_agent_start` / `turn_end` | User sees the stuck state by ~iteration 5 |
| 9 | Stray `websearch/SKILL.md` read at t=0 | Resource trimming (§3) | `--no-skills`, minimal context files | Smaller prompt, fewer lures |
| 10 | Runaway-response risk (09:22:25 truncation) | `max_tokens` 8k (§3) | request params | Bounded generations, faster loop detection |
| 11 | Byte-identical re-sampling (no escape) | Adaptive temperature (§3) | `before_provider_request` payload override | Stochastic escape after 2 identical responses |
| 12 | Task too open-ended for the model | Checkpointed decomposition (§5) | workflow / separate sessions | Bounded steps with gates; per-step verifiability |

---

## 8. Concrete pi setup for LFM2.5-2.6B (starting point)

```bash
# lean prompt, no lures
pi --model LiquidAI/LFM2.5-2.6B --thinking off --no-skills
#   (A/B: --thinking low; keep only if recovery metrics improve)
```

- **Provider params:** `temperature 0.1` (0.5 burst on loop detection), `repeat_penalty 1.1`, `max_tokens 8192`.
- **Install the `loop-guard` extension** (§1.1 + §1.2 + §1.3 + §2.3 in one; sketch below).
- **Compaction (pi-vcc):** SLM threshold ~40k; loop-aware summaries per §2.1.
- **Per-task budgets:** 60 tool calls / 2M tokens / 10 min → abort + status report.
- **Pre-flight:** `py_compile` on `.py` after write/edit.
- **System-prompt tail:** the 3-line recovery protocol from §3.
- **Workflow:** one checkpointed step per session, machine-checked gate between steps; supervisor watches the two tripwires (§5.3).

### `loop-guard` extension sketch (illustrative — verify hook signatures against `docs/extensions.md`)

```ts
// loop-guard.ts — install: pi install ./loop-guard  (or .pi/extensions/)
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";

const seen = new Map<string, number>();      // key(toolName,args,result) -> consecutive count
let identicalResponses = 0;                  // consecutive identical assistant outputs
let toolCallsThisTask = 0;
const MAX_CALLS = 60;

const key = (n: string, a: unknown, r: unknown) =>
  JSON.stringify([n, a]) + "::" + String(r).slice(0, 200);

pi.on("tool_execution_end", async (event) => {
  const k = key(event.toolName, event.args, event.result);
  seen.set(k, (seen.get(k) ?? 0) + (event.isError ? 1 : 0));
  seen.forEach((v, kk) => { if (kk !== k) seen.set(kk, Math.max(0, v - 1)); }); // only *consecutive* streaks count
  toolCallsThisTask++;
});

pi.on("tool_call", async (event, ctx) => {
  if (toolCallsThisTask > MAX_CALLS)
    return { block: true, terminate: true, reason: "Tool-call budget (60) exhausted. Stopping; write a status report." };

  if (!isToolCallEventType("bash", event)) return;
  const k = key(event.toolName, event.input, /*last result for same args*/ "");
  const streak = streakFor(event);   // consecutive same-(args,error-result) count
  if (streak >= 3) {
    const m = /File "([^"]+)", line (\d+)/.exec(lastErrorFor(event));
    if (m) {
      const excerpt = await readWindow(m[1], Number(m[2]), 15);   // fresh on-disk state
      if (streak >= 5)
        return { block: true, terminate: true,
          reason: `Same failing command ${streak} times. Stopping. Current code at the failing line:\n${excerpt}` };
      return { block: true,
        reason: `You have run the same failing command ${streak} times. Do not repeat it.\n` +
                `Current code at the failing line (${m[1]}:${m[2]}):\n${excerpt}\n` +
                `Fix the code, then re-run.` };
    }
  }
});

pi.on("context", async (event) => {
  // collapse consecutive identical (assistant toolCall -> toolResult) pairs
  return { messages: dedupeRepeatedPairs(event.messages, /*keep last*/ true) };
});

pi.on("before_provider_request", async (event) => {
  if (identicalResponses >= 2)      // attractor detected -> stochastic escape
    return { ...event.payload, temperature: 0.5 };
  return event.payload;
});

pi.on("turn_end", async (event, ctx) => {
  // track identical full assistant responses; at compaction-resume or every 5th failure,
  // queue a status-report requirement: "In one sentence: what is broken, and your next different action."
});
```

(Helper functions omitted; `dedupeRepeatedPairs` keeps one copy of each consecutive identical pair plus a `"(repeated N times)"` annotation; see §1.2.)

---

## 9. Expected impact (if P0+P1 are in place for the same task)

- The analyzed run stops or self-corrects by ~iteration 3–5 of the loop (≈09:20:15–09:20:30): **~138M of 144M tokens never spent; ~40 min of the 48.5 min saved.**
- If the model *does* self-correct (re-grounding puts `depth = 0` in front of it), the task completes in the 09:14–09:25 window.
- If it doesn't, the user gets a status report + a blocked, explained run instead of 40 minutes of silence — and the same run produces a clean training datum (loop + confabulation labels) for P5.

**Bottom line:** the model's capability gaps decide whether it *can* finish the task; the harness design above decides whether it *fails fast, visibly, and cheaply*. For a 2.6B SLM, invest in the harness first — the loop killer alone converts this session from a 48-minute 144M-token catastrophe into a 10-minute diagnostic.
