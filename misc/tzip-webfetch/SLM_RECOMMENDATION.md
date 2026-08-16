# Recommendation: improving `LiquidAI/LFM2.5-2.6B` in pi

Companion to `SLM_ANALYSIS.md`. Every recommendation below maps to an observed failure mode (FM-1…FM-7) from that analysis, and the primary vehicle is a **pi extension** — matching this repo’s stated purpose (“Reliability extension for small language models”).

## 0. Design principles

1. **Compensate priors, don’t babysit.** The SLM’s gaps are implicit-shell-semantics, multi-hop path resolution, error diagnosis, and instruction priority. A harness can supply those deterministically (cheap, locally) instead of hoping the model reasons them out (expensive, unreliable).
2. **Guard before, inform after.** Prefer fixing the command *before* execution (`tool_call`) over explaining it *after* failure (`tool_result`); use both.
3. **Stay silent when the model is fine.** All guards activate only for models on an SLM allowlist and only for the specific error classes observed; a capable model (e.g. Qwen3.8-27B) must see zero overhead and zero interference.
4. **Small context footprint.** The SLM already inflates context ~3×. Anything injected must be a few lines or less.

## 1. The extension: `slm-reliability`

Location: `.pi/extensions/slm-reliability.ts` (project) or `~/.pi/agent/extensions/slm-reliability.ts` (global). All hooks below are existing pi extension events.

### 1.0 Gating (prerequisite for everything)
- Maintain an **SLM allowlist** (model-id patterns, e.g. `LiquidAI/LFM*`, or a size rule “≤ ~4B params” via model registry metadata). Read `ctx.model` at `session_start` and on `model_select`; all guards no-op when the active model is not on the list.
- Expose `/slm-status` (registered command) showing which guards are active and the current failure counter.

### 1.1 R1 — Pre-execution command guard  *(fixes FM-1, FM-2, FM-4)*
Hook: `tool_call` (bash only; `event.input` is mutable — pi applies the mutation to the real execution and documents that later handlers see it).

| Check on `event.input.command` | Action |
|---|---|
| Bare script invocation `name.py`/`name.sh` that is **not in PATH** | If the file exists in cwd or in a loaded skill’s `scripts/` dir (skill locations come from `event.systemPromptOptions.skills` at `before_agent_start`), **rewrite in place**: `./name.py …` or, when the shebang is `env -S uv run --script` (PEP 723), `uv run --script <abs>/name.py …`. Log the rewrite via `ctx.ui.setStatus` (visible in TUI, not in the model’s context). This is exactly the transformation that turned LFM’s 8 attempts into Qwen’s 1. |
| Root-scoped scan: `find / …`, `grep -r / …`, `locate` | Rewrite to scope under `ctx.cwd` (and known project/skill roots); if the command already carries a narrower path, leave it. If it truly needs system scope, block with `{ block: true, reason: "Use find <cwd> … ; full-disk scans are not allowed in SLM mode" }` so the error teaches scoping. |
| No `timeout` field set | Inject a default (e.g. `timeout: 120`) — Qwen self-set one; LFM’s `find /` ran 19.7 s unbounded. |
| Command identical to a previously **failed** command in this session (same normalized string, same error signature) | Block with reason: “Same command already failed with: \<first 2 lines\>. Change approach (e.g. `./` prefix, absolute path, `uv run --script`).” This kills the exact LFM step-8→step-11 repeat. |

Implementation notes:
- Normalize before comparing (collapse whitespace, expand `~`, resolve `cd X && ` prefixes to the effective cwd).
- Keep the rewriter table-driven (`[{pattern, fix, hint}]`) so new SLM-specific fixes are one-liners.
- Every rewrite/block should append a **one-line note** to the returned tool result (see R2) so the model learns the rule in-context; a 2.6B model generalizes from one worked example far better than from zero.

### 1.2 R2 — Error enrichment  *(fixes FM-2, FM-3)*
Hook: `tool_result` (bash only, `isError === true`). Append at most ~3 lines of **diagnostic hints** to the error output, keyed on the error class:

| Error signature | Hint appended |
|---|---|
| `command not found` + a same-named file exists in cwd/skill dirs | `hint: not in PATH. File exists at <abs paths>. Run as ./name or with absolute path (PEP 723: uv run --script <abs>).` |
| `command not found` (uv context: `uv run name.py`) | `hint: uv needs the script path: uv run --script <path>/name.py` |
| `Permission denied` | `hint: chmod +x <file>, or invoke via interpreter (python/uv run --script)` |
| `No such file or directory` from uv | `hint: script path relative to cwd; use an absolute path` |
| timeout | `hint: command hit timeout <n>s; narrow the scope or raise the timeout explicitly` |

This hands the 2.6B model the correct hypothesis it never formed on its own (FM-3: it blamed the shebang `-S` flag for a PATH error). The hint *is* the learning; it also makes FM-7-style misattribution much harder, because the correction is in the transcript.

### 1.3 R3 — Skill materialization  *(fixes FM-1 at the source; biggest leverage)*
Hook: `context` (fires before each LLM call, deep-copied messages are safely modifiable) — or once at `before_agent_start` per skill load.

For every user message containing a `<skill name=… location=…>` block, when the active model is an SLM:
1. Resolve the skill directory to an absolute path (it’s in the header) and verify `scripts/` exists.
2. Rewrite the **Usage** examples from bare `webfetch.py <url>` to the actually-runnable command: `uv run --script <abs>/scripts/webfetch.py <url>` (shebang-aware; plain `python`/`./` otherwise).
3. Append one line: `To run in this project: <abs runnable command> <args>`.

Rationale: the SLM copies what it sees (FM-1 is a *copy* failure, not a knowledge failure). Giving it the runnable form to copy makes the first attempt the right one. This is what Qwen computed for itself in one thinking step; materialization makes it free. Do this only for SLMs — larger models do it themselves and the rewrite just burns tokens.

(Complementary, non-extension fix for the skill itself: see §2.2.)

### 1.4 R4 — Re-plan circuit breaker  *(fixes FM-3, bounds FM-4)*
Hook: `turn_end` (has `event.message` + `event.toolResults`), state kept in the extension.

- Count consecutive tool-call failures per session.
- At **2 consecutive failures**: inject (via the next `context` hook) a persistent 4-line note: the failed commands + errors so far, and the standard candidate strategies for script invocation (`./` prefix, absolute path, `uv run --script`, check `ls <skill-dir>` before any filesystem search).
- At **4**: additionally suggest escalation (see R7) and, in TUI, `ctx.ui.notify` so the human sees the loop.
- Reset the counter on any successful tool result.

### 1.5 R5 — Verify-before-claim  *(fixes FM-7 — highest-stakes fix)*
Hook: `message_end` (assistant messages) / `turn_end`.

If the final assistant text of a turn **asserts success** (“successfully”, “fetched”, “now working”, “completed”, ✅) and the turn’s tool results contain no matching success (or the last relevant result is an error), append a correction to the *displayed and stored* message: “Note: the last tool result was an error; success not verified.” In `-p`/json/rpc modes (no TUI) this still lands in the session and any downstream consumer.

This directly attacks the confabulated narrative (“running it from its directory worked” when it didn’t), which is the failure that matters most for unattended use.

### 1.6 R6 — Instruction normalization  *(fixes FM-5)*
Hook: `input` (raw user text, before expansion; can `transform`).

Detect known **trailing mode/command words** on skill-related input (e.g. a lone `full`/`lite`/`ultra`/`off` after a skill block) and append an explicit structured line: `SET TZIP MODE: full (exact mode command; reply with the mode name only)`. Trailing words were ignored twice by the SLM; a capitalized, labeled instruction is a much higher-priority cue for a small model. Same mechanism works for future skills with trailing directives.

### 1.7 R7 — Conciseness enforcement  *(fixes FM-6)*
Hook: `before_agent_start` — append a compact SLM discipline block to the system prompt **only for SLMs** (~120 words):

> When a skill or mode is activated, reply with the mode name on one line; no menus, no options lists, no “what would you like to do next”. Answer only what was asked. Before claiming a task succeeded, check that the tool result shows it. If a script is not in PATH, use `./name` or an absolute path; list the skill directory before searching the filesystem.

(Optionally: cap/trim repeated self-introductions of mode state across turns.)

### 1.8 R8 — Optional backstop: model escalation
After the R4 counter hits the escalation threshold and the human opts in (or a project setting `slm.escalateTo = <model>` is set): switch the session to the larger model (`model_select`-equivalent via session API / `ctx`), notify via `ctx.ui.notify`, and let it finish the turn with the full error history in context. Keep this off by default — it changes cost/latency characteristics and the user should choose it.

## 2. Non-extension measures (immediately applicable, no code)

### 2.1 Keep the current sampling — it works
The measured config (`temperature 0.1`, `top_k 50`, `repeat_penalty 1.1`, `reasoning_effort: high`) produced zero malformed tool calls and zero repetition degeneration over 13 calls. Don’t raise temperature for bash-heavy SLM work; the low-temp + repeat-penalty point is the stable one. `thinkingLevel: high` is fine; note the llama.cpp provider reports `reasoning` usage as 0, so budget decisions should use wall time, not reasoning tokens.

### 2.2 Fix the skill, not just the model
The webfetch skill’s Usage block teaches the wrong lesson to a literal copier: it shows `webfetch.py <url>` as if the script were on PATH. Two cheap fixes (either is sufficient; both are cheap):
- Add one canonical line to the skill: `Run: uv run --script <skill-dir>/scripts/webfetch.py <url>` (with the real relative-from-cwd path), or
- Install a PATH shim (`uvx`-style wrapper / `pi install`-ed bin) so the bare command actually resolves.
This helps *every* model and makes R3 a no-op safety net rather than the load-bearer.

### 2.3 Until the extension lands: `--append-system-prompt`
The R7 block above can be passed on the CLI today: `pi --model LiquidAI/LFM2.5-2.6B --thinking high --append-system-prompt "<SLM discipline block>"`. It covers FM-6/FM-7-ish behavior with zero code; R1–R6 still need the extension.

### 2.4 Operating rules (process, not code)
- Supervise SLM sessions for any task where a wrong script copy or a false success claim has consequences (the `find /` 10-copies finding is the cautionary example).
- Prefer tasks with a verifiable success check (build passes, tests green, file exists) over “agent says done”.

## 3. Priority & expected impact

| Priority | Item | Fixes | Effort |
|---|---|---|---|
| **P0** | R1 command guard (incl. repeat-failure block) + R2 error hints | FM-1, FM-2, FM-3, FM-4 | ~150 lines |
| **P0** | R7 system-prompt block (or §2.3 CLI version) | FM-6 | trivial |
| **P1** | R3 skill materialization (+ §2.2 skill fix) | FM-1 (root cause) | ~100 lines |
| **P1** | R4 re-plan breaker | FM-3, bounds FM-4 | ~60 lines |
| **P2** | R5 verify-before-claim | FM-7 | ~80 lines |
| **P2** | R6 input normalization | FM-5 | ~40 lines |
| **P3** | R8 model escalation backstop | all | ~80 lines |

**Measured baseline this should beat (LFM2.5-2.6B, tzip-webfetch):** 8 tool calls, 4 failures, first success on the 8th tool call, 47.7 s to working fetch, 51.2 s to summary, ~5,053 output tokens, 19.7 s `find /`, tzip mode ignored, success mis-attributed.

**Target after P0+P1 (Qwen-parity trajectory):** first bash attempt is the materialized `uv run --script <abs>` command (R3) or the guarded `./`/abs rewrite (R1) → fetch succeeds on attempt 1 (≈7 s); tzip-ack ≈1 line in the requested mode (R6/R7); no `find /`; no repeated failing command; final summary consistent with tool results (R5). Expected: 3–4 tool calls, 0 failures, ~10–15 s to summary, ~1–1.5 k output tokens — i.e. within ~2× of the 27B baseline on wall time and within ~2× on tokens, while keeping the SLM’s local, free, low-latency serving.

## 4. Acceptance tests (replay the recorded scenario)

Re-run the exact script from the analysis (tzip+`full` → webfetch skill → “Fetch a URL with webfetch”) with `LiquidAI/LFM2.5-2.6B` and the extension enabled. Pass criteria:

1. First `bash` call containing `webfetch.py` exits 0 (no 127, no uv ENOENT).
2. Zero `find /` / root-scoped scans in the session.
3. No command string repeated after having failed with the same error.
4. The tzip ack names mode `full` (not `lite`) and is ≤ 2 lines.
5. Final message contains no success assertion not backed by a successful tool result.
6. With the extension disabled on a non-allowlisted model (Qwen3.8-27B), the session byte-diff shows **no** behavioral change (gating works).
