# REPORT — tzip skill reliability on LiquidAI/LFM2.5-2.6B (skills-usage-4)

Date: 2026-07-11. Workdir: `optim/skills-usage-4/`. All scripts run with `uv run --script <file>` (PEP-723 headers; deps auto-install).

## Task

User complaint (via real pi session logs): the synthetic tzip skill (injected by the
slm extension) works for one-shot invocations, but "sometimes, for some reason, it does not
work correctly" — e.g. `tzip full` gets confirmed as `tzip lite activated`, and the model's
reasoning mixes in the previous `example` skill's invocation state ("the skill invoked earlier
was 'example' with argument 'Hi'"). Optimize the extension's step-1 Q&A pair
(`SKILLSYS_EXPLAIN`/`SKILLSYS_THINKING` in `src/slm.ts`) with dspy + GEPA against live context.

## Environment / ground truth (reconstructed)

- Student + teacher: OpenRouter `LiquidAI/LFM2.5-2.6B`, `temperature 0.1` (from
  `src/openrouter-models.json` samplingParams). Keys from env (`OPENAI_API_KEY`,
  `REFLECTOR_API_KEY`/`OPENROUTER_API_KEY`). No local vLLM — everything is API-based.
- **Real session** = `~/pi-extensions/slm/tests/reports/LiquidAI-LFM2.5-2.6B/2026-07-11/session-2026-07-11T10-12-28-718Z.jsonl`
  (user's actual failing session, 11:23 timestamps; `example` skill is active there, tzip is
  not — only the injected system-prompt section). Parsed into `real-base-context.json`
  (real system prompt + 17 Q&A messages + the real tzip user message = block + "\n\nfull").
- **Token-verified failing context** (real = 5579 input tokens):
  `base(real system + 17 Q&A + frozen seed step-1 pair) + [user: bare tzip block] +
  [assistant: 'tzip lite activated'] + [user: block + "\n\nfull"]` → renders 5581.
  The first tzip pair is missing from the session file (pi session quirk) but was in the model
  context. **The user's bug = consecutive tzip invocations; the second call repeated the stale
  mode instead of switching** (real 11:23 reply: `tzip lite activated` again, with CoT that
  cited the `example` skill).
- Template `lfm25-chat-template.jinja` (repo root copy) drops reasoning blocks for **past**
  turns; the **current** thinking the user sees in the TUI is the model's live CoT.
  `enable_thinking`/`preserve_thinking` chat_template_kwargs do NOT matter for binding (both
  tested; past thinking is stripped either way). Student max_tokens 512 (pi: maxResponseBytes 16384).
- The `example` skill block in the real context is pi's own `example` skill (run_script.py,
  "Hi" argument) — identical in shape to the synthetic one.

## Scripts

| File | Purpose | Key output |
|---|---|---|
| `repro_real.py` | Shared: builds real context, student client (thinking on/off), token check (4994 base) | — |
| `test-modes.py` | 33-case mode matrix on real context | reply accuracy |
| `probe.py` | Targeted probes (CoT leak rate etc.) | — |
| `stress_real.py` | Full matrix: 11 activations + 20 transitions, x10 (310 samples) | `stress-real/*.log` |
| `repro_live.py` | The 6 live consecutive-invocation patterns (incl. the exact failing exchange), xN | — |
| `variant_stress.py <name> <pair.json> [N]` | Focused 20-case x10 stress for candidate pairs (variant-v1/v2.json) | `stress-real/variant-*.log` |
| `train.py` | Full dspy+GEPA pipeline (teacher-judged) | `gepa_runs/`, `train.log` |
| `repeat-full.py` | Earlier: repeated 'full' probe (kept for reference) | `repeat-full.log` |

## Findings

1. **Replies are 100% correct in reconstruction** — seed pair: 310/310 (matrix), 90/90
   (live patterns), 33/33 (test-modes), 46/46 (teacher-judged suite). The user's live failure
   is a rare tail event (<0.3% at n≈1000) — not reproducibly triggerable, but consistent with
   the CoT fragility below.
2. **CoT leak (visible to the user in the TUI)**: in ~8–15% of runs the current thinking
   quotes the previous `example` skill's invocation ("skill invoked earlier was 'example'…
   'Hi'"). Seed baseline on 310-sample matrix: 25/310 (8%). This is the measurable weakness
   and the likely precursor of the stale-mode tail failures.
3. **GEPA #2 (teacher-judged, 41-item valset, full-matrix minibatch, budget 600)** produced a
   "strong isolation" pair (valset 0.963 → 0.988, single-sample estimate), but full stress
   **regressed** it: 309/310 (one EMPTY reply) and leaks tripled to 78/310 (25%). The emphatic
   "ignore earlier skill blocks…" wording makes the 2.6B model think *about* the previous
   skill more. **Rejected.** Lesson: GEPA's single-sample valset overfits this stochastic
   defect; a 0.988 on 41 single samples is noise.
4. **Light variants** (seed pair + one short positive "fresh invocation / argument is the text
   after its own closing tag" sentence; `variant-v1.json` adds it, `variant-v2.json` replaces
   the weakest seed sentence): both 200/200 correct on the focused 20-case stress; leaks
   V1 28/200 (14%), V2 26/200 (13%) vs seed ≈27/200 (13.5% on the same cases, from
   `stress-seed.log` + `repro_live`). **No improvement beyond noise.**

## Current status (at time of writing)

- **Best candidate: `pair-optimized.json`** = V2 light-touch variant (seed pair with the
  weakest sentence replaced by a concrete fresh-invocation binding rule). 200/200 correct on
  the focused 20-case stress; 13% CoT leak — lowest of all candidates, within noise of seed.
  Provenance/evidence are in the file's `meta` block.
- **`src/slm.ts` is UNCHANGED** — it still ships the original seed pair. V2 has not been
  promoted (its edge over seed is within noise on 200 samples).
- ⚠️ Stale artifacts: `skill-example-tzip-LiquidAI-LFM2.5-2.6B.json` and
  `step1-pair-optimized.json` contain the **rejected** GEPA "strong isolation" pair (written
  by the last `train.py` run). The seed pair is preserved in `real-base-context.json`
  (`step1_user`/`step1_assistant`/`step1_reasoning`) and in `src/slm.ts`. Regenerate the JSON
  file from the chosen pair if needed (write block at the end of `train.py`, or git restore).
- All evidence logs are in `stress-real/` and `train.log` (see Artifacts).

## Next steps (in order)

1. **Validate the best candidate on the full matrix** before any promotion:
   `uv run --script variant_stress.py` only covers the 20-case subset; for a go/no-go run the
   310-sample matrix with V2 (copy `stress_real.py`, point its pair at `pair-optimized.json`,
   or set the pair in `repro_real.build_context`). Go criterion: ≥309/310 correct AND leaks
   ≤ 25/310 (seed baseline: 310/310, 25/310).
2. **If it passes**: promote V2 into `src/slm.ts` (`SKILLSYS_EXPLAIN`/`SKILLSYS_THINKING`,
   ~line 187) and regenerate `skill-example-tzip-*.json` (train.py's write step). If it fails:
   keep the seed pair and delete/stamp the stale JSON files.
3. **User-side acceptance** (cheapest, do it regardless): in a real pi session with the
   extension, run `/tzip` → `tzip full` → `/tzip full` → `/tzip lite` a few times; check the
   TUI thinking for `example`/`Hi` mentions and any stale-mode reply (the live failure was a
   <0.3% tail event, not reproducible at n≈1000).
4. **If the leak still bothers** (visible in TUI, ~13% of runs): make the GEPA metric
   **multi-sample** (2–3 student samples per item, score = min, or double the leak penalty) so
   the valset can't overfit one lucky sample, and restrict reflection to light-touch edits
   (the rejected pair proved that heavy "ignore earlier skills" rewrites backfire on 2.6B —
   they make the model think about the previous skill more).
5. **Structural alternative** (extension-side, untested): the leak traces to the frozen
   example-skill history in the context; test moving the step-1 pair to AFTER that history
   (nearest-position bias) by reordering in `repro_real.build_context` and re-running the
   stress scripts.
6. Optional: keep `enable_thinking` as-is — verified irrelevant for binding (template drops
   past thinking either way); only the model's live CoT is user-visible.

## Artifacts

- `pair-optimized.json` — **best candidate** (V2) with `meta` provenance block.
- `real-base-context.json` — parsed real session context + seed pair (source of truth).
- `real-system-prompt-full.txt` — the real pi system prompt from the session.
- `lfm25-chat-template.jinja` — LFM2.5 chat template (thinking-drop verified).
- `stress-real/stress-seed.log` (seed 310-sample), `stress-real/stress-optimized.log`
  (rejected pair), `stress-real/variant-v1.log`, `stress-real/variant-v2.log`, `train.log`,
  `gepa_runs/skills-usage-4/judge/run_log.txt`.
- `REASONING-LiquidAI-LFM2.5-2.6B.md` (repo root + here) — prior reasoning-mode notes.
- `src/slm.ts` (repo) — the extension; pair at `SKILLSYS_EXPLAIN` (~line 187).
