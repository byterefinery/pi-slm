# skills-usage-5 — tzip few-shot optimization report

## Task
Optimize the ONE mutable synthetic message pair in
`skill-example-LiquidAI-LFM2.5-2.6B.json` — the "How does the skill system work?"
user question + assistant `content` + `reasoning_content` — so that the student
model `LiquidAI/LFM2.5-2.6B` (thinking ON) reliably handles `tzip` invocations
appended at the END of this exact conversation:

- default empty message, `on`, `lite`, `full`, `ultra`, `off`
- transitions between modes (6 transition cases)

Expected replies (from the tzip Usage section):
`empty`/`on`/`lite` → `tzip lite activated`; `full` → `tzip full activated`;
`ultra` → `tzip ultra activated`; `off` → `tzip deactivated`.
Trailing period ignored in matching (but should not be there).

Frozen (byte-identical, asserted at write-back): system message, skills/tools
listing Q/As, all example-skill usage turns, tools array, sampling params,
`chat_template_kwargs`, everything else in the file.

Verification: teacher `Qwen/Qwen3.8-27B` thinking OFF (leak/style judge,
teacher-as-responder baseline). Optimizer: GEPA only
(`gepa.optimize_anything`, single-task reflective mode, custom strict-JSON
proposer on the teacher). All model calls serial (shared server).

## Result

| setup (student thinking ON) | acc | CoT leak |
|---|---|---|
| teacher baseline (seed pair, deployment request) | 100% | 0% |
| student, SEED pair, deployment request | 100% (4 runs) | 0% |
| **student, OPTIMIZED pair, deployment request** | **100%** | **0%** |
| — deterministic eval | 100% | 0% |
| — 3 fresh runs (min-of-3) | 100% | 0% |
| — regime-F fresh run | 100% | 0% |
| — streaming (stream=true) fresh run | 100% | 0% |
| student, real standard-`pi` CLI, temp cwd (run 1) | 16.7% | 75.0% |
| student, real standard-`pi` CLI, temp cwd (re-run) | 16.7% | 66.7% |
| teacher, real standard-`pi` CLI, temp cwd | 58.3% | 8.3% |

"Deployment request" = the captured JSON request format:
`chat_template_kwargs {enable_thinking: true, preserve_thinking: true}` +
history `reasoning_content` present. The optimized pair: 84/84 across 7
independent 12-case runs, 0 leaks.

The seed pair was already behaviorally perfect in the deployment format; the
only defect was the student-voice style requirement (seed reasoning = 4
sentences, conditional, no committed ending → style gate FAIL). GEPA converged
in one mutation (4 metric calls) to a 2-sentence committed-ending reasoning
while preserving 100%/0. Style gate: 3/3 fresh teacher verdicts OK.

## Optimized pair (written back)
- user: "How does the skill system work? If a skill block appears in my latest
  message, what exactly should I do?"
- content: "I treat every skill block as a fresh, independent invocation. I
  identify the mode by reading the bare argument in the text immediately
  following the current block's closing tag."
- reasoning_content: "I read only the argument text after the current block's
  closing tag, ignoring any previous context. I will reply with the exact
  short confirmation line required by the Usage section."

## Standard-pi discrepancy (investigation)
The real standard-`pi` binary (0.84.2) reproduces ~16.7% acc / ~70% leak for
the student, while every raw-client reconstruction succeeds:

- Proxy-captured the exact pi request: `reasoning_effort: "high"` (no
  `chat_template_kwargs`), ALL history `reasoning_content` stripped (pi drops
  session `thinking` blocks without a `thinkingSignature`), pi-style system
  prompt, streaming.
- Raw replays: captured pi request verbatim → OK; deployment regime → OK;
  pi regime → OK; streaming (pi-style and deployment) → OK (full 12-case suite
  100%/0).
- So request params are not the cause; the pi process itself has a systematic
  effect not yet isolated. Leading hypothesis: global pi state in the real
  HOME (extensions/settings) alters the request — capture runs used a clean
  temp HOME and the proxy capture's request looked clean, but no successful
  real-HOME capture was completed before the server outage.
- Note: the user's deployment pi is the pi-slm fork that generated the
  captured JSON (it sends `chat_template_kwargs` + preserved history CoT) —
  i.e. the deployment format, in which the student is 100%/0.

## Artifacts
- `skill-example-LiquidAI-LFM2.5-2.6B.json` — optimized pair spliced in
  (only messages[5], messages[6] changed; asserted).
- `train.log` — full progress (evals, GEPA iterations, all runs).
- `runs/` — scripts, GEPA state (`gepa-run/`), per-eval traces, judge
  verdicts, pi session files, proxy captures, result JSONs.
- `train.py` — orchestrator (stages: baseline/seed/optimize/final/pi).
