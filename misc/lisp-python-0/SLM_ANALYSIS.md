# SLM Analysis — LiquidAI/LFM2.5-2.6B in pi agent harness

**Session:** `01a004af-95ea-704b-aac8-880952bea296` (2026-08-15, 09:11:27 → 09:59:57 UTC, ~48.5 min)
**Sources:** `.pi/01a004af-….jsonl` (3172 entries), `.pi/pi-llm-debugging/01a004af-…/` (1574 requests / 1573 response metas), `demo/lisp-python/` (final artifacts)

---

## 1. Executive summary

The model was asked to implement a small Lisp interpreter in Python with demos and run them. It produced a **syntactically valid but fundamentally broken** `lisp-interpreter.py` (every s-expression parses to `None`) within the first ~3 minutes, then **never fixed it again for the remaining 45+ minutes**. Instead it entered a **deterministic frozen loop**: ~1,491 turns across the final 40 minutes (≈95% of all 1,574 LLM calls) in which it re-emitted a **byte-identical "thinking" block** and the **byte-identical bash command** `python3 lisp-interpreter.py "(+ 1 2)"`, which failed with the same `TypeError` each time. The loop was sustained across 21 context compactions and ran until the session was cut off mid-request (request #1574 has no response).

Headline numbers:

| Metric | Value |
|---|---|
| LLM requests | 1574 (1573 completed, all HTTP 200; 1 pending at cutoff) |
| Tool calls | 1552: **1527 bash (98.4%)**, 14 write, 8 read, 4 edit |
| Failed bash runs | 1513 (`Command exited with code 1`) |
| Identical retry of `(+ 1 2)` | **1498** (1,499 total runs of the command) |
| Byte-identical thinking blocks | **1445 of 1573** (one text, hash `d8179af5`) |
| Compactions (pi-vcc) | 21, every ~2 min, context pinned at 123,944 tokens |
| Tokens processed | ~144M total (1.69M fresh input, 1.08M output, ~141M cache-read) |
| Non-bash tool calls (read/write/edit) after 09:18:29 | **0** (last file change 09:14:20) |
| User-facing text produced | **0** (no text block in any of the 1573 responses) |
| Task outcome | **Not completed.** Interpreter cannot evaluate any expression; no demo was ever run |

The core finding: this is not "the model struggled and gave up." It is a **complete collapse into a repeated attractor state** — identical belief text, identical action, zero belief update from 1,497 identical failure observations, plus a **confabulated memory of a successful run that never happened**.

---

## 2. Setup

- **Harness:** pi (v0.84.2), interactive session, provider `llamacpp` (local llama.cpp server behind Caddy), model `LiquidAI/LFM2.5-2.6B`, thinking level `high` → `reasoning_effort: "high"`.
- **Sampling (from `*-req.json`):** `temperature 0.1`, `top_k 50`, `repeat_penalty 1.1`, `max_tokens 49152`.
- **Tools:** 5 (`read`, `bash`, `edit`, `write`, `vcc_recall`).
- **Context:** effective ceiling ~124k tokens (compaction fired at `tokensBefore` 124,060 → then stable 123,944).
- **Compaction:** `pi-vcc` extension (custom `compaction` + `custom_message pi-vcc:resume-after-compaction` entries).
- **User task (verbatim):** implement a simple Lisp interpreter in Python with demos in `demo/lisp-python`; verify with `python`; no unit tests; support arithmetic/logical ops, conditionals, loops, variable & function definitions/calls; run all Lisp demos through the interpreter; fix all issues.

---

## 3. Timeline (phases)

| Time (UTC) | Phase | What happened |
|---|---|---|
| 09:11:29 | Orient | `ls demo/lisp-python/`; **reads unrelated `websearch/SKILL.md`** (stray action) |
| 09:11:32–09:11:51 | Create | `mkdir`; writes `lisp-interpreter.py` **v1** (uses Python `ast` module to parse Lisp; `lambda cond, then, else:` — `else` is a Python keyword → SyntaxError); writes `demo1.lisp` |
| 09:11:56–09:12:22 | Debug v1 | First run → SyntaxError. 2 edit attempts (one failed: "Could not find the exact text"). Full rewrite. `python3 -c "from lisp-interpreter import run"` → SyntaxError (**hyphenated module name**). Adds missing `if __name__` block (earlier runs printed nothing) |
| 09:12:23–09:13:34 | Flounder | `(+ 1 2 3)` runs → SyntaxError from `ast.parse`. Environment-superstition checks: `import ast` sanity, Python version (3.14.7), **`__pycache__` purge ×2**. Two more full rewrites |
| 09:14:05–09:14:20 | Final file | Writes `test-demo1.lisp` (reduced demo). Writes the **final** `lisp-interpreter.py` (broken `parse_s_expr` with dead-loop depth counter, lines 130–132). Edit adds `import ast`. `(+ 1 2)` now fails with the **final error**: `TypeError: Unsupported AST node: <class 'NoneType'>` |
| 09:14:58 | Root cause on the table | Writes `debug.py`, runs it — output **literally shows the parser bug**: `Testing: (+ 1 2)` → `Parsing list: inner=''` → `Inner is empty, returning None`. Model proceeds without fixing |
| 09:15:50–09:18:15 | Test-file churn | Reads interpreter ×3. Writes `test_simple.py` (hyphen import → SyntaxError line 4). `__pycache__` purge ×3. Writes **`test_import.py` 4 times** — first 3 with the same hyphen-import SyntaxError; final 110-line version re-implements the interpreter inline and contains a `try:` with no `except` (SyntaxError) — the exact snippet its own thinking quotes and half-analyzes |
| 09:18:29–09:20:06 | Loop wave 1 | Returns to `(+ 1 2)`. 46 iterations with thinking variant `d26d544d` |
| 09:20:08–09:59:55 | **Frozen loop (dominant)** | 1,445 iterations of byte-identical thinking `d8179af5` + identical `(+ 1 2)` command, ~1.7 s apart. 21 compactions interleaved; each resume turn produces 1–363 tokens of thinking and **no action**; loop resumes. One response (09:22:25) was truncated mid tool-call; pi correctly refused to execute it ("response hit the output token limit… Re-issue") and the loop simply continued |
| 09:59:56–57 | Cutoff | Compaction #21, then request #1574 issued; **no response recorded** (session ended) |

---

## 4. The frozen loop, in detail

**The repeated unit** (identical 1,445×): a 2,737-character thinking block ending in *"…let me just try running the original lisp-interpreter.py one more time to make sure it still works:"* followed by the tool call `cd …/demo/lisp-python && python3 lisp-interpreter.py "(+ 1 2)"`, then the identical 10-line traceback (`TypeError: Unsupported AST node: <class 'NoneType'>` at lines 182/187/192), repeat. ~669 output tokens per iteration, ~1.7–1.8 s per iteration (≈33–36/min), **~40 minutes** (09:18:46 → 09:59:55, including 46 iterations of a near-identical thinking variant `d26d544d` before the dominant 1,445× text).

Three properties make this severe:

1. **No belief update.** The identical error was present in context every single iteration (and in the compaction summaries). From 09:18:29 to the end of the session the model made **zero read/write/edit calls of any kind** — every one of those ~1,491 turns was a bash retry. A competent model would read lines 130–132 or at minimum vary the probe.
2. **Confabulated success.** The dominant thinking block contains:
   > ```
   > Usage: python lisp-interpreter.py "(+ 1 2)"
   > Result of (+ 1 2): 3
   > ```
   > "That worked! So why is it failing now?"
   
   **No tool result in the entire session contains "Result of" or `3`** — that run never happened. The model invented a memory of success and built a narrative around it ("something changed between runs… maybe… caching issue with .pyc files… something wrong with how Python is reading/writing files") — i.e., it externalized the cause to the environment instead of its own code. (The `Usage:` line is real: it's the no-argument message from 09:12:57 — the model spliced it into a fabricated transcript.)
3. **No sampling escape.** With `temperature 0.1`/`top_k 50` the sequence still reproduced byte-identically 1,445× — the model is in a low-entropy attractor where the top token at each step is always the same. Two near-identical attractor texts existed (`d26d544d` 46×, then `d8179af5` 1,445×); both end with the same plan and the same action.

The loop's context was *engineered to be uninformative*: by the end, the last request (1574, 144 messages) consisted of system prompt + a compaction summary + **71 consecutive copies of the same (thinking, tool call, traceback) triple**. All 71 assistant `reasoning_content` fields in that request are the same hash. The summary itself is degenerate — its `[Outstanding Context]` is just `[ERROR] [bash] Traceback (most recent call last):` (first line only, no diagnostic content), and its `[Earlier Turns]` is the same command listed 30+ times with the instruction *"Do not recap… Continue directly where you left off."* "Where you left off" was: run the same command.

---

## 5. Code-level issues in `demo/lisp-python/` (what the model shipped)

### `lisp-interpreter.py` (194 lines — the final artifact; still fails: `python3 lisp-interpreter.py "(+ 1 2)"` → `TypeError: Unsupported AST node: <class 'NoneType'>`)

The file is a **stack of three mutually incompatible implementations** that no version of the file ever integrated:

1. **Class `LispInterpreter`** (eval/eval_list/eval_operator/while_loop/if_statement) — a reasonable *design* but never called by anything in the `__main__` path; never instantiated.
2. **`run_lisp()`** — parses with **Python's `ast` module** (`ast.parse(code, mode='eval')`) and `ast.walk`s the result. Conceptually wrong twice over: (a) `ast.parse` parses *Python* syntax — `"(+ 1 2 3)"` is a Python SyntaxError, so this path could never parse Lisp; (b) it then calls `ast.walk` on whatever `parse_s_expr` returns (a `str`/`float`/`None`), which is not an AST, and accesses `node.value` on `ast.Name`/`ast.BinOp` nodes that have no such attribute.
3. **`parse_s_expr()` + `this_eval_node()` + `run()`** — the actual entry point, and the fatal bug:

   ```python
   depth = 0
   i = 1
   while i < len(code) and depth > 0:   # depth starts at 0 → loop body never runs
       ...
   inner = code[1:i-1]                  # code[1:0] == "" 
   if not inner:
       return None                      # ← every s-expression returns None
   ```

   - **Dead loop condition** (`depth` initialized 0, loop requires `> 0`): the parenthesis scanner never executes, so **any** input starting with `(` returns `None`. (Fix: start at `depth = 1`.)
   - **No tokenization at all**: even with the scanner fixed, the recursive scheme `return [parse_s_expr(inner)]` treats the entire inner content `"+ 1 2"` as one atom — a multi-element list could never be produced. The design is incoherent, not just buggy.
   - `this_eval_node` references `env` which is **not defined in its scope** (NameError waiting to happen), and it expects `ast.*` nodes that this parser can never produce — it is dead code for this pipeline.
   - Operator-table bugs in the (unused) class: `'/': lambda x, y: x / y if y != 0 else ZeroDivisionError("…")` **returns an exception instance instead of raising**; all lambdas are fixed 2-ary while comments claim "variadic"; `and`/`or` are not n-ary; `SymbolTable.set()` only writes `local_env`, silently dropping globals.

**Root cause of the whole session in one line:** the model confused the *concept* of an AST with **Python's `ast` module**, kept `ast.parse`-based parsing through all 7 rewrites, and when it finally wrote a hand-rolled parser (09:14:17) it shipped the depth-counter bug — which `debug.py` (09:14:58) demonstrably proved — and then looped instead of fixing it.

### `demo1.lisp`
Aspirational demo: uses bare variable assignment (`a 10`), inline `;` comments, `(c (* b 2))`, and recursive `(define (factorial n) …)` — none of which the interpreter was ever designed to support. **Never executed** (no invocation of the interpreter over any `.lisp` file exists in the session).

### `test-demo1.lisp`
A trimmed demo the model wrote at 09:14:05. Also never executed.

### `test_simple.py`, `test_import.py`, `debug.py`
- All three violate the explicit **"do not write unit-tests"** instruction.
- `test_simple.py` and the first 3 versions of `test_import.py` contain `from lisp-interpreter import run` — a **SyntaxError** (hyphen in module name), written **4 separate times** across 2 files. Each failure produced "SyntaxError … line 4"; the model never pattern-matched the error to the import line.
- Final `test_import.py` (110 lines) re-implements `SymbolTable` + `LispInterpreter` inline (copy-paste of the broken class, including the return-not-raise `/` lambda) and contains:
  ```python
  while j < len(content) and content[j:i] != '':
      try:
          args.append(self.evaluate(content[j:]))
      j += 1        # ← SyntaxError: 'try' with no 'except'/'finally'
  ```
  This exact snippet is quoted — and partially *correctly analyzed* ("the condition is always True… the logic is flawed") — in the loop thinking, yet was never fixed.

---

## 6. Catalog of model issues (LiquidAI/LFM2.5-2.6B)

### A. Agentic / reasoning failures (primary)
1. **No error-driven correction.** 1,513 failed executions; the traceback naming the exact file+line (182 → 187 → 192) was in context every turn. The model made zero edits to the failing file for 45.5 minutes. It could not map "TypeError at line 182 in this_eval_node" to "fix parse_s_expr, lines 130–132".
2. **Frozen deterministic loop.** ~1,491 identical (thinking, action) pairs across 40 minutes (interleaved with compaction-resume turns); ≈95% of the session. No behavioral variation despite sampling parameters that allow it. This is a hard failure mode for an agent: the loop is trivially detectable (identical tool-call hash) and was never broken.
3. **Confabulation.** Fabricated a "successful run" transcript (`Result of (+ 1 2): 3`) and a causal narrative (corrupted file, .pyc cache, wrong interpreter) unsupported by any observation, in place of the trivially supported fact that its own parser returns `None`.
4. **Ignored direct evidence.** `debug.py` output at 09:14:58 showed, step by step, that `(+ 1 2)` parses to `None`. The model acknowledged the surrounding context ("same issue (syntax error on line 81)") yet neither fixed the parser nor the try/except.
5. **Diagnosis by superstition.** Repeated environment checks (import ast, Python version, `__pycache__` purge ×3) — hypotheses that the evidence did not support — instead of reading the failing code.
6. **Escalation instead of minimization.** 7 full-file rewrites in 3 minutes, each preserving the core flawed approach (`ast`-based parsing), instead of a minimal isolated repro. Rewrites were used as a retry mechanism, not as refinement.
7. **Failed compaction-resume turns.** After 19 of 21 compactions the resume turn produced only 1–363 tokens of thinking and no action (the turn died); the degenerate loop only resumed on the following turn. The model cannot execute on the "continue where you left off" resume prompt.
8. **Zero communication.** No text block in any of 1,573 responses — no status, no explanation, no final answer, even at task end.
9. **Stray action at t=0.** Turn 1 read `websearch/SKILL.md` — an unrelated file, presumably latched onto from the skill catalog in the system prompt.
10. **Instruction non-compliance.** Wrote test scripts despite "do not write unit-tests"; never ran the demos as required.

### B. Code-generation quality failures (secondary, but they triggered the loop)
11. **Conceptual confusion: Python `ast` ≠ "an AST".** Using `ast.parse` as the Lisp front-end persisted across every rewrite — a category error a 2.6B model should not need to make on a task this small.
12. **Off-by-one in a 15-line parser** (dead `while depth > 0` loop) plus an incoherent recursion (no tokenization) — the parser could not parse *any* of its own demo inputs.
13. **Python fluency gaps, repeated**: `lambda …, else:` (keyword parameter); `from lisp-interpreter import …` (hyphen module) ×4; `try:` without `except`. Each was flagged by the interpreter as a SyntaxError; none was pattern-recognized.
14. **Semantic bugs in the (unused) evaluator**: return-not-raise for division by zero, fixed arity vs claimed variadic, `SymbolTable.set` dropping globals.
15. **Spec mismatch in demos**: `demo1.lisp` demands features (assignment syntax, comments, recursion) the interpreter never supported; demos were written to be run but never run.

---

## 7. Harness/systemic observations (context for the above)

- **Compaction reinforced the loop.** The pi-vcc summary of a looping session is *a summary of the loop* — it lists the same failing command 30+ times as "work in progress", truncates the error to its first line (`Traceback (most recent call last):`), and instructs "continue directly where you left off." Compaction, meant to enable long tasks, here made the degenerate state self-perpetuating and kept context permanently under the 124k ceiling, so the loop could never "crash" into a different state.
- **No repetition guardrail.** pi (by design) has no loop detection. A guard as simple as "same tool-call hash N times in a row → stop/escalate" would have broken this session at iteration ~5 and saved ≈138M tokens (the loop window alone is 137.7M of the 144.1M total) and ~40 minutes.
- **Truncated responses are handled safely.** The single mid-tool-call truncation (09:22:25) was refused with a clear re-issue instruction — the harness behaved correctly; the model just ignored the premise.
- **`reasoning_effort: high` produced shallow reasoning.** The "thinking" blocks are verbose, but they are stale (quoting errors from 5+ minutes earlier), confabulatory, and never re-derived from the latest observation. Reasoning length did not buy reasoning quality.
- **All 1,573 completed responses were HTTP 200** — no server/quantization/serving artifacts; the failure is purely model behavior.

---

## 8. Bottom line

For an agent-harness evaluation, LFM2.5-2.6B's failure here is two-layered:

1. **Capability:** it cannot (a) keep a simple parser correct (off-by-one + wrong parse strategy), (b) recognize repeated elementary Python syntax errors, or (c) connect a traceback's line numbers to the code that produced them. It cannot implement the requested Lisp interpreter at the "simple" spec level (loops, conditionals, `define`) in the first place.
2. **Behavior (worse):** when it hit its own bug it did not retry-with-variation, did not read the failing code, and did not stop. It locked into a byte-reproducible loop of ~1,500 identical failures, *confabulating a success* to keep the loop coherent, for 40 minutes, through 21 compactions, with zero edits, zero reads, and zero user communication, until the session was killed.

**Verdict:** unusable as-is for autonomous agentic coding tasks. Even a trivially successful first attempt would have been followed by an uncorrectable loop. If this model is to be evaluated further, add: (1) a repeated-tool-call circuit breaker, (2) compaction summaries that carry the *full* last error (not its first line), and (3) a task-timeout — and consider that the confabulated-success in the thinking stream is a reliable early marker that the model has left evidence-based reasoning.
