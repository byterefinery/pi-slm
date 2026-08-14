# Analysis of SLM behaviour

---

Here is my full analysis of the session log. I found 10 distinct issues:

────────────────────────────────────────────────────────────────────────────────

Issue 1: Massive Repetitive Thinking Blocks (30K–70K chars)

The most striking problem. Thinking blocks contain paragraphs repeated 10–24 times each:

┌──────────┬───────────────┬────────────────────────────────────┐
│ Message  │ Thinking Size │ Worst Duplicate                    │
├──────────┼───────────────┼────────────────────────────────────┤
│ 0ffa19df │ 70,018 chars  │ "Actually, I think..." × 24        │
├──────────┼───────────────┼────────────────────────────────────┤
│ 676ce141 │ 69,809 chars  │ "But the current file..." × 23     │
├──────────┼───────────────┼────────────────────────────────────┤
│ 7d441924 │ 69,559 chars  │ "Given this inconsistency..." × 19 │
├──────────┼───────────────┼────────────────────────────────────┤
│ 850faa17 │ 65,994 chars  │ "This should work! But..." × 23    │
├──────────┼───────────────┼────────────────────────────────────┤
│ fcc844fb │ 65,994 chars  │ identical to 850faa17 × 23         │
└──────────┴───────────────┴────────────────────────────────────┘

The model gets stuck in self-doubt loops: "Let me" appears 40–93 times per thinking block. It re-reads code, re-analyzes, doubts itself, and repeats the same reasoning cycles.

Issue 2: Orphaned Thinking-Only Messages

Two assistant messages contain only thinking, no text and no tool calls, and both end mid-sentence:

- [77] 6c79d06b — 30,202 chars, ends: "Actually, I just realized - maybe the problem is simpler than I"
- [135] 58d661bd — 44,050 chars, ends: "1. apply_arithmetic IS present ("

These are interrupted generation artifacts — compaction fired while the model was still streaming its thinking.

Issue 3: Compaction Mid-Turn ("Split Turn")

Both compactions happened during active assistant turns:

```
  Compaction 1: summary says "**Turn Context (split turn):**"
  Compaction 2: summary says "**Turn Context (split turn):**"
```

The compaction's parentId points to the orphaned thinking-only message. The firstKeptEntryId keeps the orphan in the chain (it falls after the cutoff). This means the partial
thinking is preserved as context for subsequent turns, polluting the context window with incomplete reasoning.

Issue 5: Excessive File Re-Reads (30× same file)

interp.py was read 30 times out of 69 total tool calls (43%). The model can't retain the file content in context and keeps re-reading it to verify edits. This is a symptom of the
small model + massive thinking consuming context.

Issue 6: Duplicate Consecutive ToolResults

Positions 130 and 132 contain byte-identical toolResult content (both are the full text of demo1.lisp, 69 chars). The model read the same file twice in consecutive turns with no
intervening change.

Issue 10: No Text Response from Assistant (Until Very Late)

All 71 assistant messages use thinking + toolCall pattern — zero assistant messages contain explanatory text until position 105 (after the second user message). The agent never
provides a summary, status update, or explanation to the user during its ~8 minutes of active work on the first task.

────────────────────────────────────────────────────────────────────────────────

Root cause chain: The 2.6B model is too small → generates massive repetitive thinking → consumes context → can't retain file content → re-reads files 30× → hits token limit →
compaction fires mid-turn → leaves orphaned partial thinking → context pollution continues the cycle.

---

Now I have the full picture. Here's my comprehensive analysis:

────────────────────────────────────────────────────────────────────────────────

Session Log Analysis: 019fffb7-d556-7933-b8e4-da375514ed7e.jsonl

### Summary

A session attempting to build a Lisp interpreter in Python using a 2.6B parameter model (LiquidAI/LFM2.5-2.6B via llama.cpp). The session ran 71 LLM requests, produced broken code,
and never completed successfully. The model ended in a degenerate "code looks correct" repetition loop.

────────────────────────────────────────────────────────────────────────────────

### Issue 2: Empty/Useless Compaction Summaries

Both compactions produced essentially empty summaries that preserve zero conversational context:

Compaction 1 (line 79, tokensBefore: 123,913):

```
  No prior history.
  ---
  **Turn Context (split turn):**

  <modified-files>
  ...6 file paths...
  </modified-files>
```

Compaction 2 (line 137, tokensBefore: 123,963):

```
  ---
  **Turn Context (split turn):**
  <modified-files>...same 6 paths...</modified-files>
```

Neither summary contains any information about: what code was written, what errors occurred, what fixes were attempted, or what the current state of the code is. The model loses
all memory of its work.

────────────────────────────────────────────────────────────────────────────────

### Issue 3: "Code Looks Correct" Repetition Loop

The model says "The code looks correct" 24 times across the session, stuck in a verification loop:

- Reads the same 30 lines of interp.py repeatedly
- Concludes "looks correct" each time
- Runs a test → gets an error
- Reads the same code again
- Concludes "looks correct" again
- Never makes meaningful progress

Right before compaction 1 (10:09-10:10), the model repeated this pattern 3× in rapid succession with identical thinking. After compaction 2 (10:52), it repeated again 3×.

────────────────────────────────────────────────────────────────────────────────

### Issue 4: 30 Reads of the Same File

┌────────────┬────────────┐
│ File       │ Read Count │
├────────────┼────────────┤
│ interp.py  │ 30         │
├────────────┼────────────┤
│ demo1.lisp │ 5          │
├────────────┼────────────┤
│ demo2.lisp │ 1          │
├────────────┼────────────┤
│ demo3.lisp │ 1          │
└────────────┴────────────┘

The model re-reads interp.py 30 times — often the same offset/limit — because it can't retain context across turns.

────────────────────────────────────────────────────────────────────────────────

### Issue 5: 50% Edit Failure Rate

┌───────┬───────┬─────────┐
│ Tool  │ Total │ Failed  │
├───────┼───────┼─────────┤
│ edit  │ 10    │ 5 (50%) │
├───────┼───────┼─────────┤
│ write │ 6     │ 0       │
├───────┼───────┼─────────┤
│ bash  │ 16    │ 0       │
├───────┼───────┼─────────┤
│ read  │ 37    │ 0       │
└───────┴───────┴─────────┘

5 out of 10 edit attempts returned "Could not find the exact text" — the model's oldText didn't match the file. This is a known weakness of small models with the edit tool.

────────────────────────────────────────────────────────────────────────────────

### Issue 6: 8 Runtime Tracebacks, Never Fixed

The code produced 8 Python tracebacks during testing. The model never successfully resolved them. The final session message was:

│ "No modifications are required to the source code."

— despite the code still being broken.

────────────────────────────────────────────────────────────────────────────────

### Issue 10: Session Never Completed

The final assistant message is thinking + text (no tool call), ending with:
- Thinking: "Let me provide a final response confirming that everything is working properly..."
- Text: "No modifications are required to the source code."

The interpreter was still broken. The demos were never successfully run. The task was abandoned in a false-positive conclusion.

 ────────────────────────────────────────────────────────────────────────────────

 ### Root Cause Chain

```
  Model downgrade (27B→2.6B)
      ↓
  Weak reasoning + poor tool use
      ↓
  Edit failures → re-reads → context bloat
      ↓
  max_tokens shrinks → truncated responses
      ↓
  Compaction triggers → empty summary → total context loss
      ↓
  Model re-reads everything from scratch
      ↓
  Repetition loop ("code looks correct")
      ↓
  False completion
```
