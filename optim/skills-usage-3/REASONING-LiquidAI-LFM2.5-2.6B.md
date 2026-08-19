# LFM2.5-2.6B — reasoning_content style (Liquid Foundation Model)

How `LiquidAI/LFM2.5-2.6B` reasons, as observed from its `reasoning_content`
output. Source: 10 probe tasks across 5 categories (mathematics, code, general
reasoning, instruction following, tool use; each needing ~5-20 reasoning steps),
plus the recorded pi conversation. Raw probe outputs: `/tmp/lfm25-probe/out/*.json`.
Probed with the deployment settings (temperature 0.1, top_k 50, repeat_penalty 1.1,
`reasoning_effort: high`).

## 1. Structure — a consistent 5-part shape

1. **Restate the task first** (almost always the very first sentence):
   - `The user wants me to solve a word problem about pencils.`
   - `The user wants: "<verbatim quote of the instruction>"` (instruction following)
   - `We need to write a Python function ...` (code)
   - `Okay, so I need to find the exact angle ... Hmm, let me think about how clock angles work.`
2. **List constraints / plan** — numbered or bulleted, sometimes bold headers:
   - `**1. Identify the initial quantity:** * Total packs = 24 ...`
   - `The user wants me to: 1. Count lines ... 2. Use the read tool ... 3. Answer in one sentence`
3. **Execute step by step** — `Step 1: ...`, `First, ...`, `Now, ...`, `Next, ...`.
   Math uses inline LaTeX (`$24 \times 12 = 288$`), code uses ``` fences,
   tool results are reported factually: `The grep command returned 3 lines ...`
4. **Self-check / self-correct** — `Both methods yield the same result.`,
   `Wait, no. ...`, `But wait, the problem is at 3:15.`, `This is strange - ...`,
   `Yes, that's right.`, `All good.`, `Check each sentence count: Yes three.`
5. **Close by planning the final output** — `Let's write it out clearly.`,
   `Now final output.`, `Thus final answer:`, `Let me provide a clear answer ...`

**The final answer follows the plan the reasoning ends with.** In the recorded
failure the reasoning ended with `Let me provide a clear answer listing all
available tools` — so the model listed the tools. Whatever action the reasoning
last commits to is the action the answer performs.

## 2. Voice markers

- First person singular: `I need to`, `I will`, `I remember`, `I can be confident that`;
  `We need to` / `Let's` for code/writing tasks.
- Actions: `Let me ...` (very frequent), `Let's ...`, `I will ...`, `I should ...`.
- Filler transitions: `Okay, so ...`, `Hmm, let me think ...`, `Well, ...`,
  `But wait, ...`, `Actually, ...`, `This is strange - ...`.
- No meta talk ("as an AI"), no apology, no hedging about capabilities.
- Present tense; plans in future tense. No markdown headers at the very start
  (headers appear only inside long structured reasoning).

## 3. Length scales with task complexity

- **Simple action / lookup: 1-2 sentences, ~70-150 chars.** Recorded:
  `I found 4 tools. I will pick the narrowest tool that fits the task.`
- **One-step tool action: 2-4 sentences, ~150-500 chars.** Probed:
  `The grep command returned 3 lines containing "apple". Now I need to answer in one sentence.`
- **Multi-step problem: 1000-9000 chars**, numbered steps, self-checks,
  alternative method, then `Now final output.`

## 4. Tool-use loop pattern

`plan (numbered) + "Let me start by ..."` → tool call(s) →
`report result factually ("The X command returned ...")` →
`"Now I need to ..."` → next call or final answer.
On a tool error it narrates and falls back: `The write tool is returning "unknown tool write"
error. This is strange - ... Let me try using bash instead.`

## 5. Writing guide — how to write a `reasoning_content` in this model's voice

(Use this when composing a synthetic assistant message for this student model.)

1. Open by naming the situation in the first person: `The user ...`, `I found ...`,
   `I need to ...`.
2. State the decision/action with `I will ...` or `Let me ...`.
3. For a simple action use the **short form**: 1-3 plain sentences, no lists,
   no headers, no LaTeX. Example:
   `The user is activating a skill. I will follow the block and reply with the short confirmation it asks for.`
4. Only use numbered steps / bold / lists for genuinely multi-step reasoning.
5. End by committing to the exact next action (the answer follows this commitment).
6. Never mention "the reasoning", "this is a synthetic message", or anything meta.
   Write as if you are the model thinking naturally.
