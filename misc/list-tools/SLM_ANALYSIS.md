# SLM Analysis: `LiquidAI/LFM2.5-2.6B` vs `Qwen/Qwen3.8-27B` in the `pi` harness

**Date:** 2026-08-16
**Task under test (identical for both models):** interactive `pi` session, same working
directory (`/home/mtasic/projects-b/pi-slm`), same 4373-char system prompt, same 4 built-in
tools (`read`, `write`, `edit`, `bash`), same 3 loaded skills (`git`, `webfetch`, `websearch`).

- Run 1 prompt: `list avilable tools`
- Run 2 prompt: `just list avilable tools - do not list skills` (follow-up correction)

**Inputs analyzed (read-only):**

| Run | Dir | Session JSONL | Debug dir (`pi-llm-debugging`) | HTML export |
|---|---|---|---|---|
| LFM-0 | `list-tools-lfm2.5-2.6b-0` | `01a00928-….jsonl` (5 LLM requests) | 5 req + 5 res-meta | `pi-session-…06-01-14….html` |
| LFM-1 | `list-tools-lfm2.5-2.6b-1` | `01a0092b-….jsonl` (1 LLM request) | 1 req + 1 res-meta | `pi-session-…06-03-34….html` |
| Qwen-0 | `list-tools-q3.8-27b-0` | `01a0094c-….jsonl` (1 LLM request) | 1 req + 1 res-meta | `pi-session-…06-40-08….html` |
| Qwen-1 | `list-tools-q3.8-27b-1` | `01a0094d-….jsonl` (1 LLM request) | 1 req + 1 res-meta | `pi-session-…06-41-31….html` |

All four requests were sent through **llama.cpp** (OpenAI-compatible API via Caddy).
Diffing the request payloads confirms the **system prompt, tool schemas and user message are
byte-identical** across models; only per-model sampling parameters differ (set in pi's model
config, not a model flaw):

| Param | LFM2.5-2.6B | Qwen3.8-27B |
|---|---|---|
| temperature | 0.1 | 1.0 |
| top_k | 50 | 20 |
| top_p | — | 0.95 |
| repeat_penalty | 1.1 | 1.0 |
| thinking | `reasoning_effort: high` | `enable_thinking: true`, level `xhigh` |
| max_tokens | 49152 | 65536 |

---

## 1. What happened, run by run

### LFM-0 — "list avilable tools" → **WRONG answer after 5 LLM turns / 6 tool calls**

| # | Time (UTC) | Δ (s) | Output tok | Model action |
|---|---|---|---|---|
| 1 | 06:01:32.440 | 2.6 | 244 | 3× parallel `read` of the three SKILL.md files (git, websearch, webfetch) |
| 2 | 06:01:34.996 | 2.6 | 245 | `bash: ls -la .agents/skills/` |
| 3 | 06:01:36.601 | 1.6 | 171 | `bash: ls -la .agents/skills/skman/` |
| 4 | 06:01:37.777 | 1.2 | 121 | `bash: ls -la .agents/skills/webfetch/scripts/` |
| 5 | 06:01:42.549 | 4.8 | 563 | Final answer |

The final answer is a **polished, well-structured list of the 3 skills** ("git", "websearch",
"webfetch") presented as the available tools, with detailed feature bullets. **It does not
mention the actual callable tools (`read`, `write`, `edit`, `bash`) at all**, and ends:
*"All are available through the pi-coding-agent skill system."*

Notable details from the reasoning traces:

- Turn 1 thinking: *"I can see there are several skills/tools mentioned … Let me first try to
  read the skill files to understand what they do"* — the model conflated **skills** and
  **tools** from the very first thought and chose to *verify by reading* instead of answering
  from the tool list it already had in context.
- Turn 4 thinking says *"Let me also check the scripts directory"* and then lists
  `webfetch/scripts/` (not `skman/scripts/`) — aimless exploration, no coherent plan.
- It listed the `skman/` directory but **never read `skman/SKILL.md`**, then hallucinated:
  *"the skman directory also has a SKILL.md but it seems to be a reference file for the git
  skill."* (skman is an independent skill: the Agent Skills System introduction.)
- Usage: 5 requests; 1786 input tokens on the first request; cumulative context 2030 → 4828 →
  5158 → 5413 → 6064 tokens; **1344 output tokens** total.

### LFM-1 — "just list avilable tools - do not list skills" → **CORRECT answer, 1 turn**

Single LLM call, 0 tool calls, 588 output tokens, 5.5 s. Answer:

> 1. **read** – Read file contents  2. **write** – Write content to a file
> 3. **edit** – Precisely replace text  4. **bash** – Execute bash commands

The thinking trace shows the model *does* know the four tools and can distinguish them from
skills once the ambiguity is removed — but its reasoning is shaky
(*"skills … aren't separate tool invocations here"* is imprecise). One environment artifact:
this run started ~2 min after LFM-0, and the llama.cpp server's prompt cache was warm
(`cacheRead: 1270` on the first request), so its first-request "input" count (522) looks
anomalously small versus LFM-0's 1786. Prompt size is actually identical (~1792 tokens).

### Qwen-0 — "list avilable tools" → **CORRECT answer, 1 turn, 0 tool calls**

15.1 s (27B model, `xhigh` thinking, ~33 tok/s). Answered directly with a table of the four
tools (`read`, `write`, `edit`, `bash`) and *additionally* noted the three loaded skills,
explicitly separating the two concepts. Its thinking trace shows it noticed the prompt's
typo, considered the ambiguity ("user might be asking about tools in a different sense"), and
deliberately chose the tools interpretation **without any tool calls**. 499 output tokens,
2498 cumulative context.

### Qwen-1 — "just list avilable tools - do not list skills" → **CORRECT, 1 turn**

Minimal correct 4-tool list, 79 output tokens, 2.9 s. No hedging, no tool calls.

---

## 2. Metrics comparison

| Metric | LFM-0 | LFM-1 | Qwen-0 | Qwen-1 |
|---|---|---|---|---|
| Correct answer on first prompt | **No** (listed skills) | n/a (2nd prompt) | **Yes** | n/a (2nd prompt) |
| Answer correct | No | Yes | Yes | Yes |
| LLM requests | 5 | 1 | 1 | 1 |
| Tool calls | 6 (3 read, 3 bash) | 0 | 0 | 0 |
| Output tokens | 1344 | 588 | 499 | 79 |
| Σ cumulative context tokens | 23 493 | 2380 | 2498 | 2084 |
| Wall time (user → final) | 12.7 s | 5.5 s | 15.1 s | 2.9 s |
| Extra user turn required | **Yes** | — | No | — |
| Hallucinations | 1 (skman misattribution) | 0 | 0 | 0 |
| Malformed tool calls / errors | 0 | 0 | 0 | 0 |

Per-token speed (local llama.cpp): LFM ≈ 100–118 tok/s, Qwen ≈ 27–33 tok/s — **LFM is ~3–4×
faster per token**, but its 5-turn detour consumed the entire speed advantage (12.7 s vs
15.1 s) *and* still produced the wrong answer. If LFM had answered in one turn like Qwen, it
would have finished in ~2–3 s.

---

## 3. Behavioural findings for `LiquidAI/LFM2.5-2.6B`

**F1 — Skills/tools conflation (root cause of the failure).**
The system prompt presents the 4 tools as terse one-line bullets near the top, while the
`<available_skills>` block contains three long, salient descriptions. The 2.6B model latched
onto the skills and treated "available tools" as "the capabilities described in my prompt". A
27B model disambiguated correctly with zero tool calls. This is a **salience/attention
imbalance**, not a knowledge gap: LFM-1 proves the model knows the four tools when the
question is unambiguous.

**F2 — Tool-call inflation on meta-questions.**
Instead of answering from context, LFM spent 6 tool calls "verifying" what was already in the
system prompt (re-reading skill files whose summaries it had been given, then `ls`-ing the
skill tree). Each detour cost one extra LLM round-trip (1.2–2.6 s) plus 120–250 output tokens.
Pattern: *small model treats "I should be thorough" as "I should explore the filesystem"*,
even when the exploration cannot change the answer.

**F3 — Unbounded, planless exploration.**
The exploration had no end-state: after listing `skman/` it never read `skman/SKILL.md`
despite its own thinking saying it should; it then listed `webfetch/scripts/` for no reason.
There is no internal budget on tool turns — with a real coding task this pattern scales into
long, wandering agentic loops.

**F4 — Hallucination to fill gaps.**
The skman misattribution ("seems to be a reference file for the git skill") is a classic
confabulation: an unknown was filled with a plausible guess, stated in the thinking as a
conclusion. In a coding context the same behaviour produces invented file paths, APIs or
error causes.

**F5 — Format quality masks content errors.**
The wrong answer is arguably *better formatted* than Qwen's correct one (headings, bold,
bullets, a summary line). Users skimming the structure get no signal that the content is
wrong. Small-model fluent output should not be read as confidence.

**F6 — Recoverable with an explicit correction.**
The follow-up "do not list skills" produced a fully correct answer in one turn with zero tool
calls. LFM's failures here are *interpretation* failures, corrected cheaply by negation —
useful for any harness-side mitigation (see `SLM_RECOMMENDATION.md`).

**F7 — Agentic plumbing is solid.**
What LFM2.5-2.6B got *mechanically* right: 6/6 valid tool calls (correct names, valid JSON
arguments, correct argument schema conformance), successful **parallel tool calls** in turn 1
(3 reads in one response), coherent multi-turn state tracking across 5 requests, present and
mostly-sensible reasoning traces at `high` effort, no malformed output, no loops or crashes.
The model can drive the harness; its *decisions about when to use tools* are the weak point.

**F8 — Thinking at `high` is present but not self-correcting.**
Qwen at `xhigh` explicitly debated the ambiguous interpretation and picked the right one.
LFM's `high`-effort traces are 1–5 sentences per turn that rationalize each next action
rather than challenging it. For this model class, reasoning effort raises fluency more than
decision quality.

---

## 4. Caveats

- **n = 2 prompts per model, one task class** (a meta/capability question). Coding-task
  quality of LFM2.5-2.6B is not assessed here.
- Prompt-cache state differs across runs (LFM-1 benefited from a warm llama.cpp cache);
  latency/token comparisons should be re-run cold.
- Per-model sampling parameters (temperature/top_k/repeat_penalty) are configured in pi, not
  chosen by the models; LFM's conservative settings (temp 0.1) may also suppress recovery
  from its initial misinterpretation.
- Both models were run through the same llama.cpp server, so KV-cache prefix reuse and
  throughput are environment-dependent.
