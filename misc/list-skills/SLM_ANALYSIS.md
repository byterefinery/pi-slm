# SLM Analysis — `LiquidAI/LFM2.5-2.6B` in the `pi` coding agent

Baseline: `Qwen/Qwen3.8-27B`, same harness (pi 0.84.2), same provider (llama.cpp router),
same working directory, same system prompt, same two user prompts.

Evidence base (read-only):

| Run | Dir | Session ID | Prompt | LLM calls |
|---|---|---|---|---|
| LFM-0 | `misc/list-skills-lfm2.5-2.6b-0/` | `01a0091a-…62d467498bdc` | "list available skills" | 5 |
| LFM-1 | `misc/list-skills-lfm2.5-2.6b-1/` | `01a0091c-…94a1755e4c4a` | "just list avilable skills, do not load them" | 1 |
| Q-0 | `misc/list-skills-q3.8-27b-0/` | `01a00926-…7b962960da2b` | "list available skills" | 1 |
| Q-1 | `misc/list-skills-q3.8-27b-1/` | `01a00928-…e67f5799ec9f` | "just list avilable skills, do not load them" | 1 |

Each dir contains the session JSONL (`*.jsonl`), raw provider requests
(`pi-llm-debugging/<id>/NNN-req.json`), response headers (`NNN-res-meta.json`), and an
HTML export. All numbers below come from those files.

## 1. Test setup (from the request payloads)

| | LFM2.5-2.6B | Qwen3.8-27B |
|---|---|---|
| Provider | llama.cpp (router) | llama.cpp (router) |
| Thinking level | `high` (`reasoning_effort: "high"`) | `xhigh` (`chat_template_kwargs.enable_thinking: true`) |
| Sampling | temp 0.1, top_k 50, repeat_penalty 1.1 | temp 1, top_p 0.95, top_k 20, min_p 0 |
| max_tokens | 49152 | 65536 |
| Tools in payload | read, bash, edit, write, **vcc_recall** | read, write, edit, bash |
| System prompt | pi default, **4699 chars** | pi default, **4373 chars** |

The two system prompts are the same pi default prompt. Differences:

1. LFM's runs loaded the `vcc_recall` session-history extension tool (extra tool-schema
   tokens); Qwen's runs did not.
2. Both prompts declare **exactly three skills** in `<available_skills>`: `git`,
   `webfetch`, `websearch`. The on-disk skill dir `.agents/skills/` actually contains
   **five** skill dirs — `pi-0-84-2` and `skman` exist on disk but were **not declared**
   in the system prompt of either run (LFM's own `ls` in LFM-0 confirms five dirs).

"Available skills" therefore has two candidate meanings: the three the harness declared
(what the agent can actually trigger), and the five present on disk. The models diverged
on this — and the divergence drives most of the measured gap.

## 2. Observed behaviour

### LFM2.5-2.6B, run 0 — "list available skills"

Five LLM calls, seven tool calls (3 `read` + 2 `bash` + 2 `read`), 16.4 s from user
message (05:46:10.829) to final answer (05:46:27.225):

| Call | Δt | New input | Cached input | Output | Action |
|---|---|---|---|---|---|
| 1 | 2.6 s | 2135 | 0 | 219 | parallel `read` of git, webfetch, websearch SKILL.md |
| 2 | 3.0 s | 2554 | 2353 | 294 | `bash: ls -la .agents/skills/` |
| 3 | 1.4 s | 184 | 5200 | 147 | `bash: ls` of the two unexpected dirs |
| 4 | 1.6 s | 236 | 5530 | 165 | parallel `read` of pi-0-84-2, skman SKILL.md |
| 5 | 7.8 s | 6976 | 5930 | 750 | final table |
| **Σ** | **16.4 s** | **12085** | **19013** | **1575** | |

Totals: 31 098 prompt tokens processed, 1 575 output tokens, ≈ 32.7k tokens all-in.
The tool results injected ≈ 36.4k chars (≈ 9.1k tokens) of full skill-file content
(git 3.7k, webfetch 3.6k, websearch 2.5k, pi-0-84-2 10.0k, skman 16.6k chars, plus two
`ls` outputs) into a 2.6B context to answer a question whose answer was already in the
system prompt.

Behaviour chain visible in its thinking traces:

1. Call 1 — it *saw* the three declared skills ("Looking at the available_skills, I can
   see: 1. git 2. webfetch 3. websearch") yet concluded "I should verify this by reading
   the actual skill files to make sure there aren't more". It also says "I can see from
   the **previous context**" — there is no previous context; this is a fresh session.
   The model misattributes the system prompt as conversation history (provenance
   confabulation).
2. Call 2 — after reading all three files, still unsure: "I should double-check by
   looking at the actual skill directory".
3. Calls 3–4 — the `ls` reveals two *undeclared* skill dirs; it then reads both full
   SKILL.md files (≈ 6.7k tokens) "to get the complete list".
4. Call 5 — produces a table of **five** skills and states all of them are available and
   "can be used directly".

### LFM2.5-2.6B, run 1 — "just list avilable skills, do not load them"

One call, zero tools, 2.4 s. 524 new + 1 619 cached input, 239 output (≈ 149 thinking).
It lists exactly the three declared skills and explicitly refrains from loading:
"No need to load them further — they're already defined in the system."
(Thinking again references "the previous context" — same provenance confabulation, but
the behaviour is correct.)

### Qwen3.8-27B, runs 0 and 1

One call each, zero tools, 4.0 s each. 1 997/2 005 input, 148/127 output, of which only
≈ 44/48 tokens are thinking. Its reasoning: "The system prompt has an
`<available_skills>` section with three skills. This is a simple question. I should just
list them." / "No need to read the files." Answers = exactly the three declared skills.

## 3. Head-to-head numbers (prompt "list available skills")

| Metric | LFM2.5-2.6B (LFM-0) | Qwen3.8-27B (Q-0) | Ratio |
|---|---|---|---|
| LLM calls | 5 | 1 | 5.0× |
| Tool calls | 7 (5 reads, 2 bash) | 0 | — |
| Prompt tokens processed | 31 098 | 1 997 | 15.6× |
| Output tokens | 1 575 | 148 | 10.6× |
| All-in tokens | 32 673 | 2 145 | **15.2×** |
| Wall time (user msg → answer) | 16.4 s | 4.0 s | 4.1× |
| Gen speed (avg, incl. prefill) | ≈ 96 tok/s | ≈ 37 tok/s | 2.6× faster |
| Skills listed | 5 (incl. 2 **undeclared**) | 3 (exactly the declared set) | — |
| Answer correctness re harness semantics | over-claims (see §4) | correct | — |

For the explicit "do not load them" prompt both models are equivalent: 1 call, 0 tools,
LFM 2.4 s / 2 382 tokens vs Qwen 4.0 s / 2 132 tokens. The gap exists only when the
model is left to its own judgment.

## 4. Quality assessment of LFM2.5-2.6B

### What it did well

- **Well-formed tool use.** All seven tool calls were valid JSON; it correctly batched
  parallel calls (3 parallel `read`s in one message, again 2 in call 4) — parallel
  tool-calling works in this model.
- **Fast per token.** ≈ 96 tok/s vs ≈ 37 tok/s for the 27B model; a single LFM call is
  cheaper in latency than a single Qwen call.
- **Instruction-following when explicit.** Run 1 shows it can follow "do not load them"
  perfectly, including not loading *anything*. The failure mode is the *default* policy,
  not compliance capacity.
- **Honest enumeration effort.** It did discover the two undeclared skill dirs, so its
  answer was the more *complete* one on a filesystem basis.
- **Prompt caching works.** 19 013 of 31 098 prompt tokens were cache-reads
  (llama.cpp prefix cache); per-call prefill stayed cheap.

### What it got wrong (the core finding)

1. **Distrust of declared state / over-verification.** Qwen treats
   `<available_skills>` as the complete, authoritative list; LFM treats it as a hint to
   be re-verified against the filesystem. On a question requiring **zero** tools, it
   spent 5 calls, 7 tool executions, ≈ 9.1k tokens of file content, and 12 s extra.
   This is the single largest quality gap in the sample and is likely a recurring
   pattern (any "what X is available?" question invites the same loop).
2. **Worse final answer, at 15× cost.** "Available" in a pi session means *declared in
   the system prompt* — undeclared skills are never injected or triggered, so listing
   `pi-0-84-2` and `skman` as available, and suggesting the user "can use any of them
   directly", is semantically wrong about the harness. Its extra work produced an
   over-claim, not extra accuracy.
3. **Provenance confabulation.** In a fresh session its thinking says "from the previous
   context" (both runs). It does not cleanly model "system prompt = current declared
   state" vs "conversation history". This is consistent with finding #1: it cannot
   confidently tell what it already knows.
4. **Thinking budget not productive.** `high` thinking produced ≈ 923 thinking tokens in
   run 0 (≈ 59% of all output tokens), mostly re-planning across calls ("let me verify",
   "I should double-check", "let me read these files") that a single correct decision in
   call 1 would have eliminated. Qwen used 44 thinking tokens on the same task at
   `xhigh`.
5. **Context dilution risk.** After stuffing five SKILL.md bodies (≈ 9.1k tokens) into a
   2.6B model's context, every subsequent call pays prefill for content the answer
   didn't need; on real (longer) coding tasks the same habit accelerates
   auto-compaction and degrades recall (call 5 prefilled a 12.9k-token prompt).

### Caveats

- Sample size: 4 sessions, one task family (skill enumeration), all local via llama.cpp.
- The two runs were not perfectly symmetric: LFM's requests carried the extra
  `vcc_recall` tool (≈ 0.3k extra prompt tokens) and a longer system prompt. This
  inflates LFM's absolute token counts slightly, not its call count or behaviour.
- On tasks that genuinely need file contents, LFM's instinct to read files is correct —
  the defect is that it applies it to questions already answered by declared context.

## 5. Bottom line

For the measured task class, `LiquidAI/LFM2.5-2.6B` in pi is functionally usable
(well-formed tools, fast, compliant when told explicitly) but has a systematic
**over-verification default**: it re-reads the filesystem to confirm what the system
prompt already declares, costing ≈ 15× tokens, ≈ 4× latency, and — because it mixes
on-disk skills with harness-declared skills — a less accurate answer than
`Qwen/Qwen3.8-27B`'s single no-tool call. The model follows explicit counter-instructions
perfectly (run 1), which is exactly the lever an extension can pull: move the rule from
user phrasing into the harness.
