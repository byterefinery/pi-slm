# SLM Analysis: `LiquidAI/LFM2.5-2.6B` in the pi coding agent

**Benchmark:** `tzip-webfetch` — load the `tzip` (token-pruning) and `webfetch` (page → markdown) skills, then fetch `https://tangledgroup.com/` with the webfetch script.

**Baseline:** `Qwen/Qwen3.8-27B` under the identical harness.

**Evidence (read-only):**
- `misc/tzip-webfetch-lfm2.5-2.6b-0/` — session `01a00951-…` JSONL (28 entries), 13 LLM request/response debug pairs, HTML export
- `misc/tzip-webfetch-q3.8-27b-0/` — session `01a00956-…` JSONL (12 entries), 5 LLM request/response debug pairs, HTML export

---

## 1. Harness (identical for both models)

| | LFM2.5-2.6B | Qwen3.8-27B |
|---|---|---|
| Provider / transport | `llamacpp` (llama.cpp server behind Caddy), OpenAI-completions API | same |
| System prompt | 4,848 chars (pi default) | 4,848 chars (byte-identical) |
| Tools | `read`, `write`, `edit`, `bash` | same |
| Thinking level | `high` → `reasoning_effort: "high"` | `xhigh` → `enable_thinking: true, preserve_thinking: true` |
| Sampling | temp **0.1**, top_k 50, repeat_penalty **1.1** | temp 1.0, top_p 0.95, top_k 20, min_p 0 |
| max_tokens | 49152 | 65536 |
| cwd | `/home/mtasic/projects-b/pi-slm` | same |

Both sessions: zero provider errors (all HTTP 200), no malformed tool calls, no abnormal `stopReason` (only `stop` / `toolUse`). The SLM's problems are **behavioral, not transport-level**.

Note: the llama.cpp provider reports `usage.reasoning = 0` for both models, so thinking-token volume is not observable from the logs; thinking *text* is, and is quoted below.

## 2. Task scripts (as recorded in the logs)

- **LFM session:** `tzip` skill + trailing word `full` (sent **twice**) → `webfetch` skill (ends with bare URL `https://tangledgroup.com/`) → explicit user message **“Fetch a URL with webfetch”** → (after success) `webfetch` skill re-sent.
- **Qwen session:** `tzip` skill + `full` → `webfetch` skill (+ same trailing URL). No explicit fetch command — the model started the fetch itself upon skill load.

The webfetch skill's Usage block shows the command as a bare PATH-style invocation (`webfetch.py <url>`) and says the script is self-contained PEP 723, “`uv run --script` resolves them automatically”, with “References are relative to `<skill-dir>`”. It never prints the absolute path. This matters, see §4.

## 3. Trajectories

### 3.1 LFM2.5-2.6B — 13 LLM calls, 8 tool calls, **4 failed**

| # | t (UTC) | Action | Result |
|---|---|---|---|
| 1 | 06:46:25 | *(tzip skill + `full`)* | → “**tzip lite activated**” + 4-mode menu, 862 out-tokens |
| 2 | 06:46:51 | *(tzip skill + `full` again)* | → “tzip lite activated” again (378 out-tokens) |
| 3 | 06:47:12 | *(webfetch skill)* | → verbose ack, “what would you like to do?”, 602 out-tokens |
| 4 | 06:47:34 | “Fetch a URL with webfetch” | ↓ |
| 5 | 06:47:36 | `bash: webfetch.py https://tangledgroup.com/` | **✗ 127** command not found |
| 6 | 06:47:40 | `bash: uv run webfetch.py https://tangledgroup.com/` | **✗** uv: `No such file or directory` |
| 7 | 06:47:43 | `bash: find / -name "webfetch.py" 2>/dev/null \| head -20` | ✓ **19.7 s full-disk scan**, 10 hits incl. stale copies in other projects |
| 8 | 06:48:05 | `bash: cd <skill>/scripts && webfetch.py <url>` | **✗ 127** (missing `./`) |
| 9 | 06:48:07 | `read: <skill>/scripts/webfetch.py` | ✓ sees shebang `#!/usr/bin/env -S uv run --script` |
| 10 | 06:48:12 | `bash: chmod +x <skill>/scripts/webfetch.py` | ✓ |
| 11 | 06:48:13 | `bash: cd <skill>/scripts && webfetch.py <url>` | **✗ 127 again — same command as #8** |
| 12 | 06:48:21 | `bash: <abs>/scripts/webfetch.py <url>` | ✓ **first success (attempt 8)**; “Installed 67 packages in 238ms” + page markdown |
| 13 | 06:48:26 | summary text | ✓ (but mis-attributes the fix, see FM-7) |
| 14 | 06:48:42 | *(webfetch skill re-sent)* | → re-confirms prior success, no new fetch, 215 out-tokens |

### 3.2 Qwen3.8-27B — 5 LLM calls, 3 tool calls, **0 failed**

| # | t (UTC) | Action | Result |
|---|---|---|---|
| 1 | 06:51:21 | *(tzip skill + `full`)* | → “**tzip full activated**” (118 out-tokens) |
| 2 | 06:51:41 | *(webfetch skill)* | → immediately starts the fetch (trailing URL in skill = task) |
| 3 | 06:51:45 | `bash: ls <skill>/` | ✓ `scripts`, `SKILL.md` |
| 4 | 06:51:46 | `bash: ls <skill>/scripts/` | ✓ `webfetch.py` |
| 5 | 06:51:48 | `bash: uv run --script <abs>/scripts/webfetch.py <url>` (`timeout: 120`) | ✓ **first attempt**; page markdown |
| 6 | 06:51:52 | summary text | ✓ concise, tzip-full style, accurate |

## 4. Head-to-head metrics

| Metric | LFM 2.6B | Qwen 27B | Ratio |
|---|---|---|---|
| LLM calls (session) | 13 | 5 | 2.6× |
| Tool calls (fetch task) | 8 | 3 | 2.7× |
| Failed tool calls | 4 | 0 | — |
| First-attempt success | ✗ (8th attempt) | ✓ | — |
| Time to working fetch | 47.7 s | 6.7 s | 7.1× |
| Time to final summary | 51.2 s | 10.7 s | 4.8× |
| Session wall time (incl. user typing) | ~182 s | ~36 s | 5.0× |
| Model-generation time (sum of call latencies) | ~50 s | ~14 s | 3.6× |
| Tool-execution time | ~22 s (**19.7 s = `find /`**) | <1 s | — |
| Total output tokens | ~5,053 | ~727 | 6.9× |
| Output tokens, tzip-ack turn | 862 | 118 | 7.3× |
| Final LLM request size | 58.1 k chars / ~14.6 k tokens | 15.1 k chars / ~4.7 k tokens | 3.1× |
| Full-disk `find /` | yes, no timeout | no | — |
| tzip `full` mode respected | ✗ (answered “lite” ×2, unpruned output) | ✓ | — |
| Provider errors / malformed tool calls | 0 | 0 | — |

## 5. LFM2.5-2.6B failure modes (with evidence)

### FM-1 · Literal command copying; no path resolution from skill metadata
First attempt was a verbatim copy of the skill’s Usage example: `webfetch.py https://…` → `127 command not found`. The skill header already stated `location=…/.agents/skills/webfetch/SKILL.md` and “References are relative to …/skills/webfetch”, and the model’s own thinking before the `find /` call even lists the skill directory — yet it never formed the candidate `uv run --script …/skills/webfetch/scripts/webfetch.py`. Contrast Qwen’s thinking at its first fetch step: *“The referenced webfetch.py is a relative path from that directory … resolved via `uv run --script`.”* The SLM failed to join the skill’s location metadata with the invocation syntax; the 27B did in one hop.

### FM-2 · Missing `./` prefix (bash PATH semantics) — the core repeated error
After `find /` located the script, it ran `cd <dir> && webfetch.py …` — **twice** (steps 8 and 11). In bash, a command name is resolved via `PATH`; a file in the cwd needs `./`. Both calls died with 127 for exactly this reason. The model never once tried `./webfetch.py` or `uv run --script ./webfetch.py`; it only succeeded on the **absolute path** (step 12). This is a shell-semantics gap, not an environment problem.

### FM-3 · Fixed wrong hypothesis; no error-driven learning
Across steps 9–12 the thinking traces show the model blaming the shebang: *“the `-S` flag with `uv run` might be causing the issue. In some versions of uv, you need t…”* — while the actual cause (missing `./`/PATH lookup) was never hypothesized. After `chmod +x` it reasoned *“The file is now executable. Let me try running it again from the same directory”* and **re-issued the byte-identical failing command** (step 11), expecting the permission bit to fix a PATH resolution error. Correct diagnosis would have come from reading `/bin/bash: line 1: webfetch.py: command not found` as “name not found in PATH”, not “cannot execute file”.

### FM-4 · Wasteful, unscoped search: `find /`
Instead of `ls`-ing the skill directory it already knew (the 27B’s two `ls` calls, <1 s total), it scanned the entire disk: **19.7 s, no timeout set** (the bash tool accepts `timeout`; Qwen self-set one on its fetch). The result listed **10 copies** of `webfetch.py` — including stale ones under other project trees (`projects-t/*`, `pi-slm (copy 1)`, `/tmp/slm-test-*`). LFM happened to pick the correct (cwd-matching) copy; an unscoped search in a weaker run is a real risk of executing the wrong copy of a script.

### FM-5 · Ignored explicit instruction: tzip mode `full`
The tzip injection ended with the command word `full` (twice). Both times LFM answered **“tzip lite activated”** — its thinking never registers the trailing word at all (“no specific task given … asked to reply with a mode name”). The 27B answered “tzip full activated” on the first try. Trailing-word commands are below this model’s reliable instruction-following bar; structured/keyword-form commands are safer.

### FM-6 · Verbosity; tzip discipline not actually applied
The SLM emitted 6.9–7.3× more output tokens per conversational turn, including menus (“What would you like to do? I can: …”), mode lists, emoji checklists (✅) and pleasantries — all while *claiming* “tzip lite active”. The 27B’s tzip-full output is exactly per spec (fragments, arrows, no articles, 118 tokens for the ack). Net effect: the SLM defeats the purpose of a token-pruning skill and inflates context (final request 3.1× larger).

### FM-7 · Confabulated success narrative
Final summary: *“After making it executable with `chmod +x`, running it from its directory worked.”* It never worked from the directory — steps 8 and 11 (from the directory, after and before chmod) both failed; only the **absolute path** succeeded. The model stitched a plausible causal story from its own last three actions. In an interactive session this is cosmetic; in an unattended loop, claiming success for an action that didn’t happen is the dangerous variant.

### What worked (strengths)
- **Well-formed tool calling**: 8/8 tool calls parsed and executed; no JSON errors, no stop-reason anomalies, no repetition degeneration (temp 0.1 / top_k 50 / repeat_penalty 1.1 + `reasoning_effort: high` is a workable sampling point on llama.cpp).
- **Eventual recovery with legitimate steps**: `read` the script → inspect shebang → `chmod +x` → absolute path. The *final* three-step sequence was correct debugging; it just arrived after 5 detours.
- **Correct disambiguation**: of 10 `find /` hits it chose the cwd-matching copy.
- **Faithful content**: the final page summary matches the fetched markdown (services, contact, footer) with no hallucinated facts.
- **Safe defaults**: no destructive or out-of-scope commands were attempted anywhere in the session.

## 6. Root-cause synthesis

The 27B→2.6B gap in this benchmark is not about raw knowledge (the SLM *knew* the script path and read the shebang) but about three capabilities the small model lacks under time pressure:

1. **Implicit semantics** — bash name resolution (`./`, PATH) and uv invocation syntax are not stated anywhere in the context; the 27B supplies them from priors, the 2.6B copies the visible example literally.
2. **Cross-reference** — joining “skill location header” + “Usage block” + “PEP 723 note” into one concrete command is a multi-hop inference the SLM performs across five separate attempts instead of one.
3. **Error diagnosis** — mapping `command not found` to a path fix, instead of attaching the error to an unrelated conspicuous feature (the `-S` shebang), and updating the hypothesis after a failed retry.

Combined with weaker instruction priority (FM-5), weaker output economy (FM-6) and a success-narrative bias (FM-7), the observable pattern is: **the SLM converges, but only with ~3× more calls and ~7× more tokens, and its self-reported state may not match the actual state.**

## 7. Risk assessment for using LFM2.5-2.6B in pi

| Risk | Severity | Evidence |
|---|---|---|
| Wasted time/tokens on self-fixable errors | Medium | 4 of 8 calls failed; 7.1× slower task; 6.9× output tokens |
| Executing the wrong copy of a script (unscoped search) | Medium | `find /` returned 10 candidates; luck picked the right one |
| Confabulated success / state mismatch | **High in unattended use** | FM-7; final text describes a fix that did not happen |
| Ignoring explicit user commands (mode words) | Medium | FM-5 (twice) |
| Unbounded bash commands (no timeouts, `find /`) | Low–Medium | 19.7 s scan; a `find /` on a large disk or a runaway build would be far worse |
| Destructive-command risk | Not observed | No dangerous commands attempted; small-model impulsivity not demonstrated here, but unmitigated |

**Bottom line:** LFM2.5-2.6B is *usable* in pi for short, supervised, low-stakes tasks — it converges and its tool calling is clean — but left unassisted it burns ~3× the calls, ~7× the tokens and ~5× the wall time of a 27B model, and its summaries can drift from reality. The gaps are structural (path semantics, error diagnosis, instruction priority) and are exactly the kind of thing a harness-level reliability extension can compensate for, which is the subject of `SLM_RECOMMENDATION.md`.
