# SLM Analysis — LiquidAI/LFM2.5-2.6B in pi agent harness (webfetch task family)

**Sessions (2026-08-16, local llama.cpp behind Caddy):**

| Run | Directory | Session ID | Model | Thinking | LLM reqs | Wall time | User msgs | Tool calls | Failed tools | Output tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM webfetch | `webfetch-lfm2.5-2.6b-0` | `01a00957-fc5d-7d07-a382-07003b1a1426` | LiquidAI/LFM2.5-2.6B | high | 9 | 06:52:56 → 06:53:36 | 2 | 8 (5 bash, 3 read) | 1 | 2,300 |
| LFM tzip+webfetch | `tzip-webfetch-lfm2.5-2.6b-0` | `01a00951-ae82-7e59-82c7-d241ecbc59ab` | LiquidAI/LFM2.5-2.6B | high | 13 | 06:46:25 → 06:48:45 | 5 | 8 (7 bash, 1 read) | 4 | 5,249 |
| Qwen webfetch | `webfetch-q3.8-27b-0` | `01a00967-3e98-7138-938e-8ebd80625105` | Qwen/Qwen3.8-27B | xhigh | 5 | 07:09:45 → 07:09:59 | 1 | 4 (4 bash) | 1 | 679 |
| Qwen tzip+webfetch | `tzip-webfetch-q3.8-27b-0` | `01a00956-c459-7b8d-b768-e1621b92a866` | Qwen/Qwen3.8-27B | xhigh | 5 | 06:51:21 → 06:51:52 | 2 | 3 (3 bash) | 0 | 727 |

**Sources:** session `*.jsonl`, `pi-llm-debugging/<id>/{NNN-req,NNN-res-meta}.json` (all 32 requests HTTP 200), HTML session exports.

---

## 1. Executive summary

Both models were given the same short skill-driven task: a `<skill name="webfetch">` block (the webfetch skill, whose script lives at `.agents/skills/webfetch/scripts/webfetch.py` with a PEP 723 shebang) followed by the URL `https://tangledgroup.com/` — i.e., "fetch this URL with the skill". One variant additionally activates the `tzip` token-pruning skill in mode `full` first.

**Both models eventually fetched the page in every session. The difference is first-move reliability.** LFM2.5-2.6B needed 1.8–2.6× the requests, 3.4–7.2× the output tokens, 1–4 failed tool calls, and **1–2 operator re-prompts per session** because it (a) treated the URL as passive documentation instead of the task input, and (b) did not know how to invoke the skill script (bare `webfetch.py` → `127`, detour through `find /`, `chmod +x`, and a repeated identical command before it tried the absolute path that `find` had already printed). Qwen3.8-27B made the same first path guess and same one error, then recovered in 2 cheap `ls` calls and finished in 14–31 s with clean `scrapling` output.

Headline numbers (LFM vs Qwen):

| Metric | LFM2.5-2.6B | Qwen3.8-27B |
|---|---|---|
| Requests (webfetch / tzip) | 9 / 13 | 5 / 5 |
| Output tokens (webfetch / tzip) | 2,300 / 5,249 | 679 / 727 |
| Failed tool results (webfetch / tzip) | 1 / 4 | 1 / 0 |
| Operator re-prompts needed to fetch the URL | 1 / 2 | 0 / 0 |
| Seconds user-prompt → successful fetch | 36 (after nudge) / 117 (after 3 nudges) | 9 / 27 |
| Fetcher actually used for tangledgroup.com | `requests` fallback (regex markdown, degraded) / `scrapling` | `scrapling` / `scrapling` |
| `tzip full` mode followed | **No** — answered "lite" twice | **Yes** — "tzip full activated", pruned style |
| Doom loop | None (tasks too short) | None |

The failure mode here is not the catastrophic frozen loop seen in the lisp-python session. It is a **first-move reliability deficit**: the model does not execute the obvious task on turn 1, does not adapt skill-doc commands to reality (PATH, script location, interpreter environment), and over-communicates while under-acting. All of it is measurable, and most of it is harness-addressable (see `SLM_RECOMMENDATION.md`).

---

## 2. Setup

- **Harness:** pi v0.84.2, interactive sessions, provider `llamacpp` (llama.cpp behind Caddy), 4 tools (`read`, `write`, `edit`, `bash`), system prompt 4,848 chars (identical for both models), no compaction events in any of the four sessions.
- **Skills:** `webfetch` (project `.agents/skills/webfetch/`; script at `scripts/webfetch.py`, 13,127 bytes, mode 755, shebang `#!/usr/bin/env -S uv run --script`, PEP 723 deps `scrapling[all]`, `markdownify`, `requests`) and `tzip` (`.agents/skills-byterefinery/tzip/`, a pure-text style/pruning skill with mode argument `lite|full|ultra|off`).
- **Sampling (from `*-req.json`):**
  - LFM2.5-2.6B: `temperature 0.1`, `top_k 50`, `repeat_penalty 1.1`, `max_tokens 49152`, `reasoning_effort "high"`.
  - Qwen3.8-27B: `temperature 1.0`, `top_p 0.95`, `top_k 20`, `min_p 0`, `presence_penalty 0`, `repeat_penalty 1`, `max_tokens 65536`, `chat_template_kwargs {enable_thinking, preserve_thinking}` (thinking level xhigh).
  - i.e., each model ran under its own native sampling profile — a fair "default config" comparison.
- **Token speed:** LFM ≈ 104 out-tok/s (tzip req 1: 862 tokens in 8.25 s) vs Qwen ≈ 38 out-tok/s (118 tokens in 3.1 s). LFM is ~2.7× faster per token but spends 3.4–7.2× more output tokens per task.
- **User task (verbatim, both models):** `<skill name="webfetch">…full SKILL.md…</skill>\n\nhttps://tangledgroup.com/`; the tzip variant prepends `<skill name="tzip">…</skill>\n\nfull`.

---

## 3. Session timelines

### 3.1 LFM webfetch (`01a00957`) — fetched only after operator nudge

| Time (UTC) | Turn | Action | Outcome |
|---|---|---|---|
| 06:52:56 | — | User: webfetch skill block + `https://tangledgroup.com/` | — |
| 06:52:58 | 1 | `read SKILL.md` (**content already fully in the user message**) + `ls` project root | ok |
| 06:52:59 | 2 | `read <skill-root>/webfetch.py` | **ENOENT** (script is in `scripts/`) |
| 06:53:01 | 3 | `find <project> -name webfetch.py; ls <skill dir>` | finds `scripts/webfetch.py` |
| 06:53:02 | 4 | `read` full 13 KB script | ok (never ran anything yet) |
| 06:53:07 | 5 | Text: "script already present and well-implemented… let me test it" + `python3 scripts/webfetch.py --help` | ok |
| 06:53:08 | 6 | `python3 scripts/webfetch.py --md https://example.com` — **self-chosen test URL, not the requested one** | ok |
| 06:53:14 | 7 | **Stops.** Status table (typo "Scapling") + *"Is there a specific URL or use case you'd like me to test with the webfetch script?"* | **task not performed** |
| 06:53:31 | — | User nudge #1: `` `webfetch.py https://tangledgroup.com/` `` | — |
| 06:53:32 | 8 | `python3 scripts/webfetch.py --md https://tangledgroup.com 2>&1` | ok — but via **requests fallback** (no scrapling in system python) → degraded regex markdown |
| 06:53:36 | 9 | Summary of the page; done | ok |

### 3.2 LFM tzip+webfetch (`01a00951`) — wrong mode ×2, no fetch ×1, 5 failed invocations

| Time (UTC) | Turn | Action | Outcome |
|---|---|---|---|
| 06:46:25 | — | User: tzip skill + mode `full` | — |
| 06:46:33 | 1 | **862-token** reply: *"tzip **lite** activated"* + full menu of modes | **wrong mode** (user said `full`), un-pruned verbosity |
| 06:46:51 | — | User re-sends identical tzip + `full` | — |
| 06:46:54 | 2 | *"tzip lite activated"* again (378 tokens) | **still wrong**, no correction |
| 06:47:12 | — | User: webfetch skill + URL | — |
| 06:47:18 | 3 | "Both skills are active. What would you like to do?" (602 tokens) | **no fetch** |
| 06:47:34 | — | User nudge #2: `Fetch a URL with webfetch` | — |
| 06:47:36 | 4 | `webfetch.py https://tangledgroup.com/` (literal copy of SKILL.md Usage) | **127 command not found** |
| 06:47:40 | 5 | `uv run webfetch.py …` (no script path) | **uv error 2** |
| 06:47:43 | 6 | `find / -name webfetch.py` (~19.6 s, whole filesystem) | 10 hits **incl. the correct absolute path** |
| 06:48:05 | 7 | `cd <dir> && webfetch.py …` (bare name, despite absolute path in context) | **127** |
| 06:48:07 | 8 | `read` full 13 KB script | ok |
| 06:48:12 | 9 | `chmod +x <script>` (wrong diagnosis: "not executable") | no effect |
| 06:48:13 | 10 | `cd <dir> && webfetch.py …` (**byte-identical to turn 7**) | **127** |
| 06:48:21 | 11 | 832-token thinking (hypotheses: shebang `-S` conflict, "filesystem setup", permissions — none correct) → **absolute path** `/…/scripts/webfetch.py https://tangledgroup.com/` | ok — shebang runs `uv run --script`, "Installed 67 packages", scrapling output |
| 06:48:26 | 12 | Final summary. Thinking contains: *"After making it executable with `chmod +x`, running it from its directory worked"* | **false causal claim** (post-chmod run was turn 10, failed; the fix was the absolute path, turn 11) |
| 06:48:42 | — | User re-sends webfetch skill + URL (3rd prompt for the same task) | — |
| 06:48:45 | 13 | Re-confirms earlier result with a ✅ checklist (no re-fetch) | ok |

### 3.3 Qwen webfetch (`01a00967`) — same task, 14 s

| Time (UTC) | Turn | Action | Outcome |
|---|---|---|---|
| 07:09:45 | — | User: webfetch skill block + URL | — |
| 07:09:49 | 1 | `uv run --script <skill-root>/webfetch.py https://tangledgroup.com/` (timeout 120) — same first path guess as LFM | **exit 2 ENOENT** |
| 07:09:50 | 2 | `ls <skill dir>` | finds `scripts/` |
| 07:09:52 | 3 | `ls scripts/` | finds `webfetch.py` (13,127 B, 755) |
| 07:09:54 | 4 | `uv run --script <…>/scripts/webfetch.py https://tangledgroup.com/` (timeout 180) | ok — scrapling/markdownify |
| 07:09:59 | 5 | Clean structured summary; proactively flags the broken `#` footer anchors | done |

### 3.4 Qwen tzip+webfetch (`01a00956`) — same task, 31 s

`"tzip full activated"` (118 tokens, correct mode) → `ls` skill dir → `ls scripts/` → `uv run --script …/scripts/webfetch.py https://tangledgroup.com/` → tzip-full-style summary (fragments, `docs → insights` arrows, "Footer links (dummies, `#`)"). **Zero failed tool results.**

---

## 4. Catalog of LFM2.5-2.6B issues

### A. Action bias / task extraction (primary — both sessions)

1. **The URL in the prompt was treated as documentation, not as the task input.** In *both* LFM sessions the first response to the skill+URL message was a status/confirmation ("script ready…", "both skills active…") ending in a question. The fetch happened only after the operator typed the command or "Fetch a URL with webfetch". Qwen executed the fetch as its first action in both sessions.
2. **Task drift.** Before the nudge, the webfetch session self-selected `https://example.com` as a substitute (turn 6), reported "Script Status: Ready", and stopped — the requested URL was still in context the whole time.

### B. Skill-script invocation (primary — tzip session)

3. **Literal command copy from the skill doc.** SKILL.md's Usage shows `webfetch.py https://example.com`; LFM ran exactly that — bare name, no `./`, no path, no interpreter. In a non-interactive shell a bare name is resolved via PATH only → 127.
4. **A 7-turn, 5-failure recovery chain for a one-line fix:** 127 → `uv run webfetch.py` (uv also needs a script path) → `find /` (19.6 s) → `cd <dir> && webfetch.py` (127) → read 13 KB script → `chmod +x` (wrong) → identical `cd <dir> && webfetch.py` (127) → **absolute path** (works). The correct absolute path was *already in the `find` output* three turns earlier; the model still invoked the bare name.
5. **No PATH/relative-execution mental model.** The model's error model is "file missing or not executable" — a file-system model, not a shell-resolution model. It never once hypothesized `./webfetch.py`; `chmod +x` on an already-755 file was its "fix".
6. **Environment bypass.** In the webfetch session it ran `python3 scripts/webfetch.py` instead of the shebang/PEP 723 flow (`uv run --script`) that SKILL.md explicitly prescribes ("`uv run --script` resolves them automatically"). System python has no scrapling → the script silently fell back to the regex `html_to_md` requests path → degraded markdown (`text (url)` links, ~60 stray blank lines) vs the clean markdownify output Qwen (and LFM-tzip via shebang) got. Consequence is on the model's command choice, not the environment.

### C. Instruction fidelity

7. **`tzip` mode mis-parsed, twice.** The user message ended with the mode argument `full`; LFM answered "tzip **lite** activated" both times, rationalizing in its thinking that "no specific action was requested" — the argument *was* the action. Qwen: "tzip full activated" first try.
8. **Active tzip mode not applied.** The skill claims "ACTIVE EVERY RESPONSE until `tzip off`"; LFM's responses in that session are 378–862 tokens with bullets and menus — the opposite of pruning. Its final answer is also full prose, not tzip-full style.

### D. Reasoning / confabulation

9. **False causal claim in the thinking stream (tzip turn 12):** *"After making it executable with `chmod +x`, running it from its directory worked."* The post-`chmod` run (turn 10, same `cd && webfetch.py`) failed with 127; the run that worked (turn 11) used the absolute path. The user-facing text was accurate; the reasoning stream was not — consistent with the lisp-python finding that confabulated causality in thinking is a reliable marker of reasoning drift.
10. **Wrong hypothesis cascade (832 tokens of thinking before the fix):** shebang `-S` "conflict", "something wrong with how the filesystem is set up", permissions, "a different webfetch.py taking precedence" — none correct; the actual cause (bare name → PATH lookup) never stated.
11. **Contrast: file-path errors are handled correctly.** The webfetch session's ENOENT on `read <root>/webfetch.py` was recovered in one `find`+`ls` — the search-and-adapt pattern works for "file not at this path". The breakdown is specific to the "command not found" error class.

### E. Waste / verbosity

12. **862-token confirmation** to a one-word mode activation (Qwen: 118).
13. **Redundant re-read of SKILL.md** whose full text was in the immediately preceding user message (webfetch turn 1).
14. **Reads the entire 13 KB script before ever running it** (both sessions) — "understand, then act"; Qwen never reads it.
15. **`find /`** where `ls <skill dir>` (skill location is in the prompt) would do.
16. **Output-token ratio 3.4× / 7.2×** vs Qwen for identical tasks.

### F. What LFM2.5-2.6B got right

17. **Completion:** the page was fetched in both sessions; final summaries are factually correct about the content.
18. **No doom loop** (unlike the lisp-python session): the closest approach is the byte-identical `cd && webfetch.py` ×2 (turns 7/10) — it escaped on its own at the 8th attempt. Short tasks keep it out of the attractor regime.
19. **Correct use of the shebang when given the absolute path** (`env -S uv run --script` → uv installs 67 packages → scrapling path → clean output).
20. All 22 LFM requests HTTP 200 — no serving/quantization artifacts; failures are behavioral.

---

## 5. Head-to-head interpretation (LFM2.5-2.6B vs Qwen3.8-27B)

| Dimension | LFM2.5-2.6B | Qwen3.8-27B |
|---|---|---|
| Executes the given task on turn 1 | No (both sessions) | Yes (both sessions) |
| Tool-call record | 15 calls (12 bash, 3 read), 5 failed (3× `127`, 1 uv, 1 ENOENT read); same initial path guess as Qwen in webfetch | 7 bash calls, 1 failed (same initial path guess, exit 2), recovered with 2× `ls` |
| Error-recovery style | Long, wrong-diagnosis chains (find /, chmod, re-read) | Short, correct chains (2× `ls`) |
| Canonical invocation awareness | Literal doc copy; environment bypass | `uv run --script <abs path>` + explicit timeouts, both times |
| Mode/argument parsing (`tzip full`) | Wrong twice | Correct |
| Style compliance (tzip pruning) | Not applied | Applied |
| Per-token speed | ~104 tok/s | ~38 tok/s |
| Task wall time (tzip variant) | 140 s incl. 3 user nudges | 31 s, no nudges |
| Failure regime observed | Wasteful, re-prompt-dependent | Single clean error + recovery |

**Fairness caveats.** (1) The LFM tzip session received 5 user messages vs Qwen's 2 — part of the request-count gap is operator re-prompting; but *needing* the re-prompts is itself the measured behavior (Qwen never needed one). (2) Each model ran under its native sampling/thinking config (see §2) — this is a "shipped defaults" comparison, not a tuned-vs-tuned one. (3) The degraded fetch output in the LFM webfetch session is a direct consequence of the model's `python3` vs `uv` choice and is counted as model behavior. (4) Four sessions, one day, one task family, all short — these runs do not exercise the doom-loop regime documented for the same model in `misc/lisp-python/SLM_ANALYSIS.md`; the two regimes (wasteful-but-bounded vs frozen-loop) are both "first-move reliability" failures at different severities.

---

## 6. Bottom line

For short skill-execution tasks, LFM2.5-2.6B is **capable but first-move-unreliable**: it reaches the goal, but only after (a) an operator nudge to notice that the URL in the prompt *is* the task, and (b) a multi-turn, mis-diagnosed search for the one absolute path the skill directory contains. Against Qwen3.8-27B the gap is not doom-looping (absent here) but **task extraction, canonical-invocation knowledge, and instruction fidelity** — with 3.4–7.2× the output tokens and a degraded fetch path as the measurable cost.

Crucially, none of the observed failures requires model intelligence to prevent: the correct command is derivable by the harness from the skill directory + shebang, the task is explicit in the prompt, and the 127 error names the cause. The recommendation document maps each failure to a pi extension mechanism.
