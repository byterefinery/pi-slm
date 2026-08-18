# SLM Extension Specification

Reliability extension for Small Language Models.

## Feature 1:
  - On new session, simulate a short user/assistant dialogue before the first user request:
    1. system message (pi default, untouched)
    2. user: `What are available skills?` (simulated)
    3. assistant: short synthetic thinking/reasoning, then the answer — available skills as YAML framed in natural language (header line, list, closing "These are skills, not tools.") in the main content
    4. user: `What are available tools?` (simulated)
    5. assistant: short synthetic thinking/reasoning, then the answer — available tools as YAML framed in natural language (header line, list, closing "These are tools, not skills.") in the main content
    6. user: `How can a skill be used?` (simulated)
    7. assistant: short synthetic thinking/reasoning, then the answer — a static one-shot that teaches the whole skill-invocation pattern: the `/skill:<name> [task]` trigger, pi's `<skill>` block expansion, the task after the block, and two branches — A) the block names a script file → exactly one bash tool call (4-step recipe: match the usage line, map task wording only to shown flags, take the positional argument from the task, resolve the absolute script path from the `location` attribute, run PEP 723 scripts via `uv run --script`); B) the block names no script (a mode/instruction skill such as `tzip`) → follow it and reply as it tells you (e.g. the mode name), no bash call, never create or look for a script
    8. the first-ever user request
  - Skills YAML: name (single line text), description (single line text), reference file paths as absolute paths, script file paths as absolute paths. If the model supports reasoning, insert short synthetic reasoning (on the wire it must be visible on the assistant message, not dropped). Available skills come from `pi` internal API.
  - The first real user request is sent exactly as typed; the top-level `tools` field of the provider request is pi core's native function-calling definitions (active tools), not something the extension adds.
  - Tools YAML: name (single line), description (single line), whole function signature (`parameters` JSON schema as YAML, all params and types). Available tools come from `pi` internal API.
  - Reason: small language models forget skills and tools mentioned in the system message, so we remind them what they can use — inside the conversation itself, as an actually-answered question. The natural Q/A shape (full question, first-person answer, explicit "these are skills, not tools" framing) makes the model's later use of skills and tools more deterministic than a bare listing.
  - Clarifications (agreed during implementation):
    - tools entries carry the whole function signature: the tool's `parameters` JSON schema (all params and types) converted to YAML.
    - YAML strings are emitted as plain (unquoted) scalars wherever YAML-safe, to save tokens; double quotes only as a correctness fallback.
    - the asks are the full questions "What are available skills?" / "What are available tools?" / "How can a skill be used?"; the first two answers are natural language wrapping the YAML list (header line, explicit empty-case phrase, closing disambiguation line); the third is a static one-shot example of a skill invocation; thinking lines in first person ("I found N skills…"). The Q/A shape makes the SLM follow the reminder more deterministically — the listing is not stray system noise, and skills are never conflated with tools.

---

## Feature 1: Available skills and tools (simulated dialogue)

### Problem

Small language models reliably use skills and tools only while their names
are fresh in context. Pi announces them once, in the system prompt: skills
as an `<available_skills>` XML block (name/description/location) and tools
as one-line snippets (see `system-prompt.ts` / `formatSkillsForPrompt`).
SLMs frequently forget that content by the first real request — they
hallucinate capability names, skip available skills, or fumble tool
parameter formats.

### Solution

`src/slm.ts` hooks `before_agent_start` (fires after the user submits a
prompt, before the agent loop; the user message is not persisted yet). When
the session branch contains no user message yet — i.e. this is the first
request of a new session (resumed/forked sessions and all later prompts are
skipped) — the extension persists a simulated six-message dialogue (three
ask/answer pairs) at the start of the session, right after the startup
bookkeeping entries:

```
[model_change / thinking_level_change]
[custom_message available-skills]  "What are available skills?"  (simulated user, display: true)
[message assistant]                thinking + skills answer (natural language + YAML)
[custom_message available-tools]  "What are available tools?"    (simulated user, display: true)
[message assistant]                thinking + tools answer (natural language + YAML)
[custom_message skill-usage]       "How can a skill be used?"     (simulated user, display: true)
[message assistant]                thinking + skill-usage one-shot example (static text)
[user message]                     the first real request
```

The resulting context sent to the provider is exactly:

```
1. system message            (pi default, untouched)
2. user:      "What are available skills?"
3. assistant: <skills answer: header + YAML list + "These are skills, not tools.">  (thinking block kept in pi, see below)
4. user:      "What are available tools?"
5. assistant: <tools answer: header + YAML list + "These are tools, not skills.">
6. user:      "How can a skill be used?"
7. assistant: <skill-usage one-shot: /skill:<name> trigger, <skill> block expansion, run-the-matching-usage rule>
8. user:      first real user request
```

**Mechanics — three parts:**

1. *Detection* — at `before_agent_start` time the session branch contains
   no `message` entry with role `user` (startup bookkeeping entries such as
   `model_change` / `thinking_level_change` already exist, so entry count
   is not a reliable signal).
2. *Persistence* — the simulated user messages go in with
   `pi.sendMessage()` (custom messages, `display: true`, no `triggerTurn`;
   the non-streaming path appends synchronously to the session tree and to
   the agent state, and emits `message_start`/`message_end` so the TUI
   shows them live). The synthetic assistant messages are persisted with
   `SessionManager.appendMessage()` — the runtime object behind
   `ctx.sessionManager` is the full `SessionManager` instance (the public
   context type only exposes its read-only pick), so this is the same
   public method the core uses, one cast away.
3. *Live LLM context* — persisted entries alone are not enough for the
   current run: the provider is called with the agent's in-memory state,
   which `pi.sendMessage()` populates with the three asks but nothing adds
   assistant messages to (there is no official API for that). So the
   extension also subscribes to the official `context` event
   (`transformContext`), which fires on every provider call with the full
   `AgentMessage[]`: when the session is the one this process injected
   into, it re-inserts the three synthetic assistant messages right after
   each simulated user message if they are missing from the live state.
   For resumed/continued sessions the state is restored from the session
   file and already contains the full dialogue, the per-session check
   no-ops, and nothing is duplicated.

**Answer format (natural-language Q&A)** — the dialogue is
phrased the way a real exchange would be, so the SLM treats it as an
actually-answered question it can rely on: the simulated user asks the full
questions `What are available skills?` / `What are available tools?` /
`How can a skill be used?`, and each of the first two assistant answers is
natural language wrapping the YAML list — a header line (`Available skills
are:` / `Available tools are:`), the YAML entries, and a closing
disambiguation line (`These are skills, not tools.` / `These are tools,
not skills.`); the empty case is a plain phrase (`No available skills: []`
/ `No available tools: []`). The third answer is the static skill-usage
one-shot (see below). The framing makes later skill/tool use more
deterministic: the listing is not stray system noise, and skills are never
conflated with tools. The simulated user messages carry the custom types
`available-skills` / `available-tools` / `skill-usage`.
Note: with the closing line, the answer body is prose containing a YAML
list, not a single parseable YAML document — the list itself (from the
header line through the last entry) remains valid YAML.

**Skills YAML** — skill set = the same loaded skills the system prompt is
built from (`event.systemPromptOptions.skills`), minus skills with
`disable-model-invocation` (mirrors `formatSkillsForPrompt`). Per skill:
`name`, single-line `description`, `references` (absolute paths of all
files under the skill dir's `references/`, recursive, sorted), `scripts`
(same for `scripts/`).

**Tools YAML** — tool set = the active tools (`selectedTools`, fallback
`pi.getActiveTools()`). Per tool: `name`, single-line `description` (the
one-line system-prompt snippet from `toolSnippets`, fallback flattened
`ToolInfo.description`), and the whole function signature — the tool's
`parameters` JSON schema from `pi.getAllTools()` converted to YAML (all
params and types; `null` when the tool has no parameters).

**Skill-usage one-shot (third pair)** — the simulated user asks `How can
a skill be used?` (custom type `skill-usage`) and the assistant answers
with a static one-shot that teaches the whole skill-invocation pattern in
a single exchange: the user types `/skill:<name> [task]`, pi expands it
into a user message containing the skill's SKILL.md body in a `<skill
name=... location=...>` block (plus the task after the block), and the
assistant performs the task — never explains the skill. The answer opens
with a first check that splits into two branches: **A)** the block names
a script file (a usage code block with a script name like `example.py`)
→ exactly one bash tool call, built by a 4-step recipe (match the usage
line to the task, mapping task wording only to flags the skill shows;
take the positional argument from the task — a URL for webfetch, a search
query for websearch — and quote arguments containing spaces; resolve the
script's absolute path as `dirname(location)` + `scripts/`, never a bare
or relative name; run self-contained PEP 723 Python scripts via
`uv run --script <absolute path> <args>`, never directly, never with
`python`/`python3`); **B)** the block names no script — only instructions,
for example a communication mode with commands like `tzip on` / `tzip
lite` to follow, not run → reply as the block tells you (a second
mini one-shot shows the `tzip` block plus `lite` and `ultra` tasks
producing the one-line replies `tzip lite activated` /
`tzip ultra activated`), no bash call, no listing/searching the skill
dir, never creating or running a script the block does not name. Branch
A's example uses a deliberately generic `example` skill (a name that
collides with no real or popular skill) with two harmless usages (python
script with a positional arg; same script with a flag) so the SLM sees
that the right usage must be *identified* for the current task. The
message also states that a `<skill>` message is a task to perform, never
a question about the skill. The answer text (four-backtick fence: the
text itself contains triple-backtick fences):

````
A skill is invoked by the user with /skill:<name> [task], and pi expands it into a user message with the skill's SKILL.md body in a <skill> block:
```
<skill name="example" location="/home/user/project/.agents/skills/example/SKILL.md">

# example

## Usage

```bash
example.py DIR            # list the files in DIR
example.py DIR --json     # count the files in DIR, as JSON
```
</skill>
List the files in /data.
```
The text after the </skill> block is the task - a sentence, or bare arguments like a URL or a search query.

A user message with a <skill> block is a task to perform, never a question about the skill. I perform the task; I do not explain the skill. My first check: does the <skill> block name a script file (like example.py or my.sh)?

- No: it is a mode or instruction skill - I follow the block and reply as it tells me; my reply is the action (for example the mode name). No bash call.
- Yes: I run the script with exactly one bash tool call (recipe A below):

A) If it lists how to run a script - a script file name like example.py or my.sh in a usage code block (like the example above) - I produce exactly one bash tool call:
1. I pick the usage line in the block that matches the task, mapping task wording only to flags the skill shows ("raw HTML" -> --html, "save to X" -> --file X, "JSON" -> --json, "no sanitization" -> --no-ai-targeted, "force fetcher" -> --tool TOOL). I do not invent flags.
2. I take the positional argument from the task - a URL for webfetch, a search query for websearch - and quote arguments that contain spaces.
3. I resolve the script's absolute path: the skill dir is the dirname of the <skill> location attribute, and scripts live in <skill-dir>/scripts/ (location /home/user/project/.agents/skills/example/SKILL.md -> /home/user/project/.agents/skills/example/scripts/example.py). I never use a bare script name or a relative path.
4. I run self-contained Python scripts (PEP 723 header) with uv - never directly, never with python/python3:
```bash
uv run --script /home/user/project/.agents/skills/example/scripts/example.py '/data'
```

B) If it shows no script file name - only instructions to follow, for example a communication mode with commands like `tzip on` or `tzip lite` that I follow, not run - I reply as the block tells me:

user message:
<skill name="tzip" location="/home/user/project/.agents/skills/tzip/SKILL.md">
## Usage

- `tzip` / `tzip on` / `tzip lite` → Lite (default): drop filler, keep articles and full sentences
- `tzip full` → Drop articles, fragments OK
- `tzip ultra` → Abbreviate (DB, auth, config)
- `tzip off` → Deactivate token pruning

Reply with mode name (e.g. "tzip lite activated", "tzip deactivated")
</skill>
lite
my reply: tzip lite activated

same <skill> block, task after </skill> is `ultra`:
my reply: tzip ultra activated

Activating a mode is not a script run: a mode skill names no script, so there is nothing to run - my reply is the action (the mode name), and from then on I apply the mode to my replies. The <skill> block is all I need: no bash call, no listing or searching the skill dir, and I never read, create, or run a script the block does not name.
````

The `<skill>` block in the example matches pi's real expansion shape
(`_expandSkillCommand`: `<skill name="<name>" location="<SKILL.md
path>">` + the SKILL.md body with the frontmatter stripped + the task
after the block) except the `References are relative to <skill dir>.`
line is omitted to save tokens — the path lesson is taught through the
`location` attribute instead (skill dir = `dirname(location)`, scripts
under `<skill-dir>/scripts/`), which is what lets the SLM run real skill
scripts by absolute path without a prior `read`. The fictitious paths
(`/home/user/project/.agents/skills/...`) make the SLM substitute the
real paths from the incoming block rather than reuse the example's. The
4-step recipe in branch A is the GEPA-learned instruction distilled into
first person (see `optim/skills-usage/SKILL_USAGE_OPTIM.md`); the branch
B one-shot is a direct model of the `tzip` mode skill
(`.agents/skills-byterefinery/tzip`).

**Synthetic reasoning** — if the active model supports reasoning
(`ctx.model.reasoning === true`, e.g. `LiquidAI/LFM2.5-2.6B`), each
synthetic assistant message carries one short synthetic reasoning line as a
real `thinking` content block before the answer text block:

- skills: `I found <n> skills. I will check whether the task matches a
  description, and if so read that skill's SKILL.md and the reference files
  listed below.`
- tools: `I found <n> tools. I will pick the narrowest tool that fits the
  task.`
- skill-usage: `The user asks how a skill can be used. I will show one
  example: the user invokes a skill with /skill:<name>, pi puts the
  skill's SKILL.md body in a <skill name=... location=...> block of the
  user message, and the text after the block is the task. I do not
  explain the skill - I perform the task: if the block lists how to run a
  script (a script file name like example.py in a usage code block), I
  pick the usage line that matches the task (mapping task wording to the
  flags the skill shows), take the positional argument from the task,
  resolve the script's absolute path from the skill dir (dirname of the
  location attribute, scripts under <skill-dir>/scripts/), and run the
  PEP 723 script via \`uv run --script <absolute path> <args>\` with the
  bash tool. If the block shows no script file name - only instructions,
  for example a communication mode - I follow it and reply as it tells
  me: the mode name itself is my reply (activating a mode is not a
  script run; there is nothing to run).`

For the OpenAI Completions API (the llama.cpp server path) the block
additionally carries `thinkingSignature: "reasoning_content"`. Pi's
serializer (`openai-completions.js`) then emits the thinking text as a
`reasoning_content` field on the wire message — the standard
OpenAI-compatible convention for carrying reasoning in chat history (the
replay path pi documents for llama.cpp server) — while `content` stays
the natural-language answer: `reasoning_content` on the wire is
guaranteed, not dropped. For other APIs no signature is set: the block stays in the
session/TUI and is replayed or dropped by the provider's serializer as
usual (it is never sent with a signature the provider would reject).

Sample (session JSONL, abridged — the skills assistant entry):

```json
{
  "role": "assistant",
  "content": [
    { "type": "thinking",
      "thinking": "I found 1 skills. I will check whether the task matches a description, and if so read that skill's SKILL.md and the reference files listed below.",
      "thinkingSignature": "reasoning_content" },
    { "type": "text",
      "text": "Available skills are:\n  - name: demo-skill\n    description: A demo skill for sampling.\n    references:\n      - /tmp/.../demo-skill/references/ref.md\n    scripts: []\nThese are skills, not tools." }
  ],
  "api": "openai-completions",
  "provider": "llamacpp",
  "model": "LiquidAI/LFM2.5-2.6B",
  "usage": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
             "totalTokens": 0, "cost": { ...zeros... } },
  "stopReason": "stop"
}
```

Skills answer sample (the `text` block):

```
Available skills are:
  - name: slm-alpha
    description: "Alpha test skill: handles \"quotes\", colons: and #hashes for YAML escaping checks."
    references:
      - /tmp/.../slm-alpha/references/api.md
      - /tmp/.../slm-alpha/references/deep/guide.md
    scripts:
      - /tmp/.../slm-alpha/scripts/inner/helper.py
      - /tmp/.../slm-alpha/scripts/run.sh
  - name: slm-beta
    description: Beta test skill without references or scripts.
    references: []
    scripts: []
These are skills, not tools.
```

Tools answer sample (abridged to two tools):

```
Available tools are:
  - name: read
    description: Read file contents
    parameters:
      type: object
      required: [path]
      properties:
        path:
          type: string
          description: Path to the file to read (relative or absolute)
        offset:
          type: number
          description: Line number to start reading from (1-indexed)
        limit:
          type: number
          description: Maximum number of lines to read
  - name: edit
    description: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
    parameters:
      type: object
      required: [path, edits]
      properties:
        path:
          type: string
          description: Path to the file to edit (relative or absolute)
        edits:
          type: array
          items:
            type: object
            required: [oldText, newText]
            properties:
              oldText:
                type: string
                description: Exact text for one targeted replacement. ...
              newText:
                type: string
                description: Replacement text for this targeted edit.
          description: One or more targeted replacements. ...

These are tools, not skills.
```

**Wire behavior (what the provider actually receives)** — validated against
the real provider payload (`before_provider_request`). Net wire order for
the first request (llamacpp/LFM, reasoning on):

```
system  → user "What are available skills?" → assistant <skills answer> + reasoning_content
       → user "What are available tools?"  → assistant <tools answer> + reasoning_content
       → user "How can a skill be used?"   → assistant <skill-usage one-shot> + reasoning_content
       → user <first real request>
```

The OpenAI Completions serializer sends the assistant reply `content` as a
plain string (the natural-language answer); the synthetic reasoning rides in the
`reasoning_content` field (see above). The first real user request is sent
exactly as typed (OpenAI content-block form `[{type:"text",text:...}]`,
built by pi core). The request's top-level `tools` field is pi core's
native function-calling definitions for the active tools (`context.tools` →
`llmContext.tools` in the agent loop) — the extension never touches it;
it is sent on every request while tools are active, independently of the
synthetic tools YAML (validated in S9: identical with and without the
extension).

**Display** — the simulated user messages are shown in the TUI live
(custom-message styling, `display: true`). The synthetic assistant messages
have no live streaming events in the run they were injected in, so they
render when the session is loaded/reopened (they are regular assistant
message entries in the session file, with expandable thinking blocks).

**Token economy** — strings are emitted as plain (unquoted) YAML scalars
whenever that is safe (fallback to double-quoted + escaped only when the
content would otherwise change meaning: `: `, ` #`, leading indicator
chars, leading/trailing whitespace, newlines, or values that YAML would
re-type as number/bool/null/date). Empty sets are stated explicitly
(`No available skills: []` / `No available tools: []`) — the dialogue is
always complete.

### Validation

Test suite: `tests/feature1_test.py` (Python 3 + PyYAML only; no other
deps). Run: `python3 tests/feature1_test.py`.

Each scenario runs the real `pi` with `pi -e src/slm.ts --model
LiquidAI/LFM2.5-2.6B` in a random temp dir under `/tmp`, fully isolated via
a temp `HOME` (no global skills/settings/`~/.agents/skills` interference)
and a temp session dir. A second helper extension
(`tests/payload-logger.ts`, env-gated) logs every provider request payload
via the `before_provider_request` event, so the suite validates **both**
the session JSONL (dialogue structure, ordering, assistant message shape,
YAML validity/content/quoting) **and** the actual wire context (message
roles, contents, order — including the restored dialogue on continued
sessions). Fixture skills are planted in `<tmp>/.pi/skills/`: `slm-alpha`
(nested `references/` + `scripts/`, description full of YAML-hostile
chars), `slm-beta` (bare), `slm-hidden` (`disable-model-invocation: true`).

Scenarios:

| # | Scenario | Checks |
|---|----------|--------|
| S1 | skills listed correctly (default run) | exactly one ask + one assistant per side; order `askS → asst → askT → asst → user` with only bookkeeping entries before; assistant entry shape (role, `stopReason: stop`, zero usage, model metadata, one thinking + one text block, thinking carries the `reasoning_content` signature); skills listing parses as YAML (header/note lines are prose); listed skills == `{slm-alpha, slm-beta}` (hidden excluded); descriptions single-line and exact; reference/script paths absolute, correct, existing; names/descriptions are plain (unquoted) scalars; wire: `system → ask → skills YAML → ask → tools YAML → user`, assistant wire content a plain string, wire `reasoning_content` equals the session thinking text |
| S2 | tools listed correctly | same run: `tools` == `[read, bash, edit, write]` in pi's default order; descriptions == pi's one-line snippets, emitted unquoted; each tool carries the full `parameters` JSON-schema-as-YAML signature (`type`, `required`, per-param `type`; nested `array`/`items` for `edit` verified exactly) |
| S3 | restricted tools (`-t read,grep`) | tools listing follows the active set (`[read, grep]` with signatures); skills listing unaffected; wire consistent |
| S4 | no skills (`--no-skills`) | assistant skills text is exactly `No available skills: []`; thinking says `I found 0 skills`; full tools listing still correct |
| S5 | no tools (`-nt`) | assistant tools text is exactly `No available tools: []`; full skills listing still correct |
| S6 | repeated prompts (`pi -c` continued) | dialogue injected exactly once and sits before both user messages; no re-injection around the second prompt; the continued run's wire context starts with the restored dialogue, then the restored history (first request + real reply), then the new prompt; restored assistant texts identical to run 1 |
| S7 | non-reasoning model (`reasoning: false`) | both assistant messages present and correct but with no thinking block; no `reasoning_content` fields on the wire; everything else identical |
| S8 | JSON mode (`--mode json`) | `message_start` events for both asks are emitted `display: true` with the exact ask texts, in dialogue order (ask → ask → user) — i.e. the asks are visible in the TUI; wire consistent |
| S9 | user message verbatim + tools field from pi core | first user request (`list tools`) sent verbatim in both an extension run and a baseline run without the extension; the request's top-level `tools` field (4 built-in function definitions with name + JSON-schema parameters) is identical in both runs — the extension never touches it |

Status: the suite still encodes an earlier reminder format (ask texts,
`slm-skills`/`slm-tools` custom types, `scanned …` thinking prefixes,
exact `skills: []`/`tools: []` bodies in S4/S5, and whole-body
`yaml.safe_load` in S1/S2 — the answer body is no longer a single YAML
document), so it must be synced to the natural-language Q&A format
before it passes again. The skill-usage pair (this change) is not covered
by the suite at all: S1/S6/S8-style checks would need the third ask
(`skill-usage`) + third assistant entry, the `prompt_index` shifted from
5 to 7, and the wire history expectations extended by one pair.

### Live test: real skills in a random temp dir

The dialogue is also validated end-to-end against real skills: for each
skill a random temp dir is created under `/tmp`
(`mktemp -d /tmp/slm-fN-<skill>-XXXXXX`), the skill is copied into
`<root>/work/.agents/skills/<skill>/` (pi's project skill dir), an
isolated temp `HOME` is used (minimal `models.json` with only the LFM
model, so no global skills/settings leak in), the uv cache is shared via
`UV_CACHE_DIR` (never copied per run — `/tmp` is a 1M-inode tmpfs), and
the run is `pi --offline -a -e src/slm.ts -e tests/payload-logger.ts
--model LiquidAI/LFM2.5-2.6B -p "/skill:<skill> <task>"` in
`<root>/work`. The `/skill:` expansion is pi core's (the user message in
the session file is the expanded `<skill>` block + task — the extension
touches neither), so this also validates that the one-shot's example
matches the real block shape. Because the 2.6B model's tool-call
behavior is bimodal (see below), live runs are complemented by **wire
replays**: `optim/skills-usage/replay3.py` replays the three example
invocations against the seed session (`seed.json`), and
`replay_payload.py` / `replay_tzip.py` replay the *exact* wire payload of
a finished live run (payloads.jsonl, candidate answer swapped in at the
skill-usage slot) with fresh session-affinity keys — isolating the
message from live-run noise (uv installs, tool loops, server state).

Results (model `LiquidAI/LFM2.5-2.6B`, pi 0.84.2; replay scores are
"perfect bash tool call" — single call, `uv run --script` (or the
script's own executable shebang), absolute script path, correct task
argument):

| Prompt (after the dialogue) | Replay (n=5 each) | Live pi run |
|---|---|---|
| `/skill:webfetch fetch https://tangledgroup.com/ and summarize it` | **5/5** | **success** — single `uv run --script <root>/work/.agents/skills/webfetch/scripts/webfetch.py "https://tangledgroup.com"` (one round: bare-name misfire first, then the correct call after a `which uv` check), real page fetched, answer summarizes the live content (judge: PASS). |
| `/skill:websearch look for tangled group repos` | **5/5** | **success** — single `bash` call of the script with the quoted query `"tangled group repos"` (absolute path, one run; another run used `python3` + relative path, hit the missing-dependency traceback, and improvised with curl/GitHub API — judge: FAIL on call form, but the model did act, not explain). |
| `/skill:webfetch https://tangledgroup.com/` (bare URL) | **5/5** in the seed context and in the exact-context replay of a failed live run (`replay_payload.py`) | unstable — when the live run fails, the model re-answers the preceding meta question `How can a skill be used?` (0 tool calls); the same wire payload replayed moments later scores 5/5. The difference tracks the shared inference endpoint's state (see below), not the extension. |
| `/skill:tzip <lite|full|ultra|off>` (mode skill, no scripts) | activation (mode-name reply, no tools): **lite 3/3, off 2/2, ultra 2/3, full 0/3** in the last window | **no tool calls at all** (the earlier "hallucinate and create `tzip.py`" behavior — 6–19 calls per level — is gone); `lite`/`ultra` reply with the mode name ("tzip lite activated"), `off` almost ("tzip off activated" instead of canonical "tzip deactivated"), `full` still meta-explains. Content replies after activation are correct but mostly unpruned (2.6B does not hold a style regime). |

Judge (per the user's setup): `Qwen/Qwen3.8-27B` via
`pi --no-session --no-tools -p <judge prompt>` (prompts in
`/tmp/slm-f9-logs/judge/`, harness `slm-f9-judge.py`). Final-round
verdicts: webfetch-verb **correct-call/no-explain/grounded PASS**;
tzip-**lite** **activation/no-tools/style/accuracy PASS**; tzip-ultra
activation+no-tools PASS, style FAIL (mode not applied to the content
reply); tzip-off activation FAIL (non-canonical mode name), style PASS;
tzip-full activation FAIL (claims a `tzip.py` script exists in the reply
without calling it), style FAIL. Accuracy PASS in every tzip case.

What the live runs confirm: the three-pair dialogue sits at the head of
the session and of the wire context (system → 3×(ask, assistant with
`reasoning_content`) → user request), the skills YAML lists the real
skill with its absolute `scripts/` path, and when the SLM engages the
one-shot pattern it resolves the script exactly as taught — absolute
path under the skill dir (`dirname of location`) + `scripts/`, run via
`uv run --script`, task arguments appended and quoted.

**Reliability notes (2.6B + shared endpoint).** Tool-call behavior for
the 2.6B model is bimodal and the shared inference endpoint
(`oai.tangledgroup.com`) drifts between states: the same wire payload
scores 5/5 in one window and 0/5 in another (observed for the bare-URL
invocation), while explicit-verb tasks stay ~100% across windows. The
endpoint also returns an *empty* response body when the
`x-session-affinity` header is absent (the pi client always sends it). 
Practical consequence: explicit task phrasing (verb + argument) is the
reliable form for script skills; bare-argument tasks and mode-skill
activation replies (exact mode-name line, sustained style) are the known
weak spots of the 2.6B model and of the current one-shot.

### Notes

- New-session detection uses "no user message in the session branch" at
  `before_agent_start` time — not entry count, because startup
  bookkeeping entries (`model_change`, `thinking_level_change`) already
  exist when `--model` is passed.
- The dialogue is persisted once per session (three `custom_message`
  entries + three assistant `message` entries) and stays at the head of
  the context for all later turns. Resumed/continued/forked sessions restore it from
  the session file — the `context` handler then no-ops (per-session
  guard), so nothing is duplicated. Mid-session tool/skill changes
  (`/tools`, `/reload`) do not rewrite it.
- If a compaction drops the dialogue entries from the kept context, the
  `context` handler no-ops (the asks are gone from the live state) and the
  reminder is gone too — same semantics as any compacted history.
- The `context` event is the official context-rewrite hook
  (`transformContext` in pi-agent-core, applied on every provider call);
  the extension uses it because there is no official way to push
  assistant-role messages into the agent's live state — only
  `pi.sendMessage()` (custom/user-role) writes there directly.
- `ctx.sessionManager` is typed as a read-only `Pick` of `SessionManager`,
  but the runtime object is the full instance; `appendMessage()` is the
  same public class method the core uses to persist messages.
- The request's top-level `tools` field (native function-calling
  definitions) and the OpenAI content-block form of user messages are pi
  core behavior — the extension only rewrites the `messages` array via the
  `context` event (S9 validates the `tools` field is byte-identical with
  and without the extension).
- If a session is switched to a different model mid-session, pi's
  cross-model message transform degrades the synthetic thinking block to
  plain text (standard pi behavior for foreign-model thinking); the YAML
  content is unaffected.
- `references`/`scripts` follow the Agent Skills standard layout
  (`<skill dir>/references/`, `<skill dir>/scripts/`), scanned recursively;
  other file kinds are not listed.
- Skills with `disable-model-invocation: true` are excluded, matching what
  the system prompt shows the model.
- The JSON→YAML emitter is dependency-free (single-file constraint): block
  style, flow style for scalar arrays, plain scalars when safe.
- Works in all run modes (validated via `-p` and `--mode json`; TUI/RPC
  share the same hook path).
- The skill-usage answer is static (no live data): the `example` skill
  name collides with no real or popular skill, and the paths inside the
  one-shot (`/home/user/project/.agents/skills/...`) are fictitious —
  the SLM must substitute the real skill dir from the incoming `<skill>`
  block, not reuse the example's paths. The branch-B example uses the
  real `tzip` skill shape on purpose (it is the mode-skill class the
  SLM must learn to handle without scripts).
- The one-shot's `<skill>` block mirrors pi's real expansion shape
  (`_expandSkillCommand` in pi 0.84.2) — `<skill name=... location=...>`
  tag, SKILL.md body with frontmatter stripped, task after the block —
  with the `References are relative to <skill dir>.` line omitted to
  save tokens (the path lesson is carried by the `location` attribute);
  the nested ``` fences are authentic (real SKILL.md bodies contain
  them).
- Skill usage with a 2.6B model is task-phrasing sensitive: explicit
  tasks (verb + argument) reliably trigger the script call; bare
  arguments after the `<skill>` block are performed in the wire-replay
  harness but flake in live runs (meta-explanation); mode skills (no
  scripts) are the other branch — the one-shot's branch B + the
  anti-script-creation rule removed the `tzip.py` hallucination
  entirely, leaving the exact mode-name reply and sustained style as
  the remaining 2.6B weak spots (see Live test).

---
