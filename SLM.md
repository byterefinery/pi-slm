# SLM Extension Specification

Reliability extension for Small Language Models.

## Feature 1:
  - On new session, simulate a short user/assistant dialogue before the first user request:
    1. system message (pi default, untouched)
    2. user: `What are available skills?` (simulated)
    3. assistant: short synthetic thinking/reasoning, then the answer — available skills as YAML framed in natural language (header line, list, closing "These are skills, not tools.") in the main content
    4. user: `What are available tools?` (simulated)
    5. assistant: short synthetic thinking/reasoning, then the answer — available tools as YAML framed in natural language (header line, list, closing "These are tools, not skills.") in the main content
    6. user: `How does skill system work? When a skill block is in my latest message, what do I do?` (simulated)
    7. assistant: short synthetic thinking/reasoning, then the `<skill>` block contract: a skill invocation is a `<skill> SKILL BODY </skill>` block, the text after the closing tag is the argument for this invocation, the block in the latest message is the active instruction, and the reply is exactly what the skill's Usage rules require for the current argument
    8. user: a real `<skill>` block (the `example` skill, rooted at `<cwd>/.agents/skills/example`), no argument (simulated)
    9. assistant: the exact fixed reply the skill requires: `This is an example skill.`
    10. user: same `<skill>` block + `Hello` (simulated)
    11. assistant: synthetic `read` tool call (the skill's `references/03-hello.md`, absolute path under the block's skill dir)
    12. tool result: the reference file content (says to reply exactly `world`)
    13. assistant: `world`
    14. user: same `<skill>` block + `Hi` (simulated)
    15. assistant: synthetic `bash` tool call (`bash <skill-dir>/scripts/example.sh Hi`)
    16. tool result: the script output
    17. assistant: the script output report
    18. the first-ever user request
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
[custom_message available-skills]    "What are available skills?"  (simulated user, display: true)
[message assistant]                  thinking + skills answer (natural language + YAML)
[custom_message available-tools]     "What are available tools?"   (simulated user, display: true)
[message assistant]                  thinking + tools answer (natural language + YAML)
[custom_message skill-system]        "How does skill system work? ..." (simulated user, display: true)
[message assistant]                  thinking + <skill> block contract (static text)
[custom_message skill-example-plain] <skill> block, no argument (simulated user, display: false)
[message assistant]                  thinking + "This is an example skill."
[custom_message skill-example-hello] <skill> block + "Hello"       (simulated user, display: false)
[message assistant]                  thinking + read toolCall (references/03-hello.md), stopReason: toolUse
[message toolResult]                 the reference file content
[message assistant]                  thinking + "world"
[custom_message skill-example-script] <skill> block + "Hi"        (simulated user, display: false)
[message assistant]                  thinking + bash toolCall (scripts/example.sh Hi), stopReason: toolUse
[message toolResult]                 the script output
[message assistant]                  thinking + the script output report
[user message]                       the first real request
```

The resulting context sent to the provider is exactly:

```
1.  system message            (pi default, untouched)
2.  user:      "What are available skills?"
3.  assistant: <skills answer: header + YAML list + "These are skills, not tools.">  (thinking block kept in pi, see below)
4.  user:      "What are available tools?"
5.  assistant: <tools answer: header + YAML list + "These are tools, not skills.">
6.  user:      "How does skill system work? When a skill block is in my latest message, what do I do?"
7.  assistant: <the <skill> block contract: block = active instruction, text after the block = current argument, reply exactly what the Usage rules require>
8.  user:      <skill> block (example skill, <cwd>/.agents/skills/example), no argument
9.  assistant: "This is an example skill."
10. user:      <skill> block + "Hello"
11. assistant: read toolCall (absolute path <skill-dir>/references/03-hello.md)
12. tool:      the reference file content
13. assistant: "world"
14. user:      <skill> block + "Hi"
15. assistant: bash toolCall (bash <skill-dir>/scripts/example.sh Hi)
16. tool:      the script output
17. assistant: the script output report
18. user:      first real user request
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
   into, it re-inserts the missing synthetic replies (assistant messages
   and tool results) right after each simulated user message — it matches
   the expected reply sequence against what already follows the ask and
   splices in only the missing tail.
   For resumed/continued sessions the state is restored from the session
   file and already contains the full dialogue, the per-session check
   no-ops, and nothing is duplicated.

**Answer format (natural-language Q&A)** — the dialogue is
phrased the way a real exchange would be, so the SLM treats it as an
actually-answered question it can rely on: the simulated user asks the full
questions `What are available skills?` / `What are available tools?` /
`How does skill system work? When a skill block is in my latest message,
what do I do?`, and each of the first two assistant answers is
natural language wrapping the YAML list — a header line (`Available skills
are:` / `Available tools are:`), the YAML entries, and a closing
disambiguation line (`These are skills, not tools.` / `These are tools,
not skills.`); the empty case is a plain phrase (`No available skills: []`
/ `No available tools: []`). The third answer is the static `<skill>`
block contract (see below), followed by a few-shot of three real skill
invocations covering the `example` skill's three Usage branches. The
framing makes later skill/tool use more deterministic: the listing is not
stray system noise, and skills are never conflated with tools. The
simulated user messages carry the custom types `available-skills` /
`available-tools` / `skill-system` / `skill-example-plain` /
`skill-example-hello` / `skill-example-script` (the three few-shot block
messages use `display: false` — they stay in the session file and the LLM
context without flooding the TUI).
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

**Skill-system contract + few-shot (third pair and the three invocations)** —
replaces the earlier static skill-usage one-shot (which taught the pattern
in prose; the 2.6B model failed to abstract the path-substitution recipe
out of it and reused the example's fictitious paths). The simulated user
now asks `How does skill system work? When a skill block is in my latest
message, what do I do?` (custom type `skill-system`) and the assistant
answers with the static `<skill>` block contract: a skill invocation is a
`<skill> SKILL BODY </skill>` block with the user message after it — the
block carries the skill's instructions, the text after the closing tag is
the argument for this invocation, the block in the latest message is the
active instruction (earlier questions are ignored), and the reply is
exactly what the skill's Usage rules require for the current argument
(exact fixed wording; the argument or the default wording, never an
example from inside the skill).

The dialogue then continues as a few-shot of three real invocations of
the `example` skill (`.agents/skills-byterefinery/example` in this repo),
covering its three Usage branches:

- **no argument** → the assistant replies exactly `This is an example skill.`
- **`Hello`** → the assistant makes a synthetic `read` tool call
  (content block `toolCall`, `stopReason: "toolUse"`) for
  `<skill-dir>/references/03-hello.md`, a synthetic tool result carries
  the file content, and the assistant replies exactly `world`
- **`Hi`** (any other text) → the assistant makes a synthetic `bash`
  tool call `bash <skill-dir>/scripts/example.sh Hi`, a synthetic tool
  result carries `This is example.sh output.`, and the assistant reports
  it in a code block

The three `<skill>` blocks are the exact expansion shape of pi's
`_expandSkillCommand` (tag with `name`/`location` attributes, the
`References are relative to <skill dir>.` line, the SKILL.md body with
frontmatter stripped, the argument after the block separated by a single
newline) — but rooted at the session's current working directory:
`<cwd>/.agents/skills/example/...` (pi's standard project skill
location). The few-shot's absolute paths (the block's `location`
attribute, the `read` path, the `bash` command) are therefore the
session's real absolute paths, so the SLM derives the absolute paths of a
real incoming block by the same rule it just saw applied — skill dir =
`dirname(location)`, references/scripts under it. With a fictitious
example path the 2.6B model failed this substitution (it reused the
example's path verbatim and tried to create the missing script at the
example's location); with the cwd-rooted path the same invocation
replays 5/5 perfect bash tool calls.

The contract answer text (four-backtick fence: the text contains
backticks):

````
A skill invocation is a `<skill> SKILL BODY </skill>` block with the user message after it: the block carries the skill's instructions, and the text after the closing tag is the argument for this invocation. I will treat the skill block in my latest message as the active instruction and ignore earlier questions. I will use the argument after the block as the current argument, not an example from inside the skill. If there is no argument, I will use the skill's required default wording. I will follow the skill's Usage rules for the current argument and reply text only with the exact required confirmation: skill name, current argument or default wording, and required wording.
````

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

The few-shot is a verbatim transcript (user blocks, assistant answers,
reasoning, tool calls, tool results) of one session driving the real
`example` skill, rewritten onto the cwd-rooted paths. The synthetic
tool-call assistant messages carry `stopReason: "toolUse"`; the tool
results use pi's `toolResult` role (`toolCallId`/`toolName`/`content`/
`isError`) and are serialized on the wire as `role: "tool"` messages
paired with their tool calls.

**Synthetic reasoning** — if the active model supports reasoning
(`ctx.model.reasoning === true`, e.g. `LiquidAI/LFM2.5-2.6B`), each
synthetic assistant message carries one short synthetic reasoning line as a
real `thinking` content block before the answer text block:

- skills: `I found <n> skills. I will check whether the task matches a
  description, and if so read that skill's SKILL.md and the reference files
  listed below.`
- tools: `I found <n> tools. I will pick the narrowest tool that fits the
  task.`
- skill-system: `The user wants the rule for a skill block in my latest
  message. I will explain the \`<skill> SKILL BODY </skill> [USER
  MESSAGE]\` shape: the block is the active instruction, the text after
  the block is the current argument, and the reply is the exact
  confirmation the skill requires.`
- few-shot invocations: one short first-person line per synthetic
  assistant message, each stating the Usage rule being applied and the
  exact action, e.g. `The user invoked the example skill with "Hi". That
  is any other text, so I will pass it as CLI parameters to
  scripts/example.sh and report the output. Let me run the script.` (the
  full lines live in `src/slm.ts`, constants `SKILLSYS_THINKING`,
  `SKILL_EXAMPLE_THINKING`, `SKILL_HELLO_*`, `SKILL_SCRIPT_*`)

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
system  → user "What are available skills?"  → assistant <skills answer> + reasoning_content
        → user "What are available tools?"   → assistant <tools answer> + reasoning_content
        → user "How does skill system work? ..." → assistant <contract> + reasoning_content
        → user <skill block, no argument>     → assistant "This is an example skill." + reasoning_content
        → user <skill block + "Hello">        → assistant read toolCall + reasoning_content
                                              → tool (reference file content)
                                              → assistant "world" + reasoning_content
        → user <skill block + "Hi">           → assistant bash toolCall + reasoning_content
                                              → tool (script output)
                                              → assistant <output report> + reasoning_content
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
before it passes again. The skill-system contract + few-shot (this
change) is not covered by the suite at all: S1/S6/S8-style checks would
need the `skill-system` ask + contract assistant entry plus the three
few-shot segments (each: a `skill-example-*` custom entry, then the
assistant reply — and for `skill-example-hello`/`skill-example-script`,
the toolCall assistant + toolResult + final assistant entries), and the
wire history expectations extended accordingly (the first real request
now sits at wire index 17, not 7).

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
skill-system contract slot) with fresh session-affinity keys — isolating the
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
drifts between states: the same wire payload
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
- The skill-system contract answer and the three few-shot invocations
  are static (no live data). The few-shot's `<skill>` blocks are rooted
  at the session's cwd (`<cwd>/.agents/skills/example`) — pi's standard
  project skill location — so the absolute paths inside the few-shot
  (the `location` attribute, the `read` path, the `bash` command) are
  the session's real absolute paths, and the SLM derives the real
  script/reference paths from an incoming block by the exact rule it
  just saw applied. A fictitious example path (`/home/user/project/...`,
  `/home/dev/...`) made the 2.6B model reuse the example's paths
  verbatim on live invocations (it even tried to create the missing
  script at the example's location); the cwd-rooted paths removed that
  failure mode (wire replay 5/5 perfect bash tool calls; live runs use
  the real absolute path).
- The few-shot's `<skill>` blocks mirror pi's real expansion shape
  (`_expandSkillCommand` in pi 0.84.2) — `<skill name=... location=...>`
  tag, the `References are relative to <skill dir>.` line, SKILL.md body
  with frontmatter stripped, argument after the block (single newline);
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
