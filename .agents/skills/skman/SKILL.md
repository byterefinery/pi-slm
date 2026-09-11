---
name: skman
description: Introduces the Agent Skills System — a standardized, lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. Use for scaffolding, validating, and inspecting agent skills (SKILL.md files and other skills' files and directories).
license: Apache-2.0
compatibility: Requires uv on PATH. PyYAML is declared in the PEP 723 header and auto-resolved by uv — no pip, no local venv.
metadata:
  tags:
    - meta
    - agent
    - skill
    - skills
    - agent skill system
---

# skman

Tools and guidelines for creating, validating, and managing agent skills.

## Overview

Agent Skills are a lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. A skill is a directory containing a `SKILL.md` — frontmatter metadata plus concise instructions — optionally with scripts, references, and assets, giving agents new expertise on demand without bloating the context window.

`skman` is the skill for creating, validating, and managing agent skills. It provides four functionalities:

- **`create`** — Scaffold a skill directory (SKILL.md, optional scripts and references)
- **`validate`** — Check format compliance (frontmatter, naming, structure)
- **`info`** — Inspect frontmatter, body stats, heading hierarchy
- **`generate`** — Regenerate the repo README.md with per-category skills tables and statistics (core `skills` dir plus all sibling `skills-*` dirs)

## Usage

```bash
# Quick reference — details per subcommand below
skman.py create <name> "<description>" [--with-scripts] [-o ./skills]
skman.py validate ./my-skill [--strict] # single skill
skman.py validate .agents/skills        # whole collection
skman.py info ./my-skill [--json]
skman.py generate
skman.py --help                         # or: skman.py <subcommand> --help
```

### Scaffold New Skill

```bash
skman.py create my-skill "Extracts text from PDF files"
skman.py create demo-skill "Dummy example skill" --version 2.4.1
skman.py create numpy "NumPy skill" --url https://github.com/numpy/numpy/releases/tag/v1.26.0
skman.py create my-skill "Desc" --with-scripts --with-references
skman.py create my-skill "Desc" --with-scripts --lang bash   # shell wrapper
```

Validates name and description (format, length, no `:`) before creating files. `--url` extracts name/version from GitHub, PyPI, npm, crates.io, GitLab, RubyGems URLs; the positional `name` and an explicit `--version` take precedence. LLM fallback: set `SKMAN_LLM_RESPONSE='{"version": "X.Y.Z"}'` env var. Note `--url` only extracts metadata — for code repositories, run the Creating a skill from a repository workflow first (clone the repo, then study it).

### Regenerate README

```bash
skman.py generate
skman.py generate --only python
```

Scans the `--skills-dir` collection (default `.agents/skills`) plus every sibling `skills`/`skills-*` collection in the same parent directory. Each non-empty collection gets a table (first `## Skills Table`, siblings `## <Category> Skills`) and a per-category `## Statistics` summary. Tables and statistics sit between their own HTML comment markers (`<!-- SKMAN:TABLE:<label> -->`, `<!-- SKMAN:STATS -->`), which `generate` replaces in place; new `skills-*` directories get marker blocks added automatically. `--only` (label, dir basename, or path) refreshes just that table; statistics are always recomputed.

## Skill Format

A skill is a directory containing a `SKILL.md` file. Everything else is optional.

### Directory Layout

```
<skill-name>/
├── SKILL.md              # Required: frontmatter + instructions
├── scripts/              # Optional: helper scripts (executed, not loaded into context)
│   └── <skill-name>.py   # Default: Python with PEP 723 shebang (needs uv)
│   ├── <skill-name>.sh   # Shell mode: bash wrapper (needs python)
│   └── _<skill-name>.py  # Shell mode: Python impl, no PyPI deps
├── references/           # Optional: detailed docs loaded on demand (numbered prefix)
│   └── 01-topic.md
├── assets/               # Optional: templates, images, data files, schemas
│   └── template.yaml
```

### Frontmatter Fields

| Field | Required | Rules |
|---|---|---|
| `name` | Yes | 1-64 chars, lowercase letters (including Unicode/i18n), 0-9, hyphens; no leading/trailing/consecutive hyphens; must match directory name exactly; meta skills without versions use plain name (e.g., `skman`) |
| `description` | Yes | Non-empty, max 1024 chars, third-person, must not contain XML/HTML tags (`<tag>`) or a `:` character |
| `license` | No | License name or reference to a bundled license file (e.g., `Apache-2.0`, `Proprietary. LICENSE.txt has complete terms`) |
| `compatibility` | No | Max 500 chars. Environment requirements (product, system packages, network); include only when the skill has specific needs |
| `metadata` | No | Optional object. May contain `tags` (string array, e.g., `["meta", "devops"]`). Validator warns if not a mapping or `tags` is not a string array.

All top-level text fields (`name`, `description`, `license`, `compatibility`, any unknown string field) must not contain `:` — a YAML structural character that breaks unquoted scalars and naive frontmatter parsers. Rephrase or use `;` instead; the validator errors on any `:` in a text field.

### Frontmatter Template

```yaml
---
name: my-skill
description: What this skill does and when to use it. Be specific.
license: Apache-2.0
compatibility: Requires Python 3.11+ and uv
---
```

## Creating a New Skill

Follow these steps in order:

1. **Choose a name** — lowercase, hyphens, numbers only (e.g., `pdf-processing`, `git-8-20-0`). No leading/trailing/consecutive hyphens.

2. **Write the frontmatter** — at minimum `name` and `description`. `name` must match the directory name exactly. The description determines when the agent loads the skill; make it specific. Keep text fields `:`-free (write "Use when X", not "Use when: X").

3. **Write the body** — concise instructions, under 5000 tokens. Must start with a level-1 heading matching `# <name>` or `# <name> <version>`. Structure:
   - `# <name>` (e.g., `# skman`) or `# <name> <version>` (e.g., `# demo-skill 2.4.1`)
   - `## Overview` — what it does
   - `## Usage` — Optional: how to use it with examples
   - `## Gotchas` — Optional: The most useful part of teaching a skill is listing its hidden traps. Instead of vague advice, provide specific rules that stop the agent from making predictable, common-sense mistakes in that specific environment.
   - `## References` — Optional: Provides on-demand reference material for agents. Always use a bulleted list, never a table:
     ```
     ## References

     - [01-core-expressions](references/01-core-expressions.md) — Symbols, expressions, numbers
     ```
     Each line: link to the file, a dash, brief topic summary. Local `references/NN-topic.md` files and external URLs both work.

4. **Create scripts** — only when the user explicitly requests them. Never assume a language — ask or wait for a suggestion. Conventions (default `scripts/<name>.py` with PEP 723 shebang; shell mode `scripts/<name>.sh` + `_<name>.py` for no-PyPI-deps) are in Scripting below. Scripts are **executed**, not loaded into context; include `--help` at every level. Scaffold with `--with-scripts`.

5. **Validate** — run the validation script:
   ```bash
   skman.py validate <path-to-skill>
   ```

### Manual Creation

When writing files directly, the same rules apply: the directory is named after the skill (or `<skill-name>-<version>`), `SKILL.md` sits at its root, and the frontmatter `name` plus the first H1 match the directory.

## Creating a skill from a repository

Use this workflow when creating or updating a skill whose source material is a code repository. Recognition — the user points at a project, package, or repo URL (GitHub, GitLab, Bitbucket, PyPI, npm, …) and asks for a skill around it: "make a skill for ripgrep", "create a skill from <url>", "update the <name> skill from the latest release". When that is the case, clone the repo into a temporary directory and study the local copy:

```bash
git clone --depth 1 <url> /tmp/<name>
```

Never fetch individual files over the network (slow and rate-limited); a single clone gives the full tree. Clean up the temp directory when done.

Then study the repo in this order. Documentation is the primary source; source code is the fallback — documented intent maps directly onto skill content and costs far fewer tokens than raw code.

1. **Look for documentation first** — before touching source code, find `.md`, `.rst`, and `.txt` files in doc-like directories (`docs/`, `doc/`, `documentation/`, `manual/`, …) and root-level files (`README*`, `INSTALL*`, `CHANGELOG*`, `CONTRIBUTING*`). Exact `find` commands in [02-repo-analysis](references/02-repo-analysis.md).
2. **Docs found** — mine the doc tree for the skill body: what it does, usage, commands and options, configuration, workflows, known pitfalls. Open source code only to verify specifics the docs leave ambiguous (exact flags, version behavior).
3. **No docs** — analyze the whole repo instead: entry points, CLI/argument definitions, public API, tests, config schemas. Extract the same material from the code itself.
4. **Write and validate** — follow Creating a New Skill above (naming, frontmatter, body), then run `skman.py validate <path-to-skill>`.

## Editing a Skill

Common operations:

- **Update description** — edit the frontmatter; this is what agents see in the system prompt
- **Split long content** — move sections >100 lines into `references/NN-topic.md`, link from SKILL.md
- **Add a script** — place in `scripts/` with the skill's name as base name
- **Restructure references** — keep references one level deep; all should link directly from SKILL.md

## Validation

Checks performed:
- Frontmatter: present, valid YAML (errors on parse failure or non-mapping), no duplicate top-level keys, no `:` in text fields (errors)
- Name format (case, characters, length, hyphen rules); name vs directory basename consistency (warns on mismatch)
- Description presence, length, absence of XML/HTML tags (errors)
- `metadata` structure (warns if not a mapping; `tags` must be a string array)
- Body: starts with a level-1 heading; token estimation warning (>5000 tokens)
- H1 format `# <name>` or `# <name> <version>` (errors on mismatch)
- `## Overview` presence (warns if missing); `## Usage`, `## Gotchas`, `## References` are truly optional (never warn)
- Scripts: entry scripts `<name>.py` and `<name>.sh` must be executable (warns); body must reference `<name>.sh`, not `./<name>.sh`, outside fenced code blocks (warns)

## Best Practices

### Conciseness
- Context window is shared — every token competes with conversation history
- Default assumption: the model already knows basics (what PDFs are, how libraries work)
- Challenge each paragraph: "Does this justify its token cost?"

### Scripting
- **Default entry point is `scripts/<name>.py`** — Python with PEP 723 shebang, direct execution via `uv run --script`. No bash wrapper needed.
- **Shell mode (on request):** `scripts/<name>.sh` + `scripts/_<name>.py` — bash wrapper delegates to underscore-prefixed Python. Used when the script has no PyPI dependencies (no `uv run` needed).
- **Dependent scripts use whatever language the user specifies** — Python, JS (Node/Bun/Deno), Lua, Bash, or anything else. Never assume a language; ask the user or wait for their suggestion
- **Python scripts always use `uv run --script`** — declare `requires-python` and `dependencies` in the PEP 723 `# /// script ... # ///` block so `uv run` resolves them automatically (no `pip install`, no `requirements.txt`, no venv). Full pattern in [01-python-scripts](references/01-python-scripts.md).
- Any libraries, frameworks, or dependencies are allowed when the user explicitly requests them

### Match Specificity to Task Fragility
- **High freedom** (text): multiple valid approaches, context-dependent decisions
- **Medium** (scripts with parameters): preferred pattern exists, some variation OK
- **Low** (exact commands): fragile operations, consistency is critical

### Description Writing
- Always third person ("Processes Excel files" not "I can help you")
- Include both what the skill does and when to use it
- Include relevant context (file extensions, tool names, task types) so the agent knows when to apply the skill

### Writing Style
- **Use imperative voice** — "Run this command" not "You should run this command"
- **Explain the why, avoid rigid MUST/ALWAYS/NEVER in caps** — modern models respond better to reasoning than rigid commands. If something is critical, explain why it matters

### Progressive Disclosure
Skills use a four-level loading system:

1. **Metadata** (name + description) — always in context (~100 words). Always visible to the agent.
2. **SKILL.md body** — loaded on demand (<5000 tokens ideal). Contains the core instructions.
3. **Scripts** — executed (not loaded into context). Run via `<name>.py` (default) or `<name>.sh` (shell mode).
4. **References** — loaded as needed (unlimited). Reference files load on demand.

Guidelines:
- Keep SKILL.md body under 5000 tokens
- Move detailed content to `references/` files linked from SKILL.md
- Avoid deeply nested references — all reference files should link directly from SKILL.md
- Include a table of contents in reference files longer than 100 lines
- **Reference file naming** — use incrementing numeric prefixes (`01-`, `02-`, …) so files are named `NN-topic.md`, giving deterministic ordering and easy insertion
- **Multi-domain skills** — when a skill supports multiple variants (frameworks, platforms), organize by domain in references (`00-aws.md`, `01-gcp.md`, `02-azure.md`), with SKILL.md holding the workflow and variant selection logic.

### Model Compatibility
SLMs need more explicit guidance and numbered steps; LLMs prefer concise instructions without over-explaining. Aim for clear structure and explicit rules that work across both.

## Gotchas

- **Never create `scripts/` or `assets/` automatically** — `skman.py create` does not generate them by default; use `--with-scripts` only on direct user request. Never scaffold scripts or assets without being asked.
- **Default script is Python, not bash** — `--with-scripts` creates `scripts/<name>.py` with PEP 723 shebang. Use `--lang bash` for the shell wrapper + `_<name>.py` convention.
- **Scaffolded files may lose execute permission** — `skman.py create --with-scripts` sets `chmod 0o755`, but editors or git checkouts can strip it. Always verify with `ls -l <name>.py`; the validator warns if the bit is missing.
- **`--strict` turns section warnings into errors** — only `## Overview` warns when missing; `## Usage`, `## Gotchas`, and `## References` never warn (knowledge-only skills often have no Usage). In strict mode, any warning fails validation.
- **No `:` in text-only frontmatter fields** — `description: Use when: X` is invalid YAML, and even `foo:bar` confuses naive frontmatter parsers. Keep `name`, `description`, `license`, `compatibility` free of colons; rephrase or use `;`. The validator errors on any `:`; `create` rejects colon descriptions before scaffolding.
- **Frontmatter `name` must match the directory basename exactly** — `demo-skill-2-4-1/` requires `name: demo-skill-2-4-1`; the validator warns on mismatch. Fix by renaming the directory or correcting the frontmatter.
- **H1 heading must match `# <name>` or `# <base> <version>`** — the validator errors on mismatch. For `demo-skill-2-4-1/` the H1 must be `# demo-skill 2.4.1` (version uses dots, not hyphens); for `skman/` it is `# skman`.
- **PEP 723 block is mandatory for Python scripts** — every Python script must include the `# /// script ... # ///` metadata block, at the top of the file after the shebang. `uv run` depends on it to resolve dependencies; without it, the script runs with no dependency management.
- **Clone repos locally before studying them** — when creating or updating a skill from a repo (see Creating a skill from a repository for recognition), clone it into a temporary directory first and read files from the local copy. Fetching individual files over the network is expensive in both time and rate limits; a single `git clone` gives you the full tree instantly. Clean up the temp directory after analysis.

## References

- [01-python-scripts](references/01-python-scripts.md) — PEP 723 inline dependencies, `uv run --script`, shebang patterns
- [02-repo-analysis](references/02-repo-analysis.md) — studying a repo for skill creation; doc discovery, whole-repo fallback
