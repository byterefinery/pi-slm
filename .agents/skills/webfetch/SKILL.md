---
name: webfetch
description: Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.
license: Apache-2.0
compatibility: Requires uv installed. webfetch.sh runs webfetch.py via uv, which auto-resolves dependencies via PEP 723.
metadata:
  tags:
    - web
    - scraping
    - fetch
---

# webfetch

## Overview

Fetches a web page and returns its content as clean markdown (default) or raw HTML, sanitized for LLM consumption.

One command does everything. You do not write any code — you run the webfetch.sh script once with the URL, and the command's stdout (or the file you named) is the page content.

You execute the script, you never read it. If the user's message is just a URL, the task is to fetch that URL with the script — no flags, no questions.

## Usage

The entry point is `webfetch.sh`, in the `scripts/` folder of this skill. The skill's directory is printed in the header above this text (the line `References are relative to <dir>`).

**Step 1 — Read the user's request, then pick the flags:**

- plain page content → no extra flags
- raw HTML or "HTML source" → add `--html`
- save to a file → add `--file <path>`
- "without sanitization" or unfiltered content → add `--no-ai-targeted`
- a specific fetcher (scrapling, browser, requests) → add `--tool <name>`
- impersonating chrome or firefox → add `--impersonate <name>`

Combine flags when the request needs several (e.g. `--html --file /abs/path.html`).

**Step 2 — Run exactly one bash command (no `cd`, from wherever you are):**

```bash
/home/mtasic/projects-b/skills/.agents/skills/webfetch/scripts/webfetch.sh <flags> https://example.com
```

That path is the skill's directory from the header plus `scripts/webfetch.sh`. If the header above shows a different skill directory, build the path from that directory instead. Replace `<flags>` with the flags from step 1 (leave it out when none), and replace the URL.

**Step 3 — Get the content:**

- without `--file`: the bash tool result (stdout) is the content. Answer the user's question from it. Done — there is no output file and nothing else to do.
- with `--file`: the command prints nothing — the bash tool result will be `(no output)`. That is success, not an error; the content went into the file. Verify and answer in one step: `read` the output file — the exact path you passed to `--file` (e.g. `read ./tangled.md`). The file is created in the current working directory (the user's project), not in the skill's directory: `--file ./tangled.md` lands in the project root. If the read succeeds you are done — do not re-run the fetch, do not list or search directories, do not read the scripts to debug.

Use `--file` (not a `>` redirect) when the user asks to save to a file.

## Gotchas

- **Never read the script files** — not `webfetch.sh`, not `webfetch.py` — not to learn how to use them, not to find their path, not to debug a fetch that seems to have failed. The steps above are all you need; the path comes from the header (step 2). Run `webfetch.sh` with the URL.
- **Only run webfetch.sh** — never invoke `webfetch.py` yourself (no `python3 webfetch.py`, no `uv run` by hand). `webfetch.py` needs packages that are not on the system Python and would silently fall back to a worse fetcher; `webfetch.sh` sets that up for you.
