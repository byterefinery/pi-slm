---
name: webfetch
description: Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.
license: Apache-2.0
compatibility: Requires uv installed. Script auto-resolves dependencies via PEP 723.
metadata:
  tags:
    - web
    - scraping
    - fetch
---

# webfetch

## Overview

Fetches web pages and outputs clean markdown optimized for LLM consumption.

Defaults (no flags needed):
- **Safari impersonation** — TLS fingerprint (scrapling) or user-agent (requests), rarely blocked
- **AI-targeted markdown** — strips scripts, styles, hidden elements, zero-width chars, prompt injection vectors
- **Auto-detect fetcher** — tries scrapling → browser → requests

Fetcher priority:
1. **scrapling** — `Fetcher.get()` with Safari TLS impersonation + AI-targeted sanitization via `markdownify`
2. **browser** — `DynamicFetcher.fetch()` with system Chrome/Chromium for JavaScript-rendered SPAs
3. **requests** — stdlib fallback with Safari user-agent + built-in HTML-to-markdown conversion

SPA auto-detection: if scrapling returns an empty shell (large HTML, tiny content, framework markers like `<div id="root">`, import maps, etc.), automatically retries with the browser fetcher. Use `--tool` to force a specific fetcher.

Use `--html` for raw HTML, `--impersonate` to change browser, `--no-ai-targeted` to skip sanitization, `--tool` to force a fetcher.

## Usage

```bash
# Default: Safari-impersonated, AI-targeted markdown to stdout
webfetch.py https://example.com

# Save to file
webfetch.py --file ./page.md https://example.com

# Raw HTML (skip markdown conversion)
webfetch.py --html https://example.com

# Override defaults (rarely needed)
webfetch.py --impersonate chrome https://example.com
webfetch.py --no-ai-targeted https://example.com
webfetch.py --tool requests https://example.com
```

`--file` auto-detects format from extension (`.html`/`.htm` → HTML, everything else → markdown) unless `--html`/`--md` is also given.

`--tool` accepts: `scrapling`, `browser`, `requests`. Without `--tool`, auto-detect tries scrapling → browser → requests.

## Gotchas

- **Output is LLM-optimized by default** — Safari impersonation + AI-targeted markdown (scripts, styles, hidden elements, zero-width chars, prompt injection vectors stripped). Use `--no-ai-targeted` only if you need raw content.
- **SPA auto-detection** — if scrapling returns an empty shell (large HTML, tiny content, framework markers), the script automatically retries with the browser fetcher. No `--tool browser` needed for most SPAs.
- **`--tool browser` needs system Chrome/Chromium** — uses `real_chrome=True` to find system-installed browsers. If unavailable, auto-detect falls back to requests.
- **`--impersonate` is safari by default** — change only if a site blocks Safari fingerprints. Accepts chrome, firefox, or any curl-impersonate browser name.
- **`--file` format auto-detection** — `.html`/`.htm` extensions trigger HTML output; all other extensions default to markdown. Explicit `--html`/`--md` overrides.
- **Script is self-contained** — dependencies (`scrapling[all]`, `markdownify`, `requests`) declared inline via PEP 723. `uv run --script` resolves them automatically.
