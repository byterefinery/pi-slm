---
name: websearch
description: Searches DuckDuckGo and returns LLM-optimized markdown, JSON, or YAML. Use this skill whenever the user wants to search the web, look up information online, find URLs, do web search, research a topic, or needs current/web-based answers. Uses Safari TLS impersonation and AI-targeted sanitization. Output can be markdown (default), --json, --yaml.
license: Apache-2.0
compatibility: Requires uv installed. Script auto-resolves dependencies via PEP 723.
metadata:
  tags:
    - web
    - search
    - discovery
---

# websearch

## Overview

Search the web via DuckDuckGo and return LLM-optimized results.

**Defaults** (no flags needed):
- **DuckDuckGo HTML** — GET method, clean HTML, reliable results
- **Stealth fetching** — `scrapling.Fetcher` with `impersonate="safari"` + `stealthy_headers=True` (TLS fingerprint + browser-like headers), falls back to requests
- **1-hour result cache** — caches results in `$TMPDIR/websearch-cache/` by query hash. Same query within an hour returns cached results instantly, avoiding rate-limiting
- **AI-targeted markdown** — strips scripts, styles, hidden elements, zero-width chars; converts snippets via `markdownify`
- **Markdown output** — clean, structured, ready for LLM context

Output formats: **markdown** (default), use `--json` or `--yaml` for other formats.

## Usage

```bash
# Search, markdown output (default)
websearch.py "react hooks tutorial"

# JSON output
websearch.py "web scraping tools" --json

# YAML to file
websearch.py "rust async runtime" --yaml -o results.yaml

# Skip AI sanitization (rarely needed)
websearch.py "raw html content" --no-ai-targeted
```

## Gotchas

- **DDG rate-limits aggressively** — returns HTTP 202 + challenge page when blocked. The script silently skips challenged results. Space requests 30+ seconds apart.
- **These are public websites, not APIs** — heavy use will trigger rate limits or CAPTCHAs. The script adds stealth headers but cannot guarantee unlimited access.
- **Results limited to one page** — no pagination. Use specific queries to surface relevant results.
- **Challenge pages are silently skipped** — no error output, just fewer results. This keeps the output clean.
- **`--no-ai-targeted` disables sanitization** — only use if you need raw HTML content in snippets. Default AI-targeted mode strips noise for LLM consumption.
- **Cache key is query-only** — identical queries share cache regardless of output format. Cache TTL is 1 hour.
