#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scrapling[all]>=0.4.8",
#     "pyyaml>=6.0",
#     "requests>=2.31",
#     "markdownify>=1.2.3",
#     "beautifulsoup4>=4.15.0",
# ]
# ///

"""websearch — DuckDuckGo web search with LLM-optimized output.

Searches DuckDuckGo and returns results as markdown (default), JSON, or YAML.
Uses scrapling Fetcher with Safari TLS impersonation by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import urllib.parse

# Silence scrapling info logs
logging.getLogger("scrapling").setLevel(logging.WARNING)


# ── DuckDuckGo configuration ─────────────────────────────────────────

DDG_URL = "https://html.duckduckgo.com/html/?q={query}"
DDG_REFERER = "https://duckduckgo.com/"


# ── AI-targeted HTML sanitization ────────────────────────────────────

def _ai_targeted_html(html_text: str) -> str:
    """Strip noise from HTML for AI/LLM consumption.

    Removes scripts, styles, noscript, svg, templates, aria-hidden,
    zero-width chars, control chars — same approach as webfetch.
    Lightweight regex — BeautifulSoup reserved for HTML→JSON/YAML parsing.
    """
    import re

    for tag in ("script", "style", "noscript", "svg", "template", "link", "meta"):
        html_text = re.sub(
            r"<" + tag + r"[^>]*>.*?</" + tag + r">", "", html_text,
            flags=re.DOTALL | re.I,
        )
        html_text = re.sub(r"<" + tag + r"[^>]*/>", "", html_text, flags=re.I)
    html_text = re.sub(
        r'<[^>]*aria-hidden="true"[^>]*>.*?</[^>]*>', "", html_text,
        flags=re.DOTALL | re.I,
    )
    html_text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", html_text)
    html_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", html_text)
    return html_text


def html_to_md(html_text: str, *, ai_targeted: bool = True) -> str:
    """Convert HTML to clean markdown using markdownify."""
    from markdownify import markdownify as md
    if ai_targeted:
        html_text = _ai_targeted_html(html_text)
    return md(html_text, heading_style="atx", strip=["img"]).strip()


# ── Fetching: scrapling → requests fallback + cache ──────────────────

import hashlib
import tempfile
import time
from pathlib import Path

# Cache directory and TTL
_CACHE_DIR = Path(tempfile.gettempdir()) / "websearch-cache"
_CACHE_TTL = 3600  # 1 hour


def _cache_key(query: str) -> str:
    """Generate a unique cache key for query."""
    raw = query.lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(query: str) -> str | None:
    """Get cached HTML if within TTL."""
    key = _cache_key(query)
    cache_file = _CACHE_DIR / f"{key}.html"
    if not cache_file.exists():
        return None
    try:
        meta = cache_file.stat()
        if time.time() - meta.st_mtime > _CACHE_TTL:
            cache_file.unlink(missing_ok=True)  # Remove expired
            return None
        return cache_file.read_text(encoding="utf-8")
    except Exception:
        return None


def _cache_put(query: str, html: str) -> None:
    """Cache HTML results."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(query)
    cache_file = _CACHE_DIR / f"{key}.html"
    try:
        cache_file.write_text(html, encoding="utf-8")
    except Exception:
        pass


def _is_challenge(html_text: str) -> bool:
    """Detect CAPTCHA/challenge pages."""
    markers = [
        "anomaly-modal", "challenge-form", "bots use",
        "complete the following challenge", "select all squares",
        "captcha", "verify you are human", "robot check",
    ]
    lower = html_text.lower()
    return any(m in lower for m in markers)


def _has_results(html_text: str) -> bool:
    """Check if HTML contains any search results."""
    return any(marker in html_text for marker in [
        'result__a', 'result__snippet',
    ])


def _fetch_with_scrapling(url: str, headers: dict) -> str | None:
    """Fetch with scrapling Fetcher (TLS impersonation + stealthy headers)."""
    from scrapling import Fetcher
    try:
        resp = Fetcher.get(
            url, headers=headers,
            impersonate="safari", stealthy_headers=True,
        )
        if resp.status != 200:
            return None
        html = resp.html_content
        return html if not _is_challenge(html) else None
    except Exception:
        return None


def _fetch_with_requests(url: str, headers: dict) -> str | None:
    """Fetch with requests (stdlib fallback, no TLS impersonation)."""
    import requests
    try:
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            return None
        html = resp.text
        return html if not _is_challenge(html) else None
    except Exception:
        return None


def fetch(query: str) -> str | None:
    """Fetch DuckDuckGo search results with cache: cache → scrapling → requests.

    Results cached for 1 hour to avoid rate-limiting.
    Returns HTML with results or None on failure/challenge.
    """
    # 1. Check cache first
    cached = _cache_get(query)
    if cached:
        return cached

    encoded = urllib.parse.quote_plus(query)
    url = DDG_URL.format(query=encoded)

    headers = {"Referer": DDG_REFERER}

    # 2. Try scrapling (TLS fingerprint impersonation — best anti-bot)
    html = _fetch_with_scrapling(url, headers)
    if html and _has_results(html):
        _cache_put(query, html)
        return html

    # 3. Fallback: requests (no TLS impersonation, but sometimes works)
    html = _fetch_with_requests(url, headers)
    if html and _has_results(html):
        _cache_put(query, html)
        return html

    return None


# ── Parsing ──────────────────────────────────────────────────────────

from bs4 import BeautifulSoup


def _soup(html_text: str) -> BeautifulSoup:
    """Parse HTML string with BeautifulSoup."""
    return BeautifulSoup(html_text, "html.parser")


def parse_results(html_text: str) -> list[dict]:
    """Parse DuckDuckGo HTML search results using BeautifulSoup."""
    soup = _soup(html_text)
    results = []
    seen = set()

    links = soup.find_all("a", class_="result__a")
    snippets = soup.find_all(class_="result__snippet")
    urls = soup.find_all("a", class_="result__url")

    for i, link in enumerate(links):
        href = link.get("href", "")
        title = link.get_text(strip=True)
        if not title or not href:
            continue
        norm = href.rstrip("/").lower()
        if norm in seen:
            continue
        seen.add(norm)

        snippet = ""
        if i < len(snippets):
            snip_text = snippets[i].get_text(strip=True)
            if snip_text:
                snippet = html_to_md(snip_text)

        source = ""
        if i < len(urls):
            source = urls[i].get("href", "").strip()

        results.append({
            "title": title,
            "url": href,
            "snippet": snippet,
            "source": source or href,
        })

    return results


# ── Search ───────────────────────────────────────────────────────────

def search(query: str) -> list[dict]:
    """Search DuckDuckGo and return results."""
    html_text = fetch(query)
    if html_text is None:
        return []
    return parse_results(html_text)


# ── Output formatters ────────────────────────────────────────────────

def format_markdown(results: list[dict], query: str) -> str:
    """Format results as LLM-optimized markdown."""
    if not results:
        return "No results found.\n"
    lines = [f"# Search results for: {query}\n", f"*{len(results)} results*\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r['title']}\n")
        if r.get("snippet"):
            lines.append(f"{r['snippet']}\n")
        lines.append(f"- **URL**: {r['url']}")
        lines.append("")
    return "\n".join(lines)


def format_json(results: list[dict], query: str) -> str:
    """Format results as JSON."""
    return json.dumps({"query": query, "count": len(results), "results": results},
                      indent=2, ensure_ascii=False) + "\n"


def format_yaml(results: list[dict], query: str) -> str:
    """Format results as YAML."""
    import yaml
    return yaml.dump({"query": query, "count": len(results), "results": results},
                     default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="websearch.py",
        description="Search DuckDuckGo and output LLM-optimized results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              websearch.py "react hooks tutorial"
              websearch.py "python async" --json
              websearch.py "rust vs go" --yaml -o results.yaml

            Output formats: markdown (default), --json, --yaml
        """),
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output results as JSON (default: markdown)")
    parser.add_argument("--yaml", action="store_true", default=False,
                        help="Output results as YAML")
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
                        help="Write output to file instead of stdout")
    parser.add_argument("--no-ai-targeted", action="store_false", dest="ai_targeted",
                        default=True, help="Disable AI-targeted HTML sanitization")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    results = search(args.query)

    if args.json and args.yaml:
        print("Error: --json and --yaml are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if args.json:
        output = format_json(results, args.query)
    elif args.yaml:
        output = format_yaml(results, args.query)
    else:
        output = format_markdown(results, args.query)

    if args.output:
        outdir = os.path.dirname(os.path.abspath(args.output))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
