#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scrapling[all]>=0.4.12",
#     "markdownify==1.2.3",
#     "requests>=2.34.2",
# ]
# ///

"""webfetch — Fetch web pages as markdown or HTML for LLM consumption.

Uses scrapling Fetcher with Safari impersonation by default.
Falls back to requests with built-in HTML-to-markdown conversion.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap

# ── User agents (Safari — rarely blocked) ────────────────────────────
UA_IPHONE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7_8 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/26.0 Mobile/15E148 Safari/604.1"
)

UA_MAC = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/26.0 Safari/605.1.15"
)

# ── HTML → Markdown (stdlib-only, used by requests fallback) ────────

_HTML_REPLACEMENTS = [
    # Headings
    (re.compile(r"<h1[^>]*>(.*?)</h1>", re.DOTALL | re.I), r"\n\n# \1\n\n"),
    (re.compile(r"<h2[^>]*>(.*?)</h2>", re.DOTALL | re.I), r"\n\n## \1\n\n"),
    (re.compile(r"<h3[^>]*>(.*?)</h3>", re.DOTALL | re.I), r"\n\n### \1\n\n"),
    (re.compile(r"<h4[^>]*>(.*?)</h4>", re.DOTALL | re.I), r"\n\n#### \1\n\n"),
    (re.compile(r"<h5[^>]*>(.*?)</h5>", re.DOTALL | re.I), r"\n\n##### \1\n\n"),
    (re.compile(r"<h6[^>]*>(.*?)</h6>", re.DOTALL | re.I), r"\n\n###### \1\n\n"),
    # Paragraphs & breaks
    (re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL | re.I), r"\n\n\1\n\n"),
    (re.compile(r"<br\s*/?>", re.I), r"\n"),
    # Lists
    (re.compile(r"<li[^>]*>(.*?)</li>", re.DOTALL | re.I), r"  - \1\n"),
    # Links
    (re.compile(r'<a[^>]*href="([^"]*?)"[^>]*>(.*?)</a>', re.DOTALL | re.I), r"\2 (\1)"),
    # Formatting
    (re.compile(r"<strong[^>]*>(.*?)</strong>", re.DOTALL | re.I), r"**\1**"),
    (re.compile(r"<b[^>]*>(.*?)</b>", re.DOTALL | re.I), r"**\1**"),
    (re.compile(r"<em[^>]*>(.*?)</em>", re.DOTALL | re.I), r"*\1*"),
    (re.compile(r"<i[^>]*>(.*?)</i>", re.DOTALL | re.I), r"*\1*"),
    (re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.I), r"`\1`"),
    (re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.I), r"\n\n```\n\1\n```\n\n"),
    # Tables (basic)
    (re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.I), r"\1\n"),
    (re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.I), r" \1 |"),
]


def html_to_md(html: str) -> str:
    """Convert HTML to markdown using stdlib-only regex replacements."""
    import html as html_mod

    content = html_mod.unescape(html)

    # Strip HTML comments first (contains > chars that break tag stripping)
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    # Block-level conversions
    for pat, repl in _HTML_REPLACEMENTS:
        content = pat.sub(repl, content)

    # Strip remaining tags
    content = re.sub(r"<[^>]+>", "", content)

    # Collapse whitespace
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


# ── Fetchers ─────────────────────────────────────────────────────────

def _ai_targeted_html(html: str) -> str:
    """Strip noise from HTML for AI/LLM consumption.

    Mirrors scrapling CLI --ai-targeted: removes scripts, styles, noscript,
    svg, templates, aria-hidden, zero-width chars, control chars.
    """
    # Remove block elements (with content)
    for tag in ("script", "style", "noscript", "svg", "template", "link", "meta"):
        html = re.sub(r"<" + tag + r"[^>]*>.*?</" + tag + r">", "", html, flags=re.DOTALL | re.I)
        html = re.sub(r"<" + tag + r"[^>]*/>", "", html, flags=re.I)  # self-closing
    # Remove aria-hidden elements
    html = re.sub(r'<[^>]*aria-hidden="true"[^>]*>.*?</[^>]*>', "", html, flags=re.DOTALL | re.I)
    # Strip zero-width and control characters
    html = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", html)
    html = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", html)
    return html


def _to_md(html: str, ai_targeted: bool) -> str:
    """Convert HTML to markdown, optionally with AI-targeted cleaning."""
    from markdownify import markdownify as md
    if ai_targeted:
        html = _ai_targeted_html(html)
    return md(html, heading_style="atx", strip=['img'])


def fetch_with_scrapling(url: str, output_format: str, *, impersonate: str, ai_targeted: bool) -> tuple[str, str]:
    """Fetch using scrapling Fetcher. Returns (result, raw_html)."""
    import logging
    from scrapling.fetchers import Fetcher

    logging.getLogger("scrapling").setLevel(logging.WARNING)

    page = Fetcher.get(url, impersonate=impersonate, timeout=30)
    html = page.html_content

    if output_format == "html":
        return html, html
    return _to_md(html, ai_targeted), html


def fetch_with_browser(url: str, output_format: str, *, impersonate: str, ai_targeted: bool) -> tuple[str, str]:
    """Fetch using scrapling DynamicFetcher (browser). Returns (result, raw_html)."""
    import logging
    from scrapling.fetchers import DynamicFetcher

    logging.getLogger("scrapling").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)

    page = DynamicFetcher.fetch(
        url,
        headless=True,
        disable_resources=True,
        real_chrome=True,
        timeout=30000,
    )
    html = page.html_content

    if output_format == "html":
        return html, html
    return _to_md(html, ai_targeted), html


def _ua_for_impersonate(impersonate: str) -> str:
    """Map impersonate name to a requests-compatible user-agent."""
    if "safari" in impersonate.lower():
        return UA_IPHONE
    return UA_MAC


def fetch_with_requests(url: str, output_format: str, *, impersonate: str, ai_targeted: bool) -> tuple[str, str]:
    """Fetch using requests as fallback. Returns (result, raw_html)."""
    import requests

    headers = {
        "User-Agent": _ua_for_impersonate(impersonate),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    html = resp.text

    if output_format == "html":
        return html, html
    if ai_targeted:
        html = _ai_targeted_html(html)
    return html_to_md(html), resp.text


def _looks_like_spa_shell(result: str, html: str) -> bool:
    """Heuristic: detect JS-rendered SPA shell (large HTML, tiny content).

    Returns True if the page looks like it needs a browser to render.
    """
    # Only check markdown output (HTML output is always the full HTML)
    if len(html) < 500:
        return False
    # Large HTML but very little extracted content
    if len(result) < 200 and len(html) > 1000:
        # Check for SPA markers: root div, import maps, framework scripts
        spa_markers = [
            r'<div[^>]*id="root"',
            r'<div[^>]*id="app"',
            r'type="importmap"',
            r'<script[^>]*type="module"',
            r'__NEXT_DATA__',
            r'window\.__NUXT__',
            r'__PEAKS__',
            r'__REMIX__',
        ]
        for marker in spa_markers:
            if re.search(marker, html, re.I):
                return True
    return False


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="webfetch.py",
        description="Fetch a web page and output as markdown (default) or raw HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              webfetch.py https://example.com
              webfetch.py --html https://example.com
              webfetch.py --md --file ./page.md https://example.com
              webfetch.py --impersonate chrome https://example.com
              webfetch.py --no-ai-targeted https://example.com
              webfetch.py --tool browser https://spa.example.com

            Auto-detection tries scrapling first, then browser, then requests.
            --ai-targeted is on by default; use --no-ai-targeted to disable sanitization.
            --impersonate defaults to safari; also accepts chrome, firefox, or any browser name from curl_cffi/curl-impersonate.
        """),
    )

    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--html", action="store_true", default=False,
                     help="Output raw HTML instead of markdown")
    fmt.add_argument("--md", "--markdown", action="store_true", default=False,
                     help="Output markdown (default)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stdout", action="store_true", default=False,
                      help="Print result to stdout (default)")
    mode.add_argument("--file", metavar="PATH", default=None,
                      help="Save result to file; nothing printed to stdout")

    parser.add_argument("--tool", choices=["scrapling", "browser", "requests"], default=None,
                        help="Force a specific fetcher (scrapling, browser, requests)")
    parser.add_argument("--impersonate", default="safari",
                        help="Browser to impersonate: safari (default), chrome, firefox")
    parser.add_argument("--ai-targeted", action="store_true", default=True,
                        help="Sanitize output for LLM consumption (default on; use --no-ai-targeted to disable)")
    parser.add_argument("--no-ai-targeted", action="store_false",
                        dest="ai_targeted",
                        help="Disable AI-targeted sanitization")
    parser.add_argument("url", nargs="?", default=None, help="URL to fetch")

    args = parser.parse_args(argv)

    if args.url is None:
        parser.error("missing URL. Usage: webfetch.py [OPTIONS] <URL>")

    # Derive format from --file extension if not explicitly set
    if args.file and not args.html and not args.md:
        if args.file.endswith((".html", ".htm")):
            args.html = True
        else:
            args.md = True

    return args


def main() -> None:
    args = parse_args()

    output_format = "html" if args.html else "md"
    impersonate = args.impersonate
    ai_targeted = args.ai_targeted

    # Determine which fetcher to use
    if args.tool == "requests":
        fetcher_name = "requests"
    elif args.tool == "scrapling":
        fetcher_name = "scrapling"
    else:
        # Auto-detect: try scrapling first, fall back to requests
        fetcher_name = None
        tried = []

    result = None
    raw_html = ""
    errors = []

    def try_fetch(name: str, fn) -> bool:
        nonlocal result, raw_html, fetcher_name
        try:
            result, raw_html = fn(args.url, output_format, impersonate=impersonate, ai_targeted=ai_targeted)
            fetcher_name = name
            return True
        except Exception as e:
            errors.append(f"  {name}: {e}")
            tried.append(name)
            return False

    if args.tool:
        # User forced a specific tool
        fetchers = {
            "scrapling": ("scrapling", fetch_with_scrapling),
            "browser": ("browser", fetch_with_browser),
            "requests": ("requests", fetch_with_requests),
        }
        name, fn = fetchers[args.tool]
        success = try_fetch(name, fn)

        if not success:
            print(f"webfetch.py: '{args.tool}' failed to fetch {args.url}.", file=sys.stderr)
            for err in errors:
                print(err, file=sys.stderr)
            sys.exit(1)
    else:
        # Auto-detect: scrapling → (SPA check → browser) → requests
        if not try_fetch("scrapling", fetch_with_scrapling):
            if not try_fetch("browser", fetch_with_browser):
                if not try_fetch("requests", fetch_with_requests):
                    print(f"webfetch.py: no suitable fetcher found for {args.url}.", file=sys.stderr)
                    for err in errors:
                        print(err, file=sys.stderr)
                    sys.exit(1)
        # SPA auto-detection: if scrapling succeeded but looks like empty shell, try browser
        elif _looks_like_spa_shell(result, raw_html):
            if not try_fetch("browser", fetch_with_browser):
                # Browser failed, keep scrapling result (better than nothing)
                pass

    # Output
    if args.file:
        with open(args.file, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
