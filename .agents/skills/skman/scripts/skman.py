#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML"]
# ///

"""skman — Skill Manager: scaffold, validate, and inspect agent skills.

Usage:
    skman.py --help
    skman.py create --help
    skman.py validate --help
    skman.py info --help
    skman.py generate --help
"""

import argparse
import json
import os
import re
import sys
import textwrap
import unicodedata

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NAME_LEN = 64
MAX_DESC_LEN = 1024
MAX_COMPATIBILITY_LEN = 500

FRONTMATTER_RE = re.compile(
    r'^---\s*\n(?P<content>.*?)^---\s*\n', re.MULTILINE | re.DOTALL
)

BODY_TEMPLATE = textwrap.dedent("""\
# {title}

## Overview

[Describe what this skill does and when to use it.]

## Usage

[How to use this skill. Include examples.]

""")

BODY_TEMPLATE_WITH_PYTHON_SCRIPT = textwrap.dedent("""\
# {title}

## Overview

[Describe what this skill does and when to use it.]

## Usage

```bash
{script_name}.py --help
```

""")

BODY_TEMPLATE_WITH_SHELL_SCRIPT = textwrap.dedent("""\
# {title}

## Overview

[Describe what this skill does and when to use it.]

## Usage

```bash
{script_name}.sh --help
```

""")

# Matches trailing version suffix: -<digit>[-<digit>]* at end of dir name
# e.g. "demo-skill-2-4-1" -> "2-4-1", "git-8-20-0" -> "8-20-0"
VERSION_SUFFIX_RE = re.compile(r'-(\d+(?:-\d+)+)$')

# Warns when missing — every skill should explain what it does.
# ## Usage, ## Gotchas, and ## References are truly optional (never warn).
RECOMMENDED_SECTIONS = {'## Overview'}

# ---------------------------------------------------------------------------
# URL patterns for name/version extraction
# ---------------------------------------------------------------------------
# Tried in order — first match wins. Each regex captures (name, version);
# the version group may be absent (None). Case-insensitive so the name in
# the result preserves the original casing of the URL.

_URL_PATTERNS = [
    # GitHub: /{user}/{repo} with optional /releases/tag, /tree, /-/tags
    re.compile(
        r'github\.com/[^/]+/([^/.]+?)(?:\.git)?'
        r'(?:/(?:releases/tag|tree|-/tags)/v?([^?#/]+))?'
        r'(?:[?#/].*)?$', re.IGNORECASE),
    # GitLab: /{user}/{repo} with optional /-/tags, /tags
    re.compile(
        r'gitlab\.com/[^/]+/([^/.]+?)(?:\.git)?'
        r'(?:/(?:-/tags|tags)/v?([^?#/]+))?'
        r'(?:[?#/].*)?$', re.IGNORECASE),
    # npm: npmjs.com/package/{pkg}[/v/{ver}]
    re.compile(
        r'npmjs\.com/package/([^/?#]+)(?:/v/([^?#/]+))?', re.IGNORECASE),
    # PyPI, crates.io, RubyGems, Go: {registry}/{segment}/{pkg}/[v]{ver}
    re.compile(
        r'(?:pypi\.org/project|crates\.io/crates|rubygems\.org/gems|pkg\.go\.dev)/'
        r'([^/?#]+)/v?([^/?#]+)', re.IGNORECASE),
    # crates.io without version: crates.io/crates/{pkg}
    re.compile(r'crates\.io/crates/([^/?#]+)', re.IGNORECASE),
]


def _extract_name_version(url):
    """Extract (name, version) from a package/repo URL.

    Tries known registry patterns (GitHub, GitLab, PyPI, npm, crates.io,
    RubyGems, pkg.go.dev). Returns (name, version) where version may be
    None. Raises ValueError if the URL is unrecognizable.
    """
    url = url.strip().rstrip('/')
    for pattern in _URL_PATTERNS:
        m = pattern.search(url)
        if m:
            name = m.group(1).removesuffix('.git')
            version = m.group(2)
            if version and version.startswith('v'):
                version = version[1:]
            return name, version or None

    raise ValueError(f"unrecognized URL pattern: {url}")


def _llm_extract_version(url, name):
    """Fallback for version extraction when the URL is unrecognized.

    A standalone script cannot call an LLM, so the agent supplies the
    answer by setting SKMAN_LLM_RESPONSE='{"version": "X.Y.Z"}' before
    re-running. Returns a version string or None.
    """
    env_response = os.environ.get('SKMAN_LLM_RESPONSE', '').strip()
    if env_response:
        try:
            ver = json.loads(env_response).get('version')
            if ver:
                return str(ver).lstrip('v')
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    print(
        f"create: could not extract version from URL. "
        f"Please provide version for '{name}' from: {url}",
        file=sys.stderr,
    )
    print(
        '  Set SKMAN_LLM_RESPONSE=\'{"version": "X.Y.Z"}\' or pass --version explicitly.',
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
# Char-per-token ratios based on tokenizer behavior (GPT-4 / Claude):
#   English prose:      ~4.0 chars/token
#   Code:               ~1.5–2.5 chars/token
#   CJK (Chinese/Japanese/Korean): ~1.0–1.5 chars/token
#   Mixed (prose + code fences + markdown): ~3.0–3.5 chars/token
# We use conservative (low) ratios so the estimate never undercounts.

_CHARS_PER_TOKEN_PROSE = 3.5
_CHARS_PER_TOKEN_CODE = 2.0
_CHARS_PER_TOKEN_CJK = 1.2

_CJK_RE = re.compile(
    r'['
    r'\u3000-\u303f'   # CJK symbols and punctuation
    r'\u3040-\u309f'   # Hiragana
    r'\u30a0-\u30ff'   # Katakana
    r'\u4e00-\u9fff'   # CJK unified ideographs
    r'\uf900-\ufaff'   # CJK compatibility ideographs
    r'\uac00-\ud7af'   # Hangul syllables
    r']'
)


def _chars_to_tokens(text, in_code=False):
    """Convert a chunk of text to an estimated token count."""
    chars = len(text)
    if chars == 0:
        return 0

    cjk_ratio = len(_CJK_RE.findall(text)) / chars
    if cjk_ratio > 0.3:
        return max(1, int(chars / _CHARS_PER_TOKEN_CJK))
    if in_code:
        return max(1, int(chars / _CHARS_PER_TOKEN_CODE))
    return max(1, int(chars / _CHARS_PER_TOKEN_PROSE))


def _estimate_tokens(text):
    """Estimate the number of tokens in *text*.

    Splits the text into prose regions and fenced code blocks, applying
    different char-per-token ratios. The estimate is conservative (tends
    to overcount slightly) so warnings trigger early rather than late.
    """
    if not text:
        return 0

    total = 0
    in_fence = False
    fence_chars = 0
    chunk = []

    for line in text.splitlines(keepends=True):
        if line.strip().startswith('```'):
            if chunk:
                total += _chars_to_tokens(''.join(chunk), in_code=in_fence)
                chunk = []
            in_fence = not in_fence
            fence_chars += len(line)
            continue
        chunk.append(line)

    if chunk:
        total += _chars_to_tokens(''.join(chunk), in_code=in_fence)

    # Fence delimiters themselves count as tokens too
    total += fence_chars // 3
    return max(1, total)


# ---------------------------------------------------------------------------
# Frontmatter parsing / serialization (PyYAML)
# ---------------------------------------------------------------------------

def _parse_frontmatter(text):
    """Return (frontmatter, body, yaml_error).

      (fm, body, None)   — frontmatter parsed OK ({} if the block is empty)
      (None, text, None) — no frontmatter block found
      (None, body, err)  — frontmatter block exists but is not a valid
                           YAML mapping; err explains the failure
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, text, None
    body = text[m.end():]
    try:
        fm = yaml.safe_load(m.group('content'))
    except yaml.YAMLError as e:
        return None, body, f"YAML parse error: {' '.join(str(e).split())}"
    if fm is None:
        fm = {}
    if not isinstance(fm, dict):
        return None, body, f"frontmatter must be a YAML mapping (got {type(fm).__name__})"
    return fm, body, None


def _build_frontmatter_yaml(data):
    """Serialize *data* to a YAML frontmatter block (--- ... ---)."""
    raw = yaml.dump(
        data, default_flow_style=False, sort_keys=False,
        allow_unicode=True, width=1000,  # avoid line wrapping
    ).rstrip('\n')
    return f"---\n{raw}\n---\n"


def _check_duplicate_frontmatter_keys(text):
    """Return error strings for duplicate top-level YAML keys.

    PyYAML silently accepts duplicate keys (last value wins); flag them
    since a duplicate almost always indicates a mistake. Only column-0
    keys are matched, so nested/indented keys never count.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return []
    counts = {}
    for line in m.group('content').splitlines():
        lm = re.match(r'^([A-Za-z0-9_-]+)\s*:', line)
        if lm:
            key = lm.group(1)
            counts[key] = counts.get(key, 0) + 1
    dups = sorted(k for k, n in counts.items() if n > 1)
    if dups:
        return [
            f"duplicate frontmatter key(s): {', '.join(dups)} "
            f"(each key may appear at most once)"
        ]
    return []


# ---------------------------------------------------------------------------
# Frontmatter field validation
# ---------------------------------------------------------------------------

def _validate_name(name):
    """Return list of error strings (empty = valid).

    Allows Unicode lowercase letters (i18n), digits, and hyphens.
    Must not start/end with a hyphen or contain consecutive hyphens.
    """
    errors = []
    if not name:
        errors.append("name is missing")
        return errors
    if not isinstance(name, str):
        errors.append(f"name must be a string (got {type(name).__name__})")
        return errors

    # Normalize Unicode for consistent comparison
    name = unicodedata.normalize("NFKC", name.strip())

    if len(name) > MAX_NAME_LEN:
        errors.append(f"name exceeds {MAX_NAME_LEN} characters ({len(name)})")
    if name != name.lower():
        errors.append("name must be lowercase")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name cannot start or end with a hyphen")
    if "--" in name:
        errors.append("name cannot contain consecutive hyphens")
    if not all(c.isalnum() or c == "-" for c in name):
        errors.append(
            "name must contain only letters, digits, and hyphens; "
            "no leading/trailing/consecutive hyphens"
        )
    return errors


def _validate_description(desc):
    """Return list of error strings (empty = valid)."""
    errors = []
    if desc is None:
        errors.append("description is missing (required)")
        return errors
    if not isinstance(desc, str):
        errors.append(f"description must be a string (got {type(desc).__name__})")
        return errors
    if not desc.strip():
        errors.append("description is empty (required)")
        return errors
    if len(desc) > MAX_DESC_LEN:
        errors.append(f"description exceeds {MAX_DESC_LEN} characters ({len(desc)})")
    if re.search(r'<[a-zA-Z/][^>]*>', desc):
        errors.append("description must not contain XML/HTML tags")
    return errors


def _validate_compatibility(compat):
    """Return list of error strings for the optional compatibility field."""
    if not isinstance(compat, str):
        return ["compatibility must be a string"]
    errors = []
    if len(compat) > MAX_COMPATIBILITY_LEN:
        errors.append(
            f"compatibility exceeds {MAX_COMPATIBILITY_LEN} characters ({len(compat)})"
        )
    return errors


def _validate_metadata(metadata):
    """Return list of warning strings for the optional metadata field.

    metadata should be a mapping; if present, 'tags' must be a list of
    strings. Returns warnings only (metadata is optional).
    """
    if metadata is None:
        return []
    if not isinstance(metadata, dict):
        return [f"metadata should be a mapping (got {type(metadata).__name__})"]
    tags = metadata.get('tags')
    if tags is None:
        return []
    if not isinstance(tags, list):
        return [f"metadata.tags should be an array of strings (got {type(tags).__name__})"]
    if not all(isinstance(t, str) for t in tags):
        non_str = [t for t in tags if not isinstance(t, str)]
        return [f"metadata.tags must contain only strings (found: {non_str})"]
    return []


def _validate_text_fields(fm):
    """Return list of error strings for text-only frontmatter fields.

    Top-level scalar string values must not contain a ':' character —
    colons are YAML structural characters and naive frontmatter parsers
    misread them. Rephrase or use ';' instead.
    """
    errors = []
    if not isinstance(fm, dict):
        return errors
    for key, value in fm.items():
        if isinstance(value, str) and ':' in value:
            snippet = value if len(value) <= 60 else value[:57] + '...'
            errors.append(
                f"{key} must not contain ':' (value: '{snippet}'); "
                f"rephrase or use ';' instead"
            )
    return errors


# ---------------------------------------------------------------------------
# Body / structure checks
# ---------------------------------------------------------------------------
# Each checker returns a list of (label, message) tuples where label is one
# of "PASS", "WARN", "ERROR". A PASS entry is emitted when the check runs
# and finds nothing to report.

def _iter_headings(body):
    """Yield heading lines from *body*, skipping fenced code blocks."""
    in_fence = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith('```'):
            in_fence = not in_fence
            continue
        if not in_fence and stripped.startswith('#'):
            yield stripped


def _check_sections(body):
    """Check for recommended sections.

    ## Overview warns if missing — every skill should explain what it does.
    ## Usage, ## Gotchas, and ## References are truly optional — no warning.
    """
    missing = RECOMMENDED_SECTIONS - set(_iter_headings(body))
    if missing:
        return [("WARN", f"missing recommended section(s): {', '.join(sorted(missing))}")]
    return [("PASS", "all recommended sections present")]


def _check_references_section(body):
    """Check that a ## References section (if present) uses a bulleted list.

    Tables are not allowed. Expected lines look like
    '- [01-topic](references/01-topic.md) — description'; links to local
    reference files or external URLs are both valid.
    """
    lines = body.splitlines()
    in_fence = False
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_fence = not in_fence
            continue
        if not in_fence and stripped == '## References':
            start = i
            break

    if start is None:
        return [("PASS", "references section format is correct (or absent)")]

    # Collect lines after ## References until next heading, fence, or EOF
    section = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if (stripped.startswith('```')
                or re.match(r'^#{1,6}\s', stripped)
                or re.match(r'^#{1,6}$', stripped)):
            break
        section.append(stripped)

    bullet_ref = re.compile(
        r'^-\s+\[.+\]\((?:references/[^)]+|https?://[^)]+)\)'
    )
    has_bullets = any(
        bullet_ref.match(l) for l in section if l and not l.startswith('|')
    )
    has_table = sum(1 for l in section if l.startswith('|')) > 1

    if has_table and not has_bullets:
        return [("WARN",
                 "## References uses a table format; use a bulleted list instead "
                 "(e.g. '- [01-topic](references/01-topic.md) — description')")]
    if not has_bullets:
        return [("WARN",
                 "## References section does not contain a bulleted list of reference links; "
                 "expected lines like '- [01-topic](references/01-topic.md) — description' "
                 "or '- [Title](https://example.com) — description")]
    return [("PASS", "references section format is correct (or absent)")]


def _check_reference_files(skill_dir):
    """Check that files in references/ follow NN-topic.md naming with
    sequential, gap-free prefixes.
    """
    ref_dir = os.path.join(skill_dir, 'references')
    if not os.path.isdir(ref_dir):
        return [("PASS", "reference files follow NN-topic.md naming (or absent)")]

    files = sorted(
        e for e in os.listdir(ref_dir)
        if os.path.isfile(os.path.join(ref_dir, e))
    )
    if not files:
        return [("WARN", "references/ directory is empty")]

    prefix_re = re.compile(r'^(\d{2,})-(.+\.md)$')
    found, bad = [], []
    for fname in files:
        m = prefix_re.match(fname)
        if m:
            found.append(int(m.group(1)))
        else:
            bad.append(fname)

    found.sort()
    result = []
    if bad:
        result.append(("WARN",
                       f"references/ file(s) missing numeric prefix: "
                       f"{', '.join(bad)} (expected NN-topic.md)"))

    if found:
        expected = list(range(1, len(found) + 1))
        if found != expected:
            seen, gaps, dups = set(), [], []
            for p in found:
                if p in seen:
                    dups.append(f"{p:02d}")
                seen.add(p)
            gaps = [f"{m:02d}" for m in expected if m not in seen]

            detail = []
            if gaps:
                detail.append(f"missing gaps: {', '.join(gaps)}")
            if dups:
                detail.append(f"duplicate prefixes: {', '.join(dups)}")
            if found[-1] > len(found):
                detail.append(
                    f"last prefix {found[-1]:02d} but only "
                    f"{len(found)} files (expected {len(found):02d})"
                )
            result.append(("WARN",
                           f"references/ prefixes not sequential ({', '.join(detail)}); "
                           f"found {[f'{p:02d}' for p in found]}, "
                           f"expected {[f'{e:02d}' for e in expected]}"))

    return result or [("PASS", "reference files follow NN-topic.md naming (or absent)")]


def _check_script_permissions(skill_dir, fm_name):
    """Check that direct-entry scripts are executable, if present.

    Checks scripts/<name>.py (default mode, shebang entry point) and
    scripts/<name>.sh (shell mode wrapper). Underscore-prefixed scripts
    (_<name>.py) are invoked via an interpreter, so their bit is not
    checked.
    """
    if not fm_name:
        return []
    results = []
    for ext in ('.py', '.sh'):
        path = os.path.join(skill_dir, 'scripts', f'{fm_name}{ext}')
        if not os.path.isfile(path):
            continue
        if os.access(path, os.X_OK):
            results.append(("PASS", f"scripts/{fm_name}{ext} is executable"))
        else:
            results.append(
                ("WARN", f"scripts/{fm_name}{ext} is not executable (run chmod +x)")
            )
    return results


def _check_script_usage_refs(body, fm_name):
    """Check that the body references '<name>.sh' rather than './<name>.sh'.

    Scans outside fenced code blocks, where './<name>.sh' paths are
    often structural (directory trees) rather than invocational.
    """
    if not fm_name:
        return [("PASS", "script usage references use '<name>.sh' format")]
    base_name, _ = _strip_version_suffix(fm_name)

    in_fence = False
    bad_lines = set()
    for lineno, line in enumerate(body.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for candidate in (fm_name, base_name):
            if f'./{candidate}.sh' in stripped:
                bad_lines.add(lineno)

    if bad_lines:
        lines_str = ', '.join(f"line {l}" for l in sorted(bad_lines))
        return [("WARN",
                 f"script usage reference(s) use './<name>.sh' instead of "
                 f"'{fm_name}.sh' ({lines_str})")]
    return [("PASS", "script usage references use '<name>.sh' format")]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_skill_md(path):
    """Given a path (file or dir), return the absolute path to SKILL.md."""
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        return abs_path
    if os.path.isdir(abs_path):
        candidate = os.path.join(abs_path, 'SKILL.md')
        if os.path.isfile(candidate):
            return candidate
    return None


def _discover_skill_dirs(collection_dir):
    """Find all subdirectories in collection_dir that contain a SKILL.md.

    Returns a sorted list of absolute paths to the skill directories.
    """
    skills = []
    if not os.path.isdir(collection_dir):
        return skills
    for entry in sorted(os.listdir(collection_dir)):
        candidate = os.path.join(collection_dir, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 'SKILL.md')):
            skills.append(candidate)
    return skills


def _strip_version_suffix(dir_basename):
    """Strip trailing -<version> suffix from directory basename.

    Returns (name, version_with_hyphens_or_None).
    e.g. 'demo-skill-2-4-1' -> ('demo-skill', '2-4-1')
         'skman'       -> ('skman', None)
    """
    m = VERSION_SUFFIX_RE.search(dir_basename)
    if m:
        return dir_basename[:m.start()], m.group(1)
    return dir_basename, None


def _dir_version_to_dots(version_with_hyphens):
    """Convert hyphen-separated version to dot-separated (e.g. '0-11-19' -> '0.11.19')."""
    return version_with_hyphens.replace('-', '.')


def _count_results(results):
    """Return (error_count, warn_count) for a results list."""
    errors = sum(1 for label, _ in results if label == "ERROR")
    warns = sum(1 for label, _ in results if label == "WARN")
    return errors, warns


def _die(msg):
    """Print *msg* to stderr and exit 1."""
    print(msg, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def cmd_create(args):
    """Scaffold a new skill directory."""
    name = args.name.strip()
    description = args.description.strip()
    version = (args.version or '').strip() or None
    url = (args.url or '').strip() or None

    # Extract version from URL if provided
    if url:
        try:
            _, url_version = _extract_name_version(url)
            if url_version and not version:
                version = url_version
                print(f"create: extracted version '{version}' from URL")
        except ValueError as e:
            print(f"create: regex extraction failed ({e}), trying LLM fallback…",
                  file=sys.stderr)
            if not version:
                llm_version = _llm_extract_version(url, name)
                if llm_version:
                    version = llm_version
                    print(f"create: LLM extracted version '{version}'")

    # Validate before creating anything
    errors = []
    errors.extend(_validate_name(name))
    errors.extend(_validate_description(description))
    # Keep frontmatter colon-free — colons in text values break YAML parsing
    errors.extend(_validate_text_fields({'description': description}))
    if errors:
        print("create: validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    # Build directory name and H1 title
    if version:
        dir_name = f"{name}-{version.replace('.', '-')}"
        h1_title = f"{name} {version.replace('-', '.')}"
    else:
        dir_name = name
        h1_title = name

    skill_dir = os.path.join(args.output_dir, dir_name)
    os.makedirs(skill_dir, exist_ok=True)

    skill_md_path = os.path.join(skill_dir, 'SKILL.md')
    if os.path.exists(skill_md_path):
        _die(f"create: {skill_md_path} already exists — skipping")

    # Build SKILL.md content: frontmatter (via PyYAML) + body
    shell_mode = args.lang == 'bash' or args.shell
    if args.with_scripts and shell_mode:
        body = BODY_TEMPLATE_WITH_SHELL_SCRIPT.format(title=h1_title, script_name=name)
    elif args.with_scripts:
        body = BODY_TEMPLATE_WITH_PYTHON_SCRIPT.format(title=h1_title, script_name=name)
    else:
        body = BODY_TEMPLATE.format(title=h1_title)

    frontmatter = _build_frontmatter_yaml({'name': dir_name, 'description': description})
    with open(skill_md_path, 'w') as f:
        f.write(frontmatter)
        f.write(body)

    # Create scripts
    if args.with_scripts:
        scripts_dir = os.path.join(skill_dir, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)

        if shell_mode:
            # Shell + Python: scripts/<name>.sh + scripts/_<name>.py
            # Python has no PyPI deps — bash calls it directly
            py_path = os.path.join(scripts_dir, f'_{name}.py')
            with open(py_path, 'w') as f:
                f.write(textwrap.dedent(f'''
                    #!/usr/bin/env python3

                    """{name} — {description}

                    Usage:
                        {name}.sh <subcommand> [args...]
                    """

                    import argparse
                    import sys


                    def main():
                        parser = argparse.ArgumentParser(prog="{name}")
                        parser.parse_args()
                        # TODO: implement {name}


                    if __name__ == "__main__":
                        main()
                    ''').lstrip('\n'))
            os.chmod(py_path, 0o755)
            print(f"create: created Python script at {py_path}")

            sh_path = os.path.join(scripts_dir, f'{name}.sh')
            with open(sh_path, 'w') as f:
                f.write(f'#!/usr/bin/env bash\n')
                f.write(f'# {name} — {description}\n')
                f.write(f'set -euo pipefail\n')
                f.write(f'\n')
                f.write(f'SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\n')
                f.write(f'\n')
                f.write(f'exec python3 "$SCRIPT_DIR/_{name}.py" "$@"\n')
            os.chmod(sh_path, 0o755)
            print(f"create: created bash wrapper at {sh_path}")

        else:
            # Default: scripts/<name>.py with PEP 723 shebang
            py_path = os.path.join(scripts_dir, f'{name}.py')
            with open(py_path, 'w') as f:
                f.write(textwrap.dedent(f'''
                    #!/usr/bin/env -S uv run --script
                    #
                    # /// script
                    # requires-python = ">=3.12"
                    # dependencies = [
                    #     # add dependencies here
                    # ]
                    # ///

                    """{name} — {description}

                    Usage:
                        {name}.py <subcommand> [args...]
                    """

                    import argparse
                    import sys


                    def main():
                        parser = argparse.ArgumentParser(prog="{name}")
                        parser.parse_args()
                        # TODO: implement {name}


                    if __name__ == "__main__":
                        main()
                    ''').lstrip('\n'))
            os.chmod(py_path, 0o755)
            print(f"create: created Python script at {py_path}")

    # Optionally create references directory
    if args.with_references:
        ref_dir = os.path.join(skill_dir, 'references')
        os.makedirs(ref_dir, exist_ok=True)
        placeholder = os.path.join(ref_dir, '01-reference.md')
        with open(placeholder, 'w') as f:
            f.write(f"# {h1_title} Reference\n\n[Detailed reference content.]\n")
        print(f"create: created references placeholder at {placeholder}")

    print(f"create: scaffolded skill '{dir_name}' at {skill_md_path}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _validate_single_skill(skill_path):
    """Validate one skill. Returns a list of (label, message) tuples."""
    skill_md = _find_skill_md(skill_path)
    if skill_md is None:
        return [("ERROR", "no SKILL.md found")]

    with open(skill_md, 'r') as f:
        content = f.read()

    fm, body, yaml_error = _parse_frontmatter(content)

    results = []
    skill_dir = os.path.dirname(skill_md)
    dir_basename = os.path.basename(skill_dir)
    dir_name, dir_version = _strip_version_suffix(dir_basename)

    # --- Frontmatter ---
    if fm is None:
        if yaml_error is not None:
            results.append(("ERROR", f"frontmatter is not valid YAML — {yaml_error}"))
        else:
            results.append(("ERROR", "no YAML frontmatter found (must start with ---)"))
    else:
        results.append(("PASS", "frontmatter is valid YAML"))

        # Name
        name_errors = _validate_name(fm.get('name', ''))
        if name_errors:
            results.extend(("ERROR", f"name: {e}") for e in name_errors)
        else:
            results.append(("PASS", f"name '{fm.get('name', '')}' is valid"))

        # Description
        desc_errors = _validate_description(fm.get('description'))
        if desc_errors:
            results.extend(("ERROR", f"description: {e}") for e in desc_errors)
        else:
            results.append(("PASS", "description is valid"))

        # Unknown fields
        known_fields = {'name', 'description', 'license', 'compatibility', 'metadata'}
        unknown = set(fm) - known_fields
        if unknown:
            results.append(("WARN", f"unknown frontmatter fields: {', '.join(sorted(unknown))}"))
        else:
            results.append(("PASS", "no unknown frontmatter fields"))

        # Text-only fields must not contain ':'
        text_errors = _validate_text_fields(fm)
        if text_errors:
            results.extend(("ERROR", e) for e in text_errors)
        else:
            results.append(("PASS", "text-only fields contain no ':' characters"))

        # Duplicate top-level keys
        dup_errors = _check_duplicate_frontmatter_keys(content)
        if dup_errors:
            results.extend(("ERROR", e) for e in dup_errors)
        else:
            results.append(("PASS", "no duplicate frontmatter keys"))

        # Compatibility (optional)
        compat = fm.get('compatibility')
        if compat is not None:
            compat_errors = _validate_compatibility(compat)
            if compat_errors:
                results.extend(("ERROR", f"compatibility: {e}") for e in compat_errors)
            else:
                results.append(("PASS", "compatibility is valid"))

        # Metadata (optional)
        metadata = fm.get('metadata')
        if metadata is not None:
            meta_warnings = _validate_metadata(metadata)
            if meta_warnings:
                results.extend(("WARN", f"metadata: {w}") for w in meta_warnings)
            else:
                results.append(("PASS", "metadata structure is valid"))

        # Name vs directory basename consistency
        fm_name = fm.get('name', '')
        if fm_name and fm_name != dir_basename:
            results.append((
                "WARN",
                f"directory name '{dir_basename}' does not match "
                f"frontmatter name '{fm_name}' (expected '{dir_basename}')"
            ))
        else:
            results.append(("PASS", f"directory name matches frontmatter name '{fm_name}'"))

        # Script checks
        results.extend(_check_script_permissions(skill_dir, fm_name))
        results.extend(_check_script_usage_refs(body, fm_name))

    # --- Body ---
    first_content_line = next(
        (line.strip() for line in body.splitlines() if line.strip()), None
    )
    if first_content_line is None:
        results.append(("PASS", "body is empty (no content lines)"))
    elif not first_content_line.startswith('# '):
        results.append((
            "ERROR",
            f"body must start with a level-1 heading (found: '{first_content_line[:60]}...')"
        ))
    else:
        expected_h1 = dir_name
        if dir_version:
            expected_h1 = f"{dir_name} {_dir_version_to_dots(dir_version)}"
        if first_content_line[2:] == expected_h1:
            results.append(("PASS", f"H1 heading '{first_content_line}' matches expected format"))
        else:
            results.append((
                "ERROR",
                f"H1 heading '{first_content_line}' does not match expected "
                f"'#{expected_h1}' (must be '# <name>' or '# <name> <version>')"
            ))

    # Token estimation
    estimated_tokens = _estimate_tokens(body)
    if estimated_tokens > 5000:
        results.append(("WARN", f"body is ~{estimated_tokens} tokens (recommended: under 5000)"))
    else:
        results.append(("PASS", f"body is ~{estimated_tokens} tokens (under 5000)"))

    # Sections, references, reference files
    results.extend(_check_sections(body))
    results.extend(_check_references_section(body))
    results.extend(_check_reference_files(skill_dir))

    return results


def _print_single_results(results, strict):
    """Print enumerated results; return the exit code."""
    error_count, warn_count = _count_results(results)

    if error_count:
        print("validate: FAILED")
    elif warn_count:
        print("validate: OK (with warnings)")
    else:
        print("validate: OK")

    counter = 0
    for label in ("PASS", "WARN", "ERROR"):
        for rlabel, msg in results:
            if rlabel != label:
                continue
            counter += 1
            print(f"  {counter}. [{label}] {msg}")

    passed = error_count == 0 and (warn_count == 0 or not strict)
    return 0 if passed else 1


def _validate_collection(skill_dirs, strict):
    """Validate all skills in a collection directory.

    Prints per-skill results and a summary. Exits 0 if all pass, 1 otherwise.
    """
    total = len(skill_dirs)
    passed_count = failed_count = warned_count = 0
    total_errors = total_warnings = 0

    for i, skill_dir in enumerate(skill_dirs, 1):
        print(f"{'=' * 60}")
        print(f"  {i}/{total}: {os.path.basename(skill_dir)}")
        print(f"{'=' * 60}")

        results = _validate_single_skill(skill_dir)
        error_count, warn_count = _count_results(results)
        skill_passed = error_count == 0 and (warn_count == 0 or not strict)

        if not skill_passed:
            status = "FAILED"
            failed_count += 1
        elif warn_count:
            status = "OK (with warnings)"
            warned_count += 1
        else:
            status = "OK"
            passed_count += 1

        total_errors += error_count
        total_warnings += warn_count

        print(f"  Status: {status} ({error_count} error(s), {warn_count} warning(s))")
        _print_single_results(results, strict)
        print()

    # Summary
    print(f"{'=' * 60}")
    print(f"  Summary: {total} skill(s) validated")
    print(f"  Passed: {passed_count}, Warnings: {warned_count}, Failed: {failed_count}")
    print(f"  Total errors: {total_errors}, Total warnings: {total_warnings}")
    print(f"{'=' * 60}")

    sys.exit(0 if failed_count == 0 else 1)


def cmd_validate(args):
    """Validate a SKILL.md file or a collection of skills against spec rules.

    Modes: single skill (path to skill dir or SKILL.md file), collection
    directory (dir containing skill subdirs), or auto-detect — a directory
    without SKILL.md at the root but with skill subdirs validates all of them.
    """
    target = os.path.abspath(args.path)

    if os.path.isfile(target):
        if _find_skill_md(target) is None:
            _die(f"validate: no SKILL.md found at '{args.path}'")
        sys.exit(_print_single_results(_validate_single_skill(target), args.strict))

    if os.path.isdir(target):
        if _find_skill_md(target) is not None:
            # This dir has SKILL.md — single skill mode
            sys.exit(_print_single_results(_validate_single_skill(target), args.strict))

        skill_dirs = _discover_skill_dirs(target)
        if not skill_dirs:
            _die(f"validate: no SKILL.md found at '{args.path}' "
                 f"(not a skill dir or collection)")
        _validate_collection(skill_dirs, args.strict)
        return

    _die(f"validate: path not found '{args.path}'")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args):
    """Print parsed frontmatter and structural summary of a skill."""
    skill_md = _find_skill_md(args.path)
    if skill_md is None:
        _die(f"info: no SKILL.md found at '{args.path}'")

    with open(skill_md, 'r') as f:
        content = f.read()

    fm, body, yaml_error = _parse_frontmatter(content)

    if yaml_error is not None:
        _die(f"info: {skill_md}: invalid YAML frontmatter — {yaml_error}")

    if args.json:
        print(json.dumps(fm, indent=2, ensure_ascii=False) if fm else "{}")
        return

    if args.yaml_out:
        if fm:
            print(yaml.dump(fm, default_flow_style=False, sort_keys=False, allow_unicode=True))
        else:
            print("{}")
        return

    # Default: human-readable output
    print("info: SKILL.md analysis")
    print("-" * 40)
    if fm:
        for key in ('name', 'description'):
            val = fm.get(key)
            if val is not None:
                print(f"  {key}: {val}")
        metadata = fm.get('metadata')
        if metadata is not None:
            tags = metadata.get('tags', [])
            if tags:
                print(f"  metadata.tags: {', '.join(tags)}")
        for key in ('license', 'compatibility'):
            val = fm.get(key)
            if val is not None:
                print(f"  {key}: {val}")
        known_fields = {'name', 'description', 'license', 'compatibility', 'metadata'}
        for key in sorted(set(fm) - known_fields):
            print(f"  {key}: {fm[key]}")
    else:
        print("  (no frontmatter)")

    line_count = len(body.splitlines())
    word_count = len(body.split())
    token_count = _estimate_tokens(body)

    print(f"  body lines: {line_count}")
    print(f"  body words: {word_count}")
    print(f"  body tokens: ~{token_count}")
    headings = list(_iter_headings(body))
    if headings:
        print("  headings:")
        for h in headings:
            depth = len(h) - len(h.lstrip('#'))
            text = h.lstrip('#').strip()
            indent = "    " * (depth - 1)
            print(f"  {indent}- {text}")

    skill_dir = os.path.dirname(skill_md)
    entries = sorted(os.listdir(skill_dir))
    if entries:
        print("  files:")
        for entry in entries:
            full = os.path.join(skill_dir, entry)
            if os.path.isdir(full):
                print(f"    {entry}/")
            else:
                print(f"    {entry} ({os.path.getsize(full)} bytes)")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

# Per-block HTML comment markers. Each category table and the statistics
# section live between their own open/close pair, so generate can update
# (or create) one block without touching the others:
#   <!-- SKMAN:TABLE:<label> --> ... <!-- /SKMAN:TABLE:<label> -->
#   <!-- SKMAN:STATS -->          ... <!-- /SKMAN:STATS -->
_SKMAN_OPEN_TABLE_RE = re.compile(r'^<!-- SKMAN:TABLE:([^ ]+) -->$')
_SKMAN_CLOSE_TABLE_RE = re.compile(r'^<!-- /SKMAN:TABLE:([^ ]+) -->$')
_SKMAN_OPEN_STATS_RE = re.compile(r'^<!-- SKMAN:STATS -->$')
_SKMAN_CLOSE_STATS_RE = re.compile(r'^<!-- /SKMAN:STATS -->$')

# Display names for known category labels; unknown labels fall back to a
# capitalized form (e.g. "mytools" -> "Mytools").
_CATEGORY_DISPLAY_NAMES = {
    'core': 'Core',
    'byterefinery': 'ByteRefinery',
    'general': 'General',
    'go': 'Go',
    'javascript': 'JavaScript',
    'models': 'Models',
    'python': 'Python',
}


def _category_label(dirname):
    """Label for a collection directory name.

    'skills' -> 'core', 'skills-<x>' -> '<x>', anything else -> itself.
    """
    if dirname == 'skills':
        return 'core'
    if dirname.startswith('skills-'):
        return dirname[len('skills-'):]
    return dirname


def _category_display_name(label):
    """Human-readable category name for table headings."""
    return _CATEGORY_DISPLAY_NAMES.get(label, label.capitalize())


def _discover_categories(skills_dir):
    """Return an ordered list of (dir, label) collection pairs.

    The primary --skills-dir collection comes first; then every sibling
    collection directory named 'skills' or 'skills-*' in the same parent
    directory, sorted by name.
    """
    abs_dir = os.path.abspath(skills_dir)
    categories = [(abs_dir, _category_label(os.path.basename(abs_dir)))]
    parent = os.path.dirname(abs_dir)
    if os.path.isdir(parent):
        for entry in sorted(os.listdir(parent)):
            full = os.path.join(parent, entry)
            if full == abs_dir or not os.path.isdir(full):
                continue
            if entry == 'skills' or entry.startswith('skills-'):
                categories.append((full, _category_label(entry)))
    return categories


def _discover_skills(skills_dir):
    """Return a sorted list of (name, description) for all skills in
    skills_dir. name defaults to the directory basename, description to
    the frontmatter 'description' field.
    """
    skills = []
    for skill_dir in _discover_skill_dirs(skills_dir):
        skill_md = os.path.join(skill_dir, 'SKILL.md')
        with open(skill_md, 'r') as f:
            content = f.read()
        fm, _, yaml_error = _parse_frontmatter(content)
        if fm is None and yaml_error is not None:
            print(f"generate: warning: invalid YAML frontmatter in {skill_md} — "
                  f"{yaml_error} (using directory name)", file=sys.stderr)
        name = os.path.basename(skill_dir)
        desc = ''
        if fm:
            name = fm.get('name', name)
            desc = fm.get('description', '') or ''
        # YAML block scalars (description: >) keep a trailing newline (and
        # literal | keeps internal ones); collapse all whitespace so the
        # description always fits on a single table row.
        desc = ' '.join(str(desc).split())
        skills.append((name, desc))
    return skills


def _build_table(skills, heading):
    """Build a category table markdown under *heading* from a list of
    (name, description)."""
    lines = [f"## {heading}", "", "| No | Skill | Description |", "|----|-------|-------------|"]
    for i, (name, desc) in enumerate(skills, 1):
        lines.append(f"| {i} | {name} | {desc} |")
    return "\n".join(lines)


def _build_statistics(per_category):
    """Build the Statistics section markdown.

    per_category is a list of (label, count) pairs; the total is summed.
    """
    total = sum(count for _, count in per_category)
    lines = ["## Statistics", "", "| Category | Skills |", "|----------|--------|"]
    for label, count in per_category:
        lines.append(f"| {label} | {count} |")
    lines += ["", f"- **Total Skills**: {total}"]
    return "\n".join(lines)


def _resolve_only(categories, only_arg):
    """Resolve a --only argument to a (dir, label) pair from *categories*.

    Accepts a collection path, a directory basename ('skills-python'),
    a category label ('python'), or a display name ('Python'). Returns
    None if nothing matches.
    """
    arg_abs = os.path.abspath(only_arg)
    arg_base = os.path.basename(only_arg.rstrip('/'))
    for cat_dir, label in categories:
        if os.path.abspath(cat_dir) == arg_abs:
            return (cat_dir, label)
    for cat_dir, label in categories:
        if os.path.basename(cat_dir) == arg_base:
            return (cat_dir, label)
    for cat_dir, label in categories:
        if label == only_arg or _category_display_name(label) == only_arg:
            return (cat_dir, label)
    return None


def _parse_blocks(readme):
    """Split *readme* into (prefix, blocks, suffix).

    blocks is a list of (label, content) pairs in document order — label
    is the category label for table blocks, None for the stats block;
    content is the text between the marker pair (no surrounding
    newlines). prefix is everything before the first opening marker,
    suffix everything after the last closing marker. An unterminated
    block extends to EOF; stray closing markers stay plain text.
    """
    lines = readme.splitlines(keepends=True)
    prefix_end = None
    suffix_start = len(lines)
    blocks = []
    in_block = False
    open_label = None
    content_start = None

    for i, line in enumerate(lines):
        stripped = line.rstrip('\n')
        if not in_block:
            m = _SKMAN_OPEN_TABLE_RE.match(stripped)
            if m or _SKMAN_OPEN_STATS_RE.match(stripped):
                in_block = True
                open_label = m.group(1) if m else None
                content_start = i + 1
                if prefix_end is None:
                    prefix_end = i
                continue
        else:
            if open_label is None:
                if _SKMAN_CLOSE_STATS_RE.match(stripped):
                    in_block = False
                    blocks.append((None, ''.join(lines[content_start:i]).strip('\n')))
                    suffix_start = i + 1
            else:
                m = _SKMAN_CLOSE_TABLE_RE.match(stripped)
                if m and m.group(1) == open_label:
                    in_block = False
                    blocks.append((open_label, ''.join(lines[content_start:i]).strip('\n')))
                    suffix_start = i + 1
    if in_block:
        blocks.append((open_label, ''.join(lines[content_start:]).strip('\n')))
        suffix_start = len(lines)

    if prefix_end is None:
        return readme, blocks, ''
    return ''.join(lines[:prefix_end]), blocks, ''.join(lines[suffix_start:])


def _render_blocks(blocks):
    """Render (label, content) pairs as marked blocks joined by blank
    lines. No trailing newline."""
    parts = []
    for label, content in blocks:
        if label is None:
            parts.append(f"<!-- SKMAN:STATS -->\n{content}\n<!-- /SKMAN:STATS -->")
        else:
            parts.append(
                f"<!-- SKMAN:TABLE:{label} -->\n{content}\n<!-- /SKMAN:TABLE:{label} -->"
            )
    return '\n\n'.join(parts)


def _assemble_readme(prefix, blocks_text, suffix):
    """Join prefix + generated blocks + suffix with single blank lines
    and a single trailing newline."""
    out = prefix.rstrip('\n')
    if out:
        out += '\n\n'
    out += blocks_text
    s = suffix.strip('\n')
    if s:
        out += '\n\n' + s
    return out + '\n'


def cmd_generate(args):
    """Generate per-category Skills Tables and Statistics in README.md.

    Scans the --skills-dir collection plus every sibling skills/skills-*
    collection in the same parent directory. By default every non-empty
    collection's table is (re)generated and obsolete table blocks are
    removed. With --only, just that collection's table is refreshed (its
    markers created if missing) and the other tables are kept as-is;
    statistics are recomputed from all collections in both modes.

    Tables and statistics live between per-block HTML comment markers
    ('<!-- SKMAN:TABLE:<label> -->' / '<!-- SKMAN:STATS -->') which are
    replaced in place — new skills-* directories get new marker blocks
    automatically (core first, then alphabetical).
    """
    categories = _discover_categories(args.skills_dir)

    only = None
    if args.only:
        only = _resolve_only(categories, args.only)
        if only is None:
            known = ', '.join(
                f'{label} ({os.path.basename(cat_dir)})'
                for cat_dir, label in categories
            )
            _die(f"generate: --only '{args.only}' does not match any collection "
                 f"(available: {known})")

    fresh = {}
    per_category = []
    order = []
    for i, (cat_dir, label) in enumerate(categories):
        skills = _discover_skills(cat_dir)
        if not skills:
            continue
        heading = 'Skills Table' if i == 0 else f'{_category_display_name(label)} Skills'
        fresh[label] = _build_table(skills, heading)
        per_category.append((label, len(skills)))
        order.append(label)

    if not fresh:
        _die(
            f"generate: no skills found in '{args.skills_dir}' "
            f"or in sibling skills-*/ collections"
        )

    if not os.path.isfile(args.readme):
        _die(f"generate: README not found at '{args.readme}'")
    with open(args.readme, 'r') as f:
        readme = f.read()

    prefix, existing, suffix = _parse_blocks(readme)
    existing_map = {}
    for label, content in existing:
        existing_map.setdefault(label, content)

    if only is None:
        blocks = [(label, fresh[label]) for label in order]
    else:
        only_label = only[1]
        if only_label not in fresh:
            _die(f"generate: no skills found in '{args.only}' — nothing to regenerate")
        blocks = []
        for label in order:
            if label == only_label:
                blocks.append((label, fresh[label]))
            elif label in existing_map:
                blocks.append((label, existing_map[label]))
    blocks.append((None, _build_statistics(per_category)))

    new_readme = _assemble_readme(prefix, _render_blocks(blocks), suffix)
    with open(args.readme, 'w') as f:
        f.write(new_readme)

    total = sum(count for _, count in per_category)
    new_blocks = [label for label, _ in blocks
                  if label is not None and label not in existing_map]
    msg = f"generate: updated {args.readme}"
    if only is None:
        msg += f" with {len(order)} table(s)"
    else:
        msg += f" ({only_label} table)"
    msg += f", {total} skills total"
    if new_blocks:
        msg += f", new marker block(s): {', '.join(new_blocks)}"
    print(msg)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog='skman',
        description=textwrap.dedent("""\
            Skill Manager — scaffold, validate, and inspect agent skills.

            Subcommands:
              create      Scaffold a new skill directory with SKILL.md
              validate    Check SKILL.md against spec rules
              info        Print frontmatter and structural summary
              generate    Generate per-category Skills Tables and Statistics in README.md

            Use '<subcommand> --help' for details on each subcommand.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='subcommand')

    # --- create ---
    p_create = sub.add_parser(
        'create',
        description=textwrap.dedent("""\
            Scaffold a new skill directory with SKILL.md and optional scripts/references.

            Script modes:
              --with-scripts              Default: scripts/<name>.py (PEP 723, needs uv)
              --with-scripts --lang bash  Shell: scripts/<name>.sh + scripts/_<name>.py
              --with-scripts --shell      Same as --lang bash

            Examples:
              skman.py create my-skill "Does X and Y"
              skman.py create my-skill "Desc" --with-scripts --with-references
              skman.py create my-skill "Desc" --with-scripts --lang bash
              skman.py create my-skill "Desc" -o ./custom-skills
              skman.py create demo-skill "Dummy example skill" --version 2.4.1
              skman.py create numpy "NumPy skill" --url https://github.com/numpy/numpy/releases/tag/v1.26.0
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_create.add_argument('name', help='Skill name (lowercase, hyphens, numbers)')
    p_create.add_argument('description', help='Skill description (max 1024 chars)')
    p_create.add_argument(
        '--url',
        default=None,
        help='Source URL (GitHub, PyPI, npm, crates.io, etc.). Auto-extracts name and version.',
    )
    p_create.add_argument(
        '--version', '-V',
        default=None,
        help='Optional version (e.g. 0.11.19). Dir becomes <name>-<version>, H1 is "# <name> <version>"',
    )
    p_create.add_argument(
        '--output-dir', '-o',
        default='.agents/skills',
        help='Parent directory for the skill (default: .agents/skills)',
    )
    p_create.add_argument(
        '--with-scripts',
        action='store_true',
        help='Create scripts/ directory. Default: <name>.py (PEP 723). Use --lang bash for shell wrapper.',
    )
    p_create.add_argument(
        '--lang',
        default=None,
        choices=['python', 'bash'],
        help='Script language: python (default, PEP 723), bash (shell + _<name>.py)',
    )
    p_create.add_argument(
        '--shell',
        action='store_true',
        help='Alias for --lang bash: create <name>.sh + _<name>.py',
    )
    p_create.add_argument(
        '--with-references',
        action='store_true',
        help='Also create a references/ directory with placeholder',
    )

    # --- validate ---
    p_validate = sub.add_parser(
        'validate',
        description=textwrap.dedent("""\
            Validate a SKILL.md file against the agent skills spec.

            Supports three modes:
              - Single skill: path to a skill directory or SKILL.md file
              - Collection: path to a directory containing skill subdirectories
              - Auto-detect: if the path is a directory without SKILL.md but
                contains subdirs with SKILL.md, it validates all of them

            Checks:
              - Frontmatter presence, valid YAML, no duplicate keys
              - Text-only fields (name, description, license, compatibility)
                contain no ':' character (error)
              - Name format (lowercase, hyphens, length)
              - Description presence, length, no XML/HTML tags
              - Body starts with a matching H1 heading
              - Body token estimate (warning if over 5000)
              - Recommended sections, references format, script permissions

            Examples:
              skman.py validate ./my-skill
              skman.py validate ./my-skill/SKILL.md
              skman.py validate --strict ./my-skill
              skman.py validate .agents/skills        # validate all skills
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_validate.add_argument('path', help='Path to skill directory, SKILL.md file, or skills collection directory')
    p_validate.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors (non-zero exit on any warning)',
    )

    # --- info ---
    p_info = sub.add_parser(
        'info',
        description=textwrap.dedent("""\
            Print frontmatter and structural summary of a skill.

            Shows:
              - Parsed frontmatter fields
              - Body line/word count
              - Heading outline
              - Directory listing with file sizes

            Use --json or --yaml for structured output.

            Examples:
              skman.py info ./my-skill
              skman.py info --json ./my-skill
              skman.py info --yaml ./my-skill/SKILL.md
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_info.add_argument('path', help='Path to skill directory or SKILL.md file')
    p_info.add_argument('--json', action='store_true', help='Output frontmatter as JSON')
    p_info.add_argument('--yaml', dest='yaml_out', action='store_true', help='Output frontmatter as YAML')

    # --- generate ---
    p_generate = sub.add_parser(
        'generate',
        description=textwrap.dedent("""\
            Generate the per-category Skills Tables and Statistics in README.md.

            Scans the --skills-dir collection plus every sibling skills/skills-*
            collection directory in the same parent directory. By default every
            non-empty collection's table is (re)generated; with --only just that
            collection's table is refreshed (its markers created if missing) and
            the other tables are kept as-is. --only accepts a category label
            ('python'), a directory basename ('skills-python'), or a path. ##
            Statistics is always recomputed from all collections.

            Tables and statistics live in README.md between per-block HTML
            comment markers ('<!-- SKMAN:TABLE:<label> -->' and
            '<!-- SKMAN:STATS -->'); generate replaces block contents in place
            and adds marker blocks automatically for new skills-* directories.

            Examples:
              skman.py generate
              skman.py generate --only python
              skman.py generate --only .agents/skills-python
              skman.py generate --skills-dir ./custom-skills
              skman.py generate --readme ./docs/README.md
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_generate.add_argument(
        '--skills-dir',
        default='.agents/skills',
        help='Directory containing skill subdirectories (default: .agents/skills)',
    )
    p_generate.add_argument(
        '--readme',
        default='README.md',
        help='Path to README.md to update (default: README.md)',
    )
    p_generate.add_argument(
        '--only',
        default=None,
        metavar='CATEGORY',
        help="Regenerate only this collection's table (label 'python', basename "
             "'skills-python', or path); other tables kept, Statistics recomputed",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        'create': cmd_create,
        'validate': cmd_validate,
        'info': cmd_info,
        'generate': cmd_generate,
    }
    dispatch[args.subcommand](args)


if __name__ == '__main__':
    main()
