# Repository Analysis for Skill Creation

Detailed source-study workflow backing "Creating a skill from a repository" in SKILL.md. Documentation is the primary source; source code is the fallback.

## 1. Clone Locally

```bash
git clone --depth 1 <url> /tmp/<name>
```

- Shallow clone (`--depth 1`) is enough — skill creation needs the current tree, not history.
- Use the repo name as the temp directory so it is easy to find and clean up: `rm -rf /tmp/<name>`.
- For non-git sources (tarball, PyPI sdist), extract into a temp directory instead.

## 2. Locate Documentation

Run from the repo root; exclude vendored and build noise:

```bash
# Doc-like directories
find . -maxdepth 3 -type d \
  \( -iname docs -o -iname doc -o -iname documentation \
     -o -iname manual -o -iname guides -o -iname wiki \
     -o -iname tutorials \) \
  -not -path '*/node_modules/*' -not -path '*/.git/*'

# Root-level documentation files
find . -maxdepth 2 -type f \
  \( -iname 'readme*' -o -iname 'install*' -o -iname 'changelog*' \
     -o -iname 'news*' -o -iname 'contributing*' \) \
  -not -path '*/node_modules/*' -not -path '*/.git/*'

# Remaining .md/.rst/.txt outside obvious source trees
find . -maxdepth 3 -type f \( -iname '*.md' -o -iname '*.rst' -o -iname '*.txt' \) \
  -not -path '*/node_modules/*' -not -path '*/.git/*' \
  -not -path '*/src/*' -not -path '*/lib/*'
```

### Typical Locations

| Kind | Where |
|---|---|
| Manual / guide | `docs/`, `doc/`, `documentation/`, `manual/`, `guides/`, `wiki/`, `tutorials/` |
| README | `README.md`, `README.rst`, `README.txt` at root |
| Install / setup | `INSTALL`, `SETUP.md`, `getting-started.md` |
| Change history | `CHANGELOG*`, `NEWS*`, `HISTORY*`, `CHANGES*` |
| Contributing / dev | `CONTRIBUTING*`, `DEVELOPMENT.md` (useful for internal behavior) |
| Sphinx / Docutils | `docs/*.rst` with `conf.py` (Python projects), `doc/` trees |

## 3. Mine Documentation (docs present)

Read the doc tree and extract, mapping directly onto skill sections:

- **What it does** — README intro, overview pages → `## Overview`
- **Usage patterns** — quickstart, tutorials, examples → `## Usage`
- **Commands and options** — CLI reference pages, man-page-style docs → `## Usage`
- **Configuration** — config-file references, env vars → `## Usage` or a reference file
- **Workflows** — step-by-step recipes → `## Usage` or reference files
- **Pitfalls** — FAQ, troubleshooting, known-issues pages → `## Gotchas`

Open source code only to verify specifics the docs leave ambiguous: exact flag names, version-dependent behavior, undocumented defaults.

## 4. Whole-Repo Analysis (no docs)

If step 2 finds nothing useful, derive everything from source, in this order:

1. **Entry points** — `main`/`__main__` modules, `bin/`, CLI definitions (argparse, click, clap, cobra…), shebang scripts
2. **Public API** — exported symbols, `__init__.py` exports, `public/` packages
3. **Options and configuration** — argument parsers, config schemas, env-var reads
4. **Tests** — test files show real usage patterns and edge cases
5. **Packaging** — `pyproject.toml`, `package.json`, `Cargo.toml` reveal entry points, dependencies, and version

Extract the same material as from docs: what it does, how to use it, options, traps.

## 5. Clean Up

```bash
rm -rf /tmp/<name>
```
