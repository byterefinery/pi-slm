# Python Scripts with uv

## PEP 723 Inline Dependencies

Python scripts use the PEP 723 metadata block to declare dependencies inline. `uv run` reads this block and resolves dependencies automatically — no `pip install`, no `requirements.txt`, no manual venv.

### Block Format

```python
#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["PyYAML", "requests"]
# ///

import yaml
import requests
```

The block is enclosed between `# /// script` and `# ///`. Fields:

- **`requires-python`** — minimum Python version (e.g., `">=3.12"`, `">=3.10,<4"`)
- **`dependencies`** — list of package specs, same syntax as `pyproject.toml` (e.g., `["requests", "rich>=12,<13"]`)
- **`[tool.uv]`** — optional uv-specific settings (indexes, sources, etc.)

### Shebang vs Non-Shebang

**With shebang** — script is directly executable:

```python
#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

import httpx
print(httpx.get("https://example.com"))
```

```bash
chmod +x my-script
./my-script
```

**Without shebang** — invoke via `uv run`:

```bash
uv run my-script.py
```

Or with explicit Python version:

```bash
uv run --python 3.12 my-script.py
```

### Invocation Patterns

| Pattern | Use case |
|---------|----------|
| `#!/usr/bin/env -S uv run --script` + `chmod +x` | Standalone executable on PATH |
| `uv run script.py` | Run from project/skill directory |
| `uv run --with requests script.py` | Ad-hoc dependency added at runtime |
| `uvx tool-name` | Ephemeral tool run (isolated from projects) |

### Bash Wrapper Pattern

For skills, the bash entry point delegates to the Python script:

```bash
#!/usr/bin/env bash
# my-skill — Description
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run "$SCRIPT_DIR/my-skill.py" "$@"
```

This keeps `.sh` as the canonical entry point referenced in SKILL.md while the Python script handles the actual logic with managed dependencies.

### Gotchas

- **Block must be at the top** — `uv run` only reads the PEP 723 block from the start of the file. Comments or docstrings before it break detection.
- **Dependencies are per-script** — each script gets its own ephemeral environment. Dependencies are not shared across scripts or projects.
- **`uv run` auto-downloads Python** — if `requires-python` specifies a version not available locally, uv downloads it automatically. Disable with `UV_PYTHON_DOWNLOADS=never`.
- **Shebang requires `uv` on PATH** — the `#!/usr/bin/env -S uv run --script` shebang only works if `uv` is installed and accessible. In environments without `uv`, fall back to `uv run script.py` (invoked from a wrapper that ensures `uv` is available).
- **No `pip` compatibility needed** — the PEP 723 block is understood by `uv` directly. It is not a `pip` feature.

### Reference

- [PEP 723 — Inline script metadata](https://peps.python.org/pep-0723/)
- [uv guides/scripts](https://docs.astral.sh/uv/guides/scripts/)
