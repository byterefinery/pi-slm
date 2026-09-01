#!/usr/bin/env python3
"""Parse dspy evaluation log files into per-example CSV files.

Log structure
-------------
Each example produces one tqdm progress bar ``C/T`` followed by either::

    result=Prediction(
        quality='high',
        feedback='...'
    )

or, when the example failed (``ERROR dspy.utils.parallelizer`` + Python
traceback, which we do not extract), nothing at all.  The cumulative score
printed by dspy is prefixed to the *next* progress bar line::

    Average Metric: 1.50 / 2 (75.0%):   0%|...| 2/2247 [00:14<..., ...s/it]...

so the cumulative value *after example C* is read from the prefix of bar C+1
(the final bar ``T/T`` carries the overall cumulative value).

CSV columns:
    current, total, quality, feedback, is_error, metric,
    avg_metric, avg_metric_percent

Usage:
    python parse_logs.py [LOG ...]     # default: the two eval-*.log files here
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

# tqdm bar: `P%|<bar chars>| C/T [elapsed<remaining, rate]` (bar has no `|`)
BAR_RE = re.compile(r"\d+%\|[^|\n]*\|\s*(\d+)/(\d+)\s*\[[^\]]*\]")

# cumulative prefix `Average Metric: SUM / COUNT (PCT%):` (INFO lines match too,
# but they are never adjacent to a bar)
PREFIX_RE = re.compile(
    r"Average Metric:\s*([\d.]+)\s*/\s*(\d+)\s*\(\s*([\d.]+)%\s*\):"
)

# a Python string literal (single- or double-quoted, backslash escapes allowed)
STR_RE = r"((?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"))"

# `result=Prediction(\n    quality='...',\n    feedback='...'\n)` — the closing
# `)` may be separated from the feedback literal by spaces or newlines
PRED_RE = re.compile(
    r"result=Prediction\(\s*quality=" + STR_RE + r"\s*,\s*feedback=" + STR_RE
    + r"\s*\)",
    re.DOTALL,
)

FIELDNAMES = [
    "current",
    "total",
    "quality",
    "feedback",
    "is_error",
    "metric",
    "avg_metric",
    "avg_metric_percent",
]


def quality_score(quality: str) -> float:
    match quality:
        case "very low":
            return 0.0
        case "low":
            return 0.25
        case "medium":
            return 0.5
        case "high":
            return 0.75
        case "very high":
            return 1.0
        case _:
            return 0.0


def parse_log(text: str) -> list[dict]:
    """Extract one row per example from the log text."""
    bars = [
        (int(m.group(1)), int(m.group(2)), m.start(), m.end())
        for m in BAR_RE.finditer(text)
    ]
    prefixes = [
        {"end": m.end(), "sum": m.group(1), "pct": m.group(3)}
        for m in PREFIX_RE.finditer(text)
    ]

    def prefix_before(pos: int) -> tuple[str, str] | None:
        """Cumulative prefix sitting directly in front of the bar at `pos`."""
        found = None
        for p in prefixes:
            if p["end"] > pos:
                break
            if text[p["end"]:pos].strip():
                continue  # not directly adjacent to the bar
            found = p
        if found is None:
            return None
        return found["sum"], found["pct"]

    rows: list[dict] = []
    for i, (cur, tot, _start, end) in enumerate(bars):
        if cur >= tot:
            break  # final `T/T` completion bar — no example behind it

        next_start = bars[i + 1][2] if i + 1 < len(bars) else len(text)
        content = text[end:next_start]

        is_error = "result=Prediction(" not in content
        quality = ""
        feedback = ""
        m = PRED_RE.search(content)
        if m:
            try:
                quality = str(ast.literal_eval(m.group(1)))
                feedback = str(ast.literal_eval(m.group(2)))
            except (ValueError, SyntaxError):
                pass

        cum = prefix_before(bars[i + 1][2]) if i + 1 < len(bars) else None

        rows.append(
            {
                "current": cur,
                "total": tot,
                "quality": quality,
                "feedback": feedback,
                "is_error": int(is_error),
                "metric": quality_score(quality),
                "avg_metric": cum[0] if cum else "",
                "avg_metric_percent": cum[1] if cum else "",
            }
        )
    return rows


def process_log(log_path: Path, csv_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_log(text)

    # sanity checks
    currents = [r["current"] for r in rows]
    totals = {r["total"] for r in rows}
    assert currents == list(range(len(rows))), "bar sequence is not 0..N-1"
    assert len(totals) == 1, f"inconsistent totals: {totals}"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(r["is_error"] for r in rows)
    last = rows[-1]
    print(
        f"{log_path.name}: {len(rows)} rows, total={totals.pop()}, "
        f"errors={n_errors}, "
        f"final avg_metric={last['avg_metric']} "
        f"({last['avg_metric_percent']}%)"
    )


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    if len(argv) > 1:
        logs = [Path(p) for p in argv[1:]]
    else:
        logs = sorted(here.glob("eval-*.log"))
    for log_path in logs:
        csv_path = log_path.with_suffix(".csv")
        process_log(log_path, csv_path)
        print(f"  -> {csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
