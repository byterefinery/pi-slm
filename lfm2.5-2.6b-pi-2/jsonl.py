from pathlib import Path

import jsonlines


def load_jsonl(path: str | Path) -> list[dict]:
    with jsonlines.open(path) as reader:
        return list(reader.iter(skip_empty=True)) # type: ignore
