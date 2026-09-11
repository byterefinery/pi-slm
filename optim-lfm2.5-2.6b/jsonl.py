import io
from pathlib import Path

import jsonlines


def load_jsonl(path: str | Path) -> list[dict]:
    with jsonlines.open(path) as reader:
        return list(reader.iter(skip_empty=True)) # type: ignore


def loads_jsonl(text: str) -> list[dict]:
    with jsonlines.Reader(io.StringIO(text)) as reader:
        return list(reader.iter(skip_empty=True)) # type: ignore
