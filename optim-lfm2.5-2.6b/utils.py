# ruff: noqa: I001, EXE001
import json5


def extract_json(response: str):
    # Strip markdown code blocks if the LLM wrapped the JSON
    cleaned = response.replace('```json', '').replace('```', '').strip()
    data = json5.loads(cleaned)
    return data
