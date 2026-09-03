import json


def extract_json(response: str):
    # Strip markdown code blocks if the LLM wrapped the JSON
    cleaned = response.replace('```json', '').replace('```', '').strip()
    data = json.loads(cleaned)
    return data
