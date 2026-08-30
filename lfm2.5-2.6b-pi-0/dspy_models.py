#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "gepa[full]"]
# ///

# ruff: noqa: I001, EXE001
import re
import json
import uuid
from pathlib import Path

import dspy


MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
SESSION_ID = uuid.uuid4().hex[:8]  # stable per process: keeps llama-server routing consistent


def _load_models_config(path: Path = MODELS_JSON) -> dict:
    text = path.read_text()
    # models.json may contain trailing commas (pi parses them leniently)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def _find_model(cfg: dict, model_name: str) -> tuple[dict, dict]:
    """Return the (provider, model) entries for `model_name` (matched by id)."""
    for provider in cfg["providers"].values():
        for model in provider.get("models", []):
            if model["id"] == model_name:
                return provider, model
    raise ValueError(f"model {model_name!r} not found in {MODELS_JSON}")


def create_lm(model_name: str, reasoning_effort: str="default") -> dspy.LM:
    """Create a dspy.LM for `model_name` with settings read from models.json.

    `thinking` is a thinking level name (e.g. "off", "low", "high"). The
    llama.cpp request carries the main switch `reasoning` ("on"/"off") plus
    `reasoning_effort` with the actual level. "off"/"none" map to
    `reasoning: "off"` + `reasoning_effort: "none"`; any other level maps to
    `reasoning: "on"` and, when the model's `thinkingLevelMap` resolves it to
    a non-null value, that value as `reasoning_effort`.
    """
    cfg = _load_models_config()
    provider, model = _find_model(cfg, model_name)
    sampling = model.get("samplingParams", {})

    extra_body = {
        key: sampling[key]
        for key in ("top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")
        if key in sampling
    }

    # thinking: always pass the `reasoning` on/off switch, plus
    # `reasoning_effort` with the actual level ("none", "minimal", "low", ...).
    if reasoning_effort == "none":
        extra_body["reasoning"] = "off"
        extra_body["reasoning_effort"] = reasoning_effort

        extra_body["chat_template_kwargs"] = {
            "enable_thinking": False,
        }
    else:
        extra_body["reasoning"] = "on"
        extra_body["reasoning_effort"] = reasoning_effort

    return dspy.LM(
        f"openai/{model_name}",
        api_base=provider["baseUrl"],
        api_key=provider["apiKey"],
        model_type="chat",
        temperature=sampling.get("temperature", 1.0),
        max_tokens=model.get("maxTokens"),
        extra_headers={"x-session-affinity": f"dspy-{model_name}-{SESSION_ID}"},
        extra_body=extra_body,
    )


if __name__ == "__main__":
    lm = create_lm("LiquidAI/LFM2.5-2.6B", "high")          # student model
    reflection_lm = create_lm("Qwen/Qwen3.8-27B", "none")   # teacher model

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    print(lm(messages=messages))
    print()

    print(reflection_lm)
    print(reflection_lm(messages=messages))
