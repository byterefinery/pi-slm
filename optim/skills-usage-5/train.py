#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///

'''
NOTE: this is how you instantiate dspy.LM - keep this string/comment
IMPORTANT: Radon values for API_BASE and API_KEY on demand from `~/.pi/agent/models.json`. Never leak them in text, logs or source code.

```python
lm = dspy.LM(
    f"openai/{model}", # used model
    api_base=API_BASE,
    api_key=API_KEY,
    model_type="chat",
    temperature=..., # read for used model
    max_tokens=16384,
    extra_headers={"x-session-affinity": "dspy-optim"},
    extra_body={
        "top_k": ..., # read for used model, or omit if missing
        "min_p": ..., # read for used model, or omit if missing
        "presence_penalty": ..., # read for used model, or omit if missing
        "repeat_penalty": ..., # read for used model, or omit if missing
    },
)
```
'''

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"  # thinking: ON
TEACHER_MODEL = "Qwen/Qwen3.8-27B"  # thinking: OFF
SESSION_HEADER = {"x-session-affinity": "dspy-optim"}

API_BASE = None
API_KEY = None
