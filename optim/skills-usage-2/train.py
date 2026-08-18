#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy"]
# ///

'''
# NOTE: this is how you instantiate dspy.LM - keep this string/comment
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

# Tasks:
# - You will read teacher and student messages. Last two messages are something we want to match. In order to do that, we need to insert synthetic user and assistant messages just beofre skill is invoked in student JSON messages array. We did the same for skills and tools using synthetic messages.
# - Read `tzip-full-Qwen-Qwen3.8-27B.json` and check its correct answer. Use this as teacher answer which we want to achieve. `content` is more important than `reasoning_content` because each model has specific way of reasoning.
# - Read `tzip-full-LiquidAI-LFM2.5-2.6B.json` and check its wrong answer. Use this as student answer which we want to improve. `content` is more important than `reasoning_content` because each model has specific way of reasoning.
# - `conetnet` of student model should match `content` of teacher model.
# - Your goal is to get same response(s) as teacher model after skill is invoked with use input/request/task.
'''

import os

import dspy


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.expanduser("~/.pi/agent/models.json")

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B"
TEACHER_MODEL = "Qwen/Qwen3.8-27B"

TEACHER_FILE = os.path.join(HERE, "tzip-full-Qwen-Qwen3.8-27B.json")
STUDENT_FILE = os.path.join(HERE, "tzip-full-LiquidAI-LFM2.5-2.6B.json")
