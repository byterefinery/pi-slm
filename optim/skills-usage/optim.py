#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy"]
# ///

# OPENAI_API_BASE="" OPENAI_API_KEY="" uv run optim.py

# import os
import dspy

lm = dspy.LM(
    "openai/Qwen/Qwen3.8-27B",
    # api_base=os.getenv("OPENAI_API_BASE"),
    # api_key=os.getenv("OPENAI_API_KEY"),
    model_type="chat",
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.0,
    # max_tokens=8192,
    extra_headers={
        "x-session-affinity": "any-string-you-want",
    },
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
)

dspy.configure(lm=lm)

response = lm(messages=[{"role": "user", "content": "hi"}])
print(response)
