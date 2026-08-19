#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai"]
# ///

'''
Teach the student LM (LiquidAI/LFM2.5-2.6B) to handle pi skill invocations like the teacher LM (Qwen/Qwen3.8-27B, use it without thinking/reasoning, so it works fast) using dspy with gepa optimizer.

Steps:
1. Add synthetic user/assistant pair (content + reasoning_content) inserted into the student conversation just AFTER the existing synthetic skills/tools Q&A pairs how skill system works: invoke skill `<skill> SKILL BODY </skill> [USER MESSAGE]` block with user message.
2. Add synthetic user/assistant pair (content + reasoning_content), one-shot usage of `example` skill (dummy fake skill just as reference), and answer to it: Invoked with no extra text
3. Add synthetic user/assistant pair (content + reasoning_content), one-shot usage of `example` skill (dummy fake skill just as reference), and answer to it: Invoked with "Hello"
4. Add synthetic user/assistant pair (content + reasoning_content), one-shot usage of `example` skill (dummy fake skill just as reference), and answer to it: Invoked with any other text, or asked to "call script"
5. The real skill-invocation user message happens here (it can be one message, but also more than one when testing modes transitions). We will use `tzip` skill here, it could be any skill really used in the future. Test invoking skill `tzip` using modes:
  - default / empty
  - on
  - lite
  - full
  - ultra
  - add more messages to conversation to cover transitions between all previous `tzip` modes. minimum one transitions (all combinations of mode-to-mode transition), but test up to two transitions (random three modes transitions).

Goal 1:
If file `skill-example-LiquidAI-LFM2.5-2.6B.json` is empty file, create it following these rules. If file is not empty, consider this goal fulfilled.
Since reasoning in `skill-example-Qwen-Qwen3.8-27B.json` is based on `Qwen/Qwen3.8-27B` model, in messages where `example` skill is used, convert reasoning to `LiquidAI/LFM2.5-2.6B` style based on `REASONING-LiquidAI-LFM2.5-2.6B.md` in new file `skill-example-LiquidAI-LFM2.5-2.6B.json`.
Set model and sampling params to one from LiquidAI/LFM2.5-2.6B (you can find them in `~/.pi/agent/models.json`).

Goal 2:
We need to optimize for any skill usage, so work on synthetic user/assistant pair in step 1.
You can find minimal conversations in `skill-example-LiquidAI-LFM2.5-2.6B.json` based on `Qwen/Qwen3.8-27B` but written in style of `LiquidAI/LFM2.5-2.6B`.
Do not touch system message and next two synthetic pairs of messages how to use skills and tools because that is already optimized for this model.

---

This is how you instantiate dspy.LM:

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

You can find models configurations in `~/.pi/agent/models.json`.
Never keep API_BASE and API_KEY in any other file, source code nor logs.
Never leak API_BASE and API_KEY.
Always read them from `models.json`.
'''

API_BASE = ...
API_KEY = ...

STUDENT_MODEL = "LiquidAI/LFM2.5-2.6B" # thinking: on
TEACHER_MODEL = "Qwen/Qwen3.8-27B" # thinking: off
