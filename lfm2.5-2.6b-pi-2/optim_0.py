#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "gepa[full]", "datasets", "openai", "teich", "jsonlines"]
# ///

# ruff: noqa: I001, EXE001
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

from jsonl import load_jsonl
from pi import pi # type: ignore
from gepa_models import create_lm # type: ignore


# pi(Qwen, skill, arg) -> trace_qwen: list
# pi(LFM, injected, skill, arg) -> trace_lfm: list
# metrics(trace_qwen, trace_lfm) -> score, feedback


def evaluate(candidate: dict) -> tuple[float, dict]:
    if isinstance(candidate, dict) and 'operation' in candidate and isinstance(candidate['operation'], str) and candidate['operation'] == 'sum':
        score = 1.0

        feedback = {
            'Error': None,
            'Output': 'You found right operation',
        }
    elif isinstance(candidate, dict):
        score = 0.5

        feedback = {
            'Error': 'I expect `{"operation": "..."}`. Operations are: "add", "sub", "mul", "div"',
            'Output': None,
        }
    elif not isinstance(candidate, dict):
        score = 0.0

        feedback = {
            'Error': 'You did not use dict and right operation in it. I expect `{"operation": "..."}`.',
            'Output': None,
        }
    else:
        score = 0.0

        feedback = {
            'Error': 'You did not use right candiate type and its operation.',
            'Output': None,
        }

    return score, feedback


STUDENT_MODEL = ("LiquidAI/LFM2.5-2.6B", "high")
TEACHER_MODEL = ("Qwen/Qwen3.8-27B", "none")

lm = create_lm(*STUDENT_MODEL)
reflection_lm = create_lm(*TEACHER_MODEL)

# candidate = {'operation': 'sum'}
result = optimize_anything(
    evaluator=evaluate,
    objective="Optimize for correct operations passed in a object. Do not generate code, just pass right object.",
    config=GEPAConfig(
        engine=EngineConfig(
            parallel=False,
            max_metric_calls=100,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
        ),
    ),
)

print(result.best_candidate)
