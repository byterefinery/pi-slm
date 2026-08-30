#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["gepa[full]"]
# ///

# ruff: noqa: I001, EXE001
import re
import subprocess
from typing import Any
from pprint import pprint
from random import shuffle
from tempfile import TemporaryDirectory

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
)

# STUDENT_MODEL = 'LiquidAI/LFM2.5-2.6B'
# STUDENT_MODEL_THINKING = 'high'

TEACHER_MODEL = 'LiquidAI/LFM2.5-2.6B'
TEACHER_MODEL_THINKING = 'high'
# TEACHER_MODEL = 'Qwen/Qwen3.8-27B'
# TEACHER_MODEL_THINKING = 'off' # 'xhigh'

# objective = '''\
# Generate simple python function that implements basic string comparison algorithm, which returns one of: -1, 0, 1 .
# '''
objective = '''\
Add 20 digit input integers and produce output as integer without using programming languages. Just use LLM.
'''

background = '''\
You are using `pi` coding agent harness in the background to do optimization and fulfill your objective.
You work in random temp directory. Never pollute current working directory.
If output of optimization is programming code, do not run it, instead just return it.
'''


def pi(model: str, thinking: str, prompt: str, allow_tools: bool=True, temp: bool=True) -> str:
    cmd = [
        'pi',
        '--model', model,
        '--thinking', thinking,
        '--no-extensions',
        '--no-skills',
        '--no-context-files',
        '--no-tools',
    ]

    if allow_tools:
        cmd += [
            '--tools',
            '"read,write,edit,bash"',
        ]

    cmd += [
        prompt,
        '-p',
    ]

    # print(f'{cmd=}')
    proc_kwargs = {}
    temp_dir: TemporaryDirectory | None = None

    if temp:
        temp_dir = TemporaryDirectory()
        proc_kwargs['cwd'] = temp_dir.name

    proc = subprocess.run( # type: ignore # noqa
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **proc_kwargs,
    )

    del temp_dir

    if proc.stderr:
        raise RuntimeError(proc.stderr)

    stdout = proc.stdout.decode()
    # print('reflection_lm:', stdout)
    return stdout


def reflection_lm(prompt: str) -> str:
    print('reflection_lm', prompt)
    content = pi(TEACHER_MODEL, TEACHER_MODEL_THINKING, prompt)
    return content


def extract_int(text: str) -> int | None:
    """Pull the first integer (possibly negative) from model output."""
    m = re.search(r"-?\d+", text.replace(",", "").replace(" ", ""))

    if m:
        try:
            return int(m.group())
        except ValueError:
            return None

    return None


def evaluate(candidate: str, example: dict) -> tuple[float, dict[str, Any]]:
    inputs = example["inputs"]
    expected = example["output"]

    prompt = (
        'You must solve this using only language reasoning. Do not write or execute any code.\n'
        f'<candidate>{candidate}</candidate>\n'
        f'<example>{example}</example>\n'
        'Reply with ONLY the final integer answer, nothing else.'
    )

    side_info: dict[str, Any] = {
        "inputs": inputs,
        "expected": expected,
        "candidate_preview": candidate[:300],
    }

    try:
        raw = pi(
            TEACHER_MODEL,
            TEACHER_MODEL_THINKING,
            prompt,
            allow_tools=False,
        )

        side_info["raw_response"] = raw
        predicted = extract_int(raw)
        side_info["predicted"] = predicted

        if predicted is None:
            score = 0.0
            side_info["error"] = "could not parse integer"
        elif predicted == expected:
            score = 1.0
        else:
            # soft signal for large integers (optional but helpful)
            rel_err = abs(predicted - expected) / max(abs(expected), 1)
            score = max(0.0, 1.0 - min(rel_err, 1.0))
            side_info["rel_error"] = rel_err
    except Exception as e: # noqa
        score = 0.0
        side_info["error"] = str(e)
        predicted = None

    oa.log(f'score={score} predicted={predicted} expected={expected}')
    return score, side_info


train_set = [
    {'inputs': [a, a + 19], 'output': a + a + 19}
    for a in range(1_000_000, 1_000_000 + 100, 19)
] + [
    {'inputs': [a, a + 19], 'output': a + a + 19}
    for a in range(12_345_678_901_234_567_890, 12_345_678_901_234_567_890 + 100, 19)
]

shuffle(train_set)

val_set = [
    {'inputs': [a, a + 19], 'output': a + a + 19}
    for a in range(10_000_000_000_000_000_000, 10_000_000_000_000_000_000 + 100, 19)
]

pprint(train_set)
pprint(val_set)

seed_candidate = 'Add two integers and produce output.'

result = oa.optimize_anything(
    seed_candidate=seed_candidate,
    evaluator=evaluate,
    dataset=train_set,
    valset=val_set,
    objective=objective,
    background=background,
    config=GEPAConfig(
        engine=EngineConfig(
            # display_progress_bar=True,
            parallel=False,
            max_metric_calls=100,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm, # type: ignore
        ),
    )
)

print('-' * 80)
print(result)
print('-' * 80)
print(result.best_candidate)
print('=' * 80)
