#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "gepa[full]", "datasets", "openai", "teich", "jsonlines", "rich", "tqdm"]
# ///

# ruff: noqa: I001, EXE001
import os
import json
# import shutil
# from pathlib import Path
# from copy import deepcopy
from random import Random
from typing import TypedDict
# from tempfile import TemporaryDirectory

import rich
from tqdm import tqdm
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

from pi import pi, run_isolated_pi # type: ignore
from utils import extract_json # type: ignore
from gepa_models import create_lm # type: ignore

STUDENT_MODEL = ("LiquidAI/LFM2.5-2.6B", "high")
TEACHER_MODEL = ("Qwen/Qwen3.8-27B", "low")
JUDGE_MODEL = ("Qwen/Qwen3.8-27B", "none")
REFLECTION_MODEL = ("Qwen/Qwen3.8-27B", "none")

SKILLS = {
    '.agents/skills/example': '../.agents/skills-byterefinery/example',
    '.agents/skills/tzip': '../.agents/skills-byterefinery/tzip',
    '.agents/skills/webfetch': '../.agents/skills/webfetch',
}

judge_lm = create_lm(*JUDGE_MODEL)
reflection_lm = create_lm(*REFLECTION_MODEL)

train_set = []
val_set = []


class Sample(TypedDict):
    input: str
    additional_context: dict
    answer: str


def get_teacher_samples() -> list[Sample]:
    TEACHER_SAMPLES_PATH = 'teacher-samples.json'

    if os.path.exists(TEACHER_SAMPLES_PATH):
        with open(TEACHER_SAMPLES_PATH, 'r') as f:
            samples = json.load(f)

        return samples

    samples = []

    for dst, src in tqdm(list(SKILLS.items())):
        rich.print(f'Skill: {dst=}')

        while True:
            try:
                train_input_examples, _ = run_isolated_pi(
                    model=TEACHER_MODEL[0],
                    thinking=TEACHER_MODEL[1],
                    prompt= (
                        f'Read whole skill, analyze it, and produce examples how skill can be invoked: {src}\n'
                        'Output should be just JSON (list of objects `[{"user_content": "/skill:SKILL_NAME SKILL_ARG"}, ...]`).\n'
                        'Produce 10 examples of skill usage. Do not over-complicate skill usage examples. Do not treat skill as programming tool because it has free-form of language.\n'
                        'For skill `webfetch` use URLs: https://tangledgroup.com/ , https://byterefinery.com/ .\n'
                        'Work only in current directory. Do not access user home directory.\n'
                        'Final output should be just JSON.\n'
                    ),
                    sandbox=True,
                    debug=True,
                    copy_skills=SKILLS,
                )

                train_input_examples: list[dict] = extract_json(train_input_examples)

                for n in train_input_examples:
                    sample: Sample = {
                        'input': n['user_content'],
                        'additional_context': {},
                        'answer': '',
                    }

                    samples.append(sample)
            except Exception as e:
                rich.print(f'{e=}')
                continue

            break

    for sample in tqdm(samples):
        while True:
            try:
                _, session_content = run_isolated_pi(
                    model=TEACHER_MODEL[0],
                    thinking=TEACHER_MODEL[1],
                    prompt=sample['input'],
                    extensions=['pi-slm.ts'],
                    sandbox=True,
                    debug=True,
                    copy_skills=SKILLS,
                )
            except Exception as e:
                rich.print(f'{e=}')
                continue

            break

        sample['answer'] = session_content

    with open(TEACHER_SAMPLES_PATH, 'w') as f:
        json.dump(samples, f)

    return samples


def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
    score = 0.0
    feedback = {}
    print(f'evaluate {len(candidate)=}, {example["input"]=}')

    # teacher/student pi sessions
    teacher_session = example['answer']

    while True:
        try:
            _, student_session = run_isolated_pi(
                model=STUDENT_MODEL[0],
                thinking=STUDENT_MODEL[1],
                prompt=example['input'],
                extensions=['pi-slm.ts'],
                sandbox=True,
                debug=True,
                copy_skills=SKILLS,
                override_file_content={
                    'pi-slm.json': candidate,
                }
            )
        except Exception as e:
            rich.print(f'{e=}')
            continue

        break

    # judge and produce verdict
    while True:
        try:
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.',
                },
                {
                    'role': 'user',
                    'content': (
                        'Compare teacher pi session file with student pi session file.\n'
                        f'<teacher_session>\n{teacher_session}\n</teacher_session>\n'
                        f'<student_session>\n{student_session}\n</student_session>\n'
                        "Especially, compare final assistant message in student and teacher sessions because these are their final answers. "
                        'Rate student with quality (string) and descriptive critique (string). '
                        'Rate student with following quality (string): "very low", "low", "medium", "high", "very high". '
                        'Output is just JSON with structure: `{"quality": QUALITY, "critique": CRITIQUE}`.\n'
                        'Final output is just JSON. '
                    )
                }
            ]

            # print(f'{messages=}')
            verdict: str = judge_lm(messages)
            verdict: dict = extract_json(verdict)
        except Exception as e:
            rich.print(f'{e=}')
            continue

        if not ('quality' in verdict and 'critique' in verdict):
            rich.print('Missing quality/critique in verdict, trying again...')
            continue

        break

    match verdict['quality']:
        case 'very low':
            verdict['score'] = 0.0
        case 'low':
            verdict['score'] = 0.25
        case 'medium':
            verdict['score'] = 0.5
        case 'high':
            verdict['score'] = 0.75
        case 'very high':
            verdict['score'] = 1.0
        case _:
            verdict['score'] = 0.0

    print(f'{verdict=}')
    score = verdict['score']

    feedback = {
        'Quality': verdict['quality'],
        'Critique': verdict['critique'],
    }

    return score, feedback

# train_set
train_set = get_teacher_samples()
train_set = train_set * 1
Random(0).shuffle(train_set)
rich.print(f'{len(train_set)=}')
# 1 / 0

# optimize `pi-slm.json`
with open('../pi-slm.json', 'r') as f:
    seed_candidate: str = f.read()

result = optimize_anything(
    seed_candidate=seed_candidate,
    evaluator=evaluate,
    dataset=train_set,
    objective=(
        "Optimize for student model performing like teacher model inside Pi coding agent. "
        "This is done by optimizing injected messages (keep same structure, just change `content` and/or `reasoning_content`), then asking Pi, and comparing responses after that point between student and teacher models. "
        "Do not optimize system role message. Optimize only user/assistant messages. "
        "Preserve student `reasoning_content` writing style while optimizing it. Student model is sensitive to reasoning/thinking content. "
    ),
    config=GEPAConfig(
        engine=EngineConfig(
            run_dir='./gepa_runs',
            display_progress_bar=True,
            parallel=False,
            # max_workers=2,
            max_metric_calls=100,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
        ),
    ),
)

print(result.best_candidate)
