#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["gepa[full]", "datasets", "openai", "jsonlines", "json5", "rich", "tqdm", "tenacity"]
# ///

# ruff: noqa: I001, EXE001, C408
import os
import json
# import shutil
# from pathlib import Path
# from copy import deepcopy
# from random import Random
from typing import TypedDict
# from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import rich
import json5
from tqdm import tqdm
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig, MergeConfig, RefinerConfig

from pi import run_isolated_pi # type: ignore
from utils import extract_json # type: ignore
from gepa_models import create_lm # type: ignore


TEACHER_SAMPLES_PATH = 'teacher-samples.json'
STUDENT_SAMPLES_PATH = 'student-samples.json'

STUDENT_REASONING_RULES = open('REASONING_LFM2_5_2_6B_COMPACT.md').read()

STUDENT_MODEL = ("LiquidAI/LFM2.5-2.6B", "high")
TEACHER_MODEL = ("Qwen/Qwen3.8-27B", "low")
JUDGE_MODEL = ("Qwen/Qwen3.8-27B", "none")
REFLECTION_MODEL = ("Qwen/Qwen3.8-27B", "low")

SKILLS = {
    '.agents/skills/example': '../.agents/skills-byterefinery/example',
    '.agents/skills/tzip': '../.agents/skills-byterefinery/tzip',
    '.agents/skills/webfetch': '../.agents/skills/webfetch',
}


judge_lm = create_lm(*JUDGE_MODEL)
reflection_lm = create_lm(*REFLECTION_MODEL)


class Sample(TypedDict):
    input: str
    additional_context: dict
    answer: str


def get_teacher_samples() -> list[Sample]:
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
                    prompt= f'''\
Read whole skill, analyze it, and produce examples how skill can be invoked: {src}

Output should be just JSON (list of objects `[{{"user_content": "/skill:SKILL_NAME SKILL_ARG"}}, ...]`).
Produce 10 examples of requested skill usage. Do not over-complicate skill usage examples. Do not treat skill as programming tool because it has free-form of language.
In case when skill `examples` is used, it is demo skill to see how skill directly is used, also its references, and how to use scripts.
In case when skill `webfetch`, use URLs: https://tangledgroup.com/ , https://byterefinery.com/ .
In case when skill `tzip` is used, it is skill that activates a mode (skill has list of this modes). It does not work on any files. Instead it is in-context token compressor that LLM knows how to use.
Work only in current directory. Do not access user home directory.
Since example is created in sandboxed directory (probably current working directory or temp directory), try to use local files only which are available without complaining that you do not have access to file/dir path/content.

Final output should be just JSON.
''',
                    sandbox=True,
                    debug=True,
                    copy_skills=SKILLS,
                )

                train_input_examples: list[dict] = extract_json(train_input_examples)
                rich.print(train_input_examples)

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

            if not all(sample['input'].startswith('/skill:') for sample in samples):
                rich.print('All samples need to start with "/skill:"')
                continue

            break

    for sample in tqdm(samples):
        while True:
            try:
                _, session_content = run_isolated_pi(
                    model=TEACHER_MODEL[0],
                    thinking=TEACHER_MODEL[1],
                    prompt=[
                        '''\
Work only in current directory. Do not access user home directory.
Since example is created in sandboxed directory (probably current working directory or temp directory), try to use local files only which are available without complaining that you do not have access to file/dir path/content.
Just acknowledge.''',
                        sample['input']
                    ],
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


def get_student_samples() -> list[Sample]:
    if os.path.exists(STUDENT_SAMPLES_PATH):
        with open(STUDENT_SAMPLES_PATH, 'r') as f:
            samples = json.load(f)

        return samples

    assert os.path.exists(TEACHER_SAMPLES_PATH)

    with open(TEACHER_SAMPLES_PATH, 'r') as f:
        samples = json.load(f)

    for sample in samples:
        sample['answer'] = ''

    for sample in tqdm(samples):
        while True:
            try:
                _, session_content = run_isolated_pi(
                    model=STUDENT_MODEL[0],
                    thinking=STUDENT_MODEL[1],
                    prompt=sample['input'],
                    # extensions=['pi-slm.ts'],
                    sandbox=True,
                    debug=True,
                    copy_skills=SKILLS,
                )
            except Exception as e:
                rich.print(f'{e=}')
                continue

            break

        sample['answer'] = session_content

    with open(STUDENT_SAMPLES_PATH, 'w') as f:
        json.dump(samples, f)

    return samples


def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
    print(f'evaluate {len(candidate)=}, {example["input"]=}')

    # check if candidate is valid json
    try:
        candidate: dict = json5.loads(candidate)
    except Exception as e:
        rich.print(f'Skipping f{len(candidate)=} {e=}')
        score = 0.0

        feedback = {
            'Error': f'Could not parse candidate as valid JSON: {e}',
        }

        return score, feedback

    # check candidate structure
    if not ('messages' in candidate and isinstance(candidate['messages'], list) and len(candidate['messages']) == 5):
        score = 0.0

        feedback = {
            'Error': 'Invalid JSON structure: "messages" is not an array of size 5',
        }

        return score, feedback

    # check candidate messages
    for i, message in enumerate(candidate['messages']):
        if i == 0 and not (message.get('role') == 'system' and message.get('content')):
            score = 0.0

            feedback = {
                'Error': 'System role message is not valid, and/or whole messages structure is broken probably.',
            }

            return score, feedback


        if 'role' not in message:
            score = 0.0

            feedback = {
                'Error': 'Message does not have "role" field.',
            }

            return score, feedback

        if 'content' not in message:
            score = 0.0

            feedback = {
                'Error': 'Message does not have "content" field.',
            }

            return score, feedback

        if message['role'] == 'user' and '/skill' in message['content']:
            score = 0.0

            feedback = {
                'Error': 'User role message has content that uses skill directly which is not allowed',
            }

            return score, feedback

        if message['role'] == 'assistant' and '/skill' in message['content']:
            score = 0.0

            feedback = {
                'Error': 'Assistant role message has content that uses skill directly which is not allowed',
            }

            return score, feedback

        if message['role'] == 'assistant' and 'reasoning_content' not in message:
            score = 0.0

            feedback = {
                'Error': 'Assistant role message does not have "reasoning_content" field.',
            }

            return score, feedback

        if message['role'] == 'assistant' and '/skill' in message['reasoning_content']:
            score = 0.0

            feedback = {
                'Error': 'Assistant role message has reasoning content that uses skill directly which is not allowed',
            }

            return score, feedback

    # serialize back candidate
    candidate: str = json.dumps(candidate)

    # teacher pi sessions
    teacher_session = example['answer']

    # student pi session
    is_timeout: bool = False

    while True:
        with ThreadPoolExecutor(max_workers=1) as ex:
            kwargs = dict(
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

            fut = ex.submit(run_isolated_pi, **kwargs)

            try:
                _, student_session = fut.result(timeout=300)
            except TimeoutError as e:
                rich.print(f'TimeoutError {e=}')
                is_timeout = True
                break
            except Exception as e:
                rich.print(f'{e=}')
                continue

        break

    if is_timeout:
        score = 0.0

        feedback = {
            'Error': 'Candidate has timeout, and it was running for too long.',
        }

        return score, feedback

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
                    'content': f'''\
Compare teacher pi session with student pi session.

<TEACHER_SESSION>
{teacher_session}
</TEACHER_SESSION>

<STUDENT_SESSION>
{student_session}
</STUDENT_SESSION>

Especially, compare final assistant message in student and teacher sessions because these are their final answers.
Rate student with quality (string) and descriptive critique (string).
Rate student with following quality (string): "very low", "low", "medium", "high", "very high".
Output is just JSON with structure: `{{"quality": QUALITY, "critique": CRITIQUE}}`.

Final output is just JSON.
''',
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
teacher_samples = get_teacher_samples()
train_set = teacher_samples
# Random(0).shuffle(train_set)
rich.print(f'{len(train_set)=}')
rich.print([n["input"] for n in train_set])
# 1 / 0

# student_samples = get_student_samples()
# rich.print(f'{len(student_samples)=}')
# rich.print([n["input"] for n in student_samples])
# 1 / 0

# optimize `pi-slm.json`
with open('../pi-slm.json', 'r') as f:
    seed_candidate: str = f.read()

result = optimize_anything(
    seed_candidate=seed_candidate,
    evaluator=evaluate,
    dataset=train_set,
    objective="""\
Optimize for student model performing like teacher model inside Pi coding agent.
This is done by optimizing injected messages (keep same structure, just change `content` and/or `reasoning_content`), then ask Pi and compare responses after that point between student and teacher models.

Do not optimize system role message. Optimize only user/assistant messages.

Ignore part where teacher has instructions for this:
```
Work only in current directory. Do not access user home directory.
Since example is created in sandboxed directory (probably current working directory or temp directory), try to use local files only which are available without complaining that you do not have access to file/dir path/content.
Just acknowledge.
```

Preserve student `reasoning_content` writing style while optimizing it. Student model is sensitive to reasoning/thinking content.
Student `content` and `reasoning_content` should not include anything that is any skill specific - they should be skill agnostic.
""",
    background=f"""\
These are rules for student model reasoning content:
<STUDENT_REASONING_RULES>
{STUDENT_REASONING_RULES}
</STUDENT_REASONING_RULES>
""",
    config=GEPAConfig(
        engine=EngineConfig(
            run_dir='./gepa_runs',
            display_progress_bar=True,
            # parallel=False,
            parallel=True,
            max_workers=4,
            max_metric_calls=1_000,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
        ),
        # merge=MergeConfig(max_merge_invocations=3, merge_val_overlap_floor=3),
        # refiner=RefinerConfig(),
    ),
)

print()
print(result.best_candidate)
