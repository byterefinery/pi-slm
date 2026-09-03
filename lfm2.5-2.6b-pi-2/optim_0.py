#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "gepa[full]", "datasets", "openai", "teich", "jsonlines", "rich", "tqdm"]
# ///

# ruff: noqa: I001, EXE001
import os
import json
import shutil
from pathlib import Path
from copy import deepcopy
from random import Random
from tempfile import TemporaryDirectory

import rich
from tqdm import tqdm
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

from pi import pi # type: ignore
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

EXAMPLES = []


judge_lm = create_lm(*JUDGE_MODEL)
reflection_lm = create_lm(*REFLECTION_MODEL)


def extract_json(response: str):
    # Strip markdown code blocks if the LLM wrapped the JSON
    cleaned = response.replace('```json', '').replace('```', '').strip()
    data = json.loads(cleaned)
    return data


for dst, src in tqdm(list(SKILLS.items())):
    rich.print(f'Skill: {dst=}')

    while True:
        try:
            train_input_examples = pi(
                TEACHER_MODEL[0],
                TEACHER_MODEL[1],
                (
                    f'Read whole skill, analyze it, and produce examples how skill can be invoked: {src}\n'
                    'Output should be just JSON (list of objects `[{"user_content": "/skill:SKILL_NAME SKILL_ARG"}, ...]`).\n'
                    'Produce 10 examples of skill usage. Do not treat skill as programming tool because it has free form of language.\n'
                    'For skill `webfetch` use URLs: https://tangledgroup.com/ , https://byterefinery.com/ .\n'
                    'Final output should be just JSON.\n'
                ),
                temp=False,
            )

            train_input_examples = extract_json(train_input_examples)
            EXAMPLES.extend(train_input_examples)
        except Exception as e:
            rich.print(f'{e=}')
            continue

        break


Random(0).shuffle(EXAMPLES)
rich.print(EXAMPLES)
print()


def run_isolated_pi(override_file_content: dict | None=None, *args, **kwargs) -> tuple[str, str]:
    with TemporaryDirectory(delete=False) as td, TemporaryDirectory(delete=False) as tsd: # type: ignore no-matching-overload
        # print(f'{td=}')
        # print(f'{tsd=}')
        print(f'{td=} {tsd=}')

        os.makedirs(Path(str(td)) / '.pi' / 'agent')
        os.makedirs(Path(str(td)) / '.agents' / 'skills')

        for dst, src in SKILLS.items():
            shutil.copytree(src, Path(str(td)) / dst, dirs_exist_ok=True)

        # .pi
        shutil.copy(Path.home() / '.pi' / 'agent' / 'models.json', Path(str(td)) / '.pi' / 'agent' / 'models.json')
        # shutil.copy(Path.home() / '.pi' / 'agent' / 'settings.json', Path(str(td)) / '.pi' / 'agent' / 'settings.json')

        # pi-slm.ts
        if override_file_content and 'pi-slm.ts' in override_file_content:
            with open(Path(str(td)) / 'pi-slm.ts', 'w') as f:
                f.write(override_file_content['pi-slm.ts'])
        else:
            shutil.copy('../pi-slm.ts', Path(str(td)) / 'pi-slm.ts')

        # pi-slm.json
        if override_file_content and 'pi-slm.json' in override_file_content:
            with open(Path(str(td)) / 'pi-slm.json', 'w') as f:
                f.write(override_file_content['pi-slm.json'])
        else:
            shutil.copy('../pi-slm.json', Path(str(td)) / 'pi-slm.json')

        # run pi
        pi_output: str = pi(
            *args,
            session_dir=tsd,
            cwd=td,
            env=deepcopy(os.environ) | {'PI_CODING_AGENT_DIR': Path(str(td)) / '.pi' / 'agent'},
            **kwargs,
        )

        # pi names session files `<timestamp>_<session_id>.jsonl`, so the name cannot be guessed —
        # the fresh session dir contains exactly one file, and that is the session file.
        session_files = [
            os.path.join(str(tsd), name)
            for name in os.listdir(str(tsd))
            if os.path.isfile(os.path.join(str(tsd), name))
        ]
        assert len(session_files) == 1, f'Expected exactly one session file in {tsd!r}, found: {session_files!r}'
        session_file_path: str = session_files[0]

        with open(session_file_path, 'r') as f:
            pi_session_content = f.read()

    return pi_output, pi_session_content


def evaluate(candidate: str) -> tuple[float, dict]:
    score = 0.0
    feedback = {}

    try:
        candidate: list[dict] = json.loads(candidate)
    except Exception as e:
        # raise e
        score = 0.0

        feedback = {
            'Output': None,
            'Error': f'Could not deserialize candidate JSON: {e}'
        }

        return score, feedback

    # teacher
    for example in tqdm(EXAMPLES):
        if example.get('session_content'):
            continue

        try:
            _, teacher_session_content = run_isolated_pi(
                model=TEACHER_MODEL[0],
                thinking=TEACHER_MODEL[1],
                prompt=example['user_content'],
                extensions=['pi-slm.ts'],
                debug=True,
            )
        except Exception as e:
            # raise e
            score = 0.0

            feedback = {
                'Output': None,
                'Error': f'Could not run teacher model in pi:\n{example=}\n{e}'
            }

            return score, feedback

        example['session_content'] = teacher_session_content

    # student
    student_sessions_content = []

    for example in tqdm(EXAMPLES):
        try:
            _, student_session_content = run_isolated_pi(
                model=STUDENT_MODEL[0],
                thinking=STUDENT_MODEL[1],
                prompt=example['user_content'],
                extensions=['pi-slm.ts'],
                debug=True,
                override_file_content={
                    'pi-slm.json': json.dumps(candidate),
                }
            )
        except Exception as e:
            # raise e
            score = 0.0

            feedback = {
                'Output': None,
                'Error': f'Could not run student model in pi:\n{example=}\n{e=}'
            }

            return score, feedback

        student_sessions_content.append(student_session_content)

    # rich.print(student_sessions_content)

    # judge
    teacher_sessions_content = [n['session_content'] for n in EXAMPLES if n.get('session_content')]
    assert len(teacher_sessions_content) == len(student_sessions_content)
    verdicts: list[dict] = []

    # accumulate
    for teacher_session, student_session in tqdm(list(zip(teacher_sessions_content, student_sessions_content))):
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
                            'Rate student with quality (string) and descriptive critique (string). '
                            'Rate student with following quality (string): "very low", "low", "medium", "high", "very high". '
                            'Output is just JSON with structure: `{"quality": QUALITY, "critique": CRITIQUE}`. '
                            'Final output is just JSON. '
                        )
                    }
                ]

                # print(f'{messages=}')
                verdict: str = judge_lm(messages)
                verdict: dict = extract_json(verdict)
            except Exception as e:
                print(f'{e=}')
                continue

            if not ('quality' in verdict and 'critique' in verdict):
                print('Missing quality/critique in verdict, trying again...')
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
        verdicts.append(verdict)

    # collapse
    for verdict in verdicts:
        score += verdict['score']

        if 'Critiques' not in feedback:
            feedback['Critiques'] = [] # list[tuple[score, quality, critique]]

        assert isinstance(feedback['Critiques'], list)
        item: tuple[float, str, str] = (verdict['score'], verdict['quality'], verdict['critique'])
        feedback['Critiques'].append(item)

    score = score / len(verdicts)
    feedback['Output'] = 'Individual sample\'s critique is in "Critiques" list as tuple of: score, quality, and critique.'
    feedback['Error'] = None

    return score, feedback


with open('../pi-slm.json', 'r') as f:
    seed_candidate: str = f.read()


result = optimize_anything(
    seed_candidate=seed_candidate,
    evaluator=evaluate,
    objective=(
        "Optimize for student model performing like teacher model inside Pi coding agent. "
        "This is done by optimizing injected messages (keep same structure, just change `content` and/or `reasoning_content`), then asking Pi, and comparing responses after that point between student and teacher models. "
        "Do not optimize system role message. Optimize only user/assistant messages. "
        "Preserve student `reasoning_content` writing style while optimizing it. Student model is sensitive to reasoning/thinking content. "
    ),
    config=GEPAConfig(
        engine=EngineConfig(
            run_dir='./gepa_runs',
            parallel=False,
            # max_workers=2,
            max_metric_calls=10,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
        ),
    ),
)

print(result.best_candidate)
