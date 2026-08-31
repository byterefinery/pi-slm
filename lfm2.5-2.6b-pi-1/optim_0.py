#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "datasets", "openai", "teich", "jsonlines"]
# ///

# ruff: noqa: I001, EXE001
import re
from teich.converter import _pi_reasoning_content
import os
from copy import deepcopy
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import dspy
import jsonlines
from huggingface_hub import snapshot_download

from pi import pi # type: ignore
from dspy_models import create_lm # type: ignore


def load_jsonl(path: str | Path) -> list[dict]:
    with jsonlines.open(path) as reader:
        return list(reader.iter(skip_empty=True)) # type: ignore


def load_dataset() -> dict:
    dataset_path = "TeichAI/Ox-Alpha-Pi-Traces"

    root = Path(snapshot_download(
        dataset_path,
        repo_type="dataset",
        local_files_only=True,
        allow_patterns=["*.jsonl", "**/*.jsonl"],
    ))

    raw = {
        p.name: load_jsonl(p)
        for p in root.rglob("*.jsonl")
    }

    # filter `teich-available-tools`
    for name, trace in list(raw.items()):
        trace = [
            n
            for n in trace
            if not (n['type'] == 'custom' and n['customType'] == 'teich-available-tools')
        ]

        raw[name] = trace
        last_msg = trace[-1]

        # check that last message is from `assistant`
        assert 'message' in last_msg and last_msg['type'] == 'message', last_msg
        assert last_msg['message']['role'] == 'assistant', last_msg

    return raw


def init_dataset(train: float=0.6, val: float=0.2, test: float=0.2, shuffle: bool=True) -> tuple:
    # shuffle
    dataset = load_dataset()
    dataset = list(dataset.items())

    if shuffle:
        Random(0).shuffle(dataset)

    # subset
    dataset = dataset[:10]
    tot_num = len(dataset)

    # 60%, 20%, 20%
    train_set = dataset[:int(train * tot_num)]
    val_set = dataset[-int((val + test) * tot_num):-int(test * tot_num)]
    test_set = dataset[-int(test * tot_num):]

    train_set = dict(train_set)
    val_set = dict(val_set)
    test_set = dict(test_set)

    return train_set, val_set, test_set


def get_final_assistant_message(step: dict) -> dict:
    reasoning_content: str | None = None
    content: str | None = None

    for n in step['message']['content']:
        if 'thinking' in n and n['thinking']: # noqa
            reasoning_content = n['thinking']

        if 'text' in n and n['text']: # noqa
            content = n['text']

    return {
        'reasoning_content': reasoning_content,
        'content': content,
    }


class FinalPiResponse(dspy.Signature):
    """Get final `pi` coding agent harness response."""
    filename: str = dspy.InputField(desc='Filename of session file of `pi` coding agent harness')
    steps: list[dict] = dspy.InputField(desc='Execution trace of `pi` coding agent harness')
    final_reasoning_content: dict = dspy.OutputField(desc='Final reasoning_content (thinking) of `pi` coding agent harness')
    final_content: dict = dspy.OutputField(desc='Final content of `pi` coding agent harness')


class PiModule(dspy.Module):
    def __init__(self):
        super().__init__()


    def forward(self, filename: str, steps: list[dict]) -> dspy.Prediction:
        steps = deepcopy(steps)

        with TemporaryDirectory(delete=False) as temp_dir_name: # type: ignore
            session_path: str = os.path.join(temp_dir_name, filename) # type: ignore
            print(f'{session_path=}')

            # check and fix session cwd
            assert steps[0]['type'] == 'session'

            if steps[0]['cwd'] != temp_dir_name:
                steps[0]['cwd'] = temp_dir_name

            # write JSONL file
            with jsonlines.open(session_path, mode='w') as writer:
                writer.write_all(steps)

            final_answer = pi(
                model=STUDENT_MODEL[0],
                thinking=STUDENT_MODEL[1],
                prompt='summarize',
                session=session_path,
            )

            final_steps = load_jsonl(session_path)
            message = get_final_assistant_message(final_steps[-1])
            final_reasoning_content = message['reasoning_content']
            final_content = message['content']

        return dspy.Prediction(
            final_reasoning_content=final_reasoning_content,
            final_content=final_content,
        )


STUDENT_MODEL = ("LiquidAI/LFM2.5-2.6B", "high")
TEACHER_MODEL = ("Qwen/Qwen3.8-27B", "none")

lm = create_lm(*STUDENT_MODEL)
reflection_lm = create_lm(*TEACHER_MODEL)

dspy.configure(lm=lm)

pi_module = PiModule()

# for filename, steps in train_set.items():
#     steps = steps[:-1] # skip last, predict last
#     result = pi_module(filename=filename, steps=steps)
#     print(result)
#     break

train_set, val_set, test_set = init_dataset(0.6, 0.2, 0.2, shuffle=False)
print(f'{len(train_set)=}')
print(f'{len(val_set)=}')
print(f'{len(test_set)=}')

train_set = [
    dspy.Example({
        'filename': filename,
        'steps': steps[:-1],
        '_steps': steps,
        'final_reasoning_content': get_final_assistant_message(steps[-1])['reasoning_content'],
        'final_content': get_final_assistant_message(steps[-1])['content'],
    }).with_inputs('filename', 'steps')
    for filename, steps in train_set.items()
]

val_set = [
    dspy.Example({
        'filename': filename,
        'steps': steps[:-1],
        '_steps': steps,
        'final_reasoning_content': get_final_assistant_message(steps[-1])['reasoning_content'],
        'final_content': get_final_assistant_message(steps[-1])['content'],
    }).with_inputs('filename', 'steps')
    for filename, steps in val_set.items()
]

test_set = [
    dspy.Example({
        'filename': filename,
        'steps': steps[:-1],
        '_steps': steps,
        'final_reasoning_content': get_final_assistant_message(steps[-1])['reasoning_content'],
        'final_content': get_final_assistant_message(steps[-1])['content'],
    }).with_inputs('filename', 'steps')
    for filename, steps in test_set.items()
]


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    assert example['_steps'], example
    assert 'final_reasoning_content' in prediction, prediction
    assert 'final_content' in prediction, prediction
    # correct_answer = int(example['answer'])

    # try:
    #     llm_answer = int(prediction.answer)
    # except ValueError as e:
    #     return 0

    # return int(correct_answer == llm_answer)
    return 0.5


evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=1,
    display_table=True,
    display_progress=True,
    provide_traceback=True,
)

evaluate(pi_module)
