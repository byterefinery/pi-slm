#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "gepa[full]", "openai", "jinja2"]
# ///
"""Repeat the full-start scenarios to expose intermittent failures.
Student generates, teacher judges."""
import train
from train import (load_model_configs, build_lms, make_user, user_text_of,
                   tzip_skill_block, rollout, say, EXAMPLE_FILE, PAIR_FILE)

MODES = ["on", "lite", "full", "ultra", "off"]
CONFIRM = {"on": "tzip lite activated", "lite": "tzip lite activated",
           "full": "tzip full activated", "ultra": "tzip ultra activated",
           "off": "tzip deactivated"}


def judge(teacher, history, current, reply, reasoning):
    from testjudge import JUDGE  # no
