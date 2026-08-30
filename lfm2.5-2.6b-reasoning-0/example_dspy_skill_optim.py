#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "gepa[full]", "wikipedia"]
# ///

# ruff: noqa: I001, EXE001
import dspy
import wikipedia

from dspy_models import create_lm


lm = create_lm("LiquidAI/LFM2.5-2.6B", "high")
reflection_lm = create_lm("Qwen/Qwen3.8-27B", "none")

dspy.configure(lm=lm)


class SkillSignature(dspy.Signature):
    name: str = dspy.InputField(desc="Short name of skill")
    descripton: str = dspy.InputField(desc="Description of skill and when it should be triggered")
    skill_instruction: str = dspy.InputField(desc="Skill instruction")
    skill_references: dict[str, str] = dspy.InputField(desc="Skill references where key is reference filename and value is reference file content; they live in `references/` directory")
    skill_scripts: dict[str, str] = dspy.InputField(desc="Skill scripts where key is script filename and value is script file content; they live in `scripts/` directory")
    skill_argument: str = dspy.InputField(desc="Skill argument")
    skill_output: str = dspy.OutputField(desc="Skill output")


class SkillModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.writer = dspy.Predict(SkillSignature)


    def forward(self, skill: dict) -> dspy.Prediction:
        result = self.writer(**skill)
        return dspy.Prediction(skill_output=result.skill_output)



def skill_metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    return dspy.Prediction(0.0, 'Skill is broken')


skill_module = SkillModule()


optimizer = dspy.GEPA(
    metric=skill_metric,
    reflection_lm=reflection_lm,
    auto="light",
    num_threads=1,
)

train = []
val = []
optimized_skill_module = optimizer.compile(skill_module, trainset=train, valset=val)
