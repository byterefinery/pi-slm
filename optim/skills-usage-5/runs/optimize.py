#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["gepa[full]", "openai"]
# ///
"""GEPA optimization of the synthetic skill-usage few-shot pair (skills-usage-5).

- candidate = dict of 3 fields: user_question, assistant_content, assistant_reasoning_content
- evaluator = full 12-case tzip suite (serial) + style gate -> (score, side_info)
- reflection = custom_candidate_proposer: teacher (Qwen3.8-27B, thinking OFF) with a
  strict JSON-only output contract (the default ```-block template was mis-parsed by
  the teacher, so we own the whole propose step).
- single-task reflective mode: each candidate is scored on the whole suite, so
  every reflection sees the complete failure picture.
All model calls are serial (shared server constraint).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Api, eval_candidate, hash8, log, seed_pair, teacher_call

from gepa import NoImprovementStopper, ScoreThresholdStopper, TimeoutStopCondition
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

HERE = Path(__file__).resolve().parent

MAX_METRIC_CALLS = 28
TIMEOUT_SECONDS = 6 * 3600
FIELDS = ("user_question", "assistant_content", "assistant_reasoning_content")

OBJECTIVE = (
    "Write the three fields of a synthetic few-shot message pair that teaches a 2.6B student model "
    "how to use the skill system. When a <skill> block appears in the latest user message, the model "
    "must treat it as a fresh, independent invocation: the argument is exactly the text after the "
    "block's closing </skill> tag (possibly empty), it must ignore earlier turns, and it must reply "
    "with exactly what the skill's Usage section requires for that argument - a single short line, "
    "nothing else, never empty. Score = 0.5*accuracy + 0.5*(1-leak_rate) over 12 tzip invocation "
    "cases (empty, on, lite, full, ultra, off, and mode-to-mode transitions); a style violation "
    "multiplies the score by 0.6. Target: 1.0."
)

CONSTRAINTS = (
    "HARD CONSTRAINTS - a candidate violating any of these is bad:\n"
    "1. Skill-agnostic: none of the three fields may mention 'tzip', its modes (on/lite/full/ultra/off), "
    "or any skill-specific confirmation string. The pair must teach the general protocol for ANY skill.\n"
    "2. user_question must stay a natural user question about how the skill system works when a skill "
    "block is in the latest message.\n"
    "3. assistant_reasoning_content must follow the student model's SHORT form: 1-3 plain first-person "
    "sentences ('The user ...', 'I will ...'), no lists/headers/LaTeX, no meta talk, ending by "
    "committing to the exact next action (the reply the Usage section requires).\n"
    "4. Keep the core protocol: fresh independent invocation per skill block; argument = exactly the "
    "text after the current block's closing tag (may be empty); earlier turns (including the previous "
    "skill's trailing argument word) are NOT this argument; reply = exactly what the Usage section "
    "requires - a single short line, never empty, never extra.\n"
    "5. Do not memorize or hard-code any case-specific strings from the evaluation (no 'activated', "
    "no mode names, no 'tzip'). General rules only."
)

PROPOSER_SYSTEM = (
    "You are the reflection engine of a prompt-optimization loop for a 2.6B student model. "
    "You receive the current candidate (a 3-field synthetic few-shot message pair) and evaluation "
    "feedback. You propose an improved candidate. "
    "OUTPUT FORMAT - this is critical: respond with ONLY one JSON object and nothing else. No prose "
    "before or after, no markdown fences, no commentary. The object has exactly these three string "
    "keys: \"user_question\", \"assistant_content\", \"assistant_reasoning_content\". Include all "
    "three fields in every response, even unchanged ones. Use \\n escapes for line breaks inside strings."
)


def _extract_json(text: str) -> dict | None:
    """First balanced-brace JSON object in text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
        start = text.find("{", start + 1)
    return None


def _clean_field(v: str) -> str:
    v = v.strip()
    # strip a wrapping single code fence if the LM added one
    m = re.fullmatch(r"```[a-zA-Z]*\n(.*)\n```", v, re.S)
    if m:
        v = m.group(1).strip()
    return v


def make_proposer(api: Api):
    def proposer(candidate: dict, reflective_dataset, components_to_update: list[str]) -> dict:
        # side_info is identical across components (single-task mode): take the first
        side = None
        for v in reflective_dataset.values():
            if v:
                side = v[0]
                break
        side_txt = json.dumps(side, indent=2, ensure_ascii=False) if side is not None else "(no data)"
        cand_txt = "\n\n".join(f"[{k}]\n{candidate.get(k, '')}" for k in FIELDS)
        prompt = (
            "CURRENT CANDIDATE:\n"
            f"{cand_txt}\n\n"
            f"EVALUATION FEEDBACK (JSON: score breakdown, constraints, style gate, per-case results):\n{side_txt}\n\n"
            f"{CONSTRAINTS}\n\n"
            "Objective: maximize the score (accuracy + leak-free reasoning + style compliance).\n"
            'Respond with ONLY the JSON object: {"user_question": "...", "assistant_content": "...", '
            '"assistant_reasoning_content": "..."}'
        )
        try:
            raw = teacher_call(api, PROPOSER_SYSTEM, prompt, temperature=0.7, max_tokens=4096,
                               use_cache=False)
        except Exception as e:  # noqa: BLE001
            log(f"proposer error: {e}")
            return {k: candidate[k] for k in FIELDS}
        obj = _extract_json(raw)
        if obj is None:
            log("proposer: no JSON object in response - keeping current candidate")
            return {k: candidate[k] for k in FIELDS}
        out = {}
        for k in FIELDS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = _clean_field(v)
            else:
                out[k] = candidate[k]
                log(f"proposer: field {k!r} missing/invalid - keeping current value")
        (HERE / "proposer-responses" / f"{hash8(out)}.json").write_text(
            json.dumps({"candidate": out, "raw": raw[:8000]}, indent=2, ensure_ascii=False)
        )
        return out

    return proposer


def build_side_info(res: dict) -> dict:
    lines = []
    for r in res["results"]:
        status = "OK" if (r["content_ok"] and not r["leak"]) else \
            ("WRONG-REPLY" if not r["content_ok"] else "LEAK")
        lines.append(
            f"[{r['id']}] arg={r['arg']!r} -> expected={r['expected']!r} actual={r['actual']!r} "
            f"{status}"
        )
        if not r["content_ok"]:
            lines.append(f"    student reasoning: {r['reasoning'][:400]}")
        if r["leak"]:
            lines.append(f"    LEAK verdict: {r['leak_reason']} (judge arg={r['judge_arg']!r})")
    style = "OK" if res["style"]["style_ok"] else f"FAIL: {res['style']['issues']}"
    return {
        "Score": f"accuracy={res['acc']:.3f} leak_rate={res['leak_rate']:.3f} composite={res['score']:.3f}",
        "Constraints": CONSTRAINTS,
        "StyleGate": style,
        "PerCase (12 cases, serial)": "\n".join(lines),
    }


def main() -> None:
    api = Api()
    seed = seed_pair()
    run_dir = HERE / "gepa-run"
    (HERE / "proposer-responses").mkdir(exist_ok=True)
    log(f"gepa START seed={hash8(seed)} max_metric_calls={MAX_METRIC_CALLS} timeout={TIMEOUT_SECONDS}s")

    def evaluator(candidate: dict):
        res = eval_candidate(api, candidate, use_cache=True, tag=f"cand-{hash8(candidate)}")
        side = build_side_info(res)
        (HERE / "gepa-evals" / f"{hash8(candidate)}.json").write_text(
            json.dumps({"candidate": candidate, "result": res}, indent=2, ensure_ascii=False)
        )
        return res["score"], side

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=OBJECTIVE,
        background=(
            "The message pair is spliced into a pi coding-agent conversation (system prompt + "
            "skills/tools listing + example-skill invocations ending with a finished 'example' skill "
            "exchange whose last user argument was 'Hi'). After the pair, the student receives tzip "
            "skill blocks with bare mode arguments (empty/on/lite/full/ultra/off) or transitions "
            "between them. The tzip Usage section inside each skill block already defines the exact "
            "confirmation replies; the few-shot pair only teaches the general invocation protocol. "
            "A teacher-model judge audits each student reasoning_content for CoT leaks."
        ),
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(run_dir),
                max_metric_calls=MAX_METRIC_CALLS,
                parallel=False,
                seed=5,
                display_progress_bar=True,
                raise_on_exception=True,
            ),
            reflection=ReflectionConfig(
                custom_candidate_proposer=make_proposer(api),
                module_selector="all",
            ),
            stop_callbacks=[
                NoImprovementStopper(max_iterations_without_improvement=6),
                ScoreThresholdStopper(threshold=1.0),
                TimeoutStopCondition(timeout_seconds=TIMEOUT_SECONDS),
            ],
        ),
    )

    best = result.best_candidate
    (HERE / "best_candidate.json").write_text(json.dumps(best, indent=2, ensure_ascii=False))
    log(
        f"gepa DONE best={hash8(best)} val_score={result.val_aggregate_scores[result.best_idx]:.3f} "
        f"total_metric_calls={result.total_metric_calls} candidates={len(result.candidates)}"
    )
    log("gepa BEST candidate:")
    for k, v in best.items():
        log(f"  {k}: {v!r}")


if __name__ == "__main__":
    main()
