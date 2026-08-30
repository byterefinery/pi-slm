#!/usr/bin/env python3
"""
Analyze exactly how LiquidAI/LFM2.5-2.6B (thinking `high`) reasons.

Pipeline
--------
1. For each task in TASKS, run `pi` in print mode:

       pi --model LiquidAI/LFM2.5-2.6B --thinking high ... --session-dir <tmp> "<task>" -p

   `--session-dir <tmp>` points pi at a throwaway session directory (pi stores
   sessions as JSONL under <session-dir>/<encoded-cwd>/), so after the run we
   can find and inspect the session JSONL without touching the user's real
   session store. (Verified: with only --session-dir, pi keeps stderr clean,
   unlike --session-id which warns on stderr.)

2. Parse the session JSONL. In pi's format, reasoning_content of an assistant
   message lives in the `content` list as parts of the form
       {"type": "thinking", "thinking": "<raw chain of thought>", "thinkingSignature": ...}
   together with {"type": "text", ...} parts, plus per-turn `usage` /
   `stopReason` and a `session` header entry.
   The reasoning_content (thinking parts) is preserved BYTE-EXACT (every
   space, tab, newline is part of the data). The only transformation is
   stripping <|...|>-style backend special tokens (tool-call markup) from the
   extracted text, which are protocol tokens, not model reasoning.

3. Feed the collected raw reasoning transcripts to a much larger model,
   Qwen/Qwen3.8-27B (also via `pi`; its thinkingLevelMap maps `high` -> null,
   so `xhigh` is used), with instructions to write `lfm2_5_2_6b_reasoning.md`
   documenting *exactly how* LFM2.5-2.6B thinks, with quoted evidence.

Model config (api key, base url, sampling params) is read from
~/.pi/agent/models.json at runtime; pi itself authenticates from the same file.

Side artifacts (next to this script):
  lfm2_5_2_6b_reasoning.md      <- the analysis (main deliverable)
  lfm2_5_2_6b_run/sessions/*.jsonl  <- raw session files (for inspection)
  lfm2_5_2_6b_run/transcripts.txt   <- extracted reasoning_content transcripts
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "lfm2_5_2_6b_reasoning.md"
ARTIFACT_DIR = HERE / "lfm2_5_2_6b_run"
SESSION_SAVE_DIR = ARTIFACT_DIR / "sessions"
TRANSCRIPTS_TXT = ARTIFACT_DIR / "transcripts.txt"

SMALL_MODEL = "LiquidAI/LFM2.5-2.6B"   # model under analysis
SMALL_THINKING = "high"                # only non-null level in its thinkingLevelMap
ANALYZER_MODEL = "Qwen/Qwen3.8-27B"    # larger model that writes the analysis
ANALYZER_THINKING = "xhigh"            # its thinkingLevelMap maps "high" -> null
TOOLS = "read,write,edit,bash"

# Tasks covering the requested categories: mathematics, code, general
# reasoning, instruction following, and tool use. Each entry is
# (label, prompt, allow_tools). The tool-use run runs WITH tools enabled so we
# can see how it plans/executes tool calls; the others run without tools for
# pure reasoning capture.
TASKS: list[tuple[str, str, bool]] = [
    (
        "mathematics",
        "Find the sum of all positive integers below 1000 that are not "
        "divisible by 3 or 5. Show your reasoning step by step, and give the "
        "final number.",
        False,
    ),
    (
        "code",
        "The following Python function is supposed to return the index of "
        "target in the sorted list arr, or -1 if absent. It has a bug that "
        "makes it loop forever in some cases. Find the bug, explain exactly "
        "when and why it fails, and give the fixed function.\n\n"
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid\n"
        "        else:\n"
        "            hi = mid\n"
        "    return -1",
        False,
    ),
    (
        "general_reasoning",
        "Prove that in any group of 6 people there are either 3 people who all "
        "know each other, or 3 people who are all mutual strangers. Give a "
        "complete, rigorous proof. Then state what happens for 5 people.",
        False,
    ),
    (
        "instruction_following",
        "Answer the question: Is 17 prime? Follow these rules EXACTLY in your "
        "visible reply: 1) The entire reply must be a single line. 2) It must "
        "start with the word 'Answer:'. 3) Use no punctuation other than the "
        "final period. 4) Use at most 8 words. 5) Do not mention the rules.",
        False,
    ),
    (
        "tool_use",
        "Use the bash tool to create a file named hello.txt in the current "
        "directory containing exactly the text 'hello world' (no trailing "
        "newline). Then use the read tool to read the file back and confirm "
        "its contents. Finally tell me what you did.",
        True,
    ),
]


# --------------------------------------------------------------------------
# pi runner (shape of the provided `pi()`; see notes below)
# --------------------------------------------------------------------------

def pi(
    model: str,
    thinking: str,
    prompt: str,
    allow_tools: bool = True,
    temp: bool = True,
    extra: tuple[str, ...] = (),
    cwd: str | None = None,
    timeout: float | None = None,
) -> str:
    """Run the `pi` CLI in print mode and return stdout.

    Notes vs. the original snippet:
      * `--tools` value is unquoted: without a shell, argv would contain the
        literal double quotes and no tool would match.
      * `extra`  : pass through additional flags (e.g. ["--session-dir", d]).
      * `cwd`    : fixed working directory (overrides the `temp` temp dir).
      * `timeout`: guard against hangs (None = wait forever, like the original).
    """
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
            TOOLS,
        ]

    cmd += list(extra)
    cmd += [
        prompt,
        '-p',
    ]

    # print(f'{cmd=}')
    proc_kwargs: dict[str, Any] = {}
    temp_dir: tempfile.TemporaryDirectory | None = None

    if cwd is None and temp:
        temp_dir = tempfile.TemporaryDirectory(prefix='pi-cwd-')
        proc_kwargs['cwd'] = temp_dir.name
    if cwd is not None:
        proc_kwargs['cwd'] = cwd

    try:
        proc = subprocess.run(  # type: ignore # noqa
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            **proc_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'pi timed out after {timeout}s') from exc

    del temp_dir

    if proc.stderr:
        raise RuntimeError(proc.stderr.decode(errors='replace'))
    if proc.returncode != 0:
        raise RuntimeError(f'pi exited with code {proc.returncode}')

    return proc.stdout.decode()


# --------------------------------------------------------------------------
# models.json (api key, base url, sampling params)
# --------------------------------------------------------------------------

def load_models_cfg() -> dict:
    cfg_path = Path(os.environ.get('PI_AGENT_DIR', Path.home() / '.pi' / 'agent')) / 'models.json'
    text = cfg_path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # pi's own models.json may be loose JSON (e.g. trailing commas,
        # see the thinkingLevelMap blocks); strip trailing commas and retry.
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)


def find_model(cfg: dict, model_id: str) -> tuple[str, dict, dict]:
    for provider_name, provider in cfg['providers'].items():
        for m in provider.get('models', []):
            if m.get('id') == model_id:
                return provider_name, provider, m
    raise LookupError(f'{model_id!r} not found in models.json')


# --------------------------------------------------------------------------
# Session JSONL parsing
# --------------------------------------------------------------------------

# Backend special tokens (tool-call protocol markup) are not model reasoning;
# strip them everywhere before analysis.
_TOOL_CALL_BLOCK_RE = re.compile(r'<\|tool_call_start\|>.*?<\|tool_call_end\|>', re.DOTALL)
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]*\|>')


def strip_special_tokens(s: str) -> str:
    s = _TOOL_CALL_BLOCK_RE.sub('', s)
    s = _SPECIAL_TOKEN_RE.sub('', s)
    return s


def find_session_jsonl(session_dir: Path) -> Path:
    files = sorted(session_dir.rglob('*.jsonl'))
    if len(files) != 1:
        raise RuntimeError(f'expected exactly 1 session file in {session_dir}, got: {files}')
    return files[0]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            p.get('text', '') for p in content
            if isinstance(p, dict) and p.get('type') == 'text'
        )
    return str(content)


def parse_session_jsonl(path: Path) -> dict:
    """Extract header, user prompts and assistant turns (with reasoning_content)."""
    rec: dict[str, Any] = {
        'file': path,
        'header': None,
        'thinking_level_changes': [],
        'user_prompts': [],
        'turns': [],
    }
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        t = e.get('type')
        if t == 'session':
            rec['header'] = e
        elif t == 'thinking_level_change':
            rec['thinking_level_changes'].append(e)
        elif t == 'message':
            m = e.get('message', {})
            role = m.get('role')
            if role == 'user':
                rec['user_prompts'].append(_content_to_text(m.get('content')))
            elif role == 'assistant':
                thinking: list[str] = []
                texts: list[str] = []
                tool_calls: list[dict] = []
                for p in m.get('content', []):
                    if not isinstance(p, dict):
                        continue
                    pt = p.get('type')
                    if pt == 'thinking':
                        thinking.append(strip_special_tokens(p.get('thinking') or ''))
                    elif pt == 'text':
                        texts.append(strip_special_tokens(p.get('text') or ''))
                    elif pt == 'toolCall':
                        tool_calls.append({'name': p.get('name'), 'arguments': p.get('arguments')})
                rec['turns'].append({
                    'model': m.get('model'),
                    'provider': m.get('provider'),
                    'api': m.get('api'),
                    'stop_reason': m.get('stopReason'),
                    'usage': m.get('usage'),
                    'thinking': thinking,
                    'text': texts,
                    'tool_calls': tool_calls,
                })
    return rec


# --------------------------------------------------------------------------
# Transcript rendering
# --------------------------------------------------------------------------

def whitespace_stats(s: str) -> str:
    """Coarse whitespace profile — whitespace is part of the reasoning content."""
    total = len(s)
    non_ws = sum(1 for c in s if not c.isspace())
    lines = s.split('\n')
    return (
        f"chars={total} non_ws={non_ws} newlines={s.count(chr(10))} cr={s.count(chr(13))} "
        f"tabs={s.count(chr(9))} double_spaces={s.count('  ')} "
        f"blank_lines={sum(1 for ln in lines if ln.strip() == '')} "
        f"indented_lines={sum(1 for ln in lines if ln != ln.lstrip())} "
        f"trailing_ws_lines={sum(1 for ln in lines if ln != ln.rstrip())}"
    )


def render_transcript(run_no: int, n_runs: int, task: str, sess: dict, allow_tools: bool) -> str:
    h = sess['header'] or {}
    L: list[str] = []
    bar = '=' * 78
    L.append(bar)
    L.append(f"RUN {run_no}/{n_runs} (category: {task})")
    L.append(f"session file : {sess['file']}")
    L.append(f"session id   : {h.get('id')} (started {h.get('timestamp')}, cwd {h.get('cwd')})")
    turns = sess['turns']
    if turns:
        t0 = turns[0]
        L.append(f"model          : {t0['model']} (provider {t0['provider']}, api {t0['api']})")
    L.append(f"thinking level : {SMALL_THINKING} (requested)")
    L.append(f"tools enabled  : {allow_tools}")
    tl = sess['thinking_level_changes']
    if tl:
        L.append(f"thinking_level_change entries: {json.dumps(tl)}")
    L.append(bar)
    for i, p in enumerate(sess['user_prompts'], 1):
        L.append(f'USER PROMPT {i}:')
        L.append(p)
        L.append('')
    for i, turn in enumerate(turns, 1):
        L.append(f"ASSISTANT TURN {i}")
        L.append(f"  stop_reason: {turn['stop_reason']}")
        if turn['usage']:
            u = turn['usage']
            L.append(
                "  usage      : input={input} output={output} reasoning={reasoning} total={totalTokens}".format(**{
                    k: u.get(k) for k in ('input', 'output', 'reasoning', 'totalTokens')
                })
            )
        for k, th in enumerate(turn['thinking'], 1):
            L.append(f"  --- reasoning_content (part {k}, VERBATIM, {len(th)} chars) ---")
            L.append(f"  whitespace_profile: {whitespace_stats(th)}")
            L.append(th)
            L.append('  --- end reasoning_content (verbatim) ---')
        for tc in turn['tool_calls']:
            L.append(f"  TOOL CALL: {tc['name']} {json.dumps(tc['arguments'])}")
        for tx in turn['text']:
            L.append('  --- text (visible answer) ---')
            L.append(tx)
    L.append('')
    return '\n'.join(L)


# --------------------------------------------------------------------------
# Analyzer prompt
# --------------------------------------------------------------------------

def build_analyzer_prompt(transcript: str, small_meta: dict, analyzer_meta: dict) -> str:
    sp = small_meta['sampling']
    ap = analyzer_meta['sampling']
    return f"""You are an expert LLM interpretability analyst. Your job: document EXACTLY how the small model {SMALL_MODEL} reasons (its chain-of-thought / reasoning_content) when running in the `pi` coding agent with thinking level `{SMALL_THINKING}`.

## Context
- Model under analysis: {SMALL_MODEL}
- Provider: {small_meta['provider']} (baseUrl {small_meta['base_url']}, api {small_meta['api']})
- Thinking level: `{SMALL_THINKING}`
- Sampling parameters used by the backend for this model: {json.dumps(sp)}
- Runs: mathematics, code (bug hunt), general reasoning (proof), instruction following (strict output constraints), and tool use. The first four ran with tools DISABLED; the tool-use run had read/write/edit/bash ENABLED.
- You are {ANALYZER_MODEL} (sampling {json.dumps(ap)}), analyzing the raw transcripts below. Do not do the tasks yourself; analyze the model's reasoning process, not the answers (though you may note when the model's final answer is right or wrong as part of assessing its verification behavior).

## Input
Below is the complete raw reasoning_content extracted VERBATIM (byte-exact: every space, tab and newline is preserved and is part of the data) from the pi session JSONL files (assistant message content parts of type "thinking"), together with the user prompts, visible answers, per-turn usage, stop reasons, and per-part whitespace profiles.

## Deliverable
Use the `write` tool to create the file `lfm2_5_2_6b_reasoning.md` in the current working directory. It must be a thorough, evidence-grounded markdown document titled:

    # How {SMALL_MODEL} (thinking: high) Reasons

Required content (adapt section names if you see a better structure, but cover all of it):
1. **Corpus description** — the 5 runs/categories, overall sizes (chars/tokens of reasoning per run from the usage data and whitespace profiles).
2. **Overall chain-of-thought structure** — how a reasoning turn is organized (e.g. restating the problem, planning, step-by-step execution, checking, final answer). Describe the recurring skeleton with a schematic.
3. **Planning behavior** — does it plan before executing? Does it revise plans mid-reasoning?
4. **Decomposition and step ordering** — how it breaks tasks into sub-steps, granularity, order choices.
5. **Self-checking and verification** — what kinds of checks it performs (recomputation, sanity checks, plug-in verification, alternative methods), when, and how reliable they are.
6. **Backtracking, corrections, missteps** — every place it detects and corrects an error, or fails to; quote them.
7. **Errors and failure modes** — concrete mistakes found in the transcripts (arithmetic slips, logical gaps, wrong formulas, constraint violations) with quotes and the correct answer if it ultimately self-corrects.
8. **Formatting, whitespace and layout (verbatim)** — this is a first-class analysis target: whitespace is part of the reasoning_content, not decoration. Characterize its exact layout behavior using the per-part whitespace profiles and verbatim quotes: line-break structure (stream-of-consciousness single line vs multiple lines vs paragraphs), indentation usage, blank lines, double/trailing spaces, tabs, capitalization at line starts, punctuation density, how it segments thoughts. Include at least two fenced code-block excerpts that preserve the exact whitespace so a reader can see it.
9. **Instruction following and constraint tracking** — how it tracks and enforces the explicit output constraints in the instruction-following run; does the reasoning reflect the constraints before the final answer is produced? Does it comply? Same analysis for the tool-use run: how it decides what tool to call, whether it uses exactly the tool it was told to, and how its plan unfolds across the multiple turns.
10. **Language and style** — tone, vocabulary, verbosity, redundancy, token efficiency (comment on where reasoning is bloated vs terse, using per-turn usage numbers), repetitive boilerplate, self-address, hedging language, confidence markers.
11. **Cross-run consistency** — what stays the same across all runs and what varies per task category (mathematics vs code vs proof vs constrained output vs tool use).
12. **Summary: how {SMALL_MODEL} thinks** — a compact synthesis of its "thinking personality", including notable artifacts a developer should know about (e.g. for prompt design or fine-tuning).

Rules:
- Ground every claim in evidence: quote the raw reasoning_content VERBATIM (short quotes, up to ~40 words; use fenced code blocks for any quote whose exact whitespace/line breaks you want to show) and cite which run/turn it comes from, e.g. (Run 2, Turn 1).
- Never normalize, collapse, or "clean up" quoted whitespace — reproduce it exactly, including blank lines and double spaces.
- Keep the document SIMPLE and direct — do not over-complicate it. Plain prose and short bullet points; no elaborate schematics, no nested formatting.
- Do NOT use markdown tables anywhere in the document. Use simple bullet points for all enumerations.
- Be precise and concrete; avoid generic statements that could apply to any model.
- Do not invent content that is not in the transcripts.
- Keep the document well under 3000 words.
- After writing the file, reply with a 3-5 sentence summary of what you found.

## Raw transcripts

{transcript}
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f'[lfm2.5-analyze] {msg}', flush=True)


def main() -> None:
    cfg = load_models_cfg()
    small_provider_name, small_provider, small_m = find_model(cfg, SMALL_MODEL)
    analyzer_provider_name, analyzer_provider, analyzer_m = find_model(cfg, ANALYZER_MODEL)
    small_meta = {
        'provider': small_provider_name,
        'api': small_provider.get('api'),
        'base_url': small_provider.get('baseUrl'),
        'sampling': small_m.get('samplingParams', {}),
    }
    analyzer_meta = {'sampling': analyzer_m.get('samplingParams', {}), 'thinkingLevelMap': analyzer_m.get('thinkingLevelMap', {})}
    log(f'model under analysis: {SMALL_MODEL} (provider {small_meta["provider"]}, api {small_meta["api"]}, baseUrl {small_meta["base_url"]}, sampling {small_meta["sampling"]})')
    log(f'analyzer: {ANALYZER_MODEL} (thinking {ANALYZER_THINKING}; thinkingLevelMap {analyzer_meta["thinkingLevelMap"]})')
    log(f'api key for {small_meta["provider"]} is taken from models.json/auth by pi itself (not read by this script)')

    SESSION_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    transcripts: list[str] = []
    for i, (label, task, allow_tools) in enumerate(TASKS, 1):
        log(f'run {i}/{len(TASKS)} ({label}, tools={allow_tools}): {task[:80]!r}...')
        sess = None
        for attempt in range(1, 4):  # fresh session dir per attempt (stalls leave partial files)
            with tempfile.TemporaryDirectory(prefix='pi-sessions-') as session_dir:
                try:
                    stdout = pi(
                        SMALL_MODEL,
                        SMALL_THINKING,
                        task,
                        allow_tools=allow_tools,
                        temp=True,              # throwaway working directory
                        extra=('--session-dir', session_dir),
                        timeout=900,            # normal runs take <15s; 15min is a stall
                    )
                    jsonl = find_session_jsonl(Path(session_dir))
                    sess = parse_session_jsonl(jsonl)   # parse+copy INSIDE the with-block (temp dir dies after)
                    shutil.copy2(jsonl, SESSION_SAVE_DIR / f'{label}__{jsonl.name}')
                    break
                except (RuntimeError, OSError) as exc:
                    log(f'  run failed (attempt {attempt}/3): {str(exc)[:300]!r}')
                    if attempt < 3:
                        time.sleep(10 * attempt)
        if sess is None:
            raise RuntimeError(f'task {label!r}: pi failed 3 times')
        n_thinking = sum(len(t['thinking']) for t in sess['turns'])
        n_chars = sum(len(x) for t in sess['turns'] for x in t['thinking'])
        log(f'  session saved -> {label}__{sess["file"].name}: {len(sess["turns"])} assistant turn(s), '
            f'{n_thinking} reasoning_content part(s), {n_chars} chars; stdout: {stdout.strip()[:120]!r}')
        transcripts.append(render_transcript(i, len(TASKS), task, sess, allow_tools))

    transcript = '\n\n'.join(transcripts)
    TRANSCRIPTS_TXT.write_text(transcript)
    log(f'transcripts ({len(transcript)} chars) -> {TRANSCRIPTS_TXT.relative_to(HERE)}')

    log(f'analyzing with {ANALYZER_MODEL} ({ANALYZER_THINKING}) ...')
    prompt = build_analyzer_prompt(transcript, small_meta, analyzer_meta)
    analyzer_out = None
    for attempt in range(1, 4):
        with tempfile.TemporaryDirectory(prefix='pi-sessions-') as session_dir:
            try:
                analyzer_out = pi(
                    ANALYZER_MODEL,
                    ANALYZER_THINKING,
                    prompt,
                    allow_tools=True,
                    cwd=str(HERE),     # so the `write` tool lands the file next to this script
                    extra=('--session-dir', session_dir),
                    timeout=3600,
                )
                break
            except RuntimeError as exc:
                log(f'  analyzer failed (attempt {attempt}/3): {str(exc)[:300]!r}')
                if attempt < 3:
                    time.sleep(10 * attempt)
    if analyzer_out is None:
        raise RuntimeError('analyzer failed 3 times')
    log(f'analyzer replied:\n{analyzer_out.strip()}')

    if not OUT_MD.exists():
        # Fallback: the model printed the document instead of writing the file.
        log('WARNING: file was not written by the analyzer; saving stdout instead')
        txt = analyzer_out.strip()
        txt = re.sub(r'^```(?:markdown|md)?\s*\n', '', txt)
        txt = re.sub(r'\n```\s*$', '', txt)
        OUT_MD.write_text(txt + '\n')

    log(f'DONE -> {OUT_MD}')


if __name__ == '__main__':
    main()
