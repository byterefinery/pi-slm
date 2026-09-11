# ruff: noqa: I001, EXE001
import json
import os
import shutil
import subprocess
from pathlib import Path
# from copy import deepcopy
from tempfile import TemporaryDirectory


def pi(model: str, thinking: str, prompt: str | list[str], tools: str='read,write,edit,bash', skills: list[str]=[], extensions: list[str]=[], session: str | None=None, session_dir: str | None=None, session_id: str | None=None, cwd: str | None=None, temp: bool=False, env: dict | None=None, sandbox: bool=False, debug: bool=False) -> str:
    cmd = []

    if sandbox:
        cmd += [
            'firejail',
            '--noprofile',
            '--private',
            '--quiet',
            '--deterministic-exit-code',
        ]

    cmd += [
        'pi',
    ]

    cmd += [
        '--model', model,
        '--thinking', thinking,
        '--no-extensions',
        '--no-skills',
        '--no-context-files',
        '--no-tools',
    ]

    if tools:
        cmd += [
            '--tools',
            tools,
        ]

    for skill in skills:
        cmd += [
            '--skill',
            skill,
        ]

    for extension in extensions:
        cmd += [
            '--extension',
            extension,
        ]

    if session:
        cmd += [
            '--session',
            session,
        ]

    if session_dir:
        cmd += [
            '--session-dir',
            session_dir,
        ]

    if session_id:
        cmd += [
            '--session-id',
            session_id,
        ]

    if isinstance(prompt, str):
        cmd += [prompt]
    elif isinstance(prompt, list):
        for n in prompt:
            cmd += [n]
    else:
        raise TypeError(prompt)

    cmd += ['-p']

    if debug:
        print(f'{cmd=}')

    proc_kwargs = {}
    temp_dir: TemporaryDirectory | None = None

    if cwd:
        proc_kwargs['cwd'] = cwd
    elif temp:
        temp_dir = TemporaryDirectory()
        proc_kwargs['cwd'] = temp_dir.name

    if env:
        proc_kwargs['env'] = env

    proc = subprocess.run( # type: ignore # noqa
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **proc_kwargs,
    )

    if temp:
        del temp_dir

    if proc.stderr:
        raise RuntimeError(proc.stderr)

    stdout = proc.stdout.decode()
    return stdout


def run_isolated_pi(copy_skills: dict={}, override_file_content: dict | None=None, *args, **kwargs) -> tuple[str, str]:
    with TemporaryDirectory(delete=False) as td, TemporaryDirectory(delete=False) as tsd: # type: ignore no-matching-overload
        # print(f'{td=}')
        # print(f'{tsd=}')
        print(f'{td=} {tsd=}')

        os.makedirs(Path(str(td)) / '.pi' / 'agent', exist_ok=True)
        os.makedirs(Path(str(td)) / '.agents' / 'skills', exist_ok=True)

        for dst, src in copy_skills.items():
            shutil.copytree(
                src,
                Path(str(td)) / dst,
                dirs_exist_ok=True,
            )

        # .pi
        shutil.copy(
            Path.home() / '.pi' / 'agent' / 'models.json',
            Path(str(td)) / '.pi' / 'agent' / 'models.json',
        )

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
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ["HOME"],
            "USER": os.environ.get("USER", ""),
            "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "TERM": os.environ.get("TERM", "dumb"),
            "TMPDIR": "/tmp",
            "PI_CODING_AGENT_DIR": Path(str(td)) / '.pi' / 'agent',
            "PI_SKIP_VERSION_CHECK": "1",
        }

        pi_output: str = pi(
            *args,
            session_dir=tsd,
            cwd=td,
            env=env,
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


def _text_of_content(content) -> str | None:
    """Join the text of a pi message content field; None if there is no text."""
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            block['text']
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)
        ]
        return '\n'.join(parts) if parts else None
    return None


def extract_messages(session: str) -> list[dict]:
    """Extract minimal OpenAI-compatible chat messages from a pi session file.

    `session` is the full content of a pi session `.jsonl` file (e.g. the
    `pi_session_content` returned by `run_isolated_pi`).

    Only `message` entries are kept, in file order. Everything else (session
    header, model_change, thinking_level_change, ...) and all `thinking`
    blocks are dropped. Mapping:

    - user messages            -> {'role': 'user', 'content': str}
    - assistant text           -> {'role': 'assistant', 'content': str}
    - assistant toolCall blocks-> {'role': 'assistant', 'content': str | None,
                                  'tool_calls': [{'id', 'type': 'function',
                                                  'function': {'name', 'arguments' (JSON str)}]}]
    - toolResult messages      -> {'role': 'tool', 'tool_call_id': str, 'content': str}
    """
    messages: list[dict] = []

    for line in session.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get('type') != 'message':
            continue

        msg = entry.get('message')
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content')

        if role == 'user':
            text = _text_of_content(content)
            if text is not None:
                messages.append({'role': 'user', 'content': text})

        elif role == 'assistant':
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'text' and isinstance(block.get('text'), str):
                        text_parts.append(block['text'])
                    elif block.get('type') == 'toolCall':
                        tool_calls.append({
                            'id': block.get('id', ''),
                            'type': 'function',
                            'function': {
                                'name': block.get('name', ''),
                                'arguments': json.dumps(block.get('arguments', {})),
                            },
                        })
            if tool_calls:
                messages.append({
                    'role': 'assistant',
                    'content': '\n'.join(text_parts) if text_parts else None,
                    'tool_calls': tool_calls,
                })
            elif text_parts:
                messages.append({'role': 'assistant', 'content': '\n'.join(text_parts)})

        elif role == 'toolResult':
            messages.append({
                'role': 'tool',
                'tool_call_id': msg.get('toolCallId', ''),
                'content': _text_of_content(content) or '',
            })

    return messages


if __name__ == '__main__':
    MODEL = ("Qwen/Qwen3.8-27B", "low")

    output = pi(
        model=MODEL[0],
        thinking=MODEL[1],
        prompt='Hi, print cwd and list curren dir, print home dir, list dir tree of `~/.pi`',
        sandbox=True,
        debug=True,
    )
    print(output)
    print('-' * 80)
