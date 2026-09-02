import subprocess
from tempfile import TemporaryDirectory


def pi(model: str, thinking: str, prompt: str, tools: str='read,write,edit,bash', skills: list[str]=[], extensions: list[str]=[], session: str | None=None, session_dir: str | None=None, session_id: str | None=None, cwd: str | None=None, temp: bool=False, env: dict | None=None, debug: bool=False) -> str:
    cmd = [
        'pi',
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

    cmd += [
        prompt,
        '-p',
    ]

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

    del temp_dir

    if proc.stderr:
        raise RuntimeError(proc.stderr)

    stdout = proc.stdout.decode()
    return stdout
