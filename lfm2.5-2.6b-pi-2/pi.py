import subprocess
from tempfile import TemporaryDirectory


def pi(model: str, thinking: str, prompt: str, tools: str='"read,write,edit,bash"', temp: bool=True, session: str | None=None, debug: bool=False) -> str:
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

    if session:
        cmd += [
            '--session',
            session,
        ]

    cmd += [
        prompt,
        '-p',
    ]

    if debug:
        print(f'{cmd=}')

    proc_kwargs = {}
    temp_dir: TemporaryDirectory | None = None

    if temp:
        temp_dir = TemporaryDirectory()
        proc_kwargs['cwd'] = temp_dir.name

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
