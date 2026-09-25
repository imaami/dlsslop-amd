"""Shared validation for the offline import and bounded dependency fetch tools."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT, check=True, input=None, timeout=None, env=None):
    settings = os.environ.copy()
    settings.update({'GIT_LFS_SKIP_SMUDGE': '1', 'GIT_TERMINAL_PROMPT': '0', 'LC_ALL': 'C'})
    if env:
        settings.update(env)
    process = subprocess.Popen(['git', *args], cwd=cwd, text=True,
                               stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=settings, start_new_session=True)
    output, errors = [], []
    rejected_filter = threading.Event()

    def stop():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def read_errors():
        for line in process.stderr:
            errors.append(line)
            # Repeat the capability guard on the real transfer connection and
            # on lazy blob fetches. Git otherwise warns and fetches everything.
            if 'filtering not recognized by server' in line:
                rejected_filter.set()
                stop()

    readers = [threading.Thread(target=lambda: output.append(process.stdout.read())),
               threading.Thread(target=read_errors)]
    for reader in readers:
        reader.start()
    try:
        if input is not None:
            try:
                process.stdin.write(input)
                process.stdin.close()
            except BrokenPipeError:
                pass
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
        process.stdout.close()
        process.stderr.close()
    result = subprocess.CompletedProcess(process.args, process.returncode,
                                         ''.join(output), ''.join(errors))
    if rejected_filter.is_set():
        raise RuntimeError('server dropped blob-filter support during transfer; '
                           'stopped Git rather than continuing an unfiltered fetch')
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or
                           f'git {args[0]} failed with status {result.returncode}')
    return result


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value or
            any(part in ('.', '..', '.git') for part in path.parts) or
            any(char in value for char in ('\0', '\n', '\r', '\\'))):
        raise RuntimeError(f'unsafe repository path: {value!r}')
    return path


def load_repositories():
    top = git('rev-parse', '--show-toplevel', check=False)
    if top.returncode or Path(top.stdout.strip()).resolve() != ROOT:
        raise RuntimeError('run git init in the project root first')
    lock = json.loads((ROOT / 'upstreams.lock.json').read_text())
    if lock.get('version') != 1:
        raise RuntimeError('unsupported upstream lock version')
    repositories = lock['repositories']
    paths = []
    for name, row in repositories.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
            raise RuntimeError(f'unsupported submodule name: {name!r}')
        if not re.fullmatch(r'[0-9a-f]{40}', row['commit']):
            raise RuntimeError(f'invalid pinned commit for {name}')
        path = relative_path(row['path'])
        if any(path == previous or previous in path.parents or path in previous.parents
               for previous in paths):
            raise RuntimeError('submodule paths must be distinct and non-nested')
        paths.append(path)
        for parent in [ROOT / path, *(ROOT / path).parents]:
            if parent == ROOT:
                break
            if parent.is_symlink():
                raise RuntimeError(f'submodule path contains a symlink: {parent}')
        declared = git('config', '-f', '.gitmodules', '--get', f'submodule.{name}.path')
        if declared.stdout.strip() != row['path']:
            raise RuntimeError(f'.gitmodules path differs from lock for {name}')
        if not git('config', '-f', '.gitmodules', '--get', f'submodule.{name}.url').stdout.strip():
            raise RuntimeError(f'missing submodule URL for {name}')
        sources = sorted({file['source'] for file in lock['files'].values()
                          if file['repository'] == name})
        if not sources:
            raise RuntimeError(f'no mapped source files for {name}')
        for source in sources:
            relative_path(source)
        row['sources'] = sources
    return repositories


def check_gitlink(row, required=True):
    entries = git('ls-files', '--stage', '-z', '--', row['path']).stdout.split('\0')
    entries = [entry for entry in entries if entry]
    expected = f"160000 {row['commit']} 0\t{row['path']}"
    if entries != [expected] and (entries or required):
        raise RuntimeError(f"missing or mismatched gitlink at {row['path']}; "
                           'run scripts/init-repo.py for a fresh source import')


def module_gitdir(name):
    common = Path(git('rev-parse', '--git-common-dir').stdout.strip())
    if not common.is_absolute():
        common = ROOT / common
    return common.resolve() / 'modules' / name


def inspect_worktree(name, row, timeout=300):
    """Refuse to overwrite files or local edits, including untracked files."""
    path = ROOT / row['path']
    if not path.exists():
        return False
    if not path.is_dir():
        raise RuntimeError(f'not a submodule directory: {path}')
    if not (path / '.git').exists():
        if any(path.iterdir()):
            raise RuntimeError(f'nonempty uninitialized submodule directory: {path}')
        return False
    top = git('rev-parse', '--show-toplevel', cwd=path).stdout.strip()
    actual = git('rev-parse', '--absolute-git-dir', cwd=path).stdout.strip()
    if Path(top).resolve() != path or Path(actual).resolve() != module_gitdir(name):
        raise RuntimeError(f'unexpected submodule Git directory at {path}')
    if git('status', '--porcelain', '--untracked-files=all', '--ignored=matching',
           cwd=path, timeout=timeout).stdout:
        raise RuntimeError(f'local changes in {path}; preserve them before fetching')
    return True
