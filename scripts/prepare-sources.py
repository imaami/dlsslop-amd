#!/usr/bin/env python3
"""Reconstruct build sources from pinned submodules and local patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('upstream-layer', 'kernels', 'backend/vendor')
# The staging repository must not inherit Git configuration, attributes or
# GIT_* variables: diff.noprefix, color, core.autocrlf, apply.whitespace, hooks,
# signing, eol attributes or a calling hook's GIT_DIR or GIT_INDEX_FILE would
# silently change or break the patch.
STAGE_ENV = dict({key: value for key, value in os.environ.items() if not key.startswith('GIT_')},
                 GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_ATTR_NOSYSTEM='1', GIT_CONFIG_COUNT='1',
                 GIT_CONFIG_KEY_0='core.attributesFile', GIT_CONFIG_VALUE_0=os.devnull)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(*args, cwd=ROOT, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env)

def prepare(destination):
    lock = json.loads((ROOT / 'upstreams.lock.json').read_text())
    for row in lock['repositories'].values():
        repo = ROOT / row['path']
        if run('git', 'rev-parse', 'HEAD', cwd=repo).decode().strip() != row['commit']:
            raise RuntimeError(f"wrong submodule revision: {repo}; run python3 scripts/fetch-submodules.py")
    with tempfile.TemporaryDirectory(prefix='dlsslop-amd-sources-') as temp:
        stage = Path(temp)
        run('git', 'init', '-q', cwd=stage, env=STAGE_ENV)
        for name, row in lock['files'].items():
            repo = lock['repositories'][row['repository']]
            data = run('git', 'show', repo['commit'] + ':' + row['source'], cwd=ROOT / repo['path'])
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError(f'upstream source hash mismatch: {name}')
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        run('git', 'apply', '--binary', str(ROOT / 'patches/linux-integration.patch'), cwd=stage, env=STAGE_ENV)
        staged = {path.relative_to(stage).as_posix(): digest(path)
                  for folder in DIRECTORIES for path in (stage / folder).rglob('*') if path.is_file()}
        # Never overwrite local edits: an existing file must hold what the
        # previous preparation recorded or what this one writes. A fresh output
        # directory permits inspection without touching an existing workspace.
        record = destination / '.prepared-sources.json'
        previous = json.loads(record.read_text()) if record.exists() else {}
        for folder in DIRECTORIES:
            for path in (destination / folder).rglob('*'):
                name = path.relative_to(destination).as_posix()
                if path.is_file() and digest(path) not in (previous.get(name), staged.get(name)):
                    raise RuntimeError(f'local changes at {path}; preserve them before preparing sources')
        for name in previous.keys() - staged.keys():
            (destination / name).unlink(missing_ok=True)
        for name in staged:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(stage / name, target)
        record.write_text(json.dumps(staged))
    print(f'Verified and prepared {len(staged)} source files in {destination}')

def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('-o', '--output', type=Path, default=ROOT,
                        help='destination root (default: project root)')
    parser.add_argument('-h', '--help', action='help', help='show help (default: off)')
    args = parser.parse_args()
    try:
        prepare(args.output.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'prepare-sources: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
