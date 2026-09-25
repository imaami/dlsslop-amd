#!/usr/bin/env python3
"""Reconstruct build sources from pinned submodules and local patches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('upstream-layer', 'kernels', 'backend/vendor')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd)

def prepare(destination):
    lock = json.loads((ROOT / 'upstreams.lock.json').read_text())
    for row in lock['repositories'].values():
        repo = ROOT / row['path']
        if run('git', 'rev-parse', 'HEAD', cwd=repo).decode().strip() != row['commit']:
            raise RuntimeError(f"wrong submodule revision: {repo}; run python3 scripts/fetch-submodules.py")
    with tempfile.TemporaryDirectory(prefix='dlsslop-amd-sources-') as temp:
        stage = Path(temp)
        run('git', 'init', '-q', cwd=stage)
        for name, row in lock['files'].items():
            repo = lock['repositories'][row['repository']]
            data = run('git', 'show', repo['commit'] + ':' + row['source'], cwd=ROOT / repo['path'])
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError(f'upstream source hash mismatch: {name}')
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        run('git', 'apply', '--binary', str(ROOT / 'patches/linux-integration.patch'), cwd=stage)
        for name, expected in lock['outputs'].items():
            if digest(stage / name) != expected:
                raise RuntimeError(f'patched source hash mismatch: {name}')
        # Never overwrite locally edited generated sources. A fresh output
        # directory permits inspection without touching an existing workspace.
        for folder in DIRECTORIES:
            existing = destination / folder
            if existing.exists():
                for path in existing.rglob('*'):
                    if path.is_file():
                        name = path.relative_to(destination).as_posix()
                        if name not in lock['outputs'] or digest(path) != lock['outputs'][name]:
                            raise RuntimeError(f'local changes at {path}; preserve them before preparing sources')
        for folder in DIRECTORIES:
            for path in (stage / folder).rglob('*'):
                if path.is_file():
                    name = path.relative_to(stage).as_posix()
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    target.chmod(lock.get('modes', {}).get(name, 420))
    print(f"Verified and prepared {len(lock['outputs'])} source files in {destination}")

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
