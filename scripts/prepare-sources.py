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

from submodule_support import PREPARED, ROOT, STAGE_ENV, stage_originals

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(destination):
    with tempfile.TemporaryDirectory(prefix='dlsslop-amd-sources-') as temp:
        stage = Path(temp)
        stage_originals(stage)
        subprocess.check_call(['git', 'apply', '--binary', str(ROOT / 'patches/linux-integration.patch')],
                              cwd=stage, env=STAGE_ENV)
        staged = {path.relative_to(stage).as_posix(): digest(path)
                  for folder in PREPARED for path in (stage / folder).rglob('*') if path.is_file()}
        # Never overwrite local edits: an existing file must hold what the
        # previous preparation recorded or what this one writes. A fresh output
        # directory permits inspection without touching an existing workspace.
        record = destination / '.prepared-sources.json'
        previous = json.loads(record.read_text()) if record.exists() else {}
        for folder in PREPARED:
            for path in (destination / folder, *(destination / folder).rglob('*')):
                if path.is_symlink():  # Would be read, and written, through.
                    raise RuntimeError(f'symlink in prepared source: {path}')
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
