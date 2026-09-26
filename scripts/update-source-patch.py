#!/usr/bin/env python3
"""Record edited prepared source against pinned originals; review the Git diff."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from submodule_support import PREPARED, ROOT, STAGE_ENV, stage_originals

def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('-h','--help',action='help',help='show help (default: off)')
    parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        def git(*args):
            return subprocess.check_output(['git','-C',str(stage),*args],env=STAGE_ENV)
        stage_originals(stage)
        git('add','-f','.')
        git('-c','user.name=Source snapshot','-c','user.email=snapshot@localhost','commit','-qm','Pinned originals')
        for folder in PREPARED:
            for path in (ROOT/folder).rglob('*'):
                if path.is_symlink():
                    raise RuntimeError('symlink in prepared source: '+str(path))
            shutil.rmtree(stage/folder)
            shutil.copytree(ROOT/folder,stage/folder)
        git('add','-f','.')
        patch = git('diff','--cached','--binary','--no-ext-diff','--no-renames')
        (ROOT/'patches/linux-integration.patch').write_bytes(patch)
    print('Updated patch. Review it before committing.')

if __name__=='__main__':
    try:
        main()
    except (OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'update-source-patch: {exc}',file=sys.stderr)
        sys.exit(1)
