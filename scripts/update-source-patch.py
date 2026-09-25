#!/usr/bin/env python3
"""Record edited prepared source against pinned originals; review the Git diff."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('-h','--help',action='help',help='show help (default: off)')
    parser.parse_args()
    lock = json.loads((ROOT/'upstreams.lock.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        def git(*args):
            return subprocess.check_output(['git','-C',str(stage),*args])
        git('init','-q')
        for name,row in lock['files'].items():
            repo = lock['repositories'][row['repository']]
            if subprocess.check_output(['git','-C',str(ROOT/repo['path']),'rev-parse','HEAD']).decode().strip() != repo['commit']:
                raise RuntimeError('submodule revision changed; update provenance separately')
            data = subprocess.check_output(['git','-C',str(ROOT/repo['path']),'show',repo['commit']+':'+row['source']])
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError('upstream source hash mismatch: '+name)
            path = stage/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        git('add','-f','.')
        git('-c','user.name=Source snapshot','-c','user.email=snapshot@localhost','commit','-qm','Pinned originals')
        for folder in ('upstream-layer','kernels','backend/vendor'):
            shutil.rmtree(stage/folder)
            shutil.copytree(ROOT/folder,stage/folder)
        git('add','-f','.')
        patch = git('diff','--cached','--binary','--no-ext-diff','--no-renames')
        outputs = {}; modes = {}
        for folder in ('upstream-layer','kernels','backend/vendor'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file():
                    if path.is_symlink():
                        raise RuntimeError('symlink in prepared source: '+str(path))
                    name=path.relative_to(ROOT).as_posix()
                    outputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
                    modes[name]=path.stat().st_mode & 0o777
        lock['outputs']=outputs;lock['modes']=modes
        (ROOT/'patches/linux-integration.patch').write_bytes(patch)
        (ROOT/'upstreams.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('Updated patch and hashes. Review both before committing.')

if __name__=='__main__':
    try:
        main()
    except (OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'update-source-patch: {exc}',file=sys.stderr)
        sys.exit(1)
