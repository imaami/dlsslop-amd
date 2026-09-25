#!/usr/bin/env python3
"""Exercise dlsslop-setup import and --check on sparse manifest-sized weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.parse_args()
script = Path(__file__).resolve().parent.parent / 'scripts/fetch-assets.py'


def run(*options, expected=0):
    result = subprocess.run([sys.executable, str(script), *map(str, options)],
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == expected, (options, result.returncode, result.stdout, result.stderr)
    return result


def sparse(path, size):
    with path.open('wb') as weights:
        weights.truncate(size)


manifest = json.loads(run('--print-manifest').stdout)
# FP16 weights halve the copies the import writes. One weight also has an FP32
# file, which takes precedence, so the import keeps it and drops the FP16 one.
both = 'block0-attention'
complete = (f'Complete model: {len(manifest)} weights; '
            f'{(sum(manifest.values()) + manifest[both]) * 2:,} bytes\n')

with tempfile.TemporaryDirectory(prefix='dlsslop-setup-cli-') as directory:
    root = Path(directory)
    source = root / 'native-game-tiled-assets'
    source.mkdir()
    for base, count in manifest.items():
        sparse(source / (base + '.f16'), count * 2)
    sparse(source / (both + '.f32'), manifest[both] * 4)
    # An upstream asset directory records no hashes; complete sizes pass.
    assert run('--check', source).stdout == complete

    # The import keeps an existing model directory's HIP modules and records
    # hashes in sha256sum's format, which --check then verifies.
    model = root / 'model'
    (model / 'HIP/gfx1201').mkdir(parents=True)
    run('--source', source, '--output', model)
    assert (model / 'HIP/gfx1201').is_dir()
    assert (model / (both + '.f32')).is_file() and not (model / (both + '.f16')).exists()
    sums = model / 'WEIGHTS-SHA256SUMS'
    assert len(sums.read_text().splitlines()) == len(manifest)
    subprocess.run(['sha256sum', '--quiet', '--check', sums.name], cwd=model, check=True, timeout=120)
    assert run('--check', model).stdout == complete
    result = run('--source', source, '--output', model, expected=1)
    assert 'weights already exist in' in result.stderr, result.stderr

    # An unrecorded FP32 file would take precedence at run time, so --check
    # fails although every recorded file still matches its hash.
    extra = model / 'head-matrix.f32'
    sparse(extra, manifest['head-matrix'] * 4)
    subprocess.run(['sha256sum', '--quiet', '--check', sums.name], cwd=model, check=True, timeout=120)
    result = run('--check', model, expected=1)
    assert f'{sums}: recorded hashes do not match the weights' in result.stderr, result.stderr
    extra.unlink()
    assert run('--check', model).stdout == complete

    # A corrupted weight of the right size fails against the recorded hashes.
    with (model / (both + '.f32')).open('r+b') as weights:
        weights.seek(100)
        weights.write(b'\x01')
    result = run('--check', model, expected=1)
    assert f'{sums}: recorded hashes do not match the weights' in result.stderr, result.stderr
    assert not result.stdout, result.stdout

print('setup CLI: sparse import, FP32 precedence, recorded hashes, unrecorded and corrupted weights passed')
