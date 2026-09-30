#!/usr/bin/env python3
"""Check the HIP network's weight element counts against the model import's manifest."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('test', type=Path, help='hip-weights-test executable (required; no default)')
args = parser.parse_args()
script = Path(__file__).resolve().parent.parent / 'scripts/fetch-assets.py'


def run(*command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, (command, result.returncode, result.stdout, result.stderr)
    return result.stdout


manifest = json.loads(run(sys.executable, str(script), '--print-manifest'))
# The element count the network expects of each stem it loads. --list has a
# line per packed weight, so a stem packed more than one way comes up again.
loaded = {}
for line in run(str(args.test.resolve()), '--list').splitlines():
    stem, elements, _, name = line.split('\t')
    assert loaded.setdefault(stem, int(elements)) == int(elements), f'{name}: {elements} elements'
# dlsslop-setup imports exactly the weights the network loads, at its sizes.
assert loaded == manifest, (
    {stem: (count, manifest.get(stem)) for stem, count in loaded.items() if manifest.get(stem) != count},
    sorted(manifest.keys() - loaded.keys()))
