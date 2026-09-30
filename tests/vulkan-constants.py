#!/usr/bin/env python3
"""Check the Vulkan network's plan against the fork's shader build and model tools: the markers that
build_network.py writes beside the SPIR-V from pipelines.json must be those the plan expects, and every
model entry the plan reads must be one that the model tools extract."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('--plan-test', type=Path, required=True,
                    help='vulkan-plan-test executable (required; no default)')
parser.add_argument('--pipelines', type=Path, required=True,
                    help="the fork's linux/shaders/rdna4/pipelines.json (required; no default)")
parser.add_argument('--model-files', type=Path, required=True,
                    help="the fork's linux/package/model-tools/model-files.txt (required; no default)")
args = parser.parse_args()


def run(*options):
    result = subprocess.run([str(args.plan_test.resolve()), *options], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, (options, result.returncode, result.stdout, result.stderr)
    return result.stdout.splitlines()


# build_network.py writes each marker as its lines, or its one line.
built = [f'{name} {line}' for name, text in json.loads(args.pipelines.read_text())['markers'].items()
         for line in (text if isinstance(text, list) else [text])]
expected = run('--markers')
assert sorted(built) == sorted(expected) and [l for l in built if l.startswith('shader-constants.txt ')] == \
    [l for l in expected if l.startswith('shader-constants.txt ')], (
    sorted(set(built) - set(expected)), sorted(set(expected) - set(built)))

extracted = set(args.model_files.read_text().split())
read = run('--entries')
assert read and set(read) <= extracted, sorted(set(read) - extracted)
print(f'vulkan-constants: {len(expected)} markers as the shader build writes them; '
      f'{len(read)} model entries the model tools extract')
