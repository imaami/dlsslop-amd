#!/usr/bin/env python3
"""Exercise dlsslop-gui option parsing, which must work without a display."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('gui', type=Path, help='dlsslop-gui executable (required; no default)')
args = parser.parse_args()
gui = str(args.gui.resolve())
assert Path(gui).read_bytes().startswith(b'\x7fELF'), 'GUI tests require the native ELF'

# No display server: parsing must finish before Qt looks for one.
headless = {key: value for key, value in os.environ.items()
            if key not in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM', 'DLSSNR_SHM')}


def run(*options, expected, env=headless):
    result = subprocess.run([gui, *options], env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == expected, (options, result.returncode, result.stdout, result.stderr)
    return result


helptext = run('--help', expected=0).stdout
assert run('-h', expected=0).stdout == helptext
assert f'effective: /tmp/dlsslop-amd-{os.getuid()}/shm.bin)' in helptext, helptext
assert 'numeric arrow step: 0.01;' in helptext, helptext
assert 'effective: /tmp/dlsslop-amd-' in run('--help', expected=0, env=dict(headless, DLSSNR_SHM='')).stdout
assert 'effective: /run/channel)' in run('--help', expected=0, env=dict(headless, DLSSNR_SHM='/run/channel')).stdout
assert 'effective: /tmp/other)' in run('--shm', '/tmp/other', '--help', expected=0).stdout
assert 'effective: /tmp/other)' in run('-s/tmp/other', '-h', expected=0).stdout

for options, message in ((('--shm=',), '--shm requires a nonempty path; try --help'),
                         (('-s', ''), '--shm requires a nonempty path; try --help'),
                         (('--shm',), "option '--shm' requires an argument"),
                         (('-s',), 'option requires an argument'),
                         (('--bogus',), "unrecognized option '--bogus'"),
                         (('extra',), 'Unexpected argument; try --help'),
                         (('--shm', '/tmp/other', 'extra'), 'Unexpected argument; try --help')):
    result = run(*options, expected=2)
    assert message in result.stderr and not result.stdout, (options, result.stdout, result.stderr)

print('GUI CLI tests passed: help, environment and option defaults, empty paths, missing and unknown options')
