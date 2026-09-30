#!/usr/bin/env python3
"""Check shmclient, the fake layer the tracers serve dlsslopd frames with.

Its help must give every option's default and every dlsslopctl setting as an
option; usage errors exit 2. Served by dlsslopd --test-identity, which answers
each frame with its input and needs no device, the inputs of frames A, B and f
at 1280x720 must have the FNV-1a 64 that the tracers' baselines were taken
with, as RGBA8 and FP16, the settings must be stored before the frames they
name, and --dump must write the answers. The daemon, started through a wrapper
that runs dlssnr-shmctl first, must find the settings of frame 0 in the
channel and one control generation. A tier from a later frame must be refused
with a daemon that publishes no neural raster.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-h', '--help', action='help', help='show this help and exit (default: off)')
parser.add_argument('client', type=Path, help='shmclient (required; no default)')
parser.add_argument('daemon', type=Path, help='dlsslopd (required; no default)')
parser.add_argument('shmctl', type=Path, help='dlssnr-shmctl (required; no default)')
args = parser.parse_args()

# FNV-1a 64 of the 1280x720 inputs, as the tracers' baselines record them.
INPUTS = {('A', False): '7abb62153ad78725', ('B', False): 'cf59c34e4c7eb465', ('f', False): '7c7f4e6aba8ca325',
          ('A', True): 'e00f4ae59eb4a0f1', ('B', True): 'cec010e13b0b9551'}


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run(command, expected=0, stderr=False):
    result = subprocess.run([str(c) for c in command], text=True, capture_output=True, timeout=60)
    assert result.returncode == expected, (command, result.returncode, result.stdout, result.stderr)
    return result.stderr if stderr else result.stdout


helptext = run([args.client, '--help'])
own, settings = helptext.split('\n\n', 1)[1].split('\n\nSettings', 1)
options = re.split(r'\n(?=  -)', own.strip('\n'))
assert [re.match(r'  -\w, --([\w-]+)', entry)[1] for entry in options] == \
    ['shm', 'width', 'height', 'frames', 'fp16', 'dump', 'log', 'help'], options
assert all('\n                       Default: ' in entry for entry in options), options
for name, default in (('tier', '720'), ('passes', '1'), ('mvec', '0'), ('sharpness', '0'), ('color-preserve', '0'),
                      ('intensity', '1'), ('skin-structure', '-1'), ('style', '0'), ('auto-mask', '1')):
    assert re.search(rf'\n      --{name} \[FRAME:\]VALUE\n.*\n +Range: [-0-9.]+\.\.[0-9.]+; default: {default}\n',
                     settings), (name, settings)

with tempfile.TemporaryDirectory(prefix='shmclient-test-') as directory:
    root = Path(directory)
    shm, dump = root / 'channel.bin', root / 'dump'
    dump.mkdir()
    client = [args.client, '--shm', shm, '--width', '1280', '--height', '720']
    daemon = ['--', args.daemon, '--config', '/dev/null', '--test-identity', '--shm', shm]
    for arguments in (['--width', '1'], ['--frames', 'a'], ['--frames', ''], ['--sharpness', '2'],
                      ['--sharpness', '3:0.5'], ['--tier', '800'], ['--passes', '0'], ['--mvec', '1:'],
                      ['--bogus', '1']):
        run([*client, *arguments, '--', 'true'], expected=2)
    run([args.client, '--width', '1280', '--height', '720', '--', 'true'], expected=2)
    run([*client], expected=2)

    # The daemon starts after the channel's own status and settings are written.
    status, wrapper = root / 'status.txt', root / 'dlsslopd'
    wrapper.write_text(f'#!{sys.executable}\n'
                       'import os, subprocess, sys\n'
                       f'with open({str(status)!r}, "w") as out:\n'
                       f'    subprocess.run([{str(args.shmctl)!r}, "--shm", {str(shm)!r}, "--status", "--settings"],'
                       ' stdout=out, check=True)\n'
                       f'os.execv({str(args.daemon)!r}, [{str(args.daemon)!r}, *sys.argv[1:]])\n')
    wrapper.chmod(0o755)
    out = run([*client, '--frames', 'ABf', '--dump', dump, '--mvec', '1', '--intensity', '0:0.5', '--sharpness=1:0.5',
               '--color-preserve', '1:0.25', '--sharpness', '2:0', '--log', root / 'daemon.log', '--', wrapper,
               *daemon[2:]])
    lines = out.splitlines()
    assert lines[:2] == ['settings: mvec=1 intensity=0.5 before frame 0', 'ready tier=720 state=4 fp16=0'], lines
    stored = status.read_text().splitlines()
    for line in ('control_seq=1', 'tuning_seq=1', 'mvec=1 default=0', 'intensity=0.5 default=1',
                 'sharpness=0 default=0', 'color-preserve=0 default=0'):
        assert line in stored, (line, stored)
    assert lines[3] == 'settings: sharpness=0.5 color-preserve=0.25 before frame 1', lines
    assert lines[5] == 'settings: sharpness=0 before frame 2', lines
    assert lines[-1] == 'daemon exit=0 failures=0', lines
    frames = [line for line in lines if line.startswith('frame ')]
    assert len(frames) == 3, lines
    for index, (letter, line) in enumerate(zip('ABf', frames)):
        want = INPUTS[(letter, False)]
        assert line.startswith(f'frame {index} input={letter} input_fnv={want} request={index + 1} ok=1 '
                               f'answered=1280x720 answer_fnv={want} ms='), line
        answer = (dump / f'answer-{index}.rgba8').read_bytes()
        assert len(answer) == 1280 * 720 * 4 and f'{fnv(answer):016x}' == want, index
    assert 'IDENTITY TEST' in (root / 'daemon.log').read_text()

    out = run([*client, '--fp16', '--frames', 'AB', *daemon])
    assert 'ready tier=720 state=4 fp16=1' in out, out
    for index, letter in enumerate('AB'):
        want = INPUTS[(letter, True)]
        assert f'frame {index} input={letter} input_fnv={want} request={index + 1} ok=1 answered=1280x720 ' \
               f'answer_fnv={want} ms=' in out, out
    assert not shm.exists()

    # Identity mode publishes no raster, so a later tier could not be waited for.
    err = run([*client, '--frames', 'AB', '--tier', '1:1080', *daemon], expected=2, stderr=True)
    assert 'shmclient: --tier from frame 1 needs a daemon that publishes its neural raster' in err, err
    run([*client, '--frames', 'AB', '--tier', '0:1080', *daemon])

print('PASS: shmclient options, inputs, settings and answers')
