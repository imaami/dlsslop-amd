#!/usr/bin/env python3
"""Check vulkan-contention's options and their defaults without a GPU.

Options the tool rejects must exit with 2 and name the option, before any
Vulkan call. Options it takes must reach Vulkan, which a loader without a
driver answers with no device: exit 1, and no complaint about the options.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('tool', type=Path, help='the vulkan-contention executable (required; no default)')
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix='contention-cli-') as directory:
    # No driver: the loader finds no device, and a run touches no GPU.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('VK_', 'DISPLAY', 'WAYLAND'))}
    env['VK_DRIVER_FILES'] = str(Path(directory) / 'none.json')

    def run(*options, expected):
        result = subprocess.run([str(args.tool), *options], env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == expected, (options, result.returncode, result.stdout, result.stderr)
        return result

    usage = run('--help', expected=0).stdout
    assert run('-h', expected=0).stdout == usage
    lines = ' '.join(usage.split())
    for option, default in (('-q, --queue KIND', 'compute'), ('-s, --seconds N', '40'),
                            ('-w, --workgroups N', '64'), ('-l, --lds-bytes N', '65536'),
                            ('-i, --iterations N', '1000000'),
                            ('-d, --device N', 'the first with a queue of KIND'), ('-h, --help', 'off')):
        start = lines.index(option)
        end = lines.find(' -', start + len(option))
        assert f'(default: {default})' in lines[start:end if end >= 0 else None], (option, default, usage)
    for limits in ('1..60', '1..4096', 'from 4 to 65536', '1..2000000', 'at most 64000000 / workgroups',
                   '0..255'):
        assert limits in lines, (limits, usage)

    for options, named in ((['--seconds', '0'], '--seconds'), (['--seconds', '61'], '--seconds'),
                           (['-s', 'x'], '--seconds'), (['--seconds', '-1'], '--seconds'),
                           (['--workgroups', '0'], '--workgroups'), (['-w', '4097'], '--workgroups'),
                           (['--lds-bytes', '3'], '--lds-bytes'), (['--lds-bytes', '65537'], '--lds-bytes'),
                           (['-l', '1022'], '--lds-bytes'), (['--iterations', '0'], '--iterations'),
                           (['-i', '2000001'], '--iterations'), (['--queue', 'transfer'], '--queue'),
                           (['--device', '256'], '--device'), (['-d', ''], '--device'),
                           # More work a dispatch than the defaults' 64 x 1000000 workgroup iterations.
                           (['-w', '65'], '--iterations'), (['-i', '2000000'], '--iterations'),
                           (['--workgroups', '4096', '-i', '2000000'], '--iterations'),
                           (['-l', '4', '-w', '4096', '-i', '15626'], '--iterations'),
                           (['extra'], 'extra')):
        result = run(*options, expected=2)
        assert named in result.stderr and 'vulkan-contention:' in result.stderr, (options, result.stderr)
    run('--no-such-option', expected=2)
    run('--seconds', expected=2)

    # Accepted: the run reaches Vulkan, which has no device to offer.
    for options in ([], ['--queue', 'graphics', '-s', '60', '-w', '4096', '-l', '4', '-i', '15625'],
                    ['-w', '32', '--iterations', '2000000'], ['-q', 'compute', '--seconds=1', '--device', '0'],
                    ['--device', '255']):
        result = run(*options, expected=1)
        assert 'takes' not in result.stderr and 'unexpected' not in result.stderr, (options, result.stderr)

print('contention-cli: options, ranges and defaults as --help states them')
