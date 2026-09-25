#!/usr/bin/env python3
"""Exercise real CLI validation and channel mutation without a worker/GPU."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('control', type=Path, help='dlssnr-shmctl executable (required; no default)')
args = parser.parse_args()
control = str(args.control.resolve())
assert Path(control).read_bytes().startswith(b'\x7fELF'), 'control tests require the native ELF'

with tempfile.TemporaryDirectory(prefix='dlsslopctl-cli-') as directory:
    channel = Path(directory) / 'channel with spaces.bin'
    env = dict(os.environ, DLSSNR_SHM=str(channel))

    def run(*options, expected=0):
        result = subprocess.run([control, *options], env=env, text=True, capture_output=True)
        assert result.returncode == expected, (options, result.returncode, result.stdout, result.stderr)
        return result.stdout

    def header():
        with channel.open('rb') as stream:
            return stream.read(8192)

    def settings(output):
        return {match[1]: (float(match[2]), float(match[3]))
                for match in re.finditer(r'^([a-z0-9-]+)=([^ ]+) default=([^\n]+)$', output, re.MULTILINE)}

    default_env = {key: value for key, value in env.items() if key != 'DLSSNR_SHM'}
    default_env['DLSSNR_UID'] = '12345'
    default_help = subprocess.run([control, '--help'], env=default_env, text=True,
                                  capture_output=True, check=True).stdout
    assert f'effective default: /tmp/dlsslop-amd-{os.getuid()}/shm.bin' in default_help

    helptext = run('--help')
    assert not channel.exists(), '--help created the channel'
    assert '--working-scale' in helptext and 'default: 1' in helptext
    assert 'effective default: ' + str(channel) in helptext
    assert '1 with worker --cpu-compose or --test-identity' in helptext
    shared_header = Path(__file__).resolve().parent.parent / 'upstream-layer/common/shm_protocol.h'
    tier = re.search(r'kNativeDefaultTier\s*=\s*(\d+)', shared_header.read_text())[1]
    max_passes = int(re.search(r'kMaxPasses\s*=\s*(\d+)', shared_header.read_text())[1])
    assert f'worker default: {tier}' in helptext
    assert run('-h') == helptext
    run('--settings', expected=1)
    run('--status', expected=1)
    assert not channel.exists(), 'read-only operation created the channel'
    run('--working-scale', '0.5', '--enabled', '2', expected=2)
    assert not channel.exists(), 'invalid later argument created the channel'
    run('--working-scale', '0.5', '--preset', '1', expected=2)
    assert not channel.exists(), 'unsupported captured configuration created the channel'
    run('--quit')  # Absent worker is already stopped; no creation.
    assert not channel.exists()

    output = run('--reset', '--settings')
    values = settings(output)
    assert values['working-scale'] == (1, 1)
    assert values['passes'] == (1, 1)
    assert values['transfer'] == (2, 2)
    assert values['bypass'] == (0, 0)
    assert values['sdr16-multipass'] == (1, 1)
    assert values['hdr-mode'] == (1, 1)
    assert values['mvec'] == (0, 0)
    assert len(values) == 42, values.keys()
    assert channel.stat().st_mode & 0o777 == 0o600

    # Every supported setting advertises the same default as --settings.
    for name, (_, default) in values.items():
        block = re.search(r'--' + re.escape(name) + r'\s+VALUE[^\n]*\n([^\n]*)', helptext)
        assert block, name
        advertised = re.search(r'default: ([^ (]+)', block[1])
        assert advertised and float(advertised[1]) == default, (name, block[1], default)

    original = header()
    before = channel.stat().st_mtime_ns
    run('--settings')
    assert 'initialised=1\n' in run()
    assert header() == original and channel.stat().st_mtime_ns == before
    run('--help', '--working-scale', '0.5')
    assert header() == original

    invalid_options = [
        ('--working-scale', '0.5', '--enabled', '2'),
        ('--working-scale', 'nan'), ('--working-scale', 'inf'),
        ('--working-scale', '1garbage'), ('--working-scale', '0.249'),
        ('--working-scale', '2.001'), ('--working-scale', ' 1'),
        ('--enabled', '-1'), ('--enabled', '1.0'), ('--enabled', ''),
        ('--passes', '0'), ('--passes', str(max_passes + 1)), ('--passes', '-1'),
        ('--passes', '2.5'), ('--passes', '2x'), ('--passes',),
        ('--capture', '65'), ('--capture', '4294967296'), ('--capture', '2x'),
        ('--debug-view', '6'), ('--downscaler', '0'),
        ('--enabled',), ('--unknown',),
        ('--color-preserve', '1.001'), ('--color-preserve', '-0.1'), ('--intensity', '4.001'), ('--local-tone', '-0.1'), ('--sharpness', '1.01'),
        ('--hdr-mode', '3'), ('--mvec-quality', '3'), ('--mvec-units', '3'),
        ('--mvec-pixels', '4'), ('--white-point', '0.00009'), ('--reversible', '5'),
        ('--debug-scale', '0.0099999995'),
        ('--preset', '1'), ('--style', '1'), ('--auto-mask', '0'), ('--skin-structure', '0'),
        ('--intensity', '0.5', '--preset', '1'), ('--rebuild-ms', '5001'),
        ('--toggle', 'auto-mask'), ('--toggle', 'intensity'), ('--toggle', 'hdr-mode'),
        ('--toggle', 'unknown'), ('--toggle',), ('--toggle', 'hold', '--hold', '1'),
        ('set', 'workingscale', '1'), ('toggle', 'enabled'), ('status',),
        ('--quit', '--resume'), ('--toggle-enabled', '--enabled', '1'),
        ('--working-scale', '0.5', '--', 'extra'), ('--shm', ''),
    ]
    for options in invalid_options:
        run(*options, expected=2)
        assert header() == original, ('invalid option mutated channel', options)

    # Every advertised bound is accepted, and so is the value --settings then
    # prints, although binary32 stores some float minimums below themselves.
    def printed(output):
        return dict(re.findall(r'^([a-z0-9-]+)=([^ ]+) default=', output, re.MULTILINE))

    fixed = ('preset', 'style', 'auto-mask', 'skin-structure')
    for name in values:
        if name in fixed:
            continue
        bounds = re.search(r'--' + re.escape(name) + r'\s+VALUE[^\n]*\n\s*Range: (\S+?)\.\.([^;]+);', helptext)
        assert bounds, name
        for bound in bounds.groups():
            shown = printed(run('--' + name, bound, '--settings'))[name]
            assert printed(run('--' + name, shown, '--settings'))[name] == shown, (name, bound, shown)
    # Negative zero is stored as zero.
    assert '\nsharpness=0 default=0\n' in run('--sharpness', '-0', '--settings')
    assert '\nintensity=0 default=1\n' in run('--intensity', '-0.0', '--settings')
    run('--reset')

    # Each option below has a working short form and is read back by name.
    # Captured configuration fields accept only the one real configuration.
    restored = {
        'hdr-mode': ('E', 2), 'sdr16-multipass': ('B', 0),
        'preset': ('N', 0), 'style': ('y', 0), 'auto-mask': ('M', 1),
        'intensity': ('i', 0.75), 'local-tone': ('o', 1.25),
        'local-structure': ('j', 0.5), 'skin-structure': ('K', -1),
        'color-preserve': ('L', 0.75), 'sharpness': ('n', 0.5), 'rebuild-ms': ('J', 5000),
        'mvec': ('V', 1), 'mvec-quality': ('Q', 2), 'mvec-units': ('U', 2),
        'mvec-pixels': ('F', 3), 'white-point': ('W', 100),
        'white-point-scale': ('G', 2), 'white-point-source': ('O', 1),
        'white-point-trim': ('I', 0.5), 'color-mode': ('Y', 2), 'reversible': ('Z', 4),
    }
    for name, (short, value) in restored.items():
        changed = settings(run('-' + short, str(value), '--settings'))
        assert changed[name][0] == value, (name, changed[name])
        assert f'-{short}, --{name}' in helptext
        changed = settings(run('--' + name, format(values[name][1], 'g'), '--settings'))
        assert changed[name] == values[name], (name, changed[name])

    def tuning_sequence():
        return int(re.search(r'^tuning_seq=(\d+)$', run('--status'), re.MULTILINE)[1])

    sequence = tuning_sequence()
    run('--working-scale', '0.75', '--passes', '2')
    assert tuning_sequence() == sequence
    run('--intensity', '0.75', '--local-tone', '0.5', '--local-structure', '1.25',
        '--sharpness', '0.5', '--rebuild-ms', '0')
    assert tuning_sequence() == sequence + 1
    run('--reset')
    assert tuning_sequence() == sequence + 2
    assert settings(run('--toggle', 'hold', '--settings'))['hold'][0] == 1
    assert settings(run('-A', 'hold', '-l'))['hold'][0] == 0
    assert settings(run('-A', 'sdr16-multipass', '-l'))['sdr16-multipass'][0] == 0
    assert settings(run('-A', 'sdr16-multipass', '-A', 'sdr16-multipass', '-l'))['sdr16-multipass'][0] == 1
    run('--reset')

    changed = settings(run('-w', '0.5', '-t', '0', '-e', '0', '-l'))
    assert changed['working-scale'] == (0.5, 1)
    assert changed['transfer'] == (0, 2)
    assert changed['enabled'] == (0, 1)
    assert settings(run('-T', '-l'))['enabled'] == (1, 1)
    assert settings(run('--working-scale', '0.5', '-w', '0.75', '-l'))['working-scale'][0] == 0.75
    assert settings(run('-w', '0.5', '-r', '-l'))['working-scale'][0] == 0.5
    assert settings(run('--passes', '2', '--settings'))['passes'] == (2, 1)
    assert settings(run('-P', str(max_passes), '-l'))['passes'] == (max_passes, 1)
    status = run('-S')
    assert f'passes={max_passes}\n' in status
    assert f'pass_ceiling={max_passes}\n' in status
    assert settings(run('--passes=3', '-P2', '-l'))['passes'] == (2, 1)
    run('-q')
    assert 'quit=1\n' in run('-S')
    reset = settings(run('-r', '-l'))
    assert all(current == default for current, default in reset.values())
    assert 'quit=1\n' in run('-S'), '--reset erased transport state'
    assert 'quit=0\n' in run('-R', '-S')
    run('-c', '0')
    run('--capture=64')

    # Explicit path wins over the environment; reads never initialise bad headers.
    other = Path(directory) / 'another channel.bin'
    with other.open('wb') as stream:
        stream.truncate(channel.stat().st_size)
    stat = other.stat()
    run('--shm', str(other), '--settings', expected=1)
    run('-s', str(other), '-S', expected=1)
    assert other.stat().st_mtime_ns == stat.st_mtime_ns
    with other.open('rb') as stream:
        assert not any(stream.read(8192))
    run('-s', str(other), '-r')
    assert 'initialised=1\n' in run('-s', str(other))

print('control CLI: defaults, getopt syntax, read-only access, validation and reset checks passed')
