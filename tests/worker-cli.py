#!/usr/bin/env python3
"""Exercise direct worker defaults and identity mode without HIP or model weights."""
import argparse
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import tempfile


parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('worker', type=Path, help='dlsslopd executable (required; no default)')
parser.add_argument('hip_stub', type=Path,
                    help='shared library whose dependency is unreachable (required; no default)')
args = parser.parse_args()
worker = args.worker.resolve()
assert worker.read_bytes().startswith(b'\x7fELF'), 'worker tests require the native ELF'


def run(executable, *options, env, cwd, expected=0):
    result = subprocess.run([str(executable), *options], env=env, cwd=cwd,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == expected, (options, result.returncode, result.stdout, result.stderr)
    return result


def default(helptext, option):
    match = re.search(r'--' + re.escape(option) + r' [^\n]*\n\s+Default: ([^\n]+)', helptext)
    assert match, (option, helptext)
    return match[1]


with tempfile.TemporaryDirectory(prefix='dlsslopd-cli-') as directory:
    root = Path(directory)
    prefix = root / "installed '$() ` with spaces"
    binary = prefix / 'bin/dlsslopd'
    binary.parent.mkdir(parents=True)
    shutil.copy2(worker, binary)
    installed_modules = prefix / 'share/dlsslop-amd/HIP/gfx1201'
    dev_modules = prefix / 'assets/HIP/gfx1201'
    home = root / 'home with spaces'
    xdg = root / 'data with spaces'
    home.mkdir()
    cwd = root / 'unrelated working directory'
    cwd.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DLSSLOP_', 'DLSSNR_')) and key != 'XDG_DATA_HOME'}
    env['HOME'] = str(home)
    channel = root / 'channel with spaces.bin'
    env['DLSSNR_SHM'] = str(channel)
    helptext = run(binary, '--help', env=env, cwd=cwd).stdout
    assert helptext.startswith('Usage: dlsslopd [OPTIONS]\n')
    assert default(helptext, 'assets') == str(home / '.local/share/dlsslop-amd/model')
    assert default(helptext, 'modules') == str(installed_modules)
    assert default(helptext, 'shm') == str(channel)
    assert 'DLSSNR_SHM' in helptext and 'DLSSLOP_MODULES' in helptext
    assert default(helptext, 'trace-dir').startswith('disabled')
    assert "not 'request'" in helptext and 'ln it to DIR/request' in helptext
    assert not channel.exists(), '--help created a channel'

    # Relocated native executables locate modules without a generated wrapper,
    # without depending on the working directory or on a symlink's location.
    dev_modules.mkdir(parents=True)
    helptext = run(binary, '-h', env=env, cwd=cwd).stdout
    assert default(helptext, 'modules') == str(dev_modules)
    installed_modules.mkdir(parents=True)
    link = cwd / 'worker-link'
    link.symlink_to(binary)
    helptext = run(link, '--help', env=env, cwd=cwd).stdout
    assert default(helptext, 'modules') == str(installed_modules)

    helptext = run(binary, '--help', env=dict(env, XDG_DATA_HOME=str(xdg)), cwd=cwd).stdout
    assert default(helptext, 'assets') == str(xdg / 'dlsslop-amd/model')
    helptext = run(binary, '--help', env=dict(env, XDG_DATA_HOME=''), cwd=cwd).stdout
    assert default(helptext, 'assets') == str(home / '.local/share/dlsslop-amd/model')
    no_home = dict(env)
    no_home.pop('HOME')
    helptext = run(binary, '--help', env=no_home, cwd=cwd).stdout
    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert default(helptext, 'assets') == str(account_home / '.local/share/dlsslop-amd/model')
    helptext = run(binary, '--help', env=dict(env, DLSSLOP_MODULES='custom modules'), cwd=cwd).stdout
    assert default(helptext, 'modules') == 'custom modules'
    helptext = run(binary, '--help', env=dict(env, DLSSLOP_MODULES=''), cwd=cwd).stdout
    assert default(helptext, 'modules') == str(installed_modules)
    no_channel = dict(env, DLSSNR_SHM='', DLSSNR_UID='12345')
    helptext = run(binary, '--help', env=no_channel, cwd=cwd).stdout
    assert default(helptext, 'shm') == f'/tmp/dlsslop-amd-{os.getuid()}/shm.bin'

    # Explicit empty paths must override populated defaults and fail before HIP.
    for option in ('--assets', '-a', '--modules', '-m'):
        result = run(binary, option, '', env=env, cwd=cwd, expected=1)
        assert '--assets and --modules are required for inference' in result.stderr
        assert not channel.exists(), 'invalid options created a channel'

    # Tracing needs a nonempty directory and the serving mode; parsing rejects
    # everything else before HIP, the channel or the directory is touched.
    for options in (('-R', ''), ('--trace-dir=',)):
        result = run(binary, *options, env=env, cwd=cwd, expected=1)
        assert '--trace-dir requires a nonempty directory' in result.stderr, result.stderr
    offline = ('-i', 'x', '-o', 'y', '-W', '1', '-H', '1')
    for options in (('--self-test',), ('-S', '-r', '2'), ('-D',), ('--diagnose', '-d', '0'),
                    ('--test-identity', *offline), ('-T',), offline):
        for trace in (('-R', 'd'), ('--trace-dir', 'd')):
            result = run(binary, *trace, *options, env=env, cwd=cwd, expected=1)
            assert '--trace-dir requires serving real shared-memory inference' in result.stderr, \
                (options, result.stderr)
    assert not channel.exists() and not (cwd / 'd').exists(), 'rejected tracing touched the filesystem'

    # An existing channel directory must be private to this user: mode 0700
    # and not a symlink. (trace_test checks that a created one is made so.)
    shared = root / 'shared channel directory'
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    (root / 'linked channel directory').symlink_to(root / 'private channel directory')
    (root / 'private channel directory').mkdir(mode=0o700)
    for parent, message in ((shared, 'must be private (mode 0700)'),
                            (root / 'linked channel directory', 'must be owned by the current user and not a symlink')):
        result = run(binary, '--test-identity', '-s', str(parent / 'shm.bin'), env=env, cwd=cwd, expected=1)
        assert f'shared-memory directory {message}' in result.stderr, result.stderr
        assert not (parent / 'shm.bin').exists(), 'a rejected directory received a channel'

    # Invoke the relocated ELF directly and preserve literal path arguments.
    # Identity mode must still work when HIP/model paths are unavailable.
    input_path = cwd / "input '$() ` rgba"
    output_path = cwd / "output '$() ` rgba"
    pixels = bytes((12, 34, 56, 255, 78, 90, 123, 255))
    input_path.write_bytes(pixels)
    result = run(binary, '--test-identity', '-a', 'missing weights', '-m', 'missing modules',
                 '-i', str(input_path), '-o', str(output_path), '-W', '2', '-H', '1',
                 env=dict(env, DLSSLOP_HIP_LIBRARY='missing HIP runtime'), cwd=cwd)
    assert output_path.read_bytes() == pixels
    assert 'IDENTITY TEST MODE' in result.stderr
    assert not channel.exists(), 'offline identity mode created a channel'

    # The HIP loader names why every candidate failed, so a runtime that is
    # installed but cannot load is not blamed on an unrelated absent path. A
    # set DLSSLOP_HIP_LIBRARY is the only candidate; an empty one is unset.
    stubs = root / 'broken HIP runtime'
    stubs.mkdir()
    for name in ('libamdhip64.so.7', 'libamdhip64.so.6', 'libamdhip64.so'):
        shutil.copy2(args.hip_stub, stubs / name)
    missing = 'libhip-loader-stub-dependency.so: cannot open shared object file'
    pinned = stubs / 'libamdhip64.so.7'
    result = run(binary, '--diagnose', env=dict(env, DLSSLOP_HIP_LIBRARY=str(pinned)), cwd=cwd, expected=1)
    assert result.stderr.endswith(f'DLSSLOP_HIP_LIBRARY:\n  {missing}: No such file or directory\n'), result.stderr
    if not any(Path('/opt/rocm/lib').glob('libamdhip64.so*')):
        result = run(binary, '--diagnose', env=dict(env, DLSSLOP_HIP_LIBRARY='', LD_LIBRARY_PATH=str(stubs)),
                     cwd=cwd, expected=1)
        assert result.stderr.count(missing) == 3, result.stderr
        for name in ('libamdhip64.so.7', 'libamdhip64.so.6', 'libamdhip64.so'):
            assert f'\n  /opt/rocm/lib/{name}: cannot open shared object file' in result.stderr, result.stderr

print('worker CLI: native relocation, model/module defaults, environment contracts, trace-dir parsing, '
      'HIP loader errors and identity mode passed')
