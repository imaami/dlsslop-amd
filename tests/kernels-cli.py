#!/usr/bin/env python3
"""Exercise build-kernels.py compiler selection and flags with stand-in compilers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.parse_args()
script = Path(__file__).resolve().parent.parent / 'scripts/build-kernels.py'

# Answers --version with its output and status; a compile records its
# arguments and writes the smallest AMDGPU shared-object header the script
# accepts.
FAKE = '''#!{python}
import sys
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    sys.stdout.write({output!r})
    sys.exit({status})
Path(sys.argv[0] + ".args").write_text("\\n".join(sys.argv[1:]))
header = b"\\x7fELF" + bytes(12) + (3).to_bytes(2, "little") + (224).to_bytes(2, "little")
Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(header)
'''

# Answers `path --root` with the SDK root and leaves a marker when run.
SDK = '''#!{python}
import sys
from pathlib import Path
Path(sys.argv[0] + ".ran").touch()
if sys.argv[1:] == ["path", "--root"]:
    print({root!r})
'''

# Stand-in versions: a line, empty output, a failing --version, or no runnable interpreter.
EMPTY, FAILS, UNRUNNABLE = '', None, False

ROCM_FALLBACKS = (Path('/opt/rocm/llvm/bin/clang++'), Path('/opt/rocm/bin/amdclang++'))

with tempfile.TemporaryDirectory(prefix='build-kernels-cli-') as directory:
    root = Path(directory)

    def compilers(label, versions):
        folder = root / label
        for name, version in versions.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if version is UNRUNNABLE:
                path.write_text('#!/nonexistent/interpreter\n')
            else:
                path.write_text(FAKE.format(python=sys.executable, output=version + '\n' if version else '',
                                            status=int(version is FAILS)))
            path.chmod(0o755)
        return folder

    def rocm_sdk(folder, sdk_root):
        path = folder / 'rocm-sdk'
        path.write_text(SDK.format(python=sys.executable, root=str(sdk_root)))
        path.chmod(0o755)
        return path.with_name('rocm-sdk.ran')

    def build(folder, *options, expected=0, **env):
        output = root / (folder.name + '-out')
        environment = {key: value for key, value in os.environ.items() if key != 'HIP_CLANG'}
        environment.update(PATH=str(folder), **env)
        result = subprocess.run([sys.executable, str(script), '-o', str(output), '-n', 'linux_codec', *options],
                                env=environment, text=True, capture_output=True, timeout=60)
        assert result.returncode == expected, (folder.name, options, result.stdout, result.stderr)
        if expected:
            return result.stderr
        return json.loads((output / 'modules.json').read_text())[0]['compiler']

    old = 'AMD clang version 19.0.0git (roc-6.4.0)'
    sdk_root = compilers('sdk-root', {'lib/llvm/bin/clang++': 'TheRock clang version 22.0.0',
                                      'bin/amdclang++': 'AMD clang version 23.0.0git'})
    mixed = compilers('mixed', {'amdclang++': old, 'clang++-22': 'Debian clang version 22.1.8',
                                'clang++-23': 'Debian clang version 23.1.2', 'clang++': 'clang version 20.1.8'})
    mixed_sdk_ran = rocm_sdk(mixed, sdk_root)
    assert build(mixed) == 'Debian clang version 22.1.8', 'auto-detection did not skip old Clang for clang++-22'
    arguments = (mixed / 'clang++-22.args').read_text().splitlines()
    assert '-fuse-cuid=none' in arguments, 'modules would hash the output path'
    assert '-real-true16' in arguments and arguments[arguments.index('-real-true16') - 3:][:3] == \
        ['-Xclang', '-target-feature', '-Xclang'], 'true16 is not disabled'
    assert not any(arg.startswith('--ld-path') for arg in arguments), 'no linker was requested'

    newer = compilers('newer', {'amdclang++': 'AMD clang version 22.0.0git',
                                'clang++-22': 'Debian clang version 22.1.8'})
    assert build(newer) == 'AMD clang version 22.0.0git', 'a new enough amdclang++ is no longer preferred'
    only_23 = compilers('only-23', {'clang++-23': 'Debian clang version 23.1.2'})
    assert build(only_23) == 'Debian clang version 23.1.2', 'clang++-23 alone was not selected'

    # A single-module build keeps the existing manifest's other current modules
    # and drops the ones the build no longer lists.
    stale = compilers('stale', {'clang++-22': 'Debian clang version 22.1.8'})
    (root / 'stale-out').mkdir()
    (root / 'stale-out/modules.json').write_text(json.dumps(
        [{'module': 'c32_wmma', 'compiler': 'old', 'sha256': '1' * 64},
         {'module': 'linux_color', 'compiler': 'old', 'sha256': '0' * 64}]))
    build(stale)
    rows = json.loads((root / 'stale-out/modules.json').read_text())
    assert [row['module'] for row in rows] == ['linux_codec', 'linux_color'], rows
    sums = (root / 'stale-out/SHA256SUMS').read_text()
    assert 'c32_wmma' not in sums and f'{"0" * 64}  linux_color.hsaco\n' in sums, sums

    # An explicit compiler, from the option or HIP_CLANG, is used whatever its version.
    assert build(mixed, '--compiler', 'clang++') == 'clang version 20.1.8'
    assert build(mixed, HIP_CLANG='amdclang++') == old
    assert 'compiler not found: missing' in build(mixed, '--compiler', 'missing', expected=1)
    assert not mixed_sdk_ran.exists(), 'rocm-sdk was queried although an earlier compiler qualified'

    # Candidates that cannot report a version are skipped; an explicit one is fatal.
    broken = compilers('broken', {'amdclang++': old, 'clang++-22': FAILS, 'clang++-23': EMPTY,
                                  'clang++': UNRUNNABLE})
    broken_sdk_ran = rocm_sdk(broken, sdk_root)
    assert 'returned non-zero exit status' in build(broken, '--compiler', 'clang++-22', expected=1)
    assert 'build-kernels:' in build(broken, '--compiler', 'clang++', expected=1)
    assert build(broken, '--compiler', 'clang++-23') == ''

    if any(path.is_file() for path in ROCM_FALLBACKS):
        print('note: a ROCm compiler is installed; skipping the rocm-sdk and no-compiler checks')
    else:
        assert build(broken) == 'TheRock clang version 22.0.0', 'the rocm-sdk compiler was not selected'
        assert broken_sdk_ran.exists()
        # An unparsable version counts as too old.
        too_old = compilers('too-old', {'amdclang++': old, 'clang++': 'clang version 21.1.8',
                                        'clang++-23': 'Ubuntu clang version'})
        message = build(too_old, expected=1)
        assert all(text in message for text in ('found no Clang 22 or newer', '--compiler', 'HIP_CLANG')), message

    def helptext(**env):
        return ' '.join(subprocess.run([sys.executable, str(script), '--help'], env={'PATH': str(mixed), **env},
                                       text=True, capture_output=True, check=True).stdout.split())

    auto = helptext()
    assert 'used whatever its Clang version (default: HIP_CLANG if set, else auto: the first Clang 22 or newer' \
        in auto and 'rocm-sdk' in auto, auto
    named = helptext(HIP_CLANG='amdclang++')
    assert 'used whatever its Clang version (default: amdclang++ from HIP_CLANG)' in named, named

print('build-kernels CLI tests passed: Clang 22 selection, rocm-sdk fallback, broken candidates, '
      'explicit compilers, help, reproducible and true16 flags')
