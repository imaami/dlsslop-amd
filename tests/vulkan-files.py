#!/usr/bin/env python3
"""Check that the files install.py installs for the Vulkan network are the files that the network's runtime
opens. Under strace, the network recorder test builds the network on its fake device, with every pipeline of
the runtime's own, from a directory that holds exactly those files, at extents whose plans together run every
kernel. The builds must succeed and open each file, and nothing else in the directory. Without two kernels'
files, a build must fail and name the first of them in the kernel table, whichever thread makes its pipeline.
Exit 77 without strace, after that check."""
import argparse
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# 24x24 runs ffwd3, upsview and gemmvqkvnorms, 2560x1440 ffwd3w, gemmprojw and gemmvqkvnorm, and both every
# other kernel.
EXTENTS = ("24x24", "2560x1440")

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-h', '--help', action='help', help='show help and exit (default: off)')
parser.add_argument('--recorder', type=Path, required=True,
                    help='network-recorder-test executable (required; no default)')
parser.add_argument('--spirv', type=Path, required=True,
                    help="the build's network SPIR-V directory, vulkan-nr/network (required; no default)")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('installer', ROOT / 'install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
names = installer.vulkan_shader_names(ROOT)
assert len(set(names)) == len(names), sorted(names)

with tempfile.TemporaryDirectory(prefix='dlsslop-vulkan-files-') as temporary:
    shaders = Path(temporary) / 'network'
    for name in names:
        target = shaders / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.spirv / name, target)
    # Two kernels' files, in the kernel table's order.
    missing = ('g_fswinpds64.spv', 'g_gemmvqkvs.spv')
    for name in missing:
        (shaders / name).rename(Path(temporary) / name)
    result = subprocess.run([args.recorder.resolve(), '--build', EXTENTS[0], shaders], text=True,
                            capture_output=True, timeout=300)
    assert result.returncode == 1 and f'cannot read {shaders / missing[0]}:' in result.stderr, (
        result.returncode, result.stdout, result.stderr)
    for name in missing:
        (Path(temporary) / name).rename(shaders / name)
    strace = shutil.which('strace')
    if not strace:
        print('vulkan-files: skipped: no strace')
        sys.exit(77)
    opened = set()
    for extent in EXTENTS:
        # A file of its own for each thread, so that the pipelines' threads write whole lines; long
        # strings, so that no path is cut short.
        log = Path(temporary) / f'strace-{extent}'
        result = subprocess.run([strace, '-f', '-ff', '-qq', '-s', '4096', '-e', 'trace=open,openat', '-o', log,
                                 str(args.recorder.resolve()), '--build', extent, shaders],
                                text=True, capture_output=True, timeout=300)
        assert result.returncode == 0, (extent, result.returncode, result.stdout, result.stderr)
        for path in Path(temporary).glob(f'strace-{extent}.*'):
            opened.update(re.findall(rf'"{re.escape(str(shaders))}/([^"]+)"', path.read_text()))
    assert opened == set(names), ('installed and not opened:', sorted(set(names) - opened),
                                  'opened and not installed:', sorted(opened - set(names)))
print(f'vulkan-files: the {len(names)} files that install.py installs for the Vulkan network are the files '
      'that its builds open')
