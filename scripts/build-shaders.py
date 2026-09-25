#!/usr/bin/env python3
"""Rebuild the Vulkan composition shader and its embedded byte array."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     allow_abbrev=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-h', '--help', action='help',
                        help='show help and exit (default: do not show help)')
    parser.add_argument('-d', '--dxc', default=os.environ.get('DXC') or shutil.which('dxc') or 'dxc',
                        help='DirectX Shader Compiler executable; nonempty DXC or PATH dxc')
    parser.add_argument('-s', '--source', type=Path,
                        default=root / 'upstream-layer/layer_linux/src/dlssnr/dlssnr.hlsl',
                        help='composition shader source')
    parser.add_argument('-o', '--output', type=Path,
                        default=root / 'upstream-layer/layer_linux/src/dlssnr/DlssNr_Shader_Vk.spv',
                        help='SPIR-V output')
    parser.add_argument('-H', '--header', type=Path,
                        default=root / 'upstream-layer/layer_linux/src/dlssnr/DlssNr_Shader_Vk.h',
                        help='embedded SPIR-V header output')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.dxc, '-spirv', '-D', 'VK_MODE', '-T', 'cs_6_0', '-E', 'CSMain',
                    '-Fo', str(args.output), str(args.source)], check=True)
    data = args.output.read_bytes()
    if len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
        raise RuntimeError('compiler did not produce SPIR-V')
    lines = ['    ' + ', '.join(f'0x{x:02x}' for x in data[i:i + 12]) + ','
             for i in range(0, len(data), 12)]
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text('#pragma once\n\ninline static const unsigned char dlssnr_spv[] = {\n'
                           + '\n'.join(lines).rstrip(',') + '\n};\n')
    print(f'Wrote {args.output} and {args.header} ({len(data)} bytes)')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'build-shaders: {error}', file=sys.stderr)
        sys.exit(1)
