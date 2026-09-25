#!/usr/bin/env python3
"""Rebuild every shader used by the native Linux layer from source."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('-d', '--dxc', default='dxc', help='DXC executable (default: dxc on PATH)')
    p.add_argument('-g', '--glslang', default='glslangValidator', help='GLSL compiler (default: glslangValidator on PATH)')
    p.add_argument('-v', '--validator', default='spirv-val', help='SPIR-V validator (default: spirv-val on PATH)')
    p.add_argument('-h', '--help', action='help', help='show help (default: off)')
    a = p.parse_args()
    subprocess.run([sys.executable, str(ROOT/'scripts/build-shaders.py'), '--dxc', a.dxc], check=True)
    source = ROOT/'upstream-layer/layer_linux/src'
    outputs = [source/'dlssnr/DlssNr_Shader_Vk.spv']
    for name in ('bcus', 'bcds_bicubic', 'bcds_catmull', 'bcds_lanczos2', 'bcds_lanczos3', 'bcds_kaiser2', 'bcds_kaiser3', 'bcds_magc'):
        stem = source/'scaling'/name
        output = stem.with_name(name+'_Shader_Vk.spv')
        subprocess.run([a.dxc, '-spirv', '-D', 'VK_MODE', '-T', 'cs_6_0', '-E', 'CSMain', '-Fo', str(output), str(stem.with_suffix('.hlsl'))], check=True)
        data = output.read_bytes()
        header = output.with_suffix('.h')
        lines = ['    '+', '.join(f'0x{b:02x}' for b in data[i:i+12])+',' for i in range(0,len(data),12)]
        header.write_text('#pragma once\ninline static const unsigned char '+name+'_spv[] = {\n'+'\n'.join(lines)+'\n};\n')
        outputs.append(output)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)/'meter.spv'
        subprocess.run([a.glslang, '-V', str(source/'shaders/meter_reduce.comp'), '-o', str(output)], check=True)
        subprocess.run([a.validator, '--target-env', 'vulkan1.0', str(output)], check=True)
        data = output.read_bytes()
        words = struct.unpack('<'+'I'*(len(data)//4),data)
        (source/'shaders/meter_reduce_spv.h').write_text('#pragma once\n#include <cstdint>\n#include <cstddef>\nstatic const uint32_t kMeterReduceSpv[] = {\n'+','.join(map(str,words))+'\n};\nstatic const size_t kMeterReduceSpvLen = sizeof(kMeterReduceSpv) / sizeof(kMeterReduceSpv[0]);\n')
    for output in outputs:
        subprocess.run([a.validator, '--target-env', 'vulkan1.0', str(output)], check=True)
    print('Rebuilt and validated all 10 native-layer shaders')

if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'build-all-shaders: {exc}', file=sys.stderr)
        sys.exit(1)
