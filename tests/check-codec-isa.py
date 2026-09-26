#!/usr/bin/env python3
"""Check the compiled GPU codec's binary16 conversions without a GPU.

A helper named half_round can collide with AMD Clang's OpenCL builtin
recognition: at -O3, llvm.round.f32 can replace explicit inline assembly and
quantize the encoder input to zero and one. Source-level CPU tests cannot
detect that substitution; inspect all five kernel symbols, including the
inter-pass feedback and RGBA16F boundary conversions. Exit status 77 means
the disassembler or the module is missing.

Usage: python3 tests/check-codec-isa.py --objdump /path/to/llvm-objdump \
           assets/HIP/gfx1201/linux_native.hsaco
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="show help and exit (default: do not show help)")
    parser.add_argument("module", type=Path, help="AMDGPU module to inspect (required; no default)")
    parser.add_argument("-d", "--objdump", default="llvm-objdump", help="AMDGPU-capable disassembler")
    args = parser.parse_args()
    objdump = shutil.which(args.objdump)
    if not objdump or not args.module.is_file():
        print(f"SKIP: need an AMDGPU-enabled disassembler ({args.objdump}) and {args.module}",
              file=sys.stderr)
        return 77
    result = subprocess.run([objdump, "--disassemble", str(args.module)],
                            text=True, capture_output=True)
    if result.returncode:
        print(result.stderr, file=sys.stderr, end="")
        return 1
    symbols = list(re.finditer(r"^[0-9a-f]+ <([^>]+)>:\s*$",
                               result.stdout, re.MULTILINE))
    bodies = {
        match.group(1): result.stdout[match.end():
            symbols[i + 1].start() if i + 1 < len(symbols) else len(result.stdout)]
        for i, match in enumerate(symbols)
    }
    # RGBA8 kernels and feedback round float->half->float. The FP16 encoder
    # additionally unpacks four RGB input texels; the decoder packs final RGB.
    expected_conversions = {
        "dlsslop_encode_rgba8": (3, 3),
        "dlsslop_encode_rgba16f": (3, 15),
        "dlsslop_feedback_rgb": (6, 6),
        "dlsslop_decode_rgba8": (12, 12),
        "dlsslop_decode_rgba16f": (15, 12),
    }
    for symbol, expected in expected_conversions.items():
        body = bodies.get(symbol)
        if body is None:
            print(f"FAIL: kernel symbol missing: {symbol}", file=sys.stderr)
            return 1
        down = len(re.findall(r"\bv_cvt_f16_f32(?:_e(?:32|64))?\b", body))
        up = len(re.findall(r"\bv_cvt_f32_f16(?:_e(?:32|64))?\b", body))
        if (down, up) != expected:
            print(f"FAIL: {symbol}: expected RGB binary16 conversions "
                  f"float-to-half={expected[0]}, half-to-float={expected[1]}; "
                  f"found float-to-half={down}, half-to-float={up}", file=sys.stderr)
            return 1
        print(f"PASS: {symbol}: {down} float-to-half and {up} half-to-float conversions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
