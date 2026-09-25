#!/usr/bin/env python3
"""Check LDS completion before gfx12 workgroup-barrier arrival in shipped kernels.

An execution barrier alone does not order shared-memory accesses. Multi-wave
FFN kernels need an explicit fence before signalling a barrier, so LDS stores
and reads have completed. This regression inspects production code objects
without requiring a GPU or model weights.

The check is deliberately conservative: within each disassembled kernel, any
LDS load/store requires a zero DS wait before the next barrier signal.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


# Cover FFN and attention kernels across the tiled, fast and packed modules.
KERNELS = {
    "multihead-tiled": "mh_ffn_expand_tiled",
    "prefix_fast": "dlss5_prefix_fast_fused",
    "multihead-fast": "mh_ffn_expand_fast",
    "multihead-fast-packed": "mh_ffn_expand_fast",
    "multihead-fast-padded-wave": "mh_ffn_fused_c64_project_g128_qkv",
    "multihead-fast-padded-wave-packed": "mh_ffn_fused_c64_project_g128_qkv",
    "deep_fast": "split_ffn_fused_fp8_t8",
    "deep_fast-packed": "split_ffn_fused_fp8_t8",
    "c32_fused_ffn_attention-packed": "c32_fast_ffn_attention_fused_half_prefix_finish_main8",
    "c32_fused_attention": "c32_fast_attention_fused",
    "multihead_fused_attention": "c256_attention_project_diag",
    "c32_fast": "c32_ffn_expand_fast",
    "c32_tiled": "c32_ffn_expand_tiled",
}


def check_kernel(body, label):
    pending = None
    barriers = 0
    errors = []
    for line in body.splitlines():
        instruction = line.split("//", 1)[0].strip()
        if re.match(r"ds_(?:load|store)\w*\b", instruction):
            pending = line.strip()
        wait = re.match(r"s_wait_dscnt\s+(0x[0-9a-f]+|[0-9]+)\b", instruction)
        combined = re.match(r"s_wait_(?:loadcnt|storecnt)_dscnt\s+(0x[0-9a-f]+|[0-9]+)\b",
                            instruction)
        if wait and int(wait.group(1), 0) == 0:
            pending = None
        # GFX12 DS_CNT occupies bits 0..5; LOAD/STORE_CNT starts at bit 8.
        # LLVM AMDGPUBaseInfo.cpp: getDscntBitWidth/Shift, decodeDscnt.
        if combined and (int(combined.group(1), 0) & 63) == 0:
            pending = None
        if re.match(r"s_barrier_signal(?:_isfirst)?\b", instruction):
            barriers += 1
            if pending:
                errors.append(f"{label}: outstanding LDS access before {line.strip()}\n"
                              f"  last LDS access: {pending}")
    return barriers, errors


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     allow_abbrev=False)
    parser.add_argument("-d", "--objdump", default="llvm-objdump",
                        help="AMDGPU-capable disassembler (default: llvm-objdump from PATH)")
    parser.add_argument("-m", "--modules", type=Path, default=base / "assets/HIP/gfx1201",
                        help="code-object directory (default: %(default)s)")
    parser.add_argument("-n", "--only", choices=sorted(KERNELS),
                        help="inspect one module (default: unset; inspect all listed production modules)")
    parser.add_argument("-h", "--help", action="help",
                        help="show help and exit (default: off)")
    args = parser.parse_args()
    objdump = shutil.which(args.objdump)
    if not objdump:
        parser.error(f"AMDGPU-capable disassembler not found: {args.objdump}")
    failures = 0
    for module, symbol in KERNELS.items():
        if args.only and module != args.only:
            continue
        result = subprocess.run([objdump, "--disassemble", str(args.modules / f"{module}.hsaco")],
                                text=True, capture_output=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr, end="")
            failures += 1
            continue
        # Do not split a kernel at local labels emitted by some objdump builds.
        symbols = [m for m in re.finditer(r"^[0-9a-f]+ <([^>]+)>:\s*$", result.stdout, re.M)
                   if not re.fullmatch(r"L[0-9]+", m.group(1))]
        bodies = {m.group(1): result.stdout[m.end():
                  symbols[i + 1].start() if i + 1 < len(symbols) else len(result.stdout)]
                  for i, m in enumerate(symbols)}
        if symbol not in bodies:
            print(f"FAIL: {module}: required production kernel missing: {symbol}", file=sys.stderr)
            failures += 1
            continue
        barriers, errors = check_kernel(bodies[symbol], f"{module}:{symbol}")
        if not barriers and module != "prefix_fast":
            errors.append(f"{module}:{symbol}: no workgroup-barrier signals found")
        if errors:
            failures += 1
            for error in errors[:4]:
                print(f"FAIL: {error}", file=sys.stderr)
            if len(errors) > 4:
                print(f"  plus {len(errors) - 4} further unsafe barriers", file=sys.stderr)
        else:
            print(f"PASS: {module}:{symbol}: {barriers} barrier arrivals; no outstanding LDS accesses")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
