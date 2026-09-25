#!/usr/bin/env python3
"""Check LDS completion before gfx12 workgroup-barrier arrival in shipped kernels.

An execution barrier alone does not order shared-memory accesses. Multi-wave
kernels need an explicit fence before signalling a barrier, so LDS stores and
reads have completed. This regression inspects every kernel in the production
code objects without requiring a GPU or model weights. With a compiler, it
first proves that the scan flags an unfenced probe kernel and passes a fenced
one. Exit status 77 means the disassembler or the code objects are missing.

The check is deliberately conservative: within each disassembled kernel, any
LDS load/store requires a zero DS wait before the next barrier signal.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


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


def disassemble(objdump, path):
    result = subprocess.run([objdump, "--disassemble", str(path)], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{path}: {result.stderr.strip()}")
    # Do not split a kernel at local labels emitted by some objdump builds.
    symbols = [m for m in re.finditer(r"^[0-9a-f]+ <([^>]+)>:\s*$", result.stdout, re.M)
               if not re.fullmatch(r"L[0-9]+", m.group(1))]
    return {m.group(1): result.stdout[m.end():
            symbols[i + 1].start() if i + 1 < len(symbols) else len(result.stdout)]
            for i, m in enumerate(symbols)}


def probe(objdump, compiler, base):
    """Compile the probe as build-kernels.py compiles modules; return errors."""
    # Keep these flags in step with build() in scripts/build-kernels.py.
    with tempfile.TemporaryDirectory(prefix="dlsslop-lds-probe-") as temporary:
        output = Path(temporary) / "probe.hsaco"
        subprocess.run([compiler, "-x", "hip", "--cuda-device-only", "--no-gpu-bundle-output",
                        "--offload-arch=gfx1201", "-mcode-object-version=5", "-nogpuinc", "-nogpulib",
                        "-fuse-cuid=none", "-O3", "-std=c++17",
                        "-Xclang", "-target-feature", "-Xclang", "-real-true16",
                        "-I", str(base / "kernels"), "-c",
                        str(base / "tests/lds_barrier_probe.hip"), "-o", str(output)], check=True)
        bodies = disassemble(objdump, output)
    errors = []
    for symbol, unsafe in (("unsafe_lds", True), ("fenced_lds", False)):
        barriers, found = check_kernel(bodies.get(symbol, ""), f"probe:{symbol}")
        if not barriers or bool(found) != unsafe:
            errors.append(f"probe:{symbol}: {barriers} barrier signals, {len(found)} unsafe; "
                          f"expected {'an unsafe' if unsafe else 'a fenced'} barrier")
    if not errors:
        print("PASS: probe: the scan flags unsafe_lds and passes fenced_lds")
    return errors


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     allow_abbrev=False)
    parser.add_argument("-d", "--objdump", default="llvm-objdump",
                        help="AMDGPU-capable disassembler (default: llvm-objdump from PATH)")
    parser.add_argument("-m", "--modules", type=Path, default=base / "assets/HIP/gfx1201",
                        help="code-object directory (default: %(default)s)")
    parser.add_argument("-c", "--compiler",
                        help="AMDGPU-capable clang++ for the probe control (default: unset; empty or unset skips it)")
    parser.add_argument("-h", "--help", action="help",
                        help="show help and exit (default: off)")
    args = parser.parse_args()
    objdump = shutil.which(args.objdump)
    modules = sorted(args.modules.glob("*.hsaco"))
    if not objdump or not modules:
        print(f"SKIP: need an AMDGPU-capable disassembler ({args.objdump}) "
              f"and code objects in {args.modules}", file=sys.stderr)
        return 77
    try:
        errors = probe(objdump, args.compiler, base) if args.compiler else []
        results = [check_kernel(body, f"{module.stem}:{symbol}")
                   for module in modules for symbol, body in disassemble(objdump, module).items()]
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    barriers = sum(count for count, _ in results)
    if not barriers:
        errors.append(f"no workgroup-barrier signals in {len(modules)} code objects")
    for _, found in results:
        errors += found[:4]
        if len(found) > 4:
            errors[-1] += f"\n  plus {len(found) - 4} further unsafe barriers"
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"PASS: {sum(count != 0 for count, _ in results)} of {len(results)} kernels in "
          f"{len(modules)} code objects signal {barriers} barriers; no outstanding LDS accesses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
