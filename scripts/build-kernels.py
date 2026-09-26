#!/usr/bin/env python3
"""Build the upstream production HIP modules on Linux, without a GPU or HIP SDK headers.

Only an AMDGPU-enabled Clang with gfx1201 builtins and matching ld.lld are required;
auto-detection needs the minimum Clang version stated under --compiler. ROCm's
amdclang++ and the isolated TheRock SDK are supported as well. No AMD display
driver, kernel module, Windows DLL, or proprietary model is executed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


# Upstream hip/build-modules.ps1's selections in its order, less the twelve the
# production network never loads, then the Linux kernels.
MODULES = [
    ("c32_prefix_reference", [], ["c32_reference.hip", "prefix_reference.hip"]),
    ("multihead-reference", [], ["multihead_reference.hip"]),
    ("deep_reference", [], ["deep_reference.hip"]),
    ("boundary_reference", [], ["boundary_reference.hip"]),
    ("c32_fast", [], ["c32_fast.hip"]),
    ("c32_fast_attention", [], ["c32_fast_attention.hip"]),
    ("boundary-fast", [], ["c32_fast_attention.hip", "boundary_fast.hip"]),
    ("c32_fused_ffn_attention-packed", ["HIP_C32_DIAG_WEIGHTS 1"], ["c32_fused_ffn_attention.hip"]),
    ("prefix_fast", [], ["prefix_fast.hip"]),
    ("multihead_fused_attention", ["HIP_MH_RTZ_ISA 1"], ["multihead_fused_attention.hip"]),
    ("deep_fast-packed", ["HIP_BRANCHLESS_F 1"], ["deep_fast.hip"]),
    ("multihead-fast-padded-wave-packed", ["HIP_FFN_HOIST_RES 2"], ["multihead_fast_padded.hip"]),
    ("linux_native", [], ["codec_gpu.hip", "tuning_gpu.hip", "color_gpu.hip", "temporal_gpu.hip"]),
]
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"  # The linux_ modules' sources and headers.
ARCH = "gfx1201"  # The worker accepts only gfx1201 devices.


# Upstream separates LDS phases in these sources with bare execution barriers,
# which gfx12 signals with DS operations still outstanding. Add the LDS-only
# release/acquire of __syncthreads() so global loads need not drain; the
# lds-barriers test checks the result.
# https://llvm.org/docs/AMDGPUUsage.html#execution-barriers
LDS_BARRIER = ('#define __builtin_amdgcn_s_barrier() (__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"), '
               '__builtin_amdgcn_s_barrier(), __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"))\n')
PRELUDES = {
    # Upstream's byte-output attention template calls byte_F before its later
    # definition; standard C++ lookup needs a declaration.
    "deep_fast.hip": LDS_BARRIER + "__attribute__((device)) __attribute__((always_inline)) unsigned char byte_F(float);\n",
    "multihead_fast_padded.hip": LDS_BARRIER,
    "prefix_fast.hip": LDS_BARRIER,
}


# Auto-detection order. Before Clang 22, every gfx12 workgroup barrier also
# drains global loads and stores.
MIN_CLANG = 22
PATH_COMPILERS = ["amdclang++", "clang++-22", "clang++-23", "clang++"]
ROCM_COMPILERS = ["/opt/rocm/llvm/bin/clang++", "/opt/rocm/bin/amdclang++"]
SDK_COMPILERS = ["lib/llvm/bin/clang++", "llvm/bin/clang++", "bin/amdclang++"]
AUTO_COMPILERS = (f"the first Clang {MIN_CLANG} or newer among {', '.join(PATH_COMPILERS)} on PATH, "
                  f"then {', '.join(ROCM_COMPILERS)} and the active rocm-sdk's {', '.join(SDK_COMPILERS)}")


def compiler_candidates():
    yield from (Path(path) for name in PATH_COMPILERS if (path := shutil.which(name)))
    yield from filter(Path.is_file, map(Path, ROCM_COMPILERS))
    if sdk := shutil.which("rocm-sdk"):
        root = Path(subprocess.check_output([sdk, "path", "--root"], text=True).strip())
        yield from filter(Path.is_file, (root / rel for rel in SDK_COMPILERS))


def version_line(compiler):
    return subprocess.check_output([str(compiler), "--version"], text=True).partition("\n")[0]


def find_compiler(explicit):
    """Return the explicit compiler, whatever its version, or the first new enough candidate, with its version."""
    if explicit:
        found = shutil.which(explicit)
        if not found:
            raise RuntimeError(f"compiler not found: {explicit}")
        return Path(found), version_line(found)
    for compiler in compiler_candidates():
        try:
            version = version_line(compiler)
        except (OSError, subprocess.CalledProcessError):
            continue  # A candidate that cannot report its version counts as too old.
        major = re.search(r"clang version (\d+)", version)
        if major and int(major[1]) >= MIN_CLANG:
            return compiler, version
    raise RuntimeError(f"found no Clang {MIN_CLANG} or newer: install clang-{MIN_CLANG} + lld-{MIN_CLANG} or activate "
                       f"a ROCm/TheRock SDK with Clang {MIN_CLANG} or newer, or name any other compiler with "
                       "--compiler PATH or HIP_CLANG")


def record_includes(text, directory, hashes):
    """Hash the headers text includes, found from directory as the compiler finds them, and theirs."""
    for name in re.findall(rb'^#include "([^"]+)"', text, re.M):
        header = directory / name.decode()
        key = os.path.relpath(header, BACKEND)
        if key not in hashes:
            content = header.read_bytes()
            hashes[key] = hashlib.sha256(content).hexdigest()
            record_includes(content, header.parent, hashes)


def build(args):
    compiler, version = find_compiler(args.compiler)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    modules = [row for row in MODULES if not args.only or row[0] == args.only]
    manifest = []
    for name, extra, sources in modules:
        defines = [] if name.startswith("linux_") else ["HIP_ISA_HALF 1"]
        if name.endswith("-packed"):
            defines.append("HIP_PREPACKED_WEIGHTS 1")
        defines += extra
        # The Windows COMGR frontend supplies size_t implicitly. Linux's
        # headerless HIP frontend needs its builtin type spelling explicitly.
        source = "typedef __SIZE_TYPE__ size_t;\n" + "".join(f"#define {d}\n" for d in defines)
        source += "".join(PRELUDES.get(filename, "") for filename in sources)
        source_hashes = {}
        for filename in sources:
            path = (BACKEND if name.startswith("linux_") else args.source) / filename
            content = path.read_bytes()
            source_hashes[filename] = hashlib.sha256(content).hexdigest()
            source += content.decode("utf-8") + "\n"
            record_includes(content, BACKEND, source_hashes)  # The generated copy finds them through -I.
        generated = root / f"{name}.generated.hip"
        generated.write_text(source, encoding="utf-8")
        output = root / f"{name}.hsaco"
        # Suppress the host/device bundle: HIP's module API accepts the raw
        # AMDGPU shared ELF emitted by Clang's device link step. Without a cuid
        # the module does not hash the output path. Upstream's f16 inline asm
        # takes 32-bit VGPRs, which LLVM 23's gfx12 true16 mode rejects.
        command = [str(compiler), "-x", "hip", "--cuda-device-only", "--no-gpu-bundle-output",
                   f"--offload-arch={ARCH}", f"-mcode-object-version={args.code_object_version}",
                   "-nogpuinc", "-nogpulib", "-fuse-cuid=none", "-O3", "-std=c++17",
                   "-Xclang", "-target-feature", "-Xclang", "-real-true16",
                   "-I", str(BACKEND),
                   "-c", str(generated), "-o", str(output)]
        if args.linker:
            command.append(f"--ld-path={args.linker}")
        print(f"[{len(manifest)+1}/{len(modules)}] {ARCH} {name}", flush=True)
        subprocess.run(command, check=True)
        content = output.read_bytes()
        if content[:4] != b"\x7fELF" or int.from_bytes(content[16:18], "little") != 3 or int.from_bytes(content[18:20], "little") != 224:
            raise RuntimeError(f"output is not an AMDGPU ELF shared code object: {output}")
        manifest.append({"module": name, "target": ARCH, "defines": defines, "sources": source_hashes,
                         "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "compiler": version,
                         "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "code_object_version": args.code_object_version})
        if not args.keep_generated:
            generated.unlink()
    # Single-module builds keep the existing manifest's other current modules.
    if args.only and (root / "modules.json").exists():
        old = json.loads((root / "modules.json").read_text())
        keep = {row[0] for row in MODULES} - {args.only}
        manifest += [entry for entry in old if entry["module"] in keep]
    manifest.sort(key=lambda entry: entry["module"])
    (root / "modules.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "SHA256SUMS").write_text("".join(f'{entry["sha256"]}  {entry["module"]}.hsaco\n' for entry in manifest))
    print(f"Built {len(modules)} AMDGPU code objects in {root}")


def main():
    compiler_default = os.environ.get("HIP_CLANG") or None
    compiler_help = "compiler executable, used whatever its Clang version (default: " + (
        "%(default)s from HIP_CLANG" if compiler_default else "HIP_CLANG if set, else auto: " + AUTO_COMPILERS) + ")"
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--source", type=Path, default=ROOT / "kernels",
                        help="directory containing upstream production *.hip (default: %(default)s)")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "assets/HIP" / ARCH,
                        help="output directory (default: %(default)s)")
    parser.add_argument("-V", "--code-object-version", choices=[5, 6], type=int, default=5,
                        help="AMDHSA code object version; 5 supports older ROCm loaders (default: %(default)s)")
    parser.add_argument("-c", "--compiler", default=compiler_default, help=compiler_help)
    parser.add_argument("-l", "--linker", help="linker executable (default: unset; use the compiler's default linker)")
    parser.add_argument("-n", "--only", choices=[row[0] for row in MODULES],
                        help=f"build only this module (default: unset; build all {len(MODULES)} modules)")
    parser.add_argument("-k", "--keep-generated", action="store_true",
                        help="keep generated combined HIP sources (default: off; remove them after compilation)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    try:
        build(parser.parse_args())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"build-kernels: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
