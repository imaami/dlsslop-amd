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
    ("linux_codec", [], ["codec_gpu.hip"]),
    ("linux_tuning", [], ["tuning_gpu.hip"]),
    ("linux_color", [], ["color_gpu.hip"]),
    ("linux_temporal", [], ["temporal_gpu.hip"]),
]


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


def build(args):
    compiler, version = find_compiler(args.compiler)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    modules = [row for row in MODULES if not args.only or row[0] == args.only]
    if not modules:
        raise RuntimeError(f"unknown module: {args.only}")
    manifest = []
    for name, extra, sources in modules:
        defines = [] if name.startswith("linux_") else ["HIP_ISA_HALF 1"]
        if name.endswith("-packed"):
            defines.append("HIP_PREPACKED_WEIGHTS 1")
        defines += extra
        # The Windows COMGR frontend supplies size_t implicitly. Linux's
        # headerless HIP frontend needs its builtin type spelling explicitly.
        source = "typedef __SIZE_TYPE__ size_t;\n" + "".join(f"#define {d}\n" for d in defines)
        if "deep_fast.hip" in sources:
            # Upstream's byte-output attention template calls this helper
            # before its later definition; standard C++ lookup needs a decl.
            source += "__attribute__((device)) __attribute__((always_inline)) unsigned char byte_F(float);\n"
        source_hashes = {}
        for filename in sources:
            path = (args.codec_source if name == "linux_codec" else
                    args.codec_source.parent / filename if name.startswith("linux_") else
                    args.source / filename)
            content = path.read_bytes()
            source_hashes[filename] = hashlib.sha256(content).hexdigest()
            source += content.decode("utf-8") + "\n"
            if b'#include "lds_barrier.h"' in content:
                # These shared-memory kernels depend on an explicit LDS fence
                # helper. Record its source as part of the module provenance.
                header = args.source / "lds_barrier.h"
                source_hashes[header.name] = hashlib.sha256(header.read_bytes()).hexdigest()
            # Colour's header includes tuning's: scan each recorded header too.
            for header_name in ("color_preserve_math.h", "temporal_math.h", "tuning_math.h"):
                if ('#include "' + header_name + '"').encode() in content:
                    text = (args.codec_source.parent / header_name).read_bytes()
                    content += text
                    source_hashes[header_name] = hashlib.sha256(text).hexdigest()
        generated = root / f"{name}.generated.hip"
        generated.write_text(source, encoding="utf-8")
        output = root / f"{name}.hsaco"
        # Suppress the host/device bundle: HIP's module API accepts the raw
        # AMDGPU shared ELF emitted by Clang's device link step. Without a cuid
        # the module does not hash the output path. Upstream's f16 inline asm
        # takes 32-bit VGPRs, which LLVM 23's gfx12 true16 mode rejects.
        command = [str(compiler), "-x", "hip", "--cuda-device-only", "--no-gpu-bundle-output",
                   f"--offload-arch={args.arch}", f"-mcode-object-version={args.code_object_version}",
                   "-nogpuinc", "-nogpulib", "-fuse-cuid=none", "-O3", "-std=c++17",
                   "-Xclang", "-target-feature", "-Xclang", "-real-true16",
                   "-I", str(args.source.resolve()),
                   "-I", str(args.codec_source.parent.resolve()),
                   "-c", str(generated), "-o", str(output)]
        if args.linker:
            command.append(f"--ld-path={args.linker}")
        print(f"[{len(manifest)+1}/{len(modules)}] {args.arch} {name}", flush=True)
        subprocess.run(command, check=True)
        content = output.read_bytes()
        if content[:4] != b"\x7fELF" or int.from_bytes(content[16:18], "little") != 3 or int.from_bytes(content[18:20], "little") != 224:
            raise RuntimeError(f"output is not an AMDGPU ELF shared code object: {output}")
        manifest.append({"module": name, "target": args.arch, "defines": defines, "sources": source_hashes,
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
    base = Path(__file__).resolve().parents[1]
    compiler_default = os.environ.get("HIP_CLANG") or None
    compiler_help = "compiler executable, used whatever its Clang version (default: " + (
        "%(default)s from HIP_CLANG" if compiler_default else "HIP_CLANG if set, else auto: " + AUTO_COMPILERS) + ")"
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--source", type=Path, default=base / "kernels",
                        help="directory containing upstream production *.hip (default: %(default)s)")
    parser.add_argument("-S", "--codec-source", type=Path, default=base / "backend/codec_gpu.hip",
                        help="Linux frame conversion kernel source (default: %(default)s)")
    parser.add_argument("-o", "--output", type=Path,
                        help=f"output directory (default: {base / 'assets/HIP'}/ARCH, using --arch)")
    parser.add_argument("-a", "--arch", choices=["gfx1200", "gfx1201"], default="gfx1201",
                        help="target GPU architecture (default: %(default)s)")
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
        args = parser.parse_args()
        if args.output is None:
            args.output = base / "assets/HIP" / args.arch
        build(args)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"build-kernels: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
