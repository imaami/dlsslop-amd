#!/usr/bin/env python3
"""Install a complete dlsslop-amd source build into a prefix, laid out like the binary release."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tempfile


MODULE_NAMES = (
    "multihead-reference", "deep_reference", "c32_fused_ffn_attention-packed", "multihead_fused_attention",
    "deep_fast-packed", "multihead-fast-padded-wave-packed", "linux_native",
)
MODULE_DIRECTORY = "share/dlsslop-amd/HIP/gfx1201"
LAYER_LIBRARY = "lib/dlsslop-amd/libVkLayer_DLSSLOP_amd.so"
# The in-layer network, which the layer loads from beside itself on demand.
NETWORK_LIBRARY = "lib/dlsslop-amd/libdlsslop-network.so"
# Where the Vulkan loader finds it for a ~/.local or /usr/local prefix; the
# launcher adds this directory to the loader's search for any other prefix.
LAYER_MANIFEST = "share/vulkan/implicit_layer.d/VK_LAYER_LOCAL_dlsslop_amd.json"
DOC_DIRECTORY = "share/doc/dlsslop-amd"
# The files of the Vulkan network's SPIR-V that its runtime reads, from the
# build tree, where DLSSNR-AMD's build_network.py writes them among others, and
# the extractor dlsslop-setup --dll runs.
VULKAN_DIRECTORY = "share/dlsslop-amd/vulkan"
MODEL_TOOLS_DIRECTORY = "libexec/dlsslop-amd/model-tools"
MODEL_TOOLS = ("descriptor.json", "extract_model.sh", "inspect_nr.py", "model-files.sha256", "model-files.txt",
               "pack_model.py", "unpack_postblock.py", "unpack_preblock.py", "unpack_splitswin.py",
               "unpack_swin_family.py", "unpack_vit.py")
# Every other installed file's digest; the README checks and uninstalls by it.
INVENTORY = f"{DOC_DIRECTORY}/SHA256SUMS"
NATIVE_SOURCES = {
    "bin/dlsslopd": "dlsslopd",
    "bin/dlsslopctl": "dlssnr-shmctl",
    "bin/dlsslop-gui": "gui/dlsslop-gui",
    LAYER_LIBRARY: "libVkLayer_DLSSLOP_amd.so",
    NETWORK_LIBRARY: "libdlsslop-network.so",
}
# Destination: (source, required first line, installed mode).
SCRIPT_SOURCES = {
    "bin/dlsslop-run": ("scripts/dlsslop-run", b"#!/usr/bin/bash\n", 0o755),
    "bin/dlsslop-test": ("scripts/dlsslop-test", b"#!/usr/bin/python3\n", 0o755),
    "bin/dlsslop-setup": ("scripts/fetch-assets.py", b"#!/usr/bin/python3\n", 0o755),
    "libexec/dlsslop-amd/color_metrics.py": ("scripts/color_metrics.py", b"", 0o644),
}
LICENSE_SOURCES = {
    "licenses/AGPL-3.0.txt": "LICENSE",
    "licenses/GPL-3.0.txt": "external/layer/third_party/optiscaler/LICENSE",
    "licenses/Apache-2.0.txt": "packaging/Apache-2.0.txt",
    "licenses/MIT-integration.txt": "LICENSE.integration",
    "licenses/MIT-amd.txt": "external/amd/LICENSE",
    "licenses/MIT-DLSSNR-AMD.txt": "external/vulkan/LICENSE",
    "licenses/RenoDX.txt": "external/layer/third_party/optiscaler/RenoDX_ATTRIBUTION.txt",
    "licenses/THIRD-PARTY.txt": "packaging/THIRD-PARTY.txt",
}
DOCUMENT_SOURCES = {"README.md": "packaging/README.md", **LICENSE_SOURCES}
# systemd finds user units here under ~/.local and /usr/local.
UNIT_SOURCES = {f"share/systemd/user/{name}": f"packaging/{name}" for name in ("dlsslop.socket", "dlsslop.service")}


def require_file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular file: {path}")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_elf(path, machine=62, shared=False):
    require_file(path)
    with path.open("rb") as stream:
        header = stream.read(64)
    if (len(header) != 64 or header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(header[16:18], "little") not in ((3,) if shared else (2, 3))
            or int.from_bytes(header[18:20], "little") != machine):
        kind = "gfx1201 GPU module" if machine == 224 else "x86-64 Linux binary"
        raise ValueError(f"not a {kind}: {path}")


def checksum_records(path):
    require_file(path)
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            raise ValueError(f"invalid checksum entry in {path}")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or name in records):
            raise ValueError(f"unsafe or duplicate checksum path in {path}: {name}")
        records[name] = digest
    return records


def validate_modules(directory):
    require_file(directory / "modules.json")
    entries = json.loads((directory / "modules.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError("GPU module manifest must be a list of module records")
    if len(entries) != len(MODULE_NAMES) or {row.get("module") for row in entries} != set(MODULE_NAMES):
        raise ValueError(f"GPU module manifest must contain all {len(MODULE_NAMES)} modules exactly once")
    checksums = checksum_records(directory / "SHA256SUMS")
    expected = {name + ".hsaco" for name in MODULE_NAMES}
    if set(checksums) != expected:
        raise ValueError(f"GPU checksum inventory must contain all {len(MODULE_NAMES)} modules exactly once")
    for row in entries:
        name = row["module"] + ".hsaco"
        path = directory / name
        check_elf(path, machine=224, shared=True)
        digest = sha256(path)
        if (row.get("target") != "gfx1201" or row.get("sha256") != digest
                or checksums[name] != digest or row.get("bytes") != path.stat().st_size):
            raise ValueError(f"GPU module metadata or checksum mismatch: {path}")


def vulkan_shader_names(root):
    """The files below the network's SPIR-V directory that its runtime reads, named by the runtime's tables:
    each kernel's g_STEM.spv (kKernels), the markers (kMarkers), the SPIR-V of the runtime's own pipelines
    (kAdapterBindings), and the shader constants that it checks beside the kernels' and the temporal SPIR-V."""
    # The tables are read rather than compiled: each entry starts with its stem or file, and the
    # vulkan-files test checks the result against the files that a build opens.
    def table(path, name, entry):
        text = (root / path).read_text(encoding="utf-8")
        found = re.findall(entry, text.partition(f" {name}[] = {{")[2].partition("};")[0])
        if not found:
            raise ValueError(f"no {name} table in {root / path}")
        return found
    return ([f"g_{stem}.spv" for stem in table("common/vulkan_plan.h", "kKernels", r'\{"(\w+)",')] +
            table("common/vulkan_plan.h", "kMarkers", r'\{"([\w.-]+)",') +
            table("common/vulkan_runtime.cpp", "kAdapterBindings", r'"(\w+/\w+\.spv)"') +
            ["shader-constants.txt", "temporal/shader-constants.txt"])


def runtime_files(root, build):
    """Return the exact installed runtime allowlist and validate every input."""
    files = {}
    for destination, source in NATIVE_SOURCES.items():
        path = build / source
        library = destination in (LAYER_LIBRARY, NETWORK_LIBRARY)
        check_elf(path, shared=library)
        # The Vulkan loader maps the layer, and the layer the network; nothing executes them.
        files[destination] = (path, 0o644 if library else 0o755)
    for destination, (source, first_line, mode) in SCRIPT_SOURCES.items():
        path = root / source
        require_file(path)
        if not path.read_bytes().startswith(first_line):
            raise ValueError(f"incorrect executable interpreter: {path}")
        files[destination] = (path, mode)
    for name in vulkan_shader_names(root):
        path = build / "vulkan-nr/network" / name
        require_file(path)
        files[f"{VULKAN_DIRECTORY}/{name}"] = (path, 0o644)
    for name in MODEL_TOOLS:
        path = root / "external/vulkan/linux/package/model-tools" / name
        require_file(path)
        files[f"{MODEL_TOOLS_DIRECTORY}/{name}"] = (path, 0o644)
    module_source = root / "assets/HIP/gfx1201"
    validate_modules(module_source)
    for name in (*[name + ".hsaco" for name in MODULE_NAMES], "modules.json", "SHA256SUMS"):
        files[f"{MODULE_DIRECTORY}/{name}"] = (module_source / name, 0o644)
    return files


def layer_manifest():
    """The implicit layer manifest, naming the library relative to itself so the tree relocates."""
    library = os.path.relpath(LAYER_LIBRARY, PurePosixPath(LAYER_MANIFEST).parent)
    return json.dumps({
        "file_format_version": "1.2.0",
        "layer": {
            "name": "VK_LAYER_LOCAL_dlsslop_amd",
            "type": "GLOBAL",
            "library_path": library,
            "api_version": "1.3.277",
            "implementation_version": "1",
            "description": "DLSS Linux Open Proxy for AMD presentation layer",
            "enable_environment": {"DLSSLOP_AMD_ENABLE": "1"},
            "disable_environment": {"DLSSNR_DISABLE": "1"},
        },
    }, indent=2) + "\n"


def tree(root, build, source_notice):
    """Every installed file relative to the prefix, as (content, mode); all inputs validated first."""
    files = runtime_files(root, build)
    for source in (*DOCUMENT_SOURCES.values(), *UNIT_SOURCES.values()):
        require_file(root / source)
    files.update((f"{DOC_DIRECTORY}/{name}", (root / source, 0o644)) for name, source in DOCUMENT_SOURCES.items())
    files.update((name, (root / source, 0o644)) for name, source in UNIT_SOURCES.items())
    entries = {name: (path.read_bytes(), mode) for name, (path, mode) in files.items()}
    entries[f"{DOC_DIRECTORY}/licenses/SOURCES"] = (source_notice.encode(), 0o644)
    entries[LAYER_MANIFEST] = (layer_manifest().encode(), 0o644)
    entries[INVENTORY] = ("".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n"
                                  for name, (content, _) in sorted(entries.items())).encode(), 0o644)
    return entries


def atomic_write(target, data, mode=0o644):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    parser.add_argument("-p", "--prefix", type=Path, default=Path.home() / ".local",
                        help="installation prefix (default: %(default)s)")
    parser.add_argument("-b", "--build-dir", type=Path, default=root / "build",
                        help="source-build binaries (default: %(default)s)")
    args = parser.parse_args()
    prefix = args.prefix.expanduser().resolve()
    notice = ("dlsslop-amd corresponding source\n\n"
              f"Installed from local source checkout: {root.as_uri()}\n"
              "Dependency source URLs and revisions are recorded in upstreams.lock.json\n"
              "in that checkout. See THIRD-PARTY.txt for component attribution.\n")
    try:
        # Read and validate every input before creating or changing installed files.
        files = tree(root, args.build_dir.expanduser().resolve(), notice)
        # The files that an earlier install's inventory lists and this one leaves out, removed last.
        inventory = prefix / INVENTORY
        stale = set(checksum_records(inventory)).difference(files) if inventory.exists() else ()
        for name, (content, mode) in files.items():
            atomic_write(prefix / name, content, mode)
        for name in stale:
            (prefix / name).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Installed dlsslop-amd in {prefix}")
    print("Import your model with dlsslop-setup; see share/doc/dlsslop-amd/README.md there for the worker service.")
    print(f"Steam launch options: {shlex.quote(str(prefix / 'bin/dlsslop-run'))} -- %command%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
