#!/usr/bin/env python3
"""Install a complete source build or extracted dlsslop-amd binary release."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tempfile


MODULE_NAMES = (
    "c32_prefix_reference", "multihead-reference", "deep_reference", "boundary_reference",
    "c32_fast", "c32_fast_attention", "boundary-fast", "c32_fused_ffn_attention-packed",
    "prefix_fast", "multihead_fused_attention", "deep_fast-packed",
    "multihead-fast-padded-wave-packed", "linux_codec", "linux_tuning", "linux_color",
    "linux_temporal",
)
MODULE_DIRECTORY = "share/dlsslop-amd/HIP/gfx1201"
LAYER_LIBRARY = "lib/dlsslop-amd/libVkLayer_DLSSLOP_amd.so"
NATIVE_SOURCES = {
    "bin/dlsslopd": "dlsslopd",
    "bin/dlsslopctl": "dlssnr-shmctl",
    "bin/dlsslop-gui": "gui/dlsslop-gui",
    LAYER_LIBRARY: "libVkLayer_DLSSLOP_amd.so",
}
SCRIPT_SOURCES = {
    "bin/dlsslop-run": "scripts/dlsslop-run",
    "bin/dlsslop-test": "scripts/dlsslop-test",
    "bin/dlsslop-setup": "scripts/fetch-assets.py",
    "libexec/dlsslop-amd/color_metrics.py": "scripts/color_metrics.py",
}
LICENSE_SOURCES = {
    "licenses/AGPL-3.0.txt": "LICENSE",
    "licenses/GPL-3.0.txt": "upstream-layer/third_party/optiscaler/LICENSE",
    "licenses/Apache-2.0.txt": "packaging/Apache-2.0.txt",
    "licenses/MIT-integration.txt": "LICENSE.integration",
    "licenses/MIT-amd.txt": "kernels/LICENSE",
    "licenses/RenoDX.txt": "upstream-layer/third_party/optiscaler/RenoDX_ATTRIBUTION.txt",
    "licenses/THIRD-PARTY.txt": "packaging/THIRD-PARTY.txt",
}


def require_file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular file: {path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def runtime_files(root, build, release=False):
    """Return the exact installed runtime allowlist and validate every input."""
    files = {}
    for destination, source in NATIVE_SOURCES.items():
        path = root / destination if release else build / source
        check_elf(path, shared=destination == LAYER_LIBRARY)
        files[destination] = (path, 0o755)
    for destination, source in SCRIPT_SOURCES.items():
        path = root / destination if release else root / source
        require_file(path)
        if destination.startswith("bin/"):
            expected = b"#!/usr/bin/bash\n" if destination == "bin/dlsslop-run" else b"#!/usr/bin/python3\n"
            with path.open("rb") as stream:
                if stream.readline() != expected:
                    raise ValueError(f"incorrect executable interpreter: {path}")
        files[destination] = (path, 0o755 if destination.startswith("bin/") else 0o644)
    module_source = root / (MODULE_DIRECTORY if release else "assets/HIP/gfx1201")
    validate_modules(module_source)
    for name in (*[name + ".hsaco" for name in MODULE_NAMES], "modules.json", "SHA256SUMS"):
        files[f"{MODULE_DIRECTORY}/{name}"] = (module_source / name, 0o644)
    return files


def release_names(runtime):
    return {*runtime, "install.py", "README.md", *LICENSE_SOURCES, "licenses/SOURCES"}


def validate_release(root, runtime):
    checksums = checksum_records(root / "PACKAGE-SHA256SUMS")
    if set(checksums) != release_names(runtime):
        raise ValueError("binary release checksum inventory does not match its runtime allowlist")
    for name, expected in checksums.items():
        path = root / name
        require_file(path)
        if sha256(path) != expected:
            raise ValueError(f"release checksum mismatch: {path}")


def atomic_copy(source, target, mode=0o755):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as dest, source.open("rb") as src:
            shutil.copyfileobj(src, dest)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(text, target, mode=0o644):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    root = Path(__file__).resolve().parent
    release = (root / "bin").is_dir()
    data = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    parser.add_argument("-p", "--prefix", type=Path, default=Path.home() / ".local",
                        help="installation prefix (default: %(default)s)")
    parser.add_argument("-b", "--build-dir", type=Path, default=root / "build",
                        help="source-build binaries (default: %(default)s; ignored in an extracted release)")
    parser.add_argument("-m", "--manifest-dir", type=Path, default=data / "vulkan/implicit_layer.d",
                        help="Vulkan layer manifests (default: %(default)s, using XDG_DATA_HOME or ~/.local/share)")
    args = parser.parse_args()
    prefix = args.prefix.expanduser().resolve()
    manifest_path = args.manifest_dir.expanduser().resolve() / "VK_LAYER_LOCAL_dlsslop_amd.json"
    try:
        files = runtime_files(root, args.build_dir.expanduser().resolve(), release)
        # Finish all input validation before creating or changing installed files.
        source_notice = None
        if release:
            validate_release(root, files)
            documents = {name: name for name in ("README.md", *LICENSE_SOURCES, "licenses/SOURCES")}
        else:
            documents = {"README.md": "packaging/README.md", **LICENSE_SOURCES}
            source_notice = (
                "dlsslop-amd corresponding source\n\n"
                f"Installed from local source checkout: {root.as_uri()}\n"
                "Dependency source URLs and revisions are recorded in upstreams.lock.json\n"
                "in that checkout. See THIRD-PARTY.txt for component attribution.\n"
            )
        for name, source_name in documents.items():
            source = root / source_name
            require_file(source)
            files["share/doc/dlsslop-amd/" + name] = (source, 0o644)
        for destination, (source, mode) in files.items():
            atomic_copy(source, prefix / destination, mode)
        if source_notice is not None:
            atomic_text(source_notice, prefix / "share/doc/dlsslop-amd/licenses/SOURCES")
        manifest = {
            "file_format_version": "1.2.0",
            "layer": {
                "name": "VK_LAYER_LOCAL_dlsslop_amd",
                "type": "GLOBAL",
                "library_path": str(prefix / LAYER_LIBRARY),
                "api_version": "1.3.277",
                "implementation_version": "1",
                "description": "DLSS Linux Open Proxy for AMD presentation layer",
                "enable_environment": {"DLSSLOP_AMD_ENABLE": "1"},
                "disable_environment": {"DLSSNR_DISABLE": "1"},
            },
        }
        atomic_text(json.dumps(manifest, indent=2) + "\n", manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Installed dlsslop-amd in {prefix}")
    print(f"Layer manifest: {manifest_path}")
    print("Import your model with dlsslop-setup, then start dlsslopd from a host terminal.")
    print(f"Steam launch options: {shlex.quote(str(prefix / 'bin/dlsslop-run'))} -- %command%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
