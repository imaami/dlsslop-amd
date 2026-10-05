#!/usr/bin/env python3
"""Check the binary release tree, its archive and complete source installation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from unittest.mock import patch

from shm_name import channel_name


ROOT = Path(__file__).resolve().parents[1]
# The build directory, where CMake made the files of INSTALLER.GENERATED.
BUILD = Path(sys.argv[1]).resolve()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTALLER = load_module("installer_test", ROOT / "install.py")
PACKAGER = load_module("packager_test", ROOT / "scripts/package-ci.py")


def run(arguments, env, expected=0):
    result = subprocess.run([str(argument) for argument in arguments], env=env,
                            capture_output=True, text=True)
    assert result.returncode == expected, (arguments, result.returncode, result.stdout, result.stderr)
    return result.stdout


def fixture(root):
    root.mkdir()
    paths = {"install.py", *INSTALLER.DOCUMENT_SOURCES.values(), *INSTALLER.UNIT_SOURCES.values(),
             *(source for source, *_ in INSTALLER.SCRIPT_SOURCES.values()),
             "common/vulkan_plan.c", "common/vulkan_runtime.c",
             *(f"external/vulkan/linux/package/model-tools/{name}" for name in INSTALLER.MODEL_TOOLS)}
    build = root / "build"
    for name in paths:
        target = INSTALLER.source_path(root, build, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INSTALLER.source_path(ROOT, BUILD, name), target)
    for name in INSTALLER.NATIVE_SOURCES.values():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile("/usr/bin/true", target)
        target.chmod(0o755)
    # The network's files, and two that its shader build writes and its runtime never reads.
    for name in (*INSTALLER.vulkan_shader_names(root), "g_repack.spv", "runtime/runtime_transfer.spv"):
        target = build / "vulkan-nr/network" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name.encode())
    modules = root / "assets/HIP/gfx1201"
    modules.mkdir(parents=True)
    entries = []
    for name in INSTALLER.MODULE_NAMES:
        # A minimal ELF fixture exercises architecture and hash validation
        # without needing a compiler, HIP runtime or GPU in this host test.
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[16:20] = b"\x03\x00\xe0\x00"
        content = bytes(header) + name.encode()
        (modules / (name + ".hsaco")).write_bytes(content)
        entries.append({"module": name, "target": "gfx1201", "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    (modules / "modules.json").write_text(json.dumps(entries))
    (modules / "SHA256SUMS").write_text("".join(
        f'{row["sha256"]}  {row["module"]}.hsaco\n' for row in entries))
    return build, modules


def installed(prefix, runtime, env):
    commands = {"dlsslopd", "dlsslopctl", "dlsslop-gui", "dlsslop-run", "dlsslop-test", "dlsslop-setup"}
    assert {path.name for path in (prefix / "bin").iterdir()} == commands
    for name, (source, mode) in runtime.items():
        target = prefix / name
        assert target.read_bytes() == source.read_bytes(), name
        assert target.stat().st_mode & 0o777 == mode, name
    assert (prefix / INSTALLER.LAYER_LIBRARY).stat().st_mode & 0o777 == 0o644, "the layer library is executable"
    vulkan = prefix / INSTALLER.VULKAN_DIRECTORY
    assert {path.relative_to(vulkan).as_posix() for path in vulkan.rglob("*") if path.is_file()} == set(
        INSTALLER.vulkan_shader_names(ROOT)), "a file that the network's runtime never reads is installed"
    network = prefix / INSTALLER.NETWORK_LIBRARY
    assert network.parent == (prefix / INSTALLER.LAYER_LIBRARY).parent, "the network is not beside the layer, which loads it"
    assert network.stat().st_mode & 0o777 == 0o644, "the network library is executable"
    for name in ("dlsslopd", "dlsslopctl", "dlsslop-gui"):
        assert (prefix / "bin" / name).read_bytes().startswith(b"\x7fELF")
        run([prefix / "bin" / name], env)
    assert (prefix / "bin/dlsslop-run").read_bytes().startswith(b"#!/usr/bin/bash\n")
    run(["/usr/bin/bash", "-n", prefix / "bin/dlsslop-run"], env)
    # The launcher, the color diagnostic and the socket name the default channel of this protocol.
    name = channel_name()
    assert f"channel_default=${{DLSSNR_SHM:-/tmp/dlsslop-amd-${{UID}}/{name}}}\n" in (
        prefix / "bin/dlsslop-run").read_text()
    assert f"/tmp/dlsslop-amd-{{os.getuid()}}/{name}'" in (prefix / "bin/dlsslop-test").read_text()
    assert f"\nListenSequentialPacket=/tmp/dlsslop-amd-%U/{name}.sock\n" in (
        prefix / "share/systemd/user/dlsslop.socket").read_text()
    for name in ("dlsslop-test", "dlsslop-setup"):
        assert (prefix / "bin" / name).read_bytes().startswith(b"#!/usr/bin/python3\n")
        assert "default:" in run([prefix / "bin" / name, "--help"], env)
    assert "".join(str(Path(env["XDG_DATA_HOME"]) / "dlsslop-amd/model").split()) in "".join(run(
        [prefix / "bin/dlsslop-setup", "--help"], env).split())
    inspection = run([
        sys.executable, "-c",
        "import json, runpy, sys; d = runpy.run_path(sys.argv[1]); "
        "print(json.dumps([d['metrics_module']().__file__, str(d['capture_directory']({}))]))",
        prefix / "bin/dlsslop-test"], env)
    helper, captures = json.loads(inspection)
    assert helper == str(prefix / "libexec/dlsslop-amd/color_metrics.py")
    assert captures == str(Path(env["XDG_STATE_HOME"]) / "dlssnr/captures")
    manifest = prefix / INSTALLER.LAYER_MANIFEST
    layer = json.loads(manifest.read_text())["layer"]
    # Relative to the manifest, so any prefix works, including a moved one.
    assert not layer["library_path"].startswith("/")
    assert (manifest.parent / layer["library_path"]).resolve() == (prefix / INSTALLER.LAYER_LIBRARY).resolve()
    assert layer["enable_environment"] == {"DLSSLOP_AMD_ENABLE": "1"}
    assert layer["disable_environment"] == {"DLSSNR_DISABLE": "1"}
    assert layer["name"] == "VK_LAYER_LOCAL_dlsslop_amd"


def refuses_install(source, build, base, env, label):
    prefix = base / (label + "-prefix")
    run([sys.executable, source / "install.py", "-b", build, "-p", prefix], env, 2)
    assert not prefix.exists(), "invalid input mutated installation"


def main():
    kernels = load_module("kernels_test", ROOT / "scripts/build-kernels.py")
    assert set(INSTALLER.MODULE_NAMES) == {row[0] for row in kernels.MODULES}
    with tempfile.TemporaryDirectory(prefix="dlsslop-package-") as temporary:
        base = Path(temporary)
        source = base / "source checkout"
        build, modules = fixture(source)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("DLSSNR_", "DLSSLOP_"))}
        env.update(XDG_DATA_HOME=str(base / "xdg data"), XDG_STATE_HOME=str(base / "xdg state"),
                   PYTHONDONTWRITEBYTECODE="1")
        runtime = INSTALLER.runtime_files(source, build)
        install_help = run([sys.executable, source / "install.py", "--help"], env)
        for option in ("--prefix", "--build-dir", "default:"):
            assert option in install_help
        assert "--manifest-dir" not in install_help
        prefix = base / "installed '$() ` with spaces"
        run([sys.executable, source / "install.py", "-b", build, "-p", prefix], env)
        installed(prefix, runtime, env)
        # A source installation carries the same inventory as a release.
        checked = subprocess.run(["sha256sum", "--check", INSTALLER.INVENTORY], cwd=prefix,
                                 env=env, capture_output=True, text=True)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert (prefix / "share/doc/dlsslop-amd/README.md").read_bytes() == (source / "packaging/README.md").read_bytes()
        for name, source_name in INSTALLER.LICENSE_SOURCES.items():
            assert (prefix / "share/doc/dlsslop-amd" / name).read_bytes() == (source / source_name).read_bytes()
        assert source.as_uri() in (prefix / "share/doc/dlsslop-amd/licenses/SOURCES").read_text()
        # Over an earlier install, the installer removes the files that the earlier inventory lists and
        # this install leaves out, and keeps the files that it does not list.
        stale = prefix / INSTALLER.VULKAN_DIRECTORY / "g_repack.spv"
        kept = prefix / "share/dlsslop-amd/dlssnr.bin"
        for path in (stale, kept):
            path.write_bytes(b"earlier")
        digest = hashlib.sha256(b"earlier").hexdigest()
        with (prefix / INSTALLER.INVENTORY).open("a") as inventory:
            inventory.write(f"{digest}  {stale.relative_to(prefix).as_posix()}\n")
        run([sys.executable, source / "install.py", "-b", build, "-p", prefix], env)
        assert not stale.exists() and kept.exists(), "an earlier install's file was not removed, or another was"
        installed(prefix, runtime, env)

        # A missing GUI and an incomplete module inventory each fail before
        # installing any binaries or publishing a Vulkan manifest.
        gui = build / "gui/dlsslop-gui"
        original = gui.read_bytes()
        gui.unlink()
        refuses_install(source, build, base, env, "missing-gui")
        gui.write_bytes(original)
        sums = modules / "SHA256SUMS"
        original_sums = sums.read_bytes()
        sums.write_text(sums.read_text().splitlines()[0] + "\n")
        refuses_install(source, build, base, env, "incomplete-modules")
        sums.write_bytes(original_sums)

        # A script whose first line is not exactly its interpreter line
        # (another interpreter, or a longer path sharing the prefix) is
        # refused before anything is installed.
        for name, first_line in (("dlsslop-run", b"#!/bin/sh\n"), ("dlsslop-test", b"#!/usr/bin/python3.99\n")):
            script = build / name
            original_script = script.read_bytes()
            script.write_bytes(first_line + original_script.partition(b"\n")[2])
            refuses_install(source, build, base, env, "wrong-interpreter")
            script.write_bytes(original_script)

        # Files outside the allowlist never leak into the release.
        for name in ("secret-weights.f16", "assets/model.onnx", "build/color-gpu-test", "tests/private.cpp",
                     "assets/HIP/gfx1201/unlisted.hsaco", "assets/HIP/gfx1201/linux_color.hsaco",
                     "assets/HIP/gfx1201/linux_native.generated.hip"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("must not ship\n")
        output = base / "release.tar.xz"
        source_url = "https://example.invalid/source/exact-revision"
        with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "1790000000"}):
            PACKAGER.package(source, build, output, source_url)
        tree = INSTALLER.tree(source, build, "")
        inventory = INSTALLER.INVENTORY
        with tarfile.open(output) as archive:
            members = archive.getmembers()
            # The archive root is the prefix: regular files only (no directory
            # entries to retouch an existing ~/.local), stamped with the build time.
            assert {member.name for member in members} == set(tree)
            assert all(member.isfile() and not member.name.startswith("/") and ".." not in Path(member.name).parts
                       and member.mtime == 1790000000 for member in members)
            modes = {member.name: member.mode for member in members}
            assert all(modes[name] == mode for name, (_, mode) in tree.items())
            assert modes[INSTALLER.LAYER_LIBRARY] == 0o644, "the archived layer library is executable"
        release = base / "extracted prefix"
        release.mkdir()
        with tarfile.open(output) as archive:
            archive.extractall(release, filter="data")
        checked = subprocess.run(["sha256sum", "--check", inventory], cwd=release,
                                 env=env, capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        installed(release, runtime, env)
        assert source_url in (release / INSTALLER.DOC_DIRECTORY / "licenses/SOURCES").read_text()
        for name in ("install.py", "build", "scripts", "assets"):
            assert not (release / name).exists(), name
        units = [release / name for name in INSTALLER.UNIT_SOURCES]
        assert all(unit.read_bytes() == INSTALLER.source_path(source, build, INSTALLER.UNIT_SOURCES[name]).read_bytes()
                   for name, unit in zip(INSTALLER.UNIT_SOURCES, units))
        if shutil.which("systemd-analyze"):
            verified = subprocess.run(["systemd-analyze", "--user", "verify", *map(str, units)],
                                      env=env, capture_output=True, text=True)
            assert verified.returncode == 0, verified.stderr
        # The release README's update command deletes every file that the inventory lists.
        command = next(line.strip() for line in (source / "packaging/README.md").read_text().splitlines()
                       if "| xargs" in line)
        run(["/usr/bin/bash", "-c", command.replace("~/.local", shlex.quote(str(release)))], env)
        assert [path for path in release.rglob("*") if path.is_file()] == [release / inventory]
        # Without SOURCE_DATE_EPOCH the entries carry the packaging time.
        with patch.dict(os.environ, {}, clear=True):
            before = int(time.time())
            PACKAGER.package(source, build, output, source_url)
        with tarfile.open(output) as archive:
            assert all(before <= member.mtime <= time.time() + 1 for member in archive.getmembers())

        # Package validation also finishes before replacing an existing archive.
        output.write_bytes(b"existing archive")
        (modules / "linux_native.hsaco").write_bytes(b"invalid GPU code")
        try:
            PACKAGER.package(source, build, output, source_url)
        except ValueError:
            pass
        else:
            raise AssertionError("packager accepted invalid GPU code")
        assert output.read_bytes() == b"existing archive"
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/project", "GITHUB_SHA": "a" * 40,
                                     "GITHUB_SERVER_URL": "https://github.com"}, clear=True):
            assert PACKAGER.ci_source_url() == "https://github.com/owner/project/tree/" + "a" * 40
        with patch.dict(os.environ, {}, clear=True):
            assert PACKAGER.ci_source_url() is None
    print("PASS: exact binary release, checksums, complete source/release installs, native commands and Python helpers")


if __name__ == "__main__":
    main()
