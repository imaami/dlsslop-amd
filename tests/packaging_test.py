#!/usr/bin/env python3
"""Check binary release boundaries and complete source/release installation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


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
    paths = {"install.py", "packaging/README.md", *INSTALLER.SCRIPT_SOURCES.values(),
             *INSTALLER.LICENSE_SOURCES.values()}
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    build = root / "build"
    for name in INSTALLER.NATIVE_SOURCES.values():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile("/usr/bin/true", target)
        target.chmod(0o755)
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


def installed(prefix, manifest, runtime, env):
    commands = {"dlsslopd", "dlsslopctl", "dlsslop-gui", "dlsslop-run", "dlsslop-test", "dlsslop-setup"}
    assert {path.name for path in (prefix / "bin").iterdir()} == commands
    for name, (source, mode) in runtime.items():
        target = prefix / name
        assert target.read_bytes() == source.read_bytes(), name
        assert target.stat().st_mode & 0o777 == mode, name
    for name in ("dlsslopd", "dlsslopctl", "dlsslop-gui"):
        assert (prefix / "bin" / name).read_bytes().startswith(b"\x7fELF")
        run([prefix / "bin" / name], env)
    assert (prefix / "bin/dlsslop-run").read_bytes().startswith(b"#!/usr/bin/bash\n")
    run(["/usr/bin/bash", "-n", prefix / "bin/dlsslop-run"], env)
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
    layer = json.loads((manifest / "VK_LAYER_LOCAL_dlsslop_amd.json").read_text())["layer"]
    assert layer["library_path"] == str(prefix / INSTALLER.LAYER_LIBRARY)
    assert layer["enable_environment"] == {"DLSSLOP_AMD_ENABLE": "1"}
    assert layer["disable_environment"] == {"DLSSNR_DISABLE": "1"}
    assert layer["name"] == "VK_LAYER_LOCAL_dlsslop_amd"


def refuses_install(source, build, base, env, label):
    prefix, manifest = base / (label + "-prefix"), base / (label + "-manifest")
    run([sys.executable, source / "install.py", "-b", build, "-p", prefix, "-m", manifest], env, 2)
    assert not prefix.exists() and not manifest.exists(), "invalid input mutated installation"


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
        for option in ("--prefix", "--build-dir", "--manifest-dir", "default:"):
            assert option in run([sys.executable, source / "install.py", "--help"], env)
        prefix, manifest = base / "installed '$() ` with spaces", base / "manifest path"
        run([sys.executable, source / "install.py", "-b", build, "-p", prefix, "-m", manifest], env)
        installed(prefix, manifest, runtime, env)
        assert (prefix / "share/doc/dlsslop-amd/README.md").read_bytes() == (source / "packaging/README.md").read_bytes()
        for name, source_name in INSTALLER.LICENSE_SOURCES.items():
            assert (prefix / "share/doc/dlsslop-amd" / name).read_bytes() == (source / source_name).read_bytes()
        assert source.as_uri() in (prefix / "share/doc/dlsslop-amd/licenses/SOURCES").read_text()

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

        # Files outside the allowlist never leak into the release.
        for name in ("secret-weights.f16", "assets/model.onnx", "build/color-gpu-test", "tests/private.cpp",
                     "assets/HIP/gfx1201/unlisted.hsaco", "assets/HIP/gfx1201/linux_color.generated.hip"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("must not ship\n")
        output = base / "release.tar.xz"
        source_url = "https://example.invalid/source/exact-revision"
        PACKAGER.package(source, build, output, source_url)
        with tarfile.open(output) as archive:
            names = {member.name.removeprefix("dlsslop-amd/") for member in archive.getmembers()}
            assert names == INSTALLER.release_names(runtime) | {"PACKAGE-SHA256SUMS"}
            assert all(member.isfile() and member.name.startswith("dlsslop-amd/")
                       and ".." not in Path(member.name).parts for member in archive.getmembers())
            archive.extractall(base / "extracted", filter="data")
        release = base / "extracted/dlsslop-amd"
        release_runtime = INSTALLER.runtime_files(release, build, True)
        INSTALLER.validate_release(release, release_runtime)
        checked = subprocess.run(["sha256sum", "--check", "PACKAGE-SHA256SUMS"], cwd=release,
                                 env=env, capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        assert source_url in (release / "licenses/SOURCES").read_text()
        assert not (release / "build").exists() and not (release / "scripts").exists()
        release_prefix, release_manifest = base / "release prefix", base / "release manifest"
        run([sys.executable, release / "install.py", "-p", release_prefix, "-m", release_manifest], env)
        installed(release_prefix, release_manifest, release_runtime, env)
        for name in ("README.md", *INSTALLER.LICENSE_SOURCES, "licenses/SOURCES"):
            assert (release_prefix / "share/doc/dlsslop-amd" / name).read_bytes() == (release / name).read_bytes()
        (release / "bin/dlsslopd").write_bytes(original + b"tampered")
        refuses_install(release, build, base, env, "changed-release")

        # Package validation also finishes before replacing an existing archive.
        output.write_bytes(b"existing archive")
        (modules / "linux_color.hsaco").write_bytes(b"invalid GPU code")
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
