#!/usr/bin/env python3
"""Exercise launcher channel checks, argument forwarding and environment contracts."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/dlsslop-run"
BASH = "/usr/bin/bash"


def run(argv, expected=0, env=None):
    result = subprocess.run(argv, env=env, text=True, capture_output=True)
    if result.returncode != expected:
        raise AssertionError(f"{argv!r}: exit {result.returncode}, expected {expected}\n"
                             f"stdout={result.stdout}\nstderr={result.stderr}")
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="dlsslop-amd-launcher-") as temporary:
        base = Path(temporary)
        channel = base / "worker channel" / "shm.bin"
        channel.parent.mkdir(mode=0o700)
        clean_env = {key: value for key, value in os.environ.items()
                     if not key.startswith(("DLSSLOP_", "DLSSNR_"))}
        env = dict(clean_env, DLSSNR_SHM=str(channel))
        runtime_default = f"/tmp/dlsslop-amd-{os.getuid()}/shm.bin"
        default_help = run([BASH, str(LAUNCHER), "--help"], env=clean_env).stdout
        assert runtime_default in default_help
        assert "Usage: dlsslop-run" in default_help
        assert "DLSSNR_SHM" in default_help and "DLSSNR_LOG" in default_help
        assert "dlssnr-amd" not in default_help
        help_text = run([BASH, str(LAUNCHER), "--help"], env=env).stdout
        assert "--shm" in help_text and "--log" in help_text and "default:" in help_text
        assert str(channel) in help_text
        for option in ("--help", "-h"):
            run([BASH, str(LAUNCHER), option], env=env)
        run([BASH, str(LAUNCHER)], expected=2, env=env)
        run([BASH, str(LAUNCHER), "--shm"], expected=2, env=env)
        run([BASH, str(LAUNCHER), "--shm="], expected=2, env=env)
        run([BASH, str(LAUNCHER), "--unknown"], expected=2, env=env)
        result = run([BASH, str(LAUNCHER), "/usr/bin/true"], expected=1, env=env)
        assert "Start dlsslopd on the host first" in result.stderr
        assert not channel.exists()
        channel.touch(mode=0o600)
        result = run([BASH, str(LAUNCHER), "/usr/bin/true"], expected=1, env=env)
        assert "No live native worker" in result.stderr
        result = run([BASH, str(LAUNCHER), "/usr/bin/true"], expected=1,
                     env=dict(env, PATH=str(base)))
        assert "requires flock" in result.stderr
        bad_flock = base / "flock"
        bad_flock.write_text("#!/usr/bin/bash\nexit 42\n")
        bad_flock.chmod(0o755)
        result = run([BASH, str(LAUNCHER), "/usr/bin/true"], expected=1,
                     env=dict(env, PATH=f"{base}:{os.environ['PATH']}"))
        assert "Could not inspect" in result.stderr
        bad_flock.unlink()

        # A filename beginning with '-' exercises the launcher's -- delimiter
        # and exec --; special characters exercise literal argv forwarding.
        recorder = base / "-capture-command"
        recorder.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}))\n"
            "sys.exit(int(os.environ.get('FIXTURE_EXIT', '0')))\n")
        recorder.chmod(0o755)
        launch_env = dict(env, PATH=f"{base}:{os.environ['PATH']}",
                          DLSSNR_DISABLE="1", DLSSLOP_AMD_ENABLE="0",
                          DLSSNR_ENABLE="0", DLSSLOP_BACKEND="inherited",
                          DLSSNR_DMABUF="1", DLSSNR_IDLE_REPAINT="1",
                          DLSS5_VIT_ADAPTIVE="0", DLSSLOP_HIP_LIBRARY="external library",
                          VKLayer_DLSS5="1", FIXTURE_EXIT="37")
        arguments = ["--shm", "unchanged", "", "two words", "literal '$() ` text"]
        with channel.open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run([BASH, str(LAUNCHER), "--", recorder.name, *arguments],
                         expected=37, env=launch_env)
            got = json.loads(result.stdout)
            assert got["argv"] == arguments
            assert got["env"]["DLSSNR_SHM"] == str(channel)
            assert got["env"]["DLSSNR_LOG"] == str(channel.parent / "layer.log")
            assert got["env"]["DLSSLOP_BACKEND"] == "hip"
            assert got["env"]["DLSSLOP_AMD_ENABLE"] == "1"
            assert got["env"]["DLSSNR_ENABLE"] == "1"
            assert got["env"]["DLSSNR_DMABUF"] == "0"
            assert got["env"]["DLSSNR_IDLE_REPAINT"] == "0"
            assert "DLSSNR_DISABLE" not in got["env"] and "VKLayer_DLSS5" not in got["env"]
            assert "DLSSLOP_SHM" not in got["env"] and "DLSSLOP_LOG" not in got["env"]
            assert got["env"]["DLSS5_VIT_ADAPTIVE"] == "0"
            assert got["env"]["DLSSLOP_HIP_LIBRARY"] == "external library"
            override_env = dict(launch_env, DLSSNR_SHM=str(base / "missing"),
                                DLSSNR_LOG="inherited.log")
            result = run([BASH, str(LAUNCHER), f"--shm={channel}", "-l", "selected log", str(recorder)],
                         expected=37, env=override_env)
            assert json.loads(result.stdout)["env"]["DLSSNR_LOG"] == "selected log"
            result = run([BASH, str(LAUNCHER), f"-s{channel}", "-lselected log", str(recorder)],
                         expected=37, env=override_env)
            assert json.loads(result.stdout)["env"]["DLSSNR_LOG"] == "selected log"
            result = run([BASH, str(LAUNCHER), "missing-dlsslop-amd-command"], expected=127, env=env)
            assert "could not execute" in result.stderr

    print("PASS: launcher channel checks, argv/exit forwarding and upstream environment contracts")


if __name__ == "__main__":
    main()
