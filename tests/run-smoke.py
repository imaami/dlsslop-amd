#!/usr/bin/env python3
"""Exercise real Vulkan capture/resolve/present with an explicitly fake neural worker.

Needs Pillow, a Vulkan ICD (Mesa lavapipe is sufficient), and Xvfb unless
--headless is selected. The test layer is
separately compiled with DLSSLOP_TEST_LAVAPIPE; production never admits CPU ICDs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from PIL import Image, ImageChops


def wait_for(check, process, label):
    end = time.monotonic() + 15
    while not check():
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited with {process.returncode}")
        if time.monotonic() >= end:
            raise RuntimeError(f"{label} did not become ready")
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="show help and exit (default: do not show help)")
    parser.add_argument("-b", "--build-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build", help="compiled test binaries")
    parser.add_argument("-x", "--xvfb", default=shutil.which("Xvfb"),
                        help="Xvfb executable; auto-detected on PATH, None means unavailable")
    parser.add_argument("-d", "--display", default=":98", help="X11 test display")
    parser.add_argument("-H", "--headless", action="store_true",
                        help="use VK_EXT_headless_surface without an X server")
    args = parser.parse_args()
    if not args.xvfb and not args.headless:
        parser.error("Xvfb is required")
    build = args.build_dir.resolve()
    logs = build / "smoke-logs"
    logs.mkdir(exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="dlsslop-amd-smoke-") as directory:
        root = Path(directory)
        env = dict(os.environ)
        env.update(DISPLAY=args.display, XDG_DATA_HOME=str(root / "share"),
                   XDG_RUNTIME_DIR=str(root), DLSSNR_SHM=str(root / "shm.bin"),
                   DLSSLOP_AMD_ENABLE="1", DLSSNR_ENABLE="1", DLSSLOP_BACKEND="hip",
                   DLSSNR_DMABUF="0", DLSSNR_IDLE_REPAINT="0", DLSSLOP_INPUT="off",
                   DLSSNR_LOG=str(logs / "layer.log"))
        env.pop("DLSSNR_DISABLE", None)
        env.pop("VKLayer_DLSS5", None)
        layer = root / "share/vulkan/implicit_layer.d/test.json"
        layer.parent.mkdir(parents=True)
        layer.write_text(json.dumps({"file_format_version": "1.2.0", "layer": {
            "name": "VK_LAYER_LOCAL_dlsslop_amd", "type": "GLOBAL",
            "library_path": str(build / "libVkLayer_DLSSLOP_test.so"),
            "api_version": "1.3.277", "implementation_version": "1",
            "description": "TEST ONLY - CPU ICD admitted",
            "enable_environment": {"DLSSLOP_AMD_ENABLE": "1"},
            "disable_environment": {"DLSSNR_DISABLE": "1"}}}))
        try:
            with (logs / "worker.log").open("w") as worker_log, (logs / "xvfb.log").open("w") as xvfb_log:
                worker = subprocess.Popen([str(build / "dlsslopd"), "--test-identity"],
                                          env=env, stdout=worker_log, stderr=subprocess.STDOUT)
                processes.append(worker)
                wait_for(lambda: "worker ready:" in (logs / "worker.log").read_text(), worker, "identity worker")
                if not args.headless:
                    xvfb = subprocess.Popen([args.xvfb, args.display, "-screen", "0", "640x480x24", "-nolisten", "tcp"],
                                            env=env, stdout=xvfb_log, stderr=subprocess.STDOUT)
                    processes.append(xvfb)
                    socket = Path("/tmp/.X11-unix") / ("X" + args.display.lstrip(":"))
                    wait_for(socket.exists, xvfb, "Xvfb")
                layer_log = logs / "layer.log"
                for mode in ("native", "reduced", "reduced-bgra", "native-fp16", "reduced-fp16", "native-linear",
                             "stopped-worker", "killed-worker"):
                    # A stopped worker gets no requests. A killed one still reads as
                    # running; its silent heartbeat ends the one wait within a second.
                    if mode == "stopped-worker":
                        worker.terminate()
                        if worker.wait(timeout=5):
                            raise RuntimeError(f"identity worker exited with {worker.returncode} when stopped")
                    elif mode == "killed-worker":
                        with (logs / "worker-killed.log").open("w") as killed_log:
                            worker = subprocess.Popen([str(build / "dlsslopd"), "--test-identity"],
                                                      env=env, stdout=killed_log, stderr=subprocess.STDOUT)
                        processes.append(worker)
                        wait_for(lambda: "worker ready:" in (logs / "worker-killed.log").read_text(),
                                 worker, "identity worker")
                        worker.kill()
                        worker.wait()
                    state = root / mode
                    env["XDG_STATE_HOME"] = str(state)
                    logged = layer_log.stat().st_size if layer_log.exists() else 0
                    command = [str(build / "vulkan-smoke"),
                               "--no-worker" if mode.endswith("worker") else "--contention"]
                    if args.headless:
                        command.append("--headless")
                    if mode.startswith("reduced"):
                        command.append("--reduced")
                    if mode.endswith("bgra"):
                        command.append("--bgra")
                    if mode.endswith("fp16"):
                        command.append("--proxy16")
                    # Software Vulkan accepts descriptor pools that lack a type the
                    # set layout needs; the validation layer (when installed) reports
                    # what RADV refuses.
                    mode_env = env
                    if mode.endswith("linear"):
                        command.append("--linear-hdr")
                        mode_env = dict(env, VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation")
                    result = subprocess.run(command, env=mode_env, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                    (logs / f"smoke-{mode}.log").write_text(result.stdout)
                    print(result.stdout, end="")
                    if result.returncode:
                        raise RuntimeError(f"{mode} smoke test failed ({result.returncode}); inspect {logs}")
                    if "Validation Error" in result.stdout or "AllocateDescriptorSets" in result.stdout:
                        raise RuntimeError(f"{mode} emitted Vulkan validation errors; inspect {logs}")
                    with layer_log.open() as log:
                        log.seek(logged)
                        layer_output = log.read()
                    if "cannot run here" in layer_output:
                        raise RuntimeError(f"{mode} composition could not be prepared; inspect {logs}")
                    if mode.endswith("worker"):
                        if mode == "killed-worker" and "worker did not answer frame" not in layer_output:
                            raise RuntimeError(f"layer did not log giving up on the killed worker; inspect {logs}")
                        print(f"PASS: {mode}: the layer presented without blocking on it.")
                        continue
                    captures = state / "dlssnr/captures"
                    manifest = dict(line.split(None, 1) for line in (captures / 'manifest.txt').read_text().splitlines()
                                    if line and not line.startswith('#') and len(line.split(None, 1)) == 2)
                    if manifest.get('capture_metadata_version') != '1':
                        raise RuntimeError('Capture completion lacks metadata version 1')
                    captures = captures / manifest['batch_dir']
                    for index in range(4):
                        before = Image.open(captures / f"before_{index:02}.png").convert("RGBA")
                        after = Image.open(captures / f"after_{index:02}.png").convert("RGBA")
                        error = max(high for low, high in ImageChops.difference(before, after).getextrema())
                        if error > 2:
                            raise RuntimeError(f"{mode} composed image changed identity pixels: max error {error}")
                    print(f"PASS: {mode} compositor preserved all pixels in four captured frames (error <= 2/255).")
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    print("PASS: Vulkan transport/composition only; no neural GPU inference tested.")


if __name__ == "__main__":
    main()
