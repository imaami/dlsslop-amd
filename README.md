# DLSS Linux Open Proxy for AMD

![DLSS off/on](dlss_off_on.png)

---

## DISCLAIMER

**This is slop.** Everything here, save for any externally-sourced code<sup>1</sup>,
is AI-generated. The initial implementation was done by GPT 6 Astra running in Ultra
mode after I shitposted into the prompt box.

Please note that I have close to 20 years of experience as a software dev. I
can review AI-generated code perfectly fine. Also note that I haven't read a
single god damned line of code in this repo.

*<sup>1</sup> Most of the externally-sourced code is probably slop, too.*

---

## Description

Experimental Linux source integration of [lmxxf](https://github.com/lmxxf)'s
[AMD neural implementation](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting)
and [bmitch87](https://github.com/bmitch87)'s
[Vulkan presentation layer](https://github.com/bmitch87/DLSS5VKLayer). A native
HIP worker processes a proxy of the game frame; the Vulkan layer composes the
neural edit at the original resolution. The standalone Qt 6 Widgets controller
exposes all 42 settings. Worker, layer, CLI and GUI use shared-memory protocol
**23**.

This repository contains source, patches and build tools. **No compiled binaries
or model weights are included.** See [VALIDATION.md](VALIDATION.md) for automated
tests and hardware testing instructions.

## Prepare a checkout

Clone this repository without recursively checking out its submodules, then run
these commands from the repository root:

```bash
python3 scripts/fetch-submodules.py
python3 scripts/prepare-sources.py
```

The fetch helper uses exact commits, filtered downloads and selected working-tree
paths, avoiding upstream binary payloads. A shallow clone alone does not omit
those files. The helper fails if the server cannot provide blob filtering. The
preparation script verifies pinned inputs, applies `patches/linux-integration.patch`,
and checks the resulting hashes. It generates `upstream-layer/`, `kernels/` and
`backend/vendor/`; those directories are not tracked in the parent repository.
Pins, file mappings and hashes are in `upstreams.lock.json`; attribution is in
[PROVENANCE.md](PROVENANCE.md).

## Build

The reference workflow is [.github/workflows/build.yml](.github/workflows/build.yml)
on Ubuntu 26.04. Its build dependencies are:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
    ca-certificates git curl xz-utils build-essential cmake ninja-build \
    python3 python3-numpy python3-pil qt6-base-dev libx11-dev libxi-dev \
    libvulkan-dev mesa-vulkan-drivers glslang-tools spirv-tools clang-22 lld-22
```

Install DXC 1.9.2607 using the exact download URL and SHA-256 in the workflow.
Set `DXC_DIR` to the extracted directory containing `bin/` and `lib/` (inside
the archive's top-level directory), then build from the prepared checkout:

```bash
LD_LIBRARY_PATH="$DXC_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python3 scripts/build-all-shaders.py --dxc "$DXC_DIR/bin/dxc"
python3 scripts/build-kernels.py --compiler clang++-22 --linker /usr/bin/ld.lld-22
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DDLSSLOP_BUILD_GUI=ON -DBUILD_TESTING=ON -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build --parallel 2
QT_QPA_PLATFORM=offscreen ctest --test-dir build --output-on-failure
```

This compiles and validates ten Vulkan shaders, compiles 28 `gfx1201` HIP modules
(including `linux_color`), and builds the worker, layer, CLI, Qt GUI and tests.
Compilation does not require a GPU, ROCm runtime or model weights. Qt uses the
distribution's shared Qt 6 libraries; the worker and layer have no Qt dependency.
For an independent GUI build, see [gui/README.md](gui/README.md).

On a successful push, pull request or manual Actions run, CI packages fresh
runtime outputs as `dist/dlsslop-amd-linux-gfx1201.tar.xz` in the
`dlsslop-amd-linux-gfx1201` artifact. The archive contains the installer, compiled
binaries and HIP modules, runtime Python tools, and required notices; it contains
no source tree or model weights. Extract it, enter the extracted directory and
check `sha256sum --check PACKAGE-SHA256SUMS` before installation.
`licenses/SOURCES` identifies the corresponding source revision.

## Install and import weights

After building, or after extracting a successful CI artifact, stop any running
game and worker and install:

```bash
python3 install.py
```

The same installer handles a source build or extracted CI archive and installs
the worker, CLI, GUI, diagnostic, model importer and game launcher together.
Installation defaults to `~/.local`:

| Command | Purpose |
|---|---|
| `dlsslopd` | Native HIP worker |
| `dlsslopctl` | Command-line controller |
| `dlsslop-gui` | Qt controller |
| `dlsslop-test` | Color diagnostic |
| `dlsslop-setup` | Model coefficient importer |
| `dlsslop-run` | Game launcher |

The worker, CLI and GUI are installed executables, and the diagnostic and importer
are Python programs. Only the game launcher uses Bash. The Vulkan manifest uses
`$XDG_DATA_HOME/vulkan/implicit_layer.d`, or `~/.local/share/vulkan/implicit_layer.d`.
`install.py --help` describes prefix, build-directory and manifest overrides.
The layer activates only through the launch wrapper.

Runtime inference needs an RX 9070 XT / `gfx1201` device, a compatible Linux HIP
runtime, access to `/dev/kfd` and the render device, and Mesa RADV with the normal
`amdgpu` kernel driver. Run the HIP worker on the host, outside Steam's runtime
container. `DLSSLOP_HIP_LIBRARY` can select a particular `libamdhip64.so`.

Obtain compatible model coefficients separately, using the source project's
[upstream instructions](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting).
Import a local package or extracted model directory:

```bash
dlsslop-setup --source /path/to/model-package.zip
```

The importer and worker default to `$XDG_DATA_HOME/dlsslop-amd/model`, or
`~/.local/share/dlsslop-amd/model` when `XDG_DATA_HOME` is unset or empty. Use the
importer's `--output` and worker's `--assets` options for a different location.
The importer reads coefficient files only, checks all 184 required tables and
their exact sizes, and writes a SHA-256 inventory. It does not execute packaged
Windows programs. Size checks do not establish numerical model compatibility.
Weights are external to this repository, its source licenses and CI artifacts.

## Run

Ensure `~/.local/bin` is on `PATH`, then check the device and real model:

```bash
dlsslopd --diagnose
dlsslopd --tier 720 --self-test --output neural-test.ppm
```

The self-test checks codec agreement, output sanity and repeated identical-input
inference. `--device N` selects a visible compatible HIP device. The worker finds
HIP modules under `share/dlsslop-amd/HIP/gfx1201` relative to its executable
prefix, with `assets/HIP/gfx1201` as the development fallback. Set
`DLSSLOP_MODULES` or pass `-m` / `--modules` to select another directory.
Start the worker after testing:

```bash
dlsslopd --tier 720
```

Wait for `worker ready`. With Proton GE selected, put this in the game's Steam
launch options, replacing the home-directory placeholder:

```text
/home/YOUR_USER/.local/bin/dlsslop-run -- %command%
```

The target is native Linux Steam with 64-bit Vulkan, DXVK or vkd3d-proton games.
Native Vulkan applications can also use the wrapper. OpenGL through WineD3D
does not use this layer.

From another terminal, use `dlsslop-gui` or the CLI:

```bash
dlsslopctl --status
dlsslopctl --passes 2 --color-preserve 1
dlsslopctl --enabled 0
dlsslopctl --enabled 1
dlsslopctl --quit
```

Use `--help` for options and defaults, and `--settings` for current/reset
values. See [CONTROL-OPTIONS.md](CONTROL-OPTIONS.md) and
[COLOR-PRESERVATION.md](COLOR-PRESERVATION.md). Start a fresh worker after
`--quit`. Removing the Steam launch option disables the integration for that
game. The native tools' default channel and launcher-selected layer log are
under `/tmp/dlsslop-amd-UID/`, where `UID` is the current user's numeric ID.
For separate concurrent sessions, give each worker, launcher and controller
the same distinct `DLSSNR_SHM` path inside a private directory.

The layer retains upstream `DLSSNR_*` environment names. Its shared runtime
fallback is `/tmp/dlssnr-UID/`, with `DLSSNR_UID` overriding that ID; the
native tools select their own default channel independently of `DLSSNR_UID`.

## Scope and diagnostics

`--tier` selects a 1280×720, 1600×900 or 1920×1080 neural raster; it does not
change the game's resolution. The edit is composed against the native frame.
Successive passes consume the previous output and add neural work; they do
not guarantee better quality. Motion estimation is optional and disabled by
default. HUDs are part of the image and can be altered. Capture, host staging
and inference are synchronous, so this path adds latency and competes with the
game for GPU time. It provides image transformation, not frame generation.

`--hold 1` freezes the input for comparisons; `--hold 0` resumes. `--capture 3`
requests matched before/after frames under `$XDG_STATE_HOME/dlssnr/captures`
(default `~/.local/state/dlssnr/captures`). Capture batches are preserved.
For raw per-pass color diagnosis, restart the worker with
`--trace-dir "$HOME/.local/state/dlsslop-amd/color-trace"`, then run:

```bash
dlsslop-test --trace-dir "$HOME/.local/state/dlsslop-amd/color-trace" \
    --output-dir color-report
```

The report directory must be new. The diagnostic holds one input, captures
comparison stages and restores changed controls. Do not adjust controls during
it. Raw tracing adds readbacks, disk use and timing overhead. Reports locate
color shifts. Normal inference never substitutes an identity filter;
`--test-identity` is an explicit transport-test mode. If no valid worker result
arrives, the layer presents the original frame.

When changing prepared upstream sources, use `scripts/update-source-patch.py`
to update the local patch and output hashes, then review both before committing.
Keep the pins and all license/attribution notices intact.
