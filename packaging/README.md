# dlsslop-amd

DLSS Linux Open Proxy for AMD processes game frames through a native HIP worker
and a Vulkan presentation layer. This release targets Linux x86-64 and the AMD
RX 9070 XT (`gfx1201`). It includes the worker, controller, Qt GUI, game launcher,
color diagnostic, model importer and all GPU modules. Model weights are separate.

## Runtime requirements

On Debian Sid, keep your existing Mesa RADV installation and the normal `amdgpu`
kernel driver. Install compatible HIP userspace separately; the worker needs
`libamdhip64.so` and access to `/dev/kfd` and the GPU's render device. HIP userspace
must support `gfx1201`. Run the worker in a host terminal outside Steam's runtime.
If its loader cannot find your HIP installation, select the library explicitly:

```bash
export DLSSLOP_HIP_LIBRARY=/path/to/libamdhip64.so
```

The GUI uses your distribution's shared Qt 6 Core, Gui and Widgets libraries and
its Qt platform plugin. The runtime also needs the Vulkan loader, Bash, Python 3
and `flock`. The color diagnostic needs Python NumPy and Pillow. The worker and
Vulkan layer run independently of the GUI.

## Install

Extract the archive, enter its `dlsslop-amd` directory, and stop the game and worker
before installing or updating:

```bash
sha256sum --check PACKAGE-SHA256SUMS
python3 install.py
export PATH="$HOME/.local/bin:$PATH"
```

Add `~/.local/bin` to your usual shell's `PATH` if needed. The default prefix is
`~/.local`. The Vulkan manifest goes in `$XDG_DATA_HOME/vulkan/implicit_layer.d`,
or `~/.local/share/vulkan/implicit_layer.d` when `XDG_DATA_HOME` is unset. Use
`install.py --help` for prefix and manifest overrides. Installation includes the
GUI and creates the Vulkan manifest; the layer activates through `dlsslop-run`.

## Set up the model

Obtain compatible coefficients using the
[model project's instructions](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting).
Import your local ZIP or extracted coefficient directory:

```bash
dlsslop-setup --source /path/to/model-package.zip
```

The importer checks all 184 coefficient tables and their sizes. It imports only
coefficient files. The default model directory is `$XDG_DATA_HOME/dlsslop-amd/model`,
or `~/.local/share/dlsslop-amd/model`; `dlsslopd` uses the same default. To use a
different directory, pass `--output DIR` to the importer and `--assets DIR` to
the worker. The model's terms are separate from the software licenses.

## Start a game

Check the device, run inference on a sample image, and start the worker:

```bash
dlsslopd --diagnose
dlsslopd --tier 720 --self-test --output neural-test.ppm
dlsslopd --tier 720
```

Wait for `worker ready`. With Proton GE selected, put this in the game's Steam
launch options, replacing the home-directory placeholder:

```text
/home/YOUR_USER/.local/bin/dlsslop-run -- %command%
```

The launcher supports native Linux Steam with 64-bit Vulkan, DXVK or
vkd3d-proton games. For a native Vulkan application, use
`dlsslop-run -- /path/to/application`. OpenGL through WineD3D does not use the layer.
Remove the launch option to disable the integration for a game.

Use the GUI or command line from another terminal:

```bash
dlsslop-gui
dlsslopctl --status
dlsslopctl --passes 2 --color-preserve 1
dlsslopctl --enabled 0
dlsslopctl --enabled 1
dlsslopctl --quit
```

Start a fresh worker after `--quit`. Every command has `--help`; `dlsslopctl
--settings` lists the current settings and reset values. `--tier` selects the
neural raster (720, 900 or 1080), independently of the game's resolution. Extra
passes add GPU work and latency. HUDs are part of the image and may change.

The worker, launcher and controllers use `/tmp/dlsslop-amd-UID/shm.bin`, where
`UID` is your numeric user ID. The launcher writes the layer log beside that file.
To run separate sessions, give each session's worker, launcher and controllers
the same distinct `DLSSNR_SHM` path inside a private directory.

## Capture and color checks

`dlsslopctl --hold 1` freezes input; `--hold 0` resumes it. `--capture 3` requests
matched before/after frames under `$XDG_STATE_HOME/dlssnr/captures`, or
`~/.local/state/dlssnr/captures` by default.

For per-pass color checks, restart the worker with tracing and keep a game running:

```bash
dlsslopd --tier 720 --trace-dir "$HOME/.local/state/dlsslop-amd/color-trace"
```

From another terminal:

```bash
dlsslop-test --trace-dir "$HOME/.local/state/dlsslop-amd/color-trace" \
    --output-dir color-report
```

Choose a new report directory. The diagnostic temporarily holds input, captures
comparison stages and restores its changed controls. Do not adjust controls while
it runs. Tracing adds readbacks and disk use; omit `--trace-dir` for ordinary play.
