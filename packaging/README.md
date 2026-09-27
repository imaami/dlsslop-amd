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
its Qt platform plugin. The runtime also needs the Vulkan loader, Bash, Python 3.11
or newer and `flock`. The color diagnostic needs Python NumPy and Pillow. The worker
and Vulkan layer run independently of the GUI.

## Install

The archive's root is an installation prefix. Stop the game and worker, extract
it into the prefix of your choice and check it there:

```bash
tar -xf dlsslop-amd-linux-gfx1201.tar.xz -C ~/.local
(cd ~/.local && sha256sum --check share/doc/dlsslop-amd/SHA256SUMS)
export PATH="$HOME/.local/bin:$PATH"
```

Add the prefix's `bin/` to your usual shell's `PATH` if needed. `~/.local` suits
one user; for everyone, extract into `/usr/local`, or into a directory of its own
such as `/opt/dlsslop-amd`. The worker finds its GPU modules, and the launcher its
Vulkan layer manifest (`share/vulkan/implicit_layer.d`), relative to their own
location, so the tree works from any prefix; the layer activates only through
`dlsslop-run`. Steam's runtime container, which runs Proton games, reads layer
manifests only from the Vulkan loader's standard directories: for those games,
extract into `~/.local` or `/usr/local`. The archive holds no directories of its own, so extracting over
`~/.local` changes nothing but these files. To update, extract a new archive over
the old one; to uninstall, delete the files `share/doc/dlsslop-amd/SHA256SUMS`
lists.

## Set up the model

Obtain compatible coefficients using the
[model project's instructions](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting).
Import your local ZIP or extracted coefficient directory:

```bash
dlsslop-setup --source /path/to/model-package.zip
```

The importer checks all 184 coefficient tables and their sizes, imports only
coefficient files and writes a SHA-256 inventory. It does not run packaged
Windows programs, and size checks do not prove numerical compatibility.
`dlsslop-setup --check DIR` checks an imported model's sizes and inventory
again, for example after a disk error. The default model directory is
`$XDG_DATA_HOME/dlsslop-amd/model`, or `~/.local/share/dlsslop-amd/model`;
`dlsslopd` uses the same default. To use a different directory, pass
`--output DIR` to the importer and `--assets DIR` to the worker. The model's
terms are separate from the software licenses.

## Start a game

Check the device and run inference on a sample image:

```bash
dlsslopd --diagnose
dlsslopd --tier 720 --self-test --output neural-test.ppm
```

Then let systemd start the worker whenever a game needs it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now dlsslop.socket
```

The socket listens beside the default channel, `/tmp/dlsslop-amd-UID/shm.bin`;
nothing else runs. When no worker serves, `dlsslop-run` connects there, which
starts `dlsslop.service`, and waits until the worker serves. The worker stops
after ten seconds without a frame, and the layer starts it again if a game is
still running. Its log is in `journalctl --user -u dlsslop`. A channel chosen
with `DLSSNR_SHM` or `--shm` is not started on demand: start its worker by hand.

For every user, extract into `/usr/local` and run
`sudo systemctl --global enable dlsslop.socket`: each user's own systemd
instance starts the socket at their next login. With `/opt/dlsslop-amd`, first
link both units from its `share/systemd/user` with `systemctl --user link`. The
service finds `dlsslopd` under `~/.local`, `/usr/local` or `/opt/dlsslop-amd`; for
another prefix, copy `dlsslop.service` into `~/.config/systemd/user/` and point its
`ExecStart` at that prefix's `bin/dlsslopd`. Without
systemd, run `dlsslopd` in a host terminal and wait for `worker ready`.

However it starts, the worker reads its settings from
`~/.config/dlsslop-amd/dlsslopd.conf`, or from `dlsslop-amd/dlsslopd.conf` under
`$XDG_CONFIG_HOME` when that is set; `--config FILE` names another file. An
option on its command line overrides the file. `dlsslopd --help` lists the
settings:

```ini
# ~/.config/dlsslop-amd/dlsslopd.conf
tier = 1080
passes = 2
```

With Proton GE selected, put this in the game's Steam launch options, replacing
the home-directory placeholder:

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
dlsslopctl --tier 1080
dlsslopctl --enabled 0
dlsslopctl --enabled 1
dlsslopctl --quit
```

`--quit` stops the worker, and a running game presents its own frames until a
worker starts again: at the next launch through `dlsslop-run`, with
`systemctl --user start dlsslop.service`, or by hand. `systemctl --user stop
dlsslop.socket dlsslop.service` also ends on-demand starts until the next login.

Every command has `--help`; `dlsslopctl
--settings` lists the current settings and reset values. `--tier` switches the
running worker to a 1280×720, 1600×900 or 1920×1080 neural raster,
independently of the game's resolution; the edit is composed against the native
frame. The worker rebuilds its network, which takes a few seconds while the game
presents its own frames. A restarted worker keeps the tier unless its command
line or config file sets one. Each extra pass
consumes the previous output and adds GPU work and latency without guaranteeing
better quality. Motion estimation is optional and off by default. HUDs are part
of the image and may change. Capture, transport and inference are
synchronous, so the worker adds latency and competes with the game for GPU
time. When the game renders on the worker's GPU, frames cross between the
layer and the worker in video memory; otherwise they are staged through host
memory. It transforms images; it does not generate frames.

The worker, launcher and controllers use `/tmp/dlsslop-amd-UID/shm.bin`, where
`UID` is your numeric user ID. The launcher writes the layer log beside that file.
To run separate sessions, give each session's worker, launcher and controllers
the same distinct `DLSSNR_SHM` path inside a private directory.

## Capture and color checks

`dlsslopctl --hold 1` freezes input; `--hold 0` resumes it. `--capture 3` requests
matched before/after frames under `$XDG_STATE_HOME/dlssnr/captures`, or
`~/.local/state/dlssnr/captures` by default. Each request writes a new batch
directory; earlier batches are kept.

For per-pass color checks, restart the worker with tracing and keep a game running:

```bash
dlsslopd --trace-dir "$HOME/.local/state/dlsslop-amd/color-trace"
```

From another terminal:

```bash
dlsslop-test --trace-dir "$HOME/.local/state/dlsslop-amd/color-trace" \
    --output-dir color-report
```

Choose a new report directory. The diagnostic temporarily holds input, captures
comparison stages, moves its captures and traces into the report and restores
its changed controls. Do not adjust controls while it runs. Tracing adds
readbacks and disk use; omit `--trace-dir` for ordinary play.
