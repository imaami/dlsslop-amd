# dlsslop-amd

DLSS Linux Open Proxy for AMD processes game frames through a native HIP
daemon and a Vulkan presentation layer. This release targets Linux x86-64
and the AMD RX 9070 XT (`gfx1201`). It includes the daemon, controller, Qt
GUI, game launcher, color diagnostic, model importer and all GPU modules.
Model weights are separate.

## Runtime requirements

On Debian Sid, or any other reasonably up-to-date Linux system, keep your
existing Mesa RADV installation and the normal `amdgpu` kernel driver. By
default `dlsslopd` runs the network on Vulkan, which needs nothing else: RADV
from Mesa 26.2 or newer provides the FP8 cooperative matrices it uses
(`VK_KHR_cooperative_matrix`, `VK_EXT_shader_float8`). The daemon runs on the
host, under systemd or in a terminal, never inside Steam's runtime.

The HIP backend (`dlsslopd --backend hip`) is the alternative. It needs
compatible HIP userspace installed separately: `libamdhip64.so` supporting
`gfx1201`, and access to `/dev/kfd` and the GPU's render device. If its loader
cannot find your HIP installation, select the library explicitly:

```bash
export DLSSLOP_HIP_LIBRARY=/path/to/libamdhip64.so
```

The GUI uses your distribution's shared Qt 6 Core, Gui, and Widgets libraries,
and its Qt platform plugin. The runtime also needs the Vulkan loader, `bash`,
Python 3.11 or newer, and `flock`. The color diagnostic needs Python NumPy and
Pillow. `dlsslopd` and the Vulkan layer both run independently of the GUI.

## Install

The release archive's directory structure is that of an install prefix,
and installing it simply means extracting the archive at the destination and
running a couple of `systemctl` commands. To update, follow the same steps with
the new archive.

1. Stop `dlsslopd` and any programs using it.

   ```bash
   systemctl --user stop 'dlsslop.s*'
   pkill -x dlsslopd
   ```

2. Extract the archive to the install prefix.

   ```bash
   tar -xf dlsslop-amd-linux-gfx1201.tar.xz -C ~/.local
   ```

3. Validate the extracted files by comparing
   their checksums to those in `SHA256SUMS`.

   ```bash
   (cd ~/.local && sha256sum --check share/doc/dlsslop-amd/SHA256SUMS)
   ```

4. Make the systemd units findable, unless the prefix is `~/.local` or
   `/usr/local`, where systemd already looks. For `/opt/dlsslop-amd`, link both:

   ```bash
   systemctl --user link /opt/dlsslop-amd/share/systemd/user/dlsslop.{socket,service}
   ```

   For any other prefix, link `dlsslop.socket` the same way, then copy
   `dlsslop.service` into `~/.config/systemd/user/` and point its `ExecStart` at
   that prefix's `bin/dlsslopd`: the service looks for `dlsslopd` only under
   `~/.local`, `/usr/local` and `/opt/dlsslop-amd`.

5. Load the units and enable the socket, so that systemd starts the daemon
   whenever a game needs it:

   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now dlsslop.socket
   ```

   Run `systemctl --user daemon-reload` again after any later edit to a copied
   `dlsslop.service`: systemd otherwise keeps a service it already loaded as it
   was. To enable the socket for every user of a `/usr/local` install, run
   `sudo systemctl --global enable dlsslop.socket` instead: each user's own
   systemd instance starts the socket at their next login.

Add the prefix's `bin/` to your usual shell's `PATH` if needed. `~/.local`
suits one user; for everyone, extract into `/usr/local`, or into a directory
of its own such as `/opt/dlsslop-amd`. The daemon finds its GPU modules, and
the launcher its Vulkan layer manifest (`share/vulkan/implicit_layer.d`),
relative to their own location, so the tree works from any prefix; the layer
activates only through `dlsslop-run`. Steam's runtime container, which runs
Proton games, reads layer manifests only from the Vulkan loader's standard
directories: for those games, extract into `~/.local` or `/usr/local`. The
archive holds no directories of its own, so extracting to `~/.local` changes
nothing but these files.

To uninstall, stop the units, then disable them, which also removes the links
step 4 made:

```bash
systemctl --user stop dlsslop.socket dlsslop.service
systemctl --user disable dlsslop.socket dlsslop.service
```

Delete a `dlsslop.service` you copied into `~/.config/systemd/user/`, and for a
`/usr/local` install enabled for every user, run
`sudo systemctl --global disable dlsslop.socket`. Then delete the files
`share/doc/dlsslop-amd/SHA256SUMS` lists, and finally that `SHA256SUMS` file
itself.

## Set up the model

The model is NVIDIA's and is not included. The Vulkan network reads it from your
own copy of `nvngx_dlssnr.dll`, which must be version 310.8.0, or a ZIP holding
it:

```bash
dlsslop-setup --dll /path/to/nvngx_dlssnr_310.8.0.zip
```

The extractor reads the DLL's weight data without running it, checks all 599
entries against the model the network was built for, and only then writes
`$XDG_DATA_HOME/dlsslop-amd/dlssnr.bin`, or `~/.local/share/dlsslop-amd/dlssnr.bin`,
where `dlsslopd` looks by default; `--vulkan-model FILE` selects another file for
both. A DLL of any other version is refused. `dlsslopd --backend auto`, the
default, runs the network on Vulkan once this model is there.

For the HIP backend, obtain its coefficients using the
[HIP project's instructions](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting)
and import your local ZIP or extracted coefficient directory:

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
`--output DIR` to the importer and `--assets DIR` to the daemon. The model's
terms are separate from the software licenses.

The two backends implement the same network separately, and their pictures
differ noticeably; neither is NVIDIA's. The Vulkan network runs about two to
three times faster. Its intensity, local tone and local structure settings are
NVIDIA's own model controls, not the HIP backend's residual filters. It also
has the model's style, skin structure and automatic mask, which the HIP backend
ignores.

## Start a game

Check the device and run inference on a sample image:

```bash
dlsslopd --diagnose
dlsslopd --tier 720 --self-test --output neural-test.ppm
```

The socket enabled in step 5 of the installation listens beside the default
channel, `/tmp/dlsslop-amd-UID/shm.bin`; nothing else runs. When no daemon serves, `dlsslop-run` connects there, which
starts `dlsslop.service`, and waits until the daemon serves. The daemon stops
after ten seconds without a frame, and the layer starts it again if a game is
still running. Its log is in `journalctl --user -u dlsslop`. A channel chosen
with `DLSSNR_SHM` or `--shm` is not started on demand: start its daemon by hand.
Without systemd, run `dlsslopd` in a host terminal and wait for `worker ready`.

However it starts, the daemon reads its settings from
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

`--quit` stops the daemon, and a running game presents its own frames until a
daemon starts again: at the next launch through `dlsslop-run`, with
`systemctl --user start dlsslop.service`, or by hand. `systemctl --user stop
dlsslop.socket dlsslop.service` also ends on-demand starts until the next login.

Every command has `--help`; `dlsslopctl --settings` lists the current settings
and reset values. `--tier` switches the running daemon to a 1280×720, 1600×900
or 1920×1080 neural raster, independently of the game's resolution; the edit
is composed against the native frame. The daemon rebuilds its network, which
takes a few seconds while the game presents its own frames. A restarted daemon
keeps the tier unless its command line or config file sets one. An extra pass
consumes its predecessor's output, uses more compute and adds latency without
guaranteeing better quality. Motion estimation is optional and off by default.

In-game menus and HUDs receive no special treatment: dlsslop-amd sees only
finished frames, so it cannot tell which pixels belong to them. No input pixel
is exempt from neural processing.

Capture, transport and inference are synchronous, so the daemon adds latency
and competes with the game for GPU time. When a game renders on the daemon's
GPU, frames cross between the layer and the daemon in VRAM; otherwise frames
are staged through host memory. `dlsslopd` doesn't generate new intermediate
frames.

### The network in the layer (experimental)

`dlsslop-run --layer-network -- %command%` runs the Vulkan network inside the
layer, on the game's own device, instead of in the daemon. Each frame is then
one GPU submission, capture, network and composition, and the game's present
returns without waiting for it. It needs a Vulkan 1.3 game (DXVK and
vkd3d-proton games are) on a device that runs the network, and the daemon's
model: the one `vulkan-model` in `dlsslopd.conf` names, else the default. A
`--vulkan-model` on `dlsslopd`'s command line is the daemon's alone. `dlsslopctl
--status` reports it as `layer_reason=in-layer network running`, and the GUI's
snapshot as its Layer line. While it builds, at the start and after a tier
change, the game presents its own frames. If it cannot run, the log says why and
frames go to the daemon as before, so keep the daemon available. It adds the
network's features to the game's device and runs in the game's process: remove
the option if a game misbehaves with it.

The daemon, launcher and controllers use `/tmp/dlsslop-amd-UID/shm.bin`, where
`UID` is your numeric user ID. The launcher writes the layer log beside that
file. To run separate sessions, give each session's daemon, launcher and
controllers the same distinct `DLSSNR_SHM` path inside a private directory.

## Capture and color checks

`dlsslopctl --hold 1` freezes input; `--hold 0` resumes it. `--capture 3` requests
matched before/after frames under `$XDG_STATE_HOME/dlssnr/captures`, or
`~/.local/state/dlssnr/captures` by default. Each request writes a new batch
directory; earlier batches are kept.

For per-pass color checks, restart the daemon with tracing and keep a game running.
Only the HIP network traces, so with `--trace-dir` the daemon runs HIP:

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
