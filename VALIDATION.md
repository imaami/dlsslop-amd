# Testing

Build the project as described in [README.md](README.md) before running these
checks. Automated host tests, GPU tests and game testing exercise different
parts of the integration.

## Automated tests

CTest covers CLI parsing and defaults, launchers, the worker's signal handling
and its `--diagnose` device selection against a fake HIP runtime, an identity
worker's handling of shared-memory requests, storage-image handling, CPU
codec/tuning/temporal behaviour, color preservation, trace serialization,
capture writing, color-diagnostic orchestration, offline imports with filtered
dependency fetches and source preparation, the kernel build's compiler
selection, and the model import's SHA-256 inventory, which
`dlsslop-setup --check` verifies. It checks the HIP network's weight packing
byte for byte against digests of upstream's packers on synthetic weights, and
the weight sizes the network expects against the model import's manifest. It
checks the Vulkan network's weight packing byte for byte against digests of
what upstream's graph build packed from a synthetic model pack, and its model
reader on damaged packs. At 40 frame extents from 17x17 to 5120x2880 it checks
the Vulkan network's launch plan against upstream's graph build: every dispatch
with its push constants and the activation arena's values and sizes, and at 8
of them the weight blob packed from a synthetic model pack. Frames that
upstream fails on, and frames whose arena overflows 32-bit offsets, must be
rejected. The Vulkan network's digests and goldens were recorded from the
DLSSNR-AMD fork's host code at `3dfdddc`, which the build neither fetches nor
compiles. CTest also checks each network kernel's push block and bindings in
the built SPIR-V against the plan, and the shader build's markers and the model
tools' entry list against what the plan expects. It checks that two writers
replacing the Vulkan network's pipeline cache at once, as dlsslopd and a game's
in-layer network can, never leave a reader without the file or with part of
one. At every tier and preset it checks the HIP network's launch plan against
traces of upstream's network: every launch with the buffers it uses, on the
first frame and later ones, the buffer pool, the uploaded weights, and the
gather map each ViT gather reads. It checks the tracing HIP runtime (see
[Tracing the HIP network](#tracing-the-hip-network)) against a fake runtime
line for line, and its tools and run script without a GPU. It checks the
tracing Vulkan layer (see
[Tracing the Vulkan network](#tracing-the-vulkan-network)) with a probe on
Mesa's lavapipe under the Khronos validation layer, and its tools and run
script; without lavapipe or the validation layer it skips the probe. The
tracers' fake layer, `shmclient`, must send dlsslopd's identity mode the
inputs the tracers' baselines were taken with, after storing the settings that
the daemon then finds in the channel. With the GUI enabled it also
checks the controller's option parsing without a display, its shared-memory
backend and slider, that an edit is written at once and later ones coalesced,
and that the wheel scrolls a page without editing the unfocused controls it
crosses.

The presentation smoke drives Vulkan capture, an explicit identity worker,
composition and presentation through a separate test layer that also admits
software devices. It checks transport and composition, not HIP inference.
Every mode runs under the Khronos validation layer (`vulkan-validationlayers`)
with synchronization validation and fails on any validation error; the smoke
skips when that layer is not installed. It then stops the worker, and later
kills one, and checks that presents neither publish to a stopped worker nor wait
more than a moment for a killed one. Each mode then destroys its device and
fails if the layer still holds a descriptor or mapping of the channel or its
producer lock.
The composition rebuild test changes the layer's model raster, colour domain
and downscaler in place and fails when a dispatch binds a descriptor written for
an image view that has since been destroyed. The smoke, this test and the HDR
shader test prefer software Vulkan (Mesa lavapipe) and fall back to a hardware
device; the latter two skip when no suitable device is available.

Two checks disassemble the built HIP modules and need `llvm-objdump`. The
LDS-barrier check fails when a shared-memory access can still be outstanding at
a workgroup barrier in any kernel; with `clang++-22` it first proves itself on
a deliberately unfenced probe kernel. The codec ISA check counts the binary16
conversions in the GPU codec, which a known compiler substitution breaks. Both
skip until `scripts/build-kernels.py` has run, as do the check of the HIP
network's launches against its kernels' metadata (their arguments and group
sizes) and the color GPU test, which also skips without a HIP runtime and
`gfx1201` device.

Inspect the CTest result for skips. These checks do not establish neural image
quality. After the build described in README.md:

```bash
ctest --test-dir build --output-on-failure
```

## Hardware checks

With a compatible HIP runtime and `gfx1201` GPU, CTest's `color-gpu` test
checks color correction without model weights. To run it alone:

```bash
./build/color-gpu-test --module assets/HIP/gfx1201/linux_native.hsaco
```

Exit 77 means the module, runtime or device is unavailable, not a passing
hardware test.

Software Vulkan accepts descriptor pools that the Radeon driver rejects, so
repeat the presentation smoke on RADV. It fails when the layer cannot build a
composition, including the linear-HDR one used by HDR swapchains:

```bash
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json python3 tests/run-smoke.py --headless
```

After installation and external weight import, run:

```bash
dlsslopd --diagnose
dlsslopd --tier 1080 --self-test --self-test-runs 20 --output neural-test.ppm
```

With the Vulkan model installed, the self-test runs the Vulkan network: every run
must reproduce the first byte for byte and change the image. Add `--backend hip`
for the HIP network, whose first run checks the GPU codec bit for bit against the
CPU reference. Every later run must reproduce its raw FP32 answer and decoded
output exactly, and the summary reports the last run's timings, which exclude the
reference checks.
Repeat with `--passes 2` to exercise feedback. Successful execution and
deterministic output are basic sanity checks; the self-test does not measure
visual quality or game performance.

Without a GPU, the Vulkan network's weight blob can be checked against
upstream's packing of the real model at every extent the plan test covers:

```bash
./build/vulkan-plan-test --model ~/.local/share/dlsslop-amd/dlssnr.bin
```

`vulkan-plan-test --print WxH` prints the plan for WxH frames, a line a
dispatch and a line a value, in the form of the dumps of upstream's build that
its goldens were recorded from.

When the game renders on the worker's RX 9070 XT, its layer log (beside the
channel file) must report `device-local transport ready`: proxy and answer then
stay in video memory. A game on another GPU, or a worker whose channel path is
too long for a Unix socket, logs `device-local transport declined` and stages
frames through host memory instead.

To check the network in the layer, run a Vulkan 1.3 application on the RX 9070
XT through `dlsslop-run --layer-network`. Its layer log must report `in-layer
network features enabled`, `in-layer network on the game's device` and `in-layer
network running`, and no `in-layer network off`. At the same tier, captures
(`dlsslopctl --capture 8`) of the same input must match those taken without
`--layer-network` byte for byte. With `libdlsslop-network.so` moved away from
beside the layer, the log must say the module is unavailable and frames must go
to the daemon. A Vulkan 1.1 application logs `in-layer network unavailable:
needs a Vulkan 1.3 instance`.

### Tracing the HIP network

A change to the HIP network's host code must leave its launches as they were,
unless it means to change them. `tests/hiptrace` checks this on the GPU. The
build's `libhiptrace.so` is a HIP runtime that dlsslopd loads in place of
ROCm's: it forwards every call to the real runtime and logs it. `shmclient`,
from `tests/shmclient.cpp`, is a minimal layer that serves dlsslopd frames on
a private channel. `trace.sh` runs one configuration with both and writes
`NAME.trace`:

```bash
tests/hiptrace/trace.sh --output traces --self-test   # traces/selftest-720.trace
tests/hiptrace/trace.sh --output traces               # traces/serve-720.trace: frames A, A, B
tests/hiptrace/trace.sh --output traces --tier 1080 --performance --motion --frames AABB
```

`--help` lists the options and their defaults: the build tree, the dlsslopd,
module and model directories, the output and channel directories, and the
configuration. dlsslopd reads no settings file.

To compare two builds, trace the same configuration with each and compare
the traces, each with the modules its revision builds. `--daemon` may name any
dlsslopd, including one built before 24c374d, which runs upstream's host code:
the tracing runtime also exports the entry points that code needs.

```bash
tests/hiptrace/trace.sh --output traces/old --daemon ../old/build/dlsslopd --modules ../old/assets/HIP/gfx1201
tests/hiptrace/trace.sh --output traces/new
tests/hiptrace/compare.py traces/old/serve-720.trace traces/new/serve-720.trace
```

Builds before e870ce6 also upload 32 weights (29 with `--performance`) that no
launch reads. Against such a build as A, `compare.py` reports those 32 (or 29)
payloads as only in A and prints `DIFFERENT`; the builds agree when the report
lists no other difference.

A network evaluation starts at every launch of the first kernel that is not in
`linux_native.hsaco`. `compare.py` prints `MATCH` when both traces have network
evaluations, the same number of them, and every evaluation has the same
launches in the same order. Two launches are the same when module, kernel,
grid, block, dynamic LDS and every argument are equal. Scalar arguments must
have equal bytes. A buffer argument is compared by the buffer's order of first
use in its evaluation and the byte offset, so two builds may allocate
differently but must alias buffers alike. The synchronous host-to-device copies
must also carry the same payloads (sizes and FNV-1a 64) in the same order; the
same payloads in another order are reported but pass. Otherwise it prints which
trace has no evaluation, each evaluation's first differing launch, or how the
uploads differ, then `DIFFERENT`, and exits with 1. Allocation sizes, streams,
timings and the daemon's own kernels are not compared. The first line of a
trace names the real runtime, the module directory and the hashing options, so
it differs between machines and builds.

Deep hashes also compare the data. After each launch they select, the tracing
runtime waits for the stream and logs, per buffer argument, the FNV-1a 64 of
the bytes from the argument to the end of its allocation (`uploaded` for
buffers that received a host-to-device copy, which it does not hash). These
waits and copies go to the real runtime directly and are not logged, so a deep
trace has the same calls; only the run is slower. `trace.sh
--deep-kernels c32_post_merge_head_half` hashes the network's raw FP32 output
at every evaluation, and `compare.py` then also compares the outputs of the
evaluations both traces hashed. `--deep` takes comma-separated ranges `A-B`
and ordinals `A` of launches (`n=` in the trace): `--deep 0-10,20` hashes after
launches 0 to 9 and after launch 20. `--deep all` hashes after every launch.

`analyze.py NAME.trace` writes a summary of calls, evaluations, frames,
kernels, allocations and uploads next to the trace. `plan_hashes.py` prints,
from a served trace, the values that `hip-plan-test` holds from served traces
of upstream's network. `plan_hashes.py --print 1` prints the second
evaluation's launches as `hip-plan-test --print TIER` prints the plan's, so the
two can be compared with `diff`. The first evaluation, `--print 0`, compares
with `hip-plan-test --print TIER --first`. For a trace taken with
`--performance`, add `--performance` to hip-plan-test; the third evaluation of
a trace taken with `--motion`, `--print 2`, compares with `--history`.

### Tracing the Vulkan network

A change to the Vulkan network's host code must leave its command streams as
they were, unless it means to change them. `tests/vktrace` checks this on the
GPU. The build's `libvktrace.so` is an explicit Vulkan layer,
`VK_LAYER_LOCAL_vktrace`, with its manifest in `build/vktrace`. It logs the
device work of dlsslopd with every handle renamed by kind and creation order:
the device and every feature it enables, memory, resources and their binds,
pipelines with the SPIR-V file each shader module holds, descriptor writes,
every recorded command with its push constant bytes and barriers, and every
submission and wait. It also logs the FNV-1a 64 of each copy out of mapped host
memory when it is submitted (`Upload`) and of each copy into host memory once
it has been waited for (`Readback`). `trace.sh` runs one configuration with
the layer and `shmclient` and writes `NAME.trace`:

```bash
tests/vktrace/trace.sh --output traces --self-test   # traces/selftest-720.trace and .ppm
tests/vktrace/trace.sh --output traces               # traces/serve-720.trace: frames A, A, B
tests/vktrace/trace.sh --output traces --tier 1080 --fp16 --motion --frames AABB
tests/vktrace/trace.sh --output traces --name serve-720-live --frames AAAAAA -- \
    --sharpness 1:0.5 --sharpness 2:0 --passes 3:2 --intensity 4:0.5 --mvec 5:1
```

`--help` lists the options and their defaults: the build tree, the dlsslopd,
the SPIR-V directories and the model, the output and channel directories, the
configuration and the content hashes. dlsslopd reads no settings file and keeps
its pipeline cache in `XDG_CACHE_HOME`. Arguments after `--` go to
`shmclient`, whose setting options are dlsslopctl's, with the same ranges; each
takes a value, stored before the daemon starts, or `FRAME:VALUE`, stored before
frame FRAME (from 0) is sent. dlsslopd follows a tier only between requests, so
for `--tier FRAME:VALUE` the client waits until the daemon has published the
new tier's raster before it sends frame FRAME. The layer goes first in
`VK_INSTANCE_LAYERS`, above any layer the environment already names there, so
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` also validates the layer's own
commands.

To compare two builds, trace the same configuration with each, each with the
SPIR-V its revision builds, and compare the traces:

```bash
tests/vktrace/trace.sh --output traces/old --daemon ../old/build/dlsslopd --spirv ../old/build/vulkan-nr/network
tests/vktrace/trace.sh --output traces/new
tests/vktrace/compare.py traces/old/serve-720.trace traces/new/serve-720.trace
```

`--daemon` may name any dlsslopd, including one that runs the fork's host
code, such as one built from the revision before "vulkan: run the network with
the project's own host code". Such a dlsslopd, and any built before "vulkan:
stop timing the network with queries nobody reads", also keeps upstream's
timing ring: an 8-query timestamp pool per build and, in every frame, a query
pool reset and two timestamps. Compare it with `--ignore-queries`, which leaves
the ring out; without that option, the comparison above finds every frame
different and prints `DIFFERENT`. Any dlsslopd built before "vulkan: stop
uploading an input that the first frame overwrites" also fills the network's
input with a gradient in each build whose input is RGBA8 or, with more than one
pass, RGBA32F. Its setup uploads then hold one more payload per such build, of
the frame's size in the input's format, so the comparison prints `setup uploads
differ` and `DIFFERENT` even when every frame matches; `--skip-setup` compares
the frames alone. With `--hash all`, the state after setup also differs in the
input, which newer builds leave unwritten. Any dlsslopd built before "vulkan:
create the finest flow level for sampling" creates the motion estimate's finest
flow level, an R32G32_SFLOAT image of a quarter of the frame's width and height,
without sampled usage. Every frame of a build with motion history then has the
same commands but differs in that image's usage, and the comparison prints
`DIFFERENT`.

A frame is a submission that dispatches and is not a build's one-shot, whose
command pool is created after the previous submission and destroyed before the
next. `compare.py` prints `MATCH` when both traces have the same number of
frames and each pair of frames has the same canonical command stream: every
dispatch with its pipeline's definition (SPIR-V, entry point, specialization
and layout), the contents of the descriptor sets it binds and its push constant
bytes; every barrier, copy, fill, update, clear, query and timestamp. Resources
are renamed by first use in the frame and described there (size, usage,
format, extent), and pipelines are named by their definitions, so two builds
may create objects in another order. The frames' uploads and readbacks must
have equal FNVs. Before the frames, the frames' pipelines must have the same
definitions, and the uploads outside frames must carry the same payloads;
the same payloads in another order are reported but pass. Otherwise it prints
each frame's first difference with the commands before it, and for a dispatch
which part differs (pipeline, grid, sets or push constant words), then
`DIFFERENT`, and exits with 1. `--ignore-queries` leaves query pools, their
resets, queries and timestamps out, `--frames 2-5:0-3` compares frames of A
with other frames of B, and `--span network` compares only the frames that run
the network, from its first dispatch to its last, and the answer that follows;
with `--skip-setup` it compares a game's trace of the in-layer network with
dlsslopd's. The first line of a trace names the hashing options and the SPIR-V
directories, so it differs between builds.

Content hashes also compare the data on the device. `--hash` takes
comma-separated selectors: `i2b` the bytes each image-to-buffer copy wrote,
such as the answer; `copydst` every copy, fill, update and clear destination;
`storage` every storage buffer range and storage image a dispatch bound;
`dispatch=N` those of a submission's dispatch N; `all` every buffer and image;
`bufN` and `imgN` those resources. After each submission they select, which
`--hash-submits` may limit to ranges `A-B` and ordinals `A` of the 1-based
submission numbers (`sub=` in the trace) or to `dispatch` for submissions that
dispatch, the layer waits for the queue, copies the selected resources through
its own command buffer and staging buffer and logs their FNV-1a 64 as `Hash`
lines. It logs none of its own commands, so a hashed trace has the same calls
and frames; only the run is slower. Submission numbers count a build's
submissions too, so a range that selects a baseline's build and frames does not
select the same work in a build that submits its setup another way; `dispatch`
does. `compare.py` compares the hashes of each selector kind as multisets of
size and FNV, which do not depend on the order or names of the resources: in
every frame, and in the state after setup. That state is, for each region
hashed in a submission before the first frame that is not a frame itself, its
last hash there, unless its resource is destroyed before the first frame, so a
build's staging buffers do not count however the build splits its
submissions. A kind that both traces' `--hash` selectors name must be hashed in
both or in neither, in the state after setup and in each frame; a kind only one
of them names is counted and left out. With `--hash all`, that state holds the
weights with their noise field, the zeroed activation arena and the images
after the network's build.
Past the network's values, the activation arena ends in counters the kernels
synchronise through. After a frame, they differ from one run of the same build
to the next while the answers stay equal, so each frame's hashes of `all`
differ in the arena between any two runs; the state after setup does not.
`--no-host` leaves the frames' uploads, readbacks and content hashes out and
still compares the state after setup.

`analyze.py NAME.trace` writes, next to the trace, a summary of the device,
builds, frames and answers, each frame's counts, the first and last frame of
each build in canonical form, what changes between consecutive frames, and
every allocation, pipeline and upload.

## Game and desktop testing

Run the worker and launch the game using the instructions in
[packaging/README.md](packaging/README.md). Test on
the intended distribution, desktop session and Radeon device with the intended
model coefficients.

- Use held-frame comparisons and captures to inspect scene detail, color and
  HUD handling. Check both SDR and HDR where supported by the game and display.
- Compare moving scenes with motion estimation enabled and disabled. Check
  temporal stability and transitions between scenes.
- Measure latency and sustained frame times at each intended neural raster and
  pass count, including long sessions with the game's normal GPU load.
- Switch tiers with `dlsslopctl --tier` during play. The game must keep
  presenting its own frames through the rebuild, then show the edit at the new
  raster, with `device-local transport ready` logged again.
- With `--layer-network`, repeat for a DXVK and a vkd3d-proton game. The layer
  log must report the network in the layer, frame pacing must stay even, and a
  tier switch must rebuild it while the game presents its own frames.
- Open the Qt controller in the desktop session and check rendering, input,
  connection, live controls and channel replacement behaviour.

Use [COLOR-PRESERVATION.md](COLOR-PRESERVATION.md) for the correction algorithm
and GPU test, and the color diagnostic in packaging/README.md to inspect
per-pass changes.
