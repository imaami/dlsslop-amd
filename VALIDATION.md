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
the weight sizes the network expects against the model import's manifest. At
every tier and preset it checks the HIP network's launch plan against traces
of upstream's network: every launch with the buffers it uses, on the first
frame and later ones, the buffer pool, the uploaded weights, and the gather map
each ViT gather reads. It checks the tracing HIP runtime (see
[Tracing the HIP network](#tracing-the-hip-network)) against a fake runtime
line for line, and its tools, client and run script without a GPU. With the
GUI enabled it also checks the controller's option parsing without a display,
its shared-memory backend and slider, that an edit is written at once and later
ones coalesced, and that the wheel scrolls a page without editing the
unfocused controls it crosses.

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
ROCm's: it forwards every call to the real runtime and logs it.
`hiptrace-shmclient` is a minimal layer that serves dlsslopd frames on a
private channel. `trace.sh` runs one configuration with both and writes
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
