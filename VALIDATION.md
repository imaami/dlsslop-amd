# Testing

Build the project as described in [README.md](README.md) before running these
checks. Automated host tests, GPU tests and game testing exercise different
parts of the integration.

## Automated tests

CTest covers CLI parsing and defaults, launchers, an identity worker's handling
of shared-memory requests, storage-image handling, CPU codec/tuning/temporal
behaviour, color preservation, trace serialization, capture writing,
color-diagnostic orchestration, and offline imports with filtered dependency
fetches and source preparation. With the GUI enabled it also checks the
shared-memory controller backend.

The presentation smoke drives Vulkan capture, an explicit identity worker,
composition and presentation through a separate test layer that also admits
software devices. It checks transport and composition, not HIP inference. Its
linear-HDR mode runs under the Khronos validation layer when that is installed.
It then stops the worker, and later kills one, and checks that presents neither
publish to a stopped worker nor wait more than a moment for a killed one.
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
skip until `scripts/build-kernels.py` has run, as does the color GPU test,
which also skips without a HIP runtime and `gfx1201` device.

Inspect the CTest result for skips. These checks do not establish neural image
quality. After the build described in README.md:

```bash
QT_QPA_PLATFORM=offscreen ctest --test-dir build --output-on-failure
```

## Hardware checks

With a compatible HIP runtime and `gfx1201` GPU, CTest's `color-gpu` test
checks color correction without model weights. To run it alone:

```bash
./build/color-gpu-test --module assets/HIP/gfx1201/linux_color.hsaco
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

Check that repeated raw FP32 outputs are bit-identical. Repeat with `--passes 2`
to exercise feedback. Successful execution and deterministic output are basic
sanity checks; the self-test does not measure visual quality or game performance.

## Game and desktop testing

Run the worker and launch the game using the instructions in README.md. Test on
the intended distribution, desktop session and Radeon device with the intended
model coefficients.

- Use held-frame comparisons and captures to inspect scene detail, color and
  HUD handling. Check both SDR and HDR where supported by the game and display.
- Compare moving scenes with motion estimation enabled and disabled. Check
  temporal stability and transitions between scenes.
- Measure latency and sustained frame times at each intended neural raster and
  pass count, including long sessions with the game's normal GPU load.
- Open the Qt controller in the desktop session and check rendering, input,
  connection, live controls and channel replacement behaviour.

Use [COLOR-PRESERVATION.md](COLOR-PRESERVATION.md) for the correction algorithm
and GPU test, and the color diagnostic in README.md to inspect per-pass changes.
