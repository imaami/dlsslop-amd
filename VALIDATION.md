# Testing

Build the project as described in [README.md](README.md) before running these
checks. Automated host tests, GPU tests and game testing exercise different
parts of the integration.

## Automated tests

CTest covers CLI parsing and defaults, launchers, storage-image handling, CPU
codec/tuning/temporal behaviour, color preservation, trace serialization,
capture writing and color-diagnostic orchestration. With the GUI enabled it
also checks the shared-memory controller backend. The HDR shader test uses
software Vulkan and can skip when no suitable device is available; inspect the
CTest result for skips. These checks do not establish neural image quality.

After the build described in README.md:

```bash
QT_QPA_PLATFORM=offscreen ctest --test-dir build --output-on-failure
python3 tests/run-smoke.py --build-dir build --headless
```

The optional presentation smoke uses a separate software-device test layer and
an explicit identity worker. It checks transport and composition, not HIP
inference.

## Hardware checks

With a compatible HIP runtime and `gfx1201` GPU, test color correction without
model weights:

```bash
./build/color-gpu-test --module assets/HIP/gfx1201/linux_color.hsaco
```

Exit 77 means the runtime/device is unavailable, not a passing hardware test.
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
