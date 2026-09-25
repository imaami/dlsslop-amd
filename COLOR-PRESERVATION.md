# Per-pass color preservation

`-L`, `--color-preserve 0..1` defaults to **0 (off)**. The Qt controller exposes
it under **Neural passes**. All components must use shared-memory protocol 23.
After building and installing the matching worker, layer and controls:

```bash
dlsslopctl --passes 1 --color-preserve 1
dlsslopctl --color-preserve 0
```

The first command enables full correction; the second disables it. This setting
operates after every neural pass, before temporal-history storage and feedback.
`--color` is a separate final-composition control.

## Algorithm

The original encoded input remains the reference for the entire frame. After
native intensity/tone/structure/sharpness processing, the worker smooths the
pass output's difference from that reference with a 3×3 binomial filter and
removes the selected fraction of its chroma component. Proxy-space luma uses
weights (0.2126, 0.7152, 0.0722); it is not linear-light luminance. Filtering
clamps to the fitted viewport and leaves padded pixels unchanged.

Brightness and achromatic detail remain. Uniform color shifts are removed at
full strength, but high-frequency chroma changes can remain. Chroma is
compressed toward gray when needed to keep channels in [0,1] if luma is already
in that range. Out-of-range luma remains for the existing codec. The filter can
also remove intended relighting color.

## GPU and CPU paths

The full build includes `linux_color.hsaco`. To rebuild and test that module
separately after source preparation and the host build:

```bash
python3 scripts/build-kernels.py --only linux_color \
    --compiler clang++-22 --linker /usr/bin/ld.lld-22
./build/color-gpu-test --module assets/HIP/gfx1201/linux_color.hsaco
```

The test needs a compatible HIP runtime and Radeon, but no weights or game.
It compares four strengths with the CPU reference, checks retained-reference
isolation and prints kernel timing. Exit 77 means the module or a compatible
runtime or device is unavailable. See [VALIDATION.md](VALIDATION.md) for
integration tests.

When correction is enabled, the worker logs `color preservation backend: GPU
(HIP)` if the module is available. The GPU path retains the reference using an
asynchronous device copy and adds one kernel per pass on the inference stream.
It adds no correction-related per-pass host transfers or waits. Extra storage
is 28 bytes per padded pixel, approximately 59.1 MiB at 1920×1152.

If the module is absent, the worker logs `CPU fallback`. That path reads the
original input once per frame, then reads, corrects and uploads each enabled
pass on the CPU. A present module that fails to load or launch raises an error.
Backend selection is cached until restart; reinstall and restart the worker
after building a missing module. Strength zero skips correction work.

Opt-in tracing records `color_preserve`, `color_backend` and
`pass-NN-color.pfm`. The automated color diagnostic disables preservation for
its baseline comparisons and restores the original setting afterwards; its
initial current-view capture retains the user's setting.
