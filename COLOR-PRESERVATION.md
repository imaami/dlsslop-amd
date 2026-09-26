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
full strength, but high-frequency chroma changes can remain. When luma is in
[0,1], the correction, never the pass output's own chroma, is shortened by one
factor so that no channel leaves [0,1] or goes further outside it than the pass
put it. Out-of-range luma takes the full correction and keeps its headroom for
the existing codec. The filter can also remove intended relighting color.

## HIP module

The full build includes `linux_color.hsaco`, and the worker requires it like
its other native modules. To rebuild and test that module separately after
source preparation and the host build:

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

The worker loads the module when correction is first enabled. It retains the
reference using an asynchronous device copy and adds one kernel per pass on
the inference stream, with no correction-related host transfers or waits.
Extra storage is 28 bytes per padded pixel, approximately 59.1 MiB at
1920×1152. A module that is missing or fails to load or launch stops the
worker with an error. Strength zero skips correction work.

Opt-in tracing records `color_preserve` and `pass-NN-color.pfm`. The
automated color diagnostic disables preservation for its baseline comparisons
and restores the original setting afterwards; its initial current-view capture
retains the user's setting.
