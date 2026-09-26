# Control options

`dlsslopctl --help` is the canonical option, range and default reference.
`--settings` prints current values alongside reset defaults. Omitted options leave
current settings unchanged. There are 40 settings: 39 of the 41 inherited controls
and native per-pass color preservation, all mapped below.

“Native equivalent” means an implemented AMD-side operation, not an implementation
of NVIDIA's proprietary parameter mapping. “Fixed” means the extracted network
supports only the stated configuration; requesting another value fails before
opening or modifying the control channel.

| Original setting | Current option | Implementation |
|---|---|---|
| `enabled` | `-e`, `--enabled` | Enables the rendering pipeline. |
| `hdrmode` | `-E`, `--hdr-mode` | Proxy precision: automatic, forced 8-bit, or forced binary16. |
| `sdr16multipass` | `-B`, `--sdr16-multipass` | Selects binary16 or quantized 8-bit SDR feedback between neural passes. |
| `passes` | `-P`, `--passes` | Runs successive neural evaluations; each consumes the preceding result. |
| `preset` | `-N`, `--preset` | **Fixed: 0.** Alternate model presets and their weight/parameter mappings are unavailable. |
| `style` | `-y`, `--style` | **Fixed: 0.** Natural/cinematic model conditioning is unavailable. |
| `automask` | `-M`, `--auto-mask` | **Fixed: 1.** There is no recovered switchable semantic-mask configuration. |
| `intensity` | `-i`, `--intensity` | **Native equivalent:** scales each pass's image residual. |
| `localtone` | `-o`, `--local-tone` | **Native equivalent:** scales the low-frequency part of each pass's residual. |
| `localstructure` | `-j`, `--local-structure` | **Native equivalent:** scales the high-frequency part of each pass's residual. |
| `skinstructure` | `-K`, `--skin-structure` | **Fixed: -1.** Structure applies uniformly; separate skin-specific conditioning is unavailable. |
| `sharpness` | `-n`, `--sharpness` | **Native equivalent:** adds unsharp-mask detail after each neural pass. |
| `detail` | `-d`, `--detail` | Vulkan composition strength of the final neural edit. |
| `colour` | `-C`, `--color` | Vulkan composition strength of the edit's color contribution. |
| `guard` | `-g`, `--guard` | Bounds per-pixel relighting gain in ratio-based composition modes. |
| `transfer` | `-t`, `--transfer` | Chooses classic ratio, matched residual, or native-frame-plus-edit composition. |
| `bypass` | `-b`, `--bypass` | Presents the model result instead of composing its edit. |
| `rebuildms` | None | **Not applicable:** native tuning rebuilds nothing; intensity, tone, structure and sharpness apply on the next request. |
| `ratiosmooth` | `-a`, `--ratio-smooth` | Neighbourhood contribution to ratio-based relighting. |
| `colourtrust` | `-u`, `--color-trust` | Bounds color displacement in composition. |
| `mvec` | `-V`, `--mvec` | **Native equivalent:** HIP optical-flow estimation and reprojection into the network's temporal input. |
| `mvecquality` | `-Q`, `--mvec-quality` | Sets native flow-pyramid depth and search effort. |
| `mvecunits` | None | **Not applicable:** the flow never leaves the worker, which estimates and reprojects it in pixels. |
| `mvecpixels` | `-F`, `--mvec-pixels` | Sets native flow-grid spacing. |
| `debugview` | `-v`, `--debug-view` | Selects proxy, model, residual or color-bound inspection views. |
| `debugscale` | `-D`, `--debug-scale` | Multiplies the debug-view signal. |
| `whitepoint` | `-W`, `--white-point` | Manual white point for linear-light normalization. |
| `whitepointscale` | `-G`, `--white-point-scale` | Multiplies the selected manual or measured white point. |
| `whitepointsource` | `-O`, `--white-point-source` | Selects manual white point or GPU frame measurement. |
| `whitepointtrim` | `-I`, `--white-point-trim` | Adjusts measured white point only. |
| `workingscale` | `-w`, `--working-scale` | Sets proxy scale; `dlsslopd` caps it at 1 and at its tier raster unless run with `--cpu-compose` or `--test-identity`. |
| `downscaler` | `-f`, `--downscaler` | Selects the down-leg filter when the model raster exceeds display size, which only `--working-scale` above 1 with `dlsslopd --cpu-compose` or `--test-identity` produces. |
| `compare` | `-p`, `--compare` | Enables side-by-side or wipe comparison. |
| `comparesplit` | `-x`, `--compare-split` | Sets the comparison divider. |
| `comparezoom` | `-z`, `--compare-zoom` | Sets side-by-side magnification. |
| `compareswap` | `-X`, `--compare-swap` | Swaps edited and original comparison sides. |
| `colourmode` | `-Y`, `--color-mode` | Selects automatic, display-referred or linear-light interpretation. |
| `reversible` | `-Z`, `--reversible` | Selects knee, Neutwo or hybrid proxy encoding and composition/replacement. |
| `applymodel` | `-m`, `--apply-model` | Shows the clean frame or applies the computed edit; inference still runs. |
| `hold` | `-H`, `--hold` | Freezes captured input while allowing settings to rerun processing. |
| `togglekey` | `-k`, `--toggle-key` | Selects the Linux input key watched by the layer. |
| None | `-L`, `--color-preserve` | **Native:** anchors each pass's broad chroma changes to the original input before history and feedback; see [COLOR-PRESERVATION.md](COLOR-PRESERVATION.md). |

The native residual controls operate after **each** evaluation and before its
output feeds the next pass. The low-frequency component is a 3×3 binomial blur;
structure scales the remaining residual. `--detail` and `--color` instead act
once during final Vulkan composition. Default native tuning preserves the raw
network result bit for bit.

White-point and reversible encoding curves operate on linear-light input.
The replacement choices (`--reversible 2` and `4`) also present the model result
on ordinary SDR input.
Forcing binary16 transport does not reinterpret an SDR frame as HDR. Disabling
binary16 transport also does not disable normalization of actual HDR input.
The SDR feedback precision option does not reduce HDR between-pass feedback,
which always retains binary16 precision.
Motion is estimated from consecutive presented frames; it is not supplied by the
game engine. Each neural pass retains its own temporal history.

The general actions are `--status`, `--settings`, `--reset`, `--quit`, `--resume`,
`--capture`, `--toggle`, `--shm`, and `--help`; all have short forms and explicit
defaults in help. For example:

```bash
dlsslopctl --hold 1 --passes 2
dlsslopctl --intensity 0.75 --local-tone 1 --local-structure 1.25
dlsslopctl --toggle apply-model
dlsslopctl --hold 0
```

`--toggle` accepts only writable boolean settings. In particular,
`--passes 2 --toggle auto-mask` fails without changing passes or any other field.
`--reset` restores settings while preserving transport status and stop state.
