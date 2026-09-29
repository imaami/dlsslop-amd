# Source provenance

`dlsslop-amd` means **DLSS Linux Open Proxy for AMD**. Project installation
paths and the native worker's private channel use `dlsslop-amd`. Public commands
are `dlsslopd`, `dlsslopctl`, `dlsslop-gui`, `dlsslop-test`, `dlsslop-setup`
and `dlsslop-run`. Locally introduced integration identifiers use `DLSSLOP_`, including
`DLSSLOP_AMD_ENABLE`, `DLSSLOP_MODULES` and `DLSSLOP_HIP_LIBRARY`; the local
codec namespace is `dlsslop`.

Inherited interfaces retain their upstream names, including `DLSSNR_SHM`,
`DLSSNR_LOG`, the other `DLSSNR_*` layer controls, the `dlssnr` C++ namespace
and AMD network controls `DLSS5_VIT_*`. The inherited control build target
remains `dlssnr-shmctl`; installation copies its ELF executable directly to
`bin/dlsslopctl` for the public command. The inherited shared
runtime directory is `/tmp/dlssnr-UID` and honors `DLSSNR_UID`; the native
tools' default channel is `/tmp/dlsslop-amd-UID/shm.bin`. Captures use
`dlssnr/captures` under the state directory, with `/tmp/dlssnr-captures` as the
final fallback. This distinction follows the origin of each identifier,
including local additions and inherited interfaces in the same patched source
file.

Upstream repository names, URLs, authorship, source filenames (except the GLSL
ports in `layer/dlssnr/` and `layer/scaling/`) and attribution remain unchanged,
as do references to NVIDIA DLSS and OptiScaler_DLSSNR.

`.gitmodules` declares dlsslop-amd's forks of three repositories: each fork's
`dlsslop-amd` branch holds the Linux integration as commits on top of the
upstream commit listed below. `upstreams.lock.json` records the pinned commits
and the files fetched from each repository; the build uses them in place under
`external/`.

| Component | Upstream and base commit | Fork and pinned commit | License |
|---|---|---|---|
| Vulkan presentation layer and shared protocol | [bmitch87/DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) `ab722b091071d6d59df56f10d86d4f3005bcad86` | [imaami/DLSS5VKLayer](https://github.com/imaami/DLSS5VKLayer/tree/dlsslop-amd) `125debdc8e9e5227f5dc4d800cb17d2773487e14` | AGPL-3.0; embedded dependencies keep their notices |
| Vulkan network runtime, SPIR-V sources and model extractor | [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) `743326d15f56c93ca757b18ca4d6b0d81d654113` | [imaami/DLSSNR-AMD](https://github.com/imaami/DLSSNR-AMD/tree/dlsslop-amd) `3dfdddc7c06b888685c8be4275d1eb5e8edc7334` | MIT |
| AMD neural scheduler and HIP kernels | [lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting) `ad499a8199c9ce3678d83c9be58fe3bc1bef3498` | [imaami/dlss5-on-amd-9070xt-porting](https://github.com/imaami/dlss5-on-amd-9070xt-porting/tree/dlsslop-amd) `c1908317fb7e7ee9fe4884feba4a67220d93461f` | MIT |

The layer's composition shader `layer/dlssnr/dlssnr.comp` is a GLSL port of the
layer fork's `layer_linux/src/dlssnr/dlssnr.hlsl` (AGPL-3.0). Its RenoDX-derived
composition is MIT; the notice is
`external/layer/third_party/optiscaler/RenoDX_ATTRIBUTION.txt`.

The eight scaling filters in `layer/scaling/` are GLSL ports of
`OptiScaler/shaders/output_scaling/precompile/*.hlsl` from
[optiscaler/OptiScaler](https://github.com/optiscaler/OptiScaler) at
`fb41e3e6361ca9ae55b30a821a40c2b4b346f330` (GPL-3.0; the text is
`external/layer/third_party/optiscaler/LICENSE`). `bcus.comp` also keeps the
Microsoft MiniEngine MIT notice of its original.

The layer fork adapts Linux loading and transport and extends controls and
composition. In the AMD fork, `hip_api.h` loads the Linux HIP runtime
(`libamdhip64.so.7`, `.so.6` or the unversioned soname, also from
`/opt/rocm/lib` or an explicit `DLSSLOP_HIP_LIBRARY` path) with
`dlopen`/`dlsym`. `hip_reference_network.h` ignores the Windows-only F8 hotkey
option (`DLSS5_VIT_REUSE_HOTKEY`) instead of calling Win32 keyboard APIs, and
device `Enqueue` writes the final RGB straight into the caller's buffer instead
of copying it from a pooled tensor on every pass. `LmxxfProductionOptions.h`
drops the `DLSS5_*` environment overrides: upstream's
`native_hip_env_options.h` needs Windows headers, and dlsslopd's own options
apply. Three kernel sources wrap their bare workgroup barriers in LDS-only
release/acquire fences. The scheduler and the network mathematics are otherwise
unchanged. The worker clears the production options' block-skip set unless
`--performance` restores upstream's skipped blocks 42, 43 and 46. It also clears
the WMMA, tiled, wave and fused-C32 flags: production launches nothing from their
modules, so the build ships only the 12 of upstream's 29 modules that the network
loads. It clears the wave-owned, C512 M32, ViT N64 and PDL flags as well:
`scripts/build-kernels.py` has no recipe yet for their modules.
The DLSSNR-AMD fork lets the runtime load its model, SPIR-V and pipeline cache
from explicit paths, lets the network build take its glslang, and adds
dlsslop-amd's per-pass sharpening and color preservation
(`linux/shaders/passes/pass_stages.comp`); the network mathematics are unchanged.

Preserve `external/layer/ATTRIBUTION.md` and all inherited notices of upstream
code. The layer's shader/dispatch lineage includes OptiScaler and
Dagherbou/OptiScaler_DLSSNR, with RenoDX color-composition attribution. Khronos
Vulkan/video headers and stb retain their own licenses. The BCUS shader also
identifies Microsoft MiniEngine/Minigraph and James Stanard.

The native worker integration, codec, tuning/temporal and color-preservation
modules, Qt Widgets controller, build tools and tests are local additions or
adaptations. New independent integration code uses [LICENSE.integration](LICENSE.integration)
(MIT); derivative portions retain their upstream licenses. [LICENSE](LICENSE)
is the AGPL-3.0 text. Preserve the applicable notices when redistributing source
or build outputs.

The model coefficients are a separate dependency and are never included in
source or CI artifacts. `scripts/fetch-assets.py` (installed as `dlsslop-setup`)
imports user-provided coefficients and records their hashes; its
`--print-manifest` option prints the expected names and element counts. The
source-code licenses do not license those weights.
