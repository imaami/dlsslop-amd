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

Upstream repository names, URLs, authorship, source filenames and attribution
remain unchanged, as do references to NVIDIA DLSS and OptiScaler_DLSSNR.

`.gitmodules` declares these repositories; `upstreams.lock.json` records their
exact commits, selected source paths and hashes. The parent repository stores
integration patches and local source, not complete upstream copies.

| Component | Upstream | Pinned commit | License |
|---|---|---|---|
| Vulkan presentation layer and shared protocol | [bmitch87/DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) | `ab722b091071d6d59df56f10d86d4f3005bcad86` | AGPL-3.0; embedded dependencies keep their notices |
| Vulkan network runtime, SPIR-V sources and model extractor | [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) | `743326d15f56c93ca757b18ca4d6b0d81d654113` | MIT |
| AMD neural scheduler and HIP kernels | [lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting) | `ad499a8199c9ce3678d83c9be58fe3bc1bef3498` | MIT |
| Eight scaling shader sources | [optiscaler/OptiScaler](https://github.com/optiscaler/OptiScaler) | `fb41e3e6361ca9ae55b30a821a40c2b4b346f330` | GPL-3.0; individual files retain additional notices |

`prepare-sources.py` verifies selected originals against their recorded hashes,
projects them into `upstream-layer/`, `kernels/`, `backend/vendor/` and `vulkan-nr/`, and
applies `patches/linux-integration.patch`. The patch adapts Linux loading and
transport and extends controls and composition. It leaves the upstream working
trees and HIP kernels unchanged; `scripts/build-kernels.py` wraps the bare
workgroup barriers of three kernel sources in LDS-only release/acquire fences.
In `backend/vendor/`, the patch makes `hip_api.h` load the Linux HIP runtime
(`libamdhip64.so.7`, `.so.6` or the unversioned soname, also from
`/opt/rocm/lib` or an explicit `DLSSLOP_HIP_LIBRARY` path) with
`dlopen`/`dlsym`. It makes `hip_reference_network.h` ignore the Windows-only
F8 hotkey option (`DLSS5_VIT_REUSE_HOTKEY`) instead of calling Win32 keyboard
APIs, and makes device `Enqueue` write the final RGB straight into the caller's
buffer instead of copying it from a pooled tensor on every pass. The scheduler
and the network mathematics are otherwise unchanged. The patch also drops the
`DLSS5_*` environment overrides from `LmxxfProductionOptions.h`: upstream's
`native_hip_env_options.h` needs Windows headers, and dlsslopd's own options
apply. The worker clears the production options' block-skip set unless
`--performance` restores upstream's skipped blocks 42, 43 and 46. It also clears
the WMMA, tiled, wave and fused-C32 flags: production launches nothing from their
modules, so the build ships only the 12 of upstream's 29 modules that the network
loads. It clears the wave-owned, C512 M32, ViT N64 and PDL flags as well:
`scripts/build-kernels.py` has no recipe yet for their modules.
In `vulkan-nr/`, the patch lets the runtime load its model, SPIR-V and pipeline
cache from explicit paths, lets the network build take its glslang, and adds
dlsslop-amd's per-pass sharpening and color preservation
(`shaders/passes/pass_stages.comp`); the network mathematics are unchanged.

Preserve `upstream-layer/ATTRIBUTION.md` and all inherited notices of projected
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
