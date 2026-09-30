# Source provenance

`dlsslop-amd` means **DLSS Linux Open Proxy for AMD**. Project installation
paths and the native worker's private channel use `dlsslop-amd`. Public commands
are `dlsslopd`, `dlsslopctl`, `dlsslop-gui`, `dlsslop-test`, `dlsslop-setup`
and `dlsslop-run`. Locally introduced integration identifiers use `DLSSLOP_`, including
`DLSSLOP_AMD_ENABLE`, `DLSSLOP_MODULES` and `DLSSLOP_HIP_LIBRARY`; the local
codec namespace is `dlsslop`.

Inherited interfaces retain their upstream names, including `DLSSNR_SHM`,
`DLSSNR_LOG`, the other `DLSSNR_*` layer controls, the `dlssnr` C++ namespace
and the AMD network's `DLSS5_VIT_ADAPTIVE`. The inherited control build target
remains `dlssnr-shmctl`; installation copies its ELF executable directly to
`bin/dlsslopctl` for the public command. The inherited shared
runtime directory is `/tmp/dlssnr-UID` and honors `DLSSNR_UID`; the native
tools' default channel is `/tmp/dlsslop-amd-UID/shm.bin`. Captures use
`dlssnr/captures` under the state directory, with `/tmp/dlssnr-captures` as the
final fallback. This distinction follows the origin of each identifier,
including local additions and inherited interfaces in the same patched source
file.

Upstream repository names, URLs, authorship, source filenames (except the GLSL
ports in `layer/dlssnr/` and `layer/scaling/` and the HIP host port in
`backend/`) and attribution remain unchanged, as do references to NVIDIA DLSS
and OptiScaler_DLSSNR.

`.gitmodules` declares dlsslop-amd's forks of three repositories: each fork's
`dlsslop-amd` branch holds the Linux integration as commits on top of the
upstream commit listed below. `upstreams.lock.json` records the pinned commits
and the files fetched from each repository; the build uses them in place under
`external/`.

| Component | Upstream and base commit | Fork and pinned commit | License |
|---|---|---|---|
| Vulkan presentation layer and shared protocol | [bmitch87/DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) `ab722b091071d6d59df56f10d86d4f3005bcad86` | [imaami/DLSS5VKLayer](https://github.com/imaami/DLSS5VKLayer/tree/dlsslop-amd) `125debdc8e9e5227f5dc4d800cb17d2773487e14` | AGPL-3.0; embedded dependencies keep their notices |
| Vulkan network runtime, SPIR-V sources and model extractor | [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) `743326d15f56c93ca757b18ca4d6b0d81d654113` | [imaami/DLSSNR-AMD](https://github.com/imaami/DLSSNR-AMD/tree/dlsslop-amd) `3dfdddc7c06b888685c8be4275d1eb5e8edc7334` | MIT |
| AMD HIP kernels, and the scheduler that dlsslopd ports | [lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting) `ad499a8199c9ce3678d83c9be58fe3bc1bef3498` | [imaami/dlss5-on-amd-9070xt-porting](https://github.com/imaami/dlss5-on-amd-9070xt-porting/tree/dlsslop-amd) `c1908317fb7e7ee9fe4884feba4a67220d93461f` | MIT |

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
composition. In the AMD fork, three kernel sources wrap their bare workgroup
barriers in LDS-only release/acquire fences, and `deep_fast.hip` declares
`byte_F` before a template uses it. The fork's host-side commits make
`hip_api.h` load the Linux HIP runtime, remove the Windows-only F8 hotkey from
`hip_reference_network.h`, make its network write the final RGB straight into
the caller's buffer, and remove the `DLSS5_*` environment overrides from
`LmxxfProductionOptions.h`. The build fetches only the fork's kernel sources and
license; it neither fetches nor compiles the fork's host code.
The DLSSNR-AMD fork lets the runtime load its model, SPIR-V and pipeline cache
from explicit paths, lets the network build take its glslang, and adds
dlsslop-amd's per-pass sharpening and color preservation
(`linux/shaders/passes/pass_stages.comp`); the network mathematics are unchanged.

dlsslopd runs the AMD fork's kernels with its own host code.
`backend/hip_weights.*`, `backend/hip_plan.*` and `backend/hip_network.*` port
the production path of the fork's `Development/HIP/hip_reference_network.h` and
`packed_weights.h` (MIT) at the pinned commit. The port follows upstream's
production options (`src/LmxxfProductionOptions.h`) with eight flags off.
Production launches nothing from the modules of the WMMA, tiled, wave and
fused-C32 flags, and `scripts/build-kernels.py` has no recipe yet for the
modules of the wave-owned, C512 M32, ViT N64 and PDL flags. Unlike those
options, the port runs blocks 42, 43 and 46 unless `dlsslopd --performance`
skips them. With these options, the port launches the same kernels with the
same arguments as the fork's host code at the pinned commit, and it packs the
weights byte for byte as upstream does; the network mathematics are unchanged.
Like the fork, the port writes the final RGB straight into the caller's buffer,
while upstream writes it into a pooled tensor and copies it from there on every
pass. The port leaves out the adaptive ViT reuse, and dlsslopd warns when
`DLSS5_VIT_ADAPTIVE` requests it.
`backend/hip.h` declares the HIP runtime entry points dlsslopd calls and the
layouts of the types it passes to them. It takes those layouts from ROCm's
`include/hip/hip_runtime_api.h` as the fork's `hip_api.h` and
`hip_device_properties.h` copy them from rocm-7.1.1 (MIT, Copyright (c)
2015 - 2023 Advanced Micro Devices, Inc.; the notice is in
`packaging/THIRD-PARTY.txt`). `backend/hip.cpp` loads the runtime
(`libamdhip64.so.7`, `.so.6` or the unversioned soname, also from
`/opt/rocm/lib` or an explicit `DLSSLOP_HIP_LIBRARY` path) with
`dlopen`/`dlsym`. The build ships 12 of upstream's 29 modules; the network
loads the 6 it launches.

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
