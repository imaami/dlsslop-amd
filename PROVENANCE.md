# Source provenance

`dlsslop-amd` means **DLSS Linux Open Proxy for AMD**. Project installation
paths and the native worker's private channel use `dlsslop-amd`. Public commands
are `dlsslopd`, `dlsslopctl`, `dlsslop-gui`, `dlsslop-test`, `dlsslop-setup`
and `dlsslop-run`. Locally introduced integration identifiers use `DLSSLOP_`, including
`DLSSLOP_AMD_ENABLE`, `DLSSLOP_MODULES` and `DLSSLOP_HIP_LIBRARY`; the local
codec namespace is `dlsslop`.

Inherited interfaces retain their upstream names, including `DLSSNR_SHM`,
`DLSSNR_LOG`, the other `DLSSNR_*` layer controls and the AMD network's
`DLSS5_VIT_ADAPTIVE`. The inherited control build target
remains `dlssnr-shmctl`; installation copies its ELF executable directly to
`bin/dlsslopctl` for the public command. The inherited shared
runtime directory is `/tmp/dlssnr-UID` and honors `DLSSNR_UID`; the native
tools' default channel is `/tmp/dlsslop-amd-UID/shm.bin`. Captures use
`dlssnr/captures` under the state directory, with `/tmp/dlssnr-captures` as the
final fallback. This distinction follows the origin of each identifier,
including local additions and inherited interfaces in the same patched source
file.

Upstream repository names, URLs, authorship, source filenames (except the GLSL
ports in `layer/dlssnr/` and `layer/scaling/`, the HIP host port in `backend/`,
the Vulkan host port in `common/` and the C port of the layer in `layer/`) and
attribution remain unchanged, as do references to NVIDIA DLSS and
OptiScaler_DLSSNR.

`.gitmodules` declares dlsslop-amd's forks of the two networks' repositories
and, for each fork, the branch that holds the Linux integration as commits on
top of the upstream commit listed below. `upstreams.lock.json` records the
pinned commits and the files fetched from each repository; the build uses them
in place under `external/`. The layer's sources were imported from its fork at
the commit listed below and are maintained in this repository.

| Component | Upstream and base commit | Fork and pinned commit | License |
|---|---|---|---|
| Vulkan presentation layer and shared protocol | [bmitch87/DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) `ab722b091071d6d59df56f10d86d4f3005bcad86` | [imaami/DLSS5VKLayer](https://github.com/imaami/DLSS5VKLayer/tree/dlsslop-amd) `680ec8afb96cff206cf6a7608d3a559ca1e9c2f3`, imported | AGPL-3.0; embedded dependencies keep their notices |
| Vulkan network's SPIR-V sources, shader build and model extractor, and the runtime that dlsslop-amd ports | [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) `82560c4fbfaac347fc5e22c22025191402ae916b` | [imaami/DLSSNR-AMD](https://github.com/imaami/DLSSNR-AMD/tree/dlsslop-amd-82560c4) `c77366542351732c44ce9438baba1cc1691d9476` | MIT |
| AMD HIP kernels, and the scheduler that dlsslopd ports | [lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting) `ad499a8199c9ce3678d83c9be58fe3bc1bef3498` | [imaami/dlss5-on-amd-9070xt-porting](https://github.com/imaami/dlss5-on-amd-9070xt-porting/tree/dlsslop-amd) `c1908317fb7e7ee9fe4884feba4a67220d93461f` | MIT |

The import took these files of the layer fork at `680ec8a`. It changed only
the include paths that the move broke: one line each in `layer/capture.cpp`
(now `capture.c`), `layer/composition.h` and `layer/layer.cpp` (now `layer.c`).

| Fork path | Path here |
|---|---|
| `layer_linux/src/*.cpp`, `layer_linux/src/*.h` | `layer/` |
| `layer_linux/src/dlssnr/DlssNr_Common.h`, `DlssNr_Layout.h` | `layer/dlssnr/` |
| `layer_linux/src/shaders/meter_reduce.comp` | `layer/meter_reduce.comp` |
| `layer_linux/dlssnr.map` | `layer/dlssnr.map` |
| `tools/shmctl.cpp` | `layer/shmctl.cpp` (now `shmctl.c`) |
| `common/shm_protocol.h` | `common/shm_protocol.h` (its functions now in `shm_protocol.c` and `shm_protocol.hpp`) |
| `ATTRIBUTION.md` | `layer/ATTRIBUTION.md` |
| `core/stb_image_write.h`, `standalone_runner/third_party/stb_image.h` | `third_party/stb/` |
| `third_party/optiscaler/LICENSE`, `RenoDX_ATTRIBUTION.txt` | `third_party/optiscaler/` |

The fork's `LICENSE` is the AGPL-3.0 text of [LICENSE](LICENSE). The paths that
`layer/ATTRIBUTION.md` names are those of the fork's tree. The build uses the
system's Vulkan headers, so the import left out the fork's copies of the Khronos
Vulkan and video headers.

The layer's sources are ported to C23 under the layer's license.
`layer/log.h` and `layer/log.c` are the C port of
`layer_linux/src/log.h`, `layer/capture.h` and `layer/capture.c` that of
`capture.h` and `capture.cpp`, `layer/hotkey.h`, `layer/hotkey_priv.h` and
`layer/hotkey.c` that of `hotkey.h` and `hotkey.cpp`, `layer/shader_vk.h`,
`layer/shader_vk_priv.h` and `layer/shader_vk.c` that of `shader_vk.h` and
`shader_vk.cpp`, `layer/dlssnr_pass.h` and `layer/dlssnr_pass.c` that of
`dlssnr_pass.h` and `dlssnr_pass.cpp`, `layer/scaler_vk.h` and
`layer/scaler_vk.c` that of `scaler_vk.h` and `scaler_vk.cpp`,
`layer/composition.h` and `layer/composition.c` that of `composition.h` and
`composition.cpp`, `layer/layer.c` that of `layer.cpp`, and `layer/shmctl.c`
that of `tools/shmctl.cpp`. The port renames upstream names by one rule: the
`dlssnr` namespace is dropped, CamelCase becomes snake_case, a class member
takes its class as a prefix (`Composition::RecordCompose` becomes
`composition_record_compose`), a free function with external linkage takes its
module's name as a prefix unless its name begins with it (`Verbose` in log.h
becomes `log_verbose`), and constants and enumerators become upper snake case
with the class's prefix (`DlssNrPass::kSlots` becomes `DLSS_NR_PASS_SLOTS`). Log
texts, environment variables, the loader's entry points, `layer/dlssnr.map`,
dlsslopctl's options and output, and the names in `common/shm_protocol.h`, which
dlsslopd, the GUI, the network module and the tests share, keep their names.
These upstream names do not follow the rule:

| Upstream name | Port | Why |
|---|---|---|
| `dlssnr::Log` | `log_printf` | `log` is the C library's natural logarithm |
| Data members, such as `DlssNrConstants::WhitePoint` and `CaptureWriter::_batchDir` | snake_case without a prefix or a leading underscore: `white_point`, `batch_dir` | a struct scopes its members in C as in C++ |
| `IsEightBitRgba`, `NeedsChannelSwap` and `BytesPerPixel` in capture.cpp | `encoding` in capture.c | one table of how a format's frames are written |
| `CaptureWriter::Remaining` | none; the test reads `remaining` | it had no other caller |
| The constructors and destructors of `Shader_Vk`, `DlssNrPass`, `ScalerVk` and `Composition`, the member initializers of `FrameSettings`, `InstanceChain`, `SwapchainState` and `DeviceChain`, and `Hotkeys::~Hotkeys` and `ShmMap::~ShmMap` | `shader_vk`, `dlss_nr_pass` and `dlss_nr_pass_init`, `scaler_vk` and `scaler_vk_init`, `composition` and `composition_init`, `composition_frame_settings`, `instance_chain_create`, `swapchain_state_create` and `device_chain_create`; `shader_vk_fini`, `dlss_nr_pass_fini`, `scaler_vk_fini`, `composition_fini`, `hotkeys_fini`, `shm_map_fini`, `instance_chain_destroy`, `swapchain_state_destroy` and `device_chain_destroy` | C's life cycle: a function named like the struct returns a new one, an init function makes one in place, a create function allocates one, a fini function frees what an object owns and a destroy function frees the object too |
| `Hotkeys::OpenEvdev` | part of `hotkeys_open` | it had no other caller |
| `Hotkeys::_fds`, `_known`, `_knownOrder` and `_notKeyboard` | `struct hotkey_node` entries of `nodes` | one table of the event nodes |
| `Hotkeys::_pending` | `pending` and `pending_total` | a count per key code |
| `Hotkeys::_opened` and `_announced`, `ScalerVk::_upsample`, the `bool` members of `Composition` but `_usable`, `_meterGpu` and `_superSample`, and those of `ShmMap`, `SwapchainState` and `DeviceChain` in layer.cpp | `HOTKEYS_OPENED` and `HOTKEYS_ANNOUNCED`, `SCALER_VK_UPSAMPLE`, `COMPOSITION_BLIT_SWAPCHAIN` to `COMPOSITION_FRAME_CAPTURED`, `SHM_MAP_ANSWERED` and `SHM_MAP_DEAD`, the `SWAPCHAIN_STATE_*` and the `DEVICE_CHAIN_*` in `flags` | one flags member |
| `DlssNrPass::_dummyReady` | none; the pass tests its placeholder's handles | the handles are stored only once all three exist, in the call that records the placeholder's move to `GENERAL` |
| `Composition::_meterGpu` and `_superSample` | none; the composition tests `meter_state` and `model_native.image` | each was true exactly while its object existed |
| `Composition::_usable` | `error`, which `composition_usable()` reads | the build's `VkResult`; where the class cleared `_usable` for a format it cannot write, `composition_prepare()` sets `VK_ERROR_FORMAT_NOT_SUPPORTED` |
| `Composition::HdrTransfer` and `CaptureActive` | none; composition-rebuild-test calls `capture_writer_active()` on the composition's writer | `HdrTransfer` had no caller, and `CaptureActive` only the test |
| `Composition::FormatSupportsStorage` and `FormatSupportsBlit` | `format_supports` in composition.c | one query of a format's optimal-tiling features |
| `Hotkeys::_x11`, `_xi`, `_display`, `_xiOpcode` and the `_x*` function pointers | `struct hotkey_x11` | loaded only for the XInput2 backend |
| `Shader_Vk::_init` and `CanRender` | `error` in `struct dlss_nr_pass` and `struct scaler_vk`; a dispatch tests `shader.pipeline` | the pipeline is created last, so it exists exactly when the build succeeded |
| `Shader_Vk::_descriptorSets` | `descriptor_sets` in `struct dlss_nr_pass` and `struct scaler_vk` | a fixed array in each pass |
| `Shader_Vk::CreateBinding` | `SHADER_VK_BINDING` | a constant initializer, for the passes' static tables of bindings |
| `DlssNrPass::ConstantBuffer`, `ConstantSlotStride` and `NextConstantSlot` | none; the composition reads `shader.constant_buffer`, `slot_stride` and `slot` | they had no other caller |
| `ScalerVk::_filter` | none; `scaler_vk()` selects the shader | it was read only while the pass was built |
| `DownsampleBlob` and `Blob` in scaler_vk.cpp | `SCALER_VK_AVERAGES` and `struct scaler_shader` in scaler_vk.c | one table of the averages' SPIR-V and names |
| `ScalerFilterName`, `ScalerFilter` and its `kScaler*` enumerators | `scaler_vk_filter_name`, `enum scaler_vk_filter` and `SCALER_VK_*` | the module's prefix replaces the names' own `Scaler` |
| `FrameSettings` and `FrameSettings::Read` in composition.h | `struct composition_frame_settings`, `composition_frame_settings` and `composition_frame_settings_read` | a function with external linkage carries its module's prefix, and the struct keeps the name of the function that returns one |
| `g_phys` in layer.cpp | none; `struct instance_chain` holds `physical` and `physical_count` | an instance's physical devices are freed with the instance |
| `ShmOpen`, `ShmProcessFrame` and the `ShmNeuralEnabled(ShmMap&)` overload in layer.cpp | `shm_map_open`, `shm_map_process_frame` and `shm_map_neural_enabled` | they work on a `struct shm_map`; `shm_open` is POSIX's, and `ShmNeuralEnabled` is the channel header's |
| `ShmMap::seq` and `firstHeartbeat`, `InstanceChain::next_gipa` and `DeviceChain::next_dpa` | none; the chains call their tables' `next_gipa` and `next_dpa` | nothing read the first two, and the others repeated their tables' members |
| `Initialised` in shmctl.cpp | none; `attach` and `print_status` compare the magic and the version that they read | each word of the header is read once |

The headers that C and C++ share stay valid C++23. `common/shm_protocol.h`
declares the channel for both languages, `layer/vk_table.h` the dispatch tables,
and `layer/dlssnr/DlssNr_Common.h` and `DlssNr_Layout.h` the composition
shader's constant block. `common/shm_protocol.c` defines the channel header's
functions, which both languages call. `ShmStoreString`, `ShmLoadString`,
`ShmStore64` and `ShmLoad64` take the header and an `enum shm_text` or
`enum shm_count` instead of the field's words, because a word is a
`std::atomic<uint32_t>` in C++ and an `_Atomic(uint32_t)` in C.
`common/shm_protocol.hpp` holds overloads that return a `std::string` of the
functions that write a string into the caller's buffer, but `ShmRuntimeDir`,
which only a test calls.

The layer's composition shader `layer/dlssnr/dlssnr.comp` is a GLSL port of the
layer fork's `layer_linux/src/dlssnr/dlssnr.hlsl` (AGPL-3.0). Its RenoDX-derived
composition is MIT; the notice is
`third_party/optiscaler/RenoDX_ATTRIBUTION.txt`.

The eight scaling filters in `layer/scaling/` are GLSL ports of
`OptiScaler/shaders/output_scaling/precompile/*.hlsl` from
[optiscaler/OptiScaler](https://github.com/optiscaler/OptiScaler) at
`fb41e3e6361ca9ae55b30a821a40c2b4b346f330` (GPL-3.0; the text is
`third_party/optiscaler/LICENSE`). `bcus.comp` also keeps the
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
The DLSSNR-AMD fork lets the network build take its glslang and adds
dlsslop-amd's per-pass sharpening and color preservation
(`linux/shaders/passes/pass_stages.comp`); the network mathematics are unchanged.
The fork's host-side changes let the runtime load its model, SPIR-V and
pipeline cache from explicit paths and record the new stages after every pass.
A frame with those stages does not take upstream's path that samples the
caller's color in place. The fork's two shader commits after `a75ac49`, which
leave its host code as it was there, make a tile-counter wait give up once the
error word that its record names says that a wait of the frame has given up,
and a persistent run's claims once it says that a run has given up a claim,
and wait up to 2^17 polls for a tile instead of 2^16; a frame whose waits all
finish computes the same answer. The build fetches only the fork's SPIR-V
sources, shader build, model tools and license; it neither fetches nor compiles
the fork's host code (`linux/src/core`), which its host-side changes modify. Of
the 74 files that the shader build writes, `install.py` installs the 44 that
the project's own host code names in its tables and reads.

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
`DLSS5_VIT_ADAPTIVE` requests it. It also leaves out the weights no kernel
reads, which upstream uploads: the C512 blocks' unpacked `ffwd` weights, and
the c256 blocks' attention weights in the `@fp8` layout that the c64, c128 and
c512 blocks read. The port reads a weight's `.f16` file only when there is no
`.f32` file, and only files of the exact size that `dlsslop-setup` imports.
Upstream reads the `.f16` file whenever the `.f32` file does not open, and it
accepts any downsample or decoder file that holds at least the weight matrix.
`backend/hip.hpp` declares the HIP runtime entry points dlsslopd calls and the
layouts of the types it passes to them. It takes those layouts from ROCm's
`include/hip/hip_runtime_api.h` as the fork's `hip_api.h` and
`hip_device_properties.h` copy them from rocm-7.1.1 (MIT, Copyright (c)
2015 - 2023 Advanced Micro Devices, Inc.; `backend/hip.hpp` carries the notice,
and `packaging/THIRD-PARTY.txt` repeats it for binary releases). `backend/hip.cpp` loads the runtime
(`libamdhip64.so.7`, `.so.6` or the unversioned soname, also from
`/opt/rocm/lib` or an explicit `DLSSLOP_HIP_LIBRARY` path) with
`dlopen`/`dlsym`. Of the 30 modules that upstream's `hip/build-modules.ps1`
builds (its header comment says 29), the build ships only the 6 that the
network launches.

dlsslopd and the in-layer network run the DLSSNR-AMD fork's SPIR-V with the
project's own host code. `common/vulkan_weights.*`, `common/vulkan_plan.*`,
`common/vulkan_schedule.*` and `common/vulkan_runtime.*` port the production
path of the fork's `linux/src/core` (MIT) at `a75ac49`: `nr_runtime.cpp`,
`nr_graph.cpp`, `nrvk.hpp`, `nr_native_plan.cpp` with its layer table,
`tinlayout.hpp` and `nr_activation_lut.hpp`, including the fork's changes to
`nr_runtime.cpp`. For the frame formats, pass counts, stages and motion
estimation that dlsslop-amd uses, the port records each frame's dispatches, push
constants, barriers and copies as the fork's host code does, and it packs and
uploads the weights byte for byte as upstream does; the network mathematics are
unchanged. Like upstream, it audits the model's position biases and head
scales when it builds, runs the C=32 layers whose heads never reach the upper
clamp of the attention's exponent with the kernels built without that clamp,
and writes each persistent run's layers' clamp-free heads into their records,
which the kernels do not read. It also takes the model's blend scale, by which
the temporal post block weights the history, as a constant, as upstream does.
Unlike upstream, it rejects frames whose working extent is not a
multiple of 8, on which upstream's build fails, and frames whose activation
arena needs offsets past 32 bits or does not fit the device's storage buffers,
which upstream did not check. It bounds every read of the model, sends its
messages to the caller's log instead of standard output and error, and loads
neither `runtime_transfer.spv` nor `runtime_depth.spv`, which upstream loads but
never dispatches. Its build leaves the network's input unwritten, where upstream
uploads a gradient that the first frame overwrites. It compiles up to four
pipelines at once, on no more threads than half the CPUs that its build may run
on, where upstream compiles them one at a time. It saves the pipeline cache once
it has created every pipeline or failed to create one, where upstream saves it
before it creates those of the alpha pass, the pass stages, the motion estimate
and the temporal blocks. It replaces the cache file in one rename of a file of
its own, where upstream writes a fixed `.tmp` file that another process can
truncate or rename, and removes the cache file before its rename. It creates the
motion estimate's finest flow level for sampling as well as storage and
transfers, where upstream creates it for storage and transfers only and still
binds it to the temporal blocks' samplers. It advances the motion history's
latch, parity and noise seed when its caller says that a recorded frame was
submitted, where upstream advances them when it records the frame. Its
tile-counter records all name one error word, which a wait sets when it gives
up and which the frame's other waits read to give up early, where upstream's
records that wait name one each and those that only signal none, and it gives
the persistent runs that upstream gives no record one that orders nothing and
names that word. After the
network's last dispatch, each frame reads and zeroes the words that its waits
set and, when one ran out, answers with its input; the next frame then zeroes
the persistent runs' sync regions and the tile counters and drops the motion
history. Upstream's production path never reads those words. It leaves out
what dlsslop-amd never calls, such as control masks, `record_engine`,
per-feature histories, preprocessing, model scales below 1, input formats other
than RGBA8 and RGBA16F, the GPU timing that `last_gpu_ms()` and
`average_gpu_ms()` report, and every `NR_*` environment variable. Like
upstream's OptiScaler route at `b1419b0` (`Impl::DirectSrc` in
`nr_runtime.cpp`), the in-layer network samples the composition's proxy in place
and stores its answer into the composition's model image. Unlike upstream, which
does so for one pass with motion history only and makes views and sets for each
recording in a ring, it does so for every shape, blitting an RGBA32F answer into
the model image, and writes its sets only between frames, when the composition's
generation changes. `vulkan-plan-abi` and `vulkan-constants` check the fetched
shaders and the model tools' entry list against the port. `common/vulkan_plan.hpp`
holds the constants that the plan takes from `linux/build/arch/rdna4.sh` and
from `nr_graph.cpp`'s defaults at `a75ac49`; `vulkan-constants` checks those
that the markers in `pipelines.json` record and the tile of `gemmvqkvnorms` that
its defines set, and a change of the pin must re-check the rest.

Preserve `layer/ATTRIBUTION.md` and all inherited notices of upstream
code. The layer's shader/dispatch lineage includes OptiScaler and
Dagherbou/OptiScaler_DLSSNR, with RenoDX color-composition attribution. The
Khronos Vulkan/video headers and stb retain their own licenses; stb's license
text ends each header in `third_party/stb/`. The BCUS shader also identifies
Microsoft MiniEngine/Minigraph and James Stanard.

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
