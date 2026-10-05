// SPDX-License-Identifier: MIT
#include "hip_engine.hpp"
#include "control_selftest.h"
#include "geometry.h"
#include "reference.h"
#include "tuning_math.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace dlsslop {
namespace {
// A HIP call's RESULT: nothing, or hip_fail()'s words for WHAT.
[[gnu::cold, gnu::noinline]] Result<void> hip_error(const struct hip_api& api, int result, const char* what)
{
    struct error e;
    return forward_c(hip_fail(&e, &api, result, "%s", what), e);
}
Result<void> check(const struct hip_api& api, int result, const char* what)
{
    if (!result) return {};
    return hip_error(api, result, what);
}
} // namespace

Result<int> select_device(const struct hip_api& api, int requested)
{
    DLSSLOP_TRY(check(api, api.hipInit(0), "hipInit (check /dev/kfd permissions and ROCm userspace)"));
    int count = 0, version = 0, selected = -1;
    DLSSLOP_TRY(check(api, api.hipRuntimeGetVersion(&version), "HIP runtime version"));
    DLSSLOP_TRY(check(api, api.hipGetDeviceCount(&count), "HIP device count"));
    std::fprintf(stderr, "HIP runtime=%d, visible devices=%d\n", version, count);
    for (int i = 0; i < count; ++i) {
        struct hip_device_properties p{};
        DLSSLOP_TRY(check(api, api.hipGetDevicePropertiesR0600(&p, i), "device properties"));
        std::fprintf(stderr, "  device %d: %s; arch=%s; PCI=%04x:%02x:%02x; VRAM=%.0f MiB\n",
                     i, p.name, p.gcnArchName, p.pciDomainID, p.pciBusID, p.pciDeviceID,
                     p.totalGlobalMem / 1048576.0);
        const bool gfx1201 = !std::strncmp(p.gcnArchName, "gfx1201", 7) &&
                             (!p.gcnArchName[7] || p.gcnArchName[7] == ':');
        if (gfx1201 && ((requested < 0 && selected < 0) || requested == i)) selected = i;
    }
    if (selected < 0)
        return fail(requested < 0 ? "no gfx1201 device found (RX 9070/9070 XT required); check HIP_VISIBLE_DEVICES"
                                  : "selected device is unavailable or is not gfx1201");
    return selected;
}

void HipEngine::release(Imported& slot)
{
    for (unsigned i = 0; i < 2; ++i) {
        if (slot.frame[i]) api_.hipFree(slot.frame[i]);
        if (slot.memory[i]) api_.hipDestroyExternalMemory(slot.memory[i]);
    }
    slot = {};
}

Result<struct hip_api> open_hip(Options& o)
{
    struct hip_api api;
    struct error e;
    if (const enum error_code code = hip_load(&api, &e)) return forward_c(code, e);
    o.device = DLSSLOP_TRY(select_device(api, o.device));
    // Wait for the device asleep, not spinning a core per waiting thread
    // (hipDeviceScheduleBlockingSync), before the device's context exists.
    if (api.hipSetDeviceFlags) {
        DLSSLOP_TRY(check(api, api.hipSetDevice(o.device), "select the HIP device"));
        DLSSLOP_TRY(check(api, api.hipSetDeviceFlags(4), "wait for the device without spinning"));
    }
    return api;
}

Result<void> HipEngine::fail_hip(int result, const char* what) const { return hip_error(api_, result, what); }

HipEngine::HipEngine(Options o, unsigned tier, const struct hip_api& api) : options_(std::move(o)), tier_(tier), api_(api)
{
    // Upstream's approximate ViT cache, which this variable selected, is not
    // part of the port.
    if (const char* adaptive = std::getenv("DLSS5_VIT_ADAPTIVE"); adaptive && std::strcmp(adaptive, "0"))
        std::fprintf(stderr, "DLSS5_VIT_ADAPTIVE is not supported; the HIP network runs every ViT block\n");
}

void HipEngine::release()
{
    if (!stream_) return;
    // Once the stream is done, nothing prepare() made is in use.
    api_.hipStreamSynchronize(stream_);
    for (auto& slot : imported_) release(slot);
    for (auto& event : marks_) {
        if (event) api_.hipEventDestroy(event);
        event = nullptr;
    }
    for (void** buffer : {&device_input_, &device_feedback_, &device_output_, &device_scratch_}) {
        if (*buffer) api_.hipFree(*buffer);
        *buffer = nullptr;
    }
    answer_ = nullptr;
    temporal_gpu_fini(&temporal_);
    codec_gpu_fini(&gpu_codec_);
    native_kernels_fini(&kernels_);
    hip_network_fini(&network_);
    previous_settings_ = {};
}

Result<void> HipEngine::prepare()
{
    const NativeTier& raster = *ShmNativeTier(tier_);
    // What follows is allocated on the device this thread selects.
    DLSSLOP_TRY(check(api_.hipSetDevice(options_.device), "select the HIP device"));
    // With default flags, the stream's work stays in order with the null
    // stream's synchronous copies, which the CPU codec and diagnostics make.
    if (!stream_) DLSSLOP_TRY(check(api_.hipStreamCreate(&stream_), "create the HIP stream"));
    // The plan and its placement, which the model and the network read, freed on return.
    struct Planned {
        hip_plan plan{};
        hip_placement placement{};
        ~Planned()
        {
            hip_placement_fini(&placement);
            hip_plan_fini(&plan);
        }
    } planned;
    struct error e;
    if (const enum error_code code =
            hip_plan_init(&planned.plan, raster.width, raster.networkHeight, options_.performance, &e))
        return forward_c(code, e);
    if (const enum error_code code = hip_plan_place(&planned.placement, &planned.plan, &e)) return forward_c(code, e);
    // A model that failed to load is freed, so that no later network binds it.
    if (!model_.api)
        if (const enum error_code code =
                hip_model_init(&model_, &api_, options_.modules.data(), options_.modules.size(), options_.assets.data(),
                               options_.assets.size(), planned.plan.weights, planned.plan.weight_count, &e))
            return forward_c(code, e);
    if (const enum error_code code =
            hip_network_init(&network_, &api_, &model_, stream_, &planned.plan, &planned.placement, &e))
        return forward_c(code, e);
    for (auto& event : marks_) DLSSLOP_TRY(check(api_.hipEventCreate(&event), "create timing event"));
    // Tuning, colour and motion use the module's kernels with the CPU codec too.
    if (const enum error_code code =
            native_kernels_init(&kernels_, &api_, stream_, (options_.modules + "/linux_native.hsaco").c_str(), &e))
        return forward_c(code, e);
    if (!options_.cpu_codec)
        if (const enum error_code code = codec_gpu_init(&gpu_codec_, &kernels_, &e)) return forward_c(code, e);
    temporal_ = temporal_gpu(&kernels_);
    const size_t pixels = size_t(raster.width) * raster.networkHeight;
    DLSSLOP_TRY(check(api_.hipMalloc(&device_input_, pixels * 16), "allocate network input"));
    DLSSLOP_TRY(check(api_.hipMalloc(&device_output_, pixels * 12), "allocate network output"));
    // The host copies of a pass's input and answer serve the CPU codec and the self-test's checks.
    if (!gpu_codec_.kernels || options_.self_test) {
        input_.resize(pixels * 4);
        neural_.resize(pixels * 3);
    }
    // One evaluation, without a history, before the daemon reports itself
    // ready, so that a kernel that cannot launch fails here. It is the
    // network's first frame, which has a buffer assignment of its own.
    DLSSLOP_TRY(check(api_.hipMemsetAsync(device_input_, 0, pixels * 16, stream_), "warm input"));
    if (const enum error_code code = hip_network_enqueue(&network_, device_input_, nullptr, device_output_, &e))
        return forward_c(code, e);
    DLSSLOP_TRY(synchronize());
    if (const enum error_code code = hip_network_print_memory(&network_, &e)) return forward_c(code, e);
    if (!options_.self_test) return {};
    // The kernels on synthetic inputs, independent of model weights.
    if (const enum error_code code = control_selftest_run(&kernels_, &e)) return forward_c(code, e);
    return {};
}

bool HipEngine::import_into(unsigned slot, const ShmTransportOffer& offer, Descriptor (&fds)[2])
{
    if (!gpu_codec_.kernels) return false;
    Imported next;
    for (unsigned i = 0; i < 2; ++i) {
        struct hip_memory_desc memory{};
        memory.type = 1; // hipExternalMemoryHandleTypeOpaqueFd
        memory.handle.fd = fds[i].fd;
        memory.size = offer.allocation[i];
        struct hip_buffer_desc buffer{};
        buffer.size = offer.allocation[i];
        if (api_.hipImportExternalMemory(&next.memory[i], &memory)) {
            release(next);
            return false;
        }
        fds[i].fd = -1;
        if (api_.hipExternalMemoryGetMappedBuffer(&next.frame[i], next.memory[i], &buffer)) {
            release(next);
            return false;
        }
    }
    release(imported_[slot]);
    imported_[slot] = next;
    return true;
}

Result<void> HipEngine::trace_image(FrameTrace* trace, const struct geometry& g, unsigned pass, const char* stage,
                                    const void* pointer, unsigned channels)
{
    if (!trace) return {};
    DLSSLOP_TRY(synchronize());
    std::vector<float> buffer(std::size_t(g.width) * g.height * channels);
    DLSSLOP_TRY(check(api_.hipMemcpy(buffer.data(), pointer, buffer.size() * sizeof(float), 2),
                      "read diagnostic neural stage"));
    char name[32];
    std::snprintf(name, sizeof name, "pass-%02u-%s", pass + 1, stage);
    trace->image(name, buffer.data(), g, channels);
    return {};
}

Result<void> HipEngine::infer(const Frames& io, unsigned w, unsigned h, unsigned passes,
                              const ProcessingSettings& settings, FrameTrace* trace, bool verify)
{
    const uint8_t* const input = io.proxy;
    uint8_t* const output = io.answer;
    struct error e;
    struct geometry g;
    if (const enum error_code code = geometry_init(&g, w, h, tier_, &e)) return forward_c(code, e);
    // Said once: the Vulkan model's conditioning, which this network does not have.
    if ((settings.style || !settings.auto_mask || settings.skin_structure != -1) &&
        !std::exchange(warned_conditioning_, true))
        std::fprintf(stderr, "the HIP network ignores style, skin structure and the automatic mask\n");
    const bool tuned = !native_tuning_is_default(settings.tuning);
    const bool colored = settings.color_preserve > 0;
    if ((tuned || colored) && !device_scratch_)
        DLSSLOP_TRY(check(api_.hipMalloc(&device_scratch_, size_t(g.width) * g.height * 12),
                          "allocate post-processing output"));
    if (passes > 1 && !device_feedback_)
        DLSSLOP_TRY(check(api_.hipMalloc(&device_feedback_, size_t(g.width) * g.height * 16),
                          "allocate inter-pass feedback"));
    DLSSLOP_TRY(mark(0));
    if (gpu_codec_.kernels) {
        if (const enum error_code code =
                codec_gpu_encode(&gpu_codec_, input, &g, device_input_, settings.fp16, io.slot >= 0, &e))
            return forward_c(code, e);
        if (verify) {
            std::vector<float> reference(size_t(g.width) * g.height * 4), actual(reference.size());
            if (const enum error_code code = reference_encode_proxy(input, &g, settings.fp16, reference.data(), &e))
                return forward_c(code, e);
            if (const enum error_code code =
                    control_selftest_read(&kernels_, device_input_, actual.data(), actual.size(), &e))
                return forward_c(code, e);
            if (const enum error_code code =
                    control_selftest_compare(actual.data(), reference.data(), reference.size(), "GPU encoder", &e))
                return forward_c(code, e);
            std::printf("GPU encode vs CPU reference: FP32 bit-identical\n");
            std::fflush(stdout);
        }
    } else {
        if (const enum error_code code = reference_encode_proxy(input, &g, settings.fp16, input_.data(), &e))
            return forward_c(code, e);
        DLSSLOP_TRY(check(api_.hipMemcpy(device_input_, input_.data(), input_.size() * sizeof(float), 1),
                          "upload encoded frame"));
    }
    DLSSLOP_TRY(mark(1));
    if (settings.motion) {
        // temporal_gpu_begin() itself drops the history for a new pass count, quality, grid or placement.
        const bool reset = options_.self_test || !previous_settings_.motion ||
            previous_settings_.fp16 != settings.fp16 || previous_settings_.precision16 != settings.precision16 ||
            !(previous_settings_.tuning == settings.tuning) || previous_settings_.color_preserve != settings.color_preserve;
        previous_settings_.motion = false; // Until the frame completes: a failed one leaves no history.
        // Until end(), reject nothing: the worker would serve on with the
        // history still pending, and the next begin() would fail.
        if (const enum error_code code = temporal_gpu_begin(&temporal_, device_input_, &g, settings.motion_quality,
                                                            settings.motion_grid, passes, reset, &e))
            return forward_c(code, e);
    } else {
        temporal_gpu_reset(&temporal_);
    }
    void* answer = device_output_;
    for (unsigned pass = 0; pass < passes; ++pass) {
        void* const pass_input = pass ? device_feedback_ : device_input_;
        if (pass) {
            if (gpu_codec_.kernels) {
                if (const enum error_code code =
                        codec_gpu_feedback(&gpu_codec_, answer, pass_input, settings.precision16, &e))
                    return forward_c(code, e);
                if (verify) {
                    if (const enum error_code code = reference_feedback_neural_rgb(neural_.data(), &g,
                                                                                   settings.precision16,
                                                                                   input_.data(), &e))
                        return forward_c(code, e);
                    std::vector<float> actual(input_.size());
                    if (const enum error_code code =
                            control_selftest_read(&kernels_, pass_input, actual.data(), actual.size(), &e))
                        return forward_c(code, e);
                    if (const enum error_code code = control_selftest_compare(actual.data(), input_.data(),
                                                                              input_.size(),
                                                                              "GPU inter-pass feedback", &e))
                        return forward_c(code, e);
                    std::printf("GPU feedback for pass %u/%u vs CPU reference: FP32 bit-identical\n", pass + 1, passes);
                }
            } else {
                if (const enum error_code code = reference_feedback_neural_rgb(neural_.data(), &g, settings.precision16,
                                                                               input_.data(), &e))
                    return forward_c(code, e);
                DLSSLOP_TRY(check(api_.hipMemcpy(pass_input, input_.data(), input_.size() * sizeof(float), 1),
                                  "upload inter-pass feedback"));
            }
        }
        DLSSLOP_TRY(trace_image(trace, g, pass, "input", pass_input, 4));
        // Stages alternate between two buffers, since tuning and colour read
        // neighbours; with motion, the last writes the pass's history slot.
        void* history = nullptr;
        void* stages[] = {device_output_, device_scratch_, device_output_};
        if (settings.motion) {
            if (const enum error_code code = temporal_gpu_history(&temporal_, pass, pass_input, &history, &e))
                return forward_c(code, e);
            if (const enum error_code code = temporal_gpu_target(&temporal_, pass, &stages[tuned + colored], &e))
                return forward_c(code, e);
        }
        if (const enum error_code code = hip_network_enqueue(&network_, pass_input, history, stages[0], &e))
            return forward_c(code, e);
        DLSSLOP_TRY(trace_image(trace, g, pass, "raw", stages[0], 3));
        if (tuned) {
            if (const enum error_code code =
                    native_kernels_tune(&kernels_, g, pass_input, stages[0], stages[1], settings.tuning, &e))
                return forward_c(code, e);
            DLSSLOP_TRY(trace_image(trace, g, pass, "tuned", stages[1], 3));
        }
        if (colored) {
            if (const enum error_code code = native_kernels_preserve_color(
                    &kernels_, g, device_input_, stages[tuned], stages[1 + tuned], settings.color_preserve, &e))
                return forward_c(code, e);
            DLSSLOP_TRY(trace_image(trace, g, pass, "color", stages[1 + tuned], 3));
        }
        answer = stages[tuned + colored];
        // The CPU codec, like the GPU one, rejects the nonfinite samples it reads.
        if (!gpu_codec_.kernels || verify) {
            DLSSLOP_TRY(synchronize());
            DLSSLOP_TRY(check(api_.hipMemcpy(neural_.data(), answer, neural_.size() * sizeof(float), 2),
                              "read neural answer"));
        }
    }
    answer_ = answer;
    if (settings.motion)
        if (const enum error_code code = temporal_gpu_end(&temporal_, &e)) return forward_c(code, e);
    DLSSLOP_TRY(mark(2));
    if (gpu_codec_.kernels) {
        if (const enum error_code code = codec_gpu_decode(&gpu_codec_, answer, output, &e)) return forward_c(code, e);
        DLSSLOP_TRY(mark(3));
        if (const enum error_code code = codec_gpu_finish(&gpu_codec_, &e)) return forward_c(code, e);
        if (verify) {
            const size_t bpp = settings.fp16 ? 8 : 4;
            std::vector<uint8_t> reference(size_t(w) * h * bpp);
            if (const enum error_code code = reference_decode_neural_proxy(input, &g, settings.fp16, neural_.data(),
                                                                           reference.data(), &e))
                return forward_c(code, e);
            const size_t first = std::mismatch(reference.begin(), reference.end(), output).first - reference.begin();
            if (first < reference.size()) {
                std::fprintf(stderr, "GPU decoder first mismatch: x=%zu y=%zu byte=%zu GPU=%u CPU=%u\n",
                             (first / bpp) % w, (first / bpp) / w, first % bpp,
                             unsigned(output[first]), unsigned(reference[first]));
                return fail("GPU decoder disagrees with CPU reference");
            }
            std::printf("GPU decode vs CPU reference: bit-identical\n");
            std::fflush(stdout);
        }
    } else {
        if (const enum error_code code = reference_decode_neural_proxy(input, &g, settings.fp16, neural_.data(), output,
                                                                       &e))
            return forward_c(code, e);
        DLSSLOP_TRY(mark(3));
    }
    previous_settings_ = settings;
    // codec_gpu_finish() waited for the GPU codec's stream already.
    if (!gpu_codec_.kernels) DLSSLOP_TRY(check(api_.hipEventSynchronize(marks_[3]), "timing event completion"));
    DLSSLOP_TRY(check(api_.hipEventElapsedTime(&upload_ms, marks_[0], marks_[1]), "upload interval"));
    DLSSLOP_TRY(check(api_.hipEventElapsedTime(&inference_ms, marks_[1], marks_[2]), "inference interval"));
    return check(api_.hipEventElapsedTime(&readback_ms, marks_[2], marks_[3]), "readback interval");
}

Result<const std::vector<float>*> HipEngine::raw_result()
{
    if (gpu_codec_.kernels)
        DLSSLOP_TRY(check(api_.hipMemcpy(neural_.data(), answer_, neural_.size() * sizeof(float), 2),
                          "read raw network answer"));
    return &neural_;
}

} // namespace dlsslop
