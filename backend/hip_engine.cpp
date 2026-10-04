// SPDX-License-Identifier: MIT
#include "hip_engine.hpp"
#include "codec.hpp"
#include "control_selftest.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace dlsslop {

namespace selftest = control_selftest;

Result<int> select_device(const hip::Api& api, int requested)
{
    DLSSLOP_TRY(api.check(api.hipInit(0), "hipInit (check /dev/kfd permissions and ROCm userspace)"));
    int count = 0, version = 0, selected = -1;
    DLSSLOP_TRY(api.check(api.hipRuntimeGetVersion(&version), "HIP runtime version"));
    DLSSLOP_TRY(api.check(api.hipGetDeviceCount(&count), "HIP device count"));
    std::fprintf(stderr, "HIP runtime=%d, visible devices=%d\n", version, count);
    for (int i = 0; i < count; ++i) {
        hip::DeviceProperties p{};
        DLSSLOP_TRY(api.check(api.hipGetDevicePropertiesR0600(&p, i), "device properties"));
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

Result<hip::Api> open_hip(Options& o)
{
    const auto api = DLSSLOP_TRY(hip::load());
    o.device = DLSSLOP_TRY(select_device(api, o.device));
    // Wait for the device asleep, not spinning a core per waiting thread
    // (hipDeviceScheduleBlockingSync), before the device's context exists.
    if (api.hipSetDeviceFlags) {
        DLSSLOP_TRY(api.check(api.hipSetDevice(o.device), "select the HIP device"));
        DLSSLOP_TRY(api.check(api.hipSetDeviceFlags(4), "wait for the device without spinning"));
    }
    return api;
}

HipEngine::HipEngine(Options o, unsigned tier, const hip::Api& api) : options_(std::move(o)), tier_(tier), api_(api)
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
    temporal_.reset();
    gpu_codec_.reset();
    kernels_.reset();
    network_.reset();
    previous_settings_ = {};
}

Result<void> HipEngine::prepare()
{
    const NativeTier& raster = *ShmNativeTier(tier_);
    // What follows is allocated on the device this thread selects.
    DLSSLOP_TRY(api_.check(api_.hipSetDevice(options_.device), "select the HIP device"));
    // With default flags, the stream's work stays in order with the null
    // stream's synchronous copies, which the CPU codec and diagnostics make.
    if (!stream_) DLSSLOP_TRY(api_.check(api_.hipStreamCreate(&stream_), "create the HIP stream"));
    const auto plan = DLSSLOP_TRY(hip::plan(raster.width, raster.networkHeight, options_.performance));
    const auto placement = DLSSLOP_TRY(hip::place(plan));
    if (!model_) {
        // A model that failed to load is freed, so that no later network binds it.
        if (auto loaded = model_.emplace(api_).load(options_.modules, options_.assets, plan.weights); !loaded) {
            model_.reset();
            return loaded;
        }
    }
    DLSSLOP_TRY(network_.emplace(api_, *model_, stream_).build(plan, placement));
    for (auto& event : marks_) DLSSLOP_TRY(api_.check(api_.hipEventCreate(&event), "create timing event"));
    // Tuning, colour and motion use the module's kernels with the CPU codec too.
    kernels_.emplace(api_, stream_);
    DLSSLOP_TRY(kernels_->load(options_.modules + "/linux_native.hsaco"));
    if (!options_.cpu_codec) {
        gpu_codec_.emplace(*kernels_);
        DLSSLOP_TRY(gpu_codec_->init());
    }
    temporal_.emplace(*kernels_);
    const size_t pixels = size_t(raster.width) * raster.networkHeight;
    DLSSLOP_TRY(api_.check(api_.hipMalloc(&device_input_, pixels * 16), "allocate network input"));
    DLSSLOP_TRY(api_.check(api_.hipMalloc(&device_output_, pixels * 12), "allocate network output"));
    // The host copy of the answer serves the CPU codec and the self-test's checks.
    if (!gpu_codec_ || options_.self_test) neural_.resize(pixels * 3);
    // One evaluation, without a history, before the daemon reports itself
    // ready, so that a kernel that cannot launch fails here. It is the
    // network's first frame, which has a buffer assignment of its own.
    DLSSLOP_TRY(api_.check(api_.hipMemsetAsync(device_input_, 0, pixels * 16, stream_), "warm input"));
    DLSSLOP_TRY(network_->enqueue(device_input_, nullptr, device_output_));
    DLSSLOP_TRY(synchronize());
    DLSSLOP_TRY(network_->print_memory());
    if (!options_.self_test) return {};
    // The kernels on synthetic inputs, independent of model weights.
    DLSSLOP_TRY(selftest::check_tuning(*kernels_));
    DLSSLOP_TRY(selftest::check_temporal(*kernels_));
    return selftest::check_codec(*kernels_);
}

bool HipEngine::import_into(unsigned slot, const ShmTransportOffer& offer, Descriptor (&fds)[2])
{
    if (!gpu_codec_) return false;
    Imported next;
    for (unsigned i = 0; i < 2; ++i) {
        hip::MemoryDesc memory{};
        memory.type = 1; // hipExternalMemoryHandleTypeOpaqueFd
        memory.handle.fd = fds[i].fd;
        memory.size = offer.allocation[i];
        hip::BufferDesc buffer{};
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
    DLSSLOP_TRY(api_.check(api_.hipMemcpy(buffer.data(), pointer, buffer.size() * sizeof(float), 2),
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
    const auto g = DLSSLOP_TRY(geometry(w, h, tier_));
    // Said once: the Vulkan model's conditioning, which this network does not have.
    if ((settings.style || !settings.auto_mask || settings.skin_structure != -1) &&
        !std::exchange(warned_conditioning_, true))
        std::fprintf(stderr, "the HIP network ignores style, skin structure and the automatic mask\n");
    const bool tuned = !native_tuning_is_default(settings.tuning);
    const bool colored = settings.color_preserve > 0;
    if ((tuned || colored) && !device_scratch_)
        DLSSLOP_TRY(api_.check(api_.hipMalloc(&device_scratch_, size_t(g.width) * g.height * 12),
                               "allocate post-processing output"));
    if (passes > 1 && !device_feedback_)
        DLSSLOP_TRY(api_.check(api_.hipMalloc(&device_feedback_, size_t(g.width) * g.height * 16),
                               "allocate inter-pass feedback"));
    DLSSLOP_TRY(mark(0));
    if (gpu_codec_) {
        DLSSLOP_TRY(gpu_codec_->encode(input, g, device_input_, settings.fp16, io.slot >= 0));
        if (verify) {
            std::vector<float> reference;
            DLSSLOP_TRY(encode_proxy(input, g, settings.fp16, reference));
            DLSSLOP_TRY(selftest::compare(
                DLSSLOP_TRY(selftest::Buffer::read_pointer(api_, stream_, device_input_, reference.size())), reference,
                "GPU encoder"));
            std::printf("GPU encode vs CPU reference: FP32 bit-identical\n");
            std::fflush(stdout);
        }
    } else {
        DLSSLOP_TRY(encode_proxy(input, g, settings.fp16, input_));
        DLSSLOP_TRY(api_.check(api_.hipMemcpy(device_input_, input_.data(), input_.size() * sizeof(float), 1),
                               "upload encoded frame"));
    }
    DLSSLOP_TRY(mark(1));
    if (settings.motion) {
        // GpuTemporal itself drops the history for a new pass count, quality, grid or placement.
        const bool reset = options_.self_test || !previous_settings_.motion ||
            previous_settings_.fp16 != settings.fp16 || previous_settings_.precision16 != settings.precision16 ||
            !(previous_settings_.tuning == settings.tuning) || previous_settings_.color_preserve != settings.color_preserve;
        previous_settings_.motion = false; // Until the frame completes: a failed one leaves no history.
        // Until end(), reject nothing: the worker would serve on with the
        // history still pending, and the next begin() would fail.
        DLSSLOP_TRY(temporal_->begin(device_input_, g, settings.motion_quality, settings.motion_grid, passes, reset));
    } else {
        temporal_->reset();
    }
    void* answer = device_output_;
    for (unsigned pass = 0; pass < passes; ++pass) {
        void* const pass_input = pass ? device_feedback_ : device_input_;
        if (pass) {
            if (gpu_codec_) {
                DLSSLOP_TRY(gpu_codec_->feedback(answer, pass_input, settings.precision16));
                if (verify) {
                    DLSSLOP_TRY(feedback_neural_rgb(neural_.data(), g, input_, settings.precision16));
                    DLSSLOP_TRY(selftest::compare(
                        DLSSLOP_TRY(selftest::Buffer::read_pointer(api_, stream_, pass_input, input_.size())),
                        input_, "GPU inter-pass feedback"));
                    std::printf("GPU feedback for pass %u/%u vs CPU reference: FP32 bit-identical\n", pass + 1, passes);
                }
            } else {
                DLSSLOP_TRY(feedback_neural_rgb(neural_.data(), g, input_, settings.precision16));
                DLSSLOP_TRY(api_.check(api_.hipMemcpy(pass_input, input_.data(), input_.size() * sizeof(float), 1),
                                       "upload inter-pass feedback"));
            }
        }
        DLSSLOP_TRY(trace_image(trace, g, pass, "input", pass_input, 4));
        // Stages alternate between two buffers, since tuning and colour read
        // neighbours; with motion, the last writes the pass's history slot.
        void* history = nullptr;
        void* stages[] = {device_output_, device_scratch_, device_output_};
        if (settings.motion) {
            history = DLSSLOP_TRY(temporal_->history(pass, pass_input));
            stages[tuned + colored] = DLSSLOP_TRY(temporal_->target(pass));
        }
        DLSSLOP_TRY(network_->enqueue(pass_input, history, stages[0]));
        DLSSLOP_TRY(trace_image(trace, g, pass, "raw", stages[0], 3));
        if (tuned) {
            DLSSLOP_TRY(gpu_tune(*kernels_, g, pass_input, stages[0], stages[1], settings.tuning));
            DLSSLOP_TRY(trace_image(trace, g, pass, "tuned", stages[1], 3));
        }
        if (colored) {
            DLSSLOP_TRY(gpu_preserve_color(*kernels_, g, device_input_, stages[tuned], stages[1 + tuned],
                                           settings.color_preserve));
            DLSSLOP_TRY(trace_image(trace, g, pass, "color", stages[1 + tuned], 3));
        }
        answer = stages[tuned + colored];
        // The CPU codec, like the GPU one, rejects the nonfinite samples it reads.
        if (!gpu_codec_ || verify) {
            DLSSLOP_TRY(synchronize());
            DLSSLOP_TRY(api_.check(api_.hipMemcpy(neural_.data(), answer, neural_.size() * sizeof(float), 2),
                                   "read neural answer"));
        }
    }
    answer_ = answer;
    if (settings.motion) DLSSLOP_TRY(temporal_->end());
    DLSSLOP_TRY(mark(2));
    if (gpu_codec_) {
        DLSSLOP_TRY(gpu_codec_->decode(answer, output));
        DLSSLOP_TRY(mark(3));
        DLSSLOP_TRY(gpu_codec_->finish());
        if (verify) {
            const size_t bpp = settings.fp16 ? 8 : 4;
            std::vector<uint8_t> reference(size_t(w) * h * bpp);
            DLSSLOP_TRY(decode_neural_proxy(input, g, settings.fp16, neural_.data(), reference.data()));
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
        DLSSLOP_TRY(decode_neural_proxy(input, g, settings.fp16, neural_.data(), output));
        DLSSLOP_TRY(mark(3));
    }
    previous_settings_ = settings;
    // finish() waited for the GPU codec's stream already.
    if (!gpu_codec_) DLSSLOP_TRY(api_.check(api_.hipEventSynchronize(marks_[3]), "timing event completion"));
    DLSSLOP_TRY(api_.check(api_.hipEventElapsedTime(&upload_ms, marks_[0], marks_[1]), "upload interval"));
    DLSSLOP_TRY(api_.check(api_.hipEventElapsedTime(&inference_ms, marks_[1], marks_[2]), "inference interval"));
    return api_.check(api_.hipEventElapsedTime(&readback_ms, marks_[2], marks_[3]), "readback interval");
}

Result<const std::vector<float>*> HipEngine::raw_result()
{
    if (gpu_codec_)
        DLSSLOP_TRY(api_.check(api_.hipMemcpy(neural_.data(), answer_, neural_.size() * sizeof(float), 2),
                               "read raw network answer"));
    return &neural_;
}

} // namespace dlsslop
