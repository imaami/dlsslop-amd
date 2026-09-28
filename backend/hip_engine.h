// The HIP network: the pinned lmxxf network and the daemon's own codec, tuning,
// color and motion kernels, on a gfx1201 device.
// SPDX-License-Identifier: MIT
#pragma once
#include "codec.h"
#include "codec_gpu.h"
#include "control_selftest.h"
#include "engine.h"
#include "native_kernels.h"
#include "temporal_gpu.h"
#include "tuning.h"
#include "unwrap.h"
#include "vendor/LmxxfProductionOptions.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dlsslop {
namespace selftest = dlsslop::control_selftest;

int select_device(int requested);

class Engine : public Backend {
    Options options_;
    unsigned tier_;
    std::unique_ptr<hip_reference::Network> network_;
    std::optional<dlsslop::NativeKernels> kernels_;
    std::optional<dlsslop::GpuCodec> gpu_codec_;
    std::optional<dlsslop::GpuTemporal> temporal_;
    void* device_input_ = nullptr; // The encoded frame, unchanged until the next one.
    void* device_feedback_ = nullptr; // Later passes' input, allocated for multi-pass.
    void* device_output_ = nullptr;
    void* device_scratch_ = nullptr; // Tuning or colour: the other stage output.
    void* answer_ = nullptr; // The latest frame's final network answer.
    std::vector<float> encoded_, neural_, feedback_;
    ProcessingSettings previous_settings_;
    bool warned_conditioning_ = false;
    // Stream events: frame start, uploaded, evaluated, answered. Timing never
    // stalls the stream; the intervals are read once the answer is complete.
    hip_probe::Handle marks_[4]{};
    // The pair in each import slot.
    struct Imported {
        hip_probe::Handle memory[2]{};
        void* frame[2]{}; // proxy, answer
    };
    std::array<Imported, kSlots> imported_{};

    void release(Imported& slot)
    {
        auto& api = network_->Runtime();
        for (unsigned i = 0; i < 2; ++i) {
            if (slot.frame[i]) api.hipFree(slot.frame[i]);
            if (slot.memory[i]) api.hipDestroyExternalMemory(slot.memory[i]);
        }
        slot = {};
    }

    void mark(unsigned i)
    {
        auto& api = network_->Runtime();
        api.Check(api.hipEventRecord(marks_[i], network_->Stream()), "record timing event");
    }
public:
    Engine(Options o, unsigned tier) : options_(std::move(o)), tier_(tier) {}
    const char* name() const override { return "HIP"; }
    std::string device() const override { return "device " + std::to_string(options_.device); }
    unsigned tier() const override { return tier_; }
    std::string processing() const override
    {
        const NativeTier& raster = *ShmNativeTier(tier_);
        return "processing=" + std::to_string(raster.width) + "x" + std::to_string(raster.networkHeight);
    }
    // The members go next, in reverse order: the helpers before the network whose runtime they use.
    ~Engine()
    {
        if (!network_) return;
        auto& api = network_->Runtime();
        api.hipStreamSynchronize(network_->Stream());
        for (auto& slot : imported_) release(slot);
        for (auto event : marks_) api.hipEventDestroy(event);
        for (void* buffer : {device_input_, device_feedback_, device_output_, device_scratch_})
            if (buffer) api.hipFree(buffer);
    }
    void prepare() override
    {
        if (options_.test_identity) return;
        const NativeTier& raster = *ShmNativeTier(tier_);
        auto opt = LmxxfProductionOptions(raster.width, raster.networkHeight, options_.modules, options_.assets);
        opt.device = static_cast<unsigned>(options_.device);
        if (!options_.performance) opt.skip_blocks.clear();
        // Upstream's shipped HIP configurations (scripts/hip-*-flags.txt) add
        // these bit-exact byte residual and fragment paths to the snapshot.
        opt.mh_feature_byte = opt.mh_proj_diag_fb = opt.mh_byte_stream = opt.decoder_byte = opt.mh_ffn_frag256 = true;
        // Production launches nothing from the WMMA, tiled, wave and fused-C32
        // modules these select; do not load them.
        opt.wmma = opt.tiled = opt.wave = opt.fused_c32 = false;
        network_ = std::make_unique<hip_reference::Network>(opt);
        network_->SetNoise({}); // Fast prefix uses procedural noise, not noise.f32.
        auto& api = network_->Runtime();
        for (auto& event : marks_) api.Check(api.hipEventCreate(&event), "create timing event");
        // Tuning, colour and motion use the module's kernels with the CPU codec too.
        kernels_.emplace(api, network_->Stream(), options_.modules + "/linux_native.hsaco");
        if (!options_.cpu_codec) gpu_codec_.emplace(*kernels_);
        temporal_.emplace(*kernels_);
        const size_t pixels = size_t(raster.width) * raster.networkHeight;
        api.Check(api.hipMalloc(&device_input_, pixels * 16), "allocate network input");
        api.Check(api.hipMalloc(&device_output_, pixels * 12), "allocate network output");
        // The host copy of the answer serves the CPU codec and the self-test's checks.
        if (!gpu_codec_ || options_.self_test) neural_.resize(pixels * 3);
        // Warm once before announcing readiness: upstream allocates weights and
        // scratch lazily, whatever the input; warmup has no temporal history.
        api.Check(api.hipMemsetAsync(device_input_, 0, pixels * 16, network_->Stream()), "warm input");
        network_->Enqueue(device_input_, nullptr, device_output_, 0);
        network_->Synchronize();
        network_->PrintMemory();
        if (options_.self_test) { // The kernels on synthetic inputs, independent of model weights.
            selftest::check_tuning(*kernels_);
            selftest::check_temporal(*kernels_);
            selftest::check_codec(*kernels_);
        }
    }
    // See GpuCodec::pin.
    void pin(uint8_t* input, uint8_t* output, size_t bytes) override
    {
        if (gpu_codec_) gpu_codec_->pin(input, output, bytes);
    }
    bool import_into(unsigned slot, const ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2]) override
    {
        if (!gpu_codec_) return false;
        auto& api = network_->Runtime();
        Imported next;
        for (unsigned i = 0; i < 2; ++i) {
            hip_probe::MemoryDesc memory{};
            memory.type = 1; // hipExternalMemoryHandleTypeOpaqueFd
            memory.handle.fd = fds[i].fd;
            memory.size = offer.allocation[i];
            hip_probe::BufferDesc buffer{};
            buffer.size = offer.allocation[i];
            if (api.hipImportExternalMemory(&next.memory[i], &memory)) {
                release(next);
                return false;
            }
            fds[i].fd = -1;
            if (api.hipExternalMemoryGetMappedBuffer(&next.frame[i], next.memory[i], &buffer)) {
                release(next);
                return false;
            }
        }
        release(imported_[slot]);
        imported_[slot] = next;
        return true;
    }
    Frames frames_of(unsigned slot) const override
    {
        const Imported& pair = imported_[slot];
        return {static_cast<const uint8_t*>(pair.frame[0]), static_cast<uint8_t*>(pair.frame[1]), int(slot)};
    }
    void read_back(void* host, const void* source, size_t bytes) override
    {
        auto& api = network_->Runtime();
        api.Check(api.hipMemcpy(host, source, bytes, 4), "read diagnostic frame");
    }
    void infer(const Frames& io, unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings = {},
               dlsslop::FrameTrace* trace = nullptr) override
    {
        infer(io.proxy, w, h, io.answer, passes, settings, trace);
    }
    void self_test(const Options& o) override { run_self_test(o, *this); }
    // Input and output are w * h RGBA8, or RGBA16F with settings.fp16. verify
    // checks the GPU codec against the CPU reference inside the timed frame.
    void infer(const uint8_t* input, unsigned w, unsigned h, uint8_t* output, unsigned passes,
               const ProcessingSettings& settings = {}, dlsslop::FrameTrace* trace = nullptr,
               bool verify = false)
    {
        if (options_.test_identity) {
            std::memcpy(output, input, size_t(w) * h * (settings.fp16 ? 8 : 4));
            return;
        }
        if (passes > 1 || options_.self_test || settings.motion) {
            // The optional upstream approximate cache has one history, not
            // one history per pass. Do not silently mix those states.
            const char* adaptive = std::getenv("DLSS5_VIT_ADAPTIVE");
            if (adaptive && std::strtoul(adaptive, nullptr, 10))
                throw std::range_error("multi-pass/motion/self-test requires DLSS5_VIT_ADAPTIVE=0 (uncached inference)");
        }
        const auto g = unwrap(dlsslop::geometry(w, h, tier_));
        auto& api = network_->Runtime();
        const auto trace_image = [&](unsigned pass, const char* stage, const void* pointer, unsigned channels) {
            if (!trace) return;
            network_->Synchronize();
            std::vector<float> buffer(std::size_t(g.width) * g.height * channels);
            api.Check(api.hipMemcpy(buffer.data(), pointer, buffer.size() * sizeof(float), 2),
                      "read diagnostic neural stage");
            char name[32];
            std::snprintf(name, sizeof name, "pass-%02u-%s", pass + 1, stage);
            trace->image(name, buffer.data(), g, channels);
        };
        if (settings.fp16 && options_.cpu_compose)
            throw std::range_error("FP16 proxy transport requires Vulkan composition; disable --cpu-compose");
        // Said once: the Vulkan model's conditioning, which this network does not have.
        if ((settings.style || !settings.auto_mask || settings.skin_structure != -1) &&
            !std::exchange(warned_conditioning_, true))
            std::fprintf(stderr, "the HIP network ignores style, skin structure and the automatic mask\n");
        const bool tuned = !dlsslop::native_tuning_is_default(settings.tuning);
        const bool colored = settings.color_preserve > 0;
        if ((tuned || colored) && !device_scratch_)
            api.Check(api.hipMalloc(&device_scratch_, size_t(g.width) * g.height * 12), "allocate post-processing output");
        if (passes > 1 && !device_feedback_)
            api.Check(api.hipMalloc(&device_feedback_, size_t(g.width) * g.height * 16), "allocate inter-pass feedback");
        mark(0);
        if (gpu_codec_) {
            gpu_codec_->encode(input, g, device_input_, settings.fp16);
            if (verify) {
                std::vector<float> reference;
                unwrap(dlsslop::encode_proxy(input, g, settings.fp16, reference));
                selftest::compare(selftest::Buffer::read_pointer(api, network_->Stream(), device_input_, reference.size()),
                                  reference, "GPU encoder");
                std::printf("GPU encode vs CPU reference: FP32 bit-identical\n");
                std::fflush(stdout);
            }
        } else {
            unwrap(dlsslop::encode_proxy(input, g, settings.fp16, encoded_));
            api.Check(api.hipMemcpy(device_input_, encoded_.data(), encoded_.size() * sizeof(float), 1), "upload encoded frame");
        }
        mark(1);
        if (settings.motion) {
            // GpuTemporal itself drops the history for a new pass count, quality, grid or placement.
            const bool reset = options_.self_test || !previous_settings_.motion ||
                previous_settings_.fp16 != settings.fp16 || previous_settings_.precision16 != settings.precision16 ||
                !(previous_settings_.tuning == settings.tuning) || previous_settings_.color_preserve != settings.color_preserve;
            previous_settings_.motion = false; // Until the frame completes: a rejected one leaves no history.
            // Until end(), throw no std::range_error: the worker would serve on with the
            // history still pending, and the next begin() would fail.
            temporal_->begin(device_input_, g, settings.motion_quality, settings.motion_grid, passes, reset);
        } else {
            temporal_->reset();
        }
        void* answer = device_output_;
        for (unsigned pass = 0; pass < passes; ++pass) {
            void* const pass_input = pass ? device_feedback_ : device_input_;
            if (pass) {
                if (gpu_codec_) {
                    gpu_codec_->feedback(answer, pass_input, settings.precision16);
                    if (verify) {
                        unwrap(dlsslop::feedback_neural_rgb(neural_.data(), g, feedback_, settings.precision16));
                        selftest::compare(selftest::Buffer::read_pointer(api, network_->Stream(), pass_input,
                                          feedback_.size()), feedback_, "GPU inter-pass feedback");
                        std::printf("GPU feedback for pass %u/%u vs CPU reference: FP32 bit-identical\n",
                                    pass + 1, passes);
                    }
                } else {
                    // Retain the initial encoded_ for final CPU composition.
                    unwrap(dlsslop::feedback_neural_rgb(neural_.data(), g, feedback_, settings.precision16));
                    api.Check(api.hipMemcpy(pass_input, feedback_.data(), feedback_.size() * sizeof(float), 1),
                              "upload inter-pass feedback");
                }
            }
            trace_image(pass, "input", pass_input, 4);
            // Stages alternate between two buffers, since tuning and colour read
            // neighbours; with motion, the last writes the pass's history slot.
            void* history = nullptr;
            void* stages[] = {device_output_, device_scratch_, device_output_};
            if (settings.motion) {
                history = temporal_->history(pass, pass_input);
                stages[tuned + colored] = temporal_->target(pass);
            }
            // Graph replay (upstream o.graph, off) would need one stable rgb_output.
            network_->Enqueue(pass_input, history, stages[0], 0);
            trace_image(pass, "raw", stages[0], 3);
            if (tuned) {
                dlsslop::gpu_tune(*kernels_, g, pass_input, stages[0], stages[1], settings.tuning);
                trace_image(pass, "tuned", stages[1], 3);
            }
            if (colored) {
                dlsslop::gpu_preserve_color(*kernels_, g, device_input_, stages[tuned], stages[1 + tuned],
                                            settings.color_preserve);
                trace_image(pass, "color", stages[1 + tuned], 3);
            }
            answer = stages[tuned + colored];
            // The CPU codec, like the GPU one, rejects the nonfinite samples it reads.
            if (!gpu_codec_ || verify) {
                network_->Synchronize();
                api.Check(api.hipMemcpy(neural_.data(), answer, neural_.size() * sizeof(float), 2),
                          "read neural answer");
            }
        }
        answer_ = answer;
        if (settings.motion) temporal_->end();
        mark(2);
        if (gpu_codec_) {
            gpu_codec_->decode(answer, output);
            mark(3);
            gpu_codec_->finish();
            if (verify) {
                const size_t bpp = settings.fp16 ? 8 : 4;
                std::vector<uint8_t> reference(size_t(w) * h * bpp);
                unwrap(dlsslop::decode_neural_proxy(input, g, settings.fp16, neural_.data(), reference.data()));
                const size_t first = std::mismatch(reference.begin(), reference.end(), output).first - reference.begin();
                if (first < reference.size()) {
                    std::fprintf(stderr, "GPU decoder first mismatch: x=%zu y=%zu byte=%zu GPU=%u CPU=%u\n",
                                 (first / bpp) % w, (first / bpp) / w, first % bpp,
                                 unsigned(output[first]), unsigned(reference[first]));
                    throw std::runtime_error("GPU decoder disagrees with CPU reference");
                }
                std::printf("GPU decode vs CPU reference: bit-identical\n");
                std::fflush(stdout);
            }
        } else {
            if (options_.cpu_compose)
                unwrap(dlsslop::decode_rgba8(input, g, encoded_.data(), neural_.data(), output));
            else
                unwrap(dlsslop::decode_neural_proxy(input, g, settings.fp16, neural_.data(), output));
            mark(3);
        }
        previous_settings_ = settings;
        api.Check(api.hipEventSynchronize(marks_[3]), "timing event completion");
        api.Check(api.hipEventElapsedTime(&upload_ms, marks_[0], marks_[1]), "upload interval");
        api.Check(api.hipEventElapsedTime(&inference_ms, marks_[1], marks_[2]), "inference interval");
        api.Check(api.hipEventElapsedTime(&readback_ms, marks_[2], marks_[3]), "readback interval");
    }
    // The latest infer()'s raw network answer, read back outside its timing.
    const std::vector<float>& raw_result()
    {
        if (gpu_codec_) {
            auto& api = network_->Runtime();
            api.Check(api.hipMemcpy(neural_.data(), answer_, neural_.size() * sizeof(float), 2),
                      "read raw network answer");
        }
        return neural_;
    }
};
} // namespace dlsslop
