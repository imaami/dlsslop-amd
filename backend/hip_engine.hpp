// The HIP network, a port of lmxxf's, and the daemon's own codec, tuning,
// color and motion kernels, on a gfx1201 device.
// SPDX-License-Identifier: MIT
#pragma once
#include "codec_gpu.h"
#include "engine.hpp"
#include "hip.h"
#include "hip_network.h"
#include "native_kernels.h"
#include "temporal_gpu.h"

#include <array>
#include <vector>

namespace dlsslop {

// The gfx1201 device that --device names, or the first one; every visible
// device is listed on the way.
Result<int> select_device(const struct hip_api& api, int requested);
// The HIP runtime for a HipEngine, with the gfx1201 device o.device names, or
// the first, selected and recorded there.
Result<struct hip_api> open_hip(struct options& o);

class HipEngine : public EngineBase<HipEngine> {
    // The options, which outlive the engine.
    const struct options* options_;
    unsigned tier_;
    struct hip_api api_;
    // The one stream of the network and every kernel and copy of the daemon's,
    // from the first prepare() on.
    void* stream_ = nullptr;
    // The network's code objects and weights, from the first prepare() on; not
    // loaded while its api is null. Every tier's plan reads the same weights
    // (hip-plan checks it), and they differ only with --performance, which is
    // fixed for the engine's life.
    struct hip_model model_{};
    // What prepare() makes: none of them while zeroed. The codec stays zeroed
    // with the CPU codec.
    struct hip_network network_{};
    struct native_kernels kernels_{};
    struct codec_gpu gpu_codec_{};
    struct temporal_gpu temporal_{};
    void* device_input_ = nullptr;    // The encoded frame, unchanged until the next one.
    void* device_feedback_ = nullptr; // Later passes' input, allocated for multi-pass.
    void* device_output_ = nullptr;
    void* device_scratch_ = nullptr; // Tuning or colour: the other stage output.
    void* answer_ = nullptr;         // The latest frame's final network answer.
    std::vector<float> input_, neural_; // Host copies of a pass's input and answer.
    struct processing_settings previous_settings_ = processing_settings();
    bool warned_conditioning_ = false;
    // Stream events: frame start, uploaded, evaluated, answered. Timing never
    // stalls the stream; the intervals are read once the answer is complete.
    void* marks_[4]{};
    // The pair in each import slot.
    struct Imported {
        void* memory[2]{};
        void* frame[2]{}; // proxy, answer
    };
    std::array<Imported, kSlots> imported_{};

    void release(Imported& slot);
    // Everything prepare() made but the stream and the model, and every import.
    void release();
    // Nothing unless RESULT is an error: then hip_fail()'s "WHAT: <its name> (RESULT)".
    Result<void> check(int result, const char* what) const
    {
        if (!result) return {};
        return fail_hip(result, what);
    }
    [[gnu::cold, gnu::noinline]] Result<void> fail_hip(int result, const char* what) const;
    Result<void> mark(unsigned i) { return check(api_.hipEventRecord(marks_[i], stream_), "record timing event"); }
    Result<void> synchronize() { return check(api_.hipStreamSynchronize(stream_), "network completion"); }
    // A pass's stage, read back into the trace.
    Result<void> trace_image(struct frame_trace* trace, const struct geometry& g, unsigned pass,
                             const char* stage, const void* pointer, unsigned channels);

public:
    static constexpr unsigned max_passes = kMaxPasses;
    // A tier is a raster the network is built for.
    static constexpr bool rebuilds_for_tier = true;

    HipEngine(const struct options& o, unsigned tier, const struct hip_api& api);
    HipEngine(const HipEngine&) = delete;
    ~HipEngine()
    {
        release();
        if (stream_) api_.hipStreamDestroy(stream_);
        hip_model_fini(&model_);
    }
    const char* name() const { return "HIP"; }
    std::string device() const { return "device " + std::to_string(options_->device); }
    unsigned tier() const { return tier_; }
    std::string processing() const
    {
        const NativeTier& raster = *ShmNativeTier(tier_);
        return "processing=" + std::to_string(raster.width) + "x" + std::to_string(raster.networkHeight);
    }
    Result<void> prepare();
    // Builds the network for another tier, between frames, with the loaded
    // model. The layer offers its device-local frames again.
    Result<void> retier(unsigned tier)
    {
        release();
        forget_imports();
        tier_ = tier;
        return prepare();
    }
    // See codec_gpu_pin().
    void pin(uint8_t* input, uint8_t* output, size_t bytes)
    {
        if (gpu_codec_.kernels) codec_gpu_pin(&gpu_codec_, input, output, bytes);
    }
    bool import_into(unsigned slot, const ShmTransportOffer& offer, int fds[2]);
    Frames frames_of(unsigned slot) const
    {
        const Imported& pair = imported_[slot];
        return {static_cast<const uint8_t*>(pair.frame[0]), static_cast<uint8_t*>(pair.frame[1]), int(slot)};
    }
    Result<void> read_back(void* host, const void* source, size_t bytes)
    {
        return check(api_.hipMemcpy(host, source, bytes, 4), "read diagnostic frame");
    }
    Result<void> self_test(const struct options& o) { return run_self_test(o, *this); }
    // Proxy and answer are w * h RGBA8, or RGBA16F with settings.fp16: host
    // memory, or an import slot's device frames. verify checks the GPU codec
    // of host frames against the CPU reference inside the timed frame.
    Result<void> infer(const Frames& io, unsigned w, unsigned h, unsigned passes,
                       const struct processing_settings& settings = processing_settings(),
                       struct frame_trace* trace = nullptr, bool verify = false);
    // The latest infer()'s raw network answer, read back outside its timing.
    Result<const std::vector<float>*> raw_result();
};

} // namespace dlsslop
