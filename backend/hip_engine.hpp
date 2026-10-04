// The HIP network, a port of lmxxf's, and the daemon's own codec, tuning,
// color and motion kernels, on a gfx1201 device.
// SPDX-License-Identifier: MIT
#pragma once
#include "codec_gpu.hpp"
#include "engine.hpp"
#include "hip.hpp"
#include "hip_network.hpp"
#include "native_kernels.hpp"
#include "temporal_gpu.hpp"

#include <array>
#include <optional>
#include <vector>

namespace dlsslop {

// The gfx1201 device that --device names, or the first one; every visible
// device is listed on the way.
Result<int> select_device(const hip::Api& api, int requested);
// The HIP runtime for a HipEngine, with the gfx1201 device o.device names, or
// the first, selected and recorded there.
Result<hip::Api> open_hip(Options& o);

class HipEngine : public EngineBase<HipEngine> {
    Options options_;
    unsigned tier_;
    hip::Api api_;
    // The one stream of the network and every kernel and copy of the daemon's,
    // from the first prepare() on.
    hip::Handle stream_ = nullptr;
    // The network's code objects and weights, from the first prepare() on.
    // Every tier's plan reads the same weights (hip-plan checks it), and they
    // differ only with --performance, which is fixed for the engine's life.
    std::optional<hip::Model> model_;
    std::optional<hip::Network> network_;
    std::optional<NativeKernels> kernels_;
    std::optional<GpuCodec> gpu_codec_;
    std::optional<GpuTemporal> temporal_;
    void* device_input_ = nullptr;    // The encoded frame, unchanged until the next one.
    void* device_feedback_ = nullptr; // Later passes' input, allocated for multi-pass.
    void* device_output_ = nullptr;
    void* device_scratch_ = nullptr; // Tuning or colour: the other stage output.
    void* answer_ = nullptr;         // The latest frame's final network answer.
    std::vector<float> input_, neural_; // Host copies of a pass's input and answer.
    ProcessingSettings previous_settings_;
    bool warned_conditioning_ = false;
    // Stream events: frame start, uploaded, evaluated, answered. Timing never
    // stalls the stream; the intervals are read once the answer is complete.
    hip::Handle marks_[4]{};
    // The pair in each import slot.
    struct Imported {
        hip::Handle memory[2]{};
        void* frame[2]{}; // proxy, answer
    };
    std::array<Imported, kSlots> imported_{};

    void release(Imported& slot);
    // Everything prepare() made but the stream and the model, and every import.
    void release();
    Result<void> mark(unsigned i) { return api_.check(api_.hipEventRecord(marks_[i], stream_), "record timing event"); }
    Result<void> synchronize() { return api_.check(api_.hipStreamSynchronize(stream_), "network completion"); }
    // A pass's stage, read back into the trace.
    Result<void> trace_image(FrameTrace* trace, const struct geometry& g, unsigned pass, const char* stage,
                             const void* pointer, unsigned channels);

public:
    static constexpr unsigned max_passes = kMaxPasses;
    // A tier is a raster the network is built for.
    static constexpr bool rebuilds_for_tier = true;

    HipEngine(Options o, unsigned tier, const hip::Api& api);
    HipEngine(const HipEngine&) = delete;
    ~HipEngine()
    {
        release();
        if (stream_) api_.hipStreamDestroy(stream_);
    }
    const char* name() const { return "HIP"; }
    std::string device() const { return "device " + std::to_string(options_.device); }
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
    // See GpuCodec::pin.
    void pin(uint8_t* input, uint8_t* output, size_t bytes)
    {
        if (gpu_codec_) gpu_codec_->pin(input, output, bytes);
    }
    bool import_into(unsigned slot, const ShmTransportOffer& offer, Descriptor (&fds)[2]);
    Frames frames_of(unsigned slot) const
    {
        const Imported& pair = imported_[slot];
        return {static_cast<const uint8_t*>(pair.frame[0]), static_cast<uint8_t*>(pair.frame[1]), int(slot)};
    }
    Result<void> read_back(void* host, const void* source, size_t bytes)
    {
        return api_.check(api_.hipMemcpy(host, source, bytes, 4), "read diagnostic frame");
    }
    Result<void> self_test(const Options& o) { return run_self_test(o, *this); }
    // Proxy and answer are w * h RGBA8, or RGBA16F with settings.fp16: host
    // memory, or an import slot's device frames. verify checks the GPU codec
    // of host frames against the CPU reference inside the timed frame.
    Result<void> infer(const Frames& io, unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings = {},
                       FrameTrace* trace = nullptr, bool verify = false);
    // The latest infer()'s raw network answer, read back outside its timing.
    Result<const std::vector<float>*> raw_result();
};

} // namespace dlsslop
