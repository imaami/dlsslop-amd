// SPDX-License-Identifier: MIT
#pragma once

#include "hip.h"
#include "tuning_math.h"

#include <string>

namespace dlsslop {

// The kernels of the linux_native module, from codec_gpu.hip, tuning_gpu.hip,
// color_gpu.hip and temporal_gpu.hip.
enum Kernel : unsigned {
    kEncodeRgba8, kEncodeRgba16f, kFeedbackRgb, kDecodeRgba8, kDecodeRgba16f, kTuneRgb, kPreserveColor,
    kTemporalLuma, kTemporalReduce, kTemporalFlow, kTemporalWarp, kTemporalCut, kKernelCount
};

// The module, loaded once, and its kernels, launched on one stream (the
// network's) as one-dimensional grids of 256-thread groups.
class NativeKernels {
    static constexpr const char* kNames[kKernelCount] = {
        "dlsslop_encode_rgba8", "dlsslop_encode_rgba16f", "dlsslop_feedback_rgb", "dlsslop_decode_rgba8",
        "dlsslop_decode_rgba16f", "dlsslop_tune_rgb", "dlsslop_preserve_color", "dlsslop_temporal_luma",
        "dlsslop_temporal_reduce", "dlsslop_temporal_flow", "dlsslop_temporal_warp", "dlsslop_temporal_cut"};
    hip::Handle module_{}, kernels_[kKernelCount]{};

public:
    const hip::Api& api;
    const hip::Handle stream;

    NativeKernels(const hip::Api& api, hip::Handle stream) : api(api), stream(stream) {}
    NativeKernels(const NativeKernels&) = delete;
    NativeKernels& operator=(const NativeKernels&) = delete;
    ~NativeKernels()
    {
        if (!module_) return;
        api.hipStreamSynchronize(stream);
        api.hipModuleUnload(module_);
    }
    // The module at PATH and every kernel in it.
    Result<void> load(const std::string& path);

    // One thread per item; args point to the kernel's parameters.
    Result<void> launch(Kernel kernel, unsigned count, void** args) const
    {
        return api.check(api.hipModuleLaunchKernel(kernels_[kernel], (count + 255) / 256, 1, 1, 256, 1, 1, 0, stream,
                                                   args, nullptr),
                         kNames[kernel]);
    }
};

// Native tuning of one pass's raw RGB, queued after its inference. The caller
// keeps the three buffers until the stream reaches the kernel; it reads
// neighbours, so the output is distinct from both inputs.
inline Result<void> gpu_tune(const NativeKernels& kernels, Geometry g, const void* input_rgba, const void* raw_rgb,
                             void* output_rgb, NativeTuning tuning)
{
    if (output_rgb == input_rgba || output_rgb == raw_rgb)
        return fail("GPU native tuning requires distinct input and output buffers");
    void* args[] = {&input_rgba, &raw_rgb, &output_rgb, &g, &tuning};
    return kernels.launch(kTuneRgb, g.width * g.height, args);
}

// Colour preservation of one pass against the frame's encoded input (the
// reference), with the same buffer rules as gpu_tune.
inline Result<void> gpu_preserve_color(const NativeKernels& kernels, Geometry g, const void* original_rgba,
                                       const void* raw_rgb, void* output_rgb, float strength)
{
    if (output_rgb == original_rgba || output_rgb == raw_rgb)
        return fail("GPU color preservation requires distinct input and output buffers");
    void* args[] = {&original_rgba, &raw_rgb, &output_rgb, &g, &strength};
    return kernels.launch(kPreserveColor, g.width * g.height, args);
}

} // namespace dlsslop
