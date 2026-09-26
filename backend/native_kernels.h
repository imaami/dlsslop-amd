// SPDX-License-Identifier: MIT
#pragma once

#include "tuning_math.h"
#include "vendor/hip_api.h"

#include <stdexcept>
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
    hip_probe::Handle module_{}, kernels_[kKernelCount]{};

public:
    hip_probe::Api& api;
    const hip_probe::Handle stream;

    NativeKernels(hip_probe::Api& api, hip_probe::Handle stream, const std::string& path)
        : api(api), stream(stream)
    {
        api.Check(api.LoadModule(&module_, path.c_str()), "load native module");
        for (unsigned k = 0; k < kKernelCount; ++k)
            if (const int error = api.hipModuleGetFunction(&kernels_[k], module_, kNames[k])) {
                api.hipModuleUnload(module_);
                api.Check(error, kNames[k]);
            }
    }
    NativeKernels(const NativeKernels&) = delete;
    NativeKernels& operator=(const NativeKernels&) = delete;
    ~NativeKernels()
    {
        api.hipStreamSynchronize(stream);
        api.hipModuleUnload(module_);
    }

    // One thread per item; args point to the kernel's parameters.
    void launch(Kernel kernel, unsigned count, void** args) const
    {
        api.Check(api.hipModuleLaunchKernel(kernels_[kernel], (count + 255) / 256, 1, 1, 256, 1, 1, 0,
                                            stream, args, nullptr), kNames[kernel]);
    }
};

// Native tuning of one pass's raw RGB, queued after its inference. The caller
// keeps the three buffers until the stream reaches the kernel; it reads
// neighbours, so the output is distinct from both inputs.
inline void gpu_tune(const NativeKernels& kernels, Geometry g, const void* input_rgba, const void* raw_rgb,
                     void* output_rgb, NativeTuning tuning)
{
    if (output_rgb == input_rgba || output_rgb == raw_rgb)
        throw std::invalid_argument("GPU native tuning requires distinct input and output buffers");
    void* args[] = {&input_rgba, &raw_rgb, &output_rgb, &g, &tuning};
    kernels.launch(kTuneRgb, g.width * g.height, args);
}

// Colour preservation of one pass against the frame's encoded input (the
// reference), with the same buffer rules as gpu_tune.
inline void gpu_preserve_color(const NativeKernels& kernels, Geometry g, const void* original_rgba,
                               const void* raw_rgb, void* output_rgb, float strength)
{
    if (output_rgb == original_rgba || output_rgb == raw_rgb)
        throw std::invalid_argument("GPU color preservation requires distinct input and output buffers");
    void* args[] = {&original_rgba, &raw_rgb, &output_rgb, &g, &strength};
    kernels.launch(kPreserveColor, g.width * g.height, args);
}

} // namespace dlsslop
