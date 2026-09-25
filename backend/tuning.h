// SPDX-License-Identifier: MIT
#pragma once

#include "codec.h"
#include "tuning_math.h"
#include "vendor/hip_api.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dlsslop {

static_assert(sizeof(NativeTuning) == 16 && offsetof(NativeTuning, sharpness) == 12 &&
              std::is_trivially_copyable<NativeTuning>::value, "native tuning GPU ABI");

inline void validate_native_tuning(const NativeTuning& tuning)
{
    for (float value : {tuning.intensity, tuning.tone, tuning.structure})
        if (!std::isfinite(value) || value < 0.0f || value > 4.0f)
            throw std::invalid_argument("native intensity/tone/structure must be finite and within 0..4");
    if (!std::isfinite(tuning.sharpness) || tuning.sharpness < 0.0f || tuning.sharpness > 1.0f)
        throw std::invalid_argument("native sharpness must be finite and within 0..1");
}

inline std::size_t validate_tuning_geometry(const Geometry& g)
{
    if (!g.width || !g.height || !g.fit_width || !g.fit_height ||
        g.x >= g.width || g.y >= g.height || g.fit_width > g.width - g.x ||
        g.fit_height > g.height - g.y || g.width > 16384 || g.height > 16384)
        throw std::invalid_argument("invalid native tuning geometry");
    return std::size_t(g.width) * g.height;
}

// CPU reference and --cpu-codec implementation. The vectors backing input and
// raw must be distinct from output; the neighbourhood filter is not in-place.
inline void tune_neural_rgb(const float* input_rgba, const float* raw_rgb,
                            const Geometry& g, std::vector<float>& output,
                            const NativeTuning& tuning)
{
    validate_native_tuning(tuning);
    const std::size_t pixels = validate_tuning_geometry(g);
    if (!input_rgba || !raw_rgb || (!output.empty() &&
        (input_rgba == output.data() || raw_rgb == output.data())))
        throw std::invalid_argument("native tuning requires distinct input and output buffers");
    output.resize(pixels * 3);
    const unsigned high_x = g.x + g.fit_width - 1;
    const unsigned high_y = g.y + g.fit_height - 1;
    for (unsigned y = 0; y < g.height; ++y)
        for (unsigned x = 0; x < g.width; ++x)
            for (unsigned channel = 0; channel < 3; ++channel) {
                const float value = tune_neural_component(input_rgba, raw_rgb, g.width,
                    x, y, g.x, g.y, high_x, high_y, channel, tuning);
                if (!std::isfinite(value))
                    throw std::runtime_error("native tuning produced nonfinite RGB");
                output[(std::size_t(y) * g.width + x) * 3 + channel] = value;
            }
}

class GpuTuning {
    hip_probe::Api& api_;
    hip_probe::Handle stream_{};
    hip_probe::Handle module_{}, kernel_{};

public:
    GpuTuning(hip_probe::Api& api, hip_probe::Handle stream, const std::string& module_path)
        : api_(api), stream_(stream)
    {
        api_.Check(api_.LoadModule(&module_, module_path.c_str()), "load native tuning module");
        try {
            api_.Check(api_.hipModuleGetFunction(&kernel_, module_, "dlsslop_tune_rgb"),
                       "find native tuning kernel");
        } catch (...) {
            api_.hipModuleUnload(module_);
            module_ = nullptr;
            throw;
        }
    }
    GpuTuning(const GpuTuning&) = delete;
    GpuTuning& operator=(const GpuTuning&) = delete;
    ~GpuTuning()
    {
        if (module_) {
            api_.hipStreamSynchronize(stream_);
            api_.hipModuleUnload(module_);
        }
    }

    // Queue after inference on its stream. The caller owns all three buffers
    // and retains them until stream completion. The display codec detects
    // nonfinite/FP16-overflow results when it reads the final answer.
    void apply(const Geometry& g, void* input_rgba, void* raw_rgb,
               void* output_rgb, const NativeTuning& tuning)
    {
        validate_native_tuning(tuning);
        const std::size_t pixels = validate_tuning_geometry(g);
        if (!input_rgba || !raw_rgb || !output_rgb || output_rgb == input_rgba || output_rgb == raw_rgb)
            throw std::invalid_argument("GPU native tuning requires distinct input and output buffers");
        Geometry geometry_arg = g;
        NativeTuning tuning_arg = tuning;
        void* arguments[] = {&input_rgba, &raw_rgb, &output_rgb, &geometry_arg, &tuning_arg};
        api_.Check(api_.hipModuleLaunchKernel(kernel_, unsigned((pixels + 255) / 256),
            1, 1, 256, 1, 1, 0, stream_, arguments, nullptr), "apply native per-pass residual tuning");
    }
};

} // namespace dlsslop
