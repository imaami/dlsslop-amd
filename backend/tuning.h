// SPDX-License-Identifier: MIT
#pragma once

#include "tuning_math.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dlsslop {

// Tuning outside the controls' ranges rejects a request.
inline void validate_native_tuning(const NativeTuning& tuning)
{
    for (float value : {tuning.intensity, tuning.tone, tuning.structure})
        if (!std::isfinite(value) || value < 0.0f || value > 4.0f)
            throw std::range_error("native intensity/tone/structure must be finite and within 0..4");
    if (!std::isfinite(tuning.sharpness) || tuning.sharpness < 0.0f || tuning.sharpness > 1.0f)
        throw std::range_error("native sharpness must be finite and within 0..1");
}

// CPU reference of the GPU kernel. The vectors backing input and raw must be
// distinct from output; the neighbourhood filter is not in-place.
inline void tune_neural_rgb(const float* input_rgba, const float* raw_rgb,
                            const Geometry& g, std::vector<float>& output,
                            const NativeTuning& tuning)
{
    validate_native_tuning(tuning);
    if (!fits(g) || g.width > 16384 || g.height > 16384)
        throw std::invalid_argument("invalid native tuning geometry");
    const std::size_t pixels = std::size_t(g.width) * g.height;
    if (!input_rgba || !raw_rgb || (!output.empty() &&
        (input_rgba == output.data() || raw_rgb == output.data())))
        throw std::invalid_argument("native tuning requires distinct input and output buffers");
    output.resize(pixels * 3);
    const unsigned high_x = g.x + g.fit_width - 1;
    const unsigned high_y = g.y + g.fit_height - 1;
    for (unsigned y = 0; y < g.height; ++y)
        for (unsigned x = 0; x < g.width; ++x) {
            float* const rgb = output.data() + (std::size_t(y) * g.width + x) * 3;
            tune_neural_pixel(input_rgba, raw_rgb, g.width, x, y, g.x, g.y, high_x, high_y, tuning, rgb);
            if (!std::isfinite(rgb[0]) || !std::isfinite(rgb[1]) || !std::isfinite(rgb[2]))
                throw std::runtime_error("native tuning produced nonfinite RGB");
        }
}

} // namespace dlsslop
