// SPDX-License-Identifier: MIT
#pragma once
#include "codec.h"
#include "color_preserve_math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dlsslop {
// Correct low-frequency model-space chroma drift, using the ORIGINAL encoded
// input on every pass. A 3x3 binomial filter smooths the correction rather than
// forcing original chroma onto newly generated edges. This preserves weighted
// proxy-space luma (not physical linear-light luminance). High-frequency chroma
// changes remain possible; this is not a guarantee of inference correctness.
inline void preserve_color(const float* original_rgba, const float* model_rgb,
                            const Geometry& g, float strength, std::vector<float>& result)
{
    if (!std::isfinite(strength) || strength < 0 || strength > 1)
        throw std::invalid_argument("color preservation must be finite and within 0..1");
    if (!original_rgba || !model_rgb || !g.width || !g.height || !g.fit_width || !g.fit_height ||
        g.width > 16384 || g.height > 16384 || g.x >= g.width || g.y >= g.height ||
        g.fit_width > g.width - g.x || g.fit_height > g.height - g.y)
        throw std::invalid_argument("invalid color preservation buffers or geometry");
    if (!result.empty() && (result.data() == model_rgb || result.data() == original_rgba))
        throw std::invalid_argument("color preservation requires distinct output");
    const std::size_t pixels = std::size_t(g.width) * g.height;
    result.resize(pixels * 3);
    std::memcpy(result.data(), model_rgb, pixels * 3 * sizeof(float));
    if (strength == 0) return;
    const unsigned right = g.x + g.fit_width - 1, bottom = g.y + g.fit_height - 1;
    for (unsigned y = g.y; y <= bottom; ++y) {
        for (unsigned x = g.x; x <= right; ++x) {
            preserve_color_pixel(original_rgba, model_rgb, result.data(), g.width,
                x, y, g.x, g.y, right, bottom, strength);
            const std::size_t p = (std::size_t(y) * g.width + x) * 3;
            for (unsigned c = 0; c < 3; ++c)
                if (!std::isfinite(result[p+c]))
                    throw std::runtime_error("nonfinite color preservation result");
        }
    }
}
} // namespace dlsslop
