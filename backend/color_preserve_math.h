// SPDX-License-Identifier: MIT
#pragma once
#include "tuning_math.h"
namespace dlsslop {
DLSSLOP_INLINE float color_min(float a,float b) { return a<b?a:b; }
DLSSLOP_INLINE float color_max(float a,float b) { return a>b?a:b; }
DLSSLOP_INLINE float luma(const float* rgb) { return 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2]; }
// output must not alias the inputs.
DLSSLOP_INLINE void preserve_color_pixel(const float* __restrict__ original_rgba,
    const float* __restrict__ model_rgb, float* __restrict__ output, unsigned width, unsigned x,
    unsigned y, unsigned low_x, unsigned low_y, unsigned right, unsigned bottom, float strength)
{
    const unsigned p = (y * width + x) * 3;
    const float* model = model_rgb + p;
    if (strength == 0 || x < low_x || x > right || y < low_y || y > bottom) {
        for (unsigned c = 0; c < 3; ++c) output[p+c] = model[c];
        return;
    }
            Binomial low = binomial3x3_rgb(model_rgb, original_rgba, width, x, y, low_x, low_y, right, bottom);
            float* delta = low.residual;
            // Remove the chroma of the drift: a luma-neutral correction, shortened by one factor
            // so no channel leaves [0,1], or goes further outside it than the model put it, while
            // luma is in [0,1]. The model's own chroma and out-of-range luma headroom are kept.
            const float dy = luma(delta), level = luma(model);
            float scale = 1;
            for (unsigned c = 0; c < 3; ++c) {
                const float d = delta[c] = strength * (delta[c] - dy);
                const float edge = d > 0 ? color_min(model[c], 0) : color_max(model[c], 1);
                if (d != 0) scale = color_min(scale, (model[c] - edge) / d);
            }
            if (level < 0 || level > 1) scale = 1;
            for (unsigned c = 0; c < 3; ++c) output[p + c] = model[c] - scale * delta[c];
}
} // namespace dlsslop
