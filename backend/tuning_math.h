// SPDX-License-Identifier: MIT
#pragma once

// Shared native postprocessing math: the tuning controls, and the 3x3 filter
// that colour preservation also uses. These controls operate on a pass's RGB
// residual; they are not NVIDIA NGX model conditioning or semantic masks.
#include "geometry.h"

namespace dlsslop {

struct NativeTuning {
    float intensity = 1.0f;
    float tone = 1.0f;
    float structure = 1.0f;
    float sharpness = 0.0f;
};

DLSSLOP_INLINE bool operator==(const NativeTuning& a, const NativeTuning& b)
{
    return a.intensity == b.intensity && a.tone == b.tone && a.structure == b.structure &&
           a.sharpness == b.sharpness;
}

DLSSLOP_INLINE bool native_tuning_is_default(const NativeTuning& tuning)
{
    return tuning == NativeTuning{};
}

DLSSLOP_INLINE unsigned clamp_coordinate(int position, unsigned low, unsigned high)
{
    return position < int(low) ? low : position > int(high) ? high : unsigned(position);
}

struct Binomial {
    float residual[3], model[3];
};

// Separable [1 2 1]/4 in each axis, evaluated as nine samples of the model
// and of its residual against the reference. Clamp to the fitted picture, so
// letterbox/padding cannot darken its edges.
DLSSLOP_INLINE Binomial binomial3x3_rgb(
    const float* __restrict__ model_rgb, const float* __restrict__ reference_rgba, unsigned width,
    unsigned x, unsigned y, unsigned low_x, unsigned low_y, unsigned high_x, unsigned high_y)
{
    Binomial low{};
    for (int dy = -1; dy <= 1; ++dy) {
        const unsigned sy = clamp_coordinate(int(y) + dy, low_y, high_y);
        for (int dx = -1; dx <= 1; ++dx) {
            const unsigned q = sy * width + clamp_coordinate(int(x) + dx, low_x, high_x);
            const float* m = model_rgb + q * 3;
            const float* r = reference_rgba + q * 4;
            const float weight = float((dx ? 1 : 2) * (dy ? 1 : 2)) * 0.0625f;
            for (unsigned c = 0; c < 3; ++c) {
                low.residual[c] += weight * (m[c] - r[c]);
                low.model[c] += weight * m[c];
            }
        }
    }
    return low;
}

// out receives the pixel's RGB and must not alias the inputs.
DLSSLOP_INLINE void tune_neural_pixel(
    const float* __restrict__ input_rgba, const float* __restrict__ raw_rgb, unsigned width,
    unsigned x, unsigned y, unsigned low_x, unsigned low_y,
    unsigned high_x, unsigned high_y, const NativeTuning& tuning, float* __restrict__ out)
{
    const unsigned pixel = y * width + x;
    const float* model = raw_rgb + pixel * 3;
    const float* input = input_rgba + pixel * 4;
    // Preserve default output bit for bit, including the unused padded region.
    if (native_tuning_is_default(tuning) || x < low_x || x > high_x ||
        y < low_y || y > high_y) {
        for (unsigned c = 0; c < 3; ++c) out[c] = model[c];
        return;
    }
    if (tuning.intensity == 0.0f && tuning.sharpness == 0.0f) {
        for (unsigned c = 0; c < 3; ++c) out[c] = input[c];
        return;
    }

    float edit[3];
    for (unsigned c = 0; c < 3; ++c) edit[c] = (model[c] - input[c]) * tuning.structure;
    Binomial low{};
    if (tuning.tone != tuning.structure || tuning.sharpness != 0.0f) {
        low = binomial3x3_rgb(raw_rgb, input_rgba, width, x, y, low_x, low_y, high_x, high_y);
        for (unsigned c = 0; c < 3; ++c) edit[c] += (tuning.tone - tuning.structure) * low.residual[c];
    }
    // No clipping here: subsequent passes retain floating-point headroom.
    // The final display codec is responsible for its output range.
    for (unsigned c = 0; c < 3; ++c) out[c] = input[c] + tuning.intensity * edit[c];
    if (tuning.sharpness != 0.0f)
        for (unsigned c = 0; c < 3; ++c) out[c] += tuning.sharpness * (model[c] - low.model[c]);
}

} // namespace dlsslop
