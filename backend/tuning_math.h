// SPDX-License-Identifier: MIT
#pragma once

// Shared native postprocessing math. These controls operate on a pass's RGB
// residual; they are not NVIDIA NGX model conditioning or semantic masks.
#if defined(__HIP_DEVICE_COMPILE__)
#define DLSSLOP_TUNE_INLINE __attribute__((device)) __attribute__((always_inline)) inline
#else
#define DLSSLOP_TUNE_INLINE inline
#endif

namespace dlsslop {

struct NativeTuning {
    float intensity = 1.0f;
    float tone = 1.0f;
    float structure = 1.0f;
    float sharpness = 0.0f;
};

DLSSLOP_TUNE_INLINE bool native_tuning_is_default(const NativeTuning& tuning)
{
    return tuning.intensity == 1.0f && tuning.tone == 1.0f &&
           tuning.structure == 1.0f && tuning.sharpness == 0.0f;
}

DLSSLOP_TUNE_INLINE unsigned tune_clamp_coordinate(int position, unsigned low, unsigned high)
{
    return position < int(low) ? low : position > int(high) ? high : unsigned(position);
}

DLSSLOP_TUNE_INLINE float tune_neural_component(
    const float* input_rgba, const float* raw_rgb, unsigned width,
    unsigned x, unsigned y, unsigned low_x, unsigned low_y,
    unsigned high_x, unsigned high_y, unsigned channel, const NativeTuning& tuning)
{
    const unsigned pixel = y * width + x;
    const float model = raw_rgb[pixel * 3 + channel];
    // Preserve default output bit for bit, including the unused padded region.
    if (native_tuning_is_default(tuning) || x < low_x || x > high_x ||
        y < low_y || y > high_y)
        return model;
    const float input = input_rgba[pixel * 4 + channel];
    if (tuning.intensity == 0.0f && tuning.sharpness == 0.0f)
        return input;

    const float residual = model - input;
    float edit = residual * tuning.structure;
    float model_low = 0.0f;
    if (tuning.tone != tuning.structure || tuning.sharpness != 0.0f) {
        // Separable [1 2 1]/4 in each axis, evaluated as nine samples. Clamp
        // to the fitted picture, so letterbox/padding cannot darken its edges.
        float residual_low = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            const unsigned sy = tune_clamp_coordinate(int(y) + dy, low_y, high_y);
            for (int dx = -1; dx <= 1; ++dx) {
                const unsigned sx = tune_clamp_coordinate(int(x) + dx, low_x, high_x);
                const unsigned neighbour = sy * width + sx;
                const float weight = float((dx ? 1 : 2) * (dy ? 1 : 2)) * 0.0625f;
                const float model_sample = raw_rgb[neighbour * 3 + channel];
                const float input_sample = input_rgba[neighbour * 4 + channel];
                residual_low += weight * (model_sample - input_sample);
                model_low += weight * model_sample;
            }
        }
        edit += (tuning.tone - tuning.structure) * residual_low;
    }
    // No clipping here: subsequent passes retain floating-point headroom.
    // The final display codec is responsible for its output range.
    float result = input + tuning.intensity * edit;
    if (tuning.sharpness != 0.0f)
        result += tuning.sharpness * (model - model_low);
    return result;
}

} // namespace dlsslop

#undef DLSSLOP_TUNE_INLINE
