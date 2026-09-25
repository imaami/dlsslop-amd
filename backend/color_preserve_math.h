// SPDX-License-Identifier: MIT
#pragma once
#if defined(__HIP_DEVICE_COMPILE__)
#define COLOR_INLINE __attribute__((device)) __attribute__((always_inline)) inline
#else
#define COLOR_INLINE inline
#endif
namespace dlsslop {
COLOR_INLINE int color_clamp(int v,int low,int high) { return v<low?low:v>high?high:v; }
COLOR_INLINE float color_min(float a,float b) { return a<b?a:b; }
COLOR_INLINE float color_max(float a,float b) { return a>b?a:b; }
COLOR_INLINE void preserve_color_pixel(const float* original_rgba, const float* model_rgb,
    float* output, unsigned width, unsigned x, unsigned y, unsigned low_x, unsigned low_y,
    unsigned right, unsigned bottom, float strength)
{
    const unsigned p = (y * width + x) * 3;
    const float* model = model_rgb + p;
    if (strength == 0 || x < low_x || x > right || y < low_y || y > bottom) {
        for (unsigned c = 0; c < 3; ++c) output[p+c] = model[c];
        return;
    }
            float delta[3]{};
            for (int dy = -1; dy <= 1; ++dy) {
                const unsigned sy = unsigned(color_clamp(int(y) + dy, int(low_y), int(bottom)));
                for (int dx = -1; dx <= 1; ++dx) {
                    const unsigned sx = unsigned(color_clamp(int(x) + dx, int(low_x), int(right)));
                    const unsigned q = unsigned(sy) * width + sx;
                    const float weight = float((dx ? 1 : 2) * (dy ? 1 : 2)) / 16;
                    for (unsigned c = 0; c < 3; ++c)
                        delta[c] += weight * (model_rgb[q * 3 + c] - original_rgba[q * 4 + c]);
                }
            }
            // Remove the chroma of the drift: a luma-neutral correction, shortened by one factor
            // so no channel leaves [0,1], or goes further outside it than the model put it, while
            // luma is in [0,1]. The model's own chroma and out-of-range luma headroom are kept.
            const float dy = 0.2126f * delta[0] + 0.7152f * delta[1] + 0.0722f * delta[2];
            const float luma = 0.2126f * model[0] + 0.7152f * model[1] + 0.0722f * model[2];
            float scale = 1;
            for (unsigned c = 0; c < 3; ++c) {
                const float d = delta[c] = strength * (delta[c] - dy);
                const float edge = d > 0 ? color_min(model[c], 0) : color_max(model[c], 1);
                if (d != 0) scale = color_min(scale, (model[c] - edge) / d);
            }
            if (luma < 0 || luma > 1) scale = 1;
            for (unsigned c = 0; c < 3; ++c) output[p + c] = model[c] - scale * delta[c];
}
} // namespace dlsslop
#undef COLOR_INLINE
