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
COLOR_INLINE void preserve_color_pixel(const float* original_rgba, const float* model_rgb,
    float* output, unsigned width, unsigned x, unsigned y, unsigned low_x, unsigned low_y,
    unsigned right, unsigned bottom, float strength)
{
    if (strength == 0 || x < low_x || x > right || y < low_y || y > bottom) {
        const unsigned p = (y * width + x) * 3;
        for (unsigned c = 0; c < 3; ++c) output[p+c] = model_rgb[p+c];
        return;
    }
            float delta[3]{};
            for (int dy = -1; dy <= 1; ++dy) {
                const unsigned sy = unsigned(color_clamp(int(y) + dy, int(low_y), int(bottom)));
                for (int dx = -1; dx <= 1; ++dx) {
                    const unsigned sx = unsigned(color_clamp(int(x) + dx, int(low_x), int(right)));
                    const unsigned p = unsigned(sy) * width + sx;
                    const float weight = float((dx ? 1 : 2) * (dy ? 1 : 2)) / 16;
                    for (unsigned c = 0; c < 3; ++c)
                        delta[c] += weight * (model_rgb[p * 3 + c] - original_rgba[p * 4 + c]);
                }
            }
            const float dy = 0.2126f * delta[0] + 0.7152f * delta[1] + 0.0722f * delta[2];
            const unsigned p = (unsigned(y) * width + x) * 3;
            for (unsigned c = 0; c < 3; ++c) {
                const float corrected = model_rgb[p + c] - strength * (delta[c] - dy);
                output[p + c] = corrected;
            }
            // Preserve the luma and compress chroma only where correction would
            // leave [0,1]. Preserve out-of-range luma headroom for the final codec.
            const float luma = 0.2126f * output[p] + 0.7152f * output[p + 1] + 0.0722f * output[p + 2];
            if (luma >= 0 && luma <= 1) {
                float scale = 1;
                for (unsigned c = 0; c < 3; ++c) {
                    const float chroma = output[p + c] - luma;
                    if (chroma > 0) scale = color_min(scale, (1 - luma) / chroma);
                    if (chroma < 0) scale = color_min(scale, -luma / chroma);
                }
                if (scale < 1)
                    for (unsigned c = 0; c < 3; ++c)
                        output[p + c] = luma + scale * (output[p + c] - luma);
            }
}
} // namespace dlsslop
#undef COLOR_INLINE
