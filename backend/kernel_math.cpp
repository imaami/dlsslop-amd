// SPDX-License-Identifier: MIT
// The CPU references' pixel loops over the kernels' own math, and the motion
// kernels' sampler: kernel_math.h.
// It uses nothing of the C++ runtime (no allocation, exceptions, RTTI or
// dynamic initialization), so the C programs that link it link with the C
// driver.
//
// SDR port of native_codec_encode.hlsl, native_codec_decode.hlsl and
// native_game_rgb_input.hlsl from
// https://github.com/lmxxf/dlss5-on-amd-9070xt-porting
//
// Copyright (c) 2026 Kien
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "kernel_math.h"
#include "codec_math.hpp"
#include "color_preserve_math.hpp"
#include "temporal_math.hpp"
#include "tuning_math.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dlsslop {
namespace {

// Matches conversion into the upstream intermediate RGBA16_FLOAT textures.
// Explicit IEEE round-to-nearest-even avoids requiring F16C or a HIP SDK.
inline float half_round(float input)
{
    std::uint32_t bits;
    std::memcpy(&bits, &input, sizeof bits);
    const std::uint32_t sign = bits & 0x80000000u;
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u)
        return input;
    if (magnitude >= 0x477ff000u) {
        bits = sign | 0x7f800000u;
    } else if (magnitude < 0x38800000u) {
        // Every binary16 subnormal is an integral multiple of 2^-24.
        // The power-of-two scale is exact and nearbyint need not depend on the
        // caller's floating-point rounding mode: perform ties-to-even directly.
        const float scaled = std::fabs(input) * 16777216.0f;
        const float low = std::floor(scaled);
        const float fraction = scaled - low;
        const unsigned integer = static_cast<unsigned>(low);
        const unsigned rounded = integer +
            unsigned(fraction > 0.5f || (fraction == 0.5f && (integer & 1)));
        return std::copysign(float(rounded) * 0x1p-24f, input);
    } else {
        // A normal half has ten fraction bits versus float's twenty-three.
        bits = sign | ((magnitude + 0xfffu + ((magnitude >> 13) & 1u)) &
                       0xffffe000u);
    }
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}

inline float unpack_half(const std::uint8_t* source)
{
    std::uint16_t half;
    std::memcpy(&half, source, sizeof half);
    const unsigned exponent = (half >> 10) & 31u;
    const unsigned fraction = half & 1023u;
    if (!exponent)
        return std::copysign(float(fraction) * 0x1p-24f, half & 0x8000u ? -1.0f : 1.0f);
    const std::uint32_t bits = (std::uint32_t(half & 0x8000u) << 16) |
        ((exponent == 31 ? 255u : exponent + 112u) << 23) | (fraction << 13);
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}

// The input must round to a finite binary16.
inline void pack_half(float input, std::uint8_t* output)
{
    const float rounded = half_round(input);
    std::uint32_t bits;
    std::memcpy(&bits, &rounded, sizeof bits);
    const unsigned magnitude = bits & 0x7fffffffu;
    const std::uint16_t half = std::uint16_t((bits >> 16) & 0x8000u) |
        std::uint16_t(magnitude < 0x38800000u ?
            unsigned(std::fabs(rounded) * 0x1p24f) :
            ((magnitude >> 13) - (112u << 10)));
    std::memcpy(output, &half, sizeof half);
}

struct Unorm8Floats {
    float value[256];
};

// float(i) / 255.0f for every byte i, divided at compile time: a texel's
// channels without a divide each.
constexpr Unorm8Floats unorm8_floats = [] {
    Unorm8Floats table{};
    for (unsigned i = 0; i < 256; ++i)
        table.value[i] = float(i) / 255.0f;
    return table;
}();

inline Rgb rgba8(const std::uint8_t* p)
{
    return {unorm8_floats.value[p[0]], unorm8_floats.value[p[1]], unorm8_floats.value[p[2]]};
}

// The answer at source pixel (x, y) through the upstream FP16 surface. Like
// the GPU codec, the decoder rejects it when a texel it reads is not a finite
// binary16: it is then not finite either.
Rgb answer(const float* neural_rgb, const struct geometry& g, unsigned x, unsigned y)
{
    return sample_answer([&](unsigned px, unsigned py) {
        const float* p = neural_rgb + (std::size_t(py) * g.width + px) * 3;
        return Rgb{half_round(p[0]), half_round(p[1]), half_round(p[2])};
    }, g, x, y);
}

void reflect_padding(const struct geometry& g, float* rgba)
{
    const std::size_t row = std::size_t(g.width) * 4;
    for (unsigned y = g.valid_height; y < g.height; ++y)
        std::memcpy(rgba + y * row, rgba + codec_row(g, y) * row, row * sizeof(float));
}

template<bool fp16>
bool encode_proxy(const std::uint8_t* source, const struct geometry& g, float* rgba)
{
    const auto read = [&](unsigned x, unsigned y) {
        if constexpr (fp16) {
            const std::uint8_t* p = source + (std::size_t(y) * g.source_width + x) * 8;
            return Rgb{unpack_half(p), unpack_half(p + 2), unpack_half(p + 4)};
        } else {
            return rgba8(source + (std::size_t(y) * g.source_width + x) * 4);
        }
    };
    for (unsigned y = 0; y < g.valid_height; ++y) {
        for (unsigned x = 0; x < g.width; ++x) {
            Rgb c{};
            if (fitted(g, x, y))
                c = sample_proxy(read, g, x, y);
            if (!finite(c)) // Only FP16 samples can be.
                return false;
            float* p = rgba + (std::size_t(y) * g.width + x) * 4;
            p[0] = half_round(c.r);
            p[1] = half_round(c.g);
            p[2] = half_round(c.b);
            p[3] = 1.0f;
        }
    }
    reflect_padding(g, rgba);
    return true;
}

bool feedback_neural_rgb(const float* neural_rgb, const struct geometry& g, bool precision16, float* rgba)
{
    for (unsigned y = 0; y < g.valid_height; ++y) {
        for (unsigned x = 0; x < g.width; ++x) {
            Rgb c{};
            if (fitted(g, x, y)) {
                const float* p = neural_rgb + (std::size_t(y) * g.width + x) * 3;
                const Rgb raw{p[0], p[1], p[2]};
                c = precision16 ? Rgb{half_round(p[0]), half_round(p[1]), half_round(p[2])} :
                    Rgb{half_round(float(unorm8(p[0])) / 255.0f),
                        half_round(float(unorm8(p[1])) / 255.0f),
                        half_round(float(unorm8(p[2])) / 255.0f)};
                // Binary16 feedback also rejects what rounds past binary16;
                // UNORM8 feedback, which clamps, only nonfinite input.
                if (!finite(raw) || !finite(c))
                    return false;
            }
            float* p = rgba + (std::size_t(y) * g.width + x) * 4;
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
            p[3] = 1.0f;
        }
    }
    reflect_padding(g, rgba);
    return true;
}

bool decode_neural_proxy(const std::uint8_t* original, const struct geometry& g, bool fp16,
                         const float* neural_rgb, std::uint8_t* output)
{
    const unsigned pixel_bytes = fp16 ? 8 : 4;
    for (unsigned y = 0; y < g.source_height; ++y) {
        for (unsigned x = 0; x < g.source_width; ++x) {
            const Rgb neural = answer(neural_rgb, g, x, y);
            if (!finite(neural)) return false;
            const std::size_t p = (std::size_t(y) * g.source_width + x) * pixel_bytes;
            if (fp16) {
                pack_half(neural.r, output + p);
                pack_half(neural.g, output + p + 2);
                pack_half(neural.b, output + p + 4);
                std::memcpy(output + p + 6, original + p + 6, 2);
            } else {
                output[p] = unorm8(neural.r);
                output[p + 1] = unorm8(neural.g);
                output[p + 2] = unorm8(neural.b);
                output[p + 3] = original[p + 3];
            }
        }
    }
    return true;
}

bool tune_neural_rgb(const float* input_rgba, const float* raw_rgb, const struct geometry& g,
                     const struct native_tuning& tuning, float* output)
{
    const unsigned high_x = g.x + g.fit_width - 1;
    const unsigned high_y = g.y + g.fit_height - 1;
    for (unsigned y = 0; y < g.height; ++y)
        for (unsigned x = 0; x < g.width; ++x) {
            float* const rgb = output + (std::size_t(y) * g.width + x) * 3;
            tune_neural_pixel(input_rgba, raw_rgb, g.width, x, y, g.x, g.y, high_x, high_y, tuning, rgb);
            if (!std::isfinite(rgb[0]) || !std::isfinite(rgb[1]) || !std::isfinite(rgb[2]))
                return false;
        }
    return true;
}

// Correct low-frequency model-space chroma drift, using the ORIGINAL encoded
// input on every pass. A 3x3 binomial filter smooths the correction rather than
// forcing original chroma onto newly generated edges. This preserves weighted
// proxy-space luma (not physical linear-light luminance). High-frequency chroma
// changes remain possible; this is not a guarantee of inference correctness.
bool preserve_color(const float* original_rgba, const float* model_rgb, const struct geometry& g, float strength,
                    float* result)
{
    const std::size_t pixels = std::size_t(g.width) * g.height;
    std::memcpy(result, model_rgb, pixels * 3 * sizeof(float));
    if (strength == 0) return true;
    const unsigned right = g.x + g.fit_width - 1, bottom = g.y + g.fit_height - 1;
    for (unsigned y = g.y; y <= bottom; ++y) {
        for (unsigned x = g.x; x <= right; ++x) {
            preserve_color_pixel(original_rgba, model_rgb, result, g.width,
                x, y, g.x, g.y, right, bottom, strength);
            const std::size_t p = (std::size_t(y) * g.width + x) * 3;
            for (unsigned c = 0; c < 3; ++c)
                if (!std::isfinite(result[p+c]))
                    return false;
        }
    }
    return true;
}

} // namespace
} // namespace dlsslop

bool kernel_math_encode_proxy(const std::uint8_t* source, const struct geometry* g, bool fp16, float* rgba)
{
    return fp16 ? dlsslop::encode_proxy<true>(source, *g, rgba) : dlsslop::encode_proxy<false>(source, *g, rgba);
}

bool kernel_math_feedback_neural_rgb(const float* neural_rgb, const struct geometry* g, bool precision16,
                                     float* rgba)
{
    return dlsslop::feedback_neural_rgb(neural_rgb, *g, precision16, rgba);
}

bool kernel_math_decode_neural_proxy(const std::uint8_t* original, const struct geometry* g, bool fp16,
                                     const float* neural_rgb, std::uint8_t* output)
{
    return dlsslop::decode_neural_proxy(original, *g, fp16, neural_rgb, output);
}

bool kernel_math_tune_neural_rgb(const float* input_rgba, const float* raw_rgb, const struct geometry* g,
                                 const struct native_tuning* tuning, float* output)
{
    return dlsslop::tune_neural_rgb(input_rgba, raw_rgb, *g, *tuning, output);
}

bool kernel_math_preserve_color(const float* original_rgba, const float* model_rgb, const struct geometry* g,
                                float strength, float* output)
{
    return dlsslop::preserve_color(original_rgba, model_rgb, *g, strength, output);
}

float kernel_math_temporal_sample(const float* image, struct temporal_extent e, float x, float y)
{
    return dlsslop_temporal::sample(image, e, x, y);
}
