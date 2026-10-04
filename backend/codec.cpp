// SPDX-License-Identifier: MIT
// SDR port of native_codec_encode.hlsl, native_codec_decode.hlsl,
// native_game_rgb_input.hlsl and native_input_geometry.h from
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

#include "codec.hpp"
#include "codec_math.hpp"
#include "shm_protocol.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

namespace dlsslop {
namespace {

// Matches conversion into the upstream intermediate RGBA16_FLOAT textures.
// Explicit IEEE round-to-nearest-even avoids requiring F16C or a HIP SDK.
float half_round(float input)
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

float unpack_half(const std::uint8_t* source)
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
void pack_half(float input, std::uint8_t* output)
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

Rgb rgba8(const std::uint8_t* p)
{
    return {float(p[0]) / 255.0f, float(p[1]) / 255.0f, float(p[2]) / 255.0f};
}

// The answer at source pixel (x, y) through the upstream FP16 surface. Like
// the GPU codec, the decoder rejects it when a texel it reads is not a finite
// binary16: it is then not finite either.
// Out of line: inlined, it adds 3 KB (Clang) to 8 KB (GCC) of text.
[[gnu::noinline]] Rgb answer(const float* neural_rgb, const Geometry& g, unsigned x, unsigned y)
{
    return sample_answer([&](unsigned px, unsigned py) {
        const float* p = neural_rgb + (std::size_t(py) * g.width + px) * 3;
        return Rgb{half_round(p[0]), half_round(p[1]), half_round(p[2])};
    }, g, x, y);
}

Failure<const char*> nonfinite_answer()
{
    return reject("neural output contains nonfinite or FP16-overflow samples");
}

void reflect_padding(const Geometry& g, std::vector<float>& rgba)
{
    const std::size_t row = std::size_t(g.width) * 4;
    for (unsigned y = g.valid_height; y < g.height; ++y)
        std::memcpy(rgba.data() + y * row, rgba.data() + codec_row(g, y) * row, row * sizeof(float));
}

} // namespace

Result<Geometry> geometry(unsigned source_width, unsigned source_height, unsigned tier_height)
{
    if (!source_width || !source_height || source_width > 16384 || source_height > 16384)
        return fail("source extent must be in 1..16384");
    const auto holds = [=](const NativeTier& t) { return source_width <= t.width && source_height <= t.height; };
    // The named tier, or the smallest that holds the source, else the largest.
    const NativeTier* tier = tier_height ? ShmNativeTier(tier_height)
        : std::min(std::find_if(std::begin(kNativeTiers), std::end(kNativeTiers), holds), std::end(kNativeTiers) - 1);
    if (!tier) return fail("network height must be 0, 720, 900 or 1080");
    Geometry g;
    g.source_width = source_width;
    g.source_height = source_height;
    g.width = tier->width;
    g.height = tier->networkHeight;
    g.valid_height = tier->height;
    g.fit_width = g.width;
    g.fit_height = g.valid_height;
    if (std::uint64_t(source_width) * g.valid_height >= std::uint64_t(source_height) * g.width)
        g.fit_height = unsigned((std::uint64_t(source_height) * g.width + source_width / 2) / source_width);
    else
        g.fit_width = unsigned((std::uint64_t(source_width) * g.valid_height + source_height / 2) / source_height);
    g.fit_width = std::max(g.fit_width, 1u);
    g.fit_height = std::max(g.fit_height, 1u);
    g.x = (g.width - g.fit_width) / 2;
    g.y = (g.valid_height - g.fit_height) / 2;
    return g;
}

Result<void> validate(const Geometry& g)
{
    const auto expected = geometry(g.source_width, g.source_height, g.valid_height);
    if (!expected || std::memcmp(&*expected, &g, sizeof g)) return fail("inconsistent codec geometry");
    return {};
}

Result<void> encode_proxy(const std::uint8_t* source, const Geometry& g, bool fp16, std::vector<float>& rgba)
{
    DLSSLOP_TRY(validate(g));
    if (!source) return fail("null source image");
    const auto read8 = [&](unsigned x, unsigned y) {
        return rgba8(source + (std::size_t(y) * g.source_width + x) * 4);
    };
    const auto read16 = [&](unsigned x, unsigned y) {
        const std::uint8_t* p = source + (std::size_t(y) * g.source_width + x) * 8;
        return Rgb{unpack_half(p), unpack_half(p + 2), unpack_half(p + 4)};
    };
    rgba.resize(std::size_t(g.width) * g.height * 4);
    for (unsigned y = 0; y < g.valid_height; ++y) {
        for (unsigned x = 0; x < g.width; ++x) {
            Rgb c{};
            if (fitted(g, x, y))
                c = fp16 ? sample_proxy(read16, g, x, y) : sample_proxy(read8, g, x, y);
            if (!finite(c)) // Only FP16 samples can be.
                return reject("FP16 proxy contains nonfinite RGB samples");
            float* p = rgba.data() + (std::size_t(y) * g.width + x) * 4;
            p[0] = half_round(c.r);
            p[1] = half_round(c.g);
            p[2] = half_round(c.b);
            p[3] = 1.0f;
        }
    }
    reflect_padding(g, rgba);
    return {};
}

Result<void> feedback_neural_rgb(const float* neural_rgb, const Geometry& g, std::vector<float>& rgba,
                                 bool precision16)
{
    DLSSLOP_TRY(validate(g));
    if (!neural_rgb) return fail("null neural feedback image");
    rgba.resize(std::size_t(g.width) * g.height * 4);
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
                    return fail("neural feedback contains nonfinite or FP16-overflow samples");
            }
            float* p = rgba.data() + (std::size_t(y) * g.width + x) * 4;
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
            p[3] = 1.0f;
        }
    }
    reflect_padding(g, rgba);
    return {};
}

Result<void> decode_neural_proxy(const std::uint8_t* original, const Geometry& g, bool fp16,
                                 const float* neural_rgb, std::uint8_t* output)
{
    DLSSLOP_TRY(validate(g));
    if (!original || !neural_rgb) return fail("null decode image");
    const unsigned pixel_bytes = fp16 ? 8 : 4;
    for (unsigned y = 0; y < g.source_height; ++y) {
        for (unsigned x = 0; x < g.source_width; ++x) {
            const Rgb neural = answer(neural_rgb, g, x, y);
            if (!finite(neural)) return nonfinite_answer();
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
    return {};
}

} // namespace dlsslop
