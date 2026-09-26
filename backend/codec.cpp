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

#include "codec.h"
#include "codec_math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace dlsslop {
namespace {

Rgb operator*(Rgb c, float x)
{
    return {c.r * x, c.g * x, c.b * x};
}

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

float saturate(float x)
{
    // Nonfinite output must never be allowed to reach an integer conversion.
    return std::isnan(x) ? 0.0f : std::clamp(x, 0.0f, 1.0f);
}

float srgb_decode(float x)
{
    x = saturate(x);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

Rgb decode(Rgb x)
{
    return {srgb_decode(x.r), srgb_decode(x.g), srgb_decode(x.b)};
}

float srgb_encode(float x)
{
    x = saturate(x);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

float luminance(Rgb x)
{
    return x.r * 0.212639f + x.g * 0.715169f + x.b * 0.072192f;
}

Rgb to_lab(Rgb c)
{
    const float l = std::cbrt(0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b);
    const float m = std::cbrt(0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b);
    const float s = std::cbrt(0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b);
    return {0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
            1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
            0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s};
}

Rgb from_lab(Rgb c)
{
    float l = c.r + 0.3963377774f * c.g + 0.2158037573f * c.b;
    float m = c.r - 0.1055613458f * c.g - 0.0638541728f * c.b;
    float s = c.r - 0.0894841775f * c.g - 1.2914855480f * c.b;
    l *= l * l;
    m *= m * m;
    s *= s * s;
    return {4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
            -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
            -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s};
}

Rgb clamp_ap1(Rgb c)
{
    const float r = std::max(0.0f, 0.613097f * c.r + 0.339523f * c.g + 0.047379f * c.b);
    const float g = std::max(0.0f, 0.070194f * c.r + 0.916354f * c.g + 0.013452f * c.b);
    const float b = std::max(0.0f, 0.020616f * c.r + 0.109570f * c.g + 0.869815f * c.b);
    return {1.705051f * r - 0.621792f * g - 0.083259f * b,
            -0.130256f * r + 1.140805f * g - 0.010548f * b,
            -0.024003f * r - 0.128969f * g + 1.152972f * b};
}

Rgb hue(Rgb incorrect, Rgb correct)
{
    Rgb a = to_lab(incorrect);
    const Rgb b = to_lab(correct);
    const float ca = std::hypot(a.g, a.b), cb = std::hypot(b.g, b.b);
    const float scale = cb == 0.0f ? 1.0f : ca / cb;
    a.g = b.g * scale;
    a.b = b.b * scale;
    return clamp_ap1(from_lab(a));
}

// Upstream lerps by TransferStrength, here at full strength: a + (b - a) * 1
// need not round to b, so the lerp stays (as for ColorStrength below).
Rgb upgrade(Rgb original, Rgb proxy, Rgb neural)
{
    const float oy = luminance(original), py = luminance(proxy), ny = luminance(neural);
    if (ny <= 1e-5f)
        return original;
    const float ratio = oy < py ? oy / std::max(py, 1e-6f) :
        (ny + std::max(0.0f, oy - py)) / ny;
    return lerp(original, hue(neural * ratio, neural), 1.0f);
}

Rgb rgba8(const std::uint8_t* p)
{
    return {float(p[0]) / 255.0f, float(p[1]) / 255.0f, float(p[2]) / 255.0f};
}

// The answer at source pixel (x, y) through the upstream FP16 surface. Like
// the GPU codec, reject it when a texel it reads is not a finite binary16.
// Out of line: inlining it into both decoders adds about 9 KB of text.
[[gnu::noinline]] Rgb answer(const float* neural_rgb, const Geometry& g, unsigned x, unsigned y)
{
    const Rgb c = sample_answer([&](unsigned px, unsigned py) {
        const float* p = neural_rgb + (std::size_t(py) * g.width + px) * 3;
        return Rgb{half_round(p[0]), half_round(p[1]), half_round(p[2])};
    }, g, x, y);
    if (!finite(c))
        throw std::range_error("neural output contains nonfinite or FP16-overflow samples");
    return c;
}

void reflect_padding(const Geometry& g, std::vector<float>& rgba)
{
    const std::size_t row = std::size_t(g.width) * 4;
    for (unsigned y = g.valid_height; y < g.height; ++y)
        std::memcpy(rgba.data() + y * row, rgba.data() + codec_row(g, y) * row, row * sizeof(float));
}

} // namespace

Geometry geometry(unsigned source_width, unsigned source_height, unsigned tier_height)
{
    if (!source_width || !source_height || source_width > 16384 || source_height > 16384)
        throw std::invalid_argument("source extent must be in 1..16384");
    if (!tier_height)
        tier_height = source_width <= 1280 && source_height <= 720 ? 720 :
                      source_width <= 1600 && source_height <= 900 ? 900 : 1080;
    Geometry g;
    g.source_width = source_width;
    g.source_height = source_height;
    switch (tier_height) {
    case 720: g.width = 1280; g.height = 768; break;
    case 900: g.width = 1600; g.height = 960; break;
    case 1080: g.width = 1920; g.height = 1152; break;
    default: throw std::invalid_argument("network height must be 0, 720, 900 or 1080");
    }
    g.valid_height = tier_height;
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

void validate(const Geometry& g)
{
    const Geometry expected = geometry(g.source_width, g.source_height, g.valid_height);
    if (std::memcmp(&expected, &g, sizeof g))
        throw std::invalid_argument("inconsistent codec geometry");
}

void encode_proxy(const std::uint8_t* source, const Geometry& g, bool fp16,
                  std::vector<float>& rgba)
{
    validate(g);
    if (!source)
        throw std::invalid_argument("null source image");
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
                throw std::range_error("FP16 proxy contains nonfinite RGB samples");
            float* p = rgba.data() + (std::size_t(y) * g.width + x) * 4;
            p[0] = half_round(c.r);
            p[1] = half_round(c.g);
            p[2] = half_round(c.b);
            p[3] = 1.0f;
        }
    }
    reflect_padding(g, rgba);
}

void feedback_neural_rgb(const float* neural_rgb, const Geometry& g,
                         std::vector<float>& rgba, bool precision16)
{
    validate(g);
    if (!neural_rgb)
        throw std::invalid_argument("null neural feedback image");
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
                    throw std::runtime_error("neural feedback contains nonfinite or FP16-overflow samples");
            }
            float* p = rgba.data() + (std::size_t(y) * g.width + x) * 4;
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
            p[3] = 1.0f;
        }
    }
    reflect_padding(g, rgba);
}

void decode_neural_proxy(const std::uint8_t* original, const Geometry& g, bool fp16,
                         const float* neural_rgb, std::uint8_t* output)
{
    validate(g);
    if (!original || !neural_rgb)
        throw std::invalid_argument("null decode image");
    const unsigned pixel_bytes = fp16 ? 8 : 4;
    for (unsigned y = 0; y < g.source_height; ++y) {
        for (unsigned x = 0; x < g.source_width; ++x) {
            const Rgb neural = answer(neural_rgb, g, x, y);
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
}

void decode_rgba8(const std::uint8_t* original, const Geometry& g,
                  const float* encoded_rgba, const float* neural_rgb,
                  std::uint8_t* output)
{
    validate(g);
    if (!original || !encoded_rgba || !neural_rgb)
        throw std::invalid_argument("null decode image");
    const auto encoded = [&](unsigned x, unsigned y) {
        const float* p = encoded_rgba + (std::size_t(y) * g.width + x) * 4;
        return Rgb{p[0], p[1], p[2]};
    };
    for (unsigned y = 0; y < g.source_height; ++y) {
        for (unsigned x = 0; x < g.source_width; ++x) {
            const std::size_t pixel = std::size_t(y) * g.source_width + x;
            const Rgb source = decode(rgba8(original + pixel * 4));
            const Rgb proxy = decode(sample_answer(encoded, g, x, y));
            const Rgb neural = decode(answer(neural_rgb, g, x, y));
            const Rgb upgraded = upgrade(source, proxy, neural);
            const float oy = luminance(source), uy = luminance(upgraded);
            const float ratio = oy == 0.0f ? 1.0f : std::clamp(uy / oy, 0.0f, 4.0f);
            const Rgb result = lerp(source * ratio, upgraded, 1.0f);
            output[pixel * 4] = unorm8(srgb_encode(result.r));
            output[pixel * 4 + 1] = unorm8(srgb_encode(result.g));
            output[pixel * 4 + 2] = unorm8(srgb_encode(result.b));
            output[pixel * 4 + 3] = original[pixel * 4 + 3];
        }
    }
}

} // namespace dlsslop
