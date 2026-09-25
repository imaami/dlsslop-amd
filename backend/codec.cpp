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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace dlsslop {
namespace {

struct Vec3 {
    float r, g, b;
    Vec3 operator*(float x) const { return {r * x, g * x, b * x}; }
};

Vec3 lerp(Vec3 a, Vec3 b, float f)
{
    return {a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f,
            a.b + (b.b - a.b) * f};
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

void pack_half(float input, std::uint8_t* output)
{
    const float rounded = half_round(input);
    if (!std::isfinite(rounded))
        throw std::range_error("neural output contains nonfinite or FP16-overflow samples");
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
    // A failed neural value is exposed as black rather than invoking UB.
    return std::isnan(x) ? 0.0f : std::clamp(x, 0.0f, 1.0f);
}

float srgb_decode(float x)
{
    x = saturate(x);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

Vec3 decode(Vec3 x)
{
    return {srgb_decode(x.r), srgb_decode(x.g), srgb_decode(x.b)};
}

float srgb_encode(float x)
{
    x = saturate(x);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

std::uint8_t unorm8(float x)
{
    // HLSL round() is nearest-even; do not inherit the process's fenv mode.
    const float scaled = saturate(x) * 255.0f;
    const unsigned low = static_cast<unsigned>(scaled);
    const float fraction = scaled - float(low);
    return static_cast<std::uint8_t>(low + unsigned(
        fraction > 0.5f || (fraction == 0.5f && (low & 1))));
}

float luminance(Vec3 x)
{
    return x.r * 0.212639f + x.g * 0.715169f + x.b * 0.072192f;
}

Vec3 to_lab(Vec3 c)
{
    const float l = std::cbrt(0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b);
    const float m = std::cbrt(0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b);
    const float s = std::cbrt(0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b);
    return {0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
            1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
            0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s};
}

Vec3 from_lab(Vec3 c)
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

Vec3 clamp_ap1(Vec3 c)
{
    const float r = std::max(0.0f, 0.613097f * c.r + 0.339523f * c.g + 0.047379f * c.b);
    const float g = std::max(0.0f, 0.070194f * c.r + 0.916354f * c.g + 0.013452f * c.b);
    const float b = std::max(0.0f, 0.020616f * c.r + 0.109570f * c.g + 0.869815f * c.b);
    return {1.705051f * r - 0.621792f * g - 0.083259f * b,
            -0.130256f * r + 1.140805f * g - 0.010548f * b,
            -0.024003f * r - 0.128969f * g + 1.152972f * b};
}

Vec3 hue(Vec3 incorrect, Vec3 correct)
{
    Vec3 a = to_lab(incorrect);
    const Vec3 b = to_lab(correct);
    const float ca = std::hypot(a.g, a.b), cb = std::hypot(b.g, b.b);
    const float scale = cb == 0.0f ? 1.0f : ca / cb;
    a.g = b.g * scale;
    a.b = b.b * scale;
    return clamp_ap1(from_lab(a));
}

Vec3 upgrade(Vec3 original, Vec3 proxy, Vec3 neural, float strength)
{
    const float oy = luminance(original), py = luminance(proxy), ny = luminance(neural);
    if (ny <= 1e-5f)
        return original;
    const float ratio = oy < py ? oy / std::max(py, 1e-6f) :
        (ny + std::max(0.0f, oy - py)) / ny;
    return lerp(original, hue(neural * ratio, neural), strength);
}

void validate(const Geometry& g)
{
    const Geometry expected = geometry(g.source_width, g.source_height, g.valid_height);
    if (g.width != expected.width || g.height != expected.height ||
        g.valid_width != expected.valid_width || g.valid_height != expected.valid_height ||
        g.x != expected.x ||
        g.y != expected.y || g.fit_width != expected.fit_width ||
        g.fit_height != expected.fit_height)
        throw std::invalid_argument("inconsistent codec geometry");
}

Vec3 read_rgba8(const std::uint8_t* p, std::size_t pixel)
{
    return {float(p[pixel * 4]) / 255.0f, float(p[pixel * 4 + 1]) / 255.0f,
            float(p[pixel * 4 + 2]) / 255.0f};
}

Vec3 read_proxy(const std::uint8_t* source, std::size_t pixel, bool fp16)
{
    if (!fp16)
        return read_rgba8(source, pixel);
    const std::uint8_t* p = source + pixel * 8;
    const Vec3 c{unpack_half(p), unpack_half(p + 2), unpack_half(p + 4)};
    if (!std::isfinite(c.r) || !std::isfinite(c.g) || !std::isfinite(c.b))
        throw std::range_error("FP16 proxy contains nonfinite RGB samples");
    return c;
}

Vec3 sample_source(const std::uint8_t* source, const Geometry& g, float x, float y,
                   bool fp16)
{
    x = std::clamp(x, 0.0f, float(g.source_width - 1));
    y = std::clamp(y, 0.0f, float(g.source_height - 1));
    const unsigned x0 = unsigned(x), y0 = unsigned(y);
    const unsigned x1 = std::min(x0 + 1, g.source_width - 1);
    const unsigned y1 = std::min(y0 + 1, g.source_height - 1);
    const float fx = x - float(x0), fy = y - float(y0);
    return lerp(lerp(read_proxy(source, std::size_t(y0) * g.source_width + x0, fp16),
                     read_proxy(source, std::size_t(y0) * g.source_width + x1, fp16), fx),
                lerp(read_proxy(source, std::size_t(y1) * g.source_width + x0, fp16),
                     read_proxy(source, std::size_t(y1) * g.source_width + x1, fp16), fx), fy);
}

Vec3 sample_network(const float* source, const Geometry& g, unsigned channels,
                    float x, float y, bool round_half)
{
    x = std::clamp(x, float(g.x), float(g.x + g.fit_width - 1));
    y = std::clamp(y, float(g.y), float(g.y + g.fit_height - 1));
    const unsigned x0 = unsigned(x), y0 = unsigned(y);
    const unsigned x1 = std::min(x0 + 1, g.x + g.fit_width - 1);
    const unsigned y1 = std::min(y0 + 1, g.y + g.fit_height - 1);
    const float fx = x - float(x0), fy = y - float(y0);
    const auto read = [&](unsigned px, unsigned py) {
        const float* p = source + (std::size_t(py) * g.width + px) * channels;
        return round_half ? Vec3{half_round(p[0]), half_round(p[1]), half_round(p[2])} :
                            Vec3{p[0], p[1], p[2]};
    };
    return lerp(lerp(read(x0, y0), read(x1, y0), fx),
                lerp(read(x0, y1), read(x1, y1), fx), fy);
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
    g.valid_width = g.width;
    g.valid_height = tier_height;
    g.fit_width = g.valid_width;
    g.fit_height = g.valid_height;
    if (std::uint64_t(source_width) * g.valid_height >= std::uint64_t(source_height) * g.valid_width)
        g.fit_height = unsigned((std::uint64_t(source_height) * g.valid_width + source_width / 2) / source_width);
    else
        g.fit_width = unsigned((std::uint64_t(source_width) * g.valid_height + source_height / 2) / source_height);
    g.fit_width = std::max(g.fit_width, 1u);
    g.fit_height = std::max(g.fit_height, 1u);
    g.x = (g.valid_width - g.fit_width) / 2;
    g.y = (g.valid_height - g.fit_height) / 2;
    return g;
}

void encode_rgba8(const std::uint8_t* source, const Geometry& g, std::vector<float>& rgba)
{
    encode_proxy(source, g, false, rgba);
}

void encode_proxy(const std::uint8_t* source, const Geometry& g, bool fp16,
                  std::vector<float>& rgba)
{
    validate(g);
    if (!source)
        throw std::invalid_argument("null source image");
    rgba.resize(std::size_t(g.width) * g.height * 4);
    for (unsigned y = 0; y < g.valid_height; ++y) {
        for (unsigned x = 0; x < g.valid_width; ++x) {
            Vec3 c{};
            if (x >= g.x && x < g.x + g.fit_width && y >= g.y && y < g.y + g.fit_height) {
                const float sx = (float(x) + 0.5f - float(g.x)) * float(g.source_width) / float(g.fit_width) - 0.5f;
                const float sy = (float(y) + 0.5f - float(g.y)) * float(g.source_height) / float(g.fit_height) - 0.5f;
                c = sample_source(source, g, sx, sy, fp16);
            }
            float* p = rgba.data() + (std::size_t(y) * g.width + x) * 4;
            p[0] = half_round(c.r);
            p[1] = half_round(c.g);
            p[2] = half_round(c.b);
            p[3] = 1.0f;
        }
    }
    // reflect101: 0,1,...,h-2,h-1,h-2,... . All supported tiers need less
    // than one reflected period, matching native_game_rgb_input.hlsl.
    for (unsigned y = g.valid_height; y < g.height; ++y) {
        const unsigned reflected = 2 * g.valid_height - 2 - y;
        std::memcpy(rgba.data() + std::size_t(y) * g.width * 4,
                    rgba.data() + std::size_t(reflected) * g.width * 4,
                    std::size_t(g.width) * 4 * sizeof(float));
    }
}

void feedback_neural_rgb(const float* neural_rgb, const Geometry& g,
                         std::vector<float>& rgba, bool precision16)
{
    validate(g);
    if (!neural_rgb)
        throw std::invalid_argument("null neural feedback image");
    rgba.resize(std::size_t(g.width) * g.height * 4);
    for (unsigned y = 0; y < g.valid_height; ++y) {
        for (unsigned x = 0; x < g.valid_width; ++x) {
            Vec3 c{};
            if (x >= g.x && x < g.x + g.fit_width && y >= g.y && y < g.y + g.fit_height) {
                const float* p = neural_rgb + (std::size_t(y) * g.width + x) * 3;
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
                    throw std::runtime_error("neural feedback contains nonfinite samples");
                c = precision16 ? Vec3{half_round(p[0]), half_round(p[1]), half_round(p[2])} :
                    Vec3{half_round(float(unorm8(p[0])) / 255.0f),
                         half_round(float(unorm8(p[1])) / 255.0f),
                         half_round(float(unorm8(p[2])) / 255.0f)};
                if (!std::isfinite(c.r) || !std::isfinite(c.g) || !std::isfinite(c.b))
                    throw std::runtime_error("neural feedback contains nonfinite or FP16-overflow samples");
            }
            float* p = rgba.data() + (std::size_t(y) * g.width + x) * 4;
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
            p[3] = 1.0f;
        }
    }
    for (unsigned y = g.valid_height; y < g.height; ++y) {
        const unsigned reflected = 2 * g.valid_height - 2 - y;
        std::memcpy(rgba.data() + std::size_t(y) * g.width * 4,
                    rgba.data() + std::size_t(reflected) * g.width * 4,
                    std::size_t(g.width) * 4 * sizeof(float));
    }
}

void decode_neural_rgba8(const std::uint8_t* original, const Geometry& g,
                         const float* neural_rgb, std::vector<std::uint8_t>& output)
{
    decode_neural_proxy(original, g, false, neural_rgb, output);
}

void decode_neural_proxy(const std::uint8_t* original, const Geometry& g, bool fp16,
                         const float* neural_rgb, std::vector<std::uint8_t>& output)
{
    validate(g);
    if (!original || !neural_rgb)
        throw std::invalid_argument("null decode image");
    const unsigned pixel_bytes = fp16 ? 8 : 4;
    output.resize(std::size_t(g.source_width) * g.source_height * pixel_bytes);
    for (unsigned y = 0; y < g.source_height; ++y) {
        const float ny = float(g.y) + (float(y) + 0.5f) * float(g.fit_height) / float(g.source_height) - 0.5f;
        for (unsigned x = 0; x < g.source_width; ++x) {
            const float nx = float(g.x) + (float(x) + 0.5f) * float(g.fit_width) / float(g.source_width) - 0.5f;
            const Vec3 neural = sample_network(neural_rgb, g, 3, nx, ny, true);
            const std::size_t p = (std::size_t(y) * g.source_width + x) * pixel_bytes;
            if (fp16) {
                pack_half(neural.r, output.data() + p);
                pack_half(neural.g, output.data() + p + 2);
                pack_half(neural.b, output.data() + p + 4);
                std::memcpy(output.data() + p + 6, original + p + 6, 2);
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
                  std::vector<std::uint8_t>& output, float transfer_strength,
                  float color_strength)
{
    validate(g);
    if (!original || !encoded_rgba || !neural_rgb)
        throw std::invalid_argument("null decode image");
    if (!std::isfinite(transfer_strength) || !std::isfinite(color_strength) ||
        transfer_strength < 0.0f || transfer_strength > 1.0f ||
        color_strength < 0.0f || color_strength > 1.0f)
        throw std::invalid_argument("codec strengths must be finite in [0,1]");
    output.resize(std::size_t(g.source_width) * g.source_height * 4);
    if (transfer_strength == 0.0f) {
        std::memcpy(output.data(), original, output.size());
        return;
    }
    for (unsigned y = 0; y < g.source_height; ++y) {
        const float ny = float(g.y) + (float(y) + 0.5f) * float(g.fit_height) / float(g.source_height) - 0.5f;
        for (unsigned x = 0; x < g.source_width; ++x) {
            const float nx = float(g.x) + (float(x) + 0.5f) * float(g.fit_width) / float(g.source_width) - 0.5f;
            const std::size_t pixel = std::size_t(y) * g.source_width + x;
            const Vec3 source = decode(read_rgba8(original, pixel));
            const Vec3 proxy = decode(sample_network(encoded_rgba, g, 4, nx, ny, false));
            const Vec3 neural = decode(sample_network(neural_rgb, g, 3, nx, ny, true));
            const Vec3 upgraded = upgrade(source, proxy, neural, transfer_strength);
            const float oy = luminance(source), uy = luminance(upgraded);
            const float ratio = oy == 0.0f ? 1.0f : std::clamp(uy / oy, 0.0f, 4.0f);
            const Vec3 result = lerp(source * ratio, upgraded, color_strength);
            output[pixel * 4] = unorm8(srgb_encode(result.r));
            output[pixel * 4 + 1] = unorm8(srgb_encode(result.g));
            output[pixel * 4 + 2] = unorm8(srgb_encode(result.b));
            output[pixel * 4 + 3] = original[pixel * 4 + 3];
        }
    }
}

} // namespace dlsslop
