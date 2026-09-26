// SPDX-License-Identifier: MIT
// SDR GPU boundary port of lmxxf's native_codec_encode/decode shaders.
// See codec.cpp for the retained upstream copyright and MIT permission notice.
// The complete CPU reference remains in codec.cpp.
#include "kernel.h"
#include "codec_math.h"

typedef unsigned uint;
typedef unsigned char byte;
typedef unsigned short halfword;
typedef dlsslop::Geometry Geometry;
typedef dlsslop::Rgb Rgb;

// Do not call this helper half_round: AMD Clang's OpenCL library simplifier
// recognizes that mangled name and replaces even a user-defined body with
// llvm.round.f32, rounding samples to whole integers instead of binary16.
DEV float dlsslop_round_to_binary16(float x)
{
    uint h;
    float result;
    // Two explicit conversion instructions preserve both rounding boundaries,
    // including subnormals, as in the upstream production HIP kernels.
    asm("v_cvt_f16_f32 %0, %1" : "=v"(h) : "v"(x));
    asm("v_cvt_f32_f16 %0, %1" : "=v"(result) : "v"(h));
    return result;
}

DEV float dlsslop_unpack_binary16(halfword h)
{
    float result;
    asm("v_cvt_f32_f16 %0, %1" : "=v"(result) : "v"(uint(h)));
    return result;
}

// Only ever given finite samples of finite binary16 texels, which cannot
// round past binary16.
DEV halfword dlsslop_pack_binary16(float x)
{
    uint h;
    asm("v_cvt_f16_f32 %0, %1" : "=v"(h) : "v"(x));
    return halfword(h);
}

DEV Rgb round_to_binary16(const float* p)
{
    return {dlsslop_round_to_binary16(p[0]), dlsslop_round_to_binary16(p[1]), dlsslop_round_to_binary16(p[2])};
}

// A nonfinite sample flags the frame, which the host then rejects, and
// becomes zero, so no NaN reaches an integer conversion.
DEV Rgb checked(Rgb c, uint* invalid)
{
    if (finite(c))
        return c;
    __atomic_store_n(invalid, 1u, __ATOMIC_RELAXED);
    return {};
}

template<bool fp16>
DEV void encode_proxy(const byte* __restrict__ source, float* __restrict__ rgba, uint* __restrict__ invalid,
                      Geometry g)
{
    const uint index = thread_index();
    if (index >= g.width * g.height)
        return;
    const uint x = index % g.width, y = codec_row(g, index / g.width);
    Rgb c{};
    if (fitted(g, x, y)) {
        if constexpr (fp16) {
            c = checked(sample_proxy([=](uint px, uint py) {
                const halfword* p = reinterpret_cast<const halfword*>(source) + (py * g.source_width + px) * 4;
                return Rgb{dlsslop_unpack_binary16(p[0]), dlsslop_unpack_binary16(p[1]),
                           dlsslop_unpack_binary16(p[2])};
            }, g, x, y), invalid);
        } else {
            c = sample_proxy([=](uint px, uint py) {
                const byte* p = source + (py * g.source_width + px) * 4;
                return Rgb{float(p[0]) / 255.0f, float(p[1]) / 255.0f, float(p[2]) / 255.0f};
            }, g, x, y);
        }
    }
    rgba[index * 4] = dlsslop_round_to_binary16(c.r);
    rgba[index * 4 + 1] = dlsslop_round_to_binary16(c.g);
    rgba[index * 4 + 2] = dlsslop_round_to_binary16(c.b);
    rgba[index * 4 + 3] = 1.0f;
}

KERNEL void dlsslop_encode_rgba8(const byte* source, float* rgba, uint* invalid,
                              Geometry g)
{
    encode_proxy<false>(source, rgba, invalid, g);
}

KERNEL void dlsslop_encode_rgba16f(const byte* source, float* rgba, uint* invalid,
                                Geometry g)
{
    encode_proxy<true>(source, rgba, invalid, g);
}

// Binary16 feedback flags what rounds past binary16; UNORM8 feedback, which
// clamps, only a nonfinite raw sample.
KERNEL void dlsslop_feedback_rgb(const float* __restrict__ neural_rgb, float* __restrict__ rgba,
                               uint* __restrict__ invalid, Geometry g, uint precision16)
{
    const uint index = thread_index();
    if (index >= g.width * g.height)
        return;
    const uint x = index % g.width, y = codec_row(g, index / g.width);
    Rgb c{};
    if (fitted(g, x, y)) {
        const float* p = neural_rgb + (y * g.width + x) * 3;
        if (precision16) {
            c = checked(round_to_binary16(p), invalid);
        } else {
            c = checked({p[0], p[1], p[2]}, invalid);
            c = {dlsslop_round_to_binary16(float(dlsslop::unorm8(c.r)) / 255.0f),
                 dlsslop_round_to_binary16(float(dlsslop::unorm8(c.g)) / 255.0f),
                 dlsslop_round_to_binary16(float(dlsslop::unorm8(c.b)) / 255.0f)};
        }
    }
    rgba[index * 4] = c.r;
    rgba[index * 4 + 1] = c.g;
    rgba[index * 4 + 2] = c.b;
    rgba[index * 4 + 3] = 1.0f;
}

// The answer at source pixel (x, y) through the upstream FP16 surface: all
// four texels load before the one check of their sample.
DEV Rgb answer(const float* __restrict__ neural_rgb, Geometry g, uint pixel, uint* __restrict__ invalid)
{
    return checked(sample_answer([=](uint px, uint py) { return round_to_binary16(neural_rgb + (py * g.width + px) * 3); },
                                 g, pixel % g.source_width, pixel / g.source_width), invalid);
}

// Decode replaces RGB in the uploaded proxy and keeps its alpha. Each pixel is
// one whole-word read-modify-write: a partial-word store makes the readback slower.
KERNEL void dlsslop_decode_rgba8(uint* __restrict__ proxy, const float* __restrict__ neural_rgb,
                              uint* __restrict__ invalid, Geometry g)
{
    const uint pixel = thread_index();
    if (pixel >= g.source_width * g.source_height)
        return;
    const Rgb c = answer(neural_rgb, g, pixel, invalid);
    proxy[pixel] = (proxy[pixel] & 0xff000000u) | dlsslop::unorm8(c.r) | uint(dlsslop::unorm8(c.g)) << 8 |
                   uint(dlsslop::unorm8(c.b)) << 16;
}

KERNEL void dlsslop_decode_rgba16f(unsigned long long* __restrict__ proxy, const float* __restrict__ neural_rgb,
                                uint* __restrict__ invalid, Geometry g)
{
    const uint pixel = thread_index();
    if (pixel >= g.source_width * g.source_height)
        return;
    const Rgb c = answer(neural_rgb, g, pixel, invalid);
    proxy[pixel] = (proxy[pixel] & 0xffff000000000000ull) | dlsslop_pack_binary16(c.r) |
                   uint(dlsslop_pack_binary16(c.g)) << 16 | (unsigned long long)dlsslop_pack_binary16(c.b) << 32;
}
