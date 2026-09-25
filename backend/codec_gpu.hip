// SPDX-License-Identifier: MIT
// SDR GPU boundary port of lmxxf's native_codec_encode/decode shaders.
// See codec.cpp for the retained upstream copyright and MIT permission notice.
// The complete CPU reference remains in codec.cpp.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

typedef unsigned uint;
typedef unsigned char byte;
typedef unsigned short halfword;
#define DEV __attribute__((device)) __attribute__((always_inline))
#define KERNEL extern "C" __attribute__((global))

struct CodecGeometry {
    uint source_width, source_height, width, height, valid_width, valid_height;
    uint x, y, fit_width, fit_height;
};
struct Rgb { float r, g, b; };

DEV uint thread_index()
{
    return __builtin_amdgcn_workgroup_id_x() * 256u + __builtin_amdgcn_workitem_id_x();
}

DEV uint bits(float x)
{
    union { float f; uint u; } v;
    v.f = x;
    return v.u;
}

DEV bool finite(float x) { return (bits(x) & 0x7fffffffu) < 0x7f800000u; }
DEV uint minimum(uint a, uint b) { return a < b ? a : b; }
DEV float clamp(float v, float a, float b) { return v < a ? a : v > b ? b : v; }
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

DEV halfword dlsslop_pack_binary16(float x, uint* invalid)
{
    uint h;
    asm("v_cvt_f16_f32 %0, %1" : "=v"(h) : "v"(x));
    if ((h & 0x7c00u) == 0x7c00u) {
        __atomic_store_n(invalid, 1u, __ATOMIC_RELAXED);
        return 0;
    }
    return halfword(h);
}

DEV Rgb lerp(Rgb a, Rgb b, float f)
{
    return {a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f,
            a.b + (b.b - a.b) * f};
}

template<bool fp16>
DEV Rgb read_source(const byte* source, uint pixel, uint* invalid)
{
    if constexpr (fp16) {
        const halfword* p = reinterpret_cast<const halfword*>(source) + pixel * 4;
        const Rgb c{dlsslop_unpack_binary16(p[0]), dlsslop_unpack_binary16(p[1]),
                    dlsslop_unpack_binary16(p[2])};
        if (!finite(c.r) || !finite(c.g) || !finite(c.b)) {
            __atomic_store_n(invalid, 1u, __ATOMIC_RELAXED);
            return {};
        }
        return c;
    }
    return {float(source[pixel * 4]) / 255.0f,
            float(source[pixel * 4 + 1]) / 255.0f,
            float(source[pixel * 4 + 2]) / 255.0f};
}

template<bool fp16>
DEV Rgb sample_source(const byte* source, CodecGeometry g, float x, float y, uint* invalid)
{
    x = clamp(x, 0.0f, float(g.source_width - 1));
    y = clamp(y, 0.0f, float(g.source_height - 1));
    const uint x0 = uint(x), y0 = uint(y);
    const uint x1 = minimum(x0 + 1, g.source_width - 1);
    const uint y1 = minimum(y0 + 1, g.source_height - 1);
    return lerp(lerp(read_source<fp16>(source, y0 * g.source_width + x0, invalid),
                     read_source<fp16>(source, y0 * g.source_width + x1, invalid), x - float(x0)),
                lerp(read_source<fp16>(source, y1 * g.source_width + x0, invalid),
                     read_source<fp16>(source, y1 * g.source_width + x1, invalid), x - float(x0)), y - float(y0));
}

template<bool fp16>
DEV void encode_proxy(const byte* source, float* rgba, uint* invalid, CodecGeometry g)
{
    const uint index = thread_index();
    if (index >= g.width * g.height)
        return;
    const uint x = index % g.width;
    const uint processing_y = index / g.width;
    const uint y = processing_y < g.valid_height ? processing_y :
                   2 * g.valid_height - 2 - processing_y;
    Rgb c{};
    if (x >= g.x && x < g.x + g.fit_width && y >= g.y && y < g.y + g.fit_height) {
        const float sx = (float(x) + 0.5f - float(g.x)) * float(g.source_width) / float(g.fit_width) - 0.5f;
        const float sy = (float(y) + 0.5f - float(g.y)) * float(g.source_height) / float(g.fit_height) - 0.5f;
        c = sample_source<fp16>(source, g, sx, sy, invalid);
    }
    rgba[index * 4] = dlsslop_round_to_binary16(c.r);
    rgba[index * 4 + 1] = dlsslop_round_to_binary16(c.g);
    rgba[index * 4 + 2] = dlsslop_round_to_binary16(c.b);
    rgba[index * 4 + 3] = 1.0f;
}

KERNEL void dlsslop_encode_rgba8(const byte* source, float* rgba, uint* invalid,
                              CodecGeometry g)
{
    encode_proxy<false>(source, rgba, invalid, g);
}

KERNEL void dlsslop_encode_rgba16f(const byte* source, float* rgba, uint* invalid,
                                CodecGeometry g)
{
    encode_proxy<true>(source, rgba, invalid, g);
}

DEV byte unorm8(float x)
{
    const float scaled = clamp(x, 0.0f, 1.0f) * 255.0f;
    const uint low = uint(scaled);
    const float fraction = scaled - float(low);
    return byte(low + uint(fraction > 0.5f || (fraction == 0.5f && (low & 1u))));
}

DEV Rgb read_neural(const float* neural, uint pixel, uint* invalid)
{
    Rgb c{dlsslop_round_to_binary16(neural[pixel * 3]),
          dlsslop_round_to_binary16(neural[pixel * 3 + 1]),
          dlsslop_round_to_binary16(neural[pixel * 3 + 2])};
    if (!finite(c.r) || !finite(c.g) || !finite(c.b)) {
        __atomic_store_n(invalid, 1u, __ATOMIC_RELAXED);
        return {};
    }
    return c;
}

KERNEL void dlsslop_feedback_rgb(const float* neural_rgb, float* rgba,
                               uint* invalid, CodecGeometry g, uint precision16)
{
    const uint index = thread_index();
    if (index >= g.width * g.height)
        return;
    const uint x = index % g.width;
    const uint processing_y = index / g.width;
    const uint y = processing_y < g.valid_height ? processing_y :
                   2 * g.valid_height - 2 - processing_y;
    Rgb c{};
    if (x >= g.x && x < g.x + g.fit_width && y >= g.y && y < g.y + g.fit_height) {
        const uint pixel = y * g.width + x;
        if (precision16) {
            c = read_neural(neural_rgb, pixel, invalid);
        } else {
            const float* p = neural_rgb + pixel * 3;
            if (!finite(p[0]) || !finite(p[1]) || !finite(p[2])) {
                __atomic_store_n(invalid, 1u, __ATOMIC_RELAXED);
            } else {
                c = {dlsslop_round_to_binary16(float(unorm8(p[0])) / 255.0f),
                     dlsslop_round_to_binary16(float(unorm8(p[1])) / 255.0f),
                     dlsslop_round_to_binary16(float(unorm8(p[2])) / 255.0f)};
            }
        }
    }
    rgba[index * 4] = c.r;
    rgba[index * 4 + 1] = c.g;
    rgba[index * 4 + 2] = c.b;
    rgba[index * 4 + 3] = 1.0f;
}

DEV Rgb sample_neural(const float* source, CodecGeometry g, float x, float y, uint* invalid)
{
    x = clamp(x, float(g.x), float(g.x + g.fit_width - 1));
    y = clamp(y, float(g.y), float(g.y + g.fit_height - 1));
    const uint x0 = uint(x), y0 = uint(y);
    const uint x1 = minimum(x0 + 1, g.x + g.fit_width - 1);
    const uint y1 = minimum(y0 + 1, g.y + g.fit_height - 1);
    return lerp(lerp(read_neural(source, y0 * g.width + x0, invalid),
                     read_neural(source, y0 * g.width + x1, invalid), x - float(x0)),
                lerp(read_neural(source, y1 * g.width + x0, invalid),
                     read_neural(source, y1 * g.width + x1, invalid), x - float(x0)), y - float(y0));
}

KERNEL void dlsslop_decode_rgba8(const byte* original, const float* neural_rgb,
                              byte* output, uint* invalid, CodecGeometry g)
{
    const uint pixel = thread_index();
    if (pixel >= g.source_width * g.source_height)
        return;
    const uint x = pixel % g.source_width, y = pixel / g.source_width;
    const float nx = float(g.x) + (float(x) + 0.5f) * float(g.fit_width) / float(g.source_width) - 0.5f;
    const float ny = float(g.y) + (float(y) + 0.5f) * float(g.fit_height) / float(g.source_height) - 0.5f;
    const Rgb c = sample_neural(neural_rgb, g, nx, ny, invalid);
    output[pixel * 4] = unorm8(c.r);
    output[pixel * 4 + 1] = unorm8(c.g);
    output[pixel * 4 + 2] = unorm8(c.b);
    output[pixel * 4 + 3] = original[pixel * 4 + 3];
}

KERNEL void dlsslop_decode_rgba16f(const halfword* original, const float* neural_rgb,
                                halfword* output, uint* invalid, CodecGeometry g)
{
    const uint pixel = thread_index();
    if (pixel >= g.source_width * g.source_height)
        return;
    const uint x = pixel % g.source_width, y = pixel / g.source_width;
    const float nx = float(g.x) + (float(x) + 0.5f) * float(g.fit_width) / float(g.source_width) - 0.5f;
    const float ny = float(g.y) + (float(y) + 0.5f) * float(g.fit_height) / float(g.source_height) - 0.5f;
    const Rgb c = sample_neural(neural_rgb, g, nx, ny, invalid);
    output[pixel * 4] = dlsslop_pack_binary16(c.r, invalid);
    output[pixel * 4 + 1] = dlsslop_pack_binary16(c.g, invalid);
    output[pixel * 4 + 2] = dlsslop_pack_binary16(c.b, invalid);
    output[pixel * 4 + 3] = original[pixel * 4 + 3];
}
