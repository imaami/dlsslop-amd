// SPDX-License-Identifier: MIT
// The codec's sampling math, shared by the CPU reference (codec.cpp) and the
// GPU kernels (codec_gpu.hip), which must agree bit for bit.
#pragma once

#include "geometry.h"

namespace dlsslop {

struct Rgb {
    float r, g, b;
};

DLSSLOP_INLINE Rgb lerp(Rgb a, Rgb b, float f)
{
    return {a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f, a.b + (b.b - a.b) * f};
}

DLSSLOP_INLINE bool finite(Rgb c)
{
    return __builtin_isfinite(c.r) && __builtin_isfinite(c.g) && __builtin_isfinite(c.b);
}

// Whether processing pixel (x, y) shows the picture. Unsigned wrap-around
// also excludes the pixels left of and above it.
DLSSLOP_INLINE bool fitted(const Geometry& g, unsigned x, unsigned y)
{
    return x - g.x < g.fit_width && y - g.y < g.fit_height;
}

// The viewport row a processing row holds: the padding below the viewport
// reflects it (reflect101: ..., h-2, h-1, h-2, ...), as upstream's
// native_game_rgb_input.hlsl does. Every tier pads less than one period.
DLSSLOP_INLINE unsigned codec_row(const Geometry& g, unsigned y)
{
    return y < g.valid_height ? y : 2 * g.valid_height - 2 - y;
}

// HLSL round() is nearest-even; do not inherit the rounding mode. NaN, like
// anything below zero, becomes 0 and never reaches the integer conversion.
DLSSLOP_INLINE unsigned char unorm8(float x)
{
    x = x > 0.0f ? x : 0.0f;
    const float scaled = (x < 1.0f ? x : 1.0f) * 255.0f;
    const unsigned low = unsigned(scaled);
    const float fraction = scaled - float(low);
    return (unsigned char)(low + unsigned(fraction > 0.5f || (fraction == 0.5f && (low & 1u))));
}

// Bilinear filter at (x, y), clamped to the texels low..high on each axis, of
// read(x, y), the texel at integer coordinates. A nonfinite texel makes the
// result nonfinite even at zero weight (inf * 0 is NaN), so callers check
// the result, not each texel.
template<class Read>
DLSSLOP_INLINE Rgb bilinear(const Read& read, float x, float y, unsigned low_x, unsigned low_y,
                            unsigned high_x, unsigned high_y)
{
    x = x < float(low_x) ? float(low_x) : x > float(high_x) ? float(high_x) : x;
    y = y < float(low_y) ? float(low_y) : y > float(high_y) ? float(high_y) : y;
    const unsigned x0 = unsigned(x), y0 = unsigned(y);
    const unsigned x1 = x0 < high_x ? x0 + 1 : high_x, y1 = y0 < high_y ? y0 + 1 : high_y;
    const float fx = x - float(x0), fy = y - float(y0);
    return lerp(lerp(read(x0, y0), read(x1, y0), fx), lerp(read(x0, y1), read(x1, y1), fx), fy);
}

// Encode: the proxy over fitted processing pixel (x, y). Identity and
// upscaling take the bilinear tap at its centre. A source larger than the fit
// is integrated over the pixel's footprint with area weights, as the layer's
// downscaler does, rather than dropping the texels between the taps.
template<class Read>
DLSSLOP_INLINE Rgb sample_proxy(const Read& read, const Geometry& g, unsigned x, unsigned y)
{
    const float sw = float(g.source_width), sh = float(g.source_height);
    const float fw = float(g.fit_width), fh = float(g.fit_height);
    if (g.source_width <= g.fit_width && g.source_height <= g.fit_height)
        return bilinear(read, (float(x) + 0.5f - float(g.x)) * sw / fw - 0.5f,
                        (float(y) + 0.5f - float(g.y)) * sh / fh - 0.5f,
                        0, 0, g.source_width - 1, g.source_height - 1);
    x -= g.x;
    y -= g.y;
    const float x0 = float(x) * sw / fw, x1 = float(x + 1) * sw / fw;
    const float y0 = float(y) * sh / fh, y1 = float(y + 1) * sh / fh;
    Rgb sum{};
    // Every texel overlaps the footprint; rounding may take the last one
    // past the source's edge.
    for (unsigned j = unsigned(y0); float(j) < y1; ++j) {
        const float wy = (y1 < float(j + 1) ? y1 : float(j + 1)) - (y0 > float(j) ? y0 : float(j));
        for (unsigned i = unsigned(x0); float(i) < x1; ++i) {
            const float w = ((x1 < float(i + 1) ? x1 : float(i + 1)) - (x0 > float(i) ? x0 : float(i))) * wy;
            const Rgb c = read(i < g.source_width ? i : g.source_width - 1, j < g.source_height ? j : g.source_height - 1);
            sum = {sum.r + c.r * w, sum.g + c.g * w, sum.b + c.b * w};
        }
    }
    const float area = (x1 - x0) * (y1 - y0);
    return {sum.r / area, sum.g / area, sum.b / area};
}

// Decode: the network's answer at the centre of source pixel (x, y).
template<class Read>
DLSSLOP_INLINE Rgb sample_answer(const Read& read, const Geometry& g, unsigned x, unsigned y)
{
    return bilinear(read, float(g.x) + (float(x) + 0.5f) * float(g.fit_width) / float(g.source_width) - 0.5f,
                    float(g.y) + (float(y) + 0.5f) * float(g.fit_height) / float(g.source_height) - 0.5f,
                    g.x, g.y, g.x + g.fit_width - 1, g.y + g.fit_height - 1);
}

} // namespace dlsslop
