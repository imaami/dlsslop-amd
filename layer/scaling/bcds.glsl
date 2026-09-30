// The downscalers' shared body. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_*.hlsl in OptiScaler
// (GPL-3.0; see PROVENANCE.md), with upstream's resampling replaced.
// The including file defines the kernel as Weight(x), zero for
// |x| >= SUPPORT. It may also define TRUNCATE, to weigh only the source texels
// inside the image where the others repeat the edge texel, and NO_CLAMP, to
// leave out the anti-ringing clamp. Magic defines both, as upstream's magic
// filter truncated and had no clamp.
//
// Upstream's windowed filters spaced their taps one source texel apart at any
// ratio, so they cut off at the source's Nyquist rate instead of the
// destination's and aliased. Its cubic filters stood in two bilinear samples
// for the four taps per axis, which cannot carry the kernel's negative lobes,
// and placed them a texel off centre. Every filter here resamples alike: the
// kernel is stretched by the ratio and weighs every source texel in its
// support, read with texelFetch, as a linear sampler's coordinate rounding can
// blend in a sliver of the neighbour.
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_samplerless_texture_functions : require

#include "../hlsl_ops.glsl"

layout(set = 0, binding = 0) uniform Params
{
    int _SrcWidth;
    int _SrcHeight;
    int _DstWidth;
    int _DstHeight;
};

layout(set = 0, binding = 1) uniform texture2D InputTexture;

// Formatless: the layer enables shaderStorageImageWriteWithoutFormat.
layout(set = 0, binding = 2) uniform writeonly image2D OutputTexture;

int ClampInt(int v, int lo, int hi)
{
    return min(max(v, lo), hi);
}

// With a ratio of s source texels per destination texel on an axis, the
// kernel is stretched by f = max(s, 1): it then cuts off at the destination's
// Nyquist rate, and at s <= 1 it interpolates at the source's. f stops at F_MAX
// so that the support, SUPPORT * f texels to each side of an output texel's
// centre, holds at most MAX_TAPS texels; above that ratio the filter aliases
// more but stays symmetric. The layer clamps its working scale to 2, so only
// frames under 32 texels on a side, where its 64-texel model floor sets the
// ratio, go above F_MAX = 2. Upstream magic's source tile was exact up to about
// s = 3.1, but a larger F_MAX for it would lengthen its loops at every ratio.
const int F_MAX = 2;
const int MAX_TAPS = int(2.0 * SUPPORT) * F_MAX;

// The source rows under 8 output rows at ratio F_MAX.
const int TILE_ROWS = 7 * F_MAX + MAX_TAPS;

// Weights, not yet normalised, and first tap and tap count of the group's 8
// columns (vectors 0-7) and 8 rows (8-15). Each is computed once here rather
// than by all 8 threads that use it.
shared float lds_W[16][MAX_TAPS];
shared ivec2 lds_Taps[16];

// The source rows filtered horizontally for the group's 8 columns: the
// weighted sums (planes 0-2) and, for the clamp, the minima (3-5) and maxima
// (6-8) of the texels used, per channel.
#ifdef NO_CLAMP
const int PLANES = 3;
#else
const int PLANES = 9;
#endif
shared float lds_H[PLANES][TILE_ROWS * 8];

vec3 LoadH(int plane, int h)
{
    return vec3(lds_H[plane][h], lds_H[plane + 1][h], lds_H[plane + 2][h]);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    ivec2 tid = ivec2(gl_LocalInvocationID.xy);
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outBase = ivec2(gl_WorkGroupID.xy) * 8;
    ivec2 src = ivec2(_SrcWidth, _SrcHeight);
    ivec2 dst = ivec2(_DstWidth, _DstHeight);

    // Four threads per vector.
    {
        int lane = tid.y * 8 + tid.x;
        int v = lane >> 2;
        int axis = v >> 3;
        uint o = uint(outBase[axis] + (v & 7));
        uint s = uint(src[axis]);
        uint t = uint(dst[axis]);

        // Output texel o's centre lies at c = (o + 0.5) * s / t - 0.5, where
        // source texel i has its centre at i, and the support reaches r =
        // SUPPORT * f to either side. In units of 1 / (2 t) both are integers,
        // which makes the tap range exact, and each tap's kernel argument
        // (i - c) / f is exact up to the division's error of a few ulp; a
        // float c would be off by up to 2^-24 c, 2.4e-4 texels at the far edge
        // of a 4K source. cb is c biased by MAX_TAPS texels, which keeps the
        // dividends positive so that division floors.
        uint d = 2u * t;
        uint ft = clamp(s, t, uint(F_MAX) * t); // f * t
        uint rn = uint(2.0 * SUPPORT) * ft;     // r * 2 t
        uint cb = (2u * o + 1u) * s - t + uint(MAX_TAPS) * d;

        // The taps are the source texels strictly inside (c - r, c + r).
        int first = int((cb - rn) / d) + 1 - MAX_TAPS;
        int count = int((cb + rn - 1u) / d) + 1 - MAX_TAPS - first;
#ifdef TRUNCATE
        // Only those inside the image. The texel nearest to c is one of them,
        // so at least one tap remains.
        int end = first + count;
        first = max(first, 0);
        count = min(end, int(s)) - first;
#endif
        if ((lane & 3) == 0)
            lds_Taps[v] = ivec2(first, count);

        // Taps past the count repeat the last tap, which gives every loop
        // below a fixed length, with 2^-40 of its weight: an infinite texel
        // there keeps its sign, where a weight of 0 would turn it into NaN
        // (0 * Inf), and a finite one changes the sum by at most 2^-40 of its
        // weighted value, which rounds away unless the sum has cancelled to
        // below 2^-15 of that.
        [[unroll]]
        for (int i = lane & 3; i < MAX_TAPS; i += 4)
        {
            // (tap - c) * 2 t; the wrapped difference is the signed one.
            int tap = first + min(i, count - 1);
            int xn = int(uint(tap + MAX_TAPS) * d - cb);
            float w = Weight(float(xn) / float(2u * ft));
            lds_W[v][i] = i < count ? w : ldexp(w, -40);
        }
    }
    barrier();

    ivec2 tapsX = lds_Taps[tid.x];
    ivec2 tapsY = lds_Taps[8 + tid.y];
    float wx[MAX_TAPS];
    float wy[MAX_TAPS];
    float sumWx = 0.0;
    float sumWy = 0.0;
    [[unroll]]
    for (int i = 0; i < MAX_TAPS; ++i)
    {
        wx[i] = lds_W[tid.x][i];
        wy[i] = lds_W[8 + tid.y][i];
        sumWx += wx[i];
        sumWy += wy[i];
    }
    float invSumWx = 1.0 / sumWx;
    float invSumWy = 1.0 / sumWy;
    [[unroll]]
    for (int i = 0; i < MAX_TAPS; ++i)
    {
        wx[i] *= invSumWx;
        wy[i] *= invSumWy;
    }

    // The source rows under the group's output rows fit TILE_ROWS up to ratio
    // F_MAX; above it, the group filters one output row at a time.
    int rowsOut = min(8, _DstHeight - outBase.y);
    ivec2 lastY = lds_Taps[8 + rowsOut - 1];
    int step = lastY.x + lastY.y - lds_Taps[8].x <= TILE_ROWS ? rowsOut : 1;
    for (int b = 0; b < rowsOut; b += step)
    {
        int top = lds_Taps[8 + b].x;
        ivec2 endY = lds_Taps[8 + b + step - 1];
        int rows = endY.x + endY.y - top;

        for (int row = tid.y; row < rows; row += 8)
        {
            int y = ClampInt(top + row, 0, _SrcHeight - 1);
            vec3 s[MAX_TAPS];
            [[unroll]]
            for (int i = 0; i < MAX_TAPS; ++i)
            {
                int x = ClampInt(tapsX.x + min(i, tapsX.y - 1), 0, _SrcWidth - 1);
                s[i] = texelFetch(InputTexture, ivec2(x, y), 0).rgb;
            }

            vec3 acc = s[0] * wx[0];
            [[unroll]]
            for (int i = 1; i < MAX_TAPS; ++i)
                acc += s[i] * wx[i];

            int h = row * 8 + tid.x;
            lds_H[0][h] = acc.r;
            lds_H[1][h] = acc.g;
            lds_H[2][h] = acc.b;
#ifndef NO_CLAMP
            vec3 mn = s[0];
            vec3 mx = s[0];
            [[unroll]]
            for (int i = 1; i < MAX_TAPS; ++i)
            {
                mn = NMin(mn, s[i]);
                mx = NMax(mx, s[i]);
            }
            lds_H[3][h] = mn.r;
            lds_H[4][h] = mn.g;
            lds_H[5][h] = mn.b;
            lds_H[6][h] = mx.r;
            lds_H[7][h] = mx.g;
            lds_H[8][h] = mx.b;
#endif
        }
        barrier();

        if (tid.y >= b && tid.y < b + step && id.x < _DstWidth)
        {
            int h0 = (tapsY.x - top) * 8 + tid.x;
            vec3 acc = LoadH(0, h0) * wy[0];
            [[unroll]]
            for (int j = 1; j < MAX_TAPS; ++j)
                acc += LoadH(0, h0 + 8 * min(j, tapsY.y - 1)) * wy[j];

#ifndef NO_CLAMP
            // Min/max clamp (Lanczos rings more than bicubic; helps with any
            // negative-lobe kernel).
            vec3 mn = LoadH(3, h0);
            vec3 mx = LoadH(6, h0);
            [[unroll]]
            for (int j = 1; j < MAX_TAPS; ++j)
            {
                int h = h0 + 8 * min(j, tapsY.y - 1);
                mn = NMin(mn, LoadH(3, h));
                mx = NMax(mx, LoadH(6, h));
            }
            acc = clamp(acc, mn, mx);
#endif

            imageStore(OutputTexture, id, vec4(acc, 1.0));
        }
        barrier();
    }
}
