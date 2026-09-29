// The windowed-sinc downscalers' shared body. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_{lanczos,kaiser}{2,3}.hlsl
// in OptiScaler (GPL-3.0).
// The including file defines RADIUS, TAP_COUNT and either A_LANCZOS and
// PI_OVER_A, or KAISER, A_KAISER, BETA, INV_A_SQUARED, I0_T2_SCALE and
// I0_C6_SCALED.
//
// The arithmetic is that of the HLSL as DXC compiles it (dxc -spirv
// optimizes by default). DXC turned x / c into x * (1 / c) and merged
// constant factors across operations, which rounds differently from the
// source's order, and drivers fuse multiply-adds by the shape they are given.
// This port therefore writes out DXC's merged forms, each beside the
// upstream expression; the including file derives the constants.

// Sinc with its factor pi as a parameter, for Lanczos' Sinc(x / a), whose
// division DXC merged into that factor.
float Sinc(float x, float pi)
{
    x *= pi;
    if (abs(x) < 1e-5)
        return 1.0;
    return sin(x) / x;
}

float Sinc(float x)
{
    return Sinc(x, 3.1415926535);
}

#ifdef KAISER
// Modified Bessel function I0 approximation (good enough for Kaiser window).
// Based on common Cephes-style polynomial approximations.
//
// Upstream's I0(x), for x = BETA * u only: DXC inlined KaiserWindow's
// I0(beta * t) and rewrote its small-x branch in terms of t. I0(BETA) passes
// u = 1.0; its small-x branch never runs there, as BETA >= 3.75.
float KaiserI0(float x, float u)
{
    float ax = abs(x);
    if (ax < 3.75)
    {
        // Upstream: t = x / 3.75; t2 = t * t; ... + t2 * 0.0045813.
        // With x = BETA * u, DXC made t = u * k where k = BETA * (1 / 3.75),
        // then t2 = (k * k) * (u * u) and t2 * 0.0045813 =
        // (u * u) * (k * k * 0.0045813).
        float u2 = u * u;
        float t2 = I0_T2_SCALE * u2;
        return 1.0
            + t2 * (3.5156229
            + t2 * (3.0899424
            + t2 * (1.2067492
            + t2 * (0.2659732
            + t2 * (0.0360768
            + u2 * I0_C6_SCALED)))));
    }
    else
    {
        float t = 3.75 / ax;
        // Upstream's innermost t * 0.00392377 is (3.75 / ax) * 0.00392377,
        // which DXC folded into 0.0147141377 / ax: 3.75 * 0.00392377
        // rounds to 0.0147141377 (0x3c711391).
        return (exp(ax) / sqrt(ax)) *
            (0.39894228
            + t * (0.01328592
            + t * (0.00225319
            + t * (-0.00157565
            + t * (0.00916281
            + t * (-0.02057706
            + t * (0.02635537
            + t * (-0.01647633
            + 0.014714137651026249 / ax))))))));
    }
}

float KaiserWindow(float x, float a, float beta, float invI0Beta)
{
    float ax = abs(x);
    if (ax >= a)
        return 0.0;

    // t in [0..1]
    // Upstream: r = ax / a; t = sqrt(saturate(1.0 - r * r)). DXC turned
    // ax / a into ax * (1 / a) and (ax * c) * (ax * c) into
    // (c * c) * (ax * ax), where c * c = INV_A_SQUARED for a = A_KAISER.
    float t = sqrt(clamp(1.0 - INV_A_SQUARED * (ax * ax), 0.0, 1.0));

    // w(x) = I0(beta * t) / I0(beta)
    return KaiserI0(beta * t, t) * invI0Beta;
}

float Kaiser(float x, float a, float beta, float invI0Beta)
{
    // h(x) = sinc(x) * w(x)
    return Sinc(x) * KaiserWindow(x, a, beta, invI0Beta);
}
#else
float Lanczos(float x, float a)
{
    float ax = abs(x);
    if (ax >= a)
        return 0.0;
    // Upstream: Sinc(x) * Sinc(x / a). DXC turned x / a into x * (1 / a)
    // and merged that with Sinc's x * 3.1415926535 into x * PI_OVER_A.
    return Sinc(x) * Sinc(x, PI_OVER_A);
}
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    uvec3 id = gl_GlobalInvocationID;
    uint ox = id.x;
    uint oy = id.y;
    if (ox >= uint(_DstWidth) || oy >= uint(_DstHeight))
        return;

    vec2 dst = vec2(float(ox) + 0.5, float(oy) + 0.5);
    vec2 scale = vec2(float(_SrcWidth) / float(_DstWidth),
                      float(_SrcHeight) / float(_DstHeight));

    vec2 srcPos = dst * scale - 0.5;

    vec2 ip = floor(srcPos);
    vec2 f = srcPos - ip;

    // For radius RADIUS, use taps at {1-RADIUS,...,RADIUS} around ip
    ivec2 base = ivec2(ip) - ivec2(RADIUS - 1, RADIUS - 1);

#ifdef KAISER
    // Precompute 1/I0(beta) once per pixel.
    float invI0Beta = 1.0 / KaiserI0(BETA, 1.0);
#define WEIGHT(x) Kaiser(x, A_KAISER, BETA, invI0Beta)
#else
#define WEIGHT(x) Lanczos(x, A_LANCZOS)
#endif

    float wx[TAP_COUNT];
    float wy[TAP_COUNT];
    float sumWx = 0.0;
    float sumWy = 0.0;

    [[unroll]]
    for (int i = 0; i < TAP_COUNT; ++i)
    {
        float dx = float(i) - float(RADIUS - 1) - f.x;
        wx[i] = WEIGHT(dx);
        sumWx += wx[i];

        float dy = float(i) - float(RADIUS - 1) - f.y;
        wy[i] = WEIGHT(dy);
        sumWy += wy[i];
    }

    float invSumWx = FOrdNotEqual(sumWx, 0.0) ? (1.0 / sumWx) : 0.0;
    float invSumWy = FOrdNotEqual(sumWy, 0.0) ? (1.0 / sumWy) : 0.0;

    [[unroll]]
    for (int i = 0; i < TAP_COUNT; ++i)
    {
        wx[i] *= invSumWx;
        wy[i] *= invSumWy;
    }

    vec2 invSrc = 1.0 / vec2(float(_SrcWidth), float(_SrcHeight));

    // -0.0, not upstream's 0.0: DXC unrolled the loop below and dropped the
    // first 0.0 + s * w, and -0.0 + y == y for every y, including -0.0.
    vec3 acc = vec3(-0.0);

    // Min/max clamp (Lanczos rings more than bicubic; helps with any
    // negative-lobe kernel)
    vec3 mn = vec3(1e30);
    vec3 mx = vec3(-1e30);

    [[unroll]]
    for (int j = 0; j < TAP_COUNT; ++j)
    {
        int y = ClampInt(base.y + j, 0, _SrcHeight - 1);
        float wyj = wy[j];

        [[unroll]]
        for (int i = 0; i < TAP_COUNT; ++i)
        {
            int x = ClampInt(base.x + i, 0, _SrcWidth - 1);
            float w = wx[i] * wyj;

            // Sample at exact texel centers via UV
            vec2 uv = (vec2(float(x) + 0.5, float(y) + 0.5)) * invSrc;
            vec3 s = textureLod(sampler2D(InputTexture, LinearClampSampler), uv, 0.0).rgb;

            mn = NMin(mn, s);
            mx = NMax(mx, s);

            acc += s * w;
        }
    }

    vec3 outRgb = clamp(acc, mn, mx);
    imageStore(OutputTexture, ivec2(uvec2(ox, oy)), vec4(outRgb, 1.0));
}
