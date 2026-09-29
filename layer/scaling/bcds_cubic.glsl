// The Keys cubic downscalers' shared body. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_{bicubic,catmull}.hlsl in
// OptiScaler (GPL-3.0).
// The including file defines A and SRC_OFFSET.

// Keys cubic weight for distance x in [0,2)
float CubicKeys(float x)
{
    x = abs(x);
    float x2 = x * x;
    float x3 = x2 * x;

    if (x < 1.0)
    {
        return (A + 2.0) * x3 - (A + 3.0) * x2 + 1.0;
    }
    else if (x < 2.0)
    {
        return A * x3 - 5.0 * A * x2 + 8.0 * A * x - 4.0 * A;
    }
    return 0.0;
}

// Compute two bilinear sample positions and their combined weights from 4 cubic taps.
// This is the standard "4 taps via 2 bilinear taps per axis" trick.
void BicubicAxis(float t, out float w01, out float w23, out float o01, out float o23)
{
    // t is fractional part in [0,1)
    float w0 = CubicKeys(1.0 + t);
    float w1 = CubicKeys(t);
    float w2 = CubicKeys(1.0 - t);
    float w3 = CubicKeys(2.0 - t);

    w01 = w0 + w1;
    w23 = w2 + w3;

    // Avoid division by zero; in practice w01/w23 should be >0 for these kernels.
    float invW01 = FOrdNotEqual(w01, 0.0) ? (1.0 / w01) : 0.0;
    float invW23 = FOrdNotEqual(w23, 0.0) ? (1.0 / w23) : 0.0;

    // Offsets relative to the "base" texel index (floor(pos) - 1)
    // These produce the correct mix of the two texels in each bilinear pair.
    o01 = (-1.0) + (w1 * invW01); // between base+0 and base+1
    o23 = (1.0) + (w3 * invW23); // between base+2 and base+3
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    uvec3 id = gl_GlobalInvocationID;
    uint ox = id.x;
    uint oy = id.y;
    if (ox >= uint(_DstWidth) || oy >= uint(_DstHeight))
        return;

    // Map destination pixel center to source pixel space (center-aligned when
    // SRC_OFFSET is 0.5)
    vec2 dst = vec2(float(ox) + 0.5, float(oy) + 0.5);
    vec2 scale = vec2(float(_SrcWidth) / float(_DstWidth),
                      float(_SrcHeight) / float(_DstHeight));

    // Source position in texel space, with texel centers at i+0.5
    vec2 srcPos = dst * scale - SRC_OFFSET;

    vec2 ip = floor(srcPos);
    vec2 f = srcPos - ip;

    // Bicubic uses 4 taps: base = ip - 1
    vec2 base = ip - 1.0;

    float wx01, wx23, ox01, ox23;
    float wy01, wy23, oy01, oy23;
    BicubicAxis(f.x, wx01, wx23, ox01, ox23);
    BicubicAxis(f.y, wy01, wy23, oy01, oy23);

    // Convert texel-space sample positions to UV (normalized)
    vec2 invSrc = 1.0 / vec2(float(_SrcWidth), float(_SrcHeight));

    vec2 uv00 = (base + vec2(ox01, oy01) + 0.5) * invSrc;
    vec2 uv10 = (base + vec2(ox23, oy01) + 0.5) * invSrc;
    vec2 uv01 = (base + vec2(ox01, oy23) + 0.5) * invSrc;
    vec2 uv11 = (base + vec2(ox23, oy23) + 0.5) * invSrc;

    // 4 bilinear samples (each bilinear internally mixes a 2x2 quad)
    vec3 s00 = textureLod(sampler2D(InputTexture, LinearClampSampler), uv00, 0.0).rgb;
    vec3 s10 = textureLod(sampler2D(InputTexture, LinearClampSampler), uv10, 0.0).rgb;
    vec3 s01 = textureLod(sampler2D(InputTexture, LinearClampSampler), uv01, 0.0).rgb;
    vec3 s11 = textureLod(sampler2D(InputTexture, LinearClampSampler), uv11, 0.0).rgb;

    // Combine separably
    vec3 outRgb =
        (s00 * wx01 + s10 * wx23) * wy01 +
        (s01 * wx01 + s11 * wx23) * wy23;

    // Cheap clamp (helps prevent HDR undershoot-looking pinholes)
    vec3 mn = NMin(NMin(s00, s10), NMin(s01, s11));
    vec3 mx = NMax(NMax(s00, s10), NMax(s01, s11));
    outRgb = clamp(outRgb, mn, mx);

    imageStore(OutputTexture, ivec2(uvec2(ox, oy)), vec4(outRgb, 1.0));
}
