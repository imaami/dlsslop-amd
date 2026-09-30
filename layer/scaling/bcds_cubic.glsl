// The Keys cubic kernel. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_{bicubic,catmull}.hlsl in
// OptiScaler (GPL-3.0; see PROVENANCE.md).
// The including file defines A; this file defines the kernel for bcds.glsl.

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

// CubicKeys is zero from distance 2.
const float SUPPORT = 2.0;

float Weight(float x)
{
    return CubicKeys(x);
}
