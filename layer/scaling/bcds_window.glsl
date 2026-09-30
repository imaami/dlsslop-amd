// The windowed-sinc kernels. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_{lanczos,kaiser}{2,3}.hlsl
// in OptiScaler (GPL-3.0; see PROVENANCE.md).

float Sinc(float x)
{
    x *= 3.1415926535;
    if (abs(x) < 1e-5)
        return 1.0;
    return sin(x) / x;
}

float Lanczos(float x, float a)
{
    float ax = abs(x);
    if (ax >= a)
        return 0.0;
    return Sinc(x) * Sinc(x / a);
}

// Modified Bessel function I0 approximation (good enough for Kaiser window).
// Based on common Cephes-style polynomial approximations.
float I0(float x)
{
    float ax = abs(x);
    if (ax < 3.75)
    {
        float t = x / 3.75;
        float t2 = t * t;
        return 1.0
            + t2 * (3.5156229
            + t2 * (3.0899424
            + t2 * (1.2067492
            + t2 * (0.2659732
            + t2 * (0.0360768
            + t2 * 0.0045813)))));
    }
    else
    {
        float t = 3.75 / ax;
        return (exp(ax) / sqrt(ax)) *
            (0.39894228
            + t * (0.01328592
            + t * (0.00225319
            + t * (-0.00157565
            + t * (0.00916281
            + t * (-0.02057706
            + t * (0.02635537
            + t * (-0.01647633
            + t * 0.00392377))))))));
    }
}

float KaiserWindow(float x, float a, float beta, float invI0Beta)
{
    float ax = abs(x);
    if (ax >= a)
        return 0.0;

    // t in [0..1]
    float r = ax / a;
    float t = sqrt(clamp(1.0 - r * r, 0.0, 1.0));

    // w(x) = I0(beta * t) / I0(beta)
    return I0(beta * t) * invI0Beta;
}

float Kaiser(float x, float a, float beta, float invI0Beta)
{
    // h(x) = sinc(x) * w(x)
    return Sinc(x) * KaiserWindow(x, a, beta, invI0Beta);
}
