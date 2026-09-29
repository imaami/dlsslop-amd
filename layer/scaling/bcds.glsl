// The downscalers' shared declarations. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_*.hlsl in OptiScaler
// (GPL-3.0; see PROVENANCE.md).
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

layout(set = 0, binding = 3) uniform sampler LinearClampSampler;

int ClampInt(int v, int lo, int hi)
{
    return min(max(v, lo), hi);
}
