// The downscalers' shared declarations. Ported to GLSL in 2026 from
// OptiScaler/shaders/output_scaling/precompile/bcds_*.hlsl in OptiScaler
// (GPL-3.0; see PROVENANCE.md).
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_spirv_intrinsics : require

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

// The SPIR-V operations DXC emits for HLSL's min, max and float !=. GLSL's
// min, max and != would emit FMin, FMax and FUnordNotEqual instead, which
// treat NaN differently.
spirv_instruction(set = "GLSL.std.450", id = 79) vec3 NMin(vec3 x, vec3 y);
spirv_instruction(set = "GLSL.std.450", id = 80) vec3 NMax(vec3 x, vec3 y);
spirv_instruction(id = 182) bool FOrdNotEqual(float x, float y);
