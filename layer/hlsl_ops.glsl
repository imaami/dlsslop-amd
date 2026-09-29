// The SPIR-V operations DXC emits for HLSL's float min, max and !=. GLSL's
// min, max and != would emit FMin, FMax and FUnordNotEqual instead, which
// treat NaN differently.
#extension GL_EXT_spirv_intrinsics : require

spirv_instruction(set = "GLSL.std.450", id = 79) float NMin(float x, float y);
spirv_instruction(set = "GLSL.std.450", id = 79) vec3 NMin(vec3 x, vec3 y);
spirv_instruction(set = "GLSL.std.450", id = 80) float NMax(float x, float y);
spirv_instruction(set = "GLSL.std.450", id = 80) vec3 NMax(vec3 x, vec3 y);
spirv_instruction(id = 182) bool FOrdNotEqual(float x, float y);
