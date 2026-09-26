// Positive control for tests/check-lds-barriers.py --compiler, which compiles
// it like the production modules; no GPU or HIP SDK headers required.
// unsafe_lds deliberately lacks the release/acquire ordering. Do not use it
// in production. fenced_lds, fenced like the barriers scripts/build-kernels.py
// wraps, must drain LDS before signalling the barrier.

typedef unsigned uint;
#define PROBE_KERNEL extern "C" __attribute__((global)) \
    __attribute__((amdgpu_flat_work_group_size(128,128)))

PROBE_KERNEL void unsafe_lds(uint* out, const uint* input)
{
    __attribute__((shared)) uint scratch[128];
    uint t = __builtin_amdgcn_workitem_id_x();
    scratch[t] = input[t];
    __builtin_amdgcn_s_barrier();
    out[t] = scratch[t ^ 64u];
}

PROBE_KERNEL void fenced_lds(uint* out, const uint* input)
{
    __attribute__((shared)) uint scratch[128];
    uint t = __builtin_amdgcn_workitem_id_x();
    scratch[t] = input[t];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    out[t] = scratch[t ^ 64u];
}
