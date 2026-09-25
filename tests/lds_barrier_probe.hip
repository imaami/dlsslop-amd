// Minimal compiler reproducer; no GPU or HIP SDK headers required.
// amdclang++ -x hip --cuda-device-only --no-gpu-bundle-output \
//   --offload-arch=gfx1201 -mcode-object-version=5 -nogpuinc -nogpulib \
//   -O3 -std=c++17 -I kernels -c tests/lds_barrier_probe.hip -o probe.hsaco
// llvm-objdump -d probe.hsaco
// unsafe_lds deliberately lacks the release/acquire ordering. Do not use it
// in production. fenced_lds must drain LDS before signalling the barrier.

#include "lds_barrier.h"

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
    dlsslop_lds_barrier();
    out[t] = scratch[t ^ 64u];
}
