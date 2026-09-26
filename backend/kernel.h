// SPDX-License-Identifier: MIT
// Prologue of the SDKless HIP modules. Include it first: the fp pragmas then
// also cover the math headers a module includes after it.
#pragma once
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include "geometry.h"

#define DEV DLSSLOP_INLINE
#define KERNEL extern "C" __attribute__((global))

DEV unsigned thread_index()
{
    return __builtin_amdgcn_workgroup_id_x() * 256u + __builtin_amdgcn_workitem_id_x();
}
