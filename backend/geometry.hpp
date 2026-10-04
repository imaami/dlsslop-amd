// SPDX-License-Identifier: MIT
// Dependency-free: shared by the host and the SDKless HIP modules, whose kernels
// take struct geometry (kernel_args.h) by value.
#pragma once

#include "kernel_args.h"

#if defined(__HIP_DEVICE_COMPILE__)
#define DLSSLOP_INLINE __attribute__((device)) __attribute__((always_inline)) inline
#else
#define DLSSLOP_INLINE inline
#endif
