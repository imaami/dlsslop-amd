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

namespace dlsslop {

// Whether the fitted picture is nonempty and inside the padded extent.
inline bool fits(const struct geometry& g)
{
    return g.fit_width && g.fit_height && g.x < g.width && g.y < g.height &&
           g.fit_width <= g.width - g.x && g.fit_height <= g.height - g.y;
}

} // namespace dlsslop
