// SPDX-License-Identifier: MIT
// Dependency-free: shared by the host and the SDKless HIP modules, whose kernels
// take Geometry by value.
#pragma once

#if defined(__HIP_DEVICE_COMPILE__)
#define DLSSLOP_INLINE __attribute__((device)) __attribute__((always_inline)) inline
#else
#define DLSSLOP_INLINE inline
#endif

namespace dlsslop {

struct Geometry {
    unsigned source_width{}, source_height{};
    unsigned width{}, height{};             // Padded neural processing extent.
    unsigned valid_width{}, valid_height{}; // Unpadded, fixed 16:9 viewport.
    unsigned x{}, y{}, fit_width{}, fit_height{};
};

} // namespace dlsslop
