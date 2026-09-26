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
    unsigned width{}, height{}; // Padded neural processing extent.
    unsigned valid_height{};    // The fixed 16:9 viewport: full width, unpadded height.
    unsigned x{}, y{}, fit_width{}, fit_height{};
};

// Whether the fitted picture is nonempty and inside the padded extent.
inline bool fits(const Geometry& g)
{
    return g.fit_width && g.fit_height && g.x < g.width && g.y < g.height &&
           g.fit_width <= g.width - g.x && g.fit_height <= g.height - g.y;
}

} // namespace dlsslop
