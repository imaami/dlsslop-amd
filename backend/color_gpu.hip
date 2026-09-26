// SPDX-License-Identifier: MIT
#include "kernel.h"
#include "color_preserve_math.h"

KERNEL void dlsslop_preserve_color(const float* __restrict__ original_rgba, const float* __restrict__ model_rgb,
                          float* __restrict__ output_rgb, dlsslop::Geometry g, float strength)
{
    const unsigned pixel = thread_index();
    if (pixel >= g.width * g.height) return;
    dlsslop::preserve_color_pixel(original_rgba, model_rgb, output_rgb, g.width,
        pixel % g.width, pixel / g.width, g.x, g.y,
        g.x + g.fit_width - 1, g.y + g.fit_height - 1, strength);
}
