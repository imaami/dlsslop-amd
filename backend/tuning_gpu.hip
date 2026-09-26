// SPDX-License-Identifier: MIT
// Native residual controls, not NGX neural conditioning parameters.
#include "kernel.h"
#include "tuning_math.h"

KERNEL void dlsslop_tune_rgb(const float* __restrict__ input_rgba, const float* __restrict__ raw_rgb,
                    float* __restrict__ output_rgb, dlsslop::Geometry g, dlsslop::NativeTuning tuning)
{
    const unsigned pixel = thread_index();
    if (pixel >= g.width * g.height)
        return;
    const unsigned x = pixel % g.width, y = pixel / g.width;
    dlsslop::tune_neural_pixel(input_rgba, raw_rgb, g.width, x, y, g.x, g.y,
        g.x + g.fit_width - 1, g.y + g.fit_height - 1, tuning, output_rgb + pixel * 3);
}
