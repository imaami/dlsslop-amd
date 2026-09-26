// SPDX-License-Identifier: MIT
// Native residual controls, not NGX neural conditioning parameters.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include "tuning_math.h"

struct TuningGeometry {
    unsigned source_width, source_height, width, height, valid_width, valid_height;
    unsigned x, y, fit_width, fit_height;
};

extern "C" __attribute__((global))
void dlsslop_tune_rgb(const float* __restrict__ input_rgba, const float* __restrict__ raw_rgb,
                    float* __restrict__ output_rgb, TuningGeometry g, dlsslop::NativeTuning tuning)
{
    const unsigned pixel = __builtin_amdgcn_workgroup_id_x() * 256u +
                           __builtin_amdgcn_workitem_id_x();
    if (pixel >= g.width * g.height)
        return;
    const unsigned x = pixel % g.width, y = pixel / g.width;
    dlsslop::tune_neural_pixel(input_rgba, raw_rgb, g.width, x, y, g.x, g.y,
        g.x + g.fit_width - 1, g.y + g.fit_height - 1, tuning, output_rgb + pixel * 3);
}
