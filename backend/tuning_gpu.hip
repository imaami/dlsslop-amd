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
void dlsslop_tune_rgb(const float* input_rgba, const float* raw_rgb,
                    float* output_rgb, TuningGeometry g, dlsslop::NativeTuning tuning)
{
    const unsigned pixel = __builtin_amdgcn_workgroup_id_x() * 256u +
                           __builtin_amdgcn_workitem_id_x();
    if (pixel >= g.width * g.height)
        return;
    const unsigned x = pixel % g.width, y = pixel / g.width;
    for (unsigned channel = 0; channel < 3; ++channel)
        output_rgb[pixel * 3 + channel] = dlsslop::tune_neural_component(
            input_rgba, raw_rgb, g.width, x, y, g.x, g.y,
            g.x + g.fit_width - 1, g.y + g.fit_height - 1, channel, tuning);
}
