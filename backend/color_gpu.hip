// SPDX-License-Identifier: MIT
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include "color_preserve_math.h"

struct ColorGeometry {
    unsigned source_width, source_height, width, height, valid_width, valid_height;
    unsigned x, y, fit_width, fit_height;
};

extern "C" __attribute__((global))
void dlsslop_preserve_color(const float* original_rgba, const float* model_rgb,
                          float* output_rgb, ColorGeometry g, float strength)
{
    const unsigned pixel = __builtin_amdgcn_workgroup_id_x() * 256u +
                           __builtin_amdgcn_workitem_id_x();
    if (pixel >= g.width * g.height) return;
    dlsslop::preserve_color_pixel(original_rgba, model_rgb, output_rgb, g.width,
        pixel % g.width, pixel / g.width, g.x, g.y,
        g.x + g.fit_width - 1, g.y + g.fit_height - 1, strength);
}
