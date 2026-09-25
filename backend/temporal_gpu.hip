// SPDX-License-Identifier: MIT
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include "temporal_math.h"
#define KERNEL extern "C" __attribute__((global))
#define DEV __attribute__((device)) __attribute__((always_inline))
DEV unsigned temporal_index()
{
    return __builtin_amdgcn_workgroup_id_x() * 256u + __builtin_amdgcn_workitem_id_x();
}
KERNEL void dlsslop_temporal_luma(const float* rgba, float* luma, unsigned count)
{
    const unsigned i = temporal_index();
    if (i < count) luma[i] = rgba[i * 4] * .2126f + rgba[i * 4 + 1] * .7152f + rgba[i * 4 + 2] * .0722f;
}
KERNEL void dlsslop_temporal_reduce(const float* input, float* output, dlsslop_temporal::Extent source, dlsslop_temporal::Extent target)
{
    const unsigned i = temporal_index();
    if (i >= target.width * target.height) return;
    const unsigned x = (i % target.width) * 2, y = (i / target.width) * 2;
    const unsigned xx = dlsslop_temporal::min_u(x + 1, source.width - 1), yy = dlsslop_temporal::min_u(y + 1, source.height - 1);
    output[i] = (input[y * source.width + x] + input[y * source.width + xx] +
                 input[yy * source.width + x] + input[yy * source.width + xx]) * .25f;
}
KERNEL void dlsslop_temporal_flow(const float* current, const float* previous, const dlsslop_temporal::Flow* coarse,
                               dlsslop_temporal::Flow* flow, dlsslop_temporal::Search search)
{
    const unsigned i = temporal_index();
    if (i < search.grid.width * search.grid.height) flow[i] = dlsslop_temporal::estimate(current, previous, coarse, search, i);
}
KERNEL void dlsslop_temporal_warp(const float* current, const float* previous_gray, const float* history,
                               const float* fallback, const dlsslop_temporal::Flow* flow, float* output, dlsslop_temporal::Warp geometry)
{
    const unsigned i = temporal_index();
    if (i < geometry.image.width * geometry.padded_height)
        dlsslop_temporal::warp(current, previous_gray, history, fallback, flow, output, geometry, i);
}

// A small readback detects cuts with broadly failed matches. Local disocclusions
// are still rejected independently in the warp, at every processing pixel.
KERNEL void dlsslop_temporal_cut(const float* current, const float* previous,
                              const dlsslop_temporal::Flow* flow, float* errors, dlsslop_temporal::Warp w)
{
    const unsigned i = temporal_index();
    if (i >= 64) return;
    const float x = float(w.x) + (float(i % 8) + .5f) * float(w.fit_width) / 8 - .5f;
    const float y = float(w.y) + (float(i / 8) + .5f) * float(w.fit_height) / 8 - .5f;
    dlsslop_temporal::Flow f = dlsslop_temporal::sample_flow(flow, w.grid,
        (x + .5f) / float(w.step) - .5f, (y + .5f) / float(w.step) - .5f);
    if (w.units != 1) {
        const float denominator = w.units == 0 ? 2.0f : 1.0f;
        f.x *= float(w.image.width) / denominator;
        f.y *= float(w.image.height) / denominator;
    }
    errors[i] = dlsslop_temporal::patch_cost(current, previous, w.image, x, y, f.x, f.y, 1);
}
