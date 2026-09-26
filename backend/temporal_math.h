// SPDX-License-Identifier: MIT
#pragma once

// The same scalar math is compiled into the HIP module and the CPU contract test.
// There are no shared-memory reductions or cross-wave synchronization assumptions.
#if defined(__HIP_DEVICE_COMPILE__)
#define DLSSLOP_TEMPORAL_INLINE __attribute__((device)) __attribute__((always_inline)) inline
#else
#define DLSSLOP_TEMPORAL_INLINE inline
#endif

namespace dlsslop_temporal {
struct Flow { float x, y, error; };
struct Extent { unsigned width, height; };
struct Search {
    Extent image, grid, coarse_grid;
    unsigned step, radius, patch, units, has_coarse, final_level;
};
struct Warp {
    Extent image, grid;
    unsigned padded_height, step, units, x, y, fit_width, fit_height;
};
DLSSLOP_TEMPORAL_INLINE float absolute(float v) { return v < 0 ? -v : v; }
DLSSLOP_TEMPORAL_INLINE float clamp(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
DLSSLOP_TEMPORAL_INLINE unsigned min_u(unsigned a, unsigned b) { return a < b ? a : b; }
DLSSLOP_TEMPORAL_INLINE int round_int(float v) { return int(v + (v >= 0 ? .5f : -.5f)); }
DLSSLOP_TEMPORAL_INLINE float sample(const float* image, Extent e, float x, float y)
{
    x = clamp(x, 0, float(e.width - 1));
    y = clamp(y, 0, float(e.height - 1));
    const unsigned ix = unsigned(x), iy = unsigned(y);
    const unsigned hx = min_u(ix + 1, e.width - 1), hy = min_u(iy + 1, e.height - 1);
    const float fx = x - float(ix), fy = y - float(iy);
    const float a = image[iy * e.width + ix], b = image[iy * e.width + hx];
    const float c = image[hy * e.width + ix], d = image[hy * e.width + hx];
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
}
DLSSLOP_TEMPORAL_INLINE float patch_cost(const float* current, const float* previous,
                                      Extent e, float x, float y, float dx, float dy, unsigned patch)
{
    if (x + dx < 0 || y + dy < 0 || x + dx > float(e.width - 1) || y + dy > float(e.height - 1))
        return 10;
    float cost = 0;
    for (int py = -int(patch); py <= int(patch); ++py)
        for (int px = -int(patch); px <= int(patch); ++px)
            cost += absolute(sample(current, e, x + float(px), y + float(py)) -
                             sample(previous, e, x + float(px) + dx, y + float(py) + dy));
    const unsigned side = patch * 2 + 1;
    return cost / float(side * side);
}
DLSSLOP_TEMPORAL_INLINE Flow sample_flow(const Flow* image, Extent e, float x, float y)
{
    x = clamp(x, 0, float(e.width - 1)); y = clamp(y, 0, float(e.height - 1));
    const unsigned ix = unsigned(x), iy = unsigned(y);
    const unsigned hx = min_u(ix + 1, e.width - 1), hy = min_u(iy + 1, e.height - 1);
    const float fx = x - float(ix), fy = y - float(iy);
    const Flow a = image[iy * e.width + ix], b = image[iy * e.width + hx];
    const Flow c = image[hy * e.width + ix], d = image[hy * e.width + hx];
    return {(a.x + (b.x - a.x) * fx) * (1 - fy) + (c.x + (d.x - c.x) * fx) * fy,
            (a.y + (b.y - a.y) * fx) * (1 - fy) + (c.y + (d.y - c.y) * fx) * fy,
            (a.error + (b.error - a.error) * fx) * (1 - fy) + (c.error + (d.error - c.error) * fx) * fy};
}
DLSSLOP_TEMPORAL_INLINE Flow estimate(const float* current, const float* previous,
                                   const Flow* coarse, Search s, unsigned index)
{
    const float x = clamp((float(index % s.grid.width) + .5f) * float(s.step) - .5f,
                          0, float(s.image.width - 1));
    const float y = clamp((float(index / s.grid.width) + .5f) * float(s.step) - .5f,
                          0, float(s.image.height - 1));
    Flow start{};
    if (s.has_coarse) {
        start = sample_flow(coarse, s.coarse_grid, (x + .5f) / float(s.step * 2) - .5f,
                            (y + .5f) / float(s.step * 2) - .5f);
        start.x = float(round_int(start.x * 2)); start.y = float(round_int(start.y * 2));
    }
    // Always retain zero displacement as a candidate, including at object edges.
    Flow best{0, 0, patch_cost(current, previous, s.image, x, y, 0, 0, s.patch)};
    float best_score = best.error;
    for (int dy = -int(s.radius); dy <= int(s.radius); ++dy) {
        for (int dx = -int(s.radius); dx <= int(s.radius); ++dx) {
            const float vx = start.x + float(dx), vy = start.y + float(dy);
            const float cost = patch_cost(current, previous, s.image, x, y, vx, vy, s.patch);
            // Stable tie-breaking in textureless areas; do not manufacture motion.
            const float score = cost + .00001f * (absolute(vx) + absolute(vy));
            if (score < best_score) { best = {vx, vy, cost}; best_score = score; }
        }
    }
    // Parabolic refinement is restricted to the finest level and accepted only
    // if its measured cost improves. This avoids subpixel drift on static input.
    if (s.final_level && s.radius > 1 && best.error > .000001f) {
        const float left = patch_cost(current, previous, s.image, x, y, best.x - 1, best.y, s.patch);
        const float right = patch_cost(current, previous, s.image, x, y, best.x + 1, best.y, s.patch);
        const float top = patch_cost(current, previous, s.image, x, y, best.x, best.y - 1, s.patch);
        const float bottom = patch_cost(current, previous, s.image, x, y, best.x, best.y + 1, s.patch);
        const float hx = left - 2 * best.error + right, hy = top - 2 * best.error + bottom;
        const float dx = hx > .000001f ? clamp(.5f * (left - right) / hx, -.5f, .5f) : 0;
        const float dy = hy > .000001f ? clamp(.5f * (top - bottom) / hy, -.5f, .5f) : 0;
        const float cost = patch_cost(current, previous, s.image, x, y, best.x + dx, best.y + dy, s.patch);
        if (cost < best.error) best = {best.x + dx, best.y + dy, cost};
    }
    if (s.final_level && s.units != 1) {
        const float numerator = s.units == 0 ? 2.0f : 1.0f;
        best.x *= numerator / float(s.image.width); best.y *= numerator / float(s.image.height);
    }
    return best;
}
// current_luma is the finest pyramid level, the luma of the frame's original input, which feedback
// in later passes does not change; fallback is the current pass's input. A scene cut rejects every
// pixel, which gives the network exactly the input of its no-history path.
DLSSLOP_TEMPORAL_INLINE void warp(const float* __restrict__ current_luma, const float* __restrict__ previous_gray,
                               const float* __restrict__ history, const float* __restrict__ fallback,
                               const Flow* __restrict__ flow, float* __restrict__ output, Warp w, unsigned index,
                               bool cut = false)
{
    const unsigned x = index % w.image.width, padded_y = index / w.image.width;
    const unsigned y = padded_y < w.image.height ? padded_y : 2 * w.image.height - 2 - padded_y;
    const unsigned pixel = y * w.image.width + x;
    Flow f = sample_flow(flow, w.grid, (float(x) + .5f) / float(w.step) - .5f,
                         (float(y) + .5f) / float(w.step) - .5f);
    if (w.units != 1) {
        const float denominator = w.units == 0 ? 2.0f : 1.0f;
        f.x *= float(w.image.width) / denominator; f.y *= float(w.image.height) / denominator;
    }
    const float px = float(x) + f.x, py = float(y) + f.y;
    // Both samples weight only the fitted picture; the bars' history is the network's answer for
    // black. Validity below already requires px >= w.x and py >= w.y.
    const float right = float(w.x + w.fit_width - 1), bottom = float(w.y + w.fit_height - 1);
    const float sx = px < right ? px : right, sy = py < bottom ? py : bottom;
    const float error = absolute(current_luma[pixel] - sample(previous_gray, w.image, sx, sy));
    const bool valid = !cut && x >= w.x && y >= w.y && x < w.x + w.fit_width && y < w.y + w.fit_height &&
                       px >= float(w.x) && py >= float(w.y) && px < float(w.x + w.fit_width) &&
                       py < float(w.y + w.fit_height) && f.error < .075f && error < .1f;
    float o[4] = {0, 0, 0, 1};
    if (valid) {
        const float cx = clamp(sx, 0, float(w.image.width - 1)), cy = clamp(sy, 0, float(w.image.height - 1));
        const unsigned ix = unsigned(cx), iy = unsigned(cy);
        const unsigned hx = min_u(ix + 1, w.image.width - 1), hy = min_u(iy + 1, w.image.height - 1);
        const float fx = cx - float(ix), fy = cy - float(iy);
        const unsigned top = iy * w.image.width, low = hy * w.image.width;
        const float *a = history + (top + ix) * 3, *b = history + (top + hx) * 3;
        const float *c = history + (low + ix) * 3, *d = history + (low + hx) * 3;
        for (unsigned channel = 0; channel < 3; ++channel)
            o[channel] = (a[channel] + (b[channel] - a[channel]) * fx) * (1 - fy) +
                         (c[channel] + (d[channel] - c[channel]) * fx) * fy;
    } else {
        for (unsigned channel = 0; channel < 3; ++channel) o[channel] = fallback[pixel * 4 + channel];
    }
    for (unsigned channel = 0; channel < 4; ++channel) output[index * 4 + channel] = o[channel];
}
} // namespace dlsslop_temporal
#undef DLSSLOP_TEMPORAL_INLINE
