// SPDX-License-Identifier: MIT
#include "temporal_math.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
struct Plane { dlsslop_temporal::Extent size; std::vector<float> pixels; };
struct Field { dlsslop_temporal::Extent size; std::vector<dlsslop_temporal::Flow> pixels; };
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
// The finest luma level, as dlsslop_temporal_luma builds it from the frame's original input.
std::vector<float> luma(const std::vector<float>& rgba)
{
    std::vector<float> result(rgba.size() / 4);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = rgba[i * 4] * .2126f + rgba[i * 4 + 1] * .7152f + rgba[i * 4 + 2] * .0722f;
    return result;
}
std::vector<Plane> pyramid(Plane original, unsigned quality)
{
    std::vector<Plane> result{std::move(original)};
    for (unsigned i = 1; i < 3 + quality; ++i) {
        const auto& prev = result.back();
        Plane next{{(prev.size.width + 1) / 2, (prev.size.height + 1) / 2}, {}};
        next.pixels.resize(next.size.width * next.size.height);
        for (unsigned y = 0; y < next.size.height; ++y)
            for (unsigned x = 0; x < next.size.width; ++x)
                next.pixels[y * next.size.width + x] =
                    dlsslop_temporal::sample(prev.pixels.data(), prev.size, float(x * 2) + .5f, float(y * 2) + .5f);
        result.push_back(std::move(next));
    }
    return result;
}
Field optical_flow(const Plane& current, const Plane& previous, unsigned quality, unsigned grid)
{
    const auto a = pyramid(current, quality), b = pyramid(previous, quality);
    const unsigned step = 1u << grid;
    Field coarse{};
    for (std::size_t level = a.size(); level-- > 0;) {
        const auto e = a[level].size;
        Field field{{(e.width + step - 1) / step, (e.height + step - 1) / step}, {}};
        field.pixels.resize(field.size.width * field.size.height);
        dlsslop_temporal::Search s{e, field.size, coarse.size, step, quality + 1,
            quality == 2 ? 2u : 1u, unsigned(!coarse.pixels.empty()), unsigned(!level)};
        for (unsigned p = 0; p < field.pixels.size(); ++p)
            field.pixels[p] = dlsslop_temporal::estimate(a[level].pixels.data(), b[level].pixels.data(), coarse.pixels.data(), s, p);
        coarse = std::move(field);
    }
    return coarse;
}
float random(unsigned x, unsigned y)
{
    unsigned v = x * 0x45d9f3b + y * 0x119de1f3;
    v ^= v >> 16; v *= 0x45d9f3b; v ^= v >> 16;
    return float(v & 65535) / 65535;
}
// The flow search as it was before it sampled the current patch once per grid point:
// every candidate resamples both frames. The shipped search must match it bit for bit.
float reference_cost(const float* current, const float* previous, dlsslop_temporal::Extent e,
                     float x, float y, float dx, float dy, unsigned patch)
{
    if (x + dx < 0 || y + dy < 0 || x + dx > float(e.width - 1) || y + dy > float(e.height - 1))
        return 10;
    float cost = 0;
    for (int py = -int(patch); py <= int(patch); ++py)
        for (int px = -int(patch); px <= int(patch); ++px)
            cost += dlsslop_temporal::absolute(dlsslop_temporal::sample(current, e, x + float(px), y + float(py)) -
                                               dlsslop_temporal::sample(previous, e, x + float(px) + dx, y + float(py) + dy));
    const unsigned side = patch * 2 + 1;
    return cost / float(side * side);
}
dlsslop_temporal::Flow reference_estimate(const float* current, const float* previous,
                                          const dlsslop_temporal::Flow* coarse, dlsslop_temporal::Search s, unsigned index)
{
    using dlsslop_temporal::absolute;
    using dlsslop_temporal::clamp;
    const float x = clamp((float(index % s.grid.width) + .5f) * float(s.step) - .5f, 0, float(s.image.width - 1));
    const float y = clamp((float(index / s.grid.width) + .5f) * float(s.step) - .5f, 0, float(s.image.height - 1));
    dlsslop_temporal::Flow start{};
    if (s.has_coarse) {
        start = dlsslop_temporal::sample_flow(coarse, s.coarse_grid, s.step * 2, x, y);
        start.x = float(dlsslop_temporal::round_int(start.x * 2));
        start.y = float(dlsslop_temporal::round_int(start.y * 2));
    }
    dlsslop_temporal::Flow best{0, 0, reference_cost(current, previous, s.image, x, y, 0, 0, s.patch)};
    float best_score = best.error;
    for (int dy = -int(s.radius); dy <= int(s.radius); ++dy) {
        for (int dx = -int(s.radius); dx <= int(s.radius); ++dx) {
            const float vx = start.x + float(dx), vy = start.y + float(dy);
            const float cost = reference_cost(current, previous, s.image, x, y, vx, vy, s.patch);
            const float score = cost + .00001f * (absolute(vx) + absolute(vy));
            if (score < best_score) { best = {vx, vy, cost}; best_score = score; }
        }
    }
    if (s.final_level && s.radius > 1 && best.error > .000001f) {
        const float left = reference_cost(current, previous, s.image, x, y, best.x - 1, best.y, s.patch);
        const float right = reference_cost(current, previous, s.image, x, y, best.x + 1, best.y, s.patch);
        const float top = reference_cost(current, previous, s.image, x, y, best.x, best.y - 1, s.patch);
        const float bottom = reference_cost(current, previous, s.image, x, y, best.x, best.y + 1, s.patch);
        const float hx = left - 2 * best.error + right, hy = top - 2 * best.error + bottom;
        const float dx = hx > .000001f ? clamp(.5f * (left - right) / hx, -.5f, .5f) : 0;
        const float dy = hy > .000001f ? clamp(.5f * (top - bottom) / hy, -.5f, .5f) : 0;
        const float cost = reference_cost(current, previous, s.image, x, y, best.x + dx, best.y + dy, s.patch);
        if (cost < best.error) best = {best.x + dx, best.y + dy, cost};
    }
    return best;
}
// The flow grid holds one vector per step x step pixels, vector i at pixel (i + .5) * step - .5
// as the search places it. The search's coarse start, the warp and the scene-cut kernel all read
// the grid through sample_flow: on a linear field, every pixel position, in the frame or past its
// edges, must read back its own grid coordinate, clamped to the grid.
void flow_grid()
{
    const dlsslop_temporal::Extent g{5, 4};
    std::vector<dlsslop_temporal::Flow> ramp(g.width * g.height);
    for (unsigned j = 0; j < g.height; ++j)
        for (unsigned i = 0; i < g.width; ++i) ramp[j * g.width + i] = {float(i), float(j), float(i + j)};
    unsigned positions = 0;
    for (unsigned step = 1; step <= 8; step *= 2) {
        for (float y = -.5f; y <= float(g.height * step + 1); y += .25f) {
            for (float x = -.5f; x <= float(g.width * step + 1); x += .25f, ++positions) {
                const float gx = std::clamp((x + .5f) / float(step) - .5f, 0.f, float(g.width - 1));
                const float gy = std::clamp((y + .5f) / float(step) - .5f, 0.f, float(g.height - 1));
                const auto f = dlsslop_temporal::sample_flow(ramp.data(), g, step, x, y);
                require(std::fabs(f.x - gx) < 1e-5f && std::fabs(f.y - gy) < 1e-5f &&
                        std::fabs(f.error - gx - gy) < 1e-5f, "a pixel does not read the flow at its grid coordinate");
            }
        }
    }
    std::printf("flow grid: %u pixel positions read their grid coordinate\n", positions);
}
// Random pictures, some translated, some static, at every quality, grid, coarse start and
// level: the search must return exactly what resampling every candidate returns.
void search_equivalence()
{
    std::mt19937 rng(7);
    unsigned long long vectors = 0;
    for (int trial = 0; trial < 20; ++trial) {
        const unsigned w = 4 + rng() % 36, h = 4 + rng() % 24;
        std::vector<float> current(w * h), previous(w * h);
        const int sx = int(rng() % 7) - 3, sy = int(rng() % 7) - 3;
        for (float& v : previous) v = float(rng() % 1000) / 999;
        for (unsigned y = 0; y < h; ++y)
            for (unsigned x = 0; x < w; ++x)
                current[y * w + x] = rng() % 5 ? previous[std::min(h - 1, unsigned(std::max(0, int(y) + sy))) * w +
                                                          std::min(w - 1, unsigned(std::max(0, int(x) + sx)))]
                                               : float(rng() % 1000) / 999;
        if (trial % 10 == 0) current = previous;
        for (unsigned quality = 0; quality < 3; ++quality)
            for (unsigned grid = 0; grid < 4; ++grid)
                for (unsigned has_coarse = 0; has_coarse < 2; ++has_coarse)
                    for (unsigned final_level = 0; final_level < 2; ++final_level) {
                        const unsigned step = 1u << grid;
                        const dlsslop_temporal::Extent e{w, h}, g{(w + step - 1) / step, (h + step - 1) / step};
                        const dlsslop_temporal::Extent cg{(g.width + 1) / 2, (g.height + 1) / 2};
                        std::vector<dlsslop_temporal::Flow> coarse(cg.width * cg.height);
                        for (auto& f : coarse)
                            f = {float(int(rng() % 9) - 4) * .5f, float(int(rng() % 9) - 4) * .5f, 0};
                        const dlsslop_temporal::Search s{e, g, cg, step, quality + 1, quality == 2 ? 2u : 1u,
                                                         has_coarse, final_level};
                        for (unsigned i = 0; i < g.width * g.height; ++i, ++vectors) {
                            const auto a = reference_estimate(current.data(), previous.data(), coarse.data(), s, i);
                            const auto b = dlsslop_temporal::estimate(current.data(), previous.data(), coarse.data(), s, i);
                            require(!std::memcmp(&a, &b, sizeof a), "flow search differs from resampling every candidate");
                        }
                    }
    }
    std::printf("flow search: %llu vectors bit-identical to resampling every candidate\n", vectors);
}
// A pillarboxed and letterboxed picture on a 16x8 raster. The bars' history is a sentinel that no
// fitted pixel may blend in, even when sub-pixel motion points into a bar. The bars' previous luma
// is black, as encode_proxy writes them, so blending it in fails the luma check and falls back.
void letterbox()
{
    constexpr unsigned width = 16, height = 8, left = 3, top = 2, fit_width = 10, fit_height = 4;
    constexpr unsigned right = left + fit_width - 1, bottom = top + fit_height - 1;
    constexpr float sentinel = 1000, fallback_value = .125f;
    std::vector<float> rgba(width * height * 4, .5f), gray(width * height, .5f);
    const std::vector<float> current = luma(rgba);
    std::vector<float> history(width * height * 3, sentinel), fallback(rgba.size(), fallback_value);
    std::vector<float> output(rgba.size());
    std::vector<dlsslop_temporal::Flow> flow(width * height);
    const auto inside = [](unsigned x, unsigned y) { return float(x) + float(y) * width; };
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            if (x < left || x > right || y < top || y > bottom) gray[y * width + x] = 0;
    for (unsigned y = top; y <= bottom; ++y)
        for (unsigned x = left; x <= right; ++x)
            for (unsigned c = 0; c < 3; ++c) history[(y * width + x) * 3 + c] = inside(x, y);
    const dlsslop_temporal::Warp w{{width, height}, {width, height}, height, 1, left, top, fit_width, fit_height};
    struct Case { float dx, dy; unsigned x, y; float expected; const char* message; };
    const Case cases[] = {
        {.75f, 0, right, 3, inside(right, 3), "rightward sub-pixel motion did not return the edge history"},
        {0, .75f, 5, bottom, inside(5, bottom), "downward sub-pixel motion did not return the edge history"},
        {.75f, .75f, right, bottom, inside(right, bottom), "diagonal sub-pixel motion did not return the edge history"},
        {.25f, .5f, 7, 3, inside(7, 3) + .25f + .5f * width, "interior sub-pixel motion was not interpolated"},
        {1, 0, right, 3, fallback_value, "motion from outside the fit did not fall back"},
        {-.75f, 0, left, 3, fallback_value, "motion from the left bar did not fall back"},
        {0, -.75f, 5, top, fallback_value, "motion from the top bar did not fall back"},
        {0, 0, left - 1, 3, fallback_value, "a bar pixel did not fall back"},
    };
    for (const Case& t : cases) {
        std::fill(flow.begin(), flow.end(), dlsslop_temporal::Flow{t.dx, t.dy, 0});
        const unsigned index = t.y * width + t.x;
        dlsslop_temporal::warp(current.data(), gray.data(), history.data(), fallback.data(), flow.data(), output.data(), w, index);
        for (unsigned c = 0; c < 3; ++c) require(output[index * 4 + c] == t.expected, t.message);
        require(output[index * 4 + 3] == 1, "warped history is not opaque");
    }
}
void run()
{
    constexpr unsigned width = 128, height = 96;
    Plane previous{{width, height}, std::vector<float>(width * height)};
    Plane current = previous;
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            previous.pixels[y * width + x] = random(x / 8, y / 8) * .3f + random(x / 2, y / 2) * .5f + random(x, y) * .2f;
    for (unsigned quality = 0; quality < 3; ++quality) {
        auto still = optical_flow(previous, previous, quality, 2);
        for (const auto f : still.pixels) require(f.x == 0 && f.y == 0 && f.error == 0, "static image manufactured motion");
    }
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            current.pixels[y * width + x] = dlsslop_temporal::sample(previous.pixels.data(), previous.size, float(x) - 6, float(y) + 3);
    auto pixels = optical_flow(current, previous, 2, 2);
    unsigned good = 0, tested = 0;
    for (unsigned y = 4; y + 4 < pixels.size.height; ++y) {
        for (unsigned x = 4; x + 4 < pixels.size.width; ++x) {
            const auto f = pixels.pixels[y * pixels.size.width + x];
            good += std::fabs(f.x + 6) < .5f && std::fabs(f.y - 3) < .5f;
            ++tested;
        }
    }
    std::printf("known translation: %u/%u interior vectors within 0.5 pixels\n", good, tested);
    require(good * 10 > tested * 8, "translation direction or magnitude wrong");
    // Warp a known affine RGB image using a known displacement; independently
    // verify bilinear coordinates, sign, fallback and reflected bottom padding.
    std::vector<float> rgba(width * height * 4), history(width * height * 3), fallback(rgba.size(), .125f);
    std::vector<float> output(width * (height + 8) * 4);
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            const unsigned p = y * width + x;
            for (unsigned c = 0; c < 3; ++c) {
                rgba[p * 4 + c] = current.pixels[p];
                history[p * 3 + c] = float(x) / float(width) + float(y) / float(height) + float(c);
            }
            rgba[p * 4 + 3] = 1;
        }
    }
    const std::vector<float> current_luma = luma(rgba);
    std::fill(pixels.pixels.begin(), pixels.pixels.end(), dlsslop_temporal::Flow{-6, 3, 0});
    dlsslop_temporal::Warp w{{width, height}, pixels.size, height + 8, 4, 0, 0, width, height};
    for (unsigned p = 0; p < width * (height + 8); ++p)
        dlsslop_temporal::warp(current_luma.data(), previous.pixels.data(), history.data(), fallback.data(), pixels.pixels.data(), output.data(), w, p);
    const unsigned x = 60, y = 40;
    const float expected = float(x - 6) / float(width) + float(y + 3) / float(height);
    require(std::fabs(output[(y * width + x) * 4] - expected) < .00001f, "history warp coordinate wrong");
    require(output[0] == .125f, "out-of-frame history must use corresponding current-pass input");
    const unsigned mirrored = 2 * height - 2 - (height + 7);
    require(output[((height + 7) * width + x) * 4] == output[(mirrored * width + x) * 4], "history padding is not reflected");
    std::fill(pixels.pixels.begin(), pixels.pixels.end(), dlsslop_temporal::Flow{-6, 3, .5f});
    dlsslop_temporal::warp(current_luma.data(), previous.pixels.data(), history.data(), fallback.data(), pixels.pixels.data(), output.data(), w, y * width + x);
    require(output[(y * width + x) * 4] == .125f, "rejected motion contaminated history");
    // A scene cut rejects even valid motion everywhere: every pixel, padding included,
    // is the current pass input's reflected pixel, opaque.
    std::vector<float> input(rgba.size());
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(i % 1021) / 1021;
    std::fill(pixels.pixels.begin(), pixels.pixels.end(), dlsslop_temporal::Flow{-6, 3, 0});
    for (unsigned p = 0; p < width * (height + 8); ++p) {
        dlsslop_temporal::warp(current_luma.data(), previous.pixels.data(), history.data(), input.data(), pixels.pixels.data(), output.data(), w, p, true);
        const unsigned row = p / width, source = (row < height ? row : 2 * height - 2 - row) * width + p % width;
        for (unsigned c = 0; c < 3; ++c) require(output[p * 4 + c] == input[source * 4 + c], "a scene cut kept warped history");
        require(output[p * 4 + 3] == 1, "scene cut fallback is not opaque");
    }
    letterbox();
    flow_grid();
    search_equivalence();
    std::puts("temporal tests passed");
}
}
int main()
{
    try { run(); return 0; }
    catch (const std::exception& e) { std::fprintf(stderr, "temporal-test: %s\n", e.what()); return 1; }
}
