// SPDX-License-Identifier: MIT
#include "temporal_math.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
struct Plane { dlsslop_temporal::Extent size; std::vector<float> pixels; };
struct Field { dlsslop_temporal::Extent size; std::vector<dlsslop_temporal::Flow> pixels; };
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
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
Field optical_flow(const Plane& current, const Plane& previous, unsigned quality, unsigned grid, unsigned units)
{
    const auto a = pyramid(current, quality), b = pyramid(previous, quality);
    const unsigned step = 1u << grid;
    Field coarse{};
    for (std::size_t level = a.size(); level-- > 0;) {
        const auto e = a[level].size;
        Field field{{(e.width + step - 1) / step, (e.height + step - 1) / step}, {}};
        field.pixels.resize(field.size.width * field.size.height);
        dlsslop_temporal::Search s{e, field.size, coarse.size, step, quality + 1,
            quality == 2 ? 2u : 1u, units, unsigned(!coarse.pixels.empty()), unsigned(!level)};
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
void run()
{
    constexpr unsigned width = 128, height = 96;
    Plane previous{{width, height}, std::vector<float>(width * height)};
    Plane current = previous;
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            previous.pixels[y * width + x] = random(x / 8, y / 8) * .3f + random(x / 2, y / 2) * .5f + random(x, y) * .2f;
    for (unsigned quality = 0; quality < 3; ++quality) {
        auto still = optical_flow(previous, previous, quality, 2, 1);
        for (const auto f : still.pixels) require(f.x == 0 && f.y == 0 && f.error == 0, "static image manufactured motion");
    }
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            current.pixels[y * width + x] = dlsslop_temporal::sample(previous.pixels.data(), previous.size, float(x) - 6, float(y) + 3);
    auto pixels = optical_flow(current, previous, 2, 2, 1);
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
    for (unsigned units : {0u, 2u}) {
        auto converted = optical_flow(current, previous, 2, 2, units);
        const float denominator = units == 0 ? 2.0f : 1.0f;
        for (std::size_t i = 0; i < pixels.pixels.size(); ++i) {
            require(std::fabs(converted.pixels[i].x * float(width) / denominator - pixels.pixels[i].x) < .00001f,
                    "motion X units changed physical displacement");
            require(std::fabs(converted.pixels[i].y * float(height) / denominator - pixels.pixels[i].y) < .00001f,
                    "motion Y units changed physical displacement");
        }
    }
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
    std::fill(pixels.pixels.begin(), pixels.pixels.end(), dlsslop_temporal::Flow{-6, 3, 0});
    dlsslop_temporal::Warp w{{width, height}, pixels.size, height + 8, 4, 1, 0, 0, width, height};
    for (unsigned p = 0; p < width * (height + 8); ++p)
        dlsslop_temporal::warp(rgba.data(), previous.pixels.data(), history.data(), fallback.data(), pixels.pixels.data(), output.data(), w, p);
    const unsigned x = 60, y = 40;
    const float expected = float(x - 6) / float(width) + float(y + 3) / float(height);
    require(std::fabs(output[(y * width + x) * 4] - expected) < .00001f, "history warp coordinate wrong");
    require(output[0] == .125f, "out-of-frame history must use corresponding current-pass input");
    const unsigned mirrored = 2 * height - 2 - (height + 7);
    require(output[((height + 7) * width + x) * 4] == output[(mirrored * width + x) * 4], "history padding is not reflected");
    std::fill(pixels.pixels.begin(), pixels.pixels.end(), dlsslop_temporal::Flow{-6, 3, .5f});
    dlsslop_temporal::warp(rgba.data(), previous.pixels.data(), history.data(), fallback.data(), pixels.pixels.data(), output.data(), w, y * width + x);
    require(output[(y * width + x) * 4] == .125f, "rejected motion contaminated history");
    std::puts("temporal tests passed");
}
}
int main()
{
    try { run(); return 0; }
    catch (const std::exception& e) { std::fprintf(stderr, "temporal-test: %s\n", e.what()); return 1; }
}
