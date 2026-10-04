// SPDX-License-Identifier: MIT
#include "../backend/tuning.hpp"
#include "golden.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <vector>

namespace {
void require(bool condition, const char* message)
{
    if (condition) return;
    std::fprintf(stderr, "native tuning test: %s\n", message);
    std::exit(1);
}

void tune(const std::vector<float>& input, const std::vector<float>& model, const dlsslop::Geometry& g,
          std::vector<float>& output, const dlsslop::NativeTuning& tuning)
{
    require(bool(dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, tuning)), "valid tuning refused");
}

void close(float actual, float expected, const char* message)
{
    require(std::fabs(actual - expected) < 1.0e-6f, message);
}

// The control self-test's states (control_selftest.hpp), the default one first.
const dlsslop::NativeTuning kStates[] = {{}, {0, 1, 1, 0}, {1.75f, .25f, 2.5f, .375f}, {1, 0, 1, 0}, {1, 1, 0, 1}};

// How many of the reference's outputs, one per state, moved from their goldens
// (golden.hpp).
unsigned moved_goldens(const char* fixture, const dlsslop::Geometry& g, const std::vector<float>& input,
                       const std::vector<float>& model,
                       const std::uint64_t (&goldens)[sizeof kStates / sizeof *kStates])
{
    unsigned moved = 0;
    char name[64];
    std::vector<float> output;
    for (std::size_t i = 0; i < sizeof kStates / sizeof *kStates; ++i) {
        tune(input, model, g, output, kStates[i]);
        std::snprintf(name, sizeof name, "tune_neural_rgb %s state %zu", fixture, i);
        moved += !golden::check(name, output.data(), output.size() * sizeof(float), goldens[i]);
    }
    return moved;
}

// The reference's output over the control self-test's fixture and over a
// random one, bit for bit.
void goldens()
{
    const dlsslop::Geometry self_test{7, 5, 7, 5, 5, 0, 0, 7, 5};
    std::vector<float> input(35 * 4), model(35 * 3);
    for (unsigned p = 0; p < 35; ++p) {
        for (unsigned c = 0; c < 3; ++c) {
            input[p * 4 + c] = float((p * 3 + c * 7) % 31) / 32.0f;
            model[p * 3 + c] = input[p * 4 + c] + float(int((p + c) % 7) - 3) / 64.0f;
        }
        input[p * 4 + 3] = 1;
    }
    unsigned moved = moved_goldens("self-test", self_test, input, model, {
        0xb72c2574ee5214f8u, 0x5b4b322ec30cbd50u, 0x5647251848209bf3u, 0xf65855719c16435fu, 0x6e618c8b4da1a9dbu});
    // 640x480 at the 720 tier: pillarboxed, and padded below.
    const dlsslop::Geometry pillarboxed{640, 480, 1280, 768, 720, 160, 0, 960, 720};
    const std::size_t pixels = std::size_t(pillarboxed.width) * pillarboxed.height;
    input.resize(pixels * 4);
    model.resize(pixels * 3);
    for (std::size_t p = 0; p < pixels; ++p) {
        for (unsigned c = 0; c < 3; ++c) {
            input[p * 4 + c] = golden::unit(1, p * 3 + c);
            model[p * 3 + c] = input[p * 4 + c] + (golden::unit(2, p * 3 + c) - 0.5f) * 0.5f;
        }
        input[p * 4 + 3] = 1;
    }
    moved += moved_goldens("640x480 720", pillarboxed, input, model, {
        0x1960f3736b489faeu, 0x024fa28c5f34afacu, 0x8215489abc32a561u, 0x5bc95d036f4a2798u, 0xa1d5a56f13f5c41cu});
    require(!moved, "a golden moved");
}
}

int main()
{
    dlsslop::Geometry g{5, 5, 5, 5, 5, 0, 0, 5, 5};
    std::vector<float> input(25 * 4, 0.25f), model(25 * 3, 0.5f), output;
    for (unsigned p = 0; p < 25; ++p) input[p * 4 + 3] = 1.0f;
    dlsslop::NativeTuning t;
    // A deliberately non-half sample exposes accidental default rounding.
    model[12 * 3] = 0.53123456f;
    tune(input, model, g, output, t);
    require(!std::memcmp(model.data(), output.data(), model.size() * sizeof(float)),
            "default settings must preserve every model bit");

    t.intensity = 0;
    tune(input, model, g, output, t);
    require(std::all_of(output.begin(), output.end(), [](float v) { return v == 0.25f; }),
            "zero intensity must return the input");

    model.assign(model.size(), 0.5f);
    t = {2, 1, 1, 0};
    tune(input, model, g, output, t);
    close(output[12 * 3], 0.75f, "intensity scales residual");
    t = {1, 0, 1, 0};
    tune(input, model, g, output, t);
    for (float value : output) close(value, 0.25f, "zero tone removes a constant edit");

    // A single impulse separates the low-pass (centre weight 1/4) from
    // the structural component (remaining 3/4), independently of input.
    model.assign(model.size(), 0.25f);
    model[12 * 3] += 0.25f;
    t = {1, 0, 1, 0};
    tune(input, model, g, output, t);
    close(output[12 * 3], 0.4375f, "structure retains three quarters of centre impulse");
    close(output[11 * 3], 0.21875f, "structure subtracts low-pass neighbour");
    t = {1, 1, 0, 0};
    tune(input, model, g, output, t);
    close(output[12 * 3], 0.3125f, "tone retains one quarter of centre impulse");
    close(output[11 * 3], 0.28125f, "tone spreads one eighth to axial neighbour");
    t = {1, 1, 1, 1};
    tune(input, model, g, output, t);
    close(output[12 * 3], 0.6875f, "sharpness adds model high-pass");

    // Strong residuals keep floating point headroom for the next pass.
    model.assign(model.size(), 1.0f);
    t = {4, 1, 1, 0};
    tune(input, model, g, output, t);
    close(output[12 * 3], 3.25f, "inter-pass tuning must not clamp to display range");

    // Letterbox samples are untouched and cannot leak into the picture.
    g.x = g.y = 1; g.fit_width = g.fit_height = 3;
    for (unsigned y = 0; y < 5; ++y) for (unsigned x = 0; x < 5; ++x)
        for (unsigned c = 0; c < 3; ++c)
            model[(y * 5 + x) * 3 + c] = (x && x < 4 && y && y < 4) ? 0.5f : 100.0f;
    t = {1, 0, 1, 0};
    tune(input, model, g, output, t);
    close(output[6 * 3], 0.25f, "fitted edge excludes letterbox from blur");
    close(output[0], 100.0f, "unused padding remains untouched");

    t.intensity = std::numeric_limits<float>::quiet_NaN();
    const auto nonfinite = dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
    require(!nonfinite && nonfinite.error().rejected, "nonfinite tuning must fail");
    for (const dlsslop::NativeTuning& invalid : {dlsslop::NativeTuning{4.5f, 1, 1, 0}, dlsslop::NativeTuning{1, -1, 1, 0},
                                                 dlsslop::NativeTuning{1, 1, 1, 1.5f}}) {
        const auto validated = dlsslop::validate_native_tuning(invalid);
        require(!validated && validated.error().rejected, "out-of-range tuning must reject the request");
    }
    const auto in_place = dlsslop::tune_neural_rgb(input.data(), model.data(), g, model, {});
    require(!in_place && !in_place.error().rejected, "in-place neighbourhood processing must fail");
    goldens();
    std::puts("native tuning: defaults, intensity, tone, structure, sharpness, viewport, errors and goldens passed");
    return 0;
}
