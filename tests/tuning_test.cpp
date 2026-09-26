// SPDX-License-Identifier: MIT
#include "../backend/tuning.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(float actual, float expected, const char* message)
{
    require(std::fabs(actual - expected) < 1.0e-6f, message);
}
}

int main()
{
    try {
        dlsslop::Geometry g{5, 5, 5, 5, 5, 0, 0, 5, 5};
        std::vector<float> input(25 * 4, 0.25f), model(25 * 3, 0.5f), output;
        for (unsigned p = 0; p < 25; ++p) input[p * 4 + 3] = 1.0f;
        dlsslop::NativeTuning t;
        // A deliberately non-half sample exposes accidental default rounding.
        model[12 * 3] = 0.53123456f;
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        require(!std::memcmp(model.data(), output.data(), model.size() * sizeof(float)),
                "default settings must preserve every model bit");

        t.intensity = 0;
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        require(std::all_of(output.begin(), output.end(), [](float v) { return v == 0.25f; }),
                "zero intensity must return the input");

        model.assign(model.size(), 0.5f);
        t = {2, 1, 1, 0};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        close(output[12 * 3], 0.75f, "intensity scales residual");
        t = {1, 0, 1, 0};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        for (float value : output) close(value, 0.25f, "zero tone removes a constant edit");

        // A single impulse separates the low-pass (centre weight 1/4) from
        // the structural component (remaining 3/4), independently of input.
        model.assign(model.size(), 0.25f);
        model[12 * 3] += 0.25f;
        t = {1, 0, 1, 0};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        close(output[12 * 3], 0.4375f, "structure retains three quarters of centre impulse");
        close(output[11 * 3], 0.21875f, "structure subtracts low-pass neighbour");
        t = {1, 1, 0, 0};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        close(output[12 * 3], 0.3125f, "tone retains one quarter of centre impulse");
        close(output[11 * 3], 0.28125f, "tone spreads one eighth to axial neighbour");
        t = {1, 1, 1, 1};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        close(output[12 * 3], 0.6875f, "sharpness adds model high-pass");

        // Strong residuals keep floating point headroom for the next pass.
        model.assign(model.size(), 1.0f);
        t = {4, 1, 1, 0};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        close(output[12 * 3], 3.25f, "inter-pass tuning must not clamp to display range");

        // Letterbox samples are untouched and cannot leak into the picture.
        g.x = g.y = 1; g.fit_width = g.fit_height = 3;
        for (unsigned y = 0; y < 5; ++y) for (unsigned x = 0; x < 5; ++x)
            for (unsigned c = 0; c < 3; ++c)
                model[(y * 5 + x) * 3 + c] = (x && x < 4 && y && y < 4) ? 0.5f : 100.0f;
        t = {1, 0, 1, 0};
        dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t);
        close(output[6 * 3], 0.25f, "fitted edge excludes letterbox from blur");
        close(output[0], 100.0f, "unused padding remains untouched");

        bool rejected = false;
        t.intensity = std::numeric_limits<float>::quiet_NaN();
        try { dlsslop::tune_neural_rgb(input.data(), model.data(), g, output, t); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "nonfinite tuning must fail");
        rejected = false;
        try { dlsslop::tune_neural_rgb(input.data(), model.data(), g, model, {}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "in-place neighbourhood processing must fail");
        std::puts("native tuning: defaults, intensity, tone, structure, sharpness, viewport and errors passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native tuning test: %s\n", error.what());
        return 1;
    }
}
