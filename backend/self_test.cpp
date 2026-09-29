// SPDX-License-Identifier: MIT
#include "engine.h"
#include "files.h"
#include "hip_engine.h"
#include "vulkan_engine.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dlsslop {
namespace {
// RGBA8 as a binary PPM image.
std::string ppm(const std::vector<uint8_t>& rgba, unsigned w, unsigned h)
{
    std::string image = "P6\n" + std::to_string(w) + ' ' + std::to_string(h) + "\n255\n";
    for (size_t p = 0; p < rgba.size(); p += 4) image.append(reinterpret_cast<const char*>(&rgba[p]), 3);
    return image;
}

// W x H opaque RGBA8: red and green ramps over 32-pixel blue checks of ON and OFF.
std::vector<uint8_t> gradient(unsigned w, unsigned h, uint8_t on, uint8_t off)
{
    std::vector<uint8_t> rgba(size_t(w) * h * 4);
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
            uint8_t* p = &rgba[(size_t(y) * w + x) * 4];
            p[0] = uint8_t(x * 255 / (w - 1));
            p[1] = uint8_t(y * 255 / (h - 1));
            p[2] = ((x / 32 ^ y / 32) & 1) ? on : off;
            p[3] = 255;
        }
    return rgba;
}
} // namespace

// The Vulkan network on a deterministic gradient: finite, repeatable and changed.
Result<void> run_self_test(const Options& o, VulkanEngine& engine)
{
    const unsigned passes = std::min(o.passes.value_or(kNativeDefaultPasses), VulkanEngine::max_passes);
    const unsigned w = ShmNativeTier(engine.tier())->width, h = engine.tier();
    const std::vector<uint8_t> input = gradient(w, h, 200, 60);
    std::vector<uint8_t> output(input.size()), first;
    for (unsigned run = 0; run < o.self_test_runs; ++run) {
        DLSSLOP_TRY(engine.infer({input.data(), output.data()}, w, h, passes));
        if (!run) first = output;
        else if (output != first) return fail("self-test run " + std::to_string(run + 1) + " differs from the first");
    }
    size_t changed = 0;
    for (size_t i = 0; i < input.size(); ++i) changed += (i % 4 != 3) && input[i] != output[i];
    if (!changed) return fail("self-test: the network left the input unchanged");
    if (!o.output.empty() && !write_file(o.output, ppm(output, w, h))) return fail("write " + o.output);
    std::fprintf(stderr, "Vulkan self-test PASS: %u identical runs at %ux%u; changed_components=%zu\n"
                 "tier=%u; passes=%u; upload_ms=%.3f; network_ms=%.3f; readback_ms=%.3f\n",
                 o.self_test_runs, w, h, changed, engine.tier(), passes, engine.upload_ms, engine.inference_ms,
                 engine.readback_ms);
    return {};
}

Result<void> run_self_test(const Options& o, HipEngine& engine)
{
    const unsigned passes = o.passes.value_or(kNativeDefaultPasses);
    constexpr unsigned w = 640, h = 360;
    const std::vector<uint8_t> input = gradient(w, h, 192, 64);
    std::vector<uint8_t> output(input.size());
    const unsigned repeats = o.self_test_runs;
    const auto g = DLSSLOP_TRY(dlsslop::geometry(w, h, engine.tier()));
    std::vector<float> first_raw;
    std::vector<uint8_t> first_output;
    // Only the first run checks the codec against the CPU reference, so the
    // later runs time the production path; each must reproduce the first.
    for (unsigned run = 0; run < repeats; ++run) {
        DLSSLOP_TRY(engine.infer({input.data(), output.data()}, w, h, passes, {}, nullptr, !run));
        const auto& raw = *DLSSLOP_TRY(engine.raw_result());
        if (!run) {
            first_raw = raw;
            first_output = output;
            float minimum = raw.front(), maximum = raw.front();
            size_t below_zero = 0, above_one = 0;
            for (float value : raw) {
                if (!std::isfinite(value)) return fail("network produced nonfinite values");
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                below_zero += value < 0.0f;
                above_one += value > 1.0f;
            }
            std::printf("raw network RGB range=%.9g..%.9g below_zero=%zu above_one=%zu samples=%zu\n",
                        double(minimum), double(maximum), below_zero, above_one, raw.size());
        } else if (std::memcmp(raw.data(), first_raw.data(), raw.size() * sizeof(float))) {
            size_t first = raw.size(), different = 0;
            float maximum = 0;
            uint32_t first_before = 0, first_after = 0;
            for (size_t i = 0; i < raw.size(); ++i) {
                uint32_t before, after;
                std::memcpy(&before, &first_raw[i], sizeof(before));
                std::memcpy(&after, &raw[i], sizeof(after));
                if (before == after) continue;
                if (first == raw.size()) {
                    first = i;
                    first_before = before;
                    first_after = after;
                }
                ++different;
                maximum = std::max(maximum, std::fabs(raw[i] - first_raw[i]));
            }
            std::fflush(stdout);
            std::fprintf(stderr,
                "network repeat %u/%u differs from first identical input: samples=%zu/%zu "
                "max_abs_error=%.9g; first x=%zu y=%zu channel=%zu "
                "first=%.9g (0x%08x) repeat=%.9g (0x%08x)\n",
                run + 1, repeats, different, raw.size(), double(maximum),
                (first / 3) % g.width, (first / 3) / g.width, first % 3,
                double(first_raw[first]), unsigned(first_before), double(raw[first]), unsigned(first_after));
            return fail("network is nondeterministic with identical input and fixed seed");
        } else if (const auto [a, b] = std::mismatch(first_output.begin(), first_output.end(), output.begin());
                   a != first_output.end()) {
            const size_t first = a - first_output.begin();
            std::fflush(stdout);
            std::fprintf(stderr,
                "network repeat %u/%u decoded output differs from the first run: "
                "first x=%zu y=%zu channel=%zu first=%u repeat=%u\n",
                run + 1, repeats, (first / 4) % w, (first / 4) / w, first % 4, unsigned(*a), unsigned(*b));
            return fail("decoded output differs from the verified first run");
        }
        std::printf("network repeat %u/%u: %s; passes=%u upload_ms=%.3f network_ms=%.3f readback_ms=%.3f\n",
                    run + 1, repeats, run ? "raw FP32 and output bit-identical" : "baseline", passes,
                    engine.upload_ms, engine.inference_ms, engine.readback_ms);
        std::fflush(stdout);
    }
    uint64_t hash = 14695981039346656037ull;
    unsigned low = 255, high = 0;
    size_t changed = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        hash = (hash ^ output[i]) * 1099511628211ull;
        if ((i & 3) == 3) continue;
        low = std::min(low, unsigned(output[i]));
        high = std::max(high, unsigned(output[i]));
        changed += output[i] != input[i];
    }
    if (high <= low || !changed) return fail("self-test returned constant or unchanged RGB output");
    if (!o.output.empty() && !write_file(o.output, ppm(output, w, h))) return fail("write self-test PPM");
    std::printf("real-network self-test PASS: finite output; %u identical-input runs bit-exact; RGB range=%u..%u; changed_components=%zu; fnv1a64=%016llx\n",
                repeats, low, high, changed, static_cast<unsigned long long>(hash));
    std::printf("tier=%u; passes=%u; blocks=%s; upload_ms=%.3f; network_ms=%.3f; readback_ms=%.3f\n",
                engine.tier(), passes, o.performance ? "upstream performance preset" : "all 71",
                engine.upload_ms, engine.inference_ms, engine.readback_ms);
    return {};
}
} // namespace dlsslop
