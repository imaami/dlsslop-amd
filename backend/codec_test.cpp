// SPDX-License-Identifier: MIT
#include "codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

std::vector<float> identity_neural(const std::vector<float>& rgba)
{
    std::vector<float> rgb(rgba.size() / 4 * 3);
    for (std::size_t p = 0; p < rgba.size() / 4; ++p)
        for (unsigned c = 0; c < 3; ++c)
            rgb[p * 3 + c] = rgba[p * 4 + c];
    return rgb;
}

void expect_invalid(unsigned width, unsigned height, unsigned tier)
{
    try {
        (void)dlsslop::geometry(width, height, tier);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid geometry accepted");
}

void test_geometry()
{
    const auto a = dlsslop::geometry(1280, 720);
    require(a.width == 1280 && a.height == 768 && !a.x && !a.y &&
            a.fit_width == 1280 && a.fit_height == 720, "720 geometry");
    const auto b = dlsslop::geometry(1600, 900);
    require(b.width == 1600 && b.height == 960, "900 geometry");
    const auto c = dlsslop::geometry(3840, 2160);
    require(c.width == 1920 && c.height == 1152, "large source geometry");
    const auto d = dlsslop::geometry(640, 480, 720);
    require(d.x == 160 && !d.y && d.fit_width == 960 && d.fit_height == 720,
            "4:3 letterbox geometry");
    const auto e = dlsslop::geometry(3440, 1440, 900);
    require(!e.x && e.y == 115 && e.fit_width == 1600 && e.fit_height == 670,
            "ultrawide letterbox geometry");
    const auto f = dlsslop::geometry(1, 16384, 720);
    require(f.fit_width == 1 && f.fit_height == 720, "thin image geometry");
    expect_invalid(0, 720, 720);
    expect_invalid(1920, 0, 1080);
    expect_invalid(16385, 1, 720);
    expect_invalid(1280, 720, 768);
}

void test_input_contract_and_identity()
{
    const auto g = dlsslop::geometry(1280, 720, 720);
    std::vector<std::uint8_t> source(std::size_t(g.source_width) * g.source_height * 4);
    for (unsigned y = 0; y < g.source_height; ++y) {
        for (unsigned x = 0; x < g.source_width; ++x) {
            const std::size_t p = (std::size_t(y) * g.source_width + x) * 4;
            source[p] = std::uint8_t(x);
            source[p + 1] = std::uint8_t(y);
            source[p + 2] = std::uint8_t(x + y);
            source[p + 3] = std::uint8_t(x * 17 + y * 13);
        }
    }
    std::vector<float> encoded;
    dlsslop::encode_rgba8(source.data(), g, encoded);
    require(encoded.size() == std::size_t(g.width) * g.height * 4, "encoded extent");
    // A display encoded 128 must stay around .502, not become linear .216.
    require(encoded[128 * 4] == 0.501953125f, "sRGB input was gamma decoded or not FP16 rounded");
    require(encoded[255 * 4] == 1.0f && encoded[255 * 4 + 3] == 1.0f, "white/alpha contract");
    const std::size_t stride = std::size_t(g.width) * 4;
    require(!std::memcmp(encoded.data() + 720 * stride,
                         encoded.data() + 718 * stride, stride * sizeof(float)),
            "first reflected row must be h-2");
    require(!std::memcmp(encoded.data() + 767 * stride,
                         encoded.data() + 671 * stride, stride * sizeof(float)),
            "last reflected row mismatch");
    const auto neural = identity_neural(encoded);
    std::vector<float> feedback;
    dlsslop::feedback_neural_rgb(neural.data(), g, feedback);
    require(feedback == encoded, "identity feedback must preserve the full encoded input");
    std::vector<std::uint8_t> decoded;
    dlsslop::decode_neural_rgba8(source.data(), g, neural.data(), decoded);
    require(decoded == source, "all 256 SDR codes + alpha must round-trip at native tier");
    dlsslop::decode_rgba8(source.data(), g, encoded.data(), neural.data(), decoded);
    unsigned worst = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        worst = std::max(worst, unsigned(std::abs(int(source[i]) - int(decoded[i]))));
        if (i % 4 == 3)
            require(source[i] == decoded[i], "full compose changed alpha");
    }
    require(worst <= 1, "identity full composition changed image beyond one UNORM8 code");
    dlsslop::decode_rgba8(source.data(), g, encoded.data(), neural.data(), decoded, 0.0f, 0.0f);
    require(decoded == source, "zero transfer must preserve exact source");
}

void test_fit_and_output()
{
    // Tiny constant sources test fitting/clamping without relying on a matching
    // identity sampler; colored corners below then exercise bilinear averaging.
    const auto g = dlsslop::geometry(1, 1, 720);
    const std::uint8_t source[] = {128, 64, 255, 37};
    std::vector<float> encoded;
    dlsslop::encode_rgba8(source, g, encoded);
    require(g.x == 280 && g.fit_width == 720, "square fit geometry");
    require(encoded[0] == 0.0f && encoded[3] == 1.0f, "letterbox must be opaque black");
    require(encoded[g.x * 4] == 0.501953125f, "first fitted pixel");
    const auto neural = identity_neural(encoded);
    std::vector<std::uint8_t> output;
    dlsslop::decode_neural_rgba8(source, g, neural.data(), output);
    require(output == std::vector<std::uint8_t>(source, source + 4), "constant fit decode");
    dlsslop::decode_rgba8(source, g, encoded.data(), neural.data(), output);
    require(output == std::vector<std::uint8_t>(source, source + 4), "constant full composition");

    const auto small = dlsslop::geometry(2, 2, 720);
    const std::uint8_t corners[] = {0, 0, 0, 0, 255, 0, 0, 255,
                                   0, 255, 0, 127, 255, 255, 255, 1};
    dlsslop::encode_rgba8(corners, small, encoded);
    const unsigned x = small.x + 359, y = 359;
    const std::size_t p = (std::size_t(y) * small.width + x) * 4;
    // Coordinates correspond to .498611 on both source axes: interpolation is
    // performed in display encoding, so R/G stay near .499, not ~.734.
    require(std::fabs(encoded[p] - 0.498611111f) < 0.00025f &&
            std::fabs(encoded[p + 1] - 0.498611111f) < 0.00025f &&
            std::fabs(encoded[p + 2] - 0.248613040f) < 0.00025f,
            "bilinear input samples have wrong coordinates or color space");

    std::vector<float> pathological(std::size_t(g.width) * g.height * 3);
    for (std::size_t q = 0; q < pathological.size(); q += 3) {
        pathological[q] = -1.0f;
        pathological[q + 1] = 2.0f;
        pathological[q + 2] = std::numeric_limits<float>::quiet_NaN();
    }
    dlsslop::decode_neural_rgba8(source, g, pathological.data(), output);
    require(output == std::vector<std::uint8_t>({0, 255, 0, 37}),
            "UNORM clamp / NaN guard / alpha preservation");
}

void test_feedback_precision_and_padding()
{
    for (const auto g : {dlsslop::geometry(640, 480, 720),
                         dlsslop::geometry(3440, 1440, 900),
                         dlsslop::geometry(1920, 1080, 1080)}) {
        // Poison everything except the fit rectangle: feedback must ignore raw
        // neural letterbox/padding, then reconstruct the original geometry.
        std::vector<float> neural(std::size_t(g.width) * g.height * 3,
                                  std::numeric_limits<float>::quiet_NaN());
        for (unsigned y = g.y; y < g.y + g.fit_height; ++y) {
            for (unsigned x = g.x; x < g.x + g.fit_width; ++x) {
                float* p = neural.data() + (std::size_t(y) * g.width + x) * 3;
                p[0] = -0.25f;
                p[1] = 1.5f;
                p[2] = y & 1u ? 0.25f : 0.75f;
            }
        }
        const std::size_t marker = std::size_t(g.y + 80) * g.width + g.x + 80;
        neural[marker * 3] = 0.500244140625f;     // Tie to even: .5.
        neural[marker * 3 + 1] = 0.500732421875f; // Tie to even: .5009765625.
        neural[marker * 3 + 2] = 0x1p-24f;       // Smallest binary16 subnormal.
        const auto saved = neural;
        std::vector<float> feedback;
        dlsslop::feedback_neural_rgb(neural.data(), g, feedback);
        require(!std::memcmp(saved.data(), neural.data(), neural.size() * sizeof(float)),
                "feedback modified its raw input");
        require(feedback.size() == std::size_t(g.width) * g.height * 4,
                "feedback extent differs from encode extent");
        require(feedback[marker * 4] == 0.5f &&
                feedback[marker * 4 + 1] == 0.5009765625f &&
                feedback[marker * 4 + 2] == 0x1p-24f,
                "feedback resampled, quantized to UNORM8, or rounded binary16 incorrectly");
        const std::size_t first = std::size_t(g.y) * g.width + g.x;
        require(feedback[first * 4] == -0.25f && feedback[first * 4 + 1] == 1.5f,
                "feedback must not clamp the raw neural result");
        for (std::size_t p = 0; p < feedback.size() / 4; ++p)
            require(feedback[p * 4 + 3] == 1.0f, "feedback alpha is not one");
        if (g.x)
            require(feedback[first * 4 - 4] == 0.0f &&
                    feedback[(first + g.fit_width) * 4] == 0.0f,
                    "feedback did not clear horizontal letterbox");
        if (g.y)
            require(feedback[0] == 0.0f &&
                    feedback[std::size_t(g.y + g.fit_height) * g.width * 4] == 0.0f,
                    "feedback did not clear vertical letterbox");
        const std::size_t stride = std::size_t(g.width) * 4;
        require(!std::memcmp(feedback.data() + g.valid_height * stride,
                             feedback.data() + (g.valid_height - 2) * stride,
                             stride * sizeof(float)),
                "feedback first reflected row differs from h-2");
        require(!std::memcmp(feedback.data() + (g.height - 1) * stride,
                             feedback.data() + (2 * g.valid_height - 1 - g.height) * stride,
                             stride * sizeof(float)),
                "feedback last reflected row mismatch");
    }
}

void test_feedback_invalid_samples()
{
    const auto g = dlsslop::geometry(1280, 720, 720);
    std::vector<float> neural(std::size_t(g.width) * g.height * 3, 0.5f);
    std::vector<float> feedback;
    neural[0] = 65504.0f;
    neural[1] = -65504.0f;
    dlsslop::feedback_neural_rgb(neural.data(), g, feedback);
    require(feedback[0] == 65504.0f && feedback[1] == -65504.0f,
            "feedback rejected finite binary16 limits");
    neural[0] = neural[1] = 0.5f;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(), 65520.0f, -65520.0f}) {
        for (unsigned channel = 0; channel < 3; ++channel) {
            neural[channel] = bad;
            bool rejected = false;
            try {
                dlsslop::feedback_neural_rgb(neural.data(), g, feedback);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            require(rejected, "feedback accepted nonfinite/FP16-overflow fitted sample");
            neural[channel] = 0.5f;
        }
    }
}

void test_fp16_proxy()
{
    const auto g = dlsslop::geometry(1280, 720, 720);
    const std::uint16_t samples[] = {0x0000, 0x0001, 0x03ff, 0x0400, 0x3555,
        0x3801, 0x3c00, 0x4000, 0xb400, 0xc000, 0x7bff, 0xfbff};
    std::vector<std::uint8_t> source(std::size_t(g.source_width) * g.source_height * 8);
    for (std::size_t pixel = 0; pixel < source.size() / 8; ++pixel) {
        for (unsigned channel = 0; channel < 4; ++channel) {
            const std::uint16_t value = channel == 3 ? std::uint16_t(pixel) :
                samples[(pixel + channel) % (sizeof samples / sizeof *samples)];
            std::memcpy(source.data() + pixel * 8 + channel * 2, &value, sizeof value);
        }
    }
    std::vector<float> encoded;
    dlsslop::encode_proxy(source.data(), g, true, encoded);
    require(encoded[4] == 0x1p-24f && encoded[5 * 4] == 0.50048828125f &&
            encoded[8 * 4] == -0.25f && encoded[10 * 4] == 65504.0f,
            "FP16 input was gamma decoded, clamped, or lost binary16 precision");
    const auto neural = identity_neural(encoded);
    std::vector<std::uint8_t> decoded;
    dlsslop::decode_neural_proxy(source.data(), g, true, neural.data(), decoded);
    require(decoded == source, "FP16 proxy native-tier round-trip or alpha preservation");

    const std::uint16_t nonfinite = 0x7e00;
    std::memcpy(source.data(), &nonfinite, sizeof nonfinite);
    bool rejected = false;
    try {
        dlsslop::encode_proxy(source.data(), g, true, encoded);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "FP16 proxy accepted nonfinite RGB input");

    std::vector<float> bad = neural;
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(), 65520.0f}) {
        bad[0] = value;
        rejected = false;
        try {
            dlsslop::decode_neural_proxy(source.data(), g, true, bad.data(), decoded);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "FP16 proxy decode accepted nonfinite/overflow neural sample");
    }
}

void test_feedback_unorm8()
{
    const auto g = dlsslop::geometry(1280, 720, 720);
    std::vector<float> neural(std::size_t(g.width) * g.height * 3, 0.0f);
    neural[0] = -0.25f;
    neural[1] = 1.5f;
    neural[2] = 0.4999f; // Direct UNORM8: 127; half-before-UNORM8 would become 128.
    neural[3] = 0.5f;    // 127.5 ties upward to the even code 128.
    neural[4] = std::numeric_limits<float>::max();
    neural[5] = -std::numeric_limits<float>::max();
    std::vector<float> feedback;
    dlsslop::feedback_neural_rgb(neural.data(), g, feedback, false);
    require(feedback[0] == 0.0f && feedback[1] == 1.0f &&
            feedback[2] == 0.498046875f && feedback[4] == 0.501953125f &&
            feedback[5] == 1.0f && feedback[6] == 0.0f,
            "8-bit feedback must clamp/quantize before binary16 rounding");
    const std::size_t stride = std::size_t(g.width) * 4;
    require(!std::memcmp(feedback.data() + 720 * stride,
                         feedback.data() + 718 * stride, stride * sizeof(float)),
            "8-bit feedback reflection mismatch");
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        neural[0] = bad;
        bool rejected = false;
        try {
            dlsslop::feedback_neural_rgb(neural.data(), g, feedback, false);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "8-bit feedback accepted nonfinite raw sample");
    }
}

} // namespace

int main()
{
    try {
        test_geometry();
        test_input_contract_and_identity();
        test_fit_and_output();
        test_feedback_precision_and_padding();
        test_feedback_invalid_samples();
        test_fp16_proxy();
        test_feedback_unorm8();
        std::puts("codec: geometry, SDR/FP16 transport, reflection, round-trip, fitting, composition and 8/16-bit multi-pass feedback passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "codec test failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
