// SPDX-License-Identifier: MIT
#include "codec.hpp"
#include "../tests/golden.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void require(bool ok, const char* message)
{
    if (ok) return;
    std::fprintf(stderr, "codec test failed: %s\n", message);
    std::exit(EXIT_FAILURE);
}

// A geometry the test expects to be valid.
struct geometry geometry(unsigned width, unsigned height, unsigned tier = 0)
{
    const auto g = dlsslop::geometry(width, height, tier);
    require(bool(g), "valid geometry refused");
    return *g;
}

void check(const dlsslop::Result<void>& result, const char* what)
{
    if (result) return;
    std::fprintf(stderr, "codec test failed: %s: %s\n", what, result.error().what.c_str());
    std::exit(1);
}

// A rejected frame, not a worker fault.
bool rejected(const dlsslop::Result<void>& result) { return !result && result.error().rejected; }
// A worker fault, as the invalid_argument and runtime_error it replaced.
template <class T> bool failed(const dlsslop::Result<T>& result) { return !result && !result.error().rejected; }

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
    require(failed(dlsslop::geometry(width, height, tier)), "invalid geometry accepted");
}

void test_geometry()
{
    const auto a = geometry(1280, 720);
    require(a.width == 1280 && a.height == 768 && !a.x && !a.y &&
            a.fit_width == 1280 && a.fit_height == 720, "720 geometry");
    const auto b = geometry(1600, 900);
    require(b.width == 1600 && b.height == 960, "900 geometry");
    const auto c = geometry(3840, 2160);
    require(c.width == 1920 && c.height == 1152, "large source geometry");
    const auto d = geometry(640, 480, 720);
    require(d.x == 160 && !d.y && d.fit_width == 960 && d.fit_height == 720,
            "4:3 letterbox geometry");
    const auto e = geometry(3440, 1440, 900);
    require(!e.x && e.y == 115 && e.fit_width == 1600 && e.fit_height == 670,
            "ultrawide letterbox geometry");
    const auto f = geometry(1, 16384, 720);
    require(f.fit_width == 1 && f.fit_height == 720, "thin image geometry");
    expect_invalid(0, 720, 720);
    expect_invalid(1920, 0, 1080);
    expect_invalid(16385, 1, 720);
    expect_invalid(1280, 720, 768);
    check(dlsslop::validate(e), "consistent geometry refused");
    auto skewed = e;
    ++skewed.fit_height;
    require(failed(dlsslop::validate(skewed)), "inconsistent geometry validated");
}

void test_input_contract_and_identity()
{
    const auto g = geometry(1280, 720, 720);
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
    check(dlsslop::encode_proxy(source.data(), g, false, encoded), "encode_proxy");
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
    check(dlsslop::feedback_neural_rgb(neural.data(), g, feedback), "feedback_neural_rgb");
    require(feedback == encoded, "identity feedback must preserve the full encoded input");
    std::vector<std::uint8_t> decoded(source.size());
    check(dlsslop::decode_neural_proxy(source.data(), g, false, neural.data(), decoded.data()), "decode_neural_proxy");
    require(decoded == source, "all 256 SDR codes + alpha must round-trip at native tier");
}

void test_fit_and_output()
{
    // Tiny constant sources test fitting/clamping without relying on a matching
    // identity sampler; colored corners below then exercise bilinear averaging.
    const auto g = geometry(1, 1, 720);
    const std::uint8_t source[] = {128, 64, 255, 37};
    std::vector<float> encoded;
    check(dlsslop::encode_proxy(source, g, false, encoded), "encode_proxy");
    require(g.x == 280 && g.fit_width == 720, "square fit geometry");
    require(encoded[0] == 0.0f && encoded[3] == 1.0f, "letterbox must be opaque black");
    require(encoded[g.x * 4] == 0.501953125f, "first fitted pixel");
    const auto neural = identity_neural(encoded);
    // One pixel past the source's extent stays untouched.
    std::vector<std::uint8_t> output(8, 0xa5);
    const std::vector<std::uint8_t> expected = {128, 64, 255, 37, 0xa5, 0xa5, 0xa5, 0xa5};
    check(dlsslop::decode_neural_proxy(source, g, false, neural.data(), output.data()), "decode_neural_proxy");
    require(output == expected, "constant fit decode");

    const auto small = geometry(2, 2, 720);
    const std::uint8_t corners[] = {0, 0, 0, 0, 255, 0, 0, 255,
                                   0, 255, 0, 127, 255, 255, 255, 1};
    check(dlsslop::encode_proxy(corners, small, false, encoded), "encode_proxy");
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
        pathological[q + 2] = 65519.0f; // Rounds to the largest finite binary16.
    }
    check(dlsslop::decode_neural_proxy(source, g, false, pathological.data(), output.data()), "decode_neural_proxy");
    require(output == std::vector<std::uint8_t>({0, 255, 255, 37, 0xa5, 0xa5, 0xa5, 0xa5}),
            "UNORM clamp / alpha preservation");
}

// A source larger than the fit is integrated over each pixel's footprint: a
// one-texel stripe of period 3 at exactly 3 texels per pixel encodes to its
// mean at every phase, where a bilinear tap would read all 1 or all 0.
// At 1.5 texels per pixel, the pixels alternate and keep the mean.
void test_area_downscale()
{
    const auto g = geometry(3840, 2160, 720);
    require(g.fit_width == 1280 && g.fit_height == 720, "3x downscale geometry");
    std::vector<std::uint8_t> rgba8(std::size_t(g.source_width) * g.source_height * 4);
    std::vector<std::uint8_t> fp16(rgba8.size() * 2);
    std::vector<float> encoded;
    const float mean = 0.333251953125f; // 1/3 rounded to binary16.
    for (unsigned phase = 0; phase < 3; ++phase) {
        for (std::size_t p = 0; p < rgba8.size() / 4; ++p) {
            const bool lit = (p % g.source_width) % 3 == phase;
            const std::uint16_t half = lit ? 0x3c00 : 0;
            for (unsigned c = 0; c < 4; ++c) {
                rgba8[p * 4 + c] = lit || c == 3 ? 255 : 0;
                std::memcpy(fp16.data() + p * 8 + c * 2, &half, sizeof half);
            }
        }
        for (bool half : {false, true}) {
            check(dlsslop::encode_proxy(half ? fp16.data() : rgba8.data(), g, half, encoded), "encode_proxy");
            for (std::size_t p = 0; p < std::size_t(g.width) * g.valid_height; ++p)
                require(encoded[p * 4] == mean && encoded[p * 4 + 1] == mean && encoded[p * 4 + 2] == mean,
                        "3x downscale does not encode a period-3 stripe to its mean");
        }
    }

    const auto h = geometry(1920, 1080, 720);
    require(h.fit_width == 1280 && h.fit_height == 720, "1.5x downscale geometry");
    for (std::size_t p = 0; p < std::size_t(h.source_width) * h.source_height; ++p)
        for (unsigned c = 0; c < 3; ++c)
            rgba8[p * 4 + c] = (p % h.source_width) % 3 ? 0 : 255;
    check(dlsslop::encode_proxy(rgba8.data(), h, false, encoded), "encode_proxy");
    double sum = 0;
    for (unsigned x = 0; x < h.width; ++x) {
        const float expected = x % 2 ? 0.0f : 0.66650390625f; // 2/3 rounded to binary16.
        require(encoded[(std::size_t(360) * h.width + x) * 4] == expected, "1.5x downscale weights");
        sum += encoded[(std::size_t(360) * h.width + x) * 4];
    }
    require(std::fabs(sum / h.width - 1.0 / 3.0) < 1e-3, "1.5x downscale changed the mean");

    // Every texel under a footprint is read, so a NaN at texel (0, 0), which
    // no bilinear tap at 3x reaches, still rejects the FP16 frame.
    const std::uint16_t nonfinite = 0x7e00;
    std::memcpy(fp16.data(), &nonfinite, sizeof nonfinite);
    require(rejected(dlsslop::encode_proxy(fp16.data(), g, true, encoded)),
            "3x downscale accepted a nonfinite FP16 texel");
}

void expect_decode_rejection(const std::vector<std::uint8_t>& source, const struct geometry& g,
                             const std::vector<float>& neural)
{
    std::vector<std::uint8_t> output(source.size());
    require(rejected(dlsslop::decode_neural_proxy(source.data(), g, false, neural.data(), output.data())),
            "RGBA8 decode accepted a nonfinite/FP16-overflow neural sample");
}

// Like the GPU codec, the CPU RGBA8 decode rejects an invalid neural sample
// instead of painting it and its zero-weight bilinear neighbours black.
void test_decode_invalid_samples()
{
    const auto g = geometry(1280, 720, 720);
    const std::vector<std::uint8_t> source(std::size_t(g.source_width) * g.source_height * 4, 128);
    std::vector<float> encoded;
    check(dlsslop::encode_proxy(source.data(), g, false, encoded), "encode_proxy");
    const auto neural = identity_neural(encoded);
    const std::size_t texel = (std::size_t(100) * g.width + 100) * 3;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), 65520.0f, -65520.0f}) {
        std::vector<float> poisoned = neural;
        poisoned[texel] = bad;
        expect_decode_rejection(source, g, poisoned);
        for (std::size_t q = 0; q < poisoned.size(); q += 3)
            poisoned[q] = bad;
        expect_decode_rejection(source, g, poisoned);
    }
    std::vector<float> limit = neural;
    limit[texel] = 65519.0f;
    limit[texel + 1] = -65519.0f;
    std::vector<std::uint8_t> output(source.size());
    check(dlsslop::decode_neural_proxy(source.data(), g, false, limit.data(), output.data()), "decode_neural_proxy");
    const std::size_t pixel = (std::size_t(100) * g.source_width + 100) * 4;
    require(output[pixel] == 255 && output[pixel + 1] == 0 && output[pixel + 4] == 128 &&
            output[pixel - 4] == 128, "RGBA8 decode rejected or spread the finite binary16 limits");
}

void test_feedback_precision_and_padding()
{
    for (const auto g : {geometry(640, 480, 720),
                         geometry(3440, 1440, 900),
                         geometry(1920, 1080, 1080)}) {
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
        std::vector<float> feedback;
        check(dlsslop::feedback_neural_rgb(neural.data(), g, feedback), "feedback_neural_rgb");
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
    const auto g = geometry(1280, 720, 720);
    std::vector<float> neural(std::size_t(g.width) * g.height * 3, 0.5f);
    std::vector<float> feedback;
    neural[0] = 65504.0f;
    neural[1] = -65504.0f;
    check(dlsslop::feedback_neural_rgb(neural.data(), g, feedback), "feedback_neural_rgb");
    require(feedback[0] == 65504.0f && feedback[1] == -65504.0f,
            "feedback rejected finite binary16 limits");
    neural[0] = neural[1] = 0.5f;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(), 65520.0f, -65520.0f}) {
        for (unsigned channel = 0; channel < 3; ++channel) {
            neural[channel] = bad;
            require(failed(dlsslop::feedback_neural_rgb(neural.data(), g, feedback)),
                    "feedback accepted nonfinite/FP16-overflow fitted sample");
            neural[channel] = 0.5f;
        }
    }
}

void test_fp16_proxy()
{
    const auto g = geometry(1280, 720, 720);
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
    check(dlsslop::encode_proxy(source.data(), g, true, encoded), "encode_proxy");
    require(encoded[4] == 0x1p-24f && encoded[5 * 4] == 0.50048828125f &&
            encoded[8 * 4] == -0.25f && encoded[10 * 4] == 65504.0f,
            "FP16 input was gamma decoded, clamped, or lost binary16 precision");
    const auto neural = identity_neural(encoded);
    std::vector<std::uint8_t> decoded(source.size());
    check(dlsslop::decode_neural_proxy(source.data(), g, true, neural.data(), decoded.data()), "decode_neural_proxy");
    require(decoded == source, "FP16 proxy native-tier round-trip or alpha preservation");

    const std::uint16_t nonfinite = 0x7e00;
    std::memcpy(source.data(), &nonfinite, sizeof nonfinite);
    require(rejected(dlsslop::encode_proxy(source.data(), g, true, encoded)), "FP16 proxy accepted nonfinite RGB input");

    std::vector<float> bad = neural;
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(), 65520.0f}) {
        bad[0] = value;
        require(rejected(dlsslop::decode_neural_proxy(source.data(), g, true, bad.data(), decoded.data())),
                "FP16 proxy decode accepted nonfinite/overflow neural sample");
    }
}

void test_feedback_unorm8()
{
    const auto g = geometry(1280, 720, 720);
    std::vector<float> neural(std::size_t(g.width) * g.height * 3, 0.0f);
    neural[0] = -0.25f;
    neural[1] = 1.5f;
    neural[2] = 0.4999f; // Direct UNORM8: 127; half-before-UNORM8 would become 128.
    neural[3] = 0.5f;    // 127.5 ties upward to the even code 128.
    neural[4] = std::numeric_limits<float>::max();
    neural[5] = -std::numeric_limits<float>::max();
    std::vector<float> feedback;
    check(dlsslop::feedback_neural_rgb(neural.data(), g, feedback, false), "feedback_neural_rgb");
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
        require(failed(dlsslop::feedback_neural_rgb(neural.data(), g, feedback, false)),
                "8-bit feedback accepted nonfinite raw sample");
    }
}

// A source of the extent from fixture SEED: random RGBA8, or FP16 with any
// finite binary16 in every channel, alpha included.
std::vector<std::uint8_t> fixture_source(const struct geometry& g, bool fp16, std::uint32_t seed)
{
    std::vector<std::uint8_t> source(std::size_t(g.source_width) * g.source_height * (fp16 ? 8 : 4));
    if (!fp16) {
        for (std::size_t i = 0; i < source.size(); ++i)
            source[i] = std::uint8_t(golden::bits(seed, i));
        return source;
    }
    for (std::size_t i = 0; i < source.size() / 2; ++i) {
        const std::uint16_t half = golden::half(seed, i);
        std::memcpy(source.data() + i * 2, &half, sizeof half);
    }
    return source;
}

// A network's answer at g's processing extent from fixture SEED, NaN outside
// the fitted picture, which no reference reads.
std::vector<float> fixture_neural(const struct geometry& g, std::uint32_t seed)
{
    std::vector<float> rgb(std::size_t(g.width) * g.height * 3, std::numeric_limits<float>::quiet_NaN());
    for (unsigned y = g.y; y < g.y + g.fit_height; ++y) {
        for (unsigned x = g.x; x < g.x + g.fit_width; ++x) {
            const std::size_t p = (std::size_t(y) * g.width + x) * 3;
            for (std::size_t i = p; i < p + 3; ++i)
                rgb[i] = golden::neural(seed, i);
        }
    }
    return rgb;
}

// The references' outputs over fixtures, bit for bit (golden.hpp).
void test_goldens()
{
    struct Fixture {
        unsigned width, height, tier;
        const char* name;
        std::uint64_t rgba8, fp16; // Or 8-bit and 16-bit feedback.
    };
    // Identity, enlarging (bilinear) and reducing (area-weighted) into the
    // 720 and 1080 tiers, by integral and other ratios, letterboxed or not.
    const Fixture encodes[] = {
        {1280, 720, 720, "identity 720", 0xc84be590d97083e9u, 0x7e7c0cb2abb7a392u},
        {1920, 1080, 1080, "identity 1080", 0x2a813dfd1dad093bu, 0x6f61916b9ff2a475u},
        {853, 480, 720, "upscale 720", 0x58b6349f33c931fbu, 0x18200a5975265df7u},
        {640, 480, 1080, "upscale 1080", 0xb97db8db0e3062dfu, 0x60680b606aa1a7f2u},
        {1920, 1080, 720, "downscale 720", 0x13ec170dead5e0d0u, 0x772f278d3e678ad3u},
        {3440, 1440, 1080, "downscale 1080", 0x856a00c01416d8beu, 0xd6045436a5eb871cu},
    };
    // Letterboxed and pillarboxed answers fed into another pass.
    const Fixture feedbacks[] = {
        {3440, 1440, 900, "3440x1440 900", 0x1caed93f276895a1u, 0x619a08a80adc808au},
        {640, 480, 720, "640x480 720", 0x15ab44c40c58a97eu, 0x9153a140fe206707u},
    };
    // Answers reduced to a smaller source and enlarged to a larger one.
    const Fixture decodes[] = {
        {853, 480, 720, "853x480 720", 0xdc22f9dd3cd6bafcu, 0x37c6f22cddf5d104u},
        {3440, 1440, 1080, "3440x1440 1080", 0x47b36df8fc72ba8eu, 0x734fa59fcffc5ceeu},
    };
    unsigned moved = 0;
    std::uint32_t seed = 0;
    char name[64];
    std::vector<float> rgba;
    for (const auto& f : encodes) {
        const auto g = geometry(f.width, f.height, f.tier);
        for (bool fp16 : {false, true}) {
            const auto source = fixture_source(g, fp16, ++seed);
            check(dlsslop::encode_proxy(source.data(), g, fp16, rgba), "golden encode_proxy");
            std::snprintf(name, sizeof name, "encode_proxy %s %s", fp16 ? "FP16" : "RGBA8", f.name);
            moved += !golden::check(name, rgba.data(), rgba.size() * sizeof(float), fp16 ? f.fp16 : f.rgba8);
        }
    }
    for (const auto& f : feedbacks) {
        const auto g = geometry(f.width, f.height, f.tier);
        const auto neural = fixture_neural(g, ++seed);
        for (bool precision16 : {false, true}) {
            check(dlsslop::feedback_neural_rgb(neural.data(), g, rgba, precision16), "golden feedback_neural_rgb");
            std::snprintf(name, sizeof name, "feedback_neural_rgb %s %s", precision16 ? "16-bit" : "8-bit", f.name);
            moved += !golden::check(name, rgba.data(), rgba.size() * sizeof(float), precision16 ? f.fp16 : f.rgba8);
        }
    }
    for (const auto& f : decodes) {
        const auto g = geometry(f.width, f.height, f.tier);
        const auto neural = fixture_neural(g, ++seed);
        for (bool fp16 : {false, true}) {
            const auto source = fixture_source(g, fp16, ++seed);
            std::vector<std::uint8_t> decoded(source.size());
            check(dlsslop::decode_neural_proxy(source.data(), g, fp16, neural.data(), decoded.data()),
                  "golden decode_neural_proxy");
            std::snprintf(name, sizeof name, "decode_neural_proxy %s %s", fp16 ? "FP16" : "RGBA8", f.name);
            moved += !golden::check(name, decoded.data(), decoded.size(), fp16 ? f.fp16 : f.rgba8);
        }
    }
    require(!moved, "a golden moved");
}

} // namespace

int main()
{
    test_geometry();
    test_input_contract_and_identity();
    test_fit_and_output();
    test_area_downscale();
    test_feedback_precision_and_padding();
    test_feedback_invalid_samples();
    test_decode_invalid_samples();
    test_fp16_proxy();
    test_feedback_unorm8();
    test_goldens();
    std::puts("codec: geometry, SDR/FP16 transport, reflection, round-trip, fitting, area-weighted downscale, invalid-sample rejection, 8/16-bit multi-pass feedback and goldens passed");
    return EXIT_SUCCESS;
}
