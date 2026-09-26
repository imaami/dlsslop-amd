// SPDX-License-Identifier: MIT
#pragma once

#include "codec_gpu.h"
#include "temporal_gpu.h"
#include "tuning.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dlsslop::control_selftest {

class Buffer {
    hip_probe::Api& api_;
    hip_probe::Handle stream_;
    std::size_t bytes_;
public:
    void* pointer = nullptr;
    Buffer(const NativeKernels& kernels, std::size_t bytes)
        : api_(kernels.api), stream_(kernels.stream), bytes_(bytes)
    {
        api_.Check(api_.hipMalloc(&pointer, bytes_), "allocate control self-test buffer");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer()
    {
        if (pointer) {
            api_.hipStreamSynchronize(stream_);
            api_.hipFree(pointer);
        }
    }
    void upload(const std::vector<float>& values)
    {
        if (values.size() * sizeof(float) != bytes_)
            throw std::logic_error("control self-test upload size");
        api_.Check(api_.hipStreamSynchronize(stream_), "wait before control self-test upload");
        api_.Check(api_.hipMemcpy(pointer, values.data(), bytes_, 1), "upload control self-test fixture");
    }
    std::vector<float> read() const
    {
        return read_pointer(api_, stream_, pointer, bytes_ / sizeof(float));
    }
    static std::vector<float> read_pointer(hip_probe::Api& api, hip_probe::Handle stream,
                                            const void* pointer, std::size_t samples)
    {
        if (!pointer) throw std::runtime_error("control self-test received null GPU output");
        std::vector<float> result(samples);
        api.Check(api.hipStreamSynchronize(stream), "complete control self-test kernel");
        api.Check(api.hipMemcpy(result.data(), pointer, samples * sizeof(float), 2),
                  "read control self-test output");
        return result;
    }
};

inline void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

// The GPU kernels and their CPU references compute the same IEEE operations
// without contraction, so every result must match bit for bit.
inline void compare(const std::vector<float>& actual, const std::vector<float>& expected, const char* name)
{
    require(actual.size() == expected.size(), "control self-test output size mismatch");
    if (!std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)))
        return;
    std::size_t first = 0;
    while (!std::memcmp(&actual[first], &expected[first], sizeof(float))) ++first;
    std::fprintf(stderr, "%s mismatch: first sample=%zu GPU=%.9g CPU=%.9g\n",
                 name, first, double(actual[first]), double(expected[first]));
    throw std::runtime_error(std::string(name) + " disagrees with CPU reference");
}

inline void check_tuning(const NativeKernels& kernels)
{
    const Geometry g{7, 5, 7, 5, 5, 0, 0, 7, 5};
    const std::size_t pixels = g.width * g.height;
    std::vector<float> input(pixels * 4), model(pixels * 3), reference;
    for (unsigned p = 0; p < pixels; ++p) {
        for (unsigned c = 0; c < 3; ++c) {
            input[p * 4 + c] = float((p * 3 + c * 7) % 31) / 32.0f;
            model[p * 3 + c] = input[p * 4 + c] + float(int((p + c) % 7) - 3) / 64.0f;
        }
        input[p * 4 + 3] = 1;
    }
    Buffer device_input(kernels, input.size() * sizeof(float));
    Buffer device_model(kernels, model.size() * sizeof(float));
    Buffer device_result(kernels, model.size() * sizeof(float));
    device_input.upload(input); device_model.upload(model);
    const NativeTuning states[] = {{}, {0, 1, 1, 0}, {1.75f, .25f, 2.5f, .375f},
                                  {1, 0, 1, 0}, {1, 1, 0, 1}};
    for (const auto& state : states) {
        tune_neural_rgb(input.data(), model.data(), g, reference, state);
        gpu_tune(kernels, g, device_input.pointer, device_model.pointer, device_result.pointer, state);
        compare(device_result.read(), reference, "GPU native tuning");
    }
    std::printf("GPU control self-test: native tuning 5 states exact\n");
    std::fflush(stdout);
}

inline float texture(unsigned x, unsigned y)
{
    unsigned v = x * 0x45d9f3bu + y * 0x119de1f3u;
    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;
    return float(v & 65535u) / 65535.0f;
}

inline void check_temporal(const NativeKernels& kernels)
{
    auto& api = kernels.api;
    const auto stream = kernels.stream;
    constexpr unsigned width = 128, height = 96, padded_height = 104;
    const Geometry g{width, height, width, padded_height, height, 0, 0, width, height};
    const std::size_t pixels = width * padded_height;
    std::vector<float> original(pixels * 4), shifted(pixels * 4), unrelated(pixels * 4), fallback(pixels * 4, .125f);
    std::vector<float> previous_gray(width * height), histories[2];
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            previous_gray[y * width + x] = texture(x / 8, y / 8) * .3f +
                                           texture(x / 2, y / 2) * .5f + texture(x, y) * .2f;
    for (auto& history : histories) history.resize(pixels * 3);
    for (unsigned padded_y = 0; padded_y < padded_height; ++padded_y) {
        const unsigned y = padded_y < height ? padded_y : 2 * height - 2 - padded_y;
        for (unsigned x = 0; x < width; ++x) {
            const unsigned p = padded_y * width + x;
            const float current = dlsslop_temporal::sample(previous_gray.data(), {width, height},
                                                        float(x) - 6.0f, float(y) + 3.0f);
            for (unsigned c = 0; c < 3; ++c) {
                original[p * 4 + c] = previous_gray[y * width + x];
                shifted[p * 4 + c] = current;
                unrelated[p * 4 + c] = texture(x + 4096, y + 4096);
            }
            original[p * 4 + 3] = shifted[p * 4 + 3] = unrelated[p * 4 + 3] = fallback[p * 4 + 3] = 1;
            for (unsigned pass = 0; pass < 2; ++pass) {
                histories[pass][p * 3] = float(x) / 128.0f;
                histories[pass][p * 3 + 1] = float(y) / 128.0f;
                histories[pass][p * 3 + 2] = pass ? .625f : .875f;
            }
        }
    }
    Buffer device_input(kernels, original.size() * sizeof(float));
    Buffer device_fallback(kernels, fallback.size() * sizeof(float));
    Buffer device_first(kernels, histories[0].size() * sizeof(float));
    Buffer device_second(kernels, histories[1].size() * sizeof(float));
    device_input.upload(original); device_fallback.upload(fallback);
    device_first.upload(histories[0]); device_second.upload(histories[1]);
    void* results[] = {device_first.pointer, device_second.pointer};
    GpuTemporal temporal(kernels);

    temporal.begin(device_input.pointer, g, 2, 2, 2);
    for (unsigned pass = 0; pass < 2; ++pass) {
        require(!temporal.history(pass, device_fallback.pointer), "GPU temporal first frame returned uninitialized history");
        api.Check(api.hipMemcpyAsync(temporal.target(pass), results[pass], pixels * 12, 3, stream),
                   "store GPU temporal self-test history");
    }
    temporal.end();
    temporal.begin(device_input.pointer, g, 2, 2, 2);
    for (unsigned pass = 0; pass < 2; ++pass) {
        const auto warped = Buffer::read_pointer(api, stream, temporal.history(pass, device_fallback.pointer), pixels * 4);
        std::vector<float> expected(pixels * 4);
        for (std::size_t p = 0; p < pixels; ++p) {
            for (unsigned c = 0; c < 3; ++c) expected[p * 4 + c] = histories[pass][p * 3 + c];
            expected[p * 4 + 3] = 1;
        }
        compare(warped, expected, "GPU static temporal history");
        api.Check(api.hipMemcpyAsync(temporal.target(pass), results[pass], pixels * 12, 3, stream),
                   "store GPU temporal self-test history");
    }
    temporal.end();

    device_input.upload(shifted);
    temporal.begin(device_input.pointer, g, 2, 2, 2);
    require(!temporal.cut_rejected(), "GPU temporal rejected the known translated frame as a cut");
    for (unsigned pass = 0; pass < 2; ++pass) {
        const auto warped = Buffer::read_pointer(api, stream, temporal.history(pass, device_fallback.pointer), pixels * 4);
        unsigned good = 0, tested = 0;
        for (unsigned y = 16; y + 16 < height; ++y) {
            for (unsigned x = 16; x + 16 < width; ++x) {
                const std::size_t p = (y * width + x) * 4;
                ++tested;
                good += std::fabs(warped[p] * 128.0f - float(x - 6)) < .75f &&
                        std::fabs(warped[p + 1] * 128.0f - float(y + 3)) < .75f &&
                        warped[p + 2] == (pass ? .625f : .875f) && warped[p + 3] == 1;
            }
        }
        std::printf("GPU control self-test: temporal pass %u static exact; translated history %u/%u within 0.75 pixels\n",
                    pass + 1, good, tested);
        std::fflush(stdout);
        require(good * 10 > tested * 8, "GPU temporal translation/history isolation check failed");
        api.Check(api.hipMemcpyAsync(temporal.target(pass), results[pass], pixels * 12, 3, stream),
                   "store GPU temporal self-test history");
    }
    temporal.end();

    // An unrelated frame is a scene cut: every pixel falls back to the pass
    // input (here the frame itself, as in the first pass): the no-history input.
    device_input.upload(unrelated);
    temporal.begin(device_input.pointer, g, 2, 2, 2);
    require(temporal.cut_rejected(), "GPU temporal missed a scene cut");
    for (unsigned pass = 0; pass < 2; ++pass) {
        compare(Buffer::read_pointer(api, stream, temporal.history(pass, device_input.pointer), pixels * 4),
                unrelated, "GPU temporal scene cut fallback");
        api.Check(api.hipMemcpyAsync(temporal.target(pass), results[pass], pixels * 12, 3, stream),
                   "store GPU temporal self-test history");
    }
    temporal.end();
    std::printf("GPU control self-test: temporal scene cut rejects all history\n");

    // A new source size keeps the buffers and the history;
    // a new placement of the picture in the network raster drops the history;
    // a new quality, grid or pass count reallocates and drops it too. Each
    // reallocation is followed by a frame that runs with the new sizes.
    Geometry resized = g;
    resized.source_width /= 2;
    resized.source_height /= 2;
    Geometry moved = resized;
    moved.x = 8;
    moved.fit_width -= 16;
    const struct {
        const Geometry& geometry;
        unsigned quality, grid, passes;
        bool history;
        const char* failure;
    } steps[] = {
        {resized, 2, 2, 2, true, "GPU temporal dropped history for a new source size"},
        {moved, 2, 2, 2, false, "GPU temporal kept history for a moved picture"},
        {moved, 2, 2, 2, true, "GPU temporal dropped history for a steady moved picture"},
        {moved, 1, 2, 2, false, "GPU temporal kept history across a new quality"},
        {moved, 1, 2, 2, true, "GPU temporal dropped history after a new quality"},
        {moved, 1, 1, 2, false, "GPU temporal kept history across a new grid"},
        {moved, 1, 1, 2, true, "GPU temporal dropped history after a new grid"},
        {moved, 1, 1, 1, false, "GPU temporal kept history across a new pass count"},
        {moved, 1, 1, 1, true, "GPU temporal dropped history after a new pass count"},
    };
    for (const auto& step : steps) {
        temporal.begin(device_input.pointer, step.geometry, step.quality, step.grid, step.passes);
        for (unsigned pass = 0; pass < step.passes; ++pass) {
            require((temporal.history(pass, device_input.pointer) != nullptr) == step.history, step.failure);
            temporal.target(pass);
        }
        temporal.end();
    }
    api.Check(api.hipStreamSynchronize(stream), "complete GPU temporal reconfiguration frames");
    std::printf("GPU control self-test: temporal history survives a new source size; "
                "a moved picture, quality, grid or pass count drops it\n");
    std::fflush(stdout);
}

inline float half_value(std::uint16_t value)
{
    const unsigned exponent = (value >> 10) & 31u;
    const unsigned fraction = value & 1023u;
    const float sign = value & 0x8000u ? -1.0f : 1.0f;
    if (!exponent) return sign * float(fraction) * 0x1p-24f;
    if (exponent == 31) throw std::runtime_error("FP16 control self-test produced nonfinite bits");
    return sign * std::ldexp(float(1024u + fraction), int(exponent) - 25);
}

// Runs one FP16 frame through the codec; true when finish() rejects it.
inline bool codec_rejects(GpuCodec& codec, const std::vector<std::uint8_t>& proxy, const Geometry& g,
                          void* device_input, void* device_rgb, std::vector<std::uint8_t>& decoded)
{
    codec.encode(proxy.data(), g, device_input, true);
    codec.decode(device_rgb, decoded.data());
    try {
        codec.finish();
    } catch (const std::range_error&) {
        return true;
    }
    return false;
}

inline void check_codec(const NativeKernels& kernels)
{
    const Geometry g = geometry(7, 5, 720);
    const std::size_t pixels = std::size_t(g.width) * g.height;
    std::vector<std::uint8_t> proxy(g.source_width * g.source_height * 8);
    const std::uint16_t half_samples[] = {0xb800, 0x0000, 0x3400, 0x3a00, 0x3e00, 0x4000};
    for (unsigned p = 0; p < g.source_width * g.source_height; ++p) {
        for (unsigned c = 0; c < 4; ++c) {
            const std::uint16_t value = c == 3 ? std::uint16_t(p & 1 ? 0x3800 : 0x3c00) :
                                       half_samples[(p * 3 + c) % 6];
            std::memcpy(proxy.data() + p * 8 + c * 2, &value, sizeof(value));
        }
    }
    Buffer device_input(kernels, pixels * 4 * sizeof(float));
    Buffer device_rgb(kernels, pixels * 3 * sizeof(float));
    GpuCodec codec(kernels);
    std::vector<float> reference, model(pixels * 3);
    encode_proxy(proxy.data(), g, true, reference);
    codec.encode(proxy.data(), g, device_input.pointer, true);
    compare(device_input.read(), reference, "GPU FP16 proxy encode");

    // A constant signed/extended-range RGB fixture makes all decode resampling
    // exact and verifies that the FP16 route retains alpha and never UNORM-clamps.
    for (std::size_t p = 0; p < pixels; ++p) {
        model[p * 3] = -.25f; model[p * 3 + 1] = 1.5f; model[p * 3 + 2] = .5f;
    }
    device_rgb.upload(model);
    std::vector<std::uint8_t> decoded(proxy.size()), expected(proxy.size());
    codec.decode(device_rgb.pointer, decoded.data());
    codec.finish();
    decode_neural_proxy(proxy.data(), g, true, model.data(), expected.data());
    require(decoded == expected, "GPU FP16 proxy decode disagrees on exact signed/extended-range fixture");
    std::uint16_t first_red, first_green;
    std::memcpy(&first_red, decoded.data(), sizeof(first_red));
    std::memcpy(&first_green, decoded.data() + 2, sizeof(first_green));
    require(half_value(first_red) == -.25f && half_value(first_green) == 1.5f,
            "GPU FP16 proxy decode clipped signed/extended-range values");
    // Decode overwrites the uploaded proxy's RGB in place; a repeat must agree.
    std::vector<std::uint8_t> repeated(proxy.size());
    codec.decode(device_rgb.pointer, repeated.data());
    codec.finish();
    require(repeated == expected, "GPU FP16 proxy decode changed when repeated");

    // The encoder (a NaN proxy sample) and the decoder (an answer beyond
    // binary16) each reject the frame, and every encode starts clean.
    auto poisoned = proxy;
    const std::uint16_t nan = 0x7e00;
    std::memcpy(poisoned.data(), &nan, sizeof(nan));
    require(codec_rejects(codec, poisoned, g, device_input.pointer, device_rgb.pointer, decoded),
            "GPU codec accepted a NaN FP16 proxy sample");
    require(!codec_rejects(codec, proxy, g, device_input.pointer, device_rgb.pointer, decoded),
            "GPU codec rejection outlived its frame");
    for (float& value : model) value = 65536.0f;
    device_rgb.upload(model);
    require(codec_rejects(codec, proxy, g, device_input.pointer, device_rgb.pointer, decoded),
            "GPU codec accepted an answer beyond binary16");

    for (std::size_t p = 0; p < pixels; ++p) {
        model[p * 3] = float(int(p % 29) - 3) / 16.0f;
        model[p * 3 + 1] = float((p / g.width) % 23) / 16.0f;
        model[p * 3 + 2] = float(p % 17) / 19.0f;
    }
    device_rgb.upload(model);
    for (bool precision16 : {true, false}) {
        feedback_neural_rgb(model.data(), g, reference, precision16);
        codec.feedback(device_rgb.pointer, device_input.pointer, precision16);
        compare(device_input.read(), reference, precision16 ? "GPU FP16 feedback" : "GPU UNORM8 feedback");
    }
    std::printf("GPU control self-test: FP16 proxy encode, decode, rejection and 16/8-bit feedback exact\n");

    // A source larger than the fit (4 and 4.17 texels per pixel) takes the
    // area-weighted encode, which no other check reaches.
    const Geometry large = geometry(40, 3000, 720);
    std::vector<std::uint8_t> source(std::size_t(large.source_width) * large.source_height * 8);
    for (std::size_t i = 0; i < source.size() / 2; ++i) {
        const std::uint16_t value = std::uint16_t(0x2c00u + (i * 37u) % 0x1000u); // 1/16 up to 1
        std::memcpy(source.data() + i * 2, &value, sizeof(value));
    }
    Buffer device_large(kernels, std::size_t(large.width) * large.height * 4 * sizeof(float));
    for (bool fp16 : {false, true}) {
        encode_proxy(source.data(), large, fp16, reference);
        codec.encode(source.data(), large, device_large.pointer, fp16);
        compare(device_large.read(), reference, fp16 ? "GPU FP16 downscaling encode" : "GPU RGBA8 downscaling encode");
    }
    std::printf("GPU control self-test: RGBA8 and FP16 area-weighted downscaling encode exact\n");
    std::fflush(stdout);
}

} // namespace dlsslop::control_selftest
