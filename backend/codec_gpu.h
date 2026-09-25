// SPDX-License-Identifier: MIT
#pragma once

#include "codec.h"
#include "vendor/hip_api.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dlsslop {

// Geometry is passed by value to the SDKless HIP kernels. Pin its exact ABI.
static_assert(sizeof(Geometry) == 40 && offsetof(Geometry, fit_height) == 36 &&
              std::is_trivially_copyable<Geometry>::value, "GPU codec geometry ABI");

class GpuCodec {
    hip_probe::Api& api_;
    hip_probe::Handle stream_{};
    hip_probe::Handle module_{}, encode_{}, encode16_{}, feedback_{}, decode_{}, decode16_{};
    void* source_ = nullptr;
    void* output_ = nullptr;
    void* invalid_ = nullptr;
    std::size_t capacity_ = 0;
    Geometry uploaded_{};
    bool uploaded_fp16_ = false;

    void reserve(std::size_t bytes)
    {
        if (bytes <= capacity_)
            return;
        api_.Check(api_.hipStreamSynchronize(stream_), "codec resize synchronize");
        void* source = nullptr;
        void* output = nullptr;
        api_.Check(api_.hipMalloc(&source, bytes), "allocate codec source proxy");
        try {
            api_.Check(api_.hipMalloc(&output, bytes), "allocate codec output proxy");
        } catch (...) {
            api_.hipFree(source);
            throw;
        }
        if (source_) api_.hipFree(source_);
        if (output_) api_.hipFree(output_);
        source_ = source;
        output_ = output;
        capacity_ = bytes;
    }

    void release() noexcept
    {
        api_.hipStreamSynchronize(stream_);
        if (source_) api_.hipFree(source_);
        if (output_) api_.hipFree(output_);
        if (invalid_) api_.hipFree(invalid_);
        if (module_) api_.hipModuleUnload(module_);
        source_ = output_ = invalid_ = nullptr;
        module_ = nullptr;
    }

public:
    GpuCodec(hip_probe::Api& api, hip_probe::Handle stream, const std::string& module_path)
        : api_(api), stream_(stream)
    {
        try {
            api_.Check(api_.LoadModule(&module_, module_path.c_str()), "load Linux codec module");
            api_.Check(api_.hipModuleGetFunction(&encode_, module_, "dlsslop_encode_rgba8"), "find encode kernel");
            api_.Check(api_.hipModuleGetFunction(&encode16_, module_, "dlsslop_encode_rgba16f"), "find FP16 encode kernel");
            api_.Check(api_.hipModuleGetFunction(&feedback_, module_, "dlsslop_feedback_rgb"), "find feedback kernel");
            api_.Check(api_.hipModuleGetFunction(&decode_, module_, "dlsslop_decode_rgba8"), "find decode kernel");
            api_.Check(api_.hipModuleGetFunction(&decode16_, module_, "dlsslop_decode_rgba16f"), "find FP16 decode kernel");
            api_.Check(api_.hipMalloc(&invalid_, sizeof(std::uint32_t)), "allocate codec status");
        } catch (...) {
            release();
            throw;
        }
    }
    GpuCodec(const GpuCodec&) = delete;
    GpuCodec& operator=(const GpuCodec&) = delete;
    ~GpuCodec() { release(); }

    // The caller keeps the HIP device current and the network's stream alive.
    // Upload is synchronous; encode is queued on the same stream as inference.
    void encode(const std::vector<std::uint8_t>& input, const Geometry& g,
                void* device_rgba, bool fp16 = false)
    {
        const auto expected = geometry(g.source_width, g.source_height, g.valid_height);
        if (std::memcmp(&expected, &g, sizeof g) || !device_rgba)
            throw std::invalid_argument("invalid GPU encode geometry or output");
        const std::size_t bytes = std::size_t(g.source_width) * g.source_height * (fp16 ? 8 : 4);
        if (input.size() != bytes)
            throw std::invalid_argument("GPU encode input size mismatch");
        reserve(bytes);
        api_.Check(api_.hipMemcpy(source_, input.data(), bytes, 1), "upload codec proxy");
        api_.Check(api_.hipMemsetAsync(invalid_, 0, sizeof(std::uint32_t), stream_), "reset codec status");
        Geometry parameters = g;
        void* args[] = {&source_, &device_rgba, &invalid_, &parameters};
        api_.Check(api_.hipModuleLaunchKernel(fp16 ? encode16_ : encode_, (g.width * g.height + 255u) / 256u,
            1, 1, 256, 1, 1, 0, stream_, args, nullptr), "encode SDR to neural input");
        uploaded_ = g;
        uploaded_fp16_ = fp16;
    }

    // A subsequent pass consumes the preceding raw RGB output at the existing
    // neural extent, without a host round-trip or an RGBA8 conversion. Buffers
    // must be distinct. Invalid samples remain recorded until final decode.
    void feedback(const Geometry& g, void* neural_rgb, void* device_rgba,
                  bool precision16 = true)
    {
        if (std::memcmp(&uploaded_, &g, sizeof g) || !source_ || !neural_rgb ||
            !device_rgba || neural_rgb == device_rgba)
            throw std::invalid_argument("GPU feedback without matching encode or distinct buffers");
        Geometry parameters = g;
        std::uint32_t precision = precision16 ? 1 : 0;
        void* args[] = {&neural_rgb, &device_rgba, &invalid_, &parameters, &precision};
        api_.Check(api_.hipModuleLaunchKernel(feedback_, (g.width * g.height + 255u) / 256u,
            1, 1, 256, 1, 1, 0, stream_, args, nullptr), "prepare neural input for next pass");
    }

    // decode belongs to the latest encode. Both execute on the network stream.
    // Nonfinite FP16 neural samples fail explicitly, never becoming fake output.
    void decode(const Geometry& g, void* neural_rgb, std::vector<std::uint8_t>& output)
    {
        if (std::memcmp(&uploaded_, &g, sizeof g) || !source_ || !neural_rgb)
            throw std::invalid_argument("GPU decode without matching encode");
        const std::size_t bytes = std::size_t(g.source_width) * g.source_height * (uploaded_fp16_ ? 8 : 4);
        output.resize(bytes);
        Geometry parameters = g;
        void* args[] = {&source_, &neural_rgb, &output_, &invalid_, &parameters};
        api_.Check(api_.hipModuleLaunchKernel(uploaded_fp16_ ? decode16_ : decode_, (g.source_width * g.source_height + 255u) / 256u,
            1, 1, 256, 1, 1, 0, stream_, args, nullptr), "decode neural output to SDR");
        api_.Check(api_.hipStreamSynchronize(stream_), "codec decode completion");
        std::uint32_t invalid = 0;
        api_.Check(api_.hipMemcpy(&invalid, invalid_, sizeof invalid, 2), "read codec status");
        if (invalid)
            throw std::runtime_error("proxy input, neural feedback or output contains nonfinite or FP16-overflow samples");
        api_.Check(api_.hipMemcpy(output.data(), output_, bytes, 2), "read codec proxy");
    }
};

} // namespace dlsslop
