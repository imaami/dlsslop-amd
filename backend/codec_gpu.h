// SPDX-License-Identifier: MIT
#pragma once

#include "codec.h"
#include "vendor/hip_api.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace dlsslop {

// Geometry is passed by value to the SDKless HIP kernels. Pin its exact ABI.
static_assert(sizeof(Geometry) == 40 && offsetof(Geometry, fit_height) == 36 &&
              std::is_trivially_copyable<Geometry>::value, "GPU codec geometry ABI");

class GpuCodec {
    using HostRegister = int (*)(void*, std::size_t, unsigned);
    using HostRelease = int (*)(void*);
    hip_probe::Api& api_;
    hip_probe::Handle stream_{};
    hip_probe::Handle module_{}, encode_{}, encode16_{}, feedback_{}, decode_{}, decode16_{};
    void* source_ = nullptr;
    void* output_ = nullptr;
    std::uint32_t* invalid_ = nullptr; // Pinned host status word; kernels only ever store 1.
    std::size_t capacity_ = 0;
    // Resolved here so the vendored loader stays as upstream adapted it.
    HostRegister host_register_ = reinterpret_cast<HostRegister>(dlsym(api_.dll, "hipHostRegister"));
    HostRelease host_unregister_ = reinterpret_cast<HostRelease>(dlsym(api_.dll, "hipHostUnregister"));
    HostRelease host_free_ = reinterpret_cast<HostRelease>(dlsym(api_.dll, "hipHostFree"));
    std::uint8_t* pinned_[2]{};
    std::size_t pinned_bytes_ = 0;
    Geometry uploaded_{};
    bool uploaded_fp16_ = false;

    void unpin() noexcept
    {
        if (pinned_bytes_)
            for (auto* slot : pinned_) host_unregister_(slot);
        pinned_bytes_ = 0;
    }

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
        unpin();
        if (source_) api_.hipFree(source_);
        if (output_) api_.hipFree(output_);
        if (invalid_) host_free_(invalid_);
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
            if (!host_free_) throw std::runtime_error("missing HIP export hipHostFree");
            api_.Check(api_.hipHostMalloc(reinterpret_cast<void**>(&invalid_), sizeof *invalid_, 0), "allocate codec status");
        } catch (...) {
            release();
            throw;
        }
    }
    GpuCodec(const GpuCodec&) = delete;
    GpuCodec& operator=(const GpuCodec&) = delete;
    ~GpuCodec() { release(); }

    // Serving: page-lock growing prefixes of the channel's frame slots so the
    // encode and decode copies DMA straight from and to shared memory. The
    // slots must outlive this codec; call only between frames. On failure the
    // slots stay pageable, which costs HIP staging copies but remains correct.
    void pin(std::uint8_t* input, std::uint8_t* output, std::size_t bytes)
    {
        if (!host_register_ || (bytes <= pinned_bytes_ && input == pinned_[0] && output == pinned_[1]))
            return;
        unpin();
        pinned_[0] = input;
        pinned_[1] = output;
        if (!host_register_(input, bytes, 0)) {
            if (!host_register_(output, bytes, 0)) {
                pinned_bytes_ = bytes;
                return;
            }
            host_unregister_(input);
        }
        host_register_ = nullptr;
        std::fprintf(stderr, "shared-memory pinning unavailable; using staged HIP transfers\n");
    }

    // The caller keeps the HIP device current and the network's stream alive
    // and idle here: a frame ends before encode or after finish(). Upload and
    // encode are queued on the inference stream; input must stay unchanged
    // until the stream reaches them (pageable input is staged).
    void encode(const std::uint8_t* input, const Geometry& g, void* device_rgba, bool fp16 = false)
    {
        const auto expected = geometry(g.source_width, g.source_height, g.valid_height);
        if (std::memcmp(&expected, &g, sizeof g) || !device_rgba)
            throw std::invalid_argument("invalid GPU encode geometry or output");
        const std::size_t bytes = std::size_t(g.source_width) * g.source_height * (fp16 ? 8 : 4);
        reserve(bytes);
        api_.Check(api_.hipMemcpyAsync(source_, input, bytes, 1, stream_), "upload codec proxy");
        *invalid_ = 0;
        Geometry parameters = g;
        void* args[] = {&source_, &device_rgba, &invalid_, &parameters};
        api_.Check(api_.hipModuleLaunchKernel(fp16 ? encode16_ : encode_, (g.width * g.height + 255u) / 256u,
            1, 1, 256, 1, 1, 0, stream_, args, nullptr), "encode proxy to neural input");
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

    // decode belongs to the latest encode. Both execute on the network stream;
    // output holds the proxy-sized answer once finish() returns.
    void decode(const Geometry& g, void* neural_rgb, std::uint8_t* output)
    {
        if (std::memcmp(&uploaded_, &g, sizeof g) || !source_ || !neural_rgb)
            throw std::invalid_argument("GPU decode without matching encode");
        const std::size_t bytes = std::size_t(g.source_width) * g.source_height * (uploaded_fp16_ ? 8 : 4);
        Geometry parameters = g;
        void* args[] = {&source_, &neural_rgb, &output_, &invalid_, &parameters};
        api_.Check(api_.hipModuleLaunchKernel(uploaded_fp16_ ? decode16_ : decode_, (g.source_width * g.source_height + 255u) / 256u,
            1, 1, 256, 1, 1, 0, stream_, args, nullptr), "decode neural output to proxy");
        api_.Check(api_.hipMemcpyAsync(output, output_, bytes, 2, stream_), "read codec proxy");
    }

    // Waits for the stream. Nonfinite or FP16-overflow samples anywhere since
    // the latest encode throw explicitly, never becoming a reported answer.
    void finish()
    {
        api_.Check(api_.hipStreamSynchronize(stream_), "codec decode completion");
        if (*invalid_)
            throw std::range_error("proxy input, neural feedback or output contains nonfinite or FP16-overflow samples");
    }
};

} // namespace dlsslop
