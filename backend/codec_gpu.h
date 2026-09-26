// SPDX-License-Identifier: MIT
#pragma once

#include "codec.h"
#include "native_kernels.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace dlsslop {

class GpuCodec {
    using HostRegister = int (*)(void*, std::size_t, unsigned);
    using HostRelease = int (*)(void*);
    const NativeKernels& kernels_;
    hip_probe::Api& api_;
    const hip_probe::Handle stream_;
    void* proxy_ = nullptr;
    std::uint32_t* invalid_ = nullptr; // Pinned host status word; kernels only ever store 1.
    std::size_t capacity_ = 0;
    // Resolved here so the vendored loader stays as upstream adapted it.
    HostRegister host_register_ = reinterpret_cast<HostRegister>(dlsym(api_.dll, "hipHostRegister"));
    HostRelease host_unregister_ = reinterpret_cast<HostRelease>(dlsym(api_.dll, "hipHostUnregister"));
    HostRelease host_free_ = reinterpret_cast<HostRelease>(dlsym(api_.dll, "hipHostFree"));
    std::uint8_t* pinned_[2]{};
    std::size_t pinned_bytes_ = 0;
    Geometry uploaded_{};
    // Per proxy precision, RGBA8 then RGBA16F: its kernels and bytes per pixel.
    struct Format { Kernel encode, decode; unsigned bytes; };
    static constexpr Format kFormats[2] = {{kEncodeRgba8, kDecodeRgba8, 4}, {kEncodeRgba16f, kDecodeRgba16f, 8}};
    const Format* format_ = kFormats; // The latest encode's.

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
        void* proxy = nullptr;
        api_.Check(api_.hipMalloc(&proxy, bytes), "allocate codec proxy");
        if (proxy_) api_.hipFree(proxy_);
        proxy_ = proxy;
        capacity_ = bytes;
    }

public:
    explicit GpuCodec(const NativeKernels& kernels) : kernels_(kernels), api_(kernels.api), stream_(kernels.stream)
    {
        if (!host_free_) throw std::runtime_error("missing HIP export hipHostFree");
        api_.Check(api_.hipHostMalloc(reinterpret_cast<void**>(&invalid_), sizeof *invalid_, 0), "allocate codec status");
    }
    GpuCodec(const GpuCodec&) = delete;
    GpuCodec& operator=(const GpuCodec&) = delete;
    ~GpuCodec()
    {
        api_.hipStreamSynchronize(stream_);
        unpin();
        if (proxy_) api_.hipFree(proxy_);
        host_free_(invalid_);
    }

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
    // until the stream reaches them (pageable input is staged). Input and
    // decode's output may be host memory or imported device frames: the
    // copies infer their direction.
    void encode(const std::uint8_t* input, const Geometry& g, void* device_rgba, bool fp16 = false)
    {
        validate(g);
        if (!device_rgba)
            throw std::invalid_argument("null GPU encode output");
        format_ = &kFormats[fp16];
        const std::size_t bytes = std::size_t(g.source_width) * g.source_height * format_->bytes;
        reserve(bytes);
        api_.Check(api_.hipMemcpyAsync(proxy_, input, bytes, 4, stream_), "upload codec proxy");
        *invalid_ = 0;
        uploaded_ = g;
        void* args[] = {&proxy_, &device_rgba, &invalid_, &uploaded_};
        kernels_.launch(format_->encode, g.width * g.height, args);
    }

    // A subsequent pass consumes the preceding raw RGB output at the latest
    // encode's neural extent, without a host round-trip or an RGBA8 conversion.
    // Buffers must be distinct. Invalid samples remain recorded until final decode.
    void feedback(void* neural_rgb, void* device_rgba, bool precision16 = true)
    {
        if (!proxy_ || !neural_rgb || !device_rgba || neural_rgb == device_rgba)
            throw std::invalid_argument("GPU feedback without an encode or distinct buffers");
        std::uint32_t precision = precision16 ? 1 : 0;
        void* args[] = {&neural_rgb, &device_rgba, &invalid_, &uploaded_, &precision};
        kernels_.launch(kFeedbackRgb, uploaded_.width * uploaded_.height, args);
    }

    // decode belongs to the latest encode. Both execute on the network stream;
    // output holds the proxy-sized answer once finish() returns. The answer's
    // RGB overwrites the uploaded proxy in place; its alpha passes through.
    void decode(void* neural_rgb, std::uint8_t* output)
    {
        if (!proxy_ || !neural_rgb)
            throw std::invalid_argument("GPU decode without an encode");
        const unsigned pixels = uploaded_.source_width * uploaded_.source_height;
        void* args[] = {&proxy_, &neural_rgb, &invalid_, &uploaded_};
        kernels_.launch(format_->decode, pixels, args);
        api_.Check(api_.hipMemcpyAsync(output, proxy_, std::size_t(pixels) * format_->bytes, 4, stream_),
                   "read codec proxy");
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
