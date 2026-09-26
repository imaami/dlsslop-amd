// SPDX-License-Identifier: MIT
#pragma once
#include "codec.h"
#include "vendor/hip_api.h"
#include <cmath>
#include <type_traits>

namespace dlsslop {
static_assert(sizeof(Geometry) == 40 && offsetof(Geometry, x) == 24 &&
    offsetof(Geometry, fit_height) == 36 && std::is_standard_layout<Geometry>::value,
    "color kernel geometry ABI");

class GpuColor {
    hip_probe::Api& api_;
    hip_probe::Handle stream_{}, module_{}, kernel_{};
    unsigned width_{}, height_{};
    void release() noexcept
    {
        api_.hipStreamSynchronize(stream_);
        if (module_) api_.hipModuleUnload(module_);
    }
    void validate(const Geometry& g) const
    {
        if (g.width != width_ || g.height != height_ || !g.fit_width || !g.fit_height ||
            g.x >= g.width || g.y >= g.height || g.fit_width > g.width - g.x ||
            g.fit_height > g.height - g.y)
            throw std::invalid_argument("invalid color kernel geometry");
    }
public:
    GpuColor(hip_probe::Api& api, hip_probe::Handle stream, const std::string& path,
              unsigned width, unsigned height) : api_(api), stream_(stream), width_(width), height_(height)
    {
        if (!width || !height || width > 16384 || height > 16384)
            throw std::invalid_argument("invalid color GPU extent");
        try {
            api_.Check(api_.LoadModule(&module_, path.c_str()), "load color preservation module");
            api_.Check(api_.hipModuleGetFunction(&kernel_, module_, "dlsslop_preserve_color"), "find color kernel");
        } catch (...) { release(); throw; }
    }
    ~GpuColor() { release(); }
    GpuColor(const GpuColor&) = delete;
    GpuColor& operator=(const GpuColor&) = delete;
    // Queued on the inference stream; the caller keeps the frame's encoded
    // input (the reference, original_rgba) and raw_rgb unchanged until the
    // kernel runs. It reads neighbours, so output_rgb is a distinct buffer.
    void apply(const void* original_rgba, const void* raw_rgb, void* output_rgb,
               const Geometry& g, float strength)
    {
        validate(g);
        if (!original_rgba || !raw_rgb || !output_rgb || output_rgb == original_rgba ||
            output_rgb == raw_rgb || !std::isfinite(strength) || strength < 0 || strength > 1)
            throw std::invalid_argument("invalid color kernel arguments or buffer alias");
        Geometry geometry_arg = g;
        void* arguments[] = {&original_rgba, &raw_rgb, &output_rgb, &geometry_arg, &strength};
        const std::size_t pixels = std::size_t(width_)*height_;
        api_.Check(api_.hipModuleLaunchKernel(kernel_, unsigned((pixels+255)/256),1,1,
            256,1,1,0,stream_,arguments,nullptr), "apply GPU color preservation");
    }
};
} // namespace dlsslop
