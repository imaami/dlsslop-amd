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
    void* original_{};
    void* output_{};
    unsigned width_{}, height_{};
    bool begun_ = false;
    void release() noexcept
    {
        api_.hipStreamSynchronize(stream_);
        if (output_) api_.hipFree(output_);
        if (original_) api_.hipFree(original_);
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
            api_.Check(api_.hipMalloc(&original_, std::size_t(width)*height*4*sizeof(float)), "allocate original color reference");
            api_.Check(api_.hipMalloc(&output_, std::size_t(width)*height*3*sizeof(float)), "allocate color output");
        } catch (...) { release(); throw; }
    }
    ~GpuColor() { release(); }
    GpuColor(const GpuColor&) = delete;
    GpuColor& operator=(const GpuColor&) = delete;
    void begin(void* input_rgba, const Geometry& g)
    {
        validate(g);
        if (!input_rgba || input_rgba == output_ || input_rgba == original_)
            throw std::invalid_argument("invalid original color input");
        // HIP memcpy kind 3 is device-to-device. Same stream as encoder/network:
        // no host wait and no host staging of the immutable frame reference.
        api_.Check(api_.hipMemcpyAsync(original_, input_rgba,
            std::size_t(width_)*height_*4*sizeof(float), 3, stream_), "retain GPU color reference");
        begun_ = true;
    }
    void* apply(void* raw_rgb, const Geometry& g, float strength)
    {
        validate(g);
        if (!begun_ || !raw_rgb || raw_rgb == output_ || raw_rgb == original_ ||
            !std::isfinite(strength) || strength < 0 || strength > 1)
            throw std::invalid_argument("invalid color kernel arguments or buffer alias");
        Geometry geometry_arg = g;
        void* arguments[] = {&original_, &raw_rgb, &output_, &geometry_arg, &strength};
        const std::size_t pixels = std::size_t(width_)*height_;
        api_.Check(api_.hipModuleLaunchKernel(kernel_, unsigned((pixels+255)/256),1,1,
            256,1,1,0,stream_,arguments,nullptr), "apply GPU color preservation");
        return output_;
    }
};
} // namespace dlsslop
