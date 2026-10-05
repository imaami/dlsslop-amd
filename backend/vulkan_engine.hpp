// SPDX-License-Identifier: MIT
#pragma once
#include "engine.hpp"
#include "vulkan_network.h"

#include <string>
#include <utility>

namespace dlsslop {
// DLSSNR-AMD's network on a Vulkan device of the daemon's own.
class VulkanEngine : public EngineBase<VulkanEngine> {
    // The network, which the engine owns; nullptr once the engine is moved from.
    struct vulkan_network* network_;
    unsigned tier_;

    Result<void> shape(const struct vulkan_frame& frame)
    {
        struct error e;
        if (const enum error_code code = vulkan_network_shape(network_, &frame, &e)) return forward_c(code, e);
        return {};
    }

public:
    static constexpr unsigned max_passes = VULKAN_NETWORK_MAX_PASSES;
    // The network sizes itself to each frame: a tier only changes the raster the layer targets.
    static constexpr bool rebuilds_for_tier = false;
    static_assert(kSlots == VULKAN_NETWORK_IMPORT_SLOTS);

    // Takes NETWORK over.
    VulkanEngine(struct vulkan_network* network, unsigned tier) : network_(network), tier_(tier) {}
    VulkanEngine(VulkanEngine&& other) noexcept
        : EngineBase(std::move(other)), network_(std::exchange(other.network_, nullptr)), tier_(other.tier_)
    {
    }
    ~VulkanEngine() { vulkan_network_destroy(&network_); }
    const char* name() const { return "Vulkan"; }
    std::string device() const
    {
        return "Vulkan device " + std::to_string(vulkan_network_device_index(network_)) + " (" +
               vulkan_network_device_name(network_) + ")";
    }
    unsigned tier() const { return tier_; }
    std::string processing() const
    {
        return std::string("Vulkan on ") + vulkan_network_device_name(network_) + " at each frame's extent";
    }
    Result<void> retier(unsigned tier)
    {
        tier_ = tier;
        return {};
    }
    // Built before the daemon reports itself ready, for the raster's usual frame.
    Result<void> prepare()
    {
        const struct processing_settings settings = processing_settings();
        return shape(processing_vulkan_frame(ShmNativeTier(tier_)->width, tier_, 1, &settings));
    }
    bool fits(unsigned w, unsigned h, unsigned passes, const struct processing_settings& settings) const
    {
        const struct vulkan_frame frame = processing_vulkan_frame(w, h, passes, &settings);
        return !vulkan_network_shape_differs(network_, &frame);
    }
    Result<void> admit(unsigned w, unsigned h, unsigned passes, const struct processing_settings& settings)
    {
        const struct vulkan_frame frame = processing_vulkan_frame(w, h, passes, &settings);
        struct error e;
        if (const enum error_code code = vulkan_network_plan(network_, &frame, &e)) return forward_c(code, e);
        return {};
    }
    Result<void> reshape(unsigned w, unsigned h, unsigned passes, const struct processing_settings& settings)
    {
        return shape(processing_vulkan_frame(w, h, passes, &settings));
    }
    bool import_into(unsigned slot, const ShmTransportOffer& offer, int fds[2])
    {
        return vulkan_network_import(network_, slot, &offer, fds);
    }
    Result<void> infer(const Frames& io, unsigned w, unsigned h, unsigned passes,
                       const struct processing_settings& settings = processing_settings(),
                       struct frame_trace* = nullptr)
    {
        const struct vulkan_frame frame = processing_vulkan_frame(w, h, passes, &settings);
        struct error e;
        if (const enum error_code code = vulkan_network_infer(network_, &frame, io.slot, io.proxy, io.answer, &e))
            return forward_c(code, e);
        const struct vulkan_network_times times = vulkan_network_frame_times(network_);
        upload_ms = times.upload_ms;
        inference_ms = times.inference_ms;
        readback_ms = times.readback_ms;
        return {};
    }
    Result<void> self_test(const struct options& o) { return run_self_test(o, *this); }
};
} // namespace dlsslop
