// SPDX-License-Identifier: MIT
#pragma once
#include "engine.hpp"
#include "vulkan_network.hpp"

#include <string>

namespace dlsslop {
// DLSSNR-AMD's network on a Vulkan device of the daemon's own.
class VulkanEngine : public EngineBase<VulkanEngine> {
    VulkanNetwork network_;
    unsigned tier_;

public:
    static constexpr unsigned max_passes = VulkanNetwork::kMaxPasses;
    // The network sizes itself to each frame: a tier only changes the raster the layer targets.
    static constexpr bool rebuilds_for_tier = false;
    static_assert(kSlots == VulkanNetwork::kImportSlots);

    VulkanEngine(VulkanNetwork network, unsigned tier) : network_(std::move(network)), tier_(tier) {}
    const char* name() const { return "Vulkan"; }
    std::string device() const
    {
        return "Vulkan device " + std::to_string(network_.device_index()) + " (" + network_.device_name() + ")";
    }
    unsigned tier() const { return tier_; }
    std::string processing() const { return "Vulkan on " + network_.device_name() + " at each frame's extent"; }
    Result<void> retier(unsigned tier)
    {
        tier_ = tier;
        return {};
    }
    // Built before the daemon reports itself ready, for the raster's usual frame.
    Result<void> prepare() { return network_.shape(vulkan_frame(ShmNativeTier(tier_)->width, tier_, 1, {})).transform([](bool) {}); }
    bool fits(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings) const
    {
        return !network_.shape_differs(vulkan_frame(w, h, passes, settings));
    }
    Result<void> admit(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings)
    {
        return network_.plan(vulkan_frame(w, h, passes, settings));
    }
    Result<void> reshape(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings)
    {
        return network_.shape(vulkan_frame(w, h, passes, settings)).transform([](bool) {});
    }
    bool import_into(unsigned slot, const ShmTransportOffer& offer, Descriptor (&fds)[2])
    {
        int raw[2] = {fds[0].fd, fds[1].fd};
        const bool imported = network_.import(slot, offer, raw);
        fds[0].fd = raw[0];
        fds[1].fd = raw[1];
        return imported;
    }
    Result<void> infer(const Frames& io, unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings = {},
                       FrameTrace* = nullptr)
    {
        DLSSLOP_TRY(network_.infer(vulkan_frame(w, h, passes, settings), io.slot, io.proxy, io.answer));
        upload_ms = network_.upload_ms;
        inference_ms = network_.inference_ms;
        readback_ms = network_.readback_ms;
        return {};
    }
    Result<void> self_test(const Options& o) { return run_self_test(o, *this); }
};
} // namespace dlsslop
