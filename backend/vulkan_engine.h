// SPDX-License-Identifier: MIT
#pragma once
#include "engine.h"
#include "vulkan_network.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace dlsslop {
// DLSSNR-AMD's network on a Vulkan device of the daemon's own.
class VulkanEngine : public Backend {
    std::unique_ptr<dlsslop::VulkanNetwork> network_;
    unsigned tier_;

    static dlsslop::VulkanFrame frame(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings)
    {
        dlsslop::VulkanFrame f;
        f.width = w;
        f.height = h;
        f.fp16 = settings.fp16;
        f.passes = passes;
        f.intensity = settings.tuning.intensity;
        f.local_tone = settings.tuning.tone;
        f.local_structure = settings.tuning.structure;
        f.style = settings.style;
        f.skin_structure = settings.skin_structure;
        f.auto_mask = settings.auto_mask;
        f.sharpness = settings.tuning.sharpness;
        f.color_preserve = settings.color_preserve;
        f.motion = settings.motion;
        return f;
    }
public:
    VulkanEngine(std::unique_ptr<dlsslop::VulkanNetwork> network, unsigned tier)
        : network_(std::move(network)), tier_(tier)
    {
        max_passes = dlsslop::VulkanNetwork::kMaxPasses;
    }
    const char* name() const override { return "Vulkan"; }
    std::string device() const override
    {
        return "Vulkan device " + std::to_string(network_->device_index()) + " (" + network_->device_name() + ")";
    }
    unsigned tier() const override { return tier_; }
    std::string processing() const override { return "Vulkan on " + network_->device_name() + " at each frame's extent"; }
    // The network sizes itself to each frame: a tier only changes the raster the layer targets.
    bool retier(unsigned tier) override
    {
        tier_ = tier;
        return true;
    }
    // Built before the daemon reports itself ready, for the raster's usual frame.
    void prepare() override
    {
        network_->shape(frame(ShmNativeTier(tier_)->width, tier_, 1, {}));
    }
    bool fits(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings) const override
    {
        return !network_->shape_differs(frame(w, h, passes, settings));
    }
    void reshape(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings) override
    {
        network_->shape(frame(w, h, passes, settings));
    }
    static_assert(kSlots == dlsslop::VulkanNetwork::kImportSlots);
    bool import_into(unsigned slot, const ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2]) override
    {
        int raw[2] = {fds[0].fd, fds[1].fd};
        const bool imported = network_->import(slot, offer, raw);
        fds[0].fd = raw[0];
        fds[1].fd = raw[1];
        return imported;
    }
    // Only HIP traces (hip_only).
    void read_back(void*, const void*, size_t) override
    {
        throw std::logic_error("the Vulkan network cannot trace");
    }
    void infer(const Frames& io, unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings = {},
               dlsslop::FrameTrace* = nullptr) override
    {
        network_->infer(frame(w, h, passes, settings), io.slot, io.proxy, io.answer);
        upload_ms = network_->upload_ms;
        inference_ms = network_->inference_ms;
        readback_ms = network_->readback_ms;
    }
    void self_test(const Options& o) override { run_vulkan_self_test(o, *this); }
};
} // namespace dlsslop
