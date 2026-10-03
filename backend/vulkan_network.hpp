// The network on Vulkan: DLSSNR-AMD's network (external/vulkan/linux/) on a device of the daemon's own.
// SPDX-License-Identifier: MIT
#pragma once
#include "network_recorder.hpp"
#include "result.hpp"
#include "shm_protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace dlsslop {

class VulkanNetwork {
public:
    static constexpr unsigned kMaxPasses = NetworkRecorder::kMaxPasses;
    static constexpr unsigned kImportSlots = 4;
    // Why infer() rejects a frame that it drops, as a wait of the network ran out
    // while other GPU work held the device: the network answered the frame with
    // its input and logs that itself, at most every 10 s. Serving answers it as
    // failed, so that the layer shows the game's own frame.
    static constexpr const char* kDropped = "a wait of the network ran out while other GPU work held the device";
    // device < 0: the first physical device the network can run on.
    static Result<VulkanNetwork> create(const VulkanPaths& paths, int device);
    VulkanNetwork(VulkanNetwork&&) noexcept;
    ~VulkanNetwork();

    const std::string& device_name() const;
    unsigned device_index() const;
    // Builds the network for a frame's shape now, rather than on its first request:
    // seconds of work, all of it while the caller still reports itself starting.
    // True when that took a build.
    Result<bool> shape(const VulkanFrame& frame);
    // True when shape() would build.
    bool shape_differs(const VulkanFrame& frame) const;
    // Rejects a frame of an extent the network does not take, before shape(): milliseconds of
    // work for a new extent, none after.
    Result<void> plan(const VulkanFrame& frame);
    // Imports an offered proxy/answer pair into a slot (below kImportSlots), releasing
    // the pair it held once the new one is in, and declines one of another device or
    // driver. Takes ownership of each descriptor it imports and sets it to -1; the
    // caller closes the rest.
    bool import(unsigned slot, const ShmTransportOffer& offer, int fds[2]);
    // Slot -1: the frame and answer are host memory at input and output. Otherwise
    // they are the pair imported into that slot, which the caller knows holds them.
    // A frame whose network wait ran out is rejected with kDropped.
    Result<void> infer(const VulkanFrame& frame, int slot, const uint8_t* input, uint8_t* output);

    float upload_ms = 0, inference_ms = 0, readback_ms = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit VulkanNetwork(std::unique_ptr<Impl> impl);
};

}  // namespace dlsslop
