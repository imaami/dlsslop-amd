// DLSSNR-AMD's Vulkan network, run by the project's runtime (vulkan_runtime.h)
// and recorded into a caller's command buffers on a caller's device:
// dlsslopd's own device, or a game's device in the layer. Vulkan calls go
// through the loader.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"
#include "vulkan_frame.h"
#include "vulkan_runtime.h"

#include <optional>
#include <string>

namespace dlsslop {

// The network on one device: its runtime, built for one frame shape at a time,
// and the frame's image, in the proxy's own format, which the runtime reads
// and writes back.
class NetworkRecorder {
public:
    static constexpr unsigned kMaxPasses = 16;
    static constexpr float kMaxControl = 2;

    // DEVICE's queue takes the runtime's build.
    NetworkRecorder(const vulkan::Device& device, VulkanPaths paths);
    NetworkRecorder(const NetworkRecorder&) = delete;
    // The device must have finished the recorder's work.
    ~NetworkRecorder();

    // True when shape() would build.
    bool shape_differs(const VulkanFrame& frame) const;
    // Plans the network for the frame's extent, unless the runtime is built for
    // it or the recorder holds its plan: milliseconds of work. An extent the
    // network does not take on the device is rejected, and then again without
    // planning.
    Result<void> plan(const VulkanFrame& frame);
    // Builds the network for a frame's shape, unless it has it: seconds of work
    // on first use, true after a build. The device must have finished the
    // recorder's work. A rejected extent keeps the network as it was.
    Result<bool> shape(const VulkanFrame& frame);
    // Records one frame of the shape it has: from PROXY, w x h RGBA8 or RGBA16F,
    // through the network, into ANSWER in the same form. Transfers on FAMILY's
    // queue wrote the proxy and read the last answer before, and do after.
    // EXPORTED: both belong to VK_QUEUE_FAMILY_EXTERNAL between frames. With
    // QUERIES, writes timestamps QUERY, once the frame is in the network's
    // image, and QUERY + 1, once the network is done.
    void record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const VulkanFrame& frame, uint32_t family,
                bool exported, VkQueryPool queries = VK_NULL_HANDLE, uint32_t query = 0);

private:
    vulkan::Device device_;
    VulkanPaths paths_;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory image_memory_ = VK_NULL_HANDLE;
    vulkan::Shape shape_{};
    std::optional<vulkan::Runtime> runtime_;
    std::optional<VulkanFrame> last_;  // the history's frame; none since a build
    // The device's storage buffers' limit, the plan of an extent to build, and
    // the last extent rejected.
    uint64_t storage_;
    std::optional<vulkan::Plan> plan_;
    uint32_t rejected_[2] = {};
    std::string rejection_;

    Result<void> plan_for(uint32_t width, uint32_t height);
    void drop();
};

}  // namespace dlsslop
