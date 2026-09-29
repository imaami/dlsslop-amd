// DLSSNR-AMD's Vulkan network (external/vulkan/linux/) recorded into a caller's command
// buffers on a caller's device: dlsslopd's own device, or a game's device in
// the layer. Vulkan calls go through the loader.
// SPDX-License-Identifier: MIT
#pragma once
#include "nr_runtime.hpp"
#include "result.h"
#include "vulkan_frame.h"

#include <memory>
#include <optional>
#include <string>

namespace dlsslop {

// Where the Vulkan network finds what it loads.
struct VulkanPaths {
    std::string model;    // dlssnr.bin, extracted from the user's nvngx_dlssnr.dll
    std::string shaders;  // the network's SPIR-V and markers, with runtime/ and temporal/ below
    std::string cache;    // writable pipeline cache; empty: none
};

// VkResult as a Result: WHAT, and the code, when it is not VK_SUCCESS.
Result<void> vk_check(VkResult result, const char* what);
// The first of MEMORY's types among BITS with every property in WANT.
Result<uint32_t> memory_type(const VkPhysicalDeviceMemoryProperties& memory, uint32_t bits,
                             VkMemoryPropertyFlags want);

// The network on one device: its runtime, built for one frame shape at a time,
// and the frame's image, in the proxy's own format, which the runtime reads
// and writes back.
class NetworkRecorder {
public:
    static constexpr unsigned kMaxPasses = 16;
    static constexpr float kMaxControl = 2;

    // HOST's queue takes the runtime's uploads when it is built. MEMORY is the
    // device's, for the recorder's images.
    NetworkRecorder(const nr::HostDevice& host, const VkPhysicalDeviceMemoryProperties& memory, VulkanPaths paths);
    NetworkRecorder(const NetworkRecorder&) = delete;
    // The device must have finished the recorder's work.
    ~NetworkRecorder();

    // True when shape() would build.
    bool shape_differs(const VulkanFrame& frame) const;
    // Builds the network for a frame's shape, unless it has it: seconds of work
    // on first use, true after a build. The device must have finished the
    // recorder's work.
    Result<bool> shape(const VulkanFrame& frame);
    // Records one frame of the shape it has: from PROXY, w x h RGBA8 or RGBA16F,
    // through the network, into ANSWER in the same form. Transfers on FAMILY's
    // queue wrote the proxy and read the last answer before, and do after.
    // EXPORTED: both belong to VK_QUEUE_FAMILY_EXTERNAL between frames, and are
    // given back even when recording fails. With QUERIES, writes timestamps
    // QUERY, once the frame is in the network's image, and QUERY + 1, once the
    // network is done.
    Result<void> record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const VulkanFrame& frame,
                        uint32_t family, bool exported, VkQueryPool queries = VK_NULL_HANDLE, uint32_t query = 0);

private:
    // What a built runtime serves; a frame of anything else rebuilds it.
    struct Shape {
        unsigned width = 0, height = 0;
        bool fp16 = false, motion = false;
        unsigned passes = 0;  // the most it can run
        bool stages = false;  // built with the pass stages
    };

    nr::HostDevice host_;
    VkPhysicalDeviceMemoryProperties memory_;
    VulkanPaths paths_;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory image_memory_ = VK_NULL_HANDLE;
    Shape shape_;
    std::unique_ptr<nr::Runtime> runtime_;
    std::optional<VulkanFrame> last_;  // the history's frame; none since a build

    void drop();
};

}  // namespace dlsslop
