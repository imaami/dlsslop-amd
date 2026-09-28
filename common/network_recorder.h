// DLSSNR-AMD's Vulkan network (vulkan-nr/) recorded into a caller's command
// buffers on a caller's device: dlsslopd's own device, or a game's device in
// the layer. Vulkan calls go through the loader.
// SPDX-License-Identifier: MIT
#pragma once
#include "nr_runtime.hpp"
#include "result.h"

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

// One frame's work. The frame is w x h tightly packed RGBA8, or RGBA16F with fp16,
// holding display-encoded values, and the answer comes back in the same form.
struct VulkanFrame {
    unsigned width = 0, height = 0;
    bool fp16 = false;
    // The network's own controls (NVIDIA's DLSSNR.Intensity, LocalToneStrength,
    // LocalStructureStrength), not the HIP backend's residual filters. The model
    // takes each up to kMaxControl; more counts as kMaxControl.
    unsigned passes = 1;
    float intensity = 1, local_tone = 1, local_structure = 1;
    // DLSSNR.Style, 0..2, and the automatic mask, under which skin takes its own
    // local structure; a negative one follows local_structure.
    unsigned style = 0;
    float skin_structure = -1;
    bool auto_mask = true;
    // dlsslop-amd's own stages after every pass, 0..1 each, as on HIP: sharpening,
    // and color preservation against the frame the first pass saw. Either builds
    // them in, as a larger pass count does.
    float sharpness = 0, color_preserve = 0;
    // The network's history, fed by the runtime's motion estimate. It starts over
    // whenever the frame's shape, pass count or controls change.
    bool motion = false;
};

// VkResult as a Result: WHAT, and the code, when it is not VK_SUCCESS.
Result<void> vk_check(VkResult result, const char* what);

// The network on one device: its runtime, built for one frame shape at a time,
// and the image it works in.
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
    // through the network, into ANSWER in the same form. Both buffers are read
    // and written by transfers. With QUERIES, writes timestamps QUERY, once the
    // frame is in the network's image, and QUERY + 1, once the network is done.
    Result<void> record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const VulkanFrame& frame,
                        VkQueryPool queries = VK_NULL_HANDLE, uint32_t query = 0);

private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };
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
    Image colour_, half_;  // the network's frame; with FP16 frames, the FP16 staging image
    Shape shape_;
    std::unique_ptr<nr::Runtime> runtime_;
    std::optional<VulkanFrame> last_;  // the history's frame; none since a build

    Result<void> image(Image& i, VkFormat format, unsigned w, unsigned h, VkImageUsageFlags usage);
    void drop(Image& i);
};

}  // namespace dlsslop
