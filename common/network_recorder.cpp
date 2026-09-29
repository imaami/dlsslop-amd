// SPDX-License-Identifier: MIT
#include "network_recorder.h"
#include "nr_vendor.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

namespace dlsslop {

Result<void> vk_check(VkResult result, const char* what)
{
    if (result != VK_SUCCESS) return fail(std::string(what) + " failed (VkResult " + std::to_string(int(result)) + ")");
    return {};
}

Result<uint32_t> memory_type(const VkPhysicalDeviceMemoryProperties& memory, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want) return i;
    return fail("no suitable Vulkan memory type");
}

namespace {
bool uses_stages(const VulkanFrame& f) { return f.sharpness != 0 || f.color_preserve != 0; }
// The proxy's own format: the runtime copies or blits it into the network's input itself.
VkFormat frame_format(const VulkanFrame& f) { return f.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM; }
}  // namespace

NetworkRecorder::NetworkRecorder(const nr::HostDevice& host, const VkPhysicalDeviceMemoryProperties& memory,
                                 VulkanPaths paths)
    : host_(host), memory_(memory), paths_(std::move(paths))
{
}

NetworkRecorder::~NetworkRecorder()
{
    runtime_.reset();
    drop();
}

void NetworkRecorder::drop()
{
    if (image_) vkDestroyImage(host_.device, image_, nullptr);
    if (image_memory_) vkFreeMemory(host_.device, image_memory_, nullptr);
    image_ = VK_NULL_HANDLE;
    image_memory_ = VK_NULL_HANDLE;
}

bool NetworkRecorder::shape_differs(const VulkanFrame& frame) const
{
    return !runtime_ || std::tie(shape_.width, shape_.height, shape_.fp16, shape_.motion) !=
                            std::tie(frame.width, frame.height, frame.fp16, frame.motion) ||
           shape_.passes < std::clamp(frame.passes, 1u, kMaxPasses) || (uses_stages(frame) && !shape_.stages);
}

Result<bool> NetworkRecorder::shape(const VulkanFrame& frame)
{
    if (!shape_differs(frame)) return false;
    const unsigned passes = std::clamp(frame.passes, 1u, kMaxPasses);
    runtime_.reset();
    drop();
    shape_ = {};
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = frame_format(frame);
    info.extent = {frame.width, frame.height, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    DLSSLOP_TRY(vk_check(vkCreateImage(host_.device, &info, nullptr, &image_), "create frame image"));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(host_.device, image_, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = DLSSLOP_TRY(memory_type(memory_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    DLSSLOP_TRY(vk_check(vkAllocateMemory(host_.device, &alloc, nullptr, &image_memory_), "allocate frame image"));
    DLSSLOP_TRY(vk_check(vkBindImageMemory(host_.device, image_, image_memory_, 0), "bind frame image"));
    nr::RuntimeConfig config;
    config.width = frame.width;
    config.height = frame.height;
    config.colour_format = info.format;
    // The network's own answer, as NVIDIA's DLL returns it; the layer composes it.
    config.native_compose = true;
    config.max_passes = passes;
    config.pass_stages = uses_stages(frame);
    config.model_pack = paths_.model;
    config.network_shaders = paths_.shaders;
    config.pipeline_cache = paths_.cache;
    nr::TemporalConfig temporal;
    temporal.enable = frame.motion;
    runtime_ = DLSSLOP_TRY(nr_vendor::build(host_, config, temporal));
    shape_ = {frame.width, frame.height, frame.fp16, frame.motion, passes, config.pass_stages};
    last_.reset();
    return true;
}

Result<void> NetworkRecorder::record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const VulkanFrame& frame,
                                     uint32_t family, bool exported, VkQueryPool queries, uint32_t query)
{
    // Transfers before the network wrote the proxy and read the last answer; the network's read
    // the proxy and write the answer. Exported, the pair belongs to VK_QUEUE_FAMILY_EXTERNAL
    // between frames: taken here and given back below, whatever was recorded.
    const uint32_t outside = exported ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
    const uint32_t inside = exported ? family : VK_QUEUE_FAMILY_IGNORED;
    VkBufferMemoryBarrier pair[2] = {
        {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
         outside, inside, proxy, 0, VK_WHOLE_SIZE},
        {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
         outside, inside, answer, 0, VK_WHOLE_SIZE},
    };
    // The image is rewritten whole: the last frame's copy out of it only has to be done.
    const VkImageMemoryBarrier fresh{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image_,
                                     {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, pair, 1,
                         &fresh);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {frame.width, frame.height, 1};
    vkCmdCopyBufferToImage(cmd, proxy, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    if (queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query);

    // The runtime takes the frame as the copy left it and leaves its answer ready to copy out.
    nr::ColourFrame colour;
    colour.image = image_;
    colour.format = frame_format(frame);
    colour.width = frame.width;
    colour.height = frame.height;
    colour.before = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    colour.after = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    colour.before_stage = colour.after_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    colour.before_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    colour.after_access = VK_ACCESS_TRANSFER_READ_BIT;
    colour.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    nr::Controls controls;
    controls.passes = int(std::clamp(frame.passes, 1u, shape_.passes));
    controls.intensity = std::min(frame.intensity, kMaxControl);
    controls.local_tone = std::min(frame.local_tone, kMaxControl);
    controls.local_structure = std::min(frame.local_structure, kMaxControl);
    controls.style = int(frame.style);
    controls.skin_structure = frame.skin_structure;
    controls.automatic_mask = frame.auto_mask;
    controls.sharpness = frame.sharpness;
    controls.colour_preserve = frame.color_preserve;
    Result<void> recorded;
    if (frame.motion) {
        const auto settings = [](const VulkanFrame& f) {
            return std::tie(f.passes, f.intensity, f.local_tone, f.local_structure, f.style, f.skin_structure,
                            f.auto_mask);
        };
        nr::TemporalFrame temporal;
        temporal.reset = !last_ || settings(*last_) != settings(frame);
        recorded = nr_vendor::record_temporal(*runtime_, cmd, colour, controls, temporal);
    } else {
        recorded = nr_vendor::record(*runtime_, cmd, colour, controls);
    }
    if (recorded) {
        if (queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query + 1);
        vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, answer, 1, &region);
        last_ = frame;
    }
    for (auto& b : pair) {
        std::swap(b.srcAccessMask, b.dstAccessMask);
        std::swap(b.srcQueueFamilyIndex, b.dstQueueFamilyIndex);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, pair, 0,
                         nullptr);
    return recorded;
}

}  // namespace dlsslop
