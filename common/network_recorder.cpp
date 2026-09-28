// SPDX-License-Identifier: MIT
#include "network_recorder.h"
#include "nr_vendor.h"

#include <algorithm>
#include <string>
#include <tuple>

namespace dlsslop {

Result<void> vk_check(VkResult result, const char* what)
{
    if (result != VK_SUCCESS) return fail(std::string(what) + " failed (VkResult " + std::to_string(int(result)) + ")");
    return {};
}

namespace {
bool uses_stages(const VulkanFrame& f) { return f.sharpness != 0 || f.color_preserve != 0; }
}  // namespace

NetworkRecorder::NetworkRecorder(const nr::HostDevice& host, const VkPhysicalDeviceMemoryProperties& memory,
                                 VulkanPaths paths)
    : host_(host), memory_(memory), paths_(std::move(paths))
{
}

NetworkRecorder::~NetworkRecorder()
{
    runtime_.reset();
    drop(colour_);
    drop(half_);
}

Result<void> NetworkRecorder::image(Image& i, VkFormat format, unsigned w, unsigned h, VkImageUsageFlags usage)
{
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {w, h, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    DLSSLOP_TRY(vk_check(vkCreateImage(host_.device, &info, nullptr, &i.image), "create frame image"));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(host_.device, i.image, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memory_.memoryTypeCount;
    for (uint32_t t = 0; t < memory_.memoryTypeCount && alloc.memoryTypeIndex == memory_.memoryTypeCount; ++t)
        if ((req.memoryTypeBits & (1u << t)) && (memory_.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            alloc.memoryTypeIndex = t;
    if (alloc.memoryTypeIndex == memory_.memoryTypeCount) return fail("no suitable Vulkan memory type");
    DLSSLOP_TRY(vk_check(vkAllocateMemory(host_.device, &alloc, nullptr, &i.memory), "allocate frame image"));
    return vk_check(vkBindImageMemory(host_.device, i.image, i.memory, 0), "bind frame image");
}

void NetworkRecorder::drop(Image& i)
{
    if (i.image) vkDestroyImage(host_.device, i.image, nullptr);
    if (i.memory) vkFreeMemory(host_.device, i.memory, nullptr);
    i = {};
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
    drop(colour_);
    drop(half_);
    shape_ = {};
    // The network takes 8-bit frames as they are and every other one as RGBA32F.
    const VkFormat format = frame.fp16 ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    const VkImageUsageFlags transfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    DLSSLOP_TRY(image(colour_, format, frame.width, frame.height,
                      transfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT));
    if (frame.fp16) DLSSLOP_TRY(image(half_, VK_FORMAT_R16G16B16A16_SFLOAT, frame.width, frame.height, transfer));
    nr::RuntimeConfig config;
    config.width = frame.width;
    config.height = frame.height;
    config.colour_format = format;
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
                                     VkQueryPool queries, uint32_t query)
{
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    auto layout = [&](VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
                      VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = range;
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {frame.width, frame.height, 1};
    const VkImage staged = frame.fp16 ? half_.image : colour_.image;
    layout(staged, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdCopyBufferToImage(cmd, proxy, staged, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageBlit blit{};
    blit.srcSubresource = blit.dstSubresource = region.imageSubresource;
    blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(frame.width), int32_t(frame.height), 1};
    if (frame.fp16) {
        layout(half_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
        layout(colour_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdBlitImage(cmd, half_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, colour_.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }
    layout(colour_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query);

    nr::ColourFrame colour;
    colour.image = colour_.image;
    colour.format = frame.fp16 ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    colour.width = frame.width;
    colour.height = frame.height;
    colour.before_stage = colour.after_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    colour.before_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    colour.after_access = VK_ACCESS_TRANSFER_READ_BIT;
    colour.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_STORAGE_BIT;
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
    if (frame.motion) {
        const auto settings = [](const VulkanFrame& f) {
            return std::tie(f.passes, f.intensity, f.local_tone, f.local_structure, f.style, f.skin_structure,
                            f.auto_mask);
        };
        nr::TemporalFrame temporal;
        temporal.reset = !last_ || settings(*last_) != settings(frame);
        DLSSLOP_TRY(nr_vendor::record_temporal(*runtime_, cmd, colour, controls, temporal));
    } else {
        DLSSLOP_TRY(nr_vendor::record(*runtime_, cmd, colour, controls));
    }
    if (queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query + 1);

    VkImage result = colour_.image;
    VkImageLayout result_layout = VK_IMAGE_LAYOUT_GENERAL;
    if (frame.fp16) {
        layout(half_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdBlitImage(cmd, colour_.image, VK_IMAGE_LAYOUT_GENERAL, half_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_NEAREST);
        layout(half_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
        result = half_.image;
        result_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    vkCmdCopyImageToBuffer(cmd, result, result_layout, answer, 1, &region);
    last_ = frame;
    return {};
}

}  // namespace dlsslop
