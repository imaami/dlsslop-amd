// SPDX-License-Identifier: MIT
#include "network_recorder.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

namespace dlsslop {

namespace {
bool uses_stages(const VulkanFrame& f) { return f.sharpness != 0 || f.color_preserve != 0; }
uint8_t passes_of(const VulkanFrame& f) { return uint8_t(std::clamp(f.passes, 1u, NetworkRecorder::kMaxPasses)); }
}  // namespace

NetworkRecorder::NetworkRecorder(const vulkan::Device& device, VulkanPaths paths)
    : device_(device), paths_(std::move(paths)), storage_(vulkan::storage_limit(device))
{
}

NetworkRecorder::~NetworkRecorder()
{
    runtime_.reset();
    drop();
}

void NetworkRecorder::drop()
{
    if (image_) vkDestroyImage(device_.device, image_, nullptr);
    if (image_memory_) vkFreeMemory(device_.device, image_memory_, nullptr);
    image_ = VK_NULL_HANDLE;
    image_memory_ = VK_NULL_HANDLE;
}

bool NetworkRecorder::shape_differs(const VulkanFrame& frame) const
{
    return !runtime_ || std::tie(shape_.width, shape_.height, shape_.fp16, shape_.motion) !=
                            std::tie(frame.width, frame.height, frame.fp16, frame.motion) ||
           shape_.passes < passes_of(frame) || (uses_stages(frame) && !shape_.stages);
}

Result<void> NetworkRecorder::plan(const VulkanFrame& frame)
{
    // The runtime's extent was planned before, and shape() reshapes for it without a plan.
    if (runtime_ && shape_.width == frame.width && shape_.height == frame.height) return {};
    return plan_for(frame.width, frame.height);
}

// plan_ for WIDTH x HEIGHT, unless it holds it or the extent was rejected.
Result<void> NetworkRecorder::plan_for(uint32_t width, uint32_t height)
{
    if (plan_ && plan_->width == width && plan_->height == height) return {};
    if (rejected_[0] == width && rejected_[1] == height && !rejection_.empty()) return forward(Error{rejection_, true});
    auto planned = vulkan::plan(width, height, storage_);
    if (!planned) {
        if (planned.error().rejected) {
            rejected_[0] = width;
            rejected_[1] = height;
            rejection_ = planned.error().what;
        }
        return forward(std::move(planned).error());
    }
    plan_ = std::move(*planned);
    return {};
}

Result<bool> NetworkRecorder::shape(const VulkanFrame& frame)
{
    if (!shape_differs(frame)) return false;
    const vulkan::Shape shape{frame.width, frame.height, frame.fp16, frame.motion, uses_stages(frame),
                              passes_of(frame)};
    // A runtime of the frame's extent keeps its weights and the network's pipelines.
    if (runtime_ && shape_.width == shape.width && shape_.height == shape.height) {
        DLSSLOP_TRY(make_image(shape));
        if (auto reshaped = runtime_->reshape(paths_, shape); !reshaped) {
            runtime_.reset();
            return forward(std::move(reshaped).error());
        }
    } else {
        DLSSLOP_TRY(plan_for(frame.width, frame.height));
        // The plan is not kept: the runtime copies the steps and push words it records.
        const vulkan::Plan plan = *std::exchange(plan_, std::nullopt);
        runtime_.reset();
        DLSSLOP_TRY(make_image(shape));
        runtime_.emplace(DLSSLOP_TRY(vulkan::Runtime::build(device_, paths_, shape, plan)));
    }
    shape_ = shape;
    last_.reset();
    recorded_.reset();
    return true;
}

// The frame's image for SHAPE, in place of the last shape's.
Result<void> NetworkRecorder::make_image(const vulkan::Shape& shape)
{
    drop();
    shape_ = {};
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = shape.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {shape.width, shape.height, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    DLSSLOP_TRY(vk_check(vkCreateImage(device_.device, &info, nullptr, &image_), "create frame image"));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_.device, image_, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex =
        DLSSLOP_TRY(memory_type(device_.memory, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    DLSSLOP_TRY(vk_check(vkAllocateMemory(device_.device, &alloc, nullptr, &image_memory_), "allocate frame image"));
    return vk_check(vkBindImageMemory(device_.device, image_, image_memory_, 0), "bind frame image");
}

void NetworkRecorder::record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const VulkanFrame& frame,
                             uint32_t family, bool exported, VkQueryPool queries, uint32_t query)
{
    // Transfers before the network wrote the proxy and read the last answer; the network's read
    // the proxy and write the answer. Exported, the pair belongs to VK_QUEUE_FAMILY_EXTERNAL
    // between frames: taken here and given back below.
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
    // The motion history starts over when the frame's settings change.
    const vulkan::Controls controls{std::clamp(frame.passes, 1u, unsigned(shape_.passes)),
                                    frame.style,
                                    std::min(frame.intensity, kMaxControl),
                                    std::min(frame.local_tone, kMaxControl),
                                    std::min(frame.local_structure, kMaxControl),
                                    frame.skin_structure,
                                    frame.auto_mask,
                                    frame.sharpness,
                                    frame.color_preserve};
    const auto settings = [](const VulkanFrame& f) {
        return std::tie(f.passes, f.intensity, f.local_tone, f.local_structure, f.style, f.skin_structure, f.auto_mask);
    };
    runtime_->record(cmd, image_, controls, !last_ || settings(*last_) != settings(frame));
    if (queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query + 1);
    vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, answer, 1, &region);
    recorded_ = frame;
    for (auto& b : pair) {
        std::swap(b.srcAccessMask, b.dstAccessMask);
        std::swap(b.srcQueueFamilyIndex, b.dstQueueFamilyIndex);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, pair, 0,
                         nullptr);
}

void NetworkRecorder::submitted()
{
    last_ = recorded_;
    runtime_->submitted();
}

}  // namespace dlsslop
