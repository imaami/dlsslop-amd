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
// Waits run out while other GPU work holds the device, for as long as it
// does: a line at most this often says so.
constexpr std::chrono::seconds kTimeoutLog{10};
}  // namespace

NetworkRecorder::NetworkRecorder(const vulkan::Device& device, VulkanPaths paths)
    : device_(device), paths_(std::move(paths)), storage_(vulkan::storage_limit(device))
{
}

bool NetworkRecorder::shape_differs(const VulkanFrame& frame) const
{
    // Pass stages stay built for frames without them: such a frame blits its answer out instead of copying it.
    return !runtime_ ||
           std::tuple(shape_.width, shape_.height, shape_.fp16, shape_.motion, shape_.passes) !=
               std::tuple(frame.width, frame.height, frame.fp16, frame.motion, passes_of(frame)) ||
           (uses_stages(frame) && !shape_.stages);
}

bool NetworkRecorder::has_extent(const VulkanFrame& frame) const
{
    return runtime_ && shape_.width == frame.width && shape_.height == frame.height;
}

Result<void> NetworkRecorder::plan(const VulkanFrame& frame)
{
    // The runtime's extent was planned before, and shape() reshapes for it without a plan.
    if (has_extent(frame)) return {};
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
    // A runtime of the frame's extent keeps its weights and pipelines.
    if (has_extent(frame)) {
        if (auto reshaped = runtime_->reshape(shape); !reshaped) {
            runtime_.reset();
            return forward(std::move(reshaped).error());
        }
    } else {
        DLSSLOP_TRY(plan_for(frame.width, frame.height));
        // The plan is not kept: the runtime takes the steps and push words it records.
        vulkan::Plan plan = *std::exchange(plan_, std::nullopt);
        runtime_.reset();
        runtime_.emplace(DLSSLOP_TRY(vulkan::Runtime::build(device_, paths_, shape, std::move(plan))));
    }
    shape_ = shape;
    last_.reset();
    recorded_.reset();
    return true;
}

void NetworkRecorder::record(VkCommandBuffer cmd, VkBuffer proxy, VkBuffer answer, const VulkanFrame& frame,
                             uint32_t family, bool exported, VkQueryPool queries, uint32_t query)
{
    // Transfers before the network wrote the proxy and read the last answer; the runtime's read
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
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, pair, 0,
                         nullptr);
    // The network starts over when a wait of the last frame ran out, which other GPU work
    // that holds the device for milliseconds can cause. The motion history also starts over
    // when the frame's settings change, and at a build for another pass count.
    const bool timed_out = runtime_->timed_out();
    if (timed_out) {
        const auto now = std::chrono::steady_clock::now();
        if (device_.log && now >= next_log_) {
            device_.log("a wait of the network ran out while other GPU work held the device: the network "
                        "answered that frame with its input and starts over (logged at most every 10 s)");
            next_log_ = now + kTimeoutLog;
        }
    }
    const vulkan::Controls controls{frame.style,
                                    std::min(frame.intensity, kMaxControl),
                                    std::min(frame.local_tone, kMaxControl),
                                    std::min(frame.local_structure, kMaxControl),
                                    frame.skin_structure,
                                    frame.auto_mask,
                                    frame.sharpness,
                                    frame.color_preserve};
    const auto settings = [](const VulkanFrame& f) {
        return std::tie(f.intensity, f.local_tone, f.local_structure, f.style, f.skin_structure, f.auto_mask);
    };
    runtime_->record(cmd, proxy, answer, controls, timed_out || !last_ || settings(*last_) != settings(frame), queries,
                     query);
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

bool NetworkRecorder::timed_out() const { return runtime_->timed_out(); }

}  // namespace dlsslop
