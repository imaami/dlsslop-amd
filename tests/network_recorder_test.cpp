// SPDX-License-Identifier: MIT
// Host test of the network's motion history across frames and of its
// reshapes, without a GPU. NetworkRecorder builds the network for 64x64
// frames, from a synthetic model and the build's SPIR-V, on a fake device
// whose functions log the commands recorded, each handle named by its first
// use in the frame and described by what it was made as, and each descriptor
// set by what it holds. A frame that is recorded and not submitted must leave
// the history as it was: the next frame's commands, with its motion
// parameters and push constants, must equal those of the frame recorded
// before it. The first frame after a build moves the network's images into
// their layouts, and so does the next one when that frame was not submitted.
// A network reshaped for another shape of its extent must record no command
// and make no pipeline, and then record the frames that a network built for
// that shape records. A frame of one pass without the pass stages must copy
// the proxy straight into the network's input and its answer straight out,
// and copy or blit no image. From a model whose weights free every Swin layer
// of the exponent's upper clamp, a frame must run the kernels and the temporal
// pre block without that clamp. Every frame must judge its waits after the
// network's last dispatch, with the plan's words that waits set when they run
// out, and answer with its input only over the grid that verdict writes: the
// first pass's input into the image that the answer is then copied out of. A
// verdict that says a wait ran out, poked into the fake device's memory, must
// start the next frame over: its arena zeroed from the end of the values on,
// between barriers, and its motion history dropped, as must the frame after
// it when that frame was not submitted. A network built for frames in a
// caller's images binds them only when their generation changes, which keeps
// the motion history; its frames copy no buffer, and blit at most once, into
// the caller's answer. Takes the directory of the network's SPIR-V.
// With --build, it only builds the network for one extent, which makes every
// pipeline of the runtime's own but the temporal pre block that the plan's
// pre block is not, so that tests/vulkan-files.py can see the files that a
// build opens; with --unclamped too, from a synthetic model whose weights
// free every Swin layer of the exponent's upper clamp.
#include "network_recorder.hpp"
#include "vulkan_pack.hpp"

#include <getopt.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>
// After <cstdint>: the runtime's SPIR-V, as it embeds it.
#include "network/network_fallback.hpp"
#include "network/network_verdict.hpp"

namespace {
namespace vulkan = dlsslop::vulkan;

void require(bool value, const char* message)
{
    if (value) return;
    std::fprintf(stderr, "network-recorder test: %s\n", message);
    std::exit(1);
}

// The fake device's handles are numbers from 1. It keeps the sizes of its
// buffers and allocations, and host memory for the allocations mapped; what
// each image, buffer, sampler, shader module and pipeline was made as, each
// view's image, each descriptor set's descriptors by binding, and what each
// pipeline it made was made as, in order. The runtime makes pipelines on
// several threads at once: shader modules and pipelines are recorded under
// a lock.
std::atomic<uint64_t> handles = 0;
std::mutex making;
std::map<uint64_t, VkDeviceSize> sizes;
std::map<uint64_t, std::vector<uint8_t>> host;
std::map<uint64_t, std::string> made;
std::map<uint64_t, uint64_t> views;
struct Descriptor {
    uint64_t resource, sampler;
    VkImageLayout layout;
};
std::map<uint64_t, std::map<uint32_t, Descriptor>> contents;
std::vector<std::string> pipelines_made;
// The descriptors written so far, and those of them that named no image view.
size_t descriptors_written = 0, null_views = 0;
template <class T>
uint64_t id(T handle)
{
    return reinterpret_cast<uint64_t>(handle);
}
template <class T>
VkResult make(T* handle)
{
    *handle = reinterpret_cast<T>(++handles);
    return VK_SUCCESS;
}

// What a frame recorded: its commands, a line each, the handles it named,
// and the gate of its motion parameters and the seed in the push constants of
// its pre block, the first dispatch after the parameters.
struct Frame {
    std::string commands;
    std::map<uint64_t, size_t> names;
    float gate = -1;
    uint32_t seed = UINT32_MAX;
    bool pre = false; // the next push constants are the pre block's
    // The descriptor sets bound, in order.
    std::vector<uint64_t> sets;
};
Frame frame;

// HANDLE as the frame names it: by its first use, which says what it was
// made as.
std::string name(uint64_t handle)
{
    const auto [at, first] = frame.names.try_emplace(handle, frame.names.size());
    return "#" + std::to_string(at->second) + (first ? "(" + made[handle] + ")" : "");
}

// A command: WHAT, the HANDLES it names, its VALUES and the WORDS of its data.
void log(const char* what, std::initializer_list<uint64_t> handles, std::initializer_list<uint64_t> values = {},
         std::span<const uint32_t> words = {})
{
    frame.commands += what;
    for (const uint64_t h : handles) frame.commands += " " + name(h);
    for (const uint64_t v : values) frame.commands += " " + std::to_string(v);
    for (const uint32_t w : words) frame.commands += " " + std::to_string(w);
    frame.commands += '\n';
}
} // namespace

extern "C" {
#define MAKE(name, Info, Handle)                                                                                    \
    VKAPI_ATTR VkResult VKAPI_CALL name(VkDevice, const Info*, const VkAllocationCallbacks*, Handle* handle)       \
    {                                                                                                              \
        return make(handle);                                                                                       \
    }
MAKE(vkCreateDescriptorSetLayout, VkDescriptorSetLayoutCreateInfo, VkDescriptorSetLayout)
MAKE(vkCreatePipelineLayout, VkPipelineLayoutCreateInfo, VkPipelineLayout)
MAKE(vkCreatePipelineCache, VkPipelineCacheCreateInfo, VkPipelineCache)
MAKE(vkCreateDescriptorPool, VkDescriptorPoolCreateInfo, VkDescriptorPool)
MAKE(vkCreateCommandPool, VkCommandPoolCreateInfo, VkCommandPool)
MAKE(vkCreateFence, VkFenceCreateInfo, VkFence)
#undef MAKE
#define DESTROY(name, Handle) \
    VKAPI_ATTR void VKAPI_CALL name(VkDevice, Handle, const VkAllocationCallbacks*) {}
DESTROY(vkDestroyBuffer, VkBuffer)
DESTROY(vkDestroyImage, VkImage)
DESTROY(vkDestroyImageView, VkImageView)
DESTROY(vkDestroySampler, VkSampler)
DESTROY(vkDestroyDescriptorSetLayout, VkDescriptorSetLayout)
DESTROY(vkDestroyPipelineLayout, VkPipelineLayout)
DESTROY(vkDestroyPipeline, VkPipeline)
DESTROY(vkDestroyShaderModule, VkShaderModule)
DESTROY(vkDestroyPipelineCache, VkPipelineCache)
DESTROY(vkDestroyDescriptorPool, VkDescriptorPool)
DESTROY(vkDestroyCommandPool, VkCommandPool)
DESTROY(vkDestroyFence, VkFence)
#undef DESTROY
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*)
{
    host.erase(id(memory));
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                              VkBuffer* buffer)
{
    make(buffer);
    sizes[id(*buffer)] = info->size;
    made[id(*buffer)] = "buffer " + std::to_string(info->size);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice, const VkImageCreateInfo* info, const VkAllocationCallbacks*,
                                             VkImage* image)
{
    make(image);
    made[id(*image)] = "image " + std::to_string(info->extent.width) + "x" + std::to_string(info->extent.height) +
                       " format " + std::to_string(info->format) + " usage " + std::to_string(info->usage);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice, const VkImageViewCreateInfo* info,
                                                 const VkAllocationCallbacks*, VkImageView* view)
{
    make(view);
    views[id(*view)] = id(info->image);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice, const VkSamplerCreateInfo* info, const VkAllocationCallbacks*,
                                               VkSampler* sampler)
{
    make(sampler);
    made[id(*sampler)] = "sampler " + std::to_string(info->magFilter) + " " + std::to_string(info->addressModeU);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo* info,
                                                    const VkAllocationCallbacks*, VkShaderModule* module)
{
    make(module);
    const std::lock_guard lock(making);
    made[id(*module)] = "pipeline " + std::to_string(vulkan_test::fnv1a(info->pCode, info->codeSize));
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo* info,
                                                const VkAllocationCallbacks*, VkDeviceMemory* memory)
{
    make(memory);
    sizes[id(*memory)] = info->allocationSize;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize,
                                           VkMemoryMapFlags, void** data)
{
    auto& bytes = host[id(memory)];
    bytes.resize(sizes[id(memory)]);
    *data = bytes.data();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* req)
{
    *req = {sizes[id(buffer)], 256, 3};
}
VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice, VkImage, VkMemoryRequirements* req)
{
    *req = {256, 256, 3};
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)
{
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize)
{
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t count,
                                                        const VkComputePipelineCreateInfo* infos,
                                                        const VkAllocationCallbacks*, VkPipeline* pipelines)
{
    const std::lock_guard lock(making);
    for (uint32_t i = 0; i < count; ++i) {
        make(&pipelines[i]);
        made[id(pipelines[i])] = made[id(infos[i].stage.module)];
        pipelines_made.push_back(made[id(pipelines[i])]);
    }
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice, VkPipelineCache, size_t* bytes, void*)
{
    *bytes = 0;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                                        VkDescriptorSet* sets)
{
    for (uint32_t i = 0; i < info->descriptorSetCount; ++i) make(&sets[i]);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice, uint32_t count, const VkWriteDescriptorSet* writes,
                                                  uint32_t, const VkCopyDescriptorSet*)
{
    for (const VkWriteDescriptorSet& w : std::span(writes, count)) {
        ++descriptors_written;
        null_views += w.pImageInfo && !w.pImageInfo->imageView;
        Descriptor& d = contents[id(w.dstSet)][w.dstBinding];
        d = w.pBufferInfo ? Descriptor{id(w.pBufferInfo->buffer), 0, VK_IMAGE_LAYOUT_UNDEFINED}
                          : Descriptor{views[id(w.pImageInfo->imageView)], id(w.pImageInfo->sampler),
                                       w.pImageInfo->imageLayout};
    }
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*,
                                                        VkCommandBuffer* cmd)
{
    return make(cmd);
}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
{
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t)
{
    return VK_SUCCESS;
}

// The commands a frame records.
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                                VkDependencyFlags, uint32_t memories, const VkMemoryBarrier* memory,
                                                uint32_t buffers, const VkBufferMemoryBarrier* buffer,
                                                uint32_t images, const VkImageMemoryBarrier* image)
{
    log("barrier", {}, {src, dst, memories});
    for (uint32_t i = 0; i < memories; ++i) log(" memory", {}, {memory[i].srcAccessMask, memory[i].dstAccessMask});
    for (uint32_t i = 0; i < buffers; ++i) log(" buffer", {id(buffer[i].buffer)}, {buffer[i].srcAccessMask});
    for (uint32_t i = 0; i < images; ++i)
        log(" image", {id(image[i].image)}, {uint64_t(image[i].oldLayout), uint64_t(image[i].newLayout)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer, VkBuffer from, VkBuffer to, uint32_t,
                                           const VkBufferCopy*)
{
    log("copy buffer", {id(from), id(to)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes,
                                           uint32_t)
{
    log("fill", {id(buffer)}, {offset, bytes});
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer, VkImage from, VkImageLayout, VkImage to, VkImageLayout,
                                          uint32_t, const VkImageCopy*)
{
    log("copy image", {id(from), id(to)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer, VkImage from, VkImageLayout, VkImage to, VkImageLayout,
                                          uint32_t, const VkImageBlit*, VkFilter)
{
    log("blit", {id(from), id(to)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer, VkBuffer from, VkImage to, VkImageLayout,
                                                  uint32_t, const VkBufferImageCopy*)
{
    log("copy buffer to image", {id(from), id(to)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer, VkImage from, VkImageLayout, VkBuffer to,
                                                  uint32_t, const VkBufferImageCopy*)
{
    log("copy image to buffer", {id(from), id(to)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline pipeline)
{
    log("pipeline", {id(pipeline)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t,
                                                   uint32_t count, const VkDescriptorSet* sets, uint32_t,
                                                   const uint32_t*)
{
    for (uint32_t i = 0; i < count; ++i) {
        frame.sets.push_back(id(sets[i]));
        frame.commands += "set";
        for (const auto& [binding, d] : contents[id(sets[i])])
            frame.commands += " " + std::to_string(binding) + "=" + name(d.resource) +
                              (d.sampler ? "/" + name(d.sampler) : "") + "/" + std::to_string(d.layout);
        frame.commands += '\n';
    }
}
VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t,
                                              uint32_t bytes, const void* data)
{
    std::vector<uint32_t> words(bytes / 4);
    std::memcpy(words.data(), data, bytes);
    log("push", {}, {}, words);
    constexpr size_t seed = (sizeof(vulkan::PushFSwin) + offsetof(vulkan::PushPreImage, seed)) / 4;
    if (frame.pre && seed < words.size()) frame.seed = words[seed];
    frame.pre = false;
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer, uint32_t x, uint32_t y, uint32_t z)
{
    log("dispatch", {}, {x, y, z});
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset)
{
    log("dispatch indirect", {id(buffer)}, {offset});
}
VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer, VkBuffer buffer, VkDeviceSize, VkDeviceSize bytes,
                                             const void* data)
{
    std::vector<uint32_t> words(bytes / 4);
    std::memcpy(words.data(), data, bytes);
    log("update", {id(buffer)}, {}, words);
    std::memcpy(&frame.gate, data, sizeof frame.gate);
    frame.pre = true;
}
VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t)
{
    log("timestamp", {});
}
} // extern "C"

namespace {
// The physical device: one queue family for everything, every format feature
// and storage buffers of 4 GiB. The runtime
// chains at most one structure to a query, which the fakes access through
// its own type: the link-time optimizer inlines them into the runtime, and
// access through VkBaseOutStructure breaks the aliasing rules it relies on.
VKAPI_ATTR void VKAPI_CALL queue_families(VkPhysicalDevice, uint32_t* count, VkQueueFamilyProperties* families)
{
    *count = 1;
    if (families)
        families[0] = {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1, 64, {1, 1, 1}};
}
VKAPI_ATTR void VKAPI_CALL properties(VkPhysicalDevice, VkPhysicalDeviceProperties2* properties)
{
    properties->properties.limits.maxStorageBufferRange = UINT32_MAX;
    auto* allocation = static_cast<VkPhysicalDeviceMaintenance3Properties*>(properties->pNext);
    if (allocation && allocation->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
        allocation->maxMemoryAllocationSize = 1ull << 32;
}
VKAPI_ATTR void VKAPI_CALL format_properties(VkPhysicalDevice, VkFormat, VkFormatProperties2* properties)
{
    properties->formatProperties.optimalTilingFeatures = ~VkFormatFeatureFlags(0);
    auto* features = static_cast<VkFormatProperties3*>(properties->pNext);
    if (features && features->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3)
        features->optimalTilingFeatures = ~VkFormatFeatureFlags2(0);
}

vulkan::Device fake_device()
{
    vulkan::Device d{};
    make(&d.instance);
    make(&d.physical);
    make(&d.device);
    make(&d.queue);
    d.memory.memoryTypeCount = 2;
    d.memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    d.memory.memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    d.memory.memoryHeapCount = 1;
    d.functions = {queue_families, properties, format_properties};
    return d;
}

// F recorded by RECORDER, from and into the same buffers each time.
Frame record(dlsslop::NetworkRecorder& recorder, const dlsslop::VulkanFrame& f)
{
    static VkCommandBuffer cmd;
    static VkBuffer proxy, answer;
    if (!cmd) {
        make(&cmd);
        make(&proxy);
        make(&answer);
    }
    frame = {};
    recorder.record(cmd, proxy, answer, f, 0, false);
    return frame;
}

// How many lines of COMMANDS start with HEAD and end with TAIL.
size_t lines(std::string_view commands, std::string_view head, std::string_view tail = {})
{
    size_t n = 0;
    while (!commands.empty()) {
        const std::string_view line = commands.substr(0, commands.find('\n'));
        n += line.starts_with(head) && line.ends_with(tail);
        commands.remove_prefix(std::min(line.size() + 1, commands.size()));
    }
    return n;
}

// The barrier that moves the network's images into their layouts from
// UNDEFINED: from the top of the pipe to all commands.
constexpr std::string_view kSettle = "barrier 1 65536 0";

// A frame's gate and seed as a message.
std::string history(const Frame& f)
{
    return "gate " + std::to_string(f.gate) + ", seed " + std::to_string(f.seed);
}

// A pipeline made from CODE as a frame names it at its first use.
std::string named(std::span<const uint32_t> code)
{
    return "(pipeline " + std::to_string(vulkan_test::fnv1a(code.data(), code.size_bytes())) + ")";
}

// The handle that TOKEN names, as a frame's line names it, without what its
// first use says it was made as.
std::string_view handle(std::string_view token) { return token.substr(0, token.find_first_of("(/ ")); }

// Whether COMMANDS, a frame's of WIDTH x HEIGHT, judge its waits after the
// network's last dispatch and answer with its input only over the verdict's
// grid: the verdict over one workgroup with the fallback's grid and the
// plan's TIMEOUTS, a barrier from its writes into the indirect read, the
// fallback and the host, then the fallback over the grid the verdict's set
// binds, from the first pass's input into the image the answer is then
// copied out of, and no dispatch after it. The first pass's input is what
// the last transfer before the network's first dispatch writes: the
// network's input, or with later passes the image that keeps it. In image
// mode it is FRAME, and ANSWER, the caller's answer, is the image the
// fallback stores into, with nothing after it, or what that image is blitted
// into last.
bool judged(std::string_view commands, uint32_t width, uint32_t height, const std::vector<uint32_t>& timeouts,
            std::string_view frame = {}, std::string_view answer = {})
{
    std::vector<std::string_view> l;
    for (size_t at = 0; at < commands.size();) {
        const size_t end = commands.find('\n', at);
        l.push_back(commands.substr(at, end - at));
        at = end + 1;
    }
    const auto starts = [](std::string_view head) {
        return [head](std::string_view s) { return s.starts_with(head); };
    };
    std::string_view first = frame;
    if (first.empty())
        for (auto s = l.begin(); s != l.end() && !s->starts_with("dispatch"); ++s)
            if (s->starts_with("copy buffer to image ") || s->starts_with("copy image #") || s->starts_with("blit "))
                first = handle(s->substr(s->rfind(" #") + 1));
    const auto v = std::ranges::find_if(
        l, [](std::string_view s) { return s.starts_with("pipeline #") && s.ends_with(named(kNetworkVerdictSpv)); });
    if (first.empty() || v == l.end() || l.end() - v < 10 || std::none_of(l.begin(), v, starts("dispatch ")))
        return false;
    // The handle bound at BINDING of a set's line.
    const auto bound = [](std::string_view set, std::string_view binding) {
        const size_t at = set.find(binding);
        return at == std::string_view::npos ? std::string_view() : handle(set.substr(at + binding.size()));
    };
    // The result, which the fallback stores into.
    const std::string_view result = bound(v[7], "set 0=");
    std::string push = "push " + std::to_string((width + 7) / 8) + " " + std::to_string((height + 7) / 8) + " " +
                       std::to_string(timeouts.size());
    for (size_t i = 0; i < 13; ++i) push += " " + std::to_string(i < timeouts.size() ? timeouts[i] : 0);
    constexpr VkPipelineStageFlags after =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    constexpr VkAccessFlags reads =
        VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    if (v[2] != push || v[3] != "dispatch 1 1 1" || v[4] != "barrier 2048 " + std::to_string(after) + " 1" ||
        v[5] != " memory 64 " + std::to_string(reads) || !v[6].starts_with("pipeline #") ||
        !v[6].ends_with(named(kNetworkFallbackSpv)) || result.empty() || bound(v[7], " 1=") != first ||
        v[8] != "push " + std::to_string(width) + " " + std::to_string(height) ||
        v[9] != "dispatch indirect " + std::string(bound(v[1], " 1=")) + " 0")
        return false;
    if (!answer.empty() && result == answer) return v + 10 == l.end();
    // The result leaves GENERAL for the copy out, or in image mode the blit into the answer, once
    // the fallback is done.
    if (l.end() - v < 12 || v[10] != "barrier 2048 4096 0" || v[11] != " image " + std::string(result) + " 1 6" ||
        std::any_of(v + 10, l.end(), starts("dispatch")))
        return false;
    const auto out = std::find_if(v + 12, l.end(), [](std::string_view s) {
        return s.starts_with("copy image to buffer ") || s.starts_with("blit ");
    });
    if (out == l.end() || handle(out->substr(out->find('#'))) != result) return false;
    return answer.empty() || (out == v + 12 && out->starts_with("blit ") && l.end() - v == 15 &&
                              handle(out->substr(out->rfind(" #") + 1)) == answer);
}

void check(const dlsslop::VulkanPaths& paths, unsigned passes)
{
    dlsslop::VulkanFrame f;
    f.width = f.height = 64;
    f.motion = true;
    f.passes = passes;
    dlsslop::NetworkRecorder recorder(fake_device(), paths);
    const auto built = recorder.shape(f);
    require(built && *built, built ? "the network was not built" : built.error().what.c_str());
    // The first frame, and the same frame again: the first was not
    // submitted, so the second starts the history too.
    const Frame first = record(recorder, f);
    require(first.gate == 0 && first.seed == 0, ("the first frame has " + history(first)).c_str());
    require(lines(first.commands, kSettle) == 1, "the first frame does not move the images into their layouts");
    const Frame again = record(recorder, f);
    require(again.commands == first.commands,
            ("a frame after one that was not submitted differs from that one: " + history(again)).c_str());
    // Submitted, the first frame starts the history, which the next frame reads.
    recorder.submitted();
    const Frame second = record(recorder, f);
    require(second.gate == 1 && second.seed == 1, ("the second frame has " + history(second)).c_str());
    require(lines(second.commands, kSettle) == 0, "the second frame moves the images into their layouts again");
    // A frame with another intensity starts the history over, but is not
    // submitted: the next frame is the second frame again.
    dlsslop::VulkanFrame other = f;
    other.intensity = 0.5f;
    const Frame reset = record(recorder, other);
    require(reset.gate == 0 && reset.seed == 0, ("a frame with another intensity has " + history(reset)).c_str());
    const Frame after = record(recorder, f);
    require(after.commands == second.commands,
            ("a frame after a reset that was not submitted differs from the frame before: " + history(after)).c_str());
    recorder.submitted();
    const Frame third = record(recorder, f);
    require(third.gate == 1 && third.seed == 2, ("the third frame has " + history(third)).c_str());
}

// The first two frames of F, the second following the first in the motion
// history.
std::string frames_of(dlsslop::NetworkRecorder& recorder, const dlsslop::VulkanFrame& f)
{
    std::string commands = record(recorder, f).commands;
    recorder.submitted();
    return commands + record(recorder, f).commands;
}

// A walk through shapes of 64x64 frames, each a change that the recorder
// rebuilds for: its passes, more or fewer, format, motion history and
// stages. Each shape's frames after the reshape must be those of a network
// built for the shape.
void check_reshapes(const dlsslop::VulkanPaths& paths)
{
    const struct {
        unsigned passes;
        bool fp16, motion;
        float sharpness;
    } walk[] = {{1, false, false, 0}, {1, false, false, 0.5f}, {2, false, false, 0.5f}, {2, false, true, 0.5f},
                {2, true, true, 0.5f}, {1, true, false, 0},    {1, false, true, 0},     {3, false, true, 0},
                {3, true, true, 0.5f}, {1, true, true, 0.5f},  {1, true, false, 0},     {1, false, false, 0},
                {2, false, false, 0},  {1, false, false, 0}};
    dlsslop::NetworkRecorder recorder(fake_device(), paths);
    const auto plan = vulkan::plan(64, 64);
    require(bool(plan), "cannot plan 64x64 frames");
    for (const auto& step : walk) {
        dlsslop::VulkanFrame f;
        f.width = f.height = 64;
        f.passes = step.passes;
        f.fp16 = step.fp16;
        f.motion = step.motion;
        f.sharpness = step.sharpness;
        const std::string what = std::to_string(f.passes) + " passes, " + (f.fp16 ? "FP16" : "RGBA8") +
                                 (f.motion ? ", motion" : "") + (f.sharpness != 0 ? ", pass stages" : "");
        frame = {};
        const size_t pipelines = pipelines_made.size();
        const bool first = &step == walk;
        const auto reshaped = recorder.shape(f);
        require(reshaped && *reshaped, ("the network was not rebuilt for " + what).c_str());
        // The first shape is a build; every other keeps the weights, the
        // arena and the pipelines, records nothing and makes no pipeline.
        require(first || (frame.commands.empty() && pipelines_made.size() == pipelines),
                ("the reshape for " + what + " recorded a command or made a pipeline").c_str());
        // A frame that is not submitted leaves the images' layouts to the next.
        const std::string unsubmitted = record(recorder, f).commands;
        require(lines(unsubmitted, kSettle) == 1,
                ("the first frame after the reshape for " + what + " does not move the images into their layouts")
                    .c_str());
        require(judged(unsubmitted, f.width, f.height, plan->timeouts),
                ("a frame of " + what + " does not judge its waits after the network, or answers with its input "
                 "otherwise than over the verdict's grid")
                    .c_str());
        const std::string after = frames_of(recorder, f);
        require(lines(after, kSettle) == 1,
                ("the frames after the reshape for " + what + " do not move the images into their layouts once")
                    .c_str());
        // With one pass and no pass stages, each of the two frames copies the
        // proxy, #0, into the network's input, #2, the first image it names,
        // and an image into the answer buffer, #1. It copies or blits no image.
        require(f.passes > 1 || f.sharpness != 0 ||
                    (lines(after, "copy buffer to image #0 ", " #2") == 2 &&
                     lines(after, "copy image to buffer #", " #1") == 2 && lines(after, "copy image #") == 0 &&
                     lines(after, "blit") == 0),
                ("the frames of " + what + " do not copy the proxy straight in and the answer straight out").c_str());
        dlsslop::NetworkRecorder built(fake_device(), paths);
        require(built.shape(f).value_or(false), ("cannot build the network for " + what).c_str());
        require(after == frames_of(built, f),
                ("the frames after the reshape for " + what + " differ from those of a build").c_str());
    }
}

// From a model whose weights free every Swin layer of the exponent's upper
// clamp, a frame runs the C=32 kernels without that clamp, and with motion the
// temporal pre block without it, never their twins with the clamp.
void check_unclamped(const std::string& spirv)
{
    const auto plan = vulkan::plan(64, 64);
    vulkan_test::Pack model;
    require(plan && vulkan_test::synthetic_model(*plan, model, true), "cannot make the unclamped model");
    // A pipeline as a frame names it at its first use.
    auto named = [&](const std::string& file) {
        const auto code = dlsslop::read_file(dlsslop::join(spirv, file));
        require(bool(code), ("cannot read " + file).c_str());
        return "(pipeline " + std::to_string(vulkan_test::fnv1a(code->data(), code->size())) + ")";
    };
    for (const bool motion : {false, true}) {
        dlsslop::VulkanFrame f;
        f.width = f.height = 64;
        f.motion = motion;
        dlsslop::NetworkRecorder recorder(fake_device(), {model.path, spirv, ""});
        require(recorder.shape(f).value_or(false), "cannot build the network from the unclamped model");
        const std::string commands = record(recorder, f).commands;
        std::vector<std::pair<std::string, std::string>> twins;
        if (motion) twins.push_back({"temporal/temporal_pre_fp32.spv", "temporal/temporal_pre_fp32nh.spv"});
        else
            for (const auto& [kernel, twin] : vulkan::kUnclamped)
                twins.push_back({std::string("g_") + vulkan::kKernels[size_t(kernel)].stem + ".spv",
                                 std::string("g_") + vulkan::kKernels[size_t(twin)].stem + ".spv"});
        for (const auto& [clamped, unclamped] : twins)
            require(lines(commands, "pipeline #", named(unclamped)) == 1 &&
                        !lines(commands, "pipeline #", named(clamped)),
                    ("a frame from the unclamped model does not run " + unclamped + " instead of " + clamped).c_str());
    }
}
// The caller's images of GENERATION, made on the fake device, which names
// them by what they are and their generation.
vulkan::FrameImages caller_images(uint64_t generation)
{
    vulkan::FrameImages images{generation, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImage frame_image;
    make(&frame_image);
    make(&images.answer);
    make(&images.frame);
    make(&images.answer_view);
    made[id(frame_image)] = "caller frame " + std::to_string(generation);
    made[id(images.answer)] = "caller answer " + std::to_string(generation);
    views[id(images.frame)] = id(frame_image);
    views[id(images.answer_view)] = id(images.answer);
    return images;
}

// F recorded in image mode by RECORDER.
Frame record_in_place(dlsslop::NetworkRecorder& recorder, const dlsslop::VulkanFrame& f)
{
    static VkCommandBuffer cmd;
    if (!cmd) make(&cmd);
    frame = {};
    recorder.record(cmd, f);
    return frame;
}

// The first two frames of F in image mode, the second following the first in
// the motion history.
std::string frames_in_place(dlsslop::NetworkRecorder& recorder, const dlsslop::VulkanFrame& f)
{
    std::string commands = record_in_place(recorder, f).commands;
    recorder.submitted();
    return commands + record_in_place(recorder, f).commands;
}

// How many of the sets that F bound name IMAGE.
size_t naming(const Frame& f, uint64_t image)
{
    return size_t(std::ranges::count_if(f.sets, [image](uint64_t set) {
        const auto at = contents.find(set);
        return at != contents.end() &&
               std::ranges::any_of(at->second, [image](const auto& b) { return b.second.resource == image; });
    }));
}

// IMAGE as F names it, or nothing when F does not.
std::string as_named(const Frame& f, uint64_t image)
{
    const auto at = f.names.find(image);
    return at == f.names.end() ? std::string() : "#" + std::to_string(at->second);
}

// In image mode: a build binds no images and leaves out the sets that
// would, writing no descriptor without one. A bind of the caller's images
// records nothing, makes no pipeline and writes the sets; a bind of a new
// generation keeps the motion history and the images' layouts, and the
// frames after it sample the new images only; images of the generation bound
// write nothing. Through the walk of check_reshapes, each shape with images
// of its own: no frame copies a buffer, makes an image in the frame's format
// or transfers before the network's first dispatch. One pass without the pass
// stages stores its answer in the caller's, which its fallback binds with the
// caller's frame, and copies or blits no image; any other shape blits its
// answer into the caller's last, its only blit. Only the first pass's blocks,
// the stages, the alpha pass, the fallback and the motion estimate sample the
// caller's frame. Each shape's frames after
// the reshape must be those of a network built for the shape and bound to
// the same images.
void check_images(const dlsslop::VulkanPaths& paths)
{
    const auto plan = vulkan::plan(64, 64);
    require(bool(plan), "cannot plan 64x64 frames");
    dlsslop::VulkanFrame f;
    f.width = f.height = 64;
    f.motion = true;
    dlsslop::NetworkRecorder recorder(fake_device(), paths, true);
    const size_t nulls = null_views;
    require(recorder.shape(f).value_or(false) && null_views == nulls,
            "a build in image mode failed, or wrote a descriptor without an image");
    uint64_t generation = 0;
    const vulkan::FrameImages images = caller_images(++generation);
    frame = {};
    size_t pipelines = pipelines_made.size(), written = descriptors_written;
    require(recorder.shape(f, &images).value_or(false) && frame.commands.empty() &&
                pipelines_made.size() == pipelines && descriptors_written > written,
            "a bind of the caller's images recorded a command, made a pipeline or wrote no descriptor");
    const Frame first = record_in_place(recorder, f);
    require(first.gate == 0 && first.seed == 0 && lines(first.commands, kSettle) == 1,
            ("the first frame in image mode has " + history(first) + ", or does not settle the images once").c_str());
    recorder.submitted();
    require(record_in_place(recorder, f).gate == 1, "the second frame in image mode does not follow the first");
    recorder.submitted();
    const vulkan::FrameImages next = caller_images(++generation);
    frame = {};
    written = descriptors_written;
    require(recorder.shape(f, &next).value_or(false) && frame.commands.empty() &&
                pipelines_made.size() == pipelines && descriptors_written > written,
            "a bind of new images recorded a command, made a pipeline or wrote no descriptor");
    const Frame third = record_in_place(recorder, f);
    require(third.gate == 1 && third.seed == 2 && lines(third.commands, kSettle) == 0,
            ("the frame after a bind of new images has " + history(third) + ", or settles the images again").c_str());
    require(naming(third, views[id(next.frame)]) && naming(third, id(next.answer)) &&
                !naming(third, views[id(images.frame)]) && !naming(third, id(images.answer)),
            "the frame after a bind of new images does not bind them, or binds the old ones");
    written = descriptors_written;
    require(!recorder.shape(f, &next).value_or(true) && descriptors_written == written,
            "images of the generation bound were bound again");

    const struct {
        unsigned passes;
        bool fp16, motion;
        float sharpness;
    } walk[] = {{1, false, false, 0}, {1, false, false, 0.5f}, {2, false, false, 0.5f}, {2, false, true, 0.5f},
                {2, true, true, 0.5f}, {1, true, false, 0},    {1, false, true, 0},     {3, false, true, 0},
                {3, true, true, 0.5f}, {1, true, true, 0.5f},  {1, true, false, 0},     {1, false, false, 0},
                {2, false, false, 0},  {1, false, false, 0}};
    for (const auto& step : walk) {
        f.passes = step.passes;
        f.fp16 = step.fp16;
        f.motion = step.motion;
        f.sharpness = step.sharpness;
        const std::string what = std::to_string(f.passes) + " passes, " + (f.fp16 ? "FP16" : "RGBA8") +
                                 (f.motion ? ", motion" : "") + (f.sharpness != 0 ? ", pass stages" : "");
        const vulkan::FrameImages own = caller_images(++generation);
        frame = {};
        pipelines = pipelines_made.size();
        require(recorder.shape(f, &own).value_or(false) && frame.commands.empty() &&
                    pipelines_made.size() == pipelines,
                ("the reshape in image mode for " + what + " recorded a command or made a pipeline").c_str());
        const Frame one = record_in_place(recorder, f);
        const std::string frame_name = as_named(one, views[id(own.frame)]);
        const std::string answer_name = as_named(one, id(own.answer));
        const bool direct = f.passes == 1 && f.sharpness == 0;
        const std::string format =
            "format " + std::to_string(f.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM) + " ";
        const std::string_view head = std::string_view(one.commands).substr(0, one.commands.find("\ndispatch "));
        require(!lines(one.commands, "copy buffer") && !lines(one.commands, "copy image to buffer") &&
                    one.commands.find(format) == std::string::npos && head.find("\ncopy") == std::string::npos &&
                    head.find("\nblit") == std::string::npos,
                ("a frame of " + what + " in image mode copies a buffer, makes an image in the frame's format or "
                 "transfers before the network")
                    .c_str());
        require(direct ? !lines(one.commands, "copy image") && !lines(one.commands, "blit") &&
                             naming(one, id(own.answer))
                       : lines(one.commands, "blit") == 1,
                ("a frame of " + what + " in image mode does not store into the caller's answer without a copy, or "
                 "blits other than once")
                    .c_str());
        // The sets that sample the caller's frame: the first pass's pre and post blocks, each pass's
        // stage, the alpha pass after later passes, the fallback, and with motion the finest luma
        // level of each of the four; later passes' blocks sample the input instead.
        const size_t samplers = 3 + (f.sharpness != 0 ? f.passes : 0) + (f.passes > 1) + (f.motion ? 4 : 0);
        require(naming(one, views[id(own.frame)]) == samplers,
                ("a frame of " + what + " in image mode samples the caller's frame in "
                 + std::to_string(naming(one, views[id(own.frame)])) + " sets, not " + std::to_string(samplers))
                    .c_str());
        require(judged(one.commands, f.width, f.height, plan->timeouts, frame_name, answer_name),
                ("a frame of " + what + " in image mode does not judge its waits after the network, or answers "
                 "with the caller's frame otherwise than into the caller's answer")
                    .c_str());
        const std::string after = frames_in_place(recorder, f);
        dlsslop::NetworkRecorder built(fake_device(), paths, true);
        require(built.shape(f, &own).value_or(false), ("cannot build the network in image mode for " + what).c_str());
        record_in_place(built, f);
        require(after == frames_in_place(built, f),
                ("the frames after the reshape in image mode for " + what + " differ from those of a build").c_str());
    }
    require(null_views == nulls, "image mode wrote a descriptor without an image");
}

// The verdict of the network built last, the fallback's grid, in the fake
// device's memory that its runtime keeps mapped, the only memory a built
// network keeps mapped.
std::vector<uint8_t>& verdict()
{
    for (auto m = host.rbegin(); m != host.rend(); ++m)
        if (m->second.size() == sizeof(VkDispatchIndirectCommand)) return m->second;
    require(false, "the network keeps no verdict mapped");
    std::abort();
}

// The lines the network logs that say a wait ran out.
int timeout_lines = 0;
void count_timeouts(const char* line)
{
    timeout_lines += std::string_view(line).find("ran out") != std::string_view::npos;
}

// After a frame whose wait ran out, as a verdict poked into the fake device's
// memory says, the next frame starts the network over: it zeroes the arena
// from the end of the values on before any dispatch, between barriers from
// the compute work before it and into the compute work after it, and drops
// the motion history. So does the frame after it, when that frame was not submitted. A
// line says so, but not again for the frame after it. Only a frame that
// starts over zeroes the arena.
void check_timeouts(const dlsslop::VulkanPaths& paths)
{
    const auto plan = vulkan::plan(64, 64);
    require(bool(plan), "cannot plan 64x64 frames");
    dlsslop::VulkanFrame f;
    f.width = f.height = 64;
    f.motion = true;
    vulkan::Device device = fake_device();
    device.log = count_timeouts;
    dlsslop::NetworkRecorder recorder(device, paths);
    require(recorder.shape(f).value_or(false), "cannot build the network for 64x64 frames");
    std::vector<uint8_t>& grid = verdict();
    const std::string zeroed = "(buffer " + std::to_string(plan->arena_bytes) + ") " +
                               std::to_string(plan->values_end) + " " + std::to_string(VK_WHOLE_SIZE);
    // Compute (2048) to transfer (4096) stages, shader writes (64) to transfer writes (4096), and
    // back to shader reads and writes (96).
    const std::string_view before = "barrier 2048 4096 1\n memory 64 4096\n",
                           after = "\nbarrier 4096 2048 1\n memory 4096 96\n";
    const auto zeroes = [&](const Frame& frame) {
        const std::string_view c = frame.commands;
        const size_t fill = c.find("\nfill #") + 1, end = c.find('\n', fill);
        return lines(c, "fill #", zeroed) == 1 && fill < c.find("\ndispatch ") && fill >= before.size() &&
               c.substr(fill - before.size(), before.size()) == before && c.substr(end, after.size()) == after;
    };
    // The first frame starts over, and zeroes what the build zeroed.
    require(zeroes(record(recorder, f)),
            "the first frame does not zero the arena past its values first, between barriers");
    recorder.submitted();
    const Frame second = record(recorder, f);
    require(!recorder.timed_out() && second.gate == 1 && !lines(second.commands, "fill"),
            ("the second frame has " + history(second) + " or zeroes the arena").c_str());
    // The second frame's wait ran out: its verdict gives the fallback workgroups.
    recorder.submitted();
    const uint32_t groups = 8;
    std::memcpy(grid.data(), &groups, sizeof groups);
    require(recorder.timed_out(), "a verdict that gives the fallback workgroups does not say a wait ran out");
    const Frame over = record(recorder, f);
    require(over.gate == 0 && over.seed == 0 && zeroes(over),
            ("the frame after one whose wait ran out has " + history(over) +
             ", or does not zero the arena first, between barriers")
                .c_str());
    const Frame again = record(recorder, f);
    require(again.commands == over.commands,
            "the frame after one that started over and was not submitted does not start over too");
    require(timeout_lines == 1, "a wait that ran out is not logged once");
    // Submitted, and none of its waits ran out.
    recorder.submitted();
    std::memset(grid.data(), 0, sizeof groups);
    const Frame next = record(recorder, f);
    require(!recorder.timed_out() && next.gate == 1 && next.seed == 1 && !lines(next.commands, "fill"),
            ("the frame after one that started over has " + history(next) + " or zeroes the arena").c_str());
}
} // namespace

int main(int argc, char** argv)
{
    std::string extent;
    bool unclamped = false;
    const option options[] = {{"build", required_argument, nullptr, 'b'},
                              {"unclamped", no_argument, nullptr, 'u'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+b:uh", options, nullptr)) != -1;) {
        if (code == 'b') extent = optarg;
        else if (code == 'u') unclamped = true;
        else if (code == 'h') {
            std::puts("Usage: network-recorder-test [OPTION]... SPIRV-DIRECTORY\n"
                      "Checks the Vulkan network's motion history and reshapes on a fake device, from a synthetic\n"
                      "model and the network's SPIR-V in SPIRV-DIRECTORY (required). No GPU or model needed.\n"
                      " -b, --build WxH   Instead, build the network for WxH frames, which makes every pipeline of\n"
                      "                   the runtime's own but one of the temporal pre blocks (default: unset)\n"
                      " -u, --unclamped   Build from a synthetic model whose position biases and head scales are\n"
                      "                   zero, which frees every Swin layer of the exponent's upper clamp\n"
                      "                   (default: off)\n"
                      " -h, --help        Show help (default: off)");
            return 0;
        } else
            return 2;
    }
    unsigned width = 64, height = 64;
    char end;
    if (optind + 1 != argc || (!extent.empty() && std::sscanf(extent.c_str(), "%ux%u%c", &width, &height, &end) != 2))
        return 2;
    const auto plan = vulkan::plan(width, height);
    require(bool(plan), "cannot plan the frames");
    vulkan_test::Pack model;
    require(vulkan_test::synthetic_model(*plan, model, unclamped), "cannot make the synthetic model");
    const dlsslop::VulkanPaths paths{model.path, argv[optind], ""};
    if (!extent.empty()) {
        dlsslop::VulkanFrame f;
        f.width = width;
        f.height = height;
        dlsslop::NetworkRecorder recorder(fake_device(), paths);
        const auto built = recorder.shape(f);
        require(built && *built, built ? "the network was not built" : built.error().what.c_str());
        return 0;
    }
    for (const unsigned passes : {1u, 2u}) check(paths, passes);
    check_reshapes(paths);
    check_unclamped(argv[optind]);
    check_timeouts(paths);
    check_images(paths);
    std::printf("network-recorder test: frames not submitted leave the motion history as it was, a reshaped "
                "network records the frames of a built one, weights free of the upper clamp run the kernels "
                "without it, a frame whose wait ran out answers with its input and starts the next over, and "
                "frames in a caller's images copy no buffer and blit at most once\n");
    return 0;
}
