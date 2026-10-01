// SPDX-License-Identifier: MIT
// Host test of the network's motion history across frames and of its
// reshapes, without a GPU. NetworkRecorder builds the network for 64x64
// frames, from a synthetic model and the build's SPIR-V, on a fake device
// whose functions log the commands recorded, each handle named by its first
// use in the frame and described by what it was made as, and each descriptor
// set by what it holds. A frame that is recorded and not submitted must leave
// the history as it was: the next frame's commands, with its motion
// parameters and push constants, must equal those of the frame recorded
// before it. A network reshaped for another shape of its extent must upload
// nothing, make no pipeline that it has, and record the frames that a network
// built for that shape records. A frame of one pass without the pass stages
// must copy the proxy straight into the network's input and its answer
// straight out, and copy or blit no image. Takes the directory of the
// network's SPIR-V. With --build, it only builds the network for one extent
// with every pipeline of the runtime's own, so that tests/vulkan-files.py can
// see the files that a build opens.
#include "network_recorder.h"
#include "vulkan_pack.h"

#include <getopt.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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
                                                VkDependencyFlags, uint32_t memories, const VkMemoryBarrier*,
                                                uint32_t buffers, const VkBufferMemoryBarrier* buffer,
                                                uint32_t images, const VkImageMemoryBarrier* image)
{
    log("barrier", {}, {src, dst, memories});
    for (uint32_t i = 0; i < buffers; ++i) log(" buffer", {id(buffer[i].buffer)}, {buffer[i].srcAccessMask});
    for (uint32_t i = 0; i < images; ++i)
        log(" image", {id(image[i].image)}, {uint64_t(image[i].oldLayout), uint64_t(image[i].newLayout)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer, VkBuffer from, VkBuffer to, uint32_t,
                                           const VkBufferCopy*)
{
    log("copy buffer", {id(from), id(to)});
}
VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer, VkBuffer buffer, VkDeviceSize, VkDeviceSize, uint32_t)
{
    log("fill", {id(buffer)});
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
// The physical device: one queue family for everything, FP8 cooperative
// matrices, every format feature and storage buffers of 4 GiB. The runtime
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
VKAPI_ATTR VkResult VKAPI_CALL matrix_properties(VkPhysicalDevice, uint32_t* count,
                                                 VkCooperativeMatrixPropertiesKHR* matrices)
{
    if (matrices && *count) {
        VkCooperativeMatrixPropertiesKHR& m = matrices[0];
        m.MSize = m.NSize = m.KSize = 16;
        m.AType = m.BType = VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT;
        m.CType = m.ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR;
        m.saturatingAccumulation = VK_FALSE;
        m.scope = VK_SCOPE_SUBGROUP_KHR;
    }
    *count = 1;
    return VK_SUCCESS;
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
    d.functions = {queue_families, properties, format_properties, matrix_properties};
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

// A frame's gate and seed as a message.
std::string history(const Frame& f)
{
    return "gate " + std::to_string(f.gate) + ", seed " + std::to_string(f.seed);
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
    const Frame again = record(recorder, f);
    require(again.commands == first.commands,
            ("a frame after one that was not submitted differs from that one: " + history(again)).c_str());
    // Submitted, the first frame starts the history, which the next frame reads.
    recorder.submitted();
    const Frame second = record(recorder, f);
    require(second.gate == 1 && second.seed == 1, ("the second frame has " + history(second)).c_str());
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
    std::set<std::string> kept; // what each pipeline the recorder made was made as
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
        // arena and the pipelines, and makes none that the recorder has.
        bool remade = false;
        for (size_t i = pipelines; i < pipelines_made.size(); ++i) remade |= !kept.insert(pipelines_made[i]).second;
        require(first || (frame.commands.find("copy buffer") == std::string::npos &&
                          frame.commands.find("fill") == std::string::npos && !remade),
                ("the reshape for " + what + " uploaded the weights or made a pipeline it had").c_str());
        const std::string after = frames_of(recorder, f);
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
} // namespace

int main(int argc, char** argv)
{
    std::string extent;
    const option options[] = {{"build", required_argument, nullptr, 'b'}, {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+b:h", options, nullptr)) != -1;) {
        if (code == 'b') extent = optarg;
        else if (code == 'h') {
            std::puts("Usage: network-recorder-test [OPTION]... SPIRV-DIRECTORY\n"
                      "Checks the Vulkan network's motion history and reshapes on a fake device, from a synthetic\n"
                      "model and the network's SPIR-V in SPIRV-DIRECTORY (required). No GPU or model needed.\n"
                      " -b, --build WxH   Instead, build the network for WxH frames of 2 passes with motion history\n"
                      "                   and pass stages, and so every pipeline of the runtime's own\n"
                      "                   (default: unset)\n"
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
    require(vulkan_test::synthetic_model(*plan, model), "cannot make the synthetic model");
    const dlsslop::VulkanPaths paths{model.path, argv[optind], ""};
    if (!extent.empty()) {
        dlsslop::VulkanFrame f;
        f.width = width;
        f.height = height;
        f.passes = 2;
        f.motion = true;
        f.sharpness = 0.5f;
        dlsslop::NetworkRecorder recorder(fake_device(), paths);
        const auto built = recorder.shape(f);
        require(built && *built, built ? "the network was not built" : built.error().what.c_str());
        return 0;
    }
    for (const unsigned passes : {1u, 2u}) check(paths, passes);
    check_reshapes(paths);
    std::printf("network-recorder test: frames not submitted leave the motion history as it was, and a "
                "reshaped network records the frames of a built one\n");
    return 0;
}
