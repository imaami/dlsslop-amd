// The network on Vulkan: DLSSNR-AMD's runtime (vulkan-nr/) on a device of the daemon's own.
// SPDX-License-Identifier: MIT
#include "vulkan_network.h"
#include "network_requirements.h"

#include "nr_log.hpp"
#include "nr_vendor.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>
#include <vulkan/vulkan.h>

namespace dlsslop {
namespace {

void log_line(const char* line) { std::fprintf(stderr, "%s\n", line); }

// What stops the network running on a device, or empty.
std::string unsuitable(VkPhysicalDevice physical, uint32_t& family)
{
    if (const char* missing = NetworkUnsupported(physical, vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceFeatures2,
                                                 vkEnumerateDeviceExtensionProperties))
        return std::string(missing) + " unavailable";
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
    if (!std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& e) {
            return !std::strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        }))
        return VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME " unavailable";
    // Graphics too: the runtime converts colour formats with blits.
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
    const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (family = 0; family < count && (queues[family].queueFlags & need) != need; ++family) {}
    return family < count ? std::string() : "no graphics and compute queue";
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

}  // namespace

struct VulkanNetwork::Impl {
    VulkanPaths paths;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    unsigned index = 0;
    std::string name;
    VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceMemoryProperties memory{};
    float timestamp_ns = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    Buffer upload, download;  // host transport
    std::optional<NetworkRecorder> recorder;
    // Device-local frames the layer exported, a proxy/answer pair per import slot.
    struct Imported {
        size_t bytes = 0;
        Buffer frame[2];  // proxy, answer
    };
    std::array<Imported, kImportSlots> imported{};

    Result<uint32_t> memory_type(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want) return i;
        return fail("no suitable Vulkan memory type");
    }
    void drop(Buffer& b)
    {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    // A host-visible transfer buffer of at least bytes.
    Result<void> host_buffer(Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags extra)
    {
        if (b.size >= bytes) return {};
        drop(b);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        DLSSLOP_TRY(vk_check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "create transfer buffer"));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = DLSSLOP_TRY(
            memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | extra));
        DLSSLOP_TRY(vk_check(vkAllocateMemory(device, &alloc, nullptr, &b.memory), "allocate transfer buffer"));
        DLSSLOP_TRY(vk_check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "bind transfer buffer"));
        DLSSLOP_TRY(vk_check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "map transfer buffer"));
        b.size = bytes;
        return {};
    }
    void release(Imported& slot)
    {
        for (Buffer& b : slot.frame) drop(b);
        slot = {};
    }
    ~Impl()
    {
        if (device) {
            vkDeviceWaitIdle(device);
            recorder.reset();
            for (auto& slot : imported) release(slot);
            drop(upload);
            drop(download);
            if (queries) vkDestroyQueryPool(device, queries, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

Result<VulkanNetwork> VulkanNetwork::create(const VulkanPaths& paths, int device)
{
    VulkanNetwork network(std::make_unique<Impl>());
    auto& s = *network.impl_;
    s.paths = paths;
    nr::set_log_sink(log_line);
    // The daemon has no overlay to show: keep implicit layers (Steam's, MangoHud's) out
    // of its instance, unless the environment already says otherwise.
    setenv("VK_LOADER_LAYERS_DISABLE", "~implicit~", 0);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "dlsslopd";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    DLSSLOP_TRY(vk_check(vkCreateInstance(&info, nullptr, &s.instance), "create Vulkan instance"));
    uint32_t count = 0;
    DLSSLOP_TRY(vk_check(vkEnumeratePhysicalDevices(s.instance, &count, nullptr), "enumerate Vulkan devices"));
    std::vector<VkPhysicalDevice> devices(count);
    DLSSLOP_TRY(vk_check(vkEnumeratePhysicalDevices(s.instance, &count, devices.data()), "enumerate Vulkan devices"));
    std::string reasons;
    for (uint32_t i = 0; i < count && !s.physical; ++i) {
        if (device >= 0 && i != unsigned(device)) continue;
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        const std::string why = unsuitable(devices[i], s.family);
        if (why.empty()) {
            s.physical = devices[i];
            s.index = i;
            s.name = p.deviceName;
        } else {
            reasons += "\n  Vulkan device " + std::to_string(i) + " (" + p.deviceName + "): " + why;
        }
    }
    if (!s.physical)
        return fail(device >= 0 && unsigned(device) >= count ? "no Vulkan device " + std::to_string(device)
                                                             : "no Vulkan device can run the network:" + reasons);
    NetworkFeatureChain enable;
    std::vector<const char*> extensions = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    for (const auto& f : kNetworkFeatures) {
        enable.bit(f) = VK_TRUE;
        if (f.extension && std::find_if(extensions.begin(), extensions.end(), [&f](const char* e) {
                               return !std::strcmp(e, f.extension);
                           }) == extensions.end())
            extensions.push_back(f.extension);
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = s.family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.pNext = &enable.features2();
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue;
    create.enabledExtensionCount = uint32_t(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    DLSSLOP_TRY(vk_check(vkCreateDevice(s.physical, &create, nullptr, &s.device), "create Vulkan device"));
    vkGetDeviceQueue(s.device, s.family, 0, &s.queue);
    vkGetPhysicalDeviceMemoryProperties(s.physical, &s.memory);
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &s.ids};
    vkGetPhysicalDeviceProperties2(s.physical, &p);
    s.timestamp_ns = p.properties.limits.timestampPeriod;
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = s.family;
    DLSSLOP_TRY(vk_check(vkCreateCommandPool(s.device, &pool, nullptr, &s.pool), "create command pool"));
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = s.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    DLSSLOP_TRY(vk_check(vkAllocateCommandBuffers(s.device, &alloc, &s.cmd), "allocate command buffer"));
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    DLSSLOP_TRY(vk_check(vkCreateFence(s.device, &fence, nullptr, &s.fence), "create fence"));
    VkQueryPoolCreateInfo queries{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queries.queryCount = 4;
    DLSSLOP_TRY(vk_check(vkCreateQueryPool(s.device, &queries, nullptr, &s.queries), "create timestamp queries"));
    nr::HostDevice host;
    host.instance = s.instance;
    host.physical = s.physical;
    host.device = s.device;
    host.queue = s.queue;
    host.queue_family = s.family;
    s.recorder.emplace(host, s.memory, s.paths);
    return network;
}

VulkanNetwork::VulkanNetwork(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VulkanNetwork::VulkanNetwork(VulkanNetwork&&) noexcept = default;
VulkanNetwork::~VulkanNetwork() = default;

const std::string& VulkanNetwork::device_name() const { return impl_->name; }
unsigned VulkanNetwork::device_index() const { return impl_->index; }

bool VulkanNetwork::shape_differs(const VulkanFrame& frame) const { return impl_->recorder->shape_differs(frame); }

Result<bool> VulkanNetwork::shape(const VulkanFrame& frame)
{
    auto& s = *impl_;
    if (!s.recorder->shape_differs(frame)) return false;
    DLSSLOP_TRY(vk_check(vkDeviceWaitIdle(s.device), "wait for the device"));
    return s.recorder->shape(frame);
}

bool VulkanNetwork::import(unsigned slot, const ShmTransportOffer& offer, int fds[2])
{
    auto& s = *impl_;
    if (slot >= kImportSlots) return false;
    if (std::memcmp(offer.deviceUuid, s.ids.deviceUUID, VK_UUID_SIZE) ||
        std::memcmp(offer.driverUuid, s.ids.driverUUID, VK_UUID_SIZE)) {
        std::fprintf(stderr, "device-local transport: the game's frames are on another device or driver\n");
        return false;
    }
    Impl::Imported next;
    for (unsigned i = 0; i < 2; ++i) {
        Buffer& b = next.frame[i];
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        // The layer's buffer, repeated: a dedicated import must name an identical one.
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
        info.size = offer.size[i];
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkMemoryRequirements req{};
        uint32_t type = s.memory.memoryTypeCount;
        if (vkCreateBuffer(s.device, &info, nullptr, &b.buffer) == VK_SUCCESS) {
            vkGetBufferMemoryRequirements(s.device, b.buffer, &req);
            // The layer exports from the first device-local type the buffer allows; on the
            // same GPU and driver that is this device's too.
            type = s.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).value_or(type);
        }
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.buffer = b.buffer;
        VkImportMemoryFdInfoKHR fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &dedicated};
        fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        fd.fd = fds[i];
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fd};
        alloc.allocationSize = offer.allocation[i];
        alloc.memoryTypeIndex = type;
        if (!b.buffer || type == s.memory.memoryTypeCount || req.size != offer.allocation[i] ||
            vkAllocateMemory(s.device, &alloc, nullptr, &b.memory) != VK_SUCCESS) {
            s.release(next);
            return false;
        }
        fds[i] = -1;  // Vulkan owns the descriptor it imported.
        if (vkBindBufferMemory(s.device, b.buffer, b.memory, 0) != VK_SUCCESS) {
            s.release(next);
            return false;
        }
        b.size = offer.size[i];
    }
    // An earlier frame may still be reading the slot's old import. A queue that
    // cannot go idle leaves the old pair where it is.
    if (vkQueueWaitIdle(s.queue) != VK_SUCCESS) {
        s.release(next);
        return false;
    }
    s.release(s.imported[slot]);
    s.imported[slot] = next;
    return true;
}

Result<void> VulkanNetwork::infer(const VulkanFrame& frame, int slot, const uint8_t* input, uint8_t* output)
{
    auto& s = *impl_;
    DLSSLOP_TRY(shape(frame));
    const VkDeviceSize bytes = VkDeviceSize(frame.width) * frame.height * (frame.fp16 ? 8 : 4);
    const bool exported = slot >= 0;  // the layer's device-local pair
    VkBuffer source = VK_NULL_HANDLE, target = VK_NULL_HANDLE;
    if (exported) {
        if (size_t(slot) >= s.imported.size() || !s.imported[size_t(slot)].frame[0].buffer)
            return fail("import slot " + std::to_string(slot) + " holds no frames");
        source = s.imported[size_t(slot)].frame[0].buffer;
        target = s.imported[size_t(slot)].frame[1].buffer;
    } else {
        DLSSLOP_TRY(s.host_buffer(s.upload, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0));
        DLSSLOP_TRY(s.host_buffer(s.download, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT));
        std::memcpy(s.upload.mapped, input, bytes);
        source = s.upload.buffer;
        target = s.download.buffer;
    }
    VkCommandBuffer cmd = s.cmd;
    DLSSLOP_TRY(vk_check(vkResetCommandBuffer(cmd, 0), "reset command buffer"));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    DLSSLOP_TRY(vk_check(vkBeginCommandBuffer(cmd, &begin), "begin command buffer"));
    vkCmdResetQueryPool(cmd, s.queries, 0, 4);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s.queries, 0);

    // The layer's buffers change hands at every frame: both are acquired from the layer before
    // the copies, the proxy read and the answer written, and released back to it after them.
    auto own = [&](bool acquire) {
        const VkBuffer buffers[2] = {source, target};
        const VkAccessFlags2 access[2] = {VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
        VkBufferMemoryBarrier2 b[2]{};
        for (unsigned i = 0; i < 2; ++i) {
            b[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            if (acquire) {
                b[i].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                b[i].dstAccessMask = access[i];
                b[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
                b[i].dstQueueFamilyIndex = s.family;
            } else {
                b[i].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                b[i].srcAccessMask = access[i];
                b[i].srcQueueFamilyIndex = s.family;
                b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            }
            b[i].buffer = buffers[i];
            b[i].size = VK_WHOLE_SIZE;
        }
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = 2;
        dependency.pBufferMemoryBarriers = b;
        vkCmdPipelineBarrier2(cmd, &dependency);
    };
    if (exported) own(true);
    DLSSLOP_TRY(s.recorder->record(cmd, source, target, frame, s.queries, 1));
    if (exported) {
        own(false);
    } else {
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = target;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b, 0,
                             nullptr);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.queries, 3);
    DLSSLOP_TRY(vk_check(vkEndCommandBuffer(cmd), "end command buffer"));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    DLSSLOP_TRY(vk_check(vkResetFences(s.device, 1, &s.fence), "reset fence"));
    DLSSLOP_TRY(vk_check(vkQueueSubmit(s.queue, 1, &submit, s.fence), "submit frame"));
    // A healthy frame takes milliseconds; ten seconds means the device is gone.
    DLSSLOP_TRY(vk_check(vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, 10'000'000'000ull), "wait for the frame"));
    uint64_t stamps[4] = {};
    if (vkGetQueryPoolResults(s.device, s.queries, 0, 4, sizeof stamps, stamps, sizeof stamps[0],
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        const auto ms = [&](int a, int b) { return float(double(stamps[b] - stamps[a]) * s.timestamp_ns / 1e6); };
        upload_ms = ms(0, 1);
        inference_ms = ms(1, 2);
        readback_ms = ms(2, 3);
    }
    if (!exported) std::memcpy(output, s.download.mapped, bytes);
    return {};
}

}  // namespace dlsslop
