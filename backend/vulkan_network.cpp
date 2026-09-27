// The network on Vulkan: DLSSNR-AMD's runtime (vulkan-nr/) on a device of the daemon's own.
// SPDX-License-Identifier: MIT
#include "vulkan_network.h"

#include "nr_log.hpp"
#include "nr_runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>
#include <vulkan/vulkan.h>

namespace dlsslop {
namespace {

void check(VkResult result, const char* what)
{
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(int(result)) + ")");
}

void log_line(const char* line) { std::fprintf(stderr, "%s\n", line); }

// Everything the network's shaders use (vulkan-nr/src/core/nrvk.hpp, Context::create),
// chained for one query or one vkCreateDevice.
struct Features {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR layout{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 all{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    Features(bool with_layout)
    {
        all.pNext = &coop;
        coop.pNext = &fp8;
        fp8.pNext = &f11;
        f11.pNext = &f12;
        f12.pNext = &f13;
        f13.pNext = with_layout ? &layout : nullptr;
    }
    Features(const Features&) = delete;
};

bool has_extension(const std::vector<VkExtensionProperties>& list, const char* name)
{
    return std::any_of(list.begin(), list.end(), [name](const VkExtensionProperties& e) { return !std::strcmp(e.extensionName, name); });
}

// What stops the network running on a device, or empty.
std::string unsuitable(VkPhysicalDevice physical, bool& layout, uint32_t& family)
{
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
    vkGetPhysicalDeviceProperties2(physical, &properties);
    if (properties.properties.apiVersion < VK_API_VERSION_1_3) return "Vulkan 1.3 unavailable";
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
    for (const char* name : {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
                             VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME})
        if (!has_extension(extensions, name)) return std::string(name) + " unavailable";
    layout = has_extension(extensions, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    Features f(layout);
    vkGetPhysicalDeviceFeatures2(physical, &f.all);
    layout = layout && f.layout.workgroupMemoryExplicitLayout && f.layout.workgroupMemoryExplicitLayout8BitAccess &&
             f.layout.workgroupMemoryExplicitLayout16BitAccess;
    const std::pair<VkBool32, const char*> required[] = {
        {f.coop.cooperativeMatrix, "cooperativeMatrix"}, {f.fp8.shaderFloat8, "shaderFloat8"},
        {f.fp8.shaderFloat8CooperativeMatrix, "shaderFloat8CooperativeMatrix"},
        {f.f11.storageBuffer16BitAccess, "storageBuffer16BitAccess"},
        {f.f12.storageBuffer8BitAccess, "storageBuffer8BitAccess"}, {f.f12.shaderFloat16, "shaderFloat16"},
        {f.f12.shaderInt8, "shaderInt8"}, {f.f12.vulkanMemoryModel, "vulkanMemoryModel"},
        {f.f13.subgroupSizeControl, "subgroupSizeControl"}, {f.f13.synchronization2, "synchronization2"},
    };
    for (const auto& [have, name] : required)
        if (!have) return std::string(name) + " unavailable";
    // The cooperative-matrix fragments are laid out for 32-lane subgroups.
    if (subgroup.minSubgroupSize > 32 || subgroup.maxSubgroupSize < 32 ||
        !(subgroup.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
        return "32-lane compute subgroups unavailable";
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

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

// What a built runtime serves; a request for anything else rebuilds it.
struct Shape {
    unsigned width = 0, height = 0;
    bool fp16 = false, motion = false;
    unsigned passes = 0;  // the most it can run
    auto tie() const { return std::tie(width, height, fp16, motion); }
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
    VkPhysicalDeviceMemoryProperties memory{};
    float timestamp_ns = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    Buffer upload, download;  // host transport
    Image colour, half;       // the network's frame; with FP16 requests, the FP16 staging image
    Shape shape;
    std::unique_ptr<nr::Runtime> runtime;
    std::optional<VulkanFrame> last;  // the history's frame; none since a build
    // Device-local frames the layer exported, one per producer generation, the oldest replaced first.
    struct Imported {
        uint32_t generation = 0;
        size_t bytes = 0;
        Buffer frame[2];  // proxy, answer
    };
    std::array<Imported, 4> imported{};
    unsigned next_import = 0;

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("no suitable Vulkan memory type");
    }
    void drop(Buffer& b)
    {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    void drop(Image& i)
    {
        if (i.image) vkDestroyImage(device, i.image, nullptr);
        if (i.memory) vkFreeMemory(device, i.memory, nullptr);
        i = {};
    }
    // A host-visible transfer buffer of at least bytes.
    void host_buffer(Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags extra)
    {
        if (b.size >= bytes) return;
        drop(b);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "create transfer buffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = memory_type(req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | extra);
        check(vkAllocateMemory(device, &alloc, nullptr, &b.memory), "allocate transfer buffer");
        check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "bind transfer buffer");
        check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "map transfer buffer");
        b.size = bytes;
    }
    void image(Image& i, VkFormat format, unsigned w, unsigned h, VkImageUsageFlags usage)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {w, h, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        check(vkCreateImage(device, &info, nullptr, &i.image), "create frame image");
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, i.image, &req);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device, &alloc, nullptr, &i.memory), "allocate frame image");
        check(vkBindImageMemory(device, i.image, i.memory, 0), "bind frame image");
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
            runtime.reset();
            for (auto& slot : imported) release(slot);
            drop(upload);
            drop(download);
            drop(colour);
            drop(half);
            if (queries) vkDestroyQueryPool(device, queries, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

VulkanNetwork::VulkanNetwork(const VulkanPaths& paths, int device) : impl_(std::make_unique<Impl>())
{
    auto& s = *impl_;
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
    check(vkCreateInstance(&info, nullptr, &s.instance), "create Vulkan instance");
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(s.instance, &count, nullptr), "enumerate Vulkan devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(s.instance, &count, devices.data()), "enumerate Vulkan devices");
    bool layout = false;
    std::string reasons;
    for (uint32_t i = 0; i < count && !s.physical; ++i) {
        if (device >= 0 && i != unsigned(device)) continue;
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        const std::string why = unsuitable(devices[i], layout, s.family);
        if (why.empty()) {
            s.physical = devices[i];
            s.index = i;
            s.name = p.deviceName;
        } else {
            reasons += "\n  Vulkan device " + std::to_string(i) + " (" + p.deviceName + "): " + why;
        }
    }
    if (!s.physical)
        throw std::runtime_error(device >= 0 && unsigned(device) >= count ? "no Vulkan device " + std::to_string(device)
                                                                          : "no Vulkan device can run the network:" + reasons);
    Features enable(layout);
    enable.coop.cooperativeMatrix = VK_TRUE;
    enable.fp8.shaderFloat8 = enable.fp8.shaderFloat8CooperativeMatrix = VK_TRUE;
    enable.f11.storageBuffer16BitAccess = VK_TRUE;
    enable.f12.storageBuffer8BitAccess = enable.f12.shaderFloat16 = enable.f12.shaderInt8 = VK_TRUE;
    enable.f12.vulkanMemoryModel = VK_TRUE;
    enable.f13.subgroupSizeControl = enable.f13.synchronization2 = VK_TRUE;
    enable.layout.workgroupMemoryExplicitLayout = enable.layout.workgroupMemoryExplicitLayout8BitAccess =
        enable.layout.workgroupMemoryExplicitLayout16BitAccess = layout;
    std::vector<const char*> extensions = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
                                           VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    if (layout) extensions.push_back(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = s.family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.pNext = &enable.all;
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue;
    create.enabledExtensionCount = uint32_t(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(s.physical, &create, nullptr, &s.device), "create Vulkan device");
    vkGetDeviceQueue(s.device, s.family, 0, &s.queue);
    vkGetPhysicalDeviceMemoryProperties(s.physical, &s.memory);
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(s.physical, &p);
    s.timestamp_ns = p.limits.timestampPeriod;
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = s.family;
    check(vkCreateCommandPool(s.device, &pool, nullptr, &s.pool), "create command pool");
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = s.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(s.device, &alloc, &s.cmd), "allocate command buffer");
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(s.device, &fence, nullptr, &s.fence), "create fence");
    VkQueryPoolCreateInfo queries{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queries.queryCount = 4;
    check(vkCreateQueryPool(s.device, &queries, nullptr, &s.queries), "create timestamp queries");
}

VulkanNetwork::~VulkanNetwork() = default;

const std::string& VulkanNetwork::device_name() const { return impl_->name; }
unsigned VulkanNetwork::device_index() const { return impl_->index; }

bool VulkanNetwork::shape_differs(const VulkanFrame& frame) const
{
    const auto& s = *impl_;
    const Shape want{frame.width, frame.height, frame.fp16, frame.motion};
    return !s.runtime || s.shape.tie() != want.tie() || s.shape.passes < std::clamp(frame.passes, 1u, kMaxPasses);
}

bool VulkanNetwork::shape(const VulkanFrame& frame)
{
    auto& s = *impl_;
    if (!shape_differs(frame)) return false;
    const unsigned passes = std::clamp(frame.passes, 1u, kMaxPasses);
    const Shape want{frame.width, frame.height, frame.fp16, frame.motion, passes};
    check(vkDeviceWaitIdle(s.device), "wait for the device");
    s.runtime.reset();
    s.drop(s.colour);
    s.drop(s.half);
    s.shape = {};
    // The network takes 8-bit frames as they are and every other one as RGBA32F.
    const VkFormat format = frame.fp16 ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    const VkImageUsageFlags transfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    s.image(s.colour, format, frame.width, frame.height,
            transfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    if (frame.fp16) s.image(s.half, VK_FORMAT_R16G16B16A16_SFLOAT, frame.width, frame.height, transfer);
    nr::HostDevice host;
    host.instance = s.instance;
    host.physical = s.physical;
    host.device = s.device;
    host.queue = s.queue;
    host.queue_family = s.family;
    nr::RuntimeConfig config;
    config.width = frame.width;
    config.height = frame.height;
    config.colour_format = format;
    // The network's own answer, as NVIDIA's DLL returns it; the layer composes it.
    config.native_compose = true;
    config.max_passes = passes;
    config.model_pack = s.paths.model;
    config.network_shaders = s.paths.shaders;
    config.pipeline_cache = s.paths.cache;
    nr::TemporalConfig temporal;
    temporal.enable = frame.motion;
    s.runtime = std::make_unique<nr::Runtime>(host, config, nr::ControlMaskConfig{}, temporal);
    s.shape = want;
    s.last.reset();
    return true;
}

bool VulkanNetwork::import(uint32_t generation, const uint64_t allocation[2], const uint64_t size[2], int fds[2])
{
    auto& s = *impl_;
    if (!generation) return false;
    Impl::Imported next{generation, size_t(std::min(size[0], size[1]))};
    for (unsigned i = 0; i < 2; ++i) {
        Buffer& b = next.frame[i];
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        // The layer's buffer, repeated: a dedicated import must name an identical one.
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
        info.size = size[i];
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkMemoryRequirements req{};
        uint32_t type = s.memory.memoryTypeCount;
        if (vkCreateBuffer(s.device, &info, nullptr, &b.buffer) == VK_SUCCESS) {
            vkGetBufferMemoryRequirements(s.device, b.buffer, &req);
            // The layer exports from the first device-local type the buffer allows; on the
            // same GPU and driver that is this device's too.
            try {
                type = s.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            } catch (const std::exception&) {
            }
        }
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.buffer = b.buffer;
        VkImportMemoryFdInfoKHR fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &dedicated};
        fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        fd.fd = fds[i];
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fd};
        alloc.allocationSize = allocation[i];
        alloc.memoryTypeIndex = type;
        if (!b.buffer || type == s.memory.memoryTypeCount || req.size != allocation[i] ||
            vkAllocateMemory(s.device, &alloc, nullptr, &b.memory) != VK_SUCCESS) {
            s.release(next);
            return false;
        }
        fds[i] = -1;  // Vulkan owns the descriptor it imported.
        if (vkBindBufferMemory(s.device, b.buffer, b.memory, 0) != VK_SUCCESS) {
            s.release(next);
            return false;
        }
        b.size = size[i];
    }
    Impl::Imported* slot = nullptr;
    for (auto& imported : s.imported)
        if (imported.generation == generation) slot = &imported;
    if (!slot) slot = &s.imported[s.next_import++ % s.imported.size()];
    // An earlier frame may still be reading the slot's old import.
    check(vkQueueWaitIdle(s.queue), "wait for the queue");
    s.release(*slot);
    *slot = std::move(next);
    return true;
}

bool VulkanNetwork::holds(uint32_t generation, size_t bytes) const
{
    return generation && std::any_of(impl_->imported.begin(), impl_->imported.end(), [&](const Impl::Imported& i) {
               return i.generation == generation && i.bytes >= bytes;
           });
}

void VulkanNetwork::infer(const VulkanFrame& frame, uint32_t generation, const uint8_t* input, uint8_t* output)
{
    auto& s = *impl_;
    shape(frame);
    const VkDeviceSize bytes = VkDeviceSize(frame.width) * frame.height * (frame.fp16 ? 8 : 4);
    VkBuffer source = VK_NULL_HANDLE, target = VK_NULL_HANDLE;
    for (const auto& imported : s.imported)
        if (generation && imported.generation == generation && imported.bytes >= bytes) {
            source = imported.frame[0].buffer;
            target = imported.frame[1].buffer;
        }
    if (generation && !source) throw std::range_error("request names device-local frames this daemon has not imported");
    if (!generation) {
        s.host_buffer(s.upload, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0);
        s.host_buffer(s.download, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        std::memcpy(s.upload.mapped, input, bytes);
        source = s.upload.buffer;
        target = s.download.buffer;
    }
    VkCommandBuffer cmd = s.cmd;
    check(vkResetCommandBuffer(cmd, 0), "reset command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &begin), "begin command buffer");
    vkCmdResetQueryPool(cmd, s.queries, 0, 4);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s.queries, 0);

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
    if (generation) own(true);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {frame.width, frame.height, 1};
    const VkImage staged = frame.fp16 ? s.half.image : s.colour.image;
    layout(staged, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdCopyBufferToImage(cmd, source, staged, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageBlit blit{};
    blit.srcSubresource = blit.dstSubresource = region.imageSubresource;
    blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(frame.width), int32_t(frame.height), 1};
    if (frame.fp16) {
        layout(s.half.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
        layout(s.colour.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdBlitImage(cmd, s.half.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.colour.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }
    layout(s.colour.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.queries, 1);

    nr::ColourFrame colour;
    colour.image = s.colour.image;
    colour.format = frame.fp16 ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    colour.width = frame.width;
    colour.height = frame.height;
    colour.before_stage = colour.after_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    colour.before_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    colour.after_access = VK_ACCESS_TRANSFER_READ_BIT;
    colour.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_STORAGE_BIT;
    nr::Controls controls;
    controls.passes = int(std::clamp(frame.passes, 1u, s.shape.passes));
    controls.intensity = frame.intensity;
    controls.local_tone = frame.local_tone;
    controls.local_structure = frame.local_structure;
    if (frame.motion) {
        const auto settings = [](const VulkanFrame& f) {
            return std::tie(f.passes, f.intensity, f.local_tone, f.local_structure);
        };
        nr::TemporalFrame temporal;
        temporal.reset = !s.last || settings(*s.last) != settings(frame);
        s.runtime->record_temporal(cmd, colour, controls, temporal);
    } else {
        s.runtime->record(cmd, colour, controls);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.queries, 2);

    VkImage answer = s.colour.image;
    VkImageLayout answer_layout = VK_IMAGE_LAYOUT_GENERAL;
    if (frame.fp16) {
        layout(s.half.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdBlitImage(cmd, s.colour.image, VK_IMAGE_LAYOUT_GENERAL, s.half.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_NEAREST);
        layout(s.half.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
        answer = s.half.image;
        answer_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    vkCmdCopyImageToBuffer(cmd, answer, answer_layout, target, 1, &region);
    if (generation) {
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
    check(vkEndCommandBuffer(cmd), "end command buffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    check(vkResetFences(s.device, 1, &s.fence), "reset fence");
    check(vkQueueSubmit(s.queue, 1, &submit, s.fence), "submit frame");
    // A healthy frame takes milliseconds; ten seconds means the device is gone.
    check(vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, 10'000'000'000ull), "wait for the frame");
    s.last = frame;
    uint64_t stamps[4] = {};
    if (vkGetQueryPoolResults(s.device, s.queries, 0, 4, sizeof stamps, stamps, sizeof stamps[0],
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        const auto ms = [&](int a, int b) { return float(double(stamps[b] - stamps[a]) * s.timestamp_ns / 1e6); };
        upload_ms = ms(0, 1);
        inference_ms = ms(1, 2);
        readback_ms = ms(2, 3);
    }
    if (!generation) std::memcpy(output, s.download.mapped, bytes);
}

}  // namespace dlsslop
