// Drive the layer's composition through in-place rebuilds and check that no
// dispatch binds a descriptor set written with an image view that has since
// been destroyed. Drivers recycle view handles, so a descriptor cache keyed on
// handle values can bind a set that still points at a freed image.
#include <vulkan/vulkan.h>
#include "composition.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void VkCheck(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(result));
}

// The device table's view and descriptor calls, wrapped to track every view's
// lifetime and what each descriptor set binding was last written with.
struct Tracker {
    PFN_vkCreateImageView createView = nullptr;
    PFN_vkDestroyImageView destroyView = nullptr;
    PFN_vkAllocateDescriptorSets allocate = nullptr;
    PFN_vkUpdateDescriptorSets update = nullptr;
    PFN_vkCmdBindDescriptorSets bind = nullptr;
    std::set<VkImageView> live;
    std::map<VkImageView, unsigned> generation;  // bumped whenever a handle value is created
    std::map<VkDescriptorSet, std::map<uint32_t, std::pair<VkImageView, unsigned>>> written;
    unsigned binds = 0, recycled = 0, stale = 0;
} tracker;

VKAPI_ATTR VkResult VKAPI_CALL CreateView(VkDevice device, const VkImageViewCreateInfo* info,
                                          const VkAllocationCallbacks* allocator, VkImageView* view) {
    const VkResult result = tracker.createView(device, info, allocator, view);
    if (result != VK_SUCCESS) return result;
    tracker.recycled += tracker.generation.count(*view) ? 1u : 0u;
    ++tracker.generation[*view];
    tracker.live.insert(*view);
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyView(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
    tracker.live.erase(view);
    tracker.destroyView(device, view, allocator);
}

// The scalers and the meter re-create their descriptor pools on every rebuild, so a new set can
// reuse a freed one's handle value. It starts with nothing written, whatever its predecessor held.
VKAPI_ATTR VkResult VKAPI_CALL Allocate(VkDevice device, const VkDescriptorSetAllocateInfo* info,
                                        VkDescriptorSet* sets) {
    const VkResult result = tracker.allocate(device, info, sets);
    if (result != VK_SUCCESS) return result;
    for (uint32_t i = 0; i < info->descriptorSetCount; ++i) tracker.written.erase(sets[i]);
    return result;
}

VKAPI_ATTR void VKAPI_CALL Update(VkDevice device, uint32_t count, const VkWriteDescriptorSet* writes,
                                  uint32_t copies, const VkCopyDescriptorSet* copy) {
    for (uint32_t i = 0; i < count; ++i) {
        const VkDescriptorImageInfo* image = writes[i].pImageInfo;
        if (image && image->imageView)
            tracker.written[writes[i].dstSet][writes[i].dstBinding] = {image->imageView,
                                                                       tracker.generation[image->imageView]};
    }
    tracker.update(device, count, writes, copies, copy);
}

VKAPI_ATTR void VKAPI_CALL Bind(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout,
                                uint32_t first, uint32_t count, const VkDescriptorSet* sets, uint32_t dynamicCount,
                                const uint32_t* dynamicOffsets) {
    for (uint32_t i = 0; i < count; ++i) {
        ++tracker.binds;
        for (const auto& [binding, view] : tracker.written[sets[i]]) {
            if (tracker.live.count(view.first) && tracker.generation[view.first] == view.second) continue;
            ++tracker.stale;
            std::printf("stale: binding %u of a bound set was written with view generation %u, now %u%s\n",
                        binding, view.second, tracker.generation[view.first],
                        tracker.live.count(view.first) ? "" : " (destroyed)");
        }
    }
    tracker.bind(cb, point, layout, first, count, sets, dynamicCount, dynamicOffsets);
}

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkImage swapchain = VK_NULL_HANDLE;
    VkDeviceMemory swapchainMemory = VK_NULL_HANDLE;
    dlssnr::InstanceTable instanceTable;
    dlssnr::DeviceTable deviceTable;

    ~Context() {
        if (device) vkDeviceWaitIdle(device);
        if (swapchain) vkDestroyImage(device, swapchain, nullptr);
        if (swapchainMemory) vkFreeMemory(device, swapchainMemory, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    bool Init() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "dlsslop-amd-composition-rebuild-test";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        const VkResult create = vkCreateInstance(&info, nullptr, &instance);
        if (create == VK_ERROR_INCOMPATIBLE_DRIVER) return false;
        VkCheck(create, "create instance");
        uint32_t count = 0;
        VkCheck(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate physical devices");
        std::vector<VkPhysicalDevice> devices(count);
        VkCheck(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "get physical devices");
        // Prefer software Vulkan; fall back to hardware only when no CPU device qualifies.
        std::stable_partition(devices.begin(), devices.end(), [](VkPhysicalDevice device) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(device, &properties);
            return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        });
        uint32_t family = 0;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceFeatures features{};
            vkGetPhysicalDeviceFeatures(candidate, &features);
            if (!features.shaderStorageImageWriteWithoutFormat) continue;
            uint32_t families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, queues.data());
            for (uint32_t i = 0; i < families && !physical; ++i) {
                if (!(queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
                physical = candidate;
                family = i;
            }
            if (physical) break;
        }
        if (!physical) return false;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        std::printf("composition rebuild device: %s\n", properties.deviceName);
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qinfo{};
        qinfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qinfo.queueFamilyIndex = family;
        qinfo.queueCount = 1;
        qinfo.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures enabled{};
        enabled.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        VkDeviceCreateInfo dinfo{};
        dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dinfo.queueCreateInfoCount = 1;
        dinfo.pQueueCreateInfos = &qinfo;
        dinfo.pEnabledFeatures = &enabled;
        VkCheck(vkCreateDevice(physical, &dinfo, nullptr, &device), "create device");
        vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pinfo{};
        pinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pinfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pinfo.queueFamilyIndex = family;
        VkCheck(vkCreateCommandPool(device, &pinfo, nullptr, &pool), "create command pool");
        VkCommandBufferAllocateInfo ainfo{};
        ainfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ainfo.commandPool = pool;
        ainfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ainfo.commandBufferCount = 1;
        VkCheck(vkAllocateCommandBuffers(device, &ainfo, &cmd), "allocate command buffer");
        instanceTable.next_gipa = vkGetInstanceProcAddr;
        instanceTable.Load(instance);
        deviceTable.next_dpa = vkGetDeviceProcAddr;
        deviceTable.Load(device);
        tracker.createView = std::exchange(deviceTable.vkCreateImageView, CreateView);
        tracker.destroyView = std::exchange(deviceTable.vkDestroyImageView, DestroyView);
        tracker.allocate = std::exchange(deviceTable.vkAllocateDescriptorSets, Allocate);
        tracker.update = std::exchange(deviceTable.vkUpdateDescriptorSets, Update);
        tracker.bind = std::exchange(deviceTable.vkCmdBindDescriptorSets, Bind);
        return true;
    }

    // Stands in for the swapchain image the layer composes into.
    void MakeSwapchainImage(uint32_t width, uint32_t height, VkFormat format) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkCheck(vkCreateImage(device, &info, nullptr, &swapchain), "create swapchain stand-in");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, swapchain, &requirements);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        uint32_t type = 0;
        while (type < memory.memoryTypeCount && !(requirements.memoryTypeBits & (1u << type))) ++type;
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        VkCheck(vkAllocateMemory(device, &allocation, nullptr, &swapchainMemory), "allocate swapchain stand-in");
        VkCheck(vkBindImageMemory(device, swapchain, swapchainMemory, 0), "bind swapchain stand-in");
    }

    VkCommandBuffer Begin() {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkCheck(vkBeginCommandBuffer(cmd, &begin), "begin command buffer");
        return cmd;
    }

    // Refuses before the queue sees a stale binding: executing one reads a freed image, which
    // crashes a CPU device and faults a GPU.
    void Submit() {
        VkCheck(vkEndCommandBuffer(cmd), "end command buffer");
        if (tracker.stale) throw std::runtime_error("a dispatch bound a descriptor for a destroyed view");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VkCheck(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "submit");
        VkCheck(vkQueueWaitIdle(queue), "wait for queue");
    }
};

struct Step {
    float workingScale;
    bool linearHdr;
    uint32_t downscaler;
};

}  // namespace

int main() {
    try {
        Context context;
        if (!context.Init()) {
            std::printf("SKIP: no Vulkan device with storage writes without format\n");
            return 77;
        }
        constexpr uint32_t width = 320, height = 192;
        constexpr VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
        context.MakeSwapchainImage(width, height, format);
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = context.swapchain;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(context.Begin(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        context.Submit();

        dlssnr::Composition composition(&context.deviceTable, &context.instanceTable, context.device,
                                        context.physical);
        if (!composition.Usable()) throw std::runtime_error(composition.Reason());
        // Every step changes the model raster, the colour domain or the downscaler, so Prepare
        // destroys and recreates the composition surfaces while the pass and its sets live on.
        const Step steps[] = {
            {1.0f, false, dlssnr::kScalerLanczos3}, {0.5f, false, dlssnr::kScalerLanczos3},
            {1.0f, false, dlssnr::kScalerLanczos3}, {0.5f, false, dlssnr::kScalerLanczos3},
            {0.75f, false, dlssnr::kScalerLanczos3}, {1.5f, false, dlssnr::kScalerLanczos3},
            {1.5f, false, dlssnr::kScalerCatmullRom}, {1.0f, true, dlssnr::kScalerLanczos3},
            {0.5f, true, dlssnr::kScalerLanczos3}, {1.0f, false, dlssnr::kScalerLanczos3},
            {0.5f, false, dlssnr::kScalerLanczos3}, {0.75f, false, dlssnr::kScalerLanczos3},
        };
        unsigned composed = 0;
        for (const Step& step : steps) {
            dlssnr::FrameSettings settings;
            settings.workingScale = step.workingScale;
            settings.downscaler = step.downscaler;
            for (int frame = 0; frame < 3; ++frame) {
                if (!composition.Prepare(width, height, format, settings, step.linearHdr))
                    throw std::runtime_error(std::string("prepare failed: ") + composition.Reason());
                composition.RecordCapture(context.Begin(), context.swapchain, settings);
                context.Submit();
                std::memcpy(composition.ModelPixels(), composition.ProxyPixels(), composition.ModelBytes());
                composed += composition.RecordCompose(context.Begin(), context.swapchain, settings) ? 1u : 0u;
                context.Submit();
            }
        }
        std::printf("%u frames composed, %u descriptor set binds, %u view handles recycled, "
                    "%u stale bindings bound\n", composed, tracker.binds, tracker.recycled, tracker.stale);
        if (composed != 3 * (sizeof(steps) / sizeof(steps[0])))
            throw std::runtime_error("a frame was not composed");
        if (tracker.stale) throw std::runtime_error("a dispatch bound a descriptor for a destroyed view");
        std::printf("PASS: no descriptor outlived its image view across %zu builds\n",
                    sizeof(steps) / sizeof(steps[0]));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "composition rebuild test: %s\n", error.what());
        return 1;
    }
}
