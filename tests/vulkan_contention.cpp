// SPDX-License-Identifier: MIT
// vulkan-contention: bounded compute work on a Vulkan device from a process of
// its own, for the hardware check of the Vulkan network's waits
// (VALIDATION.md). Another client whose workgroups hold the compute units'
// shared memory for milliseconds can make the network's waits run out; this
// is such a client, bounded: one dispatch a submission, of at most kWork
// workgroup iterations, until the time is up or a submission takes longer
// than kLongest.
#include <getopt.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
// After <cstdint>: the load's SPIR-V, embedded.
#include "tests/vulkan_contention.h"

namespace {

// A submission that takes longer than this stops the load: the check wants
// work of milliseconds, far from amdgpu's job timeout of 2 s.
constexpr auto kLongest = std::chrono::milliseconds(100);
// Invocations a workgroup (vulkan_contention.comp).
constexpr uint32_t kInvocations = 256;

// A number option's range and its value when not given.
struct Limit {
    unsigned low, high, initial;
};
constexpr Limit kSeconds{1, 60, 40}, kWorkgroups{1, 4096, 64}, kLdsBytes{4, 65536, 65536},
    kIterations{1, 2000000, 1000000}, kDevice{0, 255, 0};
// A dispatch's work, workgroups times iterations, at most the defaults'. On an
// otherwise idle RX 9070 XT the defaults took 9-19 ms a submission, and the
// extremes of these limits, one workgroup or 32 of kIterations.high, or 4096
// of 4 bytes or 64 KiB, at most 19 ms; beside the network's frames the
// defaults took up to 65 ms. No option asks for a dispatch near kLongest
// before the first one runs.
constexpr uint64_t kWork = uint64_t(kWorkgroups.initial) * kIterations.initial;

struct Settings {
    bool graphics = false;
    unsigned seconds = kSeconds.initial, workgroups = kWorkgroups.initial, lds_bytes = kLdsBytes.initial,
             iterations = kIterations.initial;
    int device = -1;
};

void usage()
{
    std::printf("Usage: vulkan-contention [OPTION]...\n"
                "Loads a Vulkan device from a process of its own with bounded compute work, as another GPU\n"
                "client can, for the hardware check of the Vulkan network's waits (VALIDATION.md): one\n"
                "dispatch a submission, of workgroups that each hold shared memory and run a bounded loop,\n"
                "until the time is up. A submission that takes longer than %lld ms stops the load.\n"
                " -q, --queue KIND      compute: a queue of a compute-only family, the asynchronous compute\n"
                "                       engine; graphics: one of a graphics family (default: compute)\n"
                " -s, --seconds N       How long to load the device, %u..%u (default: %u)\n"
                " -w, --workgroups N    Workgroups of %u invocations a dispatch, %u..%u (default: %u)\n"
                " -l, --lds-bytes N     Shared memory a workgroup holds, a multiple of 4 from %u to %u bytes\n"
                "                       and at most the device's limit (default: %u)\n"
                " -i, --iterations N    Dependent multiply-adds an invocation runs, %u..%u and at most\n"
                "                       %llu / workgroups (default: %u)\n"
                " -d, --device N        The Vulkan device by its index, %u..%u (default: the first with a\n"
                "                       queue of KIND)\n"
                " -h, --help            Show help (default: off)\n",
                static_cast<long long>(kLongest.count()), kSeconds.low, kSeconds.high, kSeconds.initial,
                kInvocations, kWorkgroups.low, kWorkgroups.high, kWorkgroups.initial, kLdsBytes.low, kLdsBytes.high,
                kLdsBytes.initial, kIterations.low, kIterations.high, static_cast<unsigned long long>(kWork),
                kIterations.initial, kDevice.low, kDevice.high);
}

// TEXT as a number of LIMIT's range for the option NAME, or false.
bool number(const char* name, const char* text, const Limit& limit, unsigned& value)
{
    char* end = nullptr;
    const unsigned long n = std::strtoul(text, &end, 10);
    if (*text >= '0' && *text <= '9' && !*end && n >= limit.low && n <= limit.high) {
        value = unsigned(n);
        return true;
    }
    std::fprintf(stderr, "vulkan-contention: --%s takes a number from %u to %u, not \"%s\"\n", name, limit.low,
                 limit.high, text);
    return false;
}

volatile std::sig_atomic_t stopping = 0;
void stop(int) { stopping = 1; }

bool check(VkResult result, const char* what)
{
    if (result == VK_SUCCESS) return true;
    std::fprintf(stderr, "vulkan-contention: %s failed (VkResult %d)\n", what, int(result));
    return false;
}

// What the load makes, destroyed once the device is idle.
struct Load {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Load() = default;
    Load(const Load&) = delete;
    ~Load()
    {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, commands, nullptr);
            vkDestroyDescriptorPool(device, pool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, module, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            vkDestroyBuffer(device, buffer, nullptr);
            vkFreeMemory(device, memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

// The family of PHYSICAL's queues that S asks for, or UINT32_MAX.
uint32_t family_of(VkPhysicalDevice physical, const Settings& s)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        const VkQueueFlags flags = families[i].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) && bool(flags & VK_QUEUE_GRAPHICS_BIT) == s.graphics) return i;
    }
    return UINT32_MAX;
}

int run(const Settings& s)
{
    Load l;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vulkan-contention";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance.pApplicationInfo = &app;
    if (!check(vkCreateInstance(&instance, nullptr, &l.instance), "vkCreateInstance")) return 1;
    uint32_t count = 0;
    if (!check(vkEnumeratePhysicalDevices(l.instance, &count, nullptr), "vkEnumeratePhysicalDevices")) return 1;
    std::vector<VkPhysicalDevice> devices(count);
    if (!check(vkEnumeratePhysicalDevices(l.instance, &count, devices.data()), "vkEnumeratePhysicalDevices"))
        return 1;
    const char* kind = s.graphics ? "graphics" : "compute-only";
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < count && !physical; ++i)
        if ((s.device < 0 || unsigned(s.device) == i) && (family = family_of(devices[i], s)) != UINT32_MAX)
            physical = devices[i];
    if (!physical) {
        if (s.device >= 0 && unsigned(s.device) >= count)
            std::fprintf(stderr, "vulkan-contention: no Vulkan device %d\n", s.device);
        else
            std::fprintf(stderr, "vulkan-contention: no Vulkan device with a %s queue\n", kind);
        return 1;
    }
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    if (s.lds_bytes > properties.limits.maxComputeSharedMemorySize) {
        std::fprintf(stderr, "vulkan-contention: %s holds at most %u bytes of shared memory a workgroup\n",
                     properties.deviceName, properties.limits.maxComputeSharedMemorySize);
        return 1;
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (!check(vkCreateDevice(physical, &device_info, nullptr, &l.device), "vkCreateDevice")) return 1;
    VkQueue queue;
    vkGetDeviceQueue(l.device, family, 0, &queue);

    // The results, which nothing reads, in device-local memory.
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = VkDeviceSize(s.workgroups) * kInvocations * 4;
    buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (!check(vkCreateBuffer(l.device, &buffer, nullptr, &l.buffer), "vkCreateBuffer")) return 1;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(l.device, l.buffer, &req);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    const auto local = [&](uint32_t type) {
        return (req.memoryTypeBits >> type & 1) &&
               (memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };
    uint32_t type = 0;
    while (type < memory.memoryTypeCount && !local(type)) ++type;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = req.size;
    allocation.memoryTypeIndex = type;
    if (!check(vkAllocateMemory(l.device, &allocation, nullptr, &l.memory), "vkAllocateMemory") ||
        !check(vkBindBufferMemory(l.device, l.buffer, l.memory, 0), "vkBindBufferMemory"))
        return 1;

    // The pipeline: its shared memory sized by a specialization constant.
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                               nullptr};
    VkDescriptorSetLayoutCreateInfo set_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_layout.bindingCount = 1;
    set_layout.pBindings = &binding;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &l.set_layout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module.codeSize = sizeof kContentionSpv;
    module.pCode = kContentionSpv;
    if (!check(vkCreateDescriptorSetLayout(l.device, &set_layout, nullptr, &l.set_layout),
               "vkCreateDescriptorSetLayout") ||
        !check(vkCreatePipelineLayout(l.device, &layout, nullptr, &l.layout), "vkCreatePipelineLayout") ||
        !check(vkCreateShaderModule(l.device, &module, nullptr, &l.module), "vkCreateShaderModule"))
        return 1;
    const uint32_t shared_words = s.lds_bytes / 4;
    const VkSpecializationMapEntry entry{0, 0, sizeof shared_words};
    const VkSpecializationInfo specialization{1, &entry, sizeof shared_words, &shared_words};
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = l.module;
    pipeline.stage.pName = "main";
    pipeline.stage.pSpecializationInfo = &specialization;
    pipeline.layout = l.layout;
    if (!check(vkCreateComputePipelines(l.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &l.pipeline),
               "vkCreateComputePipelines"))
        return 1;

    // Its set and the one dispatch that every submission runs.
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    if (!check(vkCreateDescriptorPool(l.device, &pool, nullptr, &l.pool), "vkCreateDescriptorPool")) return 1;
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = l.pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &l.set_layout;
    VkDescriptorSet set;
    if (!check(vkAllocateDescriptorSets(l.device, &set_info, &set), "vkAllocateDescriptorSets")) return 1;
    const VkDescriptorBufferInfo results{l.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &results;
    vkUpdateDescriptorSets(l.device, 1, &write, 0, nullptr);
    VkCommandPoolCreateInfo commands{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commands.queueFamilyIndex = family;
    if (!check(vkCreateCommandPool(l.device, &commands, nullptr, &l.commands), "vkCreateCommandPool")) return 1;
    VkCommandBufferAllocateInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = l.commands;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (!check(vkAllocateCommandBuffers(l.device, &cmd_info, &cmd), "vkAllocateCommandBuffers") ||
        !check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer"))
        return 1;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, l.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof s.iterations, &s.iterations);
    vkCmdDispatch(cmd, s.workgroups, 1, 1);
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer") ||
        !check(vkCreateFence(l.device, &fence, nullptr, &l.fence), "vkCreateFence"))
        return 1;

    std::fprintf(stderr,
                 "vulkan-contention: %u workgroups of %u bytes of shared memory and %u iterations a dispatch on "
                 "the %s queue of %s, for %u s\n",
                 s.workgroups, s.lds_bytes, s.iterations, s.graphics ? "graphics" : "compute", properties.deviceName,
                 s.seconds);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto end = start + std::chrono::seconds(s.seconds);
    unsigned submissions = 0;
    Clock::duration longest{};
    int status = 0;
    while (!stopping && Clock::now() < end) {
        const auto submitted = Clock::now();
        if (!check(vkQueueSubmit(queue, 1, &submit, l.fence), "vkQueueSubmit") ||
            !check(vkWaitForFences(l.device, 1, &l.fence, VK_TRUE, 5'000'000'000ull), "vkWaitForFences") ||
            !check(vkResetFences(l.device, 1, &l.fence), "vkResetFences"))
            return 1;
        const auto took = Clock::now() - submitted;
        longest = std::max(longest, took);
        ++submissions;
        if (took > kLongest) {
            std::fprintf(stderr,
                         "vulkan-contention: a submission took %.1f ms, longer than %lld ms; stopping (lower "
                         "--iterations or --workgroups)\n",
                         std::chrono::duration<double, std::milli>(took).count(),
                         static_cast<long long>(kLongest.count()));
            status = 1;
            break;
        }
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::fprintf(stderr, "vulkan-contention: %u submissions in %.1f s, %.3f ms each, the longest %.3f ms\n",
                 submissions, seconds, submissions ? seconds * 1e3 / submissions : 0.0,
                 std::chrono::duration<double, std::milli>(longest).count());
    return status;
}

} // namespace

int main(int argc, char** argv)
{
    Settings s;
    const option options[] = {{"queue", required_argument, nullptr, 'q'},
                              {"seconds", required_argument, nullptr, 's'},
                              {"workgroups", required_argument, nullptr, 'w'},
                              {"lds-bytes", required_argument, nullptr, 'l'},
                              {"iterations", required_argument, nullptr, 'i'},
                              {"device", required_argument, nullptr, 'd'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+q:s:w:l:i:d:h", options, nullptr)) != -1;) {
        unsigned device = 0;
        bool ok = true;
        switch (code) {
        case 'q':
            s.graphics = !std::strcmp(optarg, "graphics");
            ok = s.graphics || !std::strcmp(optarg, "compute");
            if (!ok) std::fprintf(stderr, "vulkan-contention: --queue takes compute or graphics, not \"%s\"\n", optarg);
            break;
        case 's': ok = number("seconds", optarg, kSeconds, s.seconds); break;
        case 'w': ok = number("workgroups", optarg, kWorkgroups, s.workgroups); break;
        case 'l': ok = number("lds-bytes", optarg, kLdsBytes, s.lds_bytes); break;
        case 'i': ok = number("iterations", optarg, kIterations, s.iterations); break;
        case 'd':
            ok = number("device", optarg, kDevice, device);
            s.device = int(device);
            break;
        case 'h': usage(); return 0;
        default: return 2;
        }
        if (!ok) return 2;
    }
    if (optind != argc) {
        std::fprintf(stderr, "vulkan-contention: unexpected argument \"%s\"; see --help\n", argv[optind]);
        return 2;
    }
    if (s.lds_bytes % 4) {
        std::fprintf(stderr, "vulkan-contention: --lds-bytes takes a multiple of 4, not %u\n", s.lds_bytes);
        return 2;
    }
    if (uint64_t(s.workgroups) * s.iterations > kWork) {
        std::fprintf(stderr, "vulkan-contention: --iterations takes at most %llu / workgroups, not %u with %u\n",
                     static_cast<unsigned long long>(kWork), s.iterations, s.workgroups);
        return 2;
    }
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    return run(s);
}
