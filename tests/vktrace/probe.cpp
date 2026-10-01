// SPDX-License-Identifier: MIT
// A small Vulkan compute program for checking the tracing layer
// (vktrace-test.py). A setup submission uploads a buffer through a mapped
// staging buffer and fills another; a one-shot submission, from a command pool
// made and destroyed around it as a build's are, runs probe.spv once; then two
// frames run it with different push constants, specialization and descriptor
// sets, and read the result and a storage image back. It prints the FNV-1a 64
// of what it wrote and read, so that the trace's Upload, Readback and Hash
// lines can be checked against it. One descriptor write and one copy each span
// two bindings, the result binding has an offset, and the storage image ends
// frame 0 in SHADER_READ_ONLY_OPTIMAL, from which the layer's hashing has to
// move it and to which it has to return it. With --destroy-during-hash, a
// second thread then submits a batch that waits for the host, and the probe
// destroys a buffer and an image and frees another buffer's memory while the
// layer waits for that batch before hashing what it selected at the submit.
// A sparse buffer, where the device supports one, stays and is hashed. Then a
// second thread creates, binds, destroys and frees buffers while the probe
// submits empty batches, which the layer hashes. It runs on a CPU device only,
// with the Khronos validation layer enabled below the tracing layer, and exits
// 77 without either.
#include <vulkan/vulkan.h>

#include <getopt.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
uint64_t fnv(const void* data, size_t n, uint64_t h = 14695981039346656037ull)
{
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
    return h;
}
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        const VkResult r_ = (x);                                                                                       \
        if (r_ != VK_SUCCESS) {                                                                                        \
            std::fprintf(stderr, "%s failed: %d\n", #x, int(r_));                                                      \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

struct Mem {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

constexpr const char* kUsage = "Usage: vktrace-probe [OPTION]... PROBE.SPV\n";

// The trace's path, as the tracing layer finds it: VKTRACE_FILE or its default,
// with %p replaced by the process ID.
std::string trace_path()
{
    std::string path = "/tmp/vktrace.%p.log";
    if (const char* f = std::getenv("VKTRACE_FILE"); f && *f) path = f;
    if (const size_t at = path.find("%p"); at != std::string::npos) path.replace(at, 2, std::to_string(getpid()));
    return path;
}

// Waits until the trace PATH holds a QueueSubmitResult line after the first
// submit line that waits for a semaphore, which the layer writes out before it
// waits for that submission. Returns false after 60 s.
bool submitted(const char* path)
{
    for (unsigned tries = 0; tries < 60000; ++tries) {
        std::string text;
        if (FILE* f = std::fopen(path, "r")) {
            char chunk[1 << 16];
            for (size_t n; (n = std::fread(chunk, 1, sizeof chunk, f));) text.append(chunk, n);
            std::fclose(f);
        }
        const size_t wait = text.find(" wait=[sem");
        if (wait != std::string::npos && text.find(" QueueSubmitResult ", wait) != std::string::npos) return true;
        const timespec millisecond{0, 1000000};
        nanosleep(&millisecond, nullptr);
    }
    return false;
}
} // namespace

int main(int argc, char** argv)
{
    bool split = false, destroy = false;
    const option options[] = {{"split-setup", no_argument, nullptr, 's'},
                              {"destroy-during-hash", no_argument, nullptr, 'd'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+sdh", options, nullptr)) != -1;) {
        if (code == 's') {
            split = true;
        } else if (code == 'd') {
            destroy = true;
        } else if (code == 'h') {
            std::printf("%s", kUsage);
            std::puts("Runs probe.spv on a CPU Vulkan device as vktrace-test.py describes, printing the FNV-1a 64\n"
                      "of what it uploads and reads back. Exits 77 without a CPU device or the Khronos validation\n"
                      "layer.\n"
                      " -s, --split-setup          Upload from a staging buffer of its own in a submission of its\n"
                      "                            own, fill in another, and destroy the staging buffer after the\n"
                      "                            one-shot submission (default: off; one setup submission from\n"
                      "                            the frames' upload buffer)\n"
                      " -d, --destroy-during-hash  After the frames, submit from a second thread a batch that\n"
                      "                            waits for a timeline semaphore; once the trace shows the\n"
                      "                            submission, destroy a buffer and an image, free a third\n"
                      "                            buffer's memory, then signal the semaphore. A sparse buffer,\n"
                      "                            if the device supports one, stays. Then submit empty batches\n"
                      "                            while a second thread creates, binds, destroys and frees\n"
                      "                            buffers. It reads the trace where the tracing layer writes\n"
                      "                            it: VKTRACE_FILE, or /tmp/vktrace.%p.log when that is unset,\n"
                      "                            with %p as the process ID (default: off)\n"
                      " -h, --help                 Show this help and exit (default: off)");
            return 0;
        } else {
            std::fprintf(stderr, "%s(see --help)\n", kUsage);
            return 2;
        }
    }
    if (optind + 1 != argc) {
        std::fprintf(stderr, "%s(see --help)\n", kUsage);
        return 2;
    }
    std::vector<uint32_t> code(64 << 10);
    FILE* f = std::fopen(argv[optind], "rb");
    code.resize(f ? std::fread(code.data(), 4, code.size(), f) : 0);
    if (!f || std::ferror(f) || !std::feof(f) || code.empty()) {
        std::fprintf(stderr, "probe: cannot read %s\n", argv[optind]);
        return 1;
    }
    std::fclose(f);
    // The layers the loader adds from the environment sit above those the
    // application enables: validation checks the tracing layer's work too.
    const char* validation = "VK_LAYER_KHRONOS_validation";
    uint32_t count = 0;
    CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
    const auto named = [&](const VkLayerProperties& l) { return !std::strcmp(l.layerName, validation); };
    if (std::none_of(layers.begin(), layers.end(), named)) {
        std::fprintf(stderr, "probe: skipped, %s is not installed\n", validation);
        return 77;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vktrace probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = &validation;
    VkInstance instance;
    CHECK(vkCreateInstance(&ici, nullptr, &instance));
    CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    const auto cpu = std::find_if(devices.begin(), devices.end(), [](VkPhysicalDevice d) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(d, &properties);
        return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    });
    if (cpu == devices.end()) {
        std::fprintf(stderr, "probe: skipped, no CPU Vulkan device\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    const VkPhysicalDevice physical = *cpu;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physical, &mp);
    VkPhysicalDeviceFeatures supported;
    vkGetPhysicalDeviceFeatures(physical, &supported);
    VkQueueFamilyProperties family;
    count = 1;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, &family);
    const bool sparse = destroy && supported.sparseBinding && (family.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
    const float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderInt8 = VK_TRUE;
    f12.timelineSemaphore = destroy;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.sparseBinding = sparse;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice device;
    CHECK(vkCreateDevice(physical, &dci, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    const auto allocate = [&](VkMemoryRequirements req, VkMemoryPropertyFlags want, bool map) {
        Mem m;
        uint32_t type = 0;
        while (type < mp.memoryTypeCount &&
               !((req.memoryTypeBits >> type & 1) && (mp.memoryTypes[type].propertyFlags & want) == want))
            ++type;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        CHECK(vkAllocateMemory(device, &ai, nullptr, &m.memory));
        if (map) CHECK(vkMapMemory(device, m.memory, 0, VK_WHOLE_SIZE, 0, &m.mapped));
        return m;
    };
    const auto buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, bool map, Mem& m) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        VkBuffer b;
        CHECK(vkCreateBuffer(device, &bi, nullptr, &b));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b, &req);
        m = allocate(req, want, map);
        CHECK(vkBindBufferMemory(device, b, m.memory, 0));
        return b;
    };
    constexpr uint32_t kN = 4096, kW = 64, kH = kN / kW;
    const VkDeviceSize bytes = kN * 4;
    const VkBufferUsageFlags all = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // The input a, and b, whose second half the shader writes; its first half
    // keeps the setup's fill.
    Mem mu, ma, mb, md, mi;
    VkBuffer upload = buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, true, mu);
    VkBuffer a = buffer(bytes, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, ma);
    VkBuffer b = buffer(bytes * 2, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, mb);
    VkBuffer download = buffer(bytes * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, true, md);
    auto* u = static_cast<uint32_t*>(mu.mapped);
    for (uint32_t i = 0; i < kN; ++i) u[i] = i * 2654435761u;

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {kW, kH, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage image;
    CHECK(vkCreateImage(device, &ii, nullptr, &image));
    VkMemoryRequirements ireq;
    vkGetImageMemoryRequirements(device, image, &ireq);
    mi = allocate(ireq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    CHECK(vkBindImageMemory(device, image, mi.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ii.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view;
    CHECK(vkCreateImageView(device, &vi, nullptr, &view));

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size() * 4;
    smi.pCode = code.data();
    VkShaderModule module;
    CHECK(vkCreateShaderModule(device, &smi, nullptr, &module));
    const VkDescriptorSetLayoutBinding lb[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3;
    dli.pBindings = lb;
    VkDescriptorSetLayout dsl;
    CHECK(vkCreateDescriptorSetLayout(device, &dli, nullptr, &dsl));
    const VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &layout));
    VkPipeline pipes[2];
    for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t scale = 2 + k;
        const VkSpecializationMapEntry entry{0, 0, 4};
        const VkSpecializationInfo spec{1, &entry, 4, &scale};
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = module;
        cpi.stage.pName = "main";
        cpi.stage.pSpecializationInfo = &spec;
        cpi.layout = layout;
        CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipes[k]));
    }
    const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
                                           {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(device, &dpi, nullptr, &pool));
    const VkDescriptorSetLayout dsls[2] = {dsl, dsl};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 2;
    dai.pSetLayouts = dsls;
    VkDescriptorSet sets[2];
    CHECK(vkAllocateDescriptorSets(device, &dai, sets));
    // Bindings 0 and 1 in one write: its second descriptor goes to binding 1.
    const VkDescriptorBufferInfo bi[2] = {{a, 0, VK_WHOLE_SIZE}, {b, bytes, bytes}};
    const VkDescriptorImageInfo imi{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    w[0].dstSet = sets[0];
    w[0].dstBinding = 0;
    w[0].descriptorCount = 2;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = bi;
    w[1].dstSet = sets[0];
    w[1].dstBinding = 2;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo = &imi;
    vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    // Frame 1's set is a copy, bindings 0 and 1 again in one.
    VkCopyDescriptorSet copies[2] = {{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET}};
    copies[0].srcSet = sets[0];
    copies[0].dstSet = sets[1];
    copies[0].descriptorCount = 2;
    copies[1] = copies[0];
    copies[1].srcBinding = copies[1].dstBinding = 2;
    copies[1].descriptorCount = 1;
    vkUpdateDescriptorSets(device, 0, nullptr, 2, copies);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool;
    CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cpool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(device, &cai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(device, &fci, nullptr, &fence));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 2;
    VkQueryPool qp;
    CHECK(vkCreateQueryPool(device, &qpi, nullptr, &qp));
    const VkCommandBufferBeginInfo once{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    const auto submit = [&](VkCommandBuffer c) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        CHECK(vkResetFences(device, 1, &fence));
        CHECK(vkQueueSubmit(queue, 1, &si, fence));
        CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    };

    // Setup, which dispatches nothing: the input uploaded and b filled.
    constexpr uint32_t kFill = 0x5a0f3c01;
    const std::vector<uint32_t> filled(kN * 2, kFill);
    const VkBufferCopy whole{0, 0, bytes};
    Mem ms;
    VkBuffer staging = VK_NULL_HANDLE;
    CHECK(vkBeginCommandBuffer(cb, &once));
    if (split) {
        staging = buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, true, ms);
        std::memcpy(ms.mapped, u, bytes);
        vkCmdCopyBuffer(cb, staging, a, 1, &whole);
        CHECK(vkEndCommandBuffer(cb));
        submit(cb);
        CHECK(vkResetCommandBuffer(cb, 0));
        CHECK(vkBeginCommandBuffer(cb, &once));
    } else {
        vkCmdCopyBuffer(cb, upload, a, 1, &whole);
    }
    vkCmdFillBuffer(cb, b, 0, bytes * 2, kFill);
    CHECK(vkEndCommandBuffer(cb));
    submit(cb);
    std::printf("setup upload=%016" PRIx64 " fill=%016" PRIx64 "\n", fnv(u, bytes), fnv(filled.data(), bytes * 2));

    // What the shader writes to b's second half with scale 2 and ADD, and the
    // FNV of all of b then.
    const auto b_fnv = [&](const void* result) { return fnv(result, bytes, fnv(filled.data(), bytes)); };
    std::vector<uint32_t> expected(kN);
    for (uint32_t i = 0; i < kN; ++i) expected[i] = u[i] * 2 + 5;

    // A build's one-shot dispatch: its command pool lives for this submission alone.
    VkCommandPoolCreateInfo opci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool opool;
    CHECK(vkCreateCommandPool(device, &opci, nullptr, &opool));
    VkCommandBufferAllocateInfo oai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    oai.commandPool = opool;
    oai.commandBufferCount = 1;
    VkCommandBuffer ocb;
    CHECK(vkAllocateCommandBuffers(device, &oai, &ocb));
    CHECK(vkBeginCommandBuffer(ocb, &once));
    const VkMemoryBarrier filled_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = image;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(ocb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &filled_barrier, 0, nullptr, 1, &ib);
    vkCmdBindPipeline(ocb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
    vkCmdBindDescriptorSets(ocb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[0], 0, nullptr);
    const uint32_t first[2] = {5, kW};
    vkCmdPushConstants(ocb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, first);
    vkCmdDispatch(ocb, kN / 64, 1, 1);
    CHECK(vkEndCommandBuffer(ocb));
    submit(ocb);
    vkDestroyCommandPool(device, opool, nullptr);
    if (split) {
        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, ms.memory, nullptr);
    }
    std::printf("oneshot result=%016" PRIx64 " b=%016" PRIx64 "\n", fnv(expected.data(), bytes),
                b_fnv(expected.data()));

    VkImageLayout current = VK_IMAGE_LAYOUT_GENERAL;
    for (uint32_t frame = 0; frame < 2; ++frame) {
        CHECK(vkResetCommandBuffer(cb, 0));
        CHECK(vkBeginCommandBuffer(cb, &once));
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        vkCmdCopyBuffer(cb, upload, a, 1, &whole);
        ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ib.oldLayout = current;
        const VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[frame]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[frame], 0, nullptr);
        const uint32_t push[2] = {7 + frame, kW};
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
        vkCmdDispatch(cb, kN / 64, 1, 1);
        VkMemoryBarrier2 m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        m2.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        m2.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        m2.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        m2.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        VkImageMemoryBarrier2 i2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        i2.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        i2.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        i2.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        i2.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        i2.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        i2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        i2.srcQueueFamilyIndex = i2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        i2.image = image;
        i2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &m2;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &i2;
        vkCmdPipelineBarrier2(cb, &dep);
        const VkBufferCopy result{bytes, 0, bytes};
        vkCmdCopyBuffer(cb, b, download, 1, &result);
        VkBufferImageCopy region{};
        region.bufferOffset = bytes;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kW, kH, 1};
        vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, download, 1, &region);
        const VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                             nullptr, 0, nullptr);
        current = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        if (!frame) {
            // Left where the layer's hashing cannot copy from directly.
            VkImageMemoryBarrier sampled = ib;
            sampled.srcAccessMask = 0;
            sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sampled.oldLayout = current;
            sampled.newLayout = current = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &sampled);
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        CHECK(vkEndCommandBuffer(cb));
        submit(cb);
        std::printf("frame %u upload=%016" PRIx64 " result=%016" PRIx64 " image=%016" PRIx64 " b=%016" PRIx64 "\n",
                    frame, fnv(mu.mapped, bytes), fnv(md.mapped, bytes),
                    fnv(static_cast<unsigned char*>(md.mapped) + bytes, bytes), b_fnv(md.mapped));
    }

    // Three resources that VKTRACE_HASH=all selects at the next submission,
    // after a setup submission that leaves the image in GENERAL, from which the
    // layer would copy it. The setup also fills the sparse buffer, which stays.
    if (destroy) {
        Mem gone_memory, unbound_memory, gone_image_memory, sparse_memory;
        VkBuffer gone = buffer(bytes, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, gone_memory);
        VkBuffer unbound = buffer(bytes, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, unbound_memory);
        VkImage gone_image;
        CHECK(vkCreateImage(device, &ii, nullptr, &gone_image));
        vkGetImageMemoryRequirements(device, gone_image, &ireq);
        gone_image_memory = allocate(ireq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
        CHECK(vkBindImageMemory(device, gone_image, gone_image_memory.memory, 0));
        // Bound through the queue, which the layer does not trace: it has to
        // copy the buffer as if it were bound.
        VkBuffer sparse_buffer = VK_NULL_HANDLE;
        if (sparse) {
            VkBufferCreateInfo sbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            sbi.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
            sbi.size = bytes;
            sbi.usage = all;
            CHECK(vkCreateBuffer(device, &sbi, nullptr, &sparse_buffer));
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, sparse_buffer, &req);
            sparse_memory = allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
            const VkSparseMemoryBind bind{0, req.size, sparse_memory.memory, 0, 0};
            const VkSparseBufferMemoryBindInfo binds{sparse_buffer, 1, &bind};
            VkBindSparseInfo bsi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
            bsi.bufferBindCount = 1;
            bsi.pBufferBinds = &binds;
            CHECK(vkResetFences(device, 1, &fence));
            CHECK(vkQueueBindSparse(queue, 1, &bsi, fence));
            CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        }
        CHECK(vkResetCommandBuffer(cb, 0));
        CHECK(vkBeginCommandBuffer(cb, &once));
        if (sparse) vkCmdFillBuffer(cb, sparse_buffer, 0, bytes, kFill);
        VkImageMemoryBarrier general = ib;
        general.srcAccessMask = 0;
        general.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        general.image = gone_image;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &general);
        CHECK(vkEndCommandBuffer(cb));
        submit(cb);
        // The layer selects them when the waiting batch is submitted and copies
        // them once it has completed: they are gone by then.
        VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        const VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
        VkSemaphore semaphore;
        CHECK(vkCreateSemaphore(device, &sci, nullptr, &semaphore));
        CHECK(vkResetFences(device, 1, &fence));
        std::thread waiter([&] {
            const uint64_t one = 1;
            VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            values.waitSemaphoreValueCount = 1;
            values.pWaitSemaphoreValues = &one;
            const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values};
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &semaphore;
            si.pWaitDstStageMask = &stage;
            CHECK(vkQueueSubmit(queue, 1, &si, fence));
        });
        const std::string trace = trace_path();
        if (!submitted(trace.c_str())) {
            std::fprintf(stderr, "probe: no waiting submission in %s\n", trace.c_str());
            std::exit(1);
        }
        vkDestroyBuffer(device, gone, nullptr);
        vkFreeMemory(device, gone_memory.memory, nullptr);
        vkFreeMemory(device, unbound_memory.memory, nullptr);
        vkDestroyImage(device, gone_image, nullptr);
        vkFreeMemory(device, gone_image_memory.memory, nullptr);
        const VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, nullptr, semaphore, 1};
        CHECK(vkSignalSemaphore(device, &signal));
        waiter.join();
        CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyBuffer(device, unbound, nullptr);
        vkDestroyBuffer(device, sparse_buffer, nullptr);
        vkFreeMemory(device, sparse_memory.memory, nullptr);
        vkDestroySemaphore(device, semaphore, nullptr);
        // The same at any moment: buffers that live for up to 300 us each,
        // half of them losing their memory first, while each submission is
        // hashed.
        std::atomic<bool> stop{false};
        std::thread churn([&] {
            for (uint32_t seed = 1; !stop.load(); seed = seed * 1103515245u + 12345u) {
                Mem m;
                const VkBuffer c = buffer(4 << 20, all, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, m);
                const timespec hold{0, long(seed >> 16 & 0xffff) % 301 * 1000};
                nanosleep(&hold, nullptr);
                if (seed & 1 << 16) {
                    vkFreeMemory(device, m.memory, nullptr);
                    vkDestroyBuffer(device, c, nullptr);
                } else {
                    vkDestroyBuffer(device, c, nullptr);
                    vkFreeMemory(device, m.memory, nullptr);
                }
            }
        });
        const VkSubmitInfo empty{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        for (uint32_t i = 0; i < 300; ++i) {
            CHECK(vkResetFences(device, 1, &fence));
            CHECK(vkQueueSubmit(queue, 1, &empty, fence));
            CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        }
        stop.store(true);
        churn.join();
        if (sparse) std::printf("destroyed during hash, sparse=%016" PRIx64 "\n", fnv(filled.data(), bytes));
        else std::puts("destroyed during hash, sparse=none");
    }
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, qp, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cpool, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    for (VkPipeline p : pipes) vkDestroyPipeline(device, p, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyImageView(device, view, nullptr);
    vkDestroyImage(device, image, nullptr);
    for (VkBuffer x : {upload, a, b, download}) vkDestroyBuffer(device, x, nullptr);
    for (Mem* m : {&mu, &ma, &mb, &md, &mi}) vkFreeMemory(device, m->memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
