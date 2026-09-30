// SPDX-License-Identifier: MIT
#include "vulkan_runtime.h"
#include "files.h"
#include "vulkan_weights.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace dlsslop {

Result<void> vk_check(VkResult result, const char* what)
{
    if (result != VK_SUCCESS) return fail(std::string(what) + " failed (VkResult " + std::to_string(int(result)) + ")");
    return {};
}

Result<uint32_t> memory_type(const VkPhysicalDeviceMemoryProperties& memory, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want) return i;
    return fail("no suitable Vulkan memory type");
}

} // namespace dlsslop

namespace dlsslop::vulkan {

namespace {

// The motion estimate's pyramid below the frame (upstream: kTemporalBase,
// kTemporalRadius and kTemporalReject in nr_runtime.cpp): levels from a
// quarter of the frame on, each searched within its radius.
constexpr uint32_t kMotionBase = 4;
constexpr int32_t kMotionRadius[] = {2, 3, 3, 4};
constexpr float kMotionReject = 0.5f;

// The push blocks of the runtime's own pipelines (upstream:
// Temporal::LumaPush and Temporal::FlowPush, and those record_all pushes to
// the pass stages and the alpha pass), as their GLSL declares them.
struct LumaPush {
    uint32_t dst_w, dst_h, src_w, src_h, mode;
};
struct FlowPush {
    uint32_t level_w, level_h, coarse_w, coarse_h;
    int32_t radius;
    uint32_t first, last;
    float reject;
};
struct StagePush {
    uint32_t w, h;
    float strength;
    uint32_t anchor;
};
struct AlphaPush {
    uint32_t w, h, rgba8;
};

// A pipeline's bindings: its storage buffers ('a' the activation arena, 'w'
// the weights, 'p' the motion parameters), then its images ('s' storage, 't'
// sampled), and its push range; for the runtime's own, its SPIR-V below the
// network's directory.
struct Bindings {
    const char* buffers;
    const char* images;
    uint32_t push;
    const char* file;
};
// The runtime's own pipelines, in Runtime::Adapter's order (upstream: the
// adapters of nr_runtime.cpp and the temporal variants of the pre and post
// blocks): the alpha pass, which restores the frame's alpha when later passes
// overwrote the input; dlsslop-amd's pass stages; the motion estimate's luma
// pyramid and flow; the pre and post blocks with motion history.
constexpr Bindings kAdapterBindings[] = {
    {"", "st", sizeof(AlphaPush), "runtime/runtime_alpha.spv"},
    {"", "sts", sizeof(StagePush), "runtime/pass_stages.spv"},
    {"", "tss", sizeof(LumaPush), "temporal/motion_luma.spv"},
    {"", "ssss", sizeof(FlowPush), "temporal/motion_estimate.spv"},
    {"aawwwap", "tttt", sizeof(PushFSwin) + sizeof(PushPreImage), "temporal/temporal_pre_fp32.spv"},
    {"aawwwp", "ssttt", sizeof(PushFSwin) + sizeof(PushUps) + sizeof(PushImageTail),
     "temporal/temporal_post_fp32.spv"},
};
// A kernel's images, by Images.
constexpr const char* kKernelImages[] = {"", "t", "sst"};

// Pipeline PIPELINE's: a Kernel, or one of the runtime's own after them.
constexpr Bindings bindings_of(size_t pipeline)
{
    if (pipeline >= size_t(Kernel::kCount)) return kAdapterBindings[pipeline - size_t(Kernel::kCount)];
    const KernelInfo& k = kKernels[pipeline];
    return {k.buffers, kKernelImages[size_t(k.images)], k.push, nullptr};
}

// The most bindings a pipeline has.
constexpr size_t most_bindings()
{
    size_t most = 0;
    for (size_t p = 0; p < std::size(kKernels) + std::size(kAdapterBindings); ++p) {
        const Bindings b = bindings_of(p);
        most = std::max(most, std::string_view(b.buffers).size() + std::string_view(b.images).size());
    }
    return most;
}
constexpr size_t kMostBindings = most_bindings();

constexpr VkPipelineStageFlags kTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr VkPipelineStageFlags kCompute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkAccessFlags kRead = VK_ACCESS_SHADER_READ_BIT;
constexpr VkAccessFlags kWrite = VK_ACCESS_SHADER_WRITE_BIT;
constexpr VkAccessFlags kCopyRead = VK_ACCESS_TRANSFER_READ_BIT;
constexpr VkAccessFlags kCopyWrite = VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkImageLayout kGeneral = VK_IMAGE_LAYOUT_GENERAL;
constexpr VkImageLayout kSampled = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
constexpr VkImageLayout kSource = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
constexpr VkImageLayout kTarget = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
constexpr VkImageSubresourceRange kColor{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
constexpr VkImageSubresourceLayers kColorLayer{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
constexpr VkFormat kWide = VK_FORMAT_R32G32B32A32_SFLOAT;
// Every image is copied to and from; the network samples some and stores
// into most (upstream: Context::image).
constexpr VkImageUsageFlags kCopies = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageUsageFlags kStorage = kCopies | VK_IMAGE_USAGE_STORAGE_BIT;
constexpr VkImageUsageFlags kSampledStorage = kStorage | VK_IMAGE_USAGE_SAMPLED_BIT;

void write_log(const Device& d, const char* line)
{
    if (d.log) d.log(line);
}

// DIRECTORY/shader-constants.txt, which must be this build's exactly
// (kManifest, then kShaderConstants' lines): the SPIR-V was built with the
// constants the plan's arithmetic assumes.
Result<void> check_constants(const std::string& directory)
{
    const std::string path = join(directory, "shader-constants.txt");
    const auto text = read_file(path);
    if (!text) return fail("cannot read " + path + ": " + text.error().what);
    std::string expected = std::string(kManifest) + "\n";
    for (const auto& c : kShaderConstants) expected += std::string(c.key) + " " + std::to_string(c.value) + "\n";
    if (*text == expected) return {};
    // The line that differs first, which starts at the same place in both.
    const size_t at = size_t(std::ranges::mismatch(expected, *text).in1 - expected.begin());
    const size_t newline = at ? expected.rfind('\n', at - 1) : std::string::npos;
    const size_t start = newline == std::string::npos ? 0 : newline + 1;
    const auto line = [start](const std::string& s) { return s.substr(start, s.find('\n', start) - start); };
    return fail(path + " does not match this build: \"" + line(*text) + "\" where \"" + line(expected) +
                "\" is expected; rebuild the network's shaders");
}

// The one-line markers beside the network's SPIR-V (kMarkers).
Result<void> check_markers(const std::string& directory)
{
    for (const auto& m : kMarkers) {
        const std::string path = join(directory, m.file);
        const auto text = read_file(path);
        if (!text) return fail("cannot read " + path + ": " + text.error().what);
        if (std::string_view(*text).substr(0, text->find_last_not_of(" \t\r\n") + 1) != m.line)
            return fail(path + " does not say " + m.line + "; rebuild the network's shaders");
    }
    return {};
}

// PATH's SPIR-V words (upstream: nrvk::read_spirv).
Result<std::vector<uint32_t>> read_spirv(const std::string& path)
{
    const auto text = read_file(path);
    if (!text) return fail("cannot read " + path + ": " + text.error().what);
    if (text->empty() || text->size() % 4) return fail(path + " is not SPIR-V");
    std::vector<uint32_t> code(text->size() / 4);
    std::memcpy(code.data(), text->data(), text->size());
    if (code[0] != 0x07230203u) return fail(path + " is not SPIR-V");
    return code;
}

// What the device offers the network (upstream: the checks of the Runtime
// constructor and nrvk's require_matrix_config): whether the input can be
// the frame's format filled by a copy, and whether the post block can store
// the answer in that format.
struct Capabilities {
    bool input_direct, answer_direct;
};

Result<Capabilities> query(const Device& d, const Shape& shape)
{
    const PhysicalFunctions& f = d.functions;
    // Nothing here calls vkGetPhysicalDeviceProperties2, but without it
    // storage_limit() gives plan() no limit, so the build must fail here.
    const char* missing = !f.queue_families      ? "vkGetPhysicalDeviceQueueFamilyProperties"
                          : !f.properties        ? "vkGetPhysicalDeviceProperties2"
                          : !f.format_properties ? "vkGetPhysicalDeviceFormatProperties2"
                          : !f.matrix_properties ? "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"
                                                 : nullptr;
    if (missing) return fail(std::string("the network cannot query its device: ") + missing + " is unavailable");
    // A queue that blits the frame's format into RGBA32F and back.
    uint32_t count = 0;
    f.queue_families(d.physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    f.queue_families(d.physical, &count, families.data());
    const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if (d.family >= count || (families[d.family].queueFlags & need) != need)
        return fail("the network's queue family has no graphics and compute");
    // The FP8 cooperative matrices every network kernel multiplies with.
    uint32_t configs = 0;
    DLSSLOP_TRY(vk_check(f.matrix_properties(d.physical, &configs, nullptr),
                         "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    std::vector<VkCooperativeMatrixPropertiesKHR> matrices(configs,
                                                           {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    if (const VkResult listed = f.matrix_properties(d.physical, &configs, matrices.data()); listed != VK_INCOMPLETE)
        DLSSLOP_TRY(vk_check(listed, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    matrices.resize(std::min<size_t>(configs, matrices.size()));
    if (std::ranges::none_of(matrices, [](const VkCooperativeMatrixPropertiesKHR& m) {
            return m.MSize == 16 && m.NSize == 16 && m.KSize == 16 && m.scope == VK_SCOPE_SUBGROUP_KHR &&
                   m.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT && m.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                   m.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && m.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                   !m.saturatingAccumulation;
        }))
        return fail("the device has no e4m3 16x16x16 cooperative-matrix configuration");
    // One pass samples the frame in its own format, copied into the input,
    // and without the pass stages the post block stores it in that format.
    const VkFormat frame = shape.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    VkFormatProperties3 frame3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 frame2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &frame3};
    f.format_properties(d.physical, frame, &frame2);
    VkFormatProperties2 wide{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
    f.format_properties(d.physical, kWide, &wide);
    const VkFormatFeatureFlags optimal = frame2.formatProperties.optimalTilingFeatures;
    const auto has = [](VkFormatFeatureFlags2 have, VkFormatFeatureFlags2 want) { return (have & want) == want; };
    const VkFormatFeatureFlags blits = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if (!has(optimal, blits) || !has(wide.formatProperties.optimalTilingFeatures, blits))
        return fail("the device cannot blit the frame's format and RGBA32F");
    Capabilities c;
    c.input_direct =
        shape.passes == 1 && has(optimal, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
    c.answer_direct = c.input_direct && !shape.stages &&
                      has(frame3.optimalTilingFeatures,
                          VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT | VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT);
    return c;
}

// The first of D's memory types among BITS with every property in WANT and
// none in WITHOUT, or memoryTypeCount.
uint32_t pick_memory(const Device& d, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags without = 0)
{
    uint32_t i = 0;
    for (; i < d.memory.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = d.memory.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & want) == want && !(flags & without)) break;
    }
    return i;
}

// A buffer of BYTES for USAGE in its own memory: device-local memory the host
// cannot see (upstream: Context::buffer), or with HOST coherent memory the
// host can, cached where there is such.
Result<void> make_buffer(const Device& d, VkDeviceSize bytes, VkBufferUsageFlags usage, bool host, const char* what,
                         Buffer& b)
{
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = usage;
    DLSSLOP_TRY(vk_check(vkCreateBuffer(d.device, &info, nullptr, &b.buffer), what));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d.device, b.buffer, &req);
    constexpr VkMemoryPropertyFlags visible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = host ? pick_memory(d, req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                         : pick_memory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (host && type == d.memory.memoryTypeCount) type = pick_memory(d, req.memoryTypeBits, visible);
    if (type == d.memory.memoryTypeCount) return fail(std::string("no memory type for ") + what);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    DLSSLOP_TRY(vk_check(vkAllocateMemory(d.device, &alloc, nullptr, &b.memory), what));
    return vk_check(vkBindBufferMemory(d.device, b.buffer, b.memory, 0), what);
}

// A WIDTH x HEIGHT image of FORMAT for USAGE in its own device-local memory,
// and its view (upstream: Context::image).
Result<void> make_image(const Device& d, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                        Image& i)
{
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    DLSSLOP_TRY(vk_check(vkCreateImage(d.device, &info, nullptr, &i.image), "create a network image"));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(d.device, i.image, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = pick_memory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (alloc.memoryTypeIndex == d.memory.memoryTypeCount) return fail("no device-local memory type for an image");
    DLSSLOP_TRY(vk_check(vkAllocateMemory(d.device, &alloc, nullptr, &i.memory), "allocate a network image"));
    DLSSLOP_TRY(vk_check(vkBindImageMemory(d.device, i.image, i.memory, 0), "bind a network image"));
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = i.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = kColor;
    return vk_check(vkCreateImageView(d.device, &view, nullptr, &i.view), "create a network image view");
}

// A sampler that FILTERs and addresses by ADDRESS, and nothing else.
Result<void> make_sampler(VkDevice device, VkFilter filter, VkSamplerAddressMode address, VkSampler& sampler)
{
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = info.minFilter = filter;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = info.addressModeV = info.addressModeW = address;
    return vk_check(vkCreateSampler(device, &info, nullptr, &sampler), "create a network sampler");
}

// PIPELINE's compute pipeline, through CACHE, from its SPIR-V below SHADERS:
// one set of its bindings and a push range, and 32 lanes, which the
// cooperative matrices' fragments assume (upstream: nrvk::Kernel::create).
Result<void> make_pipeline(VkDevice device, VkPipelineCache cache, const std::string& shaders, size_t pipeline,
                           Pipeline& p)
{
    const Bindings b = bindings_of(pipeline);
    const std::string file = b.file ? b.file : std::string("g_") + kKernels[pipeline].stem + ".spv";
    VkDescriptorSetLayoutBinding bindings[kMostBindings];
    uint32_t n = 0;
    for (const char* t = b.buffers; *t; ++t, ++n)
        bindings[n] = {n, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (const char* t = b.images; *t; ++t, ++n)
        bindings[n] = {n, *t == 's' ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const std::string what = "vkCreateComputePipelines " + file;
    VkDescriptorSetLayoutCreateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set.bindingCount = n;
    set.pBindings = bindings;
    DLSSLOP_TRY(vk_check(vkCreateDescriptorSetLayout(device, &set, nullptr, &p.set_layout), what.c_str()));
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, b.push};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &p.set_layout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &range;
    DLSSLOP_TRY(vk_check(vkCreatePipelineLayout(device, &layout, nullptr, &p.layout), what.c_str()));
    const auto code = DLSSLOP_TRY(read_spirv(join(shaders, file)));
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = 4 * code.size();
    module_info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    DLSSLOP_TRY(vk_check(vkCreateShaderModule(device, &module_info, nullptr, &module), what.c_str()));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo lanes{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    lanes.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &lanes};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = p.layout;
    const VkResult made = vkCreateComputePipelines(device, cache, 1, &info, nullptr, &p.pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    return vk_check(made, what.c_str());
}

void destroy(VkDevice device, Pipeline& p)
{
    if (p.pipeline) vkDestroyPipeline(device, p.pipeline, nullptr);
    if (p.layout) vkDestroyPipelineLayout(device, p.layout, nullptr);
    if (p.set_layout) vkDestroyDescriptorSetLayout(device, p.set_layout, nullptr);
    p = {};
}
void destroy(VkDevice device, Image& i)
{
    if (i.view) vkDestroyImageView(device, i.view, nullptr);
    if (i.image) vkDestroyImage(device, i.image, nullptr);
    if (i.memory) vkFreeMemory(device, i.memory, nullptr);
    i = {};
}
void destroy(VkDevice device, Buffer& b)
{
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = {};
}

// The pipeline cache from PATH, when it holds one; none when the device makes
// none (upstream: Context::load_pipeline_cache).
VkPipelineCache load_cache(VkDevice device, const std::string& path)
{
    if (path.empty()) return VK_NULL_HANDLE;
    std::string data;
    if (auto file = read_file(path); file && file->size() >= 32) data = std::move(*file);
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.empty() ? nullptr : data.data();
    VkPipelineCache cache = VK_NULL_HANDLE;
    return vkCreatePipelineCache(device, &info, nullptr, &cache) == VK_SUCCESS ? cache : VK_NULL_HANDLE;
}

// CACHE written to PATH, which dlsslopd and every game's in-layer network
// share (upstream: Context::save_pipeline_cache). A new file replaces PATH in
// one rename; upstream writes PATH.tmp, which another writer can truncate or
// rename away, and removes PATH before its rename. A failed save only makes the
// next build compile again.
void save_cache(VkDevice device, VkPipelineCache cache, const std::string& path)
{
    size_t bytes = 0;
    if (!cache || path.empty() || vkGetPipelineCacheData(device, cache, &bytes, nullptr) != VK_SUCCESS || !bytes)
        return;
    std::string data(bytes, '\0');
    if (vkGetPipelineCacheData(device, cache, &bytes, data.data()) != VK_SUCCESS) return;
    data.resize(bytes);
    (void)replace_file(path, data);
}

// An image's layout change, or a barrier in its layout (upstream: barrier in
// nr_runtime.cpp).
void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags src_stage,
             VkAccessFlags src, VkPipelineStageFlags dst_stage, VkAccessFlags dst)
{
    const VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, src, dst, from, to,
                                 VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, kColor};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Between compute work: its writes made visible to reads and writes, or with
// INVALIDATE only the reading caches invalidated (upstream: compute_barrier).
void compute_barrier(VkCommandBuffer cmd, bool invalidate = false)
{
    const VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, invalidate ? 0u : kWrite, kRead | kWrite};
    vkCmdPipelineBarrier(cmd, kCompute, kCompute, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// A WIDTH x HEIGHT copy from FROM to TO, or with CONVERT a blit that converts
// between their formats.
void transfer(VkCommandBuffer cmd, VkImage from, VkImageLayout from_layout, VkImage to, VkImageLayout to_layout,
              uint32_t width, uint32_t height, bool convert = false)
{
    if (!convert) {
        const VkImageCopy region{kColorLayer, {}, kColorLayer, {}, {width, height, 1}};
        vkCmdCopyImage(cmd, from, from_layout, to, to_layout, 1, &region);
        return;
    }
    const VkOffset3D end{int32_t(width), int32_t(height), 1};
    const VkImageBlit region{kColorLayer, {{}, end}, kColorLayer, {{}, end}};
    vkCmdBlitImage(cmd, from, from_layout, to, to_layout, 1, &region, VK_FILTER_NEAREST);
}

// A copy from FROM to TO, both in GENERAL, between compute work (upstream:
// copy_general in record_all).
void copy_general(VkCommandBuffer cmd, VkImage from, VkImage to, uint32_t width, uint32_t height)
{
    barrier(cmd, from, kGeneral, kGeneral, kCompute, kWrite | kRead, kTransfer, kCopyRead);
    barrier(cmd, to, kGeneral, kGeneral, kCompute, kRead | kWrite, kTransfer, kCopyWrite);
    transfer(cmd, from, kGeneral, to, kGeneral, width, height);
    barrier(cmd, to, kGeneral, kGeneral, kTransfer, kCopyWrite, kCompute, kRead);
    barrier(cmd, from, kGeneral, kGeneral, kTransfer, kCopyRead, kCompute, kRead | kWrite);
}

// The pre block's controls in its PushPreImage (upstream: patch_push): the
// style; the tone, which later passes do without; and the structures, under
// the automatic mask the skin's and the rest's.
void patch_pre(uint32_t* words, const Controls& c, bool later)
{
    PushPreImage p;
    std::memcpy(&p, words + sizeof(PushFSwin) / 4, sizeof p);
    p.style = float(c.style) / 128;
    p.tone = later ? 0.0f : c.tone;
    p.structure = c.auto_mask ? 1.0f : c.structure;
    p.skin = c.auto_mask ? (c.skin < 0 ? c.structure : c.skin) : -1.0f;
    p.other = c.auto_mask ? c.structure : -1.0f;
    std::memcpy(words + sizeof(PushFSwin) / 4, &p, sizeof p);
}

// The post block's controls in its PushImageTail (upstream: patch_push): the
// intensity, and whether it restores the frame's alpha itself (bit 31) and
// rounds the answer to 8 bits (bit 30).
void patch_post(uint32_t* words, const Controls& c, bool post_alpha, bool rgba8)
{
    constexpr size_t at = (sizeof(PushFSwin) + sizeof(PushUps)) / 4;
    PushImageTail p;
    std::memcpy(&p, words + at, sizeof p);
    p.intensity = std::clamp(c.intensity, 0.0f, 2.0f);
    p.w_off = (p.w_off & 0x3FFFFFFFu) | (post_alpha ? 0x80000000u | (rgba8 ? 0x40000000u : 0u) : 0u);
    std::memcpy(words + at, &p, sizeof p);
}

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// IMAGE, when it was made, from nothing into LAYOUT for anything after: a
// barrier added to LAYOUTS.
void settle(std::vector<VkImageMemoryBarrier>& layouts, const Image& image, VkImageLayout layout)
{
    if (image.image)
        layouts.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, layout,
                           VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image.image, kColor});
}

// What the build's submission uses and frees once it has run: never while
// the device may still read it, so a submission that is not known to have
// ended leaks it.
struct Setup {
    VkDevice device;
    Buffer staging{};
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool running = false;
    ~Setup()
    {
        if (running) return;
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        destroy(device, staging);
    }
};

} // namespace

uint64_t storage_limit(const Device& device)
{
    if (!device.functions.properties) return UINT64_MAX;
    VkPhysicalDeviceMaintenance3Properties allocation{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &allocation};
    device.functions.properties(device.physical, &properties);
    return std::min<uint64_t>(properties.properties.limits.maxStorageBufferRange, allocation.maxMemoryAllocationSize);
}

Result<Runtime> Runtime::build(const Device& device, const VulkanPaths& paths, const Shape& shape, const Plan& plan)
{
    Runtime runtime(device);
    DLSSLOP_TRY(runtime.make(paths, shape, plan));
    return runtime;
}

Runtime::Runtime(Runtime&& other) noexcept
    : device_(other.device_), objects_(std::exchange(other.objects_, {})), state_(other.state_),
      steps_(std::move(other.steps_)), push_(std::move(other.push_))
{
}

Runtime::~Runtime()
{
    const VkDevice d = device_.device;
    Objects& o = objects_;
    if (o.pool) vkDestroyDescriptorPool(d, o.pool, nullptr);
    for (Pipeline& p : o.pipelines) destroy(d, p);
    if (o.linear) vkDestroySampler(d, o.linear, nullptr);
    if (o.nearest) vkDestroySampler(d, o.nearest, nullptr);
    for (Image& i : o.history_store) destroy(d, i);
    destroy(d, o.depth);
    for (Image& i : o.history) destroy(d, i);
    for (Image& i : o.flow) destroy(d, i);
    for (auto& pyramid : o.luma)
        for (Image& i : pyramid) destroy(d, i);
    for (Image* i : {&o.scratch, &o.shown, &o.second, &o.answer, &o.input}) destroy(d, *i);
    for (Buffer* b : {&o.params, &o.weights, &o.arena}) destroy(d, *b);
}

Result<void> Runtime::make(const VulkanPaths& paths, const Shape& shape, const Plan& plan)
{
    const auto start = std::chrono::steady_clock::now();
    if (plan.width != shape.width || plan.height != shape.height || plan.steps.size() < 2 ||
        plan.steps.front().kernel != Kernel::kFswinImagePreds32 ||
        plan.steps.back().kernel != Kernel::kFswinImagePost32 || plan.steps.back().after != After::kFull)
        return fail("network plan: not a plan of the frames the network is built for");
    // What the build reads and what the device offers, before any object.
    DLSSLOP_TRY(check_constants(paths.shaders));
    DLSSLOP_TRY(check_markers(paths.shaders));
    if (shape.motion) DLSSLOP_TRY(check_constants(join(paths.shaders, "temporal")));
    const Model model = DLSSLOP_TRY(Model::open(paths.model));
    const Capabilities caps = DLSSLOP_TRY(query(device_, shape));

    State& s = state_;
    s.width = shape.width;
    s.height = shape.height;
    s.passes = std::clamp<uint32_t>(shape.passes, 1, kMaxPasses);
    s.motion = shape.motion;
    s.stages = shape.stages;
    s.rgba8 = !shape.fp16;
    s.input_direct = caps.input_direct;
    s.answer_direct = caps.answer_direct;
    s.post_alpha = s.passes == 1;
    s.pingpong = s.motion && s.passes == 1;
    s.level_width[0] = (s.width + kMotionBase - 1) / kMotionBase;
    s.level_height[0] = (s.height + kMotionBase - 1) / kMotionBase;
    for (uint32_t k = 1; k < kLevels; ++k) {
        s.level_width[k] = (s.level_width[k - 1] + 1) / 2;
        s.level_height[k] = (s.level_height[k - 1] + 1) / 2;
    }
    steps_ = plan.steps;
    push_ = plan.push;

    DLSSLOP_TRY(make_resources(shape, plan));
    const auto compiling = std::chrono::steady_clock::now();
    DLSSLOP_TRY(make_pipelines(paths));
    const double compiled = seconds_since(compiling);
    DLSSLOP_TRY(make_sets());
    double packed = 0;
    const auto setting_up = std::chrono::steady_clock::now();
    DLSSLOP_TRY(run_setup(plan, model, packed));
    const double ran = seconds_since(setting_up) - packed;

    size_t pipelines = 0;
    for (const Pipeline& p : objects_.pipelines) pipelines += p.pipeline != VK_NULL_HANDLE;
    char line[512];
    std::snprintf(line, sizeof line,
                  "built the network for %ux%u %s%s%s%s in %.2f s (pipelines %.2f s, weights %.2f s, GPU %.2f s): "
                  "%zu dispatches a pass, input %s, answer %s, arena %.1f MB, weights %.1f MB, %zu pipelines, "
                  "%u barriers chained",
                  s.width, s.height, shape.fp16 ? "FP16" : "RGBA8",
                  s.passes > 1 ? (", up to " + std::to_string(s.passes) + " passes").c_str() : "",
                  s.stages ? ", pass stages" : "", s.motion ? ", motion" : "", seconds_since(start), compiled, packed,
                  ran, steps_.size(), s.input_direct ? "copied" : "blitted to RGBA32F",
                  s.answer_direct ? "stored in the frame's format" : "blitted from RGBA32F",
                  double(plan.arena_bytes) / 1e6, double(plan.blob_bytes) / 1e6, pipelines, unsigned(plan.chained));
    write_log(device_, line);
    return {};
}

Result<void> Runtime::make_resources(const Shape& shape, const Plan& plan)
{
    const Device& d = device_;
    Objects& o = objects_;
    const State& s = state_;
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    DLSSLOP_TRY(make_buffer(d, plan.arena_bytes, usage, false, "the network's activation arena", o.arena));
    DLSSLOP_TRY(make_buffer(d, plan.blob_bytes, usage, false, "the network's weights", o.weights));
    // The input, sampled: the frame's format filled by a copy, or RGBA32F,
    // filled by a blit or by later passes. The answer, which the post block
    // stores in the frame's format or RGBA32F, and its second output, the
    // model's history. The first pass's input when later passes overwrite
    // it, and the pass stages' scratch.
    const VkFormat frame = shape.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    const uint32_t w = s.width, h = s.height;
    DLSSLOP_TRY(make_image(d, w, h, s.input_direct ? frame : kWide,
                           s.input_direct ? kCopies | VK_IMAGE_USAGE_SAMPLED_BIT : kSampledStorage, o.input));
    DLSSLOP_TRY(make_image(d, w, h, s.answer_direct ? frame : kWide, kStorage, o.answer));
    DLSSLOP_TRY(make_image(d, w, h, kWide, kStorage, o.second));
    if (s.passes > 1) DLSSLOP_TRY(make_image(d, w, h, kWide, kSampledStorage, o.shown));
    if (s.stages) DLSSLOP_TRY(make_image(d, w, h, kWide, kStorage, o.scratch));
    DLSSLOP_TRY(make_sampler(d.device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, o.nearest));
    if (!s.motion) return {};
    // The motion history: this frame's and the last frame's luma pyramids,
    // the flow between them, the history the pre and post blocks read, one
    // a pass with later passes, a depth nothing writes, and the parameters.
    for (auto& pyramid : o.luma)
        for (uint32_t k = 0; k < kLevels; ++k)
            DLSSLOP_TRY(make_image(d, s.level_width[k], s.level_height[k], VK_FORMAT_R32_SFLOAT, kStorage, pyramid[k]));
    for (uint32_t k = 0; k < kLevels; ++k)
        DLSSLOP_TRY(make_image(d, s.level_width[k], s.level_height[k], VK_FORMAT_R32G32_SFLOAT, kStorage, o.flow[k]));
    for (uint32_t c = 0; c < (s.pingpong ? 2u : 1u); ++c)
        DLSSLOP_TRY(make_image(d, w, h, kWide, kSampledStorage, o.history[c]));
    DLSSLOP_TRY(make_image(d, w, h, VK_FORMAT_R32_SFLOAT, kSampledStorage, o.depth));
    if (s.passes > 1)
        for (uint32_t pass = 0; pass < s.passes; ++pass)
            DLSSLOP_TRY(make_image(d, w, h, kWide, kStorage, o.history_store[pass]));
    DLSSLOP_TRY(make_buffer(d, 32, usage, false, "the network's motion parameters", o.params));
    return make_sampler(d.device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, o.linear);
}

Result<void> Runtime::make_pipelines(const VulkanPaths& paths)
{
    static_assert(std::size(kAdapterBindings) == kAdapters);
    const VkDevice d = device_.device;
    Pipeline* p = objects_.pipelines;
    const State& s = state_;
    // Every pipeline goes through the cache: the noise field's, which the
    // build runs, the kernels' that the frames run, and then the runtime's
    // own. With motion, the temporal variants replace the pre and post
    // blocks. The cache is saved once every pipeline exists or one has
    // failed; upstream saves it before it creates the runtime's own.
    const VkPipelineCache cache = load_cache(d, paths.cache);
    constexpr size_t noise = size_t(Kernel::kNoiseField);
    Result<void> made = make_pipeline(d, cache, paths.shaders, noise, p[noise]);
    for (size_t i = s.motion; made && i < steps_.size() - s.motion; ++i)
        if (const size_t k = size_t(steps_[i].kernel); !p[k].pipeline)
            made = make_pipeline(d, cache, paths.shaders, k, p[k]);
    const bool wanted[kAdapters] = {s.passes > 1, s.stages, s.motion, s.motion, s.motion, s.motion};
    for (size_t a = kAlpha; made && a < kPipelines; ++a)
        if (wanted[a - kAlpha]) made = make_pipeline(d, cache, paths.shaders, a, p[a]);
    save_cache(d, cache, paths.cache);
    if (cache) vkDestroyPipelineCache(d, cache, nullptr);
    return made;
}

Result<void> Runtime::make_sets()
{
    Objects& o = objects_;
    const State& s = state_;
    // What each set holds: the buffers its pipeline names, and its images,
    // sampled with a sampler or stored into without.
    struct ImageDescriptor {
        VkImageView view;
        VkImageLayout layout;
        VkSampler sampler;
    };
    struct SetDescriptor {
        size_t pipeline;
        VkDescriptorSet* set;
        ImageDescriptor images[5];
    };
    const ImageDescriptor input{o.input.view, kSampled, o.nearest};
    const auto stored = [](const Image& i) { return ImageDescriptor{i.view, kGeneral, VK_NULL_HANDLE}; };
    const auto linear = [&o](const Image& i) { return ImageDescriptor{i.view, kGeneral, o.linear}; };
    // A kernel's images, by Images.
    const ImageDescriptor kernel_images[][3] = {{}, {input}, {stored(o.answer), stored(o.second), input}};
    std::vector<SetDescriptor> sets;
    for (size_t k = 0; k < size_t(Kernel::kCount); ++k)
        if (o.pipelines[k].pipeline) {
            SetDescriptor& set = sets.emplace_back(SetDescriptor{k, &o.kernel_sets[k], {}});
            std::copy_n(kernel_images[size_t(kKernels[k].images)], 3, set.images);
        }
    // The first pass's input, which later passes overwrite in the input.
    ImageDescriptor first = input;
    if (s.passes > 1) {
        first = {o.shown.view, kGeneral, o.nearest};
        sets.push_back({kAlpha, &o.alpha_sets[0], {stored(o.answer), first}});
        if (s.stages) sets.push_back({kAlpha, &o.alpha_sets[1], {stored(o.scratch), first}});
    }
    if (s.stages) {
        sets.push_back({kStages, &o.stage_sets[0], {stored(o.answer), first, stored(o.scratch)}});
        sets.push_back({kStages, &o.stage_sets[1], {stored(o.scratch), first, stored(o.answer)}});
    }
    if (s.motion) {
        for (uint32_t p = 0; p < 2; ++p)
            for (uint32_t k = 0; k < kLevels; ++k) {
                // The finest level halves no finer one and the coarsest
                // flow reads no coarser one: they bind a level of their own.
                const Image& finer = o.luma[p][k ? k - 1 : kLevels - 1];
                const Image& coarser = o.flow[k + 1 < kLevels ? k + 1 : k];
                sets.push_back({kLuma, &o.luma_sets[p][k], {input, stored(finer), stored(o.luma[p][k])}});
                sets.push_back({kFlow,
                                &o.flow_sets[p][k],
                                {stored(o.luma[p][k]), stored(o.luma[1 - p][k]), stored(coarser), stored(o.flow[k])}});
            }
        // A pass reads history c and writes the next frame's into the other
        // or, with later passes, into the second output.
        for (uint32_t c = 0; c < (s.pingpong ? 2u : 1u); ++c) {
            sets.push_back({kPre, &o.pre_sets[c], {input, linear(o.flow[0]), linear(o.history[c]), linear(o.depth)}});
            sets.push_back({kPost,
                            &o.post_sets[c],
                            {stored(o.answer), stored(s.pingpong ? o.history[c ^ 1] : o.second), input,
                             linear(o.flow[0]), linear(o.history[c])}});
        }
    }

    // One pool for them all, and one allocation and one update.
    uint32_t counts[3] = {}; // storage buffers, storage images, sampled images
    std::vector<VkDescriptorSetLayout> layouts;
    for (const SetDescriptor& set : sets) {
        const Bindings b = bindings_of(set.pipeline);
        counts[0] += uint32_t(std::strlen(b.buffers));
        for (const char* t = b.images; *t; ++t) ++counts[*t == 's' ? 1 : 2];
        layouts.push_back(o.pipelines[set.pipeline].set_layout);
    }
    const VkDescriptorType types[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    VkDescriptorPoolSize sizes[3];
    uint32_t n = 0;
    for (uint32_t i = 0; i < 3; ++i)
        if (counts[i]) sizes[n++] = {types[i], counts[i]};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = uint32_t(sets.size());
    pool.poolSizeCount = n;
    pool.pPoolSizes = sizes;
    const VkDevice d = device_.device;
    DLSSLOP_TRY(vk_check(vkCreateDescriptorPool(d, &pool, nullptr, &o.pool), "create the network's descriptor pool"));
    std::vector<VkDescriptorSet> made(sets.size());
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = o.pool;
    alloc.descriptorSetCount = uint32_t(sets.size());
    alloc.pSetLayouts = layouts.data();
    DLSSLOP_TRY(vk_check(vkAllocateDescriptorSets(d, &alloc, made.data()), "allocate the network's descriptor sets"));
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    std::vector<VkWriteDescriptorSet> writes;
    buffers.reserve(counts[0]);
    images.reserve(counts[1] + counts[2]);
    for (size_t i = 0; i < sets.size(); ++i) {
        *sets[i].set = made[i];
        const Bindings b = bindings_of(sets[i].pipeline);
        uint32_t binding = 0;
        for (const char* t = b.buffers; *t; ++t) {
            const Buffer& buffer = *t == 'a' ? o.arena : *t == 'w' ? o.weights : o.params;
            buffers.push_back({buffer.buffer, 0, VK_WHOLE_SIZE});
            VkWriteDescriptorSet& w = writes.emplace_back(VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
            w.dstSet = made[i];
            w.dstBinding = binding++;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &buffers.back();
        }
        for (size_t j = 0; b.images[j]; ++j) {
            const ImageDescriptor& image = sets[i].images[j];
            images.push_back({image.sampler, image.view, image.layout});
            VkWriteDescriptorSet& w = writes.emplace_back(VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
            w.dstSet = made[i];
            w.dstBinding = binding++;
            w.descriptorCount = 1;
            w.descriptorType = types[b.images[j] == 's' ? 1 : 2];
            w.pImageInfo = &images.back();
        }
    }
    vkUpdateDescriptorSets(d, uint32_t(writes.size()), writes.data(), 0, nullptr);
    return {};
}

Result<void> Runtime::run_setup(const Plan& plan, const Model& model, double& packed)
{
    const VkDevice d = device_.device;
    Objects& o = objects_;
    // The weights packed straight into the staging memory.
    const auto packing = std::chrono::steady_clock::now();
    Setup setup{d};
    DLSSLOP_TRY(make_buffer(device_, plan.blob_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "the network's upload",
                            setup.staging));
    void* mapped = nullptr;
    DLSSLOP_TRY(
        vk_check(vkMapMemory(d, setup.staging.memory, 0, VK_WHOLE_SIZE, 0, &mapped), "map the network's upload"));
    DLSSLOP_TRY(pack(plan.segments, plan.tables, model, {static_cast<uint8_t*>(mapped), plan.blob_bytes}));
    packed = seconds_since(packing);

    // One submission: every image into the layout frames use it in, the
    // weights' upload, the arena zeroed, the noise field in the weights, and
    // a barrier before anything after it. Each frame fills the input before
    // any dispatch reads it, so the build leaves it unwritten.
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool.queueFamilyIndex = device_.family;
    DLSSLOP_TRY(vk_check(vkCreateCommandPool(d, &pool, nullptr, &setup.pool), "create the network's build commands"));
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = setup.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    DLSSLOP_TRY(vk_check(vkAllocateCommandBuffers(d, &alloc, &cmd), "allocate the network's build commands"));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    DLSSLOP_TRY(vk_check(vkBeginCommandBuffer(cmd, &begin), "begin the network's build commands"));
    std::vector<VkImageMemoryBarrier> layouts;
    settle(layouts, o.input, kSampled);
    for (const Image* i : {&o.answer, &o.second, &o.shown, &o.scratch, &o.depth}) settle(layouts, *i, kGeneral);
    for (const auto& pyramid : o.luma)
        for (const Image& i : pyramid) settle(layouts, i, kGeneral);
    for (const Image& i : o.flow) settle(layouts, i, kGeneral);
    for (const Image& i : o.history) settle(layouts, i, kGeneral);
    for (const Image& i : o.history_store) settle(layouts, i, kGeneral);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, uint32_t(layouts.size()), layouts.data());
    const VkBufferCopy weights{0, 0, plan.blob_bytes};
    vkCmdCopyBuffer(cmd, setup.staging.buffer, o.weights.buffer, 1, &weights);
    vkCmdFillBuffer(cmd, o.arena.buffer, 0, plan.arena_bytes, 0);
    const VkMemoryBarrier uploaded{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kCopyWrite, kRead | kWrite};
    vkCmdPipelineBarrier(cmd, kTransfer, kCompute, 0, 1, &uploaded, 0, nullptr, 0, nullptr);
    const size_t noise = size_t(Kernel::kNoiseField);
    dispatch(cmd, noise, o.kernel_sets[noise], (plan.noise.width + 7) / 8, (plan.noise.height + 7) / 8, &plan.noise,
             sizeof plan.noise);
    const VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                               VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &done, 0,
                         nullptr, 0, nullptr);
    DLSSLOP_TRY(vk_check(vkEndCommandBuffer(cmd), "record the network's build commands"));

    // Submitted under the queue's lock, and waited for outside it.
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    DLSSLOP_TRY(vk_check(vkCreateFence(d, &fence, nullptr, &setup.fence), "create the network's build fence"));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (device_.lock) device_.lock(device_.context);
    const VkResult submitted = vkQueueSubmit(device_.queue, 1, &submit, setup.fence);
    if (device_.unlock) device_.unlock(device_.context);
    DLSSLOP_TRY(vk_check(submitted, "submit the network's build"));
    setup.running = true;
    if (const VkResult waited = vkWaitForFences(d, 1, &setup.fence, VK_TRUE, 30'000'000'000ull); waited != VK_SUCCESS) {
        // The device may still read what the build made: it stays.
        objects_ = {};
        write_log(device_, "the network's build did not finish; its memory is left to the device");
        return vk_check(waited, "wait for the network's build");
    }
    setup.running = false;
    destroy(d, o.pipelines[noise]);
    return {};
}

void Runtime::dispatch(VkCommandBuffer cmd, size_t pipeline, VkDescriptorSet set, uint32_t x, uint32_t y,
                       const void* push, uint32_t bytes, uint32_t z) const
{
    const Pipeline& p = objects_.pipelines[pipeline];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
    vkCmdDispatch(cmd, x, y, z);
}

// STEP through PIPELINE with SET and PUSH, its words, then the barrier the
// plan puts after it.
void Runtime::run_step(VkCommandBuffer cmd, const Step& step, size_t pipeline, VkDescriptorSet set,
                       const uint32_t* push) const
{
    dispatch(cmd, pipeline, set, step.groups[0], step.groups[1], push, 4 * step.words, step.groups[2]);
    if (step.after != After::kNothing) compute_barrier(cmd, step.after == After::kInvalidate);
}

void Runtime::record(VkCommandBuffer cmd, VkImage frame, const Controls& c, bool reset)
{
    const Objects& o = objects_;
    State& s = state_;
    const uint32_t w = s.width, h = s.height, gx = (w + 7) / 8, gy = (h + 7) / 8;
    const uint32_t passes = std::clamp(c.passes, 1u, s.passes);
    // The frame into the input: copied in its own format, or blitted into
    // RGBA32F. With later passes, which overwrite the input, the first
    // pass's input is kept.
    barrier(cmd, frame, kTarget, kSource, kTransfer, kCopyWrite, kTransfer, kCopyRead);
    barrier(cmd, o.input.image, kSampled, kTarget, kCompute, kRead, kTransfer, kCopyWrite);
    transfer(cmd, frame, kSource, o.input.image, kTarget, w, h, !s.input_direct);
    barrier(cmd, o.input.image, kTarget, kSampled, kTransfer, kCopyWrite, kCompute, kRead);
    if (s.passes > 1) {
        barrier(cmd, o.input.image, kSampled, kSource, kCompute, kRead, kTransfer, kCopyRead);
        barrier(cmd, o.shown.image, kGeneral, kGeneral, kCompute, kRead, kTransfer, kCopyWrite);
        transfer(cmd, o.input.image, kSource, o.shown.image, kGeneral, w, h);
        barrier(cmd, o.shown.image, kGeneral, kGeneral, kTransfer, kCopyWrite, kCompute, kRead);
        barrier(cmd, o.input.image, kSource, kSampled, kTransfer, kCopyRead, kCompute, kRead);
    }
    compute_barrier(cmd);
    // The pre and post blocks, which carry the frame's controls, or with
    // motion their temporal variants.
    const Step& first = steps_.front();
    const Step& last = steps_.back();
    size_t pre = size_t(first.kernel), post = size_t(last.kernel);
    VkDescriptorSet pre_set = o.kernel_sets[pre], post_set = o.kernel_sets[post];
    if (s.motion) {
        pre = kPre;
        post = kPost;
        pre_set = o.pre_sets[s.current];
        post_set = o.post_sets[s.current];
        // The motion estimate: this frame's luma pyramid, then, with a last
        // frame to follow, the flow from its pyramid, coarse to fine; then
        // the parameters the pre and post blocks read, gated on that.
        const bool gate = s.latch && !reset;
        const uint32_t p = s.parity;
        for (uint32_t k = 0; k < kLevels; ++k) {
            const LumaPush push{s.level_width[k], s.level_height[k], k ? s.level_width[k - 1] : w,
                                k ? s.level_height[k - 1] : h, k ? 1u : 0u};
            dispatch(cmd, kLuma, o.luma_sets[p][k], (push.dst_w + 7) / 8, (push.dst_h + 7) / 8, &push, sizeof push);
            compute_barrier(cmd);
        }
        for (uint32_t k = kLevels; gate && k-- > 0;) {
            const bool coarsest = k == kLevels - 1;
            const FlowPush push{s.level_width[k], s.level_height[k], coarsest ? 1u : s.level_width[k + 1],
                                coarsest ? 1u : s.level_height[k + 1], kMotionRadius[k], coarsest, k == 0,
                                kMotionReject};
            dispatch(cmd, kFlow, o.flow_sets[p][k], (push.level_w + 7) / 8, (push.level_h + 7) / 8, &push, sizeof push);
            compute_barrier(cmd);
        }
        // The gate, the motion's scale, the history's strength and extent,
        // and no depth.
        const float params[8] = {gate ? 1.0f : 0.0f, 1.0f, 1.0f, 1.0f, float(w), float(h), 0.0f, 0.0f};
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, kRead, kCopyWrite,
                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, o.params.buffer, 0, VK_WHOLE_SIZE};
        vkCmdPipelineBarrier(cmd, kCompute, kTransfer, 0, 0, nullptr, 1, &b, 0, nullptr);
        vkCmdUpdateBuffer(cmd, o.params.buffer, 0, sizeof params, params);
        b.srcAccessMask = kCopyWrite;
        b.dstAccessMask = kRead;
        vkCmdPipelineBarrier(cmd, kTransfer, kCompute, 0, 0, nullptr, 1, &b, 0, nullptr);
    }
    bool in_scratch = false; // the pass stages left the answer in the scratch
    for (uint32_t pass = 0; pass < passes; ++pass) {
        if (pass) {
            // The last pass's answer is this pass's input, and this pass's
            // own history the history.
            const VkImage answer = in_scratch ? o.scratch.image : o.answer.image;
            barrier(cmd, answer, kGeneral, kSource, kCompute, kWrite, kTransfer, kCopyRead);
            barrier(cmd, o.input.image, kSampled, kTarget, kCompute, kRead, kTransfer, kCopyWrite);
            transfer(cmd, answer, kSource, o.input.image, kTarget, w, h);
            barrier(cmd, o.input.image, kTarget, kSampled, kTransfer, kCopyWrite, kCompute, kRead);
            barrier(cmd, answer, kSource, kGeneral, kTransfer, kCopyRead, kCompute, kRead | kWrite);
            if (s.motion) copy_general(cmd, o.history_store[pass].image, o.history[0].image, w, h);
        }
        uint32_t words[32];
        std::copy_n(push_.data() + first.push, first.words, words);
        patch_pre(words, c, pass > 0);
        run_step(cmd, first, pre, pre_set, words);
        for (size_t i = 1; i + 1 < steps_.size(); ++i) {
            const Step& step = steps_[i];
            const size_t kernel = size_t(step.kernel);
            run_step(cmd, step, kernel, o.kernel_sets[kernel], push_.data() + step.push);
        }
        std::copy_n(push_.data() + last.push, last.words, words);
        patch_post(words, c, s.post_alpha, s.rgba8);
        run_step(cmd, last, post, post_set, words);
        if (s.motion && !s.pingpong) {
            // The history is what the model wrote into the second output.
            const VkImage history = passes > 1 ? o.history_store[pass].image : o.history[0].image;
            barrier(cmd, o.second.image, kGeneral, kSource, kCompute, kWrite, kTransfer, kCopyRead);
            barrier(cmd, history, kGeneral, kGeneral, kCompute, kRead, kTransfer, kCopyWrite);
            transfer(cmd, o.second.image, kSource, history, kGeneral, w, h);
            barrier(cmd, history, kGeneral, kGeneral, kTransfer, kCopyWrite, kCompute, kRead);
            barrier(cmd, o.second.image, kSource, kGeneral, kTransfer, kCopyRead, kCompute, kRead | kWrite);
        }
        // dlsslop-amd's stages on the pass's answer, into the scratch and back.
        in_scratch = false;
        if (s.stages)
            for (const auto& [strength, anchor] : {std::pair(c.sharpness, 0u), std::pair(c.color_preserve, 1u)}) {
                if (strength == 0) continue;
                const StagePush push{w, h, strength, anchor};
                dispatch(cmd, kStages, o.stage_sets[in_scratch], gx, gy, &push, sizeof push);
                compute_barrier(cmd);
                in_scratch = !in_scratch;
            }
    }
    if (s.motion) {
        // The first pass's history is what the next frame's first pass reads.
        if (passes > 1) copy_general(cmd, o.history_store[0].image, o.history[0].image, w, h);
        s.latch = true;
        s.parity ^= 1;
        if (s.pingpong) s.current ^= 1;
    }
    // The answer back into the frame, with the frame's alpha, which one pass's
    // post block restores itself.
    const VkImage answer = in_scratch ? o.scratch.image : o.answer.image;
    if (!s.post_alpha) {
        const AlphaPush push{w, h, s.rgba8};
        dispatch(cmd, kAlpha, o.alpha_sets[in_scratch], gx, gy, &push, sizeof push);
    }
    barrier(cmd, answer, kGeneral, kSource, kCompute, kWrite, kTransfer, kCopyRead);
    barrier(cmd, frame, kSource, kTarget, kTransfer, kCopyRead, kTransfer, kCopyWrite);
    transfer(cmd, answer, kSource, frame, kTarget, w, h, !s.answer_direct);
    barrier(cmd, answer, kSource, kGeneral, kTransfer, kCopyRead, kCompute, kRead | kWrite);
    barrier(cmd, frame, kTarget, kSource, kTransfer, kCopyWrite, kTransfer, kCopyRead);
}

} // namespace dlsslop::vulkan
