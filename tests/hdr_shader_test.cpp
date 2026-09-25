// Execute the shipped composition SPIR-V with identity and darkening models. This checks
// transfer/domain handling on a Vulkan device, not the HIP neural network.
#include <vulkan/vulkan.h>
#include "dlssnr_pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t width = 16, height = 8;
constexpr size_t components = width * height * 4;
constexpr size_t float_bytes = components * sizeof(float);
constexpr float max_ratio = 2;  // the resolve's guard

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void VkCheck(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(result));
}

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    dlssnr::InstanceTable instance_table;
    dlssnr::DeviceTable device_table;
    std::unique_ptr<dlssnr::DlssNrPass> pass;

    ~Context() {
        if (device) vkDeviceWaitIdle(device);
        pass.reset();
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    bool Init() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "dlsslop-amd-hdr-shader-test";
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
            for (uint32_t i = 0; i < families; ++i) {
                if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physical = candidate;
                    family = i;
                    break;
                }
            }
            if (physical) break;
        }
        if (!physical) return false;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        std::printf("HDR shader device: %s\n", properties.deviceName);
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
        instance_table.next_gipa = vkGetInstanceProcAddr;
        instance_table.Load(instance);
        device_table.next_dpa = vkGetDeviceProcAddr;
        device_table.Load(device);
        pass = std::make_unique<dlssnr::DlssNrPass>(&device_table, &instance_table, device, physical);
        Check(pass->CanRender(), "composition pass initialization failed");
        return true;
    }

    uint32_t MemoryType(uint32_t mask, VkMemoryPropertyFlags wanted) const {
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &properties);
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
            if ((mask & (1u << i)) && (properties.memoryTypes[i].propertyFlags & wanted) == wanted)
                return i;
        throw std::runtime_error("required memory type unavailable");
    }
};

struct Image {
    Context& context;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

    explicit Image(Context& c) : context(c) {}
    ~Image() {
        if (view) vkDestroyImageView(context.device, view, nullptr);
        if (image) vkDestroyImage(context.device, image, nullptr);
        if (memory) vkFreeMemory(context.device, memory, nullptr);
    }

    void Init(VkFormat format, uint32_t image_width = width, uint32_t image_height = height) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {image_width, image_height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkCheck(vkCreateImage(context.device, &info, nullptr, &image), "create image");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, 0);
        VkCheck(vkAllocateMemory(context.device, &allocation, nullptr, &memory), "allocate image memory");
        VkCheck(vkBindImageMemory(context.device, image, memory, 0), "bind image memory");
        VkImageViewCreateInfo vinfo{};
        vinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vinfo.image = image;
        vinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vinfo.format = format;
        vinfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkCheck(vkCreateImageView(context.device, &vinfo, nullptr, &view), "create image view");
    }

    void Transition(VkImageLayout next) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.oldLayout = layout;
        barrier.newLayout = next;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(context.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        layout = next;
    }
};

struct Buffer {
    Context& context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    explicit Buffer(Context& c) : context(c) {}
    ~Buffer() {
        if (mapped) vkUnmapMemory(context.device, memory);
        if (buffer) vkDestroyBuffer(context.device, buffer, nullptr);
        if (memory) vkFreeMemory(context.device, memory, nullptr);
    }
    void Init(size_t bytes) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkCheck(vkCreateBuffer(context.device, &info, nullptr, &buffer), "create buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkCheck(vkAllocateMemory(context.device, &allocation, nullptr, &memory), "allocate buffer memory");
        VkCheck(vkBindBufferMemory(context.device, buffer, memory, 0), "bind buffer memory");
        VkCheck(vkMapMemory(context.device, memory, 0, bytes, 0, &mapped), "map buffer");
    }
};

float HalfToFloat(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    const float sign = bits & 0x8000u ? -1.0f : 1.0f;
    if (!exponent) return sign * std::ldexp(float(mantissa), -24);
    Check(exponent != 31, "non-finite proxy half");
    return sign * std::ldexp(float(1024u + mantissa), int(exponent) - 25);
}

// Round to nearest even; finite values within binary16 range.
uint16_t FloatToHalf(float value) {
    const uint32_t sign = std::signbit(value) ? 0x8000u : 0u;
    const float magnitude = std::abs(value);
    if (magnitude < 0x1p-14f) return uint16_t(sign | uint32_t(std::nearbyint(magnitude * 0x1p24f)));
    uint32_t bits;
    std::memcpy(&bits, &magnitude, sizeof(bits));
    uint32_t half = (((bits >> 23) - 112u) << 10) | ((bits >> 13) & 1023u);
    const uint32_t rest = bits & 0x1fffu;
    half += rest > 0x1000u || (rest == 0x1000u && (half & 1u));
    Check(half < 0x7c00u, "value exceeds binary16 range");
    return uint16_t(sign | half);
}

// One channel of a proxy-format texel buffer: UNORM8 or binary16.
float LoadTexel(const unsigned char* bytes, size_t index, bool unorm) {
    if (unorm) return float(bytes[index]) / 255.0f;
    uint16_t half;
    std::memcpy(&half, bytes + index * 2, sizeof(half));
    return HalfToFloat(half);
}

void StoreTexel(unsigned char* bytes, size_t index, bool unorm, float value) {
    if (unorm) {
        bytes[index] = uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
        return;
    }
    const uint16_t half = FloatToHalf(value);
    std::memcpy(bytes + index * 2, &half, sizeof(half));
}

struct Options {
    bool pq = false, sdr = false, reduced = false;
    float white = 1, modelScale = 1;  // the model's answer is the proxy with its RGB scaled by this
    uint32_t transfer = 2, hdrProxy = 2;  // HdrProxy 2: display-encoded native HIP model input
    uint32_t debugView = 0;
    VkFormat proxyFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
};

struct Result {
    std::array<float, components> output{};
    std::array<float, components> proxy{};
};

void Begin(Context& c) {
    VkCheck(vkResetCommandBuffer(c.cmd, 0), "reset command buffer");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkCheck(vkBeginCommandBuffer(c.cmd, &begin), "begin command buffer");
}

void Submit(Context& c) {
    VkMemoryBarrier host{};
    host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &host, 0, nullptr, 0, nullptr);
    VkCheck(vkEndCommandBuffer(c.cmd), "end command buffer");
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &c.cmd;
    VkCheck(vkQueueSubmit(c.queue, 1, &submit, VK_NULL_HANDLE), "submit shader test");
    VkCheck(vkQueueWaitIdle(c.queue), "wait shader test");
}

// Encode (and downsample) in one submission, read the proxy back, derive the model's answer on the
// CPU, then resolve it in a second submission, as the layer does around the worker.
Result Run(Context& c, const std::array<float, components>& input, const Options& o) {
    const bool unorm = o.proxyFormat == VK_FORMAT_R8G8B8A8_UNORM;
    const uint32_t model_width = o.reduced ? width / 2 : width, model_height = o.reduced ? height / 2 : height;
    const size_t texel = unorm ? 1 : 2, model_components = size_t(model_width) * model_height * 4;
    const size_t proxy_offset = float_bytes * 2, small_offset = proxy_offset + components * texel;
    const size_t model_offset = small_offset + model_components * texel;
    Image native(c), proxy(c), original(c), output(c), small(c), model(c);
    native.Init(VK_FORMAT_R32G32B32A32_SFLOAT);
    proxy.Init(o.proxyFormat);
    original.Init(VK_FORMAT_R32G32B32A32_SFLOAT);
    output.Init(VK_FORMAT_R32G32B32A32_SFLOAT);
    if (o.reduced) small.Init(o.proxyFormat, model_width, model_height);
    model.Init(o.proxyFormat, model_width, model_height);
    Buffer staging(c);
    staging.Init(model_offset + model_components * texel);
    auto* bytes = static_cast<unsigned char*>(staging.mapped);
    std::memcpy(bytes, input.data(), float_bytes);

    Begin(c);
    native.Transition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(c.cmd, staging.buffer, native.image, native.layout, 1, &copy);
    native.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    proxy.Transition(VK_IMAGE_LAYOUT_GENERAL);
    original.Transition(VK_IMAGE_LAYOUT_GENERAL);
    output.Transition(VK_IMAGE_LAYOUT_GENERAL);
    DlssNrConstants constants{};
    constants.Width = constants.GuideWidth = width;
    constants.Height = constants.GuideHeight = height;
    constants.WhitePoint = o.white;
    constants.TransferStrength = constants.ColourStrength = 1;
    constants.MaxRatio = max_ratio;
    constants.Passthrough = o.sdr;
    constants.MvScaleX = constants.MvScaleY = 1;
    constants.CompareSplit = 0.5f;
    constants.CompareZoom = 1;
    constants.Transfer = o.transfer;
    constants.DebugView = o.debugView;
    constants.DebugScale = 1;
    constants.ApplyModel = 1;
    constants.ExposurePreMul = 1;
    constants.HdrProxy = o.hdrProxy;
    constants.HdrTransfer = o.pq;
    constants.ColourTrust = 2;
    constants.RatioSmooth = 1;
    Check(c.pass->Dispatch(c.cmd, constants, width, height, native.view, VK_NULL_HANDLE,
                          VK_NULL_HANDLE, VK_NULL_HANDLE, proxy.view, original.view), "encode dispatch");
    proxy.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    original.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Image& source = o.reduced ? small : proxy;
    if (o.reduced) {
        small.Transition(VK_IMAGE_LAYOUT_GENERAL);
        constants.Mode = DlssNrMode_Downsample;
        constants.Width = model_width;
        constants.Height = model_height;
        Check(c.pass->Dispatch(c.cmd, constants, model_width, model_height, proxy.view, VK_NULL_HANDLE,
                              VK_NULL_HANDLE, VK_NULL_HANDLE, small.view, VK_NULL_HANDLE), "downsample dispatch");
        small.Transition(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy reduced = copy;
        reduced.bufferOffset = small_offset;
        reduced.imageExtent = {model_width, model_height, 1};
        vkCmdCopyImageToBuffer(c.cmd, small.image, small.layout, staging.buffer, 1, &reduced);
        small.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        constants.Width = width;
        constants.Height = height;
    }
    proxy.Transition(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    copy.bufferOffset = proxy_offset;
    vkCmdCopyImageToBuffer(c.cmd, proxy.image, proxy.layout, staging.buffer, 1, &copy);
    proxy.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Submit(c);

    Result result;
    for (size_t i = 0; i < components; ++i) result.proxy[i] = LoadTexel(bytes + proxy_offset, i, unorm);
    // The shipped downsample is an exact area average: at half size, the mean of each 2x2 block,
    // with the alpha of the block's last texel. One rounding step in the proxy's format.
    for (size_t i = 0; o.reduced && i < model_components; ++i) {
        const size_t x = i / 4 % model_width * 2, y = i / 4 / model_width * 2, channel = i % 4;
        const auto at = [&](size_t dx, size_t dy) {
            return result.proxy[((y + dy) * width + x + dx) * 4 + channel];
        };
        const float expected = channel == 3 ? at(1, 1) : (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4;
        const float tolerance = channel == 3 ? 0 : unorm ? 1.0f / 255
                                                         : std::max(std::abs(expected) * 0x1p-10f, 0x1p-24f);
        const float actual = LoadTexel(bytes + small_offset, i, unorm);
        if (std::abs(actual - expected) <= tolerance) continue;
        std::fprintf(stderr, "downsample component %zu: expected %.9g, got %.9g\n", i, expected, actual);
        throw std::runtime_error("downsample is not the proxy's area average");
    }
    const unsigned char* answer = bytes + (o.reduced ? small_offset : proxy_offset);
    for (size_t i = 0; i < model_components; ++i)
        StoreTexel(bytes + model_offset, i, unorm, LoadTexel(answer, i, unorm) * (i % 4 == 3 ? 1 : o.modelScale));

    Begin(c);
    model.Transition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy upload = copy;
    upload.bufferOffset = model_offset;
    upload.imageExtent = {model_width, model_height, 1};
    vkCmdCopyBufferToImage(c.cmd, staging.buffer, model.image, model.layout, 1, &upload);
    model.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    constants.Mode = DlssNrMode_Resolve;
    Check(c.pass->Dispatch(c.cmd, constants, width, height, source.view, model.view,
                          original.view, VK_NULL_HANDLE, output.view, VK_NULL_HANDLE), "resolve dispatch");
    output.Transition(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    copy.bufferOffset = float_bytes;
    vkCmdCopyImageToBuffer(c.cmd, output.image, output.layout, staging.buffer, 1, &copy);
    Submit(c);
    std::memcpy(result.output.data(), bytes + float_bytes, float_bytes);
    return result;
}

void Identity(const char* name, const std::array<float, components>& input, const Result& result,
              float absolute_tolerance, float relative_tolerance) {
    float maximum = 0, maximum_relative = 0;
    for (size_t i = 0; i < components; ++i) {
        const float error = std::abs(input[i] - result.output[i]);
        maximum = std::max(maximum, error);
        if (input[i] != 0) maximum_relative = std::max(maximum_relative, error / std::abs(input[i]));
        if (i % 4 == 3) Check(result.output[i] == input[i], "native alpha was not preserved");
        if (!std::isfinite(result.output[i]) ||
            error > absolute_tolerance + relative_tolerance * std::abs(input[i])) {
            std::fprintf(stderr, "%s component %zu: input %.9g output %.9g error %.9g\n",
                         name, i, input[i], result.output[i], error);
            throw std::runtime_error("identity model did not preserve native frame");
        }
    }
    std::printf("%s: identity maximum absolute error %.9g, relative %.9g\n",
                name, maximum, maximum_relative);
}

float Pq(float linear) {
    const double q = std::pow(std::max(double(linear), 0.0), 2610.0 / 16384.0);
    return float(std::pow((3424.0 / 4096.0 + (2413.0 / 128.0) * q) /
                          (1.0 + (2392.0 / 128.0) * q), 2523.0 / 32.0));
}

float PqToLinear(float code) {
    const double p = std::pow(std::max(double(code), 0.0), 32.0 / 2523.0);
    return float(std::pow(std::max(p - 3424.0 / 4096.0, 0.0) / (2413.0 / 128.0 - (2392.0 / 128.0) * p),
                          16384.0 / 2610.0));
}

// A darkening model at a hard edge: the enlarged edit is far larger than the dark side's own light.
// No output channel may fall below the frame's own: zero, or a wide-gamut pixel's own negative
// BT.709 coordinate. The PQ store clamps at zero, so a PQ run can fail this only with a non-finite
// channel, and at transfer 2 checks nothing else.
//
// The matched residual (transfer 1) also keeps every pixel's luminance above input / (MaxRatio x
// 1.6), the resolve's 1/MaxRatio floor with the per-pixel band as margin. Native plus edit
// (transfer 2) does not: at the dark side the clamped sum is the zero vector, and the colour bound,
// which lets a grey pixel swing by sqrt(3) under a ColourTrust of 2, accepts that move to black
// over the resolve's floor, as upstream does.
void NoNegativeLight(const char* name, const std::array<float, components>& input, const Result& result,
                     bool pq, bool banded) {
    // BT.709 weights for linear scRGB, BT.2020 for PQ once decoded.
    const float weights[2][3] = {{0.2126f, 0.7152f, 0.0722f}, {0.2627f, 0.6780f, 0.0593f}};
    const float band = max_ratio * 1.6f;
    float lowest = INFINITY;
    for (size_t pixel = 0; pixel < components / 4; ++pixel) {
        float in = 0, out = 0;
        for (size_t channel = 0; channel < 3; ++channel) {
            const float value = result.output[pixel * 4 + channel];
            const float source = input[pixel * 4 + channel];
            if (!std::isfinite(value) || value < std::min(source, 0.0f)) {
                std::fprintf(stderr, "%s pixel %zu channel %zu: %.9g from %.9g\n", name, pixel, channel, value,
                             source);
                throw std::runtime_error("darkening edit drove an edge channel below the frame's own");
            }
            in += weights[pq][channel] * (pq ? PqToLinear(source) : source);
            out += weights[pq][channel] * (pq ? PqToLinear(value) : value);
        }
        lowest = std::min(lowest, out / in);
        if (!banded || out >= in / band) continue;
        std::fprintf(stderr, "%s pixel %zu: luminance %.9g from %.9g\n", name, pixel, out, in);
        throw std::runtime_error("darkening edit crushed an edge pixel below the luminance band");
    }
    std::printf("%s: no channel below the frame's own, lowest luminance ratio %.9g\n", name, lowest);
}

// Debug view 4 paints how far the colour bound let the model's colour through: red held back,
// green passed, together always paper white. Checked in BT.709 linear light, so a PQ swapchain must
// receive BT.2020 PQ codes for paper white rather than linear values read as up to 10,000 nits.
void ColourBoundView(const char* name, const std::array<float, components>& input, const Result& result,
                     bool pq, float white) {
    constexpr float to709[3][3] = {{1.660491f, -0.587641f, -0.072850f}, {-0.124550f, 1.132900f, -0.008349f},
                                   {-0.018151f, -0.100579f, 1.118730f}};
    for (size_t pixel = 0; pixel < components / 4; ++pixel) {
        const float* out = &result.output[pixel * 4];
        float rgb[3] = {out[0], out[1], out[2]};
        if (pq) {
            const float bt2020[3] = {PqToLinear(out[0]), PqToLinear(out[1]), PqToLinear(out[2])};
            for (size_t channel = 0; channel < 3; ++channel)
                rgb[channel] = to709[channel][0] * bt2020[0] + to709[channel][1] * bt2020[1] +
                               to709[channel][2] * bt2020[2];
        }
        const float tolerance = white * 0.001f;
        if (std::abs(rgb[0] + rgb[1] - white) <= tolerance && std::abs(rgb[2]) <= tolerance &&
            rgb[0] >= -tolerance && rgb[1] >= -tolerance && out[3] == input[pixel * 4 + 3])
            continue;
        std::fprintf(stderr, "%s pixel %zu: %.9g %.9g %.9g %.9g, linear %.9g %.9g %.9g, expected red plus "
                     "green %.9g\n", name, pixel, out[0], out[1], out[2], out[3], rgb[0], rgb[1], rgb[2], white);
        throw std::runtime_error("colour-bound view is not paper white split between red and green");
    }
    std::printf("%s: colour-bound view sums to paper white %.9g\n", name, white);
}

} // namespace

int main() {
    try {
        Context context;
        if (!context.Init()) {
            std::puts("SKIP: no Vulkan compute device with formatless storage writes");
            return 77;
        }
        std::array<float, components> linear{}, pq{}, sdr{};
        constexpr float levels[] = {0.003f, 0.03f, 0.2f, 0.7f, 1.0f, 2.0f, 4.0f, 10.0f};
        for (size_t pixel = 0; pixel < components / 4; ++pixel) {
            const float value = levels[pixel % 8];
            const float rgb[] = {value, value * 0.8f, value * 0.6f};
            for (size_t channel = 0; channel < 3; ++channel) {
                linear[pixel * 4 + channel] = rgb[channel];
                // Encode a BT.709 color as BT.2020 PQ, matching an HDR10 swapchain.
                constexpr float to2020[3][3] = {{0.627404f, 0.329283f, 0.043313f},
                    {0.069097f, 0.919540f, 0.011362f}, {0.016391f, 0.088013f, 0.895595f}};
                float component = 0;
                for (size_t k = 0; k < 3; ++k) component += to2020[channel][k] * rgb[k];
                pq[pixel * 4 + channel] = Pq(component * 0.0203f);
                sdr[pixel * 4 + channel] = rgb[channel] / 10.0f;
            }
            linear[pixel * 4 + 3] = pq[pixel * 4 + 3] = sdr[pixel * 4 + 3] = 0.75f;
        }
        pq[0] = 0.65f;
        pq[1] = 0.60f;
        pq[2] = 0.55f;
        // BT.2020 primaries lie outside the model's BT.709 gamut. An identity
        // model must still preserve their native chroma, including black channels.
        for (size_t mask = 1; mask < 8; ++mask)
            for (size_t channel = 0; channel < 3; ++channel)
                pq[mask * 4 + channel] = mask & (size_t(1) << channel) ? 0.7f : 0.0f;
        const Result one = Run(context, linear, {});
        // FP16 proxy rounding is magnified by the nonlinear encode/composition.
        // A 0.1% relative bound admits that error while catching highlight clipping
        // and a missing PQ transfer by orders of magnitude. PQ is checked in code units.
        Identity("linear HDR, soft knee", linear, one, 0.0001f, 0.001f);
        Check(*std::max_element(one.output.begin(), one.output.end()) > 9.9f,
              "HDR highlights were clipped to SDR range");
        Options hdr10;
        hdr10.pq = true;
        Identity("HDR10 PQ/BT.2020, soft knee", pq, Run(context, pq, hdr10), 0.0002f, 0.0001f);
        Options display;
        display.sdr = true;
        Identity("SDR, FP16 encoded proxy", sdr, Run(context, sdr, display), 0.0001f, 0.001f);
        Options brighter;
        brighter.white = 2;
        const Result two = Run(context, linear, brighter);
        Identity("linear HDR, white point 2", linear, two, 0.0001f, 0.001f);
        float difference = 0;
        for (size_t i = 0; i < components; ++i)
            if (i % 4 != 3) difference = std::max(difference, std::abs(one.proxy[i] - two.proxy[i]));
        Check(difference > 0.05f, "changing white point did not change encoded HDR proxy");
        std::printf("white-point control: maximum encoded proxy change %.9g\n", difference);
        Options reduced;
        reduced.reduced = true;
        Identity("linear HDR, reduced FP16 proxy", linear, Run(context, linear, reduced), 0.0001f, 0.001f);
        Options reduced_hdr10 = reduced;
        reduced_hdr10.pq = true;
        Identity("HDR10 PQ/BT.2020, reduced FP16 proxy", pq, Run(context, pq, reduced_hdr10), 0.0002f, 0.0001f);
        // Transfer 1, the matched residual, is the non-native default and a user option.
        Options matched = reduced;
        matched.transfer = 1;
        Identity("linear HDR, reduced FP16 proxy, transfer 1", linear, Run(context, linear, matched),
                 0.0001f, 0.001f);
        matched.pq = true;
        Identity("HDR10 PQ/BT.2020, reduced FP16 proxy, transfer 1", pq, Run(context, pq, matched),
                 0.0002f, 0.0001f);
        // HdrProxy 0 is native HIP's default: HDR transport off, an RGBA8 proxy. Eight bits move a
        // native highlight by up to a couple of percent; the reduced path lays only the (zero) edit
        // on the native frame and stays exact.
        Options rgba8;
        rgba8.hdrProxy = 0;
        rgba8.proxyFormat = VK_FORMAT_R8G8B8A8_UNORM;
        Identity("linear HDR, RGBA8 proxy", linear, Run(context, linear, rgba8), 0.03f, 0.025f);
        rgba8.reduced = true;
        Identity("linear HDR, reduced RGBA8 proxy", linear, Run(context, linear, rgba8), 0.0001f, 0.001f);
        // scRGB permits negative BT.709 components for colors outside that
        // gamut. Keep their native values even though the model proxy is bounded.
        auto scrgb = linear;
        constexpr float outside709[][3] = {{-0.1f, 1.0f, 0.2f}, {2.0f, -0.1f, 0.3f},
                                           {0.2f, 0.5f, -0.05f}, {-0.2f, 4.0f, -0.1f}};
        for (size_t pixel = 0; pixel < 4; ++pixel)
            for (size_t channel = 0; channel < 3; ++channel)
                scrgb[pixel * 4 + channel] = outside709[pixel][channel];
        const Result native_scrgb = Run(context, scrgb, {});
        Identity("scRGB negative components, native FP16 proxy", scrgb, native_scrgb, 0.0001f, 0.001f);
        Check(native_scrgb.output[0] < 0, "native scRGB negative component was clipped");
        const Result reduced_scrgb = Run(context, scrgb, reduced);
        Identity("scRGB negative components, reduced FP16 proxy", scrgb, reduced_scrgb, 0.0001f, 0.001f);
        Check(reduced_scrgb.output[0] < 0, "reduced scRGB negative component was clipped");
        // A model that darkens a column-by-column edge, below frame size: every transfer and
        // transport combination the native path can take.
        std::array<float, components> edge{}, edge_pq{};
        for (size_t i = 0; i < components; ++i) {
            edge[i] = i % 4 == 3 ? 0.75f : i / 4 % 2 ? 0.01f : 0.9f;
            edge_pq[i] = i % 4 == 3 ? 0.75f : Pq(edge[i] * 0.0203f);
        }
        for (unsigned variant = 0; variant < 8; ++variant) {
            Options darker;
            darker.reduced = true;
            darker.modelScale = 0.8f;
            darker.pq = variant & 1;
            darker.hdrProxy = variant & 2 ? 2 : 0;
            darker.proxyFormat = variant & 2 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
            darker.transfer = variant & 4 ? 2 : 1;
            char name[80];
            std::snprintf(name, sizeof(name), "darkening edge, %s, HdrProxy %u, transfer %u",
                          darker.pq ? "HDR10 PQ" : "linear HDR", darker.hdrProxy, darker.transfer);
            const auto& input = darker.pq ? edge_pq : edge;
            NoNegativeLight(name, input, Run(context, input, darker), darker.pq, darker.transfer == 1);
        }
        // A BT.2020 green in scRGB beside a bright white column, under native plus edit. The native
        // transports compose it from the frame's own hue; HdrProxy 1 carries it to the model with no
        // such fallback and relies on the sum's zero clamp to keep a channel above the frame's own.
        std::array<float, components> gamut_edge{};
        constexpr float green2020[] = {-0.5876f, 1.1329f, -0.1006f};
        for (size_t i = 0; i < components; ++i)
            gamut_edge[i] = i % 4 == 3 ? 0.75f : i / 4 % 2 ? green2020[i % 4] : 4.0f;
        for (unsigned variant = 0; variant < 6; ++variant) {
            Options darker;
            darker.reduced = true;
            darker.modelScale = variant & 1 ? 0.5f : 0.8f;
            darker.hdrProxy = variant >> 1;
            darker.proxyFormat = darker.hdrProxy ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
            char name[80];
            std::snprintf(name, sizeof(name), "wide-gamut edge x%.1f, HdrProxy %u, transfer 2",
                          darker.modelScale, darker.hdrProxy);
            NoNegativeLight(name, gamut_edge, Run(context, gamut_edge, darker), false, false);
        }
        Options bound_hdr10 = hdr10;
        bound_hdr10.debugView = 4;
        ColourBoundView("debug view 4, HDR10 PQ", pq, Run(context, pq, bound_hdr10), true, 0.0203f);
        Options bound_linear = brighter;
        bound_linear.debugView = 4;
        ColourBoundView("debug view 4, linear HDR, white point 2", linear, Run(context, linear, bound_linear),
                        false, 2);
        Options bound_sdr = display;
        bound_sdr.debugView = 4;
        ColourBoundView("debug view 4, SDR", sdr, Run(context, sdr, bound_sdr), false, 1);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hdr-shader-test: %s\n", error.what());
        return 1;
    }
}
