// Execute the shipped composition SPIR-V with an identity model. This checks
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
constexpr size_t half_bytes = components * sizeof(uint16_t);

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

struct Result {
    std::array<float, components> output{};
    std::array<float, components> proxy{};
};

Result Run(Context& c, const std::array<float, components>& input, bool pq, bool sdr, float white,
           uint32_t reversible = 0, bool reduced = false) {
    Image native(c), proxy(c), original(c), output(c), small(c);
    native.Init(VK_FORMAT_R32G32B32A32_SFLOAT);
    proxy.Init(VK_FORMAT_R16G16B16A16_SFLOAT);
    original.Init(VK_FORMAT_R32G32B32A32_SFLOAT);
    output.Init(VK_FORMAT_R32G32B32A32_SFLOAT);
    if (reduced) small.Init(VK_FORMAT_R16G16B16A16_SFLOAT, width / 2, height / 2);
    Buffer staging(c);
    staging.Init(float_bytes * 2 + half_bytes);
    std::memcpy(staging.mapped, input.data(), float_bytes);
    VkCheck(vkResetCommandBuffer(c.cmd, 0), "reset command buffer");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkCheck(vkBeginCommandBuffer(c.cmd, &begin), "begin command buffer");
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
    constants.WhitePoint = white;
    constants.TransferStrength = constants.ColourStrength = 1;
    constants.MaxRatio = 2;
    constants.Passthrough = sdr;
    constants.MvScaleX = constants.MvScaleY = 1;
    constants.CompareSplit = 0.5f;
    constants.CompareZoom = 1;
    constants.Transfer = 2;
    constants.DebugScale = 1;
    constants.ReversibleMode = reversible;
    constants.ApplyModel = 1;
    constants.ExposurePreMul = 1;
    constants.HdrProxy = 2; // FP16 storage, display-encoded native HIP model input.
    constants.HdrTransfer = pq;
    constants.ColourTrust = 2;
    constants.RatioSmooth = 1;
    Check(c.pass->Dispatch(c.cmd, constants, width, height, native.view, VK_NULL_HANDLE,
                          VK_NULL_HANDLE, VK_NULL_HANDLE, proxy.view, original.view), "encode dispatch");
    proxy.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    original.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkImageView model_view = proxy.view;
    if (reduced) {
        small.Transition(VK_IMAGE_LAYOUT_GENERAL);
        constants.Mode = 2; // The shipped area-average downsample pass.
        constants.Width = width / 2;
        constants.Height = height / 2;
        Check(c.pass->Dispatch(c.cmd, constants, width / 2, height / 2, proxy.view, VK_NULL_HANDLE,
                              VK_NULL_HANDLE, VK_NULL_HANDLE, small.view, VK_NULL_HANDLE), "downsample dispatch");
        small.Transition(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        model_view = small.view;
        constants.Width = width;
        constants.Height = height;
    }
    constants.Mode = 1;
    // Binding the encoded proxy as the model output is an exact identity neural pass.
    Check(c.pass->Dispatch(c.cmd, constants, width, height, model_view, model_view,
                          original.view, VK_NULL_HANDLE, output.view, VK_NULL_HANDLE), "resolve dispatch");
    output.Transition(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    proxy.Transition(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    copy.bufferOffset = float_bytes;
    vkCmdCopyImageToBuffer(c.cmd, output.image, output.layout, staging.buffer, 1, &copy);
    copy.bufferOffset = float_bytes * 2;
    vkCmdCopyImageToBuffer(c.cmd, proxy.image, proxy.layout, staging.buffer, 1, &copy);
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
    Result result;
    const auto* bytes = static_cast<const unsigned char*>(staging.mapped);
    std::memcpy(result.output.data(), bytes + float_bytes, float_bytes);
    std::array<uint16_t, components> halves{};
    std::memcpy(halves.data(), bytes + float_bytes * 2, half_bytes);
    for (size_t i = 0; i < components; ++i) result.proxy[i] = HalfToFloat(halves[i]);
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
        const Result one = Run(context, linear, false, false, 1);
        // FP16 proxy rounding is magnified by the nonlinear encode/composition.
        // A 0.1% relative bound admits that error while catching highlight clipping
        // and a missing PQ transfer by orders of magnitude. PQ is checked in code units.
        Identity("linear HDR, soft knee", linear, one, 0.0001f, 0.001f);
        Check(*std::max_element(one.output.begin(), one.output.end()) > 9.9f,
              "HDR highlights were clipped to SDR range");
        Identity("HDR10 PQ/BT.2020, soft knee", pq, Run(context, pq, true, false, 1), 0.0002f, 0.0001f);
        Identity("SDR, FP16 encoded proxy", sdr, Run(context, sdr, false, true, 1), 0.0001f, 0.001f);
        const Result two = Run(context, linear, false, false, 2);
        Identity("linear HDR, white point 2", linear, two, 0.0001f, 0.001f);
        float difference = 0;
        for (size_t i = 0; i < components; ++i)
            if (i % 4 != 3) difference = std::max(difference, std::abs(one.proxy[i] - two.proxy[i]));
        Check(difference > 0.05f, "changing white point did not change encoded HDR proxy");
        std::printf("white-point control: maximum encoded proxy change %.9g\n", difference);
        Identity("linear HDR, reduced FP16 proxy", linear,
                 Run(context, linear, false, false, 1, 0, true), 0.0001f, 0.001f);
        Identity("HDR10 PQ/BT.2020, reduced FP16 proxy", pq,
                 Run(context, pq, true, false, 1, 0, true), 0.0002f, 0.0001f);
        // scRGB permits negative BT.709 components for colors outside that
        // gamut. Keep their native values even though the model proxy is bounded.
        auto scrgb = linear;
        constexpr float outside709[][3] = {{-0.1f, 1.0f, 0.2f}, {2.0f, -0.1f, 0.3f},
                                           {0.2f, 0.5f, -0.05f}, {-0.2f, 4.0f, -0.1f}};
        for (size_t pixel = 0; pixel < 4; ++pixel)
            for (size_t channel = 0; channel < 3; ++channel)
                scrgb[pixel * 4 + channel] = outside709[pixel][channel];
        const Result native_scrgb = Run(context, scrgb, false, false, 1);
        Identity("scRGB negative components, native FP16 proxy", scrgb, native_scrgb, 0.0001f, 0.001f);
        Check(native_scrgb.output[0] < 0, "native scRGB negative component was clipped");
        const Result reduced_scrgb = Run(context, scrgb, false, false, 1, 0, true);
        Identity("scRGB negative components, reduced FP16 proxy", scrgb, reduced_scrgb, 0.0001f, 0.001f);
        Check(reduced_scrgb.output[0] < 0, "reduced scRGB negative component was clipped");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hdr-shader-test: %s\n", error.what());
        return 1;
    }
}
