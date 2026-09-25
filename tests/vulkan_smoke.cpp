// SPDX-License-Identifier: AGPL-3.0-or-later
// Exercises Vulkan WSI -> layer capture -> native worker -> composition -> present.
// Requires X11 or VK_EXT_headless_surface and a worker launched with --test-identity.
// Tests transport and composition; does not execute HIP neural inference.
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <X11/Xlib.h>

#include "../upstream-layer/common/shm_protocol.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <getopt.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}

struct Context {
    Display* display = nullptr;
    Window window = 0;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkSemaphore rendered = VK_NULL_HANDLE;
    VkBuffer pattern = VK_NULL_HANDLE;
    VkDeviceMemory patternMemory = VK_NULL_HANDLE;
    int fd = -1;
    void* mapping = MAP_FAILED;

    ~Context() {
        if (device) vkDeviceWaitIdle(device);
        if (acquired) vkDestroySemaphore(device, acquired, nullptr);
        if (rendered) vkDestroySemaphore(device, rendered, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (pattern) vkDestroyBuffer(device, pattern, nullptr);
        if (patternMemory) vkFreeMemory(device, patternMemory, nullptr);
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        if (window) XDestroyWindow(display, window);
        if (display) XCloseDisplay(display);
        if (mapping != MAP_FAILED) munmap(mapping, ShmTotalBytes());
        if (fd >= 0) close(fd);
    }
};

class ContendingProducer {
    pid_t child_ = -1;
    int release_ = -1;
public:
    ContendingProducer(const char* path, bool enabled) {
        if (!enabled) return;
        const std::string name = std::string(path) + ".producer.lock";
        int ready[2], release[2];
        require(pipe(ready) == 0, "contention ready pipe failed");
        if (pipe(release)) {
            close(ready[0]); close(ready[1]);
            throw std::runtime_error("contention release pipe failed");
        }
        child_ = fork();
        if (!child_) {
            // No Vulkan or C++ work after forking the driver's multithreaded process.
            close(ready[0]); close(release[1]);
            const int fd = open(name.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
            const uint8_t acquired = fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0;
            if (write(ready[1], &acquired, 1) != 1) _exit(1);
            close(ready[1]);
            uint8_t unused;
            if (acquired) (void)read(release[0], &unused, 1);
            if (fd >= 0) close(fd);
            _exit(acquired ? 0 : 1);
        }
        close(ready[1]); close(release[0]);
        release_ = release[1];
        uint8_t acquired = 0;
        const bool success = child_ > 0 && read(ready[0], &acquired, 1) == 1 && acquired;
        close(ready[0]);
        if (!success) {
            finish();
            throw std::runtime_error("contention subprocess could not acquire producer lock");
        }
    }
    ~ContendingProducer() { finish(); }
    void finish() {
        if (release_ >= 0) { close(release_); release_ = -1; }
        if (child_ > 0) {
            int status;
            while (waitpid(child_, &status, 0) < 0 && errno == EINTR) {}
            child_ = -1;
        }
    }
};

static int smoke(bool headless, bool contention, bool reduced, bool bgra, bool proxy16) {
    Context c;
    const char* path = std::getenv("DLSSNR_SHM");
    require(path && *path, "set DLSSNR_SHM to a running --test-identity worker's channel");
    c.fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    require(c.fd >= 0, "open worker channel failed");
    c.mapping = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, c.fd, 0);
    require(c.mapping != MAP_FAILED, "map worker channel failed");
    auto* h = static_cast<ShmHeader*>(c.mapping);
    require(h->magic.load() == kShmMagic && h->version.load() == kShmVersion,
            "worker protocol mismatch");
    h->enabled.store(1);
    h->hdrMode.store(proxy16 ? kHdrForce : kHdrOff);
    h->colourMode.store(kColourAuto);
    h->workingScaleBits.store(FloatToBits(1.0f));
    h->compositionBypass.store(0);  // Exercise the resolve shader as well as the copy path.
    h->nativeModelMaxWidth.store(reduced ? 160 : 0);
    h->nativeModelMaxHeight.store(reduced ? 96 : 0);
    h->transfer.store(reduced ? 2 : 1);
    h->captureRequest.store(4);
    // Each smoke invocation starts a new layer process, whose frame counter
    // starts at zero even when the worker channel is reused between modes.
    h->layerFramesLo.store(0);
    h->layerFramesHi.store(0);
    h->controlSeq.fetch_add(1);

    constexpr uint32_t width = 320, height = 192, frames = 5;
    if (!headless) {
        c.display = XOpenDisplay(nullptr);
        require(c.display, "cannot open X11 display");
        c.window = XCreateSimpleWindow(c.display, DefaultRootWindow(c.display), 0, 0, width, height,
                                       0, 0, 0);
        require(c.window, "cannot create X11 window");
        XStoreName(c.display, c.window, "dlsslop-amd native transport smoke test");
        XMapWindow(c.display, c.window);
        XSync(c.display, False);
    }

    const char* instanceExtensions[] = { VK_KHR_SURFACE_EXTENSION_NAME,
        headless ? VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME : VK_KHR_XLIB_SURFACE_EXTENSION_NAME };
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "dlsslop-amd-transport-smoke";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instanceExtensions;
    check(vkCreateInstance(&ici, nullptr, &c.instance), "vkCreateInstance");

    if (headless) {
        auto create = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
            vkGetInstanceProcAddr(c.instance, "vkCreateHeadlessSurfaceEXT"));
        require(create, "VK_EXT_headless_surface is unavailable");
        VkHeadlessSurfaceCreateInfoEXT sci{};
        sci.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
        check(create(c.instance, &sci, nullptr, &c.surface), "vkCreateHeadlessSurfaceEXT");
    } else {
        VkXlibSurfaceCreateInfoKHR sci{};
        sci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        sci.dpy = c.display;
        sci.window = c.window;
        check(vkCreateXlibSurfaceKHR(c.instance, &sci, nullptr, &c.surface), "vkCreateXlibSurfaceKHR");
    }

    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(c.instance, &count, nullptr), "enumerate devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(c.instance, &count, devices.data()), "enumerate devices");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = 0;
    for (VkPhysicalDevice candidate : devices) {
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());
        for (uint32_t i = 0; i < count; ++i) {
            VkBool32 present = VK_FALSE;
            check(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, c.surface, &present),
                  "surface support");
            if (present && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                physical = candidate;
                family = i;
                break;
            }
        }
        if (physical) break;
    }
    require(physical, "no graphics/compute/present queue");
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    std::fprintf(stderr, "transport smoke GPU: %s\n", properties.deviceName);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    const char* deviceExtension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &deviceExtension;
    check(vkCreateDevice(physical, &dci, nullptr, &c.device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(c.device, family, 0, &queue);

    VkSurfaceCapabilitiesKHR caps{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, c.surface, &caps), "surface caps");
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, c.surface, &count, nullptr), "surface formats");
    std::vector<VkSurfaceFormatKHR> formats(count);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, c.surface, &count, formats.data()),
          "surface formats");
    auto format = std::find_if(formats.begin(), formats.end(), [bgra](const VkSurfaceFormatKHR& f) {
        return f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
               f.format == (bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM);
    });
    require(format != formats.end(), "no SDR UNORM surface format");
    require(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT, "no transfer destination");
    VkSwapchainCreateInfoKHR swap{};
    swap.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swap.surface = c.surface;
    swap.minImageCount = caps.minImageCount + 1;
    if (caps.maxImageCount) swap.minImageCount = std::min(swap.minImageCount, caps.maxImageCount);
    swap.imageFormat = format->format;
    swap.imageColorSpace = format->colorSpace;
    swap.imageExtent = caps.currentExtent.width == UINT32_MAX ? VkExtent2D{width, height}
                                                            : caps.currentExtent;
    swap.imageArrayLayers = 1;
    swap.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swap.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap.preTransform = caps.currentTransform;
    swap.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swap.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(c.device, &swap, nullptr, &c.swapchain), "vkCreateSwapchainKHR");
    check(vkGetSwapchainImagesKHR(c.device, c.swapchain, &count, nullptr), "swapchain images");
    std::vector<VkImage> images(count);
    check(vkGetSwapchainImagesKHR(c.device, c.swapchain, &count, images.data()), "swapchain images");

    if (reduced) {
        // One-pixel detail which the half-size proxy cannot retain. The native
        // plus edit resolve must nevertheless reproduce this exactly when the
        // worker returns its input unchanged.
        VkBufferCreateInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer.size = VkDeviceSize(swap.imageExtent.width) * swap.imageExtent.height * 4;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        check(vkCreateBuffer(c.device, &buffer, nullptr, &c.pattern), "create pattern buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(c.device, c.pattern, &requirements);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            const auto flags = memory.memoryTypes[i].propertyFlags;
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
        }
        require(type != UINT32_MAX, "no host coherent pattern buffer memory");
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        check(vkAllocateMemory(c.device, &allocation, nullptr, &c.patternMemory), "allocate pattern");
        check(vkBindBufferMemory(c.device, c.pattern, c.patternMemory, 0), "bind pattern");
        void* pointer = nullptr;
        check(vkMapMemory(c.device, c.patternMemory, 0, buffer.size, 0, &pointer), "map pattern");
        auto* pixels = static_cast<uint8_t*>(pointer);
        const bool bgra = format->format == VK_FORMAT_B8G8R8A8_UNORM;
        for (uint32_t y = 0; y < swap.imageExtent.height; ++y)
            for (uint32_t x = 0; x < swap.imageExtent.width; ++x) {
                auto* p = pixels + (size_t(y) * swap.imageExtent.width + x) * 4;
                p[bgra ? 2 : 0] = ((x ^ y) & 1) ? 208 : 48;
                p[1] = (x & 1) ? 64 : 176;
                p[bgra ? 0 : 2] = (y & 1) ? 192 : 80;
                p[3] = 255;
            }
        vkUnmapMemory(c.device, c.patternMemory);
    }

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(c.device, &pci, nullptr, &c.pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = c.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(c.device, &cai, &cb), "vkAllocateCommandBuffers");
    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    check(vkCreateSemaphore(c.device, &sem, nullptr, &c.acquired), "acquire semaphore");
    check(vkCreateSemaphore(c.device, &sem, nullptr, &c.rendered), "render semaphore");

    const uint64_t initialFrames = uint64_t(h->layerFramesLo.load()) |
                                   (uint64_t(h->layerFramesHi.load()) << 32);
    ContendingProducer contender(path, contention);
    for (uint32_t frame = 0; frame < frames + unsigned(contention); ++frame) {
        uint32_t index = 0;
        check(vkAcquireNextImageKHR(c.device, c.swapchain, UINT64_MAX, c.acquired,
                                    VK_NULL_HANDLE, &index), "acquire image");
        check(vkResetCommandBuffer(cb, 0), "reset command buffer");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check(vkBeginCommandBuffer(cb, &begin), "begin command buffer");
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = images[index];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkClearColorValue color{};
        color.float32[0] = float(frame + 1) / 8.0f;
        color.float32[1] = 0.25f;
        color.float32[2] = 0.75f;
        color.float32[3] = 1.0f;
        if (reduced) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {swap.imageExtent.width, swap.imageExtent.height, 1};
            vkCmdCopyBufferToImage(cb, c.pattern, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1, &copy);
        } else {
            vkCmdClearColorImage(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color,
                                 1, &barrier.subresourceRange);
        }
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &barrier);
        check(vkEndCommandBuffer(cb), "end command buffer");
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &c.acquired;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cb;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &c.rendered;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "submit frame");
        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &c.rendered;
        present.swapchainCount = 1;
        present.pSwapchains = &c.swapchain;
        present.pImageIndices = &index;
        const uint32_t previous = h->seq_req.load();
        check(vkQueuePresentKHR(queue, &present), "present frame");
        check(vkQueueWaitIdle(queue), "wait queue idle");
        const uint32_t request = h->seq_req.load();
        if (contention && !frame) {
            require(request == previous, "contending layer overwrote another producer's slot");
            require(h->layerFramesLo.load() == uint32_t(initialFrames),
                    "contending layer composed without owning slot");
            contender.finish();
            continue;
        }
        require(request != previous, "layer did not submit a frame");
        require(h->seq_resp.load(std::memory_order_acquire) == request &&
                h->seq_ok.load() == request, "worker did not successfully answer frame");
        const uint32_t expectedWidth = reduced ? 160 : swap.imageExtent.width;
        const uint32_t expectedHeight = reduced ? 96 : swap.imageExtent.height;
        require(h->answeredW.load() == expectedWidth &&
                h->answeredH.load() == expectedHeight, "response dimensions differ");
        const auto* input = static_cast<const uint8_t*>(c.mapping) + kHeaderBytes;
        const auto* output = input + kMaxFrame;
        for (unsigned channel = 0; !reduced && channel != 4; ++channel) {
            const int expected = int(std::lround(color.float32[channel] * 255.0f));
            if (proxy16) {
                uint16_t half;
                std::memcpy(&half, input + channel * 2, sizeof(half));
                const unsigned exponent = (half >> 10) & 31u;
                const float value = std::ldexp(float((half & 1023u) + (exponent ? 1024u : 0u)),
                                               int(exponent ? exponent : 1u) - 25);
                require(std::abs(value * 255.0f - float(expected)) <= 1.0f,
                        "FP16 proxy pixel differs from source code value");
            } else {
                require(std::abs(int(input[channel]) - expected) <= 1,
                        "GPU proxy pixel differs from clear color");
            }
        }
        require(h->hdrEncode.load() == (proxy16 ? 1u : 0u) &&
                h->proxyFormat.load() == (proxy16 ? kProxyRgba16F : kProxyRgba8),
                "request transport precision mismatch");
        const size_t bytes = size_t(expectedWidth) * expectedHeight * (proxy16 ? 8 : 4);
        require(std::memcmp(input, output, bytes) == 0, "identity worker corrupted frame pixels");
    }
    const uint64_t composed = uint64_t(h->layerFramesLo.load()) |
                              (uint64_t(h->layerFramesHi.load()) << 32);
    require(composed >= initialFrames + frames, "layer failed to compose all frames");
    std::printf("PASS: %u Vulkan frames captured, RGBA pixels verified, worker answered, "
                "composition submitted and presented (%ux%u).\n",
                frames, swap.imageExtent.width, swap.imageExtent.height);
    if (contention)
        std::printf("PASS: independent producer lock caused clean pass-through, then recovered.\n");
    if (reduced)
        std::printf("PASS: 320x192 checker frame used a 160x96 transport proxy.\n");
    return 0;
}

int main(int argc, char** argv) {
    try {
        bool headless = false, contention = false, reduced = false, bgra = false, proxy16 = false;
        const option options[] = {
            {"help", no_argument, nullptr, 'h'}, {"headless", no_argument, nullptr, 'H'},
            {"contention", no_argument, nullptr, 'c'}, {"reduced", no_argument, nullptr, 'r'},
            {"bgra", no_argument, nullptr, 'b'}, {"proxy16", no_argument, nullptr, 'f'},
            {nullptr, 0, nullptr, 0}
        };
        int value;
        while ((value = getopt_long(argc, argv, "hHcrbf", options, nullptr)) != -1) {
            switch (value) {
            case 'h':
                std::printf("Usage: vulkan-smoke [OPTIONS]\n"
                    "  -h, --help         Show help (default: no)\n"
                    "  -H, --headless     Use headless Vulkan surface (default: no, use X11)\n"
                    "  -c, --contention   Test producer lock contention (default: no)\n"
                    "  -r, --reduced      Use half-resolution model proxy (default: no)\n"
                    "  -b, --bgra         Select BGRA swapchain (default: no, RGBA)\n"
                    "  -f, --proxy16      Force FP16 encoded proxy transport (default: no, RGBA8)\n");
                return 0;
            case 'H': headless = true; break;
            case 'c': contention = true; break;
            case 'r': reduced = true; break;
            case 'b': bgra = true; break;
            case 'f': proxy16 = true; break;
            default: throw std::runtime_error("invalid option; use --help");
            }
        }
        require(optind == argc, "unexpected positional argument; use --help");
        return smoke(headless, contention, reduced, bgra, proxy16);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "transport smoke: %s\n", e.what());
        return 1;
    }
}
