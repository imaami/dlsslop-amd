// SPDX-License-Identifier: AGPL-3.0-or-later
// Exercises Vulkan WSI -> layer capture -> native worker -> composition -> present.
// Requires X11 or VK_EXT_headless_surface and a worker launched with --test-identity.
// Runs under the Khronos validation layer and exits 77 when it is not installed.
// Tests transport and composition; does not execute HIP neural inference.
// The lifecycle modes take the layer through what games do with swapchains, devices and
// instances: resizes, a second swapchain, a second device and a second instance.
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <X11/Xlib.h>

#include "shm_protocol.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

[[noreturn]] static void fail(const std::string& message) {
    std::fprintf(stderr, "transport smoke: %s\n", message.c_str());
    std::exit(1);
}

static void require(bool condition, const char* message) {
    if (!condition) fail(message);
}

static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        fail(std::string(operation) + ": " + std::to_string(result));
}

// The frame modes' swapchain, and the first swapchain of each lifecycle mode.
static constexpr VkExtent2D kExtent{320, 192};
static constexpr uint32_t kFrames = 5;
// The most swapchains one present names.
static constexpr uint32_t kPresentMax = 2;

enum Mode { kFrameMode, kResizeMode, kSecondSurfaceMode, kTwoDevicesMode, kTwoInstancesMode };

struct Options {
    bool headless = false, contention = false, reduced = false, bgra = false, proxy16 = false,
         linear = false, noWorker = false;
    Mode mode = kFrameMode;
    uint32_t resizes = 0;
};

// What every mode holds from start to end: the identity worker's channel, mapped, and the X11
// display unless the surfaces are headless.
struct Context {
    const char* path = nullptr;
    int fd = -1;
    void* mapping = MAP_FAILED;
    ShmHeader* h = nullptr;
    Display* display = nullptr;

    ~Context() {
        if (display) XCloseDisplay(display);
        if (mapping != MAP_FAILED) munmap(mapping, ShmTotalBytes());
        if (fd >= 0) close(fd);
    }
};

// Descriptor 0 is a file of the smoke's own, which no other open can produce. A teardown that
// closes a descriptor field holding 0 where it should hold -1 closes it, and the next open in the
// process gets the number for another file. main() takes it before it opens anything else, so that
// the memfd never replaces a descriptor of the smoke's that got the number 0.
static struct stat g_descriptor0;

static void TakeDescriptor0() {
    const int fd = memfd_create("vulkan-smoke-descriptor-0", 0);
    require(fd >= 0, "memfd_create failed");
    if (fd) {
        require(dup2(fd, 0) == 0, "cannot put the memfd on descriptor 0");
        close(fd);
    }
    require(fstat(0, &g_descriptor0) == 0, "fstat of descriptor 0 failed");
}

static void CheckDescriptor0(const char* teardown) {
    struct stat now{};
    if (fstat(0, &now) || now.st_dev != g_descriptor0.st_dev || now.st_ino != g_descriptor0.st_ino)
        fail(std::string(teardown) + " closed or replaced descriptor 0");
}

// Counts this process's descriptors on the channel file or its producer lock, and its mappings
// of the channel file. The kernel reports both by canonical path.
static std::pair<int, int> ChannelHandles(const char* channel) {
    char* canonical = realpath(channel, nullptr);
    require(canonical, "cannot resolve the channel path");
    const std::string path(canonical);
    std::free(canonical);
    int descriptors = 0, mappings = 0;
    if (DIR* directory = opendir("/proc/self/fd")) {
        while (const dirent* entry = readdir(directory)) {
            char target[4096];
            const std::string link = std::string("/proc/self/fd/") + entry->d_name;
            const ssize_t length = readlink(link.c_str(), target, sizeof target);
            descriptors += length > 0 && std::string(target, size_t(length)).rfind(path, 0) == 0;
        }
        closedir(directory);
    }
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);)
        mappings += line.size() > path.size() && line.compare(line.size() - path.size(), path.size(), path) == 0 &&
                    line[line.size() - path.size() - 1] == ' ';
    return {descriptors, mappings};
}

// Destroying every device releases everything the layer holds on the channel. What remains is
// this test's own descriptor and mapping.
static void CheckChannelReleased(const Context& c) {
    require(ChannelHandles(c.path) == std::make_pair(1, 1),
            "destroying the device left a layer descriptor or mapping of the channel");
}

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
            fail("contention release pipe failed");
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
            fail("contention subprocess could not acquire producer lock");
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

static void OpenChannel(Context& c, const Options& o) {
    c.path = std::getenv("DLSSNR_SHM");
    require(c.path && *c.path, "set DLSSNR_SHM to a running --test-identity worker's channel");
    c.fd = open(c.path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    require(c.fd >= 0, "open worker channel failed");
    c.mapping = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, c.fd, 0);
    require(c.mapping != MAP_FAILED, "map worker channel failed");
    ShmHeader* h = c.h = static_cast<ShmHeader*>(c.mapping);
    require(h->magic.load() == kShmMagic && h->version.load() == kShmVersion,
            "worker protocol mismatch");
    h->enabled.store(1);
    h->hdrMode.store(o.proxy16 ? kHdrForce : kHdrOff);
    h->colourMode.store(o.linear ? kColourLinearHdr : kColourAuto);
    // Linear HDR also exercises the measured white point's copy into the resolve's constants.
    h->whitePointSource.store(o.linear ? kWhitePointMeasured : kWhitePointManual);
    h->workingScaleBits.store(FloatToBits(1.0f));
    h->compositionBypass.store(0);  // Exercise the resolve shader as well as the copy path.
    h->nativeModelMaxWidth.store(o.reduced ? 160 : 0);
    h->nativeModelMaxHeight.store(o.reduced ? 96 : 0);
    h->transfer.store(o.reduced ? 2 : 1);
    // A lifecycle mode composes frames of several swapchains and captures none.
    h->captureRequest.store(o.noWorker || o.mode != kFrameMode ? 0 : 4);
    // Each smoke invocation starts a new layer process, whose frame counter
    // starts at zero even when the worker channel is reused between modes.
    h->layerFrames.store(0);
    h->layerMsBits.store(0);
    h->controlSeq.fetch_add(1);
}

// An instance that presents through X11 or a headless surface, with the Khronos validation layer
// below the implicit layer under test, so that it validates what that layer records too. False
// when the validation layer is not installed.
static bool CreateInstance(const Options& o, VkInstance* instance) {
    const char* instanceExtensions[] = { VK_KHR_SURFACE_EXTENSION_NAME,
        o.headless ? VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME : VK_KHR_XLIB_SURFACE_EXTENSION_NAME };
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "dlsslop-amd-transport-smoke";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instanceExtensions;
    const char* validation = "VK_LAYER_KHRONOS_validation";
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = &validation;
    const VkResult created = vkCreateInstance(&ici, nullptr, instance);
    if (created == VK_ERROR_LAYER_NOT_PRESENT) {
        std::fprintf(stderr, "transport smoke: skipped, %s is not installed\n", validation);
        return false;
    }
    check(created, "vkCreateInstance");
    return true;
}

// The loader unloads the layer with the last instance.
static void DestroyInstance(VkInstance instance) {
    vkDestroyInstance(instance, nullptr);
    CheckDescriptor0("vkDestroyInstance");
}

// A headless surface, or the surface of an X11 window of the given size.
struct Surface {
    Window window = 0;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
};

static Surface CreateSurface(const Context& c, VkInstance instance, VkExtent2D extent) {
    Surface s;
    if (!c.display) {
        auto create = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
        require(create, "VK_EXT_headless_surface is unavailable");
        VkHeadlessSurfaceCreateInfoEXT sci{};
        sci.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
        check(create(instance, &sci, nullptr, &s.surface), "vkCreateHeadlessSurfaceEXT");
        return s;
    }
    s.window = XCreateSimpleWindow(c.display, DefaultRootWindow(c.display), 0, 0, extent.width,
                                   extent.height, 0, 0, 0);
    require(s.window, "cannot create X11 window");
    XStoreName(c.display, s.window, "dlsslop-amd native transport smoke test");
    XMapWindow(c.display, s.window);
    XSync(c.display, False);
    VkXlibSurfaceCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    sci.dpy = c.display;
    sci.window = s.window;
    check(vkCreateXlibSurfaceKHR(instance, &sci, nullptr, &s.surface), "vkCreateXlibSurfaceKHR");
    return s;
}

// Resizes the window. A headless surface leaves the extent to the swapchain.
static void ResizeSurface(const Context& c, const Surface& s, VkExtent2D extent) {
    if (!c.display) return;
    XResizeWindow(c.display, s.window, extent.width, extent.height);
    XSync(c.display, False);
}

static void DestroySurface(const Context& c, VkInstance instance, const Surface& s) {
    vkDestroySurfaceKHR(instance, s.surface, nullptr);
    if (s.window) XDestroyWindow(c.display, s.window);
}

// A device that presents to the surface from a graphics and compute queue family. Software Vulkan
// is preferred; hardware only when no CPU device qualifies.
static VkPhysicalDevice PickDevice(VkInstance instance, VkSurfaceKHR surface, uint32_t* family) {
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
    std::stable_partition(devices.begin(), devices.end(), [](VkPhysicalDevice device) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    });
    for (VkPhysicalDevice candidate : devices) {
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());
        for (uint32_t i = 0; i < count; ++i) {
            VkBool32 present = VK_FALSE;
            check(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &present),
                  "surface support");
            if (!present || !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ||
                !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
                continue;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            std::fprintf(stderr, "transport smoke device: %s\n", properties.deviceName);
            *family = i;
            return candidate;
        }
    }
    fail("no graphics/compute/present queue");
}

// A second surface for a queue family PickDevice chose for the first.
static void RequireSupport(VkPhysicalDevice physical, uint32_t family, VkSurfaceKHR surface) {
    VkBool32 supported = VK_FALSE;
    check(vkGetPhysicalDeviceSurfaceSupportKHR(physical, family, surface, &supported), "surface support");
    require(supported, "the device cannot present to the second surface");
}

// A device with one queue, and what a frame on it needs.
struct Device {
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    // An acquire for each swapchain a present names, and the submit's signal the present waits on.
    VkSemaphore acquired[kPresentMax] = {};
    VkSemaphore rendered = VK_NULL_HANDLE;
    // The frames presented, and those the layer composed. The layer counts each device's composed
    // frames and publishes the count of the device that composed last as layerFrames.
    uint32_t presented = 0;
    uint64_t composed = 0;
};

// queue2 takes the queue from vkGetDeviceQueue2 instead of vkGetDeviceQueue: the layer has to
// remember either one to find the device of a present on that queue.
static Device CreateDevice(VkPhysicalDevice physical, uint32_t family, bool queue2) {
    Device d;
    d.physical = physical;
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
    check(vkCreateDevice(physical, &dci, nullptr, &d.device), "vkCreateDevice");
    VkDeviceQueueInfo2 qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
    qi.queueFamilyIndex = family;
    if (queue2)
        vkGetDeviceQueue2(d.device, &qi, &d.queue);
    else
        vkGetDeviceQueue(d.device, family, 0, &d.queue);
    require(d.queue, "the device returned no queue");

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(d.device, &pci, nullptr, &d.pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(d.device, &cai, &d.cb), "vkAllocateCommandBuffers");
    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (VkSemaphore& semaphore : d.acquired)
        check(vkCreateSemaphore(d.device, &sem, nullptr, &semaphore), "acquire semaphore");
    check(vkCreateSemaphore(d.device, &sem, nullptr, &d.rendered), "render semaphore");
    return d;
}

static void DestroyDevice(Device& d) {
    vkDeviceWaitIdle(d.device);
    for (VkSemaphore semaphore : d.acquired) vkDestroySemaphore(d.device, semaphore, nullptr);
    vkDestroySemaphore(d.device, d.rendered, nullptr);
    vkDestroyCommandPool(d.device, d.pool, nullptr);
    vkDestroyDevice(d.device, nullptr);
    CheckDescriptor0("vkDestroyDevice");
    d = Device{};
}

// The file of the object that a present on the device reaches first: the layer under test.
static std::string LayerObject(VkDevice device) {
    Dl_info info{};
    require(dladdr(reinterpret_cast<void*>(vkGetDeviceProcAddr(device, "vkQueuePresentKHR")), &info) &&
            info.dli_fname, "vkQueuePresentKHR is not in a loaded object");
    return info.dli_fname;
}

// The loader unloads the layer with the last instance, unless VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING
// is exactly "1". That value keeps every object loaded, which leak checkers need for full stacks.
static void CheckUnloaded(const std::string& object) {
    const char* kept = std::getenv("VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING");
    if (kept && !std::strcmp(kept, "1")) {
        std::printf("SKIP: VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1 kept the layer loaded; its unload "
                    "was not checked.\n");
        return;
    }
    void* handle = dlopen(object.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle) dlclose(handle);
    require(!handle, "the loader kept the layer loaded after the last instance");
    std::printf("PASS: the loader unloaded the layer with the last instance.\n");
}

// The layer hooks the present but leaves the application's other queue operations to the next layer
// down, unless the in-layer network is asked for: its build submits from a thread of its own. It
// never hooks the acquire.
static void CheckQueueHooks(VkDevice device) {
    const char* inLayer = std::getenv("DLSSLOP_LAYER_NETWORK");
    const bool network = inLayer && !std::strcmp(inLayer, "1");
    const std::string layer = LayerObject(device);
    for (const char* name : {"vkQueueSubmit", "vkQueueWaitIdle", "vkQueueBindSparse", "vkDeviceWaitIdle"}) {
        Dl_info next{};
        require(dladdr(reinterpret_cast<void*>(vkGetDeviceProcAddr(device, name)), &next) &&
                (layer == next.dli_fname) == network,
                network ? "the layer leaves queue operations unhooked while the in-layer network builds"
                        : "the layer hooks queue operations without the in-layer network");
    }
    Dl_info acquire{};
    require(dladdr(reinterpret_cast<void*>(vkGetDeviceProcAddr(device, "vkAcquireNextImageKHR")), &acquire) &&
                layer != acquire.dli_fname,
            "the layer hooks vkAcquireNextImageKHR");
}

struct Swapchain {
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkExtent2D extent{};
    std::vector<VkImage> images;
};

static VkSurfaceFormatKHR PickFormat(VkPhysicalDevice physical, VkSurfaceKHR surface, bool bgra) {
    uint32_t count = 0;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr), "surface formats");
    std::vector<VkSurfaceFormatKHR> formats(count);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats.data()),
          "surface formats");
    auto format = std::find_if(formats.begin(), formats.end(), [bgra](const VkSurfaceFormatKHR& f) {
        return f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
               f.format == (bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM);
    });
    require(format != formats.end(), "no SDR UNORM surface format");
    return *format;
}

// A FIFO swapchain the application writes with transfers, at the given extent: the window's, or
// the one a headless surface leaves to the swapchain.
static Swapchain CreateSwapchain(const Device& d, VkSurfaceKHR surface, VkSurfaceFormatKHR format,
                                 VkExtent2D extent, VkSwapchainKHR old) {
    VkSurfaceCapabilitiesKHR caps{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d.physical, surface, &caps), "surface caps");
    require(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT, "no transfer destination");
    VkSwapchainCreateInfoKHR swap{};
    swap.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swap.surface = surface;
    swap.minImageCount = caps.minImageCount + 1;
    if (caps.maxImageCount) swap.minImageCount = std::min(swap.minImageCount, caps.maxImageCount);
    swap.imageFormat = format.format;
    swap.imageColorSpace = format.colorSpace;
    swap.imageExtent = caps.currentExtent.width == UINT32_MAX ? extent : caps.currentExtent;
    require(swap.imageExtent.width == extent.width && swap.imageExtent.height == extent.height,
            "the surface did not take the extent asked for");
    swap.imageArrayLayers = 1;
    swap.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swap.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap.preTransform = caps.currentTransform;
    swap.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swap.clipped = VK_TRUE;
    swap.oldSwapchain = old;
    Swapchain s;
    s.extent = extent;
    check(vkCreateSwapchainKHR(d.device, &swap, nullptr, &s.handle), "vkCreateSwapchainKHR");
    uint32_t count = 0;
    check(vkGetSwapchainImagesKHR(d.device, s.handle, &count, nullptr), "swapchain images");
    s.images.resize(count);
    check(vkGetSwapchainImagesKHR(d.device, s.handle, &count, s.images.data()), "swapchain images");
    return s;
}

static void DestroySwapchain(const Device& d, Swapchain& s) {
    vkDeviceWaitIdle(d.device);
    vkDestroySwapchainKHR(d.device, s.handle, nullptr);
    CheckDescriptor0("vkDestroySwapchainKHR");
    s = Swapchain{};
}

static VkClearColorValue FrameColour(uint32_t frame) {
    VkClearColorValue color{};
    color.float32[0] = float(frame % 8 + 1) / 8.0f;
    color.float32[1] = 0.25f;
    color.float32[2] = 0.75f;
    color.float32[3] = 1.0f;
    return color;
}

// The application's frame in a swapchain image: a clear, or a copy of the pattern when there is
// one. Leaves the image in PRESENT_SRC_KHR.
static void RecordFrame(VkCommandBuffer cb, VkImage image, VkExtent2D extent, const VkClearColorValue& color,
                        VkBuffer pattern) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // The stage the acquire semaphore is waited at, so the transition follows the acquire.
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &barrier);
    if (pattern) {
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyBufferToImage(cb, pattern, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    } else {
        vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color,
                             1, &barrier.subresourceRange);
    }
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Acquires an image of each swapchain, draws the frame into each, presents them in one
// vkQueuePresentKHR and waits for the queue. Returns how long the present blocked.
static std::chrono::steady_clock::duration Draw(Device& d, std::initializer_list<const Swapchain*> swapchains,
                                                const VkClearColorValue& color, VkBuffer pattern) {
    require(swapchains.size() <= kPresentMax, "too many swapchains for one present");
    VkSwapchainKHR handles[kPresentMax];
    uint32_t indices[kPresentMax];
    const VkPipelineStageFlags waitStages[kPresentMax] = {VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                          VK_PIPELINE_STAGE_TRANSFER_BIT};
    uint32_t count = 0;
    for (const Swapchain* s : swapchains) {
        handles[count] = s->handle;
        check(vkAcquireNextImageKHR(d.device, s->handle, UINT64_MAX, d.acquired[count],
                                    VK_NULL_HANDLE, &indices[count]), "acquire image");
        ++count;
    }
    check(vkResetCommandBuffer(d.cb, 0), "reset command buffer");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    check(vkBeginCommandBuffer(d.cb, &begin), "begin command buffer");
    count = 0;
    for (const Swapchain* s : swapchains) {
        RecordFrame(d.cb, s->images[indices[count]], s->extent, color, pattern);
        ++count;
    }
    check(vkEndCommandBuffer(d.cb), "end command buffer");
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = count;
    submit.pWaitSemaphores = d.acquired;
    submit.pWaitDstStageMask = waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &d.cb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &d.rendered;
    check(vkQueueSubmit(d.queue, 1, &submit, VK_NULL_HANDLE), "submit frame");
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &d.rendered;
    present.swapchainCount = count;
    present.pSwapchains = handles;
    present.pImageIndices = indices;
    const auto started = std::chrono::steady_clock::now();
    check(vkQueuePresentKHR(d.queue, &present), "present frame");
    const auto blocked = std::chrono::steady_clock::now() - started;
    check(vkQueueWaitIdle(d.queue), "wait queue idle");
    ++d.presented;
    return blocked;
}

// The identity worker accepted request at width x height and returned the proxy byte for byte. A
// proxy of code values holds the clear colour; a pattern or linear light passes no colour.
static void CheckAnswer(const Context& c, uint32_t request, uint32_t width, uint32_t height,
                        const VkClearColorValue* color, bool proxy16) {
    const ShmHeader* h = c.h;
    require(h->seq_resp.load(std::memory_order_acquire) == request &&
            h->seq_ok.load() == request, "worker did not successfully answer frame");
    require(h->answeredW.load() == width && h->answeredH.load() == height, "response dimensions differ");
    const auto* input = static_cast<const uint8_t*>(c.mapping) + kHeaderBytes;
    const auto* output = input + kMaxFrame;
    for (unsigned channel = 0; color && channel != 4; ++channel) {
        const int expected = int(std::lround(color->float32[channel] * 255.0f));
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
    require(h->hdrEncode.load() == (proxy16 ? 1u : 0u), "request transport precision mismatch");
    const size_t bytes = size_t(width) * height * (proxy16 ? 8 : 4);
    require(std::memcmp(input, output, bytes) == 0, "identity worker corrupted frame pixels");
}

// Presents a frame on each swapchain in one vkQueuePresentKHR. The swapchain expected to compose
// publishes exactly one request, answered at its extent with the frame unchanged, and its device's
// frame count grows by one; when none is expected, neither the request nor the count moves.
static void Present(const Context& c, Device& d, std::initializer_list<const Swapchain*> swapchains,
                    const Swapchain* composes) {
    const uint32_t previous = c.h->seq_req.load();
    const uint64_t frames = c.h->layerFrames.load();
    const VkClearColorValue color = FrameColour(d.presented);
    Draw(d, swapchains, color, VK_NULL_HANDLE);
    const uint32_t request = c.h->seq_req.load();
    if (!composes) {
        require(request == previous && c.h->layerFrames.load() == frames,
                "the layer composed a present that holds no primary swapchain");
        return;
    }
    require(request == previous + 1, "the layer did not publish exactly one request for the present");
    CheckAnswer(c, request, composes->extent.width, composes->extent.height, &color, false);
    require(c.h->layerFrames.load() == ++d.composed, "the layer's frame count did not grow by one");
}

static int smoke(Context& c, const Options& o) {
    VkInstance instance = VK_NULL_HANDLE;
    if (!CreateInstance(o, &instance)) return 77;
    const Surface surface = CreateSurface(c, instance, kExtent);
    uint32_t family = 0;
    const VkPhysicalDevice physical = PickDevice(instance, surface.surface, &family);
    Device d = CreateDevice(physical, family, false);
    CheckQueueHooks(d.device);
    const VkSurfaceFormatKHR format = PickFormat(physical, surface.surface, o.bgra);
    Swapchain s = CreateSwapchain(d, surface.surface, format, kExtent, VK_NULL_HANDLE);

    VkBuffer pattern = VK_NULL_HANDLE;
    VkDeviceMemory patternMemory = VK_NULL_HANDLE;
    if (o.reduced) {
        // One-pixel detail which the half-size proxy cannot retain. The native
        // plus edit resolve must nevertheless reproduce this exactly when the
        // worker returns its input unchanged.
        VkBufferCreateInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer.size = VkDeviceSize(s.extent.width) * s.extent.height * 4;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        check(vkCreateBuffer(d.device, &buffer, nullptr, &pattern), "create pattern buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(d.device, pattern, &requirements);
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
        check(vkAllocateMemory(d.device, &allocation, nullptr, &patternMemory), "allocate pattern");
        check(vkBindBufferMemory(d.device, pattern, patternMemory, 0), "bind pattern");
        void* pointer = nullptr;
        check(vkMapMemory(d.device, patternMemory, 0, buffer.size, 0, &pointer), "map pattern");
        auto* pixels = static_cast<uint8_t*>(pointer);
        const bool bgra = format.format == VK_FORMAT_B8G8R8A8_UNORM;
        for (uint32_t y = 0; y < s.extent.height; ++y)
            for (uint32_t x = 0; x < s.extent.width; ++x) {
                auto* p = pixels + (size_t(y) * s.extent.width + x) * 4;
                p[bgra ? 2 : 0] = ((x ^ y) & 1) ? 208 : 48;
                p[1] = (x & 1) ? 64 : 176;
                p[bgra ? 0 : 2] = (y & 1) ? 192 : 80;
                p[3] = 255;
            }
        vkUnmapMemory(d.device, patternMemory);
    }

    ShmHeader* h = c.h;
    const uint64_t initialFrames = h->layerFrames.load();
    ContendingProducer contender(c.path, o.contention);
    for (uint32_t frame = 0; frame < kFrames + unsigned(o.contention); ++frame) {
        const uint32_t previous = h->seq_req.load();
        const uint32_t answered = h->seq_resp.load();
        const VkClearColorValue color = FrameColour(frame);
        const auto blocked = Draw(d, {&s}, color, pattern);
        const uint32_t request = h->seq_req.load();
        if (o.noWorker) {
            // A killed worker still reads as running: the free slot takes one request,
            // which the layer gives up on within the heartbeat window. A worker that
            // is not running gets no request at all.
            const bool running = h->helperState.load() == kHelperRunning;
            require(request == previous + unsigned(running && previous == answered),
                    running ? "layer did not publish exactly one request to the silent worker"
                            : "layer published a request to a worker that is not running");
            require(h->seq_resp.load() == answered, "a dead worker answered");
            require(blocked < std::chrono::seconds(2), "layer blocked on a worker that cannot answer");
            continue;
        }
        if (o.contention && !frame) {
            require(request == previous, "contending layer overwrote another producer's slot");
            require(h->layerFrames.load() == initialFrames,
                    "contending layer composed without owning slot");
            contender.finish();
            continue;
        }
        // One request a present: a second copy of the layer in the chain stays inert.
        require(request == previous + 1, "layer did not submit exactly one request for the frame");
        // A linear-HDR proxy carries encoded light, not the swapchain's code values.
        CheckAnswer(c, request, o.reduced ? 160 : s.extent.width, o.reduced ? 96 : s.extent.height,
                    o.reduced || o.linear ? nullptr : &color, o.proxy16);
    }
    const uint64_t composed = h->layerFrames.load();
    DestroySwapchain(d, s);
    if (pattern) vkDestroyBuffer(d.device, pattern, nullptr);
    if (patternMemory) vkFreeMemory(d.device, patternMemory, nullptr);
    DestroyDevice(d);
    CheckChannelReleased(c);
    std::printf("PASS: destroying the device released the layer's channel descriptor and mappings.\n");
    DestroySurface(c, instance, surface);
    DestroyInstance(instance);
    std::printf("PASS: descriptor 0 survived every teardown.\n");
    if (o.noWorker) {
        std::printf("PASS: %u frames presented without blocking on a worker that cannot answer.\n",
                    kFrames);
        return 0;
    }
    require(composed == initialFrames + kFrames, "layer did not compose each frame exactly once");
    // Measured on every composed frame, not only under DLSSNR_TIME.
    require(BitsToFloat(h->layerMsBits.load()) > 0, "layer did not publish its per-frame cost");
    std::printf("PASS: %u Vulkan frames captured, RGBA pixels verified, worker answered, "
                "composition submitted and presented (%ux%u).\n",
                kFrames, kExtent.width, kExtent.height);
    if (o.contention)
        std::printf("PASS: independent producer lock caused clean pass-through, then recovered.\n");
    if (o.reduced)
        std::printf("PASS: 320x192 checker frame used a 160x96 transport proxy.\n");
    return 0;
}

// Re-creates the swapchain with oldSwapchain, alternately larger and smaller, as a window resize
// does. A larger successor takes the primary role from its predecessor at once; a smaller one
// presents untouched until the predecessor is destroyed.
static int Resizes(Context& c, const Options& o) {
    static constexpr VkExtent2D kSizes[] = {{288, 176}, {352, 208}};
    VkInstance instance = VK_NULL_HANDLE;
    if (!CreateInstance(o, &instance)) return 77;
    const Surface surface = CreateSurface(c, instance, kExtent);
    uint32_t family = 0;
    const VkPhysicalDevice physical = PickDevice(instance, surface.surface, &family);
    Device d = CreateDevice(physical, family, false);
    const VkSurfaceFormatKHR format = PickFormat(physical, surface.surface, o.bgra);
    Swapchain s = CreateSwapchain(d, surface.surface, format, kExtent, VK_NULL_HANDLE);
    Present(c, d, {&s}, &s);
    for (uint32_t i = 1; i <= o.resizes; ++i) {
        const VkExtent2D extent = kSizes[i % 2];
        ResizeSurface(c, surface, extent);
        Swapchain next = CreateSwapchain(d, surface.surface, format, extent, s.handle);
        const bool larger = extent.width * extent.height > s.extent.width * s.extent.height;
        Present(c, d, {&next}, larger ? &next : nullptr);
        DestroySwapchain(d, s);
        Present(c, d, {&next}, &next);
        s = std::move(next);
    }
    DestroySwapchain(d, s);
    DestroyDevice(d);
    CheckChannelReleased(c);
    DestroySurface(c, instance, surface);
    DestroyInstance(instance);
    std::printf("PASS: %u swapchains re-created with oldSwapchain composed their frames once their "
                "predecessor was gone, at once when larger.\n", o.resizes);
    return 0;
}

// A smaller second swapchain beside the first, as an overlay's. The larger drives the channel. The
// smaller presents untouched, alone and named first in one present with the larger, until the
// larger is destroyed and it takes the primary role over.
static int TwoSurfaces(Context& c, const Options& o) {
    static constexpr VkExtent2D kSmaller{256, 160};
    VkInstance instance = VK_NULL_HANDLE;
    if (!CreateInstance(o, &instance)) return 77;
    const Surface larger = CreateSurface(c, instance, kExtent);
    const Surface smaller = CreateSurface(c, instance, kSmaller);
    uint32_t family = 0;
    const VkPhysicalDevice physical = PickDevice(instance, larger.surface, &family);
    RequireSupport(physical, family, smaller.surface);
    Device d = CreateDevice(physical, family, false);
    Swapchain primary = CreateSwapchain(d, larger.surface, PickFormat(physical, larger.surface, o.bgra),
                                        kExtent, VK_NULL_HANDLE);
    Swapchain overlay = CreateSwapchain(d, smaller.surface, PickFormat(physical, smaller.surface, o.bgra),
                                        kSmaller, VK_NULL_HANDLE);
    for (uint32_t frame = 0; frame < 2; ++frame) {
        Present(c, d, {&primary}, &primary);
        Present(c, d, {&overlay}, nullptr);
        Present(c, d, {&overlay, &primary}, &primary);
    }
    DestroySwapchain(d, primary);
    Present(c, d, {&overlay}, &overlay);
    Present(c, d, {&overlay}, &overlay);
    DestroySwapchain(d, overlay);
    DestroyDevice(d);
    CheckChannelReleased(c);
    DestroySurface(c, instance, smaller);
    DestroySurface(c, instance, larger);
    DestroyInstance(instance);
    std::printf("PASS: the smaller of two swapchains presented untouched until the larger was "
                "destroyed, then composed.\n");
    return 0;
}

// Two devices on one instance, as a game beside its launcher, each presenting to a surface of its
// own. The first composes while the second presents untouched beside it. Once the first's swapchain
// is destroyed, the second composes, before and after the first device is destroyed. With two
// devices the layer finds a present's device by its queue among those it saw handed out, so it has
// to have seen the second's, which comes from vkGetDeviceQueue2.
static int TwoDevices(Context& c, const Options& o) {
    VkInstance instance = VK_NULL_HANDLE;
    if (!CreateInstance(o, &instance)) return 77;
    const Surface firstSurface = CreateSurface(c, instance, kExtent);
    const Surface secondSurface = CreateSurface(c, instance, kExtent);
    uint32_t family = 0;
    const VkPhysicalDevice physical = PickDevice(instance, firstSurface.surface, &family);
    RequireSupport(physical, family, secondSurface.surface);
    Device first = CreateDevice(physical, family, false);
    Device second = CreateDevice(physical, family, true);
    CheckQueueHooks(second.device);
    Swapchain firstSwapchain = CreateSwapchain(first, firstSurface.surface,
                                               PickFormat(physical, firstSurface.surface, o.bgra), kExtent,
                                               VK_NULL_HANDLE);
    Swapchain s = CreateSwapchain(second, secondSurface.surface,
                                  PickFormat(physical, secondSurface.surface, o.bgra), kExtent, VK_NULL_HANDLE);
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        Present(c, first, {&firstSwapchain}, &firstSwapchain);
        Present(c, second, {&s}, nullptr);
    }
    DestroySwapchain(first, firstSwapchain);
    for (uint32_t frame = 0; frame < kFrames; ++frame) Present(c, second, {&s}, &s);
    DestroyDevice(first);
    Present(c, second, {&s}, &s);
    DestroySwapchain(second, s);
    DestroyDevice(second);
    CheckChannelReleased(c);
    DestroySurface(c, instance, secondSurface);
    DestroySurface(c, instance, firstSurface);
    DestroyInstance(instance);
    std::printf("PASS: two devices composed in turn, the second on a queue from vkGetDeviceQueue2, "
                "beside the first and after it.\n");
    return 0;
}

// An instance with one device that presents to one surface.
struct Presenter {
    VkInstance instance = VK_NULL_HANDLE;
    Surface surface;
    Device device;
    Swapchain swapchain;
};

static bool OpenPresenter(const Context& c, const Options& o, VkExtent2D extent, Presenter* p) {
    if (!CreateInstance(o, &p->instance)) return false;
    p->surface = CreateSurface(c, p->instance, extent);
    uint32_t family = 0;
    const VkPhysicalDevice physical = PickDevice(p->instance, p->surface.surface, &family);
    p->device = CreateDevice(physical, family, false);
    p->swapchain = CreateSwapchain(p->device, p->surface.surface, PickFormat(physical, p->surface.surface, o.bgra),
                                   extent, VK_NULL_HANDLE);
    return true;
}

static void ClosePresenter(const Context& c, Presenter& p) {
    DestroySwapchain(p.device, p.swapchain);
    DestroyDevice(p.device);
    DestroySurface(c, p.instance, p.surface);
    DestroyInstance(p.instance);
}

// Instances alive two at a time, then none, then one again. The loader keeps the layer loaded while
// any instance lives: the second instance keeps presenting after the first is destroyed, and a
// third, which may get the first's handles, presents beside the second. Destroying the last
// instance unloads the layer, and the next instance loads it anew. Each instance's swapchain is
// larger than those before it, so that it takes the primary role over.
static int TwoInstances(Context& c, const Options& o) {
    static constexpr VkExtent2D kExtents[] = {kExtent, {352, 208}, {384, 224}};
    Presenter first, second, third, fresh;
    if (!OpenPresenter(c, o, kExtents[0], &first)) return 77;
    Present(c, first.device, {&first.swapchain}, &first.swapchain);
    const std::string layer = LayerObject(first.device.device);
    if (!OpenPresenter(c, o, kExtents[1], &second)) return 77;
    Present(c, second.device, {&second.swapchain}, &second.swapchain);
    ClosePresenter(c, first);
    for (uint32_t frame = 0; frame < kFrames; ++frame)
        Present(c, second.device, {&second.swapchain}, &second.swapchain);
    if (!OpenPresenter(c, o, kExtents[2], &third)) return 77;
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        Present(c, third.device, {&third.swapchain}, &third.swapchain);
        Present(c, second.device, {&second.swapchain}, nullptr);
    }
    ClosePresenter(c, second);
    Present(c, third.device, {&third.swapchain}, &third.swapchain);
    ClosePresenter(c, third);
    CheckChannelReleased(c);
    CheckUnloaded(layer);

    if (!OpenPresenter(c, o, kExtent, &fresh)) return 77;
    for (uint32_t frame = 0; frame < kFrames; ++frame)
        Present(c, fresh.device, {&fresh.swapchain}, &fresh.swapchain);
    ClosePresenter(c, fresh);
    CheckChannelReleased(c);
    std::printf("PASS: instances composed two at a time and after the first was destroyed; a new "
                "instance composed after all were destroyed.\n");
    return 0;
}

static void SelectMode(Options& o, Mode mode) {
    require(o.mode == kFrameMode || o.mode == mode, "choose one lifecycle mode; use --help");
    o.mode = mode;
}

int main(int argc, char** argv) {
    Options o;
    const option options[] = {
        {"help", no_argument, nullptr, 'h'}, {"headless", no_argument, nullptr, 'H'},
        {"contention", no_argument, nullptr, 'c'}, {"reduced", no_argument, nullptr, 'r'},
        {"bgra", no_argument, nullptr, 'b'}, {"proxy16", no_argument, nullptr, 'f'},
        {"linear-hdr", no_argument, nullptr, 'l'}, {"no-worker", no_argument, nullptr, 'n'},
        {"resizes", required_argument, nullptr, 'R'}, {"second-surface", no_argument, nullptr, 's'},
        {"two-devices", no_argument, nullptr, 'd'}, {"two-instances", no_argument, nullptr, 'i'},
        {nullptr, 0, nullptr, 0}
    };
    int value;
    while ((value = getopt_long(argc, argv, "hHcrbflnR:sdi", options, nullptr)) != -1) {
        switch (value) {
        case 'h':
            std::printf("Usage: vulkan-smoke [OPTIONS]\n"
                "  -h, --help            Show help (default: no)\n"
                "  -H, --headless        Use headless Vulkan surface (default: no, use X11)\n"
                "  -c, --contention      Test producer lock contention (default: no)\n"
                "  -r, --reduced         Use half-resolution model proxy (default: no)\n"
                "  -b, --bgra            Select BGRA swapchain (default: no, RGBA)\n"
                "  -f, --proxy16         Force FP16 encoded proxy transport (default: no, RGBA8)\n"
                "  -l, --linear-hdr      Compose in linear-HDR colour mode (default: no, colour auto)\n"
                "  -n, --no-worker       Expect a stopped or killed worker: present without\n"
                "                        blocking or composing (default: no, expect answers)\n"
                "Lifecycle modes, one at a time, without -c, -r, -f, -l and -n:\n"
                "  -R, --resizes N       Re-create the swapchain N times with oldSwapchain,\n"
                "                        alternately larger and smaller (default: 0, none)\n"
                "  -s, --second-surface  Present a smaller second swapchain, alone and with the\n"
                "                        first, then after it (default: no)\n"
                "  -d, --two-devices     Present from two devices in turn, the second on a\n"
                "                        queue from vkGetDeviceQueue2 (default: no)\n"
                "  -i, --two-instances   Present from instances two at a time, the second after\n"
                "                        the first is destroyed, then from a new one after all\n"
                "                        are destroyed (default: no)\n");
            return 0;
        case 'H': o.headless = true; break;
        case 'c': o.contention = true; break;
        case 'r': o.reduced = true; break;
        case 'b': o.bgra = true; break;
        case 'f': o.proxy16 = true; break;
        case 'l': o.linear = true; break;
        case 'n': o.noWorker = true; break;
        case 'R': {
            char* end = nullptr;
            const unsigned long resizes = std::strtoul(optarg, &end, 10);
            require(std::isdigit(static_cast<unsigned char>(*optarg)) && !*end && resizes && resizes <= 1000,
                    "--resizes takes a count from 1 to 1000");
            o.resizes = uint32_t(resizes);
            SelectMode(o, kResizeMode);
            break;
        }
        case 's': SelectMode(o, kSecondSurfaceMode); break;
        case 'd': SelectMode(o, kTwoDevicesMode); break;
        case 'i': SelectMode(o, kTwoInstancesMode); break;
        default: fail("invalid option; use --help");
        }
    }
    require(optind == argc, "unexpected positional argument; use --help");
    require(o.mode == kFrameMode || !(o.contention || o.reduced || o.proxy16 || o.linear || o.noWorker),
            "a lifecycle mode takes none of -c, -r, -f, -l and -n; use --help");
    static int (*const kModes[])(Context&, const Options&) = {smoke, Resizes, TwoSurfaces, TwoDevices,
                                                              TwoInstances};
    TakeDescriptor0();
    Context c;
    OpenChannel(c, o);
    if (!o.headless) {
        c.display = XOpenDisplay(nullptr);
        require(c.display, "cannot open X11 display");
    }
    return kModes[o.mode](c, o);
}
