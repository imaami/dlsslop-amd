/** @file
 *
 * Exercises Vulkan WSI -> layer capture -> native worker -> composition -> present. Requires X11 or
 * VK_EXT_headless_surface and a worker launched with --test-identity. Runs under the Khronos
 * validation layer and exits 77 when it is not installed. Tests transport and composition; does not
 * execute HIP neural inference. The lifecycle modes take the layer through what games do with
 * swapchains, devices and instances: resizes, a second swapchain, a second device and a second
 * instance.
 */
// SPDX-License-Identifier: AGPL-3.0-or-later

// Before <vulkan/vulkan.h>, which declares the Xlib surface's entry points under it.
#define VK_USE_PLATFORM_XLIB_KHR

#include <X11/Xlib.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "shm_protocol.h"
#include "support.h"

/** @brief The frame modes' swapchain, and the first swapchain of each lifecycle mode. */
static constexpr VkExtent2D EXTENT = {320, 192};

/** @brief The frames each frame mode presents. */
static constexpr uint32_t FRAMES = 5;

/** @brief The most swapchains one present names. */
static constexpr uint32_t PRESENT_MAX = 2;

/** @brief The nanoseconds in a second. */
static constexpr int64_t NS_PER_SECOND = 1000000000;

/** @brief Ends the smoke with a message. The prefix and the newline are pasted into the format, so
 *         that glibc writes the line to unbuffered stderr in one write: it stages up to 128 bytes,
 *         more than the smoke's longest line.
 *
 * @param fmt A printf format, a string literal.
 * @param ... The format's arguments.
 */
#define fail(fmt, ...) \
	(fprintf(stderr, "transport smoke: " fmt "\n" __VA_OPT__(,) __VA_ARGS__), exit(1))

/** @brief Ends the smoke with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param message   What went wrong if it does not hold.
 */
static void
require (bool        condition,
         char const *message)
{
	if (!condition)
		fail("%s", message);
}

/** @brief Ends the smoke unless a Vulkan call succeeded; a suboptimal swapchain counts as success.
 *
 * @param result    What the call returned.
 * @param operation What it did.
 */
static void
check (VkResult    result,
       char const *operation)
{
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		fail("%s: %d", operation, (int)result);
}

/** @brief The monotonic clock.
 *
 * @return Its time in nanoseconds.
 */
static int64_t
monotonic_ns (void)
{
	struct timespec now;
	require(!clock_gettime(CLOCK_MONOTONIC, &now), "clock_gettime failed");
	return now.tv_sec * NS_PER_SECOND + now.tv_nsec;
}

/** @brief The smoke's modes: the frame modes, then one for each lifecycle. */
enum mode : uint8_t {
	MODE_FRAME,
	MODE_RESIZE,
	MODE_SECOND_SURFACE,
	MODE_TWO_DEVICES,
	MODE_TWO_INSTANCES
};

/** @brief The command line. */
struct options {
	uint32_t  resizes;    //!< The swapchain re-creations of the resize mode.
	enum mode mode;       //!< The mode.
	bool      headless;   //!< Whether the surfaces are headless instead of X11 windows.
	bool      contention; //!< Whether another producer holds the lock for the first frame.
	bool      reduced;    //!< Whether the model's proxy is half the frame.
	bool      bgra;       //!< Whether the swapchains are BGRA instead of RGBA.
	bool      proxy16;    //!< Whether the proxy is FP16 instead of RGBA8.
	bool      linear;     //!< Whether the composition is in linear-HDR colour mode.
	bool      no_worker;  //!< Whether the worker is stopped or killed.
};

/** @brief What every mode holds from start to end: the identity worker's channel, mapped, with its
 *         canonical path, and the X11 display unless the surfaces are headless.
 */
struct context {
	char const       *path;      //!< The channel's path, DLSSNR_SHM.
	char             *canonical; //!< The channel's canonical path, as the kernel reports it, or nullptr.
	char             *target;    //!< Room for a link as long as the canonical path, or nullptr.
	struct ShmHeader *h;         //!< The channel, mapped, or nullptr.
	Display          *display;   //!< The X11 display, or nullptr.
	size_t            length;    //!< The canonical path's length.
	int               fd;        //!< The channel's descriptor, or -1.
};

/** @brief Descriptor 0 as main() took it.
 *
 * Descriptor 0 is a file of the smoke's own, which no other open can produce. A teardown that
 * closes a descriptor field holding 0 where it should hold -1 closes it, and the next open in the
 * process gets the number for another file. main() takes it before it opens anything else, so that
 * the memfd never replaces a descriptor of the smoke's that got the number 0.
 */
static struct stat descriptor0;

/** @brief Puts a memfd of the smoke's own on descriptor 0 and records it. */
static void
take_descriptor0 (void)
{
	int const fd = memfd_create("vulkan-smoke-descriptor-0", 0);
	require(fd >= 0, "memfd_create failed");
	if (fd) {
		require(dup2(fd, 0) == 0, "cannot put the memfd on descriptor 0");
		require(!close(fd), "cannot close the memfd");
	}
	require(fstat(0, &descriptor0) == 0, "fstat of descriptor 0 failed");
}

/** @brief Ends the smoke unless descriptor 0 is still the smoke's memfd.
 *
 * @param teardown The call that ran last.
 */
static void
check_descriptor0 (char const *teardown)
{
	struct stat now;
	if (fstat(0, &now) || now.st_dev != descriptor0.st_dev || now.st_ino != descriptor0.st_ino)
		fail("%s closed or replaced descriptor 0", teardown);
}

/** @brief This process's hold on the channel. */
struct channel_handles {
	uint32_t descriptors; //!< Descriptors of the channel file or of its producer lock.
	uint32_t mappings;    //!< Mappings of the channel file.
};

/** @brief Counts this process's descriptors on the channel file or its producer lock, and its
 *         mappings of the channel file. The kernel reports both by canonical path.
 *
 * @param c The context, whose link room this overwrites.
 * @return  The counts.
 */
static struct channel_handles
channel_handles (struct context const *c)
{
	char const *const path = c->canonical;
	size_t const length = c->length;
	char *const target = c->target;
	struct channel_handles held = {};

	DIR *const directory = opendir("/proc/self/fd");
	require(directory, "cannot open /proc/self/fd");
	int const fds = dirfd(directory);
	require(fds >= 0, "cannot read /proc/self/fd");
	for (;;) {
		errno = 0;
		struct dirent const *const entry = readdir(directory);
		if (!entry) {
			require(!errno, "cannot read /proc/self/fd");
			break;
		}
		// A link reads as far as the canonical path, which is all that a match needs.
		ssize_t const n = readlinkat(fds, entry->d_name, target, length);
		held.descriptors += n == (ssize_t)length && !memcmp(target, path, length);
	}
	require(!closedir(directory), "cannot close /proc/self/fd");

	FILE *const maps = fopen("/proc/self/maps", "re");
	require(maps, "cannot open /proc/self/maps");
	char *line = nullptr;
	size_t room = 0;
	for (ssize_t n; (n = getline(&line, &room, maps)) > 0;) {
		size_t const end = (size_t)n - (line[n - 1] == '\n');
		held.mappings += end > length && !memcmp(line + end - length, path, length)
		                 && line[end - length - 1] == ' ';
	}
	free(line);
	line = nullptr;
	bool const failed = ferror(maps);
	require(!fclose(maps) && !failed, "cannot read /proc/self/maps");
	return held;
}

/** @brief Ends the smoke unless destroying every device released everything the layer holds on the
 *         channel: what remains is this test's own descriptor and mapping.
 *
 * @param c The context.
 */
static void
check_channel_released (struct context const *c)
{
	struct channel_handles const held = channel_handles(c);
	require(held.descriptors == 1 && held.mappings == 1,
	        "destroying the device left a layer descriptor or mapping of the channel");
}

/** @brief A process of its own that holds the channel's producer lock until it is released. */
struct contending_producer {
	pid_t child;   //!< The process, or -1.
	int   release; //!< The pipe whose closing releases the lock, or -1.
};

/** @brief Releases the lock and waits for the process to end.
 *
 * @param p The producer.
 */
static void
contending_producer_fini (struct contending_producer *p)
{
	if (p->release >= 0) {
		require(!close(p->release), "cannot close the contention release pipe");
		p->release = -1;
	}
	if (p->child > 0) {
		while (waitpid(p->child, nullptr, 0) < 0 && errno == EINTR) {}
		p->child = -1;
	}
}

/** @brief Starts a process that takes the channel's producer lock, and waits until it holds it.
 *
 * @param dest    The producer.
 * @param path    The channel's path.
 * @param enabled Whether to start it; if not, the producer holds nothing.
 */
static void
contending_producer_init (struct contending_producer *dest,
                          char const                 *path,
                          bool                        enabled)
{
	*dest = (struct contending_producer){.child = -1, .release = -1};
	if (!enabled)
		return;

	char *name = support_format(nullptr, "%s.producer.lock", path);
	require(name, "out of memory");
	int ready[2];
	int release[2];
	require(pipe(ready) == 0, "contention ready pipe failed");
	if (pipe(release)) {
		close(ready[0]);
		close(ready[1]);
		fail("contention release pipe failed");
	}
	pid_t const child = fork();
	if (!child) {
		// Only async-signal-safe calls after forking the driver's multithreaded process.
		close(ready[0]);
		close(release[1]);
		int const fd = open(name, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
		uint8_t const acquired = fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0;
		if (write(ready[1], &acquired, 1) != 1)
			_exit(1);
		close(ready[1]);
		uint8_t unused;
		if (acquired)
			(void)read(release[0], &unused, 1);
		if (fd >= 0)
			close(fd);
		_exit(acquired ? 0 : 1);
	}
	free(name);
	name = nullptr;
	// A failed fork leaves -1, which is what no process reads as.
	dest->child = child;
	dest->release = release[1];
	require(!close(ready[1]) && !close(release[0]), "cannot close the contention pipes");
	uint8_t acquired;
	bool const success = child > 0 && read(ready[0], &acquired, 1) == 1 && acquired;
	require(!close(ready[0]), "cannot close the contention ready pipe");
	if (!success) {
		contending_producer_fini(dest);
		fail("contention subprocess could not acquire producer lock");
	}
}

/** @brief Maps the identity worker's channel, resolves its canonical path, sets it up for the mode,
 *         and opens the X11 display unless the surfaces are headless.
 *
 * @param dest The context.
 * @param o    The command line.
 */
static void
context_init (struct context       *dest,
              struct options const *o)
{
	*dest = (struct context){.fd = -1};
	dest->path = getenv("DLSSNR_SHM");
	require(dest->path && *dest->path, "set DLSSNR_SHM to a running --test-identity worker's channel");
	int const fd = open(dest->path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
	require(fd >= 0, "open worker channel failed");
	dest->fd = fd;
	dest->canonical = realpath(dest->path, nullptr);
	require(dest->canonical, "cannot resolve the channel path");
	dest->length = strlen(dest->canonical);
	dest->target = malloc(dest->length);
	require(dest->target, "out of memory");
	void *const mapping = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, dest->fd, 0);
	require(mapping != MAP_FAILED, "map worker channel failed");
	struct ShmHeader *const h = dest->h = mapping;
	require(atomic_load(&h->magic) == kShmMagic && atomic_load(&h->version) == kShmVersion,
	        "worker protocol mismatch");
	atomic_store(&h->enabled, 1);
	atomic_store(&h->hdrMode, o->proxy16 ? kHdrForce : kHdrOff);
	atomic_store(&h->colourMode, o->linear ? kColourLinearHdr : kColourAuto);
	// Linear HDR also exercises the measured white point's copy into the resolve's constants.
	atomic_store(&h->whitePointSource, o->linear ? kWhitePointMeasured : kWhitePointManual);
	atomic_store(&h->workingScaleBits, FloatToBits(1.0f));
	atomic_store(&h->compositionBypass, 0); // Exercise the resolve shader as well as the copy path.
	atomic_store(&h->nativeModelMaxWidth, o->reduced ? 160 : 0);
	atomic_store(&h->nativeModelMaxHeight, o->reduced ? 96 : 0);
	atomic_store(&h->transfer, o->reduced ? 2 : 1);
	// A lifecycle mode composes frames of several swapchains and captures none.
	atomic_store(&h->captureRequest, o->no_worker || o->mode != MODE_FRAME ? 0 : 4);
	// Each smoke invocation starts a new layer process, whose frame counter
	// starts at zero even when the worker channel is reused between modes.
	atomic_store(&h->layerFrames, 0);
	atomic_store(&h->layerMsBits, 0);
	atomic_fetch_add(&h->controlSeq, 1);
	if (!o->headless) {
		dest->display = XOpenDisplay(nullptr);
		require(dest->display, "cannot open X11 display");
	}
}

/** @brief Closes the display, unmaps the channel, frees its canonical path and closes it.
 *
 * @param c The context.
 */
static void
context_fini (struct context *c)
{
	if (c->display) {
		XCloseDisplay(c->display);
		c->display = nullptr;
	}
	if (c->h) {
		require(!munmap(c->h, ShmTotalBytes()), "cannot unmap the worker channel");
		c->h = nullptr;
	}
	free(c->target);
	c->target = nullptr;
	free(c->canonical);
	c->canonical = nullptr;
	c->length = 0;
	if (c->fd >= 0) {
		require(!close(c->fd), "cannot close the worker channel");
		c->fd = -1;
	}
	c->path = nullptr;
}

/** @brief Creates an instance that presents through X11 or a headless surface, with the Khronos
 *         validation layer below the implicit layer under test, so that it validates what that
 *         layer records too.
 *
 * @param o The command line.
 * @return  The instance, or VK_NULL_HANDLE if the validation layer is not installed.
 */
static VkInstance
create_instance (struct options const *o)
{
	char const *const extensions[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
		o->headless ? VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME : VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
	};
	VkApplicationInfo const app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "dlsslop-amd-transport-smoke",
		.apiVersion = VK_API_VERSION_1_1,
	};
	char const *const validation = "VK_LAYER_KHRONOS_validation";
	VkInstanceCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
		.enabledLayerCount = 1,
		.ppEnabledLayerNames = &validation,
		.enabledExtensionCount = sizeof extensions / sizeof *extensions,
		.ppEnabledExtensionNames = extensions,
	};
	VkInstance instance;
	VkResult const created = vkCreateInstance(&info, nullptr, &instance);
	if (created == VK_ERROR_LAYER_NOT_PRESENT) {
		fprintf(stderr, "transport smoke: skipped, %s is not installed\n", validation);
		return VK_NULL_HANDLE;
	}
	check(created, "vkCreateInstance");
	return instance;
}

/** @brief Destroys an instance; the loader unloads the layer with the last one.
 *
 * @param instance The instance.
 */
static void
destroy_instance (VkInstance instance)
{
	vkDestroyInstance(instance, nullptr);
	check_descriptor0("vkDestroyInstance");
}

/** @brief A headless surface, or the surface of an X11 window. */
struct surface {
	VkSurfaceKHR surface; //!< The surface.
	Window       window;  //!< The window, or 0 for a headless surface.
};

/** @brief Creates a headless surface, or the surface of an X11 window of the given size.
 *
 * @param dest     The surface.
 * @param c        The context.
 * @param instance The instance.
 * @param extent   The window's size.
 */
static void
surface_init (struct surface       *dest,
              struct context const *c,
              VkInstance            instance,
              VkExtent2D            extent)
{
	*dest = (struct surface){};
	if (!c->display) {
		PFN_vkVoidFunction const entry = vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
		require(entry, "VK_EXT_headless_surface is unavailable");
		PFN_vkCreateHeadlessSurfaceEXT const create = (PFN_vkCreateHeadlessSurfaceEXT)entry;
		VkHeadlessSurfaceCreateInfoEXT const info = {
			.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
		};
		VkSurfaceKHR surface;
		check(create(instance, &info, nullptr, &surface), "vkCreateHeadlessSurfaceEXT");
		dest->surface = surface;
		return;
	}
	Window const window = XCreateSimpleWindow(c->display, DefaultRootWindow(c->display), 0, 0,
	                                          extent.width, extent.height, 0, 0, 0);
	require(window, "cannot create X11 window");
	dest->window = window;
	XStoreName(c->display, window, "dlsslop-amd native transport smoke test");
	XMapWindow(c->display, window);
	XSync(c->display, False);
	VkXlibSurfaceCreateInfoKHR const info = {
		.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
		.dpy = c->display,
		.window = window,
	};
	VkSurfaceKHR surface;
	check(vkCreateXlibSurfaceKHR(instance, &info, nullptr, &surface), "vkCreateXlibSurfaceKHR");
	dest->surface = surface;
}

/** @brief Resizes the window. A headless surface leaves the extent to the swapchain.
 *
 * @param s      The surface.
 * @param c      The context.
 * @param extent The window's new size.
 */
static void
surface_resize (struct surface const *s,
                struct context const *c,
                VkExtent2D            extent)
{
	if (!c->display)
		return;
	XResizeWindow(c->display, s->window, extent.width, extent.height);
	XSync(c->display, False);
}

/** @brief Destroys the surface and its window.
 *
 * @param s        The surface.
 * @param c        The context.
 * @param instance The surface's instance.
 */
static void
surface_fini (struct surface       *s,
              struct context const *c,
              VkInstance            instance)
{
	vkDestroySurfaceKHR(instance, s->surface, nullptr);
	s->surface = VK_NULL_HANDLE;
	if (s->window) {
		XDestroyWindow(c->display, s->window);
		s->window = 0;
	}
}

/** @brief Finds a queue family that presents to the surface and does graphics and compute.
 *
 * @param device  The physical device.
 * @param surface The surface.
 * @param family  Receives the family.
 * @return        true if the device has one.
 */
static bool
presents (VkPhysicalDevice  device,
          VkSurfaceKHR      surface,
          uint32_t         *family)
{
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
	VkQueueFamilyProperties *families = malloc(count * sizeof *families);
	require(families || !count, "out of memory");
	vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families);
	bool found = false;
	for (uint32_t i = 0; i < count && !found; ++i) {
		VkBool32 present;
		check(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present), "surface support");
		if (!present || !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		    || !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
			continue;
		*family = i;
		found = true;
	}
	free(families);
	families = nullptr;
	return found;
}

/** @brief Picks a device that presents to the surface from a graphics and compute queue family.
 *         Software Vulkan is preferred; hardware only when no CPU device qualifies.
 *
 * @param instance The instance.
 * @param surface  The surface.
 * @param family   Receives the queue family.
 * @return         The device.
 */
static VkPhysicalDevice
pick_device (VkInstance    instance,
             VkSurfaceKHR  surface,
             uint32_t     *family)
{
	uint32_t count = 0;
	check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
	VkPhysicalDevice *devices = malloc(count * sizeof *devices);
	require(devices || !count, "out of memory");
	check(vkEnumeratePhysicalDevices(instance, &count, devices), "enumerate devices");
	VkPhysicalDevice picked = VK_NULL_HANDLE;
	// CPU devices on the first pass, the others on the second, each in the order enumerated.
	for (uint32_t pass = 0; pass < 2 && !picked; ++pass) {
		for (uint32_t i = 0; i < count && !picked; ++i) {
			VkPhysicalDeviceProperties properties;
			vkGetPhysicalDeviceProperties(devices[i], &properties);
			if ((properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) != !pass
			    || !presents(devices[i], surface, family))
				continue;
			fprintf(stderr, "transport smoke device: %s\n", properties.deviceName);
			picked = devices[i];
		}
	}
	free(devices);
	devices = nullptr;
	require(picked, "no graphics/compute/present queue");
	return picked;
}

/** @brief Ends the smoke unless the queue family that pick_device() chose for the first surface
 *         presents to a second one.
 *
 * @param physical The device.
 * @param family   The queue family.
 * @param surface  The second surface.
 */
static void
require_support (VkPhysicalDevice physical,
                 uint32_t         family,
                 VkSurfaceKHR     surface)
{
	VkBool32 supported;
	check(vkGetPhysicalDeviceSurfaceSupportKHR(physical, family, surface, &supported), "surface support");
	require(supported, "the device cannot present to the second surface");
}

/** @brief A device with one queue, and what a frame on it needs.
 *
 * The layer counts each device's composed frames and publishes the count of the device that
 * composed last as layerFrames.
 */
struct device {
	VkPhysicalDevice physical;              //!< The physical device.
	VkDevice         device;                //!< The device.
	VkQueue          queue;                 //!< Its queue.
	VkCommandPool    pool;                  //!< The command pool.
	VkCommandBuffer  cb;                    //!< The frame's command buffer.
	VkSemaphore      acquired[PRESENT_MAX]; //!< An acquire for each swapchain a present names.
	VkSemaphore      rendered;              //!< The submit's signal, which the present waits on.
	uint64_t         composed;              //!< The frames the layer composed.
	uint64_t         presented;             //!< The frames presented.
};

/** @brief Creates a device with one queue of the family.
 *
 * @param dest     The device.
 * @param physical The physical device.
 * @param family   The queue family.
 * @param queue2   Whether the queue comes from vkGetDeviceQueue2 instead of vkGetDeviceQueue: the
 *                 layer has to remember either one to find the device of a present on that queue.
 */
static void
device_init (struct device    *dest,
             VkPhysicalDevice  physical,
             uint32_t          family,
             bool              queue2)
{
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo const queue_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = family,
		.queueCount = 1,
		.pQueuePriorities = &priority,
	};
	char const *const extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
	VkDeviceCreateInfo const device_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queue_info,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = &extension,
	};
	VkDevice device;
	check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
	*dest = (struct device){.physical = physical, .device = device};
	if (queue2) {
		VkDeviceQueueInfo2 const info = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
			.queueFamilyIndex = family,
		};
		vkGetDeviceQueue2(device, &info, &dest->queue);
	} else {
		vkGetDeviceQueue(device, family, 0, &dest->queue);
	}
	require(dest->queue, "the device returned no queue");

	VkCommandPoolCreateInfo const pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = family,
	};
	VkCommandPool pool;
	check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "vkCreateCommandPool");
	dest->pool = pool;
	VkCommandBufferAllocateInfo const buffer_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cb;
	check(vkAllocateCommandBuffers(device, &buffer_info, &cb), "vkAllocateCommandBuffers");
	dest->cb = cb;
	VkSemaphoreCreateInfo const semaphore_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	for (uint32_t i = 0; i < PRESENT_MAX; ++i) {
		VkSemaphore semaphore;
		check(vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore), "acquire semaphore");
		dest->acquired[i] = semaphore;
	}
	VkSemaphore rendered;
	check(vkCreateSemaphore(device, &semaphore_info, nullptr, &rendered), "render semaphore");
	dest->rendered = rendered;
}

/** @brief Waits for the device, then destroys it and what it holds.
 *
 * @param d The device.
 */
static void
device_fini (struct device *d)
{
	check(vkDeviceWaitIdle(d->device), "wait device idle");
	for (uint32_t i = 0; i < PRESENT_MAX; ++i)
		vkDestroySemaphore(d->device, d->acquired[i], nullptr);
	vkDestroySemaphore(d->device, d->rendered, nullptr);
	vkDestroyCommandPool(d->device, d->pool, nullptr);
	vkDestroyDevice(d->device, nullptr);
	check_descriptor0("vkDestroyDevice");
	*d = (struct device){};
}

/** @brief The file of the loaded object that holds a device's entry point.
 *
 * @param device The device.
 * @param name   The entry point.
 * @return       The file, which lasts while the object stays loaded, or nullptr if no loaded object
 *               holds the entry point.
 */
static char const *
entry_object (VkDevice    device,
              char const *name)
{
	Dl_info info;
	return dladdr((void const *)vkGetDeviceProcAddr(device, name), &info) ? info.dli_fname : nullptr;
}

/** @brief The file of the object that a present on the device reaches first: the layer under test.
 *
 * @param device The device.
 * @return       The file, which lasts while the layer stays loaded.
 */
static char const *
layer_object (VkDevice device)
{
	char const *const object = entry_object(device, "vkQueuePresentKHR");
	require(object, "vkQueuePresentKHR is not in a loaded object");
	return object;
}

/** @brief Ends the smoke unless the loader unloaded the layer with the last instance.
 *
 * It does unless VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING is exactly "1". That value keeps every
 * object loaded, which leak checkers need for full stacks.
 *
 * @param object The layer's file.
 */
static void
check_unloaded (char const *object)
{
	char const *const kept = getenv("VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING");
	if (kept && !strcmp(kept, "1")) {
		printf("SKIP: VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1 kept the layer loaded; its unload "
		       "was not checked.\n");
		return;
	}
	void *const handle = dlopen(object, RTLD_LAZY | RTLD_NOLOAD);
	if (handle)
		dlclose(handle);
	require(!handle, "the loader kept the layer loaded after the last instance");
	printf("PASS: the loader unloaded the layer with the last instance.\n");
}

/** @brief Ends the smoke unless the layer hooks the present but leaves the application's other queue
 *         operations to the next layer down, unless the in-layer network is asked for: its build
 *         submits from a thread of its own. It never hooks the acquire.
 *
 * @param device The device.
 */
static void
check_queue_hooks (VkDevice device)
{
	char const *const in_layer = getenv("DLSSLOP_LAYER_NETWORK");
	bool const network = in_layer && !strcmp(in_layer, "1");
	char const *const layer = layer_object(device);
	static char const *const operations[] = {
		"vkQueueSubmit", "vkQueueWaitIdle", "vkQueueBindSparse", "vkDeviceWaitIdle"
	};
	for (size_t i = 0; i < sizeof operations / sizeof *operations; ++i) {
		char const *const next = entry_object(device, operations[i]);
		require(next && (strcmp(layer, next) == 0) == network,
		        network ? "the layer leaves queue operations unhooked while the in-layer network builds"
		                : "the layer hooks queue operations without the in-layer network");
	}
	char const *const acquire = entry_object(device, "vkAcquireNextImageKHR");
	require(acquire && strcmp(layer, acquire), "the layer hooks vkAcquireNextImageKHR");
}

/** @brief A swapchain and its images. */
struct swapchain {
	VkSwapchainKHR handle; //!< The swapchain.
	VkImage       *images; //!< Its images.
	VkExtent2D     extent; //!< Its extent.
};

/** @brief Picks the surface's SDR UNORM format of the order asked for.
 *
 * @param physical The device.
 * @param surface  The surface.
 * @param bgra     Whether the format is BGRA instead of RGBA.
 * @return         The format.
 */
static VkSurfaceFormatKHR
pick_format (VkPhysicalDevice physical,
             VkSurfaceKHR     surface,
             bool             bgra)
{
	uint32_t count = 0;
	check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr), "surface formats");
	VkSurfaceFormatKHR *formats = malloc(count * sizeof *formats);
	require(formats || !count, "out of memory");
	check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats), "surface formats");
	VkFormat const wanted = bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
	uint32_t i = 0;
	while (i < count
	       && (formats[i].colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR || formats[i].format != wanted))
		++i;
	require(i < count, "no SDR UNORM surface format");
	VkSurfaceFormatKHR const format = formats[i];
	free(formats);
	formats = nullptr;
	return format;
}

/** @brief Creates a FIFO swapchain that the application writes with transfers, at the given extent:
 *         the window's, or the one a headless surface leaves to the swapchain.
 *
 * @param dest    The swapchain.
 * @param d       The device.
 * @param surface The surface.
 * @param format  The format.
 * @param extent  The extent.
 * @param old     The swapchain it replaces, or VK_NULL_HANDLE.
 */
static void
swapchain_init (struct swapchain    *dest,
                struct device const *d,
                VkSurfaceKHR         surface,
                VkSurfaceFormatKHR   format,
                VkExtent2D           extent,
                VkSwapchainKHR       old)
{
	VkSurfaceCapabilitiesKHR caps;
	check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->physical, surface, &caps), "surface caps");
	require(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT, "no transfer destination");
	uint32_t images = caps.minImageCount + 1;
	if (caps.maxImageCount && images > caps.maxImageCount)
		images = caps.maxImageCount;
	VkExtent2D const taken = caps.currentExtent.width == UINT32_MAX ? extent : caps.currentExtent;
	require(taken.width == extent.width && taken.height == extent.height,
	        "the surface did not take the extent asked for");
	VkSwapchainCreateInfoKHR const info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = surface,
		.minImageCount = images,
		.imageFormat = format.format,
		.imageColorSpace = format.colorSpace,
		.imageExtent = taken,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.preTransform = caps.currentTransform,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR,
		.clipped = VK_TRUE,
		.oldSwapchain = old,
	};
	VkSwapchainKHR handle;
	check(vkCreateSwapchainKHR(d->device, &info, nullptr, &handle), "vkCreateSwapchainKHR");
	*dest = (struct swapchain){.handle = handle, .extent = extent};
	uint32_t count = 0;
	check(vkGetSwapchainImagesKHR(d->device, handle, &count, nullptr), "swapchain images");
	dest->images = malloc(count * sizeof *dest->images);
	require(dest->images || !count, "out of memory");
	check(vkGetSwapchainImagesKHR(d->device, handle, &count, dest->images), "swapchain images");
}

/** @brief Waits for the device, then destroys the swapchain.
 *
 * @param s The swapchain.
 * @param d Its device.
 */
static void
swapchain_fini (struct swapchain    *s,
                struct device const *d)
{
	check(vkDeviceWaitIdle(d->device), "wait device idle");
	vkDestroySwapchainKHR(d->device, s->handle, nullptr);
	check_descriptor0("vkDestroySwapchainKHR");
	free(s->images);
	*s = (struct swapchain){};
}

/** @brief The swapchains that one present names, in order. */
struct presented {
	struct swapchain const *swapchains[PRESENT_MAX]; //!< The swapchains.
	uint32_t                count;                   //!< How many there are.
};

/** @brief A host buffer of one-pixel detail, which the transfers copy into the swapchain images. */
struct pattern {
	VkBuffer       buffer; //!< The buffer.
	VkDeviceMemory memory; //!< Its memory.
};

/** @brief Creates a pattern of one-pixel detail which the half-size proxy cannot retain. The native
 *         plus edit resolve must nevertheless reproduce it exactly when the worker returns its input
 *         unchanged.
 *
 * @param dest   The pattern.
 * @param d      The device.
 * @param extent The swapchain's extent.
 * @param format The swapchain's format.
 */
static void
pattern_init (struct pattern      *dest,
              struct device const *d,
              VkExtent2D           extent,
              VkFormat             format)
{
	VkBufferCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = (VkDeviceSize)extent.width * extent.height * 4,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
	VkBuffer buffer;
	check(vkCreateBuffer(d->device, &info, nullptr, &buffer), "create pattern buffer");
	*dest = (struct pattern){.buffer = buffer};
	VkMemoryRequirements requirements;
	vkGetBufferMemoryRequirements(d->device, buffer, &requirements);
	VkPhysicalDeviceMemoryProperties memory;
	vkGetPhysicalDeviceMemoryProperties(d->physical, &memory);
	VkMemoryPropertyFlags const wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
	                                   | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t type = 0;
	while (type < memory.memoryTypeCount
	       && (!(requirements.memoryTypeBits & (1u << type))
	           || (memory.memoryTypes[type].propertyFlags & wanted) != wanted))
		++type;
	require(type < memory.memoryTypeCount, "no host coherent pattern buffer memory");
	VkMemoryAllocateInfo const allocation = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory allocated;
	check(vkAllocateMemory(d->device, &allocation, nullptr, &allocated), "allocate pattern");
	dest->memory = allocated;
	check(vkBindBufferMemory(d->device, buffer, allocated, 0), "bind pattern");
	void *mapped;
	check(vkMapMemory(d->device, allocated, 0, info.size, 0, &mapped), "map pattern");
	uint8_t *const pixels = mapped;
	bool const bgra = format == VK_FORMAT_B8G8R8A8_UNORM;
	for (uint32_t y = 0; y < extent.height; ++y) {
		for (uint32_t x = 0; x < extent.width; ++x) {
			uint8_t *const p = pixels + ((size_t)y * extent.width + x) * 4;
			p[bgra ? 2 : 0] = ((x ^ y) & 1) ? 208 : 48;
			p[1] = (x & 1) ? 64 : 176;
			p[bgra ? 0 : 2] = (y & 1) ? 192 : 80;
			p[3] = 255;
		}
	}
	vkUnmapMemory(d->device, allocated);
}

/** @brief Destroys the pattern.
 *
 * @param p The pattern.
 * @param d Its device.
 */
static void
pattern_fini (struct pattern      *p,
              struct device const *d)
{
	if (p->buffer) {
		vkDestroyBuffer(d->device, p->buffer, nullptr);
		p->buffer = VK_NULL_HANDLE;
	}
	if (p->memory) {
		vkFreeMemory(d->device, p->memory, nullptr);
		p->memory = VK_NULL_HANDLE;
	}
}

/** @brief The clear colour of a frame.
 *
 * @param frame The frame's number.
 * @return      Its colour.
 */
static VkClearColorValue
frame_colour (uint64_t frame)
{
	return (VkClearColorValue){.float32 = {(float)(frame % 8 + 1) / 8.0f, 0.25f, 0.75f, 1.0f}};
}

/** @brief Records the application's frame in a swapchain image: a clear, or a copy of the pattern
 *         when there is one. Leaves the image in PRESENT_SRC_KHR.
 *
 * @param cb      The command buffer.
 * @param image   The swapchain image.
 * @param extent  Its extent.
 * @param color   The clear colour.
 * @param pattern The pattern, or VK_NULL_HANDLE.
 */
static void
record_frame (VkCommandBuffer          cb,
              VkImage                  image,
              VkExtent2D               extent,
              VkClearColorValue const *color,
              VkBuffer                 pattern)
{
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	// The stage the acquire semaphore is waited at, so the transition follows the acquire.
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);
	if (pattern) {
		VkBufferImageCopy const copy = {
			.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.imageExtent = {extent.width, extent.height, 1},
		};
		vkCmdCopyBufferToImage(cb, pattern, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
	} else {
		vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, color,
		                     1, &barrier.subresourceRange);
	}
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = 0;
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);
}

/** @brief Acquires an image of each swapchain, draws the frame into each, presents them in one
 *         vkQueuePresentKHR and waits for the queue.
 *
 * @param d       The device.
 * @param p       The swapchains.
 * @param color   The clear colour.
 * @param pattern The pattern, or VK_NULL_HANDLE.
 * @return        How long the present blocked, in nanoseconds.
 */
static int64_t
draw (struct device           *d,
      struct presented         p,
      VkClearColorValue const *color,
      VkBuffer                 pattern)
{
	require(p.count <= PRESENT_MAX, "too many swapchains for one present");
	VkSwapchainKHR handles[PRESENT_MAX];
	uint32_t indices[PRESENT_MAX];
	static VkPipelineStageFlags const wait_stages[PRESENT_MAX] = {
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT
	};
	for (uint32_t i = 0; i < p.count; ++i) {
		handles[i] = p.swapchains[i]->handle;
		check(vkAcquireNextImageKHR(d->device, handles[i], UINT64_MAX, d->acquired[i], VK_NULL_HANDLE,
		                            &indices[i]), "acquire image");
	}
	check(vkResetCommandBuffer(d->cb, 0), "reset command buffer");
	VkCommandBufferBeginInfo const begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	check(vkBeginCommandBuffer(d->cb, &begin), "begin command buffer");
	for (uint32_t i = 0; i < p.count; ++i)
		record_frame(d->cb, p.swapchains[i]->images[indices[i]], p.swapchains[i]->extent, color, pattern);
	check(vkEndCommandBuffer(d->cb), "end command buffer");
	VkSubmitInfo const submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = p.count,
		.pWaitSemaphores = d->acquired,
		.pWaitDstStageMask = wait_stages,
		.commandBufferCount = 1,
		.pCommandBuffers = &d->cb,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &d->rendered,
	};
	check(vkQueueSubmit(d->queue, 1, &submit, VK_NULL_HANDLE), "submit frame");
	VkPresentInfoKHR const present = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &d->rendered,
		.swapchainCount = p.count,
		.pSwapchains = handles,
		.pImageIndices = indices,
	};
	int64_t const started = monotonic_ns();
	check(vkQueuePresentKHR(d->queue, &present), "present frame");
	int64_t const blocked = monotonic_ns() - started;
	check(vkQueueWaitIdle(d->queue), "wait queue idle");
	++d->presented;
	return blocked;
}

/** @brief Ends the smoke unless the identity worker accepted the request at width x height and
 *         returned the proxy byte for byte. A proxy of code values holds the clear colour; a pattern
 *         or linear light passes no colour.
 *
 * @param c       The context.
 * @param request The request.
 * @param width   The proxy's width.
 * @param height  The proxy's height.
 * @param color   The clear colour the proxy holds, or nullptr.
 * @param proxy16 Whether the proxy is FP16.
 */
static void
check_answer (struct context const    *c,
              uint32_t                 request,
              uint32_t                 width,
              uint32_t                 height,
              VkClearColorValue const *color,
              bool                     proxy16)
{
	struct ShmHeader const *const h = c->h;
	require(atomic_load_explicit(&h->seq_resp, memory_order_acquire) == request
	        && atomic_load(&h->seq_ok) == request, "worker did not successfully answer frame");
	require(atomic_load(&h->answeredW) == width && atomic_load(&h->answeredH) == height,
	        "response dimensions differ");
	uint8_t const *const input = (uint8_t const *)h + kHeaderBytes;
	uint8_t const *const output = input + kMaxFrame;
	for (uint32_t channel = 0; color && channel != 4; ++channel) {
		long const expected = lroundf(color->float32[channel] * 255.0f);
		if (proxy16) {
			uint16_t half;
			memcpy(&half, input + channel * 2, sizeof half);
			int const exponent = (half >> 10) & 31;
			float const value = ldexpf((half & 1023) + (exponent ? 1024 : 0),
			                           (exponent ? exponent : 1) - 25);
			require(fabsf(value * 255.0f - expected) <= 1.0f,
			        "FP16 proxy pixel differs from source code value");
		} else {
			require(labs(input[channel] - expected) <= 1, "GPU proxy pixel differs from clear color");
		}
	}
	require(atomic_load(&h->hdrEncode) == (proxy16 ? 1u : 0u), "request transport precision mismatch");
	size_t const bytes = (size_t)width * height * (proxy16 ? 8 : 4);
	require(!memcmp(input, output, bytes), "identity worker corrupted frame pixels");
}

/** @brief Presents a frame on each swapchain in one vkQueuePresentKHR. The swapchain expected to
 *         compose publishes exactly one request, answered at its extent with the frame unchanged,
 *         and its device's frame count grows by one; when none is expected, neither the request nor
 *         the count moves.
 *
 * @param c        The context.
 * @param d        The device.
 * @param p        The swapchains.
 * @param composes The swapchain expected to compose, or nullptr.
 */
static void
present (struct context const   *c,
         struct device          *d,
         struct presented        p,
         struct swapchain const *composes)
{
	struct ShmHeader const *const h = c->h;
	uint32_t const previous = atomic_load(&h->seq_req);
	uint64_t const frames = atomic_load(&h->layerFrames);
	VkClearColorValue const color = frame_colour(d->presented);
	draw(d, p, &color, VK_NULL_HANDLE);
	uint32_t const request = atomic_load(&h->seq_req);
	if (!composes) {
		require(request == previous && atomic_load(&h->layerFrames) == frames,
		        "the layer composed a present that holds no primary swapchain");
		return;
	}
	require(request == previous + 1, "the layer did not publish exactly one request for the present");
	check_answer(c, request, composes->extent.width, composes->extent.height, &color, false);
	++d->composed;
	require(atomic_load(&h->layerFrames) == d->composed, "the layer's frame count did not grow by one");
}

/** @brief The swapchains given, as one present names them. */
#define PRESENTED(...) ((struct presented){ \
	.swapchains = {__VA_ARGS__}, \
	.count = sizeof (struct swapchain const *[]){__VA_ARGS__} / sizeof (struct swapchain const *), \
})

/** @brief The frame modes: frames on one swapchain, each composed once from the worker's answer,
 *         under producer contention, a reduced or FP16 proxy and linear HDR as the command line
 *         asks, or presented without blocking when the worker cannot answer.
 *
 * @param c The context.
 * @param o The command line.
 * @return  0, or 77 if the validation layer is not installed.
 */
static int
smoke (struct context const *c,
       struct options const *o)
{
	VkInstance const instance = create_instance(o);
	if (!instance)
		return 77;
	struct surface surface;
	surface_init(&surface, c, instance, EXTENT);
	uint32_t family;
	VkPhysicalDevice const physical = pick_device(instance, surface.surface, &family);
	struct device d;
	device_init(&d, physical, family, false);
	check_queue_hooks(d.device);
	VkSurfaceFormatKHR const format = pick_format(physical, surface.surface, o->bgra);
	struct swapchain s;
	swapchain_init(&s, &d, surface.surface, format, EXTENT, VK_NULL_HANDLE);
	struct pattern pattern = {};
	if (o->reduced)
		pattern_init(&pattern, &d, s.extent, format.format);

	struct ShmHeader const *const h = c->h;
	uint64_t const initial_frames = atomic_load(&h->layerFrames);
	struct contending_producer contender;
	contending_producer_init(&contender, c->path, o->contention);
	for (uint32_t frame = 0; frame < FRAMES + o->contention; ++frame) {
		uint32_t const previous = atomic_load(&h->seq_req);
		uint32_t const answered = atomic_load(&h->seq_resp);
		VkClearColorValue const color = frame_colour(frame);
		int64_t const blocked = draw(&d, PRESENTED(&s), &color, pattern.buffer);
		uint32_t const request = atomic_load(&h->seq_req);
		if (o->no_worker) {
			// A killed worker still reads as running: the free slot takes one request,
			// which the layer gives up on within the heartbeat window. A worker that
			// is not running gets no request at all.
			bool const running = atomic_load(&h->helperState) == kHelperRunning;
			require(request == previous + (running && previous == answered),
			        running ? "layer did not publish exactly one request to the silent worker"
			                : "layer published a request to a worker that is not running");
			require(atomic_load(&h->seq_resp) == answered, "a dead worker answered");
			require(blocked < 2 * NS_PER_SECOND, "layer blocked on a worker that cannot answer");
			continue;
		}
		if (o->contention && !frame) {
			require(request == previous, "contending layer overwrote another producer's slot");
			require(atomic_load(&h->layerFrames) == initial_frames,
			        "contending layer composed without owning slot");
			contending_producer_fini(&contender);
			continue;
		}
		// One request a present: a second copy of the layer in the chain stays inert.
		require(request == previous + 1, "layer did not submit exactly one request for the frame");
		// A linear-HDR proxy carries encoded light, not the swapchain's code values.
		check_answer(c, request, o->reduced ? 160 : s.extent.width, o->reduced ? 96 : s.extent.height,
		             o->reduced || o->linear ? nullptr : &color, o->proxy16);
	}
	uint64_t const composed = atomic_load(&h->layerFrames);
	swapchain_fini(&s, &d);
	pattern_fini(&pattern, &d);
	device_fini(&d);
	check_channel_released(c);
	printf("PASS: destroying the device released the layer's channel descriptor and mappings.\n");
	surface_fini(&surface, c, instance);
	destroy_instance(instance);
	printf("PASS: descriptor 0 survived every teardown.\n");
	if (o->no_worker) {
		printf("PASS: %u frames presented without blocking on a worker that cannot answer.\n", FRAMES);
	} else {
		require(composed == initial_frames + FRAMES, "layer did not compose each frame exactly once");
		// Measured on every composed frame, not only under DLSSNR_TIME.
		require(BitsToFloat(atomic_load(&h->layerMsBits)) > 0, "layer did not publish its per-frame cost");
		printf("PASS: %u Vulkan frames captured, RGBA pixels verified, worker answered, "
		       "composition submitted and presented (%ux%u).\n", FRAMES, EXTENT.width, EXTENT.height);
		if (o->contention)
			printf("PASS: independent producer lock caused clean pass-through, then recovered.\n");
		if (o->reduced)
			printf("PASS: 320x192 checker frame used a 160x96 transport proxy.\n");
	}
	contending_producer_fini(&contender);
	return 0;
}

/** @brief Re-creates the swapchain with oldSwapchain, alternately larger and smaller, as a window
 *         resize does. A larger successor takes the primary role from its predecessor at once; a
 *         smaller one presents untouched until the predecessor is destroyed.
 *
 * @param c The context.
 * @param o The command line.
 * @return  0, or 77 if the validation layer is not installed.
 */
static int
resizes (struct context const *c,
         struct options const *o)
{
	static constexpr VkExtent2D sizes[] = {{288, 176}, {352, 208}};
	VkInstance const instance = create_instance(o);
	if (!instance)
		return 77;
	struct surface surface;
	surface_init(&surface, c, instance, EXTENT);
	uint32_t family;
	VkPhysicalDevice const physical = pick_device(instance, surface.surface, &family);
	struct device d;
	device_init(&d, physical, family, false);
	VkSurfaceFormatKHR const format = pick_format(physical, surface.surface, o->bgra);
	struct swapchain s;
	swapchain_init(&s, &d, surface.surface, format, EXTENT, VK_NULL_HANDLE);
	present(c, &d, PRESENTED(&s), &s);
	for (uint32_t i = 1; i <= o->resizes; ++i) {
		VkExtent2D const extent = sizes[i % 2];
		surface_resize(&surface, c, extent);
		struct swapchain next;
		swapchain_init(&next, &d, surface.surface, format, extent, s.handle);
		bool const larger = extent.width * extent.height > s.extent.width * s.extent.height;
		present(c, &d, PRESENTED(&next), larger ? &next : nullptr);
		swapchain_fini(&s, &d);
		present(c, &d, PRESENTED(&next), &next);
		s = next;
	}
	swapchain_fini(&s, &d);
	device_fini(&d);
	check_channel_released(c);
	surface_fini(&surface, c, instance);
	destroy_instance(instance);
	printf("PASS: %u swapchains re-created with oldSwapchain composed their frames once their "
	       "predecessor was gone, at once when larger.\n", o->resizes);
	return 0;
}

/** @brief A smaller second swapchain beside the first, as an overlay's. The larger drives the
 *         channel. The smaller presents untouched, alone and named first in one present with the
 *         larger, until the larger is destroyed and it takes the primary role over.
 *
 * @param c The context.
 * @param o The command line.
 * @return  0, or 77 if the validation layer is not installed.
 */
static int
two_surfaces (struct context const *c,
              struct options const *o)
{
	static constexpr VkExtent2D smaller_extent = {256, 160};
	VkInstance const instance = create_instance(o);
	if (!instance)
		return 77;
	struct surface larger;
	surface_init(&larger, c, instance, EXTENT);
	struct surface smaller;
	surface_init(&smaller, c, instance, smaller_extent);
	uint32_t family;
	VkPhysicalDevice const physical = pick_device(instance, larger.surface, &family);
	require_support(physical, family, smaller.surface);
	struct device d;
	device_init(&d, physical, family, false);
	struct swapchain primary;
	swapchain_init(&primary, &d, larger.surface, pick_format(physical, larger.surface, o->bgra), EXTENT,
	               VK_NULL_HANDLE);
	struct swapchain overlay;
	swapchain_init(&overlay, &d, smaller.surface, pick_format(physical, smaller.surface, o->bgra),
	               smaller_extent, VK_NULL_HANDLE);
	for (uint32_t frame = 0; frame < 2; ++frame) {
		present(c, &d, PRESENTED(&primary), &primary);
		present(c, &d, PRESENTED(&overlay), nullptr);
		present(c, &d, PRESENTED(&overlay, &primary), &primary);
	}
	swapchain_fini(&primary, &d);
	present(c, &d, PRESENTED(&overlay), &overlay);
	present(c, &d, PRESENTED(&overlay), &overlay);
	swapchain_fini(&overlay, &d);
	device_fini(&d);
	check_channel_released(c);
	surface_fini(&smaller, c, instance);
	surface_fini(&larger, c, instance);
	destroy_instance(instance);
	printf("PASS: the smaller of two swapchains presented untouched until the larger was "
	       "destroyed, then composed.\n");
	return 0;
}

/** @brief Two devices on one instance, as a game beside its launcher, each presenting to a surface of
 *         its own. The first composes while the second presents untouched beside it. Once the
 *         first's swapchain is destroyed, the second composes, before and after the first device is
 *         destroyed. With two devices the layer finds a present's device by its queue among those it
 *         saw handed out, so it has to have seen the second's, which comes from vkGetDeviceQueue2.
 *
 * @param c The context.
 * @param o The command line.
 * @return  0, or 77 if the validation layer is not installed.
 */
static int
two_devices (struct context const *c,
             struct options const *o)
{
	VkInstance const instance = create_instance(o);
	if (!instance)
		return 77;
	struct surface first_surface;
	surface_init(&first_surface, c, instance, EXTENT);
	struct surface second_surface;
	surface_init(&second_surface, c, instance, EXTENT);
	uint32_t family;
	VkPhysicalDevice const physical = pick_device(instance, first_surface.surface, &family);
	require_support(physical, family, second_surface.surface);
	struct device first;
	device_init(&first, physical, family, false);
	struct device second;
	device_init(&second, physical, family, true);
	check_queue_hooks(second.device);
	struct swapchain first_swapchain;
	swapchain_init(&first_swapchain, &first, first_surface.surface,
	               pick_format(physical, first_surface.surface, o->bgra), EXTENT, VK_NULL_HANDLE);
	struct swapchain s;
	swapchain_init(&s, &second, second_surface.surface,
	               pick_format(physical, second_surface.surface, o->bgra), EXTENT, VK_NULL_HANDLE);
	for (uint32_t frame = 0; frame < FRAMES; ++frame) {
		present(c, &first, PRESENTED(&first_swapchain), &first_swapchain);
		present(c, &second, PRESENTED(&s), nullptr);
	}
	swapchain_fini(&first_swapchain, &first);
	for (uint32_t frame = 0; frame < FRAMES; ++frame)
		present(c, &second, PRESENTED(&s), &s);
	device_fini(&first);
	present(c, &second, PRESENTED(&s), &s);
	swapchain_fini(&s, &second);
	device_fini(&second);
	check_channel_released(c);
	surface_fini(&second_surface, c, instance);
	surface_fini(&first_surface, c, instance);
	destroy_instance(instance);
	printf("PASS: two devices composed in turn, the second on a queue from vkGetDeviceQueue2, "
	       "beside the first and after it.\n");
	return 0;
}

/** @brief An instance with one device that presents to one surface. */
struct presenter {
	VkInstance       instance;  //!< The instance.
	struct surface   surface;   //!< The surface.
	struct device    device;    //!< The device.
	struct swapchain swapchain; //!< The swapchain.
};

/** @brief Creates an instance with one device that presents to one surface of the given extent.
 *
 * @param dest   The presenter.
 * @param c      The context.
 * @param o      The command line.
 * @param extent The surface's extent.
 * @return       false if the validation layer is not installed.
 */
static bool
presenter_init (struct presenter     *dest,
                struct context const *c,
                struct options const *o,
                VkExtent2D            extent)
{
	*dest = (struct presenter){};
	VkInstance const instance = create_instance(o);
	if (!instance)
		return false;
	dest->instance = instance;
	surface_init(&dest->surface, c, instance, extent);
	uint32_t family;
	VkPhysicalDevice const physical = pick_device(instance, dest->surface.surface, &family);
	device_init(&dest->device, physical, family, false);
	swapchain_init(&dest->swapchain, &dest->device, dest->surface.surface,
	               pick_format(physical, dest->surface.surface, o->bgra), extent, VK_NULL_HANDLE);
	return true;
}

/** @brief Destroys the presenter's swapchain, device, surface and instance.
 *
 * @param p The presenter.
 * @param c The context.
 */
static void
presenter_fini (struct presenter     *p,
                struct context const *c)
{
	swapchain_fini(&p->swapchain, &p->device);
	device_fini(&p->device);
	surface_fini(&p->surface, c, p->instance);
	destroy_instance(p->instance);
	p->instance = VK_NULL_HANDLE;
}

/** @brief Instances alive two at a time, then none, then one again. The loader keeps the layer
 *         loaded while any instance lives: the second instance keeps presenting after the first is
 *         destroyed, and a third, which may get the first's handles, presents beside the second.
 *         Destroying the last instance unloads the layer, and the next instance loads it anew. Each
 *         instance's swapchain is larger than those before it, so that it takes the primary role
 *         over.
 *
 * @param c The context.
 * @param o The command line.
 * @return  0, or 77 if the validation layer is not installed.
 */
static int
two_instances (struct context const *c,
               struct options const *o)
{
	static constexpr VkExtent2D extents[] = {EXTENT, {352, 208}, {384, 224}};
	struct presenter first;
	if (!presenter_init(&first, c, o, extents[0]))
		return 77;
	present(c, &first.device, PRESENTED(&first.swapchain), &first.swapchain);
	// A copy: the layer's own record goes when the loader unloads it.
	char *layer = strdup(layer_object(first.device.device));
	require(layer, "out of memory");
	struct presenter second;
	int status = 77;
	if (!presenter_init(&second, c, o, extents[1]))
		goto done;
	present(c, &second.device, PRESENTED(&second.swapchain), &second.swapchain);
	presenter_fini(&first, c);
	for (uint32_t frame = 0; frame < FRAMES; ++frame)
		present(c, &second.device, PRESENTED(&second.swapchain), &second.swapchain);
	struct presenter third;
	if (!presenter_init(&third, c, o, extents[2]))
		goto done;
	for (uint32_t frame = 0; frame < FRAMES; ++frame) {
		present(c, &third.device, PRESENTED(&third.swapchain), &third.swapchain);
		present(c, &second.device, PRESENTED(&second.swapchain), nullptr);
	}
	presenter_fini(&second, c);
	present(c, &third.device, PRESENTED(&third.swapchain), &third.swapchain);
	presenter_fini(&third, c);
	check_channel_released(c);
	check_unloaded(layer);

	struct presenter fresh;
	if (!presenter_init(&fresh, c, o, EXTENT))
		goto done;
	for (uint32_t frame = 0; frame < FRAMES; ++frame)
		present(c, &fresh.device, PRESENTED(&fresh.swapchain), &fresh.swapchain);
	presenter_fini(&fresh, c);
	check_channel_released(c);
	printf("PASS: instances composed two at a time and after the first was destroyed; a new "
	       "instance composed after all were destroyed.\n");
	status = 0;
done:
	free(layer);
	layer = nullptr;
	return status;
}

#undef PRESENTED

/** @brief Runs the mode the command line chose.
 *
 * @param c The context.
 * @param o The command line.
 * @return  The mode's exit status.
 */
static int
run (struct context const *c,
     struct options const *o)
{
	switch (o->mode) {
	case MODE_FRAME:
		return smoke(c, o);
	case MODE_RESIZE:
		return resizes(c, o);
	case MODE_SECOND_SURFACE:
		return two_surfaces(c, o);
	case MODE_TWO_DEVICES:
		return two_devices(c, o);
	case MODE_TWO_INSTANCES:
		return two_instances(c, o);
	}
	unreachable();
}

/** @brief The text of --help. */
static char const USAGE[] =
	"Usage: vulkan-smoke [OPTIONS]\n"
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
	"                        are destroyed (default: no)\n";

/** @brief Chooses a lifecycle mode, unless another was chosen.
 *
 * @param o    The command line.
 * @param mode The mode.
 */
static void
select_mode (struct options *o,
             enum mode       mode)
{
	require(o->mode == MODE_FRAME || o->mode == mode, "choose one lifecycle mode; use --help");
	o->mode = mode;
}

int
main (int    argc,
      char **argv)
{
	static struct option const long_options[] = {
		{"help", no_argument, nullptr, 'h'},           {"headless", no_argument, nullptr, 'H'},
		{"contention", no_argument, nullptr, 'c'},     {"reduced", no_argument, nullptr, 'r'},
		{"bgra", no_argument, nullptr, 'b'},           {"proxy16", no_argument, nullptr, 'f'},
		{"linear-hdr", no_argument, nullptr, 'l'},     {"no-worker", no_argument, nullptr, 'n'},
		{"resizes", required_argument, nullptr, 'R'},  {"second-surface", no_argument, nullptr, 's'},
		{"two-devices", no_argument, nullptr, 'd'},    {"two-instances", no_argument, nullptr, 'i'},
		{}
	};
	struct options o = {};
	for (int value; (value = getopt_long(argc, argv, "hHcrbflnR:sdi", long_options, nullptr)) != -1;) {
		switch (value) {
		case 'h':
			require(fputs(USAGE, stdout) != EOF && !fflush(stdout), "cannot write the usage");
			return 0;
		case 'H':
			o.headless = true;
			break;
		case 'c':
			o.contention = true;
			break;
		case 'r':
			o.reduced = true;
			break;
		case 'b':
			o.bgra = true;
			break;
		case 'f':
			o.proxy16 = true;
			break;
		case 'l':
			o.linear = true;
			break;
		case 'n':
			o.no_worker = true;
			break;
		case 'R': {
			// strtoul() clamps an out-of-range count to ULONG_MAX, which the range check rejects.
			char *end;
			unsigned long const resizes = strtoul(optarg, &end, 10);
			require(*optarg >= '0' && *optarg <= '9' && !*end && resizes && resizes <= 1000,
			        "--resizes takes a count from 1 to 1000");
			o.resizes = resizes;
			select_mode(&o, MODE_RESIZE);
			break;
		}
		case 's':
			select_mode(&o, MODE_SECOND_SURFACE);
			break;
		case 'd':
			select_mode(&o, MODE_TWO_DEVICES);
			break;
		case 'i':
			select_mode(&o, MODE_TWO_INSTANCES);
			break;
		default:
			fail("invalid option; use --help");
		}
	}
	require(optind == argc, "unexpected positional argument; use --help");
	require(o.mode == MODE_FRAME || !(o.contention || o.reduced || o.proxy16 || o.linear || o.no_worker),
	        "a lifecycle mode takes none of -c, -r, -f, -l and -n; use --help");
	take_descriptor0();
	struct context c;
	context_init(&c, &o);
	int const status = run(&c, &o);
	context_fini(&c);
	require(!fflush(stdout) && !ferror(stdout), "cannot write the results");
	return status;
}

#undef fail
