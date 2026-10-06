/** @file
 *
 * The network on Vulkan: DLSSNR-AMD's network (external/vulkan/linux/) on a device of the daemon's
 * own: vulkan_network.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "error.h"
#include "network_recorder.h"
#include "network_requirements.h"
#include "shm_protocol.h"
#include "vulkan_frame.h"
#include "vulkan_network.h"
#include "vulkan_runtime.h"

static_assert(VULKAN_NETWORK_MAX_PASSES == NETWORK_RECORDER_MAX_PASSES,
              "VULKAN_NETWORK_MAX_PASSES is not the recorder's pass limit");

/** @brief A buffer in memory of its own; its handles are VK_NULL_HANDLE while it has none. */
struct vulkan_network_buffer {
	VkBuffer        buffer; //!< The buffer.
	VkDeviceMemory  memory; //!< Its memory.
	VkDeviceSize    size;   //!< Its size; 0 until it is whole.
	void           *mapped; //!< Its memory, mapped for the host, or nullptr.
};

/** @brief A proxy/answer pair that the layer exported, imported. */
struct vulkan_network_pair {
	struct vulkan_network_buffer frame[2]; //!< The proxy and the answer.
};

/** @brief The network, its device and the objects of its frames; its handles are VK_NULL_HANDLE
 *         while it has none. */
struct vulkan_network {
	struct network_recorder          recorder; //!< The network; empty until the device is made.
	VkPhysicalDeviceMemoryProperties memory;   //!< The device's memory.
	/** @brief Device-local frames that the layer exported, a proxy/answer pair per import slot. */
	struct vulkan_network_pair       imported[VULKAN_NETWORK_IMPORT_SLOTS];
	struct vulkan_network_buffer     upload;   //!< The host transport's proxy.
	struct vulkan_network_buffer     download; //!< The host transport's answer.
	VkCommandPool                    pool;     //!< The pool of cmd.
	VkFence                          fence;    //!< Each frame's end.
	VkQueryPool                      queries;  //!< Each frame's four timestamps.
	VkPhysicalDeviceIDProperties     ids;      //!< The device's and its driver's UUIDs.
	VkInstance                       instance; //!< The instance.
	VkPhysicalDevice                 physical; //!< The physical device.
	VkDevice                         device;   //!< The device.
	VkQueue                          queue;    //!< The device's queue of family.
	VkCommandBuffer                  cmd;      //!< Each frame's commands.
	struct vulkan_network_times      times;    //!< The times of the last frame that had timestamps.
	float                            period;   //!< A timestamp's period in nanoseconds.
	uint32_t                         family;   //!< The queue's family.
	uint32_t                         index;    //!< The physical device's index.
	/** @brief The physical device's name. */
	char                             name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
};

/** @brief Logs one of the runtime's lines.
 *
 * @param line The line, without its newline.
 */
static void
log_line (char const *line)
{
	fprintf(stderr, "%s\n", line);
}

/** @brief Frees a buffer and empties it.
 *
 * @param n The network, whose device has finished with the buffer.
 * @param b The buffer.
 */
static void
drop (struct vulkan_network        *n,
      struct vulkan_network_buffer *b)
{
	if (b->buffer)
		vkDestroyBuffer(n->device, b->buffer, nullptr);
	if (b->memory)
		vkFreeMemory(n->device, b->memory, nullptr);
	*b = (struct vulkan_network_buffer){};
}

/** @brief Frees an imported pair and empties it.
 *
 * @param n    The network, whose device has finished with the pair.
 * @param pair The pair.
 */
static void
release (struct vulkan_network      *n,
         struct vulkan_network_pair *pair)
{
	drop(n, &pair->frame[0]);
	drop(n, &pair->frame[1]);
}

/** @brief Returns a function's code unless it is ERROR_NONE. */
#define TRY(call) do { \
	enum error_code const code_ = (call); \
	if (code_) \
		return code_; \
} while (0)

/** @brief Makes a buffer a host-visible transfer buffer of at least some bytes, unless it is one.
 *
 * @param n     The network.
 * @param b     The buffer; keeps the handles made, also on a failure.
 * @param bytes The bytes.
 * @param usage The buffer's usage.
 * @param extra The memory's properties besides host-visible and host-coherent.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
host_buffer (struct vulkan_network        *n,
             struct vulkan_network_buffer *b,
             VkDeviceSize                  bytes,
             VkBufferUsageFlags            usage,
             VkMemoryPropertyFlags         extra,
             struct error                 *e)
{
	if (b->size >= bytes)
		return ERROR_NONE;
	drop(n, b);
	VkBufferCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size  = bytes,
		.usage = usage,
	};
	VkBuffer buffer;
	TRY(vulkan_check(vkCreateBuffer(n->device, &info, nullptr, &buffer), "create transfer buffer", e));
	b->buffer = buffer;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(n->device, buffer, &req);
	uint32_t type;
	TRY(vulkan_memory_type(&n->memory, req.memoryTypeBits,
	                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | extra,
	                       &type, e));
	VkMemoryAllocateInfo const alloc = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory memory;
	TRY(vulkan_check(vkAllocateMemory(n->device, &alloc, nullptr, &memory), "allocate transfer buffer", e));
	b->memory = memory;
	TRY(vulkan_check(vkBindBufferMemory(n->device, buffer, memory, 0), "bind transfer buffer", e));
	void *mapped;
	TRY(vulkan_check(vkMapMemory(n->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "map transfer buffer", e));
	b->mapped = mapped;
	b->size = bytes;
	return ERROR_NONE;
}

/** @brief Finds a device's first queue family with graphics and compute: the runtime converts colour
 *         formats with blits.
 *
 * @param physical The device.
 * @param family   Receives the family.
 * @return         nullptr, or the words for what stops the network on the device.
 */
static char const *
graphics_compute_family (VkPhysicalDevice  physical,
                         uint32_t         *family)
{
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
	if (!count)
		return "no graphics and compute queue";
	VkQueueFamilyProperties *queues = malloc(count * sizeof *queues);
	if (!queues)
		return NETWORK_REQUIREMENTS_NO_MEMORY " unavailable";
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
	VkQueueFlags const need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
	uint32_t f = 0;
	while (f < count && (queues[f].queueFlags & need) != need)
		++f;
	free(queues);
	queues = nullptr;
	*family = f;
	return f < count ? nullptr : "no graphics and compute queue";
}

/** @brief Chooses the physical device that the network runs on: the one that --device names, or the
 *         first that can run the network.
 *
 * @param n         The network, with its instance; receives the device, its queue family, index and
 *                  name.
 * @param p_devices The caller's pointer, which holds nullptr; receives the instance's physical
 *                  devices, if it has any, for the caller to free, also on a failure.
 * @param device    The index that --device names, or -1.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
choose_device (struct vulkan_network  *n,
               VkPhysicalDevice      **p_devices,
               int                     device,
               struct error           *e)
{
	uint32_t count = 0;
	TRY(vulkan_check(vkEnumeratePhysicalDevices(n->instance, &count, nullptr), "enumerate Vulkan devices", e));
	if (count) {
		VkPhysicalDevice *const listed = malloc(count * sizeof *listed);
		if (!listed)
			return error_fail(e, "out of memory");
		*p_devices = listed;
		TRY(vulkan_check(vkEnumeratePhysicalDevices(n->instance, &count, listed),
		                 "enumerate Vulkan devices", e));
	}
	VkPhysicalDevice const *const devices = *p_devices;
	// The devices to examine: the one that --device names, or all of them.
	uint32_t first = 0;
	uint32_t end = count;
	if (device >= 0) {
		first = (uint32_t)device;
		if (first >= count)
			return error_fail(e, "no Vulkan device %d", device);
		end = first + 1;
	}
	PFN_vkVoidFunction const lookup = vkGetInstanceProcAddr(n->instance,
	                                                        "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
	PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR const matrices =
		(PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)lookup;
	// Why each device examined cannot run the network, as far as an error's words hold them.
	char reasons[ERROR_WHAT_BYTES];
	size_t length = 0;
	reasons[0] = '\0';
	for (uint32_t i = first; i < end; ++i) {
		VkPhysicalDeviceProperties p;
		vkGetPhysicalDeviceProperties(devices[i], &p);
		// External memory: the layer's device-local frames are imported.
		char const *why = network_requirements_unsupported(devices[i], vkGetPhysicalDeviceProperties2,
		                                                   vkGetPhysicalDeviceFeatures2,
		                                                   vkEnumerateDeviceExtensionProperties, matrices,
		                                                   VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
		char const *suffix = " unavailable";
		if (!why) {
			uint32_t family;
			why = graphics_compute_family(devices[i], &family);
			suffix = "";
			if (!why) {
				n->physical = devices[i];
				n->family = family;
				n->index = i;
				static_assert(sizeof n->name == sizeof p.deviceName);
				memcpy(n->name, p.deviceName, sizeof n->name);
				return ERROR_NONE;
			}
		}
		if (length + 1 < sizeof reasons) {
			int const added = snprintf(reasons + length, sizeof reasons - length,
			                           "\n  Vulkan device %" PRIu32 " (%s): %s%s",
			                           i, p.deviceName, why, suffix);
			if (added < 0)
				reasons[length] = '\0';
			else if ((size_t)added < sizeof reasons - length)
				length += (size_t)added;
			else
				length = sizeof reasons - 1;
		}
	}
	return error_fail(e, "no Vulkan device can run the network:%s", reasons);
}

/** @brief Makes a network's instance and device, the objects of its frames, and its recorder.
 *
 * @param n      The network, empty; keeps what is made, also on a failure.
 * @param paths  Where the network's files are; the recorder copies them.
 * @param device The index of the physical device to run on, or -1 for the first that can.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
open_network (struct vulkan_network     *n,
              struct vulkan_paths const *paths,
              int                        device,
              struct error              *e)
{
	// The daemon has no overlay to show: keep implicit layers (Steam's, MangoHud's) out of its
	// instance, unless the environment already says otherwise.
	if (setenv("VK_LOADER_LAYERS_DISABLE", "~implicit~", 0)) {
		int const err = errno;
		char buf[64];
		return error_fail(e, "set VK_LOADER_LAYERS_DISABLE: %s", strerror_r(err, buf, sizeof buf));
	}
	VkApplicationInfo const app = {
		.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "dlsslopd",
		.apiVersion       = VK_API_VERSION_1_3,
	};
	VkInstanceCreateInfo const instance_info = {
		.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
	};
	VkInstance instance;
	TRY(vulkan_check(vkCreateInstance(&instance_info, nullptr, &instance), "create Vulkan instance", e));
	n->instance = instance;
	VkPhysicalDevice *devices = nullptr;
	enum error_code const code = choose_device(n, &devices, device, e);
	free(devices);
	devices = nullptr;
	if (code)
		return code;

	struct network_feature_chain enable;
	network_feature_chain_init(&enable);
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i)
		*network_feature_chain_bit(&enable, &NETWORK_FEATURES[i]) = VK_TRUE;
	char const *extensions[1 + NETWORK_FEATURE_COUNT] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
	uint32_t const extension_count = network_requirements_append_extensions(extensions, 1);
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo const queue_info = {
		.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = n->family,
		.queueCount       = 1,
		.pQueuePriorities = &priority,
	};
	VkDeviceCreateInfo const device_info = {
		.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext                   = &enable.head,
		.queueCreateInfoCount    = 1,
		.pQueueCreateInfos       = &queue_info,
		.enabledExtensionCount   = extension_count,
		.ppEnabledExtensionNames = extensions,
	};
	VkDevice d;
	TRY(vulkan_check(vkCreateDevice(n->physical, &device_info, nullptr, &d), "create Vulkan device", e));
	n->device = d;
	vkGetDeviceQueue(d, n->family, 0, &n->queue);
	vkGetPhysicalDeviceMemoryProperties(n->physical, &n->memory);
	n->ids = (VkPhysicalDeviceIDProperties){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
	VkPhysicalDeviceProperties2 p = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &n->ids,
	};
	vkGetPhysicalDeviceProperties2(n->physical, &p);
	n->period = p.properties.limits.timestampPeriod;

	VkCommandPoolCreateInfo const pool_info = {
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = n->family,
	};
	VkCommandPool pool;
	TRY(vulkan_check(vkCreateCommandPool(d, &pool_info, nullptr, &pool), "create command pool", e));
	n->pool = pool;
	VkCommandBufferAllocateInfo const cmd_info = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = pool,
		.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	TRY(vulkan_check(vkAllocateCommandBuffers(d, &cmd_info, &cmd), "allocate command buffer", e));
	n->cmd = cmd;
	VkFenceCreateInfo const fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	TRY(vulkan_check(vkCreateFence(d, &fence_info, nullptr, &fence), "create fence", e));
	n->fence = fence;
	VkQueryPoolCreateInfo const queries_info = {
		.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType  = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = 4,
	};
	VkQueryPool queries;
	TRY(vulkan_check(vkCreateQueryPool(d, &queries_info, nullptr, &queries), "create timestamp queries", e));
	n->queries = queries;

	struct vulkan_device const on = {
		.instance  = instance,
		.physical  = n->physical,
		.device    = d,
		.queue     = n->queue,
		.log       = log_line,
		.functions = {
			.queue_families    = vkGetPhysicalDeviceQueueFamilyProperties,
			.properties        = vkGetPhysicalDeviceProperties2,
			.format_properties = vkGetPhysicalDeviceFormatProperties2,
		},
		.memory    = n->memory,
		.family    = n->family,
	};
	return network_recorder_init(&n->recorder, &on, paths, false, e);
}

/** @brief Waits for a network's device and frees what the network holds, but not the network.
 *
 * @param n The network.
 */
static void
close_network (struct vulkan_network *n)
{
	if (n->device) {
		// The network goes, whatever the device says.
		(void)vkDeviceWaitIdle(n->device);
		network_recorder_fini(&n->recorder);
		for (uint32_t i = 0; i < VULKAN_NETWORK_IMPORT_SLOTS; ++i)
			release(n, &n->imported[i]);
		drop(n, &n->upload);
		drop(n, &n->download);
		if (n->queries) {
			vkDestroyQueryPool(n->device, n->queries, nullptr);
			n->queries = VK_NULL_HANDLE;
		}
		if (n->fence) {
			vkDestroyFence(n->device, n->fence, nullptr);
			n->fence = VK_NULL_HANDLE;
		}
		if (n->pool) {
			// The pool's command buffer goes with it.
			vkDestroyCommandPool(n->device, n->pool, nullptr);
			n->pool = VK_NULL_HANDLE;
			n->cmd = VK_NULL_HANDLE;
		}
		vkDestroyDevice(n->device, nullptr);
		n->device = VK_NULL_HANDLE;
	}
	if (n->instance) {
		vkDestroyInstance(n->instance, nullptr);
		n->instance = VK_NULL_HANDLE;
	}
}

enum error_code
vulkan_network_create (struct vulkan_network     **dest,
                       struct vulkan_paths const  *paths,
                       int                         device,
                       struct error               *e)
{
	struct vulkan_network *n = malloc(sizeof *n);
	if (!n) {
		*dest = nullptr;
		return error_fail(e, "out of memory");
	}
	*n = (struct vulkan_network){};
	enum error_code const code = open_network(n, paths, device, e);
	if (code)
		vulkan_network_destroy(&n);
	*dest = n;
	return code;
}

void
vulkan_network_destroy (struct vulkan_network **p_n)
{
	if (p_n && *p_n) {
		struct vulkan_network *n = *p_n;
		*p_n = nullptr;
		close_network(n);
		free(n);
		n = nullptr;
	}
}

char const *
vulkan_network_device_name (struct vulkan_network const *n)
{
	return n ? n->name : nullptr;
}

uint32_t
vulkan_network_device_index (struct vulkan_network const *n)
{
	return n ? n->index : UINT32_MAX;
}

bool
vulkan_network_shape_differs (struct vulkan_network const *n,
                              struct vulkan_frame const   *frame)
{
	return network_recorder_shape_differs(&n->recorder, frame);
}

enum error_code
vulkan_network_shape (struct vulkan_network     *n,
                      struct vulkan_frame const *frame,
                      struct error              *e)
{
	if (!network_recorder_shape_differs(&n->recorder, frame))
		return ERROR_NONE;
	TRY(vulkan_check(vkDeviceWaitIdle(n->device), "wait for the device", e));
	bool changed;
	return network_recorder_shape(&n->recorder, frame, nullptr, &changed, e);
}

enum error_code
vulkan_network_plan (struct vulkan_network     *n,
                     struct vulkan_frame const *frame,
                     struct error              *e)
{
	return network_recorder_plan(&n->recorder, frame, e);
}

/** @brief Imports one offered buffer: the layer's buffer, repeated, in the memory it exported.
 *
 * @param n          The network.
 * @param b          The buffer, empty; keeps the handles made, also on a failure.
 * @param size       The buffer's size.
 * @param allocation The memory's size.
 * @param fd         The memory's descriptor; set to -1 once Vulkan owns it.
 * @return           true if the buffer is in its memory.
 */
static bool
import_buffer (struct vulkan_network        *n,
               struct vulkan_network_buffer *b,
               VkDeviceSize                  size,
               VkDeviceSize                  allocation,
               int                          *fd)
{
	VkExternalMemoryBufferCreateInfo const external = {
		.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
	};
	// The layer's buffer, repeated: a dedicated import must name an identical one.
	VkBufferCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = &external,
		.size  = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	VkBuffer buffer;
	if (vkCreateBuffer(n->device, &info, nullptr, &buffer) != VK_SUCCESS)
		return false;
	b->buffer = buffer;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(n->device, buffer, &req);
	// The layer exports from the first device-local type the buffer allows; on the same GPU and
	// driver that is this device's too.
	uint32_t type;
	if (vulkan_memory_type(&n->memory, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type, nullptr) ||
	    req.size != allocation)
		return false;
	VkMemoryDedicatedAllocateInfo const dedicated = {
		.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.buffer = buffer,
	};
	VkImportMemoryFdInfoKHR const import = {
		.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
		.pNext      = &dedicated,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
		.fd         = *fd,
	};
	VkMemoryAllocateInfo const alloc = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext           = &import,
		.allocationSize  = allocation,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory memory;
	if (vkAllocateMemory(n->device, &alloc, nullptr, &memory) != VK_SUCCESS)
		return false;
	b->memory = memory;
	*fd = -1; // Vulkan owns the descriptor it imported.
	if (vkBindBufferMemory(n->device, buffer, memory, 0) != VK_SUCCESS)
		return false;
	b->size = size;
	return true;
}

bool
vulkan_network_import (struct vulkan_network          *n,
                       ptrdiff_t                       slot,
                       struct ShmTransportOffer const *offer,
                       int                             fds[2])
{
	if (slot < 0 || slot >= VULKAN_NETWORK_IMPORT_SLOTS)
		return false;
	static_assert(sizeof offer->deviceUuid == VK_UUID_SIZE && sizeof offer->driverUuid == VK_UUID_SIZE);
	if (memcmp(offer->deviceUuid, n->ids.deviceUUID, VK_UUID_SIZE) ||
	    memcmp(offer->driverUuid, n->ids.driverUUID, VK_UUID_SIZE)) {
		fprintf(stderr, "device-local transport: the game's frames are on another device or driver\n");
		return false;
	}
	struct vulkan_network_pair next = {};
	for (uint32_t i = 0; i < 2; ++i) {
		if (!import_buffer(n, &next.frame[i], offer->size[i], offer->allocation[i], &fds[i])) {
			release(n, &next);
			return false;
		}
	}
	// An earlier frame may still be reading the slot's old import. A queue that cannot go idle
	// leaves the old pair where it is.
	if (vkQueueWaitIdle(n->queue) != VK_SUCCESS) {
		release(n, &next);
		return false;
	}
	release(n, &n->imported[slot]);
	n->imported[slot] = next;
	return true;
}

/** @brief The time between two timestamps.
 *
 * @param from   The earlier timestamp.
 * @param to     The later one.
 * @param period A timestamp's period in nanoseconds.
 * @return       The time in milliseconds.
 */
static float
milliseconds (uint64_t from,
              uint64_t to,
              float    period)
{
	return (float)((double)(to - from) * period / 1e6);
}

enum error_code
vulkan_network_infer (struct vulkan_network     *n,
                      struct vulkan_frame const *frame,
                      ptrdiff_t                  slot,
                      uint8_t const             *input,
                      uint8_t                   *output,
                      struct error              *e)
{
	TRY(vulkan_network_shape(n, frame, e));
	VkDeviceSize const bytes = (VkDeviceSize)frame->width * frame->height * (frame->fp16 ? 8 : 4);
	bool const exported = slot >= 0; // the layer's device-local pair
	VkBuffer source;
	VkBuffer target;
	if (exported) {
		if (slot >= VULKAN_NETWORK_IMPORT_SLOTS || !n->imported[slot].frame[0].buffer)
			return error_fail(e, "import slot %td holds no frames", slot);
		source = n->imported[slot].frame[0].buffer;
		target = n->imported[slot].frame[1].buffer;
	} else {
		TRY(host_buffer(n, &n->upload, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0, e));
		TRY(host_buffer(n, &n->download, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		                VK_MEMORY_PROPERTY_HOST_CACHED_BIT, e));
		memcpy(n->upload.mapped, input, bytes);
		source = n->upload.buffer;
		target = n->download.buffer;
	}
	VkCommandBuffer const cmd = n->cmd;
	TRY(vulkan_check(vkResetCommandBuffer(cmd, 0), "reset command buffer", e));
	VkCommandBufferBeginInfo const begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	TRY(vulkan_check(vkBeginCommandBuffer(cmd, &begin), "begin command buffer", e));
	vkCmdResetQueryPool(cmd, n->queries, 0, 4);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, n->queries, 0);
	// The layer's exported buffers change hands at every frame.
	network_recorder_record_buffers(&n->recorder, cmd, source, target, frame, n->family, exported, n->queries, 1);
	if (!exported) {
		static VkMemoryBarrier const to_host = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		};
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
		                     nullptr, 0, nullptr);
	}
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, n->queries, 3);
	TRY(vulkan_check(vkEndCommandBuffer(cmd), "end command buffer", e));
	VkSubmitInfo const submit = {
		.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers    = &cmd,
	};
	TRY(vulkan_check(vkResetFences(n->device, 1, &n->fence), "reset fence", e));
	TRY(vulkan_check(vkQueueSubmit(n->queue, 1, &submit, n->fence), "submit frame", e));
	network_recorder_submitted(&n->recorder);
	// A healthy frame takes milliseconds; ten seconds means the device is gone.
	TRY(vulkan_check(vkWaitForFences(n->device, 1, &n->fence, VK_TRUE, UINT64_C(10000000000)),
	                 "wait for the frame", e));
	uint64_t stamps[4];
	if (vkGetQueryPoolResults(n->device, n->queries, 0, 4, sizeof stamps, stamps, sizeof stamps[0],
	                          VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
		n->times = (struct vulkan_network_times){
			.upload_ms    = milliseconds(stamps[0], stamps[1], n->period),
			.inference_ms = milliseconds(stamps[1], stamps[2], n->period),
			.readback_ms  = milliseconds(stamps[2], stamps[3], n->period),
		};
	}
	if (network_recorder_timed_out(&n->recorder))
		return error_drop(e, "%s", VULKAN_NETWORK_DROPPED);
	if (!exported)
		memcpy(output, n->download.mapped, bytes);
	return ERROR_NONE;
}

#undef TRY

struct vulkan_network_times
vulkan_network_frame_times (struct vulkan_network const *n)
{
	return n ? n->times : (struct vulkan_network_times){};
}
