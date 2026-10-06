/** @file
 *
 * Drives the layer's composition through in-place rebuilds and checks that no dispatch binds a
 * descriptor set written with an image view that has since been destroyed. Drivers recycle view
 * handles, so a descriptor cache keyed on handle values can bind a set that still points at a freed
 * image. Then checks that a capture whose host buffer cannot be allocated records no pair, which is
 * what keeps the layer's leg 2 asynchronous until the buffer exists.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "composition.h"
#include "support.h"

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("composition rebuild test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief Ends the test unless a Vulkan call succeeded.
 *
 * @param result    What the call returned.
 * @param operation What it did.
 */
static void
vk_check (VkResult    result,
          char const *operation)
{
	require(result == VK_SUCCESS, "%s: VkResult %d", operation, (int)result);
}

/** @brief A view handle the device has made: its build, bumped whenever the handle's value is made
 *         again, and whether it is alive. live is 32 bits wide, which fills the padding.
 */
struct view_entry {
	VkImageView view;       //!< The handle.
	uint32_t    generation; //!< How many times the handle's value was made.
	uint32_t    live;       //!< Whether it is alive.
};

/** @brief What a descriptor set's binding was last written with. */
struct written_binding {
	VkDescriptorSet set;        //!< The set.
	VkImageView     view;       //!< The view written into the binding.
	uint32_t        binding;    //!< The binding.
	uint32_t        generation; //!< The view's build when it was written.
};

/** @brief The device table's view and descriptor calls, wrapped to track every view's lifetime and
 *         what each descriptor set binding was last written with. refuse_memory is 32 bits wide,
 *         which fills the padding.
 */
static struct {
	PFN_vkCreateImageView         create_view;     //!< The device's vkCreateImageView.
	PFN_vkDestroyImageView        destroy_view;    //!< The device's vkDestroyImageView.
	PFN_vkAllocateDescriptorSets  allocate;        //!< The device's vkAllocateDescriptorSets.
	PFN_vkUpdateDescriptorSets    update;          //!< The device's vkUpdateDescriptorSets.
	PFN_vkCmdBindDescriptorSets   bind;            //!< The device's vkCmdBindDescriptorSets.
	PFN_vkAllocateMemory          allocate_memory; //!< The device's vkAllocateMemory.
	struct view_entry            *views;           //!< Every view handle made, by value.
	struct written_binding       *written;         //!< The written bindings, by set, then binding.
	size_t                        view_count;      //!< The entries of views.
	size_t                        view_room;       //!< The room in views.
	size_t                        written_count;   //!< The entries of written.
	size_t                        written_room;    //!< The room in written.
	uint32_t                      binds;           //!< Descriptor sets bound.
	uint32_t                      recycled;        //!< View handles whose value was made again.
	uint32_t                      stale;           //!< Bindings bound with a view gone or remade.
	uint32_t                      refuse_memory;   //!< Whether vkAllocateMemory() fails.
} tracker;

/** @brief A view handle's entry.
 *
 * @param view The handle.
 * @param add  Whether to add an entry of build 0 if there is none.
 * @return     The entry, or nullptr if there is none and @a add is false.
 */
static struct view_entry *
view_entry (VkImageView view,
            bool        add)
{
	for (size_t i = 0; i < tracker.view_count; ++i)
		if (tracker.views[i].view == view)
			return &tracker.views[i];
	if (!add)
		return nullptr;
	if (tracker.view_count == tracker.view_room) {
		size_t const room = tracker.view_room ? 2 * tracker.view_room : 64;
		struct view_entry *const grown = realloc(tracker.views, room * sizeof *grown);
		require(grown, "out of memory");
		tracker.views = grown;
		tracker.view_room = room;
	}
	struct view_entry *const entry = &tracker.views[tracker.view_count++];
	*entry = (struct view_entry){.view = view};
	return entry;
}

/** @brief A view handle's build, which is 0 for a handle never made.
 *
 * @param view The handle.
 * @return     Its build.
 */
static uint32_t
generation_of (VkImageView view)
{
	struct view_entry const *const entry = view_entry(view, false);
	return entry ? entry->generation : 0;
}

/** @brief Whether a view handle is alive.
 *
 * @param view The handle.
 * @return     true if it is.
 */
static bool
view_live (VkImageView view)
{
	struct view_entry const *const entry = view_entry(view, false);
	return entry && entry->live;
}

/** @brief A set's handle as a number, which orders the written bindings: two pointers to unrelated
 *         objects have no order in C.
 *
 * @param set The set.
 * @return    Its handle's bits.
 */
static uint64_t
set_key (VkDescriptorSet set)
{
	uint64_t key = 0;
	static_assert(sizeof set <= sizeof key);
	memcpy(&key, &set, sizeof set);
	return key;
}

/** @brief Where a set's binding is or goes in the written bindings, which are in order of set, then
 *         binding.
 *
 * @param set     The set.
 * @param binding The binding.
 * @return        Its index, or that of the first entry after it.
 */
static size_t
written_at (VkDescriptorSet set,
            uint32_t        binding)
{
	uint64_t const key = set_key(set);
	size_t i = 0;
	for (; i < tracker.written_count; ++i) {
		struct written_binding const *const w = &tracker.written[i];
		uint64_t const at = set_key(w->set);
		if (at > key || (at == key && w->binding >= binding))
			break;
	}
	return i;
}

/** @brief Records what a set's binding was written with.
 *
 * @param set     The set.
 * @param binding The binding.
 * @param view    The view written into it.
 */
static void
write_binding (VkDescriptorSet set,
               uint32_t        binding,
               VkImageView     view)
{
	struct written_binding const entry = {set, view, binding, generation_of(view)};
	size_t const i = written_at(set, binding);
	if (i < tracker.written_count && tracker.written[i].set == set && tracker.written[i].binding == binding) {
		tracker.written[i] = entry;
		return;
	}
	if (tracker.written_count == tracker.written_room) {
		size_t const room = tracker.written_room ? 2 * tracker.written_room : 64;
		struct written_binding *const grown = realloc(tracker.written, room * sizeof *grown);
		require(grown, "out of memory");
		tracker.written = grown;
		tracker.written_room = room;
	}
	memmove(&tracker.written[i + 1], &tracker.written[i],
	        (tracker.written_count - i) * sizeof *tracker.written);
	tracker.written[i] = entry;
	++tracker.written_count;
}

/** @brief Forgets what a set's bindings were written with.
 *
 * @param set The set.
 */
static void
forget_set (VkDescriptorSet set)
{
	size_t const first = written_at(set, 0);
	size_t end = first;
	while (end < tracker.written_count && tracker.written[end].set == set)
		++end;
	if (end == first)
		return;
	memmove(&tracker.written[first], &tracker.written[end],
	        (tracker.written_count - end) * sizeof *tracker.written);
	tracker.written_count -= end - first;
}

/** @brief vkCreateImageView(), which counts the view's build and whether its value was made before. */
static VKAPI_ATTR VkResult VKAPI_CALL
create_view (VkDevice                     device,
             VkImageViewCreateInfo const *info,
             VkAllocationCallbacks const *allocator,
             VkImageView                 *view)
{
	VkResult const result = tracker.create_view(device, info, allocator, view);
	if (result != VK_SUCCESS)
		return result;
	struct view_entry *const entry = view_entry(*view, true);
	tracker.recycled += entry->generation ? 1 : 0;
	++entry->generation;
	entry->live = true;
	return result;
}

/** @brief vkDestroyImageView(), which marks the view gone. */
static VKAPI_ATTR void VKAPI_CALL
destroy_view (VkDevice                     device,
              VkImageView                  view,
              VkAllocationCallbacks const *allocator)
{
	struct view_entry *const entry = view_entry(view, false);
	if (entry)
		entry->live = false;
	tracker.destroy_view(device, view, allocator);
}

/** @brief The meter re-creates its descriptor pool on every rebuild, so a new set can reuse a freed
 *         one's handle value. It starts with nothing written, whatever its predecessor held.
 */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate (VkDevice                           device,
          VkDescriptorSetAllocateInfo const *info,
          VkDescriptorSet                   *sets)
{
	VkResult const result = tracker.allocate(device, info, sets);
	if (result != VK_SUCCESS)
		return result;
	for (uint32_t i = 0; i < info->descriptorSetCount; ++i)
		forget_set(sets[i]);
	return result;
}

/** @brief vkUpdateDescriptorSets(), which records the view and its build that each binding is written
 *         with.
 */
static VKAPI_ATTR void VKAPI_CALL
update (VkDevice                    device,
        uint32_t                    count,
        VkWriteDescriptorSet const *writes,
        uint32_t                    copies,
        VkCopyDescriptorSet const  *copy)
{
	for (uint32_t i = 0; i < count; ++i) {
		VkDescriptorImageInfo const *const image = writes[i].pImageInfo;
		if (image && image->imageView)
			write_binding(writes[i].dstSet, writes[i].dstBinding, image->imageView);
	}
	tracker.update(device, count, writes, copies, copy);
}

/** @brief vkCmdBindDescriptorSets(), which counts the binds, and the bindings whose view is gone or was
 *         made again since they were written.
 */
static VKAPI_ATTR void VKAPI_CALL
bind (VkCommandBuffer        cb,
      VkPipelineBindPoint    point,
      VkPipelineLayout       layout,
      uint32_t               first,
      uint32_t               count,
      VkDescriptorSet const *sets,
      uint32_t               dynamic_count,
      uint32_t const        *dynamic_offsets)
{
	for (uint32_t i = 0; i < count; ++i) {
		++tracker.binds;
		size_t const end = tracker.written_count;
		for (size_t w = written_at(sets[i], 0); w < end && tracker.written[w].set == sets[i]; ++w) {
			struct written_binding const *const b = &tracker.written[w];
			bool const live = view_live(b->view);
			uint32_t const now = generation_of(b->view);
			if (live && now == b->generation)
				continue;
			++tracker.stale;
			printf("stale: binding %u of a bound set was written with view generation %u, now %u%s\n",
			       b->binding, b->generation, now, live ? "" : " (destroyed)");
		}
	}
	tracker.bind(cb, point, layout, first, count, sets, dynamic_count, dynamic_offsets);
}

/** @brief vkAllocateMemory(), which fails while tracker.refuse_memory says so. */
static VKAPI_ATTR VkResult VKAPI_CALL
allocate_memory (VkDevice                     device,
                 VkMemoryAllocateInfo const  *info,
                 VkAllocationCallbacks const *allocator,
                 VkDeviceMemory              *memory)
{
	if (tracker.refuse_memory)
		return VK_ERROR_OUT_OF_HOST_MEMORY;
	return tracker.allocate_memory(device, info, allocator, memory);
}

/** @brief The test's device, and the image that stands in for a swapchain's. */
struct context {
	VkCommandPool         pool;             //!< The command pool.
	VkImage               swapchain;        //!< The swapchain image's stand-in.
	VkDeviceMemory        swapchain_memory; //!< Its memory.
	struct instance_table instance_table;   //!< The instance's entry points.
	struct device_table   device_table;     //!< The device's, with the tracker's wrappers.
	VkInstance            instance;         //!< The instance.
	VkPhysicalDevice      physical;         //!< The physical device.
	VkDevice              device;           //!< The device.
	VkQueue               queue;            //!< Its compute queue.
	VkCommandBuffer       cmd;              //!< The command buffer.
};

/** @brief Destroys what a context holds, then leaves it empty.
 *
 * @param c The context.
 */
static void
context_fini (struct context *c)
{
	if (c->device)
		vk_check(vkDeviceWaitIdle(c->device), "wait for the device");
	if (c->swapchain)
		vkDestroyImage(c->device, c->swapchain, nullptr);
	if (c->swapchain_memory)
		vkFreeMemory(c->device, c->swapchain_memory, nullptr);
	if (c->pool)
		vkDestroyCommandPool(c->device, c->pool, nullptr);
	if (c->device)
		vkDestroyDevice(c->device, nullptr);
	if (c->instance)
		vkDestroyInstance(c->instance, nullptr);
	*c = (struct context){};
}

/** @brief Whether a physical device can run the composition: formatless storage writes and a
 *         compute queue.
 *
 * @param device The physical device.
 * @param family Receives the compute queue's family.
 * @return       true if it can.
 */
static bool
qualifies (VkPhysicalDevice  device,
           uint32_t         *family)
{
	VkPhysicalDeviceFeatures features = {};
	vkGetPhysicalDeviceFeatures(device, &features);
	if (!features.shaderStorageImageWriteWithoutFormat)
		return false;
	uint32_t families = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &families, nullptr);
	VkQueueFamilyProperties *queues = malloc(families * sizeof *queues);
	require(queues || !families, "out of memory");
	vkGetPhysicalDeviceQueueFamilyProperties(device, &families, queues);
	bool found = false;
	for (uint32_t i = 0; i < families && !found; ++i) {
		if (!(queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
			continue;
		*family = i;
		found = true;
	}
	free(queues);
	queues = nullptr;
	return found;
}

/** @brief Makes the test's device: software Vulkan if a CPU device qualifies, otherwise hardware.
 *
 * @param c The context, empty.
 * @return  true if a device qualified.
 */
static bool
context_init (struct context *c)
{
	VkApplicationInfo const app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "dlsslop-amd-composition-rebuild-test",
		.apiVersion = VK_API_VERSION_1_1,
	};
	VkInstanceCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
	};
	VkInstance instance;
	VkResult const create = vkCreateInstance(&info, nullptr, &instance);
	if (create == VK_ERROR_INCOMPATIBLE_DRIVER)
		return false;
	vk_check(create, "create instance");
	c->instance = instance;
	uint32_t count = 0;
	vk_check(vkEnumeratePhysicalDevices(c->instance, &count, nullptr), "enumerate physical devices");
	VkPhysicalDevice *devices = malloc(count * sizeof *devices);
	require(devices || !count, "out of memory");
	vk_check(vkEnumeratePhysicalDevices(c->instance, &count, devices), "get physical devices");
	// Prefer software Vulkan; fall back to hardware only when no CPU device qualifies.
	uint32_t family = 0;
	for (uint32_t pass = 0; pass < 2 && !c->physical; ++pass) {
		for (uint32_t i = 0; i < count && !c->physical; ++i) {
			VkPhysicalDeviceProperties properties = {};
			vkGetPhysicalDeviceProperties(devices[i], &properties);
			bool const cpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
			if (cpu == !pass && qualifies(devices[i], &family))
				c->physical = devices[i];
		}
	}
	free(devices);
	devices = nullptr;
	if (!c->physical)
		return false;
	VkPhysicalDeviceProperties properties = {};
	vkGetPhysicalDeviceProperties(c->physical, &properties);
	printf("composition rebuild device: %s\n", properties.deviceName);
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo const queue_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = family,
		.queueCount = 1,
		.pQueuePriorities = &priority,
	};
	VkPhysicalDeviceFeatures const enabled = {.shaderStorageImageWriteWithoutFormat = VK_TRUE};
	VkDeviceCreateInfo const device_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queue_info,
		.pEnabledFeatures = &enabled,
	};
	VkDevice device;
	vk_check(vkCreateDevice(c->physical, &device_info, nullptr, &device), "create device");
	c->device = device;
	vkGetDeviceQueue(c->device, family, 0, &c->queue);
	VkCommandPoolCreateInfo const pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = family,
	};
	VkCommandPool pool;
	vk_check(vkCreateCommandPool(c->device, &pool_info, nullptr, &pool), "create command pool");
	c->pool = pool;
	VkCommandBufferAllocateInfo const buffer_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = c->pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	vk_check(vkAllocateCommandBuffers(c->device, &buffer_info, &cmd), "allocate command buffer");
	c->cmd = cmd;
	c->instance_table.next_gipa = vkGetInstanceProcAddr;
	instance_table_load(&c->instance_table, c->instance);
	c->device_table.next_dpa = vkGetDeviceProcAddr;
	device_table_load(&c->device_table, c->device);
	tracker.create_view = c->device_table.vkCreateImageView;
	c->device_table.vkCreateImageView = create_view;
	tracker.destroy_view = c->device_table.vkDestroyImageView;
	c->device_table.vkDestroyImageView = destroy_view;
	tracker.allocate = c->device_table.vkAllocateDescriptorSets;
	c->device_table.vkAllocateDescriptorSets = allocate;
	tracker.update = c->device_table.vkUpdateDescriptorSets;
	c->device_table.vkUpdateDescriptorSets = update;
	tracker.bind = c->device_table.vkCmdBindDescriptorSets;
	c->device_table.vkCmdBindDescriptorSets = bind;
	tracker.allocate_memory = c->device_table.vkAllocateMemory;
	c->device_table.vkAllocateMemory = allocate_memory;
	return true;
}

/** @brief Makes the image that stands in for the swapchain image the layer composes into.
 *
 * @param c      The context.
 * @param width  Its width.
 * @param height Its height.
 * @param format Its format.
 */
static void
context_make_swapchain_image (struct context *c,
                              uint32_t        width,
                              uint32_t        height,
                              VkFormat        format)
{
	VkImageCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {width, height, 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	VkImage image;
	vk_check(vkCreateImage(c->device, &info, nullptr, &image), "create swapchain stand-in");
	c->swapchain = image;
	VkMemoryRequirements requirements = {};
	vkGetImageMemoryRequirements(c->device, c->swapchain, &requirements);
	VkPhysicalDeviceMemoryProperties memory = {};
	vkGetPhysicalDeviceMemoryProperties(c->physical, &memory);
	uint32_t type = 0;
	while (type < memory.memoryTypeCount && !(requirements.memoryTypeBits & (1u << type)))
		++type;
	VkMemoryAllocateInfo const allocation = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory memory_handle;
	vk_check(vkAllocateMemory(c->device, &allocation, nullptr, &memory_handle), "allocate swapchain stand-in");
	c->swapchain_memory = memory_handle;
	vk_check(vkBindImageMemory(c->device, c->swapchain, c->swapchain_memory, 0), "bind swapchain stand-in");
}

/** @brief Begins the context's command buffer.
 *
 * @param c The context.
 * @return  The command buffer.
 */
static VkCommandBuffer
context_begin (struct context *c)
{
	VkCommandBufferBeginInfo const begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vk_check(vkBeginCommandBuffer(c->cmd, &begin), "begin command buffer");
	return c->cmd;
}

/** @brief Submits the context's command buffer and waits for it. Refuses before the queue sees a
 *         stale binding: executing one reads a freed image, which crashes a CPU device and faults a
 *         GPU.
 *
 * @param c The context.
 */
static void
context_submit (struct context *c)
{
	vk_check(vkEndCommandBuffer(c->cmd), "end command buffer");
	require(!tracker.stale, "a dispatch bound a descriptor for a destroyed view");
	VkSubmitInfo const submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &c->cmd,
	};
	vk_check(vkQueueSubmit(c->queue, 1, &submit, VK_NULL_HANDLE), "submit");
	vk_check(vkQueueWaitIdle(c->queue), "wait for queue");
}

/** @brief A step of the rebuilds: the model raster and the colour domain. linear_hdr is 32 bits
 *         wide, as the working scale is, so that no padding follows it.
 */
struct step {
	float    working_scale; //!< The working scale.
	uint32_t linear_hdr;    //!< Whether the frame holds linear light.
};

/** @brief One presented frame: leg 1, the identity answer, then leg 2 with device-memory allocations
 *         refused or not.
 *
 * @param context       The context.
 * @param composition   The composition.
 * @param settings      The frame's settings.
 * @param width         The frame's width.
 * @param height        The frame's height.
 * @param format        The frame's format.
 * @param linear_hdr    Whether the frame holds linear light.
 * @param refuse_memory Whether leg 2's device-memory allocations fail.
 * @return              Whether leg 2 composed the frame.
 */
static bool
compose_frame (struct context                          *context,
               struct composition                      *composition,
               struct composition_frame_settings const *settings,
               uint32_t                                 width,
               uint32_t                                 height,
               VkFormat                                 format,
               bool                                     linear_hdr,
               bool                                     refuse_memory)
{
	require(composition_prepare(composition, width, height, format, settings, linear_hdr, false, 0, false),
	        "prepare failed: %s", composition_reason(composition));
	composition_record_capture(composition, context_begin(context), context->swapchain, settings);
	context_submit(context);
	memcpy(composition_model_pixels(composition), composition_proxy_pixels(composition),
	       composition_model_bytes(composition));
	tracker.refuse_memory = refuse_memory;
	bool const composed = composition_record_compose(composition, context_begin(context), context->swapchain,
	                                                 settings);
	tracker.refuse_memory = false;
	context_submit(context);
	return composed;
}

/** @brief The state directory that the captures go under, removed with what it holds when the test
 *         ends.
 */
static char *state_home;

/** @brief Removes the state directory, at exit; a failure ends the test with 1. */
static void
remove_state_home (void)
{
	if (!state_home)
		return;
	if (!support_remove_tree(state_home)) {
		fprintf(stderr, "composition rebuild test: cannot remove %s\n", state_home);
		_exit(1);
	}
	free(state_home);
	state_home = nullptr;
}

int
main (void)
{
	struct context context = {};
	if (!context_init(&context)) {
		context_fini(&context);
		puts("SKIP: no Vulkan device with storage writes without format");
		return 77;
	}
	uint32_t const width = 320;
	uint32_t const height = 192;
	VkFormat const format = VK_FORMAT_B8G8R8A8_UNORM;
	context_make_swapchain_image(&context, width, height, format);
	VkImageMemoryBarrier const barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = context.swapchain,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	vkCmdPipelineBarrier(context_begin(&context), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	context_submit(&context);

	struct composition composition = composition_empty();
	require(composition_init(&composition, &context.device_table, &context.instance_table, context.device,
	                         context.physical) == VK_SUCCESS,
	        "%s", composition_reason(&composition));
	// Every step changes the model raster or the colour domain, so composition_prepare() destroys
	// and recreates the composition surfaces while the pass and its sets live on.
	static struct step const steps[] = {
		{1.0f, false}, {0.5f, false}, {1.0f, false}, {0.5f, false}, {0.75f, false}, {0.25f, false},
		{0.625f, false}, {1.0f, true}, {0.5f, true}, {1.0f, false}, {0.5f, false}, {0.75f, false},
	};
	uint32_t const step_count = sizeof steps / sizeof *steps;
	uint32_t composed = 0;
	for (uint32_t i = 0; i < step_count; ++i) {
		struct composition_frame_settings settings = composition_frame_settings();
		settings.working_scale = steps[i].working_scale;
		for (uint32_t frame = 0; frame < 3; ++frame)
			composed += compose_frame(&context, &composition, &settings, width, height, format,
			                          steps[i].linear_hdr, false);
	}
	printf("%u frames composed, %u descriptor set binds, %u view handles recycled, %u stale bindings bound\n",
	       composed, tracker.binds, tracker.recycled, tracker.stale);
	require(composed == 3 * step_count, "a frame was not composed");
	require(!tracker.stale, "a dispatch bound a descriptor for a destroyed view");
	printf("PASS: no descriptor outlived its image view across %u builds\n", step_count);

	// Captures go under XDG_STATE_HOME; keep them out of the user's state directory.
	state_home = support_temp_dir("/tmp", "dlsslop-amd-composition", nullptr);
	require(state_home, "create a temporary state directory");
	// At exit, so a failed check that exits removes it too.
	require(!atexit(remove_state_home), "atexit failed");
	require(!setenv("XDG_STATE_HOME", state_home, 1), "set XDG_STATE_HOME");
	struct composition_frame_settings const settings = composition_frame_settings();
	composition_request_capture(&composition, 1, 7);
	for (uint32_t frame = 0; frame < 2; ++frame) {
		require(compose_frame(&context, &composition, &settings, width, height, format, false, true),
		        "a frame whose capture buffer failed was not composed");
		require(capture_writer_active(&composition.capture) && !composition_capture_recorded(&composition),
		        "a capture without its host buffer recorded a pair");
	}
	require(compose_frame(&context, &composition, &settings, width, height, format, false, false)
	        && composition_capture_recorded(&composition),
	        "the capture did not record its pair once the buffer existed");
	composition_write_captured_frame(&composition);
	char *manifest = support_format(nullptr, "%s/dlssnr/captures/manifest.txt", state_home);
	require(manifest, "out of memory");
	struct stat st;
	require(!capture_writer_active(&composition.capture) && !stat(manifest, &st) && S_ISREG(st.st_mode),
	        "the recorded pair did not complete the capture");
	free(manifest);
	manifest = nullptr;
	require(compose_frame(&context, &composition, &settings, width, height, format, false, false)
	        && !composition_capture_recorded(&composition),
	        "a frame after the capture recorded a pair");
	bool const printed = puts("PASS: a capture records pairs only into an allocated host buffer") != EOF;
	composition_fini(&composition);
	context_fini(&context);
	free(tracker.written);
	tracker.written = nullptr;
	free(tracker.views);
	tracker.views = nullptr;
	return printed ? 0 : 1;
}
