/** @file
 *
 * Executes the shipped composition SPIR-V with identity and darkening models. This checks
 * transfer/domain handling on a Vulkan device, not the HIP neural network.
 */
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "dlssnr_pass.h"

/** @brief The frame's width and height. */
enum {
	WIDTH  = 16,
	HEIGHT = 8
};

/** @brief The frame's float components. */
static constexpr size_t COMPONENTS = (size_t)WIDTH * HEIGHT * 4;

/** @brief The frame's size as floats. */
static constexpr size_t FLOAT_BYTES = COMPONENTS * sizeof (float);

/** @brief The resolve's guard. */
static constexpr float MAX_RATIO = 2;

/** @brief The layout the passes read their images in. */
static constexpr VkImageLayout READ_ONLY = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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
	fputs("hdr-shader-test: ", stderr);
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

/** @brief The test's device and the composition pass on it. */
struct context {
	struct dlss_nr_pass   pass;           //!< The composition pass.
	VkCommandPool         pool;           //!< The command pool.
	struct instance_table instance_table; //!< The instance's entry points.
	struct device_table   device_table;   //!< The device's.
	VkInstance            instance;       //!< The instance.
	VkPhysicalDevice      physical;       //!< The physical device.
	VkDevice              device;         //!< The device.
	VkQueue               queue;          //!< Its compute queue.
	VkCommandBuffer       cmd;            //!< The command buffer.
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
	dlss_nr_pass_fini(&c->pass);
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

/** @brief Makes the test's device, software Vulkan if a CPU device qualifies, otherwise hardware,
 *         and the composition pass on it.
 *
 * @param c The context, empty. It must not move once this returns.
 * @return  true if a device qualified.
 */
static bool
context_init (struct context *c)
{
	VkApplicationInfo const app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "dlsslop-amd-hdr-shader-test",
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
	printf("HDR shader device: %s\n", properties.deviceName);
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
	c->pass = dlss_nr_pass(&c->device_table, &c->instance_table, c->device, c->physical);
	require(c->pass.error == VK_SUCCESS, "composition pass initialization failed");
	return true;
}

/** @brief A memory type of the device.
 *
 * @param c      The context.
 * @param mask   The types that a resource can take.
 * @param wanted The properties wanted.
 * @return       The first of them that has the properties.
 */
static uint32_t
memory_type (struct context const  *c,
             uint32_t               mask,
             VkMemoryPropertyFlags  wanted)
{
	VkPhysicalDeviceMemoryProperties properties = {};
	vkGetPhysicalDeviceMemoryProperties(c->physical, &properties);
	for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
		if ((mask & (1u << i)) && (properties.memoryTypes[i].propertyFlags & wanted) == wanted)
			return i;
	require(false, "required memory type unavailable");
	return 0;
}

/** @brief An image of the test, with its view and memory, and the layout the commands recorded so
 *         far leave it in. A zeroed image is an empty one.
 */
struct image {
	VkImage        image;  //!< The image.
	VkImageView    view;   //!< Its view.
	VkDeviceMemory memory; //!< Its memory.
	VkImageLayout  layout; //!< Its layout.
};

/** @brief Makes an image that can be sampled, stored to and copied, in device memory.
 *
 * @param dest   An empty image.
 * @param c      The context.
 * @param format Its format.
 * @param width  Its width.
 * @param height Its height.
 */
static void
image_init (struct image         *dest,
            struct context const *c,
            VkFormat              format,
            uint32_t              width,
            uint32_t              height)
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
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		         | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkImage image;
	vk_check(vkCreateImage(c->device, &info, nullptr, &image), "create image");
	dest->image = image;
	VkMemoryRequirements requirements = {};
	vkGetImageMemoryRequirements(c->device, dest->image, &requirements);
	VkMemoryAllocateInfo const allocation = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size,
		.memoryTypeIndex = memory_type(c, requirements.memoryTypeBits, 0),
	};
	VkDeviceMemory memory;
	vk_check(vkAllocateMemory(c->device, &allocation, nullptr, &memory), "allocate image memory");
	dest->memory = memory;
	vk_check(vkBindImageMemory(c->device, dest->image, dest->memory, 0), "bind image memory");
	VkImageViewCreateInfo const view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = dest->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	VkImageView view;
	vk_check(vkCreateImageView(c->device, &view_info, nullptr, &view), "create image view");
	dest->view = view;
}

/** @brief Destroys what an image holds, then leaves it empty.
 *
 * @param dest The image.
 * @param c    The context.
 */
static void
image_fini (struct image         *dest,
            struct context const *c)
{
	if (dest->view)
		vkDestroyImageView(c->device, dest->view, nullptr);
	if (dest->image)
		vkDestroyImage(c->device, dest->image, nullptr);
	if (dest->memory)
		vkFreeMemory(c->device, dest->memory, nullptr);
	*dest = (struct image){};
}

/** @brief Records an image's move to another layout, after everything before it.
 *
 * @param img  The image.
 * @param c    The context.
 * @param next The layout.
 */
static void
image_transition (struct image         *img,
                  struct context const *c,
                  VkImageLayout         next)
{
	VkImageMemoryBarrier const barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = img->layout == VK_IMAGE_LAYOUT_UNDEFINED
		                 ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
		.oldLayout = img->layout,
		.newLayout = next,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = img->image,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	vkCmdPipelineBarrier(c->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
	                     nullptr, 0, nullptr, 1, &barrier);
	img->layout = next;
}

/** @brief A host-visible, coherent buffer of the test, mapped. A zeroed buffer is an empty one. */
struct buffer {
	VkBuffer        buffer; //!< The buffer.
	VkDeviceMemory  memory; //!< Its memory.
	void           *mapped; //!< The memory, mapped.
};

/** @brief Makes a buffer that copies go to and from, and maps it.
 *
 * @param dest  An empty buffer.
 * @param c     The context.
 * @param bytes Its size.
 */
static void
buffer_init (struct buffer        *dest,
             struct context const *c,
             size_t                bytes)
{
	VkBufferCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bytes,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkBuffer buffer;
	vk_check(vkCreateBuffer(c->device, &info, nullptr, &buffer), "create buffer");
	dest->buffer = buffer;
	VkMemoryRequirements requirements = {};
	vkGetBufferMemoryRequirements(c->device, dest->buffer, &requirements);
	VkMemoryPropertyFlags const host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VkMemoryAllocateInfo const allocation = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size,
		.memoryTypeIndex = memory_type(c, requirements.memoryTypeBits, host),
	};
	VkDeviceMemory memory;
	vk_check(vkAllocateMemory(c->device, &allocation, nullptr, &memory), "allocate buffer memory");
	dest->memory = memory;
	vk_check(vkBindBufferMemory(c->device, dest->buffer, dest->memory, 0), "bind buffer memory");
	void *mapped;
	vk_check(vkMapMemory(c->device, dest->memory, 0, bytes, 0, &mapped), "map buffer");
	dest->mapped = mapped;
}

/** @brief Unmaps and destroys what a buffer holds, then leaves it empty.
 *
 * @param dest The buffer.
 * @param c    The context.
 */
static void
buffer_fini (struct buffer        *dest,
             struct context const *c)
{
	if (dest->mapped)
		vkUnmapMemory(c->device, dest->memory);
	if (dest->buffer)
		vkDestroyBuffer(c->device, dest->buffer, nullptr);
	if (dest->memory)
		vkFreeMemory(c->device, dest->memory, nullptr);
	*dest = (struct buffer){};
}

/** @brief A binary16's value.
 *
 * @param bits The binary16, finite.
 * @return     Its value.
 */
static float
half_to_float (uint16_t bits)
{
	unsigned const exponent = (bits >> 10) & 31u;
	unsigned const mantissa = bits & 1023u;
	float const sign = bits & 0x8000u ? -1.0f : 1.0f;
	if (!exponent)
		return sign * ldexpf((float)mantissa, -24);
	require(exponent != 31, "non-finite proxy half");
	return sign * ldexpf((float)(1024u + mantissa), (int)exponent - 25);
}

/** @brief A float as binary16, rounded to nearest even; finite values within binary16 range.
 *
 * @param value The float.
 * @return      The binary16.
 */
static uint16_t
float_to_half (float value)
{
	uint32_t const sign = signbit(value) ? 0x8000u : 0u;
	float const magnitude = fabsf(value);
	if (magnitude < 0x1p-14f)
		return (uint16_t)(sign | (uint32_t)nearbyintf(magnitude * 0x1p24f));
	uint32_t bits;
	memcpy(&bits, &magnitude, sizeof bits);
	uint32_t half = (((bits >> 23) - 112u) << 10) | ((bits >> 13) & 1023u);
	uint32_t const rest = bits & 0x1fffu;
	half += rest > 0x1000u || (rest == 0x1000u && (half & 1u));
	require(half < 0x7c00u, "value exceeds binary16 range");
	return (uint16_t)(sign | half);
}

/** @brief One channel of a proxy-format texel buffer: UNORM8 or binary16.
 *
 * @param bytes The buffer.
 * @param index The channel.
 * @param unorm Whether the buffer is UNORM8.
 * @return      The channel's value.
 */
static float
load_texel (unsigned char const *bytes,
            size_t               index,
            bool                 unorm)
{
	if (unorm)
		return (float)bytes[index] / 255.0f;
	uint16_t half;
	memcpy(&half, bytes + index * 2, sizeof half);
	return half_to_float(half);
}

/** @brief Stores one channel of a proxy-format texel buffer: UNORM8 or binary16.
 *
 * @param bytes The buffer.
 * @param index The channel.
 * @param unorm Whether the buffer is UNORM8.
 * @param value The channel's value.
 */
static void
store_texel (unsigned char *bytes,
             size_t         index,
             bool           unorm,
             float          value)
{
	if (unorm) {
		float const clamped = value < 0.0f ? 0.0f : 1.0f < value ? 1.0f : value;
		bytes[index] = (uint8_t)lroundf(clamped * 255.0f);
		return;
	}
	uint16_t const half = float_to_half(value);
	memcpy(bytes + index * 2, &half, sizeof half);
}

/** @brief How a run composes a frame. reduced is 16 bits wide, which fills the padding. */
struct options {
	float    white;        //!< The white point.
	float    model_scale;  //!< The model's answer is the proxy with its RGB scaled by this.
	uint32_t transfer;     //!< The transfer mode.
	uint32_t hdr_proxy;    //!< HdrProxy; 2: display-encoded native HIP model input.
	uint32_t debug_view;   //!< The debug view.
	VkFormat proxy_format; //!< The proxy's format.
	uint16_t reduced;      //!< Whether the model runs at half size.
	bool     pq;           //!< Whether the frame is PQ.
	bool     sdr;          //!< Whether the frame is display-referred.
};

/** @brief The options of a run that changes nothing: linear HDR, at full size. */
static struct options
options (void)
{
	return (struct options){
		.white = 1,
		.model_scale = 1,
		.transfer = 2,
		.hdr_proxy = 2,
		.proxy_format = VK_FORMAT_R16G16B16A16_SFLOAT,
	};
}

/** @brief What a run read back: the composed frame and the proxy. */
struct result {
	float output[COMPONENTS]; //!< The composed frame.
	float proxy[COMPONENTS];  //!< The proxy.
};

/** @brief Resets and begins the context's command buffer.
 *
 * @param c The context.
 */
static void
begin (struct context const *c)
{
	vk_check(vkResetCommandBuffer(c->cmd, 0), "reset command buffer");
	VkCommandBufferBeginInfo const info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vk_check(vkBeginCommandBuffer(c->cmd, &info), "begin command buffer");
}

/** @brief Makes the copies visible to the host, submits the context's command buffer and waits.
 *
 * @param c The context.
 */
static void
submit (struct context const *c)
{
	VkMemoryBarrier const host = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	};
	vkCmdPipelineBarrier(c->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0,
	                     nullptr, 0, nullptr);
	vk_check(vkEndCommandBuffer(c->cmd), "end command buffer");
	VkSubmitInfo const info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &c->cmd,
	};
	vk_check(vkQueueSubmit(c->queue, 1, &info, VK_NULL_HANDLE), "submit shader test");
	vk_check(vkQueueWaitIdle(c->queue), "wait shader test");
}

/** @brief Encodes (and downsamples) in one submission, reads the proxy back, derives the model's
 *         answer on the CPU, then resolves it in a second submission, as the layer does around the
 *         worker.
 *
 * @param c     The context.
 * @param input The frame.
 * @param o     How to compose it.
 * @return      What the run read back.
 */
static struct result
run (struct context       *c,
     float const           input[static COMPONENTS],
     struct options const *o)
{
	bool const unorm = o->proxy_format == VK_FORMAT_R8G8B8A8_UNORM;
	uint32_t const model_width = o->reduced ? WIDTH / 2 : WIDTH;
	uint32_t const model_height = o->reduced ? HEIGHT / 2 : HEIGHT;
	size_t const texel = unorm ? 1 : 2;
	size_t const model_components = (size_t)model_width * model_height * 4;
	size_t const proxy_offset = FLOAT_BYTES * 2;
	size_t const small_offset = proxy_offset + COMPONENTS * texel;
	size_t const model_offset = small_offset + model_components * texel;
	struct image native = {};
	struct image proxy = {};
	struct image output = {};
	struct image small = {};
	struct image model = {};
	image_init(&native, c, VK_FORMAT_R32G32B32A32_SFLOAT, WIDTH, HEIGHT);
	image_init(&proxy, c, o->proxy_format, WIDTH, HEIGHT);
	image_init(&output, c, VK_FORMAT_R32G32B32A32_SFLOAT, WIDTH, HEIGHT);
	if (o->reduced)
		image_init(&small, c, o->proxy_format, model_width, model_height);
	image_init(&model, c, o->proxy_format, model_width, model_height);
	struct buffer staging = {};
	buffer_init(&staging, c, model_offset + model_components * texel);
	unsigned char *const bytes = staging.mapped;
	memcpy(bytes, input, FLOAT_BYTES);

	begin(c);
	image_transition(&native, c, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkBufferImageCopy copy = {
		.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
		.imageExtent = {WIDTH, HEIGHT, 1},
	};
	vkCmdCopyBufferToImage(c->cmd, staging.buffer, native.image, native.layout, 1, &copy);
	image_transition(&native, c, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	image_transition(&proxy, c, VK_IMAGE_LAYOUT_GENERAL);
	image_transition(&output, c, VK_IMAGE_LAYOUT_GENERAL);
	struct dlss_nr_constants constants = {
		.width = WIDTH,
		.guide_width = WIDTH,
		.height = HEIGHT,
		.guide_height = HEIGHT,
		.white_point = o->white,
		.transfer_strength = 1,
		.colour_strength = 1,
		.max_ratio = MAX_RATIO,
		.passthrough = o->sdr,
		.mv_scale_x = 1,
		.mv_scale_y = 1,
		.compare_split = 0.5f,
		.compare_zoom = 1,
		.transfer = o->transfer,
		.debug_view = o->debug_view,
		.debug_scale = 1,
		.apply_model = 1,
		.exposure_pre_mul = 1,
		.hdr_proxy = o->hdr_proxy,
		.hdr_transfer = o->pq,
		.colour_trust = 2,
		.ratio_smooth = 1,
	};
	require(dlss_nr_pass_dispatch(&c->pass, c->cmd, &constants, WIDTH, HEIGHT, native.view, VK_NULL_HANDLE,
	                              VK_NULL_HANDLE, VK_NULL_HANDLE, proxy.view, VK_NULL_HANDLE, READ_ONLY,
	                              READ_ONLY),
	        "encode dispatch");
	image_transition(&proxy, c, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	struct image const *const source = o->reduced ? &small : &proxy;
	if (o->reduced) {
		image_transition(&small, c, VK_IMAGE_LAYOUT_GENERAL);
		constants.mode = DLSS_NR_MODE_DOWNSAMPLE;
		constants.width = model_width;
		constants.height = model_height;
		require(dlss_nr_pass_dispatch(&c->pass, c->cmd, &constants, model_width, model_height, proxy.view,
		                              VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, small.view,
		                              VK_NULL_HANDLE, READ_ONLY, READ_ONLY),
		        "downsample dispatch");
		image_transition(&small, c, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		VkBufferImageCopy reduced = copy;
		reduced.bufferOffset = small_offset;
		reduced.imageExtent = (VkExtent3D){model_width, model_height, 1};
		vkCmdCopyImageToBuffer(c->cmd, small.image, small.layout, staging.buffer, 1, &reduced);
		image_transition(&small, c, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		constants.width = WIDTH;
		constants.height = HEIGHT;
	}
	image_transition(&proxy, c, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	copy.bufferOffset = proxy_offset;
	vkCmdCopyImageToBuffer(c->cmd, proxy.image, proxy.layout, staging.buffer, 1, &copy);
	image_transition(&proxy, c, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	submit(c);

	struct result result = {};
	for (size_t i = 0; i < COMPONENTS; ++i)
		result.proxy[i] = load_texel(bytes + proxy_offset, i, unorm);
	// The shipped downsample is an exact area average: at half size, the mean of each 2x2 block, with
	// the alpha of the block's last texel. One rounding step in the proxy's format.
	for (size_t i = 0; o->reduced && i < model_components; ++i) {
		size_t const x = i / 4 % model_width * 2;
		size_t const y = i / 4 / model_width * 2;
		size_t const channel = i % 4;
		float const *const at = &result.proxy[(y * WIDTH + x) * 4 + channel];
		// The block's texels: (0, 0), (1, 0), (0, 1) and (1, 1).
		float const a = at[0];
		float const b = at[4];
		float const d = at[WIDTH * 4];
		float const e = at[WIDTH * 4 + 4];
		float const expected = channel == 3 ? e : (a + b + d + e) / 4;
		float const relative = fabsf(expected) * 0x1p-10f;
		float const tolerance = channel == 3 ? 0
		                        : unorm      ? 1.0f / 255
		                        : relative < 0x1p-24f ? 0x1p-24f : relative;
		float const actual = load_texel(bytes + small_offset, i, unorm);
		if (fabsf(actual - expected) <= tolerance)
			continue;
		fprintf(stderr, "downsample component %zu: expected %.9g, got %.9g\n", i, (double)expected,
		        (double)actual);
		require(false, "downsample is not the proxy's area average");
	}
	unsigned char const *const answer = bytes + (o->reduced ? small_offset : proxy_offset);
	for (size_t i = 0; i < model_components; ++i)
		store_texel(bytes + model_offset, i, unorm,
		            load_texel(answer, i, unorm) * (i % 4 == 3 ? 1 : o->model_scale));

	begin(c);
	image_transition(&model, c, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkBufferImageCopy upload = copy;
	upload.bufferOffset = model_offset;
	upload.imageExtent = (VkExtent3D){model_width, model_height, 1};
	vkCmdCopyBufferToImage(c->cmd, staging.buffer, model.image, model.layout, 1, &upload);
	image_transition(&model, c, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	constants.mode = DLSS_NR_MODE_RESOLVE;
	require(dlss_nr_pass_dispatch(&c->pass, c->cmd, &constants, WIDTH, HEIGHT, source->view, model.view,
	                              native.view, VK_NULL_HANDLE, output.view, VK_NULL_HANDLE, READ_ONLY, READ_ONLY),
	        "resolve dispatch");
	image_transition(&output, c, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	copy.bufferOffset = FLOAT_BYTES;
	vkCmdCopyImageToBuffer(c->cmd, output.image, output.layout, staging.buffer, 1, &copy);
	submit(c);
	memcpy(result.output, bytes + FLOAT_BYTES, FLOAT_BYTES);

	buffer_fini(&staging, c);
	image_fini(&model, c);
	image_fini(&small, c);
	image_fini(&output, c);
	image_fini(&proxy, c);
	image_fini(&native, c);
	return result;
}

/** @brief Ends the test unless an identity model preserved the frame within a tolerance, its alpha
 *         exactly.
 *
 * @param name               What ran.
 * @param input              The frame.
 * @param result             What the run read back.
 * @param absolute_tolerance The absolute error allowed.
 * @param relative_tolerance The error allowed relative to the frame's.
 */
static void
identity (char const          *name,
          float const          input[static COMPONENTS],
          struct result const *result,
          float                absolute_tolerance,
          float                relative_tolerance)
{
	float maximum = 0;
	float maximum_relative = 0;
	for (size_t i = 0; i < COMPONENTS; ++i) {
		float const error = fabsf(input[i] - result->output[i]);
		if (maximum < error)
			maximum = error;
		if (input[i] != 0) {
			float const relative = error / fabsf(input[i]);
			if (maximum_relative < relative)
				maximum_relative = relative;
		}
		if (i % 4 == 3)
			require(result->output[i] == input[i], "native alpha was not preserved");
		if (!isfinite(result->output[i])
		    || error > absolute_tolerance + relative_tolerance * fabsf(input[i])) {
			fprintf(stderr, "%s component %zu: input %.9g output %.9g error %.9g\n", name, i,
			        (double)input[i], (double)result->output[i], (double)error);
			require(false, "identity model did not preserve native frame");
		}
	}
	printf("%s: identity maximum absolute error %.9g, relative %.9g\n", name, (double)maximum,
	       (double)maximum_relative);
}

/** @brief Linear light as PQ.
 *
 * @param linear The light, 1 for 10,000 nits.
 * @return       Its PQ code.
 */
static float
pq_of (float linear)
{
	double const l = (double)linear;
	double const q = pow(l < 0.0 ? 0.0 : l, 2610.0 / 16384.0);
	return (float)pow((3424.0 / 4096.0 + (2413.0 / 128.0) * q) / (1.0 + (2392.0 / 128.0) * q), 2523.0 / 32.0);
}

/** @brief A PQ code as linear light.
 *
 * @param code The code.
 * @return     The light, 1 for 10,000 nits.
 */
static float
pq_to_linear (float code)
{
	double const c = (double)code;
	double const p = pow(c < 0.0 ? 0.0 : c, 32.0 / 2523.0);
	double const lifted = p - 3424.0 / 4096.0;
	return (float)pow((lifted < 0.0 ? 0.0 : lifted) / (2413.0 / 128.0 - (2392.0 / 128.0) * p), 16384.0 / 2610.0);
}

/** @brief A darkening model at a hard edge: the enlarged edit is far larger than the dark side's own
 *         light. No output channel may fall below the frame's own: zero, or a wide-gamut pixel's own
 *         negative BT.709 coordinate. The PQ store clamps at zero, so a PQ run can fail this only
 *         with a non-finite channel.
 *
 * Every pixel's luminance must also stay above input / (MaxRatio x 1.6). The matched residual
 * (transfer 1) keeps it by the resolve's 1/MaxRatio floor, with the per-pixel band as margin. Native
 * plus edit (transfer 2) keeps it by its band's lower edge, 1/1.6 of a neighbourhood level the guard
 * holds within MaxRatio of the frame's: at the dark side the clamped sum is the zero vector, and the
 * edge makes the missing light up in the frame's own colour rather than leaving it black.
 *
 * @param name   What ran.
 * @param input  The frame.
 * @param result What the run read back.
 * @param pq     Whether the frame is PQ.
 */
static void
no_negative_light (char const          *name,
                   float const          input[static COMPONENTS],
                   struct result const *result,
                   bool                 pq)
{
	// BT.709 weights for linear scRGB, BT.2020 for PQ once decoded.
	static float const weights[2][3] = {{0.2126f, 0.7152f, 0.0722f}, {0.2627f, 0.6780f, 0.0593f}};
	float const band = MAX_RATIO * 1.6f;
	float lowest = INFINITY;
	for (size_t pixel = 0; pixel < COMPONENTS / 4; ++pixel) {
		float in = 0;
		float out = 0;
		for (size_t channel = 0; channel < 3; ++channel) {
			float const value = result->output[pixel * 4 + channel];
			float const source = input[pixel * 4 + channel];
			if (!isfinite(value) || value < (0.0f < source ? 0.0f : source)) {
				fprintf(stderr, "%s pixel %zu channel %zu: %.9g from %.9g\n", name, pixel, channel,
				        (double)value, (double)source);
				require(false, "darkening edit drove an edge channel below the frame's own");
			}
			in += weights[pq][channel] * (pq ? pq_to_linear(source) : source);
			out += weights[pq][channel] * (pq ? pq_to_linear(value) : value);
		}
		float const ratio = out / in;
		if (ratio < lowest)
			lowest = ratio;
		if (out >= in / band)
			continue;
		fprintf(stderr, "%s pixel %zu: luminance %.9g from %.9g\n", name, pixel, (double)out, (double)in);
		require(false, "darkening edit crushed an edge pixel below the luminance band");
	}
	printf("%s: no channel below the frame's own, lowest luminance ratio %.9g\n", name, (double)lowest);
}

/** @brief Debug view 4 paints how far the colour bound let the model's colour through: red held
 *         back, green passed, together always paper white. Checked in BT.709 linear light, so a PQ
 *         swapchain must receive BT.2020 PQ codes for paper white rather than linear values read as
 *         up to 10,000 nits.
 *
 * @param name   What ran.
 * @param input  The frame.
 * @param result What the run read back.
 * @param pq     Whether the frame is PQ.
 * @param white  Paper white.
 */
static void
colour_bound_view (char const          *name,
                   float const          input[static COMPONENTS],
                   struct result const *result,
                   bool                 pq,
                   float                white)
{
	static float const to709[3][3] = {{1.660491f, -0.587641f, -0.072850f}, {-0.124550f, 1.132900f, -0.008349f},
	                                  {-0.018151f, -0.100579f, 1.118730f}};
	for (size_t pixel = 0; pixel < COMPONENTS / 4; ++pixel) {
		float const *const out = &result->output[pixel * 4];
		float rgb[3] = {out[0], out[1], out[2]};
		if (pq) {
			float const bt2020[3] = {pq_to_linear(out[0]), pq_to_linear(out[1]), pq_to_linear(out[2])};
			for (size_t channel = 0; channel < 3; ++channel)
				rgb[channel] = to709[channel][0] * bt2020[0] + to709[channel][1] * bt2020[1]
				               + to709[channel][2] * bt2020[2];
		}
		float const tolerance = white * 0.001f;
		if (fabsf(rgb[0] + rgb[1] - white) <= tolerance && fabsf(rgb[2]) <= tolerance && rgb[0] >= -tolerance
		    && rgb[1] >= -tolerance && out[3] == input[pixel * 4 + 3])
			continue;
		fprintf(stderr, "%s pixel %zu: %.9g %.9g %.9g %.9g, linear %.9g %.9g %.9g, expected red plus "
		        "green %.9g\n", name, pixel, (double)out[0], (double)out[1], (double)out[2], (double)out[3],
		        (double)rgb[0], (double)rgb[1], (double)rgb[2], (double)white);
		require(false, "colour-bound view is not paper white split between red and green");
	}
	printf("%s: colour-bound view sums to paper white %.9g\n", name, (double)white);
}

/** @brief The size of a run's name. */
static constexpr size_t NAME_SIZE = 80;

/** @brief Formats a run's name, which must fit.
 *
 * @param name The name's buffer.
 * @param fmt  A printf format.
 * @param ...  Its arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
run_name (char        name[static NAME_SIZE],
          char const *fmt,
          ...)
{
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(name, NAME_SIZE, fmt, args);
	va_end(args);
	require(n >= 0 && n < (int)NAME_SIZE, "a run's name does not fit");
}

/** @brief The frames that the runs compose. */
static struct {
	float linear[COMPONENTS];     //!< Linear HDR.
	float pq[COMPONENTS];         //!< HDR10 PQ/BT.2020.
	float sdr[COMPONENTS];        //!< Display-referred.
	float scrgb[COMPONENTS];      //!< scRGB with negative components.
	float edge[COMPONENTS];       //!< A column-by-column edge, linear.
	float edge_pq[COMPONENTS];    //!< The same edge, PQ.
	float gamut_edge[COMPONENTS]; //!< A BT.2020 green in scRGB beside a bright white column.
} frames;

int
main (void)
{
	struct context context = {};
	if (!context_init(&context)) {
		context_fini(&context);
		puts("SKIP: no Vulkan compute device with formatless storage writes");
		return 77;
	}
	static float const levels[] = {0.003f, 0.03f, 0.2f, 0.7f, 1.0f, 2.0f, 4.0f, 10.0f};
	// Encode a BT.709 color as BT.2020 PQ, matching an HDR10 swapchain.
	static float const to2020[3][3] = {{0.627404f, 0.329283f, 0.043313f}, {0.069097f, 0.919540f, 0.011362f},
	                                   {0.016391f, 0.088013f, 0.895595f}};
	for (size_t pixel = 0; pixel < COMPONENTS / 4; ++pixel) {
		float const value = levels[pixel % 8];
		float const rgb[] = {value, value * 0.8f, value * 0.6f};
		for (size_t channel = 0; channel < 3; ++channel) {
			frames.linear[pixel * 4 + channel] = rgb[channel];
			float component = 0;
			for (size_t k = 0; k < 3; ++k)
				component += to2020[channel][k] * rgb[k];
			frames.pq[pixel * 4 + channel] = pq_of(component * 0.0203f);
			frames.sdr[pixel * 4 + channel] = rgb[channel] / 10.0f;
		}
		frames.linear[pixel * 4 + 3] = frames.pq[pixel * 4 + 3] = frames.sdr[pixel * 4 + 3] = 0.75f;
	}
	frames.pq[0] = 0.65f;
	frames.pq[1] = 0.60f;
	frames.pq[2] = 0.55f;
	// BT.2020 primaries lie outside the model's BT.709 gamut. An identity model must still preserve
	// their native chroma, including black channels.
	for (size_t mask = 1; mask < 8; ++mask)
		for (size_t channel = 0; channel < 3; ++channel)
			frames.pq[mask * 4 + channel] = mask & ((size_t)1 << channel) ? 0.7f : 0.0f;
	struct options const defaults = options();
	struct result const one = run(&context, frames.linear, &defaults);
	// FP16 proxy rounding is magnified by the nonlinear encode/composition. A 0.1% relative bound
	// admits that error while catching highlight clipping and a missing PQ transfer by orders of
	// magnitude. PQ is checked in code units.
	identity("linear HDR, soft knee", frames.linear, &one, 0.0001f, 0.001f);
	float largest = one.output[0];
	for (size_t i = 1; i < COMPONENTS; ++i)
		if (largest < one.output[i])
			largest = one.output[i];
	require(largest > 9.9f, "HDR highlights were clipped to SDR range");
	struct options hdr10 = options();
	hdr10.pq = true;
	struct result r = run(&context, frames.pq, &hdr10);
	identity("HDR10 PQ/BT.2020, soft knee", frames.pq, &r, 0.0002f, 0.0001f);
	struct options display = options();
	display.sdr = true;
	r = run(&context, frames.sdr, &display);
	identity("SDR, FP16 encoded proxy", frames.sdr, &r, 0.0001f, 0.001f);
	struct options brighter = options();
	brighter.white = 2;
	struct result const two = run(&context, frames.linear, &brighter);
	identity("linear HDR, white point 2", frames.linear, &two, 0.0001f, 0.001f);
	float difference = 0;
	for (size_t i = 0; i < COMPONENTS; ++i) {
		float const change = fabsf(one.proxy[i] - two.proxy[i]);
		if (i % 4 != 3 && difference < change)
			difference = change;
	}
	require(difference > 0.05f, "changing white point did not change encoded HDR proxy");
	printf("white-point control: maximum encoded proxy change %.9g\n", (double)difference);
	struct options reduced = options();
	reduced.reduced = true;
	r = run(&context, frames.linear, &reduced);
	identity("linear HDR, reduced FP16 proxy", frames.linear, &r, 0.0001f, 0.001f);
	struct options reduced_hdr10 = reduced;
	reduced_hdr10.pq = true;
	r = run(&context, frames.pq, &reduced_hdr10);
	identity("HDR10 PQ/BT.2020, reduced FP16 proxy", frames.pq, &r, 0.0002f, 0.0001f);
	// Transfer 1, the matched residual, is the non-native default and a user option.
	struct options matched = reduced;
	matched.transfer = 1;
	r = run(&context, frames.linear, &matched);
	identity("linear HDR, reduced FP16 proxy, transfer 1", frames.linear, &r, 0.0001f, 0.001f);
	matched.pq = true;
	r = run(&context, frames.pq, &matched);
	identity("HDR10 PQ/BT.2020, reduced FP16 proxy, transfer 1", frames.pq, &r, 0.0002f, 0.0001f);
	// HdrProxy 0 is native HIP's default: HDR transport off, an RGBA8 proxy. Eight bits move a native
	// highlight by up to a couple of percent; the reduced path lays only the (zero) edit on the
	// native frame and stays exact.
	struct options rgba8 = options();
	rgba8.hdr_proxy = 0;
	rgba8.proxy_format = VK_FORMAT_R8G8B8A8_UNORM;
	r = run(&context, frames.linear, &rgba8);
	identity("linear HDR, RGBA8 proxy", frames.linear, &r, 0.03f, 0.025f);
	rgba8.reduced = true;
	r = run(&context, frames.linear, &rgba8);
	identity("linear HDR, reduced RGBA8 proxy", frames.linear, &r, 0.0001f, 0.001f);
	// scRGB permits negative BT.709 components for colors outside that gamut. Keep their native
	// values even though the model proxy is bounded.
	memcpy(frames.scrgb, frames.linear, sizeof frames.scrgb);
	static float const outside709[][3] = {{-0.1f, 1.0f, 0.2f}, {2.0f, -0.1f, 0.3f}, {0.2f, 0.5f, -0.05f},
	                                      {-0.2f, 4.0f, -0.1f}};
	for (size_t pixel = 0; pixel < 4; ++pixel)
		for (size_t channel = 0; channel < 3; ++channel)
			frames.scrgb[pixel * 4 + channel] = outside709[pixel][channel];
	r = run(&context, frames.scrgb, &defaults);
	identity("scRGB negative components, native FP16 proxy", frames.scrgb, &r, 0.0001f, 0.001f);
	require(r.output[0] < 0, "native scRGB negative component was clipped");
	r = run(&context, frames.scrgb, &reduced);
	identity("scRGB negative components, reduced FP16 proxy", frames.scrgb, &r, 0.0001f, 0.001f);
	require(r.output[0] < 0, "reduced scRGB negative component was clipped");
	// A model that darkens a column-by-column edge, below frame size: every transfer and transport
	// combination the native path can take.
	for (size_t i = 0; i < COMPONENTS; ++i) {
		frames.edge[i] = i % 4 == 3 ? 0.75f : i / 4 % 2 ? 0.01f : 0.9f;
		frames.edge_pq[i] = i % 4 == 3 ? 0.75f : pq_of(frames.edge[i] * 0.0203f);
	}
	char name[NAME_SIZE];
	for (uint32_t variant = 0; variant < 8; ++variant) {
		struct options darker = options();
		darker.reduced = true;
		darker.model_scale = 0.8f;
		darker.pq = variant & 1;
		darker.hdr_proxy = variant & 2 ? 2 : 0;
		darker.proxy_format = variant & 2 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
		darker.transfer = variant & 4 ? 2 : 1;
		run_name(name, "darkening edge, %s, HdrProxy %u, transfer %u", darker.pq ? "HDR10 PQ" : "linear HDR",
		         darker.hdr_proxy, darker.transfer);
		float const *const input = darker.pq ? frames.edge_pq : frames.edge;
		r = run(&context, input, &darker);
		no_negative_light(name, input, &r, darker.pq);
	}
	// A BT.2020 green in scRGB beside a bright white column, under native plus edit. The native
	// transports compose it from the frame's own hue; HdrProxy 1 carries it to the model with no such
	// fallback and relies on the sum's zero clamp to keep a channel above the frame's own, and on the
	// band's lower edge, which adds light in the frame's positive channels only, to keep its
	// luminance.
	static float const green2020[] = {-0.5876f, 1.1329f, -0.1006f};
	for (size_t i = 0; i < COMPONENTS; ++i)
		frames.gamut_edge[i] = i % 4 == 3 ? 0.75f : i / 4 % 2 ? green2020[i % 4] : 4.0f;
	for (uint32_t variant = 0; variant < 6; ++variant) {
		struct options darker = options();
		darker.reduced = true;
		darker.model_scale = variant & 1 ? 0.5f : 0.8f;
		darker.hdr_proxy = variant >> 1;
		darker.proxy_format = darker.hdr_proxy ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
		run_name(name, "wide-gamut edge x%.1f, HdrProxy %u, transfer 2", (double)darker.model_scale,
		         darker.hdr_proxy);
		r = run(&context, frames.gamut_edge, &darker);
		no_negative_light(name, frames.gamut_edge, &r, false);
	}
	struct options bound_hdr10 = hdr10;
	bound_hdr10.debug_view = 4;
	r = run(&context, frames.pq, &bound_hdr10);
	colour_bound_view("debug view 4, HDR10 PQ", frames.pq, &r, true, 0.0203f);
	struct options bound_linear = brighter;
	bound_linear.debug_view = 4;
	r = run(&context, frames.linear, &bound_linear);
	colour_bound_view("debug view 4, linear HDR, white point 2", frames.linear, &r, false, 2);
	struct options bound_sdr = display;
	bound_sdr.debug_view = 4;
	r = run(&context, frames.sdr, &bound_sdr);
	colour_bound_view("debug view 4, SDR", frames.sdr, &r, false, 1);
	context_fini(&context);
	return 0;
}
