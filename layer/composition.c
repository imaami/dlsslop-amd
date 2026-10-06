/** @file
 *
 * The composition: the settings a frame composes with, the surfaces and passes around the round
 * trip, the white point meter, and the transport pair.
 */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <unistd.h>

#include "../common/util.h"
#include "composition.h"
#include "log.h"
#include "shaders/meter_reduce_spv.h"

/** @brief The size of the meter's state buffer and of its host mirror, which share one layout,
 *         pinned to match the MeterState block in shaders/meter_reduce.comp.
 */
static constexpr size_t COMPOSITION_METER_STATE_BYTES = 128;

/** @brief Where the meter's state holds the resolved white point. */
static constexpr VkDeviceSize COMPOSITION_METER_RESOLVED_OFFSET = 8;

/** @brief The meter's push constants. */
struct meter_push {
	float    manual;     //!< The slider's white point.
	float    scale;      //!< The white point's multiplier.
	float    trim;       //!< The trim of a measured reading.
	float    hold_value; //!< The white point a held frame keeps, or 0.
	uint32_t source;     //!< enum WhitePointSource.
	uint32_t hold;       //!< 1 while a captured frame is held.
};

static_assert(sizeof (struct meter_push) == 24, "must match the push_constant block in meter_reduce.comp");

/** @brief The layout of the source and motion views that every dispatch of the pass states. */
static constexpr VkImageLayout COMPOSITION_READ_ONLY = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

/** @brief The whole of a 2D image with one level and one layer. */
static VkImageSubresourceRange const COMPOSITION_COLOR_RANGE = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

/** @brief The level and layer that a copy of a 2D image takes. */
static VkImageSubresourceLayers const COMPOSITION_COLOR_LAYERS = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };

/** @brief The reduce pass's bindings: the grid, and the state. */
static VkDescriptorSetLayoutBinding const COMPOSITION_METER_BINDINGS[] = {
	{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
	{ 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
};

/** @brief The descriptors of the reduce pass's one set. */
static VkDescriptorPoolSize const COMPOSITION_METER_POOL_SIZES[] = {
	{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
	{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1 }
};

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

/** @brief The larger of two numbers, as std::max compares them: @a a unless it is less than @a b. */
static uint32_t
max_u (uint32_t a,
       uint32_t b)
{
	return a < b ? b : a;
}

/** @brief A number between two bounds, as std::min(std::max(v, lo), hi) gives it, NaN included.
 *
 * @param v  The number.
 * @param lo The lower bound.
 * @param hi The upper bound.
 * @return   @a v, @a lo if @a v is less, or @a hi if @a v is greater.
 */
static float
bound (float v,
       float lo,
       float hi)
{
	float const low = v < lo ? lo : v;
	return hi < low ? hi : low;
}

/** @brief A setting between two bounds, or a fallback for a setting that is not a finite number.
 *
 * @param v        The setting.
 * @param lo       The lower bound.
 * @param hi       The upper bound.
 * @param fallback What a setting that is not finite becomes.
 * @return         The bounded setting, or @a fallback.
 */
static float
clamp (float v,
       float lo,
       float hi,
       float fallback)
{
	return isfinite(v) ? bound(v, lo, hi) : fallback;
}

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

VkFormat
composition_format (VkFormat swapchain_format)
{
	switch (swapchain_format) {
	case VK_FORMAT_B8G8R8A8_UNORM:
	case VK_FORMAT_B8G8R8A8_SRGB:
		return VK_FORMAT_B8G8R8A8_UNORM;
	case VK_FORMAT_R8G8B8A8_UNORM:
	case VK_FORMAT_R8G8B8A8_SRGB:
		return VK_FORMAT_R8G8B8A8_UNORM;
	case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
	case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
		return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
	case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
		return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
		return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
	case VK_FORMAT_R16G16B16A16_SFLOAT:
		return VK_FORMAT_R16G16B16A16_SFLOAT;
	default:
		return VK_FORMAT_UNDEFINED;
	}
}

bool
composition_colour_is_linear_hdr (VkFormat swapchain_format,
                                  uint32_t colour_mode)
{
	if (colour_mode == kColourDisplay)
		return false;
	if (colour_mode == kColourLinearHdr)
		return true;
	// Auto. An 8-bit frame has been tone mapped or there would be nothing to see, and HDR10's
	// ten-bit formats carry PQ, which is display-referred as well. Only a float swapchain is light.
	return swapchain_format == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/** @brief A setting that the environment can override, and what bounds it.
 *
 * An override is a number of hundredths, or of thousandths, of the setting. A setting that is not a
 * finite number, or is negative, becomes the fallback, and one above the maximum the maximum.
 */
struct frame_override {
	char const *variable; //!< The environment variable.
	uint32_t    offset;   //!< The setting's offset in struct composition_frame_settings.
	float       divisor;  //!< What the variable's number is divided by.
	float       fallback; //!< What a negative or non-finite setting becomes.
	float       max;      //!< The largest setting.
};

/** @brief The override of a setting by a variable. */
#define FRAME_OVERRIDE(variable, setting, divisor, fallback, max) \
	{ variable, offsetof(struct composition_frame_settings, setting), divisor, fallback, max }

/** @brief The overrides, in the order upstream read their variables. */
static struct frame_override const FRAME_OVERRIDES[] = {
	FRAME_OVERRIDE("DLSSNR_GHOST_SLACK",   ghost_slack,   100.0f,  0.5f, INFINITY),
	FRAME_OVERRIDE("DLSSNR_RATIO_SMOOTH",  ratio_smooth,  100.0f,  0.0f, 1.0f),
	FRAME_OVERRIDE("DLSSNR_COLOUR_TRUST",  colour_trust,  100.0f,  1.0f, 8.0f),
	FRAME_OVERRIDE("DLSSNR_MOTION_SMOOTH", motion_smooth, 100.0f,  0.0f, 1.0f),
	FRAME_OVERRIDE("DLSSNR_EDIT_BLUR",     edit_blur,     1000.0f, 0.0f, 0.25f)
};

#undef FRAME_OVERRIDE

/** @brief The number of overrides. */
static constexpr size_t FRAME_OVERRIDE_COUNT = sizeof FRAME_OVERRIDES / sizeof *FRAME_OVERRIDES;

/** @brief Runs read_overrides(). */
static pthread_once_t frame_overrides_once = PTHREAD_ONCE_INIT;

/** @brief Each override's variable as a number, at most INT32_MAX, or -1 if it is unset, empty or
 *         negative.
 */
static int32_t frame_overrides[FRAME_OVERRIDE_COUNT];

/** @brief Reads the overrides' variables.
 *
 * A prefix of digits counts, and none is 0, as atoi() read them; strtol() saturates a number atoi()
 * could not hold, which atoi() leaves undefined.
 */
static void
read_overrides (void)
{
	for (size_t i = 0; i < FRAME_OVERRIDE_COUNT; ++i) {
		char const *const v = getenv(FRAME_OVERRIDES[i].variable);
		long const n = v && *v ? strtol(v, nullptr, 10) : -1;
		frame_overrides[i] = n < 0 ? -1 : n > INT32_MAX ? INT32_MAX : (int32_t)n;
	}
}

struct composition_frame_settings
composition_frame_settings (void)
{
	return (struct composition_frame_settings){
		.transfer_strength  = 1.0f,
		.colour_strength    = 1.0f,
		.max_ratio          = 2.0f,
		.debug_scale        = 1.0f,
		.white_point_manual = 1.0f,
		.white_point_scale  = 1.0f,
		.white_point_trim   = 1.0f,
		.white_point_source = kWhitePointManual,
		.compare_split      = 0.5f,
		.compare_zoom       = 1.0f,
		.working_scale      = 1.0f,
		.transfer           = 1,
		.reversible_mode    = kReversibleKnee,
		.apply_model        = 1,
		.ghost_slack        = 0.5f,
		.edit_blur          = 0.04f,
		.motion_smooth      = 1.0f,
		.colour_trust       = 1.0f
	};
}

struct composition_frame_settings
composition_frame_settings_read (struct ShmHeader const *h)
{
	struct composition_frame_settings s = composition_frame_settings();
	if (!h)
		return s;

	s.control_seq = atomic_load(&h->controlSeq);
	s.tuning_seq = atomic_load(&h->tuningSeq);
	s.passes = atomic_load(&h->passes);
	s.transfer_strength = BitsToFloat(atomic_load(&h->transferStrengthBits));
	s.colour_strength = BitsToFloat(atomic_load(&h->colourStrengthBits));
	s.max_ratio = BitsToFloat(atomic_load(&h->maxRatioBits));
	s.debug_scale = BitsToFloat(atomic_load(&h->debugScaleBits));
	s.compare_split = BitsToFloat(atomic_load(&h->compareSplitBits));
	s.compare_zoom = BitsToFloat(atomic_load(&h->compareZoomBits));
	s.working_scale = BitsToFloat(atomic_load(&h->workingScaleBits));
	s.native_model_max_width = atomic_load(&h->nativeModelMaxWidth);
	s.native_model_max_height = atomic_load(&h->nativeModelMaxHeight);
	if (s.native_model_max_width < kMinW || s.native_model_max_width > kMaxW
	    || s.native_model_max_height < kMinH || s.native_model_max_height > kMaxH) {
		s.native_model_max_width = 0;
		s.native_model_max_height = 0;
	}
	s.transfer = atomic_load(&h->transfer);
	s.debug_view = atomic_load(&h->debugView);
	s.compare_mode = atomic_load(&h->compareMode);
	s.compare_swap = atomic_load(&h->compareSwap);
	s.reversible_mode = atomic_load(&h->reversibleMode);
	s.apply_model = atomic_load(&h->applyModel);
	s.hold_frame = atomic_load(&h->holdFrame);
	s.composition_bypass = atomic_load(&h->compositionBypass);
	s.ratio_smooth = (float)atomic_load(&h->ratioSmoothPercent) / 100.0f;
	s.colour_trust = (float)atomic_load(&h->colourTrustPercent) / 100.0f;

	pthread_once(&frame_overrides_once, read_overrides);
	for (size_t i = 0; i < FRAME_OVERRIDE_COUNT; ++i) {
		struct frame_override const *const o = &FRAME_OVERRIDES[i];
		float *const v = (float *)((unsigned char *)&s + o->offset);
		if (frame_overrides[i] >= 0)
			*v = (float)frame_overrides[i] / o->divisor;
		if (!isfinite(*v) || *v < 0.0f)
			*v = o->fallback;
		if (*v > o->max)
			*v = o->max;
	}

	s.white_point_manual = BitsToFloat(atomic_load(&h->whitePointBits));
	s.white_point_scale = BitsToFloat(atomic_load(&h->whitePointScaleBits));
	s.white_point_trim = BitsToFloat(atomic_load(&h->whitePointTrimBits));
	s.white_point_source = atomic_load(&h->whitePointSource);

	// Clamped here rather than trusted, because these come from a file any process can write.
	s.transfer_strength = clamp(s.transfer_strength, 0.0f, 4.0f, 1.0f);
	s.colour_strength = clamp(s.colour_strength, 0.0f, 4.0f, 1.0f);
	s.max_ratio = clamp(s.max_ratio, 1.0f, (float)kMaxPasses, 2.0f);
	s.debug_scale = clamp(s.debug_scale, 0.01f, 100.0f, 1.0f);
	s.white_point_manual = clamp(s.white_point_manual, 1e-4f, 2000.0f, 1.0f);
	s.white_point_scale = clamp(s.white_point_scale, 0.01f, 100.0f, 1.0f);
	s.white_point_trim = clamp(s.white_point_trim, 0.01f, 100.0f, 1.0f);
	if (s.white_point_source > kWhitePointMeasured)
		s.white_point_source = kWhitePointManual;
	s.compare_split = clamp(s.compare_split, 0.0f, 1.0f, 0.5f);
	s.compare_zoom = clamp(s.compare_zoom, 1.0f, 2.0f, 1.0f);

	// Upstream goes up to 2, where the model supersamples; here the model works at most at the
	// frame's raster.
	s.working_scale = clamp(s.working_scale, 0.25f, 1.0f, 1.0f);

	// Native + edit is mode 2; the clamp used to stop at 1 and silently killed it.
	if (s.transfer > 2)
		s.transfer = 2;
	// 4 and 5 are the two views of the colour bound. This clamp is why they did nothing when they
	// were added: the shader grew the cases and the validation did not, so the GUI offered them, the
	// header carried them, and the layer quietly rewrote them to 0 on the way past.
	if (s.debug_view > 5)
		s.debug_view = 0;
	if (s.compare_mode > 2)
		s.compare_mode = 0;
	if (s.reversible_mode >= kReversibleModeCount)
		s.reversible_mode = kReversibleKnee;
	return s;
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/** @brief Whether the device's optimal tiling of a format has features.
 *
 * @param c        The composition.
 * @param format   The format.
 * @param features The features.
 * @return         true if it has all of them.
 */
static bool
format_supports (struct composition const *c,
                 VkFormat                  format,
                 VkFormatFeatureFlags      features)
{
	VkFormatProperties props;
	c->instance->vkGetPhysicalDeviceFormatProperties(c->physical_device, format, &props);
	return (props.optimalTilingFeatures & features) == features;
}

/** @brief The first device-local memory type of some.
 *
 * @param c    The composition.
 * @param bits The memory types to choose from, one bit each.
 * @return     The type's index, or UINT32_MAX if none of them is device-local.
 */
static uint32_t
device_local_type (struct composition const *c,
                   uint32_t                  bits)
{
	VkPhysicalDeviceMemoryProperties mp;
	c->instance->vkGetPhysicalDeviceMemoryProperties(c->physical_device, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
		VkMemoryPropertyFlags const f = mp.memoryTypes[i].propertyFlags;
		if ((bits & (1u << i)) && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
			return i;
	}
	return UINT32_MAX;
}

/** @brief Destroys an image, then leaves it empty.
 *
 * @param c   The composition.
 * @param img The image.
 */
static void
drop_image (struct composition       *c,
            struct composition_image *img)
{
	if (img->view)
		c->vk->vkDestroyImageView(c->device, img->view, nullptr);
	if (img->image)
		c->vk->vkDestroyImage(c->device, img->image, nullptr);
	if (img->memory)
		c->vk->vkFreeMemory(c->device, img->memory, nullptr);
	*img = (struct composition_image){0};
}

/** @brief Makes an image of the composition's, in device-local memory, with a view.
 *
 * Destroys the image's previous objects first, and what the function created if a step fails.
 *
 * @param c      The composition.
 * @param img    The image.
 * @param w      Its width.
 * @param h      Its height.
 * @param format Its format.
 * @param usage  Its usage.
 * @return       true if the image, its memory and its view exist, in UNDEFINED.
 */
static bool
make_image (struct composition       *c,
            struct composition_image *img,
            uint32_t                  w,
            uint32_t                  h,
            VkFormat                  format,
            VkImageUsageFlags         usage)
{
	drop_image(c, img);

	VkImageCreateInfo const image_info = {
		.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType     = VK_IMAGE_TYPE_2D,
		.format        = format,
		.extent        = { w, h, 1 },
		.mipLevels     = 1,
		.arrayLayers   = 1,
		.samples       = VK_SAMPLE_COUNT_1_BIT,
		.tiling        = VK_IMAGE_TILING_OPTIMAL,
		.usage         = usage,
		.sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};
	VkImage image;
	if (c->vk->vkCreateImage(c->device, &image_info, nullptr, &image) != VK_SUCCESS) {
		log_printf("[comp] vkCreateImage %ux%u fmt=%d failed", w, h, (int)format);
		return false;
	}
	img->image = image;

	VkMemoryRequirements req;
	c->vk->vkGetImageMemoryRequirements(c->device, image, &req);
	uint32_t const type = device_local_type(c, req.memoryTypeBits);
	if (type == UINT32_MAX) {
		drop_image(c, img);
		return false;
	}

	VkMemoryAllocateInfo const allocate_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = type
	};
	VkDeviceMemory memory;
	if (c->vk->vkAllocateMemory(c->device, &allocate_info, nullptr, &memory) != VK_SUCCESS) {
		log_printf("[comp] out of device memory for a %ux%u surface", w, h);
		drop_image(c, img);
		return false;
	}
	img->memory = memory;
	if (c->vk->vkBindImageMemory(c->device, image, memory, 0) != VK_SUCCESS) {
		drop_image(c, img);
		return false;
	}

	VkImageViewCreateInfo const view_info = {
		.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image            = image,
		.viewType         = VK_IMAGE_VIEW_TYPE_2D,
		.format           = format,
		.subresourceRange = COMPOSITION_COLOR_RANGE
	};
	VkImageView view;
	if (c->vk->vkCreateImageView(c->device, &view_info, nullptr, &view) != VK_SUCCESS) {
		drop_image(c, img);
		return false;
	}
	img->view = view;
	img->format = format;
	img->width = w;
	img->height = h;
	img->layout = VK_IMAGE_LAYOUT_UNDEFINED;
	return true;
}

/** @brief Unmaps and destroys a buffer, then leaves it empty.
 *
 * @param c   The composition.
 * @param buf The buffer.
 */
static void
drop_host_buffer (struct composition             *c,
                  struct composition_host_buffer *buf)
{
	if (buf->mapped)
		c->vk->vkUnmapMemory(c->device, buf->memory);
	if (buf->buffer)
		c->vk->vkDestroyBuffer(c->device, buf->buffer, nullptr);
	if (buf->memory)
		c->vk->vkFreeMemory(c->device, buf->memory, nullptr);
	*buf = (struct composition_host_buffer){0};
}

/** @brief Makes a buffer in host-visible, coherent memory, and maps it.
 *
 * Destroys the buffer's previous objects first, and what the function created if a step fails.
 *
 * @param c     The composition.
 * @param buf   The buffer.
 * @param bytes Its size.
 * @param usage Its usage.
 * @return      true if the buffer exists and is mapped.
 */
static bool
make_host_buffer (struct composition             *c,
                  struct composition_host_buffer *buf,
                  size_t                          bytes,
                  VkBufferUsageFlags              usage)
{
	drop_host_buffer(c, buf);

	VkBufferCreateInfo const buffer_info = {
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = bytes,
		.usage       = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};
	VkBuffer buffer;
	if (c->vk->vkCreateBuffer(c->device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
		return false;
	buf->buffer = buffer;

	VkMemoryRequirements req;
	c->vk->vkGetBufferMemoryRequirements(c->device, buffer, &req);

	VkPhysicalDeviceMemoryProperties mp;
	c->instance->vkGetPhysicalDeviceMemoryProperties(c->physical_device, &mp);

	// Host visible and coherent is the requirement; cached is a large win on the readback and
	// harmless on the upload, so it is preferred rather than demanded. A type that is lazily
	// allocated and not cached scores below the first choice and is never taken.
	VkMemoryPropertyFlags const required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
	                                       | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t best = UINT32_MAX;
	int32_t best_score = -1;
	for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
		VkMemoryPropertyFlags const f = mp.memoryTypes[i].propertyFlags;
		if (!(req.memoryTypeBits & (1u << i)) || (f & required) != required)
			continue;
		int32_t const score = ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 100 : 0)
		                      - ((f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? 50 : 0);
		if (score > best_score) {
			best_score = score;
			best = i;
		}
	}
	if (best == UINT32_MAX) {
		drop_host_buffer(c, buf);
		return false;
	}

	VkMemoryAllocateInfo const allocate_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = best
	};
	VkDeviceMemory memory;
	if (c->vk->vkAllocateMemory(c->device, &allocate_info, nullptr, &memory) != VK_SUCCESS) {
		drop_host_buffer(c, buf);
		return false;
	}
	buf->memory = memory;

	void *mapped;
	if (c->vk->vkBindBufferMemory(c->device, buffer, memory, 0) != VK_SUCCESS
	    || c->vk->vkMapMemory(c->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
		drop_host_buffer(c, buf);
		return false;
	}
	buf->mapped = mapped;
	buf->size = bytes;
	return true;
}

/** @brief Makes a buffer of the native transport: device-local memory exported as an opaque fd the
 *         daemon imports, so neither side copies the frame through host memory.
 *
 * Both buffers are made as ShmTransportOffer states, for a Vulkan importer to repeat. Destroys the
 * buffer's previous objects first, and what the function created if a step fails.
 *
 * @param c     The composition.
 * @param buf   The buffer.
 * @param bytes Its size.
 * @return      true if the buffer and its dedicated, exportable memory exist.
 */
static bool
make_export_buffer (struct composition             *c,
                    struct composition_host_buffer *buf,
                    size_t                          bytes)
{
	drop_host_buffer(c, buf);

	VkExternalMemoryBufferCreateInfo const external_info = {
		.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
	};
	VkBufferCreateInfo const buffer_info = {
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext       = &external_info,
		.size        = bytes,
		.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};
	VkBuffer buffer;
	if (c->vk->vkCreateBuffer(c->device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
		return false;
	buf->buffer = buffer;

	VkMemoryRequirements req;
	c->vk->vkGetBufferMemoryRequirements(c->device, buffer, &req);
	uint32_t const type = device_local_type(c, req.memoryTypeBits);
	if (type == UINT32_MAX) {
		drop_host_buffer(c, buf);
		return false;
	}

	VkMemoryDedicatedAllocateInfo const dedicated_info = {
		.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.buffer = buffer
	};
	VkExportMemoryAllocateInfo const export_info = {
		.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
		.pNext       = &dedicated_info,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
	};
	VkMemoryAllocateInfo const allocate_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext           = &export_info,
		.allocationSize  = req.size,
		.memoryTypeIndex = type
	};
	VkDeviceMemory memory;
	if (c->vk->vkAllocateMemory(c->device, &allocate_info, nullptr, &memory) != VK_SUCCESS) {
		drop_host_buffer(c, buf);
		return false;
	}
	buf->memory = memory;
	if (c->vk->vkBindBufferMemory(c->device, buffer, memory, 0) != VK_SUCCESS) {
		drop_host_buffer(c, buf);
		return false;
	}
	buf->size = bytes;
	buf->allocation = req.size;
	return true;
}

/** @brief Records the move of an exported buffer between the queue family and the daemon.
 *
 * Exported memory changes hands with the worker at every leg: acquired before a copy touches it and
 * released after. The worker's own accesses are ordered by the fences and futexes around them. A
 * buffer that is not exported records nothing.
 *
 * @param c      The composition.
 * @param cb     The command buffer to record into.
 * @param buf    The buffer.
 * @param from   The family that releases it.
 * @param to     The family that acquires it.
 * @param access The copy's access.
 */
static void
external_ownership (struct composition const             *c,
                    VkCommandBuffer                       cb,
                    struct composition_host_buffer const *buf,
                    uint32_t                              from,
                    uint32_t                              to,
                    VkAccessFlags                         access)
{
	if (!buf->allocation)
		return;

	VkBufferMemoryBarrier const barrier = {
		.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask       = access,
		.dstAccessMask       = access,
		.srcQueueFamilyIndex = from,
		.dstQueueFamilyIndex = to,
		.buffer              = buf->buffer,
		.size                = VK_WHOLE_SIZE
	};
	c->vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
	                            0, nullptr, 1, &barrier, 0, nullptr);
}

/** @brief Builds the transport pair at the model size: exported device-local memory while the
 *         export is on, private host-visible staging otherwise.
 *
 * Called from composition_prepare(), which knows the model size, and from
 * composition_disable_export(), after each has dropped the old pair. Does nothing without a frame,
 * and makes no pair for the in-layer network.
 *
 * @param c The composition.
 * @return  true if both buffers exist, or there is no frame or no pair to build them for.
 */
static bool
ensure_transport (struct composition *c)
{
	if (!c->frame.image)
		return true;

	c->flags &= ~COMPOSITION_TRANSPORT_READY;
	composition_withdraw_offer(c); // New buffers: any offer was of the old ones.
	// The in-layer network takes the proxy and gives the answer on the device: no pair.
	if (c->flags & COMPOSITION_NETWORK)
		return true;

	size_t const bytes = composition_model_bytes(c);
	if ((c->flags & COMPOSITION_EXPORT) && make_export_buffer(c, &c->download, bytes)
	    && make_export_buffer(c, &c->upload, bytes))
		return true;

	// make_host_buffer() drops an exported buffer that a failed pair left behind.
	// Both ways: the in-layer network reads the proxy and writes the answer between the copies.
	VkBufferUsageFlags const transfer = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	return make_host_buffer(c, &c->download, bytes, transfer)
	       && make_host_buffer(c, &c->upload, bytes, transfer);
}

// ---------------------------------------------------------------------------
// The white-point meter, on the GPU
// ---------------------------------------------------------------------------

/** @brief Destroys the reduce pass's pipeline, layouts, pool and sampler, then forgets them. */
static void
drop_meter_objects (struct composition *c)
{
	struct device_table const *const vk = c->vk;
	if (c->meter_pipeline && vk->vkDestroyPipeline)
		vk->vkDestroyPipeline(c->device, c->meter_pipeline, nullptr);
	if (c->meter_pipeline_layout)
		vk->vkDestroyPipelineLayout(c->device, c->meter_pipeline_layout, nullptr);
	if (c->meter_descriptor_layout)
		vk->vkDestroyDescriptorSetLayout(c->device, c->meter_descriptor_layout, nullptr);
	if (c->meter_descriptor_pool)
		vk->vkDestroyDescriptorPool(c->device, c->meter_descriptor_pool, nullptr);
	if (c->meter_sampler)
		vk->vkDestroySampler(c->device, c->meter_sampler, nullptr);
	c->meter_pipeline = VK_NULL_HANDLE;
	c->meter_pipeline_layout = VK_NULL_HANDLE;
	c->meter_descriptor_layout = VK_NULL_HANDLE;
	c->meter_descriptor_pool = VK_NULL_HANDLE;
	c->meter_descriptor_set = VK_NULL_HANDLE;
	c->meter_sampler = VK_NULL_HANDLE;
}

/** @brief Builds the reduce pass's objects around its shader module, the pipeline last.
 *
 * @param c      The composition, which has none of them.
 * @param module The reduce shader.
 * @return       true if all of them exist; otherwise the composition holds those that do.
 */
static bool
build_meter_objects (struct composition *c,
                     VkShaderModule      module)
{
	struct device_table const *const vk = c->vk;
	VkDescriptorSetLayoutCreateInfo const layout_info = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof COMPOSITION_METER_BINDINGS / sizeof *COMPOSITION_METER_BINDINGS,
		.pBindings    = COMPOSITION_METER_BINDINGS
	};
	VkDescriptorSetLayout set_layout;
	if (vk->vkCreateDescriptorSetLayout(c->device, &layout_info, nullptr, &set_layout) != VK_SUCCESS)
		return false;
	c->meter_descriptor_layout = set_layout;

	VkPushConstantRange const push_range = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.size       = sizeof (struct meter_push)
	};
	VkPipelineLayoutCreateInfo const pipeline_layout_info = {
		.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount         = 1,
		.pSetLayouts            = &c->meter_descriptor_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &push_range
	};
	VkPipelineLayout pipeline_layout;
	if (vk->vkCreatePipelineLayout(c->device, &pipeline_layout_info, nullptr, &pipeline_layout)
	    != VK_SUCCESS)
		return false;
	c->meter_pipeline_layout = pipeline_layout;

	VkSamplerCreateInfo const sampler_info = {
		.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter    = VK_FILTER_NEAREST,
		.minFilter    = VK_FILTER_NEAREST,
		.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
	};
	VkSampler sampler;
	if (vk->vkCreateSampler(c->device, &sampler_info, nullptr, &sampler) != VK_SUCCESS)
		return false;
	c->meter_sampler = sampler;

	VkDescriptorPoolCreateInfo const pool_info = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets       = 1,
		.poolSizeCount = sizeof COMPOSITION_METER_POOL_SIZES / sizeof *COMPOSITION_METER_POOL_SIZES,
		.pPoolSizes    = COMPOSITION_METER_POOL_SIZES
	};
	VkDescriptorPool pool;
	if (vk->vkCreateDescriptorPool(c->device, &pool_info, nullptr, &pool) != VK_SUCCESS)
		return false;
	c->meter_descriptor_pool = pool;

	VkDescriptorSetAllocateInfo const set_info = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool     = pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &c->meter_descriptor_layout
	};
	VkDescriptorSet set;
	if (vk->vkAllocateDescriptorSets(c->device, &set_info, &set) != VK_SUCCESS)
		return false;
	c->meter_descriptor_set = set;

	VkComputePipelineCreateInfo const pipeline_info = {
		.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage  = {
			.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage  = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = module,
			.pName  = "main"
		},
		.layout = pipeline_layout
	};
	VkPipeline pipeline;
	if (vk->vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline)
	    != VK_SUCCESS || !pipeline)
		return false;
	c->meter_pipeline = pipeline;
	return true;
}

/** @brief Builds the reduce pass, once.
 *
 * The reduce pipeline is size-independent and survives rebuilds; only the state buffer, the mirror
 * and the descriptor's image binding follow the meter image.
 *
 * @param c The composition.
 * @return  true if the pass exists.
 */
static bool
build_meter_pipeline (struct composition *c)
{
	if (c->meter_pipeline)
		return true;
	drop_meter_objects(c); // whatever a failed attempt left behind

	struct device_table const *const vk = c->vk;
	if (!vk->vkCreateShaderModule || !vk->vkCmdPushConstants || !vk->vkCmdFillBuffer)
		return false;

	VkShaderModuleCreateInfo const module_info = {
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = kMeterReduceSpvLen * sizeof (uint32_t),
		.pCode    = kMeterReduceSpv
	};
	VkShaderModule module;
	if (vk->vkCreateShaderModule(c->device, &module_info, nullptr, &module) != VK_SUCCESS)
		return false;

	bool const built = build_meter_objects(c, module);
	vk->vkDestroyShaderModule(c->device, module, nullptr);
	return built;
}

/** @brief Writes the reduce pass's set: the meter's grid and its state.
 *
 * @param c The composition.
 * @return  true if the set, the grid's view and the state exist.
 */
static bool
build_meter_descriptors (struct composition const *c)
{
	if (!c->meter_descriptor_set || !c->meter.view || !c->meter_state)
		return false;

	VkDescriptorImageInfo const grid = { c->meter_sampler, c->meter.view, VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorBufferInfo const state = { c->meter_state, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet const writes[2] = {
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = c->meter_descriptor_set,
			.dstBinding      = 0,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo      = &grid
		},
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = c->meter_descriptor_set,
			.dstBinding      = 1,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = &state
		}
	};
	c->vk->vkUpdateDescriptorSets(c->device, 2, writes, 0, nullptr);
	return true;
}

/** @brief Destroys the meter's state and its mirror, and stops the meter. */
static void
drop_meter_state (struct composition *c)
{
	drop_host_buffer(c, &c->meter_mirror);
	if (c->meter_state)
		c->vk->vkDestroyBuffer(c->device, c->meter_state, nullptr);
	if (c->meter_state_memory)
		c->vk->vkFreeMemory(c->device, c->meter_state_memory, nullptr);
	c->meter_state = VK_NULL_HANDLE;
	c->meter_state_memory = VK_NULL_HANDLE;
}

/** @brief Makes the meter's device-local state (percentile, history, resolved value) plus the
 *         128-byte host mirror the CPU reads after leg 1's fence -- for the frame-hold snapshot and
 *         the status field -- and the reduce pass that runs on them.
 *
 * @param c The composition, whose meter image exists.
 * @return  true if the meter runs on the GPU.
 */
static bool
make_meter_state (struct composition *c)
{
	drop_meter_state(c);
	if (!build_meter_pipeline(c))
		return false;

	VkBufferCreateInfo const buffer_info = {
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = COMPOSITION_METER_STATE_BYTES,
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		               | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // vkCmdFillBuffer clears it
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};
	VkBuffer buffer;
	if (c->vk->vkCreateBuffer(c->device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
		return false;
	c->meter_state = buffer;

	VkMemoryRequirements req;
	c->vk->vkGetBufferMemoryRequirements(c->device, buffer, &req);
	uint32_t const type = device_local_type(c, req.memoryTypeBits);
	if (type == UINT32_MAX) {
		drop_meter_state(c);
		return false;
	}

	VkMemoryAllocateInfo const allocate_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = type
	};
	VkDeviceMemory memory;
	if (c->vk->vkAllocateMemory(c->device, &allocate_info, nullptr, &memory) != VK_SUCCESS) {
		drop_meter_state(c);
		return false;
	}
	c->meter_state_memory = memory;

	if (c->vk->vkBindBufferMemory(c->device, buffer, memory, 0) != VK_SUCCESS
	    || !make_host_buffer(c, &c->meter_mirror, COMPOSITION_METER_STATE_BYTES,
	                         VK_BUFFER_USAGE_TRANSFER_DST_BIT)
	    || !build_meter_descriptors(c)) {
		drop_meter_state(c);
		return false;
	}
	// New memory holds anything, and the mirror is read before a leg 1 copies the state in: zero
	// reads as no reading.
	memset(c->meter_mirror.mapped, 0, COMPOSITION_METER_STATE_BYTES);
	c->flags &= ~COMPOSITION_METER_STATE_CLEARED;
	return true;
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

/** @brief The generation of the last build of a composition's surfaces in the process. */
static _Atomic(uint64_t) generations;

/** @brief Destroys every surface, the transport pair and the meter's state, closes the connection
 *         of an offer, and forgets the frame's shape and the build's generation.
 *
 * The meter's pass and the composition's pass stay.
 *
 * @param c The composition.
 */
static void
drop_all (struct composition *c)
{
	drop_host_buffer(c, &c->capture_buf);
	drop_image(c, &c->frame);
	drop_image(c, &c->proxy);
	drop_image(c, &c->work);
	drop_image(c, &c->model);
	drop_image(c, &c->composed);
	drop_image(c, &c->meter);
	drop_meter_state(c);
	drop_host_buffer(c, &c->download);
	drop_host_buffer(c, &c->upload);
	composition_withdraw_offer(c);
	c->flags &= ~(COMPOSITION_FRAME_CAPTURED | COMPOSITION_CAPTURE_RECORDED);
	c->generation = 0;
	c->width = c->height = c->model_w = c->model_h = 0;
	c->measured_white_point = 0.0f;
}

struct composition
composition (struct device_table const   *vk,
             struct instance_table const *instance,
             VkDevice                     device,
             VkPhysicalDevice             physical_device)
{
	struct composition ret = {
		.capture_metadata = capture_metadata(),
		.vk               = vk,
		.instance         = instance,
		.device           = device,
		.physical_device  = physical_device,
		.held_white_point = 1.0f,
		.offer            = -1
	};
	if (!device_table_complete(vk)) {
		ret.reason = "the device does not expose everything a compute pass needs";
		ret.error = VK_ERROR_INITIALIZATION_FAILED;
		log_printf("[comp] %s", ret.reason);
		return ret;
	}

	ret.error = dlss_nr_pass_init(&ret.pass, vk, instance, device, physical_device);
	if (ret.error != VK_SUCCESS) {
		ret.reason = "the composition pipeline could not be built";
		dlss_nr_pass_fini(&ret.pass);
	}
	return ret;
}

VkResult
composition_init (struct composition          *dest,
                  struct device_table const   *vk,
                  struct instance_table const *instance,
                  VkDevice                     device,
                  VkPhysicalDevice             physical_device)
{
	if (!dest)
		return VK_ERROR_INITIALIZATION_FAILED;
	*dest = composition(vk, instance, device, physical_device);
	return dest->error;
}

void
composition_fini (struct composition *dest)
{
	if (!dest)
		return;

	drop_all(dest);
	drop_meter_objects(dest);
	dlss_nr_pass_fini(&dest->pass);
	capture_writer_fini(&dest->capture);
	*dest = composition_empty();
}

// ---------------------------------------------------------------------------
// The transport pair's offer
// ---------------------------------------------------------------------------

void
composition_withdraw_offer (struct composition *c)
{
	if (!c || c->offer < 0)
		return;
	close(c->offer);
	c->offer = -1;
}

void
composition_await_answer (struct composition *c,
                          int                 connection)
{
	// Without a composition to own the connection, nothing would close it.
	if (!c) {
		if (connection >= 0) {
			close(connection);
			connection = -1;
		}
		return;
	}
	composition_withdraw_offer(c);
	if (connection < 0)
		return;
	c->offer = connection;
}

void
composition_set_transport_ready (struct composition *c,
                                 bool                ready)
{
	if (!c)
		return;
	c->flags = ready ? c->flags | COMPOSITION_TRANSPORT_READY
	                 : c->flags & ~COMPOSITION_TRANSPORT_READY;
	composition_withdraw_offer(c);
}

/** @brief Exports the memory of one buffer of the transport pair as an opaque descriptor.
 *
 * @param c   The composition.
 * @param buf The buffer.
 * @return    A new descriptor, or -1.
 */
static int
export_memory (struct composition const             *c,
               struct composition_host_buffer const *buf)
{
	VkMemoryGetFdInfoKHR const info = {
		.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
		.memory     = buf->memory,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
	};
	// A failed call leaves its output undefined.
	int fd = -1;
	return c->vk->vkGetMemoryFdKHR(c->device, &info, &fd) == VK_SUCCESS ? fd : -1;
}

bool
composition_export_transport (struct composition       *c,
                              int                       fds[2],
                              struct ShmTransportOffer *offer)
{
	fds[0] = -1;
	fds[1] = -1;
	// Opaque fds import only on the device and driver that made them: the offer names both. Only
	// memory made for export has a descriptor to give.
	if (!composition_usable(c) || !composition_transport_exported(c) || !c->vk->vkGetMemoryFdKHR
	    || !c->instance->vkGetPhysicalDeviceProperties2)
		return false;

	VkPhysicalDeviceIDProperties ids = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
	VkPhysicalDeviceProperties2 properties = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &ids
	};
	c->instance->vkGetPhysicalDeviceProperties2(c->physical_device, &properties);
	memcpy(offer->deviceUuid, ids.deviceUUID, sizeof offer->deviceUuid);
	memcpy(offer->driverUuid, ids.driverUUID, sizeof offer->driverUuid);

	int proxy = export_memory(c, &c->download);
	if (proxy < 0)
		return false;
	int answer = export_memory(c, &c->upload);
	if (answer < 0) {
		close(proxy);
		proxy = -1;
		return false;
	}
	offer->allocation[0] = c->download.allocation;
	offer->allocation[1] = c->upload.allocation;
	offer->size[0] = c->download.size;
	offer->size[1] = c->upload.size;

	// A fresh number per offer: a restarted worker's acknowledgement can never be an old one.
	// No random source means no offer; the low bit keeps the number nonzero (zero is the host
	// transport).
	uint32_t generation;
	if (getrandom(&generation, sizeof generation, 0) != (ssize_t)sizeof generation) {
		close(proxy);
		proxy = -1;
		close(answer);
		answer = -1;
		return false;
	}
	c->transport_gen = generation | 1u;
	offer->generation = c->transport_gen;
	fds[0] = proxy;
	fds[1] = answer;
	return true;
}

void
composition_disable_export (struct composition *c)
{
	if (!c)
		return;
	c->flags &= ~COMPOSITION_EXPORT;
	drop_host_buffer(c, &c->download);
	drop_host_buffer(c, &c->upload);
	// Without a pair, no leg may record against the surfaces: the next frame builds them again.
	if (!ensure_transport(c))
		drop_all(c);
}

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------

VkExtent2D
composition_model_extent (uint32_t                                 width,
                          uint32_t                                 height,
                          struct composition_frame_settings const *s)
{
	// The model works at this fraction of the frame.
	//
	// No rounding to a workgroup multiple: every dispatch here covers a partial group and the shader
	// bounds-checks against gWidth/gHeight, so alignment buys nothing -- and rounding *up* was worse
	// than nothing, because at a scale of exactly 1.0 it pushed a 500-pixel frame to 504, a model
	// raster above the frame's on a setting that means "leave it alone". A floor of 64 only stops a
	// pathologically small window from producing a degenerate raster.
	if (s->native_model_max_width && s->native_model_max_height && width && height) {
		// Keep the low-resolution source and answer together. Upscaling the
		// answer inside the worker hid its true resolution from the resolve
		// shader and bypassed the native-detail-preserving transfer branch.
		double const scale = min_d(min_d((double)s->working_scale,
		                                 (double)s->native_model_max_width / width),
		                           (double)s->native_model_max_height / height);
		return (VkExtent2D){
			max_u(kMinW, (uint32_t)lround(width * scale)),
			max_u(kMinH, (uint32_t)lround(height * scale))
		};
	}
	if (s->working_scale == 1.0f)
		return (VkExtent2D){ width, height };

	double const scale = (double)s->working_scale;
	return (VkExtent2D){
		max_u(64, (uint32_t)lround((double)width * scale)),
		max_u(64, (uint32_t)lround((double)height * scale))
	};
}

/** @brief What a frame holds, as composition_prepare() builds for it. */
struct frame_domain {
	uint64_t    flag; //!< Its flag in struct composition.
	char const *log;  //!< What the build's log line calls it.
};

/** @brief The frame's domain, by whether it holds linear light. */
static struct frame_domain const FRAME_DOMAINS[2] = {
	{ 0,                      "display-referred" },
	{ COMPOSITION_LINEAR_HDR, "linear HDR" }
};

/** @brief The surfaces that cross, as composition_prepare() builds them. */
struct proxy_kind {
	char const *log[2]; //!< What the build's log line says of them, by hdr_transfer.
	VkFormat    format; //!< Their format.
	uint32_t    flag;   //!< Their flag in struct composition.
};

/** @brief The surfaces that cross, by whether the proxy is float16. */
static struct proxy_kind const PROXY_KINDS[2] = {
	{
		.log    = { "", "" },
		.format = VK_FORMAT_R8G8B8A8_UNORM
	},
	{
		.log    = { ", float16 proxy", ", float16 proxy (PQ in)" },
		.format = VK_FORMAT_R16G16B16A16_SFLOAT,
		.flag   = COMPOSITION_HDR_PROXY
	}
};

/** @brief How the answer reaches the model image, as composition_prepare() builds for it. */
struct answer_kind {
	char const        *log;   //!< What the build's log line says of it.
	VkImageUsageFlags  usage; //!< The model image's usage.
	uint32_t           flag;  //!< Its flag in struct composition.
};

/** @brief How the answer reaches the model image, by whether the in-layer network gives it: copied
 *         in from the transport pair, or stored or blitted into it by the network.
 */
static struct answer_kind const ANSWER_KINDS[2] = {
	{
		.log   = "",
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		         | VK_IMAGE_USAGE_TRANSFER_DST_BIT
	},
	{
		.log   = ", in-layer network",
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
		         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.flag  = COMPOSITION_NETWORK
	}
};

bool
composition_prepare (struct composition                      *c,
                     uint32_t                                 width,
                     uint32_t                                 height,
                     VkFormat                                 swapchain_format,
                     struct composition_frame_settings const *s,
                     bool                                     linear_hdr,
                     bool                                     hdr_proxy,
                     uint32_t                                 hdr_transfer,
                     bool                                     network)
{
	if (!composition_usable(c))
		return false;

	VkFormat work = composition_format(swapchain_format);
	if (work == VK_FORMAT_UNDEFINED) {
		c->reason = "unsupported swapchain format";
		return false;
	}

	// The composed surface is written as a storage image and then handed back to the swapchain. When
	// the swapchain's own UNORM twin can be written that way, that is what everything internal uses:
	// it shares the swapchain's bit layout, so both ends are a byte-for-byte copy and nothing is
	// reinterpreted.
	//
	// Not every presentable format can be written as a storage image, though. NVIDIA does not expose
	// A2R10G10B10_UNORM_PACK32 that way, and that is exactly what a 10-bit desktop hands most games
	// -- so the pass used to switch itself off, for the whole run, on the machines it was written
	// for. Compose in half float in that case and blit at both ends instead: the blit converts, and
	// sixteen bits a channel hold more than the ten the swapchain can show, so nothing is lost that
	// the display could have displayed.
	//
	// The device's answer does not change, and this runs every frame: the swapchain format of the
	// last build keeps what was decided for it.
	uint64_t blit = 0;
	if (swapchain_format == c->swapchain_format) {
		work = c->work_format;
		blit = c->flags & COMPOSITION_BLIT_SWAPCHAIN;
	} else if (!format_supports(c, work, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
		VkFormat const wide = VK_FORMAT_R16G16B16A16_SFLOAT;
		VkFormatFeatureFlags const blit_both = VK_FORMAT_FEATURE_BLIT_SRC_BIT
		                                       | VK_FORMAT_FEATURE_BLIT_DST_BIT;
		if (!format_supports(c, wide, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
		    || !format_supports(c, wide, blit_both)
		    || !format_supports(c, swapchain_format, blit_both)) {
			c->reason = "this device cannot write the swapchain's format as a storage image";
			c->error = VK_ERROR_FORMAT_NOT_SUPPORTED;
			log_printf("[comp] %s (format %d)", c->reason, (int)work);
			return false;
		}
		log_printf("[comp] format %d cannot be written as a storage image here; composing in half "
		           "float", (int)work);
		work = wide;
		blit = COMPOSITION_BLIT_SWAPCHAIN;
	}

	// The build declares write-only storage images formatless and the device enables
	// shaderStorageImageWriteWithoutFormat, so each declaration matches the bound view's
	// actual format (RGBA8, BGRA8 or float) without changing shader arithmetic.
	VkExtent2D const model = composition_model_extent(width, height, s);

	// The float16 proxy needs a surface the shader can write and sample as float. Where the device
	// says it cannot, the request quietly becomes the 8-bit arrangement that shipped before -- the
	// same shape as every other capability step in this file. A build with the float16 proxy already
	// found it supported.
	VkFormatFeatureFlags const fp16_need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
	                                       | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
	                                       | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
	                                       | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
	if (hdr_proxy && !(c->flags & COMPOSITION_HDR_PROXY)
	    && !format_supports(c, VK_FORMAT_R16G16B16A16_SFLOAT, fp16_need)) {
		log_printf("[comp] float16 proxy requested but not supported here; staying 8-bit");
		hdr_proxy = false;
	}

	struct frame_domain const *const domain = &FRAME_DOMAINS[linear_hdr];
	struct proxy_kind const *const proxy = &PROXY_KINDS[hdr_proxy];
	struct answer_kind const *const answer = &ANSWER_KINDS[network];
	uint64_t const shape = domain->flag | proxy->flag | answer->flag;
	if (c->width == width && c->height == height && c->swapchain_format == swapchain_format
	    && c->model_w == model.width && c->model_h == model.height
	    && (c->flags & (COMPOSITION_LINEAR_HDR | COMPOSITION_HDR_PROXY | COMPOSITION_NETWORK)) == shape
	    && c->hdr_transfer == hdr_transfer && c->frame.image)
		return true;

	log_printf("[comp] building %ux%u, model %ux%u, %s%s%s", width, height, model.width, model.height,
	           domain->log, proxy->log[hdr_transfer != 0], answer->log);

	// Keep the captured frame across a rebuild that does not change its shape.
	//
	// Almost everything here is rebuilt because the *model's* raster changed -- passes, model
	// resolution -- while the captured frame stays the swapchain's size in the swapchain's working
	// format. Dropping it anyway costs nothing while the frame is not held, because the next frame
	// captures another one. A held frame would be lost: the capture unfreezes the moment
	// COMPOSITION_FRAME_CAPTURED goes, so the next composition would read the swapchain image
	// instead of the held picture.
	struct composition_image kept = {0};
	if ((c->flags & COMPOSITION_FRAME_CAPTURED) && c->frame.image && c->width == width
	    && c->height == height && c->work_format == work) {
		kept = c->frame;
		c->frame = (struct composition_image){0}; // detached, so drop_all() leaves it alone
	}

	drop_all(c);

	VkImageUsageFlags const sampled = VK_IMAGE_USAGE_SAMPLED_BIT;
	VkImageUsageFlags const storage = VK_IMAGE_USAGE_STORAGE_BIT;
	VkImageUsageFlags const src = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	VkImageUsageFlags const dst = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	if (kept.image) {
		c->frame = kept;
		c->flags |= COMPOSITION_FRAME_CAPTURED;
	}
	bool const made_frame = kept.image
	                        || make_image(c, &c->frame, width, height, work, sampled | src | dst);

	c->swapchain_format = swapchain_format;
	c->work_format = work;
	c->flags = (c->flags & ~(COMPOSITION_BLIT_SWAPCHAIN | COMPOSITION_LINEAR_HDR
	                         | COMPOSITION_HDR_PROXY | COMPOSITION_NETWORK))
	           | blit | shape;
	c->hdr_transfer = hdr_transfer;

	// A display-referred UNORM frame is its own proxy: the encode would only copy it, so there is no
	// proxy and the downsample reads the frame, converting it into the model's format even at equal
	// size. A float frame still goes through the encode, which clamps its negative channels.
	bool const direct = !linear_hdr && work != VK_FORMAT_R16G16B16A16_SFLOAT;

	bool const ok = made_frame
	                && (direct
	                    || make_image(c, &c->proxy, width, height, proxy->format,
	                                  sampled | storage | src))
	                && make_image(c, &c->model, model.width, model.height, proxy->format, answer->usage)
	                && make_image(c, &c->composed, width, height, work, storage | src);

	// The transport pair is sized to the model raster: exported device-local memory while the
	// export is on, staging otherwise.
	c->model_w = model.width;
	c->model_h = model.height;
	bool const ok_transport = ensure_transport(c);

	// The meter is a fixed 64x64 grid whatever the frame is, and is only built when there is
	// something to measure: on a frame the game already tone mapped there is no white point to find,
	// so the dispatch and its reduction are skipped entirely rather than run and ignored. The
	// reduction -- percentile, gates, history, resolve -- runs on the GPU; only a 128-byte mirror
	// of its answer ever reaches the CPU.
	bool const ok_meter = !linear_hdr
	                      || (make_image(c, &c->meter, DLSS_NR_METER_GRID, DLSS_NR_METER_GRID,
	                                     VK_FORMAT_R32_SFLOAT, storage | sampled)
	                          && make_meter_state(c));

	bool const need_work = direct || model.width != width || model.height != height;
	bool const ok_work = !need_work
	                     || make_image(c, &c->work, model.width, model.height, proxy->format,
	                                   sampled | storage | src);

	if (!ok || !ok_transport || !ok_work || !ok_meter) {
		c->reason = "could not allocate the composition surfaces";
		drop_all(c);
		return false;
	}

	c->width = width;
	c->height = height;
	c->model_w = model.width;
	c->model_h = model.height;
	c->generation = atomic_fetch_add_explicit(&generations, 1, memory_order_relaxed) + 1;
	c->reason = nullptr;
	return true;
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/** @brief Records a layout transition of one of the composition's images, unless it is already in
 *         the layout, with the pass's conservative barrier.
 *
 * @param c   The composition.
 * @param cb  The command buffer to record into.
 * @param img The image.
 * @param to  The layout it moves to.
 */
static void
transition (struct composition       *c,
            VkCommandBuffer           cb,
            struct composition_image *img,
            VkImageLayout             to)
{
	if (img->layout == to)
		return;
	shader_vk_set_image_layout(&c->pass.shader, cb, img->image, img->layout, to, COMPOSITION_COLOR_RANGE);
	img->layout = to;
}

/** @brief The access that a layout of the swapchain image carries.
 *
 * A transfer layout carries its transfer access. PRESENT_SRC needs only an execution dependency.
 *
 * @param layout The layout.
 * @return       Its access.
 */
static VkAccessFlags
swapchain_access (VkImageLayout layout)
{
	if (layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
		return VK_ACCESS_TRANSFER_READ_BIT;
	if (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		return VK_ACCESS_TRANSFER_WRITE_BIT;
	return 0;
}

/** @brief The stage of a swapchain access: the transfer, or, for none, BOTTOM_OF_PIPE, which is
 *         every prior command as a source and nothing as a destination.
 *
 * @param access The access.
 * @return       Its stage.
 */
static VkPipelineStageFlags
swapchain_stage (VkAccessFlags access)
{
	return access ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
}

/** @brief Records a layout transition of the swapchain image.
 *
 * @param c     The composition.
 * @param cb    The command buffer to record into.
 * @param image The swapchain image.
 * @param from  Its layout.
 * @param to    The layout it moves to.
 */
static void
transition_swapchain (struct composition const *c,
                      VkCommandBuffer           cb,
                      VkImage                   image,
                      VkImageLayout             from,
                      VkImageLayout             to)
{
	VkImageMemoryBarrier const barrier = {
		.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask       = swapchain_access(from),
		.dstAccessMask       = swapchain_access(to),
		.oldLayout           = from,
		.newLayout           = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = image,
		.subresourceRange    = COMPOSITION_COLOR_RANGE
	};
	c->vk->vkCmdPipelineBarrier(cb, swapchain_stage(barrier.srcAccessMask),
	                            swapchain_stage(barrier.dstAccessMask), 0, 0, nullptr, 0, nullptr, 1,
	                            &barrier);
}

/** @brief Records one step in or out of the swapchain.
 *
 * A copy when the two formats share a bit layout, which is the usual case and moves the bytes
 * untouched; a blit when the working format had to differ, which converts between them. Same extent
 * either way -- this never resamples.
 *
 * @param c          The composition.
 * @param cb         The command buffer to record into.
 * @param src        The source image.
 * @param src_layout Its layout.
 * @param dst        The destination image.
 * @param dst_layout Its layout.
 * @param w          The images' width.
 * @param h          Their height.
 */
static void
copy_whole_image (struct composition const *c,
                  VkCommandBuffer           cb,
                  VkImage                   src,
                  VkImageLayout             src_layout,
                  VkImage                   dst,
                  VkImageLayout             dst_layout,
                  uint32_t                  w,
                  uint32_t                  h)
{
	if (c->flags & COMPOSITION_BLIT_SWAPCHAIN) {
		VkImageBlit const blit = {
			.srcSubresource = COMPOSITION_COLOR_LAYERS,
			.srcOffsets     = { { 0, 0, 0 }, { (int32_t)w, (int32_t)h, 1 } },
			.dstSubresource = COMPOSITION_COLOR_LAYERS,
			.dstOffsets     = { { 0, 0, 0 }, { (int32_t)w, (int32_t)h, 1 } }
		};
		c->vk->vkCmdBlitImage(cb, src, src_layout, dst, dst_layout, 1, &blit, VK_FILTER_NEAREST);
		return;
	}

	VkImageCopy const copy = {
		.srcSubresource = COMPOSITION_COLOR_LAYERS,
		.dstSubresource = COMPOSITION_COLOR_LAYERS,
		.extent         = { w, h, 1 }
	};
	c->vk->vkCmdCopyImage(cb, src, src_layout, dst, dst_layout, 1, &copy);
}

/** @brief Where the white point comes from, in one place.
 *
 * The measured reading is only taken when there is one: the meter needs a lit scene and a linear
 * frame to say anything, and until it has spoken the slider is the answer rather than zero. The scale
 * applies to whichever was chosen, because it is the user saying what the model should treat as white
 * rather than a property of the measurement.
 *
 * @param c The composition.
 * @param s The frame's settings.
 * @return  The white point, from 1e-4 to 2000.
 */
static float
resolved_white_point (struct composition const                *c,
                      struct composition_frame_settings const *s)
{
	float base = s->white_point_manual;
	if (s->white_point_source == kWhitePointMeasured && c->measured_white_point > 0.0f)
		base = c->measured_white_point * s->white_point_trim;
	return bound(base * s->white_point_scale, 1e-4f, 2000.0f);
}

/** @brief The constants every dispatch of a frame shares.
 *
 * @param c The composition.
 * @param s The frame's settings.
 * @return  The constants, whose mode, width and height the dispatch sets.
 */
static struct dlss_nr_constants
base_constants (struct composition const                *c,
                struct composition_frame_settings const *s)
{
	// The empty initializer zeroes the block's padding too, all of which the pass copies.
	struct dlss_nr_constants k = {};

	k.white_point = resolved_white_point(c, s);
	k.transfer_strength = s->transfer_strength;
	k.colour_strength = s->colour_strength;
	k.max_ratio = s->max_ratio;
	k.debug_view = s->debug_view;
	k.debug_scale = s->debug_scale;
	k.transfer = s->transfer;
	k.compare_mode = s->compare_mode;
	k.compare_split = s->compare_split;
	k.compare_zoom = s->compare_zoom;
	k.compare_swap = s->compare_swap;
	k.reversible_mode = s->reversible_mode;
	k.apply_model = s->apply_model;

	// The model's answer IS the frame. The raw-answer debug path returns the model's picture ahead of
	// every step of the composition -- no ratio, no guard, no blend, no compare -- and it returns
	// before the normalisation step, so the scale that step would have applied has to come from here.
	//
	// That early return is also why compare did nothing under a bypass: the overlay lives at the tail
	// of the resolve, past every return, so the raw path showed one picture with a divider across it
	// and nothing to compare. The replace modes are the only route that presents the model's answer
	// without returning early -- pure inverse of the encode, none of the composition -- so comparing
	// under a bypass borrows them: soft knee and Neutwo pair with NeutwoDecode, Hybrid with
	// HybridDecode. The encode reads these same constants, so the pair stays an exact inverse. The
	// price is that the proxy shown to the model while comparing is the replace mode's own rather
	// than the user's, which is acceptable for a diagnostic view and why this is not done when the
	// composition is on, where the overlay already runs on the composed picture.
	if (s->composition_bypass) {
		if (s->compare_mode != 0) {
			k.reversible_mode = s->reversible_mode >= 3 ? 4u : 2u;
		} else {
			k.debug_view = 2;
			k.debug_scale = 1.0f; // resolve applies the white-point/transfer scale once
		}
	}

	// A frame the game already tone mapped goes through the encode untouched, and the composition
	// works in its units rather than normalising by a white point that means nothing here.
	k.passthrough = (c->flags & COMPOSITION_LINEAR_HDR) ? 0u : 1u;

	// No motion vectors and no exposure reach a present-time layer. The guides are declared at the
	// frame's own size so nothing downstream scales by a ratio that does not exist.
	k.mv_scale_x = 1.0f;
	k.mv_scale_y = 1.0f;
	k.guide_width = c->width;
	k.guide_height = c->height;
	k.use_game_exposure = 0;
	k.exposure_pre_mul = 1.0f;
	// Surface precision and model color domain are independent for native HIP.
	k.hdr_proxy = (c->flags & COMPOSITION_HDR_PROXY) ? 2u : 0u;
	k.hdr_transfer = c->hdr_transfer;
	k.colour_trust = s->colour_trust;
	k.ratio_smooth = s->ratio_smooth;
	return k;
}

/** @brief Records the meter's reduce after its grid: the clear of the state the first time, the
 *         reduce, and the copy of the state into the mirror.
 *
 * @param c  The composition, whose meter runs on the GPU.
 * @param cb The command buffer to record into.
 * @param s  The frame's settings.
 */
static void
record_meter_reduce (struct composition                      *c,
                     VkCommandBuffer                          cb,
                     struct composition_frame_settings const *s)
{
	struct device_table const *const vk = c->vk;
	if (!(c->flags & COMPOSITION_METER_STATE_CLEARED)) {
		vk->vkCmdFillBuffer(cb, c->meter_state, 0, VK_WHOLE_SIZE, 0);
		// The reduce reads and writes what the fill cleared.
		VkMemoryBarrier const cleared = {
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
		};
		vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &cleared, 0, nullptr, 0,
		                         nullptr);
		c->flags |= COMPOSITION_METER_STATE_CLEARED;
	}

	struct meter_push push = {
		.manual = s->white_point_manual,
		.scale  = s->white_point_scale,
		.trim   = s->white_point_trim,
		.source = s->white_point_source
	};
	// Leg 1 has copied the frame by now, so a held frame is a captured one.
	if (c->flags & COMPOSITION_HOLDING) {
		push.hold_value = resolved_white_point(c, s);
		push.hold = 1;
	}
	vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->meter_pipeline);
	vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->meter_pipeline_layout, 0, 1,
	                            &c->meter_descriptor_set, 0, nullptr);
	vk->vkCmdPushConstants(cb, c->meter_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push,
	                       &push);
	vk->vkCmdDispatch(cb, 1, 1, 1);

	// The mirror must show this frame's answer, so the transfer read waits on the reduce's write.
	// The state buffer stays in its default layout; buffers need no transition.
	VkBufferMemoryBarrier const to_mirror = {
		.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer              = c->meter_state,
		.offset              = 0,
		.size                = VK_WHOLE_SIZE
	};
	vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
	                         0, nullptr, 1, &to_mirror, 0, nullptr);
	VkBufferCopy const mirror = { 0, 0, COMPOSITION_METER_STATE_BYTES };
	vk->vkCmdCopyBuffer(cb, c->meter_state, c->meter_mirror.buffer, 1, &mirror);
}

/** @brief Notes a change of the frame hold, on the edge rather than the level, so the white point is
 *         snapshotted once at the moment it comes on rather than re-read every frame it stays on.
 *
 * The snapshot comes from the meter's mirror -- the resolved value the GPU settled on for the frame
 * just captured -- so a hold freezes the same number the resolve has been using, not a host-side
 * approximation.
 *
 * @param c The composition.
 * @param s The frame's settings.
 */
static void
note_hold (struct composition                      *c,
           struct composition_frame_settings const *s)
{
	bool const holding = c->flags & COMPOSITION_HOLDING;
	if (!s->hold_frame == !holding)
		return;

	if (holding) {
		c->flags &= ~COMPOSITION_HOLDING;
		log_printf("[comp] frame released");
		return;
	}
	c->flags |= COMPOSITION_HOLDING;
	float const *const mirror = c->meter_mirror.mapped;
	c->held_white_point = mirror && mirror[2] > 0.0f ? mirror[2] : resolved_white_point(c, s);
	log_printf("[comp] frame held (white point %.3f)", (double)c->held_white_point);
}

/** @brief Records what the model is handed when it is not the proxy: the downsample.
 *
 * @param c      The composition, whose work image exists.
 * @param cb     The command buffer to record into.
 * @param base   The frame's constants (base_constants()).
 * @param source The frame or the proxy.
 * @return       true if the dispatch was recorded.
 */
static bool
record_work (struct composition             *c,
             VkCommandBuffer                 cb,
             struct dlss_nr_constants const *base,
             struct composition_image       *source)
{
	transition(c, cb, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	transition(c, cb, &c->work, VK_IMAGE_LAYOUT_GENERAL);

	// Reduce, with the module's own area filter -- the model then works on fewer pixels and less
	// crosses the shared memory.
	struct dlss_nr_constants down = *base;
	down.mode = DLSS_NR_MODE_DOWNSAMPLE;
	down.width = c->model_w;
	down.height = c->model_h;
	return dlss_nr_pass_dispatch(&c->pass, cb, &down, c->model_w, c->model_h, source->view,
	                             VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, c->work.view,
	                             VK_NULL_HANDLE, COMPOSITION_READ_ONLY, COMPOSITION_READ_ONLY);
}

/** @brief Records the barrier that hands the proxy and the model image to the in-layer network: the
 *         proxy from leg 1's writes into the network's reads, in SHADER_READ_ONLY_OPTIMAL, and the
 *         model image from the last frame's reads into the network's writes, in GENERAL.
 *
 * The network writes the model image with compute shaders or with a blit. shader_vk_set_image_layout()
 * would give a transition into GENERAL to compute writes only, so this barrier names both writers.
 *
 * @param c      The composition, built for the in-layer network.
 * @param cb     The command buffer to record into.
 * @param source The proxy, which leg 1 wrote with compute shaders.
 */
static void
hand_to_network (struct composition       *c,
                 VkCommandBuffer           cb,
                 struct composition_image *source)
{
	VkPipelineStageFlags const stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
	                                    | VK_PIPELINE_STAGE_TRANSFER_BIT;
	VkImageMemoryBarrier const barriers[2] = {
		{
			.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout           = source->layout,
			.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image               = source->image,
			.subresourceRange    = COMPOSITION_COLOR_RANGE
		},
		{
			.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout           = c->model.layout,
			.newLayout           = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image               = c->model.image,
			.subresourceRange    = COMPOSITION_COLOR_RANGE
		}
	};
	c->vk->vkCmdPipelineBarrier(cb, stages, stages, 0, 0, nullptr, 0, nullptr, 2, barriers);
	source->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	c->model.layout = VK_IMAGE_LAYOUT_GENERAL;
}

// ---------------------------------------------------------------------------
// Leg 1: the frame the model is shown
// ---------------------------------------------------------------------------

bool
composition_record_capture (struct composition                      *c,
                            VkCommandBuffer                          cb,
                            VkImage                                  swapchain_image,
                            struct composition_frame_settings const *s)
{
	if (!composition_usable(c) || !c->frame.image)
		return false;

	note_hold(c, s);

	// While held the frame is not re-read, but everything downstream of it still runs: the encode
	// re-encodes, the model re-evaluates and the resolve re-composes, so a setting changed now is
	// answered on the same picture. Freezing the proxy instead would be wrong -- settings must still
	// re-encode -- and freezing it would also desynchronise it from the frame the resolve reads.
	uint64_t const held = COMPOSITION_HOLDING | COMPOSITION_FRAME_CAPTURED;
	bool const freeze = (c->flags & held) == held;
	if (!freeze) {
		transition_swapchain(c, cb, swapchain_image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		transition(c, cb, &c->frame, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copy_whole_image(c, cb, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c->frame.image,
		                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c->width, c->height);

		// Straight back, so that every path out of this leg -- including the ones that give up --
		// leaves the image in the layout the present engine requires.
		transition_swapchain(c, cb, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
		c->flags |= COMPOSITION_FRAME_CAPTURED;
	}

	transition(c, cb, &c->frame, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	// The constants every dispatch of this leg shares; each sets its mode and size.
	struct dlss_nr_constants const base = base_constants(c, s);
	struct composition_image *source = &c->frame;
	if (c->proxy.image) {
		transition(c, cb, &c->proxy, VK_IMAGE_LAYOUT_GENERAL);
		struct dlss_nr_constants enc = base;
		enc.mode = DLSS_NR_MODE_ENCODE;
		enc.width = c->width;
		enc.height = c->height;
		if (!dlss_nr_pass_dispatch(&c->pass, cb, &enc, c->width, c->height, c->frame.view,
		                           VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, c->proxy.view,
		                           VK_NULL_HANDLE, COMPOSITION_READ_ONLY, COMPOSITION_READ_ONLY))
			return false;
		source = &c->proxy;
	}

	// The meter, measured off the captured frame and never off anything this pass writes. That
	// distinction is the whole reason it is safe: an earlier white point meter upstream read its own
	// output and chased it, walking one session from 0.010 to 97.910. There is no path from what this
	// pass writes back into what this reads.
	//
	// The grid lands in a device-local state buffer and the reduce pass turns it into the resolved
	// white point without the bytes ever crossing to the host; the resolve reads the answer through
	// a four-byte copy into its own constant block (composition_record_compose()). The 128-byte
	// mirror the transfer copies at the end is for the CPU's frame-hold snapshot and status field
	// only.
	if (c->meter_state) {
		transition(c, cb, &c->meter, VK_IMAGE_LAYOUT_GENERAL);

		struct dlss_nr_constants meter = base;
		meter.mode = DLSS_NR_MODE_CALIBRATE;
		meter.width = DLSS_NR_METER_GRID;
		meter.height = DLSS_NR_METER_GRID;
		if (dlss_nr_pass_dispatch(&c->pass, cb, &meter, DLSS_NR_METER_GRID, DLSS_NR_METER_GRID,
		                          c->frame.view, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
		                          c->meter.view, VK_NULL_HANDLE, COMPOSITION_READ_ONLY,
		                          COMPOSITION_READ_ONLY))
			record_meter_reduce(c, cb, s);
	}

	// What the model is actually handed: the full-resolution proxy, or a reduction of it.
	if (c->work.image) {
		if (!record_work(c, cb, &base, source))
			return false;
		source = &c->work;
	}

	if (c->flags & COMPOSITION_NETWORK) {
		hand_to_network(c, cb, source);
		return true;
	}

	transition(c, cb, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

	VkBufferImageCopy const region = {
		.imageSubresource = COMPOSITION_COLOR_LAYERS,
		.imageExtent      = { c->model_w, c->model_h, 1 }
	};
	external_ownership(c, cb, &c->download, VK_QUEUE_FAMILY_EXTERNAL, c->export_family,
	                   VK_ACCESS_TRANSFER_WRITE_BIT);
	c->vk->vkCmdCopyImageToBuffer(cb, source->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                              c->download.buffer, 1, &region);
	external_ownership(c, cb, &c->download, c->export_family, VK_QUEUE_FAMILY_EXTERNAL,
	                   VK_ACCESS_TRANSFER_WRITE_BIT);
	return true;
}

// ---------------------------------------------------------------------------
// Leg 2: the model's answer, composed back
// ---------------------------------------------------------------------------

/** @brief The size of one image of a capture pair, in the working format.
 *
 * @param c The composition.
 * @return  The size in bytes.
 */
static size_t
pair_bytes (struct composition const *c)
{
	return (size_t)c->width * c->height * (c->work_format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4);
}

/** @brief Records the readback of a capture pair, and what the manifest says about it.
 *
 * The pair is taken in leg 2 because this is the one place that holds both the frame as the game
 * presented it and the frame the model edited, for the same frame. A capture whose host buffer cannot
 * be made records none.
 *
 * @param c  The composition.
 * @param cb The command buffer to record into.
 * @param s  The frame's settings.
 */
static void
record_capture_pair (struct composition                      *c,
                     VkCommandBuffer                          cb,
                     struct composition_frame_settings const *s)
{
	struct capture_metadata *const m = &c->capture_metadata;
	m->frame_control_seq = s->control_seq;
	m->tuning_seq = s->tuning_seq;
	m->passes = s->passes;
	m->debug_view = s->debug_view;
	m->apply_model = s->apply_model;
	m->bypass = s->composition_bypass;
	m->hold = s->hold_frame;
	m->compare = s->compare_mode;
	m->transfer = s->transfer;
	m->detail = s->transfer_strength;
	m->color = s->colour_strength;
	m->debug_scale = s->debug_scale;
	m->model_width = c->model_w;
	m->model_height = c->model_h;
	m->hdr_proxy = (c->flags & COMPOSITION_HDR_PROXY) != 0;
	m->linear_hdr = (c->flags & COMPOSITION_LINEAR_HDR) != 0;
	m->hdr_transfer = c->hdr_transfer;

	size_t const bytes = pair_bytes(c);
	if (!c->capture_buf.buffer
	    && !make_host_buffer(c, &c->capture_buf, bytes * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT))
		return;

	transition(c, cb, &c->frame, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	transition(c, cb, &c->composed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	VkBufferImageCopy region = {
		.imageSubresource = COMPOSITION_COLOR_LAYERS,
		.imageExtent      = { c->width, c->height, 1 }
	};
	c->vk->vkCmdCopyImageToBuffer(cb, c->frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                              c->capture_buf.buffer, 1, &region);
	region.bufferOffset = bytes;
	c->vk->vkCmdCopyImageToBuffer(cb, c->composed.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                              c->capture_buf.buffer, 1, &region);
	c->flags |= COMPOSITION_CAPTURE_RECORDED;
}

/** @brief Records the copy of the model's raw answer into the swapchain image, for a bypass whose
 *         answer is the frame's own size and format.
 *
 * @param c               The composition.
 * @param cb              The command buffer to record into.
 * @param swapchain_image The swapchain image, in PRESENT_SRC_KHR.
 */
static void
record_raw_copy (struct composition *c,
                 VkCommandBuffer     cb,
                 VkImage             swapchain_image)
{
	transition(c, cb, &c->model, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	transition_swapchain(c, cb, swapchain_image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copy_whole_image(c, cb, c->model.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_image,
	                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c->width, c->height);
	transition_swapchain(c, cb, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

/** @brief Records the copy of the meter's resolved white point into the constant slot that the next
 *         dispatch of the pass takes.
 *
 * When the meter feeds the white point, the resolve must see the GPU's resolved value, not the host's
 * one-frame-stale mirror of it: copy the four bytes straight from the meter state into the constant
 * slot this dispatch is about to take, over the placeholder the host wrote. The host memcpy ran
 * before submit, so this transfer write is the last writer ahead of the dispatch's uniform read, and
 * the barrier states that.
 *
 * @param c  The composition, whose meter runs on the GPU.
 * @param cb The command buffer to record into.
 */
static void
record_white_point_copy (struct composition const *c,
                         VkCommandBuffer           cb)
{
	VkDeviceSize const slot_offset = c->pass.slot_stride * c->pass.slot
	                                 + offsetof(struct dlss_nr_constants, white_point);
	VkBufferCopy const patch = { COMPOSITION_METER_RESOLVED_OFFSET, slot_offset, sizeof (float) };
	c->vk->vkCmdCopyBuffer(cb, c->meter_state, c->pass.shader.constant_buffer, 1, &patch);
	VkBufferMemoryBarrier const barrier = {
		.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask       = VK_ACCESS_UNIFORM_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer              = c->pass.shader.constant_buffer,
		.offset              = slot_offset,
		.size                = sizeof (float)
	};
	c->vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                            0, 0, nullptr, 1, &barrier, 0, nullptr);
}

/** @brief Records the barrier that takes the model image back from the in-layer network: from the
 *         network's compute-shader and transfer writes into the composition's reads, in
 *         SHADER_READ_ONLY_OPTIMAL.
 *
 * @param c  The composition, built for the in-layer network.
 * @param cb The command buffer to record into.
 */
static void
take_from_network (struct composition *c,
                   VkCommandBuffer     cb)
{
	VkImageMemoryBarrier const barrier = {
		.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout           = c->model.layout,
		.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = c->model.image,
		.subresourceRange    = COMPOSITION_COLOR_RANGE
	};
	VkPipelineStageFlags const writers = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
	                                     | VK_PIPELINE_STAGE_TRANSFER_BIT;
	c->vk->vkCmdPipelineBarrier(cb, writers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
	                            nullptr, 1, &barrier);
	c->model.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

/** @brief Records the upload of the model's answer from the transport pair into the model image.
 *
 * @param c  The composition.
 * @param cb The command buffer to record into.
 */
static void
record_upload (struct composition *c,
               VkCommandBuffer     cb)
{
	transition(c, cb, &c->model, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkBufferImageCopy const region = {
		.imageSubresource = COMPOSITION_COLOR_LAYERS,
		.imageExtent      = { c->model_w, c->model_h, 1 }
	};
	external_ownership(c, cb, &c->upload, VK_QUEUE_FAMILY_EXTERNAL, c->export_family,
	                   VK_ACCESS_TRANSFER_READ_BIT);
	c->vk->vkCmdCopyBufferToImage(cb, c->upload.buffer, c->model.image,
	                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	external_ownership(c, cb, &c->upload, c->export_family, VK_QUEUE_FAMILY_EXTERNAL,
	                   VK_ACCESS_TRANSFER_READ_BIT);
	transition(c, cb, &c->model, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

/** @brief Records the resolve of the model's answer against the frame, and the copy of the composed
 *         frame into the swapchain image.
 *
 * @param c               The composition, whose model image holds the answer.
 * @param cb              The command buffer to record into.
 * @param swapchain_image The swapchain image, in PRESENT_SRC_KHR.
 * @param s               The frame's settings.
 * @return                true if the resolve was recorded.
 */
static bool
record_resolve (struct composition                      *c,
                VkCommandBuffer                          cb,
                VkImage                                  swapchain_image,
                struct composition_frame_settings const *s)
{
	struct composition_image *const answer = &c->model;
	struct composition_image *const source = c->work.image ? &c->work : &c->proxy;

	transition(c, cb, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	transition(c, cb, &c->frame, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	transition(c, cb, &c->composed, VK_IMAGE_LAYOUT_GENERAL);

	struct dlss_nr_constants res = base_constants(c, s);
	res.mode = DLSS_NR_MODE_RESOLVE;
	res.width = c->width;
	res.height = c->height;

	if (c->meter_state && s->white_point_source == kWhitePointMeasured)
		record_white_point_copy(c, cb);

	if (!dlss_nr_pass_dispatch(&c->pass, cb, &res, c->width, c->height, source->view, answer->view,
	                           c->frame.view, VK_NULL_HANDLE, c->composed.view, VK_NULL_HANDLE,
	                           COMPOSITION_READ_ONLY, COMPOSITION_READ_ONLY))
		return false;

	transition(c, cb, &c->composed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	transition_swapchain(c, cb, swapchain_image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copy_whole_image(c, cb, c->composed.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_image,
	                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c->width, c->height);
	transition_swapchain(c, cb, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	return true;
}

bool
composition_record_compose (struct composition                      *c,
                            VkCommandBuffer                          cb,
                            VkImage                                  swapchain_image,
                            struct composition_frame_settings const *s)
{
	if (!composition_usable(c) || !c->composed.image)
		return false;

	c->flags &= ~COMPOSITION_CAPTURE_RECORDED;
	if (c->flags & COMPOSITION_NETWORK)
		take_from_network(c, cb);
	else
		record_upload(c, cb);

	// A capture pair holds the composed frame, so a frame being captured is resolved even under a
	// bypass.
	if (capture_writer_active(&c->capture)) {
		if (!record_resolve(c, cb, swapchain_image, s))
			return false;
		record_capture_pair(c, cb, s);
		return true;
	}

	bool const raw_copy = s->composition_bypass != 0 && s->compare_mode == 0
	                      && !(c->flags & (COMPOSITION_HDR_PROXY | COMPOSITION_LINEAR_HDR))
	                      && c->model_w == c->width && c->model_h == c->height
	                      && c->model.format == c->work_format;
	if (!raw_copy)
		return record_resolve(c, cb, swapchain_image, s);

	record_raw_copy(c, cb, swapchain_image);
	return true;
}

void
composition_write_captured_frame (struct composition *c)
{
	if (!c || !(c->flags & COMPOSITION_CAPTURE_RECORDED) || !c->capture_buf.mapped)
		return;

	c->flags &= ~COMPOSITION_CAPTURE_RECORDED;
	unsigned char const *const base = c->capture_buf.mapped;
	capture_writer_write_frame(&c->capture, base, base + pair_bytes(c), c->width, c->height,
	                           (uint32_t)c->work_format, &c->capture_metadata);
}

void
composition_consume_meter (struct composition *c)
{
	if (!c || !c->meter_mirror.mapped)
		return;

	// The percentile, the gates and the history now run on the GPU (shaders/meter_reduce.comp); this
	// reads the 128-byte mirror of that state, recorded in leg 1 and landed by leg 1's fence. The
	// reasoning the shader reproduces: the 90th percentile of tile peaks, not the maximum (a sun or a
	// specular hit would normalise the whole picture into the dark) and not the mean (scene
	// brightness says nothing about the buffer's scale); offered only when enough of the frame
	// carries light against its own brightest tile, since the units are the game's and there is no
	// absolute scale. A rejected reading carries the previous answer forward on the GPU, so the value
	// here never needs to be zeroed by a dark or torn frame.
	float const *const mirror = c->meter_mirror.mapped;
	c->measured_white_point = mirror[1];
}
