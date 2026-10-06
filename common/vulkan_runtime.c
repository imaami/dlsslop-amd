/** @file
 *
 * The Vulkan network's host runtime: vulkan_runtime.h.
 */
// SPDX-License-Identifier: MIT
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "error.h"
#include "files.h"
#include "network/network_fallback.h"
#include "network/network_verdict.h"
#include "util.h"
#include "vulkan_plan.h"
#include "vulkan_runtime.h"
#include "vulkan_weights.h"

/* The motion estimate's pyramid below the frame (upstream: kTemporalBase, kTemporalRadius and
 * kTemporalReject in nr_runtime.cpp): levels from a quarter of the frame on, each searched within its
 * radius. */
static constexpr uint32_t MOTION_BASE = 4;                                    //!< The finest level's divisor.
static constexpr int32_t MOTION_RADIUS[VULKAN_RUNTIME_LEVELS] = {2, 3, 3, 4}; //!< Each level's radius.
static constexpr float MOTION_REJECT = 0.5f;                                  //!< The flow's rejection threshold.

/** @brief The temporal pre and post blocks' parameters, NrTemporal's three vec4s (upstream: the
 *         params record_all writes): the gate, the motion's scale and the history's weight scale;
 *         the history's extent and the depth's presence and direction; and where the frame's uv
 *         lands in the motion field. */
static constexpr uint32_t TEMPORAL_PARAMS = 12;

/** @brief The post block weights the history by sigmoid(net) times the model's own blend_scale of
 *         block70.layer0, f16 0x39EB, as NVIDIA's DLL does (upstream: kPostBlendScale).
 *
 * The model tools do not extract it, and the model is checked against its SHA-256, so it is a
 * constant.
 */
static constexpr float BLEND_SCALE = 0.73974609375f;

/* The push blocks of the runtime's own pipelines (upstream: Temporal::LumaPush and
 * Temporal::FlowPush, and those record_all pushes to the pass stages and the alpha pass), as their
 * GLSL declares them. */

/** @brief The luma pyramid's block. */
struct luma_push {
	uint32_t dst_w, dst_h, src_w, src_h, mode;
};

/** @brief The flow's block. */
struct flow_push {
	uint32_t level_w, level_h, coarse_w, coarse_h;
	int32_t  radius;
	uint32_t first, last;
	float    reject;
};

/** @brief The pass stages' block. */
struct stage_push {
	uint32_t w, h;
	float    strength;
	uint32_t anchor;
};

/** @brief The alpha pass's block. */
struct alpha_push {
	uint32_t w, h, rgba8;
};

/** @brief The most arena words that the verdict reads. */
static constexpr size_t VERDICT_WORDS = 13;

/** @brief The verdict's block (network_verdict.comp): the fallback's grid, and the arena's words
 *         that the waits set when they run out. */
struct verdict_push {
	uint32_t groups_x, groups_y, count, words[VERDICT_WORDS];
};

/** @brief The fallback's block (network_fallback.comp). */
struct fallback_push {
	uint32_t w, h;
};

/** @brief A pipeline's bindings and push range, and where its SPIR-V is.
 *
 * Its storage buffers come first ('a' the activation arena, 'w' the weights, 'p' the motion
 * parameters, 'v' the verdict), then its images ('s' storage, 't' sampled). A kernel's SPIR-V is
 * g_STEM.spv below the network's directory; one of the runtime's own is the file there, or the
 * SPIR-V code it embeds from the source file.
 */
struct bindings {
	char const     *buffers;     //!< Its storage buffers in binding order.
	char const     *images;      //!< Its images after them.
	char const     *file;        //!< Its SPIR-V's file, or its source's; nullptr for a kernel.
	uint32_t const *code;        //!< The SPIR-V it embeds, or nullptr.
	size_t          code_size;   //!< The embedded SPIR-V's bytes.
	uint32_t        push;        //!< The bytes of its push range.
	uint32_t        file_length; //!< The length of the file's name.
};

/** @brief A struct bindings of the runtime's own: the file's length is its literal's. */
#define ADAPTER(buffers, images, file, code, code_size, push) \
	{buffers, images, file, code, code_size, push, sizeof (file) - 1}

/** @brief The runtime's own pipelines, by enum vulkan_runtime_adapter after the kernels (upstream:
 *         the adapters of nr_runtime.cpp and the temporal variants of the pre and post blocks).
 *
 * The alpha pass, which restores the frame's alpha when later passes overwrote the input;
 * dlsslop-amd's pass stages; the motion estimate's luma pyramid and flow; the pre block with motion
 * history, with the upper clamp and without; the post block with motion history; dlsslop-amd's
 * verdict on the frame's waits and the fallback that answers with the input when one ran out.
 * install.py reads the SPIR-V files from this table.
 */
static struct bindings const ADAPTER_BINDINGS[] = {
	ADAPTER("", "st", "runtime/runtime_alpha.spv", nullptr, 0, sizeof (struct alpha_push)),
	ADAPTER("", "sts", "runtime/pass_stages.spv", nullptr, 0, sizeof (struct stage_push)),
	ADAPTER("", "tss", "temporal/motion_luma.spv", nullptr, 0, sizeof (struct luma_push)),
	ADAPTER("", "ssss", "temporal/motion_estimate.spv", nullptr, 0, sizeof (struct flow_push)),
	ADAPTER("aawwwap", "tttt", "temporal/temporal_pre_fp32.spv", nullptr, 0,
	        sizeof (struct push_f_swin) + sizeof (struct push_pre_image)),
	ADAPTER("aawwwap", "tttt", "temporal/temporal_pre_fp32nh.spv", nullptr, 0,
	        sizeof (struct push_f_swin) + sizeof (struct push_pre_image)),
	ADAPTER("aawwwp", "ssttt", "temporal/temporal_post_fp32.spv", nullptr, 0,
	        sizeof (struct push_f_swin) + sizeof (struct push_ups) + sizeof (struct push_image_tail)),
	ADAPTER("av", "", "network_verdict.comp", kNetworkVerdictSpv, sizeof kNetworkVerdictSpv,
	        sizeof (struct verdict_push)),
	ADAPTER("", "st", "network_fallback.comp", kNetworkFallbackSpv, sizeof kNetworkFallbackSpv,
	        sizeof (struct fallback_push)),
};

#undef ADAPTER

static_assert(sizeof ADAPTER_BINDINGS / sizeof *ADAPTER_BINDINGS ==
              VULKAN_RUNTIME_PIPELINES - VULKAN_RUNTIME_ALPHA);

/** @brief A kernel's images, by enum vulkan_images. */
static char const *const KERNEL_IMAGES[] = {
	[VULKAN_IMAGES_NONE]   = "",
	[VULKAN_IMAGES_INPUT]  = "t",
	[VULKAN_IMAGES_OUTPUT] = "sst",
};

/** @brief The most bindings a pipeline has: a kernel's at most VULKAN_KERNEL_MOST_BINDINGS, the
 *         temporal blocks' 11. make_pipeline() refuses a pipeline of more. */
static constexpr size_t MOST_BINDINGS = VULKAN_KERNEL_MOST_BINDINGS < 11 ? 11 : VULKAN_KERNEL_MOST_BINDINGS;

/** @brief The most images a pipeline's set binds: the temporal post block's 5. make_pipeline()
 *         refuses a pipeline of more. */
static constexpr size_t SET_IMAGES = 5;

/** @brief The most descriptor sets a runtime has: a kernel's each, two each of the alpha pass, the
 *         pass stages and the fallback, the verdict's, a luma level's and a flow level's for each
 *         parity, the temporal pre and post blocks' for each history, and those of later passes. */
static constexpr size_t MOST_SETS = VULKAN_KERNEL_COUNT + 2 + 2 + 2 + 1 + 2 * 2 * VULKAN_RUNTIME_LEVELS + 2 * 2 + 2;

/** @brief The most images that a frame moves into their layouts: the input, the answer, the second
 *         output, the first pass's input, the scratch, the depth, the luma pyramids, the flow, the
 *         histories and each pass's history. */
static constexpr size_t MOST_SETTLED = 6 + 2 * VULKAN_RUNTIME_LEVELS + VULKAN_RUNTIME_LEVELS + 2 +
                                       VULKAN_RUNTIME_MAX_PASSES;

/** @brief The most threads that create pipelines at once, the build's own included: the largest
 *         pipelines take seconds each to compile, and in the layer the build runs in the game's
 *         process. */
static constexpr size_t COMPILERS = 4;

static constexpr VkPipelineStageFlags TRANSFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
static constexpr VkPipelineStageFlags COMPUTE = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
static constexpr VkAccessFlags READ = VK_ACCESS_SHADER_READ_BIT;
static constexpr VkAccessFlags WRITE = VK_ACCESS_SHADER_WRITE_BIT;
static constexpr VkAccessFlags COPY_READ = VK_ACCESS_TRANSFER_READ_BIT;
static constexpr VkAccessFlags COPY_WRITE = VK_ACCESS_TRANSFER_WRITE_BIT;
static constexpr VkImageLayout UNDEFINED = VK_IMAGE_LAYOUT_UNDEFINED;
static constexpr VkImageLayout GENERAL = VK_IMAGE_LAYOUT_GENERAL;
static constexpr VkImageLayout SAMPLED = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
static constexpr VkImageLayout SOURCE = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
static constexpr VkImageLayout TARGET = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
static constexpr VkImageSubresourceRange COLOR = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
static constexpr VkImageSubresourceLayers COLOR_LAYER = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
static constexpr VkFormat WIDE = VK_FORMAT_R32G32B32A32_SFLOAT;
/* Every image is copied to and from; the network samples some and stores into most (upstream:
 * Context::image). */
static constexpr VkImageUsageFlags COPIES = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
static constexpr VkImageUsageFlags STORAGE = COPIES | VK_IMAGE_USAGE_STORAGE_BIT;
static constexpr VkImageUsageFlags SAMPLED_STORAGE = STORAGE | VK_IMAGE_USAGE_SAMPLED_BIT;
/** @brief What make_image() takes for an image that a shape does without. */
static constexpr VkFormat NONE = VK_FORMAT_UNDEFINED;
static constexpr VkBufferUsageFlags BUFFERS = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT;

/** @brief A pipeline's bindings: a kernel's, or one of the runtime's own after them.
 *
 * @param pipeline The pipeline.
 * @return         Its bindings.
 */
static struct bindings
bindings_of (size_t pipeline)
{
	if (pipeline >= VULKAN_KERNEL_COUNT)
		return ADAPTER_BINDINGS[pipeline - VULKAN_KERNEL_COUNT];
	struct vulkan_kernel_info const *const k = &VULKAN_PLAN_KERNELS[pipeline];
	return (struct bindings){.buffers = k->buffers, .images = KERNEL_IMAGES[k->images], .push = k->push};
}

/** @brief The seconds since a time.
 *
 * @param start The time, from now_ns().
 * @return      The seconds.
 */
static double
seconds_since (uint64_t start)
{
	return (double)(now_ns() - start) / 1e9;
}

/** @brief Writes a line to a device's log, if it has one; a line that does not fit ends in "...".
 *
 * @param d   The device.
 * @param fmt A printf format.
 * @param ... The format's arguments.
 */
[[gnu::cold, gnu::format(printf, 2, 3)]]
static void
log_line (struct vulkan_device const *d,
          char const                 *fmt,
          ...)
{
	if (!d->log)
		return;

	char line[512];
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(line, sizeof line, fmt, args);
	va_end(args);
	if (n < 0)
		return;
	if (n >= (int)sizeof line)
		memcpy(line + sizeof line - sizeof "...", "...", sizeof "...");
	d->log(line);
}

/** @brief The shape that frames are recorded for, as the log says it. */
struct description {
	char text[192]; //!< The words.
};

/** @brief Says how a runtime records frames.
 *
 * @param s The runtime's state.
 * @return  The words: the extent and format, the passes, the stages and motion when there are, and
 *          how the input and the answer go in and out.
 */
[[gnu::cold]]
static struct description
describe (struct vulkan_runtime_state const *s)
{
	// The input and the answer, by the mode and whether each is in the frame's format.
	static char const *const inputs[2][2] = {{"blitted to RGBA32F", "copied"},
	                                         {"sampled in place", "sampled in place"}};
	static char const *const answers[2][2] = {{"blitted from RGBA32F", "stored in the frame's format"},
	                                          {"blitted from RGBA32F into the caller's image",
	                                           "stored in place"}};
	char passes[24] = "";
	if (s->passes > 1) {
		int const length = snprintf(passes, sizeof passes, ", %" PRIu32 " passes", s->passes);
		if (length < 0 || length >= (int)sizeof passes)
			passes[0] = '\0';
	}
	struct description d;
	int const n = snprintf(d.text, sizeof d.text, "%" PRIu32 "x%" PRIu32 "%s%s%s%s: input %s, answer %s",
	                       s->width, s->height, s->rgba8 ? " RGBA8" : " FP16", passes,
	                       s->stages ? ", pass stages" : "", s->motion ? ", motion" : "",
	                       inputs[s->external][s->input_direct], answers[s->external][s->answer_direct]);
	if (n < 0)
		d.text[0] = '\0';
	else if (n >= (int)sizeof d.text)
		memcpy(d.text + sizeof d.text - sizeof "...", "...", sizeof "...");
	return d;
}

enum error_code
vulkan_check (VkResult      result,
              char const   *what,
              struct error *e)
{
	if (result == VK_SUCCESS)
		return ERROR_NONE;
	return error_fail(e, "%s failed (VkResult %d)", what, (int)result);
}

enum error_code
vulkan_memory_type (VkPhysicalDeviceMemoryProperties const *memory,
                    uint32_t                                bits,
                    VkMemoryPropertyFlags                   want,
                    uint32_t                               *type,
                    struct error                           *e)
{
	for (uint32_t i = 0; i < memory->memoryTypeCount; ++i) {
		if ((bits & (1u << i)) && (memory->memoryTypes[i].propertyFlags & want) == want) {
			*type = i;
			return ERROR_NONE;
		}
	}
	return error_fail(e, "no suitable Vulkan memory type");
}

/** @brief Checks a line of a file against the line expected there.
 *
 * @param text     The file.
 * @param size     Its bytes.
 * @param at       The line's start in the file; moves past its newline when it matches.
 * @param expected The line expected, without its newline.
 * @param length   Its length.
 * @return         true if the file holds the line and a newline at @a at.
 */
static bool
line_matches (uint8_t const *text,
              size_t         size,
              size_t        *at,
              char const    *expected,
              size_t         length)
{
	if (size - *at <= length || memcmp(text + *at, expected, length) || text[*at + length] != '\n')
		return false;
	*at += length + 1;
	return true;
}

/** @brief The length of a file's line, up to its newline or the file's end, for a "%.*s".
 *
 * @param text The file.
 * @param size Its bytes.
 * @param at   The line's start.
 * @return     The line's length, at most ERROR_WHAT_BYTES.
 */
static int
line_length (uint8_t const *text,
             size_t         size,
             size_t         at)
{
	uint8_t const *const newline = memchr(text + at, '\n', size - at);
	size_t const length = newline ? (size_t)(newline - text) - at : size - at;
	return length < ERROR_WHAT_BYTES ? (int)length : ERROR_WHAT_BYTES;
}

/** @brief Writes a line that shader-constants.txt must hold.
 *
 * @param i    The line: 0 for VULKAN_PLAN_MANIFEST, then VULKAN_PLAN_SHADER_CONSTANTS' keys and
 *             values.
 * @param line Receives the line, without its newline.
 * @param size Its bytes.
 * @return     The line's length, or -1 if it does not fit.
 */
static int
constant_line (size_t i,
               char  *line,
               size_t size)
{
	struct vulkan_shader_constant const *const c = i ? &VULKAN_PLAN_SHADER_CONSTANTS[i - 1] : nullptr;
	int const length = c ? snprintf(line, size, "%s %" PRIu32, c->key, c->value)
	                     : snprintf(line, size, "%s", VULKAN_PLAN_MANIFEST);
	return length >= 0 && length < (int)size ? length : -1;
}

/** @brief The words for a line of shader-constants.txt that differs from the line expected there.
 *
 * @param path     The file's path.
 * @param text     The file.
 * @param size     Its bytes.
 * @param at       Where the line starts, in the file and in what is expected.
 * @param expected The line expected, empty where the file should end.
 * @param e        Receives the words, or nullptr.
 * @return         ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
constants_differ (char const    *path,
                  uint8_t const *text,
                  size_t         size,
                  size_t         at,
                  char const    *expected,
                  struct error  *e)
{
	return error_fail(e, "%s does not match this build: \"%.*s\" where \"%s\" is expected; rebuild the "
	                  "network's shaders", path, line_length(text, size, at), (char const *)text + at, expected);
}

/** @brief Checks the lines of shader-constants.txt: VULKAN_PLAN_MANIFEST, then
 *         VULKAN_PLAN_SHADER_CONSTANTS' lines, and nothing after them.
 *
 * @param path The file's path, for the words.
 * @param text The file.
 * @param size Its bytes.
 * @param e    Receives the words for the first line that differs and the line expected there, or
 *             nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
check_constant_lines (char const    *path,
                      uint8_t const *text,
                      size_t         size,
                      struct error  *e)
{
	// The first line that differs starts at the same place in the file and in what is expected.
	size_t at = 0;
	for (size_t i = 0; i <= VULKAN_PLAN_SHADER_CONSTANT_COUNT; ++i) {
		char expected[64];
		int const length = constant_line(i, expected, sizeof expected);
		if (length < 0)
			return error_fail(e, "a line of shader-constants.txt is too long for this build");
		if (!line_matches(text, size, &at, expected, (size_t)length))
			return constants_differ(path, text, size, at, expected, e);
	}
	// A file that goes on differs where no line is expected.
	return at == size ? ERROR_NONE : constants_differ(path, text, size, at, "", e);
}

/** @brief Checks DIRECTORY/shader-constants.txt, which must be this build's exactly: the SPIR-V was
 *         built with the constants that the plan's arithmetic assumes.
 *
 * @param directory The directory.
 * @param length    The length of its path.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
check_constants (char const   *directory,
                 size_t        length,
                 struct error *e)
{
	static char const name[] = "shader-constants.txt";
	char *path = files_join(directory, length, name, sizeof name - 1, nullptr);
	if (!path)
		return error_fail(e, "out of memory");
	struct files_data text;
	enum error_code code = files_read(&text, path, e);
	if (code)
		error_wrap(e, "cannot read %s: ", path);
	else
		code = check_constant_lines(path, text.bytes, text.size, e);
	files_data_fini(&text);
	free(path);
	path = nullptr;
	return code;
}

/** @brief Checks the one-line markers beside the network's SPIR-V (VULKAN_PLAN_MARKERS).
 *
 * @param directory The SPIR-V's directory.
 * @param length    The length of its path.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
check_markers (char const   *directory,
               size_t        length,
               struct error *e)
{
	for (size_t i = 0; i < VULKAN_PLAN_MARKER_COUNT; ++i) {
		struct vulkan_marker const *const m = &VULKAN_PLAN_MARKERS[i];
		char *path = files_join(directory, length, m->file, m->file_length, nullptr);
		if (!path)
			return error_fail(e, "out of memory");
		struct files_data text;
		enum error_code code = files_read(&text, path, e);
		if (code) {
			error_wrap(e, "cannot read %s: ", path);
		} else {
			// The file's text without the white space after it.
			static char const space[] = " \t\r\n";
			size_t said = text.size;
			while (said && memchr(space, text.bytes[said - 1], sizeof space - 1))
				--said;
			if (said != m->line_length || memcmp(text.bytes, m->line, said))
				code = error_fail(e, "%s does not say %s; rebuild the network's shaders", path,
				                  m->line);
		}
		files_data_fini(&text);
		free(path);
		path = nullptr;
		if (code)
			return code;
	}
	return ERROR_NONE;
}

/** @brief What the device offers the network (upstream: the checks of the Runtime constructor). */
struct capabilities {
	bool input_direct;  //!< The input can be the frame's format, filled by a copy.
	bool answer_direct; //!< The post block can store the answer in that format.
};

/** @brief Queries what a device offers a shape.
 *
 * @param d     The device.
 * @param shape The shape.
 * @param caps  Receives what it offers; nothing on a failure.
 * @param e     Receives the words for what it lacks, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
query (struct vulkan_device const *d,
       struct vulkan_shape const  *shape,
       struct capabilities        *caps,
       struct error               *e)
{
	*caps = (struct capabilities){};
	struct vulkan_physical_functions const *const f = &d->functions;
	// Nothing here calls vkGetPhysicalDeviceProperties2, but without it vulkan_storage_limit() gives
	// the plan no limit, so the build must fail here.
	char const *const missing = !f->queue_families    ? "vkGetPhysicalDeviceQueueFamilyProperties"
	                          : !f->properties        ? "vkGetPhysicalDeviceProperties2"
	                          : !f->format_properties ? "vkGetPhysicalDeviceFormatProperties2"
	                                                  : nullptr;
	if (missing)
		return error_fail(e, "the network cannot query its device: %s is unavailable", missing);

	// A queue that blits the frame's format into RGBA32F and back.
	uint32_t count = 0;
	f->queue_families(d->physical, &count, nullptr);
	VkQueueFamilyProperties *families = calloc(count ? count : 1, sizeof *families);
	if (!families)
		return error_fail(e, "out of memory");
	f->queue_families(d->physical, &count, families);
	VkQueueFlags const need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
	bool const able = d->family < count && (families[d->family].queueFlags & need) == need;
	free(families);
	families = nullptr;
	if (!able)
		return error_fail(e, "the network's queue family has no graphics and compute");

	// One pass samples the frame in its own format, copied into the input, and without the pass
	// stages the post block stores it in that format.
	VkFormat const frame = shape->fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
	VkFormatProperties3 frame3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
	VkFormatProperties2 frame2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &frame3};
	f->format_properties(d->physical, frame, &frame2);
	VkFormatProperties2 wide = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
	f->format_properties(d->physical, WIDE, &wide);
	VkFormatFeatureFlags const optimal = frame2.formatProperties.optimalTilingFeatures;
	VkFormatFeatureFlags const blits = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
	if ((optimal & blits) != blits || (wide.formatProperties.optimalTilingFeatures & blits) != blits)
		return error_fail(e, "the device cannot blit the frame's format and RGBA32F");
	VkFormatFeatureFlags const sampled = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
	VkFormatFeatureFlags2 const stored = VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT |
	                                     VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT;
	caps->input_direct = shape->passes == 1 && (optimal & sampled) == sampled;
	caps->answer_direct = caps->input_direct && !shape->stages &&
	                      (frame3.optimalTilingFeatures & stored) == stored;
	return ERROR_NONE;
}

/** @brief Picks a memory type.
 *
 * @param d       The device.
 * @param bits    The types allowed, bit i for type i.
 * @param want    The properties wanted, all of them.
 * @param without The properties not wanted, none of them.
 * @return        The first type among @a bits with every property in @a want and none in
 *                @a without, or memoryTypeCount.
 */
static uint32_t
pick_memory (struct vulkan_device const *d,
             uint32_t                    bits,
             VkMemoryPropertyFlags       want,
             VkMemoryPropertyFlags       without)
{
	uint32_t i = 0;
	for (; i < d->memory.memoryTypeCount; ++i) {
		VkMemoryPropertyFlags const flags = d->memory.memoryTypes[i].propertyFlags;
		if ((bits & (1u << i)) && (flags & want) == want && !(flags & without))
			break;
	}
	return i;
}

/** @brief Destroys a pipeline's handles that are set and empties it.
 *
 * @param device The device.
 * @param p      The pipeline.
 */
static void
pipeline_fini (VkDevice                        device,
               struct vulkan_runtime_pipeline *p)
{
	if (p->pipeline)
		vkDestroyPipeline(device, p->pipeline, nullptr);
	if (p->layout)
		vkDestroyPipelineLayout(device, p->layout, nullptr);
	if (p->set_layout)
		vkDestroyDescriptorSetLayout(device, p->set_layout, nullptr);
	*p = (struct vulkan_runtime_pipeline){};
}

/** @brief Destroys an image's handles that are set and empties it.
 *
 * @param device The device.
 * @param i      The image.
 */
static void
image_fini (VkDevice                     device,
            struct vulkan_runtime_image *i)
{
	if (i->view)
		vkDestroyImageView(device, i->view, nullptr);
	if (i->image)
		vkDestroyImage(device, i->image, nullptr);
	if (i->memory)
		vkFreeMemory(device, i->memory, nullptr);
	*i = (struct vulkan_runtime_image){};
}

/** @brief Destroys a buffer's handles that are set and empties it.
 *
 * @param device The device.
 * @param b      The buffer.
 */
static void
buffer_fini (VkDevice                      device,
             struct vulkan_runtime_buffer *b)
{
	if (b->buffer)
		vkDestroyBuffer(device, b->buffer, nullptr);
	if (b->memory)
		vkFreeMemory(device, b->memory, nullptr);
	*b = (struct vulkan_runtime_buffer){};
}

/** @brief The words of a failed pipeline creation.
 *
 * @param result The result.
 * @param file   The pipeline's SPIR-V file.
 * @param e      Receives the words, or nullptr.
 * @return       ERROR_NONE for VK_SUCCESS, otherwise ERROR_FAILED.
 */
static enum error_code
pipeline_check (VkResult      result,
                char const   *file,
                struct error *e)
{
	if (result == VK_SUCCESS)
		return ERROR_NONE;
	return error_fail(e, "vkCreateComputePipelines %s failed (VkResult %d)", file, (int)result);
}

/** @brief Reads a SPIR-V file (upstream: nrvk::read_spirv).
 *
 * @param path The file.
 * @param dest Receives its bytes, a whole number of words that starts with SPIR-V's magic number;
 *             empty on a failure.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
read_spirv (char const        *path,
            struct files_data *dest,
            struct error      *e)
{
	enum error_code const code = files_read(dest, path, e);
	if (code) {
		error_wrap(e, "cannot read %s: ", path);
		return code;
	}
	uint32_t magic = 0;
	if (dest->size >= sizeof magic)
		memcpy(&magic, dest->bytes, sizeof magic);
	if (!dest->size || dest->size % 4 || magic != UINT32_C(0x07230203)) {
		files_data_fini(dest);
		return error_fail(e, "%s is not SPIR-V", path);
	}
	return ERROR_NONE;
}

/** @brief Makes a pipeline's compute pipeline through a cache, from its SPIR-V below the network's
 *         directory or embedded: one set of its bindings and a push range, and 32 lanes, which the
 *         cooperative matrices' fragments assume (upstream: nrvk::Kernel::create).
 *
 * @param device         The device.
 * @param cache          The pipeline cache, or VK_NULL_HANDLE.
 * @param shaders        The network's SPIR-V directory.
 * @param shaders_length The length of its path.
 * @param pipeline       The pipeline.
 * @param p              The pipeline's handles, empty; receives those made, also on a failure.
 * @param e              Receives the words for what stopped it, or nullptr.
 * @return               ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_pipeline (VkDevice                        device,
               VkPipelineCache                 cache,
               char const                     *shaders,
               size_t                          shaders_length,
               size_t                          pipeline,
               struct vulkan_runtime_pipeline *p,
               struct error                   *e)
{
	struct bindings const b = bindings_of(pipeline);
	char kernel_file[64];
	char const *file = b.file;
	size_t file_length = b.file_length;
	if (!file) {
		int const n = snprintf(kernel_file, sizeof kernel_file, "g_%s.spv", VULKAN_PLAN_KERNELS[pipeline].stem);
		if (n < 0 || n >= (int)sizeof kernel_file)
			return error_fail(e, "the name of a kernel's SPIR-V is too long");
		file = kernel_file;
		file_length = (size_t)n;
	}
	VkDescriptorSetLayoutBinding bindings[MOST_BINDINGS];
	uint32_t n = 0;
	for (char const *t = b.buffers; *t; ++t, ++n) {
		if (n == MOST_BINDINGS)
			return error_fail(e, "%s has too many bindings", file);
		bindings[n] = (VkDescriptorSetLayoutBinding){n, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
		                                              VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	}
	for (char const *t = b.images; *t; ++t, ++n) {
		if (n == MOST_BINDINGS || (size_t)(t - b.images) == SET_IMAGES)
			return error_fail(e, "%s has too many bindings", file);
		bindings[n] = (VkDescriptorSetLayoutBinding){
			n, *t == 's' ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
			VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	}
	VkDescriptorSetLayoutCreateInfo const set = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = n,
		.pBindings    = bindings,
	};
	VkDescriptorSetLayout set_layout;
	enum error_code code = pipeline_check(vkCreateDescriptorSetLayout(device, &set, nullptr, &set_layout), file, e);
	if (code)
		return code;
	p->set_layout = set_layout;
	VkPushConstantRange const range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, b.push};
	VkPipelineLayoutCreateInfo const layout_info = {
		.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount         = 1,
		.pSetLayouts            = &p->set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &range,
	};
	VkPipelineLayout layout;
	code = pipeline_check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), file, e);
	if (code)
		return code;
	p->layout = layout;

	struct files_data read = {};
	if (!b.code) {
		char *path = files_join(shaders, shaders_length, file, file_length, nullptr);
		if (!path)
			return error_fail(e, "out of memory");
		code = read_spirv(path, &read, e);
		free(path);
		path = nullptr;
		if (code)
			return code;
	}
	VkShaderModuleCreateInfo const module_info = {
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = b.code ? b.code_size : read.size,
		.pCode    = b.code ? b.code : (uint32_t const *)(void const *)read.bytes,
	};
	VkShaderModule module;
	code = pipeline_check(vkCreateShaderModule(device, &module_info, nullptr, &module), file, e);
	files_data_fini(&read);
	if (code)
		return code;
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo lanes = {
		.sType                = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
		.requiredSubgroupSize = 32,
	};
	VkComputePipelineCreateInfo const info = {
		.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage  = {
			.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext  = &lanes,
			.stage  = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = module,
			.pName  = "main",
		},
		.layout = layout,
	};
	VkPipeline made;
	code = pipeline_check(vkCreateComputePipelines(device, cache, 1, &info, nullptr, &made), file, e);
	vkDestroyShaderModule(device, module, nullptr);
	if (code)
		return code;
	p->pipeline = made;
	return ERROR_NONE;
}

/** @brief How many threads create pipelines: half the CPUs that the build may run on, from 1 to
 *         COMPILERS.
 *
 * @return The number of threads.
 */
static size_t
compilers (void)
{
	cpu_set_t cpus;
	size_t const usable = sched_getaffinity(0, sizeof cpus, &cpus) ? 0 : (size_t)CPU_COUNT(&cpus);
	size_t const half = usable / 2;
	return half < 1 ? 1 : COMPILERS < half ? COMPILERS : half;
}

/** @brief Pipelines being made by several threads, each taking the next one.
 *
 * Once one has failed, no thread takes another. failed is as wide as a pointer, which fills the
 * padding.
 */
struct compile {
	VkPipelineCache                 cache;          //!< The pipeline cache, or VK_NULL_HANDLE.
	char const                     *shaders;        //!< The network's SPIR-V directory.
	size_t const                   *which;          //!< The pipelines to make.
	struct vulkan_runtime_pipeline *pipelines;      //!< Every pipeline, by index.
	VkDevice                        device;         //!< The device.
	size_t                          shaders_length; //!< The length of the directory's path.
	size_t                          count;          //!< The number of pipelines to make.
	atomic_size_t                   next;           //!< The next of which to take.
	atomic_uintptr_t                failed;         //!< Whether one has failed: 1 if one has.
};

/** @brief A thread that makes pipelines of a struct compile. */
struct compiler {
	struct compile *pool; //!< The pipelines.
	/** @brief The index in which of the pipeline that failed, the one this thread took last; SIZE_MAX
	 *         when none did. */
	size_t          failed;
	struct error    e;    //!< Why it failed.
};

/** @brief Makes pipelines until none is left or one has failed.
 *
 * @param arg The thread's struct compiler.
 * @return    nullptr.
 */
static void *
compile_run (void *arg)
{
	struct compiler *const c = arg;
	struct compile *const pool = c->pool;
	for (size_t i; !atomic_load_explicit(&pool->failed, memory_order_relaxed) &&
	               (i = atomic_fetch_add_explicit(&pool->next, 1, memory_order_relaxed)) < pool->count;) {
		size_t const pipeline = pool->which[i];
		if (make_pipeline(pool->device, pool->cache, pool->shaders, pool->shaders_length, pipeline,
		                  &pool->pipelines[pipeline], &c->e)) {
			c->failed = i;
			atomic_store_explicit(&pool->failed, 1, memory_order_relaxed);
		}
	}
	return nullptr;
}

/** @brief Loads the pipeline cache from a file, when it holds one (upstream:
 *         Context::load_pipeline_cache).
 *
 * @param device The device.
 * @param paths  The cache's path.
 * @return       The cache, empty when the file holds none; VK_NULL_HANDLE without a path or when the
 *               device makes none.
 */
static VkPipelineCache
load_cache (VkDevice                   device,
            struct vulkan_paths const *paths)
{
	if (!paths->cache_length)
		return VK_NULL_HANDLE;
	struct files_data data;
	if (files_read(&data, paths->cache, nullptr) == ERROR_NONE && data.size < 32)
		files_data_fini(&data);
	VkPipelineCacheCreateInfo const info = {
		.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
		.initialDataSize = data.size,
		.pInitialData    = data.size ? data.bytes : nullptr,
	};
	VkPipelineCache cache;
	VkResult const created = vkCreatePipelineCache(device, &info, nullptr, &cache);
	files_data_fini(&data);
	return created == VK_SUCCESS ? cache : VK_NULL_HANDLE;
}

/** @brief Writes the pipeline cache to its file, which dlsslopd and every game's in-layer network
 *         share (upstream: Context::save_pipeline_cache).
 *
 * A new file replaces the file in one rename; upstream writes PATH.tmp, which another writer can
 * truncate or rename away, and removes the file before its rename. A failed save only makes the
 * next build compile again.
 *
 * @param device The device.
 * @param cache  The cache, or VK_NULL_HANDLE.
 * @param paths  The cache's path.
 */
static void
save_cache (VkDevice                   device,
            VkPipelineCache            cache,
            struct vulkan_paths const *paths)
{
	size_t bytes = 0;
	if (!cache || !paths->cache_length || vkGetPipelineCacheData(device, cache, &bytes, nullptr) != VK_SUCCESS ||
	    !bytes)
		return;
	uint8_t *data = malloc(bytes);
	if (!data)
		return;
	if (vkGetPipelineCacheData(device, cache, &bytes, data) == VK_SUCCESS)
		(void)files_replace(paths->cache, paths->cache_length, data, bytes, nullptr);
	free(data);
	data = nullptr;
}

/** @brief Records an image's layout change, or a barrier in its layout (upstream: barrier in
 *         nr_runtime.cpp).
 *
 * @param cmd       The command buffer.
 * @param image     The image.
 * @param from      Its layout before.
 * @param to        Its layout after.
 * @param src_stage The stages before.
 * @param src       Their accesses.
 * @param dst_stage The stages after.
 * @param dst       Their accesses.
 */
static void
barrier (VkCommandBuffer      cmd,
         VkImage              image,
         VkImageLayout        from,
         VkImageLayout        to,
         VkPipelineStageFlags src_stage,
         VkAccessFlags        src,
         VkPipelineStageFlags dst_stage,
         VkAccessFlags        dst)
{
	VkImageMemoryBarrier const b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, src, dst, from, to,
	                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, COLOR};
	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

/** @brief Records a barrier between compute work (upstream: compute_barrier).
 *
 * @param cmd        The command buffer.
 * @param invalidate Only invalidate the reading caches, instead of making the work's writes visible
 *                   to reads and writes.
 */
static void
compute_barrier (VkCommandBuffer cmd,
                 bool            invalidate)
{
	VkMemoryBarrier const b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, invalidate ? 0u : WRITE, READ | WRITE};
	vkCmdPipelineBarrier(cmd, COMPUTE, COMPUTE, 0, 1, &b, 0, nullptr, 0, nullptr);
}

/** @brief Records a copy of an extent from one image to another, or a blit that converts between
 *         their formats.
 *
 * @param cmd         The command buffer.
 * @param from        The source.
 * @param from_layout Its layout.
 * @param to          The target.
 * @param to_layout   Its layout.
 * @param width       The extent's width.
 * @param height      Its height.
 * @param convert     Blit instead of copying.
 */
static void
transfer (VkCommandBuffer cmd,
          VkImage         from,
          VkImageLayout   from_layout,
          VkImage         to,
          VkImageLayout   to_layout,
          uint32_t        width,
          uint32_t        height,
          bool            convert)
{
	if (!convert) {
		VkImageCopy const region = {COLOR_LAYER, {}, COLOR_LAYER, {}, {width, height, 1}};
		vkCmdCopyImage(cmd, from, from_layout, to, to_layout, 1, &region);
		return;
	}
	VkOffset3D const end = {(int32_t)width, (int32_t)height, 1};
	VkImageBlit const region = {COLOR_LAYER, {{}, end}, COLOR_LAYER, {{}, end}};
	vkCmdBlitImage(cmd, from, from_layout, to, to_layout, 1, &region, VK_FILTER_NEAREST);
}

/** @brief Records a copy from one image to another, both in GENERAL, between compute work
 *         (upstream: copy_general in record_all).
 *
 * @param cmd    The command buffer.
 * @param from   The source.
 * @param to     The target.
 * @param width  The images' width.
 * @param height Their height.
 */
static void
copy_general (VkCommandBuffer cmd,
              VkImage         from,
              VkImage         to,
              uint32_t        width,
              uint32_t        height)
{
	barrier(cmd, from, GENERAL, GENERAL, COMPUTE, WRITE | READ, TRANSFER, COPY_READ);
	barrier(cmd, to, GENERAL, GENERAL, COMPUTE, READ | WRITE, TRANSFER, COPY_WRITE);
	transfer(cmd, from, GENERAL, to, GENERAL, width, height, false);
	barrier(cmd, to, GENERAL, GENERAL, TRANSFER, COPY_WRITE, COMPUTE, READ);
	barrier(cmd, from, GENERAL, GENERAL, TRANSFER, COPY_READ, COMPUTE, READ | WRITE);
}

/** @brief Puts a frame's controls in the pre block's push_pre_image (upstream: patch_push).
 *
 * The style; the tone, which later passes do without; and the structures, under the automatic mask
 * the skin's and the rest's. A nonzero seed has the shader compute the noise for it instead of
 * reading the noise field, which the build computed for seed 0 (upstream: pre_seed in record_all).
 *
 * @param words The pre block's push words.
 * @param c     The controls.
 * @param later The pass is not the first.
 * @param seed  The noise seed.
 */
static void
patch_pre (uint32_t                     *words,
           struct vulkan_controls const *c,
           bool                          later,
           uint32_t                      seed)
{
	struct push_pre_image p;
	memcpy(&p, words + sizeof (struct push_f_swin) / 4, sizeof p);
	p.style = (float)c->style / 128;
	p.tone = later ? 0.0f : c->tone;
	p.structure = c->auto_mask ? 1.0f : c->structure;
	p.skin = c->auto_mask ? (c->skin < 0 ? c->structure : c->skin) : -1.0f;
	p.other = c->auto_mask ? c->structure : -1.0f;
	if (seed) {
		p.seed = seed;
		p.noise_off = 0;
	}
	memcpy(words + sizeof (struct push_f_swin) / 4, &p, sizeof p);
}

/** @brief Puts a frame's controls in the post block's push_image_tail (upstream: patch_push): the
 *         intensity, and whether it restores the frame's alpha itself (bit 31) and rounds the answer
 *         to 8 bits (bit 30).
 *
 * @param words      The post block's push words.
 * @param c          The controls.
 * @param post_alpha The post block restores the alpha.
 * @param rgba8      The frames are RGBA8.
 */
static void
patch_post (uint32_t                     *words,
            struct vulkan_controls const *c,
            bool                          post_alpha,
            bool                          rgba8)
{
	constexpr size_t at = (sizeof (struct push_f_swin) + sizeof (struct push_ups)) / 4;
	struct push_image_tail p;
	memcpy(&p, words + at, sizeof p);
	p.intensity = c->intensity < 0.0f ? 0.0f : 2.0f < c->intensity ? 2.0f : c->intensity;
	p.w_off = (p.w_off & UINT32_C(0x3FFFFFFF)) |
	          (post_alpha ? UINT32_C(0x80000000) | (rgba8 ? UINT32_C(0x40000000) : 0) : 0);
	memcpy(words + at, &p, sizeof p);
}

/** @brief Adds a barrier that moves an image, when there is one, into a layout for anything after,
 *         from nothing as though new.
 *
 * @param layouts The barriers.
 * @param count   Their number, which grows by one for an image.
 * @param image   The image.
 * @param layout  The layout.
 */
static void
settle (VkImageMemoryBarrier              *layouts,
        uint32_t                          *count,
        struct vulkan_runtime_image const *image,
        VkImageLayout                      layout)
{
	if (image->image)
		layouts[(*count)++] = (VkImageMemoryBarrier){
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, layout,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image->image, COLOR};
}

/** @brief What a build's submission uses, and frees once it has run: never while the device may
 *         still read it, so a submission that is not known to have ended leaks it. running is as
 *         wide as a pointer, which fills the padding. */
struct setup {
	struct vulkan_runtime_buffer staging; //!< The weights' upload.
	VkCommandPool                pool;    //!< The commands' pool.
	VkFence                      fence;   //!< The submission's fence.
	VkCommandBuffer              cmd;     //!< The commands.
	uintptr_t                    running; //!< Whether the submission may still run: 1 if it may.
};

/** @brief Frees what a build's submission used, unless it may still run.
 *
 * @param device The device.
 * @param setup  The submission's objects.
 */
static void
setup_fini (VkDevice      device,
            struct setup *setup)
{
	if (setup->running)
		return;
	if (setup->fence)
		vkDestroyFence(device, setup->fence, nullptr);
	if (setup->pool)
		vkDestroyCommandPool(device, setup->pool, nullptr);
	buffer_fini(device, &setup->staging);
	*setup = (struct setup){};
}

uint64_t
vulkan_storage_limit (struct vulkan_device const *device)
{
	if (!device->functions.properties)
		return UINT64_MAX;
	VkPhysicalDeviceMaintenance3Properties allocation = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
	};
	VkPhysicalDeviceProperties2 properties = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &allocation,
	};
	device->functions.properties(device->physical, &properties);
	uint64_t const range = properties.properties.limits.maxStorageBufferRange;
	uint64_t const most = allocation.maxMemoryAllocationSize;
	return most < range ? most : range;
}

/** @brief The pre block with motion history.
 *
 * @param rt The runtime.
 * @return   VULKAN_RUNTIME_PRE_NH when the plan's pre block is without the upper clamp, else
 *           VULKAN_RUNTIME_PRE.
 */
static size_t
temporal_pre (struct vulkan_runtime const *rt)
{
	return rt->steps[0].kernel == VULKAN_KERNEL_FSWIN_IMAGE_PREDS32_NH ? VULKAN_RUNTIME_PRE_NH : VULKAN_RUNTIME_PRE;
}

/** @brief Makes the noise field's pipeline, which the build runs, the kernels' of the steps, and
 *         every one of the runtime's own but the temporal pre block that the steps' pre block is not,
 *         so that no reshape makes a pipeline.
 *
 * With motion, the temporal variants replace the pre and post blocks, whose kernels' pipelines are
 * made all the same. Every pipeline goes through the cache, compiled at once on this thread and
 * compilers() - 1 more; a thread that cannot start leaves its share to the others. The cache is
 * saved once every pipeline exists or one has failed; upstream creates them one at a time and saves
 * the cache before it creates the runtime's own. Of the pipelines that failed, the words name the
 * first in the order of the pipelines' table, whichever thread took it: each thread takes them in
 * that order, so every pipeline before one that failed was taken, and finished.
 *
 * @param rt    The runtime.
 * @param paths Where the SPIR-V and the cache are.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_pipelines (struct vulkan_runtime     *rt,
                struct vulkan_paths const *paths,
                struct error              *e)
{
	VkDevice const d = rt->device->device;
	bool wanted[VULKAN_RUNTIME_PIPELINES] = {};
	wanted[VULKAN_KERNEL_NOISE_FIELD] = true;
	for (size_t i = 0; i < rt->step_count; ++i)
		wanted[rt->steps[i].kernel] = true;
	for (size_t i = VULKAN_RUNTIME_ALPHA; i < VULKAN_RUNTIME_PIPELINES; ++i)
		wanted[i] = true;
	wanted[temporal_pre(rt) == VULKAN_RUNTIME_PRE ? VULKAN_RUNTIME_PRE_NH : VULKAN_RUNTIME_PRE] = false;
	size_t which[VULKAN_RUNTIME_PIPELINES];
	size_t n = 0;
	for (size_t i = 0; i < VULKAN_RUNTIME_PIPELINES; ++i)
		if (wanted[i])
			which[n++] = i;

	VkPipelineCache const cache = load_cache(d, paths);
	struct compile pool = {
		.shaders        = paths->shaders,
		.which          = which,
		.pipelines      = rt->objects.pipelines,
		.device         = d,
		.cache          = cache,
		.shaders_length = paths->shaders_length,
		.count          = n,
		.next           = 0,
		.failed         = 0,
	};
	struct compiler workers[COMPILERS];
	size_t const most = compilers();
	size_t const threads = (n < most ? n : most) - 1;
	for (size_t t = 0; t <= threads; ++t) {
		workers[t].pool = &pool;
		workers[t].failed = SIZE_MAX;
	}
	pthread_t helpers[COMPILERS - 1];
	size_t started = 0;
	while (started < threads && !pthread_create(&helpers[started], nullptr, compile_run, &workers[started + 1]))
		++started;
	compile_run(&workers[0]);
	// A thread that pthread_create() started is joinable, and no other thread joins it.
	for (size_t t = 0; t < started; ++t)
		(void)pthread_join(helpers[t], nullptr);
	save_cache(d, cache, paths);
	if (cache)
		vkDestroyPipelineCache(d, cache, nullptr);

	struct compiler const *first = nullptr;
	for (size_t t = 0; t <= started; ++t)
		if (workers[t].failed != SIZE_MAX && (!first || workers[t].failed < first->failed))
			first = &workers[t];
	if (!first)
		return ERROR_NONE;
	if (e)
		*e = first->e;
	return ERROR_FAILED;
}

/** @brief An image as a set binds it. */
struct image_descriptor {
	VkImageView   view;    //!< The image's view.
	VkSampler     sampler; //!< Its sampler; VK_NULL_HANDLE when it is stored into.
	VkImageLayout layout;  //!< Its layout.
};

/** @brief What a set holds: the buffers that its pipeline names, and its images, sampled with a
 *         sampler or stored into without. */
struct set_descriptor {
	struct image_descriptor images[SET_IMAGES]; //!< Its images, in binding order.
	VkDescriptorSet        *set;                //!< Receives the set.
	size_t                  pipeline;           //!< The pipeline it is for.
};

/** @brief An image as a set binds it to be stored into.
 *
 * @param i The image.
 * @return  The image in GENERAL, without a sampler.
 */
static struct image_descriptor
stored (struct vulkan_runtime_image const *i)
{
	return (struct image_descriptor){i->view, VK_NULL_HANDLE, GENERAL};
}

/** @brief An image as a set binds it to be sampled linearly.
 *
 * @param o The runtime's objects.
 * @param i The image.
 * @return  The image in GENERAL, with the linear sampler.
 */
static struct image_descriptor
linear (struct vulkan_runtime_objects const *o,
        struct vulkan_runtime_image const   *i)
{
	return (struct image_descriptor){i->view, o->linear, GENERAL};
}

/** @brief Writes a runtime's descriptor sets: every set that its state wants, in one pool, which
 *         replaces the last shape's, made by one allocation and written by one update.
 *
 * @param rt The runtime.
 * @param e  Receives the words for what stopped it, or nullptr.
 * @return   ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_sets (struct vulkan_runtime *rt,
           struct error          *e)
{
	struct vulkan_runtime_objects *const o = &rt->objects;
	struct vulkan_runtime_state const *const s = &rt->state;
	// What frames go through: the frame, which the first pass and the motion estimate sample; the
	// input of later passes; the target, which the post block and the fallback store into; and the
	// first pass's input, which the pass stages, the alpha pass and the fallback sample, kept when
	// later passes overwrite the input. In buffer mode the frame is the input; in image mode the frame
	// is the caller's, so is the target when the post block stores in the frame's format, and the
	// frame stays the first pass's input.
	struct image_descriptor const input = {o->input.view, o->nearest, SAMPLED};
	struct image_descriptor const caller_frame = {rt->images.frame, o->nearest, SAMPLED};
	struct image_descriptor const caller_answer = {rt->images.answer_view, VK_NULL_HANDLE, GENERAL};
	struct image_descriptor const frame = s->external ? caller_frame : input;
	struct image_descriptor const target = s->external && s->answer_direct ? caller_answer : stored(&o->answer);
	struct image_descriptor const first = !s->external && s->passes > 1
	                                    ? (struct image_descriptor){o->shown.view, o->nearest, GENERAL}
	                                    : frame;
	struct set_descriptor sets[MOST_SETS];
	size_t n = 0;
	for (size_t k = 0; k < VULKAN_KERNEL_COUNT; ++k) {
		if (!o->pipelines[k].pipeline)
			continue;
		struct set_descriptor *const set = &sets[n++];
		*set = (struct set_descriptor){.set = &o->kernel_sets[k], .pipeline = k};
		switch (VULKAN_PLAN_KERNELS[k].images) {
		case VULKAN_IMAGES_NONE:
			break;
		case VULKAN_IMAGES_INPUT:
			set->images[0] = frame;
			break;
		case VULKAN_IMAGES_OUTPUT:
			set->images[0] = target;
			set->images[1] = stored(&o->second);
			set->images[2] = frame;
			break;
		}
	}
	if (s->passes > 1) {
		sets[n++] = (struct set_descriptor){
			.images   = {stored(&o->answer), first},
			.set      = &o->alpha_sets[0],
			.pipeline = VULKAN_RUNTIME_ALPHA,
		};
		if (s->stages)
			sets[n++] = (struct set_descriptor){
				.images   = {stored(&o->scratch), first},
				.set      = &o->alpha_sets[1],
				.pipeline = VULKAN_RUNTIME_ALPHA,
			};
	}
	if (s->stages) {
		sets[n++] = (struct set_descriptor){
			.images   = {stored(&o->answer), first, stored(&o->scratch)},
			.set      = &o->stage_sets[0],
			.pipeline = VULKAN_RUNTIME_STAGES,
		};
		sets[n++] = (struct set_descriptor){
			.images   = {stored(&o->scratch), first, stored(&o->answer)},
			.set      = &o->stage_sets[1],
			.pipeline = VULKAN_RUNTIME_STAGES,
		};
	}
	sets[n++] = (struct set_descriptor){.set = &o->verdict_set, .pipeline = VULKAN_RUNTIME_VERDICT};
	sets[n++] = (struct set_descriptor){
		.images   = {target, first},
		.set      = &o->fallback_sets[0],
		.pipeline = VULKAN_RUNTIME_FALLBACK,
	};
	if (s->stages)
		sets[n++] = (struct set_descriptor){
			.images   = {stored(&o->scratch), first},
			.set      = &o->fallback_sets[1],
			.pipeline = VULKAN_RUNTIME_FALLBACK,
		};
	if (s->motion) {
		for (uint32_t p = 0; p < 2; ++p) {
			for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k) {
				// The finest level halves no finer one and the coarsest flow reads no coarser one:
				// they bind a level of their own.
				struct vulkan_runtime_image const *const finer =
					&o->luma[p][k ? k - 1 : VULKAN_RUNTIME_LEVELS - 1];
				struct vulkan_runtime_image const *const coarser =
					&o->flow[k + 1 < VULKAN_RUNTIME_LEVELS ? k + 1 : k];
				sets[n++] = (struct set_descriptor){
					.images   = {frame, stored(finer), stored(&o->luma[p][k])},
					.set      = &o->luma_sets[p][k],
					.pipeline = VULKAN_RUNTIME_LUMA,
				};
				sets[n++] = (struct set_descriptor){
					.images   = {stored(&o->luma[p][k]), stored(&o->luma[1 - p][k]),
					             stored(coarser), stored(&o->flow[k])},
					.set      = &o->flow_sets[p][k],
					.pipeline = VULKAN_RUNTIME_FLOW,
				};
			}
		}
		// A pass reads history c and writes the next frame's into the other or, with later passes,
		// into the second output.
		for (uint32_t c = 0; c < (s->pingpong ? 2u : 1u); ++c) {
			struct vulkan_runtime_image const *const next = s->pingpong ? &o->history[c ^ 1] : &o->second;
			sets[n++] = (struct set_descriptor){
				.images   = {frame, linear(o, &o->flow[0]), linear(o, &o->history[c]),
				             linear(o, &o->depth)},
				.set      = &o->pre_sets[c],
				.pipeline = temporal_pre(rt),
			};
			sets[n++] = (struct set_descriptor){
				.images   = {target, stored(next), frame, linear(o, &o->flow[0]),
				             linear(o, &o->history[c])},
				.set      = &o->post_sets[c],
				.pipeline = VULKAN_RUNTIME_POST,
			};
		}
	}
	// The pre and post blocks of later passes, which read history 0 and store into the answer: in
	// image mode sets of their own, which sample the input; otherwise the first pass's.
	size_t const pre = s->motion ? temporal_pre(rt) : rt->steps[0].kernel;
	size_t const post = s->motion ? VULKAN_RUNTIME_POST : rt->steps[rt->step_count - 1].kernel;
	bool const later = s->external && s->passes > 1;
	if (later && s->motion) {
		sets[n++] = (struct set_descriptor){
			.images   = {input, linear(o, &o->flow[0]), linear(o, &o->history[0]), linear(o, &o->depth)},
			.set      = &o->later_sets[0],
			.pipeline = pre,
		};
		sets[n++] = (struct set_descriptor){
			.images   = {stored(&o->answer), stored(&o->second), input, linear(o, &o->flow[0]),
			             linear(o, &o->history[0])},
			.set      = &o->later_sets[1],
			.pipeline = post,
		};
	} else if (later) {
		sets[n++] = (struct set_descriptor){
			.images   = {input},
			.set      = &o->later_sets[0],
			.pipeline = pre,
		};
		sets[n++] = (struct set_descriptor){
			.images   = {stored(&o->answer), stored(&o->second), input},
			.set      = &o->later_sets[1],
			.pipeline = post,
		};
	}
	// A build in image mode binds none of the caller's images: the sets that would bind them are left
	// out until a reshape binds them.
	if (s->external && !rt->images.generation) {
		size_t kept = 0;
		for (size_t i = 0; i < n; ++i) {
			char const *const images = bindings_of(sets[i].pipeline).images;
			size_t j = 0;
			while (images[j] && sets[i].images[j].view)
				++j;
			if (!images[j])
				sets[kept++] = sets[i];
		}
		n = kept;
	}

	// One pool for them all: storage buffers, storage images and sampled images.
	VkDescriptorType const types[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
	uint32_t counts[3] = {};
	VkDescriptorSetLayout layouts[MOST_SETS];
	for (size_t i = 0; i < n; ++i) {
		struct bindings const b = bindings_of(sets[i].pipeline);
		for (char const *t = b.buffers; *t; ++t)
			++counts[0];
		for (char const *t = b.images; *t; ++t)
			++counts[*t == 's' ? 1 : 2];
		layouts[i] = o->pipelines[sets[i].pipeline].set_layout;
	}
	VkDescriptorPoolSize sizes[3];
	uint32_t size_count = 0;
	for (uint32_t i = 0; i < 3; ++i)
		if (counts[i])
			sizes[size_count++] = (VkDescriptorPoolSize){types[i], counts[i]};
	// The writes, and what they write: the verdict's set binds buffers in every shape.
	size_t const buffer_total = counts[0];
	size_t const image_total = (size_t)counts[1] + counts[2];
	VkWriteDescriptorSet *writes = malloc((buffer_total + image_total) * sizeof *writes);
	VkDescriptorBufferInfo *buffers = malloc(buffer_total * sizeof *buffers);
	VkDescriptorImageInfo *images = image_total ? malloc(image_total * sizeof *images) : nullptr;
	enum error_code code = !writes || !buffers || (image_total && !images) ? error_fail(e, "out of memory")
	                                                                       : ERROR_NONE;

	VkDevice const d = rt->device->device;
	if (!code && o->pool) {
		vkDestroyDescriptorPool(d, o->pool, nullptr);
		o->pool = VK_NULL_HANDLE;
	}
	VkDescriptorPool pool = VK_NULL_HANDLE;
	if (!code) {
		VkDescriptorPoolCreateInfo const pool_info = {
			.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets       = (uint32_t)n,
			.poolSizeCount = size_count,
			.pPoolSizes    = sizes,
		};
		code = vulkan_check(vkCreateDescriptorPool(d, &pool_info, nullptr, &pool),
		                    "create the network's descriptor pool", e);
	}
	VkDescriptorSet made[MOST_SETS];
	if (!code) {
		o->pool = pool;
		VkDescriptorSetAllocateInfo const alloc = {
			.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool     = pool,
			.descriptorSetCount = (uint32_t)n,
			.pSetLayouts        = layouts,
		};
		code = vulkan_check(vkAllocateDescriptorSets(d, &alloc, made),
		                    "allocate the network's descriptor sets", e);
	}
	if (!code) {
		uint32_t write_count = 0, buffer_count = 0, written_images = 0;
		for (size_t i = 0; i < n; ++i) {
			*sets[i].set = made[i];
			struct bindings const b = bindings_of(sets[i].pipeline);
			uint32_t binding = 0;
			for (char const *t = b.buffers; *t; ++t) {
				struct vulkan_runtime_buffer const *const buffer = *t == 'a' ? &o->arena
				                                                 : *t == 'w' ? &o->weights
				                                                 : *t == 'v' ? &o->verdict
				                                                             : &o->params;
				buffers[buffer_count] = (VkDescriptorBufferInfo){buffer->buffer, 0, VK_WHOLE_SIZE};
				writes[write_count++] = (VkWriteDescriptorSet){
					.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet          = made[i],
					.dstBinding      = binding++,
					.descriptorCount = 1,
					.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo     = &buffers[buffer_count++],
				};
			}
			for (size_t j = 0; b.images[j]; ++j) {
				struct image_descriptor const *const image = &sets[i].images[j];
				images[written_images] =
					(VkDescriptorImageInfo){image->sampler, image->view, image->layout};
				writes[write_count++] = (VkWriteDescriptorSet){
					.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet          = made[i],
					.dstBinding      = binding++,
					.descriptorCount = 1,
					.descriptorType  = types[b.images[j] == 's' ? 1 : 2],
					.pImageInfo      = &images[written_images++],
				};
			}
		}
		vkUpdateDescriptorSets(d, write_count, writes, 0, nullptr);
		if (!later) {
			o->later_sets[0] = s->motion ? o->pre_sets[0] : o->kernel_sets[pre];
			o->later_sets[1] = s->motion ? o->post_sets[0] : o->kernel_sets[post];
		}
	}
	free(images);
	images = nullptr;
	free(buffers);
	buffers = nullptr;
	free(writes);
	writes = nullptr;
	return code;
}

/** @brief Records the barrier that moves every image of a runtime's own into the layout that frames
 *         use it in, from nothing as though new. Each frame fills the input before any dispatch
 *         reads it.
 *
 * @param rt  The runtime.
 * @param cmd The command buffer.
 */
static void
settle_images (struct vulkan_runtime const *rt,
               VkCommandBuffer              cmd)
{
	struct vulkan_runtime_objects const *const o = &rt->objects;
	VkImageMemoryBarrier layouts[MOST_SETTLED];
	uint32_t n = 0;
	settle(layouts, &n, &o->input, SAMPLED);
	settle(layouts, &n, &o->answer, GENERAL);
	settle(layouts, &n, &o->second, GENERAL);
	settle(layouts, &n, &o->shown, GENERAL);
	settle(layouts, &n, &o->scratch, GENERAL);
	settle(layouts, &n, &o->depth, GENERAL);
	for (uint32_t p = 0; p < 2; ++p)
		for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k)
			settle(layouts, &n, &o->luma[p][k], GENERAL);
	for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k)
		settle(layouts, &n, &o->flow[k], GENERAL);
	for (uint32_t c = 0; c < 2; ++c)
		settle(layouts, &n, &o->history[c], GENERAL);
	for (uint32_t pass = 0; pass < VULKAN_RUNTIME_MAX_PASSES; ++pass)
		settle(layouts, &n, &o->history_store[pass], GENERAL);
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
	                     0, nullptr, n, layouts);
}

/** @brief Binds a pipeline with its set and pushes its constants.
 *
 * @param rt       The runtime.
 * @param cmd      The command buffer.
 * @param pipeline The pipeline.
 * @param set      Its set.
 * @param push     Its push constants.
 * @param bytes    Their bytes.
 */
static void
bind (struct vulkan_runtime const *rt,
      VkCommandBuffer              cmd,
      size_t                       pipeline,
      VkDescriptorSet              set,
      void const                  *push,
      uint32_t                     bytes)
{
	struct vulkan_runtime_pipeline const *const p = &rt->objects.pipelines[pipeline];
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &set, 0, nullptr);
	vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
}

/** @brief Records a dispatch of a pipeline with its set and push constants.
 *
 * @param rt       The runtime.
 * @param cmd      The command buffer.
 * @param pipeline The pipeline.
 * @param set      Its set.
 * @param x        The workgroups in x.
 * @param y        The workgroups in y.
 * @param z        The workgroups in z.
 * @param push     Its push constants.
 * @param bytes    Their bytes.
 */
static void
dispatch (struct vulkan_runtime const *rt,
          VkCommandBuffer              cmd,
          size_t                       pipeline,
          VkDescriptorSet              set,
          uint32_t                     x,
          uint32_t                     y,
          uint32_t                     z,
          void const                  *push,
          uint32_t                     bytes)
{
	bind(rt, cmd, pipeline, set, push, bytes);
	vkCmdDispatch(cmd, x, y, z);
}

/** @brief Records a step through a pipeline with a set and push words, then the barrier that the
 *         plan puts after it.
 *
 * @param rt       The runtime.
 * @param cmd      The command buffer.
 * @param step     The step.
 * @param pipeline The pipeline.
 * @param set      Its set.
 * @param push     The step's push words.
 */
static void
run_step (struct vulkan_runtime const *rt,
          VkCommandBuffer              cmd,
          struct vulkan_step const    *step,
          size_t                       pipeline,
          VkDescriptorSet              set,
          uint32_t const              *push)
{
	dispatch(rt, cmd, pipeline, set, step->groups[0], step->groups[1], step->groups[2], push, 4u * step->words);
	if (step->after != VULKAN_AFTER_NOTHING)
		compute_barrier(cmd, step->after == VULKAN_AFTER_INVALIDATE);
}

/** @brief Starts a frame: the images into their layouts until a frame was submitted, and to start
 *         over the arena's sync regions and tile counters zeroed.
 *
 * A frame that starts over zeroes them as the build left them. After a wait of a persistent run ran
 * out, they stay short of their counts until then, and every later frame's waits would run out.
 *
 * @param rt    The runtime.
 * @param cmd   The command buffer.
 * @param reset Start the frame over.
 */
static void
begin (struct vulkan_runtime const *rt,
       VkCommandBuffer              cmd,
       bool                         reset)
{
	if (!rt->settled)
		settle_images(rt, cmd);
	if (!reset)
		return;
	VkMemoryBarrier const before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, WRITE, COPY_WRITE};
	vkCmdPipelineBarrier(cmd, COMPUTE, TRANSFER, 0, 1, &before, 0, nullptr, 0, nullptr);
	vkCmdFillBuffer(cmd, rt->objects.arena.buffer, rt->values_end, VK_WHOLE_SIZE, 0);
	VkMemoryBarrier const after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, COPY_WRITE, READ | WRITE};
	vkCmdPipelineBarrier(cmd, TRANSFER, COMPUTE, 0, 1, &after, 0, nullptr, 0, nullptr);
}

/** @brief Records the network's dispatches of a frame whose first pass's input is in place, the
 *         alpha pass, and the verdict and the fallback, which answers with the first pass's input
 *         when a wait ran out.
 *
 * @param rt      The runtime; its recorded history becomes what the frame leaves.
 * @param cmd     The command buffer.
 * @param c       The frame's controls.
 * @param reset   Start the frame over.
 * @param queries Timestamp queries, or VK_NULL_HANDLE.
 * @param query   The query before the one written once the network is done.
 * @return        The image of the runtime's own that holds the answer, or none when the answer is
 *                the caller's.
 */
static VkImage
record_network (struct vulkan_runtime        *rt,
                VkCommandBuffer               cmd,
                struct vulkan_controls const *c,
                bool                          reset,
                VkQueryPool                   queries,
                uint32_t                      query)
{
	struct vulkan_runtime_objects const *const o = &rt->objects;
	struct vulkan_runtime_state const *const s = &rt->state;
	uint32_t const w = s->width, h = s->height, gx = (w + 7) / 8, gy = (h + 7) / 8;
	compute_barrier(cmd, false);
	// The pre and post blocks, which carry the frame's controls, or with motion their temporal
	// variants.
	struct vulkan_step const *const first = &rt->steps[0];
	struct vulkan_step const *const last = &rt->steps[rt->step_count - 1];
	size_t pre = first->kernel, post = last->kernel;
	VkDescriptorSet pre_set = o->kernel_sets[pre], post_set = o->kernel_sets[post];
	uint32_t seed = 0;
	if (s->motion) {
		struct vulkan_runtime_history const *const history = &rt->history;
		pre = temporal_pre(rt);
		post = VULKAN_RUNTIME_POST;
		pre_set = o->pre_sets[history->current];
		post_set = o->post_sets[history->current];
		// The motion estimate: this frame's luma pyramid, then, with a last frame to follow, the flow
		// from its pyramid, coarse to fine; then the parameters the pre and post blocks read, gated on
		// that.
		bool const gate = history->latch && !reset;
		uint32_t const p = history->parity;
		// The pre block's noise seed, as NVIDIA's DLL counts it: frames since the history's first
		// frame or last reset.
		seed = gate ? history->seed : 0;
		// The history this frame leaves, which the next frame reads once vulkan_runtime_submitted()
		// says that this one was submitted.
		rt->recorded = (struct vulkan_runtime_history){
			.parity  = p ^ 1,
			.current = s->pingpong ? history->current ^ 1 : history->current,
			.seed    = seed + 1,
			.latch   = true,
		};
		for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k) {
			struct luma_push const push = {s->level_width[k], s->level_height[k],
			                               k ? s->level_width[k - 1] : w, k ? s->level_height[k - 1] : h,
			                               k ? 1u : 0u};
			dispatch(rt, cmd, VULKAN_RUNTIME_LUMA, o->luma_sets[p][k], (push.dst_w + 7) / 8,
			         (push.dst_h + 7) / 8, 1, &push, sizeof push);
			compute_barrier(cmd, false);
		}
		for (uint32_t k = VULKAN_RUNTIME_LEVELS; gate && k-- > 0;) {
			bool const coarsest = k == VULKAN_RUNTIME_LEVELS - 1;
			struct flow_push const push = {s->level_width[k], s->level_height[k],
			                               coarsest ? 1u : s->level_width[k + 1],
			                               coarsest ? 1u : s->level_height[k + 1],
			                               MOTION_RADIUS[k], coarsest, k == 0, MOTION_REJECT};
			dispatch(rt, cmd, VULKAN_RUNTIME_FLOW, o->flow_sets[p][k], (push.level_w + 7) / 8,
			         (push.level_h + 7) / 8, 1, &push, sizeof push);
			compute_barrier(cmd, false);
		}
		// The gate, the motion's scale, the history's weight scale and extent, no depth, and the
		// estimator's field, which covers the frame.
		float const params[TEMPORAL_PARAMS] = {gate ? 1.0f : 0.0f, 1.0f, 1.0f, BLEND_SCALE, (float)w, (float)h,
		                                       0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
		VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, READ, COPY_WRITE,
		                           VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
		                           o->params.buffer, 0, VK_WHOLE_SIZE};
		vkCmdPipelineBarrier(cmd, COMPUTE, TRANSFER, 0, 0, nullptr, 1, &b, 0, nullptr);
		vkCmdUpdateBuffer(cmd, o->params.buffer, 0, sizeof params, params);
		b.srcAccessMask = COPY_WRITE;
		b.dstAccessMask = READ;
		vkCmdPipelineBarrier(cmd, TRANSFER, COMPUTE, 0, 0, nullptr, 1, &b, 0, nullptr);
	}
	bool in_scratch = false; // the pass stages left the answer in the scratch
	for (uint32_t pass = 0; pass < s->passes; ++pass) {
		if (pass) {
			// The last pass's answer is this pass's input, and this pass's own history the history.
			// From the second pass on, the blocks sample the input.
			VkImage const previous = in_scratch ? o->scratch.image : o->answer.image;
			barrier(cmd, previous, GENERAL, SOURCE, COMPUTE, WRITE, TRANSFER, COPY_READ);
			barrier(cmd, o->input.image, SAMPLED, TARGET, COMPUTE, READ, TRANSFER, COPY_WRITE);
			transfer(cmd, previous, SOURCE, o->input.image, TARGET, w, h, false);
			barrier(cmd, o->input.image, TARGET, SAMPLED, TRANSFER, COPY_WRITE, COMPUTE, READ);
			barrier(cmd, previous, SOURCE, GENERAL, TRANSFER, COPY_READ, COMPUTE, READ | WRITE);
			if (s->motion)
				copy_general(cmd, o->history_store[pass].image, o->history[0].image, w, h);
			pre_set = o->later_sets[0];
			post_set = o->later_sets[1];
		}
		uint32_t words[32];
		memcpy(words, rt->push + first->push, first->words * sizeof *words);
		patch_pre(words, c, pass > 0, seed);
		run_step(rt, cmd, first, pre, pre_set, words);
		for (size_t i = 1; i + 1 < rt->step_count; ++i) {
			struct vulkan_step const *const step = &rt->steps[i];
			run_step(rt, cmd, step, step->kernel, o->kernel_sets[step->kernel], rt->push + step->push);
		}
		memcpy(words, rt->push + last->push, last->words * sizeof *words);
		patch_post(words, c, s->post_alpha, s->rgba8);
		run_step(rt, cmd, last, post, post_set, words);
		if (s->stored) {
			// The pass's history is what the model wrote into the second output.
			VkImage const history = o->history_store[pass].image;
			barrier(cmd, o->second.image, GENERAL, SOURCE, COMPUTE, WRITE, TRANSFER, COPY_READ);
			barrier(cmd, history, GENERAL, GENERAL, COMPUTE, READ, TRANSFER, COPY_WRITE);
			transfer(cmd, o->second.image, SOURCE, history, GENERAL, w, h, false);
			barrier(cmd, history, GENERAL, GENERAL, TRANSFER, COPY_WRITE, COMPUTE, READ);
			barrier(cmd, o->second.image, SOURCE, GENERAL, TRANSFER, COPY_READ, COMPUTE, READ | WRITE);
		}
		// dlsslop-amd's stages on the pass's answer, into the scratch and back.
		in_scratch = false;
		if (s->stages) {
			float const strengths[2] = {c->sharpness, c->color_preserve};
			for (uint32_t anchor = 0; anchor < 2; ++anchor) {
				if (strengths[anchor] == 0)
					continue;
				struct stage_push const push = {w, h, strengths[anchor], anchor};
				dispatch(rt, cmd, VULKAN_RUNTIME_STAGES, o->stage_sets[in_scratch], gx, gy, 1, &push,
				         sizeof push);
				compute_barrier(cmd, false);
				in_scratch = !in_scratch;
			}
		}
	}
	// The first pass's history is what the next frame's first pass reads.
	if (s->stored)
		copy_general(cmd, o->history_store[0].image, o->history[0].image, w, h);
	// The frame's alpha, which one pass's post block restores itself.
	if (!s->post_alpha) {
		struct alpha_push const push = {w, h, s->rgba8};
		dispatch(rt, cmd, VULKAN_RUNTIME_ALPHA, o->alpha_sets[in_scratch], gx, gy, 1, &push, sizeof push);
	}
	if (queries)
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query + 1);
	// Whether a wait of the frame ran out, which the host reads once the frame has run; then the first
	// pass's input into the result, over the grid that the verdict leaves empty when none did.
	struct verdict_push verdict = {gx, gy, (uint32_t)rt->timeout_count, {}};
	memcpy(verdict.words, rt->timeouts, rt->timeout_count * sizeof *verdict.words);
	dispatch(rt, cmd, VULKAN_RUNTIME_VERDICT, o->verdict_set, 1, 1, 1, &verdict, sizeof verdict);
	VkMemoryBarrier const judged = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, WRITE,
	                                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | WRITE | VK_ACCESS_HOST_READ_BIT};
	vkCmdPipelineBarrier(cmd, COMPUTE, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | COMPUTE | VK_PIPELINE_STAGE_HOST_BIT,
	                     0, 1, &judged, 0, nullptr, 0, nullptr);
	struct fallback_push const fallback = {w, h};
	bind(rt, cmd, VULKAN_RUNTIME_FALLBACK, o->fallback_sets[in_scratch], &fallback, sizeof fallback);
	vkCmdDispatchIndirect(cmd, o->verdict.buffer, 0);
	return in_scratch ? o->scratch.image : o->answer.image;
}

void
vulkan_runtime_record_buffers (struct vulkan_runtime        *runtime,
                               VkCommandBuffer               cmd,
                               VkBuffer                      proxy,
                               VkBuffer                      answer,
                               struct vulkan_controls const *controls,
                               bool                          reset,
                               VkQueryPool                   queries,
                               uint32_t                      query)
{
	struct vulkan_runtime_objects const *const o = &runtime->objects;
	struct vulkan_runtime_state const *const s = &runtime->state;
	uint32_t const w = s->width, h = s->height;
	begin(runtime, cmd, reset);
	// The proxy into the input: copied straight in the frame's format, or copied into the frame's
	// image and blitted into RGBA32F. With later passes, which overwrite the input, the first pass's
	// input is kept. The frame's image is rewritten whole wherever it is used.
	VkBufferImageCopy const region = {0, 0, 0, COLOR_LAYER, {}, {w, h, 1}};
	barrier(cmd, o->input.image, SAMPLED, TARGET, COMPUTE, READ, TRANSFER, COPY_WRITE);
	if (s->input_direct) {
		vkCmdCopyBufferToImage(cmd, proxy, o->input.image, TARGET, 1, &region);
	} else {
		barrier(cmd, o->frame.image, UNDEFINED, TARGET, TRANSFER, 0, TRANSFER, COPY_WRITE);
		vkCmdCopyBufferToImage(cmd, proxy, o->frame.image, TARGET, 1, &region);
		barrier(cmd, o->frame.image, TARGET, SOURCE, TRANSFER, COPY_WRITE, TRANSFER, COPY_READ);
		transfer(cmd, o->frame.image, SOURCE, o->input.image, TARGET, w, h, true);
	}
	barrier(cmd, o->input.image, TARGET, SAMPLED, TRANSFER, COPY_WRITE, COMPUTE, READ);
	if (s->passes > 1) {
		barrier(cmd, o->input.image, SAMPLED, SOURCE, COMPUTE, READ, TRANSFER, COPY_READ);
		barrier(cmd, o->shown.image, GENERAL, GENERAL, COMPUTE, READ, TRANSFER, COPY_WRITE);
		transfer(cmd, o->input.image, SOURCE, o->shown.image, GENERAL, w, h, false);
		barrier(cmd, o->shown.image, GENERAL, GENERAL, TRANSFER, COPY_WRITE, COMPUTE, READ);
		barrier(cmd, o->input.image, SOURCE, SAMPLED, TRANSFER, COPY_READ, COMPUTE, READ);
	}
	if (queries)
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, query);
	// Then the answer out into the answer buffer: copied straight in the frame's format, or blitted
	// from RGBA32F into the frame's image and copied from there.
	VkImage const result = record_network(runtime, cmd, controls, reset, queries, query);
	barrier(cmd, result, GENERAL, SOURCE, COMPUTE, WRITE, TRANSFER, COPY_READ);
	if (s->answer_direct) {
		vkCmdCopyImageToBuffer(cmd, result, SOURCE, answer, 1, &region);
	} else {
		barrier(cmd, o->frame.image, UNDEFINED, TARGET, TRANSFER, 0, TRANSFER, COPY_WRITE);
		transfer(cmd, result, SOURCE, o->frame.image, TARGET, w, h, true);
		barrier(cmd, o->frame.image, TARGET, SOURCE, TRANSFER, COPY_WRITE, TRANSFER, COPY_READ);
		vkCmdCopyImageToBuffer(cmd, o->frame.image, SOURCE, answer, 1, &region);
	}
	barrier(cmd, result, SOURCE, GENERAL, TRANSFER, COPY_READ, COMPUTE, READ | WRITE);
}

void
vulkan_runtime_record_images (struct vulkan_runtime        *runtime,
                              VkCommandBuffer               cmd,
                              struct vulkan_controls const *controls,
                              bool                          reset)
{
	begin(runtime, cmd, reset);
	// The post block stored the answer in the caller's image, or it is blitted there from RGBA32F.
	VkImage const result = record_network(runtime, cmd, controls, reset, VK_NULL_HANDLE, 0);
	struct vulkan_runtime_state const *const s = &runtime->state;
	if (s->answer_direct)
		return;
	barrier(cmd, result, GENERAL, SOURCE, COMPUTE, WRITE, TRANSFER, COPY_READ);
	transfer(cmd, result, SOURCE, runtime->images.answer, GENERAL, s->width, s->height, true);
	barrier(cmd, result, SOURCE, GENERAL, TRANSFER, COPY_READ, COMPUTE, READ | WRITE);
}

void
vulkan_runtime_submitted (struct vulkan_runtime *runtime)
{
	runtime->history = runtime->recorded;
	runtime->settled = true;
}

bool
vulkan_runtime_timed_out (struct vulkan_runtime const *runtime)
{
	return runtime->objects.grid[0] != 0;
}

uint64_t
vulkan_runtime_bound (struct vulkan_runtime const *runtime)
{
	return runtime->images.generation;
}

/** @brief Returns a function's code unless it is ERROR_NONE. */
#define TRY(call) do { \
	enum error_code const code_ = (call); \
	if (code_) \
		return code_; \
} while (0)

/** @brief Makes a buffer in its own memory: device-local memory the host cannot see (upstream:
 *         Context::buffer), or host-visible coherent memory, cached where there is such.
 *
 * @param d     The device.
 * @param bytes The buffer's bytes.
 * @param usage Its usage.
 * @param host  Whether the host sees its memory.
 * @param what  What it is, for the words.
 * @param b     The buffer, empty; receives the handles made, also on a failure.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_buffer (struct vulkan_device const   *d,
             VkDeviceSize                  bytes,
             VkBufferUsageFlags            usage,
             bool                          host,
             char const                   *what,
             struct vulkan_runtime_buffer *b,
             struct error                 *e)
{
	VkBufferCreateInfo const info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size  = bytes,
		.usage = usage,
	};
	VkBuffer buffer;
	TRY(vulkan_check(vkCreateBuffer(d->device, &info, nullptr, &buffer), what, e));
	b->buffer = buffer;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(d->device, buffer, &req);
	VkMemoryPropertyFlags const visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t type = host ? pick_memory(d, req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)
	                     : pick_memory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	if (host && type == d->memory.memoryTypeCount)
		type = pick_memory(d, req.memoryTypeBits, visible, 0);
	if (type == d->memory.memoryTypeCount)
		return error_fail(e, "no memory type for %s", what);
	VkMemoryAllocateInfo const alloc = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = type,
	};
	VkDeviceMemory memory;
	TRY(vulkan_check(vkAllocateMemory(d->device, &alloc, nullptr, &memory), what, e));
	b->memory = memory;
	return vulkan_check(vkBindBufferMemory(d->device, buffer, memory, 0), what, e);
}

/** @brief Makes an image what a shape wants: a width x height image of a format for a usage in its
 *         own device-local memory, with its view when it is sampled or stored into (upstream:
 *         Context::image), or none for NONE. An image that is so already stays.
 *
 * @param d      The device.
 * @param width  The image's width.
 * @param height Its height.
 * @param format Its format, or NONE.
 * @param usage  Its usage.
 * @param i      The image; receives the handles made, also on a failure.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_image (struct vulkan_device const  *d,
            uint32_t                     width,
            uint32_t                     height,
            VkFormat                     format,
            VkImageUsageFlags            usage,
            struct vulkan_runtime_image *i,
            struct error                *e)
{
	if (i->width == width && i->height == height && i->format == format && i->usage == usage)
		return ERROR_NONE;
	image_fini(d->device, i);
	if (format == NONE)
		return ERROR_NONE;

	i->width = width;
	i->height = height;
	i->format = format;
	i->usage = usage;
	VkImageCreateInfo const info = {
		.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType   = VK_IMAGE_TYPE_2D,
		.format      = format,
		.extent      = {width, height, 1},
		.mipLevels   = 1,
		.arrayLayers = 1,
		.samples     = VK_SAMPLE_COUNT_1_BIT,
		.tiling      = VK_IMAGE_TILING_OPTIMAL,
		.usage       = usage,
	};
	VkImage image;
	TRY(vulkan_check(vkCreateImage(d->device, &info, nullptr, &image), "create a network image", e));
	i->image = image;
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(d->device, image, &req);
	VkMemoryAllocateInfo const alloc = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = req.size,
		.memoryTypeIndex = pick_memory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0),
	};
	if (alloc.memoryTypeIndex == d->memory.memoryTypeCount)
		return error_fail(e, "no device-local memory type for an image");
	VkDeviceMemory memory;
	TRY(vulkan_check(vkAllocateMemory(d->device, &alloc, nullptr, &memory), "allocate a network image", e));
	i->memory = memory;
	TRY(vulkan_check(vkBindImageMemory(d->device, image, memory, 0), "bind a network image", e));
	if (!(usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)))
		return ERROR_NONE;
	VkImageViewCreateInfo const view_info = {
		.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image            = image,
		.viewType         = VK_IMAGE_VIEW_TYPE_2D,
		.format           = format,
		.subresourceRange = COLOR,
	};
	VkImageView view;
	TRY(vulkan_check(vkCreateImageView(d->device, &view_info, nullptr, &view), "create a network image view", e));
	i->view = view;
	return ERROR_NONE;
}

/** @brief Makes a sampler that filters and addresses as given, and does nothing else.
 *
 * @param device  The device.
 * @param filter  Its filter.
 * @param address Its address mode.
 * @param sampler Receives the sampler; untouched on a failure.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_sampler (VkDevice             device,
              VkFilter             filter,
              VkSamplerAddressMode address,
              VkSampler           *sampler,
              struct error        *e)
{
	VkSamplerCreateInfo const info = {
		.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter    = filter,
		.minFilter    = filter,
		.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = address,
		.addressModeV = address,
		.addressModeW = address,
	};
	VkSampler made;
	TRY(vulkan_check(vkCreateSampler(device, &info, nullptr, &made), "create a network sampler", e));
	*sampler = made;
	return ERROR_NONE;
}

/** @brief Makes a runtime's images what its state wants.
 *
 * @param rt The runtime.
 * @param e  Receives the words for what stopped it, or nullptr.
 * @return   ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_images (struct vulkan_runtime *rt,
             struct error          *e)
{
	struct vulkan_device const *const d = rt->device;
	struct vulkan_runtime_objects *const o = &rt->objects;
	struct vulkan_runtime_state const *const s = &rt->state;
	// The input, sampled: the frame's format filled by a copy, or RGBA32F, filled by a blit or by
	// later passes. The answer, which the post block stores in the frame's format or RGBA32F, and its
	// second output, the model's history, which only the temporal post block of later passes writes:
	// the post block's SPIR-V never stores into it, and with one pass the temporal one stores into the
	// next frame's history. Elsewhere 1x1 stands in for it. The first pass's input when later passes
	// overwrite it, and the pass stages' scratch. The frame's image, through which blits convert the
	// answer from RGBA32F and, when the input is RGBA32F, the proxy into it; an RGBA32F input implies
	// an RGBA32F answer. In image mode the caller's frame is the first pass's input, kept, and the
	// caller's answer takes the answer in the frame's format or a blit of it: the input exists for
	// later passes only, and neither the frame's image nor the copy of the first pass's input does.
	VkFormat const frame = s->rgba8 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT;
	uint32_t const w = s->width, h = s->height;
	uint32_t const second = s->stored ? w : 1, second_height = s->stored ? h : 1;
	bool const own = !s->external;
	VkFormat const input = s->input_direct ? frame : WIDE;
	TRY(make_image(d, w, h, own || s->passes > 1 ? input : NONE,
	               s->input_direct ? COPIES | VK_IMAGE_USAGE_SAMPLED_BIT : SAMPLED_STORAGE, &o->input, e));
	TRY(make_image(d, w, h, s->answer_direct ? (own ? frame : NONE) : WIDE, STORAGE, &o->answer, e));
	TRY(make_image(d, second, second_height, WIDE, STORAGE, &o->second, e));
	TRY(make_image(d, w, h, own && s->passes > 1 ? WIDE : NONE, SAMPLED_STORAGE, &o->shown, e));
	TRY(make_image(d, w, h, s->stages ? WIDE : NONE, STORAGE, &o->scratch, e));
	TRY(make_image(d, w, h, own && !s->answer_direct ? frame : NONE, COPIES, &o->frame, e));
	// The motion history: this frame's and the last frame's luma pyramids, the flow between them, the
	// history the pre and post blocks read, one a pass with later passes, a depth nothing writes, and
	// the parameters. The pre and post blocks also sample the finest flow level, which upstream
	// creates without sampled usage.
	VkFormat const r32 = s->motion ? VK_FORMAT_R32_SFLOAT : NONE;
	VkFormat const rg32 = s->motion ? VK_FORMAT_R32G32_SFLOAT : NONE;
	for (uint32_t p = 0; p < 2; ++p)
		for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k)
			TRY(make_image(d, s->level_width[k], s->level_height[k], r32, STORAGE, &o->luma[p][k], e));
	for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k)
		TRY(make_image(d, s->level_width[k], s->level_height[k], rg32, k ? STORAGE : SAMPLED_STORAGE,
		               &o->flow[k], e));
	for (uint32_t c = 0; c < 2; ++c)
		TRY(make_image(d, w, h, s->motion && (!c || s->pingpong) ? WIDE : NONE, SAMPLED_STORAGE,
		               &o->history[c], e));
	TRY(make_image(d, w, h, r32, SAMPLED_STORAGE, &o->depth, e));
	for (uint32_t pass = 0; pass < VULKAN_RUNTIME_MAX_PASSES; ++pass)
		TRY(make_image(d, w, h, s->stored && pass < s->passes ? WIDE : NONE, STORAGE,
		               &o->history_store[pass], e));
	// The parameters and the sampler stay once made.
	if (!s->motion || o->linear)
		return ERROR_NONE;
	TRY(make_buffer(d, 4 * TEMPORAL_PARAMS, BUFFERS, false, "the network's motion parameters", &o->params, e));
	return make_sampler(d->device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, &o->linear, e);
}

/** @brief Makes a runtime's state that of a shape, from what the device offers it, and its images.
 *
 * @param rt    The runtime.
 * @param shape The shape.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
adopt (struct vulkan_runtime     *rt,
       struct vulkan_shape const *shape,
       struct error              *e)
{
	struct capabilities caps;
	TRY(query(rt->device, shape, &caps, e));
	rt->shape = *shape;
	struct vulkan_runtime_state *const s = &rt->state;
	s->width = shape->width;
	s->height = shape->height;
	s->passes = shape->passes < 1 ? 1 : VULKAN_RUNTIME_MAX_PASSES < shape->passes ? VULKAN_RUNTIME_MAX_PASSES
	                                                                              : shape->passes;
	s->motion = shape->motion;
	s->stages = shape->stages;
	s->rgba8 = !shape->fp16;
	s->external = shape->external;
	s->input_direct = caps.input_direct;
	s->answer_direct = caps.answer_direct;
	s->post_alpha = s->passes == 1;
	s->pingpong = s->motion && s->passes == 1;
	s->stored = s->motion && s->passes > 1;
	s->level_width[0] = (s->width + MOTION_BASE - 1) / MOTION_BASE;
	s->level_height[0] = (s->height + MOTION_BASE - 1) / MOTION_BASE;
	for (uint32_t k = 1; k < VULKAN_RUNTIME_LEVELS; ++k) {
		s->level_width[k] = (s->level_width[k - 1] + 1) / 2;
		s->level_height[k] = (s->level_height[k - 1] + 1) / 2;
	}
	return make_images(rt, e);
}

/** @brief Starts a build's commands.
 *
 * @param rt    The runtime.
 * @param setup The submission's objects, empty; receives those made, also on a failure.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
begin_setup (struct vulkan_runtime const *rt,
             struct setup                *setup,
             struct error                *e)
{
	VkDevice const d = rt->device->device;
	VkCommandPoolCreateInfo const pool_info = {
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = rt->device->family,
	};
	VkCommandPool pool;
	TRY(vulkan_check(vkCreateCommandPool(d, &pool_info, nullptr, &pool), "create the network's build commands", e));
	setup->pool = pool;
	VkCommandBufferAllocateInfo const alloc = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = pool,
		.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	TRY(vulkan_check(vkAllocateCommandBuffers(d, &alloc, &cmd), "allocate the network's build commands", e));
	setup->cmd = cmd;
	VkCommandBufferBeginInfo const begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	return vulkan_check(vkBeginCommandBuffer(cmd, &begin), "begin the network's build commands", e);
}

/** @brief Records a build's part of its submission: the weights packed straight into the staging
 *         memory and uploaded, the arena zeroed, and the noise field in the weights.
 *
 * @param rt    The runtime.
 * @param plan  The plan.
 * @param model The model.
 * @param setup The submission's objects; receives the staging buffer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
upload (struct vulkan_runtime const *rt,
        struct vulkan_plan const    *plan,
        struct vulkan_model const   *model,
        struct setup                *setup,
        struct error                *e)
{
	VkDevice const d = rt->device->device;
	struct vulkan_runtime_objects const *const o = &rt->objects;
	TRY(make_buffer(rt->device, plan->blob_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "the network's upload",
	                &setup->staging, e));
	void *mapped;
	TRY(vulkan_check(vkMapMemory(d, setup->staging.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
	                 "map the network's upload", e));
	TRY(vulkan_weights_pack(plan->segments, plan->segment_count, plan->tables, plan->table_count, model, mapped,
	                        plan->blob_bytes, e));
	VkCommandBuffer const cmd = setup->cmd;
	VkBufferCopy const weights = {0, 0, plan->blob_bytes};
	vkCmdCopyBuffer(cmd, setup->staging.buffer, o->weights.buffer, 1, &weights);
	vkCmdFillBuffer(cmd, o->arena.buffer, 0, plan->arena_bytes, 0);
	VkMemoryBarrier const uploaded = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, COPY_WRITE, READ | WRITE};
	vkCmdPipelineBarrier(cmd, TRANSFER, COMPUTE, 0, 1, &uploaded, 0, nullptr, 0, nullptr);
	size_t const noise = VULKAN_KERNEL_NOISE_FIELD;
	dispatch(rt, cmd, noise, o->kernel_sets[noise], (plan->noise.width + 7) / 8, (plan->noise.height + 7) / 8, 1,
	         &plan->noise, sizeof plan->noise);
	return ERROR_NONE;
}

/** @brief Ends a build's commands with a barrier before anything after them, submits them under the
 *         queue's lock and waits for them outside it.
 *
 * A submission that does not finish leaves the runtime's objects and the submission's to the device.
 *
 * @param rt    The runtime.
 * @param setup The submission's objects; receives the fence.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
end_setup (struct vulkan_runtime *rt,
           struct setup          *setup,
           struct error          *e)
{
	struct vulkan_device const *const device = rt->device;
	VkDevice const d = device->device;
	VkMemoryBarrier const done = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
	                              VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
	vkCmdPipelineBarrier(setup->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
	                     &done, 0, nullptr, 0, nullptr);
	TRY(vulkan_check(vkEndCommandBuffer(setup->cmd), "record the network's build commands", e));
	VkFenceCreateInfo const fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	TRY(vulkan_check(vkCreateFence(d, &fence_info, nullptr, &fence), "create the network's build fence", e));
	setup->fence = fence;
	VkSubmitInfo const submit = {
		.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers    = &setup->cmd,
	};
	if (device->lock)
		device->lock(device->context);
	VkResult const submitted = vkQueueSubmit(device->queue, 1, &submit, fence);
	if (device->unlock)
		device->unlock(device->context);
	TRY(vulkan_check(submitted, "submit the network's build", e));
	setup->running = 1;
	VkResult const waited = vkWaitForFences(d, 1, &fence, VK_TRUE, UINT64_C(30000000000));
	if (waited != VK_SUCCESS) {
		// The device may still read what the build made: it stays.
		rt->objects = (struct vulkan_runtime_objects){};
		log_line(device, "the network's build did not finish; its memory is left to the device");
		return vulkan_check(waited, "wait for the network's build", e);
	}
	setup->running = 0;
	return ERROR_NONE;
}

/** @brief Builds a runtime with an open model: the network's buffers, images, pipelines and sets,
 *         and its submission, then logs what it built.
 *
 * @param rt    The runtime, holding its device.
 * @param paths Where the SPIR-V and the cache are.
 * @param shape The shape.
 * @param plan  The plan, whose steps, push words and timeouts the runtime takes.
 * @param model The model.
 * @param start When the build started.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make_with_model (struct vulkan_runtime     *rt,
                 struct vulkan_paths const *paths,
                 struct vulkan_shape const *shape,
                 struct vulkan_plan        *plan,
                 struct vulkan_model const *model,
                 uint64_t                   start,
                 struct error              *e)
{
	struct vulkan_clamp_free clamp_free;
	TRY(vulkan_weights_clamp_free(plan->segments, plan->segment_count, model, &clamp_free, e));
	vulkan_plan_unclamp(plan, &clamp_free);
	// The runtime takes the steps, the push words and the timeouts.
	rt->steps = plan->steps;
	plan->steps = nullptr;
	rt->step_count = plan->step_count;
	plan->step_count = 0;
	rt->push = plan->push;
	plan->push = nullptr;
	plan->push_count = 0;
	rt->values_end = plan->values_end;
	rt->timeouts = plan->timeouts;
	plan->timeouts = nullptr;
	rt->timeout_count = plan->timeout_count;
	plan->timeout_count = 0;
	TRY(adopt(rt, shape, e));
	// What stays while the runtime lives: the activation arena, the weights and the input's sampler.
	struct vulkan_device const *const d = rt->device;
	struct vulkan_runtime_objects *const o = &rt->objects;
	TRY(make_buffer(d, plan->arena_bytes, BUFFERS, false, "the network's activation arena", &o->arena, e));
	TRY(make_buffer(d, plan->blob_bytes, BUFFERS, false, "the network's weights", &o->weights, e));
	// The verdict, which the host reads after each frame: no wait ran out before the first.
	TRY(make_buffer(d, sizeof (VkDispatchIndirectCommand),
	                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, true,
	                "the network's verdict", &o->verdict, e));
	void *grid;
	TRY(vulkan_check(vkMapMemory(d->device, o->verdict.memory, 0, VK_WHOLE_SIZE, 0, &grid),
	                 "map the network's verdict", e));
	o->grid = memset(grid, 0, sizeof (VkDispatchIndirectCommand));
	TRY(make_sampler(d->device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, &o->nearest, e));
	uint64_t const compiling = now_ns();
	TRY(make_pipelines(rt, paths, e));
	double const compiled = seconds_since(compiling);
	TRY(make_sets(rt, e));
	// One submission: the weights and the arena, and a barrier before anything after it.
	uint64_t const setting_up = now_ns();
	struct setup setup = {};
	enum error_code code = begin_setup(rt, &setup, e);
	if (!code)
		code = upload(rt, plan, model, &setup, e);
	double const packed = seconds_since(setting_up);
	if (!code)
		code = end_setup(rt, &setup, e);
	if (code) {
		setup_fini(d->device, &setup);
		return code;
	}
	double const ran = seconds_since(setting_up) - packed;
	pipeline_fini(d->device, &o->pipelines[VULKAN_KERNEL_NOISE_FIELD]);

	size_t pipelines = 0;
	for (size_t i = 0; i < VULKAN_RUNTIME_PIPELINES; ++i)
		pipelines += o->pipelines[i].pipeline != VK_NULL_HANDLE;
	log_line(d, "built the network in %.2f s (pipelines %.2f s, weights %.2f s, GPU %.2f s) for %s, %zu dispatches "
	         "a pass, arena %.1f MB, weights %.1f MB, %zu pipelines, %" PRIu32 " barriers chained",
	         seconds_since(start), compiled, packed, ran, describe(&rt->state).text, rt->step_count,
	         (double)plan->arena_bytes / 1e6, (double)plan->blob_bytes / 1e6, pipelines, plan->chained);
	setup_fini(d->device, &setup);
	return ERROR_NONE;
}

/** @brief Builds a runtime: checks what the build reads and opens the model, then builds with it.
 *
 * @param rt    The runtime, holding its device.
 * @param paths Where the network's files are.
 * @param shape The shape.
 * @param plan  The plan.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
make (struct vulkan_runtime     *rt,
      struct vulkan_paths const *paths,
      struct vulkan_shape const *shape,
      struct vulkan_plan        *plan,
      struct error              *e)
{
	uint64_t const start = now_ns();
	if (plan->width != shape->width || plan->height != shape->height || plan->step_count < 2 ||
	    plan->steps[0].kernel != VULKAN_KERNEL_FSWIN_IMAGE_PREDS32 ||
	    plan->steps[plan->step_count - 1].kernel != VULKAN_KERNEL_FSWIN_IMAGE_POST32 ||
	    plan->steps[plan->step_count - 1].after != VULKAN_AFTER_FULL ||
	    plan->timeout_count > VERDICT_WORDS)
		return error_fail(e, "network plan: not a plan of the frames the network is built for");
	// What the build reads and what the device offers, before any object.
	TRY(check_constants(paths->shaders, paths->shaders_length, e));
	size_t temporal_length;
	char *temporal = files_join(paths->shaders, paths->shaders_length, "temporal", sizeof "temporal" - 1,
	                            &temporal_length);
	if (!temporal)
		return error_fail(e, "out of memory");
	enum error_code code = check_constants(temporal, temporal_length, e);
	free(temporal);
	temporal = nullptr;
	if (code)
		return code;
	TRY(check_markers(paths->shaders, paths->shaders_length, e));
	struct vulkan_model model;
	code = vulkan_model_open(&model, paths->model, e);
	if (!code)
		code = make_with_model(rt, paths, shape, plan, &model, start, e);
	vulkan_model_fini(&model);
	return code;
}

enum error_code
vulkan_runtime_reshape (struct vulkan_runtime            *runtime,
                        struct vulkan_shape const        *shape,
                        struct vulkan_frame_images const *images,
                        struct error                     *e)
{
	uint64_t const start = now_ns();
	struct vulkan_shape const *const own = &runtime->shape;
	if (shape->width != runtime->state.width || shape->height != runtime->state.height)
		return error_fail(e, "network reshape: frames of another extent need a build");
	if (images)
		runtime->images = *images;
	// The runtime's own shape: the sets alone, which bind the images, while the images, their contents
	// and the motion history stay.
	if (shape->width == own->width && shape->height == own->height && shape->passes == own->passes &&
	    shape->fp16 == own->fp16 && shape->motion == own->motion && shape->stages == own->stages &&
	    shape->external == own->external)
		return make_sets(runtime, e);
	TRY(adopt(runtime, shape, e));
	TRY(make_sets(runtime, e));
	runtime->history = runtime->recorded = (struct vulkan_runtime_history){};
	runtime->settled = false;
	log_line(runtime->device, "reshaped the network in %.3f ms for %s; its weights and pipelines stay",
	         seconds_since(start) * 1e3, describe(&runtime->state).text);
	return ERROR_NONE;
}

#undef TRY

enum error_code
vulkan_runtime_build (struct vulkan_runtime      *dest,
                      struct vulkan_device const *device,
                      struct vulkan_paths const  *paths,
                      struct vulkan_shape const  *shape,
                      struct vulkan_plan         *plan,
                      struct error               *e)
{
	*dest = (struct vulkan_runtime){.device = device};
	enum error_code const code = make(dest, paths, shape, plan, e);
	vulkan_plan_fini(plan);
	if (code)
		vulkan_runtime_fini(dest);
	return code;
}

void
vulkan_runtime_fini (struct vulkan_runtime *runtime)
{
	if (!runtime)
		return;
	if (runtime->device) {
		VkDevice const d = runtime->device->device;
		struct vulkan_runtime_objects *const o = &runtime->objects;
		if (o->pool)
			vkDestroyDescriptorPool(d, o->pool, nullptr);
		for (size_t i = 0; i < VULKAN_RUNTIME_PIPELINES; ++i)
			pipeline_fini(d, &o->pipelines[i]);
		if (o->linear)
			vkDestroySampler(d, o->linear, nullptr);
		if (o->nearest)
			vkDestroySampler(d, o->nearest, nullptr);
		for (uint32_t pass = 0; pass < VULKAN_RUNTIME_MAX_PASSES; ++pass)
			image_fini(d, &o->history_store[pass]);
		image_fini(d, &o->depth);
		for (uint32_t c = 0; c < 2; ++c)
			image_fini(d, &o->history[c]);
		for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k)
			image_fini(d, &o->flow[k]);
		for (uint32_t p = 0; p < 2; ++p)
			for (uint32_t k = 0; k < VULKAN_RUNTIME_LEVELS; ++k)
				image_fini(d, &o->luma[p][k]);
		image_fini(d, &o->frame);
		image_fini(d, &o->scratch);
		image_fini(d, &o->shown);
		image_fini(d, &o->second);
		image_fini(d, &o->answer);
		image_fini(d, &o->input);
		buffer_fini(d, &o->verdict);
		buffer_fini(d, &o->params);
		buffer_fini(d, &o->weights);
		buffer_fini(d, &o->arena);
	}
	free(runtime->timeouts);
	runtime->timeouts = nullptr;
	free(runtime->push);
	runtime->push = nullptr;
	free(runtime->steps);
	runtime->steps = nullptr;
	*runtime = (struct vulkan_runtime){};
}
