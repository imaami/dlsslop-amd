/** @file
 *
 * DLSSNR-AMD's Vulkan network on a caller's device: network_recorder.h.
 */
// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "error.h"
#include "network_recorder.h"
#include "util.h"
#include "vulkan_frame.h"
#include "vulkan_plan.h"
#include "vulkan_runtime.h"

/** @brief Waits run out while other GPU work holds the device, for as long as it does: a line at
 *         most this often says so, in nanoseconds. */
static constexpr uint64_t TIMEOUT_LOG = UINT64_C(10000000000);

/** @brief Whether a frame uses the pass stages.
 *
 * @param f The frame.
 * @return  true if its sharpness or color preservation is not 0.
 */
static bool
uses_stages (struct vulkan_frame const *f)
{
	return f->sharpness != 0 || f->color_preserve != 0;
}

/** @brief A frame's passes, in the range the recorder takes.
 *
 * @param f The frame.
 * @return  Its passes, from 1 to NETWORK_RECORDER_MAX_PASSES.
 */
static uint32_t
passes_of (struct vulkan_frame const *f)
{
	return f->passes < 1 ? 1 : NETWORK_RECORDER_MAX_PASSES < f->passes ? NETWORK_RECORDER_MAX_PASSES : f->passes;
}

/** @brief The smaller of a control and NETWORK_RECORDER_MAX_CONTROL, as std::min compares them: a
 *         NaN control stays NaN.
 *
 * @param control The control.
 * @return        NETWORK_RECORDER_MAX_CONTROL if it is less than @a control, otherwise @a control.
 */
static float
capped (float control)
{
	return NETWORK_RECORDER_MAX_CONTROL < control ? NETWORK_RECORDER_MAX_CONTROL : control;
}

enum error_code
network_recorder_init (struct network_recorder    *dest,
                       struct vulkan_device const *device,
                       struct vulkan_paths const  *paths,
                       bool                        external,
                       struct error               *e)
{
	*dest = (struct network_recorder){};
	// The three paths, each with its null, in one block.
	size_t const model = paths->model_length + 1;
	size_t const shaders = paths->shaders_length + 1;
	char *const copies = malloc(model + shaders + paths->cache_length + 1);
	if (!copies)
		return error_fail(e, "out of memory");
	dest->copies = copies;
	memcpy(copies, paths->model, paths->model_length);
	copies[paths->model_length] = '\0';
	memcpy(copies + model, paths->shaders, paths->shaders_length);
	copies[model + paths->shaders_length] = '\0';
	if (paths->cache_length)
		memcpy(copies + model + shaders, paths->cache, paths->cache_length);
	copies[model + shaders + paths->cache_length] = '\0';
	dest->device = *device;
	dest->paths = (struct vulkan_paths){
		.model          = copies,
		.shaders        = copies + model,
		.cache          = copies + model + shaders,
		.model_length   = paths->model_length,
		.shaders_length = paths->shaders_length,
		.cache_length   = paths->cache_length,
	};
	dest->storage = vulkan_storage_limit(device);
	dest->external = external;
	return ERROR_NONE;
}

void
network_recorder_fini (struct network_recorder *recorder)
{
	if (!recorder)
		return;
	vulkan_runtime_fini(&recorder->runtime);
	vulkan_plan_fini(&recorder->plan);
	free(recorder->copies);
	recorder->copies = nullptr;
	*recorder = (struct network_recorder){};
}

bool
network_recorder_shape_differs (struct network_recorder const *recorder,
                                struct vulkan_frame const     *frame)
{
	if (!recorder)
		return true;
	struct vulkan_shape const *const s = &recorder->shape;
	// Pass stages stay built for frames without them: such a frame blits its answer out instead of
	// copying it.
	return !recorder->runtime.device || s->width != frame->width || s->height != frame->height ||
	       s->fp16 != frame->fp16 || s->motion != frame->motion || s->passes != passes_of(frame) ||
	       (uses_stages(frame) && !s->stages);
}

bool
network_recorder_has_extent (struct network_recorder const *recorder,
                             struct vulkan_frame const     *frame)
{
	return recorder && recorder->runtime.device && recorder->shape.width == frame->width &&
	       recorder->shape.height == frame->height;
}

/** @brief Plans the network for an extent, unless the recorder holds its plan or the extent was
 *         rejected.
 *
 * @param r      The recorder.
 * @param width  The extent's width.
 * @param height Its height.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE; ERROR_REJECTED for an extent the network does not take; ERROR_FAILED.
 */
static enum error_code
plan_for (struct network_recorder *r,
          uint32_t                 width,
          uint32_t                 height,
          struct error            *e)
{
	if (r->plan.width && r->plan.width == width && r->plan.height == height)
		return ERROR_NONE;
	if (r->rejected[0] == width && r->rejected[1] == height && r->rejection.what[0])
		return error_reject(e, "%s", r->rejection.what);
	struct vulkan_plan planned;
	struct error words;
	enum error_code const code = vulkan_plan_init(&planned, width, height, r->storage, &words);
	if (code) {
		if (code != ERROR_FAILED) {
			r->rejected[0] = width;
			r->rejected[1] = height;
			r->rejection = words;
		}
		if (e)
			*e = words;
		return code;
	}
	vulkan_plan_fini(&r->plan);
	r->plan = planned;
	return ERROR_NONE;
}

enum error_code
network_recorder_plan (struct network_recorder   *recorder,
                       struct vulkan_frame const *frame,
                       struct error              *e)
{
	// The runtime's extent was planned before, and network_recorder_shape() reshapes for it without a
	// plan.
	if (network_recorder_has_extent(recorder, frame))
		return ERROR_NONE;
	return plan_for(recorder, frame->width, frame->height, e);
}

/** @brief Reshapes the runtime for a shape with images; a runtime that fails to is freed.
 *
 * @param r      The recorder.
 * @param shape  The shape.
 * @param images The images, or nullptr.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
reshape (struct network_recorder          *r,
         struct vulkan_shape const        *shape,
         struct vulkan_frame_images const *images,
         struct error                     *e)
{
	enum error_code const code = vulkan_runtime_reshape(&r->runtime, shape, images, e);
	if (code)
		vulkan_runtime_fini(&r->runtime);
	return code;
}

enum error_code
network_recorder_shape (struct network_recorder          *recorder,
                        struct vulkan_frame const        *frame,
                        struct vulkan_frame_images const *images,
                        bool                             *changed,
                        struct error                     *e)
{
	*changed = false;
	if (!network_recorder_shape_differs(recorder, frame)) {
		// Images of another generation: a reshape for the runtime's own shape binds them.
		if (!images || images->generation == vulkan_runtime_bound(&recorder->runtime))
			return ERROR_NONE;
		struct vulkan_shape const own = recorder->shape;
		enum error_code const code = reshape(recorder, &own, images, e);
		*changed = !code;
		return code;
	}
	struct vulkan_shape const shape = {
		.width    = frame->width,
		.height   = frame->height,
		.passes   = passes_of(frame),
		.fp16     = frame->fp16,
		.motion   = frame->motion,
		.stages   = uses_stages(frame),
		.external = recorder->external,
	};
	enum error_code code;
	// A runtime of the frame's extent keeps its weights and pipelines.
	if (network_recorder_has_extent(recorder, frame)) {
		code = reshape(recorder, &shape, images, e);
	} else {
		code = plan_for(recorder, frame->width, frame->height, e);
		if (code)
			return code;
		// The plan is not kept: the runtime takes the steps and push words it records.
		vulkan_runtime_fini(&recorder->runtime);
		code = vulkan_runtime_build(&recorder->runtime, &recorder->device, &recorder->paths, &shape,
		                            &recorder->plan, e);
		// A build binds no images.
		if (!code && images)
			code = reshape(recorder, &shape, images, e);
	}
	if (code)
		return code;
	recorder->shape = shape;
	recorder->last.width = 0;
	recorder->recorded.width = 0;
	*changed = true;
	return ERROR_NONE;
}

/** @brief A frame's controls, and whether it starts the network over.
 *
 * It does when a wait of the last frame ran out, which other GPU work that holds the device for
 * milliseconds can cause, and which is logged. The motion history also starts over when the frame's
 * settings change, and at a build for another pass count.
 *
 * @param r        The recorder.
 * @param frame    The frame.
 * @param controls Receives the frame's controls.
 * @return         true if the frame starts the network over.
 */
static bool
begin (struct network_recorder   *r,
       struct vulkan_frame const *frame,
       struct vulkan_controls    *controls)
{
	bool const timed_out = vulkan_runtime_timed_out(&r->runtime);
	if (timed_out) {
		uint64_t const now = now_ns();
		if (r->device.log && now >= r->next_log) {
			r->device.log("a wait of the network ran out while other GPU work held the device: the network "
			              "answered that frame with its input and starts over (logged at most every 10 s)");
			r->next_log = now + TIMEOUT_LOG;
		}
	}
	*controls = (struct vulkan_controls){
		.style          = frame->style,
		.intensity      = capped(frame->intensity),
		.tone           = capped(frame->local_tone),
		.structure      = capped(frame->local_structure),
		.skin           = frame->skin_structure,
		.sharpness      = frame->sharpness,
		.color_preserve = frame->color_preserve,
		.auto_mask      = frame->auto_mask,
	};
	// The settings that the history follows; a NaN control never equals itself and starts it over.
	struct vulkan_frame const *const last = &r->last;
	return timed_out || !last->width || last->intensity != frame->intensity ||
	       last->local_tone != frame->local_tone || last->local_structure != frame->local_structure ||
	       last->style != frame->style || last->skin_structure != frame->skin_structure ||
	       last->auto_mask != frame->auto_mask;
}

void
network_recorder_record_buffers (struct network_recorder   *recorder,
                                 VkCommandBuffer            cmd,
                                 VkBuffer                   proxy,
                                 VkBuffer                   answer,
                                 struct vulkan_frame const *frame,
                                 uint32_t                   family,
                                 bool                       exported,
                                 VkQueryPool                queries,
                                 uint32_t                   query)
{
	// Transfers before the network wrote the proxy and read the last answer; the runtime's read the
	// proxy and write the answer. Exported, the pair belongs to VK_QUEUE_FAMILY_EXTERNAL between
	// frames: taken here and given back below.
	uint32_t const outside = exported ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
	uint32_t const inside = exported ? family : VK_QUEUE_FAMILY_IGNORED;
	VkBufferMemoryBarrier pair[2] = {
		{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
		 VK_ACCESS_TRANSFER_READ_BIT, outside, inside, proxy, 0, VK_WHOLE_SIZE},
		{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT,
		 VK_ACCESS_TRANSFER_WRITE_BIT, outside, inside, answer, 0, VK_WHOLE_SIZE},
	};
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
	                     2, pair, 0, nullptr);
	struct vulkan_controls controls;
	bool const reset = begin(recorder, frame, &controls);
	vulkan_runtime_record_buffers(&recorder->runtime, cmd, proxy, answer, &controls, reset, queries, query);
	recorder->recorded = *frame;
	for (size_t i = 0; i < 2; ++i) {
		VkBufferMemoryBarrier *const b = &pair[i];
		VkAccessFlags const access = b->srcAccessMask;
		b->srcAccessMask = b->dstAccessMask;
		b->dstAccessMask = access;
		uint32_t const queue = b->srcQueueFamilyIndex;
		b->srcQueueFamilyIndex = b->dstQueueFamilyIndex;
		b->dstQueueFamilyIndex = queue;
	}
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
	                     2, pair, 0, nullptr);
}

void
network_recorder_record_images (struct network_recorder   *recorder,
                                VkCommandBuffer            cmd,
                                struct vulkan_frame const *frame)
{
	struct vulkan_controls controls;
	bool const reset = begin(recorder, frame, &controls);
	vulkan_runtime_record_images(&recorder->runtime, cmd, &controls, reset);
	recorder->recorded = *frame;
}

void
network_recorder_submitted (struct network_recorder *recorder)
{
	recorder->last = recorder->recorded;
	vulkan_runtime_submitted(&recorder->runtime);
}

bool
network_recorder_timed_out (struct network_recorder const *recorder)
{
	return recorder && vulkan_runtime_timed_out(&recorder->runtime);
}
