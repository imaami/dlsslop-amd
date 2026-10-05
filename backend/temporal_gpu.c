/** @file
 *
 * GPU optical flow and the network's temporal RGB input: temporal_gpu.h.
 */
// SPDX-License-Identifier: MIT
#include "temporal_gpu.h"

/** @brief Allocates a device buffer.
 *
 * @param temporal The state.
 * @param buffer   Receives the buffer; unchanged on a failure.
 * @param bytes    Its bytes.
 * @param what     What it is for, as the words of a failure say.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
allocate (struct temporal_gpu const  *temporal,
          void                      **buffer,
          size_t                      bytes,
          char const                 *what,
          struct error               *e)
{
	void *allocated = nullptr;
	int const result = temporal->kernels->api->hipMalloc(&allocated, bytes);
	if (result)
		return hip_fail(e, temporal->kernels->api, result, "%s", what);
	*buffer = allocated;
	return ERROR_NONE;
}

/** @brief Frees a device buffer, if there is one, and forgets it.
 *
 * @param api    The runtime.
 * @param buffer The buffer, or nullptr.
 */
static void
release (struct hip_api const  *api,
         void                 **buffer)
{
	if (*buffer) {
		api->hipFree(*buffer);
		*buffer = nullptr;
	}
}

/** @brief Waits for the stream, then frees every buffer and drops the history.
 *
 * @param temporal The state.
 */
static void
release_buffers (struct temporal_gpu *temporal)
{
	struct hip_api const *const api = temporal->kernels->api;
	api->hipStreamSynchronize(temporal->kernels->stream);
	for (unsigned i = 0; i < temporal->level_count; ++i) {
		struct temporal_gpu_level *const level = &temporal->levels[i];
		release(api, &level->current);
		release(api, &level->previous);
		release(api, &level->flow);
		*level = (struct temporal_gpu_level){0};
	}
	for (size_t pass = 0; pass < sizeof temporal->history / sizeof *temporal->history; ++pass)
		release(api, &temporal->history[pass]);
	release(api, &temporal->warped);
	release(api, &temporal->scene_cut);
	temporal->level_count = 0;
	temporal->valid = false;
	temporal->pending = false;
}

/** @brief Returns a failed call's code from the function that makes the call. */
#define TRY(call) \
	do { \
		enum error_code const try_code_ = (call); \
		if (try_code_) \
			return try_code_; \
	} while (0)

/** @brief Allocates the buffers of a frame size, quality, grid spacing and pass count.
 *
 * @param temporal The state, without buffers; what it allocates before a failure stays for
 *                 release_buffers().
 * @param g        The geometry.
 * @param quality  The motion quality.
 * @param grid     The grid spacing.
 * @param passes   The passes.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
allocate_buffers (struct temporal_gpu   *temporal,
                  struct geometry const *g,
                  unsigned               quality,
                  unsigned               grid,
                  unsigned               passes,
                  struct error          *e)
{
	size_t const pixels = (size_t)g->width * g->height;
	TRY(allocate(temporal, &temporal->warped, pixels * 16, "allocate warped temporal history", e));
	TRY(allocate(temporal, &temporal->scene_cut, sizeof (uint32_t), "allocate temporal scene cut flag", e));
	for (unsigned pass = 0; pass < passes; ++pass)
		TRY(allocate(temporal, &temporal->history[pass], pixels * 12, "allocate per-pass neural history", e));
	unsigned width = g->width;
	unsigned height = g->valid_height;
	unsigned const count = 3 + quality;
	unsigned const step = 1u << grid;
	for (unsigned i = 0; i < count; ++i) {
		struct temporal_gpu_level *const level = &temporal->levels[temporal->level_count++];
		*level = (struct temporal_gpu_level){
			.image = {width, height},
			.grid  = {(width + step - 1) / step, (height + step - 1) / step},
		};
		size_t const luma = (size_t)width * height * 4;
		TRY(allocate(temporal, &level->current, luma, "allocate current luma pyramid", e));
		TRY(allocate(temporal, &level->previous, luma, "allocate previous luma pyramid", e));
		TRY(allocate(temporal, &level->flow,
		             (size_t)level->grid.width * level->grid.height * sizeof (struct temporal_flow),
		             "allocate flow pyramid", e));
		if (width < 8 || height < 8)
			break;
		width = (width + 1) / 2;
		height = (height + 1) / 2;
	}
	return ERROR_NONE;
}

#undef TRY

/** @brief Takes a frame's geometry and settings: keeps the buffers for the same sizes, and the
 *         history unless the picture moved, else allocates new ones without a history.
 *
 * @param temporal The state.
 * @param g        The geometry.
 * @param quality  The motion quality.
 * @param grid     The grid spacing.
 * @param passes   The passes.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
configure (struct temporal_gpu   *temporal,
           struct geometry const *g,
           unsigned               quality,
           unsigned               grid,
           unsigned               passes,
           struct error          *e)
{
	if (!g->width || !g->valid_height || g->valid_height > g->height || g->height > 2 * g->valid_height - 2)
		return error_fail(e, "invalid temporal geometry");
	// The levels and the histories are arrays of these bounds.
	if (quality > kMVecQuality || passes > kMaxPasses)
		return error_fail(e, "invalid temporal quality or passes");
	// Buffer sizes do not depend on the source size or the picture's placement; only a new
	// placement invalidates the history. The buffers exist while the levels do.
	struct geometry const *const was = &temporal->geometry;
	bool const same_size = temporal->level_count && g->width == was->width && g->height == was->height &&
	                       g->valid_height == was->valid_height && quality == temporal->quality &&
	                       grid == temporal->grid && passes == temporal->passes;
	bool const moved = g->x != was->x || g->y != was->y || g->fit_width != was->fit_width ||
	                   g->fit_height != was->fit_height;
	temporal->geometry = *g;
	if (same_size) {
		temporal->valid = temporal->valid && !moved;
		return ERROR_NONE;
	}
	release_buffers(temporal);
	temporal->quality = quality;
	temporal->grid = grid;
	temporal->passes = passes;
	enum error_code const code = allocate_buffers(temporal, g, quality, grid, passes, e);
	if (code)
		release_buffers(temporal);
	return code;
}

/** @brief The geometry of the history warp and the scene cut check.
 *
 * @param temporal The state, configured.
 * @return         The geometry.
 */
static struct temporal_warp
warp_geometry (struct temporal_gpu const *temporal)
{
	struct geometry const *const g = &temporal->geometry;
	return (struct temporal_warp){
		.image         = {g->width, g->valid_height},
		.grid          = temporal->levels[0].grid,
		.padded_height = g->height,
		.step          = 1u << temporal->grid,
		.x             = g->x,
		.y             = g->y,
		.fit_width     = g->fit_width,
		.fit_height    = g->fit_height,
	};
}

struct temporal_gpu
temporal_gpu (struct native_kernels const *kernels)
{
	return (struct temporal_gpu){.kernels = kernels};
}

void
temporal_gpu_fini (struct temporal_gpu *temporal)
{
	if (!temporal || !temporal->kernels)
		return;
	release_buffers(temporal);
	*temporal = (struct temporal_gpu){0};
}

void
temporal_gpu_reset (struct temporal_gpu *temporal)
{
	temporal->valid = false;
	temporal->pending = false;
	temporal->completed = 0;
}

enum error_code
temporal_gpu_cut_rejected (struct temporal_gpu *temporal,
                           bool                *cut,
                           struct error        *e)
{
	*cut = false;
	if (!temporal->valid)
		return ERROR_NONE;
	struct hip_api const *const api = temporal->kernels->api;
	int result = api->hipStreamSynchronize(temporal->kernels->stream);
	if (result)
		return hip_fail(e, api, result, "complete temporal scene cut");
	uint32_t word = 0;
	result = api->hipMemcpy(&word, temporal->scene_cut, sizeof word, 2);
	if (result)
		return hip_fail(e, api, result, "read temporal scene cut");
	*cut = word != 0;
	return ERROR_NONE;
}

/** @brief Returns a failed call's code from the function that makes the call. */
#define TRY(call) \
	do { \
		enum error_code const try_code_ = (call); \
		if (try_code_) \
			return try_code_; \
	} while (0)

enum error_code
temporal_gpu_begin (struct temporal_gpu   *temporal,
                    void                  *rgba,
                    struct geometry const *g,
                    unsigned               quality,
                    unsigned               grid,
                    unsigned               passes,
                    bool                   reset_history,
                    struct error          *e)
{
	if (!rgba || temporal->pending)
		return error_fail(e, "temporal begin without previous end or input");
	TRY(configure(temporal, g, quality, grid, passes, e));
	if (reset_history)
		temporal_gpu_reset(temporal);
	temporal->pending = true;
	temporal->completed = 0;

	struct native_kernels const *const kernels = temporal->kernels;
	struct temporal_gpu_level *const levels = temporal->levels;
	unsigned count = g->width * g->valid_height;
	void *luma_args[] = {&rgba, &levels[0].current, &count};
	TRY(native_kernels_launch(kernels, NATIVE_KERNEL_TEMPORAL_LUMA, count, luma_args, e));
	for (unsigned i = 1; i < temporal->level_count; ++i) {
		struct temporal_gpu_level *const source = &levels[i - 1];
		struct temporal_gpu_level *const target = &levels[i];
		void *args[] = {&source->current, &target->current, &source->image, &target->image};
		TRY(native_kernels_launch(kernels, NATIVE_KERNEL_TEMPORAL_REDUCE, target->image.width * target->image.height,
		                          args, e));
	}
	if (!temporal->valid)
		return ERROR_NONE;

	for (unsigned i = temporal->level_count; i-- > 0;) {
		struct temporal_gpu_level *const level = &levels[i];
		bool const coarse = i + 1 < temporal->level_count;
		void *coarse_flow = coarse ? levels[i + 1].flow : level->flow;
		struct temporal_search search = {
			.image       = level->image,
			.grid        = level->grid,
			.coarse_grid = coarse ? levels[i + 1].grid : level->grid,
			.step        = 1u << temporal->grid,
			.radius      = temporal->quality + 1,
			.patch       = temporal->quality == 2 ? 2u : 1u,
			.has_coarse  = coarse,
			.final_level = i == 0,
		};
		void *args[] = {&level->current, &level->previous, &coarse_flow, &level->flow, &search};
		TRY(native_kernels_launch(kernels, NATIVE_KERNEL_TEMPORAL_FLOW, level->grid.width * level->grid.height,
		                          args, e));
	}
	struct temporal_warp warp = warp_geometry(temporal);
	void *cut_args[] = {&levels[0].current, &levels[0].previous, &levels[0].flow, &temporal->scene_cut, &warp};
	return native_kernels_launch(kernels, NATIVE_KERNEL_TEMPORAL_CUT, 32, cut_args, e);
}

#undef TRY

enum error_code
temporal_gpu_history (struct temporal_gpu  *temporal,
                      unsigned              pass,
                      void                 *current_pass_rgba,
                      void                **history,
                      struct error         *e)
{
	if (!temporal->pending || pass != temporal->completed || pass >= temporal->passes || !current_pass_rgba)
		return error_fail(e, "temporal history pass order");
	if (!temporal->valid) {
		*history = nullptr;
		return ERROR_NONE;
	}
	struct temporal_gpu_level *const finest = &temporal->levels[0];
	struct temporal_warp warp = warp_geometry(temporal);
	void *args[] = {&finest->current, &finest->previous, &temporal->history[pass], &current_pass_rgba,
	                &finest->flow, &temporal->warped, &temporal->scene_cut, &warp};
	enum error_code const code = native_kernels_launch(temporal->kernels, NATIVE_KERNEL_TEMPORAL_WARP,
	                                                   temporal->geometry.width * temporal->geometry.height,
	                                                   args, e);
	if (code)
		return code;
	*history = temporal->warped;
	return ERROR_NONE;
}

enum error_code
temporal_gpu_target (struct temporal_gpu  *temporal,
                     unsigned              pass,
                     void                **target,
                     struct error         *e)
{
	if (!temporal->pending || pass != temporal->completed || pass >= temporal->passes)
		return error_fail(e, "temporal target pass order");
	++temporal->completed;
	*target = temporal->history[pass];
	return ERROR_NONE;
}

enum error_code
temporal_gpu_end (struct temporal_gpu *temporal,
                  struct error        *e)
{
	if (!temporal->pending || temporal->completed != temporal->passes)
		return error_fail(e, "incomplete temporal frame");
	for (unsigned i = 0; i < temporal->level_count; ++i) {
		struct temporal_gpu_level *const level = &temporal->levels[i];
		void *const current = level->current;
		level->current = level->previous;
		level->previous = current;
	}
	temporal->valid = true;
	temporal->pending = false;
	return ERROR_NONE;
}
