/** @file
 *
 * GPU optical flow and the network's real temporal RGB input, by the linux_native module's kernels
 * on the network's stream. Flow is measured on consecutive original frames; each neural pass owns
 * its own previous output. temporal_gpu.c defines the functions.
 *
 * A frame is temporal_gpu_begin(), then for each pass temporal_gpu_history() right before the
 * pass's network evaluation and temporal_gpu_target() after it, then temporal_gpu_end().
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_TEMPORAL_GPU_H_
#define DLSSLOP_AMD_BACKEND_TEMPORAL_GPU_H_

#include <stdint.h>

#include "error.h"
#include "kernel_args.h"
#include "native_kernels.h"
#include "shm_protocol.h"

/** @brief The most levels of the luma pyramid: three, and one more for each step of quality. */
#define TEMPORAL_GPU_MAX_LEVELS (3 + kMVecQuality)

/** @brief One level of the luma pyramid and its flow grid. */
struct temporal_gpu_level {
	void                  *current;  //!< This frame's luma.
	void                  *previous; //!< The previous frame's.
	void                  *flow;     //!< The flow grid.
	struct temporal_extent image;    //!< The luma's extent.
	struct temporal_extent grid;     //!< The flow grid's.
};

/** @brief The pyramids, the passes' histories and the frame in progress.
 *
 * temporal_gpu() makes one, which allocates its buffers with its first frame, and
 * temporal_gpu_fini() frees it; a zeroed one is none.
 */
struct temporal_gpu {
	struct native_kernels const *kernels;                         //!< The kernels; nullptr for none.
	void                        *warped;                          //!< The warped history that a pass reads.
	void                        *scene_cut;                       //!< The scene cut flag, a device word.
	void                        *history[kMaxPasses];             //!< Each pass's last answer.
	struct temporal_gpu_level    levels[TEMPORAL_GPU_MAX_LEVELS]; //!< The pyramid, finest first.
	struct geometry              geometry;                        //!< The latest frame's geometry.
	unsigned                     quality;                         //!< The buffers' motion quality.
	unsigned                     grid;                            //!< Their spacing: 2^grid pixels a vector.
	unsigned                     passes;                          //!< Their passes.
	unsigned                     completed;                       //!< The frame's passes done so far.
	uint16_t                     level_count;                     //!< The levels allocated; 0 without buffers.
	bool                         valid;                           //!< Whether the previous frame is a history.
	bool                         pending;                         //!< Whether a frame is in progress.
};

/** @brief Makes a temporal state with no buffers.
 *
 * @param kernels The kernels, which must outlive it.
 * @return        The state.
 */
extern struct temporal_gpu
temporal_gpu (struct native_kernels const *kernels);

/** @brief Waits for the stream, frees the buffers and zeroes the state.
 *
 * @param temporal The state, or nullptr.
 */
extern void
temporal_gpu_fini (struct temporal_gpu *temporal);

/** @brief Forgets the history and any frame in progress.
 *
 * @param temporal The state.
 */
extern void
temporal_gpu_reset (struct temporal_gpu *temporal);

/** @brief Self-test only, as it waits for the stream: whether the latest temporal_gpu_begin()
 *         found a scene cut.
 *
 * @param temporal The state.
 * @param cut      Receives whether it did; false without a history.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
temporal_gpu_cut_rejected (struct temporal_gpu *temporal,
                           bool                *cut,
                           struct error        *e);

/** @brief Begins a frame: queues its luma pyramid and, with a history, its flow and scene cut
 *         check.
 *
 * The input is float4 raster data in the network's encoding; begin reads it before multi-pass
 * feedback overwrites it. A new pass count, quality, grid spacing or placement of the picture drops
 * the history; a new source size keeps it.
 *
 * @param temporal      The state.
 * @param rgba          The frame's encoded input.
 * @param g             Its geometry.
 * @param quality       The motion quality, 0..kMVecQuality.
 * @param grid          The grid spacing, as ShmMVecPixelSize() gives it.
 * @param passes        The frame's passes, 1..kMaxPasses.
 * @param reset_history Whether to drop the history first.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
temporal_gpu_begin (struct temporal_gpu   *temporal,
                    void                  *rgba,
                    struct geometry const *g,
                    unsigned               quality,
                    unsigned               grid,
                    unsigned               passes,
                    bool                   reset_history,
                    struct error          *e);

/** @brief Queues a pass's warped history, right before its network evaluation.
 *
 * @param temporal          The state.
 * @param pass              The pass.
 * @param current_pass_rgba The pass's input, which stands in where the history falls back.
 * @param history           Receives the warped history, valid until the next call, or nullptr
 *                          without a history.
 * @param e                 Receives the words for what stopped it, or nullptr.
 * @return                  ERROR_NONE, or ERROR_FAILED out of pass order.
 */
extern enum error_code
temporal_gpu_history (struct temporal_gpu  *temporal,
                      unsigned              pass,
                      void                 *current_pass_rgba,
                      void                **history,
                      struct error         *e);

/** @brief Where a pass's final RGB answer goes, after temporal_gpu_history() for the pass: the
 *         next frame's warp for the same pass reads it there.
 *
 * @param temporal The state.
 * @param pass     The pass.
 * @param target   Receives the buffer.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED out of pass order.
 */
extern enum error_code
temporal_gpu_target (struct temporal_gpu  *temporal,
                     unsigned              pass,
                     void                **target,
                     struct error         *e);

/** @brief Ends a frame whose passes are all done: it becomes the next frame's history.
 *
 * @param temporal The state.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED for an incomplete frame.
 */
extern enum error_code
temporal_gpu_end (struct temporal_gpu *temporal,
                  struct error        *e);

#endif /* DLSSLOP_AMD_BACKEND_TEMPORAL_GPU_H_ */
