/** @file
 *
 * DLSSNR-AMD's Vulkan network, run by the project's runtime (vulkan_runtime.h) and recorded into a
 * caller's command buffers on a caller's device: dlsslopd's own device, or a game's device in the
 * layer. Vulkan calls go through the loader. network_recorder.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_NETWORK_RECORDER_H_
#define DLSSLOP_AMD_COMMON_NETWORK_RECORDER_H_

#include <stdint.h>

#include <vulkan/vulkan.h>

#include "error.h"
#include "vulkan_frame.h"
#include "vulkan_plan.h"
#include "vulkan_runtime.h"

/** @brief The most passes a frame chains; more count as this many. */
#define NETWORK_RECORDER_MAX_PASSES UINT32_C(16)

/** @brief The most that the model takes of each of the network's own controls; more counts as this
 *         much. */
#define NETWORK_RECORDER_MAX_CONTROL 2.0f

/** @brief The network on one device: its runtime, built for one frame shape at a time.
 *
 * network_recorder_init() makes one and network_recorder_fini() frees it; a zeroed one is empty.
 * Its runtime points at its device, so it is only ever initialized in place and never copied. The
 * device must have finished the recorder's work before the recorder is freed. external is as wide as
 * a pointer, which fills the padding.
 */
struct network_recorder {
	struct vulkan_runtime runtime;     //!< The network; none built while its device is nullptr.
	struct vulkan_plan    plan;        //!< The plan of an extent to build; width 0 for none.
	struct vulkan_device  device;      //!< The device, whose queue takes the runtime's build.
	uint64_t              storage;     //!< The device's storage buffers' limit.
	/** @brief When the next line about a wait that ran out may be logged, in nanoseconds of
	 *         CLOCK_MONOTONIC. */
	uint64_t              next_log;
	struct vulkan_paths   paths;       //!< Where the network's files are, in copies.
	char                 *copies;      //!< The paths' copies, one after another in one heap block.
	uintptr_t             external;    //!< Whether frames go through a caller's images, not buffers.
	/** @brief The history's last frame submitted; width 0 for none since a build. */
	struct vulkan_frame   last;
	struct vulkan_frame   recorded;    //!< The last frame recorded; width 0 for none since a build.
	struct vulkan_shape   shape;       //!< The shape that the runtime has.
	uint32_t              rejected[2]; //!< The last extent rejected.
	struct error          rejection;   //!< Why it was; empty for none.
};

/** @brief Makes a recorder on a device, with no network built.
 *
 * @param dest     Receives the recorder; empty on a failure.
 * @param device   The device; the recorder copies it.
 * @param paths    Where the network's files are; the recorder copies them.
 * @param external Frames go through a caller's images (struct vulkan_frame_images) instead of
 *                 buffers.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
network_recorder_init (struct network_recorder    *dest,
                       struct vulkan_device const *device,
                       struct vulkan_paths const  *paths,
                       bool                        external,
                       struct error               *e);

/** @brief Frees a recorder and empties it; the device must have finished its work.
 *
 * @param recorder The recorder, or nullptr.
 */
extern void
network_recorder_fini (struct network_recorder *recorder);

/** @brief Whether network_recorder_shape() would build or reshape for a frame.
 *
 * @param recorder The recorder.
 * @param frame    The frame.
 * @return         true if it would.
 */
extern bool
network_recorder_shape_differs (struct network_recorder const *recorder,
                                struct vulkan_frame const     *frame);

/** @brief Whether the runtime is built for a frame's extent, so that network_recorder_shape() at
 *         most reshapes it: a fraction of a millisecond, and no GPU work.
 *
 * @param recorder The recorder.
 * @param frame    The frame.
 * @return         true if it is.
 */
extern bool
network_recorder_has_extent (struct network_recorder const *recorder,
                             struct vulkan_frame const     *frame);

/** @brief Plans the network for a frame's extent, unless the runtime is built for it or the
 *         recorder holds its plan: milliseconds of work.
 *
 * An extent that the network does not take on the device is rejected, and then again without
 * planning.
 *
 * @param recorder The recorder.
 * @param frame    The frame.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE; ERROR_REJECTED for an extent the network does not take; ERROR_FAILED.
 */
extern enum error_code
network_recorder_plan (struct network_recorder   *recorder,
                       struct vulkan_frame const *frame,
                       struct error              *e);

/** @brief Builds the network for a frame's shape, unless it has it.
 *
 * A new extent takes seconds of work; another shape of the same extent a reshape that keeps the
 * weights and pipelines. In image mode it also binds images when their generation is not the one
 * bound, which a build does not: a bind alone keeps the motion history. The device must have
 * finished the recorder's work. A rejected extent keeps the network as it was, and a runtime that
 * fails to reshape is freed.
 *
 * @param recorder The recorder.
 * @param frame    The frame.
 * @param images   In image mode, the caller's images, or nullptr for those bound before.
 * @param changed  Receives whether it built, reshaped or bound.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE; ERROR_REJECTED for an extent the network does not take; ERROR_FAILED.
 */
extern enum error_code
network_recorder_shape (struct network_recorder          *recorder,
                        struct vulkan_frame const        *frame,
                        struct vulkan_frame_images const *images,
                        bool                             *changed,
                        struct error                     *e);

/** @brief Records one frame of the shape it has, from a proxy buffer through the network into an
 *         answer buffer, or the proxy unchanged when a wait of the frame runs out.
 *
 * The proxy and the answer hold width x height RGBA8 or RGBA16F. Transfers on the family's queue
 * wrote the proxy and read the last answer before, and do after. The frame submitted last must have
 * finished: when a wait of it ran out, this frame starts the network over.
 *
 * @param recorder The recorder.
 * @param cmd      The command buffer.
 * @param proxy    The proxy.
 * @param answer   The answer.
 * @param frame    The frame.
 * @param family   The queue family that the command buffer is for.
 * @param exported The proxy and the answer belong to VK_QUEUE_FAMILY_EXTERNAL between frames.
 * @param queries  Timestamp queries, or VK_NULL_HANDLE for none.
 * @param query    The query written once the frame is in the network's input; query + 1 is written
 *                 once the network is done.
 */
extern void
network_recorder_record_buffers (struct network_recorder   *recorder,
                                 VkCommandBuffer            cmd,
                                 VkBuffer                   proxy,
                                 VkBuffer                   answer,
                                 struct vulkan_frame const *frame,
                                 uint32_t                   family,
                                 bool                       exported,
                                 VkQueryPool                queries,
                                 uint32_t                   query);

/** @brief Records one frame of the shape it has in image mode, from the frame image that
 *         network_recorder_shape() bound through the network into its answer image, or the frame
 *         unchanged when a wait of the frame runs out.
 *
 * Before it, the caller's barriers make the frame readable by compute shaders in
 * SHADER_READ_ONLY_OPTIMAL and the answer writable by compute shaders and transfers in GENERAL;
 * after it, they take the answer's writes. The frame submitted last must have finished.
 *
 * @param recorder The recorder.
 * @param cmd      The command buffer.
 * @param frame    The frame.
 */
extern void
network_recorder_record_images (struct network_recorder   *recorder,
                                VkCommandBuffer            cmd,
                                struct vulkan_frame const *frame);

/** @brief Says that the frame recorded last was submitted, so that the next frame follows it in the
 *         motion history. A frame that is recorded and not submitted leaves the history as it was.
 *
 * @param recorder The recorder.
 */
extern void
network_recorder_submitted (struct network_recorder *recorder);

/** @brief Whether a wait of the frame submitted last ran out, so that its answer is its proxy; that
 *         frame must have finished.
 *
 * @param recorder The recorder.
 * @return         true if one ran out.
 */
extern bool
network_recorder_timed_out (struct network_recorder const *recorder);

#endif /* DLSSLOP_AMD_COMMON_NETWORK_RECORDER_H_ */
