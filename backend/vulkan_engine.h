/** @file
 *
 * DLSSNR-AMD's network on a Vulkan device of the daemon's own (vulkan_network.h). The engine's
 * operations (engine.h) are the functions below: vulkan_engine.c defines those that are not
 * trivial, and self_test.c the self-test.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_VULKAN_ENGINE_H_
#define DLSSLOP_AMD_BACKEND_VULKAN_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

#include "engine.h"
#include "error.h"
#include "processing.h"
#include "shm_protocol.h"
#include "vulkan_network.h"

struct frame_trace;
struct options;

/** @brief The engine's name. */
#define VULKAN_ENGINE_NAME "Vulkan"

/** @brief The engine's traits (engine.h). */
enum : uint32_t {
	VULKAN_ENGINE_MAX_PASSES = VULKAN_NETWORK_MAX_PASSES, //!< The most passes a frame chains.
	/** @brief The network sizes itself to each frame: a tier only changes the raster the layer
	 *         targets. */
	VULKAN_ENGINE_REBUILDS_FOR_TIER = 0,
	VULKAN_ENGINE_NEURAL = 1, //!< It runs the network.
};

static_assert(ENGINE_SLOTS == VULKAN_NETWORK_IMPORT_SLOTS, "an engine slot is not a network slot");

/** @brief The engine: vulkan_engine() makes one and vulkan_engine_fini() frees it. */
struct vulkan_engine {
	struct vulkan_network *network; //!< The network, which the engine owns; nullptr for none.
	struct engine_slots    slots;   //!< The generations in the network's import slots.
	struct engine_times    times;   //!< The latest frame's times.
	uint32_t               tier;    //!< The tier whose raster the layer targets.
};

/** @brief Makes the engine.
 *
 * @param network The network, which the engine takes over.
 * @param tier    The tier.
 * @return        The engine.
 */
extern struct vulkan_engine
vulkan_engine (struct vulkan_network *network,
               uint32_t               tier);

/** @brief Frees the engine's network and empties the engine.
 *
 * @param engine The engine, or nullptr.
 */
extern void
vulkan_engine_fini (struct vulkan_engine *engine);

/** @brief The device, as --device names it: "Vulkan device INDEX (NAME)".
 *
 * @param engine The engine.
 * @param dest   Receives the description.
 * @return       @a dest.
 */
extern char const *
vulkan_engine_describe_device (struct vulkan_engine const *engine,
                               char                        dest[ENGINE_DESCRIPTION_BYTES]);

/** @brief What the network runs at: "Vulkan on NAME at each frame's extent".
 *
 * @param engine The engine.
 * @param dest   Receives the description.
 * @return       @a dest.
 */
extern char const *
vulkan_engine_describe_processing (struct vulkan_engine const *engine,
                                   char                        dest[ENGINE_DESCRIPTION_BYTES]);

/** @brief Follows another tier: only the raster that the layer targets changes.
 *
 * @param engine The engine.
 * @param tier   The tier.
 * @param e      Unused.
 * @return       ERROR_NONE.
 */
static inline enum error_code
vulkan_engine_retier (struct vulkan_engine *engine,
                      uint32_t              tier,
                      struct error         *e)
{
	engine->tier = tier;
	return ERROR_NONE;
}

/** @brief Builds the network before the daemon reports itself ready, for the raster's usual
 *         frame: the tier's extent, one pass and the default settings.
 *
 * @param engine The engine.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, ERROR_REJECTED or ERROR_FAILED, as vulkan_network_shape() says.
 */
extern enum error_code
vulkan_engine_prepare (struct vulkan_engine *engine,
                       struct error         *e);

/** @brief Whether a frame needs no build first.
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @return         true if the network has the frame's shape.
 */
extern bool
vulkan_engine_fits (struct vulkan_engine const       *engine,
                    uint32_t                          width,
                    uint32_t                          height,
                    uint32_t                          passes,
                    struct processing_settings const *settings);

/** @brief Rejects a shape that the network does not take, before vulkan_engine_reshape().
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, ERROR_REJECTED or ERROR_FAILED, as vulkan_network_plan() says.
 */
extern enum error_code
vulkan_engine_admit (struct vulkan_engine             *engine,
                     uint32_t                          width,
                     uint32_t                          height,
                     uint32_t                          passes,
                     struct processing_settings const *settings,
                     struct error                     *e);

/** @brief Builds the network for a frame's shape.
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, ERROR_REJECTED or ERROR_FAILED, as vulkan_network_shape() says.
 */
extern enum error_code
vulkan_engine_reshape (struct vulkan_engine             *engine,
                       uint32_t                          width,
                       uint32_t                          height,
                       uint32_t                          passes,
                       struct processing_settings const *settings,
                       struct error                     *e);

/** @brief Imports an offered pair into a slot (vulkan_network_import()).
 *
 * @param engine The engine.
 * @param slot   The slot.
 * @param offer  The offer.
 * @param fds    The offered memory; each descriptor imported is the network's and set to -1, and
 *               the caller closes the rest.
 * @return       true if the pair is in the slot.
 */
extern bool
vulkan_engine_import_into (struct vulkan_engine           *engine,
                           ptrdiff_t                       slot,
                           struct ShmTransportOffer const *offer,
                           int                             fds[2]);

/** @brief A slot's frames: the slot alone, whose pair the network finds itself.
 *
 * @param engine The engine.
 * @param slot   The slot.
 * @return       The frames.
 */
static inline struct engine_frames
vulkan_engine_frames_of (struct vulkan_engine const *engine,
                         ptrdiff_t                   slot)
{
	return (struct engine_frames){nullptr, nullptr, slot};
}

/** @brief Locks the channel's frame slots: the network copies them itself.
 *
 * @param engine The engine.
 * @param input  The input slot.
 * @param output The output slot.
 * @param bytes  The bytes of each.
 */
static inline void
vulkan_engine_pin (struct vulkan_engine *engine,
                   uint8_t              *input,
                   uint8_t              *output,
                   size_t                bytes)
{
}

/** @brief Runs a frame through the network and keeps its times.
 *
 * @param engine   The engine.
 * @param io       The frames: host memory, or an import slot's pair.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings: RGBA8 frames, or RGBA16F with fp16.
 * @param trace    Unused: this engine cannot trace.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, ERROR_REJECTED, ERROR_DROPPED or ERROR_FAILED, as
 *                 vulkan_network_infer() says.
 */
extern enum error_code
vulkan_engine_infer (struct vulkan_engine             *engine,
                     struct engine_frames const       *io,
                     uint32_t                          width,
                     uint32_t                          height,
                     uint32_t                          passes,
                     struct processing_settings const *settings,
                     struct frame_trace               *trace,
                     struct error                     *e);

/** @brief Copies memory to the host for a trace: this engine cannot trace.
 *
 * @param engine The engine.
 * @param host   Unused.
 * @param source Unused.
 * @param bytes  Unused.
 * @param e      Receives the words, or nullptr.
 * @return       ERROR_FAILED.
 */
static inline enum error_code
vulkan_engine_read_back (struct vulkan_engine *engine,
                         void                 *host,
                         void const           *source,
                         size_t                bytes,
                         struct error         *e)
{
	return error_fail(e, "this network cannot trace");
}

/** @brief The network on a deterministic gradient at the tier's raster: finite, repeatable and
 *         changed.
 *
 * A run whose wait ran out under other GPU work is dropped, as serving answers such a frame as
 * failed, up to o->self_test_drops of them; the runs it keeps must all be equal. self_test.c
 * defines it.
 *
 * @param engine The engine, prepared.
 * @param o      The options: the passes, the runs, the drops and the output file.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
extern enum error_code
vulkan_engine_self_test (struct vulkan_engine *engine,
                         struct options const *o,
                         struct error         *e);

#endif /* DLSSLOP_AMD_BACKEND_VULKAN_ENGINE_H_ */
