/** @file
 *
 * DIAGNOSTIC ONLY (--test-identity): each frame answered with itself, without a network, HIP or a
 * device. The engine's operations (engine.h) are the functions below.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_IDENTITY_ENGINE_H_
#define DLSSLOP_AMD_BACKEND_IDENTITY_ENGINE_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "engine.h"
#include "error.h"
#include "processing.h"
#include "shm_protocol.h"

struct frame_trace;
struct options;

/** @brief The engine's name. */
#define IDENTITY_ENGINE_NAME "identity"

/** @brief The engine's traits (engine.h). */
enum : uint32_t {
	IDENTITY_ENGINE_MAX_PASSES = kMaxPasses, //!< The most passes a frame chains.
	/** @brief Like the HIP engine, so that a GPU-less daemon follows a tier through its rebuild. */
	IDENTITY_ENGINE_REBUILDS_FOR_TIER = 1,
	IDENTITY_ENGINE_NEURAL = 0, //!< It runs no network: its answer is the frame.
};

/** @brief The engine: identity_engine() makes one; it holds nothing to free. */
struct identity_engine {
	struct engine_slots slots; //!< The import slots, which it never fills.
	struct engine_times times; //!< The latest frame's times: none.
	uint32_t            tier;  //!< The tier it follows.
};

/** @brief Makes the engine.
 *
 * @param tier The tier.
 * @return     The engine.
 */
static inline struct identity_engine
identity_engine (uint32_t tier)
{
	return (struct identity_engine){.tier = tier};
}

/** @brief The device, as --device names it: none.
 *
 * @param engine The engine.
 * @param dest   Unused.
 * @return       The description.
 */
static inline char const *
identity_engine_describe_device (struct identity_engine const *engine,
                                 char                          dest[ENGINE_DESCRIPTION_BYTES])
{
	return "no device (identity test)";
}

/** @brief What the network runs at: nothing.
 *
 * @param engine The engine.
 * @param dest   Unused.
 * @return       The description.
 */
static inline char const *
identity_engine_describe_processing (struct identity_engine const *engine,
                                     char                          dest[ENGINE_DESCRIPTION_BYTES])
{
	return "no processing";
}

/** @brief Follows another tier.
 *
 * @param engine The engine.
 * @param tier   The tier.
 * @param e      Unused.
 * @return       ERROR_NONE.
 */
static inline enum error_code
identity_engine_retier (struct identity_engine *engine,
                        uint32_t                tier,
                        struct error           *e)
{
	engine->tier = tier;
	return ERROR_NONE;
}

/** @brief Builds nothing.
 *
 * @param engine The engine.
 * @param e      Unused.
 * @return       ERROR_NONE.
 */
static inline enum error_code
identity_engine_prepare (struct identity_engine *engine,
                         struct error           *e)
{
	return ERROR_NONE;
}

/** @brief Whether a frame needs no build first: none ever does.
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @return         true.
 */
static inline bool
identity_engine_fits (struct identity_engine const     *engine,
                      uint32_t                          width,
                      uint32_t                          height,
                      uint32_t                          passes,
                      struct processing_settings const *settings)
{
	return true;
}

/** @brief Takes a shape: every one.
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @param e        Unused.
 * @return         ERROR_NONE.
 */
static inline enum error_code
identity_engine_admit (struct identity_engine           *engine,
                       uint32_t                          width,
                       uint32_t                          height,
                       uint32_t                          passes,
                       struct processing_settings const *settings,
                       struct error                     *e)
{
	return ERROR_NONE;
}

/** @brief Builds for a shape: nothing.
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @param e        Unused.
 * @return         ERROR_NONE.
 */
static inline enum error_code
identity_engine_reshape (struct identity_engine           *engine,
                         uint32_t                          width,
                         uint32_t                          height,
                         uint32_t                          passes,
                         struct processing_settings const *settings,
                         struct error                     *e)
{
	return ERROR_NONE;
}

/** @brief Imports an offer: never.
 *
 * @param engine The engine.
 * @param slot   The slot.
 * @param offer  The offer.
 * @param fds    The offered memory, which the caller closes.
 * @return       false.
 */
static inline bool
identity_engine_import_into (struct identity_engine         *engine,
                             ptrdiff_t                       slot,
                             struct ShmTransportOffer const *offer,
                             int                             fds[2])
{
	return false;
}

/** @brief A slot's frames: the slot alone.
 *
 * @param engine The engine.
 * @param slot   The slot.
 * @return       The frames.
 */
static inline struct engine_frames
identity_engine_frames_of (struct identity_engine const *engine,
                           ptrdiff_t                     slot)
{
	return (struct engine_frames){nullptr, nullptr, slot};
}

/** @brief Locks the channel's frame slots: nothing to lock.
 *
 * @param engine The engine.
 * @param input  The input slot.
 * @param output The output slot.
 * @param bytes  The bytes of each.
 */
static inline void
identity_engine_pin (struct identity_engine *engine,
                     uint8_t                *input,
                     uint8_t                *output,
                     size_t                  bytes)
{
}

/** @brief Answers a frame with itself.
 *
 * @param engine   The engine.
 * @param io       The frames, in host memory.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Unused.
 * @param settings The frame's settings: RGBA8, or RGBA16F with fp16.
 * @param trace    Unused.
 * @param e        Unused.
 * @return         ERROR_NONE.
 */
static inline enum error_code
identity_engine_infer (struct identity_engine           *engine,
                       struct engine_frames const       *io,
                       uint32_t                          width,
                       uint32_t                          height,
                       uint32_t                          passes,
                       struct processing_settings const *settings,
                       struct frame_trace               *trace,
                       struct error                     *e)
{
	memcpy(io->answer, io->proxy, (size_t)width * height * (settings->fp16 ? 8 : 4));
	return ERROR_NONE;
}

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
identity_engine_read_back (struct identity_engine *engine,
                           void                   *host,
                           void const             *source,
                           size_t                  bytes,
                           struct error           *e)
{
	return error_fail(e, "this network cannot trace");
}

/** @brief Runs the self-test: this engine has none.
 *
 * @param engine The engine.
 * @param o      Unused.
 * @param e      Receives the words, or nullptr.
 * @return       ERROR_FAILED.
 */
static inline enum error_code
identity_engine_self_test (struct identity_engine *engine,
                           struct options const   *o,
                           struct error           *e)
{
	return error_fail(e, "--self-test requires the real network");
}

#endif /* DLSSLOP_AMD_BACKEND_IDENTITY_ENGINE_H_ */
