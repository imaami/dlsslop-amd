/** @file
 *
 * The HIP network, a port of lmxxf's, and the daemon's own codec, tuning, color and motion kernels,
 * on a gfx1201 device. The engine's operations (engine.h) are the functions below: hip_engine.c
 * defines those that are not trivial, and self_test.c the self-test.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_HIP_ENGINE_H_
#define DLSSLOP_AMD_BACKEND_HIP_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

#include "codec_gpu.h"
#include "engine.h"
#include "error.h"
#include "hip.h"
#include "hip_network.h"
#include "native_kernels.h"
#include "processing.h"
#include "shm_protocol.h"
#include "temporal_gpu.h"

struct frame_trace;
struct options;

/** @brief The engine's name. */
#define HIP_ENGINE_NAME "HIP"

/** @brief The engine's traits (engine.h). */
enum : uint32_t {
	HIP_ENGINE_MAX_PASSES = kMaxPasses, //!< The most passes a frame chains.
	HIP_ENGINE_REBUILDS_FOR_TIER = 1,   //!< A tier is a raster that the network is built for.
};

/** @brief The device-local pair in an import slot; nullptr where it has none. */
struct hip_engine_import {
	void *memory[2]; //!< The proxy's and the answer's external memory.
	void *frame[2];  //!< The proxy and the answer, mapped.
};

/** @brief The engine: hip_engine() makes one, hip_engine_prepare() builds it for its tier and
 *         hip_engine_fini() frees it.
 *
 * The network binds addresses inside the engine, so the engine is never copied once prepared.
 */
struct hip_engine {
	struct options const      *options;                //!< The options, which outlive the engine.
	/** @brief The one stream of the network and of every kernel and copy of the daemon's, from the
	 *         first hip_engine_prepare() on; nullptr before. */
	void                      *stream;
	void                      *device_input;           //!< The encoded frame, unchanged until the next one.
	void                      *device_feedback;        //!< Later passes' input, allocated for multi-pass.
	void                      *device_output;          //!< The network's answer.
	void                      *device_scratch;         //!< Tuning or colour: the other stage output.
	void                      *answer;                 //!< The latest frame's final network answer.
	float                     *input;                  //!< The host copy of a pass's input, or nullptr.
	float                     *neural;                 //!< The host copy of a pass's answer, or nullptr.
	size_t                     input_count;            //!< The floats of input; 0 without it.
	size_t                     neural_count;           //!< The floats of neural; 0 without it.
	/** @brief Stream events: frame start, uploaded, evaluated, answered. Timing never stalls the
	 *         stream; the intervals are read once the answer is complete. */
	void                      *marks[4];
	struct hip_engine_import   imported[ENGINE_SLOTS]; //!< The pair in each import slot.
	struct hip_api             api;                    //!< The runtime.
	/** @brief The network's code objects and weights, from the first hip_engine_prepare() on; not
	 *         loaded while its api is nullptr. Every tier's plan reads the same weights (hip-plan
	 *         checks it), and they differ only with --performance, which is fixed for the engine's
	 *         life. */
	struct hip_model           model;
	struct hip_network         network;                //!< The tier's network; none while zeroed.
	struct native_kernels      kernels;                //!< The daemon's kernels; none while zeroed.
	struct codec_gpu           gpu_codec;              //!< The GPU codec; none while zeroed, as with --cpu-codec.
	struct temporal_gpu        temporal;               //!< The motion history; none while zeroed.
	struct engine_slots        slots;                  //!< The generations in the import slots.
	struct processing_settings previous_settings;      //!< The latest completed frame's settings.
	struct engine_times        times;                  //!< The latest frame's times.
	uint32_t                   tier;                   //!< The tier the network is built for.
	uint64_t                   warned_conditioning;    //!< 1 once the ignored conditioning was mentioned, else 0.
};

/** @brief Makes the engine, with nothing built.
 *
 * @param o    The options, which must outlive the engine.
 * @param tier The tier.
 * @param api  The runtime, from hip_engine_open().
 * @return     The engine.
 */
extern struct hip_engine
hip_engine (struct options const *o,
            uint32_t              tier,
            struct hip_api const *api);

/** @brief Frees what the engine holds and empties it.
 *
 * @param engine The engine, or nullptr.
 */
extern void
hip_engine_fini (struct hip_engine *engine);

/** @brief Loads the HIP runtime for a HIP engine, with the gfx1201 device that o->device names, or
 *         the first, selected and recorded there; every visible device is listed on the way. A
 *         failure after the runtime's first call leaves it loaded.
 *
 * @param dest Receives the runtime, which stays loaded for the process; unchanged on a failure.
 * @param o    The options, whose device receives the device selected.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_engine_open (struct hip_api *dest,
                 struct options *o,
                 struct error   *e);

/** @brief The device, as --device names it: "device INDEX".
 *
 * @param engine The engine.
 * @param dest   Receives the description.
 * @return       @a dest.
 */
extern char const *
hip_engine_describe_device (struct hip_engine const *engine,
                            char                     dest[ENGINE_DESCRIPTION_BYTES]);

/** @brief What the network runs at: "processing=WIDTHxHEIGHT", the tier's padded raster.
 *
 * @param engine The engine.
 * @param dest   Receives the description.
 * @return       @a dest.
 */
extern char const *
hip_engine_describe_processing (struct hip_engine const *engine,
                                char                     dest[ENGINE_DESCRIPTION_BYTES]);

/** @brief Builds the network for the tier: its plan, the model on the first call, the network,
 *         the kernels and the buffers; then one evaluation, so that a kernel that cannot launch
 *         fails here, and with --self-test the kernels' checks.
 *
 * @param engine The engine; on a failure it keeps what it made, which hip_engine_fini() frees.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
extern enum error_code
hip_engine_prepare (struct hip_engine *engine,
                    struct error      *e);

/** @brief Builds the network for another tier, between frames, with the loaded model. The layer
 *         offers its device-local frames again.
 *
 * @param engine The engine.
 * @param tier   The tier.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
extern enum error_code
hip_engine_retier (struct hip_engine *engine,
                   uint32_t           tier,
                   struct error      *e);

/** @brief Whether a frame needs no build first: none does, as the tier's network takes every
 *         frame.
 *
 * @param engine   The engine.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes.
 * @param settings Its settings.
 * @return         true.
 */
static inline bool
hip_engine_fits (struct hip_engine const          *engine,
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
hip_engine_admit (struct hip_engine                *engine,
                  uint32_t                          width,
                  uint32_t                          height,
                  uint32_t                          passes,
                  struct processing_settings const *settings,
                  struct error                     *e)
{
	return ERROR_NONE;
}

/** @brief Builds for a shape: nothing, as no frame needs it.
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
hip_engine_reshape (struct hip_engine                *engine,
                    uint32_t                          width,
                    uint32_t                          height,
                    uint32_t                          passes,
                    struct processing_settings const *settings,
                    struct error                     *e)
{
	return ERROR_NONE;
}

/** @brief Imports an offered pair into a slot, releasing the pair it held once the new one is in.
 *
 * Only the GPU codec reads device-local frames.
 *
 * @param engine The engine.
 * @param slot   The slot.
 * @param offer  The offer.
 * @param fds    The offered memory; each descriptor imported is the runtime's and set to -1, and
 *               the caller closes the rest.
 * @return       true if the pair is in the slot.
 */
extern bool
hip_engine_import_into (struct hip_engine              *engine,
                        ptrdiff_t                       slot,
                        struct ShmTransportOffer const *offer,
                        int                             fds[2]);

/** @brief A slot's frames: its imported pair.
 *
 * @param engine The engine.
 * @param slot   The slot.
 * @return       The frames.
 */
static inline struct engine_frames
hip_engine_frames_of (struct hip_engine const *engine,
                      ptrdiff_t                slot)
{
	struct hip_engine_import const *pair = &engine->imported[slot];
	return (struct engine_frames){pair->frame[0], pair->frame[1], slot};
}

/** @brief Serving: page-locks the channel's frame slots for the GPU codec (codec_gpu_pin()).
 *
 * @param engine The engine.
 * @param input  The input slot.
 * @param output The output slot.
 * @param bytes  The bytes of each to lock.
 */
static inline void
hip_engine_pin (struct hip_engine *engine,
                uint8_t           *input,
                uint8_t           *output,
                size_t             bytes)
{
	if (engine->gpu_codec.kernels)
		codec_gpu_pin(&engine->gpu_codec, input, output, bytes);
}

/** @brief Runs a frame through the codec, the network and the post-processing stages.
 *
 * @param engine   The engine, prepared.
 * @param io       The frames: host memory, or an import slot's device frames.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes, 1..HIP_ENGINE_MAX_PASSES.
 * @param settings Its settings: RGBA8 frames, or RGBA16F with fp16.
 * @param trace    The trace that receives each pass's stages, or nullptr.
 * @param verify   Whether to check the GPU codec of host frames against the CPU reference inside
 *                 the timed frame.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, ERROR_REJECTED for a frame at fault, or ERROR_FAILED.
 */
extern enum error_code
hip_engine_frame (struct hip_engine                *engine,
                  struct engine_frames const       *io,
                  uint32_t                          width,
                  uint32_t                          height,
                  uint32_t                          passes,
                  struct processing_settings const *settings,
                  struct frame_trace               *trace,
                  bool                              verify,
                  struct error                     *e);

/** @brief Runs a frame: hip_engine_frame() without the check against the CPU reference.
 *
 * @param engine   The engine, prepared.
 * @param io       The frames: host memory, or an import slot's device frames.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param passes   Its passes, 1..HIP_ENGINE_MAX_PASSES.
 * @param settings Its settings: RGBA8 frames, or RGBA16F with fp16.
 * @param trace    The trace that receives each pass's stages, or nullptr.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, ERROR_REJECTED for a frame at fault, or ERROR_FAILED.
 */
static inline enum error_code
hip_engine_infer (struct hip_engine                *engine,
                  struct engine_frames const       *io,
                  uint32_t                          width,
                  uint32_t                          height,
                  uint32_t                          passes,
                  struct processing_settings const *settings,
                  struct frame_trace               *trace,
                  struct error                     *e)
{
	return hip_engine_frame(engine, io, width, height, passes, settings, trace, false, e);
}

/** @brief Reads the latest frame's raw network answer into the engine's neural, outside its
 *         timing; with the CPU codec it is there already.
 *
 * @param engine The engine, which has the host copies (the CPU codec or --self-test).
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_engine_read_raw (struct hip_engine *engine,
                     struct error      *e);

/** @brief Copies host or device memory, such as an imported frame, to the host for a trace.
 *
 * @param engine The engine.
 * @param host   Receives the bytes.
 * @param source The memory.
 * @param bytes  The bytes.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_engine_read_back (struct hip_engine *engine,
                      void              *host,
                      void const        *source,
                      size_t             bytes,
                      struct error      *e);

/** @brief The real network on a deterministic 640x360 gradient: finite, bit-exact over the runs
 *         and changed, with the first run's codec checked against the CPU reference.
 *
 * self_test.c defines it.
 *
 * @param engine The engine, prepared.
 * @param o      The options: the passes, the runs and the output file.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
extern enum error_code
hip_engine_self_test (struct hip_engine    *engine,
                      struct options const *o,
                      struct error         *e);

#endif /* DLSSLOP_AMD_BACKEND_HIP_ENGINE_H_ */
