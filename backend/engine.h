/** @file
 *
 * The network as serving, the offline mode and the self-test see it: one of the engines, chosen
 * once at startup (open.h), each compiled into what it runs (run_engine.inc). None is dispatched at
 * run time.
 *
 * Each engine is a module with the same operations under its own prefix: identity_engine.h,
 * vulkan_engine.h and hip_engine.h. Its struct has the members that every engine shares: slots,
 * times and tier. The operations, with ENGINE for the prefix:
 *
 * - ENGINE_describe_device() and ENGINE_describe_processing(): the device, as --device names it,
 *   for --diagnose, and what the network runs at, for the log;
 * - ENGINE_retier() and ENGINE_prepare(): another tier, and the build before the daemon reports
 *   itself ready;
 * - ENGINE_fits(), ENGINE_admit() and ENGINE_reshape(): whether a frame needs a build first, the
 *   rejection of a shape the engine does not take, and the build;
 * - ENGINE_import_into() and ENGINE_frames_of(): an offer imported into a slot, and a slot's
 *   frames;
 * - ENGINE_pin(), ENGINE_infer() and ENGINE_read_back(): the channel's frame slots locked for DMA,
 *   a frame, and memory copied to the host for a trace;
 * - ENGINE_self_test().
 *
 * Its traits are constants: ENGINE_NAME, the most passes a frame chains (ENGINE_MAX_PASSES),
 * whether a tier change rebuilds it, with the layer presenting its own frames meanwhile
 * (ENGINE_REBUILDS_FOR_TIER), and whether it runs a network (ENGINE_NEURAL): only the identity
 * engine of --test-identity does not.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_ENGINE_H_
#define DLSSLOP_AMD_BACKEND_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

#include "shm_protocol.h"

/** @brief The import slots: device-local frames that the layer exported (ShmTransportOffer), one
 *         proxy/answer pair per producer generation in as many slots, the oldest replaced first. */
#define ENGINE_SLOTS 4

/** @brief The bytes of an engine's description and its null: every one fits. */
#define ENGINE_DESCRIPTION_BYTES 320

/** @brief A request's frames: host memory, or the device-local pair in an import slot. */
struct engine_frames {
	uint8_t const *proxy;  //!< The proxy, or nullptr where the engine finds a slot's itself.
	uint8_t       *answer; //!< The answer, or nullptr where the engine finds a slot's itself.
	ptrdiff_t      slot;   //!< The import slot, or -1 for host memory.
};

/** @brief The generations held in the import slots: engine_slots_choose() picks a slot for an
 *         offer, engine_slots_hold() records the pair it imported and engine_slots_find() finds a
 *         request's. All zero, it holds none. */
struct engine_slots {
	/** @brief The slot that the next offer of a generation not held takes: the one filled longest ago. */
	ptrdiff_t oldest;
	size_t    bytes[ENGINE_SLOTS];      //!< The bytes that each slot's frames hold.
	uint32_t  generation[ENGINE_SLOTS]; //!< Each slot's generation; 0 for none.
};

/** @brief The GPU time of the latest frame's parts, in milliseconds. */
struct engine_times {
	float upload_ms;    //!< From the frame's start until it is in the network's input.
	float inference_ms; //!< The network.
	float readback_ms;  //!< From the network's end until the answer is out.
};

/** @brief The slot for an offer: the one that holds its generation, else the oldest.
 *
 * @param slots      The slots.
 * @param generation The offer's generation, not 0.
 * @return           The slot.
 */
static inline ptrdiff_t
engine_slots_choose (struct engine_slots *slots,
                     uint32_t             generation)
{
	for (ptrdiff_t slot = 0; slot < ENGINE_SLOTS; ++slot)
		if (slots->generation[slot] == generation)
			return slot;
	ptrdiff_t const oldest = slots->oldest;
	slots->oldest = (oldest + 1) % ENGINE_SLOTS;
	return oldest;
}

/** @brief Records the pair that an engine imported into a slot.
 *
 * @param slots The slots.
 * @param slot  The slot.
 * @param offer The offer.
 */
static inline void
engine_slots_hold (struct engine_slots            *slots,
                   ptrdiff_t                       slot,
                   struct ShmTransportOffer const *offer)
{
	slots->generation[slot] = offer->generation;
	slots->bytes[slot] = offer->size[1] < offer->size[0] ? offer->size[1] : offer->size[0];
}

/** @brief The slot of a generation whose frames hold some bytes.
 *
 * @param slots      The slots.
 * @param generation The generation, not 0.
 * @param bytes      The bytes.
 * @return           The slot, or -1 for none.
 */
static inline ptrdiff_t
engine_slots_find (struct engine_slots const *slots,
                   uint32_t                   generation,
                   size_t                     bytes)
{
	for (ptrdiff_t slot = 0; slot < ENGINE_SLOTS; ++slot)
		if (slots->generation[slot] == generation && slots->bytes[slot] >= bytes)
			return slot;
	return -1;
}

/** @brief Forgets the pairs held, which a rebuild dropped: the layer offers them again.
 *
 * @param slots The slots.
 */
static inline void
engine_slots_forget (struct engine_slots *slots)
{
	for (ptrdiff_t slot = 0; slot < ENGINE_SLOTS; ++slot) {
		slots->generation[slot] = 0;
		slots->bytes[slot] = 0;
	}
}

#endif /* DLSSLOP_AMD_BACKEND_ENGINE_H_ */
