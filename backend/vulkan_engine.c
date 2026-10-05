/** @file
 *
 * The Vulkan engine: vulkan_engine.h.
 */
// SPDX-License-Identifier: MIT
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <vulkan/vulkan.h>

#include "engine.h"
#include "error.h"
#include "processing.h"
#include "shm_protocol.h"
#include "vulkan_engine.h"
#include "vulkan_frame.h"
#include "vulkan_network.h"

// A device's name, nulls included, is at most VK_MAX_PHYSICAL_DEVICE_NAME_SIZE bytes, so every
// description fits.
static_assert(sizeof "Vulkan device 4294967295 ()" - 1 + VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
              <= ENGINE_DESCRIPTION_BYTES);
static_assert(sizeof "Vulkan on  at each frame's extent" - 1 + VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
              <= ENGINE_DESCRIPTION_BYTES);

struct vulkan_engine
vulkan_engine (struct vulkan_network *network,
               uint32_t               tier)
{
	return (struct vulkan_engine){.network = network, .tier = tier};
}

void
vulkan_engine_fini (struct vulkan_engine *engine)
{
	if (engine) {
		vulkan_network_destroy(&engine->network);
		*engine = (struct vulkan_engine){};
	}
}

char const *
vulkan_engine_describe_device (struct vulkan_engine const *engine,
                               char                        dest[ENGINE_DESCRIPTION_BYTES])
{
	snprintf(dest, ENGINE_DESCRIPTION_BYTES, "Vulkan device %" PRIu32 " (%s)",
	         vulkan_network_device_index(engine->network), vulkan_network_device_name(engine->network));
	return dest;
}

char const *
vulkan_engine_describe_processing (struct vulkan_engine const *engine,
                                   char                        dest[ENGINE_DESCRIPTION_BYTES])
{
	snprintf(dest, ENGINE_DESCRIPTION_BYTES, "Vulkan on %s at each frame's extent",
	         vulkan_network_device_name(engine->network));
	return dest;
}

enum error_code
vulkan_engine_prepare (struct vulkan_engine *engine,
                       struct error         *e)
{
	struct processing_settings const settings = processing_settings();
	struct vulkan_frame const frame = processing_vulkan_frame(ShmNativeTier(engine->tier)->width,
	                                                          engine->tier, 1, &settings);
	return vulkan_network_shape(engine->network, &frame, e);
}

bool
vulkan_engine_fits (struct vulkan_engine const       *engine,
                    uint32_t                          width,
                    uint32_t                          height,
                    uint32_t                          passes,
                    struct processing_settings const *settings)
{
	struct vulkan_frame const frame = processing_vulkan_frame(width, height, passes, settings);
	return !vulkan_network_shape_differs(engine->network, &frame);
}

enum error_code
vulkan_engine_admit (struct vulkan_engine             *engine,
                     uint32_t                          width,
                     uint32_t                          height,
                     uint32_t                          passes,
                     struct processing_settings const *settings,
                     struct error                     *e)
{
	struct vulkan_frame const frame = processing_vulkan_frame(width, height, passes, settings);
	return vulkan_network_plan(engine->network, &frame, e);
}

enum error_code
vulkan_engine_reshape (struct vulkan_engine             *engine,
                       uint32_t                          width,
                       uint32_t                          height,
                       uint32_t                          passes,
                       struct processing_settings const *settings,
                       struct error                     *e)
{
	struct vulkan_frame const frame = processing_vulkan_frame(width, height, passes, settings);
	return vulkan_network_shape(engine->network, &frame, e);
}

bool
vulkan_engine_import_into (struct vulkan_engine           *engine,
                           ptrdiff_t                       slot,
                           struct ShmTransportOffer const *offer,
                           int                             fds[2])
{
	return vulkan_network_import(engine->network, slot, offer, fds);
}

enum error_code
vulkan_engine_infer (struct vulkan_engine             *engine,
                     struct engine_frames const       *io,
                     uint32_t                          width,
                     uint32_t                          height,
                     uint32_t                          passes,
                     struct processing_settings const *settings,
                     struct frame_trace               *trace,
                     struct error                     *e)
{
	struct vulkan_frame const frame = processing_vulkan_frame(width, height, passes, settings);
	enum error_code const code = vulkan_network_infer(engine->network, &frame, io->slot, io->proxy,
	                                                  io->answer, e);
	if (code)
		return code;
	struct vulkan_network_times const times = vulkan_network_frame_times(engine->network);
	engine->times = (struct engine_times){times.upload_ms, times.inference_ms, times.readback_ms};
	return ERROR_NONE;
}
