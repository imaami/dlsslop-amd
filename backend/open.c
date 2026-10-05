/** @file
 *
 * Which engine runs the network: open.h.
 */
// SPDX-License-Identifier: MIT
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "hip.h"
#include "hip_engine.h"
#include "identity_engine.h"
#include "open.h"
#include "options.h"
#include "paths.h"
#include "serve.h"
#include "vulkan_engine.h"
#include "vulkan_network.h"
#include "vulkan_runtime.h"

/** @brief The engine that --backend selects, opened: open_engine() makes it. With neither, the
 *         identity engine runs. */
struct opened {
	struct hip_api       hip;    //!< The runtime for a HIP engine; not loaded while hipInit is nullptr.
	struct vulkan_engine vulkan; //!< The Vulkan engine, prepared; none while its network is nullptr.
};

/** @brief Opens the Vulkan engine, prepared. The pipeline cache is a convenience.
 *
 * @param dest Receives the engine; none on a failure.
 * @param o    The options.
 * @param tier The tier.
 * @param e    Receives the words for why not, or nullptr.
 * @return     ERROR_NONE, or the code of what failed.
 */
static enum error_code
open_vulkan (struct vulkan_engine *dest,
             struct options const *o,
             uint32_t              tier,
             struct error         *e)
{
	enum error_code code = paths_require_vulkan_model(o->vulkan_model, e);
	if (code)
		return code;
	struct paths_home const home = paths_home();
	char *cache;
	size_t cache_length;
	code = paths_vulkan_cache(&cache, &cache_length, &home, e);
	if (code)
		return code;
	// The network copies the paths, which borrow these strings.
	struct vulkan_paths const paths = {
		.model = o->vulkan_model,
		.shaders = o->shaders,
		.cache = cache ? cache : "",
		.model_length = o->vulkan_model_length,
		.shaders_length = o->shaders_length,
		.cache_length = cache_length,
	};
	struct vulkan_network *network;
	code = vulkan_network_create(&network, &paths, o->device, e);
	free(cache);
	cache = nullptr;
	if (code)
		return code;
	fprintf(stderr, "Vulkan network on %s\n", vulkan_network_device_name(network));
	struct vulkan_engine engine = vulkan_engine(network, tier);
	code = vulkan_engine_prepare(&engine, e);
	if (code) {
		vulkan_engine_fini(&engine);
		return code;
	}
	*dest = engine;
	return ERROR_NONE;
}

/** @brief Opens the engine that --backend selects.
 *
 * @param dest Receives the engine; a zeroed one for the identity engine; none on a failure.
 * @param o    The options; the HIP device selected is recorded in them.
 * @param tier The tier.
 * @param e    Receives the words for what failed, or nullptr.
 * @return     ERROR_NONE, or the code of what failed.
 */
static enum error_code
open_engine (struct opened  *dest,
             struct options *o,
             uint32_t        tier,
             struct error   *e)
{
	*dest = (struct opened){};
	if (o->test_identity)
		return ERROR_NONE;
	char const *const hip = options_hip_only(o);
	if (hip && o->backend == OPTIONS_BACKEND_AUTO) {
		fprintf(stderr, "%s needs the HIP network; using HIP\n", hip);
	} else if (o->backend != OPTIONS_BACKEND_HIP) {
		struct error why;
		enum error_code const code = open_vulkan(&dest->vulkan, o, tier, &why);
		if (!code)
			return ERROR_NONE;
		if (o->backend == OPTIONS_BACKEND_VULKAN) {
			if (e)
				*e = why;
			return code;
		}
		fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", why.what);
	}
	return hip_engine_open(&dest->hip, o, e);
}

enum error_code
open_run_engine (struct options       *o,
                 uint32_t              tier,
                 struct serving const *serving,
                 struct error         *e)
{
	struct opened opened;
	enum error_code code = open_engine(&opened, o, tier, e);
	if (code)
		return code;
	if (opened.vulkan.network) {
		code = run_vulkan_engine(o, &opened.vulkan, serving, e);
		vulkan_engine_fini(&opened.vulkan);
	} else if (opened.hip.hipInit) {
		struct hip_engine engine = hip_engine(o, tier, &opened.hip);
		code = run_hip_engine(o, &engine, serving, e);
		hip_engine_fini(&engine);
	} else {
		struct identity_engine engine = identity_engine(tier);
		code = run_identity_engine(o, &engine, serving, e);
	}
	return code;
}
