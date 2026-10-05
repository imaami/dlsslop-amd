/** @file
 *
 * libdlsslop-network.so: the in-layer network (network_module.h).
 */
// SPDX-License-Identifier: MIT
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "error.h"
#include "files.h"
#include "network_module.h"
#include "network_recorder.h"
#include "options.h"
#include "paths.h"
#include "processing.h"
#include "shm_protocol.h"
#include "vulkan_frame.h"
#include "vulkan_runtime.h"

/** @brief The network on a game's device.
 *
 * dlsslop_network_open() makes one and dlsslop_network_close() frees it. While building is set, a
 * thread of its own builds the recorder for target; it sets failed and error before it clears
 * building. error is read only after dlsslop_network_prepare() said REJECTED or FAILED, so the
 * functions that may write it get it directly.
 */
struct DlsslopNetwork {
	/** @brief The network, whose paths hold the model that dlsslopd's config file names; empty
	 *         when it could not be made, which fails the network. */
	struct network_recorder recorder;
	pthread_t               builder;  //!< The build's thread, while joinable.
	struct vulkan_frame     prepared; //!< The frame that dlsslop_network_prepare() readied last.
	struct vulkan_frame     target;   //!< The frame that the build is for.
	struct error            error;    //!< Why the network rejected a frame or failed.
	atomic_bool             building; //!< A build runs in the background.
	bool                    joinable; //!< builder is a thread to join.
	bool                    failed;   //!< The network cannot run: error says why.
};

uint64_t const dlsslop_network_interface = DLSSLOP_NETWORK_INTERFACE;

/** @brief Fails a network for good.
 *
 * @param n The network, whose error says why.
 * @return  DLSSLOP_NETWORK_FAILED.
 */
static enum dlsslop_network_state
fail (struct DlsslopNetwork *n)
{
	n->failed = true;
	return DLSSLOP_NETWORK_FAILED;
}

/** @brief Finds the network's SPIR-V beside the module, installed or in a source build.
 *
 * @param dest   Receives the SPIR-V's directory, which the caller frees; nullptr on a failure.
 * @param length Receives its length.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
module_shaders (char         **dest,
                size_t        *length,
                struct error  *e)
{
	// The module's directory, and the prefix two levels above it; both empty if dladdr() cannot
	// tell.
	char *module = nullptr;
	size_t directory = 0;
	Dl_info self;
	if (dladdr((void const *)&dlsslop_network_open, &self) && self.dli_fname) {
		size_t module_length;
		module = files_absolute(self.dli_fname, strlen(self.dli_fname), &module_length);
		if (!module) {
			*dest = nullptr;
			return error_fail(e, "out of memory");
		}
		directory = files_parent(module, module_length);
	}
	char const *const path = module ? module : "";
	size_t const prefix = files_parent(path, files_parent(path, directory));
	enum error_code const code = paths_vulkan_shaders_at(dest, length, path, prefix, path, directory, e);
	free(module);
	module = nullptr;
	return code;
}

/** @brief The game's device, with the next layer's physical-device functions that the build queries,
 *         looked up now: a lookup on the build's thread would go through the loader, which takes the
 *         loader's lock, and vkDestroyDevice holds that lock while the layer waits for the build to
 *         end.
 *
 * @param d The device, as the layer describes it.
 * @return  The device, for the recorder.
 */
static struct vulkan_device
network_device (struct dlsslop_network_device const *d)
{
	// One lookup per statement, in this order: an initializer's are evaluated in no set order.
	PFN_vkVoidFunction const queue_families = d->physical_dispatch(d->instance,
	                                                               "vkGetPhysicalDeviceQueueFamilyProperties");
	PFN_vkVoidFunction const properties = d->physical_dispatch(d->instance, "vkGetPhysicalDeviceProperties2");
	PFN_vkVoidFunction const format_properties = d->physical_dispatch(d->instance,
	                                                                  "vkGetPhysicalDeviceFormatProperties2");
	return (struct vulkan_device){
		.instance  = d->instance,
		.physical  = d->physical,
		.device    = d->device,
		.queue     = d->queue,
		.lock      = d->lock_queue,
		.unlock    = d->unlock_queue,
		.context   = d->context,
		.log       = d->log,
		.functions = {
			.queue_families    = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)queue_families,
			.properties        = (PFN_vkGetPhysicalDeviceProperties2)properties,
			.format_properties = (PFN_vkGetPhysicalDeviceFormatProperties2)format_properties,
		},
		.memory    = d->memory,
		.family    = d->family,
	};
}

/** @brief Makes a network's recorder on the game's device, with the model that dlsslopd's config
 *         file names, the SPIR-V beside the module and the user's pipeline cache.
 *
 * An unknown model fails it, but the recorder is made all the same, without a model.
 *
 * @param dest   Receives the recorder; empty if it could not be made.
 * @param device The game's device.
 * @param e      Receives the words for what stopped it, or nullptr; an unknown model's win over the
 *               rest.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
open_recorder (struct network_recorder             *dest,
               struct dlsslop_network_device const *device,
               struct error                        *e)
{
	struct paths_home const home = paths_home();
	char *model;
	size_t model_length;
	enum error_code const unknown = options_configured_vulkan_model(&model, &model_length, &home, e);
	// The rest is found and made all the same, its words in e only if the model's are not.
	struct error *const rest = unknown ? nullptr : e;
	char *shaders;
	size_t shaders_length;
	char *cache = nullptr;
	size_t cache_length = 0;
	enum error_code code = module_shaders(&shaders, &shaders_length, rest);
	if (!code)
		code = paths_vulkan_cache(&cache, &cache_length, &home, rest);
	if (code) {
		*dest = (struct network_recorder){};
	} else {
		struct vulkan_device const d = network_device(device);
		// The recorder copies the paths.
		struct vulkan_paths const paths = {
			.model          = model ? model : "",
			.shaders        = shaders,
			.cache          = cache,
			.model_length   = model_length,
			.shaders_length = shaders_length,
			.cache_length   = cache_length,
		};
		code = network_recorder_init(dest, &d, &paths, true, rest);
	}
	free(cache);
	cache = nullptr;
	free(shaders);
	shaders = nullptr;
	free(model);
	model = nullptr;
	return unknown ? unknown : code;
}

/** @brief Builds a network's recorder for its target, on a thread of its own.
 *
 * @param network The network.
 * @return        nullptr.
 */
static void *
build (void *network)
{
	struct DlsslopNetwork *const n = network;
	bool built;
	if (network_recorder_shape(&n->recorder, &n->target, nullptr, &built, &n->error))
		n->failed = true;
	atomic_store_explicit(&n->building, false, memory_order_release);
	return nullptr;
}

struct DlsslopNetwork *
dlsslop_network_open (struct dlsslop_network_device const *device)
{
	// A layer of another interface describes its device otherwise: an older one's begins with its
	// VkInstance, which never equals the interface.
	if (device->interface != DLSSLOP_NETWORK_INTERFACE)
		return nullptr;
	struct DlsslopNetwork *const n = malloc(sizeof *n);
	if (!n)
		return nullptr;
	n->prepared = (struct vulkan_frame)VULKAN_FRAME_DEFAULTS;
	n->target = (struct vulkan_frame)VULKAN_FRAME_DEFAULTS;
	n->error.what[0] = '\0';
	atomic_init(&n->building, false);
	n->joinable = false;
	n->failed = open_recorder(&n->recorder, device, &n->error) != ERROR_NONE;
	return n;
}

enum dlsslop_network_state
dlsslop_network_prepare (struct DlsslopNetwork               *n,
                         struct ShmHeader const              *channel,
                         struct dlsslop_network_images const *images)
{
	if (atomic_load_explicit(&n->building, memory_order_acquire))
		return DLSSLOP_NETWORK_BUILDING;
	if (n->joinable) {
		// A thread that pthread_create() started is joinable, and no other thread joins it.
		(void)pthread_join(n->builder, nullptr);
		n->joinable = false;
	}
	if (n->failed)
		return DLSSLOP_NETWORK_FAILED;
	// A channel of another protocol has its settings elsewhere.
	uint32_t const version = atomic_load_explicit(&channel->version, memory_order_relaxed);
	if (version != kShmVersion) {
		error_fail(&n->error, "the channel is of protocol v%u, the network of v%u", version, kShmVersion);
		return fail(n);
	}
	// The frame's format: the composition's proxy, RGBA8 or the float16 proxy.
	bool const fp16 = images->format == VK_FORMAT_R16G16B16A16_SFLOAT;
	if (!fp16 && images->format != VK_FORMAT_R8G8B8A8_UNORM) {
		error_fail(&n->error, "the network takes frames of RGBA8 or RGBA16F, not of format %u", images->format);
		return fail(n);
	}
	if (!images->generation) {
		error_fail(&n->error, "the composition's images have no generation");
		return fail(n);
	}
	struct processing_settings settings;
	if (processing_read(channel, &settings, &n->error))
		return DLSSLOP_NETWORK_REJECTED;
	settings.fp16 = fp16;
	uint32_t const asked = ShmPasses(channel);
	uint32_t const passes = NETWORK_RECORDER_MAX_PASSES < asked ? NETWORK_RECORDER_MAX_PASSES : asked;
	struct vulkan_frame const frame = processing_vulkan_frame(images->width, images->height, passes, &settings);
	// A frame of the network's extent: of its shape, or of another that it is reshaped for here,
	// between frames, with no GPU work, and in images that are bound here when they are new.
	if (network_recorder_has_extent(&n->recorder, &frame)) {
		struct vulkan_frame_images const surfaces = {
			.generation  = images->generation,
			.frame       = images->input_view,
			.answer      = images->answer,
			.answer_view = images->answer_view,
		};
		bool shaped;
		if (network_recorder_shape(&n->recorder, &frame, &surfaces, &shaped, &n->error))
			return fail(n);
		n->prepared = frame;
		return DLSSLOP_NETWORK_READY;
	}
	if (paths_require_vulkan_model(n->recorder.paths.model, &n->error))
		return fail(n);
	// A new extent is planned here, so that one the network does not take on the device is refused
	// without a build; the build's thread builds from that plan.
	enum error_code const code = network_recorder_plan(&n->recorder, &frame, &n->error);
	if (code == ERROR_FAILED)
		return fail(n);
	if (code)
		return DLSSLOP_NETWORK_REJECTED;
	n->target = frame;
	atomic_store_explicit(&n->building, true, memory_order_relaxed);
	int const started = pthread_create(&n->builder, nullptr, build, n);
	if (started) {
		atomic_store_explicit(&n->building, false, memory_order_relaxed);
		char buf[64];
		error_fail(&n->error, "start the network's build: %s", strerror_r(started, buf, sizeof buf));
		return fail(n);
	}
	n->joinable = true;
	return DLSSLOP_NETWORK_BUILDING;
}

enum dlsslop_network_state
dlsslop_network_record (struct DlsslopNetwork *n,
                        VkCommandBuffer        cmd)
{
	network_recorder_record_images(&n->recorder, cmd, &n->prepared);
	return DLSSLOP_NETWORK_READY;
}

void
dlsslop_network_submitted (struct DlsslopNetwork *n)
{
	network_recorder_submitted(&n->recorder);
}

char const *
dlsslop_network_error (struct DlsslopNetwork const *n)
{
	return n->error.what;
}

void
dlsslop_network_close (struct DlsslopNetwork *n)
{
	if (!n)
		return;
	// As in dlsslop_network_prepare().
	if (n->joinable)
		(void)pthread_join(n->builder, nullptr);
	network_recorder_fini(&n->recorder);
	free(n);
	n = nullptr;
}
