/** @file
 *
 * The HIP network run by dlsslopd's own host code: hip_network.h.
 */
// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "files.h"
#include "hip_network.h"

// A kernel takes each argument's bytes from the start of its 8-byte value.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && sizeof (void *) == sizeof (uint64_t),
              "an argument's bytes start its 8-byte value");

/** @brief A launch, bound: its kernel and its arguments' addresses. */
struct hip_network_launch {
	void            *function; //!< The kernel's function.
	void           **argv[2];  //!< The arguments' addresses for the first frame, and for later ones.
	uint32_t         grid;     //!< The groups.
	uint16_t         threads;  //!< The threads of a group.
	enum hip_kernel  kernel;   //!< The kernel, for the words of a failed launch.
};

/** @brief Reads a weight, packs it and uploads it into a model.
 *
 * @param model         The model, whose weights have room for it.
 * @param assets        The directory of the weights; it need not be null-terminated.
 * @param assets_length The length of its path.
 * @param spec          The weight.
 * @param file          Receives the weight's file, which the caller frees.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
upload (struct hip_model             *model,
        char const                   *assets,
        size_t                        assets_length,
        struct hip_weight_spec const *spec,
        struct hip_weight_file       *file,
        struct error                 *e)
{
	enum error_code const code = hip_weights_load(file, assets, assets_length, spec, e);
	if (code)
		return code;
	struct hip_api const *const api = model->api;
	size_t const bytes = file->count * sizeof (float);
	void *weight = nullptr;
	int result = api->hipMalloc(&weight, bytes);
	if (result)
		return hip_fail(e, api, result, "allocate weight %s", file->path);
	model->weights[model->count++] = weight;
	model->bytes += bytes;
	result = api->hipMemcpy(weight, file->values, bytes, 1);
	if (result)
		return hip_fail(e, api, result, "upload weight %s", file->path);
	return ERROR_NONE;
}

/** @brief Loads a model's modules and kernels, then its weights.
 *
 * @param model          The model, holding its runtime; what it loads before a failure stays
 *                       for hip_model_fini().
 * @param modules        The directory of the modules; it need not be null-terminated.
 * @param modules_length The length of its path.
 * @param assets         The directory of the weights; it need not be null-terminated.
 * @param assets_length  The length of its path.
 * @param weights        The weights.
 * @param count          Their number.
 * @param e              Receives the words for what stopped it, or nullptr.
 * @return               ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
load (struct hip_model             *model,
      char const                   *modules,
      size_t                        modules_length,
      char const                   *assets,
      size_t                        assets_length,
      struct hip_weight_spec const *weights,
      size_t                        count,
      struct error                 *e)
{
	struct hip_api const *const api = model->api;
	for (size_t m = 0; m < HIP_MODULE_COUNT; ++m) {
		struct hip_module_file const *const file = &HIP_PLAN_MODULE_FILES[m];
		char *path = files_join(modules, modules_length, file->name, file->length, nullptr);
		if (!path)
			return error_fail(e, "out of memory");
		enum error_code const code = hip_load_module(api, path, &model->modules[m], e);
		free(path);
		path = nullptr;
		if (code)
			return code;
	}
	for (size_t k = 0; k < HIP_KERNEL_COUNT; ++k) {
		struct hip_kernel_info const *const kernel = &HIP_PLAN_KERNELS[k];
		void *function = nullptr;
		int const result = api->hipModuleGetFunction(&function, model->modules[kernel->module],
		                                             kernel->name);
		if (result)
			return hip_fail(e, api, result, "%s", kernel->name);
		model->functions[k] = function;
	}

	model->weights = malloc(count * sizeof *model->weights);
	if (!model->weights)
		return error_fail(e, "out of memory");
	for (size_t w = 0; w < count; ++w) {
		struct hip_weight_file file;
		enum error_code const code = upload(model, assets, assets_length, &weights[w], &file, e);
		hip_weight_file_fini(&file);
		if (code)
			return code;
	}
	return ERROR_NONE;
}

enum error_code
hip_model_init (struct hip_model             *dest,
                struct hip_api const         *api,
                char const                   *modules,
                size_t                        modules_length,
                char const                   *assets,
                size_t                        assets_length,
                struct hip_weight_spec const *weights,
                size_t                        count,
                struct error                 *e)
{
	*dest = (struct hip_model){.api = api};
	enum error_code const code = load(dest, modules, modules_length, assets, assets_length, weights, count, e);
	if (code)
		hip_model_fini(dest);
	return code;
}

void
hip_model_fini (struct hip_model *model)
{
	if (!model)
		return;
	if (model->api) {
		for (size_t w = 0; w < model->count; ++w)
			model->api->hipFree(model->weights[w]);
		for (size_t m = 0; m < HIP_MODULE_COUNT; ++m) {
			if (model->modules[m]) {
				model->api->hipModuleUnload(model->modules[m]);
				model->modules[m] = nullptr;
			}
		}
	}
	free(model->weights);
	model->weights = nullptr;
	*model = (struct hip_model){0};
}

/** @brief Uploads the ViT's gather map and its inverse into a network.
 *
 * @param network The network.
 * @param tokens  The ViT's tokens.
 * @param map     Room for the host's copy of a map: tokens * 1024 indices.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
upload_gather_maps (struct hip_network *network,
                    unsigned            tokens,
                    uint32_t           *map,
                    struct error       *e)
{
	struct hip_api const *const api = network->api;
	size_t const bytes = (size_t)tokens * 1024 * sizeof *map;
	for (unsigned inverse = 0; inverse < 2; ++inverse) {
		hip_plan_gather_map(map, tokens, inverse);
		void *buffer = nullptr;
		int result = api->hipMalloc(&buffer, bytes);
		if (result)
			return hip_fail(e, api, result, "allocate gather map");
		network->gather[inverse] = buffer;
		network->bytes += bytes;
		result = api->hipMemcpy(buffer, map, bytes, 1);
		if (result)
			return hip_fail(e, api, result, "upload gather map");
	}
	return ERROR_NONE;
}

/** @brief Allocates a network's buffers and maps and binds its launches.
 *
 * @param network   The network, holding its runtime, model and stream; what it makes before a
 *                  failure stays for hip_network_fini().
 * @param plan      The plan.
 * @param placement Its placement.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
build (struct hip_network         *network,
       struct hip_plan const      *plan,
       struct hip_placement const *placement,
       struct error               *e)
{
	struct hip_api const *const api = network->api;
	size_t const buffer_count = placement->buffer_count;
	network->buffers = malloc(buffer_count * sizeof *network->buffers);
	if (!network->buffers)
		return error_fail(e, "out of memory");
	for (size_t b = 0; b < buffer_count; ++b) {
		size_t const bytes = placement->buffers[b];
		void *buffer = nullptr;
		int const result = api->hipMalloc(&buffer, bytes);
		if (result)
			return hip_fail(e, api, result, "allocate network buffer");
		network->buffers[network->buffer_count++] = buffer;
		network->bytes += bytes;
	}

	uint32_t *map = malloc((size_t)plan->tokens * 1024 * sizeof *map);
	if (!map)
		return error_fail(e, "out of memory");
	enum error_code const code = upload_gather_maps(network, plan->tokens, map, e);
	free(map);
	map = nullptr;
	if (code)
		return code;

	size_t const launch_count = plan->launch_count;
	size_t arguments = 0;
	for (size_t l = 0; l < launch_count; ++l)
		arguments += plan->launches[l].count;
	// A null argument keeps its zero.
	network->values = calloc(2 * arguments, sizeof *network->values);
	network->argv = malloc(2 * arguments * sizeof *network->argv);
	network->launches = malloc(launch_count * sizeof *network->launches);
	if (!network->values || !network->argv || !network->launches)
		return error_fail(e, "out of memory");
	network->launch_count = launch_count;

	// The frame's arguments, in the order of their kinds.
	static_assert(HIP_ARG_HISTORY == HIP_ARG_RGBA + 1 && HIP_ARG_TEMPORAL == HIP_ARG_RGBA + 2 &&
	              HIP_ARG_OUTPUT == HIP_ARG_RGBA + 3, "the frame's arguments follow HIP_ARG_RGBA");
	void *const frame[] = {&network->frame.rgba, &network->frame.history, &network->frame.temporal,
	                       &network->frame.output};
	size_t at = 0;
	for (size_t l = 0; l < launch_count; ++l) {
		struct hip_launch const *const launch = &plan->launches[l];
		network->launches[l] = (struct hip_network_launch){
			.function = network->model->functions[launch->kernel],
			.argv     = {&network->argv[at], &network->argv[arguments + at]},
			.grid     = launch->grid,
			.threads  = HIP_PLAN_KERNELS[launch->kernel].threads,
			.kernel   = launch->kernel,
		};
		for (unsigned later = 0; later < 2; ++later) {
			uint16_t const *const buffer_of = later ? placement->later : placement->first;
			for (unsigned i = 0; i < launch->count; ++i) {
				struct hip_arg const arg = launch->args[i];
				size_t const slot = later * arguments + at + i;
				uint64_t *const value = &network->values[slot];
				network->argv[slot] = value;
				switch (arg.kind) {
				case HIP_ARG_U32:
				case HIP_ARG_F32:
					*value = arg.value;
					break;
				case HIP_ARG_NULL:
					break;
				case HIP_ARG_TENSOR:
					memcpy(value, &network->buffers[buffer_of[arg.value]], sizeof *value);
					break;
				case HIP_ARG_WEIGHT:
					memcpy(value, &network->model->weights[arg.value], sizeof *value);
					break;
				case HIP_ARG_GATHER:
					memcpy(value, &network->gather[arg.value], sizeof *value);
					break;
				case HIP_ARG_RGBA:
				case HIP_ARG_HISTORY:
				case HIP_ARG_TEMPORAL:
				case HIP_ARG_OUTPUT:
					network->argv[slot] = frame[arg.kind - HIP_ARG_RGBA];
					break;
				}
			}
		}
		at += launch->count;
	}
	return ERROR_NONE;
}

enum error_code
hip_network_init (struct hip_network         *dest,
                  struct hip_api const       *api,
                  struct hip_model const     *model,
                  void                       *stream,
                  struct hip_plan const      *plan,
                  struct hip_placement const *placement,
                  struct error               *e)
{
	*dest = (struct hip_network){.api = api, .model = model, .stream = stream};
	enum error_code const code = build(dest, plan, placement, e);
	if (code)
		hip_network_fini(dest);
	return code;
}

void
hip_network_fini (struct hip_network *network)
{
	if (!network)
		return;
	if (network->api) {
		for (size_t b = 0; b < network->buffer_count; ++b)
			network->api->hipFree(network->buffers[b]);
		for (size_t i = 0; i < sizeof network->gather / sizeof *network->gather; ++i) {
			if (network->gather[i]) {
				network->api->hipFree(network->gather[i]);
				network->gather[i] = nullptr;
			}
		}
	}
	free(network->buffers);
	network->buffers = nullptr;
	free(network->launches);
	network->launches = nullptr;
	free(network->values);
	network->values = nullptr;
	free(network->argv);
	network->argv = nullptr;
	*network = (struct hip_network){0};
}

enum error_code
hip_network_enqueue (struct hip_network *network,
                     void               *rgba,
                     void               *history,
                     void               *output,
                     struct error       *e)
{
	network->frame = (struct hip_network_frame){history != nullptr, rgba, history ? history : rgba, output};
	// What every launch reads, read once instead of again after each call.
	struct hip_api const *const api = network->api;
	typeof (api->hipModuleLaunchKernel) const launch = api->hipModuleLaunchKernel;
	void *const stream = network->stream;
	size_t const warm = network->warm;
	struct hip_network_launch const *const end = network->launches + network->launch_count;
	for (struct hip_network_launch const *b = network->launches; b < end; ++b) {
		int const result = launch(b->function, b->grid, 1, 1, b->threads, 1, 1, 0, stream, b->argv[warm],
		                          nullptr);
		if (result)
			return hip_fail(e, api, result, "%s", HIP_PLAN_KERNELS[b->kernel].name);
	}
	network->warm = 1;
	return ERROR_NONE;
}

enum error_code
hip_network_print_memory (struct hip_network const *network,
                          struct error             *e)
{
	struct hip_api const *const api = network->api;
	size_t available = 0;
	size_t total = 0;
	int const result = api->hipMemGetInfo(&available, &total);
	if (result)
		return hip_fail(e, api, result, "memory stats");
	struct hip_model const *const model = network->model;
	printf("memory owned_MiB=%.1f allocations=%zu device_free_MiB=%.1f total_MiB=%.1f\n",
	       (model->bytes + network->bytes) / 1048576.,
	       model->count + network->buffer_count + sizeof network->gather / sizeof *network->gather,
	       available / 1048576., total / 1048576.);
	return ERROR_NONE;
}
