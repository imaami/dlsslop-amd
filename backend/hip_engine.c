/** @file
 *
 * The HIP engine: hip_engine.h.
 */
// SPDX-License-Identifier: MIT
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec_gpu.h"
#include "control_selftest.h"
#include "engine.h"
#include "error.h"
#include "geometry.h"
#include "hip.h"
#include "hip_engine.h"
#include "hip_network.h"
#include "hip_plan.h"
#include "kernel_args.h"
#include "kernel_math.h"
#include "native_kernels.h"
#include "options.h"
#include "processing.h"
#include "reference.h"
#include "shm_protocol.h"
#include "temporal_gpu.h"
#include "trace.h"

// "processing=WIDTHxHEIGHT" and "device INDEX" fit.
static_assert(sizeof "processing=4294967295x4294967295" <= ENGINE_DESCRIPTION_BYTES);
static_assert(sizeof "device -2147483648" <= ENGINE_DESCRIPTION_BYTES);

/** @brief The gfx1201 device that a request names, or the first one; every visible device is
 *         listed on the way.
 *
 * @param api       The runtime.
 * @param requested The device's index, or -1 for the first gfx1201 device.
 * @param dest      Receives the device's index; untouched on a failure.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
select_device (struct hip_api const *api,
               int                   requested,
               int                  *dest,
               struct error         *e)
{
	int r = api->hipInit(0);
	if (r)
		return hip_fail(e, api, r, "hipInit (check /dev/kfd permissions and ROCm userspace)");
	int count = 0;
	int version = 0;
	r = api->hipRuntimeGetVersion(&version);
	if (r)
		return hip_fail(e, api, r, "HIP runtime version");
	r = api->hipGetDeviceCount(&count);
	if (r)
		return hip_fail(e, api, r, "HIP device count");
	fprintf(stderr, "HIP runtime=%d, visible devices=%d\n", version, count);
	int selected = -1;
	for (int i = 0; i < count; ++i) {
		struct hip_device_properties p = {0};
		r = api->hipGetDevicePropertiesR0600(&p, i);
		if (r)
			return hip_fail(e, api, r, "device properties");
		fprintf(stderr, "  device %d: %s; arch=%s; PCI=%04x:%02x:%02x; VRAM=%.0f MiB\n",
		        i, p.name, p.gcnArchName, p.pciDomainID, p.pciBusID, p.pciDeviceID,
		        p.totalGlobalMem / 1048576.0);
		bool const gfx1201 = !strncmp(p.gcnArchName, "gfx1201", 7)
		                     && (!p.gcnArchName[7] || p.gcnArchName[7] == ':');
		if (gfx1201 && ((requested < 0 && selected < 0) || requested == i))
			selected = i;
	}
	if (selected < 0)
		return error_fail(e, "%s", requested < 0
		                  ? "no gfx1201 device found (RX 9070/9070 XT required); check HIP_VISIBLE_DEVICES"
		                  : "selected device is unavailable or is not gfx1201");
	*dest = selected;
	return ERROR_NONE;
}

enum error_code
hip_engine_open (struct hip_api *dest,
                 struct options *o,
                 struct error   *e)
{
	struct hip_api api;
	enum error_code code = hip_load(&api, e);
	if (code)
		return code;
	code = select_device(&api, o->device, &o->device, e);
	if (code)
		return code;
	// Wait for the device asleep, not spinning a core per waiting thread
	// (hipDeviceScheduleBlockingSync), before the device's context exists.
	if (api.hipSetDeviceFlags) {
		int r = api.hipSetDevice(o->device);
		if (r)
			return hip_fail(e, &api, r, "select the HIP device");
		r = api.hipSetDeviceFlags(4);
		if (r)
			return hip_fail(e, &api, r, "wait for the device without spinning");
	}
	*dest = api;
	return ERROR_NONE;
}

struct hip_engine
hip_engine (struct options const *o,
            uint32_t              tier,
            struct hip_api const *api)
{
	// Upstream's approximate ViT cache, which this variable selected, is not
	// part of the port.
	char const *const adaptive = getenv("DLSS5_VIT_ADAPTIVE");
	if (adaptive && strcmp(adaptive, "0"))
		fprintf(stderr, "DLSS5_VIT_ADAPTIVE is not supported; the HIP network runs every ViT block\n");
	return (struct hip_engine){
		.options = o,
		.api = *api,
		.previous_settings = processing_settings(),
		.tier = tier,
	};
}

/** @brief Frees an import slot's pair and empties the slot.
 *
 * @param engine The engine, whose stream is done with the pair.
 * @param slot   The slot.
 */
static void
release_import (struct hip_engine        *engine,
                struct hip_engine_import *slot)
{
	for (unsigned i = 0; i < 2; ++i) {
		if (slot->frame[i])
			engine->api.hipFree(slot->frame[i]);
		if (slot->memory[i])
			engine->api.hipDestroyExternalMemory(slot->memory[i]);
	}
	*slot = (struct hip_engine_import){};
}

/** @brief Frees everything hip_engine_prepare() made but the stream and the model, and every
 *         import.
 *
 * @param engine The engine.
 */
static void
release (struct hip_engine *engine)
{
	if (!engine->stream)
		return;
	// Once the stream is done, nothing prepare made is in use.
	engine->api.hipStreamSynchronize(engine->stream);
	for (unsigned slot = 0; slot < ENGINE_SLOTS; ++slot)
		release_import(engine, &engine->imported[slot]);
	for (unsigned i = 0; i < 4; ++i) {
		if (engine->marks[i])
			engine->api.hipEventDestroy(engine->marks[i]);
		engine->marks[i] = nullptr;
	}
	void **const buffers[] = {
		&engine->device_input, &engine->device_feedback, &engine->device_output, &engine->device_scratch,
	};
	for (unsigned i = 0; i < sizeof buffers / sizeof *buffers; ++i) {
		if (*buffers[i])
			engine->api.hipFree(*buffers[i]);
		*buffers[i] = nullptr;
	}
	engine->answer = nullptr;
	free(engine->input);
	engine->input = nullptr;
	engine->input_count = 0;
	free(engine->neural);
	engine->neural = nullptr;
	engine->neural_count = 0;
	temporal_gpu_fini(&engine->temporal);
	codec_gpu_fini(&engine->gpu_codec);
	native_kernels_fini(&engine->kernels);
	hip_network_fini(&engine->network);
	engine->previous_settings = processing_settings();
}

void
hip_engine_fini (struct hip_engine *engine)
{
	if (engine) {
		release(engine);
		if (engine->stream)
			engine->api.hipStreamDestroy(engine->stream);
		hip_model_fini(&engine->model);
		*engine = (struct hip_engine){};
	}
}

char const *
hip_engine_describe_device (struct hip_engine const *engine,
                            char                     dest[ENGINE_DESCRIPTION_BYTES])
{
	snprintf(dest, ENGINE_DESCRIPTION_BYTES, "device %d", engine->options->device);
	return dest;
}

char const *
hip_engine_describe_processing (struct hip_engine const *engine,
                                char                     dest[ENGINE_DESCRIPTION_BYTES])
{
	struct NativeTier const *const raster = ShmNativeTier(engine->tier);
	snprintf(dest, ENGINE_DESCRIPTION_BYTES, "processing=%" PRIu32 "x%" PRIu32, raster->width,
	         raster->networkHeight);
	return dest;
}

/** @brief Allocates device memory for the engine.
 *
 * @param engine The engine.
 * @param dest   Receives the memory; untouched on a failure.
 * @param bytes  The bytes.
 * @param what   What the memory is for, for the words.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
allocate (struct hip_engine  *engine,
          void              **dest,
          size_t              bytes,
          char const         *what,
          struct error       *e)
{
	void *memory;
	int const r = engine->api.hipMalloc(&memory, bytes);
	if (r)
		return hip_fail(e, &engine->api, r, "%s", what);
	*dest = memory;
	return ERROR_NONE;
}

/** @brief Waits for the engine's stream.
 *
 * @param engine The engine.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
synchronize (struct hip_engine *engine,
             struct error      *e)
{
	int const r = engine->api.hipStreamSynchronize(engine->stream);
	return r ? hip_fail(e, &engine->api, r, "network completion") : ERROR_NONE;
}

/** @brief The native kernels' code object, in the modules' directory. */
#define NATIVE_MODULE "/linux_native.hsaco"

/** @brief hip_engine_prepare() once the plan and its placement are made.
 *
 * @param engine    The engine, with its stream.
 * @param raster    The tier's raster.
 * @param plan      The tier's plan.
 * @param placement Its placement.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or the code of what failed.
 */
static enum error_code
prepare_planned (struct hip_engine          *engine,
                 struct NativeTier const    *raster,
                 struct hip_plan const      *plan,
                 struct hip_placement const *placement,
                 struct error               *e)
{
	struct options const *const o = engine->options;
	struct hip_api const *const api = &engine->api;
	enum error_code code;
	// A model that failed to load is freed, so that no later network binds it.
	if (!engine->model.api) {
		code = hip_model_init(&engine->model, api, o->modules, o->modules_length, o->assets,
		                      o->assets_length, plan->weights, plan->weight_count, e);
		if (code)
			return code;
	}
	code = hip_network_init(&engine->network, api, &engine->model, engine->stream, plan, placement, e);
	if (code)
		return code;
	for (unsigned i = 0; i < 4; ++i) {
		void *event;
		int const r = api->hipEventCreate(&event);
		if (r)
			return hip_fail(e, api, r, "create timing event");
		engine->marks[i] = event;
	}
	// Tuning, colour and motion use the module's kernels with the CPU codec too.
	char *native = malloc(o->modules_length + sizeof NATIVE_MODULE);
	if (!native)
		return error_fail(e, "out of memory");
	memcpy(native, o->modules, o->modules_length);
	memcpy(native + o->modules_length, NATIVE_MODULE, sizeof NATIVE_MODULE);
	code = native_kernels_init(&engine->kernels, api, engine->stream, native, e);
	free(native);
	native = nullptr;
	if (code)
		return code;
	if (!o->cpu_codec) {
		code = codec_gpu_init(&engine->gpu_codec, &engine->kernels, e);
		if (code)
			return code;
	}
	engine->temporal = temporal_gpu(&engine->kernels);
	size_t const pixels = (size_t)raster->width * raster->networkHeight;
	code = allocate(engine, &engine->device_input, pixels * 16, "allocate network input", e);
	if (code)
		return code;
	code = allocate(engine, &engine->device_output, pixels * 12, "allocate network output", e);
	if (code)
		return code;
	// The host copies of a pass's input and answer serve the CPU codec and the self-test's checks.
	if (!engine->gpu_codec.kernels || o->self_test) {
		engine->input = calloc(pixels * 4, sizeof *engine->input);
		engine->neural = calloc(pixels * 3, sizeof *engine->neural);
		if (!engine->input || !engine->neural)
			return error_fail(e, "out of memory");
		engine->input_count = pixels * 4;
		engine->neural_count = pixels * 3;
	}
	// One evaluation, without a history, before the daemon reports itself
	// ready, so that a kernel that cannot launch fails here. It is the
	// network's first frame, which has a buffer assignment of its own.
	int const r = api->hipMemsetAsync(engine->device_input, 0, pixels * 16, engine->stream);
	if (r)
		return hip_fail(e, api, r, "warm input");
	code = hip_network_enqueue(&engine->network, engine->device_input, nullptr, engine->device_output, e);
	if (code)
		return code;
	code = synchronize(engine, e);
	if (code)
		return code;
	code = hip_network_print_memory(&engine->network, e);
	if (code)
		return code;
	if (!o->self_test)
		return ERROR_NONE;
	// The kernels on synthetic inputs, independent of model weights.
	return control_selftest_run(&engine->kernels, e);
}

#undef NATIVE_MODULE

enum error_code
hip_engine_prepare (struct hip_engine *engine,
                    struct error      *e)
{
	struct NativeTier const *const raster = ShmNativeTier(engine->tier);
	struct hip_api const *const api = &engine->api;
	// What follows is allocated on the device this thread selects.
	int r = api->hipSetDevice(engine->options->device);
	if (r)
		return hip_fail(e, api, r, "select the HIP device");
	// With default flags, the stream's work stays in order with the null
	// stream's synchronous copies, which the CPU codec and diagnostics make.
	if (!engine->stream) {
		void *stream;
		r = api->hipStreamCreate(&stream);
		if (r)
			return hip_fail(e, api, r, "create the HIP stream");
		engine->stream = stream;
	}
	// The plan and its placement, which the model and the network read, freed on return.
	struct hip_plan plan;
	enum error_code code = hip_plan_init(&plan, raster->width, raster->networkHeight,
	                                     engine->options->performance, e);
	if (code)
		return code;
	struct hip_placement placement;
	code = hip_plan_place(&placement, &plan, e);
	if (!code)
		code = prepare_planned(engine, raster, &plan, &placement, e);
	hip_placement_fini(&placement);
	hip_plan_fini(&plan);
	return code;
}

enum error_code
hip_engine_retier (struct hip_engine *engine,
                   uint32_t           tier,
                   struct error      *e)
{
	release(engine);
	engine_slots_forget(&engine->slots);
	engine->tier = tier;
	return hip_engine_prepare(engine, e);
}

bool
hip_engine_import_into (struct hip_engine              *engine,
                        ptrdiff_t                       slot,
                        struct ShmTransportOffer const *offer,
                        int                             fds[2])
{
	if (!engine->gpu_codec.kernels)
		return false;
	struct hip_api const *const api = &engine->api;
	struct hip_engine_import next = {};
	for (unsigned i = 0; i < 2; ++i) {
		struct hip_memory_desc memory = {0};
		memory.type = 1; // hipExternalMemoryHandleTypeOpaqueFd
		memory.handle.fd = fds[i];
		memory.size = offer->allocation[i];
		struct hip_buffer_desc buffer = {0};
		buffer.size = offer->allocation[i];
		void *imported;
		if (api->hipImportExternalMemory(&imported, &memory)) {
			release_import(engine, &next);
			return false;
		}
		next.memory[i] = imported;
		fds[i] = -1;
		void *mapped;
		if (api->hipExternalMemoryGetMappedBuffer(&mapped, next.memory[i], &buffer)) {
			release_import(engine, &next);
			return false;
		}
		next.frame[i] = mapped;
	}
	release_import(engine, &engine->imported[slot]);
	engine->imported[slot] = next;
	return true;
}

/** @brief Records one of the frame's timing events on the stream.
 *
 * @param engine The engine.
 * @param i      The event: frame start, uploaded, evaluated or answered.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
mark (struct hip_engine *engine,
      unsigned           i,
      struct error      *e)
{
	int const r = engine->api.hipEventRecord(engine->marks[i], engine->stream);
	return r ? hip_fail(e, &engine->api, r, "record timing event") : ERROR_NONE;
}

/** @brief Copies device memory to the host or the host to the device.
 *
 * @param engine The engine.
 * @param dest   Where the bytes go.
 * @param source Where they come from.
 * @param bytes  The bytes.
 * @param kind   hipMemcpyKind: 1 host to device, 2 device to host, 4 inferred.
 * @param what   What the copy is, for the words.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
copy (struct hip_engine *engine,
      void              *dest,
      void const        *source,
      size_t             bytes,
      int                kind,
      char const        *what,
      struct error      *e)
{
	int const r = engine->api.hipMemcpy(dest, source, bytes, kind);
	return r ? hip_fail(e, &engine->api, r, "%s", what) : ERROR_NONE;
}

/** @brief Reads a pass's stage back into the trace, if there is one.
 *
 * @param engine   The engine.
 * @param trace    The trace, or nullptr.
 * @param g        The frame's geometry.
 * @param pass     The pass.
 * @param stage    The stage's name: input, raw, tuned or color.
 * @param pointer  The stage's device memory: g->width x g->height pixels of @a channels floats.
 * @param channels 3 for RGB, 4 for RGBA.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
trace_image (struct hip_engine     *engine,
             struct frame_trace    *trace,
             struct geometry const *g,
             uint32_t               pass,
             char const            *stage,
             void const            *pointer,
             unsigned               channels,
             struct error          *e)
{
	if (!trace)
		return ERROR_NONE;
	enum error_code code = synchronize(engine, e);
	if (code)
		return code;
	size_t const count = (size_t)g->width * g->height * channels;
	float *buffer = malloc(count * sizeof *buffer);
	if (!buffer)
		return error_fail(e, "out of memory");
	code = copy(engine, buffer, pointer, count * sizeof *buffer, 2, "read diagnostic neural stage", e);
	if (!code) {
		char name[TRACE_STAGE_BYTES];
		int const length = snprintf(name, sizeof name, "pass-%02" PRIu32 "-%s", pass + 1, stage);
		if (length < 0 || length >= (int)sizeof name)
			code = error_fail(e, "name diagnostic neural stage");
		else
			frame_trace_image(trace, name, (size_t)length, buffer, g, channels);
	}
	free(buffer);
	buffer = nullptr;
	return code;
}

/** @brief Checks the GPU encode of a host frame against the CPU reference.
 *
 * @param engine The engine.
 * @param input  The frame.
 * @param g      Its geometry.
 * @param fp16   Whether it is RGBA16F.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
verify_encode (struct hip_engine     *engine,
               uint8_t const         *input,
               struct geometry const *g,
               bool                   fp16,
               struct error          *e)
{
	size_t const count = (size_t)g->width * g->height * 4;
	float *reference = calloc(count, sizeof *reference);
	float *actual = calloc(count, sizeof *actual);
	if (!reference || !actual) {
		free(actual);
		actual = nullptr;
		free(reference);
		reference = nullptr;
		return error_fail(e, "out of memory");
	}
	enum error_code code = reference_encode_proxy(input, g, fp16, reference, e);
	if (!code)
		code = control_selftest_read(&engine->kernels, engine->device_input, actual, count, e);
	if (!code)
		code = control_selftest_compare(actual, reference, count, "GPU encoder", e);
	free(actual);
	actual = nullptr;
	free(reference);
	reference = nullptr;
	if (code)
		return code;
	printf("GPU encode vs CPU reference: FP32 bit-identical\n");
	fflush(stdout);
	return ERROR_NONE;
}

/** @brief Checks the GPU feedback into a later pass against the CPU reference.
 *
 * @param engine      The engine, whose neural holds the previous pass's answer.
 * @param pass_input  The pass's input, which the GPU feedback wrote.
 * @param g           The frame's geometry.
 * @param precision16 Whether the feedback keeps binary16 precision.
 * @param pass        The pass.
 * @param passes      The frame's passes.
 * @param e           Receives the words for what stopped it, or nullptr.
 * @return            ERROR_NONE, or the code of what failed.
 */
static enum error_code
verify_feedback (struct hip_engine     *engine,
                 void const            *pass_input,
                 struct geometry const *g,
                 bool                   precision16,
                 uint32_t               pass,
                 uint32_t               passes,
                 struct error          *e)
{
	enum error_code code = reference_feedback_neural_rgb(engine->neural, g, precision16, engine->input, e);
	if (code)
		return code;
	float *actual = calloc(engine->input_count, sizeof *actual);
	if (!actual)
		return error_fail(e, "out of memory");
	code = control_selftest_read(&engine->kernels, pass_input, actual, engine->input_count, e);
	if (!code)
		code = control_selftest_compare(actual, engine->input, engine->input_count, "GPU inter-pass feedback",
		                                e);
	free(actual);
	actual = nullptr;
	if (code)
		return code;
	printf("GPU feedback for pass %" PRIu32 "/%" PRIu32 " vs CPU reference: FP32 bit-identical\n", pass + 1,
	       passes);
	return ERROR_NONE;
}

/** @brief Checks the GPU decode of a host frame against the CPU reference.
 *
 * @param engine The engine, whose neural holds the last pass's answer.
 * @param input  The frame.
 * @param output The GPU's decode of the answer.
 * @param width  The frame's width.
 * @param height Its height.
 * @param g      Its geometry.
 * @param fp16   Whether the frames are RGBA16F.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
verify_decode (struct hip_engine     *engine,
               uint8_t const         *input,
               uint8_t const         *output,
               size_t                 width,
               size_t                 height,
               struct geometry const *g,
               bool                   fp16,
               struct error          *e)
{
	size_t const bpp = fp16 ? 8 : 4;
	size_t const bytes = width * height * bpp;
	uint8_t *reference = calloc(bytes, 1);
	if (!reference)
		return error_fail(e, "out of memory");
	enum error_code code = reference_decode_neural_proxy(input, g, fp16, engine->neural, reference, e);
	if (!code) {
		size_t first = 0;
		while (first < bytes && reference[first] == output[first])
			++first;
		if (first < bytes) {
			fprintf(stderr, "GPU decoder first mismatch: x=%zu y=%zu byte=%zu GPU=%u CPU=%u\n",
			        (first / bpp) % width, (first / bpp) / width, first % bpp, (unsigned)output[first],
			        (unsigned)reference[first]);
			code = error_fail(e, "GPU decoder disagrees with CPU reference");
		}
	}
	free(reference);
	reference = nullptr;
	if (code)
		return code;
	printf("GPU decode vs CPU reference: bit-identical\n");
	fflush(stdout);
	return ERROR_NONE;
}

/** @brief Uploads and encodes a frame into the network's input: the GPU codec, checked against the
 *         CPU reference with @a verify, or the CPU codec.
 *
 * @param engine   The engine.
 * @param io       The frame.
 * @param g        Its geometry.
 * @param settings Its settings.
 * @param verify   Whether to check the GPU codec of a host frame against the CPU reference.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or the code of what failed.
 */
static enum error_code
encode (struct hip_engine                *engine,
        struct engine_frames const       *io,
        struct geometry const            *g,
        struct processing_settings const *settings,
        bool                              verify,
        struct error                     *e)
{
	if (!engine->gpu_codec.kernels) {
		enum error_code const code = reference_encode_proxy(io->proxy, g, settings->fp16, engine->input, e);
		if (code)
			return code;
		return copy(engine, engine->device_input, engine->input, engine->input_count * sizeof (float), 1,
		            "upload encoded frame", e);
	}
	enum error_code const code = codec_gpu_encode(&engine->gpu_codec, io->proxy, g, engine->device_input,
	                                              settings->fp16, io->slot >= 0, e);
	if (code || !verify)
		return code;
	return verify_encode(engine, io->proxy, g, settings->fp16, e);
}

/** @brief Feeds a pass's answer into the next pass's input: the GPU codec, checked against the CPU
 *         reference with @a verify, or the CPU codec from the engine's neural.
 *
 * @param engine     The engine.
 * @param answer     The pass's answer.
 * @param pass_input Receives the next pass's input.
 * @param g          The frame's geometry.
 * @param settings   Its settings: whether the feedback keeps binary16 precision.
 * @param pass       The next pass.
 * @param passes     The frame's passes.
 * @param verify     Whether to check the GPU codec against the CPU reference.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or the code of what failed.
 */
static enum error_code
feed_back (struct hip_engine                *engine,
           void                             *answer,
           void                             *pass_input,
           struct geometry const            *g,
           struct processing_settings const *settings,
           uint32_t                          pass,
           uint32_t                          passes,
           bool                              verify,
           struct error                     *e)
{
	bool const precision16 = settings->precision16;
	if (!engine->gpu_codec.kernels) {
		enum error_code const code = reference_feedback_neural_rgb(engine->neural, g, precision16,
		                                                           engine->input, e);
		if (code)
			return code;
		return copy(engine, pass_input, engine->input, engine->input_count * sizeof (float), 1,
		            "upload inter-pass feedback", e);
	}
	enum error_code const code = codec_gpu_feedback(&engine->gpu_codec, answer, pass_input, precision16, e);
	if (code || !verify)
		return code;
	return verify_feedback(engine, pass_input, g, precision16, pass, passes, e);
}

/** @brief Decodes the last pass's answer into the frame's answer and marks the frame answered:
 *         the GPU codec, which waits for the stream and is checked against the CPU reference with
 *         @a verify, or the CPU codec from the engine's neural.
 *
 * @param engine   The engine.
 * @param io       The frame.
 * @param answer   The last pass's answer.
 * @param width    The frame's width.
 * @param height   Its height.
 * @param g        Its geometry.
 * @param settings Its settings.
 * @param verify   Whether to check the GPU codec of a host frame against the CPU reference.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or the code of what failed.
 */
static enum error_code
decode (struct hip_engine                *engine,
        struct engine_frames const       *io,
        void                             *answer,
        uint32_t                          width,
        uint32_t                          height,
        struct geometry const            *g,
        struct processing_settings const *settings,
        bool                              verify,
        struct error                     *e)
{
	enum error_code code;
	if (!engine->gpu_codec.kernels) {
		code = reference_decode_neural_proxy(io->proxy, g, settings->fp16, engine->neural, io->answer, e);
		return code ? code : mark(engine, 3, e);
	}
	code = codec_gpu_decode(&engine->gpu_codec, answer, io->answer, e);
	if (code)
		return code;
	code = mark(engine, 3, e);
	if (code)
		return code;
	code = codec_gpu_finish(&engine->gpu_codec, e);
	if (code || !verify)
		return code;
	return verify_decode(engine, io->proxy, io->answer, width, height, g, settings->fp16, e);
}

enum error_code
hip_engine_frame (struct hip_engine                *engine,
                  struct engine_frames const       *io,
                  uint32_t                          width,
                  uint32_t                          height,
                  uint32_t                          passes,
                  struct processing_settings const *settings,
                  struct frame_trace               *trace,
                  bool                              verify,
                  struct error                     *e)
{
	struct geometry g;
	enum error_code code = geometry_init(&g, width, height, engine->tier, e);
	if (code)
		return code;
	// Said once: the Vulkan model's conditioning, which this network does not have.
	if ((settings->style || !settings->auto_mask || settings->skin_structure != -1)
	    && !engine->warned_conditioning) {
		engine->warned_conditioning = 1;
		fprintf(stderr, "the HIP network ignores style, skin structure and the automatic mask\n");
	}
	bool const tuned = !kernel_math_tuning_is_default(&settings->tuning);
	bool const colored = settings->color_preserve > 0;
	if ((tuned || colored) && !engine->device_scratch) {
		code = allocate(engine, &engine->device_scratch, (size_t)g.width * g.height * 12,
		                "allocate post-processing output", e);
		if (code)
			return code;
	}
	if (passes > 1 && !engine->device_feedback) {
		code = allocate(engine, &engine->device_feedback, (size_t)g.width * g.height * 16,
		                "allocate inter-pass feedback", e);
		if (code)
			return code;
	}
	code = mark(engine, 0, e);
	if (code)
		return code;
	code = encode(engine, io, &g, settings, verify, e);
	if (code)
		return code;
	code = mark(engine, 1, e);
	if (code)
		return code;
	struct processing_settings *const previous = &engine->previous_settings;
	if (settings->motion) {
		// temporal_gpu_begin() itself drops the history for a new pass count, quality, grid or placement.
		bool const reset = engine->options->self_test || !previous->motion
		                   || previous->fp16 != settings->fp16
		                   || previous->precision16 != settings->precision16
		                   || !kernel_math_tuning_equal(&previous->tuning, &settings->tuning)
		                   || previous->color_preserve != settings->color_preserve;
		previous->motion = false; // Until the frame completes: a failed one leaves no history.
		// Until end(), reject nothing: the worker would serve on with the
		// history still pending, and the next begin() would fail.
		code = temporal_gpu_begin(&engine->temporal, engine->device_input, &g, settings->motion_quality,
		                          settings->motion_grid, passes, reset, e);
		if (code)
			return code;
	} else {
		temporal_gpu_reset(&engine->temporal);
	}
	void *answer = engine->device_output;
	for (uint32_t pass = 0; pass < passes; ++pass) {
		void *const pass_input = pass ? engine->device_feedback : engine->device_input;
		if (pass) {
			code = feed_back(engine, answer, pass_input, &g, settings, pass, passes, verify, e);
			if (code)
				return code;
		}
		code = trace_image(engine, trace, &g, pass, "input", pass_input, 4, e);
		if (code)
			return code;
		// Stages alternate between two buffers, since tuning and colour read
		// neighbours; with motion, the last writes the pass's history slot.
		void *history = nullptr;
		void *stages[] = {engine->device_output, engine->device_scratch, engine->device_output};
		if (settings->motion) {
			code = temporal_gpu_history(&engine->temporal, pass, pass_input, &history, e);
			if (code)
				return code;
			code = temporal_gpu_target(&engine->temporal, pass, &stages[tuned + colored], e);
			if (code)
				return code;
		}
		code = hip_network_enqueue(&engine->network, pass_input, history, stages[0], e);
		if (code)
			return code;
		code = trace_image(engine, trace, &g, pass, "raw", stages[0], 3, e);
		if (code)
			return code;
		if (tuned) {
			code = native_kernels_tune(&engine->kernels, g, pass_input, stages[0], stages[1],
			                           settings->tuning, e);
			if (code)
				return code;
			code = trace_image(engine, trace, &g, pass, "tuned", stages[1], 3, e);
			if (code)
				return code;
		}
		if (colored) {
			code = native_kernels_preserve_color(&engine->kernels, g, engine->device_input, stages[tuned],
			                                     stages[1 + tuned], settings->color_preserve, e);
			if (code)
				return code;
			code = trace_image(engine, trace, &g, pass, "color", stages[1 + tuned], 3, e);
			if (code)
				return code;
		}
		answer = stages[tuned + colored];
		// The CPU codec, like the GPU one, rejects the nonfinite samples it reads.
		if (!engine->gpu_codec.kernels || verify) {
			code = synchronize(engine, e);
			if (code)
				return code;
			code = copy(engine, engine->neural, answer, engine->neural_count * sizeof (float), 2,
			            "read neural answer", e);
			if (code)
				return code;
		}
	}
	engine->answer = answer;
	if (settings->motion) {
		code = temporal_gpu_end(&engine->temporal, e);
		if (code)
			return code;
	}
	code = mark(engine, 2, e);
	if (code)
		return code;
	code = decode(engine, io, answer, width, height, &g, settings, verify, e);
	if (code)
		return code;
	*previous = *settings;
	struct hip_api const *const api = &engine->api;
	// codec_gpu_finish() waited for the GPU codec's stream already.
	if (!engine->gpu_codec.kernels) {
		int const r = api->hipEventSynchronize(engine->marks[3]);
		if (r)
			return hip_fail(e, api, r, "timing event completion");
	}
	struct engine_times *const times = &engine->times;
	int r = api->hipEventElapsedTime(&times->upload_ms, engine->marks[0], engine->marks[1]);
	if (r)
		return hip_fail(e, api, r, "upload interval");
	r = api->hipEventElapsedTime(&times->inference_ms, engine->marks[1], engine->marks[2]);
	if (r)
		return hip_fail(e, api, r, "inference interval");
	r = api->hipEventElapsedTime(&times->readback_ms, engine->marks[2], engine->marks[3]);
	return r ? hip_fail(e, api, r, "readback interval") : ERROR_NONE;
}

enum error_code
hip_engine_read_raw (struct hip_engine *engine,
                     struct error      *e)
{
	if (!engine->gpu_codec.kernels)
		return ERROR_NONE;
	return copy(engine, engine->neural, engine->answer, engine->neural_count * sizeof (float), 2,
	            "read raw network answer", e);
}

enum error_code
hip_engine_read_back (struct hip_engine *engine,
                      void              *host,
                      void const        *source,
                      size_t             bytes,
                      struct error      *e)
{
	return copy(engine, host, source, bytes, 4, "read diagnostic frame", e);
}
