/** @file
 *
 * Independent HIP test: no model weights, game or worker required. The colour preservation kernel
 * against its CPU reference for two references, then timed at the 1080 tier.
 */
// SPDX-License-Identifier: MIT
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../backend/hip.h"
#include "../backend/native_kernels.h"
#include "../backend/reference.h"

/** @brief The test's stream, events, buffers, kernels and host images, which resources_fini()
 *         releases however the test ends. */
struct resources {
	struct native_kernels kernels;  //!< The kernels; zeroed until loaded.
	struct hip_api const *api;      //!< The runtime.
	void                 *stream;   //!< The stream; nullptr for none.
	void                 *start;    //!< The timing's start event; nullptr for none.
	void                 *end;      //!< Its end event; nullptr for none.
	void                 *original; //!< The reference on the device; nullptr for none.
	void                 *raw;      //!< The model output on the device; nullptr for none.
	void                 *output;   //!< The corrected output on the device; nullptr for none.
	float                *host;     //!< The host images; nullptr for none.
};

/** @brief Releases what the test holds, the kernels first.
 *
 * @param r The resources.
 */
static void
resources_fini (struct resources *r)
{
	native_kernels_fini(&r->kernels);
	struct hip_api const *const api = r->api;
	if (r->stream)
		api->hipStreamSynchronize(r->stream);
	void **const events[] = {&r->end, &r->start};
	for (size_t i = 0; i < sizeof events / sizeof *events; ++i) {
		if (*events[i]) {
			api->hipEventDestroy(*events[i]);
			*events[i] = nullptr;
		}
	}
	void **const buffers[] = {&r->output, &r->raw, &r->original};
	for (size_t i = 0; i < sizeof buffers / sizeof *buffers; ++i) {
		if (*buffers[i]) {
			api->hipFree(*buffers[i]);
			*buffers[i] = nullptr;
		}
	}
	if (r->stream) {
		api->hipStreamDestroy(r->stream);
		r->stream = nullptr;
	}
	free(r->host);
	r->host = nullptr;
}

/** @brief A HIP call's result.
 *
 * @param api    The runtime.
 * @param result The result.
 * @param what   What the call did.
 * @param e      Receives hip_fail()'s words for an error, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
check (struct hip_api const *api,
       int                   result,
       char const           *what,
       struct error         *e)
{
	return result ? hip_fail(e, api, result, "%s", what) : ERROR_NONE;
}

/** @brief Returns a failed call's code from the function that makes the call. */
#define TRY(call) \
	do { \
		enum error_code const try_code_ = (call); \
		if (try_code_) \
			return try_code_; \
	} while (0)

/** @brief Allocates device memory.
 *
 * @param api    The runtime.
 * @param buffer Receives the memory; unchanged on a failure.
 * @param bytes  Its size.
 * @param what   What the allocation is, for the words of a failure.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
allocate (struct hip_api const  *api,
          void                 **buffer,
          size_t                 bytes,
          char const            *what,
          struct error          *e)
{
	void *allocated = nullptr;
	TRY(check(api, api->hipMalloc(&allocated, bytes), what, e));
	*buffer = allocated;
	return ERROR_NONE;
}

/** @brief Creates an event.
 *
 * @param api   The runtime.
 * @param event Receives the event; unchanged on a failure.
 * @param what  What the event is for, for the words of a failure.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
create_event (struct hip_api const  *api,
              void                 **event,
              char const            *what,
              struct error          *e)
{
	void *created = nullptr;
	TRY(check(api, api->hipEventCreate(&created), what, e));
	*event = created;
	return ERROR_NONE;
}

/** @brief The test, on a selected device, with what it holds.
 *
 * @param r      Receives what the test holds; the caller releases it.
 * @param module The kernel module.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
run_with (struct resources *r,
          char const       *module,
          struct error     *e)
{
	struct hip_api const *const api = r->api;
	void *stream = nullptr;
	TRY(check(api, api->hipStreamCreate(&stream), "create stream", e));
	r->stream = stream;
	struct geometry const g = {1920, 1080, 1920, 1152, 1080, 0, 0, 1920, 1080};
	size_t const pixels = (size_t)g.width * g.height;
	r->host = malloc(pixels * (4 + 3 + 3 + 3) * sizeof *r->host);
	if (!r->host)
		return error_fail(e, "out of memory");
	float *const input = r->host;
	float *const model = input + pixels * 4;
	float *const expected = model + pixels * 3;
	float *const actual = expected + pixels * 3;
	for (size_t p = 0; p < pixels; ++p) {
		for (unsigned c = 0; c < 3; ++c) {
			input[p * 4 + c] = (float)((p * 11 + c * 71) % 1000) / 999;
			model[p * 3 + c] = (float)((p * 37 + c * 113) % 1000) / 999;
		}
		input[p * 4 + 3] = 1;
	}
	TRY(allocate(api, &r->original, pixels * 4 * sizeof *input, "allocate test input", e));
	TRY(allocate(api, &r->raw, pixels * 3 * sizeof *model, "allocate test model", e));
	TRY(allocate(api, &r->output, pixels * 3 * sizeof *model, "allocate test output", e));
	TRY(check(api, api->hipMemcpy(r->original, input, pixels * 4 * sizeof *input, 1), "upload test reference", e));
	TRY(check(api, api->hipMemcpy(r->raw, model, pixels * 3 * sizeof *model, 1), "upload test model", e));
	TRY(native_kernels_init(&r->kernels, api, r->stream, module, e));

	// The kernel reads a neighbourhood of the model output, so it cannot run in place.
	struct error in_place;
	if (!native_kernels_preserve_color(&r->kernels, g, r->original, r->raw, r->raw, 1, &in_place) ||
	    !strstr(in_place.what, "distinct input and output"))
		return error_fail(e, "GPU correction accepted an in-place output");
	static float const strengths[] = {0.f, .25f, .5f, 1.f};
	for (unsigned reference = 0; reference < 2; ++reference) {
		if (reference) {
			// Each apply reads the caller's reference as it is when the kernel runs.
			for (size_t p = 0; p < pixels; ++p)
				for (unsigned c = 0; c < 3; ++c)
					input[p * 4 + c] = (float)((p * 53 + c * 29) % 1000) / 999;
			TRY(check(api, api->hipMemcpy(r->original, input, pixels * 4 * sizeof *input, 1),
			          "replace test reference", e));
		}
		for (size_t s = 0; s < sizeof strengths / sizeof *strengths; ++s) {
			float const strength = strengths[s];
			TRY(reference_preserve_color(input, model, &g, strength, expected, e));
			TRY(native_kernels_preserve_color(&r->kernels, g, r->original, r->raw, r->output, strength, e));
			TRY(check(api, api->hipStreamSynchronize(r->stream), "finish correction", e));
			TRY(check(api, api->hipMemcpy(actual, r->output, pixels * 3 * sizeof *actual, 2), "read correction", e));
			float worst = 0;
			for (size_t i = 0; i < pixels * 3; ++i) {
				if (!isfinite(actual[i]))
					return error_fail(e, "nonfinite GPU correction");
				float const difference = fabsf(actual[i] - expected[i]);
				worst = worst < difference ? difference : worst;
			}
			printf("reference=%u strength=%g max_abs_error=%.9g\n", reference, (double)strength, (double)worst);
			if (worst > 2e-6f)
				return error_fail(e, "GPU correction differs from CPU reference");
		}
	}

	TRY(create_event(api, &r->start, "create start event", e));
	TRY(create_event(api, &r->end, "create end event", e));
	TRY(check(api, api->hipEventRecord(r->start, r->stream), "record start", e));
	for (unsigned i = 0; i < 20; ++i)
		TRY(native_kernels_preserve_color(&r->kernels, g, r->original, r->raw, r->output, 1, e));
	TRY(check(api, api->hipEventRecord(r->end, r->stream), "record end", e));
	TRY(check(api, api->hipEventSynchronize(r->end), "wait for timing", e));
	float ms = 0;
	TRY(check(api, api->hipEventElapsedTime(&ms, r->start, r->end), "measure correction", e));
	printf("GPU correction mean_ms=%.6f over 20 runs; excludes inference\n", (double)ms / 20);
	return ERROR_NONE;
}

/** @brief The test on a device.
 *
 * @param api    The runtime.
 * @param device The device.
 * @param module The kernel module.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
run (struct hip_api const *api,
     int                   device,
     char const           *module,
     struct error         *e)
{
	TRY(check(api, api->hipSetDevice(device), "select device", e));
	struct resources r = {.api = api};
	enum error_code const code = run_with(&r, module, e);
	resources_fini(&r);
	return code;
}

/** @brief Finds the devices and, unless one is chosen, the first gfx1201 among them.
 *
 * @param api     The runtime.
 * @param devices Receives the number of devices.
 * @param device  The chosen device, or -1, which receives the first gfx1201 one if there is one.
 * @param e       Receives the words for what failed, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
find_device (struct hip_api const *api,
             int                  *devices,
             int                  *device,
             struct error         *e)
{
	TRY(check(api, api->hipInit(0), "initialize HIP", e));
	TRY(check(api, api->hipGetDeviceCount(devices), "enumerate devices", e));
	for (int i = 0; *device < 0 && i < *devices; ++i) {
		struct hip_device_properties properties = {};
		TRY(check(api, api->hipGetDevicePropertiesR0600(&properties, i), "device properties", e));
		if (!strncmp(properties.gcnArchName, "gfx1201", sizeof "gfx1201" - 1))
			*device = i;
	}
	return ERROR_NONE;
}

#undef TRY

int
main (int    argc,
      char **argv)
{
	char const *module = "assets/HIP/gfx1201/linux_native.hsaco";
	int device = -1;
	static struct option const options[] = {
		{"module", required_argument, nullptr, 'm'},
		{"device", required_argument, nullptr, 'd'},
		{"help", no_argument, nullptr, 'h'},
		{nullptr, 0, nullptr, 0},
	};
	int code;
	while ((code = getopt_long(argc, argv, "+m:d:h", options, nullptr)) != -1) {
		if (code == 'h') {
			puts("Usage: color-gpu-test [OPTION]...\n"
			     " -m, --module PATH  Kernel module (default: assets/HIP/gfx1201/linux_native.hsaco)\n"
			     " -d, --device N     HIP device (default: auto, first gfx1201)\n"
			     " -h, --help         Show help (default: off)\n"
			     "Tests correction against CPU reference for two references, then times a 1080-tier kernel.\n"
			     "No model assets required. Exit 77 means the module, HIP runtime or device is unavailable.");
			return 0;
		}
		if (code == 'm') {
			module = optarg;
			continue;
		}
		if (code == 'd') {
			char *end;
			long const n = strtol(optarg, &end, 10);
			if (!*optarg || *end || n < 0 || n > 1024)
				return 2;
			device = (int)n;
			continue;
		}
		return 2;
	}
	if (optind != argc || !*module)
		return 2;
	if (access(module, R_OK)) {
		fprintf(stderr, "SKIP: cannot read module: %s\n", module);
		return 77;
	}
	struct error e;
	struct hip_api api;
	if (hip_load(&api, &e)) {
		fprintf(stderr, "SKIP: %s\n", e.what);
		return 77;
	}
	// A runtime that loads but fails is a failure; only an absent device skips.
	int devices = 0;
	if (find_device(&api, &devices, &device, &e)) {
		fprintf(stderr, "%s\n", e.what);
		return 1;
	}
	if (!devices) {
		fprintf(stderr, "SKIP: no HIP devices\n");
		return 77;
	}
	if (device < 0) {
		fprintf(stderr, "SKIP: no gfx1201 device\n");
		return 77;
	}
	if (run(&api, device, module, &e)) {
		fprintf(stderr, "%s\n", e.what);
		return 1;
	}
	return 0;
}
