/** @file
 *
 * The HIP self-test's checks of the linux_native module's kernels: control_selftest.h.
 */
// SPDX-License-Identifier: MIT
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec_gpu.h"
#include "control_selftest.h"
#include "geometry.h"
#include "kernel_math.h"
#include "reference.h"
#include "temporal_gpu.h"

/** @brief A device buffer of a check's. buffer_init() allocates one and buffer_fini() frees it; a
 *         zeroed one is none. */
struct buffer {
	struct native_kernels const *kernels; //!< The kernels, whose stream uses the buffer.
	void                        *pointer; //!< The device memory; nullptr for none.
	size_t                       bytes;   //!< Its size.
};

/** @brief Allocates a buffer.
 *
 * @param dest    Receives the buffer; unchanged on a failure.
 * @param kernels The kernels.
 * @param bytes   Its size.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
buffer_init (struct buffer               *dest,
             struct native_kernels const *kernels,
             size_t                       bytes,
             struct error                *e)
{
	void *pointer = nullptr;
	int const result = kernels->api->hipMalloc(&pointer, bytes);
	if (result)
		return hip_fail(e, kernels->api, result, "allocate control self-test buffer");
	*dest = (struct buffer){.kernels = kernels, .pointer = pointer, .bytes = bytes};
	return ERROR_NONE;
}

/** @brief Waits for the stream, frees a buffer and zeroes it.
 *
 * @param buffer The buffer.
 */
static void
buffer_fini (struct buffer *buffer)
{
	if (buffer->pointer) {
		struct hip_api const *const api = buffer->kernels->api;
		api->hipStreamSynchronize(buffer->kernels->stream);
		api->hipFree(buffer->pointer);
		buffer->pointer = nullptr;
	}
	*buffer = (struct buffer){0};
}

/** @brief Waits for the stream, then fills a buffer from the host.
 *
 * @param buffer The buffer.
 * @param values Its floats, as many as fill it.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
buffer_upload (struct buffer const *buffer,
               float const         *values,
               struct error        *e)
{
	struct hip_api const *const api = buffer->kernels->api;
	int result = api->hipStreamSynchronize(buffer->kernels->stream);
	if (result)
		return hip_fail(e, api, result, "wait before control self-test upload");
	result = api->hipMemcpy(buffer->pointer, values, buffer->bytes, 1);
	if (result)
		return hip_fail(e, api, result, "upload control self-test fixture");
	return ERROR_NONE;
}

/** @brief Waits for the stream, then reads a whole buffer back.
 *
 * @param buffer The buffer.
 * @param values Receives its floats.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
buffer_read (struct buffer const *buffer,
             float               *values,
             struct error        *e)
{
	return control_selftest_read(buffer->kernels, buffer->pointer, values, buffer->bytes / sizeof *values, e);
}

/** @brief Fails with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param message   The words of its failure.
 * @param e         Receives them, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
require (bool          condition,
         char const   *message,
         struct error *e)
{
	if (condition)
		return ERROR_NONE;
	return error_fail(e, "%s", message);
}

enum error_code
control_selftest_read (struct native_kernels const *kernels,
                       void const                  *pointer,
                       float                       *samples,
                       size_t                       count,
                       struct error                *e)
{
	if (!pointer)
		return error_fail(e, "control self-test received null GPU output");
	struct hip_api const *const api = kernels->api;
	int result = api->hipStreamSynchronize(kernels->stream);
	if (result)
		return hip_fail(e, api, result, "complete control self-test kernel");
	result = api->hipMemcpy(samples, pointer, count * sizeof *samples, 2);
	if (result)
		return hip_fail(e, api, result, "read control self-test output");
	return ERROR_NONE;
}

enum error_code
control_selftest_compare (float const  *actual,
                          float const  *expected,
                          size_t        count,
                          char const   *name,
                          struct error *e)
{
	if (!memcmp(actual, expected, count * sizeof *actual))
		return ERROR_NONE;
	size_t first = 0;
	while (!memcmp(&actual[first], &expected[first], sizeof *actual))
		++first;
	fprintf(stderr, "%s mismatch: first sample=%zu GPU=%.9g CPU=%.9g\n", name, first, (double)actual[first],
	        (double)expected[first]);
	return error_fail(e, "%s disagrees with CPU reference", name);
}

/** @brief The tuning check's picture: its extent, also the raster's. */
enum : unsigned {
	TUNING_WIDTH  = 7,                            //!< The width.
	TUNING_HEIGHT = 5,                            //!< The height.
	TUNING_PIXELS = TUNING_WIDTH * TUNING_HEIGHT, //!< The pixels.
};

/** @brief Returns a failed call's code from the function that makes the call. */
#define TRY(call) \
	do { \
		enum error_code const try_code_ = (call); \
		if (try_code_) \
			return try_code_; \
	} while (0)

/** @brief The tuning check with its device buffers.
 *
 * @param kernels The kernels.
 * @param device  Receives the input, the answer and the result; the caller frees them.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
check_tuning_with (struct native_kernels const *kernels,
                   struct buffer                device[3],
                   struct error                *e)
{
	struct geometry const g = {TUNING_WIDTH, TUNING_HEIGHT, TUNING_WIDTH, TUNING_HEIGHT, TUNING_HEIGHT, 0, 0,
	                           TUNING_WIDTH, TUNING_HEIGHT};
	float input[TUNING_PIXELS * 4];
	float model[TUNING_PIXELS * 3];
	for (unsigned p = 0; p < TUNING_PIXELS; ++p) {
		for (unsigned c = 0; c < 3; ++c) {
			input[p * 4 + c] = (float)((p * 3 + c * 7) % 31) / 32.0f;
			model[p * 3 + c] = input[p * 4 + c] + (float)((int)((p + c) % 7) - 3) / 64.0f;
		}
		input[p * 4 + 3] = 1;
	}
	TRY(buffer_init(&device[0], kernels, sizeof input, e));
	TRY(buffer_init(&device[1], kernels, sizeof model, e));
	TRY(buffer_init(&device[2], kernels, sizeof model, e));
	TRY(buffer_upload(&device[0], input, e));
	TRY(buffer_upload(&device[1], model, e));

	static struct native_tuning const states[] = {
		NATIVE_TUNING_DEFAULTS, {0, 1, 1, 0}, {1.75f, .25f, 2.5f, .375f}, {1, 0, 1, 0}, {1, 1, 0, 1},
	};
	float reference[TUNING_PIXELS * 3];
	float actual[TUNING_PIXELS * 3];
	for (size_t i = 0; i < sizeof states / sizeof *states; ++i) {
		TRY(reference_tune_neural_rgb(input, model, &g, &states[i], reference, e));
		TRY(native_kernels_tune(kernels, g, device[0].pointer, device[1].pointer, device[2].pointer, states[i],
		                        e));
		TRY(buffer_read(&device[2], actual, e));
		TRY(control_selftest_compare(actual, reference, TUNING_PIXELS * 3, "GPU native tuning", e));
	}
	printf("GPU control self-test: native tuning 5 states exact\n");
	fflush(stdout);
	return ERROR_NONE;
}

#undef TRY

/** @brief The native tuning in five states against its CPU reference.
 *
 * @param kernels The kernels.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
check_tuning (struct native_kernels const *kernels,
              struct error                *e)
{
	struct buffer device[3] = {0};
	enum error_code const code = check_tuning_with(kernels, device, e);
	for (size_t i = sizeof device / sizeof *device; i-- > 0;)
		buffer_fini(&device[i]);
	return code;
}

/** @brief A value noise texel.
 *
 * @param x The column.
 * @param y The row.
 * @return  A value in 0..1.
 */
static float
texture (unsigned x,
         unsigned y)
{
	uint32_t v = x * 0x45d9f3bu + y * 0x119de1f3u;
	v ^= v >> 16;
	v *= 0x45d9f3bu;
	v ^= v >> 16;
	return (float)(v & 65535u) / 65535.0f;
}

/** @brief The temporal check's picture and raster, and the floats of its host images: four RGBA
 *         rasters (original, shifted, unrelated, fallback), two RGB histories, two more RGBA
 *         rasters (expected, warped) and the picture's luma. */
enum : size_t {
	TEMPORAL_WIDTH         = 128, //!< The width of picture and raster.
	TEMPORAL_HEIGHT        = 96,  //!< The picture's height.
	TEMPORAL_PADDED_HEIGHT = 104, //!< The raster's height.
	/** @brief The raster's pixels. */
	TEMPORAL_PIXELS        = TEMPORAL_WIDTH * TEMPORAL_PADDED_HEIGHT,
	/** @brief The floats of every host image. */
	TEMPORAL_FLOATS        = TEMPORAL_PIXELS * (4 * 4 + 2 * 3 + 2 * 4) + TEMPORAL_WIDTH * TEMPORAL_HEIGHT,
};

/** @brief The temporal check's device buffers. */
struct temporal_buffers {
	struct buffer input;    //!< The frame.
	struct buffer fallback; //!< A pass input that a missing history falls back on.
	struct buffer first;    //!< The first pass's answer.
	struct buffer second;   //!< The second pass's.
};

/** @brief A frame of the temporal check's reconfiguration steps. */
struct temporal_step {
	struct geometry const *geometry; //!< Its geometry.
	char const            *failure;  //!< The words if its passes find the wrong history.
	unsigned               quality;  //!< Its motion quality.
	unsigned               grid;     //!< Its grid spacing.
	unsigned               passes;   //!< Its passes.
	unsigned               history;  //!< 1 when its passes find a history, else 0.
};

/** @brief Returns a failed call's code from the function that makes the call. */
#define TRY(call) \
	do { \
		enum error_code const try_code_ = (call); \
		if (try_code_) \
			return try_code_; \
	} while (0)

/** @brief Queues a pass's answer into its history, as a pass's last stage writes it.
 *
 * @param kernels  The kernels.
 * @param temporal The temporal state.
 * @param pass     The pass.
 * @param answer   The answer: the raster's RGB32F.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
store (struct native_kernels const *kernels,
       struct temporal_gpu         *temporal,
       unsigned                     pass,
       void const                  *answer,
       struct error                *e)
{
	void *target;
	TRY(temporal_gpu_target(temporal, pass, &target, e));
	int const result = kernels->api->hipMemcpyAsync(target, answer, TEMPORAL_PIXELS * 12, 3, kernels->stream);
	if (result)
		return hip_fail(e, kernels->api, result, "store GPU temporal self-test history");
	return ERROR_NONE;
}

/** @brief The temporal check with its host images, device buffers and temporal state.
 *
 * @param kernels  The kernels.
 * @param host     Room for TEMPORAL_FLOATS floats.
 * @param device   Receives the device buffers; the caller frees them.
 * @param temporal The temporal state, without buffers; the caller frees it.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
check_temporal_with (struct native_kernels const *kernels,
                     float                       *host,
                     struct temporal_buffers     *device,
                     struct temporal_gpu         *temporal,
                     struct error                *e)
{
	unsigned const width = TEMPORAL_WIDTH;
	unsigned const height = TEMPORAL_HEIGHT;
	size_t const pixels = TEMPORAL_PIXELS;
	struct geometry const g = {width, height, width, TEMPORAL_PADDED_HEIGHT, height, 0, 0, width, height};
	float *const original = host;
	float *const shifted = original + pixels * 4;
	float *const unrelated = shifted + pixels * 4;
	float *const fallback = unrelated + pixels * 4;
	float *const histories[2] = {fallback + pixels * 4, fallback + pixels * 7};
	float *const expected = histories[1] + pixels * 3;
	float *const warped = expected + pixels * 4;
	float *const gray = warped + pixels * 4;

	for (unsigned y = 0; y < height; ++y)
		for (unsigned x = 0; x < width; ++x)
			gray[y * width + x] = texture(x / 8, y / 8) * .3f + texture(x / 2, y / 2) * .5f +
			                      texture(x, y) * .2f;
	for (unsigned padded_y = 0; padded_y < TEMPORAL_PADDED_HEIGHT; ++padded_y) {
		unsigned const y = padded_y < height ? padded_y : 2 * height - 2 - padded_y;
		for (unsigned x = 0; x < width; ++x) {
			size_t const p = (size_t)padded_y * width + x;
			// The picture moved 6 pixels right and 3 up, sampled as the kernels sample it.
			float const current = kernel_math_temporal_sample(gray, (struct temporal_extent){width, height},
			                                                  (float)x - 6.0f, (float)y + 3.0f);
			float const still = gray[y * width + x];
			float const other = texture(x + 4096, y + 4096);
			for (unsigned c = 0; c < 3; ++c) {
				original[p * 4 + c] = still;
				shifted[p * 4 + c] = current;
				unrelated[p * 4 + c] = other;
				fallback[p * 4 + c] = .125f;
			}
			original[p * 4 + 3] = shifted[p * 4 + 3] = unrelated[p * 4 + 3] = fallback[p * 4 + 3] = 1;
			for (unsigned pass = 0; pass < 2; ++pass) {
				histories[pass][p * 3] = (float)x / 128.0f;
				histories[pass][p * 3 + 1] = (float)y / 128.0f;
				histories[pass][p * 3 + 2] = pass ? .625f : .875f;
			}
		}
	}
	TRY(buffer_init(&device->input, kernels, pixels * 4 * sizeof (float), e));
	TRY(buffer_init(&device->fallback, kernels, pixels * 4 * sizeof (float), e));
	TRY(buffer_init(&device->first, kernels, pixels * 3 * sizeof (float), e));
	TRY(buffer_init(&device->second, kernels, pixels * 3 * sizeof (float), e));
	TRY(buffer_upload(&device->input, original, e));
	TRY(buffer_upload(&device->fallback, fallback, e));
	TRY(buffer_upload(&device->first, histories[0], e));
	TRY(buffer_upload(&device->second, histories[1], e));
	void const *const answers[] = {device->first.pointer, device->second.pointer};

	// The first frame has no history; the second, the same picture, finds each pass's own.
	TRY(temporal_gpu_begin(temporal, device->input.pointer, &g, 2, 2, 2, false, e));
	for (unsigned pass = 0; pass < 2; ++pass) {
		void *history;
		TRY(temporal_gpu_history(temporal, pass, device->fallback.pointer, &history, e));
		TRY(require(!history, "GPU temporal first frame returned uninitialized history", e));
		TRY(store(kernels, temporal, pass, answers[pass], e));
	}
	TRY(temporal_gpu_end(temporal, e));
	TRY(temporal_gpu_begin(temporal, device->input.pointer, &g, 2, 2, 2, false, e));
	for (unsigned pass = 0; pass < 2; ++pass) {
		void *history;
		TRY(temporal_gpu_history(temporal, pass, device->fallback.pointer, &history, e));
		TRY(control_selftest_read(kernels, history, warped, pixels * 4, e));
		for (size_t p = 0; p < pixels; ++p) {
			for (unsigned c = 0; c < 3; ++c)
				expected[p * 4 + c] = histories[pass][p * 3 + c];
			expected[p * 4 + 3] = 1;
		}
		TRY(control_selftest_compare(warped, expected, pixels * 4, "GPU static temporal history", e));
		TRY(store(kernels, temporal, pass, answers[pass], e));
	}
	TRY(temporal_gpu_end(temporal, e));

	TRY(buffer_upload(&device->input, shifted, e));
	TRY(temporal_gpu_begin(temporal, device->input.pointer, &g, 2, 2, 2, false, e));
	bool cut;
	TRY(temporal_gpu_cut_rejected(temporal, &cut, e));
	TRY(require(!cut, "GPU temporal rejected the known translated frame as a cut", e));
	for (unsigned pass = 0; pass < 2; ++pass) {
		void *history;
		TRY(temporal_gpu_history(temporal, pass, device->fallback.pointer, &history, e));
		TRY(control_selftest_read(kernels, history, warped, pixels * 4, e));
		unsigned good = 0;
		unsigned tested = 0;
		for (unsigned y = 16; y + 16 < height; ++y) {
			for (unsigned x = 16; x + 16 < width; ++x) {
				size_t const p = ((size_t)y * width + x) * 4;
				++tested;
				good += fabsf(warped[p] * 128.0f - (float)(x - 6)) < .75f &&
				        fabsf(warped[p + 1] * 128.0f - (float)(y + 3)) < .75f &&
				        warped[p + 2] == (pass ? .625f : .875f) && warped[p + 3] == 1;
			}
		}
		printf("GPU control self-test: temporal pass %u static exact; translated history %u/%u within 0.75 "
		       "pixels\n", pass + 1, good, tested);
		fflush(stdout);
		TRY(require(good * 10 > tested * 8, "GPU temporal translation/history isolation check failed", e));
		TRY(store(kernels, temporal, pass, answers[pass], e));
	}
	TRY(temporal_gpu_end(temporal, e));

	// An unrelated frame is a scene cut: every pixel falls back to the pass input (here the frame
	// itself, as in the first pass): the no-history input.
	TRY(buffer_upload(&device->input, unrelated, e));
	TRY(temporal_gpu_begin(temporal, device->input.pointer, &g, 2, 2, 2, false, e));
	TRY(temporal_gpu_cut_rejected(temporal, &cut, e));
	TRY(require(cut, "GPU temporal missed a scene cut", e));
	for (unsigned pass = 0; pass < 2; ++pass) {
		void *history;
		TRY(temporal_gpu_history(temporal, pass, device->input.pointer, &history, e));
		TRY(control_selftest_read(kernels, history, warped, pixels * 4, e));
		TRY(control_selftest_compare(warped, unrelated, pixels * 4, "GPU temporal scene cut fallback", e));
		TRY(store(kernels, temporal, pass, answers[pass], e));
	}
	TRY(temporal_gpu_end(temporal, e));
	printf("GPU control self-test: temporal scene cut rejects all history\n");

	// A new source size keeps the buffers and the history; a new placement of the picture in the
	// network raster drops the history; a new quality, grid or pass count reallocates and drops it
	// too. Each reallocation is followed by a frame that runs with the new sizes.
	struct geometry resized = g;
	resized.source_width /= 2;
	resized.source_height /= 2;
	struct geometry moved = resized;
	moved.x = 8;
	moved.fit_width -= 16;
	struct temporal_step const steps[] = {
		{&resized, "GPU temporal dropped history for a new source size", 2, 2, 2, 1},
		{&moved, "GPU temporal kept history for a moved picture", 2, 2, 2, 0},
		{&moved, "GPU temporal dropped history for a steady moved picture", 2, 2, 2, 1},
		{&moved, "GPU temporal kept history across a new quality", 1, 2, 2, 0},
		{&moved, "GPU temporal dropped history after a new quality", 1, 2, 2, 1},
		{&moved, "GPU temporal kept history across a new grid", 1, 1, 2, 0},
		{&moved, "GPU temporal dropped history after a new grid", 1, 1, 2, 1},
		{&moved, "GPU temporal kept history across a new pass count", 1, 1, 1, 0},
		{&moved, "GPU temporal dropped history after a new pass count", 1, 1, 1, 1},
	};
	for (size_t i = 0; i < sizeof steps / sizeof *steps; ++i) {
		struct temporal_step const *const step = &steps[i];
		TRY(temporal_gpu_begin(temporal, device->input.pointer, step->geometry, step->quality, step->grid,
		                       step->passes, false, e));
		for (unsigned pass = 0; pass < step->passes; ++pass) {
			void *history;
			TRY(temporal_gpu_history(temporal, pass, device->input.pointer, &history, e));
			TRY(require((history != nullptr) == step->history, step->failure, e));
			void *target;
			TRY(temporal_gpu_target(temporal, pass, &target, e));
		}
		TRY(temporal_gpu_end(temporal, e));
	}
	int const result = kernels->api->hipStreamSynchronize(kernels->stream);
	if (result)
		return hip_fail(e, kernels->api, result, "complete GPU temporal reconfiguration frames");
	printf("GPU control self-test: temporal history survives a new source size; "
	       "a moved picture, quality, grid or pass count drops it\n");
	fflush(stdout);
	return ERROR_NONE;
}

#undef TRY

/** @brief The motion history: static frames find their own history bit for bit, a translated one
 *         its history moved, an unrelated one none, and new sizes and placements keep or drop it.
 *
 * @param kernels The kernels.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
check_temporal (struct native_kernels const *kernels,
                struct error                *e)
{
	float *host = malloc(TEMPORAL_FLOATS * sizeof *host);
	if (!host)
		return error_fail(e, "out of memory");
	struct temporal_buffers device = {0};
	struct temporal_gpu temporal = temporal_gpu(kernels);
	enum error_code const code = check_temporal_with(kernels, host, &device, &temporal, e);
	temporal_gpu_fini(&temporal);
	buffer_fini(&device.second);
	buffer_fini(&device.first);
	buffer_fini(&device.fallback);
	buffer_fini(&device.input);
	free(host);
	host = nullptr;
	return code;
}

/** @brief The codec check's small source, of 7 x 5 FP16 pixels. */
enum : unsigned {
	CODEC_WIDTH  = 7,                              //!< Its width.
	CODEC_HEIGHT = 5,                              //!< Its height.
	CODEC_BYTES  = CODEC_WIDTH * CODEC_HEIGHT * 8, //!< Its bytes.
};

/** @brief The codec check's large source, of 40 x 3000 pixels: more texels than pixels in the
 *         fit, 4 and 4.17 of them per pixel. */
enum : size_t {
	LARGE_WIDTH  = 40,                             //!< Its width.
	LARGE_HEIGHT = 3000,                           //!< Its height.
	LARGE_BYTES  = LARGE_WIDTH * LARGE_HEIGHT * 8, //!< Its bytes as FP16, which RGBA8 reads half of.
};

/** @brief What the codec check holds. */
struct codec_fixture {
	struct codec_gpu codec;  //!< The codec.
	struct buffer    input;  //!< The network's input.
	struct buffer    rgb;    //!< A network answer.
	struct buffer    large;  //!< The large source's network input.
	float           *floats; //!< The host floats: a reference, a read back and an answer.
	uint8_t         *source; //!< The large source.
};

/** @brief Returns a failed call's code from the function that makes the call. */
#define TRY(call) \
	do { \
		enum error_code const try_code_ = (call); \
		if (try_code_) \
			return try_code_; \
	} while (0)

/** @brief Runs one FP16 frame of the small source through the codec.
 *
 * @param codec        The codec.
 * @param proxy        The source.
 * @param g            Its geometry.
 * @param device_input The network's input.
 * @param device_rgb   The network's answer.
 * @param decoded      Receives the decoded frame.
 * @param rejected     Receives whether codec_gpu_finish() rejected the frame.
 * @param e            Receives the words for what stopped it, or nullptr.
 * @return             ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
codec_rejects (struct codec_gpu      *codec,
               uint8_t const         *proxy,
               struct geometry const *g,
               void                  *device_input,
               void                  *device_rgb,
               uint8_t               *decoded,
               bool                  *rejected,
               struct error          *e)
{
	TRY(codec_gpu_encode(codec, proxy, g, device_input, true, false, e));
	TRY(codec_gpu_decode(codec, device_rgb, decoded, e));
	enum error_code const code = codec_gpu_finish(codec, e);
	*rejected = code == ERROR_REJECTED;
	return *rejected ? ERROR_NONE : code;
}

/** @brief The codec check with what it holds.
 *
 * @param kernels The kernels.
 * @param f       Receives what the check holds; the caller frees it.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
check_codec_with (struct native_kernels const *kernels,
                  struct codec_fixture        *f,
                  struct error                *e)
{
	struct geometry g;
	TRY(geometry_init(&g, CODEC_WIDTH, CODEC_HEIGHT, 720, e));
	struct geometry large;
	TRY(geometry_init(&large, LARGE_WIDTH, LARGE_HEIGHT, 720, e));
	size_t const pixels = (size_t)g.width * g.height;
	size_t const large_pixels = (size_t)large.width * large.height;
	size_t const most = pixels < large_pixels ? large_pixels : pixels;
	f->floats = malloc(most * (4 + 4 + 3) * sizeof *f->floats);
	f->source = malloc(LARGE_BYTES);
	if (!f->floats || !f->source)
		return error_fail(e, "out of memory");
	float *const reference = f->floats;
	float *const actual = reference + most * 4;
	float *const model = actual + most * 4;

	uint8_t proxy[CODEC_BYTES];
	static uint16_t const half_samples[] = {0xb800, 0x0000, 0x3400, 0x3a00, 0x3e00, 0x4000};
	for (unsigned p = 0; p < CODEC_WIDTH * CODEC_HEIGHT; ++p) {
		for (unsigned c = 0; c < 4; ++c) {
			uint16_t const value = c == 3 ? (p & 1 ? 0x3800 : 0x3c00) : half_samples[(p * 3 + c) % 6];
			memcpy(proxy + p * 8 + c * 2, &value, sizeof value);
		}
	}
	TRY(buffer_init(&f->input, kernels, pixels * 4 * sizeof (float), e));
	TRY(buffer_init(&f->rgb, kernels, pixels * 3 * sizeof (float), e));
	TRY(codec_gpu_init(&f->codec, kernels, e));
	TRY(reference_encode_proxy(proxy, &g, true, reference, e));
	TRY(codec_gpu_encode(&f->codec, proxy, &g, f->input.pointer, true, false, e));
	TRY(buffer_read(&f->input, actual, e));
	TRY(control_selftest_compare(actual, reference, pixels * 4, "GPU FP16 proxy encode", e));

	// A constant signed/extended-range RGB fixture makes all decode resampling exact and verifies
	// that the FP16 route retains alpha and never UNORM-clamps.
	for (size_t p = 0; p < pixels; ++p) {
		model[p * 3] = -.25f;
		model[p * 3 + 1] = 1.5f;
		model[p * 3 + 2] = .5f;
	}
	TRY(buffer_upload(&f->rgb, model, e));
	uint8_t decoded[CODEC_BYTES];
	uint8_t expected[CODEC_BYTES];
	TRY(codec_gpu_decode(&f->codec, f->rgb.pointer, decoded, e));
	TRY(codec_gpu_finish(&f->codec, e));
	TRY(reference_decode_neural_proxy(proxy, &g, true, model, expected, e));
	TRY(require(!memcmp(decoded, expected, sizeof expected),
	            "GPU FP16 proxy decode disagrees on exact signed/extended-range fixture", e));
	// Decode overwrites the uploaded proxy's RGB in place; a repeat must agree.
	uint8_t repeated[CODEC_BYTES];
	TRY(codec_gpu_decode(&f->codec, f->rgb.pointer, repeated, e));
	TRY(codec_gpu_finish(&f->codec, e));
	TRY(require(!memcmp(repeated, expected, sizeof expected), "GPU FP16 proxy decode changed when repeated", e));

	// The encoder (a NaN proxy sample) and the decoder (an answer beyond binary16) each reject the
	// frame, and every encode starts clean.
	uint8_t poisoned[CODEC_BYTES];
	memcpy(poisoned, proxy, sizeof poisoned);
	uint16_t const nan = 0x7e00;
	memcpy(poisoned, &nan, sizeof nan);
	bool rejected;
	TRY(codec_rejects(&f->codec, poisoned, &g, f->input.pointer, f->rgb.pointer, decoded, &rejected, e));
	TRY(require(rejected, "GPU codec accepted a NaN FP16 proxy sample", e));
	TRY(codec_rejects(&f->codec, proxy, &g, f->input.pointer, f->rgb.pointer, decoded, &rejected, e));
	TRY(require(!rejected, "GPU codec rejection outlived its frame", e));
	for (size_t i = 0; i < pixels * 3; ++i)
		model[i] = 65536.0f;
	TRY(buffer_upload(&f->rgb, model, e));
	TRY(codec_rejects(&f->codec, proxy, &g, f->input.pointer, f->rgb.pointer, decoded, &rejected, e));
	TRY(require(rejected, "GPU codec accepted an answer beyond binary16", e));

	for (size_t p = 0; p < pixels; ++p) {
		model[p * 3] = (float)((int)(p % 29) - 3) / 16.0f;
		model[p * 3 + 1] = (float)((p / g.width) % 23) / 16.0f;
		model[p * 3 + 2] = (float)(p % 17) / 19.0f;
	}
	TRY(buffer_upload(&f->rgb, model, e));
	for (unsigned i = 0; i < 2; ++i) {
		bool const precision16 = !i;
		TRY(reference_feedback_neural_rgb(model, &g, precision16, reference, e));
		TRY(codec_gpu_feedback(&f->codec, f->rgb.pointer, f->input.pointer, precision16, e));
		TRY(buffer_read(&f->input, actual, e));
		TRY(control_selftest_compare(actual, reference, pixels * 4,
		                             precision16 ? "GPU FP16 feedback" : "GPU UNORM8 feedback", e));
	}
	printf("GPU control self-test: FP16 proxy encode, decode, rejection and 16/8-bit feedback exact\n");

	// A source larger than the fit takes the area-weighted encode, which no other check reaches.
	for (size_t i = 0; i < LARGE_BYTES / 2; ++i) {
		uint16_t const value = (uint16_t)(0x2c00u + (i * 37u) % 0x1000u); // 1/16 up to 1
		memcpy(f->source + i * 2, &value, sizeof value);
	}
	TRY(buffer_init(&f->large, kernels, large_pixels * 4 * sizeof (float), e));
	for (unsigned i = 0; i < 2; ++i) {
		bool const fp16 = i;
		TRY(reference_encode_proxy(f->source, &large, fp16, reference, e));
		TRY(codec_gpu_encode(&f->codec, f->source, &large, f->large.pointer, fp16, false, e));
		TRY(buffer_read(&f->large, actual, e));
		TRY(control_selftest_compare(actual, reference, large_pixels * 4,
		                             fp16 ? "GPU FP16 downscaling encode" : "GPU RGBA8 downscaling encode", e));
	}
	printf("GPU control self-test: RGBA8 and FP16 area-weighted downscaling encode exact\n");
	fflush(stdout);
	return ERROR_NONE;
}

#undef TRY

/** @brief The codec's FP16 encode, decode, rejection and feedback, and its downscaling encode,
 *         against the CPU references.
 *
 * @param kernels The kernels.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or the code of what stopped it.
 */
static enum error_code
check_codec (struct native_kernels const *kernels,
             struct error                *e)
{
	struct codec_fixture f = {0};
	enum error_code const code = check_codec_with(kernels, &f, e);
	buffer_fini(&f.large);
	codec_gpu_fini(&f.codec);
	buffer_fini(&f.rgb);
	buffer_fini(&f.input);
	free(f.source);
	f.source = nullptr;
	free(f.floats);
	f.floats = nullptr;
	return code;
}

enum error_code
control_selftest_run (struct native_kernels const *kernels,
                      struct error                *e)
{
	enum error_code code = check_tuning(kernels, e);
	if (code)
		return code;
	code = check_temporal(kernels, e);
	if (code)
		return code;
	return check_codec(kernels, e);
}
