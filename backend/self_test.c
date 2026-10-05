/** @file
 *
 * The engines' self-tests: vulkan_engine_self_test() and hip_engine_self_test(), built for size
 * apart from the engines' frame paths.
 */
// SPDX-License-Identifier: MIT
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine.h"
#include "error.h"
#include "files.h"
#include "geometry.h"
#include "hip_engine.h"
#include "options.h"
#include "processing.h"
#include "shm_protocol.h"
#include "vulkan_engine.h"
#include "vulkan_network.h"

/** @brief Draws opaque RGBA8: red and green ramps over 32-pixel blue checks of two levels.
 *
 * @param rgba Receives w x h pixels.
 * @param w    The width, at least 2.
 * @param h    The height, at least 2.
 * @param on   The blue of one check.
 * @param off  The blue of the other.
 */
static void
gradient (uint8_t  *rgba,
          uint32_t  w,
          uint32_t  h,
          uint8_t   on,
          uint8_t   off)
{
	for (uint32_t y = 0; y < h; ++y) {
		for (uint32_t x = 0; x < w; ++x) {
			uint8_t *const p = &rgba[((size_t)y * w + x) * 4];
			p[0] = (uint8_t)(x * 255 / (w - 1));
			p[1] = (uint8_t)(y * 255 / (h - 1));
			p[2] = ((x / 32 ^ y / 32) & 1) ? on : off;
			p[3] = 255;
		}
	}
}

/** @brief Writes RGBA8 as a binary PPM file, created or truncated: the header, then each pixel's RGB.
 *
 * @param path The file.
 * @param rgba The image: w x h pixels.
 * @param w    Its width.
 * @param h    Its height.
 * @param e    Receives the words for what failed, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
write_ppm (char const    *path,
           uint8_t const *rgba,
           uint32_t       w,
           uint32_t       h,
           struct error  *e)
{
	char header[sizeof "P6\n4294967295 4294967295\n255\n"];
	int const length = snprintf(header, sizeof header, "P6\n%" PRIu32 " %" PRIu32 "\n255\n", w, h);
	if (length < 0 || length >= (int)sizeof header)
		return error_fail(e, "no PPM header for %" PRIu32 "x%" PRIu32, w, h);
	size_t const header_bytes = (size_t)length;
	size_t const pixels = (size_t)w * h;
	size_t const bytes = header_bytes + pixels * 3;
	uint8_t *image = malloc(bytes);
	if (!image)
		return error_fail(e, "out of memory");
	memcpy(image, header, header_bytes);
	uint8_t *const rgb = &image[header_bytes];
	for (size_t i = 0; i < pixels; ++i)
		memcpy(&rgb[i * 3], &rgba[i * 4], 3);
	enum error_code const code = files_write(path, image, bytes, e);
	free(image);
	image = nullptr;
	return code;
}

/** @brief The Vulkan self-test's runs, with its buffers made.
 *
 * @param engine The engine, prepared.
 * @param o      The options.
 * @param input  The gradient: w x h pixels.
 * @param output Receives each run's answer.
 * @param first  Receives the first answer that was not dropped.
 * @param w      The tier's width.
 * @param h      Its height.
 * @param passes The passes.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
vulkan_runs (struct vulkan_engine *engine,
             struct options const *o,
             uint8_t const        *input,
             uint8_t              *output,
             uint8_t              *first,
             uint32_t              w,
             uint32_t              h,
             uint32_t              passes,
             struct error         *e)
{
	size_t const bytes = (size_t)w * h * 4;
	struct engine_frames const io = {input, output, -1};
	struct processing_settings const settings = processing_settings();
	uint32_t dropped = 0;
	bool kept = false;
	for (uint32_t run = 0; run < o->self_test_runs; ++run) {
		enum error_code const code = vulkan_engine_infer(engine, &io, w, h, passes, &settings, nullptr, e);
		if (code) {
			if (code != ERROR_DROPPED)
				return code;
			if (++dropped > o->self_test_drops)
				return error_fail(e, "self-test run %" PRIu32 " dropped, one more than "
				                  "--self-test-drops %" PRIu32 " allows: %s", run + 1,
				                  o->self_test_drops, VULKAN_NETWORK_DROPPED);
		} else if (!kept) {
			memcpy(first, output, bytes);
			kept = true;
		} else if (memcmp(output, first, bytes)) {
			return error_fail(e, "self-test run %" PRIu32 " differs from the first that was not dropped",
			                  run + 1);
		}
	}
	if (!kept)
		return error_fail(e, "self-test: every run was dropped: " VULKAN_NETWORK_DROPPED);
	size_t changed = 0;
	for (size_t i = 0; i < bytes; ++i)
		changed += (i % 4 != 3) && input[i] != output[i];
	if (!changed)
		return error_fail(e, "self-test: the network left the input unchanged");
	if (o->output_length && write_ppm(o->output, output, w, h, nullptr))
		return error_fail(e, "write %s", o->output);
	fprintf(stderr, "Vulkan self-test PASS: %" PRIu32 " identical runs at %" PRIu32 "x%" PRIu32 "; dropped=%"
	        PRIu32 "; changed_components=%zu\ntier=%" PRIu32 "; passes=%" PRIu32 "; upload_ms=%.3f; "
	        "network_ms=%.3f; readback_ms=%.3f\n", o->self_test_runs - dropped, w, h, dropped, changed,
	        engine->tier, passes, engine->times.upload_ms, engine->times.inference_ms,
	        engine->times.readback_ms);
	return ERROR_NONE;
}

enum error_code
vulkan_engine_self_test (struct vulkan_engine *engine,
                         struct options const *o,
                         struct error         *e)
{
	uint32_t const wanted = o->passes ? o->passes : kNativeDefaultPasses;
	uint32_t const passes = VULKAN_ENGINE_MAX_PASSES < wanted ? VULKAN_ENGINE_MAX_PASSES : wanted;
	uint32_t const w = ShmNativeTier(engine->tier)->width;
	uint32_t const h = engine->tier;
	size_t const bytes = (size_t)w * h * 4;
	uint8_t *input = malloc(bytes);
	uint8_t *output = calloc(bytes, 1);
	uint8_t *first = malloc(bytes);
	enum error_code code;
	if (input && output && first) {
		gradient(input, w, h, 200, 60);
		code = vulkan_runs(engine, o, input, output, first, w, h, passes, e);
	} else {
		code = error_fail(e, "out of memory");
	}
	free(first);
	first = nullptr;
	free(output);
	output = nullptr;
	free(input);
	input = nullptr;
	return code;
}

/** @brief The HIP self-test's buffers. */
struct hip_runs {
	uint8_t *input;        //!< The gradient.
	uint8_t *output;       //!< Each run's answer.
	uint8_t *first_output; //!< The first run's answer.
	float   *first_raw;    //!< The first run's raw network answer.
};

/** @brief Checks the first run's raw network answer and prints its range.
 *
 * @param raw   The answer.
 * @param count Its floats, at least 1.
 * @param e     Receives the words for what failed, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED for a value that is not finite.
 */
static enum error_code
print_range (float const  *raw,
             size_t        count,
             struct error *e)
{
	float minimum = raw[0];
	float maximum = raw[0];
	size_t below_zero = 0;
	size_t above_one = 0;
	for (size_t i = 0; i < count; ++i) {
		float const value = raw[i];
		if (!isfinite(value))
			return error_fail(e, "network produced nonfinite values");
		minimum = value < minimum ? value : minimum;
		maximum = maximum < value ? value : maximum;
		below_zero += value < 0.0f;
		above_one += value > 1.0f;
	}
	printf("raw network RGB range=%.9g..%.9g below_zero=%zu above_one=%zu samples=%zu\n",
	       (double)minimum, (double)maximum, below_zero, above_one, count);
	return ERROR_NONE;
}

/** @brief Reports how a later run's raw network answer differs from the first's.
 *
 * @param raw       The later run's answer.
 * @param first_raw The first run's answer.
 * @param count     Their floats.
 * @param g         The geometry, whose raster they cover.
 * @param run       The later run.
 * @param repeats   The runs.
 * @param e         Receives the words, or nullptr.
 * @return          ERROR_FAILED.
 */
static enum error_code
raw_differs (float const           *raw,
             float const           *first_raw,
             size_t                 count,
             struct geometry const *g,
             uint32_t               run,
             uint32_t               repeats,
             struct error          *e)
{
	size_t first = count;
	size_t different = 0;
	float maximum = 0;
	uint32_t first_before = 0;
	uint32_t first_after = 0;
	for (size_t i = 0; i < count; ++i) {
		uint32_t before;
		uint32_t after;
		memcpy(&before, &first_raw[i], sizeof before);
		memcpy(&after, &raw[i], sizeof after);
		if (before == after)
			continue;
		if (first == count) {
			first = i;
			first_before = before;
			first_after = after;
		}
		++different;
		float const error = fabsf(raw[i] - first_raw[i]);
		maximum = maximum < error ? error : maximum;
	}
	fflush(stdout);
	fprintf(stderr,
	        "network repeat %" PRIu32 "/%" PRIu32 " differs from first identical input: samples=%zu/%zu "
	        "max_abs_error=%.9g; first x=%zu y=%zu channel=%zu "
	        "first=%.9g (0x%08" PRIx32 ") repeat=%.9g (0x%08" PRIx32 ")\n",
	        run + 1, repeats, different, count, (double)maximum,
	        (first / 3) % g->width, (first / 3) / g->width, first % 3,
	        (double)first_raw[first], first_before, (double)raw[first], first_after);
	return error_fail(e, "network is nondeterministic with identical input and fixed seed");
}

/** @brief The HIP self-test's runs, with its buffers made.
 *
 * @param engine The engine, prepared.
 * @param o      The options.
 * @param b      The buffers.
 * @param g      The gradient's geometry at the engine's tier.
 * @param w      The gradient's width.
 * @param h      Its height.
 * @param passes The passes.
 * @param e      Receives the words for what failed, or nullptr.
 * @return       ERROR_NONE, or the code of what failed.
 */
static enum error_code
hip_runs (struct hip_engine     *engine,
          struct options const  *o,
          struct hip_runs       *b,
          struct geometry const *g,
          uint32_t               w,
          uint32_t               h,
          uint32_t               passes,
          struct error          *e)
{
	uint32_t const repeats = o->self_test_runs;
	size_t const bytes = (size_t)w * h * 4;
	struct engine_frames const io = {b->input, b->output, -1};
	struct processing_settings const settings = processing_settings();
	// Only the first run checks the codec against the CPU reference, so the
	// later runs time the production path; each must reproduce the first.
	for (uint32_t run = 0; run < repeats; ++run) {
		enum error_code code = hip_engine_frame(engine, &io, w, h, passes, &settings, nullptr, !run, e);
		if (code)
			return code;
		code = hip_engine_read_raw(engine, e);
		if (code)
			return code;
		float const *const raw = engine->neural;
		size_t const count = engine->neural_count;
		if (!run) {
			b->first_raw = malloc(count * sizeof *b->first_raw);
			if (!b->first_raw)
				return error_fail(e, "out of memory");
			memcpy(b->first_raw, raw, count * sizeof *b->first_raw);
			memcpy(b->first_output, b->output, bytes);
			code = print_range(raw, count, e);
			if (code)
				return code;
		} else if (memcmp(raw, b->first_raw, count * sizeof *raw)) {
			return raw_differs(raw, b->first_raw, count, g, run, repeats, e);
		} else {
			size_t first = 0;
			while (first < bytes && b->first_output[first] == b->output[first])
				++first;
			if (first < bytes) {
				fflush(stdout);
				fprintf(stderr,
				        "network repeat %" PRIu32 "/%" PRIu32 " decoded output differs from the first "
				        "run: first x=%zu y=%zu channel=%zu first=%u repeat=%u\n",
				        run + 1, repeats, (first / 4) % w, (first / 4) / w, first % 4,
				        (unsigned)b->first_output[first], (unsigned)b->output[first]);
				return error_fail(e, "decoded output differs from the verified first run");
			}
		}
		printf("network repeat %" PRIu32 "/%" PRIu32 ": %s; passes=%" PRIu32 " upload_ms=%.3f network_ms=%.3f "
		       "readback_ms=%.3f\n", run + 1, repeats, run ? "raw FP32 and output bit-identical" : "baseline",
		       passes, engine->times.upload_ms, engine->times.inference_ms, engine->times.readback_ms);
		fflush(stdout);
	}
	uint64_t hash = UINT64_C(14695981039346656037);
	unsigned low = 255;
	unsigned high = 0;
	size_t changed = 0;
	for (size_t i = 0; i < bytes; ++i) {
		hash = (hash ^ b->output[i]) * UINT64_C(1099511628211);
		if ((i & 3) == 3)
			continue;
		low = b->output[i] < low ? b->output[i] : low;
		high = high < b->output[i] ? b->output[i] : high;
		changed += b->output[i] != b->input[i];
	}
	if (high <= low || !changed)
		return error_fail(e, "self-test returned constant or unchanged RGB output");
	if (o->output_length && write_ppm(o->output, b->output, w, h, nullptr))
		return error_fail(e, "write self-test PPM");
	printf("real-network self-test PASS: finite output; %" PRIu32 " identical-input runs bit-exact; "
	       "RGB range=%u..%u; changed_components=%zu; fnv1a64=%016" PRIx64 "\n",
	       repeats, low, high, changed, hash);
	printf("tier=%" PRIu32 "; passes=%" PRIu32 "; blocks=%s; upload_ms=%.3f; network_ms=%.3f; readback_ms=%.3f\n",
	       engine->tier, passes, o->performance ? "upstream performance preset" : "all 71",
	       engine->times.upload_ms, engine->times.inference_ms, engine->times.readback_ms);
	return ERROR_NONE;
}

enum error_code
hip_engine_self_test (struct hip_engine    *engine,
                      struct options const *o,
                      struct error         *e)
{
	uint32_t const passes = o->passes ? o->passes : kNativeDefaultPasses;
	uint32_t const w = 640;
	uint32_t const h = 360;
	struct geometry g;
	enum error_code code = geometry_init(&g, w, h, engine->tier, e);
	if (code)
		return code;
	size_t const bytes = (size_t)w * h * 4;
	struct hip_runs b = {0};
	b.input = malloc(bytes);
	b.output = calloc(bytes, 1);
	b.first_output = malloc(bytes);
	if (b.input && b.output && b.first_output) {
		gradient(b.input, w, h, 192, 64);
		code = hip_runs(engine, o, &b, &g, w, h, passes, e);
	} else {
		code = error_fail(e, "out of memory");
	}
	free(b.first_raw);
	b.first_raw = nullptr;
	free(b.first_output);
	b.first_output = nullptr;
	free(b.output);
	b.output = nullptr;
	free(b.input);
	b.input = nullptr;
	return code;
}
