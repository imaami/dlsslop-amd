/** @file
 *
 * Host test of the native tuning's CPU reference (reference_tune_neural_rgb()) and of its check
 * (tuning_validate()): the defaults, intensity, tone, structure and sharpness, the viewport, the
 * errors, and the goldens of its outputs (golden.h).
 */
// SPDX-License-Identifier: MIT
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "golden.h"
#include "reference.h"
#include "tuning.h"

/** @brief The control self-test's states (control_selftest.c), the default one first. */
static struct native_tuning const states[] = {
	NATIVE_TUNING_DEFAULTS, {0, 1, 1, 0}, {1.75f, .25f, 2.5f, .375f}, {1, 0, 1, 0}, {1, 1, 0, 1},
};

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param condition The condition.
 * @param message   The message.
 */
static void
require (bool        condition,
         char const *message)
{
	if (condition)
		return;
	fprintf(stderr, "native tuning test: %s\n", message);
	exit(1);
}

/** @brief Ends the test without memory for a value.
 *
 * @param p The value's memory, or nullptr.
 * @return  @a p.
 */
static void *
allocated (void *p)
{
	require(p, "out of memory");
	return p;
}

/** @brief Tunes an answer that the test expects to be tuned.
 *
 * @param input  The pass's input, RGBA32F.
 * @param model  The pass's answer, RGB32F.
 * @param g      The geometry.
 * @param output Receives the tuned answer, RGB32F.
 * @param tuning The tuning.
 */
static void
tune (float const                *input,
      float const                *model,
      struct geometry const      *g,
      float                      *output,
      struct native_tuning const *tuning)
{
	require(!reference_tune_neural_rgb(input, model, g, tuning, output, nullptr), "valid tuning refused");
}

/** @brief Ends the test with a message unless a value is within 1e-6 of what it should be.
 *
 * @param actual   The value.
 * @param expected What it should be.
 * @param message  The message.
 */
static void
require_near (float       actual,
              float       expected,
              char const *message)
{
	require(fabsf(actual - expected) < 1.0e-6f, message);
}

/** @brief Sets every float of an array to one value.
 *
 * @param v     The array.
 * @param count Its floats.
 * @param value The value.
 */
static void
fill (float  *v,
      size_t  count,
      float   value)
{
	for (size_t i = 0; i < count; ++i)
		v[i] = value;
}

/** @brief How many of the reference's outputs, one per state, moved from their goldens.
 *
 * @param fixture The fixture's name in a message.
 * @param g       The geometry.
 * @param input   The pass's input, RGBA32F.
 * @param model   The pass's answer, RGB32F.
 * @param goldens The goldens, one per state.
 * @return        The number that moved.
 */
static unsigned
moved_goldens (char const            *fixture,
               struct geometry const *g,
               float const           *input,
               float const           *model,
               uint64_t const         goldens[static sizeof states / sizeof *states])
{
	size_t const floats = (size_t)g->width * g->height * 3;
	float *output = allocated(malloc(floats * sizeof *output));
	unsigned moved = 0;
	for (size_t i = 0; i < sizeof states / sizeof *states; ++i) {
		tune(input, model, g, output, &states[i]);
		moved += !golden_check(output, floats * sizeof *output, goldens[i], "tune_neural_rgb %s state %zu",
		                       fixture, i);
	}
	free(output);
	output = nullptr;
	return moved;
}

/** @brief The reference's output over the control self-test's fixture and over a random one, bit
 *         for bit. */
static void
goldens (void)
{
	struct geometry const self_test = {7, 5, 7, 5, 5, 0, 0, 7, 5};
	float input[35 * 4], model[35 * 3];
	for (unsigned p = 0; p < 35; ++p) {
		for (unsigned c = 0; c < 3; ++c) {
			input[p * 4 + c] = (float)((p * 3 + c * 7) % 31) / 32.0f;
			model[p * 3 + c] = input[p * 4 + c] + (float)((int)((p + c) % 7) - 3) / 64.0f;
		}
		input[p * 4 + 3] = 1;
	}
	unsigned moved = moved_goldens("self-test", &self_test, input, model, (uint64_t const[]){
		0xb72c2574ee5214f8u, 0x5b4b322ec30cbd50u, 0x5647251848209bf3u, 0xf65855719c16435fu,
		0x6e618c8b4da1a9dbu});
	// 640x480 at the 720 tier: pillarboxed, and padded below.
	struct geometry const pillarboxed = {640, 480, 1280, 768, 720, 160, 0, 960, 720};
	size_t const pixels = (size_t)pillarboxed.width * pillarboxed.height;
	float *large_input = allocated(malloc(pixels * 4 * sizeof *large_input));
	float *large_model = allocated(malloc(pixels * 3 * sizeof *large_model));
	for (size_t p = 0; p < pixels; ++p) {
		for (unsigned c = 0; c < 3; ++c) {
			large_input[p * 4 + c] = golden_unit(1, p * 3 + c);
			large_model[p * 3 + c] = large_input[p * 4 + c] + (golden_unit(2, p * 3 + c) - 0.5f) * 0.5f;
		}
		large_input[p * 4 + 3] = 1;
	}
	moved += moved_goldens("640x480 720", &pillarboxed, large_input, large_model, (uint64_t const[]){
		0x1960f3736b489faeu, 0x024fa28c5f34afacu, 0x8215489abc32a561u, 0x5bc95d036f4a2798u,
		0xa1d5a56f13f5c41cu});
	free(large_model);
	large_model = nullptr;
	free(large_input);
	large_input = nullptr;
	require(!moved, "a golden moved");
}

int
main (void)
{
	struct geometry g = {5, 5, 5, 5, 5, 0, 0, 5, 5};
	float input[25 * 4], model[25 * 3], output[25 * 3];
	fill(input, 25 * 4, 0.25f);
	fill(model, 25 * 3, 0.5f);
	for (unsigned p = 0; p < 25; ++p)
		input[p * 4 + 3] = 1.0f;
	struct native_tuning t = NATIVE_TUNING_DEFAULTS;
	// A deliberately non-half sample exposes accidental default rounding.
	model[12 * 3] = 0.53123456f;
	tune(input, model, &g, output, &t);
	require(!memcmp(model, output, sizeof model), "default settings must preserve every model bit");

	t.intensity = 0;
	tune(input, model, &g, output, &t);
	for (size_t i = 0; i < 25 * 3; ++i)
		require(output[i] == 0.25f, "zero intensity must return the input");

	fill(model, 25 * 3, 0.5f);
	t = (struct native_tuning){2, 1, 1, 0};
	tune(input, model, &g, output, &t);
	require_near(output[12 * 3], 0.75f, "intensity scales residual");
	t = (struct native_tuning){1, 0, 1, 0};
	tune(input, model, &g, output, &t);
	for (size_t i = 0; i < 25 * 3; ++i)
		require_near(output[i], 0.25f, "zero tone removes a constant edit");

	// A single impulse separates the low-pass (centre weight 1/4) from
	// the structural component (remaining 3/4), independently of input.
	fill(model, 25 * 3, 0.25f);
	model[12 * 3] += 0.25f;
	t = (struct native_tuning){1, 0, 1, 0};
	tune(input, model, &g, output, &t);
	require_near(output[12 * 3], 0.4375f, "structure retains three quarters of centre impulse");
	require_near(output[11 * 3], 0.21875f, "structure subtracts low-pass neighbour");
	t = (struct native_tuning){1, 1, 0, 0};
	tune(input, model, &g, output, &t);
	require_near(output[12 * 3], 0.3125f, "tone retains one quarter of centre impulse");
	require_near(output[11 * 3], 0.28125f, "tone spreads one eighth to axial neighbour");
	t = (struct native_tuning){1, 1, 1, 1};
	tune(input, model, &g, output, &t);
	require_near(output[12 * 3], 0.6875f, "sharpness adds model high-pass");

	// Strong residuals keep floating point headroom for the next pass.
	fill(model, 25 * 3, 1.0f);
	t = (struct native_tuning){4, 1, 1, 0};
	tune(input, model, &g, output, &t);
	require_near(output[12 * 3], 3.25f, "inter-pass tuning must not clamp to display range");

	// Letterbox samples are untouched and cannot leak into the picture.
	g.x = g.y = 1;
	g.fit_width = g.fit_height = 3;
	for (unsigned y = 0; y < 5; ++y)
		for (unsigned x = 0; x < 5; ++x)
			for (unsigned c = 0; c < 3; ++c)
				model[(y * 5 + x) * 3 + c] = (x && x < 4 && y && y < 4) ? 0.5f : 100.0f;
	t = (struct native_tuning){1, 0, 1, 0};
	tune(input, model, &g, output, &t);
	require_near(output[6 * 3], 0.25f, "fitted edge excludes letterbox from blur");
	require_near(output[0], 100.0f, "unused padding remains untouched");

	t.intensity = NAN;
	require(reference_tune_neural_rgb(input, model, &g, &t, output, nullptr) == ERROR_REJECTED,
	        "nonfinite tuning must fail");
	static struct native_tuning const invalid[] = {{4.5f, 1, 1, 0}, {1, -1, 1, 0}, {1, 1, 1, 1.5f}};
	for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i)
		require(tuning_validate(&invalid[i], nullptr) == ERROR_REJECTED,
		        "out-of-range tuning must reject the request");
	require(reference_tune_neural_rgb(input, model, &g, &(struct native_tuning const)NATIVE_TUNING_DEFAULTS, model,
	                                  nullptr) == ERROR_FAILED,
	        "in-place neighbourhood processing must fail");
	goldens();
	puts("native tuning: defaults, intensity, tone, structure, sharpness, viewport, errors and goldens passed");
	return 0;
}
