/** @file
 *
 * Host test of the color preservation's CPU reference (reference_preserve_color()): bypass,
 * strength, luma, detail, padding, repeated bias, gamut, the model's own excursions, headroom,
 * validation, and the goldens of its outputs (golden.h).
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

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param value   The condition.
 * @param message The message.
 */
static void
check (bool        value,
       char const *message)
{
	if (value)
		return;
	fprintf(stderr, "%s\n", message);
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
	check(p, "out of memory");
	return p;
}

/** @brief Corrects an answer that the test expects to be corrected.
 *
 * @param original The frame's encoded input, RGBA32F.
 * @param raw      The pass's answer, RGB32F.
 * @param g        The geometry.
 * @param strength The correction's strength.
 * @param result   Receives the corrected answer, RGB32F.
 */
static void
preserve (float const           *original,
          float const           *raw,
          struct geometry const *g,
          float                  strength,
          float                 *result)
{
	check(!reference_preserve_color(original, raw, g, strength, result, nullptr), "valid color preservation refused");
}

/** @brief The weighted proxy-space luma of an RGB sample.
 *
 * @param rgb The sample.
 * @return    Its luma.
 */
static float
luma (float const *rgb)
{
	return .2126f * rgb[0] + .7152f * rgb[1] + .0722f * rgb[2];
}

/** @brief Corrects a uniform frame of one original and one model color.
 *
 * @param original_rgb The original color.
 * @param model_rgb    The model's color.
 * @param strength     The correction's strength.
 * @param centre       Receives the corrected centre pixel.
 */
static void
uniform (float const *original_rgb,
         float const *model_rgb,
         float        strength,
         float        centre[static 3])
{
	struct geometry const g = {8, 8, 8, 8, 8, 1, 1, 6, 6};
	float original[8 * 8 * 4], raw[8 * 8 * 3], result[8 * 8 * 3];
	for (unsigned p = 0; p < 64; ++p) {
		original[p * 4 + 3] = 1;
		for (unsigned c = 0; c < 3; ++c) {
			original[p * 4 + c] = original_rgb[c];
			raw[p * 3 + c] = model_rgb[c];
		}
	}
	preserve(original, raw, &g, strength, result);
	memcpy(centre, result + 27 * 3, 3 * sizeof *centre);
}

/** @brief The reference's output at four strengths for a random letterboxed fixture, a little outside
 *         0..1, and a model that drifted from it, bit for bit (golden.h). 3440x1440 at the 720 tier:
 *         letterboxed, and padded below. */
static void
goldens (void)
{
	struct geometry const g = {3440, 1440, 1280, 768, 720, 0, 92, 1280, 536};
	size_t const pixels = (size_t)g.width * g.height;
	float *original = allocated(malloc(pixels * 4 * sizeof *original));
	float *model = allocated(malloc(pixels * 3 * sizeof *model));
	float *result = allocated(malloc(pixels * 3 * sizeof *result));
	for (size_t p = 0; p < pixels; ++p) {
		for (unsigned c = 0; c < 3; ++c) {
			original[p * 4 + c] = golden_unit(1, p * 3 + c) * 1.25f - .125f;
			model[p * 3 + c] = original[p * 4 + c] + (golden_unit(2, p * 3 + c) - .5f) * .25f;
		}
		original[p * 4 + 3] = 1;
	}
	static struct {
		uint64_t golden;
		float    strength;
	} const strengths[] = {
		{0xf44c4ac95f2260f1u, 0}, {0x928f9929da24f922u, .25f},
		{0xfadb9dc5a658363bu, .5f}, {0x57b4e5d13a255e46u, 1},
	};
	unsigned moved = 0;
	for (size_t i = 0; i < sizeof strengths / sizeof *strengths; ++i) {
		preserve(original, model, &g, strengths[i].strength, result);
		moved += !golden_check(result, pixels * 3 * sizeof *result, strengths[i].golden, "preserve_color %g",
		                       (double)strengths[i].strength);
	}
	free(result);
	result = nullptr;
	free(model);
	model = nullptr;
	free(original);
	original = nullptr;
	check(!moved, "a golden moved");
}

int
main (void)
{
	struct geometry const g = {8, 8, 8, 8, 8, 1, 1, 6, 6};
	float original[8 * 8 * 4], raw[8 * 8 * 3], result[8 * 8 * 3], half[8 * 8 * 3];
	for (size_t i = 0; i < 8 * 8 * 4; ++i)
		original[i] = .4f;
	for (unsigned p = 0; p < 64; ++p) {
		raw[p * 3] = .6f;
		raw[p * 3 + 1] = .4f;
		raw[p * 3 + 2] = .2f;
	}
	preserve(original, raw, &g, 0, result);
	check(!memcmp(raw, result, sizeof raw), "disabled not bit identical");
	preserve(original, raw, &g, 1, result);
	preserve(original, raw, &g, .5f, half);
	for (unsigned y = 0; y < 8; ++y) {
		for (unsigned x = 0; x < 8; ++x) {
			unsigned const p = (y * 8 + x) * 3;
			if (x < 1 || x > 6 || y < 1 || y > 6) {
				check(!memcmp(raw + p, result + p, 3 * sizeof *raw), "padding modified");
				continue;
			}
			check(fabsf(result[p] - result[p + 2]) < 1e-6f, "uniform cast not removed");
			check(fabsf(luma(raw + p) - luma(result + p)) < 1e-6f, "luma changed");
			check(fabsf((half[p] - half[p + 2]) - .2f) < 1e-6f, "strength not proportional");
		}
	}
	// Repeated warm bias must be removed against the same original reference.
	for (unsigned pass = 0; pass < 3; ++pass) {
		for (unsigned p = 0; p < 64; ++p) {
			raw[p * 3] = result[p * 3] + .04f;
			raw[p * 3 + 1] = result[p * 3 + 1];
			raw[p * 3 + 2] = result[p * 3 + 2] - .04f;
		}
		preserve(original, raw, &g, 1, result);
		check(fabsf(result[3 * 27] - result[3 * 27 + 2]) < 1e-6f, "repeated cast accumulated");
	}
	// Achromatic high-frequency detail is retained, even on colored reference.
	for (unsigned p = 0; p < 64; ++p) {
		original[p * 4] = .3f;
		original[p * 4 + 1] = .4f;
		original[p * 4 + 2] = .5f;
		float const light = p % 2 ? .1f : -.1f;
		for (unsigned c = 0; c < 3; ++c)
			raw[p * 3 + c] = original[p * 4 + c] + light;
	}
	preserve(original, raw, &g, 1, result);
	for (size_t i = 0; i < 8 * 8 * 3; ++i)
		check(fabsf(result[i] - raw[i]) < 1e-6f, "achromatic detail lost");
	// Out-of-gamut corrected chroma compresses without changing in-range luma.
	for (unsigned p = 0; p < 64; ++p) {
		original[p * 4] = 1;
		original[p * 4 + 1] = 0;
		original[p * 4 + 2] = 0;
		raw[p * 3] = raw[p * 3 + 1] = raw[p * 3 + 2] = .8f;
	}
	preserve(original, raw, &g, 1, result);
	for (unsigned c = 0; c < 3; ++c)
		check(result[27 * 3 + c] >= 0 && result[27 * 3 + c] <= 1, "gamut excursion");
	check(fabsf(luma(result + 27 * 3) - .8f) < 1e-6f, "gamut mapping changed luma");
	// The model's own excursions outside [0,1] are not the correction's to undo: with no
	// drift the result is the model, bit for bit, at any strength.
	static float const excursions[][3] = {{1.2f, .9f, .9f}, {-.05f, .02f, .6f}};
	static float const excursion_strengths[] = {1e-6f, .5f, 1.f};
	for (size_t i = 0; i < sizeof excursions / sizeof *excursions; ++i) {
		for (size_t j = 0; j < sizeof excursion_strengths / sizeof *excursion_strengths; ++j) {
			float kept[3];
			uniform(excursions[i], excursions[i], excursion_strengths[j], kept);
			check(!memcmp(kept, excursions[i], sizeof excursions[i]), "model excursion changed without drift");
		}
	}
	// Tuning keeps headroom above 1; a negligible strength must not clip it.
	static float const tuned_original[] = {.9f, .5f, .5f}, tuned[] = {1.1f, .5f, .5f};
	float slight[3];
	uniform(tuned_original, tuned, 1e-6f, slight);
	for (unsigned c = 0; c < 3; ++c)
		check(fabsf(slight[c] - tuned[c]) < 1e-6f, "negligible strength jumped");
	// Saturated SDR blue whose model undershoots red: chroma only moves toward the original's,
	// in proportion to strength, and luma stays.
	static float const blue[] = {0, .05f, .6f}, undershoot[] = {-.02f, .05f, .6f};
	static float const blue_strengths[] = {.01f, .5f, 1.f};
	for (size_t i = 0; i < sizeof blue_strengths / sizeof *blue_strengths; ++i) {
		float const strength = blue_strengths[i];
		float moved[3];
		uniform(blue, undershoot, strength, moved);
		for (unsigned c = 0; c < 3; ++c) {
			float const from = undershoot[c] - luma(undershoot), to = blue[c] - luma(blue);
			float const chroma = moved[c] - luma(moved);
			check(fabsf(chroma - (from + strength * (to - from))) < 1e-6f, "chroma moved away from original");
		}
		check(fabsf(luma(moved) - luma(undershoot)) < 1e-6f, "SDR correction changed luma");
	}
	// An HDR highlight keeps its headroom: out-of-range luma takes the full correction.
	static float const highlight_original[] = {2.2f, 1.9f, 1.9f}, highlight[] = {2, 2, 2};
	float lit[3];
	uniform(highlight_original, highlight, 1, lit);
	static float const drift[] = {-.2f, .1f, .1f};
	for (unsigned c = 0; c < 3; ++c)
		check(fabsf(lit[c] - (highlight[c] - drift[c] + luma(drift))) < 1e-5f, "HDR highlight lost its correction");
	check(fabsf(luma(lit) - 2) < 1e-6f, "HDR correction changed luma");
	struct error e;
	enum error_code const refused = reference_preserve_color(original, raw, &g, NAN, result, &e);
	check(refused == ERROR_FAILED && !strcmp(e.what, "color preservation must be finite and within 0..1"),
	      "NaN strength accepted");
	goldens();
	puts("Color preservation: bypass, strength, luma, detail, padding, repeated bias, gamut, "
	     "model excursions, headroom, validation and goldens passed");
	return 0;
}
