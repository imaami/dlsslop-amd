/** @file
 *
 * Host test of the CPU codec (reference.h, geometry.h): the geometry, the SDR and FP16 transport,
 * the padding's reflection, round trips, fitting, the area-weighted downscale, the rejection of
 * invalid samples, 8-bit and 16-bit feedback between passes, and the goldens of its outputs
 * (golden.h).
 */
// SPDX-License-Identifier: MIT
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "geometry.h"
#include "golden.h"
#include "reference.h"

/** @brief Ends the test with a message unless a condition holds.
 *
 * @param ok      The condition.
 * @param message The message.
 */
static void
require (bool        ok,
         char const *message)
{
	if (ok)
		return;
	fprintf(stderr, "codec test failed: %s\n", message);
	exit(EXIT_FAILURE);
}

/** @brief Ends the test with a reference's words unless it succeeded.
 *
 * @param code What the reference returned.
 * @param e    Its error.
 * @param what The reference's name.
 */
static void
check (enum error_code     code,
       struct error const *e,
       char const         *what)
{
	if (!code)
		return;
	fprintf(stderr, "codec test failed: %s: %s\n", what, e->what);
	exit(EXIT_FAILURE);
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

/** @brief A geometry that the test expects to be valid.
 *
 * @param width  The source's width.
 * @param height Its height.
 * @param tier   The tier, or 0 for the smallest that holds the source.
 * @return       The geometry.
 */
static struct geometry
valid_geometry (unsigned width,
                unsigned height,
                unsigned tier)
{
	struct geometry g;
	require(!geometry_init(&g, width, height, tier, nullptr), "valid geometry refused");
	return g;
}

/** @brief The floats behind a reference's output in a buffer from nan_floats(), which it must leave
 *         alone. */
#define GUARD_FLOATS 4

/** @brief A buffer of floats for a reference's output, and GUARD_FLOATS more behind them, all NaN.
 *
 * @param count The floats of the output.
 * @return      The buffer, which the caller frees.
 */
static float *
nan_floats (size_t count)
{
	float *const v = allocated(malloc((count + GUARD_FLOATS) * sizeof *v));
	for (size_t i = 0; i < count + GUARD_FLOATS; ++i)
		v[i] = NAN;
	return v;
}

/** @brief Whether a reference wrote a number into every float of its output in a buffer from
 *         nan_floats(), and nothing behind it.
 *
 * @param v     The buffer.
 * @param count The floats of the output.
 * @return      true if it did.
 */
static bool
filled (float const *v,
        size_t       count)
{
	for (size_t i = 0; i < count; ++i)
		if (isnan(v[i]))
			return false;
	for (size_t i = count; i < count + GUARD_FLOATS; ++i)
		if (!isnan(v[i]))
			return false;
	return true;
}

#undef GUARD_FLOATS

/** @brief Whether two arrays of floats are equal, each pair compared by ==.
 *
 * @param a     An array.
 * @param b     Another.
 * @param count Their floats.
 * @return      true if they are.
 */
static bool
equal_floats (float const *a,
              float const *b,
              size_t       count)
{
	for (size_t i = 0; i < count; ++i)
		if (a[i] != b[i])
			return false;
	return true;
}

/** @brief The answer of a network that returns its input: a proxy's RGB.
 *
 * @param rgba   The proxy, RGBA32F.
 * @param pixels Its pixels.
 * @return       The answer, RGB32F, which the caller frees.
 */
static float *
identity_neural (float const *rgba,
                 size_t       pixels)
{
	float *const rgb = allocated(malloc(pixels * 3 * sizeof *rgb));
	for (size_t p = 0; p < pixels; ++p)
		for (unsigned c = 0; c < 3; ++c)
			rgb[p * 3 + c] = rgba[p * 4 + c];
	return rgb;
}

/** @brief Expects geometry_init() to fail for a source extent and tier.
 *
 * @param width  The source's width.
 * @param height Its height.
 * @param tier   The tier.
 */
static void
expect_invalid (unsigned width,
                unsigned height,
                unsigned tier)
{
	struct geometry g;
	require(geometry_init(&g, width, height, tier, nullptr) == ERROR_FAILED, "invalid geometry accepted");
}

/** @brief The geometry: tiers, fits and letterboxes, the ranges it refuses, and its validation. */
static void
test_geometry (void)
{
	struct geometry const a = valid_geometry(1280, 720, 0);
	require(a.width == 1280 && a.height == 768 && !a.x && !a.y &&
	        a.fit_width == 1280 && a.fit_height == 720, "720 geometry");
	struct geometry const b = valid_geometry(1600, 900, 0);
	require(b.width == 1600 && b.height == 960, "900 geometry");
	struct geometry const c = valid_geometry(3840, 2160, 0);
	require(c.width == 1920 && c.height == 1152, "large source geometry");
	struct geometry const d = valid_geometry(640, 480, 720);
	require(d.x == 160 && !d.y && d.fit_width == 960 && d.fit_height == 720,
	        "4:3 letterbox geometry");
	struct geometry const ultrawide = valid_geometry(3440, 1440, 900);
	require(!ultrawide.x && ultrawide.y == 115 && ultrawide.fit_width == 1600 && ultrawide.fit_height == 670,
	        "ultrawide letterbox geometry");
	struct geometry const f = valid_geometry(1, 16384, 720);
	require(f.fit_width == 1 && f.fit_height == 720, "thin image geometry");
	expect_invalid(0, 720, 720);
	expect_invalid(1920, 0, 1080);
	expect_invalid(16385, 1, 720);
	expect_invalid(1280, 720, 768);
	struct error e;
	check(geometry_validate(&ultrawide, &e), &e, "consistent geometry refused");
	struct geometry skewed = ultrawide;
	++skewed.fit_height;
	require(geometry_validate(&skewed, nullptr) == ERROR_FAILED, "inconsistent geometry validated");
}

/** @brief The proxy's input contract and padding, and an identity network's round trip at the native
 *         tier. */
static void
test_input_contract_and_identity (void)
{
	struct geometry const g = valid_geometry(1280, 720, 720);
	size_t const bytes = (size_t)g.source_width * g.source_height * 4;
	uint8_t *source = allocated(malloc(bytes));
	for (unsigned y = 0; y < g.source_height; ++y) {
		for (unsigned x = 0; x < g.source_width; ++x) {
			size_t const p = ((size_t)y * g.source_width + x) * 4;
			source[p] = (uint8_t)x;
			source[p + 1] = (uint8_t)y;
			source[p + 2] = (uint8_t)(x + y);
			source[p + 3] = (uint8_t)(x * 17 + y * 13);
		}
	}
	size_t const pixels = (size_t)g.width * g.height;
	float *encoded = nan_floats(pixels * 4);
	struct error e;
	check(reference_encode_proxy(source, &g, false, encoded, &e), &e, "encode_proxy");
	require(filled(encoded, pixels * 4), "encoded extent");
	// A display encoded 128 must stay around .502, not become linear .216.
	require(encoded[128 * 4] == 0.501953125f, "sRGB input was gamma decoded or not FP16 rounded");
	require(encoded[255 * 4] == 1.0f && encoded[255 * 4 + 3] == 1.0f, "white/alpha contract");
	size_t const stride = (size_t)g.width * 4;
	require(!memcmp(encoded + 720 * stride, encoded + 718 * stride, stride * sizeof *encoded),
	        "first reflected row must be h-2");
	require(!memcmp(encoded + 767 * stride, encoded + 671 * stride, stride * sizeof *encoded),
	        "last reflected row mismatch");
	float *neural = identity_neural(encoded, pixels);
	float *feedback = allocated(malloc(pixels * 4 * sizeof *feedback));
	check(reference_feedback_neural_rgb(neural, &g, true, feedback, &e), &e, "feedback_neural_rgb");
	require(equal_floats(feedback, encoded, pixels * 4), "identity feedback must preserve the full encoded input");
	uint8_t *decoded = allocated(malloc(bytes));
	check(reference_decode_neural_proxy(source, &g, false, neural, decoded, &e), &e, "decode_neural_proxy");
	require(!memcmp(decoded, source, bytes), "all 256 SDR codes + alpha must round-trip at native tier");
	free(decoded);
	decoded = nullptr;
	free(feedback);
	feedback = nullptr;
	free(neural);
	neural = nullptr;
	free(encoded);
	encoded = nullptr;
	free(source);
	source = nullptr;
}

/** @brief A tiny source's fit, bilinear sampling between a source's texels, and the decode's clamp
 *         and alpha. */
static void
test_fit_and_output (void)
{
	// Tiny constant sources test fitting/clamping without relying on a matching
	// identity sampler; colored corners below then exercise bilinear averaging.
	struct geometry const g = valid_geometry(1, 1, 720);
	uint8_t const source[] = {128, 64, 255, 37};
	size_t const pixels = (size_t)g.width * g.height;
	float *encoded = allocated(malloc(pixels * 4 * sizeof *encoded));
	struct error e;
	check(reference_encode_proxy(source, &g, false, encoded, &e), &e, "encode_proxy");
	require(g.x == 280 && g.fit_width == 720, "square fit geometry");
	require(encoded[0] == 0.0f && encoded[3] == 1.0f, "letterbox must be opaque black");
	require(encoded[g.x * 4] == 0.501953125f, "first fitted pixel");
	float *neural = identity_neural(encoded, pixels);
	// One pixel past the source's extent stays untouched.
	uint8_t output[8];
	memset(output, 0xa5, sizeof output);
	uint8_t const expected[] = {128, 64, 255, 37, 0xa5, 0xa5, 0xa5, 0xa5};
	check(reference_decode_neural_proxy(source, &g, false, neural, output, &e), &e, "decode_neural_proxy");
	require(!memcmp(output, expected, sizeof output), "constant fit decode");

	// The same tier: the same processing extent as g's.
	struct geometry const small = valid_geometry(2, 2, 720);
	uint8_t const corners[] = {0, 0, 0, 0, 255, 0, 0, 255,
	                           0, 255, 0, 127, 255, 255, 255, 1};
	check(reference_encode_proxy(corners, &small, false, encoded, &e), &e, "encode_proxy");
	unsigned const x = small.x + 359, y = 359;
	size_t const p = ((size_t)y * small.width + x) * 4;
	// Coordinates correspond to .498611 on both source axes: interpolation is
	// performed in display encoding, so R/G stay near .499, not ~.734.
	require(fabsf(encoded[p] - 0.498611111f) < 0.00025f &&
	        fabsf(encoded[p + 1] - 0.498611111f) < 0.00025f &&
	        fabsf(encoded[p + 2] - 0.248613040f) < 0.00025f,
	        "bilinear input samples have wrong coordinates or color space");

	float *pathological = allocated(malloc(pixels * 3 * sizeof *pathological));
	for (size_t q = 0; q < pixels * 3; q += 3) {
		pathological[q] = -1.0f;
		pathological[q + 1] = 2.0f;
		pathological[q + 2] = 65519.0f; // Rounds to the largest finite binary16.
	}
	check(reference_decode_neural_proxy(source, &g, false, pathological, output, &e), &e, "decode_neural_proxy");
	uint8_t const clamped[] = {0, 255, 255, 37, 0xa5, 0xa5, 0xa5, 0xa5};
	require(!memcmp(output, clamped, sizeof output), "UNORM clamp / alpha preservation");
	free(pathological);
	pathological = nullptr;
	free(neural);
	neural = nullptr;
	free(encoded);
	encoded = nullptr;
}

/** @brief A source larger than the fit is integrated over each pixel's footprint: a one-texel stripe
 *         of period 3 at exactly 3 texels per pixel encodes to its mean at every phase, where a
 *         bilinear tap would read all 1 or all 0. At 1.5 texels per pixel, the pixels alternate and
 *         keep the mean. */
static void
test_area_downscale (void)
{
	struct geometry const g = valid_geometry(3840, 2160, 720);
	require(g.fit_width == 1280 && g.fit_height == 720, "3x downscale geometry");
	size_t const source_pixels = (size_t)g.source_width * g.source_height;
	uint8_t *rgba8 = allocated(malloc(source_pixels * 4));
	uint8_t *fp16 = allocated(malloc(source_pixels * 8));
	float *encoded = allocated(malloc((size_t)g.width * g.height * 4 * sizeof *encoded));
	float const mean = 0.333251953125f; // 1/3 rounded to binary16.
	struct error e;
	for (unsigned phase = 0; phase < 3; ++phase) {
		for (size_t p = 0; p < source_pixels; ++p) {
			bool const lit = (p % g.source_width) % 3 == phase;
			uint16_t const half = lit ? 0x3c00 : 0;
			for (unsigned c = 0; c < 4; ++c) {
				rgba8[p * 4 + c] = lit || c == 3 ? 255 : 0;
				memcpy(fp16 + p * 8 + c * 2, &half, sizeof half);
			}
		}
		for (unsigned half = 0; half < 2; ++half) {
			check(reference_encode_proxy(half ? fp16 : rgba8, &g, half, encoded, &e), &e, "encode_proxy");
			for (size_t p = 0; p < (size_t)g.width * g.valid_height; ++p)
				require(encoded[p * 4] == mean && encoded[p * 4 + 1] == mean && encoded[p * 4 + 2] == mean,
				        "3x downscale does not encode a period-3 stripe to its mean");
		}
	}

	// The same tier: the same processing extent as g's.
	struct geometry const h = valid_geometry(1920, 1080, 720);
	require(h.fit_width == 1280 && h.fit_height == 720, "1.5x downscale geometry");
	for (size_t p = 0; p < (size_t)h.source_width * h.source_height; ++p)
		for (unsigned c = 0; c < 3; ++c)
			rgba8[p * 4 + c] = (p % h.source_width) % 3 ? 0 : 255;
	check(reference_encode_proxy(rgba8, &h, false, encoded, &e), &e, "encode_proxy");
	double sum = 0;
	for (unsigned x = 0; x < h.width; ++x) {
		float const expected = x % 2 ? 0.0f : 0.66650390625f; // 2/3 rounded to binary16.
		require(encoded[((size_t)360 * h.width + x) * 4] == expected, "1.5x downscale weights");
		sum += encoded[((size_t)360 * h.width + x) * 4];
	}
	require(fabs(sum / h.width - 1.0 / 3.0) < 1e-3, "1.5x downscale changed the mean");

	// Every texel under a footprint is read, so a NaN at texel (0, 0), which
	// no bilinear tap at 3x reaches, still rejects the FP16 frame.
	uint16_t const nonfinite = 0x7e00;
	memcpy(fp16, &nonfinite, sizeof nonfinite);
	require(reference_encode_proxy(fp16, &g, true, encoded, nullptr) == ERROR_REJECTED,
	        "3x downscale accepted a nonfinite FP16 texel");
	free(encoded);
	encoded = nullptr;
	free(fp16);
	fp16 = nullptr;
	free(rgba8);
	rgba8 = nullptr;
}

/** @brief Expects the RGBA8 decode to reject an answer.
 *
 * @param source The source, RGBA8.
 * @param bytes  Its bytes.
 * @param g      Its geometry.
 * @param neural The answer.
 */
static void
expect_decode_rejection (uint8_t const         *source,
                         size_t                 bytes,
                         struct geometry const *g,
                         float const           *neural)
{
	uint8_t *output = allocated(malloc(bytes));
	require(reference_decode_neural_proxy(source, g, false, neural, output, nullptr) == ERROR_REJECTED,
	        "RGBA8 decode accepted a nonfinite/FP16-overflow neural sample");
	free(output);
	output = nullptr;
}

/** @brief Like the GPU codec, the CPU RGBA8 decode rejects an invalid neural sample instead of
 *         painting it and its zero-weight bilinear neighbours black. */
static void
test_decode_invalid_samples (void)
{
	struct geometry const g = valid_geometry(1280, 720, 720);
	size_t const bytes = (size_t)g.source_width * g.source_height * 4;
	uint8_t *source = allocated(malloc(bytes));
	memset(source, 128, bytes);
	size_t const pixels = (size_t)g.width * g.height;
	float *encoded = allocated(malloc(pixels * 4 * sizeof *encoded));
	struct error e;
	check(reference_encode_proxy(source, &g, false, encoded, &e), &e, "encode_proxy");
	float *neural = identity_neural(encoded, pixels);
	size_t const texel = ((size_t)100 * g.width + 100) * 3;
	float const bad[] = {NAN, INFINITY, -INFINITY, 65520.0f, -65520.0f};
	float *poisoned = allocated(malloc(pixels * 3 * sizeof *poisoned));
	for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
		memcpy(poisoned, neural, pixels * 3 * sizeof *poisoned);
		poisoned[texel] = bad[i];
		expect_decode_rejection(source, bytes, &g, poisoned);
		for (size_t q = 0; q < pixels * 3; q += 3)
			poisoned[q] = bad[i];
		expect_decode_rejection(source, bytes, &g, poisoned);
	}
	float *limit = allocated(malloc(pixels * 3 * sizeof *limit));
	memcpy(limit, neural, pixels * 3 * sizeof *limit);
	limit[texel] = 65519.0f;
	limit[texel + 1] = -65519.0f;
	uint8_t *output = allocated(malloc(bytes));
	check(reference_decode_neural_proxy(source, &g, false, limit, output, &e), &e, "decode_neural_proxy");
	size_t const pixel = ((size_t)100 * g.source_width + 100) * 4;
	require(output[pixel] == 255 && output[pixel + 1] == 0 && output[pixel + 4] == 128 &&
	        output[pixel - 4] == 128, "RGBA8 decode rejected or spread the finite binary16 limits");
	free(output);
	output = nullptr;
	free(limit);
	limit = nullptr;
	free(poisoned);
	poisoned = nullptr;
	free(neural);
	neural = nullptr;
	free(encoded);
	encoded = nullptr;
	free(source);
	source = nullptr;
}

/** @brief 16-bit feedback at three tiers: its extent, rounding, letterbox and padding, without a
 *         clamp. */
static void
test_feedback_precision_and_padding (void)
{
	struct geometry const geometries[] = {
		valid_geometry(640, 480, 720),
		valid_geometry(3440, 1440, 900),
		valid_geometry(1920, 1080, 1080),
	};
	struct error e;
	for (size_t i = 0; i < sizeof geometries / sizeof *geometries; ++i) {
		struct geometry const *const g = &geometries[i];
		size_t const pixels = (size_t)g->width * g->height;
		// Poison everything except the fit rectangle: feedback must ignore raw
		// neural letterbox/padding, then reconstruct the original geometry.
		float *neural = allocated(malloc(pixels * 3 * sizeof *neural));
		for (size_t q = 0; q < pixels * 3; ++q)
			neural[q] = NAN;
		for (unsigned y = g->y; y < g->y + g->fit_height; ++y) {
			for (unsigned x = g->x; x < g->x + g->fit_width; ++x) {
				float *const p = neural + ((size_t)y * g->width + x) * 3;
				p[0] = -0.25f;
				p[1] = 1.5f;
				p[2] = y & 1u ? 0.25f : 0.75f;
			}
		}
		size_t const marker = (size_t)(g->y + 80) * g->width + g->x + 80;
		neural[marker * 3] = 0.500244140625f;     // Tie to even: .5.
		neural[marker * 3 + 1] = 0.500732421875f; // Tie to even: .5009765625.
		neural[marker * 3 + 2] = 0x1p-24f;        // Smallest binary16 subnormal.
		float *feedback = nan_floats(pixels * 4);
		check(reference_feedback_neural_rgb(neural, g, true, feedback, &e), &e, "feedback_neural_rgb");
		require(filled(feedback, pixels * 4), "feedback extent differs from encode extent");
		require(feedback[marker * 4] == 0.5f &&
		        feedback[marker * 4 + 1] == 0.5009765625f &&
		        feedback[marker * 4 + 2] == 0x1p-24f,
		        "feedback resampled, quantized to UNORM8, or rounded binary16 incorrectly");
		size_t const first = (size_t)g->y * g->width + g->x;
		require(feedback[first * 4] == -0.25f && feedback[first * 4 + 1] == 1.5f,
		        "feedback must not clamp the raw neural result");
		for (size_t p = 0; p < pixels; ++p)
			require(feedback[p * 4 + 3] == 1.0f, "feedback alpha is not one");
		if (g->x)
			require(feedback[first * 4 - 4] == 0.0f &&
			        feedback[(first + g->fit_width) * 4] == 0.0f,
			        "feedback did not clear horizontal letterbox");
		if (g->y)
			require(feedback[0] == 0.0f &&
			        feedback[(size_t)(g->y + g->fit_height) * g->width * 4] == 0.0f,
			        "feedback did not clear vertical letterbox");
		size_t const stride = (size_t)g->width * 4;
		require(!memcmp(feedback + g->valid_height * stride,
		                feedback + (g->valid_height - 2) * stride,
		                stride * sizeof *feedback),
		        "feedback first reflected row differs from h-2");
		require(!memcmp(feedback + (g->height - 1) * stride,
		                feedback + (2 * g->valid_height - 1 - g->height) * stride,
		                stride * sizeof *feedback),
		        "feedback last reflected row mismatch");
		free(feedback);
		feedback = nullptr;
		free(neural);
		neural = nullptr;
	}
}

/** @brief Feedback takes the finite binary16 limits and fails on samples past them. */
static void
test_feedback_invalid_samples (void)
{
	struct geometry const g = valid_geometry(1280, 720, 720);
	size_t const pixels = (size_t)g.width * g.height;
	float *neural = allocated(malloc(pixels * 3 * sizeof *neural));
	for (size_t q = 0; q < pixels * 3; ++q)
		neural[q] = 0.5f;
	float *feedback = allocated(malloc(pixels * 4 * sizeof *feedback));
	neural[0] = 65504.0f;
	neural[1] = -65504.0f;
	struct error e;
	check(reference_feedback_neural_rgb(neural, &g, true, feedback, &e), &e, "feedback_neural_rgb");
	require(feedback[0] == 65504.0f && feedback[1] == -65504.0f,
	        "feedback rejected finite binary16 limits");
	neural[0] = neural[1] = 0.5f;
	float const bad[] = {NAN, INFINITY, -INFINITY, 65520.0f, -65520.0f};
	for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
		for (unsigned channel = 0; channel < 3; ++channel) {
			neural[channel] = bad[i];
			require(reference_feedback_neural_rgb(neural, &g, true, feedback, nullptr) == ERROR_FAILED,
			        "feedback accepted nonfinite/FP16-overflow fitted sample");
			neural[channel] = 0.5f;
		}
	}
	free(feedback);
	feedback = nullptr;
	free(neural);
	neural = nullptr;
}

/** @brief The FP16 proxy keeps binary16 unclamped, round-trips with its alpha, and rejects nonfinite
 *         samples and answers past binary16. */
static void
test_fp16_proxy (void)
{
	struct geometry const g = valid_geometry(1280, 720, 720);
	uint16_t const samples[] = {0x0000, 0x0001, 0x03ff, 0x0400, 0x3555,
	                            0x3801, 0x3c00, 0x4000, 0xb400, 0xc000, 0x7bff, 0xfbff};
	size_t const source_pixels = (size_t)g.source_width * g.source_height;
	size_t const bytes = source_pixels * 8;
	uint8_t *source = allocated(malloc(bytes));
	for (size_t pixel = 0; pixel < source_pixels; ++pixel) {
		for (unsigned channel = 0; channel < 4; ++channel) {
			uint16_t const value = channel == 3 ? (uint16_t)pixel :
			                       samples[(pixel + channel) % (sizeof samples / sizeof *samples)];
			memcpy(source + pixel * 8 + channel * 2, &value, sizeof value);
		}
	}
	size_t const pixels = (size_t)g.width * g.height;
	float *encoded = allocated(malloc(pixels * 4 * sizeof *encoded));
	struct error e;
	check(reference_encode_proxy(source, &g, true, encoded, &e), &e, "encode_proxy");
	require(encoded[4] == 0x1p-24f && encoded[5 * 4] == 0.50048828125f &&
	        encoded[8 * 4] == -0.25f && encoded[10 * 4] == 65504.0f,
	        "FP16 input was gamma decoded, clamped, or lost binary16 precision");
	float *neural = identity_neural(encoded, pixels);
	uint8_t *decoded = allocated(malloc(bytes));
	check(reference_decode_neural_proxy(source, &g, true, neural, decoded, &e), &e, "decode_neural_proxy");
	require(!memcmp(decoded, source, bytes), "FP16 proxy native-tier round-trip or alpha preservation");

	uint16_t const nonfinite = 0x7e00;
	memcpy(source, &nonfinite, sizeof nonfinite);
	require(reference_encode_proxy(source, &g, true, encoded, nullptr) == ERROR_REJECTED,
	        "FP16 proxy accepted nonfinite RGB input");

	float const values[] = {NAN, INFINITY, 65520.0f};
	for (size_t i = 0; i < sizeof values / sizeof *values; ++i) {
		neural[0] = values[i];
		require(reference_decode_neural_proxy(source, &g, true, neural, decoded, nullptr) == ERROR_REJECTED,
		        "FP16 proxy decode accepted nonfinite/overflow neural sample");
	}
	free(decoded);
	decoded = nullptr;
	free(neural);
	neural = nullptr;
	free(encoded);
	encoded = nullptr;
	free(source);
	source = nullptr;
}

/** @brief 8-bit feedback clamps and quantizes before rounding to binary16, and fails on nonfinite
 *         samples. */
static void
test_feedback_unorm8 (void)
{
	struct geometry const g = valid_geometry(1280, 720, 720);
	size_t const pixels = (size_t)g.width * g.height;
	float *neural = allocated(malloc(pixels * 3 * sizeof *neural));
	for (size_t q = 0; q < pixels * 3; ++q)
		neural[q] = 0.0f;
	neural[0] = -0.25f;
	neural[1] = 1.5f;
	neural[2] = 0.4999f; // Direct UNORM8: 127; half-before-UNORM8 would become 128.
	neural[3] = 0.5f;    // 127.5 ties upward to the even code 128.
	neural[4] = FLT_MAX;
	neural[5] = -FLT_MAX;
	float *feedback = allocated(malloc(pixels * 4 * sizeof *feedback));
	struct error e;
	check(reference_feedback_neural_rgb(neural, &g, false, feedback, &e), &e, "feedback_neural_rgb");
	require(feedback[0] == 0.0f && feedback[1] == 1.0f &&
	        feedback[2] == 0.498046875f && feedback[4] == 0.501953125f &&
	        feedback[5] == 1.0f && feedback[6] == 0.0f,
	        "8-bit feedback must clamp/quantize before binary16 rounding");
	size_t const stride = (size_t)g.width * 4;
	require(!memcmp(feedback + 720 * stride, feedback + 718 * stride, stride * sizeof *feedback),
	        "8-bit feedback reflection mismatch");
	float const bad[] = {NAN, INFINITY, -INFINITY};
	for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
		neural[0] = bad[i];
		require(reference_feedback_neural_rgb(neural, &g, false, feedback, nullptr) == ERROR_FAILED,
		        "8-bit feedback accepted nonfinite raw sample");
	}
	free(feedback);
	feedback = nullptr;
	free(neural);
	neural = nullptr;
}

/** @brief A source of a geometry's extent from a fixture: random RGBA8, or FP16 with any finite
 *         binary16 in every channel, alpha included.
 *
 * @param g     The geometry.
 * @param fp16  Whether the source is RGBA16F.
 * @param seed  The fixture.
 * @param bytes Receives the source's bytes.
 * @return      The source, which the caller frees.
 */
static uint8_t *
fixture_source (struct geometry const *g,
                bool                   fp16,
                uint32_t               seed,
                size_t                *bytes)
{
	size_t const n = (size_t)g->source_width * g->source_height * (fp16 ? 8 : 4);
	uint8_t *const source = allocated(malloc(n));
	*bytes = n;
	if (!fp16) {
		for (size_t i = 0; i < n; ++i)
			source[i] = (uint8_t)golden_bits(seed, i);
		return source;
	}
	for (size_t i = 0; i < n / 2; ++i) {
		uint16_t const half = golden_half(seed, i);
		memcpy(source + i * 2, &half, sizeof half);
	}
	return source;
}

/** @brief A network's answer at a geometry's processing extent from a fixture, NaN outside the
 *         fitted picture, which no reference reads.
 *
 * @param g    The geometry.
 * @param seed The fixture.
 * @return     The answer, RGB32F, which the caller frees.
 */
static float *
fixture_neural (struct geometry const *g,
                uint32_t               seed)
{
	size_t const n = (size_t)g->width * g->height * 3;
	float *const rgb = allocated(malloc(n * sizeof *rgb));
	for (size_t i = 0; i < n; ++i)
		rgb[i] = NAN;
	for (unsigned y = g->y; y < g->y + g->fit_height; ++y) {
		for (unsigned x = g->x; x < g->x + g->fit_width; ++x) {
			size_t const p = ((size_t)y * g->width + x) * 3;
			for (size_t i = p; i < p + 3; ++i)
				rgb[i] = golden_neural(seed, i);
		}
	}
	return rgb;
}

/** @brief A source extent at a tier, and the goldens of a reference's outputs for it. */
struct fixture {
	uint64_t    rgba8;  //!< The golden of the RGBA8 source, or of 8-bit feedback.
	uint64_t    fp16;   //!< The golden of the FP16 source, or of 16-bit feedback.
	char const *name;   //!< The fixture's name in a message.
	unsigned    width;  //!< The source's width.
	unsigned    height; //!< Its height.
	uint64_t    tier;   //!< The tier.
};

/** @brief The references' outputs over fixtures, bit for bit (golden.h). */
static void
test_goldens (void)
{
	// Identity, enlarging (bilinear) and reducing (area-weighted) into the
	// 720 and 1080 tiers, by integral and other ratios, letterboxed or not.
	static struct fixture const encodes[] = {
		{0xc84be590d97083e9u, 0x7e7c0cb2abb7a392u, "identity 720", 1280, 720, 720},
		{0x2a813dfd1dad093bu, 0x6f61916b9ff2a475u, "identity 1080", 1920, 1080, 1080},
		{0x58b6349f33c931fbu, 0x18200a5975265df7u, "upscale 720", 853, 480, 720},
		{0xb97db8db0e3062dfu, 0x60680b606aa1a7f2u, "upscale 1080", 640, 480, 1080},
		{0x13ec170dead5e0d0u, 0x772f278d3e678ad3u, "downscale 720", 1920, 1080, 720},
		{0x856a00c01416d8beu, 0xd6045436a5eb871cu, "downscale 1080", 3440, 1440, 1080},
	};
	// Letterboxed and pillarboxed answers fed into another pass.
	static struct fixture const feedbacks[] = {
		{0x1caed93f276895a1u, 0x619a08a80adc808au, "3440x1440 900", 3440, 1440, 900},
		{0x15ab44c40c58a97eu, 0x9153a140fe206707u, "640x480 720", 640, 480, 720},
	};
	// Answers reduced to a smaller source and enlarged to a larger one.
	static struct fixture const decodes[] = {
		{0xdc22f9dd3cd6bafcu, 0x37c6f22cddf5d104u, "853x480 720", 853, 480, 720},
		{0x47b36df8fc72ba8eu, 0x734fa59fcffc5ceeu, "3440x1440 1080", 3440, 1440, 1080},
	};
	unsigned moved = 0;
	uint32_t seed = 0;
	struct error e;
	for (size_t i = 0; i < sizeof encodes / sizeof *encodes; ++i) {
		struct fixture const *const f = &encodes[i];
		struct geometry const g = valid_geometry(f->width, f->height, f->tier);
		size_t const floats = (size_t)g.width * g.height * 4;
		float *rgba = allocated(malloc(floats * sizeof *rgba));
		for (unsigned fp16 = 0; fp16 < 2; ++fp16) {
			size_t bytes;
			uint8_t *source = fixture_source(&g, fp16, ++seed, &bytes);
			check(reference_encode_proxy(source, &g, fp16, rgba, &e), &e, "golden encode_proxy");
			moved += !golden_check(rgba, floats * sizeof *rgba, fp16 ? f->fp16 : f->rgba8, "encode_proxy %s %s",
			                       fp16 ? "FP16" : "RGBA8", f->name);
			free(source);
			source = nullptr;
		}
		free(rgba);
		rgba = nullptr;
	}
	for (size_t i = 0; i < sizeof feedbacks / sizeof *feedbacks; ++i) {
		struct fixture const *const f = &feedbacks[i];
		struct geometry const g = valid_geometry(f->width, f->height, f->tier);
		size_t const floats = (size_t)g.width * g.height * 4;
		float *neural = fixture_neural(&g, ++seed);
		float *rgba = allocated(malloc(floats * sizeof *rgba));
		for (unsigned precision16 = 0; precision16 < 2; ++precision16) {
			check(reference_feedback_neural_rgb(neural, &g, precision16, rgba, &e), &e,
			      "golden feedback_neural_rgb");
			moved += !golden_check(rgba, floats * sizeof *rgba, precision16 ? f->fp16 : f->rgba8,
			                       "feedback_neural_rgb %s %s", precision16 ? "16-bit" : "8-bit", f->name);
		}
		free(rgba);
		rgba = nullptr;
		free(neural);
		neural = nullptr;
	}
	for (size_t i = 0; i < sizeof decodes / sizeof *decodes; ++i) {
		struct fixture const *const f = &decodes[i];
		struct geometry const g = valid_geometry(f->width, f->height, f->tier);
		float *neural = fixture_neural(&g, ++seed);
		for (unsigned fp16 = 0; fp16 < 2; ++fp16) {
			size_t bytes;
			uint8_t *source = fixture_source(&g, fp16, ++seed, &bytes);
			uint8_t *decoded = allocated(malloc(bytes));
			check(reference_decode_neural_proxy(source, &g, fp16, neural, decoded, &e), &e,
			      "golden decode_neural_proxy");
			moved += !golden_check(decoded, bytes, fp16 ? f->fp16 : f->rgba8, "decode_neural_proxy %s %s",
			                       fp16 ? "FP16" : "RGBA8", f->name);
			free(decoded);
			decoded = nullptr;
			free(source);
			source = nullptr;
		}
		free(neural);
		neural = nullptr;
	}
	require(!moved, "a golden moved");
}

int
main (void)
{
	test_geometry();
	test_input_contract_and_identity();
	test_fit_and_output();
	test_area_downscale();
	test_feedback_precision_and_padding();
	test_feedback_invalid_samples();
	test_decode_invalid_samples();
	test_fp16_proxy();
	test_feedback_unorm8();
	test_goldens();
	puts("codec: geometry, SDR/FP16 transport, reflection, round-trip, fitting, area-weighted downscale, "
	     "invalid-sample rejection, 8/16-bit multi-pass feedback and goldens passed");
	return EXIT_SUCCESS;
}
