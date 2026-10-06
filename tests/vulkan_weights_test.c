/** @file
 *
 * Host test of the Vulkan network's weight packing, without a GPU or a model. The primitives are
 * checked against upstream's definitions, transcribed from DLSSNR-AMD's tinlayout.hpp and
 * nr_graph.cpp at 3dfdddc, and every recipe against digests of what upstream's NrSession::build
 * packed from a synthetic model pack. The model reader is checked on packs in memory, good and
 * damaged, and vulkan_weights_pack() on segments that do not fit.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "vulkan_pack.h"
#include "vulkan_plan.h"
#include "vulkan_weights.h"

/** @brief The checks that failed. */
static unsigned failures;

/** @brief Fails the test with a message unless a condition holds.
 *
 * @param ok  The condition.
 * @param fmt A printf format for the message.
 * @param ... The format's arguments.
 * @return    @a ok.
 */
[[gnu::format(printf, 2, 3)]]
static bool
expect (bool        ok,
        char const *fmt,
        ...)
{
	if (ok)
		return true;

	va_list args;
	va_start(args, fmt);
	fputs("vulkan-weights test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	++failures;
	return false;
}

/** @brief Expects a call to have failed with given words.
 *
 * @param code  What the call returned.
 * @param e     Its error.
 * @param words The words it should have put there.
 */
static void
expect_error (enum error_code     code,
              struct error const *e,
              char const         *words)
{
	char const *const what = code ? e->what : "no error";
	expect(!strcmp(what, words), "expected \"%s\", got \"%s\"", words, what);
}

/** @brief Ends the test without memory for a value.
 *
 * @param p The value's memory, or nullptr.
 * @return  @a p.
 */
static void *
allocated (void *p)
{
	if (!p) {
		fputs("vulkan-weights test: out of memory\n", stderr);
		exit(1);
	}
	return p;
}

/** @brief Expects a model pack in memory to have been made and written. */
static void
expect_pack (struct vulkan_pack const *pack)
{
	expect(pack->ok, "cannot make a model pack");
}

/* Upstream's conversions, transcribed from tinlayout.hpp at 3dfdddc. */

/** @brief An E4M3 code as a float (upstream: tin::e4m3_to_f). */
static float
tin_e4m3_to_f (uint8_t b)
{
	float const s = (b & 0x80) ? -1.0f : 1.0f;
	int const e = (b >> 3) & 0xF;
	int const m = b & 0x7;
	if (e == 0xF && m == 0x7)
		return NAN;
	if (e == 0)
		return s * (float)m * 0.001953125f;
	return s * (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
}

/** @brief A float as the nearest E4M3 code, ties to even (upstream: tin::f_to_e4m3). */
static uint8_t
tin_f_to_e4m3 (float x)
{
	if (isnan(x))
		return 0x7F;
	if (x > 448.0f)
		return 0x7E;
	if (x < -448.0f)
		return 0xFE;
	int best = 0;
	float bd = INFINITY;
	for (int i = 0; i < 256; ++i) {
		if (i == 0x7F || i == 0xFF)
			continue;
		float const d = fabsf(tin_e4m3_to_f((uint8_t)i) - x);
		if (d < bd || (d == bd && (i & 1) == 0)) {
			bd = d;
			best = i;
		}
	}
	return (uint8_t)best;
}

/** @brief Binary16 as a float (upstream: tin::f16_to_f). */
static float
tin_f16_to_f (uint16_t h)
{
	uint32_t const s = (uint32_t)(h & 0x8000u) << 16;
	int const e = (h >> 10) & 0x1F;
	uint32_t const m = h & 0x3FFu;
	if (e == 0) {
		float const v = ldexpf((float)m, -24);
		return s ? -v : v;
	}
	if (e == 31)
		return m ? NAN : (s ? -INFINITY : INFINITY);
	float out;
	uint32_t const o = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
	memcpy(&out, &o, 4);
	return out;
}

/** @brief A float as binary16, ties to even (upstream: tin::f_to_f16). */
static uint16_t
tin_f_to_f16 (float x)
{
	uint32_t u;
	memcpy(&u, &x, 4);
	uint32_t const sg = (u >> 16) & 0x8000u;
	int32_t const ex = (int32_t)((u >> 23) & 0xFF) - 127;
	uint32_t const mn = u & 0x7FFFFFu;
	if (ex == 128)
		return (uint16_t)(sg | 0x7C00u | (mn ? 0x200u : 0u));
	if (ex < -127)
		return (uint16_t)sg;
	int32_t he = ex + 15;
	uint32_t const m24 = mn | 0x800000u;
	int shift = 13;
	if (he <= 0) {
		shift = 14 - he;
		he = 0;
		if (shift > 24)
			return (uint16_t)sg;
	}
	uint32_t hi = m24 >> shift;
	uint32_t const lo = m24 & ((1u << shift) - 1u);
	uint32_t const half = 1u << (shift - 1);
	if (lo > half || (lo == half && (hi & 1u)))
		++hi;
	if (he == 0)
		return (uint16_t)(hi >= 0x400u ? (sg | 0x400u) : (sg | hi));
	if (hi >= 0x800u) {
		hi >>= 1;
		++he;
	}
	if (he >= 31)
		return (uint16_t)(sg | 0x7C00u);
	return (uint16_t)(sg | ((uint32_t)he << 10) | (hi & 0x3FFu));
}

/** @brief The bits of a float. */
static uint32_t
bits_of (float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof bits);
	return bits;
}

/** @brief The conversions against upstream's round trips. */
static void
check_conversions (void)
{
	for (unsigned code = 0; code < 256; ++code) {
		uint8_t const expected = tin_f_to_e4m3(tin_e4m3_to_f((uint8_t)code));
		uint8_t const got = vulkan_weights_requantise((uint8_t)code);
		expect(got == expected, "requantise(%#x) is %#x, upstream's round trip %#x", code, got, expected);
	}
	for (unsigned h = 0; h < 65536; ++h) {
		float const wide = tin_f16_to_f((uint16_t)h);
		if (!expect(vulkan_weights_widen_half((uint16_t)h) == bits_of(wide),
		            "widen_half(%#x) differs from tin::f16_to_f", h) ||
		    !expect(vulkan_weights_recode_half((uint16_t)h) == tin_f_to_f16(wide),
		            "recode_half(%#x) differs from tin::f_to_f16 of tin::f16_to_f", h))
			break;
	}
}

/** @brief Which element of a matrix a layout puts at each byte.
 *
 * @param rows    The matrix's rows.
 * @param cols    Its columns.
 * @param layout  The layout: vulkan_weights_tile_blocked() or vulkan_weights_npair_blocked().
 * @param element Receives each byte's element.
 */
static void
element_at (size_t    rows,
            size_t    cols,
            void      layout (uint8_t const *, size_t, size_t, uint8_t *),
            uint32_t *element)
{
	size_t const bytes = rows * cols;
	uint8_t *plane = allocated(malloc(bytes));
	uint8_t *out = allocated(malloc(bytes));
	memset(element, 0, bytes * sizeof *element);
	for (unsigned p = 0; p < 4; ++p) {
		// Byte p of each element's index.
		for (size_t i = 0; i < bytes; ++i)
			plane[i] = (uint8_t)(i >> 8 * p);
		layout(plane, rows, cols, out);
		for (size_t i = 0; i < bytes; ++i)
			element[i] |= (uint32_t)out[i] << 8 * p;
	}
	free(out);
	out = nullptr;
	free(plane);
	plane = nullptr;
}

/** @brief The layouts against upstream's orders. */
static void
check_layouts (void)
{
	static struct {
		size_t rows;
		size_t cols;
	} const shapes[] = {{32, 16}, {64, 48}, {96, 32}, {32, 128}, {128, 32}};
	for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
		size_t const rows = shapes[s].rows;
		size_t const cols = shapes[s].cols;
		size_t const bytes = rows * cols;
		size_t const ktiles = cols / 16;
		// tin::tile_blocked's order, and the weight layout 3 pass on it (nr_graph.cpp:3270-3281).
		uint32_t *tiled = allocated(malloc(bytes * sizeof *tiled));
		uint32_t *paired = allocated(malloc(bytes * sizeof *paired));
		uint32_t *got = allocated(malloc(bytes * sizeof *got));
		bool *seen = allocated(calloc(bytes, sizeof *seen));
		for (size_t r = 0; r < rows; ++r)
			for (size_t k = 0; k < cols; ++k) {
				size_t const at = ((r / 16) * ktiles + k / 16) * 256 + (r % 16) * 16 + (k % 16);
				tiled[at] = (uint32_t)(r * cols + k);
			}
		for (size_t n = 0; n < rows / 16; n += 2)
			for (size_t k = 0; k < ktiles; ++k)
				for (size_t lane = 0; lane < 32; ++lane)
					for (size_t j = 0; j < 2; ++j) {
						size_t const to = ((n / 2) * ktiles + k) * 512 + lane * 16 + j * 8;
						size_t const from = ((n + j) * ktiles + k) * 256 + (lane % 16) * 16 +
						                    (lane / 16) * 8;
						for (size_t c = 0; c < 8; ++c)
							paired[to + c] = tiled[from + c];
					}
		for (size_t i = 0; i < bytes; ++i)
			seen[paired[i]] = true;
		size_t count = 0;
		for (size_t i = 0; i < bytes; ++i)
			count += seen[i];
		expect(count == bytes, "upstream's layouts of %zux%zu are not bijections", rows, cols);
		element_at(rows, cols, vulkan_weights_tile_blocked, got);
		expect(!memcmp(got, tiled, bytes * sizeof *got),
		       "tile_blocked of %zux%zu differs from upstream's order", rows, cols);
		element_at(rows, cols, vulkan_weights_npair_blocked, got);
		expect(!memcmp(got, paired, bytes * sizeof *got),
		       "npair_blocked of %zux%zu differs from upstream's order", rows, cols);
		free(seen);
		seen = nullptr;
		free(got);
		got = nullptr;
		free(paired);
		paired = nullptr;
		free(tiled);
		tiled = nullptr;
	}
}

/** @brief Whether a gather over a x b x c names each of count bytes from first on once.
 *
 * @param first The first byte.
 * @param count The bytes.
 * @param a     The first index's range.
 * @param b     The second's.
 * @param c     The third's.
 * @param byte  The gather: which byte of a record holds an element.
 * @return      true if it does.
 */
static bool
covers (size_t first,
        size_t count,
        size_t a,
        size_t b,
        size_t c,
        size_t byte (size_t, size_t, size_t))
{
	unsigned *hits = allocated(calloc(first + count, sizeof *hits));
	bool inside = true;
	for (size_t i = 0; i < a && inside; ++i)
		for (size_t j = 0; j < b && inside; ++j)
			for (size_t k = 0; k < c && inside; ++k) {
				size_t const at = byte(i, j, k);
				inside = at >= first && at < first + count;
				if (inside)
					++hits[at];
			}
	size_t once = 0;
	for (size_t i = first; inside && i < first + count; ++i)
		once += hits[i] == 1;
	free(hits);
	hits = nullptr;
	return inside && once == count;
}

/** @brief vulkan_weights_bias_source() as a gather of one group. */
static size_t
bias (size_t group,
      size_t i,
      size_t j)
{
	return vulkan_weights_bias_source(i, j);
}

/** @brief The gathers are bijections onto their parts. */
static void
check_gathers (void)
{
	expect(covers(0, 262144, 8, 64, 512, vulkan_weights_ff_a), "ff_a is not a bijection onto bytes 0..262143");
	expect(covers(262144, 131072, 8, 256, 64, vulkan_weights_ff_q0),
	       "ff_q0 is not a bijection onto bytes 262144..393215");
	expect(covers(393216, 131072, 8, 64, 256, vulkan_weights_ff_q2),
	       "ff_q2 is not a bijection onto bytes 393216..524287");
	expect(covers(128, 3145728, 3, 1024, 1024, vulkan_weights_vit_qkv_weight_byte),
	       "vit_qkv_weight_byte is not a bijection onto bytes 128..3145855");
	expect(covers(0, 4096, 1, 64, 64, bias), "bias_source is not a permutation of 4096 values");
}

/** @brief A word rotated right. */
static uint32_t
rotr (uint32_t x,
      unsigned n)
{
	return x >> n | x << (32 - n);
}

/** @brief SHA-256 (FIPS 180-4) of bytes, in hexadecimal.
 *
 * @param data  The bytes.
 * @param bytes Their number.
 * @param hex   Receives the digest and a null.
 */
static void
sha256 (uint8_t const *data,
        size_t         bytes,
        char           hex[65])
{
	static uint32_t const k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
	};
	uint32_t h[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
	};
	// The message, 0x80, zeros to 56 bytes past a multiple of 64, and its length in bits.
	size_t const size = (bytes + 1 + 8 + 63) / 64 * 64;
	uint8_t *message = allocated(calloc(size, 1));
	memcpy(message, data, bytes);
	message[bytes] = 0x80;
	for (int i = 0; i < 8; ++i)
		message[size - 1 - i] = (uint8_t)((uint64_t)bytes * 8 >> 8 * i);
	for (size_t block = 0; block < size; block += 64) {
		uint32_t w[64];
		for (size_t i = 0; i < 16; ++i)
			w[i] = (uint32_t)message[block + 4 * i] << 24 | (uint32_t)message[block + 4 * i + 1] << 16 |
			       (uint32_t)message[block + 4 * i + 2] << 8 | message[block + 4 * i + 3];
		for (size_t i = 16; i < 64; ++i) {
			uint32_t const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
			uint32_t const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
			w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		}
		uint32_t v[8];
		memcpy(v, h, sizeof v);
		for (size_t i = 0; i < 64; ++i) {
			uint32_t const s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
			uint32_t const t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
			uint32_t const s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
			uint32_t const t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
			memmove(v + 1, v, 7 * sizeof *v);
			v[4] += t1;
			v[0] = t1 + t2;
		}
		for (size_t i = 0; i < 8; ++i)
			h[i] += v[i];
	}
	free(message);
	message = nullptr;
	for (size_t i = 0; i < 8; ++i)
		snprintf(hex + 8 * i, 9, "%08x", h[i]);
}

/** @brief The activation table against the digest that upstream states. */
static void
check_activation_table (void)
{
	uint8_t table[VULKAN_WEIGHTS_ACTIVATION_BYTES];
	vulkan_weights_activation_table(table);
	char hex[65];
	sha256((uint8_t const *)"abc", 3, hex);
	expect(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
	       "the test's SHA-256 is wrong");
	// The digest that nr_activation_lut.hpp states.
	sha256(table, sizeof table, hex);
	expect(!strcmp(hex, "71eed43d50e6447a80db431d73f7d8000f0829b63c9f795f5019a037755558d5"),
	       "the activation table differs from activation_lut_v1");
	size_t runs = 1;
	for (size_t i = 1; i < sizeof table; ++i)
		runs += table[i] != table[i - 1];
	expect(runs == 220, "the activation table has %zu runs, not 220", runs);
}

/** @brief A source of the model pack by its directory's and suffix's short names. */
#define SOURCE(directory_, block_, layer_, suffix_) \
	{VULKAN_DIRECTORY_##directory_, block_, layer_, VULKAN_SUFFIX_##suffix_}

/** @brief An entry that a recipe case reads, and its size in the real model. */
#define ENTRY(directory_, block_, layer_, suffix_, bytes_) {bytes_, SOURCE(directory_, block_, layer_, suffix_)}

/** @brief The entries that the recipe cases read, with their sizes in the real model. */
static struct {
	uint32_t             bytes;
	struct vulkan_source source;
} const ENTRIES[] = {
	ENTRY(UNPACKED, 1, 0, MLP_CONTRACT, 4096),
	ENTRY(UNPACKED, 1, 0, RESIDUAL_SCALE, 96),
	ENTRY(UNPACKED, 1, 0, ATTN_RESIDUAL_SCALE, 80),
	ENTRY(UNPACKED, 1, 0, SCALARS_B, 16),
	ENTRY(UNPACKED, 4, 0, RESAMPLE, 2048),
	ENTRY(UNPACKED, 5, 0, MLP_MID, 8192),
	ENTRY(UNPACKED, 9, 0, ATTN_OUT_PROJ, 16384),
	ENTRY(UNPACKED, 15, 0, QKV, 196608),
	ENTRY(UNPACKED, 15, 0, ATTN_POS_BIAS, 65536),
	ENTRY(UNPACKED, 22, 0, RESAMPLE, 131072),
	ENTRY(UNPACKED, 48, 0, RESIDUAL_SCALE, 544),
	ENTRY(UNPACKED, 48, 0, ATTN_RESIDUAL_SCALE, 528),
	ENTRY(UNPACKED, 48, 0, SCALARS_B, 32),
	ENTRY(UNPACKED, 48, 0, RESAMPLE, 131072),
	ENTRY(UNPACKED, 48, 0, UPSAMPLE_GAIN, 480),
	ENTRY(UNPACKED, 62, 0, RESIDUAL_SCALE, 160),
	ENTRY(UNPACKED, 62, 0, UPSAMPLE_GAIN, 96),
	ENTRY(UNPACKED, 66, 0, RESIDUAL_SCALE, 96),
	ENTRY(UNPACKED, 66, 0, RESAMPLE, 2048),
	ENTRY(UNPACKED, 66, 0, UPSAMPLE_GAIN, 64),
	ENTRY(PREBLOCK, 0, 0, MLP_EXPAND, 4096),
	ENTRY(PREBLOCK, 0, 0, ATTN_POS_BIAS, 8192),
	ENTRY(PREBLOCK, 0, 0, RESIDUAL_SCALE, 80),
	ENTRY(PREBLOCK, 0, 0, INPUT_LIFT, 1024),
	ENTRY(POSTBLOCK, 70, 0, QKV, 3072),
	ENTRY(POSTBLOCK, 70, 0, RESIDUAL_SCALE, 64),
	ENTRY(POSTBLOCK, 70, 0, OUT_PROJECT, 1024),
	ENTRY(POSTBLOCK, 70, 0, SKIP_GAIN, 64),
	ENTRY(POSTBLOCK, 70, 0, MAIN_GAIN, 64),
	ENTRY(SPLIT_SWIN, 23, 1, WEIGHT, 262144),
	ENTRY(SPLIT_SWIN, 23, 1, SKIP_WEIGHT, 1024),
	ENTRY(SPLIT_SWIN, 23, 2, QKV, 786432),
	ENTRY(SPLIT_SWIN, 23, 2, ATTN_POS_BIAS, 131072),
	ENTRY(SPLIT_SWIN, 23, 2, TAIL, 64),
	ENTRY(SPLIT_SWIN, 30, 3, WEIGHT, 262144),
	ENTRY(SPLIT_SWIN, 30, 4, WEIGHT, 524288),
	ENTRY(SPLIT_SWIN, 39, 0, WEIGHT, 524288),
	ENTRY(SPLIT_SWIN, 39, 0, SKIP_WEIGHT, 1024),
	ENTRY(VIT, 31, 0, WEIGHT, 4194304),
	ENTRY(VIT, 31, 1, WEIGHT, 4194304),
	ENTRY(VIT, 31, 1, SKIP_WEIGHT, 2048),
	ENTRY(VIT, 31, 4, WEIGHT, 1048576),
	ENTRY(RECORDS, 23, 0, NONE, 524288),
	ENTRY(RECORDS, 31, 2, NONE, 3145856),
};

#undef ENTRY

/* The flags in the recipe cases. */
#define REQUANTISE VULKAN_SEGMENT_REQUANTISE
#define NPAIR      VULKAN_SEGMENT_NPAIR
#define AFFINE     VULKAN_SEGMENT_AFFINE
#define PADDED     VULKAN_SEGMENT_PADDED
#define GAIN_TAIL  VULKAN_SEGMENT_GAIN_TAIL
#define EXACT      VULKAN_SEGMENT_EXACT

/** @brief A recipe case: a segment and its digest. */
#define CASE(bytes_, recipe_, flags_, directory_, block_, layer_, suffix_, rows_, cols_, index_, digest_) \
	{UINT64_C(digest_), \
	 {.bytes = bytes_, .index = index_, .rows = rows_, .cols = cols_, \
	  .source = SOURCE(directory_, block_, layer_, suffix_), .recipe = VULKAN_RECIPE_##recipe_, \
	  .flags = flags_}}

/** @brief Segments as the network's plan makes them, each with the FNV-1a 64 of the bytes that
 *         upstream's NrSession::build (nr_graph.cpp at 3dfdddc, built with its rdna4.sh defines)
 *         packed for it from the synthetic pack of the real pack's names and sizes
 *         (vulkan_pack_synthetic_entry()). Offsets are the test's. */
static struct {
	uint64_t              digest;
	struct vulkan_segment segment;
} const UPSTREAM_CASES[] = {
	CASE(4096, ACTIVATIONS, 0, UNPACKED, 0, 0, NONE, 0, 0, 0, 0x8802ec9dcae156c8),
	CASE(4096, MATRIX, REQUANTISE | NPAIR, PREBLOCK, 0, 0, MLP_EXPAND, 128, 32, 0, 0x4a1dd5bc7b7a043f),
	CASE(4096, MATRIX, REQUANTISE | NPAIR, UNPACKED, 1, 0, MLP_CONTRACT, 32, 128, 0, 0x9d7bdaef5f4f02df),
	CASE(8192, MATRIX, REQUANTISE | NPAIR, UNPACKED, 5, 0, MLP_MID, 64, 128, 0, 0x97e5e1cb579c9af9),
	CASE(16384, MATRIX, REQUANTISE | NPAIR, UNPACKED, 9, 0, ATTN_OUT_PROJ, 128, 128, 0, 0x3ab19af565304963),
	CASE(196608, MATRIX, REQUANTISE | NPAIR, UNPACKED, 15, 0, QKV, 768, 256, 0, 0xbc91680bcd6a8cdb),
	CASE(3072, MATRIX, REQUANTISE | NPAIR, POSTBLOCK, 70, 0, QKV, 96, 32, 0, 0xfe5f8de627dacc64),
	CASE(262144, MATRIX, NPAIR, SPLIT_SWIN, 23, 1, WEIGHT, 512, 512, 0, 0x7e6bf9d7e27d6949),
	CASE(4194304, MATRIX, NPAIR, VIT, 31, 0, WEIGHT, 4096, 1024, 0, 0xcf1abf13d93ba57f),
	CASE(4194304, MATRIX, NPAIR, VIT, 31, 1, WEIGHT, 1024, 4096, 0, 0xa763bab1010b8eed),
	CASE(1048576, MATRIX, NPAIR, VIT, 31, 4, WEIGHT, 1024, 1024, 0, 0x6864acfdddab50a4),
	CASE(524288, MATRIX, NPAIR, SPLIT_SWIN, 39, 0, WEIGHT, 512, 1024, 0, 0x5dc371446e7d5da6),
	CASE(786432, MATRIX, 0, SPLIT_SWIN, 23, 2, QKV, 1536, 512, 0, 0xef411e3648765299),
	CASE(2048, MATRIX, 0, UNPACKED, 4, 0, RESAMPLE, 64, 32, 0, 0x2b72403c8fa72eeb),
	CASE(131072, MATRIX, 0, UNPACKED, 22, 0, RESAMPLE, 512, 256, 0, 0x8d6b636def36599c),
	CASE(131072, MATRIX, 0, UNPACKED, 48, 0, RESAMPLE, 256, 512, 0, 0x1af391239f32bbcc),
	CASE(2048, MATRIX, 0, UNPACKED, 66, 0, RESAMPLE, 32, 64, 0, 0xf651d9728df2fb38),
	CASE(524288, MATRIX, 0, SPLIT_SWIN, 30, 4, WEIGHT, 1024, 512, 0, 0xbc8f88d92dd05d10),
	CASE(262144, MATRIX, 0, SPLIT_SWIN, 30, 3, WEIGHT, 512, 512, 0, 0x334ebdaaf7fc9b88),
	CASE(262144, FFWD, NPAIR, RECORDS, 23, 0, NONE, 0, 0, 0, 0x8a4b63b5a7d453dc),
	CASE(131072, FFWD, NPAIR, RECORDS, 23, 0, NONE, 0, 0, 1, 0x51be1a24e88a0d6a),
	CASE(131072, FFWD, NPAIR, RECORDS, 23, 0, NONE, 0, 0, 2, 0xa961486978e0c959),
	CASE(3145728, VIT_QKV, NPAIR, RECORDS, 31, 2, NONE, 3072, 1024, 0, 0x10543654d8c5a0db),
	CASE(16384, BIAS, AFFINE, PREBLOCK, 0, 0, ATTN_POS_BIAS, 0, 0, 0, 0x29d00d9b91cd07ec),
	CASE(131072, BIAS, AFFINE, UNPACKED, 15, 0, ATTN_POS_BIAS, 0, 0, 0, 0x6824effc03b242ed),
	CASE(262144, BIAS, 0, SPLIT_SWIN, 23, 2, ATTN_POS_BIAS, 0, 0, 0, 0x42e70e2381b2ca5b),
	CASE(128, SCALES, PADDED, UNPACKED, 1, 0, RESIDUAL_SCALE, 0, 0, 0, 0x566aaf25b9e292b4),
	CASE(1024, SCALES, PADDED, UNPACKED, 48, 0, RESIDUAL_SCALE, 0, 0, 0, 0x7b6555748f951705),
	CASE(128, SCALES, PADDED, PREBLOCK, 0, 0, RESIDUAL_SCALE, 0, 0, 0, 0x9c063e8c9f9d7a01),
	CASE(128, SCALES, PADDED, POSTBLOCK, 70, 0, RESIDUAL_SCALE, 0, 0, 0, 0xd9b3365b18e4cfce),
	CASE(128, SCALES, 0, UNPACKED, 1, 0, ATTN_RESIDUAL_SCALE, 0, 0, 0, 0x397d969de084bff7),
	CASE(1024, HALF, 0, SPLIT_SWIN, 23, 1, SKIP_WEIGHT, 0, 0, 0, 0xc53e7b0d685f0ac1),
	CASE(2048, HALF, 0, VIT, 31, 1, SKIP_WEIGHT, 0, 0, 0, 0x407ca1cfdef06e7c),
	CASE(1024, HALF, 0, SPLIT_SWIN, 39, 0, SKIP_WEIGHT, 0, 0, 0, 0x6b03ceedce682fa6),
	CASE(64, HALF, EXACT, POSTBLOCK, 70, 0, SKIP_GAIN, 0, 0, 0, 0x1aa798b71578e659),
	CASE(64, HALF, EXACT, POSTBLOCK, 70, 0, MAIN_GAIN, 0, 0, 0, 0xbb4ceca14b37fa79),
	CASE(512, HALF, GAIN_TAIL, UNPACKED, 48, 0, UPSAMPLE_GAIN, 0, 0, 0, 0xa87dde9855554156),
	CASE(128, HALF, GAIN_TAIL, UNPACKED, 62, 0, UPSAMPLE_GAIN, 0, 0, 0, 0x02f6426babd71be1),
	CASE(64, HALF, GAIN_TAIL, UNPACKED, 66, 0, UPSAMPLE_GAIN, 0, 0, 0, 0x047542155cf3bf2c),
	CASE(1024, DIAGONAL, PADDED, UNPACKED, 1, 0, RESIDUAL_SCALE, 0, 0, 0, 0x1ccf2228fcda20f6),
	CASE(1024, DIAGONAL, PADDED, POSTBLOCK, 70, 0, RESIDUAL_SCALE, 0, 0, 0, 0x5a01315ad67633f3),
	CASE(8192, DIAGONAL, 0, UNPACKED, 48, 0, ATTN_RESIDUAL_SCALE, 0, 0, 0, 0x58c442f0c6b64f76),
	CASE(16384, DIAGONAL, 0, SPLIT_SWIN, 23, 1, SKIP_WEIGHT, 0, 0, 0, 0x06331066856cf841),
	CASE(32, BYTES, EXACT, UNPACKED, 48, 0, SCALARS_B, 0, 0, 0, 0x9528a1d79173b478),
	CASE(16, BYTES, EXACT, UNPACKED, 1, 0, SCALARS_B, 0, 0, 0, 0xbc7a39464f7bacb7),
	CASE(64, BYTES, EXACT, SPLIT_SWIN, 23, 2, TAIL, 0, 0, 0, 0xfff8d3891fe7d480),
	CASE(1024, BYTES, EXACT, PREBLOCK, 0, 0, INPUT_LIFT, 0, 0, 0, 0xb3355eeae2e0d1ac),
	CASE(1024, BYTES, EXACT, POSTBLOCK, 70, 0, OUT_PROJECT, 0, 0, 0, 0x474e79c7770e7da1),
	CASE(128, BYTES, 0, RECORDS, 31, 2, NONE, 0, 0, 0, 0xc492902d0a05fb75),
	CASE(1024, LIFT, EXACT, PREBLOCK, 0, 0, INPUT_LIFT, 0, 0, 0, 0x93ddcf3b4ed94d70),
};

#undef CASE
#undef EXACT
#undef GAIN_TAIL
#undef PADDED
#undef AFFINE
#undef NPAIR
#undef REQUANTISE

/** @brief The number of recipe cases. */
#define CASE_COUNT (sizeof UPSTREAM_CASES / sizeof *UPSTREAM_CASES)

/** @brief An entry's name in a buffer. */
struct name {
	size_t length;                          //!< The name's length.
	char   text[VULKAN_WEIGHTS_NAME_BYTES]; //!< The name.
};

/** @brief A source's name. */
static struct name
name_of (struct vulkan_source source)
{
	struct name n;
	n.length = vulkan_weights_entry_name(&source, n.text);
	return n;
}

/** @brief Writes a pack of the case entries, in the order of ENTRIES, of synthetic data.
 *
 * @param pack The pack, empty.
 */
static void
synthetic_pack (struct vulkan_pack *pack)
{
	size_t const count = sizeof ENTRIES / sizeof *ENTRIES;
	struct name names[sizeof ENTRIES / sizeof *ENTRIES];
	struct vulkan_pack_entry entries[sizeof ENTRIES / sizeof *ENTRIES];
	for (size_t i = 0; i < count; ++i) {
		names[i] = name_of(ENTRIES[i].source);
		uint8_t *const data = allocated(malloc(ENTRIES[i].bytes));
		vulkan_pack_synthetic_entry(names[i].text, names[i].length, data, ENTRIES[i].bytes);
		entries[i] = (struct vulkan_pack_entry){names[i].text, data, names[i].length, ENTRIES[i].bytes};
	}
	vulkan_pack_entries(pack, entries, count);
	for (size_t i = 0; i < count; ++i) {
		free((void *)entries[i].data);
		entries[i].data = nullptr;
	}
}

/** @brief Every recipe against upstream's digests.
 *
 * @param model The synthetic pack.
 */
static void
check_recipes (struct vulkan_model const *model)
{
	// Every case after the one before, with a gap of 16 bytes between them.
	struct vulkan_segment segments[CASE_COUNT];
	size_t at = 0;
	for (size_t i = 0; i < CASE_COUNT; ++i) {
		segments[i] = UPSTREAM_CASES[i].segment;
		segments[i].offset = (uint32_t)at;
		at += UPSTREAM_CASES[i].segment.bytes + 16;
	}
	size_t const size = at + 7;
	uint8_t *blob = allocated(malloc(size));
	memset(blob, 0xaa, size);
	struct error e;
	enum error_code const code = vulkan_weights_pack(segments, CASE_COUNT, nullptr, 0, model, blob, size, &e);
	if (expect(!code, "packing the cases: %s", code ? e.what : "")) {
		for (size_t i = 0; i < CASE_COUNT; ++i) {
			struct vulkan_segment const *const s = &segments[i];
			struct name const name = name_of(s->source);
			uint8_t const *const bytes = blob + s->offset;
			uint64_t const digest = vulkan_pack_fnv1a(VULKAN_PACK_FNV1A_BASIS, bytes, s->bytes);
			expect(digest == UPSTREAM_CASES[i].digest,
			       "case %zu, %s: packed bytes differ from upstream's", i,
			       s->recipe == VULKAN_RECIPE_ACTIVATIONS ? "the activation table" : name.text);
			bool zeroed = true;
			for (size_t g = 0; g < 16; ++g)
				zeroed = zeroed && !blob[s->offset + s->bytes + g];
			expect(zeroed, "case %zu: the gap after it is not zeroed", i);
		}
		bool zeroed = true;
		for (size_t g = size - 7; g < size; ++g)
			zeroed = zeroed && !blob[g];
		expect(zeroed, "the blob's end is not zeroed");
	}
	free(blob);
	blob = nullptr;
}

#undef CASE_COUNT

/** @brief A table and zeros are packed as words.
 *
 * @param model The synthetic pack.
 */
static void
check_tables (struct vulkan_model const *model)
{
	uint32_t const tables[] = {1, 2, 3, 0xdeadbeef, 5};
	struct vulkan_segment const segments[] = {
		{.offset = 4, .bytes = 8, .index = 2, .recipe = VULKAN_RECIPE_TABLE},
		{.offset = 16, .bytes = 4, .recipe = VULKAN_RECIPE_ZEROS},
	};
	uint8_t blob[24];
	memset(blob, 0xaa, sizeof blob);
	struct error e;
	enum error_code const code = vulkan_weights_pack(segments, 2, tables, 5, model, blob, sizeof blob, &e);
	uint32_t words[6];
	memcpy(words, blob, sizeof words);
	expect(!code && words[0] == 0 && words[1] == 3 && words[2] == 0xdeadbeef && !words[3] && !words[4] &&
	       !words[5],
	       "a table and zeros are not packed as words 0 3 0xdeadbeef 0 0 0");
}

/** @brief The blob of check_pack_errors(). */
#define ERROR_BLOB (UINT32_C(1) << 22)

/** @brief Expects packing segments to fail with given words.
 *
 * @param model    The synthetic pack.
 * @param blob     A blob of ERROR_BLOB bytes.
 * @param segments The segments.
 * @param count    Their number.
 * @param fmt      A printf format for the words.
 * @param ...      The format's arguments.
 */
[[gnu::format(printf, 5, 6)]]
static void
expect_pack_error (struct vulkan_model const   *model,
                   uint8_t                     *blob,
                   struct vulkan_segment const *segments,
                   size_t                       count,
                   char const                  *fmt,
                   ...)
{
	uint32_t const tables[4] = {0};
	char words[ERROR_WHAT_BYTES];
	va_list args;
	va_start(args, fmt);
	vsnprintf(words, sizeof words, fmt, args);
	va_end(args);
	struct error e;
	expect_error(vulkan_weights_pack(segments, count, tables, 4, model, blob, ERROR_BLOB, &e), &e, words);
}

/** @brief A segment of check_pack_errors() at an offset. */
#define SEGMENT(offset_, bytes_, recipe_, flags_, source_, rows_, cols_, index_) \
	(struct vulkan_segment){.offset = offset_, .bytes = bytes_, .index = index_, .rows = rows_, .cols = cols_, \
	                        .source = source_, .recipe = VULKAN_RECIPE_##recipe_, .flags = flags_}

/** @brief Expects packing one segment to fail with given words. */
#define EXPECT_ONE(segment_, ...) expect_pack_error(model, blob, &(segment_), 1, __VA_ARGS__)

/** @brief Pack errors: segments that overlap, leave the blob or do not hold what their recipe
 *         writes, and entries of the wrong size.
 *
 * @param model The synthetic pack.
 */
static void
check_pack_errors (struct vulkan_model const *model)
{
	uint8_t *blob = allocated(malloc(ERROR_BLOB));
	char const *const bad = "network plan: bad weight segment at %u";
	struct vulkan_source const none = {0};
	struct vulkan_segment const zeros = SEGMENT(16, 32, ZEROS, 0, none, 0, 0, 0);
	struct vulkan_segment pair[2] = {zeros, SEGMENT(40, 8, ZEROS, 0, none, 0, 0, 0)};
	expect_pack_error(model, blob, pair, 2, bad, 40);
	pair[1] = SEGMENT(8, 8, ZEROS, 0, none, 0, 0, 0);
	expect_pack_error(model, blob, pair, 2, bad, 8);
	EXPECT_ONE(SEGMENT(4194302, 4, ZEROS, 0, none, 0, 0, 0), bad, 4194302);
	EXPECT_ONE(SEGMENT(4194305, 0, ZEROS, 0, none, 0, 0, 0), bad, 4194305);
	EXPECT_ONE(SEGMENT(0, 12, TABLE, 0, none, 0, 0, 2), bad, 0);
	struct vulkan_source const expand = SOURCE(PREBLOCK, 0, 0, MLP_EXPAND);
	EXPECT_ONE(SEGMENT(0, 4096, MATRIX, VULKAN_SEGMENT_NPAIR, expand, 16, 256, 0), bad, 0);
	EXPECT_ONE(SEGMENT(0, 4000, MATRIX, 0, expand, 128, 32, 0), bad, 0);
	struct vulkan_source const ffwd = SOURCE(RECORDS, 23, 0, NONE);
	EXPECT_ONE(SEGMENT(0, 256, FFWD, 0, ffwd, 0, 0, 3), bad, 0);
	struct vulkan_source const scales = SOURCE(UNPACKED, 1, 0, RESIDUAL_SCALE);
	EXPECT_ONE(SEGMENT(0, 100, DIAGONAL, 0, scales, 0, 0, 0), bad, 0);

	// Short entries, and entries of the wrong size where it must be exact.
	char const *const path = model->path;
	EXPECT_ONE(SEGMENT(0, 8192, MATRIX, 0, expand, 256, 32, 0),
	           "%s: unpacked-preblock/block0.layer0.layer.mlp_expand.bin has 4096 bytes, 8192 needed", path);
	struct vulkan_source const scalars = SOURCE(UNPACKED, 1, 0, SCALARS_B);
	struct vulkan_source const scalars48 = SOURCE(UNPACKED, 48, 0, SCALARS_B);
	EXPECT_ONE(SEGMENT(0, 32, BYTES, VULKAN_SEGMENT_EXACT, scalars, 0, 0, 0),
	           "%s: unpacked/block1.layer0.layer.scalars_b.bin has 16 bytes, expected 32", path);
	EXPECT_ONE(SEGMENT(0, 16, BYTES, VULKAN_SEGMENT_EXACT, scalars48, 0, 0, 0),
	           "%s: unpacked/block48.layer0.layer.scalars_b.bin has 32 bytes, expected 16", path);
	struct vulkan_source const bias_source = SOURCE(PREBLOCK, 0, 0, ATTN_POS_BIAS);
	EXPECT_ONE(SEGMENT(0, 32768, BIAS, 0, bias_source, 0, 0, 0),
	           "%s: unpacked-preblock/block0.layer0.layer.attn_pos_bias.bin has 8192 bytes, 16384 needed", path);
	struct vulkan_source const weight = SOURCE(SPLIT_SWIN, 23, 1, WEIGHT);
	EXPECT_ONE(SEGMENT(0, 131072, FFWD, 0, weight, 0, 0, 1),
	           "%s: unpacked-splitswin/block23.layer1.layer.weight.bin has 262144 bytes, 524288 needed", path);
	EXPECT_ONE(SEGMENT(0, 3145728, VIT_QKV, 0, ffwd, 3072, 1024, 0),
	           "%s: inventory/weights/block23.layer0.layer.bin has 524288 bytes, 3145856 needed", path);
	// Block 1's residual scales start with eight zeros, so 41 of them are read from the ninth on;
	// its attention residual scales from the first.
	EXPECT_ONE(SEGMENT(0, 164, SCALES, VULKAN_SEGMENT_PADDED, scales, 0, 0, 0),
	           "%s: unpacked/block1.layer0.layer.residual_scale.bin has 96 bytes, 98 needed", path);
	struct vulkan_source const attention = SOURCE(UNPACKED, 1, 0, ATTN_RESIDUAL_SCALE);
	EXPECT_ONE(SEGMENT(0, 1536, DIAGONAL, VULKAN_SEGMENT_PADDED, attention, 0, 0, 0),
	           "%s: unpacked/block1.layer0.layer.attn_residual_scale.bin has 80 bytes, 96 needed", path);
	// Block 62's upsample gain holds 48 values: for 272 it needs the last 224 of its 80 residual
	// scales.
	struct vulkan_source const gain62 = SOURCE(UNPACKED, 62, 0, UPSAMPLE_GAIN);
	struct vulkan_source const gain48 = SOURCE(UNPACKED, 48, 0, UPSAMPLE_GAIN);
	EXPECT_ONE(SEGMENT(0, 544, HALF, VULKAN_SEGMENT_GAIN_TAIL, gain62, 0, 0, 0),
	           "%s: unpacked/block62.layer0.layer.residual_scale.bin has 160 bytes, 448 needed", path);
	EXPECT_ONE(SEGMENT(0, 1024, HALF, 0, gain48, 0, 0, 0),
	           "%s: unpacked/block48.layer0.layer.upsample_gain.bin has 480 bytes, 1024 needed", path);
	struct vulkan_source const qkv = SOURCE(POSTBLOCK, 70, 0, QKV);
	EXPECT_ONE(SEGMENT(0, 1024, LIFT, VULKAN_SEGMENT_EXACT, qkv, 0, 0, 0),
	           "%s: unpacked-postblock/block70.layer0.layer.qkv.bin has 3072 bytes, expected 1024", path);
	struct vulkan_source const vit = SOURCE(VIT, 31, 3, WEIGHT);
	EXPECT_ONE(SEGMENT(0, 64, BYTES, 0, vit, 0, 0, 0), "%s: missing unpacked-vit/block31.layer3.layer.weight.bin",
	           path);
	free(blob);
	blob = nullptr;
}

#undef ERROR_BLOB

#undef EXPECT_ONE

/** @brief Residual scales whose first eight values are -0: upstream compared each with 0.0f
 *         (nr_graph.cpp:1825-1826), so it read them from the ninth on too. */
static void
check_negative_zeros (void)
{
	static constexpr size_t COUNT = 32;
	struct vulkan_source const scales = SOURCE(UNPACKED, 1, 0, RESIDUAL_SCALE);
	uint16_t halves[8 + COUNT];
	for (size_t i = 0; i < 8; ++i)
		halves[i] = 0x8000;
	struct vulkan_pack_random random = {0x5eed};
	for (size_t i = 8; i < 8 + COUNT; ++i)
		halves[i] = (uint16_t)vulkan_pack_random_next(&random);
	struct name const name = name_of(scales);
	struct vulkan_pack pack;
	vulkan_pack_init(&pack);
	vulkan_pack_entries(&pack, &(struct vulkan_pack_entry){name.text, (uint8_t const *)halves, name.length,
	                                                       sizeof halves}, 1);
	expect_pack(&pack);
	struct vulkan_model model;
	struct error e;
	enum error_code code = vulkan_model_open(&model, pack.path, &e);
	if (expect(!code, "opening the pack of -0 scales: %s", code ? e.what : "")) {
		struct vulkan_segment const segments[] = {
			SEGMENT(0, 4 * COUNT, SCALES, VULKAN_SEGMENT_PADDED, scales, 0, 0, 0),
			SEGMENT(4 * COUNT, 32 * COUNT, DIAGONAL, VULKAN_SEGMENT_PADDED, scales, 0, 0, 0),
		};
		uint8_t blob[36 * COUNT];
		uint8_t expected[36 * COUNT] = {0};
		memset(blob, 0xaa, sizeof blob);
		for (size_t i = 0; i < COUNT; ++i) {
			float const value = tin_f16_to_f(halves[8 + i]);
			uint16_t const half = tin_f_to_f16(value);
			memcpy(&expected[4 * i], &value, 4);
			// Block i / 16, row and column i % 16.
			memcpy(&expected[4 * COUNT + 2 * (i / 16 * 256 + i % 16 * 17)], &half, 2);
		}
		code = vulkan_weights_pack(segments, 2, nullptr, 0, &model, blob, sizeof blob, &e);
		expect(!code && !memcmp(blob, expected, sizeof blob),
		       "residual scales after eight -0 values are not read from the ninth on");
	}
	vulkan_model_fini(&model);
	vulkan_pack_fini(&pack);
}

/** @brief An entry of check_clamp_free()'s packs. */
struct audited {
	struct name name; //!< Its name.
	uint8_t    *data; //!< Its data.
	size_t      size; //!< Their bytes.
};

/** @brief Writes check_clamp_free()'s entries as a pack.
 *
 * @param pack    The pack, empty.
 * @param entries The entries.
 * @param count   Their number.
 */
static void
write_audited (struct vulkan_pack          *pack,
               struct audited const        *entries,
               size_t                       count)
{
	struct vulkan_pack_entry *list = allocated(malloc(count * sizeof *list));
	for (size_t i = 0; i < count; ++i)
		list[i] = (struct vulkan_pack_entry){entries[i].name.text, entries[i].data, entries[i].name.length,
		                                     entries[i].size};
	vulkan_pack_entries(pack, list, count);
	free(list);
	list = nullptr;
}

/** @brief The audit of the exponent's upper clamp on crafted Swin layers of one or two heads.
 *
 * A head is free when each of its biases, baked, plus 0.044921875 * (1.2 |s| + 0.05) for its scale
 * s is at most 1.5693359375. A last bias of 0x4554, 5.328125, bakes to 1.540130615234375: with the
 * largest scale that frees it, 0x1.000ed6p-1, it reaches 1.56933593559..., and with one ulp more,
 * 1.56933593880..., past the limit, also with the scale negative. The other biases, 0x4500, stay
 * below it. A NaN frees no head. With s = 0 the largest free bias is 0x45ed, and the heads of a
 * layer are free each on its own. The C=512 attention's biases, which have no affine, are not
 * audited.
 */
static void
check_clamp_free (void)
{
	static struct {
		float    scales[4];
		uint32_t free;
		uint16_t biases[2]; // each head's biases
		uint16_t last;      // head 0's last one
		uint8_t  block;
		uint8_t  heads;
	} const layers[] = {
		{{0x1.000ed6p-1f}, 1, {0x4500}, 0x4554, 1, 1},
		{{0x1.000ed8p-1f}, 0, {0x4500}, 0x4554, 2, 1},
		{{-0x1.000ed8p-1f}, 0, {0x4500}, 0x4554, 3, 1},
		{{0}, 0, {0}, 0x7e00, 4, 1},
		{{0}, 1, {0x45ed, 0x45ee}, 0x45ed, 5, 2},
	};
	static constexpr size_t LAYERS = sizeof layers / sizeof *layers;
	static constexpr size_t ENTRIES = 2 * LAYERS + 1;
	struct audited entries[ENTRIES];
	struct vulkan_segment segments[LAYERS + 1];
	struct vulkan_clamp_free expected = {0};
	for (size_t l = 0; l < LAYERS; ++l) {
		struct vulkan_source const bias_source = {VULKAN_DIRECTORY_UNPACKED, layers[l].block, 0,
		                                          VULKAN_SUFFIX_ATTN_POS_BIAS};
		struct vulkan_source const scales = {
			VULKAN_DIRECTORY_UNPACKED, layers[l].block, 0, VULKAN_SUFFIX_SCALARS_B,
		};
		size_t const size = (size_t)layers[l].heads * 8192;
		uint8_t *const biases = allocated(malloc(size));
		for (uint32_t h = 0; h < layers[l].heads; ++h)
			for (size_t i = 0; i < 4096; ++i) {
				uint16_t const value = i == 4095 && !h ? layers[l].last : layers[l].biases[h];
				memcpy(biases + 8192 * h + 2 * i, &value, 2);
			}
		uint8_t *const scale_bytes = allocated(malloc(sizeof layers[l].scales));
		memcpy(scale_bytes, layers[l].scales, sizeof layers[l].scales);
		entries[2 * l] = (struct audited){name_of(bias_source), biases, size};
		entries[2 * l + 1] = (struct audited){name_of(scales), scale_bytes, sizeof layers[l].scales};
		segments[l] = SEGMENT(0, 16384u * layers[l].heads, BIAS, VULKAN_SEGMENT_AFFINE, bias_source, 0, 0, 0);
		expected.heads[layers[l].block] = layers[l].free;
	}
	struct vulkan_source const attention = SOURCE(SPLIT_SWIN, 23, 2, ATTN_POS_BIAS);
	entries[2 * LAYERS] = (struct audited){name_of(attention), allocated(calloc(16 * 8192, 1)), 16 * 8192};
	segments[LAYERS] = SEGMENT(0, 16 * 16384, BIAS, 0, attention, 0, 0, 0);
	struct error e;
	{
		struct vulkan_pack pack;
		vulkan_pack_init(&pack);
		write_audited(&pack, entries, ENTRIES);
		expect_pack(&pack);
		struct vulkan_model model;
		enum error_code code = vulkan_model_open(&model, pack.path, &e);
		if (expect(!code, "opening the pack of audited layers: %s", code ? e.what : "")) {
			struct vulkan_clamp_free free_heads;
			code = vulkan_weights_clamp_free(segments, LAYERS + 1, &model, &free_heads, &e);
			expect(!code && !memcmp(free_heads.heads, expected.heads, sizeof expected.heads),
			       "the heads free of the upper clamp differ from upstream's audit%s%s", code ? ": " : "",
			       code ? e.what : "");
		}
		vulkan_model_fini(&model);
		vulkan_pack_fini(&pack);
	}
	// Entries short of what the audit reads, and one missing: block 1's scale, block 5's biases
	// and block 2's scales.
	struct name const short_scale = entries[1].name;
	struct name const short_biases = entries[8].name;
	struct name const missing = entries[3].name;
	entries[1].size = 2;
	entries[8].size = 16383;
	free(entries[3].data);
	entries[3].data = nullptr;
	memmove(&entries[3], &entries[4], (ENTRIES - 4) * sizeof *entries);
	struct vulkan_pack pack;
	vulkan_pack_init(&pack);
	write_audited(&pack, entries, ENTRIES - 1);
	expect_pack(&pack);
	struct vulkan_model model;
	enum error_code code = vulkan_model_open(&model, pack.path, &e);
	if (expect(!code, "opening the damaged pack: %s", code ? e.what : "")) {
		struct vulkan_clamp_free free_heads;
		char words[ERROR_WHAT_BYTES];
		snprintf(words, sizeof words, "%s: %s has 2 bytes, 4 needed", pack.path, short_scale.text);
		expect_error(vulkan_weights_clamp_free(&segments[0], 1, &model, &free_heads, &e), &e, words);
		snprintf(words, sizeof words, "%s: %s has 16383 bytes, 16384 needed", pack.path, short_biases.text);
		expect_error(vulkan_weights_clamp_free(&segments[4], 1, &model, &free_heads, &e), &e, words);
		snprintf(words, sizeof words, "%s: missing %s", pack.path, missing.text);
		expect_error(vulkan_weights_clamp_free(&segments[1], 1, &model, &free_heads, &e), &e, words);
	}
	vulkan_model_fini(&model);
	vulkan_pack_fini(&pack);
	for (size_t i = 0; i < ENTRIES - 1; ++i) {
		free(entries[i].data);
		entries[i].data = nullptr;
	}
}

#undef SEGMENT

/** @brief Expects a pack to be refused with given words after its path.
 *
 * @param pack  The pack, written.
 * @param words The words.
 */
static void
expect_refused (struct vulkan_pack *pack,
                char const         *words)
{
	expect_pack(pack);
	char expected[ERROR_WHAT_BYTES];
	snprintf(expected, sizeof expected, "%s: %s", pack->path, words);
	struct vulkan_model model;
	struct error e;
	expect_error(vulkan_model_open(&model, pack->path, &e), &e, expected);
	vulkan_model_fini(&model);
	vulkan_pack_fini(pack);
}

/** @brief The bytes of a string literal, without its null. */
#define LITERAL(text) (uint8_t const *)"" text, sizeof "" text - 1

/** @brief The model reader on packs good and damaged. */
static void
check_model (void)
{
	struct vulkan_source const tail = SOURCE(SPLIT_SWIN, 23, 2, TAIL);
	struct vulkan_source const record = SOURCE(RECORDS, 31, 2, NONE);
	struct name const tail_name = name_of(tail);
	struct name const record_name = name_of(record);
	expect(!strcmp(tail_name.text, "unpacked-splitswin/block23.layer2.layer.tail.bin") &&
	       !strcmp(record_name.text, "inventory/weights/block31.layer2.layer.bin"),
	       "entry names differ from upstream's: %s, %s", tail_name.text, record_name.text);
	struct error e;
	{
		struct vulkan_pack pack;
		vulkan_pack_init(&pack);
		struct vulkan_pack_entry const good[] = {
			{tail_name.text, (uint8_t const *)"tail bytes", tail_name.length, 10},
			{record_name.text, (uint8_t const *)"record", record_name.length, 6},
		};
		vulkan_pack_entries(&pack, good, 2);
		expect_pack(&pack);
		struct vulkan_model model;
		enum error_code code = vulkan_model_open(&model, pack.path, &e);
		if (expect(!code, "a good pack is refused: %s", code ? e.what : "")) {
			struct vulkan_scratch scratch = {allocated(malloc(64)), 64};
			memset(scratch.bytes, 0xaa, 64);
			size_t size = 0;
			code = vulkan_model_read(&model, &record, &scratch, &size, &e);
			bool good_read = !code && size == 6 && !memcmp(scratch.bytes, "record", 6);
			code = vulkan_model_read(&model, &tail, &scratch, &size, &e);
			good_read = good_read && !code && size == 10 && !memcmp(scratch.bytes, "tail bytes", 10);
			expect(good_read && scratch.capacity == 64,
			       "a good pack's entries are not read into the start of the scratch");
			char words[ERROR_WHAT_BYTES];
			snprintf(words, sizeof words, "%s: missing unpacked-splitswin/block23.layer2.layer.qkv.bin",
			         pack.path);
			struct vulkan_source const qkv = SOURCE(SPLIT_SWIN, 23, 2, QKV);
			expect_error(vulkan_model_read(&model, &qkv, &scratch, &size, &e), &e, words);
			vulkan_scratch_fini(&scratch);
		}
		vulkan_model_fini(&model);
		vulkan_pack_fini(&pack);
	}
	uint8_t entry[4 + VULKAN_WEIGHTS_NAME_BYTES + 16];
	size_t const entry_bytes = vulkan_pack_index_entry(tail_name.text, tail_name.length,
	                                                   16 + 4 + tail_name.length + 16, 10, entry);
	struct vulkan_pack pack;
	char words[ERROR_WHAT_BYTES];

	vulkan_pack_init(&pack);
	vulkan_pack_write(&pack, LITERAL("NRMODEL"));
	expect_refused(&pack, "not a model pack");

	vulkan_pack_init(&pack);
	vulkan_pack_write(&pack, LITERAL("NRMODEL2"));
	vulkan_pack_write(&pack, &(uint64_t){0}, 8);
	expect_refused(&pack, "not a model pack");

	vulkan_pack_init(&pack);
	vulkan_pack_header(&pack, 2);
	vulkan_pack_write(&pack, entry, entry_bytes);
	vulkan_pack_write(&pack, LITERAL("tail bytes"));
	expect_refused(&pack, "index is damaged");

	vulkan_pack_init(&pack);
	vulkan_pack_header(&pack, 1);
	vulkan_pack_write(&pack, entry, entry_bytes - 1);
	expect_refused(&pack, "index is damaged");

	vulkan_pack_init(&pack);
	vulkan_pack_header(&pack, 1);
	vulkan_pack_write(&pack, &(uint32_t){4097}, 4);
	uint8_t *long_name = allocated(malloc(4097 + 16));
	memset(long_name, 'a', 4097 + 16);
	vulkan_pack_write(&pack, long_name, 4097 + 16);
	free(long_name);
	long_name = nullptr;
	expect_refused(&pack, "index is damaged");

	vulkan_pack_init(&pack);
	vulkan_pack_header(&pack, 1);
	vulkan_pack_write(&pack, entry, entry_bytes);
	vulkan_pack_write(&pack, LITERAL("tail byte"));
	snprintf(words, sizeof words, "entry %s past the end", tail_name.text);
	expect_refused(&pack, words);

	uint8_t far[4 + VULKAN_WEIGHTS_NAME_BYTES + 16];
	size_t const far_bytes = vulkan_pack_index_entry(tail_name.text, tail_name.length, UINT64_MAX, 2, far);
	vulkan_pack_init(&pack);
	vulkan_pack_header(&pack, 1);
	vulkan_pack_write(&pack, far, far_bytes);
	expect_refused(&pack, words);

	vulkan_pack_init(&pack);
	struct vulkan_pack_entry const twice[] = {
		{tail_name.text, (uint8_t const *)"a", tail_name.length, 1},
		{record_name.text, (uint8_t const *)"b", record_name.length, 1},
		{tail_name.text, (uint8_t const *)"c", tail_name.length, 1},
	};
	vulkan_pack_entries(&pack, twice, 3);
	snprintf(words, sizeof words, "entry %s listed twice", tail_name.text);
	expect_refused(&pack, words);

	struct vulkan_model model;
	snprintf(words, sizeof words, "cannot read /nonexistent/dlssnr.bin: %s", strerror(ENOENT));
	expect_error(vulkan_model_open(&model, "/nonexistent/dlssnr.bin", &e), &e, words);
	vulkan_model_fini(&model);
}

#undef LITERAL
#undef SOURCE

int
main (int    argc,
      char **argv)
{
	static struct option const options[] = {
		{"help", no_argument, nullptr, 'h'},
		{},
	};
	for (int code; (code = getopt_long(argc, argv, "+h", options, nullptr)) != -1;) {
		if (code != 'h')
			return 2;
		puts("Usage: vulkan-weights-test [OPTION]...\n"
		     "Checks the Vulkan network's weight packing against upstream's. No GPU or model needed.\n"
		     " -h, --help  Show help (default: off)");
		return 0;
	}
	if (optind != argc)
		return 2;
	check_conversions();
	check_layouts();
	check_gathers();
	check_activation_table();
	check_model();
	check_negative_zeros();
	check_clamp_free();
	struct vulkan_pack pack;
	vulkan_pack_init(&pack);
	synthetic_pack(&pack);
	expect_pack(&pack);
	struct vulkan_model model;
	struct error e;
	enum error_code const code = vulkan_model_open(&model, pack.path, &e);
	if (expect(!code, "opening the synthetic pack: %s", code ? e.what : "")) {
		check_recipes(&model);
		check_tables(&model);
		check_pack_errors(&model);
	}
	vulkan_model_fini(&model);
	vulkan_pack_fini(&pack);
	if (!failures)
		puts("vulkan-weights test: every check passed");
	return failures ? 1 : 0;
}
