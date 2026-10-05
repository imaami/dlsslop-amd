/** @file
 *
 * The HIP network's weights, read and packed as upstream's loaders pack them.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "files.h"
#include "hip_weights.h"

static_assert(sizeof (float) == sizeof (uint32_t), "a float is 32 bits");

char const *const HIP_WEIGHTS_RECIPE_SUFFIX[] = {
	"", "c32fp8", "ds-cast", "ds-frag", "fp8-g128", "fp8", "fp8-diag", "ffn-frag", "qkv-frag-only",
	"split-mix-f16", "proj-frag", "qkv-frag", "vit-frag", "qkv-f16-frag", "vit-proj-frag",
	"decoder-f16r",
};
static_assert(sizeof HIP_WEIGHTS_RECIPE_SUFFIX / sizeof *HIP_WEIGHTS_RECIPE_SUFFIX == HIP_RECIPE_COUNT,
              "a suffix per recipe");

char const *const HIP_WEIGHTS_FLAW_WORDS[] = {
	"", "nonfinite FP8 matrix weight", "matrix weight not exact FP8 subnormal",
	"matrix weight not exact finite FP8", "nonfinite half weight", "half weight overflow",
	"weight not exact half", "weight not exact half subnormal",
	"nonzero outside grouped contraction",
};
static_assert(sizeof HIP_WEIGHTS_FLAW_WORDS / sizeof *HIP_WEIGHTS_FLAW_WORDS == HIP_FLAW_COUNT,
              "words per flaw");

/** @brief A float's bits. */
static uint32_t
bits (float value)
{
	uint32_t b;
	memcpy(&b, &value, sizeof b);
	return b;
}

/** @brief The float of some bits. */
static float
from_bits (uint32_t b)
{
	float value;
	memcpy(&value, &b, sizeof value);
	return value;
}

/** @brief A stem's type: what follows blockN- or post70-. head-matrix is a ds and
 *         decoder39-weights a weights; post70-scales and post70-head have their own. */
enum stem_type : uint8_t {
	STEM_DS,
	STEM_WEIGHTS,
	STEM_FFWD,
	STEM_FFWD_PROJECTION,
	STEM_FFN,
	STEM_ATTENTION,
	STEM_EXPAND,
	STEM_CONTRACT,
	STEM_QKV,
	STEM_PROJECTION,
	STEM_SCALES,
	STEM_HEAD,
};

/** @brief The names that parse() looks up after blockN- or post70-, by enum stem_type. */
static char const *const type_names[] = {
	"ds", "weights", "ffwd", "ffwd-projection", "ffn", "attention", "expand", "contract", "qkv",
	"projection",
};
static_assert(sizeof type_names / sizeof *type_names == STEM_SCALES, "a name per type but two");

/** @brief The channel counts that a loader applies to, as bits. */
enum channels : uint8_t {
	CHANNELS_32       = 1, //!< 32.
	CHANNELS_ABOVE_32 = 2, //!< More than 32.
	CHANNELS_ANY      = 3, //!< Any.
};

/** @brief Every enum stem_type, as bits. */
#define ANY_TYPE UINT16_C(0xffff)

/** @brief The stems that upstream applies a recipe's loader to: their types and channel counts. */
struct domain {
	uint16_t      types;    //!< Bit enum stem_type of each type.
	enum channels channels; //!< The channel counts.
};

/** @brief The stems of each recipe, by enum hip_recipe. */
static struct domain const domains[] = {
	{ANY_TYPE, CHANNELS_ANY},
	{1 << STEM_FFN | 1 << STEM_ATTENTION, CHANNELS_32},
	{1 << STEM_DS, CHANNELS_32},
	{1 << STEM_DS, CHANNELS_ABOVE_32},
	{1 << STEM_FFN, CHANNELS_ABOVE_32},
	{1 << STEM_ATTENTION, CHANNELS_ABOVE_32},
	{1 << STEM_ATTENTION, CHANNELS_ABOVE_32},
	{1 << STEM_FFN, CHANNELS_ABOVE_32},
	{1 << STEM_ATTENTION, CHANNELS_ABOVE_32},
	{1 << STEM_FFWD, CHANNELS_ABOVE_32},
	{1 << STEM_FFWD_PROJECTION, CHANNELS_ABOVE_32},
	{1 << STEM_ATTENTION, CHANNELS_ABOVE_32},
	{1 << STEM_EXPAND | 1 << STEM_CONTRACT, CHANNELS_ABOVE_32},
	{1 << STEM_QKV, CHANNELS_ABOVE_32},
	{1 << STEM_PROJECTION, CHANNELS_ABOVE_32},
	{1 << STEM_WEIGHTS, CHANNELS_ANY},
};
static_assert(sizeof domains / sizeof *domains == HIP_RECIPE_COUNT, "a domain per recipe");

#undef ANY_TYPE

/** @brief The channel count of a stage's blocks, by its last block (upstream: WeightElements).
 *         Later blocks, and post70 as block 70, have 32. */
struct stage {
	uint16_t c;    //!< The channels.
	uint16_t last; //!< The stage's last block.
};

/** @brief The stages, first to last. */
static struct stage const stages[] = {
	{32, 4}, {64, 8}, {128, 14}, {256, 22}, {512, 30}, {1024, 38}, {512, 47}, {256, 55},
	{128, 61}, {64, 65},
};

/** @brief A stem's element count, channel count and type; the counts are 0 for a stem upstream
 *         does not know. */
struct stem {
	size_t         elements; //!< The values in its file.
	unsigned       c;        //!< Its channels.
	enum stem_type type;     //!< Its type.
};

/** @brief The values in a file of a type at a channel count.
 *
 * @param type The type.
 * @param c    The channels.
 * @return     The count; 0 for the types that parse() knows by name.
 */
static size_t
type_elements (enum stem_type type,
               size_t         c)
{
	size_t const cc = c * c;
	switch (type) {
	case STEM_DS:              return 2 * cc;
	case STEM_WEIGHTS:         return 2 * cc + c;
	case STEM_FFWD:            return 524288;
	case STEM_FFWD_PROJECTION: return 262656;
	case STEM_FFN:             return c == 32 ? 8736 : 9 * cc + c;
	case STEM_ATTENTION:       return c == 32 ? 8225 : 4 * cc + (c / 32) * 4096 + c / 32 + c;
	case STEM_EXPAND:          return 4194304;
	case STEM_CONTRACT:        return 4195328;
	case STEM_QKV:             return 3145760;
	case STEM_PROJECTION:      return 1049600;
	case STEM_SCALES:
	case STEM_HEAD:            break; // parse() knows these stems by name.
	}
	return 0;
}

/** @brief A stem's counts and type (upstream: WeightElements). Its stage rule also gives the
 *         channel count that upstream passes the stem's loaders.
 *
 * @param stem The stem, null-terminated.
 * @return     Its counts and type; zeroed for a stem upstream does not know.
 */
static struct stem
parse (char const *stem)
{
	if (!strcmp(stem, "head-matrix"))
		return (struct stem){524288, 512, STEM_DS};
	if (!strcmp(stem, "decoder39-weights"))
		return (struct stem){524800, 512, STEM_WEIGHTS};
	if (!strcmp(stem, "post70-scales"))
		return (struct stem){64, 32, STEM_SCALES};
	if (!strcmp(stem, "post70-head"))
		return (struct stem){96, 32, STEM_HEAD};

	unsigned block = 70;
	char const *name;
	if (!strncmp(stem, "block", 5)) {
		// As std::from_chars reads an unsigned: one digit at least, no sign, no overflow.
		char const *at = stem + 5;
		if (*at < '0' || *at > '9')
			return (struct stem){0};
		for (block = 0; *at >= '0' && *at <= '9'; ++at) {
			unsigned const digit = (unsigned)(*at - '0');
			if (block > (UINT_MAX - digit) / 10)
				return (struct stem){0};
			block = 10 * block + digit;
		}
		if (*at != '-')
			return (struct stem){0};
		name = at + 1;
	} else if (!strncmp(stem, "post70-", 7)) {
		name = stem + 7;
	} else {
		return (struct stem){0};
	}

	enum stem_type type = STEM_SCALES;
	for (uint8_t t = 0; t < STEM_SCALES; ++t) {
		if (!strcmp(name, type_names[t])) {
			type = (enum stem_type)t;
			break;
		}
	}
	if (type == STEM_SCALES)
		return (struct stem){0};

	unsigned c = 32;
	for (size_t i = 0; i < sizeof stages / sizeof *stages; ++i) {
		if (block <= stages[i].last) {
			c = stages[i].c;
			break;
		}
	}
	return (struct stem){type_elements(type, c), c, type};
}

/** @brief A run of a weight's values: first and count index its floats. */
struct region {
	size_t first; //!< The first.
	size_t count; //!< Their number.
};

/** @brief A matrix of rows x columns bytes from float start on, for fragment tiles. */
struct tiles {
	size_t start;   //!< The float it starts at.
	size_t rows;    //!< Its rows.
	size_t columns; //!< Its columns.
};

/** @brief Parts matrices of rows x columns values as exact binary16 16x16 B-fragment tiles, which
 *         replace the values; the tail floats after the matrices follow. */
struct half_tiles {
	size_t parts;   //!< The matrices.
	size_t rows;    //!< Their rows.
	size_t columns; //!< Their columns.
	size_t tail;    //!< The floats after them.
};

/** @brief The diagonal tiles that a recipe appends. */
enum diagonals : uint8_t {
	DIAGONALS_NONE, //!< None.
	DIAGONALS_C32,  //!< A C32 FFN weight's.
	DIAGONALS_MH,   //!< An MH attention weight's.
};

/** @brief What a recipe does, in the order upstream's loaders do it: the grouped contraction
 *         check, the encodings, then the layouts of the encoded bytes. Each binary16 or E4M3
 *         region is written from its first byte on. */
struct layout {
	struct region     rounded;    //!< Binary16, rounded to nearest even (upstream: HalfR).
	struct region     halves;     //!< Exact binary16 (upstream: HalfExact).
	struct region     saturated;  //!< E4M3 of the values saturated to its range.
	struct region     fp8[3];     //!< Exact E4M3 (upstream: Fp8).
	struct tiles      tiles[3];   //!< Fragment tiles of E4M3 bytes.
	struct half_tiles half_tiles; //!< Binary16 fragment tiles.
	bool              grouped;    //!< Whether the contraction is checked first.
	enum diagonals    diagonals;  //!< The diagonal tiles appended last.
};

#define ATTENTION {{0, 3 * cc}, {3 * cc, cc}}
#define FFN {{0, 4 * cc}, {4 * cc, 4 * cc}, {8 * cc, cc}}

/** @brief What a recipe does to a stem's file.
 *
 * @param recipe The recipe.
 * @param stem   The stem.
 * @return       The layout.
 */
static struct layout
layout (enum hip_recipe    recipe,
        struct stem const *stem)
{
	size_t const c = stem->c, cc = c * c;
	switch (recipe) {
	case HIP_RECIPE_RAW:
		break;
	case HIP_RECIPE_C32:
		if (stem->type == STEM_ATTENTION)
			return (struct layout){.fp8 = {{0, 4096}}};
		return (struct layout){.fp8 = {{512, 4096}, {4608, 4096}}, .diagonals = DIAGONALS_C32};
	case HIP_RECIPE_DS_CAST:
		// Upstream clamps the values to ±448 first, which hip_weights_saturate_fp8() does too.
		return (struct layout){.saturated = {0, 2 * cc}};
	case HIP_RECIPE_DS_FRAG:
		return (struct layout){.half_tiles = {1, 2 * c, c, 0}};
	case HIP_RECIPE_MH_FFN:
		return (struct layout){.fp8 = FFN, .grouped = true};
	case HIP_RECIPE_MH_ATTENTION:
		return (struct layout){.fp8 = ATTENTION};
	case HIP_RECIPE_MH_ATTENTION_DIAG:
		return (struct layout){.fp8 = ATTENTION, .diagonals = DIAGONALS_MH};
	case HIP_RECIPE_FFN_FRAG:
		return (struct layout){
			.fp8 = FFN,
			.tiles = {{0, 4 * c, c}, {4 * cc, c, 4 * c}, {8 * cc, c, c}},
			.grouped = true,
		};
	case HIP_RECIPE_QKV_FRAG_ONLY:
		return (struct layout){.fp8 = ATTENTION, .tiles = {{0, 3 * c, c}}};
	case HIP_RECIPE_SPLIT_MIX_F16:
		return (struct layout){
			.rounded = {0, 262144},
			.halves = {262144, 131072},
			.fp8 = {{393216, 131072}},
		};
	case HIP_RECIPE_PROJ_FRAG:
		return (struct layout){.fp8 = {{0, 262144}}, .tiles = {{0, 512, 512}}};
	case HIP_RECIPE_QKV_FRAG:
		return (struct layout){.fp8 = ATTENTION, .tiles = {{0, 3 * c, c}, {3 * cc, c, c}}};
	case HIP_RECIPE_VIT_FRAG: {
		// The expansion is 4096 rows of 1024, the contraction 1024 of 4096.
		size_t const rows = stem->type == STEM_EXPAND ? 4096 : 1024;
		return (struct layout){.fp8 = {{0, 4194304}}, .tiles = {{0, rows, 4194304 / rows}}};
	}
	case HIP_RECIPE_QKV_F16_FRAG:
		return (struct layout){.half_tiles = {3, 1024, 1024, 32}};
	case HIP_RECIPE_VIT_PROJ_FRAG:
		return (struct layout){.fp8 = {{0, 1048576}}, .tiles = {{0, 1024, 1024}}};
	case HIP_RECIPE_DECODER_F16R:
		return (struct layout){.rounded = {0, 2 * cc}};
	case HIP_RECIPE_COUNT:
		break;
	}
	return (struct layout){0};
}

#undef FFN
#undef ATTENTION

/** @brief The floats of the image that a layout makes of a stem's file.
 *
 * @param l    The layout.
 * @param stem The stem.
 * @return     The floats.
 */
static size_t
packed_floats (struct layout const *l,
               struct stem const   *stem)
{
	struct half_tiles const *const t = &l->half_tiles;
	if (t->parts)
		return t->parts * t->rows * t->columns / 2 + t->tail;
	size_t const diagonals = l->diagonals == DIAGONALS_C32 ? HIP_WEIGHTS_C32_DIAGONAL_FLOATS
	                       : l->diagonals == DIAGONALS_MH  ? 24 * (size_t)stem->c
	                       : 0;
	return stem->elements + diagonals;
}

/** @brief hip_weights_exact_fp8(), which the packing loops inline: gcc does not inline the
 *         public function. */
static inline enum hip_flaw
exact_fp8 (float    value,
           uint8_t *code)
{
	uint32_t const a = bits(value) & 0x7fffffff;
	uint8_t const sign = (uint8_t)(bits(value) >> 24 & 128);
	if (!a) {
		*code = sign;
		return HIP_FLAW_NONE;
	}
	if (a >= 0x7f800000)
		return HIP_FLAW_FP8_NONFINITE;
	if (a < 0x3c800000) {
		// Below 2^-6: an E4M3 subnormal, q * 2^-9 with q 1..7.
		float const q = from_bits(a) * 512.f;
		if (q < 1 || q > 7 || q != (float)(uint32_t)q)
			return HIP_FLAW_FP8_SUBNORMAL;
		*code = (uint8_t)(sign | (uint8_t)q);
		return HIP_FLAW_NONE;
	}
	// The biased exponent is 121 or more here, so E4M3's is 1 or more.
	uint32_t const exponent = (a >> 23) - 120, mantissa = a >> 20 & 7;
	// Exponent 15 with mantissa 7 is NaN.
	if (exponent > 15 || a & 0xfffff || (exponent == 15 && mantissa == 7))
		return HIP_FLAW_FP8_INEXACT;
	*code = (uint8_t)(sign | exponent << 3 | mantissa);
	return HIP_FLAW_NONE;
}

/** @brief The E4M3 code of a value that hip_weights_saturate_fp8() or hip_weights_scale_piece()
 *         made: always exact. */
static uint8_t
fp8_code (float exact)
{
	uint8_t code = 0;
	exact_fp8(exact, &code);
	return code;
}

/** @brief Puts the words for a value that has no exact encoding in an error, out of line so that
 *         the loops that find it build no words.
 *
 * @param element The value's index.
 * @param flaw    Why it has none.
 * @param e       The error, or nullptr.
 * @return        ERROR_FAILED.
 */
[[gnu::cold, gnu::noinline]]
static enum error_code
bad_element (size_t        element,
             enum hip_flaw flaw,
             struct error *e)
{
	return error_fail(e, "element %zu: %s", element, HIP_WEIGHTS_FLAW_WORDS[flaw]);
}

/** @brief Puts "ACTION PATH: " and an errno value's words in an error.
 *
 * @param action What failed.
 * @param path   The file.
 * @param err    The errno value.
 * @param e      The error, or nullptr.
 * @return       ERROR_FAILED.
 */
[[gnu::cold, gnu::noinline]]
static enum error_code
file_error (char const   *action,
            char const   *path,
            int           err,
            struct error *e)
{
	char buf[64];
	return error_fail(e, "%s %s: %s", action, path, strerror_r(err, buf, sizeof buf));
}

size_t
hip_weights_file_elements (struct hip_weight_spec const *spec)
{
	return parse(spec->stem).elements;
}

unsigned
hip_weights_channels (struct hip_weight_spec const *spec)
{
	return parse(spec->stem).c;
}

size_t
hip_weights_packed_bytes (struct hip_weight_spec const *spec)
{
	struct stem const stem = parse(spec->stem);
	struct layout const l = layout(spec->recipe, &stem);
	return 4 * packed_floats(&l, &stem);
}

void
hip_weight_file_fini (struct hip_weight_file *file)
{
	if (!file)
		return;
	free(file->path);
	file->path = nullptr;
	free(file->values);
	file->values = nullptr;
	file->count = 0;
	file->capacity = 0;
}

/** @brief Reads a weight's file once its path is made.
 *
 * @param file     The file, whose path is ASSETS/STEM.f32 of length; the rest is filled in.
 * @param length   The path's length.
 * @param stem     The weight's stem.
 * @param capacity The floats to allocate, at least the stem's elements.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
read_file (struct hip_weight_file *file,
           size_t                  length,
           struct stem const      *stem,
           size_t                  capacity,
           struct error           *e)
{
	int fd = open(file->path, O_RDONLY | O_CLOEXEC);
	bool const full = fd >= 0;
	if (!full) {
		int const err = errno;
		if (err != ENOENT)
			return file_error("open", file->path, err, e);
		memcpy(file->path + length - 2, "16", 2);
		fd = open(file->path, O_RDONLY | O_CLOEXEC);
		if (fd < 0) {
			int const missing = errno;
			if (missing != ENOENT)
				return file_error("open", file->path, missing, e);
			return error_fail(e, "missing weight %.*s (neither .f32 nor .f16)", (int)(length - 4),
			                  file->path);
		}
	}

	enum error_code code = ERROR_NONE;
	struct stat st;
	size_t const bytes = stem->elements * (full ? 4 : 2);
	if (fstat(fd, &st)) {
		code = file_error("read", file->path, errno, e);
	} else if (st.st_size != (off_t)bytes) {
		code = error_fail(e, "%s: %jd bytes, expected %zu", file->path, (intmax_t)st.st_size, bytes);
	} else if (!(file->values = malloc(capacity * sizeof *file->values))) {
		code = error_fail(e, "out of memory");
	} else {
		file->count = stem->elements;
		file->capacity = capacity;
		// A binary16 file goes into the second half of the floats and is widened from the
		// front: each float overwrites only halves widened already.
		uint8_t *const data = (uint8_t *)file->values + stem->elements * 4 - bytes;
		code = files_read_all(fd, data, bytes, e);
		if (code) {
			error_wrap(e, "read %s: ", file->path);
		} else if (!full) {
			for (size_t i = 0; i < stem->elements; ++i) {
				uint16_t half;
				memcpy(&half, data + 2 * i, sizeof half);
				file->values[i] = hip_weights_widen_half(half);
			}
		}
	}
	// Nothing was written to it: close() has nothing to report.
	close(fd);
	fd = -1;
	return code;
}

/** @brief Binary16 of a region's values, rounded to nearest even, from its first byte on.
 *
 * @param values The weight's floats.
 * @param region The region.
 */
static void
round_halves (float               *values,
              struct region const *region)
{
	uint8_t *const halves = (uint8_t *)(values + region->first);
	for (size_t i = 0; i < region->count; ++i) {
		uint16_t const half = hip_weights_round_half(values[region->first + i]);
		memcpy(halves + 2 * i, &half, sizeof half);
	}
}

/** @brief Exact binary16 of a region's values, from its first byte on.
 *
 * @param values The weight's floats.
 * @param region The region.
 * @param e      Receives the words for the first value that has none, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
exact_halves (float               *values,
              struct region const *region,
              struct error        *e)
{
	uint8_t *const halves = (uint8_t *)(values + region->first);
	for (size_t i = 0; i < region->count; ++i) {
		uint16_t half;
		enum hip_flaw const flaw = hip_weights_exact_half(values[region->first + i], &half);
		if (flaw)
			return bad_element(region->first + i, flaw, e);
		memcpy(halves + 2 * i, &half, sizeof half);
	}
	return ERROR_NONE;
}

/** @brief E4M3 of a region's values saturated to its range, one byte each, from its first byte on.
 *
 * @param values The weight's floats.
 * @param region The region.
 */
static void
saturate_fp8s (float               *values,
               struct region const *region)
{
	uint8_t *const codes = (uint8_t *)(values + region->first);
	for (size_t i = 0; i < region->count; ++i)
		codes[i] = fp8_code(hip_weights_saturate_fp8(values[region->first + i]));
}

/** @brief Each region's values as exact E4M3, one byte each, from the region's first byte on
 *         (upstream: PackWeightRegions).
 *
 * @param values  The weight's floats.
 * @param regions The regions.
 * @param count   Their number.
 * @param e       Receives the words for the first value that has none, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
pack_fp8_regions (float               *values,
                  struct region const *regions,
                  size_t               count,
                  struct error        *e)
{
	uint8_t *const codes = (uint8_t *)values;
	// Byte 4*first + i overwrites only values at or before first + i, which are read already.
	for (size_t r = 0; r < count; ++r) {
		// A copy: the stores through codes could otherwise alias the region, which is reloaded.
		struct region const region = regions[r];
		for (size_t i = 0; i < region.count; ++i) {
			uint8_t code;
			enum hip_flaw const flaw = exact_fp8(values[region.first + i], &code);
			if (flaw)
				return bad_element(region.first + i, flaw, e);
			codes[4 * region.first + i] = code;
		}
	}
	return ERROR_NONE;
}

/** @brief Exact binary16 16x16 B-fragment tiles of a weight's matrices, which replace its values;
 *         the tail floats follow them (upstream: the loops of PackedDsWeightFrag and
 *         PackedVitQkvWeightFrag).
 *
 * @param file The file.
 * @param t    The tiles.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
half_tiles (struct hip_weight_file  *file,
            struct half_tiles const *t,
            struct error            *e)
{
	float *const values = file->values;
	size_t const matrix = t->rows * t->columns, halves = t->parts * matrix;
	uint16_t *tiles = malloc(halves * sizeof *tiles);
	if (!tiles)
		return error_fail(e, "out of memory");
	size_t const across = t->columns / 16;
	for (size_t part = 0; part < t->parts; ++part) {
		for (size_t nt = 0; nt < t->rows / 16; ++nt) {
			for (size_t kt = 0; kt < across; ++kt) {
				// Lane (g, rc) = g*16 + rc of tile (nt, kt) holds k = g*8 .. g*8+7 of row
				// nt*16+rc.
				uint16_t *const tile = tiles + part * matrix + (nt * across + kt) * 256;
				for (size_t lane = 0; lane < 32; ++lane) {
					size_t const g = lane / 16, rc = lane % 16;
					size_t const from = part * matrix + (nt * 16 + rc) * t->columns + kt * 16 +
					                    g * 8;
					float const *const run = values + from;
					for (size_t i = 0; i < 8; ++i) {
						uint16_t half;
						enum hip_flaw const flaw = hip_weights_exact_half(run[i], &half);
						if (flaw) {
							free(tiles);
							tiles = nullptr;
							return bad_element(from + i, flaw, e);
						}
						tile[lane * 8 + i] = half;
					}
				}
			}
		}
	}
	memmove(values + halves / 2, values + halves, t->tail * 4);
	file->count = halves / 2 + t->tail;
	memcpy(values, tiles, halves * 2);
	free(tiles);
	tiles = nullptr;
	return ERROR_NONE;
}

/** @brief Packs a file's values as a layout says.
 *
 * @param l    The layout.
 * @param file The file, whose memory holds the image.
 * @param c    The weight's channels.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
pack_values (struct layout const    *l,
             struct hip_weight_file *file,
             unsigned                c,
             struct error           *e)
{
	float *const values = file->values;
	enum error_code code;
	if (l->grouped && (code = hip_weights_check_grouped_contract(values, c, e)))
		return code;
	round_halves(values, &l->rounded);
	if ((code = exact_halves(values, &l->halves, e)))
		return code;
	saturate_fp8s(values, &l->saturated);
	if ((code = pack_fp8_regions(values, l->fp8, sizeof l->fp8 / sizeof *l->fp8, e)))
		return code;
	for (size_t i = 0; i < sizeof l->tiles / sizeof *l->tiles; ++i) {
		struct tiles const *const t = &l->tiles[i];
		if ((code = hip_weights_fragment_tiles(values, t->start, t->rows, t->columns, e)))
			return code;
	}
	if (l->half_tiles.parts && (code = half_tiles(file, &l->half_tiles, e)))
		return code;
	if (l->diagonals == DIAGONALS_C32) {
		hip_weights_append_c32_diagonals(values);
		file->count = HIP_WEIGHTS_C32_FFN_FLOATS + HIP_WEIGHTS_C32_DIAGONAL_FLOATS;
	} else if (l->diagonals == DIAGONALS_MH) {
		hip_weights_append_mh_diagonals(values, file->count, c);
		file->count += 24 * (size_t)c;
	}
	return ERROR_NONE;
}

enum error_code
hip_weights_load (struct hip_weight_file       *dest,
                  char const                   *assets,
                  size_t                        assets_length,
                  struct hip_weight_spec const *spec,
                  struct error                 *e)
{
	*dest = (struct hip_weight_file){0};
	struct stem const stem = parse(spec->stem);
	if (!stem.c)
		return error_fail(e, "unknown weight %s", spec->stem);
	struct domain const *const domain = &domains[spec->recipe];
	if (!(domain->types >> stem.type & 1) ||
	    !(domain->channels & (stem.c == 32 ? CHANNELS_32 : CHANNELS_ABOVE_32)))
		return error_fail(e, "no %s packing of %s", HIP_WEIGHTS_RECIPE_SUFFIX[spec->recipe], spec->stem);

	size_t length = 0;
	dest->path = files_join(assets, assets_length, spec->stem, strlen(spec->stem), &length);
	char *const path = dest->path ? realloc(dest->path, length + 5) : nullptr;
	if (!path) {
		hip_weight_file_fini(dest);
		return error_fail(e, "out of memory");
	}
	dest->path = path;
	memcpy(dest->path + length, ".f32", 5);
	length += 4;

	// The values are read into memory that holds the image too, and packed there.
	struct layout const l = layout(spec->recipe, &stem);
	size_t const packed = packed_floats(&l, &stem);
	enum error_code code = read_file(dest, length, &stem,
	                                 packed < stem.elements ? stem.elements : packed, e);
	if (!code && (code = pack_values(&l, dest, stem.c, e)))
		error_wrap(e, "%s: ", dest->path);
	if (code)
		hip_weight_file_fini(dest);
	return code;
}

enum error_code
hip_weights_pack (struct hip_weight_spec const *spec,
                  struct hip_weight_file       *file,
                  struct error                 *e)
{
	struct stem const stem = parse(spec->stem);
	struct layout const l = layout(spec->recipe, &stem);
	size_t const packed = packed_floats(&l, &stem);
	if (file->capacity < packed) {
		float *const grown = realloc(file->values, packed * sizeof *grown);
		if (!grown)
			return error_fail(e, "out of memory");
		file->values = grown;
		file->capacity = packed;
	}
	enum error_code const code = pack_values(&l, file, stem.c, e);
	if (code)
		error_wrap(e, "%s: ", file->path);
	return code;
}

float
hip_weights_widen_half (uint16_t half)
{
	uint32_t const sign = (uint32_t)(half & 0x8000) << 16, exponent = half >> 10 & 31,
	               mantissa = half & 1023;
	if (exponent)
		return from_bits(sign | (exponent == 31 ? 255 : exponent + 112) << 23 | mantissa << 13);
	if (!mantissa)
		return from_bits(sign);
	// A subnormal: shifted until bit 10 is set, the exponent lowered as often.
	uint32_t const shift = (uint32_t)__builtin_clz(mantissa) - 21;
	return from_bits(sign | (113 - shift) << 23 | (mantissa << shift & 1023) << 13);
}

enum hip_flaw
hip_weights_exact_fp8 (float    value,
                       uint8_t *code)
{
	return exact_fp8(value, code);
}

enum hip_flaw
hip_weights_exact_half (float     value,
                        uint16_t *half)
{
	uint32_t const a = bits(value) & 0x7fffffff;
	uint16_t const sign = (uint16_t)(bits(value) >> 16 & 0x8000);
	if (!a) {
		*half = sign;
		return HIP_FLAW_NONE;
	}
	if (a >= 0x7f800000)
		return HIP_FLAW_HALF_NONFINITE;
	// The biased exponent: binary16's normals are 113 to 142.
	uint32_t const exponent = a >> 23;
	if (exponent > 142)
		return HIP_FLAW_HALF_OVERFLOW;
	if (exponent >= 113) {
		if (a & 8191)
			return HIP_FLAW_HALF_INEXACT;
		*half = (uint16_t)(sign | (exponent - 112) << 10 | (a >> 13 & 1023));
		return HIP_FLAW_NONE;
	}
	// A binary16 subnormal: q * 2^-24 with q 1..1023.
	float const q = from_bits(a) * 16777216.f;
	if (q < 1 || q > 1023 || q != (float)(uint32_t)q)
		return HIP_FLAW_HALF_SUBNORMAL;
	*half = (uint16_t)(sign | (uint16_t)q);
	return HIP_FLAW_NONE;
}

uint16_t
hip_weights_round_half (float value)
{
	uint32_t const a = bits(value) & 0x7fffffff;
	uint16_t const sign = (uint16_t)(bits(value) >> 16 & 0x8000);
	if (a >= 0x7f800000)
		return (uint16_t)(sign | 0x7c00 | (a & 0x7fffff ? 0x200 : 0));
	// The biased exponent: binary16's subnormals are 102 to 112, its normals 113 to 142.
	uint32_t const exponent = a >> 23;
	if (exponent < 102)
		return sign;
	if (exponent < 113) {
		// A binary16 subnormal.
		uint32_t const m = (a & 0x7fffff) | 0x800000, n = 126 - exponent, rest = m & ((1u << n) - 1),
		               half_way = 1u << (n - 1);
		uint32_t const q = (m >> n) + (rest > half_way || (rest == half_way && m >> n & 1));
		return (uint16_t)(sign | q);
	}
	if (exponent > 142)
		return (uint16_t)(sign | 0x7c00);
	uint32_t const h = ((a + 0xfff + (a >> 13 & 1)) >> 13) - 0x1c000;
	return (uint16_t)(sign | (h >= 0x7c00 ? 0x7c00 : h));
}

float
hip_weights_saturate_fp8 (float value)
{
	uint32_t const a = bits(value) & 0x7fffffff, sign = bits(value) & 0x80000000;
	if (!a)
		return 0.f;
	if (a >= 0x43e00000)
		return from_bits(sign | 0x43e00000);
	if (a < 0x3c800000) {
		// An E4M3 subnormal, q * 2^-9, rounded to nearest even.
		float const scaled = from_bits(a) * 512.f;
		uint32_t q = (uint32_t)scaled;
		float const rest = scaled - (float)q;
		q += rest > .5f || (rest == .5f && q & 1);
		float const magnitude = (float)q / 512.f;
		return sign ? -magnitude : magnitude;
	}
	uint32_t const rounded = (a + 0x7ffff + (a >> 20 & 1)) & 0xfff00000;
	return from_bits(sign | (rounded > 0x43e00000 ? 0x43e00000 : rounded));
}

float
hip_weights_scale_piece (float value)
{
	float const magnitude = fabsf(value);
	if (!(magnitude < .015625f))
		return hip_weights_saturate_fp8(value);
	float const scaled = magnitude * 512.f;
	uint32_t q = (uint32_t)scaled;
	float const rest = scaled - (float)q;
	q += rest > .5f || (rest == .5f && q & 1);
	// The device keeps a piece below 2^-6 subnormal too: 8 * 2^-9 becomes 7 * 2^-9.
	if (q > 7)
		q = 7;
	return from_bits(bits((float)q / 512.f) | (bits(value) & 0x80000000));
}

enum error_code
hip_weights_fragment_tiles (float        *values,
                            size_t        start,
                            size_t        rows,
                            size_t        columns,
                            struct error *e)
{
	size_t const bytes = rows * columns;
	if (!bytes)
		return ERROR_NONE;
	// Tile (n/16, k/32) is 512 bytes: runs of 8 k of one row, by k%32/8, then n%16.
	uint8_t *const tiles = (uint8_t *)(values + start);
	uint8_t *matrix = malloc(bytes);
	if (!matrix)
		return error_fail(e, "out of memory");
	memcpy(matrix, tiles, bytes);
	for (size_t n = 0; n < rows; ++n)
		for (size_t k = 0; k < columns; k += 8)
			memcpy(tiles + ((n / 16) * (columns / 32) + k / 32) * 512 + (k % 32 / 8 * 16 + n % 16) * 8,
			       matrix + n * columns + k, 8);
	free(matrix);
	matrix = nullptr;
	return ERROR_NONE;
}

void
hip_weights_append_c32_diagonals (float *values)
{
	// Byte (gr*16 + rc)*8 + e of tile (part*2 + ci)*2 + kt is piece PART of the scale of
	// column c = ci*16 + rc where k = kt*16 + gr*8 + e is c, else 0.
	uint8_t *const tiles = (uint8_t *)(values + HIP_WEIGHTS_C32_FFN_FLOATS);
	memset(tiles, 0, 4 * HIP_WEIGHTS_C32_DIAGONAL_FLOATS);
	for (unsigned c = 0; c < 32; ++c) {
		unsigned const ci = c / 16, rc = c % 16;
		float remaining = values[8704 + c];
		for (unsigned part = 0; part < 3; ++part) {
			float const piece = hip_weights_scale_piece(remaining);
			remaining -= piece;
			tiles[((part * 2 + ci) * 2 + ci) * 512 + (rc / 8 * 16 + rc) * 8 + rc % 8] = fp8_code(piece);
		}
	}
}

void
hip_weights_append_mh_diagonals (float   *values,
                                 size_t   count,
                                 unsigned c)
{
	// Byte (gr*16 + rc)*8 + e of tile part*(c/16) + ct is piece PART of the scale of column
	// ct*16 + rc where k = gr*8 + e is rc, else 0.
	size_t const cc = (size_t)c * c, heads = c / 32, scales = 4 * cc + heads * 4096 + heads;
	uint8_t *const tiles = (uint8_t *)(values + count);
	memset(tiles, 0, 4 * 24 * (size_t)c);
	for (unsigned column = 0; column < c; ++column) {
		unsigned const ct = column / 16, rc = column % 16;
		float remaining = values[scales + column];
		for (unsigned part = 0; part < 3; ++part) {
			float const piece = hip_weights_scale_piece(remaining);
			remaining -= piece;
			tiles[(part * (c / 16) + ct) * 512 + (rc / 8 * 16 + rc) * 8 + rc % 8] = fp8_code(piece);
		}
	}
}

enum error_code
hip_weights_check_grouped_contract (float const  *values,
                                    unsigned      c,
                                    struct error *e)
{
	// The contraction, c rows of 4c after the expansion: row r may be nonzero only in
	// k 128*(r/32) .. 128*(r/32)+127.
	size_t const width = 4 * (size_t)c, matrix = width * c;
	for (size_t row = 0; row < c; ++row)
		for (size_t k = 0; k < width; ++k)
			if (k / 128 != row / 32 && values[matrix + row * width + k] != 0.f)
				return bad_element(matrix + row * width + k, HIP_FLAW_UNGROUPED, e);
	return ERROR_NONE;
}
