/** @file
 *
 * The Vulkan network's weights: vulkan_weights.h.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "error.h"
#include "files.h"
#include "util.h"
#include "vulkan_plan.h"
#include "vulkan_weights.h"

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the blob holds little-endian values");

/** @brief The model pack's directories, by enum vulkan_directory. */
static char const *const DIRECTORY_NAMES[] = {
	[VULKAN_DIRECTORY_UNPACKED]   = "unpacked",
	[VULKAN_DIRECTORY_PREBLOCK]   = "unpacked-preblock",
	[VULKAN_DIRECTORY_POSTBLOCK]  = "unpacked-postblock",
	[VULKAN_DIRECTORY_SPLIT_SWIN] = "unpacked-splitswin",
	[VULKAN_DIRECTORY_VIT]        = "unpacked-vit",
	[VULKAN_DIRECTORY_RECORDS]    = "inventory/weights",
};
static_assert(sizeof DIRECTORY_NAMES / sizeof *DIRECTORY_NAMES == VULKAN_DIRECTORY_RECORDS + 1);

/** @brief The entries' suffixes, by enum vulkan_suffix. */
static char const *const SUFFIX_NAMES[] = {
	[VULKAN_SUFFIX_NONE]                = "",
	[VULKAN_SUFFIX_MLP_EXPAND]          = ".mlp_expand",
	[VULKAN_SUFFIX_MLP_CONTRACT]        = ".mlp_contract",
	[VULKAN_SUFFIX_MLP_MID]             = ".mlp_mid",
	[VULKAN_SUFFIX_QKV]                 = ".qkv",
	[VULKAN_SUFFIX_ATTN_OUT_PROJ]       = ".attn_out_proj",
	[VULKAN_SUFFIX_ATTN_POS_BIAS]       = ".attn_pos_bias",
	[VULKAN_SUFFIX_RESIDUAL_SCALE]      = ".residual_scale",
	[VULKAN_SUFFIX_ATTN_RESIDUAL_SCALE] = ".attn_residual_scale",
	[VULKAN_SUFFIX_SCALARS_B]           = ".scalars_b",
	[VULKAN_SUFFIX_RESAMPLE]            = ".resample",
	[VULKAN_SUFFIX_UPSAMPLE_GAIN]       = ".upsample_gain",
	[VULKAN_SUFFIX_INPUT_LIFT]          = ".input_lift",
	[VULKAN_SUFFIX_OUT_PROJECT]         = ".out_project",
	[VULKAN_SUFFIX_SKIP_GAIN]           = ".skip_gain",
	[VULKAN_SUFFIX_MAIN_GAIN]           = ".main_gain",
	[VULKAN_SUFFIX_WEIGHT]              = ".weight",
	[VULKAN_SUFFIX_SKIP_WEIGHT]         = ".skip_weight",
	[VULKAN_SUFFIX_TAIL]                = ".tail",
};
static_assert(sizeof SUFFIX_NAMES / sizeof *SUFFIX_NAMES == VULKAN_SUFFIX_TAIL + 1);

/** @brief A run of one value in the activation table. */
#define RUN(value_, count_) {.count = count_, .value = value_}

/** @brief The activation table as runs of one value: its 4096 bytes in 220 runs. */
static struct {
	uint16_t count; //!< The run's bytes.
	uint8_t  value; //!< Its value.
} const ACTIVATION_RUNS[] = {
	RUN(0x00, 328), RUN(0x01, 99), RUN(0x02, 46), RUN(0x03, 36), RUN(0x04, 19), RUN(0x05, 18),
	RUN(0x06, 17), RUN(0x07, 16), RUN(0x08, 8), RUN(0x09, 9), RUN(0x0a, 9), RUN(0x0b, 9),
	RUN(0x0c, 8), RUN(0x0d, 9), RUN(0x0e, 9), RUN(0x0f, 4), RUN(0x10, 7), RUN(0x11, 8),
	RUN(0x12, 9), RUN(0x13, 8), RUN(0x14, 9), RUN(0x15, 8), RUN(0x16, 9), RUN(0x17, 5),
	RUN(0x18, 6), RUN(0x19, 9), RUN(0x1a, 8), RUN(0x1b, 8), RUN(0x1c, 8), RUN(0x1d, 8),
	RUN(0x1e, 8), RUN(0x1f, 7), RUN(0x20, 6), RUN(0x21, 8), RUN(0x22, 8), RUN(0x23, 7),
	RUN(0x24, 8), RUN(0x25, 7), RUN(0x26, 8), RUN(0x27, 7), RUN(0x28, 7), RUN(0x29, 8),
	RUN(0x2a, 6), RUN(0x2b, 7), RUN(0x2c, 7), RUN(0x2d, 7), RUN(0x2e, 6), RUN(0x2f, 6),
	RUN(0x30, 10), RUN(0x31, 8), RUN(0x32, 6), RUN(0x33, 6), RUN(0x34, 6), RUN(0x35, 5),
	RUN(0x36, 6), RUN(0x37, 5), RUN(0x38, 8), RUN(0x39, 10), RUN(0x3a, 9), RUN(0x3b, 5),
	RUN(0x3c, 5), RUN(0x3d, 4), RUN(0x3e, 5), RUN(0x3f, 4), RUN(0x40, 7), RUN(0x41, 8),
	RUN(0x42, 9), RUN(0x43, 8), RUN(0x44, 8), RUN(0x45, 4), RUN(0x46, 4), RUN(0x47, 4),
	RUN(0x48, 5), RUN(0x49, 8), RUN(0x4a, 8), RUN(0x4b, 8), RUN(0x4c, 8), RUN(0x4d, 8),
	RUN(0x4e, 8), RUN(0x4f, 4), RUN(0x50, 7), RUN(0x51, 9), RUN(0x52, 9), RUN(0x53, 9),
	RUN(0x54, 9), RUN(0x55, 9), RUN(0x56, 8), RUN(0x57, 4), RUN(0x58, 7), RUN(0x59, 9),
	RUN(0x5a, 9), RUN(0x5b, 9), RUN(0x5c, 9), RUN(0x5d, 9), RUN(0x5e, 8), RUN(0x5f, 4),
	RUN(0x60, 7), RUN(0x61, 9), RUN(0x62, 9), RUN(0x63, 9), RUN(0x64, 9), RUN(0x65, 9),
	RUN(0x66, 8), RUN(0x67, 4), RUN(0x68, 7), RUN(0x69, 9), RUN(0x6a, 9), RUN(0x6b, 9),
	RUN(0x6c, 9), RUN(0x6d, 9), RUN(0x6e, 8), RUN(0x6f, 4), RUN(0x70, 7), RUN(0x71, 9),
	RUN(0x72, 9), RUN(0x73, 9), RUN(0x74, 9), RUN(0x75, 9), RUN(0x76, 8), RUN(0x77, 4),
	RUN(0x78, 7), RUN(0x79, 9), RUN(0x7a, 9), RUN(0x7b, 9), RUN(0x7c, 9), RUN(0x7d, 9),
	RUN(0x7e, 463), RUN(0x7f, 56), RUN(0xff, 64), RUN(0x80, 328), RUN(0x81, 99), RUN(0x82, 47),
	RUN(0x83, 36), RUN(0x84, 19), RUN(0x85, 18), RUN(0x86, 18), RUN(0x87, 15), RUN(0x88, 9),
	RUN(0x89, 9), RUN(0x8a, 9), RUN(0x8b, 9), RUN(0x8c, 9), RUN(0x8d, 10), RUN(0x8e, 7),
	RUN(0x8f, 4), RUN(0x90, 7), RUN(0x91, 10), RUN(0x92, 9), RUN(0x93, 10), RUN(0x94, 9),
	RUN(0x95, 9), RUN(0x96, 7), RUN(0x97, 5), RUN(0x98, 7), RUN(0x99, 10), RUN(0x9a, 10),
	RUN(0x9b, 9), RUN(0x9c, 11), RUN(0x9d, 9), RUN(0x9e, 5), RUN(0x9f, 6), RUN(0xa0, 8),
	RUN(0xa1, 10), RUN(0xa2, 11), RUN(0xa3, 11), RUN(0xa4, 12), RUN(0xa5, 6), RUN(0xa6, 7),
	RUN(0xa7, 6), RUN(0xa8, 10), RUN(0xa9, 14), RUN(0xaa, 15), RUN(0xab, 11), RUN(0xac, 10),
	RUN(0xad, 10), RUN(0xae, 12), RUN(0xaf, 16), RUN(0xb0, 64), RUN(0xaf, 10), RUN(0xae, 5),
	RUN(0xad, 4), RUN(0xac, 4), RUN(0xab, 4), RUN(0xaa, 3), RUN(0xa9, 3), RUN(0xa8, 3),
	RUN(0xa7, 1), RUN(0xa6, 2), RUN(0xa5, 2), RUN(0xa4, 2), RUN(0xa3, 2), RUN(0xa2, 1),
	RUN(0xa1, 2), RUN(0xa0, 1), RUN(0x9f, 2), RUN(0x9e, 1), RUN(0x9d, 1), RUN(0x9c, 1),
	RUN(0x9b, 1), RUN(0x9a, 1), RUN(0x99, 1), RUN(0x98, 1), RUN(0x97, 1), RUN(0x95, 1),
	RUN(0x94, 1), RUN(0x92, 1), RUN(0x91, 1), RUN(0x90, 1), RUN(0x8d, 1), RUN(0x8c, 1),
	RUN(0x89, 1), RUN(0x87, 1), RUN(0x86, 1), RUN(0x85, 1), RUN(0x83, 2), RUN(0x81, 1),
	RUN(0x80, 1), RUN(0x81, 1), RUN(0x80, 897), RUN(0xff, 64),
};

#undef RUN

/** @brief A C=512 FFN record's bytes. */
static constexpr size_t FFWD_RECORD = 524288;

/** @brief A C=512 FFN record's parts as matrices of eight groups of rows (upstream: A, Q0 and Q2 of
 *         the F_FFWD3 lowering, nr_graph.cpp:2852-2893). */
static struct {
	uint16_t rows; //!< The part's rows.
	uint16_t cols; //!< Its columns.
} const FFWD_PARTS[] = {{8 * 64, 512}, {8 * 256, 64}, {8 * 64, 256}};

/* A ViT QKV record: 128 bytes of scales, then Q, K and V of 1024 x 1024. */
static constexpr size_t QKV_SCALES = 128; //!< The scales' bytes.
static constexpr size_t QKV_ROWS = 3072;  //!< The matrices' rows.
static constexpr size_t QKV_COLS = 1024;  //!< Their columns.

/* Math profile 3's exponent affine on a Swin position bias (nr_graph.cpp:1775-1776). */
static constexpr float BIAS_SCALE = 0.044921875f; //!< The affine's scale.
static constexpr float BIAS_OFFSET = 1.30078125f; //!< Its offset.

/* A Swin head's logits are at most 1.2 |s| + 0.05 for its scale s, and the exponent's input, its
 * baked bias plus BIAS_SCALE times its logit, is clamped at EXP_UPPER (upstream: the audit of
 * NR_EXP_NOHI). */
static constexpr double LOGIT_PER_SCALE = 1.2;     //!< The logits' bound per unit of scale.
static constexpr double LOGIT_MARGIN = 0.05;       //!< The bound's margin.
static constexpr double EXP_UPPER = 1.5693359375; //!< The exponent's upper clamp.

size_t
vulkan_weights_entry_name (struct vulkan_source const *source,
                           char                        name[VULKAN_WEIGHTS_NAME_BYTES])
{
	int const n = snprintf(name, VULKAN_WEIGHTS_NAME_BYTES, "%s/block%u.layer%u.layer%s.bin",
	                       DIRECTORY_NAMES[source->directory], source->block, source->layer,
	                       SUFFIX_NAMES[source->suffix]);
	if (n < 0) {
		name[0] = '\0';
		return 0;
	}
	return n < VULKAN_WEIGHTS_NAME_BYTES ? (size_t)n : VULKAN_WEIGHTS_NAME_BYTES - 1;
}

/** @brief A binary16 value from two bytes. */
static uint16_t
load_half (uint8_t const *p)
{
	uint16_t h;
	memcpy(&h, p, 2);
	return h;
}

/** @brief A binary16 value into two bytes. */
static void
store_half (uint8_t  *p,
            uint16_t  h)
{
	memcpy(p, &h, 2);
}

/** @brief A word into four bytes. */
static void
store_word (uint8_t  *p,
            uint32_t  w)
{
	memcpy(p, &w, 4);
}

/** @brief A position bias as the blob holds it: widened, and with an affine through math profile
 *         3's affine. A NaN stays the NaN that widening makes.
 *
 * @param half   The bias.
 * @param affine Whether the affine applies.
 * @return       The f32 bits.
 */
static uint32_t
baked (uint16_t half,
       bool     affine)
{
	uint32_t const wide = vulkan_weights_widen_half(half);
	float value;
	memcpy(&value, &wide, sizeof value);
	if (!affine || value != value)
		return wide;
	float const folded = fmaf(value, BIAS_SCALE, BIAS_OFFSET);
	uint32_t bits;
	memcpy(&bits, &folded, sizeof bits);
	return bits;
}

/** @brief An f32 from its bits. */
static float
float_of (uint32_t bits)
{
	float value;
	memcpy(&value, &bits, sizeof value);
	return value;
}

void
vulkan_scratch_fini (struct vulkan_scratch *scratch)
{
	if (!scratch)
		return;
	free(scratch->bytes);
	scratch->bytes = nullptr;
	scratch->capacity = 0;
}

/** @brief Makes a scratch hold at least some bytes.
 *
 * @param scratch The scratch.
 * @param size    The bytes.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
reserve (struct vulkan_scratch *scratch,
         size_t                 size,
         struct error          *e)
{
	if (scratch->capacity >= size)
		return ERROR_NONE;
	uint8_t *const grown = realloc(scratch->bytes, size);
	if (!grown)
		return error_fail(e, "out of memory");
	scratch->bytes = grown;
	scratch->capacity = size;
	return ERROR_NONE;
}

/** @brief An entry of a model's index. */
struct vulkan_model_entry {
	uint64_t offset; //!< Where its data start in the pack.
	uint64_t size;   //!< Their bytes.
	union {
		size_t      at;   //!< While the index is read: where its name starts in the names.
		char const *name; //!< Then its name, without a null.
	};
	size_t   length; //!< The name's length.
};

/** @brief Orders entries by name, as std::string_view does: by bytes, then the shorter first. */
static int
by_name (void const *a,
         void const *b)
{
	struct vulkan_model_entry const *const x = a;
	struct vulkan_model_entry const *const y = b;
	int const order = memcmp(x->name, y->name, x->length < y->length ? x->length : y->length);
	if (order)
		return order;
	return (x->length > y->length) - (x->length < y->length);
}

/** @brief Puts strerror()'s words for a pack that cannot be read in an error.
 *
 * @param e    The error, or nullptr.
 * @param path The pack.
 * @param err  The errno value.
 * @return     ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
unreadable (struct error *e,
            char const   *path,
            int           err)
{
	char buf[64];
	return error_fail(e, "cannot read %s: %s", path, strerror_r(err, buf, sizeof buf));
}

/** @brief Reads bytes of a pack at an offset.
 *
 * @param fd     The pack.
 * @param path   Its path.
 * @param offset Where the bytes start.
 * @param data   Receives them.
 * @param size   Their number.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
load (int           fd,
      char const   *path,
      uint64_t      offset,
      void         *data,
      size_t        size,
      struct error *e)
{
	enum error_code const code = files_read_at(fd, offset, data, size, e);
	if (code)
		error_wrap(e, "cannot read %s: ", path);
	return code;
}

/** @brief Reads a pack's index into a model whose path and descriptor are set.
 *
 * @param model The model.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
read_index (struct vulkan_model *model,
            struct error        *e)
{
	char const *const path = model->path;
	int const fd = model->fd;
	struct stat st;
	if (fstat(fd, &st)) {
		int const err = errno;
		return unreadable(e, path, err);
	}
	uint64_t const size = (uint64_t)st.st_size;
	char header[16];
	if (size < sizeof header)
		return error_fail(e, "%s: not a model pack", path);
	enum error_code code = load(fd, path, 0, header, sizeof header, e);
	if (code)
		return code;
	if (memcmp(header, "NRMODEL1", 8))
		return error_fail(e, "%s: not a model pack", path);
	uint32_t count;
	memcpy(&count, header + 8, 4);
	// An entry takes at least 20 bytes of the index, so no more fit the pack.
	uint64_t const most = (size - sizeof header) / 20;
	size_t const capacity = count < most ? count : (size_t)most;
	size_t names_size = 0;
	size_t names_capacity = 4096;
	if (capacity) {
		model->entries = malloc(capacity * sizeof *model->entries);
		model->names = malloc(names_capacity);
		if (!model->entries || !model->names)
			return error_fail(e, "out of memory");
	}
	char name[4096 + 16];
	for (uint64_t at = sizeof header; count--;) {
		uint32_t length;
		if (size - at < 4)
			return error_fail(e, "%s: index is damaged", path);
		code = load(fd, path, at, &length, 4, e);
		if (code)
			return code;
		at += 4;
		if (length > 4096 || size - at < length + UINT64_C(16))
			return error_fail(e, "%s: index is damaged", path);
		code = load(fd, path, at, name, length + 16, e);
		if (code)
			return code;
		at += length + 16;
		struct vulkan_model_entry entry = {.length = length};
		memcpy(&entry.offset, name + length, 8);
		memcpy(&entry.size, name + length + 8, 8);
		if (entry.offset > size || entry.size > size - entry.offset)
			return error_fail(e, "%s: entry %.*s past the end", path, (int)length, name);
		if (length > names_capacity - names_size) {
			size_t grown_capacity = names_capacity;
			while (length > grown_capacity - names_size)
				grown_capacity *= 2;
			char *const grown = realloc(model->names, grown_capacity);
			if (!grown)
				return error_fail(e, "out of memory");
			model->names = grown;
			names_capacity = grown_capacity;
		}
		memcpy(model->names + names_size, name, length);
		entry.at = names_size;
		names_size += length;
		model->entries[model->count++] = entry;
	}
	if (!model->count)
		return ERROR_NONE;
	// The names stop moving: each entry points at its own.
	for (size_t i = 0; i < model->count; ++i)
		model->entries[i].name = model->names + model->entries[i].at;
	qsort(model->entries, model->count, sizeof *model->entries, by_name);
	for (size_t i = 1; i < model->count; ++i)
		if (!by_name(&model->entries[i - 1], &model->entries[i]))
			return error_fail(e, "%s: entry %.*s listed twice", path, (int)model->entries[i].length,
			                  model->entries[i].name);
	return ERROR_NONE;
}

enum error_code
vulkan_model_open (struct vulkan_model *dest,
                   char const          *path,
                   struct error        *e)
{
	*dest = (struct vulkan_model){.fd = -1};
	struct vulkan_model model = {.path = path, .fd = open(path, O_RDONLY | O_CLOEXEC)};
	if (model.fd < 0) {
		int const err = errno;
		return unreadable(e, path, err);
	}
	enum error_code const code = read_index(&model, e);
	if (code) {
		vulkan_model_fini(&model);
		return code;
	}
	*dest = model;
	return ERROR_NONE;
}

void
vulkan_model_fini (struct vulkan_model *model)
{
	if (!model)
		return;
	if (model->fd >= 0) {
		close(model->fd);
		model->fd = -1;
	}
	free(model->entries);
	model->entries = nullptr;
	free(model->names);
	model->names = nullptr;
	*model = (struct vulkan_model){.fd = -1};
}

enum error_code
vulkan_model_read (struct vulkan_model const  *model,
                   struct vulkan_source const *source,
                   struct vulkan_scratch      *scratch,
                   size_t                     *size,
                   struct error               *e)
{
	char text[VULKAN_WEIGHTS_NAME_BYTES];
	struct vulkan_model_entry const wanted = {
		.name   = text,
		.length = vulkan_weights_entry_name(source, text),
	};
	struct vulkan_model_entry const *const found =
		model->count ? bsearch(&wanted, model->entries, model->count, sizeof *model->entries, by_name)
		             : nullptr;
	if (!found)
		return error_fail(e, "%s: missing %s", model->path, text);
	enum error_code const code = reserve(scratch, found->size, e);
	if (code)
		return code;
	*size = found->size;
	return load(model->fd, model->path, found->offset, scratch->bytes, found->size, e);
}

/** @brief Puts the words for a segment that is not the plan's in an error. */
[[gnu::cold]]
static enum error_code
bad_segment (struct error                *e,
             struct vulkan_segment const *s)
{
	return error_fail(e, "network plan: bad weight segment at %u", s->offset);
}

/** @brief Whether a segment holds what its recipe writes, inside the tables for TABLE.
 *
 * @param s      The segment.
 * @param tables The tables' words.
 * @return       true if it does.
 */
static bool
shaped (struct vulkan_segment const *s,
        size_t                       tables)
{
	switch (s->recipe) {
	case VULKAN_RECIPE_ZEROS:
	case VULKAN_RECIPE_BYTES:
		return true;
	case VULKAN_RECIPE_TABLE:
		return s->bytes % 4 == 0 && s->index <= tables && s->bytes / 4 <= tables - s->index;
	case VULKAN_RECIPE_ACTIVATIONS:
		return s->bytes == VULKAN_WEIGHTS_ACTIVATION_BYTES;
	case VULKAN_RECIPE_MATRIX:
		return s->rows % (s->flags & VULKAN_SEGMENT_NPAIR ? 32 : 16) == 0 && s->cols % 16 == 0 &&
		       s->bytes == (size_t)s->rows * s->cols;
	case VULKAN_RECIPE_FFWD:
		return s->index < sizeof FFWD_PARTS / sizeof *FFWD_PARTS &&
		       s->bytes == (size_t)FFWD_PARTS[s->index].rows * FFWD_PARTS[s->index].cols;
	case VULKAN_RECIPE_VIT_QKV:
		return s->bytes == QKV_ROWS * QKV_COLS;
	case VULKAN_RECIPE_BIAS:
		return s->bytes % 16384 == 0;
	case VULKAN_RECIPE_SCALES:
		return s->bytes % 4 == 0;
	case VULKAN_RECIPE_HALF:
		return s->bytes % 2 == 0;
	case VULKAN_RECIPE_DIAGONAL:
		return s->bytes % 512 == 0;
	case VULKAN_RECIPE_LIFT:
		return s->bytes == 1024;
	}
	return false;
}

/** @brief A matrix of groups x rows rows of cols codes, each code the byte of a record that a
 *         gather names.
 *
 * Every call passes its gather as a constant, so each inlined copy calls its gather directly.
 *
 * @param record The record.
 * @param groups The groups.
 * @param rows   The rows a group.
 * @param cols   The columns.
 * @param matrix Receives the codes.
 * @param byte   The gather: which byte holds group, row, column.
 */
force_inline void
gather (uint8_t const *record,
        size_t         groups,
        size_t         rows,
        size_t         cols,
        uint8_t       *matrix,
        size_t         byte (size_t, size_t, size_t))
{
	for (size_t g = 0; g < groups; ++g)
		for (size_t r = 0; r < rows; ++r)
			for (size_t c = 0; c < cols; ++c)
				*matrix++ = record[byte(g, r, c)];
}

/** @brief Binary16 values through vulkan_weights_recode_half().
 *
 * @param in    The values.
 * @param count Their number.
 * @param out   Receives them.
 */
static void
recode_halves (uint8_t const *in,
               size_t         count,
               uint8_t       *out)
{
	for (size_t i = 0; i < count; ++i)
		store_half(out + 2 * i, vulkan_weights_recode_half(load_half(in + 2 * i)));
}

/** @brief Whether the first eight binary16 values at a place are zero. */
static bool
zero_eight (uint8_t const *p)
{
	for (size_t i = 0; i < 8; ++i)
		if (load_half(p + 2 * i) & 0x7fff)
			return false;
	return true;
}

/** @brief Packs segment after segment, with scratch for what they read. */
struct packer {
	struct vulkan_model const *model;       //!< The model.
	uint32_t const            *tables;      //!< The words that TABLE segments copy.
	size_t                     table_count; //!< Their number.
	struct vulkan_scratch      entry;       //!< A segment's entry.
	struct vulkan_scratch      tail;        //!< Another entry of its layer.
	struct vulkan_scratch      matrix;      //!< A gathered matrix.
};

/** @brief Frees a packer's scratch. */
static void
packer_fini (struct packer *p)
{
	vulkan_scratch_fini(&p->entry);
	vulkan_scratch_fini(&p->tail);
	vulkan_scratch_fini(&p->matrix);
}

/** @brief Checks an entry's size against the bytes that its recipe reads.
 *
 * @param p      The packer.
 * @param source The entry.
 * @param size   Its bytes.
 * @param needed The bytes that its recipe reads.
 * @param exact  Whether it must hold exactly those.
 * @param e      Receives the words for an entry of the wrong size, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
packer_check (struct packer const        *p,
              struct vulkan_source const *source,
              size_t                      size,
              size_t                      needed,
              bool                        exact,
              struct error               *e)
{
	if (exact ? size == needed : size >= needed)
		return ERROR_NONE;
	char name[VULKAN_WEIGHTS_NAME_BYTES];
	vulkan_weights_entry_name(source, name);
	return exact ? error_fail(e, "%s: %s has %zu bytes, expected %zu", p->model->path, name, size, needed)
	             : error_fail(e, "%s: %s has %zu bytes, %zu needed", p->model->path, name, size, needed);
}

/** @brief Reads a segment's entry, which must hold at least some bytes, or exactly those with
 *         VULKAN_SEGMENT_EXACT.
 *
 * @param p      The packer.
 * @param s      The segment.
 * @param needed The bytes.
 * @param bytes  Receives the entry's bytes, in the packer's entry scratch.
 * @param size   Receives their number, or nullptr.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
packer_read (struct packer               *p,
             struct vulkan_segment const *s,
             size_t                       needed,
             uint8_t                    **bytes,
             size_t                      *size,
             struct error                *e)
{
	size_t got;
	enum error_code code = vulkan_model_read(p->model, &s->source, &p->entry, &got, e);
	if (code)
		return code;
	code = packer_check(p, &s->source, got, needed, s->flags & VULKAN_SEGMENT_EXACT, e);
	if (code)
		return code;
	*bytes = p->entry.bytes;
	if (size)
		*size = got;
	return ERROR_NONE;
}

/** @brief A segment's binary16 values (upstream: load_f16): the first ones, or with
 *         VULKAN_SEGMENT_PADDED those from the ninth on when the first eight are zero
 *         (nr_graph.cpp:1817-1834).
 *
 * @param p      The packer.
 * @param s      The segment.
 * @param count  The values.
 * @param values Receives them, in the packer's entry scratch.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
packer_halves (struct packer               *p,
               struct vulkan_segment const *s,
               size_t                       count,
               uint8_t const              **values,
               struct error                *e)
{
	size_t size;
	enum error_code code = vulkan_model_read(p->model, &s->source, &p->entry, &size, e);
	if (code)
		return code;
	bool const padded = s->flags & VULKAN_SEGMENT_PADDED && size >= 16 && zero_eight(p->entry.bytes);
	size_t const first = padded ? 8 : 0;
	code = packer_check(p, &s->source, size, 2 * (first + count), s->flags & VULKAN_SEGMENT_EXACT, e);
	if (code)
		return code;
	*values = p->entry.bytes + 2 * first;
	return ERROR_NONE;
}

/** @brief A segment's matrix of codes, tile-blocked, and N-paired with VULKAN_SEGMENT_NPAIR. */
static void
blocked (struct vulkan_segment const *s,
         uint8_t const               *matrix,
         size_t                       rows,
         size_t                       cols,
         uint8_t                     *out)
{
	if (s->flags & VULKAN_SEGMENT_NPAIR)
		vulkan_weights_npair_blocked(matrix, rows, cols, out);
	else
		vulkan_weights_tile_blocked(matrix, rows, cols, out);
}

/** @brief Packs a HALF segment. With VULKAN_SEGMENT_GAIN_TAIL, an entry of fewer values than the
 *         segment's is completed in front by the last values of its layer's residual_scale
 *         (nr_graph.cpp:2442-2477).
 *
 * @param p   The packer.
 * @param s   The segment.
 * @param out Receives its bytes.
 * @param e   Receives the words for what stopped it, or nullptr.
 * @return    ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
packer_put_halves (struct packer               *p,
                   struct vulkan_segment const *s,
                   uint8_t                     *out,
                   struct error                *e)
{
	size_t const count = s->bytes / 2;
	size_t size;
	enum error_code code = vulkan_model_read(p->model, &s->source, &p->entry, &size, e);
	if (code)
		return code;
	uint8_t const *const gains = p->entry.bytes;
	size_t const have = size / 2;
	if (!(s->flags & VULKAN_SEGMENT_GAIN_TAIL) || have >= count) {
		code = packer_check(p, &s->source, size, s->bytes, s->flags & VULKAN_SEGMENT_EXACT, e);
		if (code)
			return code;
		recode_halves(gains, count, out);
		return ERROR_NONE;
	}
	struct vulkan_source const residual = {
		s->source.directory, s->source.block, s->source.layer, VULKAN_SUFFIX_RESIDUAL_SCALE,
	};
	size_t scales_size;
	code = vulkan_model_read(p->model, &residual, &p->tail, &scales_size, e);
	if (code)
		return code;
	size_t const missing = count - have;
	code = packer_check(p, &residual, scales_size, 2 * missing, false, e);
	if (code)
		return code;
	recode_halves(p->tail.bytes + 2 * (scales_size / 2 - missing), missing, out);
	recode_halves(gains, have, out + 2 * missing);
	return ERROR_NONE;
}

/** @brief Packs a segment.
 *
 * @param p   The packer.
 * @param s   The segment, which shaped() accepts.
 * @param out Receives its bytes.
 * @param e   Receives the words for what stopped it, or nullptr.
 * @return    ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
packer_put (struct packer               *p,
            struct vulkan_segment const *s,
            uint8_t                     *out,
            struct error                *e)
{
	// Read once: a store through a byte pointer may alias the segment, which would make every loop
	// below read its size again.
	size_t const bytes = s->bytes;
	switch (s->recipe) {
	case VULKAN_RECIPE_ZEROS:
		memset(out, 0, bytes);
		return ERROR_NONE;
	case VULKAN_RECIPE_TABLE:
		memcpy(out, p->tables + s->index, bytes);
		return ERROR_NONE;
	case VULKAN_RECIPE_ACTIVATIONS:
		vulkan_weights_activation_table(out);
		return ERROR_NONE;
	case VULKAN_RECIPE_MATRIX: {
		uint8_t *codes;
		enum error_code code = packer_read(p, s, bytes, &codes, nullptr, e);
		if (code)
			return code;
		if (s->flags & VULKAN_SEGMENT_REQUANTISE)
			for (size_t i = 0; i < bytes; ++i)
				codes[i] = vulkan_weights_requantise(codes[i]);
		blocked(s, codes, s->rows, s->cols, out);
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_FFWD: {
		uint8_t *record;
		enum error_code code = packer_read(p, s, FFWD_RECORD, &record, nullptr, e);
		if (code)
			return code;
		size_t const rows = FFWD_PARTS[s->index].rows;
		size_t const cols = FFWD_PARTS[s->index].cols;
		code = reserve(&p->matrix, bytes, e);
		if (code)
			return code;
		if (s->index == 0)
			gather(record, 8, rows / 8, cols, p->matrix.bytes, vulkan_weights_ff_a);
		else if (s->index == 1)
			gather(record, 8, rows / 8, cols, p->matrix.bytes, vulkan_weights_ff_q0);
		else
			gather(record, 8, rows / 8, cols, p->matrix.bytes, vulkan_weights_ff_q2);
		blocked(s, p->matrix.bytes, rows, cols, out);
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_VIT_QKV: {
		uint8_t *record;
		enum error_code code = packer_read(p, s, QKV_SCALES + bytes, &record, nullptr, e);
		if (code)
			return code;
		code = reserve(&p->matrix, bytes, e);
		if (code)
			return code;
		gather(record, 3, QKV_ROWS / 3, QKV_COLS, p->matrix.bytes, vulkan_weights_vit_qkv_weight_byte);
		blocked(s, p->matrix.bytes, QKV_ROWS, QKV_COLS, out);
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_BIAS: {
		// Each head's values reordered to [i][j] (upstream: deswizzle_bias) and baked.
		size_t const heads = bytes / 16384;
		uint8_t *biases;
		enum error_code code = packer_read(p, s, heads * 8192, &biases, nullptr, e);
		if (code)
			return code;
		bool const affine = s->flags & VULKAN_SEGMENT_AFFINE;
		for (size_t h = 0; h < heads; ++h, biases += 8192)
			for (size_t i = 0; i < 64; ++i)
				for (size_t j = 0; j < 64; ++j, out += 4) {
					size_t const at = vulkan_weights_bias_source(i, j);
					store_word(out, baked(load_half(biases + 2 * at), affine));
				}
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_SCALES: {
		uint8_t const *values;
		enum error_code code = packer_halves(p, s, bytes / 4, &values, e);
		if (code)
			return code;
		for (size_t i = 0; i < bytes / 4; ++i)
			store_word(out + 4 * i, vulkan_weights_widen_half(load_half(values + 2 * i)));
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_HALF:
		return packer_put_halves(p, s, out, e);
	case VULKAN_RECIPE_DIAGONAL: {
		// Block b, row r, column c holds value 16b + r where r is c, else zero.
		uint8_t const *values;
		enum error_code code = packer_halves(p, s, bytes / 32, &values, e);
		if (code)
			return code;
		for (size_t i = 0; i < bytes / 2; ++i) {
			size_t const row = i % 256 / 16;
			uint8_t const *const value = values + 2 * (i / 256 * 16 + row);
			uint16_t const half = row == i % 16 ? vulkan_weights_recode_half(load_half(value)) : 0;
			store_half(out + 2 * i, half);
		}
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_BYTES: {
		uint8_t *entry;
		enum error_code code = packer_read(p, s, bytes, &entry, nullptr, e);
		if (code)
			return code;
		memcpy(out, entry, bytes);
		return ERROR_NONE;
	}
	case VULKAN_RECIPE_LIFT: {
		// The lift's [channel][k] as the input kernel addressed its packed operand
		// (nr_graph.cpp:2133-2140).
		uint8_t *lift;
		enum error_code code = packer_read(p, s, bytes, &lift, nullptr, e);
		if (code)
			return code;
		for (size_t ch = 0; ch < 32; ++ch)
			for (size_t k = 0; k < 16; ++k, out += 2) {
				size_t const from = ch / 16 * 256 + 32 * (ch % 8) + 8 * (k % 8 / 2) +
				                    4 * (ch / 8 % 2) + k % 2 + 2 * (k / 8);
				memcpy(out, lift + 2 * from, 2);
			}
		return ERROR_NONE;
	}
	}
	return ERROR_NONE;
}

/** @brief Packs the segments with a packer.
 *
 * @param p             The packer.
 * @param segments      The segments.
 * @param segment_count Their number.
 * @param blob          Receives the blob.
 * @param blob_size     Its bytes.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
pack_segments (struct packer               *p,
               struct vulkan_segment const *segments,
               size_t                       segment_count,
               uint8_t                     *blob,
               size_t                       blob_size,
               struct error                *e)
{
	size_t at = 0;
	for (size_t i = 0; i < segment_count; ++i) {
		struct vulkan_segment const *const s = &segments[i];
		if (s->offset < at || s->offset > blob_size || s->bytes > blob_size - s->offset ||
		    !shaped(s, p->table_count))
			return bad_segment(e, s);
		memset(blob + at, 0, s->offset - at);
		enum error_code const code = packer_put(p, s, blob + s->offset, e);
		if (code)
			return code;
		at = (size_t)s->offset + s->bytes;
	}
	memset(blob + at, 0, blob_size - at);
	return ERROR_NONE;
}

enum error_code
vulkan_weights_pack (struct vulkan_segment const *segments,
                     size_t                       segment_count,
                     uint32_t const              *tables,
                     size_t                       table_count,
                     struct vulkan_model const   *model,
                     uint8_t                     *blob,
                     size_t                       blob_size,
                     struct error                *e)
{
	struct packer p = {model, tables, table_count, {0}, {0}, {0}};
	enum error_code const code = pack_segments(&p, segments, segment_count, blob, blob_size, e);
	packer_fini(&p);
	return code;
}

/** @brief Audits the segments with a packer.
 *
 * @param p        The packer.
 * @param segments The segments.
 * @param count    Their number.
 * @param dest     Receives the free heads, zeroed before.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
audit (struct packer               *p,
       struct vulkan_segment const *segments,
       size_t                       count,
       struct vulkan_clamp_free    *dest,
       struct error                *e)
{
	for (size_t i = 0; i < count; ++i) {
		struct vulkan_segment const *const s = &segments[i];
		if (s->recipe != VULKAN_RECIPE_BIAS || !(s->flags & VULKAN_SEGMENT_AFFINE))
			continue;
		size_t const heads = s->bytes / 16384;
		uint8_t *biases;
		enum error_code code = packer_read(p, s, heads * 8192, &biases, nullptr, e);
		if (code)
			return code;
		struct vulkan_source const of = {
			s->source.directory, s->source.block, s->source.layer, VULKAN_SUFFIX_SCALARS_B,
		};
		size_t scales_size;
		code = vulkan_model_read(p->model, &of, &p->tail, &scales_size, e);
		if (code)
			return code;
		code = packer_check(p, &of, scales_size, 4 * heads, false, e);
		if (code)
			return code;
		for (size_t h = 0; h < heads; ++h, biases += 8192) {
			float scale;
			memcpy(&scale, p->tail.bytes + 4 * h, 4);
			double const reach = BIAS_SCALE * (LOGIT_PER_SCALE * fabs((double)scale) +
			                                   LOGIT_MARGIN);
			bool clear = true;
			for (size_t b = 0; b < 4096 && clear; ++b) {
				float const baked_bias = float_of(baked(load_half(biases + 2 * b), true));
				clear = (double)baked_bias + reach <= EXP_UPPER;
			}
			dest->heads[s->source.block] |= (uint32_t)clear << h;
		}
	}
	return ERROR_NONE;
}

enum error_code
vulkan_weights_clamp_free (struct vulkan_segment const *segments,
                           size_t                       count,
                           struct vulkan_model const   *model,
                           struct vulkan_clamp_free    *dest,
                           struct error                *e)
{
	*dest = (struct vulkan_clamp_free){0};
	struct packer p = {model, nullptr, 0, {0}, {0}, {0}};
	enum error_code const code = audit(&p, segments, count, dest, e);
	packer_fini(&p);
	return code;
}

uint8_t
vulkan_weights_requantise (uint8_t code)
{
	return code == 0 ? 0x80 : code == 0xff ? 0x7f : code;
}

uint32_t
vulkan_weights_widen_half (uint16_t half)
{
	uint32_t const sign = (uint32_t)(half & 0x8000) << 16;
	uint32_t const e = half >> 10 & 31;
	uint32_t const m = half & 1023;
	if (e == 31)
		return m ? 0x7fc00000 : sign | 0x7f800000;
	if (e)
		return sign | (e + 112) << 23 | m << 13;
	if (!m)
		return sign;
	// A subnormal: shifted until bit 10 is set, the exponent lowered as often.
	uint32_t const shift = (uint32_t)__builtin_clz(m) - 21;
	return sign | (113 - shift) << 23 | (m << shift & 1023) << 13;
}

uint16_t
vulkan_weights_recode_half (uint16_t half)
{
	return (half & 0x7c00) == 0x7c00 && half & 1023 ? 0x7e00 : half;
}

void
vulkan_weights_tile_blocked (uint8_t const *matrix,
                             size_t         rows,
                             size_t         cols,
                             uint8_t       *out)
{
	// Tile (r/16, c/16) is the 16-byte runs of its 16 rows, one after another.
	for (size_t r = 0; r < rows; r += 16)
		for (size_t c = 0; c < cols; c += 16)
			for (size_t row = r; row < r + 16; ++row, out += 16)
				memcpy(out, matrix + row * cols + c, 16);
}

void
vulkan_weights_npair_blocked (uint8_t const *matrix,
                              size_t         rows,
                              size_t         cols,
                              uint8_t       *out)
{
	// Lane l of the pair of tiles (r/16, c/16) and (r/16 + 1, c/16) is 8 columns of row r + l%16
	// of each: c + 8*(l/16) on.
	for (size_t r = 0; r < rows; r += 32)
		for (size_t c = 0; c < cols; c += 16)
			for (size_t lane = 0; lane < 32; ++lane) {
				uint8_t const *const run = matrix + (r + lane % 16) * cols + c + lane / 16 * 8;
				memcpy(out, run, 8);
				memcpy(out + 8, run + 16 * cols, 8);
				out += 16;
			}
}

size_t
vulkan_weights_ff_a (size_t g,
                     size_t row,
                     size_t k)
{
	return ((k & 1) << 0) | ((k >> 1 & 1) << 4) | ((k >> 2 & 1) << 5) | ((k >> 3 & 1) << 1) |
	       ((k >> 4 & 1) << 2) | ((k >> 5 & 1) << 14) | ((k >> 6 & 1) << 15) | ((k >> 7 & 1) << 16) |
	       ((k >> 8 & 1) << 17) | ((row & 1) << 3) | ((row >> 1 & 1) << 6) | ((row >> 2 & 1) << 7) |
	       ((row >> 3 & 1) << 8) | ((row >> 4 & 1) << 9) | ((row >> 5 & 1) << 10) | ((g & 1) << 11) |
	       ((g >> 1 & 1) << 12) | ((g >> 2 & 1) << 13);
}

size_t
vulkan_weights_ff_q0 (size_t g,
                      size_t j,
                      size_t row)
{
	return 262144 + g * 16384 +
	       (((row & 1) << 1) | ((row >> 1 & 1) << 0) | ((row >> 2 & 1) << 4) | ((row >> 3 & 1) << 5) |
	        ((row >> 4 & 1) << 2) | ((row >> 5 & 1) << 13) | ((j & 1) << 6) | ((j >> 1 & 1) << 3) |
	        ((j >> 2 & 1) << 9) | ((j >> 3 & 1) << 7) | ((j >> 4 & 1) << 8) | ((j >> 5 & 1) << 10) |
	        ((j >> 6 & 1) << 11) | ((j >> 7 & 1) << 12));
}

size_t
vulkan_weights_ff_q2 (size_t g,
                      size_t n,
                      size_t j)
{
	return 393216 + g * 16384 +
	       (((j & 1) << 0) | ((j >> 1 & 1) << 1) | ((j >> 2 & 1) << 2) | ((j >> 3 & 1) << 4) |
	        ((j >> 4 & 1) << 5) | ((j >> 5 & 1) << 11) | ((j >> 6 & 1) << 12) | ((j >> 7 & 1) << 13) |
	        ((n & 1) << 6) | ((n >> 1 & 1) << 7) | ((n >> 2 & 1) << 8) | ((n >> 3 & 1) << 3) |
	        ((n >> 4 & 1) << 9) | ((n >> 5 & 1) << 10));
}

size_t
vulkan_weights_vit_qkv_weight_byte (size_t which,
                                    size_t row,
                                    size_t k)
{
	size_t const b = (k & 1) | ((k >> 1 & 1) << 4) | ((k >> 2 & 1) << 5) | ((k >> 3 & 1) << 1) |
	                 ((k >> 4 & 1) << 2) | ((row & 1) << 6) | ((row >> 1 & 1) << 7) |
	                 ((row >> 2 & 1) << 8) | ((row >> 3 & 1) << 3) | ((row >> 4 & 1) << 9);
	size_t const tile = (row >> 5) | ((k >> 5) << 5);
	return QKV_SCALES + (3 * tile + which) * 1024 + b;
}

size_t
vulkan_weights_bias_source (size_t i,
                            size_t j)
{
	size_t const lane = 4 * (i % 8) + j % 8 / 2;
	size_t const slot = (j % 2) | (i % 16 / 8) << 1 | (j % 16 / 8) << 2;
	return (4 * (i / 16) + j / 16) * 256 + lane * 8 + slot;
}

void
vulkan_weights_activation_table (uint8_t table[VULKAN_WEIGHTS_ACTIVATION_BYTES])
{
	for (size_t i = 0; i < sizeof ACTIVATION_RUNS / sizeof *ACTIVATION_RUNS; ++i) {
		memset(table, ACTIVATION_RUNS[i].value, ACTIVATION_RUNS[i].count);
		table += ACTIVATION_RUNS[i].count;
	}
}
