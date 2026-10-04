/** @file
 *
 * Model packs for the Vulkan network's host tests: vulkan_pack.h.
 */
// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "error.h"
#include "files.h"
#include "vulkan_pack.h"
#include "vulkan_plan.h"
#include "vulkan_weights.h"

uint64_t
vulkan_pack_fnv1a (uint64_t    hash,
                   void const *data,
                   size_t      bytes)
{
	for (unsigned char const *at = data; bytes--; ++at)
		hash = (hash ^ *at) * UINT64_C(0x100000001b3);
	return hash;
}

uint64_t
vulkan_pack_random_next (struct vulkan_pack_random *random)
{
	uint64_t z = random->state += UINT64_C(0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
	return z ^ (z >> 31);
}

/** @brief Whether a name of a length starts with a literal. */
#define STARTS_WITH(name, length, literal) \
	((length) >= sizeof literal - 1 && !memcmp((name), literal, sizeof literal - 1))

/** @brief Whether a name of a length ends with a literal. */
#define ENDS_WITH(name, length, literal) \
	((length) >= sizeof literal - 1 && \
	 !memcmp((name) + (length) - (sizeof literal - 1), literal, sizeof literal - 1))

void
vulkan_pack_synthetic_entry (char const *name,
                             size_t      length,
                             uint8_t    *data,
                             size_t      bytes)
{
	struct vulkan_pack_random random = {vulkan_pack_fnv1a(VULKAN_PACK_FNV1A_BASIS, name, length)};
	for (size_t at = 0; at < bytes; at += 8) {
		uint64_t const x = vulkan_pack_random_next(&random);
		memcpy(data + at, &x, bytes - at < 8 ? bytes - at : 8);
	}
	if (STARTS_WITH(name, length, "unpacked/") && ENDS_WITH(name, length, ".residual_scale.bin") &&
	    !STARTS_WITH(name, length, "unpacked/block48.") &&
	    !STARTS_WITH(name, length, "unpacked/block56.") &&
	    !STARTS_WITH(name, length, "unpacked/block62."))
		memset(data, 0, bytes < 16 ? bytes : 16);
}

#undef ENDS_WITH
#undef STARTS_WITH

void
vulkan_pack_init (struct vulkan_pack *dest)
{
	*dest = (struct vulkan_pack){.fd = memfd_create("vulkan-test-pack", MFD_CLOEXEC)};
	int const n = snprintf(dest->path, sizeof dest->path, "/proc/self/fd/%d", dest->fd);
	dest->ok = n > 0 && n < (int)sizeof dest->path;
}

void
vulkan_pack_fini (struct vulkan_pack *pack)
{
	if (!pack)
		return;
	if (pack->fd >= 0) {
		close(pack->fd);
		pack->fd = -1;
	}
	pack->ok = false;
}

void
vulkan_pack_write (struct vulkan_pack *pack,
                   void const         *data,
                   size_t              size)
{
	pack->ok = pack->ok && pack->fd >= 0 && files_write_all(pack->fd, data, size, nullptr) == ERROR_NONE;
}

void
vulkan_pack_header (struct vulkan_pack *pack,
                    uint32_t            count)
{
	uint8_t header[16] = "NRMODEL1";
	memcpy(header + 8, &count, 4);
	vulkan_pack_write(pack, header, sizeof header);
}

size_t
vulkan_pack_index_entry (char const *name,
                         size_t      length,
                         uint64_t    offset,
                         uint64_t    size,
                         uint8_t     out[4 + VULKAN_WEIGHTS_NAME_BYTES + 16])
{
	uint32_t const name_length = (uint32_t)length;
	memcpy(out, &name_length, 4);
	memcpy(out + 4, name, length);
	memcpy(out + 4 + length, &offset, 8);
	memcpy(out + 4 + length + 8, &size, 8);
	return 4 + length + 16;
}

void
vulkan_pack_entries (struct vulkan_pack             *pack,
                     struct vulkan_pack_entry const *entries,
                     size_t                          count)
{
	vulkan_pack_header(pack, (uint32_t)count);
	uint64_t offset = 16;
	for (size_t i = 0; i < count; ++i)
		offset += 4 + entries[i].length + 16;
	for (size_t i = 0; i < count; ++i) {
		uint8_t entry[4 + VULKAN_WEIGHTS_NAME_BYTES + 16];
		size_t const bytes = vulkan_pack_index_entry(entries[i].name, entries[i].length, offset,
		                                             entries[i].size, entry);
		vulkan_pack_write(pack, entry, bytes);
		offset += entries[i].size;
	}
	for (size_t i = 0; i < count; ++i)
		vulkan_pack_write(pack, entries[i].data, entries[i].size);
}

uint32_t
vulkan_pack_model_bytes (struct vulkan_source const *s)
{
	uint32_t const b = s->block;
	if (s->directory == VULKAN_DIRECTORY_RECORDS)
		return b >= 31 && b <= 38 ? 3145856 : 524288;
	if (s->directory == VULKAN_DIRECTORY_VIT)
		return s->suffix == VULKAN_SUFFIX_SKIP_WEIGHT ? 2048 : s->layer == 4 ? 1048576 : 4194304;
	if (s->directory == VULKAN_DIRECTORY_SPLIT_SWIN)
		switch (s->suffix) {
		case VULKAN_SUFFIX_QKV:
			return 786432;
		case VULKAN_SUFFIX_ATTN_POS_BIAS:
			return 131072;
		case VULKAN_SUFFIX_SKIP_WEIGHT:
			return 1024;
		case VULKAN_SUFFIX_TAIL:
			return 64;
		default:
			return (b == 30 && s->layer == 4) || b == 39 ? 524288 : 262144;
		}
	// The Swin blocks: at C=32 the pre and post blocks and the first and last levels, then 64, 128
	// and 256 down to block 22 and back from block 48.
	bool const edge = s->directory != VULKAN_DIRECTORY_UNPACKED;
	uint32_t const c = edge || b <= 4 || b >= 66 ? 32
	                 : b <= 8 || b >= 62         ? 64
	                 : b <= 14 || b >= 56        ? 128
	                                             : 256;
	uint32_t const heads = c / 32;
	switch (s->suffix) {
	case VULKAN_SUFFIX_MLP_EXPAND:
		return 4 * c * c;
	case VULKAN_SUFFIX_MLP_CONTRACT:
		return c * (heads > 1 ? c : 4 * c);
	case VULKAN_SUFFIX_MLP_MID:
		return c * 128;
	case VULKAN_SUFFIX_QKV:
		return 3 * c * c;
	case VULKAN_SUFFIX_ATTN_OUT_PROJ:
		return c * c;
	case VULKAN_SUFFIX_ATTN_POS_BIAS:
		return heads * 8192;
	case VULKAN_SUFFIX_RESIDUAL_SCALE:
		return s->directory == VULKAN_DIRECTORY_PREBLOCK ? 80
		     : s->directory == VULKAN_DIRECTORY_POSTBLOCK ? 64
		                                                  : 2 * (c + 16);
	case VULKAN_SUFFIX_ATTN_RESIDUAL_SCALE:
		return edge ? 80 : b == 4 || b == 8 || b == 14 || b == 22 ? 2 * c : 2 * (c + 8);
	case VULKAN_SUFFIX_SCALARS_B:
		return 4 * (4 < heads ? heads : 4);
	case VULKAN_SUFFIX_RESAMPLE:
		return 2 * c * c;
	case VULKAN_SUFFIX_UPSAMPLE_GAIN:
		return c >= 64 ? 2 * (c - 16) : 2 * c;
	case VULKAN_SUFFIX_SKIP_GAIN:
	case VULKAN_SUFFIX_MAIN_GAIN:
		return 64;
	default:
		return 1024; // input_lift, out_project
	}
}

/** @brief Orders entries by name. */
static int
by_name (void const *a,
         void const *b)
{
	struct vulkan_pack_source const *const x = a;
	struct vulkan_pack_source const *const y = b;
	return strcmp(x->name, y->name);
}

/** @brief An entry by its source. */
static struct vulkan_pack_source
source_entry (struct vulkan_source source)
{
	struct vulkan_pack_source entry = {.source = source};
	entry.length = vulkan_weights_entry_name(&source, entry.name);
	return entry;
}

struct vulkan_pack_source *
vulkan_pack_plan_entries (struct vulkan_plan const *plan,
                          size_t                   *count)
{
	// Each segment reads one entry, and a short upsample gain the tail of its layer's residual
	// scales too.
	struct vulkan_pack_source *const read = malloc((2 * plan->segment_count + 1) * sizeof *read);
	if (!read)
		return nullptr;
	size_t n = 0;
	for (size_t i = 0; i < plan->segment_count; ++i) {
		struct vulkan_segment const *const s = &plan->segments[i];
		if (s->recipe == VULKAN_RECIPE_ZEROS || s->recipe == VULKAN_RECIPE_TABLE ||
		    s->recipe == VULKAN_RECIPE_ACTIVATIONS)
			continue;
		read[n++] = source_entry(s->source);
		if (s->flags & VULKAN_SEGMENT_GAIN_TAIL) {
			struct vulkan_source scales = s->source;
			scales.suffix = VULKAN_SUFFIX_RESIDUAL_SCALE;
			read[n++] = source_entry(scales);
		}
	}
	qsort(read, n, sizeof *read, by_name);
	size_t unique = 0;
	for (size_t i = 0; i < n; ++i)
		if (!unique || strcmp(read[unique - 1].name, read[i].name))
			read[unique++] = read[i];
	*count = unique;
	return read;
}

bool
vulkan_pack_synthetic_model (struct vulkan_plan const *plan,
                             struct vulkan_pack       *pack,
                             bool                      unclamped)
{
	size_t count = 0;
	struct vulkan_pack_source *read = vulkan_pack_plan_entries(plan, &count);
	if (!read)
		return false;
	vulkan_pack_header(pack, (uint32_t)count);
	uint64_t offset = 16;
	size_t largest = 0;
	for (size_t i = 0; i < count; ++i) {
		offset += 4 + read[i].length + 16;
		size_t const bytes = vulkan_pack_model_bytes(&read[i].source);
		if (largest < bytes)
			largest = bytes;
	}
	for (size_t i = 0; i < count; ++i) {
		uint8_t entry[4 + VULKAN_WEIGHTS_NAME_BYTES + 16];
		uint32_t const bytes = vulkan_pack_model_bytes(&read[i].source);
		size_t const entry_bytes = vulkan_pack_index_entry(read[i].name, read[i].length, offset, bytes,
		                                                   entry);
		vulkan_pack_write(pack, entry, entry_bytes);
		offset += bytes;
	}
	uint8_t *data = malloc(largest ? largest : 1);
	if (data) {
		for (size_t i = 0; i < count; ++i) {
			uint32_t const bytes = vulkan_pack_model_bytes(&read[i].source);
			enum vulkan_suffix const suffix = read[i].source.suffix;
			bool const audited = suffix == VULKAN_SUFFIX_ATTN_POS_BIAS ||
			                     suffix == VULKAN_SUFFIX_SCALARS_B;
			if (unclamped && audited)
				memset(data, 0, bytes);
			else
				vulkan_pack_synthetic_entry(read[i].name, read[i].length, data, bytes);
			vulkan_pack_write(pack, data, bytes);
		}
	}
	bool const ok = data && pack->ok;
	free(data);
	data = nullptr;
	free(read);
	read = nullptr;
	return ok;
}
