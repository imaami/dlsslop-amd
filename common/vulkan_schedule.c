/** @file
 *
 * The Vulkan network's schedule: vulkan_plan_priv.h's merge of the Swin bodies into persistent runs,
 * the arena shared by lifetime, the trimmed rows and the tile counters.
 */
// SPDX-License-Identifier: MIT
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "vulkan_plan.h"
#include "vulkan_plan_priv.h"

/** @brief A number rounded up to a multiple of an alignment. */
static uint64_t
align (uint64_t n,
       uint64_t alignment)
{
	return (n + alignment - 1) / alignment * alignment;
}

/** @brief Whether a kernel is a downsampling persistent run. */
static bool
downsampling_run (enum vulkan_kernel k)
{
	return k >= VULKAN_KERNEL_FSWIN_PDS64 && k <= VULKAN_KERNEL_FSWIN_PDS256;
}

/** @brief Whether a kernel is an upsampling persistent run. */
static bool
upsampling_run (enum vulkan_kernel k)
{
	return k >= VULKAN_KERNEL_FSWIN_PUP64 && k <= VULKAN_KERNEL_FSWIN_PUP256;
}

/** @brief Whether a kernel is a GEMM. */
static bool
gemm (enum vulkan_kernel k)
{
	return k >= VULKAN_KERNEL_GEMM_PROJC && k <= VULKAN_KERNEL_GEMM_VQKVS;
}

/** @brief Whether a kernel is the C=512 levels' projection. */
static bool
c512 (enum vulkan_kernel k)
{
	return k == VULKAN_KERNEL_GEMM_PROJC;
}

/** @brief Whether a kernel is the C=512 FFN. */
static bool
ffn (enum vulkan_kernel k)
{
	return k == VULKAN_KERNEL_FFWD3 || k == VULKAN_KERNEL_FFWD3W;
}

/** @brief Whether a kernel is one of the ViT's residual projections. */
static bool
projection (enum vulkan_kernel k)
{
	return k == VULKAN_KERNEL_GEMM_PROJ || k == VULKAN_KERNEL_GEMM_PROJW || k == VULKAN_KERNEL_GEMM_PROJT;
}

/** @brief Whether a kernel is the ViT's QKV. */
static bool
qkv (enum vulkan_kernel k)
{
	return k == VULKAN_KERNEL_GEMM_VQKV_NORM || k == VULKAN_KERNEL_GEMM_VQKV_NORMS;
}

/** @brief Whether a tile counter can order one step before the next, by their kernels (upstream:
 *         tchain_pair).
 *
 * The C=512 levels' attention, projections and FFNs; the ViT's expansion, contraction and QKV, and
 * its attention, projection and next expansion, but not its QKV and attention; the encoder's
 * downsampling runs down to the first C=512 FFN; and the decoder from the last C=512 projection up
 * through its upsampling runs to the fused C=32 upsample.
 *
 * @param p The first step's kernel.
 * @param q The next step's.
 * @return  true if a counter can order them.
 */
static bool
counted_pair (enum vulkan_kernel p,
              enum vulkan_kernel q)
{
	return (p == VULKAN_KERNEL_ATTN && c512(q)) || (c512(p) && ffn(q)) || (ffn(p) && c512(q)) ||
	       (c512(p) && q == VULKAN_KERNEL_ATTN) || (p == VULKAN_KERNEL_GEMM_VACT && projection(q)) ||
	       (projection(p) && qkv(q)) || (p == VULKAN_KERNEL_VIT_ATTN && projection(q)) ||
	       (projection(p) && q == VULKAN_KERNEL_GEMM_VACT) || (c512(p) && upsampling_run(q)) ||
	       (upsampling_run(p) && (upsampling_run(q) || q == VULKAN_KERNEL_FSWIN_FUSED_UP32)) ||
	       (p == VULKAN_KERNEL_FSWIN_DSP32 && q == VULKAN_KERNEL_FSWIN_PDS64) ||
	       (p == VULKAN_KERNEL_FSWIN_PDS64 && q == VULKAN_KERNEL_FSWIN_PDS128) ||
	       (p == VULKAN_KERNEL_FSWIN_PDS128 && q == VULKAN_KERNEL_FSWIN_PDS256) ||
	       (p == VULKAN_KERNEL_FSWIN_PDS256 && ffn(q));
}

/** @brief Whether a kernel's push block ends with the index of its tile-counter record (upstream:
 *         tchain_kern). */
static bool
counted (enum vulkan_kernel k)
{
	return k == VULKAN_KERNEL_ATTN || k == VULKAN_KERNEL_GEMM_PROJC || k == VULKAN_KERNEL_FFWD3 ||
	       k == VULKAN_KERNEL_FFWD3W || k == VULKAN_KERNEL_GEMM_VACT || k == VULKAN_KERNEL_GEMM_PROJ ||
	       k == VULKAN_KERNEL_GEMM_PROJW || k == VULKAN_KERNEL_GEMM_PROJT ||
	       k == VULKAN_KERNEL_GEMM_VQKV_NORM || k == VULKAN_KERNEL_GEMM_VQKV_NORMS ||
	       k == VULKAN_KERNEL_VIT_ATTN || upsampling_run(k) || k == VULKAN_KERNEL_FSWIN_FUSED_UP32 ||
	       downsampling_run(k) || k == VULKAN_KERNEL_FSWIN_DSP32 || k == VULKAN_KERNEL_FSWIN32;
}

/** @brief The pairs of the persistent runs, the fused C=32 downsample and upsample that tile
 *         counters order on big frames all the same (upstream: NR_TC_BIG_ALLOW's default).
 *
 * The decoder's C=128 -> C=64 run and C=64 run -> fused C=32 upsample, and the encoder's C=128 ->
 * C=256 run and C=256 run -> first C=512 FFN.
 */
static struct {
	enum vulkan_kernel p;
	enum vulkan_kernel q;
} const BIG_FRAME_PAIRS[] = {
	{VULKAN_KERNEL_FSWIN_PUP128, VULKAN_KERNEL_FSWIN_PUP64},
	{VULKAN_KERNEL_FSWIN_PUP64, VULKAN_KERNEL_FSWIN_FUSED_UP32},
	{VULKAN_KERNEL_FSWIN_PDS128, VULKAN_KERNEL_FSWIN_PDS256},
	{VULKAN_KERNEL_FSWIN_PDS256, VULKAN_KERNEL_FFWD3W},
};

/** @brief The words of a push_f_swin. */
#define SWIN_WORDS (sizeof (struct push_f_swin) / 4)

/** @brief A dispatch's Swin body block. */
static struct push_f_swin
swin_of (struct vulkan_dispatch const *d)
{
	struct push_f_swin f;
	memcpy(&f, d->push, sizeof f);
	return f;
}

/** @brief A dispatch's persistent run block. */
static struct push_persist
persist_of (struct vulkan_dispatch const *d)
{
	struct push_persist p;
	memcpy(&p, d->push, sizeof p);
	return p;
}

/** @brief Whether two Swin blocks share their tile raster. */
static bool
same_tiles (struct push_f_swin const *a,
            struct push_f_swin const *b)
{
	return a->tiles_x == b->tiles_x && a->tiles_y == b->tiles_y;
}

/* Words of a persist_rec in the blob: its layer's tile raster and window grid. */
static_assert(offsetof(struct persist_rec, p) == 0);
#define TILES_X (offsetof(struct push_f_swin, tiles_x) / 4)
#define TILES_Y (offsetof(struct push_f_swin, tiles_y) / 4)
#define GRID_X  (offsetof(struct persist_rec, gx) / 4)
#define GRID_Y  (offsetof(struct persist_rec, gy) / 4)

/** @brief The words of a persist_rec. */
#define REC_WORDS (sizeof (struct persist_rec) / 4)

/** @brief The most layers of a persistent run. */
#define RUN_LAYERS 16

/** @brief The items of a persistent run's layer that a window of the next layer waits for, in
 *         order, once each.
 *
 * Tile (tx, ty) of a layer comes from window floor((tx - shift) / 2), floor((ty - shift_y) / 2) of
 * the layer before.
 *
 * @param r         The layer.
 * @param q         The layer before.
 * @param w         The window of the layer.
 * @param producers Receives the items.
 * @return          Their number.
 */
static size_t
producers_of (struct persist_rec const *r,
              struct persist_rec const *q,
              uint32_t                  w,
              uint32_t                  producers[4])
{
	int const wx = (int)(w % r->gx);
	int const wy = (int)(w / r->gx);
	size_t n = 0;
	for (int t = 0; t < 4; ++t) {
		int const tx = 2 * wx + r->p.shift + (t & 1);
		int const ty = 2 * wy + r->p.shift_y + (t >> 1);
		if (tx < 0 || ty < 0 || tx >= (int)r->p.tiles_x || ty >= (int)r->p.tiles_y)
			continue;
		int const ax = tx - q->p.shift;
		int const ay = ty - q->p.shift_y;
		int const px = ax >= 0 ? ax / 2 : -((1 - ax) / 2);
		int const py = ay >= 0 ? ay / 2 : -((1 - ay) / 2);
		if (px < 0 || py < 0 || px >= (int)q->gx || py >= (int)q->gy)
			continue;
		uint32_t const producer = q->flag_base + (uint32_t)py * q->gx + (uint32_t)px;
		bool seen = false;
		for (size_t i = 0; i < n && !seen; ++i)
			seen = producers[i] == producer;
		if (seen)
			continue;
		size_t at = n++;
		for (; at && producers[at - 1] > producer; --at)
			producers[at] = producers[at - 1];
		producers[at] = producer;
	}
	return n;
}

/** @brief Adds a consumer to a producer's four slots.
 *
 * @param slots The producer's slots, UINT32_MAX where empty.
 * @param item  The consumer.
 * @return      Whether a slot was empty.
 */
static bool
add_consumer (uint32_t slots[4],
              uint32_t item)
{
	for (size_t slot = 0; slot < 4; ++slot)
		if (slots[slot] == UINT32_MAX) {
			slots[slot] = item;
			return true;
		}
	return false;
}

/** @brief A persistent run's tables of dependencies between its items, a window of one of its
 *         layers each (upstream: the NR_PERSIST_DF block, nr_graph.cpp:3538-3602).
 *
 * Each item's producers in the layer before, the up to four consumers each has, and the items no
 * producer holds up; then, for the straggler queue, the layer of each item, and each layer's items
 * that wait with the rank from which the oldest workgroups take them, a fraction straggle of them.
 *
 * @param rec        The run's layers.
 * @param count      Their number, at most RUN_LAYERS.
 * @param windows    Their windows.
 * @param straggle   The fraction of the waiting items that the oldest workgroups take.
 * @param free_items Receives the items no producer holds up.
 * @param dest       Receives the tables, which the caller frees.
 * @param size       Receives their words.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
dependencies (struct persist_rec const *rec,
              size_t                    count,
              uint32_t                  windows,
              double                    straggle,
              uint32_t                 *free_items,
              uint32_t                **dest,
              size_t                   *size,
              struct error             *e)
{
	// need, cons, init padded to 6 windows, each item's layer, 16 layers' counts and ranks, and
	// the totals.
	size_t const words = 7 * (size_t)windows + 2 * RUN_LAYERS + 2;
	uint32_t *table = malloc(words * sizeof *table);
	if (!table)
		return error_fail(e, "out of memory");
	uint32_t *const need = table;
	uint32_t *const cons = table + windows;
	memset(need, 0, windows * sizeof *need);
	memset(cons, 0xff, 4 * (size_t)windows * sizeof *cons);
	for (size_t k = 1; k < count; ++k) {
		struct persist_rec const *const r = &rec[k];
		for (uint32_t w = 0; w < r->windows; ++w) {
			uint32_t producers[4];
			size_t const n = producers_of(r, &rec[k - 1], w, producers);
			uint32_t const item = r->flag_base + w;
			need[item] = (uint32_t)n;
			bool added = true;
			for (size_t i = 0; i < n && added; ++i)
				added = add_consumer(&cons[(size_t)producers[i] * 4], item);
			if (!added) {
				free(table);
				table = nullptr;
				return error_fail(e, "network plan: a persistent run's item has more than four "
				                     "consumers");
			}
		}
	}
	size_t at = 5 * (size_t)windows;
	for (uint32_t item = 0; item < windows; ++item)
		if (!need[item])
			table[at++] = item;
	uint32_t const init = (uint32_t)(at - 5 * (size_t)windows);
	memset(table + at, 0, (6 * (size_t)windows - at) * sizeof *table);
	at = 6 * (size_t)windows;
	uint32_t queued[RUN_LAYERS] = {0};
	for (size_t k = 0; k < count; ++k)
		for (uint32_t w = 0; w < rec[k].windows; ++w) {
			table[at++] = (uint32_t)k;
			queued[k] += need[rec[k].flag_base + w] != 0;
		}
	uint32_t first = init;
	uint32_t late = 0;
	for (size_t k = 0; k < RUN_LAYERS; ++k) {
		uint32_t const q = queued[k];
		uint32_t const threshold = (uint32_t)ceil(straggle * (double)q);
		uint32_t const taken = q < threshold ? q : threshold;
		table[at++] = q;
		table[at++] = threshold;
		first += taken;
		late += q - taken;
	}
	table[at++] = first;
	table[at++] = late;
	*free_items = init;
	*dest = table;
	*size = at;
	return ERROR_NONE;
}

/** @brief The word that a dispatch reads from its predecessor (upstream: io).
 *
 * A persistent run's input is its first layer's, or its upsampling table's; the fused C=32
 * upsample's input is its blend's.
 *
 * @param d    The dispatch.
 * @param blob The blob.
 * @param word Receives the word.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
input (struct vulkan_dispatch const *d,
       struct vulkan_blob const     *blob,
       uint32_t                     *word,
       struct error                 *e)
{
	if (gemm(d->kernel)) {
		struct push_gemm g;
		memcpy(&g, d->push, sizeof g);
		*word = g.x_off;
		return ERROR_NONE;
	}
	if (vulkan_plan_persistent(d->kernel)) {
		struct push_persist const p = persist_of(d);
		uint32_t const at = upsampling_run(d->kernel) && p.ds_off ? p.ds_off : p.layers_off;
		return vulkan_blob_word(blob, at, word, e);
	}
	if (d->kernel == VULKAN_KERNEL_FSWIN_FUSED_UP32) {
		struct push_ups u;
		memcpy(&u, d->push + SWIN_WORDS, sizeof u);
		*word = u.p_off;
		return ERROR_NONE;
	}
	*word = d->push[0];
	return ERROR_NONE;
}

/** @brief The word that a dispatch writes for its successor (upstream: io).
 *
 * A persistent run's output is its last layer's, or its downsampling table's.
 *
 * @param d    The dispatch.
 * @param blob The blob.
 * @param word Receives the word.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
output (struct vulkan_dispatch const *d,
        struct vulkan_blob const     *blob,
        uint32_t                     *word,
        struct error                 *e)
{
	if (gemm(d->kernel)) {
		struct push_gemm g;
		memcpy(&g, d->push, sizeof g);
		*word = g.o_off;
		return ERROR_NONE;
	}
	if (vulkan_plan_persistent(d->kernel)) {
		struct push_persist const p = persist_of(d);
		uint64_t const at = downsampling_run(d->kernel) && p.ds_off
		                  ? p.ds_off + 1
		                  : p.layers_off + (p.n_layers - 1) * REC_WORDS + 1;
		return vulkan_blob_word(blob, at, word, e);
	}
	if (d->kernel == VULKAN_KERNEL_FSWIN_DSP32) {
		struct push_ds_proj q;
		memcpy(&q, d->push + SWIN_WORDS, sizeof q);
		*word = q.o_off;
		return ERROR_NONE;
	}
	if (d->kernel == VULKAN_KERNEL_FSWIN_FUSED_UP32) {
		*word = 0xfffffffeu;
		return ERROR_NONE;
	}
	*word = d->push[1];
	return ERROR_NONE;
}

/** @brief The tile raster in which an upsampling consumer gathers its lower-level input (upstream:
 *         ups_grid).
 *
 * @param d    The dispatch.
 * @param blob The blob.
 * @param has  Receives whether it has one.
 * @param x    Receives its width in tiles.
 * @param y    Receives its height in tiles.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
gather_raster (struct vulkan_dispatch const *d,
               struct vulkan_blob const     *blob,
               bool                         *has,
               uint32_t                     *x,
               uint32_t                     *y,
               struct error                 *e)
{
	if (vulkan_plan_persistent(d->kernel)) {
		struct push_persist const p = persist_of(d);
		if (!p.ds_off) {
			*has = false;
			return ERROR_NONE;
		}
		uint64_t const x_at = p.ds_off + offsetof(struct push_ups, itiles_x) / 4;
		uint64_t const y_at = p.ds_off + offsetof(struct push_ups, itiles_y) / 4;
		enum error_code code = vulkan_blob_word(blob, x_at, x, e);
		if (code)
			return code;
		code = vulkan_blob_word(blob, y_at, y, e);
		if (code)
			return code;
	} else {
		struct push_ups u;
		memcpy(&u, d->push + SWIN_WORDS, sizeof u);
		*x = u.itiles_x;
		*y = u.itiles_y;
	}
	*has = *x && *y;
	return ERROR_NONE;
}

/** @brief A word of a persistent run's last layer's record, or of its first.
 *
 * @param d     The run's dispatch.
 * @param blob  The blob.
 * @param at    The word in the record.
 * @param first Whether the first layer's.
 * @param word  Receives the word.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
layer_word (struct vulkan_dispatch const *d,
            struct vulkan_blob const     *blob,
            size_t                        at,
            bool                          first,
            uint32_t                     *word,
            struct error                 *e)
{
	struct push_persist const p = persist_of(d);
	return vulkan_blob_word(blob, p.layers_off + (first ? 0 : (p.n_layers - 1) * REC_WORDS) + at, word, e);
}

/** @brief A free block of the arena, and the last dispatch of what it held. */
struct block {
	uint64_t offset; //!< Where it starts.
	uint64_t size;   //!< Its bytes.
	size_t   freed;  //!< The last dispatch of what it held.
};

/** @brief A placed value's bytes, and its last dispatch. */
struct live {
	uint64_t offset; //!< Where it starts.
	uint64_t size;   //!< Its bytes.
	size_t   last;   //!< Its last dispatch.
};

/** @brief The arena's free blocks in offset order, merged where they touch. */
struct free_list {
	struct block block[VULKAN_PLAN_KEYS]; //!< The blocks.
	size_t       count;                   //!< Their number.
};

/** @brief Releases a value's bytes into the free list, merged with the blocks next to them; a free
 *         block that ends at the top lowers the top instead.
 *
 * @param l    The value.
 * @param list The free list.
 * @param top  The arena's top.
 */
static void
release (struct live const *l,
         struct free_list  *list,
         uint64_t          *top)
{
	// In offset order: the blocks never overlap, so their offsets are distinct.
	size_t at = list->count++;
	for (; at && list->block[at - 1].offset > l->offset; --at)
		list->block[at] = list->block[at - 1];
	list->block[at] = (struct block){l->offset, l->size, l->last};
	size_t merged = 0;
	for (size_t i = 0; i < list->count; ++i) {
		struct block const b = list->block[i];
		if (merged && list->block[merged - 1].offset + list->block[merged - 1].size == b.offset) {
			struct block *const m = &list->block[merged - 1];
			m->size += b.size;
			if (m->freed < b.freed)
				m->freed = b.freed;
		} else {
			list->block[merged++] = b;
		}
	}
	list->count = merged;
	if (list->count && list->block[list->count - 1].offset + list->block[list->count - 1].size == *top)
		*top = list->block[--list->count].offset;
}

/** @brief Sets a table's entry for a tile: a counter, and the units of it that make the tile whole.
 *
 * @param table   The table.
 * @param size    Its entries.
 * @param tile    The tile.
 * @param counter The counter.
 * @param need    The units.
 * @return        Whether both fit it.
 */
static bool
set_entry (uint32_t *table,
           size_t    size,
           uint64_t  tile,
           uint32_t  counter,
           uint32_t  need)
{
	if (tile >= size || counter >= UINT32_C(1) << 20 || need >= UINT32_C(1) << 12)
		return false;
	table[tile] = counter | need << 20;
	return true;
}

enum error_code
vulkan_schedule_merge (struct vulkan_dispatches       *dest,
                       struct vulkan_dispatches const *lowered,
                       struct vulkan_blob             *blob,
                       uint64_t                       *arena,
                       bool                           *epoch,
                       struct error                   *e)
{
	static constexpr uint32_t FOLD_WORDS = (sizeof (struct push_f_swin) + sizeof (struct push_ds_proj)) / 4;
	// A fused upsample's words too.
	static_assert(sizeof (struct push_f_swin) + sizeof (struct push_ups) + 4 == 4 * FOLD_WORDS);
	struct vulkan_dispatch const *const d = lowered->dispatch;
	size_t const count = lowered->count;
	struct vulkan_dispatches out = {malloc(count * sizeof *out.dispatch), 0, count};
	if (!out.dispatch)
		return error_fail(e, "out of memory");
	*epoch = false;
	for (size_t i = 0; i < count;) {
		// A maximal run of plain bodies of one width at one tile raster, each reading what the
		// one before wrote.
		unsigned const c = d[i].body == VULKAN_BODY_SWIN && d[i].words == SWIN_WORDS ? d[i].c : 0;
		if (!c) {
			out.dispatch[out.count++] = d[i++];
			continue;
		}
		struct push_f_swin const head = swin_of(&d[i]);
		size_t j = i;
		while (j + 1 < count && d[j + 1].body == VULKAN_BODY_SWIN && d[j + 1].c == c &&
		       d[j + 1].words == SWIN_WORDS) {
			struct push_f_swin const next = swin_of(&d[j + 1]);
			if (!same_tiles(&next, &head) || next.x_off != swin_of(&d[j]).o_off)
				break;
			++j;
		}
		if (j == i) {
			out.dispatch[out.count++] = d[i++];
			continue;
		}
		// The downsampling body after the run joins it as its last layer, or else the upsampling
		// one before it as its first.
		bool ds_fold = false;
		if (j + 1 < count && d[j + 1].body == VULKAN_BODY_SWIN_DS && d[j + 1].c == c &&
		    d[j + 1].words == FOLD_WORDS) {
			struct push_f_swin const next = swin_of(&d[j + 1]);
			ds_fold = same_tiles(&next, &head) && next.x_off == swin_of(&d[j]).o_off;
		}
		if (ds_fold)
			++j;
		bool up_fold = false;
		if (!ds_fold && out.count) {
			struct vulkan_dispatch const *const back = &out.dispatch[out.count - 1];
			if (back->body == VULKAN_BODY_SWIN_UP && back->c == c && back->words == FOLD_WORDS) {
				struct push_f_swin const before = swin_of(back);
				up_fold = same_tiles(&before, &head) && before.o_off == head.x_off;
			}
		}
		if (!ds_fold && !up_fold) {
			free(out.dispatch);
			out.dispatch = nullptr;
			return error_fail(e, "network plan: a persistent run neither downsamples nor "
			                     "upsamples");
		}
		struct vulkan_dispatch const up_body = up_fold ? out.dispatch[out.count - 1]
		                                               : (struct vulkan_dispatch){0};
		if (up_fold)
			--out.count;
		size_t const layers = up_fold + (j - i + 1);
		if (layers > RUN_LAYERS) {
			free(out.dispatch);
			out.dispatch = nullptr;
			return error_fail(e, "network plan: a persistent run has more than 16 layers");
		}
		struct vulkan_dispatch const *layer[RUN_LAYERS];
		size_t n = 0;
		if (up_fold)
			layer[n++] = &up_body;
		for (size_t t = i; t <= j; ++t)
			layer[n++] = &d[t];
		struct persist_rec rec[RUN_LAYERS];
		uint32_t windows = 0;
		uint32_t most = 0;
		for (size_t k = 0; k < n; ++k) {
			rec[k] = (struct persist_rec){
				swin_of(layer[k]), layer[k]->groups[0], layer[k]->groups[1],
				layer[k]->groups[0] * layer[k]->groups[1], windows,
			};
			windows += rec[k].windows;
			if (most < rec[k].windows)
				most = rec[k].windows;
		}
		// Workgroups by the width's occupancy. A downsampling or upsampling run of C=128 or 256
		// of 65 to VULKAN_PLAN_PERSIST_ONE_MAX windows a layer takes one an item, which leaves its
		// straggler queue nothing.
		uint32_t const bit = c == 64 ? 1 : c == 128 ? 2 : 4;
		uint32_t const cap = c == 64 ? 512 : c == 128 ? 256 : 128;
		bool const one = (VULKAN_PLAN_PERSIST_ONE_MASK & bit) && most > 64 &&
		                 most <= VULKAN_PLAN_PERSIST_ONE_MAX;
		uint32_t const groups = one ? windows : cap < most ? cap : most;
		struct push_persist p = {0};
		uint32_t words[RUN_LAYERS * REC_WORDS];
		memcpy(words, rec, n * sizeof *rec);
		p.layers_off = (uint32_t)(vulkan_blob_put_words(blob, words, n * REC_WORDS) / 4);
		struct vulkan_dispatch const *const fold = ds_fold ? &d[j] : &up_body;
		uint32_t const *const folded = fold->push + SWIN_WORDS;
		p.ds_off = (uint32_t)(vulkan_blob_put_words(blob, folded, FOLD_WORDS - SWIN_WORDS) / 4);
		uint32_t *table = nullptr;
		size_t table_size = 0;
		double const straggle = one ? 1.0 : VULKAN_PLAN_STRAGGLER_PERCENT / 100.0;
		enum error_code const code = dependencies(rec, n, windows, straggle, &p.df_n0, &table,
		                                          &table_size, e);
		if (code) {
			free(out.dispatch);
			out.dispatch = nullptr;
			return code;
		}
		p.df_off = (uint32_t)(vulkan_blob_put_words(blob, table, table_size) / 4);
		free(table);
		table = nullptr;
		// The sync region, in the arena: counters, flags, broadcast slots, the ready queue and the
		// straggler queues.
		*arena = align(*arena, 256);
		p.sync_off = (uint32_t)(*arena / 4);
		*epoch = *epoch || c == 256;
		*arena += align((uint64_t)(4 + windows + groups + windows + 1 + 18 + windows + 8) * 4, 256);
		p.n_layers = (uint32_t)n;
		p.spin_limit = VULKAN_PLAN_SPIN_LIMIT;
		p.total_windows = windows;
		int const width = c == 64 ? 0 : c == 128 ? 1 : 2;
		enum vulkan_kernel const narrowest = ds_fold ? VULKAN_KERNEL_FSWIN_PDS64
		                                             : VULKAN_KERNEL_FSWIN_PUP64;
		struct vulkan_dispatch m = {
			.groups = {groups, 1, 1},
			.kernel = (enum vulkan_kernel)(narrowest + width),
			.after  = VULKAN_AFTER_INVALIDATE,
			.block  = d[i].block,
			.layer  = d[i].layer,
			.first  = layer[0]->first,
			.last   = layer[0]->last,
		};
		for (size_t k = 0; k < n; ++k) {
			if (layer[k]->first < m.first)
				m.first = layer[k]->first;
			if (m.last < layer[k]->last)
				m.last = layer[k]->last;
		}
		memcpy(m.push, &p, sizeof p);
		m.words = sizeof p / 4;
		out.dispatch[out.count++] = m;
		i = j + 1;
	}
	for (size_t i = 0; i < out.count; ++i)
		if (out.dispatch[i].kernel == VULKAN_KERNEL_COUNT) {
			free(out.dispatch);
			out.dispatch = nullptr;
			return error_fail(e, "network plan: a Swin layer at C>=64 is outside a persistent run");
		}
	if (blob->error) {
		free(out.dispatch);
		out.dispatch = nullptr;
		return error_fail(e, "out of memory");
	}
	*dest = out;
	return ERROR_NONE;
}

/** @brief A span of dispatches. */
struct span {
	size_t first; //!< The first dispatch.
	size_t last;  //!< The last.
};

/** @brief Whether one key goes before another in the arena's order: by first dispatch, the larger
 *         value first, then by key.
 *
 * @param span The keys' spans.
 * @param v    The values.
 * @param a    A key.
 * @param b    Another key.
 * @return     true if a goes before b.
 */
static bool
before (struct span const          *span,
        struct vulkan_values const *v,
        size_t                      a,
        size_t                      b)
{
	if (span[a].first != span[b].first)
		return span[a].first < span[b].first;
	if (v->size[a] != v->size[b])
		return v->size[a] > v->size[b];
	return a < b;
}

/** @brief The live values and the free blocks of the arena as values are placed. */
struct arena_lists {
	struct free_list free;                   //!< The free blocks.
	struct live      live[VULKAN_PLAN_KEYS]; //!< The placed values, in the order placed.
	size_t           live_count;             //!< Their number.
};

enum error_code
vulkan_schedule_share (struct vulkan_dispatches const *dispatches,
                       bool                            chains,
                       struct vulkan_values           *v,
                       uint64_t                       *end,
                       struct error                   *e)
{
	struct vulkan_dispatch const *const d = dispatches->dispatch;
	size_t const count = dispatches->count;
	// Each value's dispatches: every one whose layers meet the layers that name it, or the whole
	// frame when none does, which leaves first at count.
	struct span span[VULKAN_PLAN_KEYS] = {0};
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k) {
		if (v->first[k] < 0)
			continue;
		span[k] = (struct span){count, 0};
		for (size_t f = 0; f < count; ++f)
			if (d[f].first <= v->last[k] && d[f].last >= v->first[k]) {
				if (f < span[k].first)
					span[k].first = f;
				if (span[k].last < f)
					span[k].last = f;
			}
		if (span[k].first == count)
			span[k] = (struct span){0, count};
	}
	// A key with no size aliases the value that its plain offset lies in, which lives for it too
	// (skip(30) and block 0's output). VULKAN_PLAN_KEYS is no key.
	size_t owner[VULKAN_PLAN_KEYS];
	uint64_t delta[VULKAN_PLAN_KEYS] = {0};
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
		owner[k] = VULKAN_PLAN_KEYS;
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k) {
		if (v->first[k] < 0 || v->size[k])
			continue;
		for (size_t o = 0; o < VULKAN_PLAN_KEYS; ++o)
			if (v->size[o] && v->offset[o] <= v->offset[k] &&
			    v->offset[k] < v->offset[o] + v->size[o])
				owner[k] = o;
		if (owner[k] == VULKAN_PLAN_KEYS)
			return error_fail(e, "network plan: a value without a size aliases no value");
		delta[k] = v->offset[k] - v->offset[owner[k]];
		struct span *const o = &span[owner[k]];
		if (span[k].first < o->first)
			o->first = span[k].first;
		if (o->last < span[k].last)
			o->last = span[k].last;
	}
	// Values read past their size live all frame, so the bytes past it stay zero.
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
		if (v->overread[k] && v->first[k] >= 0)
			span[k] = (struct span){0, count};
	// The keys in their order, sorted by insertion.
	size_t order[VULKAN_PLAN_KEYS];
	size_t ordered = 0;
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k) {
		if (v->first[k] < 0 || !v->size[k])
			continue;
		size_t at = ordered++;
		for (; at && before(span, v, k, order[at - 1]); --at)
			order[at] = order[at - 1];
		order[at] = k;
	}
	// Steps that tile counters order overlap: a value read in such a stretch stays until its end.
	if (chains) {
		size_t *stretch_end = malloc((count + 1) * sizeof *stretch_end);
		if (!stretch_end)
			return error_fail(e, "out of memory");
		for (size_t f = count + 1; f--;) {
			bool const counted_next = f + 1 < count && counted_pair(d[f].kernel, d[f + 1].kernel);
			stretch_end[f] = counted_next ? stretch_end[f + 1] : f;
		}
		for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
			if (v->first[k] >= 0 && span[k].last < count)
				span[k].last = stretch_end[span[k].last];
		free(stretch_end);
		stretch_end = nullptr;
	}
	// First fit by first use, the smallest block that holds a value; a block is free once its last
	// dispatch is behind a barrier. A value smaller than VULKAN_PLAN_ARENA_COLD_MAX takes only a
	// block freed at least VULKAN_PLAN_ARENA_COLD dispatches before its first.
	struct arena_lists *lists = malloc(sizeof *lists);
	if (!lists)
		return error_fail(e, "out of memory");
	lists->free.count = 0;
	lists->live_count = 0;
	struct free_list *const list = &lists->free;
	uint64_t top = 0;
	uint64_t high = 0;
	uint64_t offset[VULKAN_PLAN_KEYS] = {0};
	for (size_t i = 0; i < ordered; ++i) {
		size_t const k = order[i];
		size_t const first = span[k].first;
		size_t kept = 0;
		for (size_t l = 0; l < lists->live_count; ++l)
			if (lists->live[l].last < first)
				release(&lists->live[l], list, &top);
			else
				lists->live[kept++] = lists->live[l];
		lists->live_count = kept;
		uint64_t const need = align(v->size[k] < v->overread[k] ? v->overread[k] : v->size[k], 256);
		bool const cold = need < VULKAN_PLAN_ARENA_COLD_MAX;
		size_t best = list->count;
		for (size_t b = 0; b < list->count; ++b)
			if (list->block[b].size >= need &&
			    !(cold && list->block[b].freed + VULKAN_PLAN_ARENA_COLD >= first) &&
			    (best == list->count || list->block[b].size < list->block[best].size))
				best = b;
		if (best < list->count) {
			struct block *const b = &list->block[best];
			offset[k] = b->offset;
			if (b->size > need) {
				*b = (struct block){b->offset + need, b->size - need, b->freed};
			} else {
				memmove(b, b + 1, (list->count - best - 1) * sizeof *b);
				--list->count;
			}
		} else {
			offset[k] = top;
			top += need;
		}
		lists->live[lists->live_count++] = (struct live){offset[k], need, span[k].last};
		if (high < top)
			high = top;
	}
	free(lists);
	lists = nullptr;
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
		if (owner[k] < VULKAN_PLAN_KEYS)
			offset[k] = offset[owner[k]] + delta[k];
	memcpy(v->offset, offset, sizeof offset);
	*end = align(high, 256);
	return ERROR_NONE;
}

void
vulkan_schedule_trim (struct vulkan_dispatches *dispatches,
                      uint32_t                  height)
{
	struct vulkan_dispatch *const d = dispatches->dispatch;
	// The post block's window rows that start inside the picture: 8 wy + 4 shift_y < height.
	struct vulkan_dispatch *const post = &d[dispatches->count - 1];
	struct push_f_swin const f = swin_of(post);
	int const post_rows = ((int)height - 4 * f.shift_y + 7) / 8;
	uint32_t const post_groups = (uint32_t)(post_rows < 1 ? 1 : post_rows);
	if (post_groups < post->groups[1])
		post->groups[1] = post_groups;
	// The rows of its input, at half its resolution, that those read; then, back through the
	// plain C=32 layers that write that input, each read by the next alone, the window rows that
	// start inside the rows read.
	int rows = (8 * (int)post->groups[1] + 4 * f.shift_y + 1) / 2;
	struct push_ups u;
	memcpy(&u, post->push + SWIN_WORDS, sizeof u);
	uint32_t input = u.p_off;
	for (size_t i = dispatches->count - 1; i-- && d[i].kernel == VULKAN_KERNEL_FSWIN32;) {
		struct push_f_swin const g = swin_of(&d[i]);
		int const read = (rows - 4 * g.shift_y + 7) / 8;
		uint32_t const wanted = (uint32_t)(read < 1 ? 1 : read);
		uint32_t const groups = wanted < d[i].groups[1] ? wanted : d[i].groups[1];
		if (g.o_off != input || groups == d[i].groups[1])
			break;
		d[i].groups[1] = groups;
		rows = 8 * (int)groups + 4 * g.shift_y;
		input = g.x_off;
	}
}

/** @brief The words of a tile-counter record. */
#define RECORD_WORDS 7

/** @brief What a tile-counter chain holds on the heap until it ends. */
struct chain {
	uint32_t (*rec)[RECORD_WORDS]; //!< Each dispatch's record.
	uint32_t  *table;              //!< A pair's table.
	uint32_t  *need;               //!< A pair's units a tile.
	size_t     capacity;           //!< The entries that the table and the units hold.
};

/** @brief Makes room in a chain's table and units for a number of tiles.
 *
 * @param c     The chain.
 * @param tiles The tiles.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
reserve (struct chain *c,
         size_t        tiles,
         struct error *e)
{
	if (tiles <= c->capacity)
		return ERROR_NONE;
	uint32_t *const table = realloc(c->table, tiles * sizeof *table);
	if (!table)
		return error_fail(e, "out of memory");
	c->table = table;
	uint32_t *const need = realloc(c->need, tiles * sizeof *need);
	if (!need)
		return error_fail(e, "out of memory");
	c->need = need;
	c->capacity = tiles;
	return ERROR_NONE;
}

/** @brief Fills a table for an attention producer: a window of four tiles a workgroup.
 *
 * @param p        The producer.
 * @param table    The table.
 * @param tiles    Its entries.
 * @param counters Receives the producer's counters.
 * @return         Whether every entry fits the table.
 */
static bool
attention_table (struct vulkan_dispatch const *p,
                 uint32_t                     *table,
                 uint32_t                      tiles,
                 uint32_t                     *counters)
{
	struct push_attn a;
	memcpy(&a, p->push, sizeof a);
	*counters = p->groups[0] * p->groups[1];
	bool fits = true;
	for (uint32_t wy = 0; wy < p->groups[1]; ++wy)
		for (uint32_t wx = 0; wx < p->groups[0]; ++wx)
			for (int t = 0; t < 4; ++t) {
				int const tx = 2 * (int)wx + a.shift + (t & 1);
				int const ty = 2 * (int)wy + a.shift_y + (t >> 1);
				if (tx < 0 || ty < 0 || tx >= (int)a.tiles_x || ty >= (int)a.tiles_y)
					continue;
				uint64_t const tile = (uint64_t)ty * a.tiles_x + (uint64_t)tx;
				fits &= set_entry(table, tiles, tile, wy * p->groups[0] + wx, p->groups[2]);
			}
	return fits;
}

/** @brief Fills a table for a GEMM producer: every M tile, signalled by each wave of each N tile.
 *
 * @param p     The producer.
 * @param table The table.
 * @param tiles Its entries.
 * @return      Whether every entry fits the table.
 */
static bool
gemm_table (struct vulkan_dispatch const *p,
            uint32_t                     *table,
            uint32_t                      tiles)
{
	bool const wide = p->kernel == VULKAN_KERNEL_GEMM_VACT || p->kernel == VULKAN_KERNEL_GEMM_VQKV_NORM;
	bool const projw = p->kernel == VULKAN_KERNEL_GEMM_PROJW;
	uint32_t const mt = wide    ? VULKAN_PLAN_GEMM_WIDE_MT
	                  : projw   ? VULKAN_PLAN_GEMM_PROJW_MT
	                            : VULKAN_PLAN_GEMM_PROJ_MT;
	uint32_t const waves = wide    ? VULKAN_PLAN_GEMM_WIDE_NT / 64
	                     : projw   ? VULKAN_PLAN_GEMM_PROJW_NT / 64
	                               : VULKAN_PLAN_GEMM_PROJ_NT / 32;
	bool fits = true;
	for (uint32_t k = 0; k < p->groups[0] * (mt / 16); ++k)
		fits &= set_entry(table, tiles, k, k, p->groups[1] * waves);
	return fits;
}

/** @brief Fills a table for the ViT's attention: a query tile, by each head.
 *
 * @param p     The producer.
 * @param table The table.
 * @param tiles Its entries.
 * @return      Whether every entry fits the table.
 */
static bool
vit_attention_table (struct vulkan_dispatch const *p,
                     uint32_t                     *table,
                     uint32_t                      tiles)
{
	struct push_v_attn a;
	memcpy(&a, p->push, sizeof a);
	bool fits = true;
	for (uint32_t k = 0; k < (a.tokens + 15) / 16; ++k)
		fits &= set_entry(table, tiles, k, k, p->groups[1]);
	return fits;
}

/** @brief Fills a table for the FFN: a wave a weight group and tile.
 *
 * @param p     The producer.
 * @param table The table.
 * @param tiles Its entries.
 * @return      Whether every entry fits the table.
 */
static bool
ffn_table (struct vulkan_dispatch const *p,
           uint32_t                     *table,
           uint32_t                      tiles)
{
	struct push_ffwd3 f;
	memcpy(&f, p->push, sizeof f);
	bool fits = true;
	for (uint32_t k = 0; k < (f.M + 15) / 16; ++k)
		fits &= set_entry(table, tiles, k, k, 8);
	return fits;
}

/** @brief Fills a table for a downsampling producer: its windows, onto the next level's tiles, when
 *         it writes them in the consumer's own raster.
 *
 * @param p     The producer: the fused C=32 downsample or a downsampling run.
 * @param q     The consumer.
 * @param blob  The blob.
 * @param c     The chain, whose table has tiles entries.
 * @param tiles The table's entries.
 * @param skip  Receives whether counters cannot order the pair.
 * @param fits  Receives whether every entry fits the table.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
downsampling_table (struct vulkan_dispatch const *p,
                    struct vulkan_dispatch const *q,
                    struct vulkan_blob const     *blob,
                    struct chain                 *c,
                    uint32_t                      tiles,
                    bool                         *skip,
                    bool                         *fits,
                    struct error                 *e)
{
	struct push_f_swin f;
	struct push_ds_proj ds;
	uint32_t gx = p->groups[0];
	uint32_t gy = p->groups[1];
	*skip = true;
	if (p->kernel == VULKAN_KERNEL_FSWIN_DSP32) {
		f = swin_of(p);
		memcpy(&ds, p->push + SWIN_WORDS, sizeof ds);
	} else {
		struct push_persist const r = persist_of(p);
		if (!r.ds_off)
			return ERROR_NONE;
		uint32_t w[REC_WORDS];
		uint32_t x[sizeof ds / 4];
		for (size_t k = 0; k < REC_WORDS; ++k) {
			enum error_code const code = layer_word(p, blob, k, false, &w[k], e);
			if (code)
				return code;
		}
		for (size_t k = 0; k < sizeof ds / 4; ++k) {
			enum error_code const code = vulkan_blob_word(blob, r.ds_off + k, &x[k], e);
			if (code)
				return code;
		}
		memcpy(&f, w, sizeof f);
		memcpy(&ds, x, sizeof ds);
		gx = w[GRID_X];
		gy = w[GRID_Y];
	}
	uint32_t const rows = ds.writer_rows ? ds.writer_rows : ds.rows;
	if (ds.mode == 2 || ds.raster != ds.crow || rows != ds.rows)
		return ERROR_NONE;
	if (vulkan_plan_persistent(q->kernel)) {
		uint32_t x = 0;
		enum error_code const code = layer_word(q, blob, TILES_X, true, &x, e);
		if (code)
			return code;
		if (x != ds.otx)
			return ERROR_NONE;
	}
	*skip = false;
	int const pool_x = (int)(f.pool_tiles_x * 4);
	int const raster = (int)ds.raster;
	int const otx = (int)(ds.otx * 4);
	int const nearer = raster < pool_x ? raster : pool_x;
	int const lx = (otx < nearer ? otx : nearer) - 1;
	int const pool_y = (int)(f.pool_tiles_y * 4);
	int const ly = ((int)rows < pool_y ? (int)rows : pool_y) - 1;
	uint32_t *const need = c->need;
	memset(need, 0, tiles * sizeof *need);
	*fits = true;
	for (uint32_t wy = 0; wy < gy; ++wy)
		for (uint32_t wx = 0; wx < gx; ++wx) {
			int const px = (2 * (int)wx + f.shift) * 2;
			int const py = (2 * (int)wy + f.shift_y) * 2;
			int const x0 = px < 0 ? 0 : px;
			int const x1 = lx < px + 3 ? lx : px + 3;
			int const y0 = py < 0 ? 0 : py;
			int const y1 = ly < py + 3 ? ly : py + 3;
			if (x0 > x1 || y0 > y1)
				continue;
			for (int ty = y0 / 4; ty <= y1 / 4; ++ty)
				for (int tx = x0 / 4; tx <= x1 / 4; ++tx) {
					uint64_t const tile = (uint64_t)ty * ds.otx + (uint64_t)tx;
					if (tile < tiles)
						++need[tile];
					else
						*fits = false;
				}
		}
	for (uint32_t k = 0; k < tiles; ++k)
		if (need[k])
			*fits &= set_entry(c->table, tiles, k, k, need[k]);
	return ERROR_NONE;
}

/** @brief Fills a table for an upsampling run's last layer: a window a tile.
 *
 * @param p     The producer.
 * @param blob  The blob.
 * @param up    Whether the consumer gathers its input in a raster of its own.
 * @param itx   The width in tiles of that raster.
 * @param table The table.
 * @param tiles Its entries.
 * @param skip  Receives whether counters cannot order the pair.
 * @param fits  Receives whether every entry fits the table.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
run_table (struct vulkan_dispatch const *p,
           struct vulkan_blob const     *blob,
           bool                          up,
           uint32_t                      itx,
           uint32_t                     *table,
           uint32_t                      tiles,
           bool                         *skip,
           bool                         *fits,
           struct error                 *e)
{
	uint32_t tx = 0;
	uint32_t ty = 0;
	enum error_code code = layer_word(p, blob, TILES_X, false, &tx, e);
	if (code)
		return code;
	code = layer_word(p, blob, TILES_Y, false, &ty, e);
	if (code)
		return code;
	*skip = up && itx != tx;
	*fits = true;
	if (*skip)
		return ERROR_NONE;
	for (uint32_t k = 0; k < tx * ty; ++k)
		*fits &= set_entry(table, tiles, k, k, 1);
	return ERROR_NONE;
}

/** @brief The tile raster that a persistent run consumes in at its first layer or produces in at its
 *         last.
 *
 * @param d     The run's dispatch.
 * @param blob  The blob.
 * @param first Whether the first layer's.
 * @param tiles Receives its tiles.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
run_tiles (struct vulkan_dispatch const *d,
           struct vulkan_blob const     *blob,
           bool                          first,
           uint32_t                     *tiles,
           struct error                 *e)
{
	uint32_t x = 0;
	uint32_t y = 0;
	enum error_code code = layer_word(d, blob, TILES_X, first, &x, e);
	if (code)
		return code;
	code = layer_word(d, blob, TILES_Y, first, &y, e);
	if (code)
		return code;
	*tiles = x * y;
	return ERROR_NONE;
}

/** @brief Whether a kernel's barriers stay on a big frame but between BIG_FRAME_PAIRS: a persistent
 *         run, or the fused C=32 downsample or upsample. */
static bool
run_or_fused (enum vulkan_kernel k)
{
	return vulkan_plan_persistent(k) || k == VULKAN_KERNEL_FSWIN_FUSED_UP32 ||
	       k == VULKAN_KERNEL_FSWIN_DSP32;
}

/** @brief Chains the dispatches with tile counters, with the heap's work in a struct chain.
 *
 * @param ds    The dispatches.
 * @param c     The chain, whose records are those of an idle dispatch.
 * @param blob  The blob.
 * @param base  The counters' first word in the arena.
 * @param tick  The dispatch that ticks the frame counter; the count when there is none.
 * @param words The words the counters take, from 1 on.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, ERROR_REJECTED or ERROR_FAILED.
 */
static enum error_code
chain_pairs (struct vulkan_dispatches *ds,
             struct chain             *c,
             struct vulkan_blob       *blob,
             uint64_t                  base,
             size_t                    tick,
             uint64_t                 *words,
             struct error             *e)
{
#define TRY(call) do { enum error_code const code_ = (call); if (code_) return code_; } while (0)
	struct vulkan_dispatch *const d = ds->dispatch;
	size_t const count = ds->count;
	// On a big frame, with a persistent run of more than 4096 windows a layer, the runs and the
	// fused C=32 downsample and upsample keep their barriers but between BIG_FRAME_PAIRS.
	bool big = false;
	for (size_t i = 0; i < count; ++i)
		if (vulkan_plan_persistent(d[i].kernel)) {
			struct push_persist const p = persist_of(&d[i]);
			big = big || (p.n_layers && p.total_windows / p.n_layers > 4096);
		}
	for (size_t i = 0; i < count && tick < count; ++i) {
		if (i + 1 == count || i < tick)
			continue;
		struct vulkan_dispatch *const p = &d[i];
		struct vulkan_dispatch *const q = &d[i + 1];
		bool kept = false;
		for (size_t b = 0; b < sizeof BIG_FRAME_PAIRS / sizeof *BIG_FRAME_PAIRS && !kept; ++b)
			kept = BIG_FRAME_PAIRS[b].p == p->kernel && BIG_FRAME_PAIRS[b].q == q->kernel;
		if (!counted_pair(p->kernel, q->kernel) ||
		    (big && !kept && (run_or_fused(p->kernel) || run_or_fused(q->kernel))))
			continue;
		uint32_t in = 0;
		uint32_t out = 0;
		TRY(input(q, blob, &in, e));
		TRY(output(p, blob, &out, e));
		if (in != out)
			continue;
		uint32_t itx = 0;
		uint32_t ity = 0;
		bool const up = upsampling_run(q->kernel) || q->kernel == VULKAN_KERNEL_FSWIN_FUSED_UP32;
		if (up) {
			bool has = false;
			TRY(gather_raster(q, blob, &has, &itx, &ity, e));
			if (!has)
				continue;
		}
		// A table entry per token tile the consumer reads: the counter of the producer's unit that
		// writes it, and the units that make it whole. A persistent run's tile raster is that of
		// its first layer as a consumer, of its last as a producer.
		uint32_t tiles = 4096;
		if (vulkan_plan_persistent(q->kernel)) {
			uint32_t run = 0;
			TRY(run_tiles(q, blob, true, &run, e));
			if (tiles < run)
				tiles = run;
		}
		if (vulkan_plan_persistent(p->kernel)) {
			uint32_t run = 0;
			TRY(run_tiles(p, blob, false, &run, e));
			if (tiles < run)
				tiles = run;
		}
		if (up && tiles < itx * ity)
			tiles = itx * ity;
		TRY(reserve(c, tiles, e));
		uint32_t *const table = c->table;
		for (uint32_t k = 0; k < tiles; ++k)
			table[k] = UINT32_MAX;
		uint32_t counters = tiles;
		bool fits = true;
		bool skip = false;
		if (p->kernel == VULKAN_KERNEL_ATTN)
			fits = attention_table(p, table, tiles, &counters);
		else if (gemm(p->kernel))
			fits = gemm_table(p, table, tiles);
		else if (p->kernel == VULKAN_KERNEL_VIT_ATTN)
			fits = vit_attention_table(p, table, tiles);
		else if (p->kernel == VULKAN_KERNEL_FSWIN_DSP32 || downsampling_run(p->kernel))
			TRY(downsampling_table(p, q, blob, c, tiles, &skip, &fits, e));
		else if (vulkan_plan_persistent(p->kernel))
			TRY(run_table(p, blob, up, itx, table, tiles, &skip, &fits, e));
		else
			fits = ffn_table(p, table, tiles);
		if (skip)
			continue;
		if (!fits)
			return error_reject(e, "its tile counters overflow their tables");
		uint32_t const counter = (uint32_t)(base + *words + 1);
		*words += 1 + (uint64_t)counters;
		// A C=512 projection that waits takes its M tiles in order.
		if (q->kernel == VULKAN_KERNEL_GEMM_PROJC) {
			struct push_gemm g;
			memcpy(&g, q->push, sizeof g);
			g.remap = 1;
			memcpy(q->push, &g, sizeof g);
		}
		c->rec[i][3] = counter;
		c->rec[i + 1][0] = counter;
		c->rec[i + 1][2] = (uint32_t)(vulkan_blob_put_words(blob, table, tiles) / 4);
		p->after = VULKAN_AFTER_NOTHING;
	}
	return ERROR_NONE;
#undef TRY
}

enum error_code
vulkan_schedule_chain (struct vulkan_dispatches *ds,
                       struct vulkan_blob       *blob,
                       uint64_t                 *arena,
                       uint32_t                 *error,
                       uint32_t                 *counter_words,
                       struct error             *e)
{
	struct vulkan_dispatch *const d = ds->dispatch;
	size_t const count = ds->count;
	*arena = align(*arena, 256);
	uint64_t const base = *arena / 4;
	// Every record names one error word, the first pair's, so that one word says whether a wait
	// of the frame ran out: a consumer's tile waits and a persistent run's claims set it when they
	// give up, and read it to give up early. A tile wait that ran out stops the frame's tile
	// waits, and a run that gave up a claim stops every wait. Upstream's records name each
	// consumer pair's own, and none for a dispatch that only signals. Each pair keeps its word
	// before its counters, which stay where upstream puts them.
	*error = (uint32_t)(base + 1);
	// The frame counter, which the first fswin32 ticks.
	size_t tick = 0;
	while (tick < count && d[tick].kernel != VULKAN_KERNEL_FSWIN32)
		++tick;
	uint64_t words = 1;
	// Each dispatch's record: the counters it waits on, the units it needs of each, its table, the
	// counters it signals, the frame counter, its error word and the magic word.
	uint32_t const idle[RECORD_WORDS] = {
		UINT32_MAX, 0, UINT32_MAX, UINT32_MAX, (uint32_t)base, *error, 0x54434852u,
	};
	struct chain c = {malloc(count * sizeof *c.rec), nullptr, nullptr, 0};
	if (!c.rec)
		return error_fail(e, "out of memory");
	for (size_t i = 0; i < count; ++i)
		memcpy(c.rec[i], idle, sizeof idle);
	if (tick < count)
		c.rec[tick][1] = 1;
	enum error_code code = chain_pairs(ds, &c, blob, base, tick, &words, e);
	free(c.table);
	c.table = nullptr;
	free(c.need);
	c.need = nullptr;
	if (!code) {
		for (size_t i = 0; i < count; ++i) {
			if (!counted(d[i].kernel))
				continue;
			uint32_t record = UINT32_MAX;
			if (memcmp(c.rec[i], idle, sizeof idle))
				record = (uint32_t)(vulkan_blob_put_words(blob, c.rec[i], RECORD_WORDS) / 4);
			d[i].push[d[i].words++] = record;
		}
		// A persistent run that neither waits nor signals, which upstream leaves without a record
		// (on a big frame, fswinpds64 and fswinpup256), names one that orders nothing, put after
		// upstream's: its claims too must give up once a run of the frame has given up a claim,
		// and stop the frame's waits when they give up.
		uint32_t nothing = UINT32_MAX;
		for (size_t i = 0; i < count; ++i)
			if (vulkan_plan_persistent(d[i].kernel) && d[i].push[d[i].words - 1] == UINT32_MAX) {
				if (nothing == UINT32_MAX) {
					uint64_t const at = vulkan_blob_put_words(blob, idle, RECORD_WORDS);
					nothing = (uint32_t)(at / 4);
				}
				d[i].push[d[i].words - 1] = nothing;
			}
		*arena += align(words * 4, 256);
		if (words > UINT32_MAX)
			code = error_reject(e, "its tile counters overflow 32-bit indices");
		else if (blob->error)
			code = error_fail(e, "out of memory");
		else
			*counter_words = (uint32_t)words;
	}
	free(c.rec);
	c.rec = nullptr;
	return code;
}

#undef RECORD_WORDS
#undef RUN_LAYERS
#undef REC_WORDS
#undef GRID_Y
#undef GRID_X
#undef TILES_Y
#undef TILES_X
#undef SWIN_WORDS
