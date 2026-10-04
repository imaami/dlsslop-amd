/** @file
 *
 * Inside the Vulkan network's plan: the dispatches that the lowering of its layers makes, the weight
 * blob as it is put together, and the schedule that merges the dispatches into persistent runs,
 * shares the activation arena among their values and chains them with tile counters. vulkan_plan.c
 * and vulkan_schedule.c share it.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_VULKAN_PLAN_PRIV_H_
#define DLSSLOP_AMD_COMMON_VULKAN_PLAN_PRIV_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "vulkan_plan.h"

/** @brief A Swin body at C=64, 128 or 256 as the lowering leaves it, for a persistent run to take
 *         as a layer. */
enum vulkan_body : uint8_t {
	VULKAN_BODY_NONE,    //!< Not a body.
	VULKAN_BODY_SWIN,    //!< A plain one (upstream: fswin<C>).
	VULKAN_BODY_SWIN_DS, //!< One with the downsample fused (fswindsp<C>).
	VULKAN_BODY_SWIN_UP, //!< One with the upsample fused (fswinfusedup<C>).
};

/** @brief A dispatch (upstream: Disp).
 *
 * It runs kernel, or a body of width c when kernel is VULKAN_KERNEL_COUNT, over groups with the
 * push block of words words. It runs layers first to last, starting with block's layer layer; after
 * orders it before the next.
 */
struct vulkan_dispatch {
	uint32_t           groups[3]; //!< Its workgroups.
	uint32_t           push[32];  //!< Its push block.
	uint16_t           c;         //!< Its body's width.
	enum vulkan_kernel kernel;    //!< Its kernel; VULKAN_KERNEL_COUNT for a body.
	enum vulkan_body   body;      //!< Its body.
	enum vulkan_after  after;     //!< What orders it before the next.
	uint8_t            words;     //!< Its push words.
	uint8_t            block;     //!< The block of its first layer.
	uint8_t            layer;     //!< Its first layer in that block.
	uint8_t            first;     //!< Its first layer in the layer table.
	uint8_t            last;      //!< Its last.
};

/** @brief Dispatches in order. */
struct vulkan_dispatches {
	struct vulkan_dispatch *dispatch; //!< The dispatches.
	size_t                  count;    //!< Their number.
	size_t                  capacity; //!< The number that their memory holds.
};

/** @brief The activation arena's keys: 8 a block (see struct vulkan_value). */
#define VULKAN_PLAN_KEYS (8 * VULKAN_PLAN_BLOCKS)

/** @brief The activation arena's values by key (upstream: vsize, overread, voff.m and voff.used). */
struct vulkan_values {
	uint64_t size[VULKAN_PLAN_KEYS];     //!< Their sizes.
	uint64_t overread[VULKAN_PLAN_KEYS]; //!< The bytes their readers reach past them.
	uint64_t offset[VULKAN_PLAN_KEYS];   //!< Their offsets.
	int16_t  first[VULKAN_PLAN_KEYS];    //!< The first layer whose lowering names each, -1 when none does.
	int16_t  last[VULKAN_PLAN_KEYS];     //!< The last.
};

/** @brief The weight blob as it is put together (upstream: wblob and its put()).
 *
 * A put that runs out of memory sets error and puts nothing; what it returns is then meaningless,
 * and the build fails once its stage ends.
 */
struct vulkan_blob {
	struct vulkan_segment *segments;         //!< The segments in offset order.
	uint32_t              *tables;           //!< The words that TABLE segments copy.
	size_t                 segment_count;    //!< The number of segments.
	size_t                 segment_capacity; //!< The number that their memory holds.
	size_t                 table_count;      //!< The number of table words.
	size_t                 table_capacity;   //!< The number that their memory holds.
	uint64_t               bytes;            //!< The blob's bytes so far.
	enum error_code        error;            //!< ERROR_FAILED once a put ran out of memory.
};

/** @brief Puts a segment at the next multiple of an alignment.
 *
 * @param blob    The blob.
 * @param segment The segment; its offset is set.
 * @param align   The alignment.
 * @return        Its offset.
 */
extern uint64_t
vulkan_blob_put (struct vulkan_blob    *blob,
                 struct vulkan_segment  segment,
                 uint32_t               align);

/** @brief Puts words as a table at the next multiple of 16.
 *
 * @param blob  The blob.
 * @param words The words.
 * @param count Their number.
 * @return      Its offset.
 */
extern uint64_t
vulkan_blob_put_words (struct vulkan_blob *blob,
                       uint32_t const     *words,
                       size_t              count);

/** @brief A word of the blob, which a table holds (upstream: wword).
 *
 * @param blob  The blob.
 * @param index The word's index in the blob.
 * @param word  Receives the word.
 * @param e     Receives the words for a word outside the tables, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_blob_word (struct vulkan_blob const *blob,
                  uint64_t                  index,
                  uint32_t                 *word,
                  struct error             *e);

/** @brief Merges the Swin bodies at C>=64 into persistent runs (upstream: nr_graph.cpp:3354-3637).
 *
 * @param dest    Receives the dispatches, which the caller frees; untouched on a failure.
 * @param lowered The lowered dispatches.
 * @param blob    The blob, which takes the runs' tables.
 * @param arena   The arena's end, which grows by the runs' sync regions.
 * @param epoch   Receives whether there is a run at C=256, whose sync words count the frames that
 *                tile counters need.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_schedule_merge (struct vulkan_dispatches       *dest,
                       struct vulkan_dispatches const *lowered,
                       struct vulkan_blob             *blob,
                       uint64_t                       *arena,
                       bool                           *epoch,
                       struct error                   *e);

/** @brief Shares the arena among the values by lifetime (upstream: the arena probe,
 *         nr_graph.cpp:3657-3799).
 *
 * @param dispatches The dispatches, whose layers the values' lifetimes follow.
 * @param chains     Whether to keep a value that a stretch of tile-counted steps reads until its end.
 * @param values     The values in their plain layout, whose offsets become the shared ones.
 * @param end        Receives the end of the values.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_schedule_share (struct vulkan_dispatches const *dispatches,
                       bool                            chains,
                       struct vulkan_values           *values,
                       uint64_t                       *end,
                       struct error                   *e);

/** @brief Trims the post block's window rows that start past the picture, and the window rows of
 *         the plain C=32 layers before it that only those read (upstream: the dead rows of
 *         NrSession::build, NR_DEAD_ROWS).
 *
 * @param dispatches The dispatches, which end with the post block.
 * @param height     The picture's height.
 */
extern void
vulkan_schedule_trim (struct vulkan_dispatches *dispatches,
                      uint32_t                  height);

/** @brief Puts tile counters in place of the barriers between dispatches that they can replace
 *         (upstream: nr_graph.cpp:3844-4247).
 *
 * Their tables and records go in the blob, their counters after the arena's end, and each dispatch
 * they order gets VULKAN_AFTER_NOTHING.
 *
 * @param dispatches The dispatches.
 * @param blob       The blob.
 * @param arena      The arena's end, which grows by the counters.
 * @param error      Receives the arena's u32 word that every record names, and that the waits set
 *                   when they give up.
 * @param words      Receives the counters' words.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE; ERROR_REJECTED when the counters overflow their tables or 32-bit
 *                   indices; ERROR_FAILED.
 */
extern enum error_code
vulkan_schedule_chain (struct vulkan_dispatches *dispatches,
                       struct vulkan_blob       *blob,
                       uint64_t                 *arena,
                       uint32_t                 *error,
                       uint32_t                 *words,
                       struct error             *e);

/** @brief The word of a persistent run's sync region, from its sync_off on, that the run's waits
 *         set when they run out (fswin_t.comp). */
#define VULKAN_PLAN_RUN_ERROR 3

#endif /* DLSSLOP_AMD_COMMON_VULKAN_PLAN_PRIV_H_ */
