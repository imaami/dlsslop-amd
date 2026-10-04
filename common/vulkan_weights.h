/** @file
 *
 * The Vulkan network's weights: dlssnr.bin read and packed into the blob that the network's kernels
 * read, byte for byte as upstream packs it. A port of the weight lowering of DLSSNR-AMD's
 * linux/src/core/nr_graph.cpp and tinlayout.hpp (MIT). vulkan_weights.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_VULKAN_WEIGHTS_H_
#define DLSSLOP_AMD_COMMON_VULKAN_WEIGHTS_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "vulkan_plan.h"

/** @brief The bytes that the longest name of a source takes, its null included. */
#define VULKAN_WEIGHTS_NAME_BYTES 80

/** @brief Writes a source's name in the model pack, upstream's:
 *         DIRECTORY/blockB.layerL.layer.SUFFIX.bin, or DIRECTORY/blockB.layerL.layer.bin for a record.
 *
 * @param source The source.
 * @param name   Receives the name and a null.
 * @return       The name's length.
 */
extern size_t
vulkan_weights_entry_name (struct vulkan_source const *source,
                           char                        name[VULKAN_WEIGHTS_NAME_BYTES]);

struct vulkan_model_entry;

/** @brief A model pack's index, in linux/package/model-tools' NRMODEL1 format (upstream:
 *         open_model_pack).
 *
 * "NRMODEL1", a u32 entry count and a u32 that is not read, then for each entry a u32 name length
 * of at most 4096, the name, and its data's u64 offset and u64 size. Only vulkan_model_open() makes
 * one, and vulkan_model_fini() frees it.
 */
struct vulkan_model {
	char const                *path;    //!< The pack's path, the caller's.
	char                      *names;   //!< The entries' names, one after another, without nulls.
	struct vulkan_model_entry *entries; //!< The entries, sorted by name.
	size_t                     count;   //!< Their number.
	int                        fd;      //!< The pack, or -1.
};

/** @brief Memory that grows to hold what is read into it. */
struct vulkan_scratch {
	uint8_t *bytes;    //!< The memory, or nullptr.
	size_t   capacity; //!< Its bytes.
};

/** @brief Frees a scratch's memory and empties it.
 *
 * @param scratch The scratch, or nullptr.
 */
extern void
vulkan_scratch_fini (struct vulkan_scratch *scratch);

/** @brief Opens a model pack and reads its index.
 *
 * It is refused when the pack lacks the magic, when its index ends early or holds a name longer
 * than 4096 bytes, when an entry reaches past the end of the pack, and when a name is listed twice;
 * the errors name the path.
 *
 * @param dest Receives the model, which vulkan_model_fini() frees; empty, with no descriptor, on a
 *             failure.
 * @param path The pack's path, which the model keeps for its errors: it must outlive the model.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_model_open (struct vulkan_model *dest,
                   char const          *path,
                   struct error        *e);

/** @brief Closes a model pack and frees its index.
 *
 * @param model The model, or nullptr.
 */
extern void
vulkan_model_fini (struct vulkan_model *model);

/** @brief Reads a source's bytes into the start of a scratch, which grows to hold them.
 *
 * @param model   The model.
 * @param source  The source.
 * @param scratch The scratch.
 * @param size    Receives the bytes read.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_model_read (struct vulkan_model const  *model,
                   struct vulkan_source const *source,
                   struct vulkan_scratch      *scratch,
                   size_t                     *size,
                   struct error               *e);

/** @brief Packs segments of a blob, in offset order, from a model's entries; the bytes between and
 *         after them are zeroed.
 *
 * An entry must hold at least the bytes that its recipe reads, or exactly those with
 * VULKAN_SEGMENT_EXACT; the errors name the model's path and the entry. A segment out of order,
 * outside the blob or of a size that its recipe does not write is the plan's error.
 *
 * @param segments      The segments.
 * @param segment_count Their number.
 * @param tables        The words that TABLE segments copy.
 * @param table_count   Their number.
 * @param model         The model.
 * @param blob          Receives the blob.
 * @param blob_size     Its bytes.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_weights_pack (struct vulkan_segment const *segments,
                     size_t                       segment_count,
                     uint32_t const              *tables,
                     size_t                       table_count,
                     struct vulkan_model const   *model,
                     uint8_t                     *blob,
                     size_t                       blob_size,
                     struct error                *e);

/** @brief Audits which heads of the Swin layers whose position biases a plan's segments bake (BIAS
 *         with VULKAN_SEGMENT_AFFINE) a model's weights keep below the exponent's upper clamp
 *         (struct vulkan_clamp_free), from the head's biases as the blob holds them and its scale in
 *         the layer's scalars_b (upstream: the audit in NrSession::build).
 *
 * A NaN bias frees no head. An entry must hold at least the bytes the audit reads; the errors name
 * the model's path and the entry.
 *
 * @param segments The segments.
 * @param count    Their number.
 * @param model    The model.
 * @param dest     Receives the free heads.
 * @param e        Receives the words for what stopped it, or nullptr.
 * @return         ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
vulkan_weights_clamp_free (struct vulkan_segment const *segments,
                           size_t                       count,
                           struct vulkan_model const   *model,
                           struct vulkan_clamp_free    *dest,
                           struct error                *e);

/* The packing's primitives, exposed for the tests. */

/** @brief An E4M3 code through upstream's round trip (upstream: e4m3_requantise_table): 0x80 (-0)
 *         for 0x00 (+0), 0x7f for 0xff (both NaN), and every other code as it is. */
extern uint8_t
vulkan_weights_requantise (uint8_t code);

/** @brief Binary16 widened to f32 bits as tin::f16_to_f widens it: NaN becomes the quiet
 *         0x7fc00000. */
extern uint32_t
vulkan_weights_widen_half (uint16_t half);

/** @brief Binary16 widened and narrowed again (upstream: tin::f_to_f16 of tin::f16_to_f): NaN
 *         becomes 0x7e00, every other value is kept. */
extern uint16_t
vulkan_weights_recode_half (uint16_t half);

/** @brief A rows x cols row-major matrix as 16x16 tiles, 256 bytes each (upstream:
 *         tin::tile_blocked).
 *
 * @param matrix The matrix.
 * @param rows   Its rows, a multiple of 16.
 * @param cols   Its columns, a multiple of 16.
 * @param out    Receives the tiles.
 */
extern void
vulkan_weights_tile_blocked (uint8_t const *matrix,
                             size_t         rows,
                             size_t         cols,
                             uint8_t       *out);

/** @brief The same as vulkan_weights_tile_blocked(), then each pair of 16-row tiles interleaved in
 *         8-byte runs (upstream: pack_matrix in NrSession::build, nr_graph.cpp:3270-3281).
 *
 * @param matrix The matrix.
 * @param rows   Its rows, a multiple of 32.
 * @param cols   Its columns, a multiple of 16.
 * @param out    Receives the tiles.
 */
extern void
vulkan_weights_npair_blocked (uint8_t const *matrix,
                              size_t         rows,
                              size_t         cols,
                              uint8_t       *out);

/** @brief The byte of a C=512 FFN record that holds A[g][row][k], upstream's gather. */
extern size_t
vulkan_weights_ff_a (size_t g,
                     size_t row,
                     size_t k);

/** @brief The byte of a C=512 FFN record that holds Q0[g][j][row], upstream's gather. */
extern size_t
vulkan_weights_ff_q0 (size_t g,
                      size_t j,
                      size_t row);

/** @brief The byte of a C=512 FFN record that holds Q2[g][n][j], upstream's gather. */
extern size_t
vulkan_weights_ff_q2 (size_t g,
                      size_t n,
                      size_t j);

/** @brief The byte of a ViT QKV record that holds row row, column k of matrix which (0 Q, 1 K,
 *         2 V), upstream's gather. */
extern size_t
vulkan_weights_vit_qkv_weight_byte (size_t which,
                                    size_t row,
                                    size_t k);

/** @brief Which of a head's 4096 biases, in MMA C-fragment order, is bias [i][j] (upstream:
 *         deswizzle_bias). */
extern size_t
vulkan_weights_bias_source (size_t i,
                            size_t j);

/** @brief The bytes of the activation table. */
#define VULKAN_WEIGHTS_ACTIVATION_BYTES 4096

/** @brief Writes the activation table (upstream: nr::detail::activation_lut_v1).
 *
 * @param table Receives the table.
 */
extern void
vulkan_weights_activation_table (uint8_t table[VULKAN_WEIGHTS_ACTIVATION_BYTES]);

#endif /* DLSSLOP_AMD_COMMON_VULKAN_WEIGHTS_H_ */
