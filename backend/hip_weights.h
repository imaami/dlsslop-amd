/** @file
 *
 * The HIP network's weights: each model file read and packed in place into the image its kernels
 * read, byte for byte as upstream packs it. A port of the production path of lmxxf's
 * packed_weights.h and the weight loaders of hip_reference_network.h (MIT). hip_weights.c defines
 * the functions and the tables.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_HIP_WEIGHTS_H_
#define DLSSLOP_AMD_BACKEND_HIP_WEIGHTS_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"

/** @brief How a weight file becomes its image: each is one of upstream's loaders.
 *
 * Their other arguments follow from the stem: the channel count, whether a C32 weight is attention
 * or FFN, and a ViT weight's shape.
 */
enum hip_recipe : uint8_t {
	HIP_RECIPE_RAW,               //!< Weight: the values as read.
	HIP_RECIPE_C32,               //!< PackedC32Weight.
	HIP_RECIPE_DS_CAST,           //!< PackedDsWeightCast, at 32 channels.
	HIP_RECIPE_DS_FRAG,           //!< PackedDsWeightFrag.
	HIP_RECIPE_MH_FFN,            //!< PackedFusedMhWeight below the tiled width: PackedMhWeight(ffn).
	HIP_RECIPE_MH_ATTENTION,      //!< PackedMhWeight(attention).
	HIP_RECIPE_MH_ATTENTION_DIAG, //!< PackedMhWeightDiag.
	HIP_RECIPE_FFN_FRAG,          //!< PackedFusedMhWeightFrag.
	HIP_RECIPE_QKV_FRAG_ONLY,     //!< PackedMhWeightQkvFragOnly.
	HIP_RECIPE_SPLIT_MIX_F16,     //!< PackedSplitFfnWeightMixHalf.
	HIP_RECIPE_PROJ_FRAG,         //!< PackedSplitProjectionFrag.
	HIP_RECIPE_QKV_FRAG,          //!< PackedMhWeightQkvFrag.
	HIP_RECIPE_VIT_FRAG,          //!< PackedVitWeight, fragment tiles.
	HIP_RECIPE_QKV_F16_FRAG,      //!< PackedVitQkvWeightFrag.
	HIP_RECIPE_VIT_PROJ_FRAG,     //!< PackedVitProjectionFrag.
	HIP_RECIPE_DECODER_F16R,      //!< PackedDecoderHalf.
	HIP_RECIPE_COUNT              //!< The number of recipes; no recipe.
};

/** @brief Upstream's key suffix for each recipe, by enum hip_recipe, a different one each.
 *
 * Logs name a packed weight STEM@SUFFIX, and a raw one by its stem alone: its suffix is empty.
 */
extern char const *const HIP_WEIGHTS_RECIPE_SUFFIX[];

/** @brief The bytes of a weight's stem, its terminating null included. */
#define HIP_WEIGHTS_STEM_BYTES 24

/** @brief A weight: its file's stem and one of the recipes upstream applies to such a stem. */
struct hip_weight_spec {
	char            stem[HIP_WEIGHTS_STEM_BYTES]; //!< The stem, such as block23-ffwd-projection, and a null.
	enum hip_recipe recipe;                       //!< The recipe.
	uint8_t         length;                       //!< The stem's length.
};

/** @brief The initializer of a struct hip_weight_spec of a literal stem and a recipe. */
#define HIP_WEIGHT_SPEC(stem, recipe) {stem, (recipe), sizeof (stem) - 1}

/** @brief The values in a weight's file (upstream: WeightElements).
 *
 * @param spec The weight.
 * @return     The count, or 0 for a stem that upstream does not know.
 */
extern size_t
hip_weights_file_elements (struct hip_weight_spec const *spec);

/** @brief The channel count of a weight's stem, which upstream passes the recipe's loader.
 *
 * @param spec The weight.
 * @return     The count, or 0 for a stem that upstream does not know.
 */
extern unsigned
hip_weights_channels (struct hip_weight_spec const *spec);

/** @brief The size of the image that hip_weights_load() makes of a file of
 *         hip_weights_file_elements() values: the bytes upstream uploads.
 *
 * @param spec The weight.
 * @return     The bytes.
 */
extern size_t
hip_weights_packed_bytes (struct hip_weight_spec const *spec);

/** @brief A weight file's values, and then the image they are packed into, which
 *         hip_weight_file_fini() frees. */
struct hip_weight_file {
	char  *path;     //!< The file's path, or nullptr.
	float *values;   //!< Its values, binary16 ones widened; then the image. Or nullptr.
	size_t count;    //!< The floats that the values or the image take.
	size_t capacity; //!< The floats that their memory holds.
};

/** @brief Loads a weight: reads its file, ASSETS/STEM.f32 or ASSETS/STEM.f16 when there is no
 *         .f32 (upstream: ReadWeights), and packs its values in place into the image upstream
 *         uploads, of hip_weights_packed_bytes() bytes (upstream: the recipe's loader).
 *
 * The stem must be one upstream knows and applies the weight's recipe to, and the file must hold
 * exactly hip_weights_file_elements() values. Upstream reads the .f16 whenever the .f32 does not
 * open, and its ds-cast, ds-frag and decoder loaders accept longer files. Bytes that a recipe does
 * not rewrite keep the file's f32 bytes, which upstream uploads as they are. The packers' errors
 * name the file and the element.
 *
 * @param dest          Receives the image, which hip_weight_file_fini() frees; empty on a failure.
 * @param assets        The directory of the weights; it need not be null-terminated.
 * @param assets_length The length of its path.
 * @param spec          The weight.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_weights_load (struct hip_weight_file       *dest,
                  char const                   *assets,
                  size_t                        assets_length,
                  struct hip_weight_spec const *spec,
                  struct error                 *e);

/** @brief Frees a weight file's path and values and empties it.
 *
 * @param file The file, or nullptr.
 */
extern void
hip_weight_file_fini (struct hip_weight_file *file);

/* The packing primitives, exposed for the tests. Those over many values fail with
 * "element N: WORDS" for the first value N they reject. */

/** @brief Packs a weight's values in place into its image, as hip_weights_load() packs those it
 *         reads.
 *
 * The file holds hip_weights_file_elements() values of a weight whose stem and recipe
 * hip_weights_load() accepts. The errors name the file's path and the element.
 *
 * @param spec The weight.
 * @param file The values and the path; its memory grows if it does not hold the image.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_weights_pack (struct hip_weight_spec const *spec,
                  struct hip_weight_file       *file,
                  struct error                 *e);

/** @brief Why a value has no exact encoding. The encodings of single values return one, so that
 *         their loops build no words. */
enum hip_flaw : uint8_t {
	HIP_FLAW_NONE,           //!< The value has an exact encoding.
	HIP_FLAW_FP8_NONFINITE,  //!< NaN or infinity, for E4M3.
	HIP_FLAW_FP8_SUBNORMAL,  //!< Below E4M3's normals and not one of its subnormals.
	HIP_FLAW_FP8_INEXACT,    //!< In E4M3's normal range or above it, and not one of its values.
	HIP_FLAW_HALF_NONFINITE, //!< NaN or infinity, for binary16.
	HIP_FLAW_HALF_OVERFLOW,  //!< Above binary16's range.
	HIP_FLAW_HALF_INEXACT,   //!< In binary16's normal range, and not one of its values.
	HIP_FLAW_HALF_SUBNORMAL, //!< Below binary16's normals and not one of its subnormals.
	HIP_FLAW_UNGROUPED,      //!< Nonzero outside an FFN weight's grouped contraction.
	HIP_FLAW_COUNT           //!< The number of flaws and none.
};

/** @brief Upstream's words for each flaw, by enum hip_flaw; empty for HIP_FLAW_NONE. */
extern char const *const HIP_WEIGHTS_FLAW_WORDS[];

/** @brief Binary16 widened exactly; NaN payloads are kept (upstream: Half).
 *
 * @param half The binary16 bits.
 * @return     The value.
 */
extern float
hip_weights_widen_half (uint16_t half);

/** @brief E4M3 of an exactly representable value (upstream: ExactWeightFp8).
 *
 * @param value The value.
 * @param code  Receives its E4M3 code; untouched when it has none.
 * @return      HIP_FLAW_NONE, or why the value has no code.
 */
extern enum hip_flaw
hip_weights_exact_fp8 (float    value,
                       uint8_t *code);

/** @brief Binary16 of an exactly representable value (upstream: ExactWeightHalf).
 *
 * @param value The value.
 * @param half  Receives its binary16 bits; untouched when it has none.
 * @return      HIP_FLAW_NONE, or why the value has no binary16.
 */
extern enum hip_flaw
hip_weights_exact_half (float     value,
                        uint16_t *half);

/** @brief Binary16 rounded to nearest even, subnormals included, overflow made infinity and NaN
 *         0x7e00 with its sign (upstream: RoundWeightHalf).
 *
 * @param value The value.
 * @return      The binary16 bits.
 */
extern uint16_t
hip_weights_round_half (float value);

/** @brief A value rounded to nearest even E4M3 and saturated at ±448, NaN and infinity included
 *         (upstream: HostF).
 *
 * @param value The value.
 * @return      The E4M3 value.
 */
extern float
hip_weights_saturate_fp8 (float value);

/** @brief The next E4M3 piece of a residual scale, as the device splits it (upstream:
 *         HostScalePiece).
 *
 * @param value What remains of the scale.
 * @return      The piece.
 */
extern float
hip_weights_scale_piece (float value);

/** @brief A matrix of bytes as fragment-native 512-byte tiles (upstream: FragmentPackedMatrix).
 *
 * @param values  The floats whose bytes hold the matrix.
 * @param start   The float at which the matrix starts.
 * @param rows    Its rows, a multiple of 16.
 * @param columns Its columns, a multiple of 32.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED without memory for a copy of the matrix.
 */
extern enum error_code
hip_weights_fragment_tiles (float        *values,
                            size_t        start,
                            size_t        rows,
                            size_t        columns,
                            struct error *e);

/** @brief The floats of a C32 FFN weight's file. */
#define HIP_WEIGHTS_C32_FFN_FLOATS 8736

/** @brief The floats that a C32 FFN weight's diagonal tiles take: 12 tiles of 512 bytes. */
#define HIP_WEIGHTS_C32_DIAGONAL_FLOATS 1536

/** @brief Appends a C32 FFN weight's residual scales' pieces as diagonal E4M3 tiles (upstream:
 *         AppendC32ResidualDiagonals).
 *
 * @param values The weight's HIP_WEIGHTS_C32_FFN_FLOATS floats, with room for
 *               HIP_WEIGHTS_C32_DIAGONAL_FLOATS more, which receive the tiles.
 */
extern void
hip_weights_append_c32_diagonals (float *values);

/** @brief Appends an MH attention weight's residual scales' pieces as diagonal E4M3 tiles
 *         (upstream: AppendMhResidualDiagonals).
 *
 * @param values The weight's floats, with room for 24 * c more, which receive the tiles.
 * @param count  The number of the weight's floats.
 * @param c      Its channels.
 */
extern void
hip_weights_append_mh_diagonals (float    *values,
                                 size_t    count,
                                 unsigned  c);

/** @brief Checks that an FFN weight contracts in groups only, which its kernels assume (upstream:
 *         ValidateGroupedMhContract).
 *
 * @param values The weight's floats.
 * @param c      Its channels.
 * @param e      Receives the words for the first value outside the groups, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_weights_check_grouped_contract (float const  *values,
                                    unsigned      c,
                                    struct error *e);

#endif /* DLSSLOP_AMD_BACKEND_HIP_WEIGHTS_H_ */
