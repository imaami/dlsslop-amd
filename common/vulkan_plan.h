/** @file
 *
 * The Vulkan network's plan: what the host works out from the frame's extent before it makes any
 * Vulkan object. Its steps with their push constants, the segments of the weight blob and the
 * activation arena's size, all of which depend on the extent alone. A port of the production path
 * of DLSSNR-AMD's linux/src/core (MIT): the layer table and walk of nr_native_plan.cpp, and
 * NrSession::build in nr_graph.cpp up to the device. vulkan_plan.c and vulkan_schedule.c define the
 * functions and the tables.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_VULKAN_PLAN_H_
#define DLSSLOP_AMD_COMMON_VULKAN_PLAN_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
#else
# include <stddef.h>
# include <stdint.h>
#endif

#include "error.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/* The shader build's constants that the plan's arithmetic depends on (upstream:
 * linux/build/arch/rdna4.sh, and nr_graph.cpp's defaults for those it leaves unset). The SPIR-V
 * records them in shader-constants.txt. */

/** @brief The ViT's wide GEMM tile's tokens. */
#define VULKAN_PLAN_GEMM_WIDE_MT UINT32_C(64)

/** @brief The ViT's wide GEMM tile's outputs. */
#define VULKAN_PLAN_GEMM_WIDE_NT UINT32_C(256)

/** @brief The residual projections' tile's tokens. */
#define VULKAN_PLAN_GEMM_PROJ_MT UINT32_C(32)

/** @brief The residual projections' tile's outputs. */
#define VULKAN_PLAN_GEMM_PROJ_NT UINT32_C(128)

/** @brief gemmprojw's tile's tokens. */
#define VULKAN_PLAN_GEMM_PROJW_MT UINT32_C(64)

/** @brief gemmprojw's tile's outputs. */
#define VULKAN_PLAN_GEMM_PROJW_NT UINT32_C(128)

/** @brief The ViT tokens from which on gemmprojw runs. */
#define VULKAN_PLAN_PROJW_MIN_TOKENS UINT32_C(768)

/** @brief gemmvqkvnorms' tile's tokens. */
#define VULKAN_PLAN_GEMM_QKVS_MT UINT32_C(32)

/** @brief gemmvqkvnorms' tile's outputs. */
#define VULKAN_PLAN_GEMM_QKVS_NT UINT32_C(384)

/** @brief The tokens up to which the ViT's QKV runs gemmvqkvnorms. */
#define VULKAN_PLAN_QKVS_MAX_TOKENS UINT32_C(768)

/** @brief ffwd3's subgroups a workgroup. */
#define VULKAN_PLAN_FFWD_WGW UINT32_C(4)

/** @brief The tokens from which on ffwd3w runs. */
#define VULKAN_PLAN_FFWD_FM2_MIN_TOKENS UINT32_C(2560)

/** @brief decups' channels an invocation. */
#define VULKAN_PLAN_DECUPS_VEC UINT32_C(16)

/** @brief vitattn's query tokens a workgroup. */
#define VULKAN_PLAN_VATTN_QT UINT32_C(32)

/** @brief A persistent run's straggler threshold, in percent. */
#define VULKAN_PLAN_STRAGGLER_PERCENT UINT32_C(25)

/** @brief The widths whose persistent runs of 65 to VULKAN_PLAN_PERSIST_ONE_MAX windows a layer take
 *         a workgroup an item: bit 2 C=128, 4 C=256. */
#define VULKAN_PLAN_PERSIST_ONE_MASK UINT32_C(6)

/** @brief The most windows a layer of a run that takes a workgroup an item. */
#define VULKAN_PLAN_PERSIST_ONE_MAX UINT32_C(400)

/** @brief A persistent run's polls before it gives up. */
#define VULKAN_PLAN_SPIN_LIMIT UINT32_C(4194304)

/** @brief The steps before its first use that a value smaller than VULKAN_PLAN_ARENA_COLD_MAX takes
 *         only memory freed by. */
#define VULKAN_PLAN_ARENA_COLD UINT32_C(1)

/** @brief The bytes below which a value is cold. */
#define VULKAN_PLAN_ARENA_COLD_MAX (UINT64_C(8) << 20)

/** @brief attn's head groups: its grid's z. */
#define VULKAN_PLAN_ATTENTION_SPLIT UINT32_C(16)

/** @brief The network's blocks: 0 the pre block to 70 the post block. */
#define VULKAN_PLAN_BLOCKS 71

/** @brief The first line of shader-constants.txt, as linux/build/build_network.py writes it from
 *         pipelines.json's markers. */
extern char const VULKAN_PLAN_MANIFEST[];

/** @brief A line of shader-constants.txt after VULKAN_PLAN_MANIFEST: a constant's key and value. */
struct vulkan_shader_constant {
	char const    *key;   //!< The constant's key.
	STD(uint32_t)  value; //!< Its value.
};

/** @brief The number of VULKAN_PLAN_SHADER_CONSTANTS. */
#define VULKAN_PLAN_SHADER_CONSTANT_COUNT 29

/** @brief shader-constants.txt's constants, in its order.
 *
 * The SPIR-V that the plan's kernels load must carry them exactly; the fixed paths that the plan
 * takes (weight layout 3, math profile 3 and the fusions) are among them.
 */
extern struct vulkan_shader_constant const VULKAN_PLAN_SHADER_CONSTANTS[VULKAN_PLAN_SHADER_CONSTANT_COUNT];

/** @brief A marker beside the SPIR-V: a file of one line. */
struct vulkan_marker {
	char const *file; //!< The file's name.
	char const *line; //!< Its line.
};

/** @brief The number of VULKAN_PLAN_MARKERS. */
#define VULKAN_PLAN_MARKER_COUNT 3

/** @brief The markers beside the SPIR-V other than shader-constants.txt. */
extern struct vulkan_marker const VULKAN_PLAN_MARKERS[VULKAN_PLAN_MARKER_COUNT];

/** @brief The network's pipelines: those that the plan's steps run, and the noise field that the
 *         build fills the weight blob's noise region with. Each is the SPIR-V g_STEM.spv. */
enum vulkan_kernel : STD(uint8_t) {
	VULKAN_KERNEL_FSWIN32,
	VULKAN_KERNEL_FSWIN_IMAGE_PREDS32,
	VULKAN_KERNEL_FSWIN_DSP32,
	VULKAN_KERNEL_FSWIN_PDS64,
	VULKAN_KERNEL_FSWIN_PDS128,
	VULKAN_KERNEL_FSWIN_PDS256,
	VULKAN_KERNEL_FSWIN_PUP64,
	VULKAN_KERNEL_FSWIN_PUP128,
	VULKAN_KERNEL_FSWIN_PUP256,
	VULKAN_KERNEL_FSWIN_FUSED_UP32,
	VULKAN_KERNEL_FSWIN_IMAGE_POST32,
	VULKAN_KERNEL_FSWIN32_NH,
	VULKAN_KERNEL_FSWIN_IMAGE_PREDS32_NH,
	VULKAN_KERNEL_FSWIN_DSP32_NH,
	VULKAN_KERNEL_FSWIN_FUSED_UP32_NH,
	VULKAN_KERNEL_FFWD3,
	VULKAN_KERNEL_FFWD3W,
	VULKAN_KERNEL_ATTN,
	VULKAN_KERNEL_GEMM_PROJC,
	VULKAN_KERNEL_GEMM_POOL,
	VULKAN_KERNEL_GEMM_NORES,
	VULKAN_KERNEL_GEMM_VACT,
	VULKAN_KERNEL_GEMM_PROJ,
	VULKAN_KERNEL_GEMM_PROJW,
	VULKAN_KERNEL_GEMM_PROJT,
	VULKAN_KERNEL_GEMM_VQKV_NORM,
	VULKAN_KERNEL_GEMM_VQKV_NORMS,
	VULKAN_KERNEL_GEMM_VQKVS,
	VULKAN_KERNEL_VIT_ATTN,
	VULKAN_KERNEL_DEC_UPS,
	VULKAN_KERNEL_UPS_VIEW,
	VULKAN_KERNEL_NOISE_FIELD,
	VULKAN_KERNEL_COUNT //!< The number of kernels; no kernel.
};

/** @brief The images that a kernel binds after its buffers. */
enum vulkan_images : STD(uint8_t) {
	VULKAN_IMAGES_NONE,   //!< None.
	VULKAN_IMAGES_INPUT,  //!< The input, sampled.
	VULKAN_IMAGES_OUTPUT, //!< The answer and the second output as storage, then the input, sampled.
};

/** @brief What a kernel binds and pushes (upstream: the Kernel::create calls of NrSession::build,
 *         nr_graph.cpp:4487-4546).
 *
 * Its storage buffers are named in binding order: 'a' the activation arena, 'w' the weight blob.
 */
struct vulkan_kernel_info {
	char const         *stem;    //!< Its SPIR-V's stem.
	char const         *buffers; //!< Its storage buffers in binding order.
	enum vulkan_images  images;  //!< Its images after them.
	STD(uint8_t)        push;    //!< The bytes of its push-constant range.
};

/** @brief The most bindings, buffers and images, that a kernel of VULKAN_PLAN_KERNELS takes. */
#define VULKAN_KERNEL_MOST_BINDINGS 8

/** @brief Each kernel's bindings and push range, by enum vulkan_kernel. */
extern struct vulkan_kernel_info const VULKAN_PLAN_KERNELS[VULKAN_KERNEL_COUNT];

/** @brief A C=32 kernel and its twin built without the upper clamp of the Swin attention's exponent
 *         (upstream: the "nh" pipelines of NR_EXP_NOHI), which takes the same push block and
 *         bindings. */
struct vulkan_unclamped {
	enum vulkan_kernel kernel; //!< The kernel.
	enum vulkan_kernel twin;   //!< Its twin without the clamp.
};

/** @brief The number of VULKAN_PLAN_UNCLAMPED. */
#define VULKAN_PLAN_UNCLAMPED_COUNT 4

/** @brief The kernels that have a twin without the upper clamp. */
extern struct vulkan_unclamped const VULKAN_PLAN_UNCLAMPED[VULKAN_PLAN_UNCLAMPED_COUNT];

/* The push blocks, as upstream's host declares them (nr_graph.cpp:144 and 776-858) and the GLSL
 * does: the SPIR-V's contract, field for field. A fused kernel's block is several of them in a row.
 * A kernel that waits on or signals tile counters takes one more word, the u32 index in the weight
 * blob of its record, or ~0 for none. Offsets name the activation arena and the weight blob in
 * bytes (_off), or in binary16, f32 or u32 elements where upstream's comments say so. */

/** @brief A Swin body's block (upstream: PushFSwin). */
struct push_f_swin {
	STD(uint32_t) x_off, o_off, e_off, ct_off, qkv_off, op_off, b_off, rs_off, ars_off, s_off;
	STD(uint32_t) rsd_off, ard_off, mid_off, tiles_x, tiles_y;
	STD(int32_t)  shift, shift_y;
	STD(uint32_t) pool_off, pool_tiles_x, pool_tiles_y;
};

/** @brief A downsampling body's resample projection, fswindsp (upstream: PushDsProj). */
struct push_ds_proj {
	STD(uint32_t) w_off, o_off, otx, raster, crow, rows, mode, writer_rows, n, clear_x, clear_y;
};

/** @brief An upsampling body's blend of the skip, fswinfusedup and fswinimagepost (upstream:
 *         PushUps). */
struct push_ups {
	STD(uint32_t) p_off, s_off, o_off, g_off, tiles_x, tiles_y, itiles_x, itiles_y, mode, stiles_x;
};

/** @brief The pre block's image features, fswinimagepreds32 (upstream: PushPreImage). */
struct push_pre_image {
	STD(uint32_t) lift, source_W, source_H, seed;
	float         style, tone, structure, skin, other, noise, constant;
	STD(uint32_t) noise_off;
};

/** @brief The post block's output projection, fswinimagepost32 (upstream: PushImageTail). */
struct push_image_tail {
	STD(uint32_t) w_off;
	float         intensity;
};

/** @brief A persistent run of layers, fswinpds and fswinpup: its persist_rec array, sync region,
 *         dependency tables and DS or UPS table, as u32 indices (upstream: PushPersist). */
struct push_persist {
	STD(uint32_t) layers_off, sync_off, n_layers, spin_limit, total_windows, df_off, df_n0, ds_off;
};

/** @brief A layer of a persistent run, in the weight blob (upstream: PersistRec). */
struct persist_rec {
	struct push_f_swin p;
	STD(uint32_t)      gx, gy, windows, flag_base;
};

/** @brief A GEMM's block (upstream: PushGemm). */
struct push_gemm {
	STD(uint32_t) gd_off, x_off, r_off, o_off, d_off, w_off, g_off, M, N, K, p_off, W;
	STD(uint32_t) otx, raster, crow, rows, ds_raster, writer_rows, clear_x, clear_y, remap;
};

/** @brief The C=512 FFN's block (upstream: PushFfwd3). */
struct push_ffwd3 {
	STD(uint32_t) x_off, o_off, a_off, q0_off, q2_off, M, C;
};

/** @brief The C=512 windowed attention's block (upstream: PushAttn). */
struct push_attn {
	STD(uint32_t) x_off, o_off, w_off, b_off, s_off, C, wins_x, tiles_x, tiles_y;
	STD(int32_t)  shift, shift_y;
};

/** @brief The ViT's attention's block (upstream: PushVAttn). */
struct push_v_attn {
	STD(uint32_t) x_off, o_off, tokens, s_off, mode;
};

/** @brief The decoder's upsample's block (upstream: PushDecoderUps). */
struct push_decoder_ups {
	STD(uint32_t) src, skip, dst, gain_off, IW, OW, OH;
};

/** @brief The post block's input when it is not its predecessor's raster, upsview (upstream:
 *         PushUpsView). */
struct push_ups_view {
	STD(uint32_t) x_off, o_off, M, C, W, H, RW, RH;
};

/** @brief The noise field's region of the weight blob, noisefield's push block (upstream:
 *         NoiseJob). */
struct noise_job {
	STD(uint32_t) off, width, height, seed;
	float         noise;
};

static_assert(sizeof (struct push_f_swin) == 80 && offsetof(struct push_f_swin, shift) == 60 &&
              offsetof(struct push_f_swin, pool_off) == 68);
static_assert(sizeof (struct push_ds_proj) == 44 && sizeof (struct push_ups) == 40 &&
              sizeof (struct push_image_tail) == 8);
static_assert(sizeof (struct push_pre_image) == 48 && offsetof(struct push_pre_image, style) == 16);
static_assert(sizeof (struct push_persist) == 32 && sizeof (struct persist_rec) == 96 &&
              offsetof(struct persist_rec, gx) == 80);
static_assert(sizeof (struct push_gemm) == 84 && offsetof(struct push_gemm, p_off) == 40 &&
              offsetof(struct push_gemm, remap) == 80);
static_assert(sizeof (struct push_ffwd3) == 28 && sizeof (struct push_attn) == 44 &&
              offsetof(struct push_attn, shift) == 36);
static_assert(sizeof (struct push_v_attn) == 20 && sizeof (struct push_decoder_ups) == 28 &&
              sizeof (struct push_ups_view) == 32);
static_assert(sizeof (struct noise_job) == 20 && offsetof(struct noise_job, noise) == 16);
// The fused blocks fill their ranges: 128 bytes, with the counter word where the kernel takes one.
static_assert(sizeof (struct push_f_swin) + sizeof (struct push_pre_image) == 128);
static_assert(sizeof (struct push_f_swin) + sizeof (struct push_ups) + sizeof (struct push_image_tail) == 128);
static_assert(sizeof (struct push_f_swin) + sizeof (struct push_ds_proj) + 4 == 128);
static_assert(sizeof (struct push_f_swin) + sizeof (struct push_ups) + 4 + 4 == 128);

/** @brief A directory of the model pack (upstream: the tree under --unpacked). */
enum vulkan_directory : STD(uint8_t) {
	VULKAN_DIRECTORY_UNPACKED,
	VULKAN_DIRECTORY_PREBLOCK,
	VULKAN_DIRECTORY_POSTBLOCK,
	VULKAN_DIRECTORY_SPLIT_SWIN,
	VULKAN_DIRECTORY_VIT,
	VULKAN_DIRECTORY_RECORDS,
};

/** @brief What follows a layer's name in the names of its entries; a record has none. */
enum vulkan_suffix : STD(uint8_t) {
	VULKAN_SUFFIX_NONE,
	VULKAN_SUFFIX_MLP_EXPAND,
	VULKAN_SUFFIX_MLP_CONTRACT,
	VULKAN_SUFFIX_MLP_MID,
	VULKAN_SUFFIX_QKV,
	VULKAN_SUFFIX_ATTN_OUT_PROJ,
	VULKAN_SUFFIX_ATTN_POS_BIAS,
	VULKAN_SUFFIX_RESIDUAL_SCALE,
	VULKAN_SUFFIX_ATTN_RESIDUAL_SCALE,
	VULKAN_SUFFIX_SCALARS_B,
	VULKAN_SUFFIX_RESAMPLE,
	VULKAN_SUFFIX_UPSAMPLE_GAIN,
	VULKAN_SUFFIX_INPUT_LIFT,
	VULKAN_SUFFIX_OUT_PROJECT,
	VULKAN_SUFFIX_SKIP_GAIN,
	VULKAN_SUFFIX_MAIN_GAIN,
	VULKAN_SUFFIX_WEIGHT,
	VULKAN_SUFFIX_SKIP_WEIGHT,
	VULKAN_SUFFIX_TAIL,
};

/** @brief A model entry: one of a layer's weights, or the layer's record. */
struct vulkan_source {
	enum vulkan_directory directory; //!< Its directory.
	STD(uint8_t)          block;     //!< The layer's block.
	STD(uint8_t)          layer;     //!< The layer in its block.
	enum vulkan_suffix    suffix;    //!< The weight; none for the record.
};

/** @brief How a segment of the weight blob is made from its entry.
 *
 * Each recipe is one of the put() paths of upstream's NrSession::build (nr_graph.cpp:1700-3259,
 * 3525-3600 and 3966-4221) as the network takes it.
 */
enum vulkan_recipe : STD(uint8_t) {
	/** @brief Zeros: the noise field that the build fills on the GPU, and the post block's unread
	 *         gain. */
	VULKAN_RECIPE_ZEROS,
	VULKAN_RECIPE_TABLE,       //!< Words of the plan's tables, from word index on.
	VULKAN_RECIPE_ACTIVATIONS, //!< The 4096-byte activation table (activation_lut_v1).
	VULKAN_RECIPE_MATRIX,      //!< rows x cols E4M3 codes, tile-blocked.
	/** @brief Part index (0 A, 1 Q0, 2 Q2) of a C=512 FFN record, gathered and tile-blocked. */
	VULKAN_RECIPE_FFWD,
	VULKAN_RECIPE_VIT_QKV,     //!< The 3072 x 1024 weight of a ViT QKV record, gathered and tile-blocked.
	/** @brief Position biases, 4096 binary16 values a head in MMA C-fragment order, as f32 in [i][j]
	 *         order. */
	VULKAN_RECIPE_BIAS,
	VULKAN_RECIPE_SCALES,      //!< Binary16 values widened to f32.
	VULKAN_RECIPE_HALF,        //!< Binary16 values through upstream's f32 round trip.
	VULKAN_RECIPE_DIAGONAL,    //!< 16x16 binary16 blocks with the values on their diagonals.
	VULKAN_RECIPE_BYTES,       //!< The entry's first bytes.
	VULKAN_RECIPE_LIFT,        //!< The input lift, 32 x 16 binary16, in [channel][k] order.
};

/** @brief What a recipe does besides, in a segment's flags. */
enum vulkan_segment_flag : STD(uint8_t) {
	VULKAN_SEGMENT_REQUANTISE = 1,  //!< MATRIX: each E4M3 code through upstream's round trip.
	/** @brief MATRIX, FFWD, VIT_QKV: pairs of 16-row tiles interleaved in 8-byte runs (weight
	 *         layout 3). */
	VULKAN_SEGMENT_NPAIR      = 2,
	VULKAN_SEGMENT_AFFINE     = 4,  //!< BIAS: math profile 3's exponent affine folded in.
	/** @brief SCALES, DIAGONAL: the values from the ninth on when the first eight are zero. */
	VULKAN_SEGMENT_PADDED     = 8,
	/** @brief HALF: an entry short of values is completed in front by the last values of its
	 *         layer's residual_scale. */
	VULKAN_SEGMENT_GAIN_TAIL  = 16,
	VULKAN_SEGMENT_EXACT      = 32, //!< The entry holds exactly the bytes the recipe reads.
};

/** @brief Bytes of the weight blob, made of a source's entry by a recipe.
 *
 * Counts of values follow from the bytes: 16384 bytes a head for BIAS, 4 a value for SCALES, 2 for
 * HALF and 32 for DIAGONAL.
 */
struct vulkan_segment {
	STD(uint32_t)         offset; //!< Where in the blob it starts.
	STD(uint32_t)         bytes;  //!< Its bytes.
	STD(uint32_t)         index;  //!< TABLE's first word; FFWD's part.
	STD(uint16_t)         rows;   //!< MATRIX's and VIT_QKV's rows.
	STD(uint16_t)         cols;   //!< Their columns.
	struct vulkan_source  source; //!< The entry it is made of.
	enum vulkan_recipe    recipe; //!< How.
	STD(uint8_t)          flags;  //!< enum vulkan_segment_flag bits.
};

/** @brief What orders a step before the next one. */
enum vulkan_after : STD(uint8_t) {
	VULKAN_AFTER_NOTHING,    //!< Nothing, where tile counters do.
	/** @brief A barrier that makes the arena's writes visible to the next step (upstream: compute
	 *         to compute, source access 0, destination shader reads and writes, which gfx1201's
	 *         write-through caches make enough). */
	VULKAN_AFTER_INVALIDATE,
	VULKAN_AFTER_FULL,       //!< After the last step, a full one.
};

/** @brief A dispatch of the network.
 *
 * It runs kernel over groups with words push words from the plan's push words at push on. It runs
 * the network's layers first to last, by their place in the layer table, starting with layer
 * layer of block block.
 */
struct vulkan_step {
	STD(uint32_t)      groups[3]; //!< Its workgroups.
	STD(uint16_t)      push;      //!< Its first push word.
	enum vulkan_kernel kernel;    //!< Its kernel.
	enum vulkan_after  after;     //!< What orders it before the next step.
	STD(uint8_t)       words;     //!< Its push words.
	STD(uint8_t)       block;     //!< The block of its first layer.
	STD(uint8_t)       layer;     //!< Its first layer in that block.
	STD(uint8_t)       first;     //!< Its first layer in the layer table.
	STD(uint8_t)       last;      //!< Its last.
};

/** @brief A value in the activation arena: its bytes and those its readers reach past them, which
 *         stay zero (upstream: voff, vsize and overread).
 *
 * Its key is 8 * block + slot, the slot a layer's output (its layer), or the block's input lift or
 * upsampled input (5), pooled output (6) or skip output (7).
 */
struct vulkan_value {
	STD(uint64_t) offset;   //!< Where it is.
	STD(uint64_t) bytes;    //!< Its bytes.
	STD(uint64_t) overread; //!< The bytes its readers reach.
	STD(uint64_t) key;      //!< Its key.
};

/** @brief The network at one frame extent, which the working extent pads.
 *
 * The weight blob is blob_bytes in all: the segments in offset order, the gaps between them zero.
 *
 * The activation arena is zeroed at build: the values up to values_end, shared by lifetime; the
 * persistent runs' sync regions; counter_words words of tile counters. Zeroed again,
 * [values_end, arena_bytes) starts the sync regions and counters over as the build left them.
 *
 * The timeouts are the arena's u32 words that a frame's waits set when they give up: the one that
 * every record names, which every wait sets and reads, then each persistent run's; nothing else
 * writes them. A wait that runs out leaves its frame wrong; one of a persistent run also leaves its
 * sync region and the counters it signals short of their counts.
 *
 * Zeroed, it is no plan: a plan's width is never 0.
 */
struct vulkan_plan {
	struct vulkan_step    *steps;         //!< The dispatches.
	STD(uint32_t)         *push;          //!< Their push words.
	struct vulkan_segment *segments;      //!< The weight blob's segments in offset order.
	STD(uint32_t)         *tables;        //!< The words that TABLE segments copy.
	struct vulkan_value   *values;        //!< The activation arena's values.
	STD(uint32_t)         *timeouts;      //!< The arena's words that a frame's waits set when they give up.
	STD(size_t)            step_count;    //!< The number of steps.
	STD(size_t)            push_count;    //!< The number of push words.
	STD(size_t)            segment_count; //!< The number of segments.
	STD(size_t)            table_count;   //!< The number of table words.
	STD(size_t)            value_count;   //!< The number of values.
	STD(size_t)            timeout_count; //!< The number of timeouts.
	STD(uint64_t)          values_end;    //!< Where the values end in the arena.
	STD(uint64_t)          arena_bytes;   //!< The arena's bytes.
	STD(uint32_t)          width;         //!< The frames' width.
	STD(uint32_t)          height;        //!< Their height.
	STD(uint32_t)          work_width;    //!< The working extent's width.
	STD(uint32_t)          work_height;   //!< Its height.
	STD(uint32_t)          blob_bytes;    //!< The weight blob's bytes.
	STD(uint32_t)          counter_words; //!< The tile counters' words after the sync regions.
	struct noise_job       noise;         //!< The noise field, which the build writes on the GPU.
	STD(uint32_t)          chained;       //!< The steps that tile counters order instead of a barrier.
};

/** @brief Each block's Swin heads, bit h for head h, whose attention never reaches the upper clamp
 *         of its exponent (upstream: g_nohi_heads), as the model's weights bound it.
 *
 * A head is free when every baked position bias of the head, plus 0.044921875 times 1.2 |s| + 0.05
 * with s the head's scale, is at most 1.5693359375. vulkan_weights_clamp_free() audits a model.
 */
struct vulkan_clamp_free {
	STD(uint32_t) heads[VULKAN_PLAN_BLOCKS]; //!< The free heads by block.
};

/** @brief Plans the network for frames of an extent.
 *
 * A frame that the network cannot take is rejected: a side of 0 or above 16384, a working extent
 * whose sides are not multiples of 8 (a side of 16 pixels or less, where upstream fails), one whose
 * arena or tile counters overflow the 32-bit offsets and indices of the push constants (upstream
 * wrapped them), and one whose arena or weights exceed the device's storage buffers (upstream did
 * not check). The arena's size is known only once the network is lowered, so a rejection costs as
 * much as a plan: a caller keeps the plan it checked an extent with, and remembers an extent that
 * was rejected.
 *
 * @param dest    Receives the plan, which vulkan_plan_fini() frees; zeroed on a failure.
 * @param width   The frames' width.
 * @param height  Their height.
 * @param storage The most bytes one of the device's storage buffers holds; UINT64_MAX for no limit.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE; ERROR_REJECTED for a frame the network cannot take; ERROR_FAILED.
 */
extern enum error_code
vulkan_plan_init (struct vulkan_plan *dest,
                  STD(uint32_t)       width,
                  STD(uint32_t)       height,
                  STD(uint64_t)       storage,
                  struct error       *e);

/** @brief Frees what a plan holds and zeroes it.
 *
 * @param plan The plan, or nullptr.
 */
extern void
vulkan_plan_fini (struct vulkan_plan *plan);

/** @brief Takes the kernels without the upper clamp where a model's weights allow it (upstream:
 *         NR_EXP_NOHI).
 *
 * The C=32 steps whose block's one head is free take their VULKAN_PLAN_UNCLAMPED twins, and the
 * records of a persistent run's layers carry their blocks' free heads in their window counts, which
 * the kernels do not read. fswinfusedup32nh also leaves out the tests of its tiles: the fused
 * upsample's grid is unshifted and covers its tile raster whole at every extent, which upstream
 * checks before it takes that twin.
 *
 * @param plan       The plan.
 * @param clamp_free The heads that the model frees.
 */
extern void
vulkan_plan_unclamp (struct vulkan_plan             *plan,
                     struct vulkan_clamp_free const *clamp_free);

/** @brief Whether a kernel runs a persistent run of Swin layers.
 *
 * @param kernel The kernel.
 * @return       true for fswinpds64 to fswinpup256.
 */
extern bool
vulkan_plan_persistent (enum vulkan_kernel kernel);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_VULKAN_PLAN_H_ */
