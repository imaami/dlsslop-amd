/** @file
 *
 * The Vulkan network's plan: vulkan_plan.h. The layer table and the walk that places it on the
 * working extent, the lowering of the layers into dispatches and of their weights into the blob's
 * segments, and the plan that vulkan_schedule.c's schedule makes of them.
 */
// SPDX-License-Identifier: MIT
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "vulkan_plan.h"
#include "vulkan_plan_priv.h"

char const VULKAN_PLAN_MANIFEST[] = "nr_shader_manifest v3";

struct vulkan_shader_constant const VULKAN_PLAN_SHADER_CONSTANTS[] = {
	{"weight_layout", 3},
	{"math_profile", 3},
	{"vattn_qt", VULKAN_PLAN_VATTN_QT},
	{"gemm_wide_mt", VULKAN_PLAN_GEMM_WIDE_MT},
	{"gemm_wide_nt", VULKAN_PLAN_GEMM_WIDE_NT},
	{"qkv_fused_norm", 1},
	{"ups_fused_mode", 2},
	{"wide_ups_fused_mode", 2},
	{"gemm_proj_mt", VULKAN_PLAN_GEMM_PROJ_MT},
	{"gemm_proj_nt", VULKAN_PLAN_GEMM_PROJ_NT},
	{"gemm_projw_mt", VULKAN_PLAN_GEMM_PROJW_MT},
	{"gemm_projw_nt", VULKAN_PLAN_GEMM_PROJW_NT},
	{"ffwd_wgw", VULKAN_PLAN_FFWD_WGW},
	{"ffwd_gmajor", 1},
	{"gemm_remap", 1},
	{"upsview_vec", 1},
	{"repack_vec", 1},
	{"wide_ups_mask", 7},
	{"persist_df", 1},
	{"ds_fuse", 15},
	{"decups_vec", VULKAN_PLAN_DECUPS_VEC},
	{"persist_ds", 7},
	{"persist_up", 7},
	{"persist_strag", VULKAN_PLAN_STRAGGLER_PERCENT},
	{"persist_one", VULKAN_PLAN_PERSIST_ONE_MASK},
	{"noise_field", 1},
	{"ffwd_fm2_min", VULKAN_PLAN_FFWD_FM2_MIN_TOKENS},
	{"post_alpha", 1},
	{"tchain", 1},
};

static_assert(sizeof VULKAN_PLAN_SHADER_CONSTANTS / sizeof *VULKAN_PLAN_SHADER_CONSTANTS
              == VULKAN_PLAN_SHADER_CONSTANT_COUNT,
              "VULKAN_PLAN_SHADER_CONSTANT_COUNT is not the number of VULKAN_PLAN_SHADER_CONSTANTS");

/** @brief A marker of a file named by a literal and of a literal line, with their lengths. */
#define MARKER(file, line) {file, line, sizeof (file) - 1, sizeof (line) - 1}

struct vulkan_marker const VULKAN_PLAN_MARKERS[] = {
	MARKER("accumulation.txt", "fp32"),
	MARKER("coherent-act.txt", "0"),
	MARKER("swin-bias-storage.txt", "fp32"),
};

#undef MARKER

static_assert(sizeof VULKAN_PLAN_MARKERS / sizeof *VULKAN_PLAN_MARKERS == VULKAN_PLAN_MARKER_COUNT,
              "VULKAN_PLAN_MARKER_COUNT is not the number of VULKAN_PLAN_MARKERS");

struct vulkan_kernel_info const VULKAN_PLAN_KERNELS[] = {
	[VULKAN_KERNEL_FSWIN32]              = {"fswin32", "aawwwa", VULKAN_IMAGES_NONE, 84},
	[VULKAN_KERNEL_FSWIN_IMAGE_PREDS32]  = {"fswinimagepreds32", "aawwwa", VULKAN_IMAGES_INPUT, 128},
	[VULKAN_KERNEL_FSWIN_DSP32]          = {"fswindsp32", "aawwwa", VULKAN_IMAGES_NONE, 128},
	[VULKAN_KERNEL_FSWIN_PDS64]          = {"fswinpds64", "aawwwa", VULKAN_IMAGES_NONE, 36},
	[VULKAN_KERNEL_FSWIN_PDS128]         = {"fswinpds128", "aawwwa", VULKAN_IMAGES_NONE, 36},
	[VULKAN_KERNEL_FSWIN_PDS256]         = {"fswinpds256", "aawwwa", VULKAN_IMAGES_NONE, 36},
	[VULKAN_KERNEL_FSWIN_PUP64]          = {"fswinpup64", "aawwwa", VULKAN_IMAGES_NONE, 36},
	[VULKAN_KERNEL_FSWIN_PUP128]         = {"fswinpup128", "aawwwa", VULKAN_IMAGES_NONE, 36},
	[VULKAN_KERNEL_FSWIN_PUP256]         = {"fswinpup256", "aawwwa", VULKAN_IMAGES_NONE, 36},
	[VULKAN_KERNEL_FSWIN_FUSED_UP32]     = {"fswinfusedup32", "aawww", VULKAN_IMAGES_NONE, 128},
	[VULKAN_KERNEL_FSWIN_IMAGE_POST32]   = {"fswinimagepost32", "aawww", VULKAN_IMAGES_OUTPUT, 128},
	[VULKAN_KERNEL_FSWIN32_NH]           = {"fswin32nh", "aawwwa", VULKAN_IMAGES_NONE, 84},
	[VULKAN_KERNEL_FSWIN_IMAGE_PREDS32_NH] = {"fswinimagepreds32nh", "aawwwa", VULKAN_IMAGES_INPUT, 128},
	[VULKAN_KERNEL_FSWIN_DSP32_NH]       = {"fswindsp32nh", "aawwwa", VULKAN_IMAGES_NONE, 128},
	[VULKAN_KERNEL_FSWIN_FUSED_UP32_NH]  = {"fswinfusedup32nh", "aawww", VULKAN_IMAGES_NONE, 128},
	[VULKAN_KERNEL_FFWD3]                = {"ffwd3", "aaawwa", VULKAN_IMAGES_NONE, 32},
	[VULKAN_KERNEL_FFWD3W]               = {"ffwd3w", "aaawwa", VULKAN_IMAGES_NONE, 32},
	[VULKAN_KERNEL_ATTN]                 = {"attn", "aawwa", VULKAN_IMAGES_NONE, 48},
	[VULKAN_KERNEL_GEMM_PROJC]           = {"gemmprojc", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_POOL]            = {"gemmpool", "aaawwawa", VULKAN_IMAGES_NONE, 84},
	[VULKAN_KERNEL_GEMM_NORES]           = {"gemmnores", "aaawwawa", VULKAN_IMAGES_NONE, 84},
	[VULKAN_KERNEL_GEMM_VACT]            = {"gemmvact", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_PROJ]            = {"gemmproj", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_PROJW]           = {"gemmprojw", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_PROJT]           = {"gemmprojt", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_VQKV_NORM]       = {"gemmvqkvnorm", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_VQKV_NORMS]      = {"gemmvqkvnorms", "aaawwawa", VULKAN_IMAGES_NONE, 88},
	[VULKAN_KERNEL_GEMM_VQKVS]           = {"gemmvqkvs", "aaawwawa", VULKAN_IMAGES_NONE, 84},
	[VULKAN_KERNEL_VIT_ATTN]             = {"vitattn", "awa", VULKAN_IMAGES_NONE, 24},
	[VULKAN_KERNEL_DEC_UPS]              = {"decups", "aaw", VULKAN_IMAGES_NONE, 28},
	[VULKAN_KERNEL_UPS_VIEW]             = {"upsview", "a", VULKAN_IMAGES_NONE, 32},
	[VULKAN_KERNEL_NOISE_FIELD]          = {"noisefield", "w", VULKAN_IMAGES_NONE, 20},
};

static_assert(sizeof VULKAN_PLAN_KERNELS / sizeof *VULKAN_PLAN_KERNELS == VULKAN_KERNEL_COUNT,
              "VULKAN_KERNEL_COUNT is not the number of VULKAN_PLAN_KERNELS");

struct vulkan_unclamped const VULKAN_PLAN_UNCLAMPED[] = {
	{VULKAN_KERNEL_FSWIN32, VULKAN_KERNEL_FSWIN32_NH},
	{VULKAN_KERNEL_FSWIN_IMAGE_PREDS32, VULKAN_KERNEL_FSWIN_IMAGE_PREDS32_NH},
	{VULKAN_KERNEL_FSWIN_DSP32, VULKAN_KERNEL_FSWIN_DSP32_NH},
	{VULKAN_KERNEL_FSWIN_FUSED_UP32, VULKAN_KERNEL_FSWIN_FUSED_UP32_NH},
};

static_assert(sizeof VULKAN_PLAN_UNCLAMPED / sizeof *VULKAN_PLAN_UNCLAMPED == VULKAN_PLAN_UNCLAMPED_COUNT,
              "VULKAN_PLAN_UNCLAMPED_COUNT is not the number of VULKAN_PLAN_UNCLAMPED");

bool
vulkan_plan_persistent (enum vulkan_kernel kernel)
{
	return kernel >= VULKAN_KERNEL_FSWIN_PDS64 && kernel <= VULKAN_KERNEL_FSWIN_PUP256;
}

/** @brief What a layer computes (upstream: its type column, and the "_ds_" and "_upsample" of its
 *         kernel column).
 *
 * A Swin block, one that downsamples after it and one that upsamples before it; the pre and post
 * blocks; the C=512 split Swin's FFN, its projection, attention and output projections, the last of
 * them pooled, and the head after them; the ViT's layers; and the decoder's input.
 */
enum family : uint8_t {
	FAMILY_SWIN,
	FAMILY_SWIN_DS,
	FAMILY_SWIN_UP,
	FAMILY_PRE,
	FAMILY_POST,
	FAMILY_FFWD,
	FAMILY_FFWD_PROJ,
	FAMILY_QKV_ATTN,
	FAMILY_PROJ,
	FAMILY_PROJ_POOL,
	FAMILY_FINAL_HEAD,
	FAMILY_VIT_EXPAND,
	FAMILY_VIT_CONTRACT,
	FAMILY_VIT_QKV,
	FAMILY_VIT_ATTENTION,
	FAMILY_VIT_PROJECTION,
	FAMILY_DECODER_UP,
};

/** @brief A layer of the network (upstream: the records of nr_native_plan_data.hpp, from the frame
 *         plan and layer descriptor of nvngx_dlssnr 310.8.0). */
struct row {
	uint16_t    c;        //!< Its width C.
	uint16_t    ci;       //!< Its input channels.
	uint16_t    co;       //!< Its output channels.
	uint8_t     block;    //!< Its block.
	uint8_t     layer;    //!< Its layer in the block.
	enum family family;   //!< What it computes.
	uint8_t     heads;    //!< Its heads.
	int8_t      in0;      //!< The block it reads, -1 for none.
	int8_t      in1;      //!< The other block it reads, -1 for none.
	int8_t      skip;     //!< For an upsampling layer, the row whose extent it takes.
	int8_t      sx;       //!< Its window shift in tiles along x,
	int8_t      sy;       //!< and along y.
	bool        original; //!< Whether it takes skip's extent as it was before that row ran.
};

/** @brief The layer table's rows. */
#define LAYERS 152

/** @brief A row in the layer table's column order. */
#define ROW(block_, layer_, family_, c_, heads_, ci_, co_, in0_, in1_, skip_, original_, sx_, sy_) \
	{.c = c_, .ci = ci_, .co = co_, .block = block_, .layer = layer_, .family = family_, .heads = heads_, \
	 .in0 = in0_, .in1 = in1_, .skip = skip_, .sx = sx_, .sy = sy_, .original = original_}

/** @brief The network's layers in order. */
static struct row const ROWS[] = {
	ROW(0, 0, FAMILY_PRE, 3, 0, 3, 32, -1, -1, -1, false, 0, 0),
	ROW(1, 0, FAMILY_SWIN, 32, 1, 32, 32, 0, -1, -1, false, 0, 0),
	ROW(2, 0, FAMILY_SWIN, 32, 1, 32, 32, 1, -1, -1, false, -1, -1),
	ROW(3, 0, FAMILY_SWIN, 32, 1, 32, 32, 2, -1, -1, false, -1, 0),
	ROW(4, 0, FAMILY_SWIN_DS, 32, 1, 32, 64, 3, -1, -1, false, 0, -1),
	ROW(5, 0, FAMILY_SWIN, 64, 2, 64, 64, 4, -1, -1, false, 0, 0),
	ROW(6, 0, FAMILY_SWIN, 64, 2, 64, 64, 5, -1, -1, false, -1, -1),
	ROW(7, 0, FAMILY_SWIN, 64, 2, 64, 64, 6, -1, -1, false, -1, 0),
	ROW(8, 0, FAMILY_SWIN_DS, 64, 2, 64, 128, 7, -1, -1, false, 0, -1),
	ROW(9, 0, FAMILY_SWIN, 128, 4, 128, 128, 8, -1, -1, false, 0, 0),
	ROW(10, 0, FAMILY_SWIN, 128, 4, 128, 128, 9, -1, -1, false, -1, -1),
	ROW(11, 0, FAMILY_SWIN, 128, 4, 128, 128, 10, -1, -1, false, -1, 0),
	ROW(12, 0, FAMILY_SWIN, 128, 4, 128, 128, 11, -1, -1, false, 0, -1),
	ROW(13, 0, FAMILY_SWIN, 128, 4, 128, 128, 12, -1, -1, false, 0, 0),
	ROW(14, 0, FAMILY_SWIN_DS, 128, 4, 128, 256, 13, -1, -1, false, -1, -1),
	ROW(15, 0, FAMILY_SWIN, 256, 8, 256, 256, 14, -1, -1, false, 0, 0),
	ROW(16, 0, FAMILY_SWIN, 256, 8, 256, 256, 15, -1, -1, false, -1, -1),
	ROW(17, 0, FAMILY_SWIN, 256, 8, 256, 256, 16, -1, -1, false, -1, 0),
	ROW(18, 0, FAMILY_SWIN, 256, 8, 256, 256, 17, -1, -1, false, 0, -1),
	ROW(19, 0, FAMILY_SWIN, 256, 8, 256, 256, 18, -1, -1, false, 0, 0),
	ROW(20, 0, FAMILY_SWIN, 256, 8, 256, 256, 19, -1, -1, false, -1, -1),
	ROW(21, 0, FAMILY_SWIN, 256, 8, 256, 256, 20, -1, -1, false, -1, 0),
	ROW(22, 0, FAMILY_SWIN_DS, 256, 8, 256, 512, 21, -1, -1, false, 0, -1),
	ROW(23, 0, FAMILY_FFWD, 512, 4, 512, 512, 22, -1, -1, false, 0, 0),
	ROW(23, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 23, -1, -1, false, 0, 0),
	ROW(23, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 23, -1, -1, false, 0, 0),
	ROW(23, 3, FAMILY_PROJ, 512, 8, 512, 512, 23, -1, -1, false, 0, 0),
	ROW(24, 0, FAMILY_FFWD, 512, 4, 512, 512, 23, -1, -1, false, 0, 0),
	ROW(24, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 24, -1, -1, false, 0, 0),
	ROW(24, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 24, -1, -1, false, -1, -1),
	ROW(24, 3, FAMILY_PROJ, 512, 8, 512, 512, 24, -1, -1, false, 0, 0),
	ROW(25, 0, FAMILY_FFWD, 512, 4, 512, 512, 24, -1, -1, false, 0, 0),
	ROW(25, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 25, -1, -1, false, 0, 0),
	ROW(25, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 25, -1, -1, false, -1, 0),
	ROW(25, 3, FAMILY_PROJ, 512, 8, 512, 512, 25, -1, -1, false, 0, 0),
	ROW(26, 0, FAMILY_FFWD, 512, 4, 512, 512, 25, -1, -1, false, 0, 0),
	ROW(26, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 26, -1, -1, false, 0, 0),
	ROW(26, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 26, -1, -1, false, 0, -1),
	ROW(26, 3, FAMILY_PROJ, 512, 8, 512, 512, 26, -1, -1, false, 0, 0),
	ROW(27, 0, FAMILY_FFWD, 512, 4, 512, 512, 26, -1, -1, false, 0, 0),
	ROW(27, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 27, -1, -1, false, 0, 0),
	ROW(27, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 27, -1, -1, false, 0, 0),
	ROW(27, 3, FAMILY_PROJ, 512, 8, 512, 512, 27, -1, -1, false, 0, 0),
	ROW(28, 0, FAMILY_FFWD, 512, 4, 512, 512, 27, -1, -1, false, 0, 0),
	ROW(28, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 28, -1, -1, false, 0, 0),
	ROW(28, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 28, -1, -1, false, -1, -1),
	ROW(28, 3, FAMILY_PROJ, 512, 8, 512, 512, 28, -1, -1, false, 0, 0),
	ROW(29, 0, FAMILY_FFWD, 512, 4, 512, 512, 28, -1, -1, false, 0, 0),
	ROW(29, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 29, -1, -1, false, 0, 0),
	ROW(29, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 29, -1, -1, false, -1, 0),
	ROW(29, 3, FAMILY_PROJ, 512, 8, 512, 512, 29, -1, -1, false, 0, 0),
	ROW(30, 0, FAMILY_FFWD, 512, 4, 512, 512, 29, -1, -1, false, 0, 0),
	ROW(30, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 30, -1, -1, false, 0, 0),
	ROW(30, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 30, -1, -1, false, 0, -1),
	ROW(30, 3, FAMILY_PROJ_POOL, 512, 4, 512, 512, 30, -1, -1, false, 0, 0),
	ROW(30, 4, FAMILY_FINAL_HEAD, 512, 8, 512, 1024, 30, -1, -1, false, 0, 0),
	ROW(31, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 30, -1, -1, false, 0, 0),
	ROW(31, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 31, -1, -1, false, 0, 0),
	ROW(31, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 31, -1, -1, false, 0, 0),
	ROW(31, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 31, -1, -1, false, 0, 0),
	ROW(31, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 31, -1, -1, false, 0, 0),
	ROW(32, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 31, -1, -1, false, 0, 0),
	ROW(32, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 32, -1, -1, false, 0, 0),
	ROW(32, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 32, -1, -1, false, 0, 0),
	ROW(32, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 32, -1, -1, false, 0, 0),
	ROW(32, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 32, -1, -1, false, 0, 0),
	ROW(33, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 32, -1, -1, false, 0, 0),
	ROW(33, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 33, -1, -1, false, 0, 0),
	ROW(33, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 33, -1, -1, false, 0, 0),
	ROW(33, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 33, -1, -1, false, 0, 0),
	ROW(33, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 33, -1, -1, false, 0, 0),
	ROW(34, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 33, -1, -1, false, 0, 0),
	ROW(34, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 34, -1, -1, false, 0, 0),
	ROW(34, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 34, -1, -1, false, 0, 0),
	ROW(34, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 34, -1, -1, false, 0, 0),
	ROW(34, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 34, -1, -1, false, 0, 0),
	ROW(35, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 34, -1, -1, false, 0, 0),
	ROW(35, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 35, -1, -1, false, 0, 0),
	ROW(35, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 35, -1, -1, false, 0, 0),
	ROW(35, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 35, -1, -1, false, 0, 0),
	ROW(35, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 35, -1, -1, false, 0, 0),
	ROW(36, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 35, -1, -1, false, 0, 0),
	ROW(36, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 36, -1, -1, false, 0, 0),
	ROW(36, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 36, -1, -1, false, 0, 0),
	ROW(36, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 36, -1, -1, false, 0, 0),
	ROW(36, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 36, -1, -1, false, 0, 0),
	ROW(37, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 36, -1, -1, false, 0, 0),
	ROW(37, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 37, -1, -1, false, 0, 0),
	ROW(37, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 37, -1, -1, false, 0, 0),
	ROW(37, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 37, -1, -1, false, 0, 0),
	ROW(37, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 37, -1, -1, false, 0, 0),
	ROW(38, 0, FAMILY_VIT_EXPAND, 1024, 4, 1024, 4096, 37, -1, -1, false, 0, 0),
	ROW(38, 1, FAMILY_VIT_CONTRACT, 1024, 4, 4096, 1024, 38, -1, -1, false, 0, 0),
	ROW(38, 2, FAMILY_VIT_QKV, 1024, 4, 1024, 1024, 38, -1, -1, false, 0, 0),
	ROW(38, 3, FAMILY_VIT_ATTENTION, 1024, 4, 1024, 1024, 38, -1, -1, false, 0, 0),
	ROW(38, 4, FAMILY_VIT_PROJECTION, 1024, 4, 1024, 1024, 38, -1, -1, false, 0, 0),
	ROW(39, 0, FAMILY_DECODER_UP, 512, 1, 1024, 512, 38, 30, 54, false, 0, 0),
	ROW(40, 0, FAMILY_FFWD, 512, 4, 512, 512, 39, -1, -1, false, 0, 0),
	ROW(40, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 40, -1, -1, false, 0, 0),
	ROW(40, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 40, -1, -1, false, 0, 0),
	ROW(40, 3, FAMILY_PROJ, 512, 8, 512, 512, 40, -1, -1, false, 0, 0),
	ROW(41, 0, FAMILY_FFWD, 512, 4, 512, 512, 40, -1, -1, false, 0, 0),
	ROW(41, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 41, -1, -1, false, 0, 0),
	ROW(41, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 41, -1, -1, false, -1, -1),
	ROW(41, 3, FAMILY_PROJ, 512, 8, 512, 512, 41, -1, -1, false, 0, 0),
	ROW(42, 0, FAMILY_FFWD, 512, 4, 512, 512, 41, -1, -1, false, 0, 0),
	ROW(42, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 42, -1, -1, false, 0, 0),
	ROW(42, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 42, -1, -1, false, -1, 0),
	ROW(42, 3, FAMILY_PROJ, 512, 8, 512, 512, 42, -1, -1, false, 0, 0),
	ROW(43, 0, FAMILY_FFWD, 512, 4, 512, 512, 42, -1, -1, false, 0, 0),
	ROW(43, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 43, -1, -1, false, 0, 0),
	ROW(43, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 43, -1, -1, false, 0, -1),
	ROW(43, 3, FAMILY_PROJ, 512, 8, 512, 512, 43, -1, -1, false, 0, 0),
	ROW(44, 0, FAMILY_FFWD, 512, 4, 512, 512, 43, -1, -1, false, 0, 0),
	ROW(44, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 44, -1, -1, false, 0, 0),
	ROW(44, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 44, -1, -1, false, 0, 0),
	ROW(44, 3, FAMILY_PROJ, 512, 8, 512, 512, 44, -1, -1, false, 0, 0),
	ROW(45, 0, FAMILY_FFWD, 512, 4, 512, 512, 44, -1, -1, false, 0, 0),
	ROW(45, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 45, -1, -1, false, 0, 0),
	ROW(45, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 45, -1, -1, false, -1, -1),
	ROW(45, 3, FAMILY_PROJ, 512, 8, 512, 512, 45, -1, -1, false, 0, 0),
	ROW(46, 0, FAMILY_FFWD, 512, 4, 512, 512, 45, -1, -1, false, 0, 0),
	ROW(46, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 46, -1, -1, false, 0, 0),
	ROW(46, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 46, -1, -1, false, -1, 0),
	ROW(46, 3, FAMILY_PROJ, 512, 8, 512, 512, 46, -1, -1, false, 0, 0),
	ROW(47, 0, FAMILY_FFWD, 512, 4, 512, 512, 46, -1, -1, false, 0, 0),
	ROW(47, 1, FAMILY_FFWD_PROJ, 512, 4, 512, 512, 47, -1, -1, false, 0, 0),
	ROW(47, 2, FAMILY_QKV_ATTN, 512, 4, 512, 512, 47, -1, -1, false, 0, -1),
	ROW(47, 3, FAMILY_PROJ, 512, 4, 512, 512, 47, -1, -1, false, 0, 0),
	ROW(48, 0, FAMILY_SWIN_UP, 256, 8, 512, 256, 47, 22, 22, true, 0, 0),
	ROW(49, 0, FAMILY_SWIN, 256, 8, 256, 256, 48, -1, -1, false, -1, -1),
	ROW(50, 0, FAMILY_SWIN, 256, 8, 256, 256, 49, -1, -1, false, -1, 0),
	ROW(51, 0, FAMILY_SWIN, 256, 8, 256, 256, 50, -1, -1, false, 0, -1),
	ROW(52, 0, FAMILY_SWIN, 256, 8, 256, 256, 51, -1, -1, false, 0, 0),
	ROW(53, 0, FAMILY_SWIN, 256, 8, 256, 256, 52, -1, -1, false, -1, -1),
	ROW(54, 0, FAMILY_SWIN, 256, 8, 256, 256, 53, -1, -1, false, -1, 0),
	ROW(55, 0, FAMILY_SWIN, 256, 8, 256, 256, 54, -1, -1, false, 0, -1),
	ROW(56, 0, FAMILY_SWIN_UP, 128, 4, 256, 128, 55, 14, 14, true, -1, 0),
	ROW(57, 0, FAMILY_SWIN, 128, 4, 128, 128, 56, -1, -1, false, 0, -1),
	ROW(58, 0, FAMILY_SWIN, 128, 4, 128, 128, 57, -1, -1, false, 0, 0),
	ROW(59, 0, FAMILY_SWIN, 128, 4, 128, 128, 58, -1, -1, false, -1, -1),
	ROW(60, 0, FAMILY_SWIN, 128, 4, 128, 128, 59, -1, -1, false, -1, 0),
	ROW(61, 0, FAMILY_SWIN, 128, 4, 128, 128, 60, -1, -1, false, 0, -1),
	ROW(62, 0, FAMILY_SWIN_UP, 64, 2, 128, 64, 61, 8, 8, true, 0, 0),
	ROW(63, 0, FAMILY_SWIN, 64, 2, 64, 64, 62, -1, -1, false, -1, -1),
	ROW(64, 0, FAMILY_SWIN, 64, 2, 64, 64, 63, -1, -1, false, -1, 0),
	ROW(65, 0, FAMILY_SWIN, 64, 2, 64, 64, 64, -1, -1, false, 0, -1),
	ROW(66, 0, FAMILY_SWIN_UP, 32, 1, 64, 32, 65, 4, 4, true, 0, 0),
	ROW(67, 0, FAMILY_SWIN, 32, 1, 32, 32, 66, -1, -1, false, -1, -1),
	ROW(68, 0, FAMILY_SWIN, 32, 1, 32, 32, 67, -1, -1, false, -1, 0),
	ROW(69, 0, FAMILY_SWIN, 32, 1, 32, 32, 68, -1, -1, false, 0, -1),
	ROW(70, 0, FAMILY_POST, 3, 0, 32, 3, 69, 0, -1, false, -1, -1),
};

#undef ROW

static_assert(sizeof ROWS / sizeof *ROWS == LAYERS);

/** @brief A number rounded up to a multiple of an alignment. */
static uint32_t
up (uint32_t n,
    uint32_t alignment)
{
	return (n + alignment - 1) / alignment * alignment;
}

/** @brief A number rounded up to a multiple of an alignment. */
static uint64_t
align (uint64_t n,
       uint64_t alignment)
{
	return (n + alignment - 1) / alignment * alignment;
}

/** @brief A number rounded up to a multiple of 4. */
static uint64_t
pad4 (uint64_t n)
{
	return align(n, 4);
}

/** @brief A number rounded up to a multiple of 8. */
static uint64_t
pad8 (uint64_t n)
{
	return align(n, 8);
}

/** @brief The 4x4-pixel tiles that a row of pixels holds whole (upstream: tiles_of). */
static uint32_t
tiles_of (uint32_t px)
{
	return px / 4;
}

/** @brief A Swin layer's body by family: at C=32 its kernel, which the pre and post blocks replace;
 *         at C>=64 the body of its persistent run. */
static struct {
	enum vulkan_kernel kernel;
	enum vulkan_body   body;
} const SWIN_BODIES[] = {
	[FAMILY_SWIN]    = {VULKAN_KERNEL_FSWIN32, VULKAN_BODY_SWIN},
	[FAMILY_SWIN_DS] = {VULKAN_KERNEL_FSWIN_DSP32, VULKAN_BODY_SWIN_DS},
	[FAMILY_SWIN_UP] = {VULKAN_KERNEL_FSWIN_FUSED_UP32, VULKAN_BODY_SWIN_UP},
	[FAMILY_PRE]     = {VULKAN_KERNEL_FSWIN32, VULKAN_BODY_NONE},
	[FAMILY_POST]    = {VULKAN_KERNEL_FSWIN32, VULKAN_BODY_NONE},
};

/* Keys of the arena's values (upstream: key_of, lift_key, ups_key, pool_key and skip_key): a
 * layer's output, the input lift of the pre block and the ViT or the upsampled input of an
 * upsampling block, a pooled output and the skip output of a downsampling block. */

/** @brief The key of a layer's output. */
static size_t
key (size_t block,
     size_t layer)
{
	return 8 * block + layer;
}

/** @brief The key of a block's input lift. */
static size_t
lift (size_t block)
{
	return 8 * block + 5;
}

/** @brief The key of a block's upsampled input. */
static size_t
ups (size_t block)
{
	return 8 * block + 5;
}

/** @brief The key of a block's pooled output. */
static size_t
pool (size_t block)
{
	return 8 * block + 6;
}

/** @brief The key of a block's skip output. */
static size_t
skip (size_t block)
{
	return 8 * block + 7;
}

/** @brief A layer where the walk over the working extent puts it (upstream: the plan row that
 *         make_native_plan writes).
 *
 * W and H keep upstream's order: W is the level's height and H its width.
 */
struct layer {
	uint64_t   tokens; //!< Its tokens.
	uint32_t   W;      //!< The level's height.
	uint32_t   H;      //!< Its width.
	uint32_t   gx;     //!< Its window grid's width,
	uint32_t   gy;     //!< and height.
	struct row row;    //!< Its row.
};

/** @brief The network over a working extent, and what the lowering asks of its layer table
 *         (upstream: plan, last_layer and reader_tokens). */
struct network {
	struct layer layers[LAYERS];                    //!< Its layers.
	uint64_t     reader_tokens[VULKAN_PLAN_BLOCKS]; //!< Each block's reader's tokens; 0: no other block's.
	uint32_t     width;                             //!< The frames' width.
	uint32_t     height;                            //!< Their height.
	uint32_t     work_width;                        //!< The working extent's width.
	uint32_t     work_height;                       //!< Its height.
	uint8_t      last_layer[VULKAN_PLAN_BLOCKS];    //!< Each block's last layer.
};

/** @brief A layer's lowering's place.
 *
 * @param net   The network.
 * @param block The layer's block.
 * @param layer The layer in that block.
 * @return      Its index in the layer table.
 */
static uint8_t
network_step (struct network const *net,
              uint8_t               block,
              uint8_t               layer)
{
	uint8_t i = 0;
	while (net->layers[i].row.block != block || net->layers[i].row.layer != layer)
		++i;
	return i;
}

/** @brief Where a layer is.
 *
 * @param net   The network.
 * @param block The layer's block.
 * @param layer The layer in that block.
 * @return      The layer.
 */
static struct layer const *
network_at (struct network const *net,
            uint8_t               block,
            uint8_t               layer)
{
	return &net->layers[network_step(net, block, layer)];
}

/** @brief The output of a block's last layer (upstream: blk_out). */
static size_t
network_block_out (struct network const *net,
                   uint8_t               block)
{
	return key(block, net->last_layer[block]);
}

/** @brief An extent in upstream's order. */
struct extent {
	uint64_t w; //!< Upstream's W.
	uint64_t h; //!< Upstream's H.
};

/** @brief The extent that an upsampling layer works at: its encoder skip's (upstream: up_extent),
 *         padded to whole windows at C=32.
 *
 * @param net The network.
 * @param l   The layer.
 * @return    The extent, as upstream's (W, H).
 */
static struct extent
network_up_extent (struct network const *net,
                   struct layer const   *l)
{
	struct layer const *const enc = network_at(net, l->row.in1, 0);
	return l->row.co == 32 ? (struct extent){pad8(enc->W), pad8(enc->H)} : (struct extent){enc->W, enc->H};
}

/** @brief The strict halvings along each axis of the walk over a raster. */
struct halvings {
	uint32_t x; //!< Along x.
	uint32_t y; //!< Along y.
};

/** @brief Upstream's host tensor scalars: w is the height and h the width. */
struct shape {
	uint32_t w; //!< The height.
	uint32_t h; //!< The width.
};

/** @brief The walk down and up the levels from a raster (upstream: walk() in nr_native_plan.cpp).
 *
 * @param width  The raster's width.
 * @param height Its height.
 * @param layers Receives each layer where the walk puts it, or nullptr.
 * @return       The halvings.
 */
static struct halvings
walk (uint32_t      width,
      uint32_t      height,
      struct layer *layers)
{
	struct shape current = {height, width};
	struct shape previous = current;
	struct shape before[LAYERS];
	struct shape after[LAYERS];
	struct halvings halvings = {0};
	int last_block = -1;
	for (size_t i = 0; i < LAYERS; ++i) {
		struct row const *const r = &ROWS[i];
		if (r->block != last_block) {
			halvings.x += current.h < previous.h;
			halvings.y += current.w < previous.w;
			previous = current;
			last_block = r->block;
		}
		struct shape out = current;
		if (r->family == FAMILY_PRE || r->family == FAMILY_SWIN_DS || r->family == FAMILY_FINAL_HEAD) {
			out = (struct shape){up((current.w + 1) / 2, 4), up((current.h + 1) / 2, 4)};
		} else if (r->family == FAMILY_SWIN_UP || r->family == FAMILY_DECODER_UP) {
			out = r->original ? before[r->skip] : after[r->skip];
			if (r->family == FAMILY_SWIN_UP && r->co == 32)
				out = (struct shape){up(out.w, 8), up(out.h, 8)};
		} else if (r->family == FAMILY_POST) {
			out = (struct shape){height, width};
		}
		if (layers) {
			struct shape plan = current;
			if (r->family == FAMILY_FINAL_HEAD)
				plan = out;
			if (r->family == FAMILY_DECODER_UP)
				plan = (struct shape){up(out.w, 4), up(out.h, 4)};
			if (r->family == FAMILY_POST)
				plan = (struct shape){height / 2, width / 2};
			// The window grid pads the extent for a negative shift.
			bool const grid_out = r->family == FAMILY_POST || r->family == FAMILY_SWIN_UP;
			struct shape const grid = grid_out ? out : plan;
			bool const vit_block = r->block >= 31 && r->block <= 38;
			layers[i] = (struct layer){
				.tokens = vit_block ? (uint64_t)plan.w * plan.h
				                    : (uint64_t)up(plan.w, 8) * up(plan.h, 8),
				.W      = plan.w,
				.H      = plan.h,
				.gx     = (uint32_t)(((int64_t)grid.h - 4 * r->sx + 7) / 8),
				.gy     = (uint32_t)(((int64_t)grid.w - 4 * r->sy + 7) / 8),
				.row    = *r,
			};
		}
		before[i] = current;
		after[i] = out;
		current = out;
	}
	return halvings;
}

/** @brief What a rejected frame's words start with. */
#define REFUSED "the network does not take %" PRIu32 "x%" PRIu32 " frames: "

/** @brief Places the network over the working extent of a frame (upstream: make_native_plan).
 *
 * The working extent is the raster padded to its halvings, at least 320 pixels a side, and one
 * halving's step wider when both sides would divide by four of them.
 *
 * @param width  The frame's width.
 * @param height Its height.
 * @param net    Receives the network.
 * @param e      Receives the words for a frame the network cannot take, or nullptr.
 * @return       ERROR_NONE, or ERROR_REJECTED.
 */
static enum error_code
place (uint32_t        width,
       uint32_t        height,
       struct network *net,
       struct error   *e)
{
	if (!width || !height || width > 16384 || height > 16384)
		return error_reject(e, REFUSED "a side is 0 or above 16384", width, height);
	struct halvings const halvings = walk(width, height, nullptr);
	uint32_t const ax = UINT32_C(1) << halvings.x;
	uint32_t const ay = UINT32_C(1) << halvings.y;
	net->width = width;
	net->height = height;
	uint32_t const work_width = up(width, ax);
	uint32_t const work_height = up(height, ay);
	net->work_width = work_width < 320 ? 320 : work_width;
	net->work_height = work_height < 320 ? 320 : work_height;
	if (!(net->work_height % (4 * ay)) && !(net->work_width % (4 * ax)))
		net->work_width += ax;
	// Upstream's graph needs the input lift fused, which takes whole windows.
	if (net->work_width % 8 || net->work_height % 8)
		return error_reject(e, REFUSED "its working extent %" PRIu32 "x%" PRIu32
		                    " is not a multiple of 8",
		                    width, height, net->work_width, net->work_height);
	walk(net->work_width, net->work_height, net->layers);
	memset(net->last_layer, 0, sizeof net->last_layer);
	memset(net->reader_tokens, 0, sizeof net->reader_tokens);
	for (size_t i = 0; i < LAYERS; ++i) {
		struct layer const *const l = &net->layers[i];
		if (net->last_layer[l->row.block] < l->row.layer)
			net->last_layer[l->row.block] = l->row.layer;
		int const ins[2] = {l->row.in0, l->row.in1};
		for (size_t j = 0; j < 2; ++j) {
			int const in = ins[j];
			if (in >= 0 && in != l->row.block && !net->reader_tokens[in])
				net->reader_tokens[in] = l->tokens;
		}
	}
	return ERROR_NONE;
}

/** @brief The width of a layer's output (upstream: shape_of's N). */
static uint64_t
outputs (struct row const *r)
{
	switch (r->family) {
	case FAMILY_SWIN:
		return r->c;
	case FAMILY_SWIN_DS:
		return 2 * r->c;
	case FAMILY_SWIN_UP:
	case FAMILY_DECODER_UP:
		return r->co;
	case FAMILY_PRE:
	case FAMILY_POST:
		return 32;
	case FAMILY_FFWD:
	case FAMILY_FFWD_PROJ:
	case FAMILY_QKV_ATTN:
	case FAMILY_PROJ:
	case FAMILY_PROJ_POOL:
		return 512;
	case FAMILY_FINAL_HEAD:
	case FAMILY_VIT_CONTRACT:
	case FAMILY_VIT_ATTENTION:
	case FAMILY_VIT_PROJECTION:
		return 1024;
	case FAMILY_VIT_EXPAND:
		return 4096;
	case FAMILY_VIT_QKV:
		return 3072;
	}
	return 0;
}

/** @brief Whether a layer is one of the ViT's. */
static bool
vit (struct row const *r)
{
	return r->family >= FAMILY_VIT_EXPAND && r->family <= FAMILY_VIT_PROJECTION;
}

/** @brief Grows a size to at least some bytes. */
static void
grow (uint64_t *at,
      uint64_t  bytes)
{
	if (*at < bytes)
		*at = bytes;
}

/** @brief Sizes each value, and the bytes read past it (upstream: the vsize and overread tables,
 *         nr_graph.cpp:1284-1395).
 *
 * @param net The network.
 * @param v   The values.
 */
static void
size_values (struct network const *net,
             struct vulkan_values *v)
{
	for (size_t i = 0; i < LAYERS; ++i) {
		struct layer const *const l = &net->layers[i];
		struct row const *const r = &l->row;
		uint8_t const b = r->block;
		size_t const k = key(b, r->layer);
		// A downsampling layer's output is sized by its reader's tokens.
		uint64_t const out_tokens = net->reader_tokens[b] ? net->reader_tokens[b] : l->tokens / 4;
		switch (r->family) {
		case FAMILY_SWIN_UP: {
			struct extent const x = network_up_extent(net, l);
			uint64_t const tokens = pad8(x.w) * pad8(x.h);
			grow(&v->size[pool(b)], l->tokens * r->co * 2);
			grow(&v->size[ups(b)], tokens * r->co * (r->co == 32 ? 2 : 1));
			grow(&v->size[k], tokens * r->co);
			continue;
		}
		case FAMILY_SWIN_DS:
			grow(&v->size[skip(b)], l->tokens * r->c);
			grow(&v->size[pool(b)], l->tokens / 4 * r->c);
			grow(&v->size[k], out_tokens * 2 * r->c);
			// The pooled grid is padded to its reader's, which reads the padding as zero.
			grow(&v->overread[k], v->size[k]);
			continue;
		case FAMILY_PRE:
			grow(&v->size[lift(b)], l->tokens * 32 * 2);
			grow(&v->size[skip(b)], l->tokens * 32);
			grow(&v->size[k], out_tokens * 32);
			continue;
		case FAMILY_POST:
			v->size[pool(b)] = l->tokens * 32;
			grow(&v->size[ups(b)], l->tokens * 4 * 32 * 2);
			grow(&v->size[k], l->tokens * 4 * 32 * 2);
			break;
		case FAMILY_DECODER_UP: {
			struct layer const *const prev = network_at(net, r->in0, 4);
			v->size[pool(b)] = align((uint64_t)prev->W * prev->H, 32) * 512 * 2;
			break;
		}
		case FAMILY_VIT_QKV:
			v->size[pool(b)] = align(l->tokens, 32) * 3072 * 2;
			break;
		case FAMILY_PROJ_POOL:
			v->size[pool(b)] = pad4((l->H + 1) / 2) * pad4((l->W + 1) / 2) * 512;
			grow(&v->overread[pool(b)], v->size[pool(b)]);
			break;
		case FAMILY_FINAL_HEAD:
			// FinalHead reads its plan's tokens off the smaller pooled grid.
			grow(&v->overread[pool(b)], l->tokens * 512);
			break;
		default:
			break;
		}
		if (b == 31 && r->layer == 0)
			v->size[lift(31)] = align(l->tokens, 32) * 1024;
		if (b == 38 && r->layer == 4)
			v->size[ups(38)] = align(l->tokens, 32) * 1024;
		// A partial ViT token tile still spans every channel group.
		grow(&v->size[k], (vit(r) ? align(l->tokens, 32) : l->tokens) * outputs(r));
	}
}

/** @brief Places a value, unless it is placed already.
 *
 * @param v      The values.
 * @param placed Which values are placed.
 * @param k      The value's key.
 * @param at     Where it goes.
 * @return       Where the next value goes.
 */
static uint64_t
place_value (struct vulkan_values *v,
             bool                  placed[VULKAN_PLAN_KEYS],
             size_t                k,
             uint64_t              at)
{
	if (placed[k])
		return at;
	placed[k] = true;
	v->offset[k] = at;
	return at + align(v->size[k], 256);
}

/** @brief Gives every value its own bytes: the layers' outputs in order, then the others by key
 *         (upstream: the plain layout, nr_graph.cpp:1524-1536). A key that is not sized stays at 0.
 *
 * @param net The network.
 * @param v   The values.
 */
static void
plain_layout (struct network const *net,
              struct vulkan_values *v)
{
	bool placed[VULKAN_PLAN_KEYS] = {0};
	uint64_t at = 0;
	for (size_t i = 0; i < LAYERS; ++i)
		at = place_value(v, placed, key(net->layers[i].row.block, net->layers[i].row.layer), at);
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
		if (v->size[k])
			at = place_value(v, placed, k, at);
}

/** @brief Adds a dispatch.
 *
 * @param dispatches The dispatches.
 * @param d          The dispatch.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
push_dispatch (struct vulkan_dispatches     *dispatches,
               struct vulkan_dispatch const *d,
               struct error                 *e)
{
	if (dispatches->count == dispatches->capacity) {
		size_t const capacity = dispatches->capacity ? 2 * dispatches->capacity : 2 * LAYERS;
		struct vulkan_dispatch *const grown = realloc(dispatches->dispatch,
		                                              capacity * sizeof *grown);
		if (!grown)
			return error_fail(e, "out of memory");
		dispatches->dispatch = grown;
		dispatches->capacity = capacity;
	}
	dispatches->dispatch[dispatches->count++] = *d;
	return ERROR_NONE;
}

/** @brief The lowering of the network's layers into dispatches, and of their weights into the
 *         blob's segments (upstream: the loop of NrSession::build, nr_graph.cpp:1727-3254), on the
 *         path that dlsslopd's configuration takes. */
struct lowering {
	struct network const     *net;        //!< The network.
	struct vulkan_values     *values;     //!< Its values, whose first and last layers the lowering sets.
	struct vulkan_blob       *blob;       //!< The weight blob.
	struct vulkan_dispatches *dispatches; //!< Receives the dispatches.
	struct noise_job          noise;      //!< Receives the noise field.
	uint8_t                   step;       //!< The layer it lowers.
};

/** @brief The offset of a value that the current layer's lowering names (upstream: voff[KEY]).
 *
 * @param l The lowering.
 * @param k The value's key.
 * @return  Its offset.
 */
static uint64_t
at (struct lowering *l,
    size_t           k)
{
	int16_t const step = l->step;
	int16_t *const first = &l->values->first[k];
	int16_t *const last = &l->values->last[k];
	*first = *first < 0 ? step : step < *first ? step : *first;
	if (*last < step)
		*last = step;
	return l->values->offset[k];
}

/** @brief A value's offset as a push constant. */
static uint32_t
off (struct lowering *l,
     size_t           k)
{
	return (uint32_t)at(l, k);
}

/** @brief The value that a layer reads (upstream: x_src). */
static size_t
x_src (struct lowering const *l,
       struct layer const    *layer)
{
	struct row const *const r = &layer->row;
	if (r->family == FAMILY_FINAL_HEAD)
		return pool(r->block);
	if (r->block == 31 && r->layer == 0)
		return lift(31);
	if (r->block == 39)
		return ups(38);
	if (r->layer > 0)
		return key(r->block, r->layer - 1);
	return network_block_out(l->net, r->in0 >= 0 ? r->in0 : r->block);
}

/** @brief The value that a layer's residual adds (upstream: r_src). */
static size_t
r_src (struct lowering const *l,
       struct layer const    *layer)
{
	struct row const *const r = &layer->row;
	if (r->family == FAMILY_PROJ || r->family == FAMILY_PROJ_POOL || r->family == FAMILY_VIT_PROJECTION)
		return key(r->block, 1);
	if (r->family == FAMILY_FFWD_PROJ || r->family == FAMILY_VIT_CONTRACT)
		return x_src(l, network_at(l->net, r->block, 0));
	return r->layer > 1 ? key(r->block, r->layer - 2) : network_block_out(l->net, r->in0);
}

/** @brief A dispatch of a kernel for a layer over workgroups. */
static struct vulkan_dispatch
start (struct lowering const *l,
       struct layer const    *layer,
       enum vulkan_kernel     kernel,
       uint32_t               gx,
       uint32_t               gy,
       uint32_t               gz)
{
	return (struct vulkan_dispatch){
		.groups = {gx, gy, gz},
		.kernel = kernel,
		.after  = VULKAN_AFTER_INVALIDATE,
		.block  = layer->row.block,
		.layer  = layer->row.layer,
		.first  = l->step,
		.last   = l->step,
	};
}

/** @brief Puts a segment at the next multiple of an alignment.
 *
 * @param blob      The blob.
 * @param segment   The segment; its offset is set.
 * @param alignment The alignment.
 * @return          Its offset.
 */
static uint64_t
vulkan_blob_put (struct vulkan_blob    *blob,
                 struct vulkan_segment  segment,
                 uint32_t               alignment)
{
	blob->bytes = align(blob->bytes, alignment);
	segment.offset = (uint32_t)blob->bytes;
	if (blob->segment_count == blob->segment_capacity) {
		size_t const capacity = blob->segment_capacity ? 2 * blob->segment_capacity : 1024;
		struct vulkan_segment *const grown = realloc(blob->segments, capacity * sizeof *grown);
		if (!grown) {
			blob->error = ERROR_FAILED;
			return segment.offset;
		}
		blob->segments = grown;
		blob->segment_capacity = capacity;
	}
	blob->segments[blob->segment_count++] = segment;
	blob->bytes += segment.bytes;
	return segment.offset;
}

/** @brief Puts a segment of the blob that has rows, columns or an index.
 *
 * @param l      The lowering.
 * @param bytes  The segment's bytes.
 * @param recipe Its recipe.
 * @param flags  Its flags.
 * @param source Its entry.
 * @param align  Its alignment.
 * @param rows   Its rows.
 * @param cols   Its columns.
 * @param index  Its index.
 * @return       Its offset.
 */
static uint32_t
put_shaped (struct lowering      *l,
            uint32_t              bytes,
            enum vulkan_recipe    recipe,
            uint8_t               flags,
            struct vulkan_source  source,
            uint32_t              align,
            uint16_t              rows,
            uint16_t              cols,
            uint32_t              index)
{
	struct vulkan_segment const segment = {
		.bytes  = bytes,
		.index  = index,
		.rows   = rows,
		.cols   = cols,
		.source = source,
		.recipe = recipe,
		.flags  = flags,
	};
	return (uint32_t)vulkan_blob_put(l->blob, segment, align);
}

/** @brief Puts a segment of the blob.
 *
 * @param l      The lowering.
 * @param bytes  The segment's bytes.
 * @param recipe Its recipe.
 * @param flags  Its flags.
 * @param source Its entry.
 * @param align  Its alignment.
 * @return       Its offset.
 */
static uint32_t
put (struct lowering      *l,
     uint32_t              bytes,
     enum vulkan_recipe    recipe,
     uint8_t               flags,
     struct vulkan_source  source,
     uint32_t              align)
{
	return put_shaped(l, bytes, recipe, flags, source, align, 0, 0, 0);
}

/** @brief A layer's entry. */
static struct vulkan_source
source_of (enum vulkan_directory  directory,
           struct row const      *r,
           enum vulkan_suffix     suffix)
{
	return (struct vulkan_source){directory, r->block, r->layer, suffix};
}

/** @brief A Swin layer's matrix, which every Swin kernel reads N-paired (weight layout 3). */
static uint32_t
swin_matrix (struct lowering       *l,
             enum vulkan_directory  directory,
             struct row const      *r,
             enum vulkan_suffix     suffix,
             uint32_t               rows,
             uint32_t               cols)
{
	uint8_t const flags = VULKAN_SEGMENT_REQUANTISE | VULKAN_SEGMENT_NPAIR;
	return put_shaped(l, rows * cols, VULKAN_RECIPE_MATRIX, flags, source_of(directory, r, suffix), 256,
	                  (uint16_t)rows, (uint16_t)cols, 0);
}

/** @brief The words of a push_f_swin. */
#define SWIN_WORDS (sizeof (struct push_f_swin) / 4)

/** @brief The entry of a weight of the layer that a lowering function lowers, from its directory. */
#define ENTRY(suffix) source_of(directory, r, VULKAN_SUFFIX_##suffix)

/** @brief Lowers a Swin body, or the pre or post block, downsampling and upsampling around it
 *         (upstream: nr_graph.cpp:1734-2734).
 *
 * @param l     The lowering.
 * @param layer The layer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower_swin (struct lowering    *l,
            struct layer const *layer,
            struct error       *e)
{
	struct row const *const r = &layer->row;
	uint8_t const b = r->block;
	bool const pre = r->family == FAMILY_PRE;
	bool const post = r->family == FAMILY_POST;
	bool const ds = r->family == FAMILY_SWIN_DS;
	bool const upsample = r->family == FAMILY_SWIN_UP;
	uint32_t const c = pre || post ? 32 : r->c;
	uint32_t const heads = pre || post ? 1 : r->heads;
	uint32_t const hidden = 4 * c;
	enum vulkan_directory const directory = pre ? VULKAN_DIRECTORY_PREBLOCK
	                                      : post ? VULKAN_DIRECTORY_POSTBLOCK
	                                             : VULKAN_DIRECTORY_UNPACKED;
	struct push_f_swin p = {0};
	p.e_off = swin_matrix(l, directory, r, VULKAN_SUFFIX_MLP_EXPAND, hidden, c);
	p.ct_off = swin_matrix(l, directory, r, VULKAN_SUFFIX_MLP_CONTRACT, c, heads > 1 ? c : hidden);
	p.qkv_off = swin_matrix(l, directory, r, VULKAN_SUFFIX_QKV, 3 * c, c);
	p.op_off = swin_matrix(l, directory, r, VULKAN_SUFFIX_ATTN_OUT_PROJ, c, c);
	p.mid_off = heads > 1 ? swin_matrix(l, directory, r, VULKAN_SUFFIX_MLP_MID, c, hidden / heads) : 0;
	p.b_off = put(l, heads * 16384, VULKAN_RECIPE_BIAS, VULKAN_SEGMENT_AFFINE, ENTRY(ATTN_POS_BIAS),
	              16) / 4;
	p.rs_off = put(l, 4 * c, VULKAN_RECIPE_SCALES, VULKAN_SEGMENT_PADDED, ENTRY(RESIDUAL_SCALE), 16) / 4;
	p.ars_off = put(l, 4 * c, VULKAN_RECIPE_SCALES, 0, ENTRY(ATTN_RESIDUAL_SCALE), 16) / 4;
	p.s_off = put(l, 4 * (4 < heads ? heads : 4), VULKAN_RECIPE_BYTES, VULKAN_SEGMENT_EXACT,
	              ENTRY(SCALARS_B), 16) / 4;
	// Unread: math profile 3 points rsd_off at the activation table instead.
	put(l, 32 * c, VULKAN_RECIPE_DIAGONAL, VULKAN_SEGMENT_PADDED, ENTRY(RESIDUAL_SCALE), 16);
	p.rsd_off = 0;
	p.ard_off = put(l, 32 * c, VULKAN_RECIPE_DIAGONAL, 0, ENTRY(ATTN_RESIDUAL_SCALE), 16) / 2;
	p.x_off = off(l, upsample || post ? ups(b) : pre ? lift(b) : x_src(l, layer));
	p.o_off = off(l, ds || pre ? skip(b) : key(b, r->layer));
	// An upsampling body's tiles are its output level's: its encoder skip's, or in the post block
	// twice its own extent.
	uint32_t const scale = post ? 2 : 1;
	struct extent const out = upsample ? network_up_extent(l->net, layer)
	                                   : (struct extent){scale * layer->W, scale * layer->H};
	p.tiles_x = tiles_of((uint32_t)out.h);
	p.tiles_y = tiles_of((uint32_t)out.w);
	p.shift = r->sx;
	p.shift_y = r->sy;
	if (pre || ds) {
		p.pool_off = off(l, pre ? key(b, r->layer) : pool(b));
		p.pool_tiles_x = pre ? tiles_of(layer->H / 2) : (uint32_t)(pad4((layer->H + 1) / 2) / 4);
		p.pool_tiles_y = (p.tiles_y + 1) / 2;
	}
	struct vulkan_dispatch d = start(l, layer, SWIN_BODIES[r->family].kernel, layer->gx, layer->gy, 1);
	if (c != 32) {
		d.kernel = VULKAN_KERNEL_COUNT;
		d.body = SWIN_BODIES[r->family].body;
		d.c = (uint16_t)c;
	}
	memcpy(d.push, &p, sizeof p);
	d.words = SWIN_WORDS;

	if (pre) {
		// Upstream's standalone input lift, whose weights stay in the blob; fused into the body,
		// it needs the input's extent in whole windows.
		put(l, 1024, VULKAN_RECIPE_BYTES, VULKAN_SEGMENT_EXACT, ENTRY(INPUT_LIFT), 16);
		off(l, lift(b));
		if (layer->H % 8 || layer->W % 8 || p.tiles_x * 4 != layer->H || p.tiles_y * 4 != layer->W ||
		    p.shift || p.shift_y || layer->gx != (layer->H + 7) / 8 || layer->gy != (layer->W + 7) / 8)
			return error_fail(e, "network plan: the pre block's input lift does not fuse");
		// The noise features the build writes, a pixel of the window grid four binary16s.
		uint32_t const noise_width = layer->gx * 8;
		uint32_t const noise_height = layer->gy * 8;
		uint32_t const noise_off = put(l, noise_width * noise_height * 8, VULKAN_RECIPE_ZEROS, 0,
		                               (struct vulkan_source){0}, 256) / 4;
		l->noise = (struct noise_job){noise_off, noise_width, noise_height, 0, 1.0f};
		// Upstream's defaults, which each frame's controls replace.
		uint32_t const lift_off = put(l, 1024, VULKAN_RECIPE_LIFT, VULKAN_SEGMENT_EXACT,
		                              ENTRY(INPUT_LIFT), 16) / 2;
		struct push_pre_image const image = {
			lift_off, l->net->width, l->net->height, 0,
			0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			noise_off,
		};
		d.kernel = VULKAN_KERNEL_FSWIN_IMAGE_PREDS32;
		memcpy(d.push + SWIN_WORDS, &image, sizeof image);
		d.words = 32;
		// The 2x2 mean's slots, pooled inside the body.
		off(l, skip(b));
		off(l, key(b, r->layer));
		return push_dispatch(l->dispatches, &d, e);
	}
	if (post) {
		// Unread: upstream's --post-skip gain, zero.
		put(l, 64, VULKAN_RECIPE_ZEROS, 0, (struct vulkan_source){0}, 16);
		struct push_ups u = {0};
		u.p_off = off(l, x_src(l, layer));
		u.s_off = off(l, skip(r->in1 >= 0 ? r->in1 : b));
		u.o_off = off(l, ups(b));
		u.tiles_x = p.tiles_x;
		u.tiles_y = p.tiles_y;
		u.itiles_y = tiles_of(layer->W);
		u.itiles_x = tiles_of(layer->H);
		u.g_off = put(l, 64, VULKAN_RECIPE_HALF, VULKAN_SEGMENT_EXACT, ENTRY(SKIP_GAIN), 16) / 2;
		put(l, 64, VULKAN_RECIPE_HALF, VULKAN_SEGMENT_EXACT, ENTRY(MAIN_GAIN), 16);
		u.mode = 6;
		// The input's own raster, as its producer wrote it, unless it is the body's.
		struct layer const *const prev = network_at(l->net, r->in0, l->net->last_layer[r->in0]);
		uint32_t const view_x = off(l, x_src(l, layer));
		uint32_t const view_o = off(l, pool(b));
		struct push_ups_view const view = {
			view_x, view_o, layer->H * layer->W, 32, prev->H, prev->W, layer->H, layer->W,
		};
		if (view.W == view.RW && view.H == view.RH && view.W % 4 == 0 && view.H % 4 == 0 &&
		    view.M == view.W * view.H) {
			u.p_off = view.x_off;
		} else {
			struct vulkan_dispatch v = start(l, layer, VULKAN_KERNEL_UPS_VIEW,
			                                 (view.M * (view.C / 16) + 63) / 64, 1, 1);
			memcpy(v.push, &view, sizeof view);
			v.words = sizeof view / 4;
			enum error_code const code = push_dispatch(l->dispatches, &v, e);
			if (code)
				return code;
			u.p_off = view.o_off;
		}
		uint32_t const tail_off = put(l, 1024, VULKAN_RECIPE_BYTES, VULKAN_SEGMENT_EXACT,
		                              ENTRY(OUT_PROJECT), 16) / 2;
		struct push_image_tail const tail = {tail_off, 1.0f};
		// Upstream's standalone output projection's input.
		off(l, key(b, r->layer));
		d.kernel = VULKAN_KERNEL_FSWIN_IMAGE_POST32;
		memcpy(d.push + SWIN_WORDS, &u, sizeof u);
		memcpy(d.push + (sizeof p + sizeof u) / 4, &tail, sizeof tail);
		d.words = 32;
		return push_dispatch(l->dispatches, &d, e);
	}
	if (upsample) {
		// At C>=64 the resample reads its input in the view's raster, which the producer writes
		// straight into when the view is a copy.
		bool const identity = layer->H % 4 == 0 && layer->W % 4 == 0 &&
		                      layer->tokens == (uint64_t)layer->H * layer->W;
		bool const view = c >= 64 && !identity;
		if (view) {
			uint32_t const x = off(l, x_src(l, layer));
			uint32_t const o = off(l, ups(b));
			uint32_t const tokens = (uint32_t)layer->tokens;
			struct vulkan_dispatch *producer = nullptr;
			if (layer->H % 4 == 0 && layer->W % 4 == 0 && tokens >= layer->H * layer->W) {
				for (size_t i = l->dispatches->count; i-- && !producer;) {
					struct vulkan_dispatch *const q = &l->dispatches->dispatch[i];
					if (q->kernel == VULKAN_KERNEL_GEMM_PROJ ||
					    q->kernel == VULKAN_KERNEL_GEMM_PROJC) {
						struct push_gemm g;
						memcpy(&g, q->push, sizeof g);
						if (q->words == sizeof (struct push_gemm) / 4 && g.o_off == x &&
						    g.M == layer->H * layer->W)
							producer = q;
					} else if (q->body == VULKAN_BODY_SWIN && q->c == 2 * c &&
					           q->words == SWIN_WORDS) {
						struct push_f_swin f;
						memcpy(&f, q->push, sizeof f);
						if (f.o_off == x)
							producer = q;
					}
				}
			}
			if (!producer)
				return error_fail(e, "network plan: an upsampling layer's view has no producer "
				                     "to fold into");
			if (producer->kernel != VULKAN_KERNEL_COUNT) {
				struct push_gemm g;
				memcpy(&g, producer->push, sizeof g);
				g.o_off = o;
				memcpy(producer->push, &g, sizeof g);
			} else {
				struct push_f_swin f;
				memcpy(&f, producer->push, sizeof f);
				f.o_off = o;
				memcpy(producer->push, &f, sizeof f);
			}
			// The view's padding past the producer's tokens is read as zero.
			uint64_t const size = l->values->size[ups(b)];
			uint64_t const read = (uint64_t)tokens * 2 * c;
			grow(&l->values->overread[ups(b)], size < read ? read : size);
		}
		uint32_t const w_off = put_shaped(l, 2 * c * c, VULKAN_RECIPE_MATRIX, 0, ENTRY(RESAMPLE), 256,
		                                  (uint16_t)c, (uint16_t)(2 * c), 0);
		uint32_t const x = off(l, view ? ups(b) : x_src(l, layer));
		// The resample's output, which the blend read: fused away.
		off(l, pool(b));
		struct push_ups u = {0};
		u.s_off = off(l, skip(r->in1 >= 0 ? r->in1 : b));
		u.o_off = off(l, ups(b));
		u.g_off = put(l, 2 * c, VULKAN_RECIPE_HALF, VULKAN_SEGMENT_GAIN_TAIL, ENTRY(UPSAMPLE_GAIN),
		              16) / 2;
		u.tiles_x = p.tiles_x;
		u.tiles_y = p.tiles_y;
		u.itiles_x = tiles_of(layer->H);
		u.itiles_y = tiles_of(layer->W);
		u.stiles_x = tiles_of(network_at(l->net, r->in1, 0)->H);
		// Fused: the body reads the resample's input and projects it itself.
		u.p_off = x;
		memcpy(d.push + SWIN_WORDS, &u, sizeof u);
		d.push[(sizeof p + sizeof u) / 4] = w_off;
		d.words = (sizeof p + sizeof u) / 4 + 1;
		return push_dispatch(l->dispatches, &d, e);
	}
	if (ds) {
		// The 2x2 mean's slots, the second the resample's input: pooled and projected inside the
		// body.
		off(l, skip(b));
		off(l, pool(b));
		struct push_ds_proj q = {0};
		q.w_off = put_shaped(l, 2 * c * c, VULKAN_RECIPE_MATRIX, 0, ENTRY(RESAMPLE), 256,
		                     (uint16_t)(2 * c), (uint16_t)c, 0);
		q.o_off = off(l, key(b, r->layer));
		q.otx = (uint32_t)(pad4((layer->H + 1) / 2) / 4);
		q.crow = q.raster = q.otx * 4;
		q.rows = (uint32_t)pad4((layer->W + 1) / 2);
		if (c == 32) {
			q.writer_rows = layer->W / 2;
			q.raster = layer->H / 2;
		} else {
			// The view's padding, zeroed each frame.
			q.clear_x = (layer->H + 1) / 2;
			q.clear_y = (layer->W + 1) / 2;
		}
		q.n = 2 * c;
		if (p.pool_tiles_x != q.otx)
			return error_fail(e, "network plan: a downsampling layer's pooled grid is not its own");
		memcpy(d.push + SWIN_WORDS, &q, sizeof q);
		d.words = (sizeof p + sizeof q) / 4;
	}
	return push_dispatch(l->dispatches, &d, e);
}

#undef SWIN_WORDS

/** @brief Lowers the decoder's input: a projection into binary16, then upsampled with the encoder's
 *         skip added (upstream: nr_graph.cpp:2735-2799).
 *
 * @param l     The lowering.
 * @param layer The layer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower_decoder (struct lowering    *l,
               struct layer const *layer,
               struct error       *e)
{
	struct row const *const r = &layer->row;
	uint8_t const b = r->block;
	uint32_t const n = r->co;
	uint32_t const k = r->ci;
	enum vulkan_directory const directory = VULKAN_DIRECTORY_SPLIT_SWIN;
	struct push_gemm p = {0};
	p.w_off = put_shaped(l, n * k, VULKAN_RECIPE_MATRIX, VULKAN_SEGMENT_NPAIR, ENTRY(WEIGHT), 256,
	                     (uint16_t)n, (uint16_t)k, 0);
	p.g_off = put(l, 2 * n, VULKAN_RECIPE_HALF, 0, ENTRY(SKIP_WEIGHT), 16) / 2;
	p.gd_off = put(l, n / 16 * 512, VULKAN_RECIPE_DIAGONAL, 0, ENTRY(SKIP_WEIGHT), 16) / 2;
	p.r_off = off(l, skip(r->in1 >= 0 ? r->in1 : b));
	p.x_off = off(l, x_src(l, layer));
	off(l, key(b, r->layer));
	struct layer const *const prev = network_at(l->net, r->in0, 4);
	p.M = prev->W * prev->H;
	p.N = n;
	p.K = k;
	p.W = layer->W;
	p.o_off = (uint32_t)(at(l, pool(b)) / 2);
	if (n % VULKAN_PLAN_GEMM_PROJ_NT)
		return error_fail(e, "network plan: the decoder's projection does not divide its tile");
	struct vulkan_dispatch d = start(l, layer, VULKAN_KERNEL_GEMM_VQKVS,
	                                 (p.M + VULKAN_PLAN_GEMM_PROJ_MT - 1) / VULKAN_PLAN_GEMM_PROJ_MT,
	                                 n / VULKAN_PLAN_GEMM_PROJ_NT, 1);
	memcpy(d.push, &p, sizeof p);
	d.words = sizeof p / 4;
	enum error_code const code = push_dispatch(l->dispatches, &d, e);
	if (code)
		return code;
	uint32_t const skip_off = off(l, key(r->in1, 3));
	uint32_t const dst_off = off(l, key(b, r->layer));
	struct push_decoder_ups const u = {p.o_off, skip_off, dst_off, p.g_off, prev->H, layer->H, layer->W};
	// Whole 4x4 token tiles take the tile form of VULKAN_PLAN_DECUPS_VEC 16; others the shader's
	// per-element fallback.
	static_assert(VULKAN_PLAN_DECUPS_VEC == 16);
	uint32_t const groups = u.OW % 4 == 0 && u.OH % 4 == 0 ? (u.OW * u.OH / 16 * 1024 + 63) / 64
	                                                       : (u.OW * u.OH * 512 + 63) / 64;
	struct vulkan_dispatch v = start(l, layer, VULKAN_KERNEL_DEC_UPS, groups, 1, 1);
	memcpy(v.push, &u, sizeof u);
	v.words = sizeof u / 4;
	return push_dispatch(l->dispatches, &v, e);
}

/** @brief Lowers the ViT's global attention over its tokens, W x H of them, with the temperatures
 *         of the QKV layer before it (upstream: nr_graph.cpp:2802-2851).
 *
 * @param l     The lowering.
 * @param layer The layer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower_vit_attention (struct lowering    *l,
                     struct layer const *layer,
                     struct error       *e)
{
	struct row const *const r = &layer->row;
	struct push_v_attn p = {0};
	p.x_off = off(l, x_src(l, layer));
	p.o_off = off(l, key(r->block, r->layer));
	p.tokens = layer->W * layer->H;
	struct vulkan_source const temperatures = {
		VULKAN_DIRECTORY_RECORDS, r->block, (uint8_t)(r->layer - 1), VULKAN_SUFFIX_NONE,
	};
	p.s_off = put(l, 128, VULKAN_RECIPE_BYTES, 0, temperatures, 16) / 4;
	// Q and K arrive normalized and scaled; the weights rounded to E4M3.
	p.mode = 4;
	struct vulkan_dispatch d = start(l, layer, VULKAN_KERNEL_VIT_ATTN,
	                                 (p.tokens + VULKAN_PLAN_VATTN_QT - 1) / VULKAN_PLAN_VATTN_QT, 32, 1);
	memcpy(d.push, &p, sizeof p);
	d.words = sizeof p / 4;
	return push_dispatch(l->dispatches, &d, e);
}

/** @brief Lowers the C=512 FFN, over the tile grid (upstream: nr_graph.cpp:2852-2927).
 *
 * @param l     The lowering.
 * @param layer The layer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower_ffwd (struct lowering    *l,
            struct layer const *layer,
            struct error       *e)
{
	struct row const *const r = &layer->row;
	struct vulkan_source const record = source_of(VULKAN_DIRECTORY_RECORDS, r, VULKAN_SUFFIX_NONE);
	struct push_ffwd3 p = {0};
	p.a_off = put_shaped(l, 8 * 64 * 512, VULKAN_RECIPE_FFWD, VULKAN_SEGMENT_NPAIR, record, 256, 0, 0, 0);
	p.q0_off = put_shaped(l, 8 * 256 * 64, VULKAN_RECIPE_FFWD, VULKAN_SEGMENT_NPAIR, record, 256, 0, 0, 1);
	p.q2_off = put_shaped(l, 8 * 64 * 256, VULKAN_RECIPE_FFWD, VULKAN_SEGMENT_NPAIR, record, 256, 0, 0, 2);
	p.x_off = off(l, x_src(l, layer));
	p.o_off = off(l, key(r->block, r->layer));
	p.M = tiles_of(layer->H) * tiles_of(layer->W) * 16;
	p.C = 512;
	// ffwd3w: two token tiles a subgroup. Workgroups are group-major, eight weight groups times
	// the token units VULKAN_PLAN_FFWD_WGW a workgroup.
	bool const two = p.M >= VULKAN_PLAN_FFWD_FM2_MIN_TOKENS;
	uint32_t const units = (p.M / 16 + (two ? 1 : 0)) / (two ? 2 : 1);
	struct vulkan_dispatch d = start(l, layer, two ? VULKAN_KERNEL_FFWD3W : VULKAN_KERNEL_FFWD3,
	                                 8 * ((units + VULKAN_PLAN_FFWD_WGW - 1) / VULKAN_PLAN_FFWD_WGW), 1, 1);
	memcpy(d.push, &p, sizeof p);
	d.words = sizeof p / 4;
	return push_dispatch(l->dispatches, &d, e);
}

/** @brief Lowers the C=512 windowed attention, its 16 heads split VULKAN_PLAN_ATTENTION_SPLIT ways
 *         (upstream: nr_graph.cpp:2928-2984).
 *
 * @param l     The lowering.
 * @param layer The layer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower_attention (struct lowering    *l,
                 struct layer const *layer,
                 struct error       *e)
{
	struct row const *const r = &layer->row;
	enum vulkan_directory const directory = VULKAN_DIRECTORY_SPLIT_SWIN;
	struct push_attn p = {0};
	// Its QKV matrix N-paired, as attn.comp reads it (NR_ATTN_WPAIR).
	p.w_off = put_shaped(l, 1536 * 512, VULKAN_RECIPE_MATRIX, VULKAN_SEGMENT_NPAIR, ENTRY(QKV), 256,
	                     1536, 512, 0);
	p.b_off = put(l, 16 * 16384, VULKAN_RECIPE_BIAS, 0, ENTRY(ATTN_POS_BIAS), 16) / 4;
	p.s_off = put(l, 64, VULKAN_RECIPE_BYTES, VULKAN_SEGMENT_EXACT, ENTRY(TAIL), 16) / 4;
	p.x_off = off(l, x_src(l, layer));
	p.o_off = off(l, key(r->block, r->layer));
	p.C = 512;
	p.wins_x = layer->gx;
	p.tiles_x = tiles_of(layer->H);
	p.tiles_y = tiles_of(layer->W);
	p.shift = r->sx;
	p.shift_y = r->sy;
	struct vulkan_dispatch d = start(l, layer, VULKAN_KERNEL_ATTN, layer->gx, layer->gy,
	                                 VULKAN_PLAN_ATTENTION_SPLIT);
	memcpy(d.push, &p, sizeof p);
	d.words = sizeof p / 4;
	return push_dispatch(l->dispatches, &d, e);
}

/** @brief Lowers a GEMM: the C=512 projections and head, and the ViT's layers (upstream:
 *         nr_graph.cpp:2985-3254).
 *
 * @param l     The lowering.
 * @param layer The layer.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower_gemm (struct lowering    *l,
            struct layer const *layer,
            struct error       *e)
{
	struct row const *const r = &layer->row;
	uint8_t const b = r->block;
	enum family const family = r->family;
	if (b == 31 && r->layer == 0) {
		// FinalHead stores the ViT's input in its raster itself, and the value's life starts
		// there.
		if (layer->W % 4 || layer->H % 4)
			return error_fail(e, "network plan: the ViT's input does not fold into FinalHead");
		struct vulkan_dispatch *head = nullptr;
		for (size_t i = l->dispatches->count; i-- && !head;) {
			struct vulkan_dispatch *const d = &l->dispatches->dispatch[i];
			if (d->block == 30 && d->layer == 4 && d->kernel == VULKAN_KERNEL_GEMM_NORES)
				head = d;
		}
		if (!head)
			return error_fail(e, "network plan: the ViT's input has no FinalHead to fold into");
		struct push_gemm q;
		memcpy(&q, head->push, sizeof q);
		uint8_t const now = l->step;
		l->step = network_step(l->net, 30, 4);
		q.p_off = off(l, lift(31));
		l->step = now;
		q.W = layer->H;
		q.rows = layer->W;
		memcpy(head->push, &q, sizeof q);
	}
	bool const split = family <= FAMILY_FINAL_HEAD;
	bool const residual = family == FAMILY_FFWD_PROJ || family == FAMILY_PROJ ||
	                      family == FAMILY_PROJ_POOL || family == FAMILY_VIT_CONTRACT ||
	                      family == FAMILY_VIT_PROJECTION;
	enum vulkan_directory const directory = split ? VULKAN_DIRECTORY_SPLIT_SWIN : VULKAN_DIRECTORY_VIT;
	struct vulkan_source const record = source_of(VULKAN_DIRECTORY_RECORDS, r, VULKAN_SUFFIX_NONE);
	uint32_t const n = (uint32_t)outputs(r);
	uint32_t const k = family == FAMILY_FINAL_HEAD ? 512
	                 : family == FAMILY_VIT_CONTRACT ? 4096
	                 : split ? 512
	                         : 1024;
	// The C=512 layers compute the tile grid, the ViT its tokens.
	uint32_t const m = split ? tiles_of(layer->H) * tiles_of(layer->W) * 16 : (uint32_t)layer->tokens;
	struct push_gemm p = {0};
	size_t const weight = l->blob->segment_count;
	p.w_off = family == FAMILY_VIT_QKV
	        ? put_shaped(l, n * k, VULKAN_RECIPE_VIT_QKV, 0, record, 256, (uint16_t)n, (uint16_t)k, 0)
	        : put_shaped(l, n * k, VULKAN_RECIPE_MATRIX, 0, ENTRY(WEIGHT), 256, (uint16_t)n, (uint16_t)k,
	                     0);
	if (residual) {
		p.g_off = put(l, 2 * n, VULKAN_RECIPE_HALF, 0, ENTRY(SKIP_WEIGHT), 16) / 2;
		p.gd_off = put(l, n / 16 * 512, VULKAN_RECIPE_DIAGONAL, 0, ENTRY(SKIP_WEIGHT), 16) / 2;
		p.r_off = off(l, r_src(l, layer));
	}
	p.x_off = off(l, x_src(l, layer));
	p.o_off = off(l, key(b, r->layer));
	p.M = m;
	p.N = n;
	p.K = k;
	p.W = layer->W;
	// The ViT's products without a residual take the wide tile, but its QKV at up to
	// VULKAN_PLAN_QKVS_MAX_TOKENS tokens gemmvqkvnorms'; the residual projections take theirs, the
	// rest 64x128. At many ViT tokens, its contractions and its projections but the last, which
	// stores the decoder's raster, take gemmprojw's.
	bool const qkv = family == FAMILY_VIT_QKV;
	bool const qkvs = qkv && m <= VULKAN_PLAN_QKVS_MAX_TOKENS;
	bool const wide = vit(r) && !residual;
	bool const projection = !wide && residual && family != FAMILY_PROJ_POOL && family != FAMILY_VIT_EXPAND;
	bool const last = b == 38 && r->layer == 4;
	bool const projw = projection && m >= VULKAN_PLAN_PROJW_MIN_TOKENS &&
	                   (family == FAMILY_VIT_CONTRACT || (family == FAMILY_VIT_PROJECTION && !last));
	uint32_t const mt = qkvs         ? VULKAN_PLAN_GEMM_QKVS_MT
	                  : wide         ? VULKAN_PLAN_GEMM_WIDE_MT
	                  : projw        ? VULKAN_PLAN_GEMM_PROJW_MT
	                  : projection   ? VULKAN_PLAN_GEMM_PROJ_MT
	                                 : 64;
	uint32_t const nt = qkvs         ? VULKAN_PLAN_GEMM_QKVS_NT
	                  : wide         ? VULKAN_PLAN_GEMM_WIDE_NT
	                  : projw        ? VULKAN_PLAN_GEMM_PROJW_NT
	                  : projection   ? VULKAN_PLAN_GEMM_PROJ_NT
	                                 : 128;
	if (n % nt)
		return error_fail(e, "network plan: a GEMM's outputs do not divide its tile");
	enum vulkan_kernel kernel = family == FAMILY_VIT_EXPAND ? VULKAN_KERNEL_GEMM_VACT
	                          : qkvs                        ? VULKAN_KERNEL_GEMM_VQKV_NORMS
	                          : qkv                         ? VULKAN_KERNEL_GEMM_VQKV_NORM
	                          : !residual                   ? VULKAN_KERNEL_GEMM_NORES
	                          : projw                       ? VULKAN_KERNEL_GEMM_PROJW
	                          : split                       ? VULKAN_KERNEL_GEMM_PROJC
	                                                        : VULKAN_KERNEL_GEMM_PROJ;
	if (family == FAMILY_PROJ_POOL) {
		kernel = VULKAN_KERNEL_GEMM_POOL;
		p.p_off = off(l, pool(b));
		p.W = layer->H;
		p.rows = layer->W;
	}
	// Two token tiles, then every output tile, then the next two.
	if (family == FAMILY_VIT_CONTRACT ||
	    (family == FAMILY_VIT_PROJECTION && m < VULKAN_PLAN_PROJW_MIN_TOKENS) ||
	    (family == FAMILY_VIT_EXPAND && m <= 1024))
		p.remap = 2;
	if (qkv) {
		// Q normalized and scaled in the GEMM's epilogue, with the record's first 128 bytes, its
		// temperatures.
		p.o_off = (uint32_t)(at(l, pool(b)) / 2);
		uint32_t const scales = put(l, 128, VULKAN_RECIPE_BYTES, 0, record, 16) / 4;
		if (n != 3072 || k != 1024)
			return error_fail(e, "network plan: the ViT's QKV is not 3072x1024");
		p.o_off = off(l, key(b, r->layer));
		p.r_off = scales;
	}
	if (last) {
		// The last projection stores in the raster the decoder reads.
		if (kernel != VULKAN_KERNEL_GEMM_PROJ || layer->W % 4 || layer->H % 4)
			return error_fail(e, "network plan: the ViT's output does not fold into its "
			                     "projection");
		p.o_off = off(l, ups(38));
		p.W = layer->H;
		p.rows = layer->W;
		kernel = VULKAN_KERNEL_GEMM_PROJT;
	}
	// Weight layout 3 pairs every GEMM's weights but FinalHead's and ProjPool's.
	if (kernel != VULKAN_KERNEL_GEMM_NORES && kernel != VULKAN_KERNEL_GEMM_POOL &&
	    weight < l->blob->segment_count)
		l->blob->segments[weight].flags |= VULKAN_SEGMENT_NPAIR;
	struct vulkan_dispatch d = start(l, layer, kernel, (m + mt - 1) / mt, n / nt, 1);
	memcpy(d.push, &p, sizeof p);
	d.words = sizeof p / 4;
	return push_dispatch(l->dispatches, &d, e);
}

#undef ENTRY

/** @brief Lowers every layer.
 *
 * @param l The lowering.
 * @param e Receives the words for what stopped it, or nullptr.
 * @return  ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
lower (struct lowering *l,
       struct error    *e)
{
	// The activation table comes first: the Swin bodies find it at offset 0.
	put(l, 4096, VULKAN_RECIPE_ACTIVATIONS, 0, (struct vulkan_source){0}, 256);
	for (l->step = 0; l->step < LAYERS; ++l->step) {
		struct layer const *const layer = &l->net->layers[l->step];
		enum error_code code;
		switch (layer->row.family) {
		case FAMILY_SWIN:
		case FAMILY_SWIN_DS:
		case FAMILY_SWIN_UP:
		case FAMILY_PRE:
		case FAMILY_POST:
			code = lower_swin(l, layer, e);
			break;
		case FAMILY_DECODER_UP:
			code = lower_decoder(l, layer, e);
			break;
		case FAMILY_VIT_ATTENTION:
			code = lower_vit_attention(l, layer, e);
			break;
		case FAMILY_FFWD:
			code = lower_ffwd(l, layer, e);
			break;
		case FAMILY_QKV_ATTN:
			code = lower_attention(l, layer, e);
			break;
		default:
			code = lower_gemm(l, layer, e);
			break;
		}
		if (code)
			return code;
	}
	return l->blob->error ? error_fail(e, "out of memory") : ERROR_NONE;
}

#undef LAYERS

uint64_t
vulkan_blob_put_words (struct vulkan_blob *blob,
                       uint32_t const     *words,
                       size_t              count)
{
	uint32_t const index = (uint32_t)blob->table_count;
	if (count > blob->table_capacity - blob->table_count) {
		size_t capacity = blob->table_capacity ? blob->table_capacity : 4096;
		while (count > capacity - blob->table_count)
			capacity *= 2;
		uint32_t *const grown = realloc(blob->tables, capacity * sizeof *grown);
		if (!grown) {
			blob->error = ERROR_FAILED;
			return 0;
		}
		blob->tables = grown;
		blob->table_capacity = capacity;
	}
	if (count) {
		memcpy(blob->tables + blob->table_count, words, count * sizeof *words);
		blob->table_count += count;
	}
	struct vulkan_segment const segment = {
		.bytes  = (uint32_t)(count * 4),
		.index  = index,
		.recipe = VULKAN_RECIPE_TABLE,
	};
	return vulkan_blob_put(blob, segment, 16);
}

enum error_code
vulkan_blob_word (struct vulkan_blob const *blob,
                  uint64_t                  index,
                  uint32_t                 *word,
                  struct error             *e)
{
	for (size_t i = blob->segment_count; i--;) {
		struct vulkan_segment const *const s = &blob->segments[i];
		if (s->recipe == VULKAN_RECIPE_TABLE && 4 * index >= s->offset &&
		    4 * index < (uint64_t)s->offset + s->bytes) {
			*word = blob->tables[s->index + (4 * index - s->offset) / 4];
			return ERROR_NONE;
		}
	}
	return error_fail(e, "network plan: a table word is outside the tables");
}

/** @brief Frees what a blob holds and empties it. */
static void
blob_fini (struct vulkan_blob *blob)
{
	free(blob->segments);
	blob->segments = nullptr;
	free(blob->tables);
	blob->tables = nullptr;
	*blob = (struct vulkan_blob){0};
}

/** @brief Frees dispatches and empties them. */
static void
dispatches_fini (struct vulkan_dispatches *dispatches)
{
	free(dispatches->dispatch);
	dispatches->dispatch = nullptr;
	*dispatches = (struct vulkan_dispatches){0};
}

/** @brief What a plan's build holds on the heap until it ends. */
struct work {
	struct vulkan_blob       probe_blob;    //!< The probe's blob, whose weights are not read.
	struct vulkan_blob       blob;          //!< The plan's blob, which the plan takes.
	struct vulkan_dispatches probe_lowered; //!< The probe's lowering.
	struct vulkan_dispatches probed;        //!< The probe's lowering merged.
	struct vulkan_dispatches lowered;       //!< The lowering over the shared arena.
	struct vulkan_dispatches merged;        //!< That lowering merged.
};

/** @brief Plans the network, with the heap's work in a struct work.
 *
 * @param p       Receives the plan, zeroed before.
 * @param w       The work, zeroed before.
 * @param width   The frames' width.
 * @param height  Their height.
 * @param storage The most bytes one of the device's storage buffers holds.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, ERROR_REJECTED or ERROR_FAILED.
 */
static enum error_code
build (struct vulkan_plan *p,
       struct work        *w,
       uint32_t            width,
       uint32_t            height,
       uint64_t            storage,
       struct error       *e)
{
	struct network net;
	enum error_code code = place(width, height, &net, e);
	if (code)
		return code;
	struct vulkan_values values = {0};
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k) {
		values.first[k] = -1;
		values.last[k] = -1;
	}
	size_values(&net, &values);
	plain_layout(&net, &values);

	// Upstream's layout probe: the network lowered over every value in its own bytes, for the
	// layers that name each one. Its weights are not read.
	struct lowering probe = {&net, &values, &w->probe_blob, &w->probe_lowered, {0}, 0};
	code = lower(&probe, e);
	if (code)
		return code;
	uint64_t probe_arena = 0;
	bool epoch = false;
	code = vulkan_schedule_merge(&w->probed, &w->probe_lowered, &w->probe_blob, &probe_arena, &epoch, e);
	if (code)
		return code;
	if (!epoch)
		return error_fail(e, "network plan: no persistent run at C=256 to count frames with");

	p->width = width;
	p->height = height;
	p->work_width = net.work_width;
	p->work_height = net.work_height;
	code = vulkan_schedule_share(&w->probed, epoch, &values, &p->values_end, e);
	if (code)
		return code;
	if (p->values_end > UINT32_MAX)
		return error_reject(e, REFUSED "its activation arena of at least %" PRIu64
		                    " bytes overflows 32-bit offsets", width, height, p->values_end);
	// The network lowered again over the shared arena, which must lower the same way.
	struct lowering lowering = {&net, &values, &w->blob, &w->lowered, {0}, 0};
	code = lower(&lowering, e);
	if (code)
		return code;
	uint64_t arena = p->values_end;
	code = vulkan_schedule_merge(&w->merged, &w->lowered, &w->blob, &arena, &epoch, e);
	if (code)
		return code;
	struct vulkan_dispatches *const dispatches = &w->merged;
	bool same = dispatches->count == w->probed.count;
	for (size_t i = 0; same && i < dispatches->count; ++i) {
		struct vulkan_dispatch const *const a = &dispatches->dispatch[i];
		struct vulkan_dispatch const *const b = &w->probed.dispatch[i];
		same = a->kernel == b->kernel && a->first == b->first && a->last == b->last;
	}
	if (!same)
		return error_fail(e, "network plan: the network lowered differently over the shared arena");
	vulkan_schedule_trim(dispatches, height);
	uint32_t error = 0;
	code = vulkan_schedule_chain(dispatches, &w->blob, &arena, &error, &p->counter_words, e);
	if (code == ERROR_REJECTED) {
		error_wrap(e, REFUSED, width, height);
		return code;
	}
	if (code)
		return code;
	if (arena > UINT32_MAX)
		return error_reject(e, REFUSED "its activation arena of %" PRIu64
		                    " bytes overflows 32-bit offsets", width, height, arena);
	if (w->blob.bytes > UINT32_MAX)
		return error_fail(e, "network plan: the weight blob overflows 32-bit offsets");
	// The arena and the weights are each one storage buffer, bound whole.
	if (arena > storage || w->blob.bytes > storage)
		return error_reject(e, REFUSED "its activation arena of %" PRIu64 " bytes or weights of %"
		                    PRIu64 " bytes exceed the device's storage buffers of %" PRIu64 " bytes",
		                    width, height, arena, w->blob.bytes, storage);
	p->arena_bytes = arena;
	p->blob_bytes = (uint32_t)w->blob.bytes;
	p->noise = lowering.noise;

	// The steps, their push words, and the words that the waits set: the one that every record
	// names, then each persistent run's.
	size_t push_count = 0;
	size_t runs = 0;
	for (size_t i = 0; i < dispatches->count; ++i) {
		struct vulkan_dispatch const *const d = &dispatches->dispatch[i];
		if (d->kernel == VULKAN_KERNEL_COUNT || d->words * 4 > VULKAN_PLAN_KERNELS[d->kernel].push)
			return error_fail(e, "network plan: a dispatch has no kernel, or a push block past its "
			                     "range");
		push_count += d->words;
		runs += vulkan_plan_persistent(d->kernel);
	}
	size_t value_count = 0;
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
		value_count += values.size[k] || values.first[k] >= 0;
	p->steps = malloc(dispatches->count * sizeof *p->steps);
	p->push = malloc(push_count * sizeof *p->push);
	p->timeouts = malloc((1 + runs) * sizeof *p->timeouts);
	p->values = malloc(value_count * sizeof *p->values);
	if (!p->steps || !p->push || !p->timeouts || !p->values)
		return error_fail(e, "out of memory");
	p->timeouts[p->timeout_count++] = error;
	for (size_t i = 0; i < dispatches->count; ++i) {
		struct vulkan_dispatch const *const d = &dispatches->dispatch[i];
		p->steps[p->step_count++] = (struct vulkan_step){
			.groups = {d->groups[0], d->groups[1], d->groups[2]},
			.push   = (uint16_t)p->push_count,
			.kernel = d->kernel,
			.after  = d->after,
			.words  = d->words,
			.block  = d->block,
			.layer  = d->layer,
			.first  = d->first,
			.last   = d->last,
		};
		memcpy(p->push + p->push_count, d->push, d->words * sizeof *d->push);
		p->push_count += d->words;
		p->chained += d->after == VULKAN_AFTER_NOTHING;
		if (vulkan_plan_persistent(d->kernel)) {
			struct push_persist r;
			memcpy(&r, d->push, sizeof r);
			p->timeouts[p->timeout_count++] = r.sync_off + VULKAN_PLAN_RUN_ERROR;
		}
	}
	p->steps[p->step_count - 1].after = VULKAN_AFTER_FULL;
	for (size_t k = 0; k < VULKAN_PLAN_KEYS; ++k)
		if (values.size[k] || values.first[k] >= 0)
			p->values[p->value_count++] = (struct vulkan_value){
				.offset   = values.offset[k],
				.bytes    = values.size[k],
				.overread = values.overread[k],
				.key      = k,
			};
	// The plan takes the blob.
	p->segments = w->blob.segments;
	p->segment_count = w->blob.segment_count;
	p->tables = w->blob.tables;
	p->table_count = w->blob.table_count;
	w->blob = (struct vulkan_blob){0};
	return ERROR_NONE;
}

#undef REFUSED

enum error_code
vulkan_plan_init (struct vulkan_plan *dest,
                  uint32_t            width,
                  uint32_t            height,
                  uint64_t            storage,
                  struct error       *e)
{
	*dest = (struct vulkan_plan){0};
	struct work w = {0};
	enum error_code const code = build(dest, &w, width, height, storage, e);
	blob_fini(&w.probe_blob);
	blob_fini(&w.blob);
	dispatches_fini(&w.probe_lowered);
	dispatches_fini(&w.probed);
	dispatches_fini(&w.lowered);
	dispatches_fini(&w.merged);
	if (code)
		vulkan_plan_fini(dest);
	return code;
}

void
vulkan_plan_fini (struct vulkan_plan *plan)
{
	if (!plan)
		return;
	free(plan->steps);
	plan->steps = nullptr;
	free(plan->push);
	plan->push = nullptr;
	free(plan->segments);
	plan->segments = nullptr;
	free(plan->tables);
	plan->tables = nullptr;
	free(plan->values);
	plan->values = nullptr;
	free(plan->timeouts);
	plan->timeouts = nullptr;
	*plan = (struct vulkan_plan){0};
}

void
vulkan_plan_unclamp (struct vulkan_plan             *plan,
                     struct vulkan_clamp_free const *clamp_free)
{
	for (size_t i = 0; i < plan->step_count; ++i) {
		struct vulkan_step *const s = &plan->steps[i];
		for (size_t t = 0; t < VULKAN_PLAN_UNCLAMPED_COUNT; ++t)
			if (VULKAN_PLAN_UNCLAMPED[t].kernel == s->kernel) {
				if (clamp_free->heads[s->block] & 1)
					s->kernel = VULKAN_PLAN_UNCLAMPED[t].twin;
				break;
			}
	}
	// A run's records in the blob, a layer each from its first on.
	for (size_t i = 0; i < plan->step_count; ++i) {
		struct vulkan_step const *const s = &plan->steps[i];
		if (!vulkan_plan_persistent(s->kernel))
			continue;
		struct push_persist r;
		memcpy(&r, &plan->push[s->push], sizeof r);
		for (size_t g = 0; g < plan->segment_count; ++g) {
			struct vulkan_segment const *const records = &plan->segments[g];
			if (records->recipe != VULKAN_RECIPE_TABLE ||
			    records->offset != UINT64_C(4) * r.layers_off)
				continue;
			size_t const windows = records->index + offsetof(struct persist_rec, windows) / 4;
			for (uint32_t k = 0; k < r.n_layers; ++k) {
				uint32_t const heads = clamp_free->heads[ROWS[s->first + k].block];
				plan->tables[windows + k * (sizeof (struct persist_rec) / 4)] = heads;
			}
			break;
		}
	}
}
