/** @file
 *
 * The parameters that the host passes by value to the linux_native module's kernels, and the
 * element of their flow buffers: the types that the kernels' math (geometry.hpp, codec_math.hpp,
 * tuning_math.hpp, temporal_math.hpp) takes. The SDKless HIP modules have no system headers, so
 * this header includes nothing and holds only types, macros and static assertions, which C and C++
 * (HIP included) read alike. The kernels read the parameters from their argument segment and the
 * flow from global memory, each struct at its C layout with 32-bit members; the assertions pin the
 * sizes.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_KERNEL_ARGS_H_
#define DLSSLOP_AMD_BACKEND_KERNEL_ARGS_H_

/** @brief Where a source picture sits in the network's padded raster.
 *
 * The picture is fitted, aspect kept, into the tier's 16:9 viewport, which the raster pads below.
 */
struct geometry {
	unsigned source_width;  //!< The source picture's width.
	unsigned source_height; //!< Its height.
	unsigned width;         //!< The padded neural processing extent's width.
	unsigned height;        //!< Its height.
	unsigned valid_height;  //!< The viewport's height, unpadded; the viewport spans the full width.
	unsigned x;             //!< The fitted picture's left edge in the raster.
	unsigned y;             //!< Its top edge.
	unsigned fit_width;     //!< Its width.
	unsigned fit_height;    //!< Its height.
};

static_assert(sizeof (struct geometry) == 36);

/** @brief The native tuning of a pass's RGB residual: dlsslop-amd's own filters, not NVIDIA NGX
 *         model conditioning or semantic masks.
 *
 * NATIVE_TUNING_DEFAULTS initializes one; at the defaults the network's answer stays unchanged,
 * bit for bit.
 */
struct native_tuning {
	float intensity; //!< The scale of the whole edit, 0..4.
	float tone;      //!< The scale of the residual's low-pass part, 0..4.
	float structure; //!< The scale of the rest of the residual, 0..4.
	float sharpness; //!< How much of the model's own high-pass is added, 0..1.
};

static_assert(sizeof (struct native_tuning) == 16);

#define NATIVE_TUNING_INTENSITY 1.0f //!< The default intensity.
#define NATIVE_TUNING_TONE      1.0f //!< The default tone.
#define NATIVE_TUNING_STRUCTURE 1.0f //!< The default structure.
#define NATIVE_TUNING_SHARPNESS 0.0f //!< The default sharpness.

/** @brief The defaults as an initializer, in member order: the HIP modules are C++17, which has no
 *         designated initializers. */
#define NATIVE_TUNING_DEFAULTS {NATIVE_TUNING_INTENSITY, NATIVE_TUNING_TONE, NATIVE_TUNING_STRUCTURE, \
                                NATIVE_TUNING_SHARPNESS}

/** @brief The extent of a luma image or of a flow grid. */
struct temporal_extent {
	unsigned width;  //!< The width.
	unsigned height; //!< The height.
};

static_assert(sizeof (struct temporal_extent) == 8);

/** @brief One vector of a flow grid: where its pixel was in the previous frame, and how well it
 *         matched there. */
struct temporal_flow {
	float x;     //!< The horizontal displacement, in pixels.
	float y;     //!< The vertical displacement, in pixels.
	float error; //!< The match's mean absolute luma difference.
};

static_assert(sizeof (struct temporal_flow) == 12);

/** @brief One pyramid level's flow search (dlsslop_temporal_flow). */
struct temporal_search {
	struct temporal_extent image;       //!< The level's luma extent.
	struct temporal_extent grid;        //!< The level's flow grid.
	struct temporal_extent coarse_grid; //!< The next coarser level's flow grid.
	unsigned               step;        //!< Pixels per grid vector on each axis.
	unsigned               radius;      //!< The search window's radius around its start.
	unsigned               patch;       //!< The compared patches' radius, 1 or 2.
	unsigned               has_coarse;  //!< Nonzero: the search starts from the coarser level's flow.
	unsigned               final_level; //!< Nonzero: the finest level, which refines below a pixel.
};

static_assert(sizeof (struct temporal_search) == 44);

/** @brief The geometry of the history warp and of the scene cut check (dlsslop_temporal_warp,
 *         dlsslop_temporal_cut). */
struct temporal_warp {
	/** @brief The finest level's extent: the raster's width and the viewport's height. */
	struct temporal_extent image;
	struct temporal_extent grid;          //!< The finest level's flow grid.
	unsigned               padded_height; //!< The raster's height, padding included.
	unsigned               step;          //!< Pixels per grid vector on each axis.
	unsigned               x;             //!< The fitted picture's left edge in the raster.
	unsigned               y;             //!< Its top edge.
	unsigned               fit_width;     //!< Its width.
	unsigned               fit_height;    //!< Its height.
};

static_assert(sizeof (struct temporal_warp) == 40);

#endif /* DLSSLOP_AMD_BACKEND_KERNEL_ARGS_H_ */
