/** @file
 *
 * Everything about the Neural Rendering composition pass that is not Direct3D 12.
 *
 * The constants the shader reads live here, so a Vulkan implementation can share the struct rather
 * than redefine it and drift.
 *
 * The Direct3D 12 side is DlssNr_Dx12, which implements Shader_Dx12 the way RCAS and Output Scaling
 * do. The model itself is separate again: creating and evaluating an NGX feature is not a dispatch,
 * so it does not belong in a shader class.
 *
 * Plain C API, consumable from C++.
 */
#ifndef DLSSLOP_AMD_LAYER_DLSSNR_DLSSNR_COMMON_H_
#define DLSSLOP_AMD_LAYER_DLSSNR_DLSSNR_COMMON_H_

#ifdef __cplusplus
# include <cstdint>
# define DLSS_NR_STD(x) std::x
extern "C" {
#else
# include <stdint.h>
# define DLSS_NR_STD(x) x
#endif

/** @brief Which of the passes a dispatch is.
 *
 * One shader, because they read and write the same set of resources and differ only in what they
 * compute.
 */
enum dlss_nr_mode : DLSS_NR_STD(uint32_t) {
	DLSS_NR_MODE_ENCODE = 0,     //!< the frame -> a tone-mapped proxy, plus an untouched copy
	DLSS_NR_MODE_RESOLVE = 1,    //!< proxy + the model's answer + the untouched copy -> the edited frame
	DLSS_NR_MODE_DOWNSAMPLE = 2, //!< the proxy -> a smaller proxy, when the model works below full size
	DLSS_NR_MODE_METER = 3,      //!< the exposure texture -> tile (0,0), for the white point
	DLSS_NR_MODE_CALIBRATE = 4   //!< the untouched frame -> a grid of tile peak luminances
};

/** @brief The meter's grid. 64 x 64 tiles over the whole frame, whatever its size.
 *
 * Tiles rather than pixels because the number wanted is where white sits, not how bright the
 * brightest pixel is: a single specular hit or a sky pixel is not the white point, and a frame's
 * maximum is exactly the statistic that would be dominated by one. Averaging each tile first means
 * anything smaller than a four-thousandth of the frame cannot decide the answer on its own.
 *
 * 4096 values is also small enough to read back and take a real percentile of on the CPU, rather
 * than approximating one on the GPU.
 */
static constexpr DLSS_NR_STD(uint32_t) DLSS_NR_METER_GRID = 64;

/** @brief What the composition shader reads.
 *
 * The model does not replace the frame. It is shown a tone-mapped proxy of the picture, and its
 * answer is transferred back onto the real frame -- so most of these describe how much of that answer
 * to take, not what the model should do.
 *
 * Aligned to 256 because a constant buffer view's size must be a multiple of it. Without this the
 * buffer is created at the struct's natural size, the view is invalid, and the device is removed a
 * few milliseconds later -- with nothing in any log to say why. Every other shader here does the
 * same thing; it is not optional. The first member carries the alignment, which C and C++ both
 * allow there.
 */
struct dlss_nr_constants {
	alignas(256) DLSS_NR_STD(uint32_t) mode;
	float                 white_point;

	DLSS_NR_STD(uint32_t) width;
	DLSS_NR_STD(uint32_t) height;

	// How much of the model's edit lands, and how much of it is allowed to be colour rather than
	// luminance. Separating the two is what keeps saturated highlights from shifting hue.
	float                 transfer_strength;
	float                 colour_strength;

	DLSS_NR_STD(uint32_t) debug_view;

	// A ceiling on how far a pixel may be brightened. The transfer is a ratio, and a ratio against a
	// near-black proxy pixel is unbounded without one.
	float                 max_ratio;

	// Set when the game's buffer is already tone-mapped, in which case there is nothing to convert
	// and the transfer is the identity.
	DLSS_NR_STD(uint32_t) passthrough;

	float                 mv_scale_x;
	float                 mv_scale_y;

	// Depth and motion vectors come from the upscaler's inputs and so may be at render resolution
	// while colour and output are at display resolution.
	DLSS_NR_STD(uint32_t) guide_width;
	DLSS_NR_STD(uint32_t) guide_height;

	// Showing the pass against itself. 0 off, 1 side by side, 2 a wipe.
	//
	// Both are drawn by the resolve rather than by a pass of their own, because the resolve is the
	// one place that already holds the frame as the upscaler produced it and the frame the model
	// edited. Comparing them anywhere else would mean keeping a second copy of one of them.
	DLSS_NR_STD(uint32_t) compare_mode;
	float                 compare_split;

	// How much of the frame side by side shows. 1 fits the whole thing at its right shape and
	// letterboxes what is left over; 2 fills the half and crops to the middle instead.
	float                 compare_zoom;

	// Which side the edited frame is on. Swapping matters because the eye is not even-handed about
	// left and right, so a difference can look like an improvement purely from where it sits.
	DLSS_NR_STD(uint32_t) compare_swap;

	// How a model that worked below the frame's size is brought back. 0 classic, 1 matched residual,
	// 2 native + edit.
	//
	// Classic composes the model's own low-resolution picture against the full-resolution frame, so
	// the two disagree by the blur the downsample introduced as well as by the edit -- and the
	// composition reads that disagreement as headroom the frame has and the model never saw. Matched
	// residual takes only the model's *difference* from low resolution and lays it on the frame's own
	// full-resolution proxy, so the two pictures being compared are at the same scale and the only
	// thing carried up from small is the edit itself. Native + edit composes nothing: the frame's own
	// pixels are the result and only the model's difference is added to them, so what the model left
	// alone never passes through the enlargement. A luminance guard bounds the sum.
	//
	// The idea and the cube-scaled residual are hhkbble's, from the multi-pass PR against this fork.
	// Native + edit is the technique from xenmods' DLSSNR-Cost-Scaler (MIT).
	DLSS_NR_STD(uint32_t) transfer;

	// What the debug views are multiplied by on their way out.
	//
	// They have to be scaled into the frame's units or the game's tonemapper shows them wrong, but
	// scaling them by the live white point makes the instrument move with the thing being measured:
	// two captures at different exposures then differ by the exposure, whatever the edit did. This
	// is the user's own multiplier, which holds still while the meter works.
	float                 debug_scale;

	// The reversible-proxy mode. 0 soft knee + our composition (default), 1 unclipped Neutwo proxy +
	// our composition, 2 Neutwo proxy + pure-inverse replace (model's answer straight back, no
	// composition). Trailing field, mirroring the shader's cbuffer, so the layout stays a flat run of
	// 4-byte scalars that C++ and HLSL agree on.
	DLSS_NR_STD(uint32_t) reversible_mode;

	// 0 = output the clean upscaler frame (the pass still runs, so Hold frame keeps a frozen frame
	// to A/B against), 1 = apply the model's edit. Trailing scalar, mirrored in the shader cbuffer.
	DLSS_NR_STD(uint32_t) apply_model;

	// D3D12 source-1 zero-latency exposure. use_game_exposure = 1 makes the shader read the game's
	// live exposure texture (bound at t4) instead of the CPU-resolved white point; exposure_pre_mul
	// is preExposure * trim, so the live white point is exposure_pre_mul / exposure. Mirrored in the
	// cbuffer.
	DLSS_NR_STD(uint32_t) use_game_exposure;
	float                 exposure_pre_mul;

	// The float16 HDR proxy. hdr_proxy = 1 makes the encode write linear light normalised by the white
	// point into a float16 surface -- no knee, no sRGB, no ceiling -- and the resolve reads it back
	// without the sRGB decode. hdr_transfer = 1 says the swapchain itself carries PQ, so the frame is
	// PQ-decoded on the way in and PQ-encoded on the way out. Both zero keeps the SDR path
	// byte-identical. Trailing, mirroring the cbuffer.
	// 2 selects native HIP encoded FP16: same color domain as RGBA8, finer precision.
	DLSS_NR_STD(uint32_t) hdr_proxy;
	// Native HIP also uses this PQ conversion with its display-encoded proxy.
	DLSS_NR_STD(uint32_t) hdr_transfer;

	// How much of the chroma-agreement gate to apply, 0..1. See colourTrustPercent.
	float                 colour_trust;

	// How much of the relighting ratio comes from the neighbourhood. See ratioSmoothPercent.
	float                 ratio_smooth;
};

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef DLSS_NR_STD

#endif /* DLSSLOP_AMD_LAYER_DLSSNR_DLSSNR_COMMON_H_ */
