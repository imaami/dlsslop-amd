/** @file
 *
 * The CPU references' pixel loops: the HIP kernels' own math (codec_math.hpp, tuning_math.hpp,
 * color_preserve_math.hpp) run on the host, and the motion kernels' sampler (temporal_math.hpp).
 * kernel_math.cpp, which includes that math, defines the functions; reference.h checks the loops'
 * arguments and words their failures.
 *
 * A loop writes into a caller's buffer of the size its function states, which must not overlap its
 * inputs. It returns false at the first sample that is not finite, its output then incomplete.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_KERNEL_MATH_H_
#define DLSSLOP_AMD_BACKEND_KERNEL_MATH_H_

#ifdef __cplusplus
# include <cstdint>
#else
# include <stdint.h>
#endif

#include "kernel_args.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief Encodes a source picture into the network's padded proxy, as codec_gpu.hip does.
 *
 * Each RGB sample is rounded through binary16, as upstream's RGBA16_FLOAT encoding texture rounds
 * it, and alpha is one. A source no larger than the fit is sampled bilinearly, a larger one averaged
 * over each pixel's footprint. The padding below the viewport reflects it.
 *
 * @param source The source: g->source_width x g->source_height tightly packed pixels of SDR sRGB
 *               RGBA8, or with @a fp16 of native-endian RGBA16F, neither decoded nor clamped.
 * @param g      The geometry, which geometry_validate() accepts.
 * @param fp16   Whether the source is RGBA16F.
 * @param rgba   Receives the proxy: g->width x g->height pixels of RGBA32F.
 * @return       false if a sample of an FP16 source was not finite.
 */
extern bool
kernel_math_encode_proxy (STD(uint8_t) const    *source,
                          struct geometry const *g,
                          bool                   fp16,
                          float                 *rgba);

/** @brief Feeds a network's answer into another pass as its proxy, as codec_gpu.hip does.
 *
 * The fitted picture's RGB is rounded through binary16 without being resampled, and with
 * @a precision16 false clamped and rounded to UNORM8 first. Alpha is one, the bars are black and the
 * padding below the viewport reflects it.
 *
 * @param neural_rgb  The answer: g->width x g->height pixels of RGB32F.
 * @param g           The geometry, which geometry_validate() accepts.
 * @param precision16 Whether the feedback keeps binary16 precision rather than UNORM8.
 * @param rgba        Receives the proxy: g->width x g->height pixels of RGBA32F.
 * @return            false if a sample of the fitted picture was not finite or, rounded, not a
 *                    finite binary16.
 */
extern bool
kernel_math_feedback_neural_rgb (float const           *neural_rgb,
                                 struct geometry const *g,
                                 bool                   precision16,
                                 float                 *rgba);

/** @brief Decodes a network's answer into the source's extent and format, as codec_gpu.hip does.
 *
 * The fitted picture is resampled bilinearly through upstream's binary16 neural surface, then
 * clamped and rounded to UNORM8, or with @a fp16 rounded to binary16, keeping signed values and those
 * above one. Alpha is the original's.
 *
 * @param original   The source that was encoded, in its format.
 * @param g          The geometry, which geometry_validate() accepts.
 * @param fp16       Whether the source is RGBA16F.
 * @param neural_rgb The answer: g->width x g->height pixels of RGB32F.
 * @param output     Receives the decoded picture: as many bytes as @a original holds.
 * @return           false if a sample that the resampling read was not a finite binary16.
 */
extern bool
kernel_math_decode_neural_proxy (STD(uint8_t) const    *original,
                                 struct geometry const *g,
                                 bool                   fp16,
                                 float const           *neural_rgb,
                                 STD(uint8_t)          *output);

/** @brief Tunes a pass's answer, as tuning_gpu.hip does.
 *
 * @param input_rgba The pass's input: g->width x g->height pixels of RGBA32F.
 * @param raw_rgb    The pass's answer: g->width x g->height pixels of RGB32F.
 * @param g          The geometry; the fitted picture lies inside the padded extent.
 * @param tuning     The tuning, which tuning_validate() accepts.
 * @param output     Receives the tuned answer: g->width x g->height pixels of RGB32F.
 * @return           false if a tuned sample was not finite.
 */
extern bool
kernel_math_tune_neural_rgb (float const                *input_rgba,
                             float const                *raw_rgb,
                             struct geometry const      *g,
                             struct native_tuning const *tuning,
                             float                      *output);

/** @brief Corrects the chroma drift of a pass's answer against the frame's encoded input, as
 *         color_gpu.hip does.
 *
 * @param original_rgba The frame's encoded input: g->width x g->height pixels of RGBA32F.
 * @param model_rgb     The pass's answer: g->width x g->height pixels of RGB32F.
 * @param g             The geometry; the fitted picture lies inside the padded extent.
 * @param strength      The correction's strength, in 0..1.
 * @param output        Receives the corrected answer: g->width x g->height pixels of RGB32F.
 * @return              false if a corrected sample was not finite.
 */
extern bool
kernel_math_preserve_color (float const           *original_rgba,
                            float const           *model_rgb,
                            struct geometry const *g,
                            float                  strength,
                            float                 *output);

/** @brief Samples a luma image bilinearly, its point clamped into the image, as temporal_gpu.hip
 *         samples a pyramid level.
 *
 * @param image The image: e.width x e.height floats.
 * @param e     Its extent.
 * @param x     The point's column, in pixels.
 * @param y     Its row.
 * @return      The sample.
 */
extern float
kernel_math_temporal_sample (float const            *image,
                             struct temporal_extent  e,
                             float                   x,
                             float                   y);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_BACKEND_KERNEL_MATH_H_ */
