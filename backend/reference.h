/** @file
 *
 * The CPU references of the daemon's kernels: the codec, the native tuning and the color
 * preservation, which the HIP self-test checks the GPU against and --cpu-codec serves with. Each
 * checks its arguments, then runs its kernel's own math through kernel_math.h. reference.c defines
 * the functions.
 *
 * Each writes into a caller's buffer of the size it states, which must not overlap its inputs. On
 * failure the buffer's contents are unspecified.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_REFERENCE_H_
#define DLSSLOP_AMD_BACKEND_REFERENCE_H_

#include <stdint.h>

#include "error.h"
#include "kernel_args.h"

/** @brief Encodes a source picture into the network's padded proxy (kernel_math_encode_proxy()).
 *
 * @param source The source: g->source_width x g->source_height tightly packed pixels of SDR sRGB
 *               RGBA8, or with @a fp16 of native-endian RGBA16F.
 * @param g      The geometry, as geometry_init() makes it.
 * @param fp16   Whether the source is RGBA16F.
 * @param rgba   Receives the proxy: g->width x g->height pixels of RGBA32F.
 * @param e      Receives the words of a failure, or nullptr.
 * @return       ERROR_NONE; ERROR_FAILED for an inconsistent geometry or a null source;
 *               ERROR_REJECTED for an FP16 source with a sample that is not finite.
 */
extern enum error_code
reference_encode_proxy (uint8_t const         *source,
                        struct geometry const *g,
                        bool                   fp16,
                        float                 *rgba,
                        struct error          *e);

/** @brief Feeds a network's answer into another pass as its proxy
 *         (kernel_math_feedback_neural_rgb()).
 *
 * @param neural_rgb  The answer: g->width x g->height pixels of RGB32F.
 * @param g           The geometry, as geometry_init() makes it.
 * @param precision16 Whether the feedback keeps binary16 precision rather than UNORM8.
 * @param rgba        Receives the proxy: g->width x g->height pixels of RGBA32F.
 * @param e           Receives the words of a failure, or nullptr.
 * @return            ERROR_NONE, or ERROR_FAILED for an inconsistent geometry, a null answer or a
 *                    sample of the fitted picture that is not finite or past binary16.
 */
extern enum error_code
reference_feedback_neural_rgb (float const           *neural_rgb,
                               struct geometry const *g,
                               bool                   precision16,
                               float                 *rgba,
                               struct error          *e);

/** @brief Decodes a network's answer into the source's extent and format
 *         (kernel_math_decode_neural_proxy()).
 *
 * @param original   The source that was encoded, in its format.
 * @param g          The geometry, as geometry_init() makes it.
 * @param fp16       Whether the source is RGBA16F.
 * @param neural_rgb The answer: g->width x g->height pixels of RGB32F.
 * @param output     Receives the decoded picture: as many bytes as @a original holds,
 *                   g->source_width x g->source_height pixels of 4 bytes, or with @a fp16 of 8.
 * @param e          Receives the words of a failure, or nullptr.
 * @return           ERROR_NONE; ERROR_FAILED for an inconsistent geometry or a null image;
 *                   ERROR_REJECTED for an answer that the resampling reads a sample of that is not
 *                   a finite binary16.
 */
extern enum error_code
reference_decode_neural_proxy (uint8_t const         *original,
                               struct geometry const *g,
                               bool                   fp16,
                               float const           *neural_rgb,
                               uint8_t               *output,
                               struct error          *e);

/** @brief Tunes a pass's answer (kernel_math_tune_neural_rgb()).
 *
 * @param input_rgba The pass's input: g->width x g->height pixels of RGBA32F.
 * @param raw_rgb    The pass's answer: g->width x g->height pixels of RGB32F.
 * @param g          The geometry: a fitted picture inside a padded extent of at most 16384 per
 *                   dimension.
 * @param tuning     The tuning.
 * @param output     Receives the tuned answer: g->width x g->height pixels of RGB32F.
 * @param e          Receives the words of a failure, or nullptr.
 * @return           ERROR_NONE; ERROR_REJECTED for a tuning that tuning_validate() rejects;
 *                   ERROR_FAILED for an invalid geometry, a null input, an output that is an input
 *                   or a tuned sample that is not finite.
 */
extern enum error_code
reference_tune_neural_rgb (float const                *input_rgba,
                           float const                *raw_rgb,
                           struct geometry const      *g,
                           struct native_tuning const *tuning,
                           float                      *output,
                           struct error               *e);

/** @brief Corrects the chroma drift of a pass's answer against the frame's encoded input
 *         (kernel_math_preserve_color()).
 *
 * @param original_rgba The frame's encoded input: g->width x g->height pixels of RGBA32F.
 * @param model_rgb     The pass's answer: g->width x g->height pixels of RGB32F.
 * @param g             The geometry: a fitted picture inside a padded extent of at most 16384 per
 *                      dimension.
 * @param strength      The correction's strength.
 * @param output        Receives the corrected answer: g->width x g->height pixels of RGB32F.
 * @param e             Receives the words of a failure, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED for a strength that is not finite and within
 *                      0..1, a null input, an invalid geometry, an output that is an input or a
 *                      corrected sample that is not finite.
 */
extern enum error_code
reference_preserve_color (float const           *original_rgba,
                          float const           *model_rgb,
                          struct geometry const *g,
                          float                  strength,
                          float                 *output,
                          struct error          *e);

#endif /* DLSSLOP_AMD_BACKEND_REFERENCE_H_ */
