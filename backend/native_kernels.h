/** @file
 *
 * The linux_native module's kernels, from codec_gpu.hip, tuning_gpu.hip, color_gpu.hip and
 * temporal_gpu.hip: the module, loaded once, and its kernels, launched on one stream (the HIP
 * network's) as one-dimensional grids of 256-thread groups. native_kernels.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_NATIVE_KERNELS_H_
#define DLSSLOP_AMD_BACKEND_NATIVE_KERNELS_H_

#include <stdint.h>

#include "error.h"
#include "hip.h"
#include "kernel_args.h"

/** @brief The module's kernels. */
enum native_kernel : uint8_t {
	NATIVE_KERNEL_ENCODE_RGBA8,     //!< dlsslop_encode_rgba8.
	NATIVE_KERNEL_ENCODE_RGBA16F,   //!< dlsslop_encode_rgba16f.
	NATIVE_KERNEL_FEEDBACK_RGB,     //!< dlsslop_feedback_rgb.
	NATIVE_KERNEL_DECODE_RGBA8,     //!< dlsslop_decode_rgba8.
	NATIVE_KERNEL_DECODE_RGBA16F,   //!< dlsslop_decode_rgba16f.
	NATIVE_KERNEL_TUNE_RGB,         //!< dlsslop_tune_rgb.
	NATIVE_KERNEL_PRESERVE_COLOR,   //!< dlsslop_preserve_color.
	NATIVE_KERNEL_TEMPORAL_LUMA,    //!< dlsslop_temporal_luma.
	NATIVE_KERNEL_TEMPORAL_REDUCE,  //!< dlsslop_temporal_reduce.
	NATIVE_KERNEL_TEMPORAL_FLOW,    //!< dlsslop_temporal_flow.
	NATIVE_KERNEL_TEMPORAL_WARP,    //!< dlsslop_temporal_warp.
	NATIVE_KERNEL_TEMPORAL_CUT,     //!< dlsslop_temporal_cut.
	NATIVE_KERNEL_COUNT             //!< The number of kernels; no kernel.
};

/** @brief The module and its kernels, launched on a stream.
 *
 * native_kernels_init() loads one and native_kernels_fini() unloads it; a zeroed one is not loaded.
 */
struct native_kernels {
	struct hip_api const *api;                            //!< The runtime.
	void                 *stream;                         //!< The stream every kernel runs on.
	void                 *module;                         //!< The module; nullptr when none is loaded.
	void                 *functions[NATIVE_KERNEL_COUNT]; //!< Each kernel, by enum native_kernel.
};

/** @brief Loads the module and finds every kernel in it.
 *
 * @param dest   Receives the kernels; zeroed on a failure.
 * @param api    The runtime, which must outlive them.
 * @param stream The stream to launch them on, which must outlive them.
 * @param path   The module's file.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
native_kernels_init (struct native_kernels *dest,
                     struct hip_api const  *api,
                     void                  *stream,
                     char const            *path,
                     struct error          *e);

/** @brief Waits for the stream, unloads the module and zeroes the kernels.
 *
 * @param kernels The kernels, or nullptr.
 */
extern void
native_kernels_fini (struct native_kernels *kernels);

/** @brief Queues a kernel with one thread per item.
 *
 * @param kernels The kernels.
 * @param kernel  The kernel.
 * @param count   The items.
 * @param args    Pointers to the kernel's parameters, which the runtime copies as it queues it.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
native_kernels_launch (struct native_kernels const *kernels,
                       enum native_kernel           kernel,
                       unsigned                     count,
                       void                       **args,
                       struct error                *e);

/** @brief Queues the native tuning of one pass's raw RGB after its inference.
 *
 * The caller keeps the three buffers until the stream reaches the kernel. It reads neighbours, so
 * the output is distinct from both inputs.
 *
 * @param kernels    The kernels.
 * @param g          The geometry.
 * @param input_rgba The pass's input: g.width x g.height pixels of RGBA32F.
 * @param raw_rgb    The pass's answer: g.width x g.height pixels of RGB32F.
 * @param output_rgb Receives the tuned answer, in the same layout.
 * @param tuning     The tuning.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
native_kernels_tune (struct native_kernels const *kernels,
                     struct geometry              g,
                     void const                  *input_rgba,
                     void const                  *raw_rgb,
                     void                        *output_rgb,
                     struct native_tuning         tuning,
                     struct error                *e);

/** @brief Queues the colour preservation of one pass against the frame's encoded input (the
 *         reference), with the same buffer rules as native_kernels_tune().
 *
 * @param kernels       The kernels.
 * @param g             The geometry.
 * @param original_rgba The frame's encoded input: g.width x g.height pixels of RGBA32F.
 * @param raw_rgb       The pass's answer: g.width x g.height pixels of RGB32F.
 * @param output_rgb    Receives the corrected answer, in the same layout.
 * @param strength      The correction's strength, in 0..1.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
native_kernels_preserve_color (struct native_kernels const *kernels,
                               struct geometry              g,
                               void const                  *original_rgba,
                               void const                  *raw_rgb,
                               void                        *output_rgb,
                               float                        strength,
                               struct error                *e);

#endif /* DLSSLOP_AMD_BACKEND_NATIVE_KERNELS_H_ */
