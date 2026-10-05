/** @file
 *
 * The GPU codec: a frame's proxy encoded into the network's input, a pass's answer fed into the
 * next pass, and the last answer decoded into the proxy's format, by the linux_native module's
 * kernels on the network's stream. codec_gpu.c defines the functions.
 *
 * The caller keeps the HIP device current and the stream alive and idle between frames: a frame
 * ends before codec_gpu_encode() or after codec_gpu_finish().
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_CODEC_GPU_H_
#define DLSSLOP_AMD_BACKEND_CODEC_GPU_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "kernel_args.h"
#include "native_kernels.h"

/** @brief The codec's buffers and the latest encode.
 *
 * codec_gpu_init() makes one and codec_gpu_fini() frees it; a zeroed one is none.
 */
struct codec_gpu {
	struct native_kernels const *kernels;      //!< The kernels and their stream; nullptr for none.
	void                        *proxy;        //!< A host frame's upload, which decode overwrites, then reads back.
	void const                  *source;       //!< The latest encode's proxy: proxy, or the caller's frame.
	uint32_t                    *invalid;      //!< Pinned host status word; kernels only ever store 1.
	uint8_t                     *pinned[2];    //!< The channel's input and output slots, page-locked.
	size_t                       capacity;     //!< The bytes of proxy.
	size_t                       pinned_bytes; //!< The bytes locked in each slot; 0 for none.
	struct geometry              uploaded;     //!< The latest encode's geometry.
	bool                         fp16;         //!< The latest encode's proxy is RGBA16F, not RGBA8.
	bool                         device;       //!< The latest encode's frames are device memory.
	uint16_t                     pinning;      //!< 1 while the runtime can page-lock the slots, else 0.
};

/** @brief Makes a codec: its status word.
 *
 * @param dest    Receives the codec; zeroed on a failure.
 * @param kernels The kernels, which must outlive the codec.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
codec_gpu_init (struct codec_gpu            *dest,
                struct native_kernels const *kernels,
                struct error                *e);

/** @brief Waits for the stream, unlocks the slots, frees the codec's buffers and zeroes it.
 *
 * @param codec The codec, or nullptr.
 */
extern void
codec_gpu_fini (struct codec_gpu *codec);

/** @brief Serving: page-locks growing prefixes of the channel's frame slots, so that the encode
 *         and decode copies DMA straight from and to shared memory.
 *
 * The slots must outlive the codec; call only between frames. On a failure the slots stay
 * pageable, which costs HIP staging copies but remains correct.
 *
 * @param codec  The codec.
 * @param input  The input slot.
 * @param output The output slot.
 * @param bytes  The bytes of each to lock.
 */
extern void
codec_gpu_pin (struct codec_gpu *codec,
               uint8_t          *input,
               uint8_t          *output,
               size_t            bytes);

/** @brief Queues a frame's upload and encode on the stream.
 *
 * The input must stay unchanged until the decode is done (pageable input is staged). Input and
 * decode's output are host memory, copied through the codec's own proxy, or with @a device the
 * caller's device frames, which the kernels read and write.
 *
 * @param codec       The codec.
 * @param input       The proxy: g->source_width x g->source_height pixels of RGBA8 or RGBA16F.
 * @param g           The geometry, which geometry_validate() checks.
 * @param device_rgba Receives the network's input: g->width x g->height pixels of RGBA32F.
 * @param fp16        Whether the proxy is RGBA16F.
 * @param device      Whether the proxy and the answer are the caller's device memory.
 * @param e           Receives the words for what stopped it, or nullptr.
 * @return            ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
codec_gpu_encode (struct codec_gpu      *codec,
                  uint8_t const         *input,
                  struct geometry const *g,
                  void                  *device_rgba,
                  bool                   fp16,
                  bool                   device,
                  struct error          *e);

/** @brief Queues the feedback of a pass's raw RGB into the next pass's input at the latest
 *         encode's extent, without a host round trip or an RGBA8 conversion.
 *
 * The buffers must be distinct. Invalid samples stay recorded until codec_gpu_finish().
 *
 * @param codec       The codec.
 * @param neural_rgb  The pass's answer: RGB32F.
 * @param device_rgba Receives the next pass's input: RGBA32F.
 * @param precision16 Whether the feedback keeps binary16 precision rather than UNORM8.
 * @param e           Receives the words for what stopped it, or nullptr.
 * @return            ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
codec_gpu_feedback (struct codec_gpu *codec,
                    void             *neural_rgb,
                    void             *device_rgba,
                    bool              precision16,
                    struct error     *e);

/** @brief Queues the decode of the latest encode's answer, and with a host frame its readback.
 *
 * The output holds the proxy-sized answer once codec_gpu_finish() returns: the answer's RGB with
 * the proxy's alpha. A host frame's is written over the upload, then read back.
 *
 * @param codec      The codec.
 * @param neural_rgb The last pass's answer: RGB32F.
 * @param output     Receives the answer in the proxy's format and extent.
 * @param e          Receives the words for what stopped it, or nullptr.
 * @return           ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
codec_gpu_decode (struct codec_gpu *codec,
                  void             *neural_rgb,
                  uint8_t          *output,
                  struct error     *e);

/** @brief Waits for the stream. Nonfinite or FP16-overflow samples anywhere since the latest
 *         encode reject the frame, never becoming a reported answer.
 *
 * @param codec The codec.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, ERROR_FAILED, or ERROR_REJECTED for such a sample.
 */
extern enum error_code
codec_gpu_finish (struct codec_gpu *codec,
                  struct error     *e);

#endif /* DLSSLOP_AMD_BACKEND_CODEC_GPU_H_ */
