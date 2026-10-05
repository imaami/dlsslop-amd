/** @file
 *
 * The GPU codec: codec_gpu.h.
 */
// SPDX-License-Identifier: MIT
#include <stdio.h>

#include "codec_gpu.h"
#include "geometry.h"

/** @brief A proxy precision: its kernels and bytes per pixel. */
struct format {
	enum native_kernel encode; //!< The encoder.
	enum native_kernel decode; //!< The decoder.
	uint8_t            bytes;  //!< The bytes of a pixel.
};

/** @brief RGBA8 and RGBA16F, by codec_gpu::fp16. */
static struct format const formats[2] = {
	{NATIVE_KERNEL_ENCODE_RGBA8, NATIVE_KERNEL_DECODE_RGBA8, 4},
	{NATIVE_KERNEL_ENCODE_RGBA16F, NATIVE_KERNEL_DECODE_RGBA16F, 8},
};

/** @brief Unlocks the slots that codec_gpu_pin() locked.
 *
 * @param codec The codec.
 */
static void
unpin (struct codec_gpu *codec)
{
	if (codec->pinned_bytes) {
		for (size_t i = 0; i < sizeof codec->pinned / sizeof *codec->pinned; ++i)
			codec->kernels->api->hipHostUnregister(codec->pinned[i]);
	}
	codec->pinned_bytes = 0;
}

/** @brief Grows the proxy upload's buffer to hold some bytes, once the stream is done with it.
 *
 * @param codec The codec.
 * @param bytes The bytes.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
reserve (struct codec_gpu *codec,
         size_t            bytes,
         struct error     *e)
{
	if (bytes <= codec->capacity)
		return ERROR_NONE;
	struct hip_api const *const api = codec->kernels->api;
	int result = api->hipStreamSynchronize(codec->kernels->stream);
	if (result)
		return hip_fail(e, api, result, "codec resize synchronize");
	void *proxy = nullptr;
	result = api->hipMalloc(&proxy, bytes);
	if (result)
		return hip_fail(e, api, result, "allocate codec proxy");
	if (codec->proxy)
		api->hipFree(codec->proxy);
	codec->proxy = proxy;
	codec->capacity = bytes;
	return ERROR_NONE;
}

enum error_code
codec_gpu_init (struct codec_gpu            *dest,
                struct native_kernels const *kernels,
                struct error                *e)
{
	*dest = (struct codec_gpu){0};
	struct hip_api const *const api = kernels->api;
	if (!api->hipHostFree)
		return error_fail(e, "missing HIP export hipHostFree");
	void *invalid = nullptr;
	int const result = api->hipHostMalloc(&invalid, sizeof *dest->invalid, 0);
	if (result)
		return hip_fail(e, api, result, "allocate codec status");
	*dest = (struct codec_gpu){
		.kernels = kernels,
		.invalid = invalid,
		.pinning = api->hipHostRegister && api->hipHostUnregister,
	};
	return ERROR_NONE;
}

void
codec_gpu_fini (struct codec_gpu *codec)
{
	if (!codec || !codec->kernels)
		return;
	struct hip_api const *const api = codec->kernels->api;
	api->hipStreamSynchronize(codec->kernels->stream);
	unpin(codec);
	if (codec->proxy) {
		api->hipFree(codec->proxy);
		codec->proxy = nullptr;
	}
	api->hipHostFree(codec->invalid);
	codec->invalid = nullptr;
	*codec = (struct codec_gpu){0};
}

void
codec_gpu_pin (struct codec_gpu *codec,
               uint8_t          *input,
               uint8_t          *output,
               size_t            bytes)
{
	if (!codec->pinning ||
	    (bytes <= codec->pinned_bytes && input == codec->pinned[0] && output == codec->pinned[1]))
		return;
	unpin(codec);
	codec->pinned[0] = input;
	codec->pinned[1] = output;
	struct hip_api const *const api = codec->kernels->api;
	if (!api->hipHostRegister(input, bytes, 0)) {
		if (!api->hipHostRegister(output, bytes, 0)) {
			codec->pinned_bytes = bytes;
			return;
		}
		api->hipHostUnregister(input);
	}
	codec->pinning = 0;
	fprintf(stderr, "shared-memory pinning unavailable; using staged HIP transfers\n");
}

enum error_code
codec_gpu_encode (struct codec_gpu      *codec,
                  uint8_t const         *input,
                  struct geometry const *g,
                  void                  *device_rgba,
                  bool                   fp16,
                  bool                   device,
                  struct error          *e)
{
	enum error_code const code = geometry_validate(g, e);
	if (code)
		return code;
	if (!device_rgba)
		return error_fail(e, "null GPU encode output");
	struct format const *const format = &formats[fp16];
	codec->fp16 = fp16;
	codec->source = input;
	codec->device = device;
	if (!device) {
		size_t const bytes = (size_t)g->source_width * g->source_height * format->bytes;
		enum error_code const reserved = reserve(codec, bytes, e);
		if (reserved)
			return reserved;
		struct native_kernels const *const kernels = codec->kernels;
		int const result = kernels->api->hipMemcpyAsync(codec->proxy, input, bytes, 1, kernels->stream);
		if (result)
			return hip_fail(e, kernels->api, result, "upload codec proxy");
		codec->source = codec->proxy;
	}
	*codec->invalid = 0;
	codec->uploaded = *g;
	void *args[] = {&codec->source, &device_rgba, &codec->invalid, &codec->uploaded};
	return native_kernels_launch(codec->kernels, format->encode, g->width * g->height, args, e);
}

enum error_code
codec_gpu_feedback (struct codec_gpu *codec,
                    void             *neural_rgb,
                    void             *device_rgba,
                    bool              precision16,
                    struct error     *e)
{
	if (!codec->source || !neural_rgb || !device_rgba || neural_rgb == device_rgba)
		return error_fail(e, "GPU feedback without an encode or distinct buffers");
	uint32_t precision = precision16;
	void *args[] = {&neural_rgb, &device_rgba, &codec->invalid, &codec->uploaded, &precision};
	return native_kernels_launch(codec->kernels, NATIVE_KERNEL_FEEDBACK_RGB,
	                             codec->uploaded.width * codec->uploaded.height, args, e);
}

enum error_code
codec_gpu_decode (struct codec_gpu *codec,
                  void             *neural_rgb,
                  uint8_t          *output,
                  struct error     *e)
{
	if (!codec->source || !neural_rgb)
		return error_fail(e, "GPU decode without an encode");
	struct format const *const format = &formats[codec->fp16];
	unsigned const pixels = codec->uploaded.source_width * codec->uploaded.source_height;
	void *target = codec->device ? output : codec->proxy;
	void *args[] = {&codec->source, &target, &neural_rgb, &codec->invalid, &codec->uploaded};
	enum error_code const code = native_kernels_launch(codec->kernels, format->decode, pixels, args, e);
	if (code || codec->device)
		return code;
	struct native_kernels const *const kernels = codec->kernels;
	int const result = kernels->api->hipMemcpyAsync(output, codec->proxy, (size_t)pixels * format->bytes, 2,
	                                                kernels->stream);
	if (result)
		return hip_fail(e, kernels->api, result, "read codec proxy");
	return ERROR_NONE;
}

enum error_code
codec_gpu_finish (struct codec_gpu *codec,
                  struct error     *e)
{
	struct native_kernels const *const kernels = codec->kernels;
	int const result = kernels->api->hipStreamSynchronize(kernels->stream);
	if (result)
		return hip_fail(e, kernels->api, result, "codec decode completion");
	if (*codec->invalid)
		return error_reject(e, "proxy input, neural feedback or output contains nonfinite or FP16-overflow "
		                    "samples");
	return ERROR_NONE;
}
