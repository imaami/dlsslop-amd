/** @file
 *
 * The linux_native module's kernels: native_kernels.h.
 */
// SPDX-License-Identifier: MIT
#include "native_kernels.h"

/** @brief Each kernel's symbol, by enum native_kernel. */
static char const *const names[] = {
	"dlsslop_encode_rgba8", "dlsslop_encode_rgba16f", "dlsslop_feedback_rgb", "dlsslop_decode_rgba8",
	"dlsslop_decode_rgba16f", "dlsslop_tune_rgb", "dlsslop_preserve_color", "dlsslop_temporal_luma",
	"dlsslop_temporal_reduce", "dlsslop_temporal_flow", "dlsslop_temporal_warp", "dlsslop_temporal_cut",
};
static_assert(sizeof names / sizeof *names == NATIVE_KERNEL_COUNT, "a symbol per kernel");

enum error_code
native_kernels_init (struct native_kernels *dest,
                     struct hip_api const  *api,
                     void                  *stream,
                     char const            *path,
                     struct error          *e)
{
	*dest = (struct native_kernels){0};
	struct native_kernels kernels = {.api = api, .stream = stream};
	enum error_code const code = hip_load_module(api, path, &kernels.module, e);
	if (code)
		return code;
	for (unsigned k = 0; k < NATIVE_KERNEL_COUNT; ++k) {
		void *function = nullptr;
		int const result = api->hipModuleGetFunction(&function, kernels.module, names[k]);
		if (result) {
			api->hipModuleUnload(kernels.module);
			return hip_fail(e, api, result, "%s", names[k]);
		}
		kernels.functions[k] = function;
	}
	*dest = kernels;
	return ERROR_NONE;
}

void
native_kernels_fini (struct native_kernels *kernels)
{
	if (!kernels)
		return;
	if (kernels->module) {
		kernels->api->hipStreamSynchronize(kernels->stream);
		kernels->api->hipModuleUnload(kernels->module);
	}
	*kernels = (struct native_kernels){0};
}

enum error_code
native_kernels_launch (struct native_kernels const *kernels,
                       enum native_kernel           kernel,
                       unsigned                     count,
                       void                       **args,
                       struct error                *e)
{
	int const result = kernels->api->hipModuleLaunchKernel(kernels->functions[kernel], (count + 255) / 256, 1, 1,
	                                                        256, 1, 1, 0, kernels->stream, args, nullptr);
	if (result)
		return hip_fail(e, kernels->api, result, "%s", names[kernel]);
	return ERROR_NONE;
}

enum error_code
native_kernels_tune (struct native_kernels const *kernels,
                     struct geometry              g,
                     void const                  *input_rgba,
                     void const                  *raw_rgb,
                     void                        *output_rgb,
                     struct native_tuning         tuning,
                     struct error                *e)
{
	if (output_rgb == input_rgba || output_rgb == raw_rgb)
		return error_fail(e, "GPU native tuning requires distinct input and output buffers");
	void *args[] = {&input_rgba, &raw_rgb, &output_rgb, &g, &tuning};
	return native_kernels_launch(kernels, NATIVE_KERNEL_TUNE_RGB, g.width * g.height, args, e);
}

enum error_code
native_kernels_preserve_color (struct native_kernels const *kernels,
                               struct geometry              g,
                               void const                  *original_rgba,
                               void const                  *raw_rgb,
                               void                        *output_rgb,
                               float                        strength,
                               struct error                *e)
{
	if (output_rgb == original_rgba || output_rgb == raw_rgb)
		return error_fail(e, "GPU color preservation requires distinct input and output buffers");
	void *args[] = {&original_rgba, &raw_rgb, &output_rgb, &g, &strength};
	return native_kernels_launch(kernels, NATIVE_KERNEL_PRESERVE_COLOR, g.width * g.height, args, e);
}
