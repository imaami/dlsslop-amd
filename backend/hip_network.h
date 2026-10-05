/** @file
 *
 * The HIP network run by dlsslopd's own host code: the model's code objects and packed weights, and
 * a tier's pool buffers with every launch of its plan bound once, so that a frame is its kernel
 * launches alone. hip_network.c defines the functions.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_HIP_NETWORK_H_
#define DLSSLOP_AMD_BACKEND_HIP_NETWORK_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "hip.h"
#include "hip_plan.h"
#include "hip_weights.h"

/** @brief The network's code objects, every kernel of HIP_PLAN_KERNELS, and the weights of a plan,
 *         packed and uploaded in its order.
 *
 * hip_model_init() loads one and hip_model_fini() frees it; a zeroed one is not loaded.
 */
struct hip_model {
	struct hip_api const *api;                         //!< The runtime; nullptr when not loaded.
	void                 *modules[HIP_MODULE_COUNT];   //!< Each module, by enum hip_module.
	void                 *functions[HIP_KERNEL_COUNT]; //!< Each kernel, by enum hip_kernel.
	void                **weights;                     //!< Each weight's image, by its index in the plan.
	size_t                count;                       //!< The weights uploaded.
	size_t                bytes;                       //!< The device memory they take.
};

/** @brief Loads the modules with their kernels, then reads, packs and uploads each weight.
 *
 * @param dest           Receives the model, which hip_model_fini() frees; zeroed on a failure.
 * @param api            The runtime, which must outlive the model.
 * @param modules        The directory of the modules; it need not be null-terminated.
 * @param modules_length The length of its path.
 * @param assets         The directory of the weights; it need not be null-terminated.
 * @param assets_length  The length of its path.
 * @param weights        The weights, in the plan's order.
 * @param count          Their number.
 * @param e              Receives the words for what stopped it, or nullptr.
 * @return               ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_model_init (struct hip_model             *dest,
                struct hip_api const         *api,
                char const                   *modules,
                size_t                        modules_length,
                char const                   *assets,
                size_t                        assets_length,
                struct hip_weight_spec const *weights,
                size_t                        count,
                struct error                 *e);

/** @brief Frees a model's weights, unloads its modules and zeroes it; no stream may still use it.
 *
 * @param model The model, or nullptr.
 */
extern void
hip_model_fini (struct hip_model *model);

/** @brief A launch, bound: its kernel and its arguments' addresses. */
struct hip_network_launch;

/** @brief A frame's own arguments, which the bound arguments point to. HIP copies each argument as
 *         a launch is queued, so the next frame may change them. */
struct hip_network_frame {
	void    *rgba;     //!< The frame's input.
	void    *history;  //!< Its history, or its input when it has none.
	void    *output;   //!< Its RGB output.
	uint64_t temporal; //!< 1 when it has a history, else 0; 8 bytes, as every argument's value.
};

/** @brief A tier's network: its pool buffers and gather maps, and each launch of its plan with the
 *         arguments bound twice, once for the first frame's buffers and once for every later
 *         frame's.
 *
 * Within a frame, a pool buffer holds several tensors in turn, as upstream's allocator assigns
 * them. That is correct only because every network launch, and every kernel or copy of dlsslopd's
 * that touches the network's buffers, is ordered on the one stream. A frame's rgba must stay
 * unchanged until the network's last kernel has run, since block 0 and the post kernel both read
 * it; its history is read only by block 0, and its output is written only by the post kernel. The
 * caller keeps all three until the stream is past the network. hip_model_init() and
 * hip_network_init() make every allocation, on the thread that selected the device; a frame makes
 * none.
 *
 * hip_network_init() builds one and hip_network_fini() frees it; a zeroed one is not built. The
 * bound arguments point into the network, so it is only ever initialized in place and never copied.
 */
struct hip_network {
	struct hip_api const      *api;          //!< The runtime; nullptr when not built.
	struct hip_model const    *model;        //!< The model whose kernels and weights it binds.
	void                      *stream;       //!< The stream every launch is queued on.
	void                     **buffers;      //!< The pool's buffers.
	struct hip_network_launch *launches;     //!< The plan's launches, bound.
	uint64_t                  *values;       //!< Each argument's value, 8 bytes whatever its size.
	void                     **argv;         //!< Each argument's address: in values, or in frame.
	void                      *gather[2];    //!< The ViT's gather map, and its inverse.
	size_t                     buffer_count; //!< The buffers allocated.
	size_t                     launch_count; //!< The launches.
	size_t                     bytes;        //!< The device memory of the buffers and maps.
	struct hip_network_frame   frame;        //!< The latest frame's own arguments.
	size_t                     warm;         //!< Which argv of a launch a frame uses: 0 for the first, then 1.
};

/** @brief Allocates a plan's pool buffers as its placement sizes them, uploads the gather maps and
 *         binds the plan's launches.
 *
 * @param dest      Receives the network, which hip_network_fini() frees; zeroed on a failure.
 * @param api       The runtime, which must outlive the network.
 * @param model     The model, which must outlive the network.
 * @param stream    The stream to queue the launches on.
 * @param plan      The plan.
 * @param placement Its placement.
 * @param e         Receives the words for what stopped it, or nullptr.
 * @return          ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_network_init (struct hip_network         *dest,
                  struct hip_api const       *api,
                  struct hip_model const     *model,
                  void                       *stream,
                  struct hip_plan const      *plan,
                  struct hip_placement const *placement,
                  struct error               *e);

/** @brief Frees a network's buffers and maps and zeroes it; the stream must be done with it.
 *
 * @param network The network, or nullptr.
 */
extern void
hip_network_fini (struct hip_network *network);

/** @brief Queues one evaluation on the stream.
 *
 * @param network The network.
 * @param rgba    The frame: W x H x 4 floats.
 * @param history The frame at the same pass before, in the same layout, or nullptr for none.
 * @param output  Receives W x H x 3 floats.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_network_enqueue (struct hip_network *network,
                     void               *rgba,
                     void               *history,
                     void               *output,
                     struct error       *e);

/** @brief Prints the device memory of the network and its model, and the device's, to stdout.
 *
 * @param network The network.
 * @param e       Receives the words for what stopped it, or nullptr.
 * @return        ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_network_print_memory (struct hip_network const *network,
                          struct error             *e);

#endif /* DLSSLOP_AMD_BACKEND_HIP_NETWORK_H_ */
