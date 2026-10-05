/** @file
 *
 * The HIP network's launch plan: upstream's production RunGraph turned into data once per tier and
 * preset, with the buffers its pool settles on. A port of the production path of lmxxf's
 * hip_reference_network.h (MIT). hip_plan.c defines the functions and the tables.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_HIP_PLAN_H_
#define DLSSLOP_AMD_BACKEND_HIP_PLAN_H_

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "hip_weights.h"

/** @brief The code objects the network launches kernels from. */
enum hip_module : uint8_t {
	HIP_MODULE_MH_REFERENCE,   //!< multihead-reference.
	HIP_MODULE_DEEP_REFERENCE, //!< deep_reference.
	HIP_MODULE_C32_FUSED,      //!< c32_fused_ffn_attention-packed.
	HIP_MODULE_MH_FUSED,       //!< multihead_fused_attention.
	HIP_MODULE_DEEP_FAST,      //!< deep_fast-packed.
	HIP_MODULE_MH_FAST,        //!< multihead-fast-padded-wave-packed.
	HIP_MODULE_COUNT           //!< The number of modules; no module.
};

/** @brief Each module's file, by enum hip_module. */
extern char const *const HIP_PLAN_MODULE_FILES[];

/** @brief How upstream's Run() makes a launch's groups of the count it is given. */
enum hip_grid : uint8_t {
	HIP_GRID_GROUPS,     //!< The count is of groups: windows.
	HIP_GRID_DEFAULT,    //!< A group per 256 elements, rounded up, whatever the group's size.
	HIP_GRID_512,        //!< A group per 512 elements.
	HIP_GRID_1024,       //!< A group per 1024 elements.
	HIP_GRID_FFN,        //!< A group per 16 tokens of c channels, for 2c threads.
	HIP_GRID_POOL_GROUP, //!< A group per 16 output tokens of 2c channels, rounded up, for c threads.
	HIP_GRID_SCALAR,     //!< 64 tokens by 64 channels a group, c from argument 5.
	HIP_GRID_DECODER,    //!< 16 tokens by 16 channels a group, c from argument 9.
};

/** @brief Every kernel the network can launch at dlsslopd's tiers. */
enum hip_kernel : uint8_t {
	HIP_KERNEL_C32_PREFIX,
	HIP_KERNEL_C32_MAPPED,
	HIP_KERNEL_C32_CHAIN,
	HIP_KERNEL_C32_CHAIN_FINISH,
	HIP_KERNEL_C32_CHAIN_FINISH_DCROP,
	HIP_KERNEL_C32_POST,
	HIP_KERNEL_POOL32,
	HIP_KERNEL_FFN_C64,
	HIP_KERNEL_FFN_C64_BYTEIN,
	HIP_KERNEL_FFN_C64_MAPPED,
	HIP_KERNEL_FFN_C64_MAPPED_BYTEIN,
	HIP_KERNEL_FFN_C128,
	HIP_KERNEL_FFN_C128_BYTEIN,
	HIP_KERNEL_FFN_C128_MAPPED,
	HIP_KERNEL_FFN_C128_MAPPED_BYTEIN,
	HIP_KERNEL_FFN_C256,
	HIP_KERNEL_FFN_C256_BYTEIN,
	HIP_KERNEL_FFN_C256_MAPPED,
	HIP_KERNEL_FFN_C256_MAPPED_BYTEIN,
	HIP_KERNEL_ATTENTION_C64,
	HIP_KERNEL_ATTENTION_C64_BOUT,
	HIP_KERNEL_ATTENTION_C128,
	HIP_KERNEL_ATTENTION_C128_BOUT,
	HIP_KERNEL_ATTENTION_C256,
	HIP_KERNEL_ATTENTION_C256_BOUT,
	HIP_KERNEL_POOL_GROUP_C64,
	HIP_KERNEL_POOL_GROUP_C128,
	HIP_KERNEL_POOL_GROUP_C256,
	HIP_KERNEL_POOL_GROUP_C512,
	HIP_KERNEL_SHIFT_PACK,
	HIP_KERNEL_SPLIT_MIX,
	HIP_KERNEL_SPLIT_FFN,
	HIP_KERNEL_SPLIT_PROJECTION,
	HIP_KERNEL_QKV_C512,
	HIP_KERNEL_ATTENTION_C512,
	HIP_KERNEL_PROJECT_C512,
	HIP_KERNEL_PROJECT_SCALAR,
	HIP_KERNEL_VIT_GATHER,
	HIP_KERNEL_VIT_PACK,
	HIP_KERNEL_VIT_EXPAND,
	HIP_KERNEL_VIT_CONTRACT,
	HIP_KERNEL_VIT_QKV,
	HIP_KERNEL_VIT_ATTENTION_256,
	HIP_KERNEL_VIT_ATTENTION_400,
	HIP_KERNEL_VIT_ATTENTION_640,
	HIP_KERNEL_VIT_PROJECT,
	HIP_KERNEL_DECODER,
	HIP_KERNEL_DECODER_BYTEOUT,
	HIP_KERNEL_COUNT //!< The number of kernels; no kernel.
};

/** @brief A kernel: its symbol, its module, its groups' size and how its grid is made. */
struct hip_kernel_info {
	char const      *name;    //!< Its symbol.
	uint16_t         threads; //!< The threads of a group.
	enum hip_module  module;  //!< Its module.
	enum hip_grid    grid;    //!< How a launch's count becomes its groups.
};

/** @brief Each kernel, by enum hip_kernel. */
extern struct hip_kernel_info const HIP_PLAN_KERNELS[];

/** @brief What a kernel argument is. The frame's own ones are known only as it is queued. */
enum hip_arg_kind : uint8_t {
	HIP_ARG_U32,      //!< The value.
	HIP_ARG_F32,      //!< The float with the value's bits.
	HIP_ARG_NULL,     //!< A null pointer.
	HIP_ARG_TENSOR,   //!< The buffer in the pool of the tensor whose id is the value.
	HIP_ARG_WEIGHT,   //!< The image of the plan's weight at the value.
	HIP_ARG_GATHER,   //!< The ViT gather map, the inverse one when the value is 1.
	HIP_ARG_RGBA,     //!< The frame's input.
	HIP_ARG_HISTORY,  //!< The frame's history, or its input when it has none.
	HIP_ARG_TEMPORAL, //!< 1 when the frame has a history, else 0.
	HIP_ARG_OUTPUT,   //!< The frame's RGB output.
};

/** @brief A kernel argument. */
struct hip_arg {
	uint32_t          value; //!< Its value, as its kind says.
	enum hip_arg_kind kind;  //!< Its kind.
};

/** @brief The most arguments a launch passes. */
#define HIP_LAUNCH_ARGS 15

/** @brief A kernel launch: grid one-dimensional groups of the kernel's threads, and its count
 *         arguments, each passed as a 4-byte value or an 8-byte pointer. */
struct hip_launch {
	struct hip_arg  args[HIP_LAUNCH_ARGS]; //!< The arguments; those after count are zeroed.
	uint32_t        grid;                  //!< The groups.
	uint16_t        count;                 //!< The arguments it passes.
	enum hip_kernel kernel;                //!< The kernel.
};

/** @brief One evaluation of the network at a geometry, as upstream's production RunGraph queues it
 *         on its stream.
 *
 * Zeroed, it is no plan: a plan's width is never 0.
 */
struct hip_plan {
	struct hip_launch      *launches;     //!< The launches, in order.
	uint32_t               *floats;       //!< The floats of each tensor, by its id (upstream's New(n)).
	int32_t                *events;       //!< Tensors taken from the pool (id) and returned (~id), in order.
	struct hip_weight_spec *weights;      //!< The weights the launches read, in the order upstream uploads them.
	size_t                  launch_count; //!< The number of launches.
	size_t                  tensor_count; //!< The number of tensors.
	size_t                  event_count;  //!< The number of events.
	size_t                  weight_count; //!< The number of weights.
	uint32_t                tokens;       //!< The ViT's tokens; each gather map holds 1024 of them.
	uint16_t                width;        //!< The geometry's width.
	uint16_t                height;       //!< Its height.
};

/** @brief Plans the network at one of its padded tiers.
 *
 * @param dest        Receives the plan, which hip_plan_fini() frees; zeroed on a failure.
 * @param width       The tier's width.
 * @param height      Its padded height.
 * @param performance Whether to skip the blocks that dlsslopd --performance skips (42, 43 and 46).
 * @param e           Receives the words for what stopped it, or nullptr.
 * @return            ERROR_NONE, or ERROR_FAILED for another geometry or without memory.
 */
extern enum error_code
hip_plan_init (struct hip_plan *dest,
               unsigned         width,
               unsigned         height,
               bool             performance,
               struct error    *e);

/** @brief Frees what a plan holds and zeroes it.
 *
 * @param plan The plan, or nullptr.
 */
extern void
hip_plan_fini (struct hip_plan *plan);

/** @brief Writes the ViT's gather map for some tokens of 1024 channels: for each element, the one
 *         it takes, with tokens permuted within 16 and channels within 32 into the order the
 *         ViT's kernels read, or back (upstream: Gather).
 *
 * @param map     Receives tokens * 1024 indices.
 * @param tokens  The tokens.
 * @param inverse Whether to map back.
 */
extern void
hip_plan_gather_map (uint32_t *map,
                     unsigned  tokens,
                     bool      inverse);

/** @brief The pool buffers a plan's tensors live in, as upstream's allocator assigns them: the
 *         smallest free buffer that holds a tensor, the earliest of equal ones, or a new buffer of
 *         its exact size.
 *
 * The first frame starts with no buffers and sizes the pool. Each later frame starts with all of
 * them free and settles on another assignment.
 */
struct hip_placement {
	size_t   *buffers;      //!< Each buffer's bytes, in creation order.
	uint16_t *first;        //!< Each tensor's buffer in the first frame.
	uint16_t *later;        //!< Each tensor's buffer in later frames.
	size_t    buffer_count; //!< The number of buffers.
};

/** @brief Places a plan's tensors in the pool's buffers.
 *
 * @param dest Receives the placement, which hip_placement_fini() frees; zeroed on a failure.
 * @param plan The plan.
 * @param e    Receives the words for what stopped it, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED when a later frame needs more buffers than the first, or
 *             without memory.
 */
extern enum error_code
hip_plan_place (struct hip_placement  *dest,
                struct hip_plan const *plan,
                struct error          *e);

/** @brief Frees what a placement holds and zeroes it.
 *
 * @param placement The placement, or nullptr.
 */
extern void
hip_placement_fini (struct hip_placement *placement);

#endif /* DLSSLOP_AMD_BACKEND_HIP_PLAN_H_ */
