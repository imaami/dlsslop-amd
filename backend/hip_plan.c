/** @file
 *
 * The HIP network's launch plan, recorded from a port of upstream's production RunGraph, and the
 * pool's buffers.
 */
// SPDX-License-Identifier: MIT
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip_plan.h"

char const *const HIP_PLAN_MODULE_FILES[] = {
	"multihead-reference.hsaco",
	"deep_reference.hsaco",
	"c32_fused_ffn_attention-packed.hsaco",
	"multihead_fused_attention.hsaco",
	"deep_fast-packed.hsaco",
	"multihead-fast-padded-wave-packed.hsaco",
};
static_assert(sizeof HIP_PLAN_MODULE_FILES / sizeof *HIP_PLAN_MODULE_FILES == HIP_MODULE_COUNT,
              "a file per module");

struct hip_kernel_info const HIP_PLAN_KERNELS[] = {
	{"c32_fast_ffn_attention_fused_half_prefix_finish_main8", 128, HIP_MODULE_C32_FUSED, HIP_GRID_GROUPS},
	{"c32_fast_ffn_attention_fused_half_mapped", 128, HIP_MODULE_C32_FUSED, HIP_GRID_GROUPS},
	{"c32_fast_ffn_attention_fused_half_chain", 128, HIP_MODULE_C32_FUSED, HIP_GRID_GROUPS},
	{"c32_fast_ffn_attention_fused_half_chain_finish", 128, HIP_MODULE_C32_FUSED, HIP_GRID_GROUPS},
	{"c32_fast_ffn_attention_fused_half_chain_finish_dcrop", 128, HIP_MODULE_C32_FUSED, HIP_GRID_GROUPS},
	{"c32_post_merge_head_half", 128, HIP_MODULE_C32_FUSED, HIP_GRID_GROUPS},
	{"mh_pool_project_production_h16w", 32, HIP_MODULE_MH_FAST, HIP_GRID_DEFAULT},
	{"mh_ffn_fused_c64_project_g128_qkv_fb", 128, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c64_project_g128_qkv_bytein_fb", 128, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c64_project_mapped_g128_qkv_fb", 128, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c64_project_mapped_g128_qkv_bytein_fb", 128, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c128_project_g128_qkv_fb", 256, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c128_project_g128_qkv_bytein_fb", 256, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c128_project_mapped_g128_qkv_fb", 256, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c128_project_mapped_g128_qkv_bytein_fb", 256, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c256_frag_project_g128_qkv_fb", 512, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c256_frag_project_g128_qkv_bytein_fb", 512, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c256_frag_project_mapped_g128_qkv_fb", 512, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"mh_ffn_fused_c256_frag_project_mapped_g128_qkv_bytein_fb", 512, HIP_MODULE_MH_FAST, HIP_GRID_FFN},
	{"c64_attention_project_fb_diag", 256, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"c64_attention_project_fb_bout_diag", 256, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"c128_attention_project_fb_diag", 512, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"c128_attention_project_fb_bout_diag", 512, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"c256_attention_project_fb_diag", 512, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"c256_attention_project_fb_bout_diag", 512, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"mh_pool_project_group_c64", 64, HIP_MODULE_MH_FAST, HIP_GRID_POOL_GROUP},
	{"mh_pool_project_group_c128", 128, HIP_MODULE_MH_FAST, HIP_GRID_POOL_GROUP},
	{"mh_pool_project_group_c256", 256, HIP_MODULE_MH_FAST, HIP_GRID_POOL_GROUP},
	{"mh_pool_project_group_c512", 512, HIP_MODULE_MH_FAST, HIP_GRID_POOL_GROUP},
	{"mh_shift_pack", 256, HIP_MODULE_MH_REFERENCE, HIP_GRID_DEFAULT},
	{"split_mix_blocked_h16w", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_1024},
	{"split_ffn_fused_fp8_t8", 128, HIP_MODULE_DEEP_FAST, HIP_GRID_1024},
	{"split_projection_frag", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_1024},
	{"mh_qkv_normalize_frag_c512", 32, HIP_MODULE_MH_FAST, HIP_GRID_1024},
	{"mh_attention_fused_fp8_out", 128, HIP_MODULE_MH_FUSED, HIP_GRID_GROUPS},
	{"mh_attention_project_frag_c512", 32, HIP_MODULE_MH_FAST, HIP_GRID_1024},
	{"mh_attention_project_fast_scalar_fp8", 512, HIP_MODULE_MH_FAST, HIP_GRID_SCALAR},
	{"vit_gather", 256, HIP_MODULE_DEEP_REFERENCE, HIP_GRID_DEFAULT},
	{"vit_pack_input", 256, HIP_MODULE_DEEP_FAST, HIP_GRID_DEFAULT},
	{"vit_expand_blocked_fp8_frag_bytein", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_1024},
	{"vit_contract_blocked_fp8_frag", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_1024},
	{"vit_qkv_project_normalize_fused_f16compact_fp8_frag", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_512},
	{"vit_attention_fused_256_bytein", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_DEFAULT},
	{"vit_attention_fused_400_bytein", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_DEFAULT},
	{"vit_attention_fused_640_bytein", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_DEFAULT},
	{"vit_project_frag", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_DEFAULT},
	{"decoder_project2x_h16w", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_DECODER},
	{"decoder_project2x_h16w_byteout", 32, HIP_MODULE_DEEP_FAST, HIP_GRID_DECODER},
};
static_assert(sizeof HIP_PLAN_KERNELS / sizeof *HIP_PLAN_KERNELS == HIP_KERNEL_COUNT, "an entry per kernel");

/** @brief The shift of each block of an encoder level, and of the C512 encoder, by its place in
 *         the group (upstream: RunGraph's shifts). */
static uint8_t const encoder_shift[8] = {0, 3, 1, 2, 0, 3, 1, 2};

/** @brief The shift of decoder blocks 40 to 69 (upstream: Shift). */
static uint8_t const decoder_shift[30] = {
	0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2,
	1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2,
};

/** @brief The multihead blocks of a level. */
struct level {
	uint16_t c;     //!< Their channels.
	uint8_t  first; //!< The first block.
	uint8_t  last;  //!< The last.
};

/** @brief The encoder's levels: c64, c128 and c256 down. */
static struct level const encoder_levels[3] = {{64, 5, 8}, {128, 9, 14}, {256, 15, 22}};

/** @brief The decoder's levels: c256, c128 and c64 back up. */
static struct level const decoder_levels[3] = {{256, 48, 55}, {128, 56, 61}, {64, 62, 65}};

/** @brief The fused FFN and QKV kernel of a level, for an identity block's input or a mapped one,
 *         read as floats or as bytes. */
static enum hip_kernel const ffn_kernels[3][2][2] = {
	{{HIP_KERNEL_FFN_C64, HIP_KERNEL_FFN_C64_BYTEIN},
	 {HIP_KERNEL_FFN_C64_MAPPED, HIP_KERNEL_FFN_C64_MAPPED_BYTEIN}},
	{{HIP_KERNEL_FFN_C128, HIP_KERNEL_FFN_C128_BYTEIN},
	 {HIP_KERNEL_FFN_C128_MAPPED, HIP_KERNEL_FFN_C128_MAPPED_BYTEIN}},
	{{HIP_KERNEL_FFN_C256, HIP_KERNEL_FFN_C256_BYTEIN},
	 {HIP_KERNEL_FFN_C256_MAPPED, HIP_KERNEL_FFN_C256_MAPPED_BYTEIN}},
};

/** @brief The attention and projection kernel of a level, writing floats or bytes. */
static enum hip_kernel const attention_kernels[3][2] = {
	{HIP_KERNEL_ATTENTION_C64, HIP_KERNEL_ATTENTION_C64_BOUT},
	{HIP_KERNEL_ATTENTION_C128, HIP_KERNEL_ATTENTION_C128_BOUT},
	{HIP_KERNEL_ATTENTION_C256, HIP_KERNEL_ATTENTION_C256_BOUT},
};

/** @brief The pooling kernel by its input's channels: 64, 128, 256 and 512. */
static enum hip_kernel const pool_group_kernels[4] = {
	HIP_KERNEL_POOL_GROUP_C64, HIP_KERNEL_POOL_GROUP_C128, HIP_KERNEL_POOL_GROUP_C256,
	HIP_KERNEL_POOL_GROUP_C512,
};

/** @brief The ViT's attention kernel by the most tokens it attends over: 256, 400 and 640. */
static enum hip_kernel const vit_attention_kernels[3] = {
	HIP_KERNEL_VIT_ATTENTION_256, HIP_KERNEL_VIT_ATTENTION_400, HIP_KERNEL_VIT_ATTENTION_640,
};

/** @brief A geometry upstream accepts that dlsslopd's tiers use. */
struct geometry_size {
	uint16_t width;  //!< Its width.
	uint16_t height; //!< Its height.
};

/** @brief The geometries of the tiers. */
static struct geometry_size const geometries[] = {{1280, 768}, {1600, 960}, {1920, 1152}};

/** @brief The groups of a launch of some elements (upstream: Run()).
 *
 * @param k     The kernel.
 * @param count The elements.
 * @param args  The launch's arguments.
 * @return      The groups.
 */
static uint32_t
groups (struct hip_kernel_info const *k,
        size_t                        count,
        struct hip_arg const         *args)
{
	switch (k->grid) {
	case HIP_GRID_GROUPS:     return (uint32_t)count;
	case HIP_GRID_DEFAULT:    return (uint32_t)((count + 255) / 256);
	case HIP_GRID_512:        return (uint32_t)(count / 512);
	case HIP_GRID_1024:       return (uint32_t)(count / 1024);
	case HIP_GRID_FFN:        return (uint32_t)(count / (8 * (size_t)k->threads));
	case HIP_GRID_POOL_GROUP: return (uint32_t)((count / (2 * (size_t)k->threads) + 15) / 16);
	case HIP_GRID_SCALAR: {
		size_t const c = args[5].value;
		return (uint32_t)((count / c + 63) / 64 * ((c + 63) / 64));
	}
	case HIP_GRID_DECODER: {
		size_t const c = args[9].value;
		return (uint32_t)((count / c + 15) / 16 * (c / 16));
	}
	}
	return 0;
}

/** @brief Records upstream's RunGraph as it queues its production path: each launch, each weight
 *         a launch reads when upstream first asks for it, and each tensor it takes from its pool
 *         and returns.
 *
 * A failure to grow an array is kept in error, after which nothing more is recorded.
 */
struct builder {
	struct hip_plan *plan;              //!< The plan.
	uint32_t        *references;        //!< The references to each tensor.
	struct error    *e;                 //!< Receives the words for the first failure, or nullptr.
	size_t           launch_capacity;   //!< The launches that the plan's memory holds.
	size_t           tensor_capacity;   //!< The tensors that the plan's and references' memory hold.
	size_t           event_capacity;    //!< The events that the plan's memory holds.
	size_t           weight_capacity;   //!< The weights that the plan's memory holds.
	bool             performance;       //!< Whether blocks 42, 43 and 46 are skipped.
	enum error_code  error;             //!< The first failure.
};

/** @brief Upstream's pooled Tensor: a reference to a tensor, or none. When the last reference is
 *         dropped, the pool may hand the tensor's buffer to the next. Copying the struct does not
 *         make a reference: hip_plan_tensor_ref() does. */
struct hip_plan_tensor {
	uint32_t ref; //!< The tensor's id plus 1, or 0 for none.
};

/** @brief A weight of the plan. */
struct hip_plan_weight {
	uint32_t index; //!< Its index in the plan's weights.
};

/** @brief Records a failure to grow an array, once.
 *
 * @param b The builder.
 */
[[gnu::cold, gnu::noinline]]
static void
out_of_memory (struct builder *b)
{
	if (!b->error)
		b->error = error_fail(b->e, "out of memory");
}

/** @brief The doubled capacity of a full array.
 *
 * @param capacity Its capacity.
 * @param first    The capacity of an empty array.
 * @return         The new capacity.
 */
static size_t
grown (size_t capacity,
       size_t first)
{
	return capacity ? 2 * capacity : first;
}

/** @brief Adds a launch.
 *
 * @param b The builder.
 * @param l The launch.
 */
static void
push_launch (struct builder          *b,
             struct hip_launch const *l)
{
	struct hip_plan *const p = b->plan;
	if (p->launch_count == b->launch_capacity) {
		size_t const capacity = grown(b->launch_capacity, 256);
		struct hip_launch *const launches = realloc(p->launches, capacity * sizeof *launches);
		if (!launches) {
			out_of_memory(b);
			return;
		}
		p->launches = launches;
		b->launch_capacity = capacity;
	}
	p->launches[p->launch_count++] = *l;
}

/** @brief Adds an event.
 *
 * @param b     The builder.
 * @param event The event.
 */
static void
push_event (struct builder *b,
            int32_t         event)
{
	struct hip_plan *const p = b->plan;
	if (p->event_count == b->event_capacity) {
		size_t const capacity = grown(b->event_capacity, 1024);
		int32_t *const events = realloc(p->events, capacity * sizeof *events);
		if (!events) {
			out_of_memory(b);
			return;
		}
		p->events = events;
		b->event_capacity = capacity;
	}
	p->events[p->event_count++] = event;
}

/** @brief Adds a tensor of one reference.
 *
 * @param b      The builder.
 * @param floats Its floats.
 * @return       Its id, or UINT32_MAX without memory for it.
 */
static uint32_t
push_tensor (struct builder *b,
             uint32_t        floats)
{
	struct hip_plan *const p = b->plan;
	if (p->tensor_count == b->tensor_capacity) {
		size_t const capacity = grown(b->tensor_capacity, 512);
		uint32_t *const sizes = realloc(p->floats, capacity * sizeof *sizes);
		if (!sizes) {
			out_of_memory(b);
			return UINT32_MAX;
		}
		p->floats = sizes;
		uint32_t *const references = realloc(b->references, capacity * sizeof *references);
		if (!references) {
			out_of_memory(b);
			return UINT32_MAX;
		}
		b->references = references;
		b->tensor_capacity = capacity;
	}
	b->references[p->tensor_count] = 1;
	p->floats[p->tensor_count] = floats;
	return (uint32_t)p->tensor_count++;
}

/** @brief Adds a weight.
 *
 * @param b    The builder.
 * @param spec The weight.
 * @return     Its index, or UINT32_MAX without memory for it.
 */
static uint32_t
push_weight (struct builder               *b,
             struct hip_weight_spec const *spec)
{
	struct hip_plan *const p = b->plan;
	if (p->weight_count == b->weight_capacity) {
		size_t const capacity = grown(b->weight_capacity, 256);
		struct hip_weight_spec *const weights = realloc(p->weights, capacity * sizeof *weights);
		if (!weights) {
			out_of_memory(b);
			return UINT32_MAX;
		}
		p->weights = weights;
		b->weight_capacity = capacity;
	}
	p->weights[p->weight_count] = *spec;
	return (uint32_t)p->weight_count++;
}

/** @brief A new tensor from the pool, of one reference (upstream's New(floats)).
 *
 * @param b      The builder.
 * @param floats Its floats.
 * @return       The reference; none once the builder failed.
 */
static struct hip_plan_tensor
tensor (struct builder *b,
        size_t          floats)
{
	if (b->error)
		return (struct hip_plan_tensor){0};
	uint32_t const id = push_tensor(b, (uint32_t)floats);
	if (id == UINT32_MAX)
		return (struct hip_plan_tensor){0};
	push_event(b, (int32_t)id);
	return (struct hip_plan_tensor){id + 1};
}

/** @brief Another reference to a tensor, where upstream copies a Tensor.
 *
 * @param b A builder.
 * @param t The reference, or none.
 * @return  @a t.
 */
static struct hip_plan_tensor
hip_plan_tensor_ref (struct builder         *b,
                     struct hip_plan_tensor  t)
{
	if (t.ref)
		++b->references[t.ref - 1];
	return t;
}

/** @brief Drops a reference to a tensor, where upstream destroys or resets a Tensor; the last one
 *         returns the tensor to the pool.
 *
 * @param b A builder.
 * @param t The reference, or none.
 */
static void
hip_plan_tensor_drop (struct builder         *b,
                      struct hip_plan_tensor  t)
{
	if (t.ref && !--b->references[t.ref - 1])
		push_event(b, ~(int32_t)(t.ref - 1));
}

/** @brief A weight of the plan, listed the first time upstream asks for it.
 *
 * @param b    The builder.
 * @param spec The weight, its stem's bytes after its null zero, as the plan's are.
 * @return     The weight; index 0 once the builder failed.
 */
static struct hip_plan_weight
listed_weight (struct builder               *b,
               struct hip_weight_spec const *spec)
{
	if (b->error)
		return (struct hip_plan_weight){0};
	// No weight is listed twice. Upstream asks again mostly for one it asked for a moment ago,
	// so the search starts at the end.
	struct hip_plan const *const p = b->plan;
	for (uint32_t i = (uint32_t)p->weight_count; i--;)
		if (p->weights[i].recipe == spec->recipe && !memcmp(p->weights[i].stem, spec->stem, sizeof spec->stem))
			return (struct hip_plan_weight){i};
	uint32_t const index = push_weight(b, spec);
	return (struct hip_plan_weight){index == UINT32_MAX ? 0 : index};
}

/** @brief The weight of a literal stem packed by a recipe. */
#define NAMED_WEIGHT(b, stem, recipe) listed_weight((b), &(struct hip_weight_spec){stem, (recipe)})

/** @brief The weight blockBLOCK-PART packed by a recipe, listed the first time upstream asks for
 *         it.
 *
 * @param b      The builder.
 * @param block  The block.
 * @param part   The part.
 * @param recipe The recipe.
 * @return       The weight; index 0 once the builder failed.
 */
static struct hip_plan_weight
weight (struct builder  *b,
        unsigned         block,
        char const      *part,
        enum hip_recipe  recipe)
{
	struct hip_weight_spec spec = {.recipe = recipe};
	int const length = snprintf(spec.stem, sizeof spec.stem, "block%u-%s", block, part);
	if (length < 0 || length >= (int)sizeof spec.stem) {
		if (!b->error)
			b->error = error_fail(b->e, "weight stem block%u-%s is too long", block, part);
		return (struct hip_plan_weight){0};
	}
	return listed_weight(b, &spec);
}

/** @brief An argument of a tensor: its buffer, or a null pointer for none. */
static struct hip_arg
tensor_arg (struct hip_plan_tensor t)
{
	if (!t.ref)
		return (struct hip_arg){.kind = HIP_ARG_NULL};
	return (struct hip_arg){.value = t.ref - 1, .kind = HIP_ARG_TENSOR};
}

/** @brief An argument of a weight: its image. */
static struct hip_arg
weight_arg (struct hip_plan_weight w)
{
	return (struct hip_arg){.value = w.index, .kind = HIP_ARG_WEIGHT};
}

/** @brief An argument of a 4-byte unsigned value. */
static struct hip_arg
u32_arg (unsigned value)
{
	return (struct hip_arg){.value = value, .kind = HIP_ARG_U32};
}

/** @brief An argument of a 4-byte float value. */
static struct hip_arg
f32_arg (float value)
{
	uint32_t b;
	memcpy(&b, &value, sizeof b);
	return (struct hip_arg){.value = b, .kind = HIP_ARG_F32};
}

/** @brief A null pointer argument. */
static struct hip_arg
null_arg (nullptr_t)
{
	return (struct hip_arg){.kind = HIP_ARG_NULL};
}

/** @brief An argument as it is. */
static struct hip_arg
same_arg (struct hip_arg a)
{
	return a;
}

/** @brief A kernel argument of a value. A literal's type must say whether it is a value or a
 *         pointer: an int has no association. */
#define arg(x) _Generic((x),                     \
	struct hip_plan_tensor: tensor_arg,      \
	struct hip_plan_weight: weight_arg,      \
	unsigned:               u32_arg,         \
	float:                  f32_arg,         \
	nullptr_t:              null_arg,        \
	struct hip_arg:         same_arg)(x)

/** @brief The frame's input. */
#define RGBA ((struct hip_arg){.kind = HIP_ARG_RGBA})

/** @brief The frame's history, or its input. */
#define HISTORY ((struct hip_arg){.kind = HIP_ARG_HISTORY})

/** @brief Whether the frame has a history. */
#define TEMPORAL ((struct hip_arg){.kind = HIP_ARG_TEMPORAL})

/** @brief The frame's RGB output. */
#define OUTPUT ((struct hip_arg){.kind = HIP_ARG_OUTPUT})

/** @brief Records a launch of a kernel over some elements (upstream: Run()).
 *
 * @param b      The builder.
 * @param kernel The kernel.
 * @param count  The elements.
 * @param args   Its arguments.
 * @param n      Their number.
 */
static void
launch (struct builder       *b,
        enum hip_kernel       kernel,
        size_t                count,
        struct hip_arg const *args,
        uint16_t              n)
{
	if (b->error)
		return;
	struct hip_launch l = {.kernel = kernel, .count = n};
	memcpy(l.args, args, n * sizeof *args);
	l.grid = groups(&HIP_PLAN_KERNELS[kernel], count, l.args);
	push_launch(b, &l);
}

/** @brief Records a launch of a kernel over some elements with the arguments that follow, each an
 *         arg(). */
#define LAUNCH(b, kernel, count, ...) do {                                                    \
	struct hip_arg const args_[] = {__VA_ARGS__};                                         \
	static_assert(sizeof args_ / sizeof *args_ <= HIP_LAUNCH_ARGS, "too many arguments"); \
	launch((b), (kernel), (count), args_, sizeof args_ / sizeof *args_);                  \
} while (0)

/** @brief Upstream's C32Result, less what production never reads. */
struct c32 {
	struct hip_plan_tensor main;  //!< The main output of a finishing block.
	struct hip_plan_tensor down;  //!< Its pooled quarter.
	struct hip_plan_tensor raw;   //!< The raw window tiles of a block that passes them on.
	unsigned               workw; //!< The shifted lattice's width.
	unsigned               sx;    //!< The shift across.
	unsigned               sy;    //!< The shift down.
};

/** @brief Drops a C32Result's references in the order of its members, as upstream's assignment
 *         over it destroys them.
 *
 * @param b The builder.
 * @param r The result, which is emptied.
 */
static void
c32_drop (struct builder *b,
          struct c32     *r)
{
	hip_plan_tensor_drop(b, r->main);
	hip_plan_tensor_drop(b, r->down);
	hip_plan_tensor_drop(b, r->raw);
	*r = (struct c32){0};
}

/** @brief The tokens a pooling keeps: ow x oh, of which vw x vh hold the input when those are
 *         nonzero, the rest being zero padding. */
struct pooled {
	unsigned ow; //!< The output's width.
	unsigned oh; //!< Its height.
	unsigned vw; //!< The width of its input part, or 0 for all of it.
	unsigned vh; //!< The height of its input part.
};

/** @brief A block of the C32 chain at w x h, which passes on its raw window tiles; a finishing
 *         block writes the main output or its pooled quarter instead (upstream: C32Chain).
 *
 * Upstream finishes the first block of a chain in a kernel of its own, but no first block
 * finishes.
 *
 * @param b      The builder.
 * @param input  The chain's input, which only the first block reads.
 * @param prev   The previous block's result, or nullptr for the first block.
 * @param w      The width.
 * @param h      The height.
 * @param shift  The block's shift.
 * @param block  The block.
 * @param finish Whether it writes the main output.
 * @param down   Whether it writes the pooled quarter.
 * @return       The result, holding new references.
 */
static struct c32
c32_chain (struct builder           *b,
           struct hip_plan_tensor    input,
           struct c32 const         *prev,
           unsigned                  w,
           unsigned                  h,
           unsigned                  shift,
           unsigned                  block,
           bool                      finish,
           bool                      down)
{
	unsigned const sx = shift & 1 ? 4 : 0, sy = shift & 2 ? 4 : 0, ww = w + 2 * sx, hh = h + 2 * sy,
	               windows = ww * hh / 64;
	// Upstream asks for the weights as it evaluates the launch's arguments, right to left.
	struct hip_plan_weight const attention = weight(b, block, "attention", HIP_RECIPE_C32);
	struct hip_plan_weight const ffn = weight(b, block, "ffn", HIP_RECIPE_C32);
	if (prev && (finish || down)) {
		struct c32 result = {.workw = ww, .sx = sx, .sy = sy};
		if (finish)
			result.main = tensor(b, (size_t)w * h * 32);
		if (down)
			result.down = tensor(b, (size_t)(w / 2) * (h / 2) * 32);
		LAUNCH(b, down ? HIP_KERNEL_C32_CHAIN_FINISH_DCROP : HIP_KERNEL_C32_CHAIN_FINISH, windows,
		       arg(prev->raw), arg(ffn), arg(attention), arg(result.main), arg(result.down),
		       arg(windows), arg(3u), arg(1u), arg(w), arg(h), arg(sx), arg(sy), arg(prev->workw),
		       arg(prev->sx), arg(prev->sy));
		return result;
	}
	// Upstream asks for 16 floats a token; the kernels use half of them.
	struct c32 const result = {
		.raw = tensor(b, (size_t)ww * hh * 16), .workw = ww, .sx = sx, .sy = sy,
	};
	if (prev)
		LAUNCH(b, HIP_KERNEL_C32_CHAIN, windows, arg(prev->raw), arg(ffn), arg(attention),
		       arg(result.raw), arg(windows), arg(3u), arg(1u), arg(w), arg(h), arg(sx), arg(sy),
		       arg(prev->workw), arg(prev->sx), arg(prev->sy));
	else
		LAUNCH(b, HIP_KERNEL_C32_MAPPED, windows, arg(input), arg(ffn), arg(attention), arg(result.raw),
		       arg(windows), arg(0u), arg(1u), arg(w), arg(h), arg(sx), arg(sy));
	return result;
}

/** @brief A block below 512 channels, with AttentionFast (upstream: Body).
 *
 * The input is mapped onto the block's shifted lattice as the FFN reads it, and the attention
 * crops back; within a level the blocks pass bytes.
 *
 * @param b        The builder.
 * @param input    The input.
 * @param w        The width.
 * @param h        The height.
 * @param c        The channels.
 * @param shift    The block's shift.
 * @param block    The block.
 * @param raw      Whether it ends an encoder level, whose output is a skip.
 * @param byte_in  Whether its input is bytes.
 * @param byte_out Whether its output is bytes.
 * @return         The output, a new reference.
 */
static struct hip_plan_tensor
mh_block (struct builder         *b,
          struct hip_plan_tensor  input,
          unsigned                w,
          unsigned                h,
          unsigned                c,
          unsigned                shift,
          unsigned                block,
          bool                    raw,
          bool                    byte_in,
          bool                    byte_out)
{
	unsigned const sx = shift & 1 ? 4 : 0, sy = shift & 2 ? 4 : 0, ww = (w + sx + 7) & ~7u,
	               hh = (h + sy + 7) & ~7u, n = ww * hh, level = (unsigned)__builtin_ctz(c) - 6;
	bool const identity = !sx && !sy && ww == w && hh == h;
	struct hip_plan_tensor const ffn = tensor(b, (size_t)n * c / 4);
	struct hip_plan_tensor const norm = tensor(b, (size_t)n * 3 * c / 4);
	bool const frag = c == 256;
	// At c256 upstream also uploads the attention weight as HIP_RECIPE_MH_ATTENTION packs it,
	// which no kernel reads.
	struct hip_plan_weight const ffn_weights =
		weight(b, block, "ffn", frag ? HIP_RECIPE_FFN_FRAG : HIP_RECIPE_MH_FFN);
	struct hip_plan_weight const qkv =
		weight(b, block, "attention", frag ? HIP_RECIPE_QKV_FRAG_ONLY : HIP_RECIPE_MH_ATTENTION);
	LAUNCH(b, ffn_kernels[level][!identity][byte_in], (size_t)n * c, arg(input), arg(ffn_weights),
	       arg(qkv), arg(ffn), arg(norm), arg(n), arg(w), arg(h), arg(ww), arg(sx), arg(sy));
	struct hip_plan_tensor const out = tensor(b, (size_t)(identity ? n : w * h) * c / (byte_out ? 4 : 1));
	// The output's rounding: Hrtz at the end of an encoder level, which is a skip; F of Hrtz at
	// blocks 48, 55, 61 and 65; F elsewhere.
	unsigned const post = raw ? 3 : block == 48 || block == 55 || block == 61 || block == 65 ? 0 : 4;
	struct hip_plan_weight const attention = weight(b, block, "attention", HIP_RECIPE_MH_ATTENTION_DIAG);
	LAUNCH(b, attention_kernels[level][byte_out], n / 64, arg(norm), arg(attention), arg(ffn),
	       arg(out), arg(ww), arg(hh), arg(post), arg(identity ? 0u : w), arg(identity ? 0u : h),
	       arg(sx), arg(sy));
	hip_plan_tensor_drop(b, norm);
	hip_plan_tensor_drop(b, ffn);
	return out;
}

/** @brief A block at 512 channels, with AttentionFast (upstream: Body).
 *
 * Unless the block is an identity (unshifted, on a grid that is already a multiple of 8, as at
 * 1280x768), its input is packed onto the padded, shifted lattice first.
 *
 * @param b     The builder.
 * @param input The input.
 * @param w     The width.
 * @param h     The height.
 * @param shift The block's shift.
 * @param block The block.
 * @param raw   Whether it ends the encoder, whose output is a skip.
 * @return      The output, a new reference: the input's for a block that --performance skips.
 */
static struct hip_plan_tensor
c512_block (struct builder         *b,
            struct hip_plan_tensor  input,
            unsigned                w,
            unsigned                h,
            unsigned                shift,
            unsigned                block,
            bool                    raw)
{
	if (b->performance && (block == 42 || block == 43 || block == 46))
		return hip_plan_tensor_ref(b, input);
	unsigned const sx = shift & 1 ? 4 : 0, sy = shift & 2 ? 4 : 0, ww = (w + sx + 7) & ~7u,
	               hh = (h + sy + 7) & ~7u, n = ww * hh;
	bool const identity = !sx && !sy && ww == w && hh == h;
	struct hip_plan_tensor const packed = identity ? hip_plan_tensor_ref(b, input)
	                                               : tensor(b, (size_t)n * 512);
	if (!identity)
		LAUNCH(b, HIP_KERNEL_SHIFT_PACK, (size_t)n * 512, arg(input), arg(packed), arg(w), arg(h),
		       arg(ww), arg(hh), arg(sx), arg(sy), arg(512u), arg(0u));
	struct hip_plan_tensor const ffn = tensor(b, (size_t)n * 512);
	struct hip_plan_tensor norm;
	{
		struct hip_plan_tensor const mixed = tensor(b, (size_t)n * 512);
		struct hip_plan_tensor const contract = tensor(b, (size_t)n * 512);
		// Upstream also uploads the unpacked ffwd weights, which no kernel reads.
		struct hip_plan_weight const mix = weight(b, block, "ffwd", HIP_RECIPE_SPLIT_MIX_F16);
		LAUNCH(b, HIP_KERNEL_SPLIT_MIX, (size_t)n * 512, arg(packed), arg(mix), arg(mixed), arg(n));
		struct hip_plan_tensor const contract8 = tensor(b, (size_t)n * 128);
		// The kernel also writes CONTRACT, which nothing reads.
		LAUNCH(b, HIP_KERNEL_SPLIT_FFN, (size_t)n * 512, arg(mixed), arg(mix), arg(contract),
		       arg(contract8), arg(n));
		hip_plan_tensor_drop(b, mixed);
		struct hip_plan_tensor const ffn8 = tensor(b, (size_t)n * 128);
		struct hip_plan_weight const projection =
			weight(b, block, "ffwd-projection", HIP_RECIPE_PROJ_FRAG);
		LAUNCH(b, HIP_KERNEL_SPLIT_PROJECTION, (size_t)n * 512, arg(contract8), arg(projection),
		       arg(packed), arg(ffn), arg(ffn8), arg(n));
		norm = tensor(b, (size_t)n * 384);
		struct hip_plan_weight const qkv = weight(b, block, "attention", HIP_RECIPE_QKV_FRAG);
		LAUNCH(b, HIP_KERNEL_QKV_C512, (size_t)n * 1536, arg(ffn8), arg(qkv), arg(norm), arg(n));
		hip_plan_tensor_drop(b, ffn8);
		hip_plan_tensor_drop(b, contract8);
		hip_plan_tensor_drop(b, contract);
	}
	hip_plan_tensor_drop(b, packed);
	struct hip_plan_weight const attention = weight(b, block, "attention", HIP_RECIPE_MH_ATTENTION);
	struct hip_plan_tensor const av = tensor(b, (size_t)n * 128);
	struct hip_plan_tensor const out = tensor(b, (size_t)(identity ? n : w * h) * 512);
	LAUNCH(b, HIP_KERNEL_ATTENTION_C512, (size_t)(n / 64) * 16, arg(norm), arg(attention), arg(av),
	       arg(ww), arg(hh), arg(512u));
	// F of Hrtz, or Hrtz at the end of the encoder, which is a skip.
	if (identity) {
		LAUNCH(b, HIP_KERNEL_PROJECT_SCALAR, (size_t)n * 512, arg(av), arg(ffn), arg(attention),
		       arg(out), arg(n), arg(512u), arg(raw ? 3u : 0u));
	} else {
		struct hip_plan_weight const projection = weight(b, block, "attention", HIP_RECIPE_QKV_FRAG);
		LAUNCH(b, HIP_KERNEL_PROJECT_C512, (size_t)n * 512, arg(av), arg(ffn), arg(projection),
		       arg(out), arg(n), arg(raw ? 3u : 0u), arg(w), arg(h), arg(ww), arg(sx), arg(sy));
	}
	hip_plan_tensor_drop(b, av);
	hip_plan_tensor_drop(b, norm);
	hip_plan_tensor_drop(b, ffn);
	return out;
}

/** @brief Pools a w-wide level's output and projects it to twice its channels (upstream: Down).
 *
 * @param b       The builder.
 * @param raw     The level's output.
 * @param w       Its width.
 * @param c       Its channels.
 * @param weights The projection's weights.
 * @param p       The tokens kept.
 * @return        The output, a new reference.
 */
static struct hip_plan_tensor
down (struct builder         *b,
      struct hip_plan_tensor  raw,
      unsigned                w,
      unsigned                c,
      struct hip_plan_weight  weights,
      struct pooled           p)
{
	struct hip_plan_tensor const out = tensor(b, (size_t)p.ow * p.oh * c * 2);
	LAUNCH(b, pool_group_kernels[__builtin_ctz(c) - 6], (size_t)p.ow * p.oh * c * 2, arg(raw),
	       arg(weights), arg(out), arg(p.ow), arg(p.oh), arg(w), arg(p.vw), arg(p.vh));
	return out;
}

/** @brief Permutes the tokens into or out of the ViT's layout (upstream: Gather).
 *
 * @param b       The builder.
 * @param input   The tokens.
 * @param tokens  Their number.
 * @param inverse Whether out of it.
 * @return        The output, a new reference.
 */
static struct hip_plan_tensor
gather (struct builder         *b,
        struct hip_plan_tensor  input,
        unsigned                tokens,
        bool                    inverse)
{
	struct hip_plan_tensor const out = tensor(b, (size_t)tokens * 1024);
	struct hip_arg const map = {.value = inverse, .kind = HIP_ARG_GATHER};
	LAUNCH(b, HIP_KERNEL_VIT_GATHER, (size_t)tokens * 1024, arg(input), arg(map), arg(out),
	       arg(tokens * 1024));
	return out;
}

/** @brief A ViT block; each kernel takes a trailing gate for the adaptive reuse, null without it
 *         (upstream: Vit).
 *
 * @param b     The builder.
 * @param input The input.
 * @param n     The tokens.
 * @param block The block.
 * @return      The output, a new reference.
 */
static struct hip_plan_tensor
vit_block (struct builder         *b,
           struct hip_plan_tensor  input,
           unsigned                n,
           unsigned                block)
{
	struct hip_plan_tensor const packed = tensor(b, (size_t)n * 256);
	LAUNCH(b, HIP_KERNEL_VIT_PACK, (size_t)n * 256, arg(input), arg(packed), arg(n * 1024), arg(nullptr));
	struct hip_plan_tensor const hidden = tensor(b, (size_t)n * 1024);
	struct hip_plan_tensor const contract = tensor(b, (size_t)n * 1024);
	struct hip_plan_weight const expand = weight(b, block, "expand", HIP_RECIPE_VIT_FRAG);
	LAUNCH(b, HIP_KERNEL_VIT_EXPAND, (size_t)n * 4096, arg(packed), arg(expand), arg(hidden), arg(n),
	       arg(1024u), arg(4096u), arg(nullptr));
	hip_plan_tensor_drop(b, packed);
	struct hip_plan_weight const contraction = weight(b, block, "contract", HIP_RECIPE_VIT_FRAG);
	LAUNCH(b, HIP_KERNEL_VIT_CONTRACT, (size_t)n * 1024, arg(hidden), arg(contraction), arg(input),
	       arg(contract), arg(n), arg(4096u), arg(1024u), arg(nullptr));
	hip_plan_tensor_drop(b, hidden);
	struct hip_plan_tensor const norm = tensor(b, (size_t)n * 768);
	struct hip_plan_weight const qkv = weight(b, block, "qkv", HIP_RECIPE_QKV_F16_FRAG);
	LAUNCH(b, HIP_KERNEL_VIT_QKV, (size_t)n * 3072, arg(contract), arg(qkv), arg(norm), arg(n),
	       arg(nullptr));
	struct hip_plan_tensor const av = tensor(b, (size_t)n * 1024);
	LAUNCH(b, vit_attention_kernels[n <= 256 ? 0 : n <= 400 ? 1 : 2], (size_t)n * 512, arg(norm),
	       arg(av), arg(n), arg(nullptr));
	hip_plan_tensor_drop(b, norm);
	struct hip_plan_tensor const out = tensor(b, (size_t)n * 1024);
	struct hip_plan_weight const projection = weight(b, block, "projection", HIP_RECIPE_VIT_PROJ_FRAG);
	LAUNCH(b, HIP_KERNEL_VIT_PROJECT, (size_t)n * 1024, arg(av), arg(projection), arg(contract),
	       arg(out), arg(n), arg(1024u), arg(1024u), arg(nullptr));
	hip_plan_tensor_drop(b, av);
	hip_plan_tensor_drop(b, contract);
	return out;
}

/** @brief The ViT's blocks 31 to 38 (upstream: AdaptiveVitGroup without its approximate reuse).
 *
 * @param b      The builder.
 * @param input  The input.
 * @param tokens The tokens.
 * @return       The output, a new reference.
 */
static struct hip_plan_tensor
vit (struct builder         *b,
     struct hip_plan_tensor  input,
     unsigned                tokens)
{
	struct hip_plan_tensor full = hip_plan_tensor_ref(b, input);
	for (unsigned block = 31; block <= 38; ++block) {
		struct hip_plan_tensor const next = vit_block(b, full, tokens, block);
		hip_plan_tensor_drop(b, full);
		full = next;
	}
	return full;
}

/** @brief Projects iw x ih tokens of ic channels onto the skip's ow x oh of oc channels, twice the
 *         size (upstream: Up).
 *
 * @param b        The builder.
 * @param input    The input.
 * @param skip     The skip.
 * @param iw       The input's width.
 * @param ih       Its height.
 * @param ow       The output's width.
 * @param oh       Its height.
 * @param ic       The input's channels.
 * @param oc       The output's.
 * @param weights  The projection's weights.
 * @param byte_out Whether the output is bytes.
 * @return         The output, a new reference.
 */
static struct hip_plan_tensor
up (struct builder         *b,
    struct hip_plan_tensor  input,
    struct hip_plan_tensor  skip,
    unsigned                iw,
    unsigned                ih,
    unsigned                ow,
    unsigned                oh,
    unsigned                ic,
    unsigned                oc,
    struct hip_plan_weight  weights,
    bool                    byte_out)
{
	struct hip_plan_tensor const out = tensor(b, (size_t)ow * oh * oc / (byte_out ? 4 : 1));
	LAUNCH(b, byte_out ? HIP_KERNEL_DECODER_BYTEOUT : HIP_KERNEL_DECODER, (size_t)iw * ih * oc,
	       arg(input), arg(weights), arg(skip), arg(out), arg(iw), arg(ih), arg(ow), arg(oh), arg(ic),
	       arg(oc));
	return out;
}

/** @brief Replaces a reference with another, as upstream's assignment of a new Tensor does: the
 *         old one is dropped once the new one exists.
 *
 * @param b    The builder.
 * @param dest The reference to replace.
 * @param next The new one.
 */
static void
replace (struct builder         *b,
         struct hip_plan_tensor *dest,
         struct hip_plan_tensor  next)
{
	hip_plan_tensor_drop(b, *dest);
	*dest = next;
}

/** @brief Drops a reference and empties it (upstream: Tensor::reset).
 *
 * @param b    The builder.
 * @param dest The reference.
 */
static void
reset (struct builder         *b,
       struct hip_plan_tensor *dest)
{
	hip_plan_tensor_drop(b, *dest);
	*dest = (struct hip_plan_tensor){0};
}

/** @brief Records upstream's RunGraph's production path, which reads the frame in place, with the
 *         prefix fused into block 0 and the RGB head into the post. Every tensor is returned when
 *         it returns, so the plan is complete.
 *
 * @param b The builder.
 */
static void
run (struct builder *b)
{
	unsigned const W = b->plan->width, H = b->plan->height;
	// Block 0 reads the frame's input, and its history when it has one.
	unsigned const windows = W * H / 64;
	struct hip_plan_tensor const skip0 = tensor(b, (size_t)W * H * 8);
	struct hip_plan_tensor source = tensor(b, (size_t)W * H / 4 * 32);
	{
		struct hip_plan_weight const attention = weight(b, 0, "attention", HIP_RECIPE_C32);
		struct hip_plan_weight const ffn = weight(b, 0, "ffn", HIP_RECIPE_C32);
		LAUNCH(b, HIP_KERNEL_C32_PREFIX, windows, arg(RGBA), arg(HISTORY), arg(ffn), arg(attention),
		       arg(skip0), arg(source), arg(windows), arg(0u), arg(1u), arg(W), arg(H), arg(0u),
		       arg(TEMPORAL));
	}
	struct hip_plan_tensor skips[5] = {0};
	struct c32 last = {0};
	for (unsigned block = 1; block <= 4; ++block) {
		struct c32 const next = c32_chain(b, source, last.raw.ref ? &last : nullptr, W / 2, H / 2,
		                                  encoder_shift[block - 1], block, block == 4, block == 4);
		c32_drop(b, &last);
		last = next;
		replace(b, &source, hip_plan_tensor_ref(b, last.main));
	}
	skips[0] = hip_plan_tensor_ref(b, source);
	replace(b, &source, tensor(b, (size_t)(W / 4) * (H / 4) * 64));
	struct hip_plan_weight const ds = weight(b, 4, "ds", HIP_RECIPE_DS_CAST);
	LAUNCH(b, HIP_KERNEL_POOL32, (size_t)(W / 4) * (H / 4) * 64, arg(last.down), arg(ds), arg(source),
	       arg(W / 4), arg(H / 4), arg(0u), arg(0u), arg(32u));
	c32_drop(b, &last);

	for (unsigned g = 0; g < 3; ++g) {
		struct level const *const l = &encoder_levels[g];
		unsigned const w = W / (4u << g), h = H / (4u << g);
		for (unsigned block = l->first; block <= l->last; ++block)
			replace(b, &source, mh_block(b, source, w, h, l->c, encoder_shift[block - l->first], block,
			                             block == l->last, block > l->first, block < l->last));
		skips[g + 1] = hip_plan_tensor_ref(b, source);
		struct hip_plan_weight const weights = weight(b, l->last, "ds", HIP_RECIPE_DS_FRAG);
		replace(b, &source, down(b, source, w, l->c, weights, (struct pooled){w / 2, h / 2, 0, 0}));
	}
	for (unsigned j = 0; j < 8; ++j)
		replace(b, &source, c512_block(b, source, W / 32, H / 32, encoder_shift[j], 23 + j, j == 7));
	skips[4] = hip_plan_tensor_ref(b, source);

	// The head pools to the ViT's token grid, padded to a multiple of 16 tokens: at 1600x960 by
	// a row, at 1920x1152 to 32x20. Upstream works the grid out again for the ViT by another
	// rule, which agrees at every tier.
	unsigned const w = W / 32, h = H / 32, rw = w / 2, rh = h / 2;
	struct pooled head = w == 60 && h == 36 ? (struct pooled){32, 20, 0, 0}
	                                        : (struct pooled){rw, (rw * rh) % 16 ? rh + 1 : rh, 0, 0};
	if (head.ow * head.oh != rw * rh) {
		head.vw = rw;
		head.vh = rh;
	}
	unsigned const tokens = head.ow * head.oh;
	b->plan->tokens = tokens;
	struct hip_plan_weight const matrix = NAMED_WEIGHT(b, "head-matrix", HIP_RECIPE_DS_FRAG);
	replace(b, &source, down(b, source, w, 512, matrix, head));
	replace(b, &source, gather(b, source, tokens, false));
	replace(b, &source, vit(b, source, tokens));
	replace(b, &source, gather(b, source, tokens, true));
	struct hip_plan_weight const decoder = NAMED_WEIGHT(b, "decoder39-weights", HIP_RECIPE_DECODER_F16R);
	replace(b, &source, up(b, source, skips[4], head.ow, head.oh, w, h, 1024, 512, decoder, false));
	reset(b, &skips[4]);

	for (unsigned block = 40; block <= 47; ++block)
		replace(b, &source, c512_block(b, source, w, h, decoder_shift[block - 40], block, false));
	for (unsigned g = 0; g < 3; ++g) {
		struct level const *const l = &decoder_levels[g];
		unsigned const ow = W / (16u >> g), oh = H / (16u >> g);
		struct hip_plan_weight const weights = weight(b, l->first, "weights", HIP_RECIPE_DECODER_F16R);
		replace(b, &source, up(b, source, skips[3 - g], ow / 2, oh / 2, ow, oh, 2u * l->c, l->c,
		                       weights, true));
		reset(b, &skips[3 - g]);
		for (unsigned block = l->first; block <= l->last; ++block)
			replace(b, &source, mh_block(b, source, ow, oh, l->c, decoder_shift[block - 40], block,
			                             false, true, block < l->last));
	}
	struct hip_plan_weight const weights = weight(b, 66, "weights", HIP_RECIPE_DECODER_F16R);
	replace(b, &source, up(b, source, skips[0], W / 4, H / 4, W / 2, H / 2, 64, 32, weights, false));
	reset(b, &skips[0]);
	struct c32 chain = {0};
	for (unsigned block = 66; block <= 69; ++block) {
		struct c32 const next = c32_chain(b, source, chain.raw.ref ? &chain : nullptr, W / 2, H / 2,
		                                  decoder_shift[block - 40], block, block == 69, false);
		c32_drop(b, &chain);
		chain = next;
		replace(b, &source, hip_plan_tensor_ref(b, chain.main));
	}
	c32_drop(b, &chain);

	// The post merges block 0's bytes back in, and writes the frame's RGB.
	unsigned const post_windows = (W + 8) * (H + 8) / 64;
	struct hip_plan_weight const head_weights = NAMED_WEIGHT(b, "post70-head", HIP_RECIPE_RAW);
	struct hip_plan_weight const attention = NAMED_WEIGHT(b, "post70-attention", HIP_RECIPE_C32);
	struct hip_plan_weight const ffn = NAMED_WEIGHT(b, "post70-ffn", HIP_RECIPE_C32);
	struct hip_plan_weight const scales = NAMED_WEIGHT(b, "post70-scales", HIP_RECIPE_RAW);
	LAUNCH(b, HIP_KERNEL_C32_POST, post_windows, arg(source), arg(skip0), arg(scales), arg(ffn),
	       arg(attention), arg(RGBA), arg(head_weights), arg(OUTPUT), arg(post_windows), arg(W), arg(H),
	       arg(4u), arg(4u), arg(.03125f));
	hip_plan_tensor_drop(b, source);
	hip_plan_tensor_drop(b, skip0);
}

#undef NAMED_WEIGHT
#undef LAUNCH
#undef OUTPUT
#undef TEMPORAL
#undef HISTORY
#undef RGBA
#undef arg

enum error_code
hip_plan_init (struct hip_plan *dest,
               unsigned         width,
               unsigned         height,
               bool             performance,
               struct error    *e)
{
	*dest = (struct hip_plan){0};
	bool supported = false;
	for (size_t i = 0; i < sizeof geometries / sizeof *geometries; ++i)
		supported |= geometries[i].width == width && geometries[i].height == height;
	if (!supported)
		return error_fail(e, "unsupported processing geometry %ux%u", width, height);

	dest->width = (uint16_t)width;
	dest->height = (uint16_t)height;
	struct builder b = {.plan = dest, .e = e, .performance = performance};
	run(&b);
	free(b.references);
	b.references = nullptr;
	if (b.error)
		hip_plan_fini(dest);
	return b.error;
}

void
hip_plan_fini (struct hip_plan *plan)
{
	if (!plan)
		return;
	free(plan->launches);
	plan->launches = nullptr;
	free(plan->floats);
	plan->floats = nullptr;
	free(plan->events);
	plan->events = nullptr;
	free(plan->weights);
	plan->weights = nullptr;
	*plan = (struct hip_plan){0};
}

void
hip_plan_gather_map (uint32_t *map,
                     unsigned  tokens,
                     bool      inverse)
{
	for (uint32_t t = 0; t < tokens; ++t) {
		for (uint32_t c = 0; c < 1024; ++c) {
			uint32_t const raster = (t & ~15u) | ((t & 1u) << 3) | ((t & 14u) >> 1),
			               channel = (c & ~31u) | ((c & 1u) << 1) | ((c & 2u) >> 1) | ((c & 4u) << 2) |
			                         ((c & 24u) >> 1),
			               to = t * 1024 + c, from = raster * 1024 + channel;
			if (inverse)
				map[from] = to;
			else
				map[to] = from;
		}
	}
}

/** @brief Assigns a plan's tensors to the pool's buffers as upstream's New() does over its pool,
 *         which only grows.
 *
 * @param p         The placement, whose buffers the assignment starts with, all free.
 * @param plan      The plan.
 * @param buffer_of Receives each tensor's buffer.
 * @param taken     Scratch of a flag per tensor.
 * @param grow      Whether a tensor that no free buffer holds adds a buffer; else it is a failure.
 * @return          false on such a failure.
 */
static bool
assign (struct hip_placement  *p,
        struct hip_plan const *plan,
        uint16_t              *buffer_of,
        bool                  *taken,
        bool                   grow)
{
	memset(buffer_of, 0, plan->tensor_count * sizeof *buffer_of);
	memset(taken, 0, p->buffer_count * sizeof *taken);
	for (size_t i = 0; i < plan->event_count; ++i) {
		int32_t const event = plan->events[i];
		if (event < 0) {
			taken[buffer_of[~event]] = false;
			continue;
		}
		size_t const bytes = (size_t)plan->floats[event] * 4, none = p->buffer_count;
		size_t best = none;
		for (size_t j = 0; j < p->buffer_count; ++j)
			if (!taken[j] && p->buffers[j] >= bytes && (best == none || p->buffers[j] < p->buffers[best]))
				best = j;
		if (best == none) {
			if (!grow)
				return false;
			// One buffer at most per tensor, which the arrays hold.
			p->buffers[p->buffer_count] = bytes;
			taken[p->buffer_count++] = false;
		}
		taken[best] = true;
		buffer_of[event] = (uint16_t)best;
	}
	return true;
}

enum error_code
hip_plan_place (struct hip_placement  *dest,
                struct hip_plan const *plan,
                struct error          *e)
{
	*dest = (struct hip_placement){0};
	size_t const n = plan->tensor_count;
	if (!n)
		return ERROR_NONE;
	dest->buffers = malloc(n * sizeof *dest->buffers);
	dest->first = malloc(n * sizeof *dest->first);
	dest->later = malloc(n * sizeof *dest->later);
	bool *taken = malloc(n * sizeof *taken);
	enum error_code code = ERROR_NONE;
	if (!dest->buffers || !dest->first || !dest->later || !taken) {
		code = error_fail(e, "out of memory");
	} else {
		assign(dest, plan, dest->first, taken, true);
		if (!assign(dest, plan, dest->later, taken, false))
			code = error_fail(e, "the HIP network's buffers do not suffice after its first frame");
	}
	free(taken);
	taken = nullptr;
	if (code)
		hip_placement_fini(dest);
	return code;
}

void
hip_placement_fini (struct hip_placement *placement)
{
	if (!placement)
		return;
	free(placement->buffers);
	placement->buffers = nullptr;
	free(placement->first);
	placement->first = nullptr;
	free(placement->later);
	placement->later = nullptr;
	placement->buffer_count = 0;
}
