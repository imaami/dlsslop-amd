/** @file
 *
 * Host test of the HIP network's launch plan, without a GPU or a model. At every tier and preset
 * its launches, buffers and uploads are checked against traces that the tracing HIP runtime
 * (tests/hiptrace) took of upstream's network (hip_reference_network.h at c190831) in dlsslopd:
 * hashes of the launch lists with buffers named by first use, the pool's buffers, the weights and
 * the gather maps each launch reads. tests/hiptrace/plan_hashes.py prints these values from a
 * trace. --modules checks the launches against the built kernels' metadata instead.
 */
// SPDX-License-Identifier: MIT
#include <elf.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "files.h"
#include "hip_plan.h"
#include "shm_protocol.h"
#include "support.h"

/** @brief What the traces show at a tier and preset. */
struct traced {
	size_t   launches;    //!< The launches per frame.
	size_t   kernels;     //!< The distinct kernels among them.
	uint64_t first;       //!< The FNV-1a 64 of the first frame's canonical launch list.
	uint64_t later;       //!< That of a later frame.
	uint64_t history;     //!< That of a later frame with a history; 0: not traced.
	uint64_t uploads;     //!< That of the weight uploads that a launch reads (upload_text()).
	uint32_t tier;        //!< The tier.
	uint32_t performance; //!< 1 with --performance, else 0.
};

/** @brief The traced tiers and presets. */
static struct traced const traced_frames[] = {
	{250, 43, 0xa2dae6593e416beeu, 0x1f7c6f29d1c28a84u, 0x240a88c7eb4b11a6u, 0x0ee01bec21d79f03u, 720, 0},
	{229, 43, 0x834bc420b12694dcu, 0x892cd4d1d3db66a5u, 0, 0xfd99729c88d07861u, 720, 1},
	{254, 41, 0x32d0cdfcc08b67e2u, 0x0c823eac9f698da2u, 0, 0xf023500cbbedf100u, 900, 0},
	{233, 41, 0xff9623a799b34742u, 0x42c5410c76dcbbb9u, 0, 0x38d51bf165148771u, 900, 1},
	{254, 42, 0x18bff6041650c106u, 0xdffb8c9588486891u, 0x4f31a38dc45a9974u, 0xf023500cbbedf100u, 1080, 0},
	{233, 42, 0x1f5cb0ebce0e78d8u, 0xfb73f9fd4c0be465u, 0, 0x38d51bf165148771u, 1080, 1},
};

/** @brief Per tier, with either preset: the pool's buffers in creation order, ending with 0, and
 *         the FNV-1a 64 of the gather maps in the order the launches first read them. */
struct traced_tier {
	size_t   buffers[16]; //!< The buffers' bytes.
	uint64_t gather[2];   //!< The maps' hashes.
	size_t   tier;        //!< The tier.
};

/** @brief The traced tiers. */
static struct traced_tier const traced_tiers[] = {
	{{31457280, 31457280, 15728640, 15925248, 11796480, 3932160, 1966080, 7864320, 1966080, 491520,
	  1474560, 3145728, 3145728, 3145728, 31457280},
	 {0x9089c42dbe84bd25u, 0xa6bc7ecb30496925u}, 720},
	{{49152000, 49152000, 24576000, 24821760, 18432000, 6144000, 3072000, 12288000, 3670016, 3670016,
	  917504, 2752512, 4587520, 4587520, 49152000},
	 {0x87ab221ca3c6a125u, 0xc8863327344d5b25u}, 900},
	{{70778880, 70778880, 35389440, 35684352, 26542080, 8847360, 4423680, 17694720, 5242880, 5242880,
	  1310720, 3932160, 5242880, 70778880},
	 {0xd4e847b0089b1b25u, 0xe7cf48cabb5e6325u}, 1080},
};

/** @brief Without and with --performance: the weights uploaded, their bytes, and the FNV-1a 64 of
 *         identity_text().
 *
 * The traces hold only the payloads. The names are those under which the pinned upstream packers,
 * run on the real model, made the same payloads in the same order; no two payloads are alike.
 * Upstream also uploads 32 weights, 29 with --performance, that no launch reads, and dlsslopd
 * leaves them out: these values and the upload hashes above are of the traced uploads without
 * them. A weight that no launch reads fails the upload hash, as upload_text() writes "BYTES -" for
 * it.
 */
struct traced_weights {
	size_t   count;    //!< The weights.
	size_t   bytes;    //!< Their bytes.
	uint64_t identity; //!< The hash of their names.
};

/** @brief The traced weights without and with --performance. */
static struct traced_weights const traced_weights[] = {
	{236, 644222504, 0x675dbc4f8229a0b1u},
	{224, 608027816, 0xec181f70bf0130c9u},
};

/** @brief The checks that failed. */
static unsigned failures;

/** @brief Fails the test with a message unless a condition holds.
 *
 * @param ok  The condition.
 * @param fmt A printf format for the message.
 * @param ... The format's arguments.
 * @return    @a ok.
 */
[[gnu::format(printf, 2, 3)]]
static bool
expect (bool        ok,
        char const *fmt,
        ...)
{
	if (ok)
		return true;

	va_list args;
	va_start(args, fmt);
	fputs("hip-plan test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	++failures;
	return false;
}

/** @brief Ends the test without memory for a value.
 *
 * @param ok Whether the value's memory was allocated.
 */
static void
allocated (bool ok)
{
	if (!ok) {
		fputs("hip-plan test: out of memory\n", stderr);
		exit(1);
	}
}

/** @brief Appends formatted words to a text, or ends the test without memory for them.
 *
 * @param text The text.
 * @param fmt  A printf format.
 * @param ...  The format's arguments.
 */
[[gnu::format(printf, 2, 3)]]
static void
add (struct support_text *text,
     char const          *fmt,
     ...)
{
	va_list args;
	va_start(args, fmt);
	bool const ok = support_text_vprintf(text, fmt, args);
	va_end(args);
	allocated(ok);
}

/** @brief The FNV-1a 64 of bytes.
 *
 * @param data  The bytes.
 * @param bytes Their number.
 * @return      The hash.
 */
static uint64_t
fnv1a (void const *data,
       size_t      bytes)
{
	uint64_t hash = 0xcbf29ce484222325u;
	for (unsigned char const *at = data; bytes--; ++at)
		hash = (hash ^ *at) * 0x100000001b3u;
	return hash;
}

/** @brief The FNV-1a 64 of a text's words. */
static uint64_t
text_hash (struct support_text const *text)
{
	return fnv1a(support_text_string(text), text->length);
}

/** @brief Whether an argument is passed as a pointer. */
static bool
pointer (struct hip_arg a)
{
	return a.kind != HIP_ARG_U32 && a.kind != HIP_ARG_F32 && a.kind != HIP_ARG_TEMPORAL;
}

/** @brief Writes the launch list as the trace tools write a traced one: a line per launch,
 *         "MODULE:KERNEL\tGRIDx1x1\tTHREADSx1x1\t0\tARGUMENTS", with each buffer named B<n>+0 in
 *         the order of first use. A frame without a history passes its input in its stead.
 *
 * @param plan      The plan.
 * @param buffer_of Each tensor's buffer.
 * @param history   Whether the frame has a history.
 * @param text      Receives the list.
 */
static void
canonical_text (struct hip_plan const *plan,
                uint16_t const        *buffer_of,
                bool                   history,
                struct support_text   *text)
{
	// Each buffer by first use: at most one per argument.
	uint64_t *names = malloc(plan->launch_count * HIP_LAUNCH_ARGS * sizeof *names);
	allocated(names);
	size_t named = 0;
	for (size_t n = 0; n < plan->launch_count; ++n) {
		struct hip_launch const *const l = &plan->launches[n];
		struct hip_kernel_info const *const k = &HIP_PLAN_KERNELS[l->kernel];
		add(text, "%s:%s\t%" PRIu32 "x1x1\t%ux1x1\t0\t", HIP_PLAN_MODULE_FILES[k->module], k->name,
		    l->grid, (unsigned)k->threads);
		for (uint16_t i = 0; i < l->count; ++i) {
			struct hip_arg const a = l->args[i];
			if (i)
				add(text, ",");
			if (a.kind == HIP_ARG_NULL) {
				add(text, "null");
				continue;
			}
			if (!pointer(a)) {
				add(text, "%" PRIu32, a.kind == HIP_ARG_TEMPORAL ? (uint32_t)history : a.value);
				continue;
			}
			enum hip_arg_kind const kind = a.kind == HIP_ARG_HISTORY && !history ? HIP_ARG_RGBA : a.kind;
			uint64_t const buffer = (uint64_t)kind << 32 |
			                        (kind == HIP_ARG_TENSOR ? buffer_of[a.value] : a.value);
			size_t name = 0;
			while (name < named && names[name] != buffer)
				++name;
			if (name == named)
				names[named++] = buffer;
			add(text, "B%zu+0", name);
		}
		add(text, "\n");
	}
	free(names);
	names = nullptr;
}

/** @brief Writes the weight uploads as the trace tools list them: a line per weight, in upload
 *         order, of its bytes and the first launch and argument that read it in the first frame,
 *         "BYTES LAUNCH ARGUMENT", or "BYTES -" when none does.
 *
 * @param plan The plan.
 * @param text Receives the list.
 */
static void
upload_text (struct hip_plan const *plan,
             struct support_text   *text)
{
	for (uint32_t w = 0; w < plan->weight_count; ++w) {
		add(text, "%zu", hip_weights_packed_bytes(&plan->weights[w]));
		bool used = false;
		for (size_t n = 0; n < plan->launch_count && !used; ++n) {
			for (uint16_t i = 0; i < plan->launches[n].count; ++i) {
				struct hip_arg const a = plan->launches[n].args[i];
				if (a.kind == HIP_ARG_WEIGHT && a.value == w) {
					add(text, " %zu %u", n, (unsigned)i);
					used = true;
					break;
				}
			}
		}
		add(text, used ? "\n" : " -\n");
	}
}

/** @brief Writes which weight each upload is, a line per weight in upload order: "STEM@SUFFIX C",
 *         or "STEM C" for a raw one, with the channel count upstream passes its loader.
 *
 * @param plan The plan.
 * @param text Receives the list.
 */
static void
identity_text (struct hip_plan const *plan,
               struct support_text   *text)
{
	for (size_t w = 0; w < plan->weight_count; ++w) {
		struct hip_weight_spec const *const spec = &plan->weights[w];
		char const *const suffix = HIP_WEIGHTS_RECIPE_SUFFIX[spec->recipe];
		add(text, "%s%s%s %u\n", spec->stem, *suffix ? "@" : "", suffix, hip_weights_channels(spec));
	}
}

/** @brief The traced values of a tier and preset, or nullptr. */
static struct traced const *
traced_of (unsigned tier,
           bool     performance)
{
	for (size_t i = 0; i < sizeof traced_frames / sizeof *traced_frames; ++i)
		if (traced_frames[i].tier == tier && traced_frames[i].performance == performance)
			return &traced_frames[i];
	return nullptr;
}

/** @brief The traced pool of a tier, or nullptr. */
static struct traced_tier const *
traced_tier_of (unsigned tier)
{
	for (size_t i = 0; i < sizeof traced_tiers / sizeof *traced_tiers; ++i)
		if (traced_tiers[i].tier == tier)
			return &traced_tiers[i];
	return nullptr;
}

/** @brief Checks a tier's and preset's placement, launch lists, weights and gather maps.
 *
 * @param plan     The plan.
 * @param tier     The tier.
 * @param preset   " --performance" or "".
 * @param traced   The traced values.
 * @param pool     The traced pool.
 * @param w        The traced weights of the preset.
 * @param identity Receives the plan's identity_text().
 * @return         false if the plan's tensors could not be placed, which ends the checks.
 */
static bool
check_plan (struct hip_plan const        *plan,
            unsigned                      tier,
            char const                   *preset,
            struct traced const          *traced,
            struct traced_tier const     *pool,
            struct traced_weights const  *w,
            struct support_text          *identity)
{
	struct error e;
	struct hip_placement placement;
	if (!expect(!hip_plan_place(&placement, plan, &e), "tier %u%s: %s", tier, preset, e.what))
		return false;
	size_t buffers = 0;
	while (pool->buffers[buffers])
		++buffers;
	expect(buffers == placement.buffer_count &&
	       !memcmp(pool->buffers, placement.buffers, buffers * sizeof *placement.buffers),
	       "tier %u%s: the pool's buffers differ from upstream's", tier, preset);
	// The first frame, a later one, and a later one with a history.
	char const *const frames[] = {"the first frame", "later frames", "frames with a history"};
	uint64_t const hashes[] = {traced->first, traced->later, traced->history};
	for (size_t f = 0; f < sizeof frames / sizeof *frames; ++f) {
		struct support_text text = {0};
		canonical_text(plan, f ? placement.later : placement.first, f == 2, &text);
		uint64_t const hash = text_hash(&text);
		support_text_fini(&text);
		expect(!hashes[f] || hash == hashes[f],
		       "tier %u%s: the launches of %s hash to %016" PRIx64 ", upstream's to %016" PRIx64, tier,
		       preset, frames[f], hash, hashes[f]);
	}
	hip_placement_fini(&placement);

	size_t bytes = 0;
	for (size_t i = 0; i < plan->weight_count; ++i)
		bytes += hip_weights_packed_bytes(&plan->weights[i]);
	expect(plan->weight_count == w->count && bytes == w->bytes,
	       "tier %u%s: %zu weights of %zu bytes; upstream's launches read %zu of %zu", tier, preset,
	       plan->weight_count, bytes, w->count, w->bytes);
	identity_text(plan, identity);
	expect(text_hash(identity) == w->identity, "tier %u%s: other weights uploaded than upstream's launches read",
	       tier, preset);
	struct support_text uploads = {0};
	upload_text(plan, &uploads);
	expect(text_hash(&uploads) == traced->uploads,
	       "tier %u%s: weights uploaded otherwise than upstream's launches read them", tier, preset);
	support_text_fini(&uploads);

	// The gather maps, in the order the launches first read them.
	uint32_t maps[2];
	size_t read = 0;
	uint32_t *map = malloc((size_t)plan->tokens * 1024 * sizeof *map);
	allocated(map);
	for (size_t n = 0; n < plan->launch_count; ++n) {
		for (uint16_t i = 0; i < plan->launches[n].count; ++i) {
			struct hip_arg const a = plan->launches[n].args[i];
			if (a.kind != HIP_ARG_GATHER)
				continue;
			bool seen = false;
			for (size_t m = 0; m < read && m < 2; ++m)
				seen |= maps[m] == a.value;
			if (seen)
				continue;
			hip_plan_gather_map(map, plan->tokens, a.value);
			expect(read < 2 && fnv1a(map, (size_t)plan->tokens * 1024 * sizeof *map) == pool->gather[read],
			       "tier %u%s: the gather map launch %zu reads differs from upstream's", tier, preset, n);
			if (read < 2)
				maps[read] = a.value;
			++read;
		}
	}
	free(map);
	map = nullptr;
	expect(read == 2, "tier %u%s: %zu gather maps read; upstream reads 2", tier, preset, read);
	return true;
}

/** @brief Checks the plans at every tier and preset against the traces. */
static void
check_plans (void)
{
	struct error e;
	struct hip_plan plan;
	enum error_code code = hip_plan_init(&plan, 1280, 720, false, &e);
	expect(code && !strcmp(e.what, "unsupported processing geometry 1280x720"),
	       "a 1280x720 plan does not fail as unsupported");
	if (!code)
		hip_plan_fini(&plan);

	bool launched[HIP_KERNEL_COUNT] = {0};
	struct support_text weights[2] = {0};
	for (size_t t = 0; t < kNativeTierCount; ++t) {
		struct NativeTier const *const tier = &kNativeTiers[t];
		for (uint8_t performance = 0; performance < 2; ++performance) {
			struct traced const *const traced = traced_of(tier->height, performance);
			struct traced_tier const *const pool = traced_tier_of(tier->height);
			char const *const preset = performance ? " --performance" : "";
			if (!expect(traced && pool, "tier %u%s has no traced values", tier->height, preset))
				continue;
			if (!expect(!hip_plan_init(&plan, tier->width, tier->networkHeight, performance, &e),
			            "tier %u%s: %s", tier->height, preset, e.what))
				continue;
			bool kernels[HIP_KERNEL_COUNT] = {0};
			size_t distinct = 0;
			for (size_t n = 0; n < plan.launch_count; ++n) {
				distinct += !kernels[plan.launches[n].kernel];
				kernels[plan.launches[n].kernel] = true;
				launched[plan.launches[n].kernel] = true;
			}
			expect(plan.launch_count == traced->launches && distinct == traced->kernels,
			       "tier %u%s: %zu launches of %zu kernels, upstream's %zu of %zu", tier->height, preset,
			       plan.launch_count, distinct, traced->launches, traced->kernels);

			struct support_text identity = {0};
			if (check_plan(&plan, tier->height, preset, traced, pool, &traced_weights[performance],
			               &identity)) {
				// One model serves every tier.
				if (!weights[performance].length) {
					weights[performance] = identity;
					identity = (struct support_text){0};
				} else {
					expect(support_text_equal(&identity, &weights[performance]),
					       "tier %u%s: weights differ from tier %u's", tier->height, preset,
					       kNativeTiers[0].height);
				}
			}
			support_text_fini(&identity);
			hip_plan_fini(&plan);
		}
	}
	support_text_fini(&weights[0]);
	support_text_fini(&weights[1]);

	// At the tiers, the first block of the c64 and c128 levels is never mapped.
	for (uint8_t k = 0; k < HIP_KERNEL_COUNT; ++k) {
		bool const unused = k == HIP_KERNEL_FFN_C64_MAPPED || k == HIP_KERNEL_FFN_C128_MAPPED;
		expect(launched[k] != unused, "%s is %slaunched", HIP_PLAN_KERNELS[k].name, unused ? "" : "never ");
	}
}

/** @brief A msgpack reader, for as much of it as the AMDGPU metadata uses. */
struct msgpack {
	uint8_t const *at;  //!< The next byte, or nullptr once a value read was not whole.
	uint8_t const *end; //!< The end of the bytes.
};

/** @brief Whether every value a reader read was whole. */
static bool
msgpack_ok (struct msgpack const *m)
{
	return m->at;
}

/** @brief A value, but for a container's items: a string's text, an unsigned integer, or a
 *         container's length. */
struct msgpack_head {
	char const *text;   //!< A string's text, not null-terminated, or nullptr.
	size_t      length; //!< Its length.
	uint64_t    value;  //!< A number, or a container's items.
	enum : uint8_t {
		MSGPACK_OTHER,
		MSGPACK_STRING,
		MSGPACK_NUMBER,
		MSGPACK_ARRAY,
		MSGPACK_MAP,
	} type; //!< What it is.
};

/** @brief The next bytes, big-endian.
 *
 * @param m The reader.
 * @param n Their number, at most 8.
 * @return  Their value; 0 when the bytes end first, which the reader keeps.
 */
static uint64_t
msgpack_take (struct msgpack *m,
              size_t          n)
{
	if (!m->at || (size_t)(m->end - m->at) < n) {
		m->at = nullptr;
		return 0;
	}
	uint64_t value = 0;
	for (size_t i = 0; i < n; ++i)
		value = value << 8 | m->at[i];
	m->at += n;
	return value;
}

/** @brief Reads the next head.
 *
 * @param m The reader.
 * @return  The head.
 */
static struct msgpack_head
msgpack_head (struct msgpack *m)
{
	uint8_t const c = (uint8_t)msgpack_take(m, 1);
	size_t bytes = 0;
	bool string = false;
	if (c <= 0x7f)
		return (struct msgpack_head){.type = MSGPACK_NUMBER, .value = c};
	if ((c & 0xf0) == 0x80)
		return (struct msgpack_head){.type = MSGPACK_MAP, .value = c & 15u};
	if ((c & 0xf0) == 0x90)
		return (struct msgpack_head){.type = MSGPACK_ARRAY, .value = c & 15u};
	if ((c & 0xe0) == 0xa0) {
		bytes = c & 31u;
		string = true;
	} else if (c >= 0xe0 || c == 0xc0 || c == 0xc2 || c == 0xc3) {
		return (struct msgpack_head){.type = MSGPACK_OTHER};
	} else if (c >= 0xcc && c <= 0xcf) {
		return (struct msgpack_head){.type = MSGPACK_NUMBER, .value = msgpack_take(m, (size_t)1 << (c - 0xcc))};
	} else if (c >= 0xd9 && c <= 0xdb) {
		bytes = msgpack_take(m, (size_t)1 << (c - 0xd9));
		string = true;
	} else if (c == 0xdc || c == 0xdd) {
		return (struct msgpack_head){.type = MSGPACK_ARRAY, .value = msgpack_take(m, c == 0xdc ? 2 : 4)};
	} else if (c == 0xde || c == 0xdf) {
		return (struct msgpack_head){.type = MSGPACK_MAP, .value = msgpack_take(m, c == 0xde ? 2 : 4)};
	} else if (c >= 0xd0 && c <= 0xd3) {
		bytes = (size_t)1 << (c - 0xd0); // Signed integers.
	} else if (c == 0xca || c == 0xcb) {
		bytes = c == 0xca ? 4 : 8;
	} else if (c >= 0xc4 && c <= 0xc6) {
		bytes = msgpack_take(m, (size_t)1 << (c - 0xc4)); // Binary.
	} else {
		m->at = nullptr; // Extension types.
		return (struct msgpack_head){.type = MSGPACK_OTHER};
	}
	if (!m->at || (size_t)(m->end - m->at) < bytes) {
		m->at = nullptr;
		return (struct msgpack_head){.type = MSGPACK_OTHER};
	}
	char const *const text = (char const *)m->at;
	m->at += bytes;
	if (!string)
		return (struct msgpack_head){.type = MSGPACK_OTHER};
	return (struct msgpack_head){.type = MSGPACK_STRING, .text = text, .length = bytes};
}

/** @brief Skips the items of the container whose head was read.
 *
 * @param m    The reader.
 * @param head The head.
 */
static void
msgpack_skip (struct msgpack            *m,
              struct msgpack_head const *head)
{
	uint64_t const items = head->type == MSGPACK_MAP ? 2 * head->value
	                     : head->type == MSGPACK_ARRAY ? head->value
	                     : 0;
	for (uint64_t i = 0; i < items && msgpack_ok(m); ++i) {
		struct msgpack_head const item = msgpack_head(m);
		msgpack_skip(m, &item);
	}
}

/** @brief A kernel argument as the metadata describes it: its kind and bytes. */
struct abi_arg {
	char const *kind;   //!< Its value kind, not null-terminated, in the module's bytes.
	size_t      length; //!< The kind's length.
	uint64_t    size;   //!< Its bytes.
};

/** @brief A kernel as the metadata describes it: its explicit arguments and its largest group. */
struct kernel_abi {
	char const     *name;      //!< Its name, not null-terminated, in the module's bytes.
	struct abi_arg *args;      //!< Its explicit arguments.
	size_t          length;    //!< The name's length.
	size_t          count;     //!< The number of arguments.
	size_t          capacity;  //!< The arguments that their memory holds.
	uint64_t        max_group; //!< The most threads of a group.
};

/** @brief The kernels of a module, and its bytes, which their names point into. */
struct module_abi {
	struct files_data  image;    //!< The module's bytes.
	struct kernel_abi *kernels;  //!< Its kernels.
	size_t             count;    //!< Their number.
	size_t             capacity; //!< The kernels that their memory holds.
};

/** @brief Frees a module's kernels and bytes.
 *
 * @param m The module.
 */
static void
module_abi_fini (struct module_abi *m)
{
	for (size_t k = 0; k < m->count; ++k) {
		free(m->kernels[k].args);
		m->kernels[k].args = nullptr;
	}
	free(m->kernels);
	m->kernels = nullptr;
	files_data_fini(&m->image);
	m->count = 0;
	m->capacity = 0;
}

/** @brief A kernel of a module by its name, or nullptr. */
static struct kernel_abi const *
module_abi_find (struct module_abi const *m,
                 char const              *name)
{
	size_t const length = strlen(name);
	for (size_t k = 0; k < m->count; ++k)
		if (m->kernels[k].length == length && !memcmp(m->kernels[k].name, name, length))
			return &m->kernels[k];
	return nullptr;
}

/** @brief Adds an explicit argument to a kernel. */
static void
kernel_abi_add (struct kernel_abi    *k,
                struct abi_arg const *a)
{
	if (k->count == k->capacity) {
		size_t const capacity = k->capacity ? 2 * k->capacity : 16;
		struct abi_arg *const args = realloc(k->args, capacity * sizeof *args);
		allocated(args);
		k->args = args;
		k->capacity = capacity;
	}
	k->args[k->count++] = *a;
}

/** @brief Adds a kernel to a module unless one of its name is there, whose arguments it then
 *         frees. */
static void
module_abi_add (struct module_abi *m,
                struct kernel_abi *k)
{
	for (size_t i = 0; i < m->count; ++i) {
		if (m->kernels[i].length == k->length && !memcmp(m->kernels[i].name, k->name, k->length)) {
			free(k->args);
			k->args = nullptr;
			return;
		}
	}
	if (m->count == m->capacity) {
		size_t const capacity = m->capacity ? 2 * m->capacity : 64;
		struct kernel_abi *const kernels = realloc(m->kernels, capacity * sizeof *kernels);
		allocated(kernels);
		m->kernels = kernels;
		m->capacity = capacity;
	}
	m->kernels[m->count++] = *k;
	*k = (struct kernel_abi){0};
}

/** @brief Whether a head is a string of a text.
 *
 * @param h      The head.
 * @param text   The text.
 * @param length Its length.
 * @return       true if it is.
 */
static bool
msgpack_equals (struct msgpack_head const *h,
                char const                *text,
                size_t                     length)
{
	return h->type == MSGPACK_STRING && h->length == length && !memcmp(h->text, text, length);
}

/** @brief Whether a head is a string of a literal's text. */
#define MSGPACK_IS(h, literal) msgpack_equals((h), (literal), sizeof (literal) - 1)

/** @brief Reads one kernel's map from the metadata.
 *
 * @param r The reader, at the kernel's map.
 * @param m Receives the kernel.
 */
static void
read_kernel (struct msgpack    *r,
             struct module_abi *m)
{
	struct msgpack_head const fields = msgpack_head(r);
	struct kernel_abi kernel = {0};
	for (uint64_t f = 0; f < (fields.type == MSGPACK_MAP ? fields.value : 0) && msgpack_ok(r); ++f) {
		struct msgpack_head const field = msgpack_head(r);
		struct msgpack_head const value = msgpack_head(r);
		if (MSGPACK_IS(&field, ".name")) {
			kernel.name = value.text;
			kernel.length = value.length;
		}
		if (MSGPACK_IS(&field, ".max_flat_workgroup_size"))
			kernel.max_group = value.value;
		if (!MSGPACK_IS(&field, ".args") || value.type != MSGPACK_ARRAY) {
			msgpack_skip(r, &value);
			continue;
		}
		for (uint64_t a = 0; a < value.value && msgpack_ok(r); ++a) {
			struct msgpack_head const arg = msgpack_head(r);
			struct abi_arg described = {.kind = "", .length = 0};
			for (uint64_t p = 0; p < (arg.type == MSGPACK_MAP ? arg.value : 0) && msgpack_ok(r); ++p) {
				struct msgpack_head const property = msgpack_head(r);
				struct msgpack_head const setting = msgpack_head(r);
				if (MSGPACK_IS(&property, ".value_kind")) {
					described.kind = setting.text ? setting.text : "";
					described.length = setting.length;
				}
				if (MSGPACK_IS(&property, ".size"))
					described.size = setting.value;
				msgpack_skip(r, &setting);
			}
			if (described.length < 7 || memcmp(described.kind, "hidden_", 7))
				kernel_abi_add(&kernel, &described);
		}
	}
	if (msgpack_ok(r)) {
		if (!kernel.name)
			kernel.name = "";
		module_abi_add(m, &kernel);
	} else {
		free(kernel.args);
		kernel.args = nullptr;
	}
}

/** @brief Reads the kernels that the NT_AMDGPU_METADATA note of a code object describes; none
 *         when it has none.
 *
 * @param m The module, whose image is read.
 */
static void
read_kernels (struct module_abi *m)
{
	uint8_t const *const image = m->image.bytes;
	size_t const size = m->image.size;
	Elf64_Ehdr eh;
	if (size < sizeof eh)
		return;
	memcpy(&eh, image, sizeof eh);
	if (memcmp(eh.e_ident, ELFMAG, SELFMAG) || eh.e_ident[EI_CLASS] != ELFCLASS64)
		return;
	for (unsigned s = 0; s < eh.e_shnum; ++s) {
		Elf64_Shdr sh;
		size_t const at = eh.e_shoff + (size_t)s * eh.e_shentsize;
		if (at + sizeof sh > size)
			break;
		memcpy(&sh, image + at, sizeof sh);
		if (sh.sh_type != SHT_NOTE || sh.sh_offset + sh.sh_size > size)
			continue;
		uint8_t const *const notes = image + sh.sh_offset;
		for (size_t note = 0; note + 12 <= sh.sh_size;) {
			uint32_t word[3]; // Name bytes, description bytes, type.
			memcpy(word, notes + note, 12);
			size_t const name = note + 12, description = name + ((word[0] + 3) & ~3u);
			note = description + ((word[1] + 3) & ~3u);
			if (note > sh.sh_size)
				break;
			if (word[2] != 32 || word[0] != 7 || memcmp(notes + name, "AMDGPU", 7))
				continue;
			struct msgpack r = {notes + description, notes + description + word[1]};
			struct msgpack_head const root = msgpack_head(&r);
			for (uint64_t i = 0; i < (root.type == MSGPACK_MAP ? root.value : 0) && msgpack_ok(&r); ++i) {
				struct msgpack_head const key = msgpack_head(&r);
				struct msgpack_head const list = msgpack_head(&r);
				if (!MSGPACK_IS(&key, "amdhsa.kernels") || list.type != MSGPACK_ARRAY) {
					msgpack_skip(&r, &list);
					continue;
				}
				for (uint64_t k = 0; k < list.value && msgpack_ok(&r); ++k)
					read_kernel(&r, m);
			}
		}
	}
}

#undef MSGPACK_IS

/** @brief Whether an argument's metadata is a kind of a size.
 *
 * @param a      The argument.
 * @param kind   The kind.
 * @param length Its length.
 * @param size   The size.
 * @return       true if it is.
 */
static bool
abi_arg_equals (struct abi_arg const *a,
                char const           *kind,
                size_t                length,
                uint64_t              size)
{
	return a->length == length && !memcmp(a->kind, kind, length) && a->size == size;
}

/** @brief Whether an argument's metadata is the kind of a literal and a size. */
#define ABI_ARG_IS(a, literal, size) abi_arg_equals((a), (literal), sizeof (literal) - 1, (size))

/** @brief Checks every planned launch against its kernel's metadata in the modules in a directory:
 *         its arguments, pointers or 4-byte values, and its group size.
 *
 * @param directory The directory.
 * @return          0 if every launch fits, 77 when a module is missing, else 1.
 */
static int
check_abi (char const *directory)
{
	struct module_abi modules[HIP_MODULE_COUNT] = {0};
	size_t const directory_length = strlen(directory);
	int status = 0;
	for (size_t m = 0; m < HIP_MODULE_COUNT && !status; ++m) {
		char *path = files_join(directory, directory_length, HIP_PLAN_MODULE_FILES[m],
		                        strlen(HIP_PLAN_MODULE_FILES[m]), nullptr);
		allocated(path);
		struct error e;
		if (!files_is_regular_file(path)) {
			printf("hip-plan test: skipped: no module %s\n", path);
			status = 77;
		} else if (!expect(!files_read(&modules[m].image, path, &e), "%s", e.what)) {
			status = 1;
		} else {
			read_kernels(&modules[m]);
			expect(modules[m].count, "%s: no kernel metadata", path);
		}
		free(path);
		path = nullptr;
	}
	if (status) {
		for (size_t m = 0; m < HIP_MODULE_COUNT; ++m)
			module_abi_fini(&modules[m]);
		return status;
	}

	// Each kernel's metadata, or nullptr.
	struct kernel_abi const *abis[HIP_KERNEL_COUNT];
	for (uint8_t k = 0; k < HIP_KERNEL_COUNT; ++k) {
		struct hip_kernel_info const *const i = &HIP_PLAN_KERNELS[k];
		abis[k] = module_abi_find(&modules[i->module], i->name);
		expect(abis[k], "%s has no kernel %s", HIP_PLAN_MODULE_FILES[i->module], i->name);
	}
	bool reported[HIP_KERNEL_COUNT] = {0};
	for (size_t t = 0; t < kNativeTierCount; ++t) {
		struct NativeTier const *const tier = &kNativeTiers[t];
		for (uint8_t performance = 0; performance < 2; ++performance) {
			struct error e;
			struct hip_plan plan;
			if (!expect(!hip_plan_init(&plan, tier->width, tier->networkHeight, performance, &e),
			            "tier %u: %s", tier->height, e.what))
				continue;
			for (size_t n = 0; n < plan.launch_count; ++n) {
				struct hip_launch const *const l = &plan.launches[n];
				struct hip_kernel_info const *const i = &HIP_PLAN_KERNELS[l->kernel];
				struct kernel_abi const *const abi = abis[l->kernel];
				if (!abi || reported[l->kernel])
					continue;
				bool same = abi->count == l->count;
				for (uint16_t a = 0; same && a < l->count; ++a)
					same = pointer(l->args[a]) ? ABI_ARG_IS(&abi->args[a], "global_buffer", 8)
					                           : ABI_ARG_IS(&abi->args[a], "by_value", 4);
				if (!expect(same, "%s takes other arguments than the plan passes", i->name) ||
				    !expect(i->threads <= abi->max_group, "%s: groups of %u threads, at most %" PRIu64,
				            i->name, (unsigned)i->threads, abi->max_group))
					reported[l->kernel] = true;
			}
			hip_plan_fini(&plan);
		}
	}
	for (size_t m = 0; m < HIP_MODULE_COUNT; ++m)
		module_abi_fini(&modules[m]);
	if (!failures)
		puts("hip-plan test: every launch fits its kernel");
	return failures ? 1 : 0;
}

#undef ABI_ARG_IS

/** @brief Prints the canonical launch list the checks hash.
 *
 * @param height      The tier's height, as given.
 * @param performance Whether with --performance.
 * @param first       Whether with the first frame's buffers.
 * @param history     Whether the frame has a history.
 * @return            0, 1 for a plan that fails, or 2 for no such tier.
 */
static int
print (char const *height,
       bool        performance,
       bool        first,
       bool        history)
{
	struct NativeTier const *tier = nullptr;
	for (size_t t = 0; t < kNativeTierCount; ++t) {
		char decimal[16];
		int const length = snprintf(decimal, sizeof decimal, "%" PRIu32, kNativeTiers[t].height);
		if (length > 0 && length < (int)sizeof decimal && !strcmp(decimal, height))
			tier = &kNativeTiers[t];
	}
	if (!tier) {
		fprintf(stderr, "hip-plan-test: no tier %s\n", height);
		return 2;
	}
	struct error e;
	struct hip_plan plan;
	if (hip_plan_init(&plan, tier->width, tier->networkHeight, performance, &e)) {
		fprintf(stderr, "hip-plan-test: %s\n", e.what);
		return 1;
	}
	struct hip_placement placement;
	if (hip_plan_place(&placement, &plan, &e)) {
		fprintf(stderr, "hip-plan-test: %s\n", e.what);
		hip_plan_fini(&plan);
		return 1;
	}
	struct support_text text = {0};
	canonical_text(&plan, first ? placement.first : placement.later, history, &text);
	fputs(support_text_string(&text), stdout);
	support_text_fini(&text);
	hip_placement_fini(&placement);
	hip_plan_fini(&plan);
	return 0;
}

/** @brief The usage that --help prints. */
static char const USAGE[] =
	"Usage: hip-plan-test [OPTION]...\n"
	"Checks the HIP network's launch plan against traces of upstream's network. No GPU or model\n"
	"needed.\n"
	" -m, --modules DIR  Instead, check each planned launch's arguments and group size against\n"
	"                    the kernels in the HIP modules in DIR; exit 77 when one is missing\n"
	"                    (default: unset)\n"
	" -P, --print TIER   Instead, print the launch list at TIER (720, 900 or 1080) as the checks\n"
	"                    hash it, buffers named by first use (default: unset)\n"
	" -p, --performance  With --print, leave out the blocks that dlsslopd --performance skips\n"
	"                    (default: off)\n"
	" -f, --first        With --print, the first frame's buffers, not a later frame's\n"
	"                    (default: off)\n"
	" -H, --history      With --print, a frame with a history (default: off)\n"
	" -h, --help         Show help (default: off)\n";

int
main (int    argc,
      char **argv)
{
	char const *modules = "";
	char const *tier = "";
	bool performance = false;
	bool first = false;
	bool history = false;
	static struct option const options[] = {
		{"modules",     required_argument, nullptr, 'm'},
		{"print",       required_argument, nullptr, 'P'},
		{"performance", no_argument,       nullptr, 'p'},
		{"first",       no_argument,       nullptr, 'f'},
		{"history",     no_argument,       nullptr, 'H'},
		{"help",        no_argument,       nullptr, 'h'},
		{},
	};
	for (int code; (code = getopt_long(argc, argv, "+m:P:pfHh", options, nullptr)) != -1;) {
		switch (code) {
		case 'm':
			modules = optarg;
			break;
		case 'P':
			tier = optarg;
			break;
		case 'p':
			performance = true;
			break;
		case 'f':
			first = true;
			break;
		case 'H':
			history = true;
			break;
		case 'h':
			fputs(USAGE, stdout);
			return 0;
		default:
			return 2;
		}
	}
	if (optind != argc || (*tier && *modules))
		return 2;
	if (*tier)
		return print(tier, performance, first, history);
	if (*modules)
		return check_abi(modules);
	check_plans();
	if (!failures)
		puts("hip-plan test: every check passed");
	return failures ? 1 : 0;
}
