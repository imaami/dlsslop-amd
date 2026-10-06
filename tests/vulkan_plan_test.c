/** @file
 *
 * Host test of the Vulkan network's plan, without a GPU or a model. At 40 frame extents from 17x17
 * to 5120x2880, the plan's steps with their push constants, as planned and with the kernels without
 * the exponent's upper clamp that the real model's weights allow, the tables of its weight blob with
 * the heads that model frees of that clamp, and its activation arena's values and sizes are checked
 * against goldens of upstream's own NrSession::build (DLSSNR-AMD's nr_graph.cpp at 82560c4, built
 * with its rdna4.sh defines and stopped before the device), and the weight blob packed from a
 * synthetic model pack against upstream's. Where upstream's tile-counter records that wait each name
 * an error word of their own and those that only signal none, the plan's all name one they share,
 * and the persistent runs that upstream gives no record share one more, after the others: the
 * goldens are checked with upstream's words put back and that record taken out. The frames the
 * network cannot take are rejected.
 */
// SPDX-License-Identifier: MIT
#include <getopt.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "files.h"
#include "support.h"
#include "vulkan_pack.h"
#include "vulkan_plan.h"
#include "vulkan_weights.h"

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
	fputs("vulkan-plan test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	++failures;
	return false;
}

/** @brief Ends the test without memory for a value.
 *
 * @param p The value's memory, or nullptr.
 * @return  @a p.
 */
static void *
allocated (void *p)
{
	if (!p) {
		fputs("vulkan-plan test: out of memory\n", stderr);
		exit(1);
	}
	return p;
}

/** @brief Upstream's plan of WIDTH x HEIGHT frames.
 *
 * Its working extent, dispatches, activation arena and values' end, weight blob, tile-counter
 * words, persistent runs and dispatches ordered by tile counters; the FNV-1a 64 of the tables in its
 * weight blob for the real model, whose persistent runs' records carry the heads that model frees of
 * the exponent's upper clamp, real_tables; of its dispatches, "kernel gx gy gz chained
 * push-words...\n" each, as it planned them for the synthetic pack, which frees no head of that
 * clamp, and for the real model, clamped; of its values, "key bBLOCK/SLOT off OFFSET size BYTES
 * overread BYTES\n" each; and of the weight blob it packed from the synthetic pack and from the real
 * model. Recorded from upstream's build by a probe that dumps them where the build meets the device,
 * the tables from the weight blob where the plan puts them.
 */
struct golden {
	uint64_t arena;          //!< The activation arena's bytes.
	uint64_t values_end;     //!< The values' end.
	uint64_t real_tables;    //!< The tables' hash for the real model.
	uint64_t dispatches;     //!< The dispatches' hash for the synthetic pack.
	uint64_t clamped;        //!< The dispatches' hash for the real model.
	uint64_t values;         //!< The values' hash.
	uint64_t synthetic_blob; //!< The blob's hash from the synthetic pack.
	uint64_t real_blob;      //!< The blob's hash from the real model.
	uint32_t width;          //!< The frames' width.
	uint32_t height;         //!< Their height.
	uint32_t work_width;     //!< The working extent's width.
	uint32_t work_height;    //!< Its height.
	uint32_t steps;          //!< The dispatches.
	uint32_t blob;           //!< The weight blob's bytes.
	uint32_t counters;       //!< The tile-counter words.
	uint32_t runs;           //!< The persistent runs.
	uint32_t chained;        //!< The dispatches ordered by tile counters.
};

/** @brief A golden in the order of upstream's dumps. */
#define GOLDEN(width_, height_, work_width_, work_height_, steps_, arena_, values_end_, blob_, counters_, \
               runs_, chained_, real_tables_, dispatches_, clamped_, values_, synthetic_blob_, real_blob_) \
	{.arena = arena_, .values_end = values_end_, .real_tables = UINT64_C(real_tables_), \
	 .dispatches = UINT64_C(dispatches_), .clamped = UINT64_C(clamped_), .values = UINT64_C(values_), \
	 .synthetic_blob = UINT64_C(synthetic_blob_), .real_blob = UINT64_C(real_blob_), .width = width_, \
	 .height = height_, .work_width = work_width_, .work_height = work_height_, .steps = steps_, \
	 .blob = blob_, .counters = counters_, .runs = runs_, .chained = chained_}

/** @brief Upstream's plans at the golden extents. */
static struct golden const GOLDENS[] = {
	GOLDEN(17, 17, 328, 320, 124, 27948288, 26550272, 155469644, 344224, 6, 99,
	       0x4cf188352abd04ad, 0xac36df2d91920ebf, 0x3a78a163f2c62f99,
	       0x7c39a7c135a6f2ab, 0xafae60b012a87738, 0xbbb5ac1f3aee438e),
	GOLDEN(1, 400, 320, 448, 123, 36382976, 34963456, 155804972, 348335, 6, 100,
	       0xaebd9fc79b1f33c8, 0xf6d036c2e947b105, 0x0976322fb7cd4e7f,
	       0x2cf2a0137495ca85, 0xb446e89512b6f371, 0x30ebc0897abc17d7),
	GOLDEN(400, 1, 448, 320, 123, 36382976, 34963456, 155804972, 348337, 6, 100,
	       0xbb144751d798c970, 0x7f533198b8b8012b, 0x06f07580efaf4d39,
	       0x2cf2a0137495ca85, 0xfb702c0473262c71, 0xaa0d893a65db36cb),
	GOLDEN(32, 32, 328, 320, 124, 27948288, 26550272, 155469644, 344224, 6, 99,
	       0x4cf188352abd04ad, 0x869b9f39fefc592f, 0x1fb32f4ab8a5fc9d,
	       0x7c39a7c135a6f2ab, 0xafae60b012a87738, 0xbbb5ac1f3aee438e),
	GOLDEN(64, 64, 336, 320, 123, 28210432, 26796032, 155506540, 348321, 6, 100,
	       0x86b7b010ceb4ca38, 0x2026062dedaeeef7, 0xb087316315fb19f1,
	       0xb494e00c5a47741f, 0x6e830263ad05ae39, 0x93fd711bff4bb9b3),
	GOLDEN(160, 120, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0x1856adec0aa2f035, 0xd3638c3ecc405a9b,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(200, 100, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0x09d80048b4ed851a, 0xce7b64232426ada0,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(256, 144, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0xca589d388cdd3cd9, 0x9d98e096eb2ed17b,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(128, 320, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0x947919b23ced93ec, 0x2e152e7050faa166,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(320, 128, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0x864937e722be9f78, 0xdc2315105520ab1a,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(320, 129, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0xaee336ab0f2c3d81, 0xa8a8238280b6b26b,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(320, 180, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
	       0x7500245e6afda6b6, 0x4c90c77f72d49464, 0xbf79b370fb94a3a2,
	       0xc331ba7fb0e0bb1f, 0x8e2cf06f232d449f, 0xdf6111acc943189d),
	GOLDEN(333, 333, 384, 384, 123, 36973824, 35553280, 155839068, 348321, 6, 100,
	       0xfb828ac519522972, 0x2139df0bb258c75b, 0xcb6ffcb66120cde5,
	       0x6b33d951c0bcfbc7, 0xdfd3e82f50838b5b, 0x315a6453da11cb99),
	GOLDEN(640, 360, 640, 384, 123, 60683520, 59244544, 156659772, 348351, 6, 100,
	       0xe1a9e58807e3095f, 0x5911d840006ddb38, 0xf9501a487a69041e,
	       0xd08ba5e9c77522d6, 0x43997cb9a9682362, 0x298e1e72ff78de6c),
	GOLDEN(640, 480, 640, 512, 123, 80357888, 78905344, 157343020, 348372, 6, 100,
	       0xae97cdf54e807756, 0xc97f2bd563a789ea, 0x7ca3cbff0a3c61b8,
	       0x4d6a79d85edfccd5, 0x05327c7118b5518f, 0x4b45520c4aa31d35),
	GOLDEN(800, 450, 832, 512, 123, 104950272, 103481344, 158163244, 348409, 6, 100,
	       0x74409f22d32a02bd, 0x687acb224f8f437e, 0xec870c63f184a2fc,
	       0xc573bd1a9fee5da2, 0x93d20ac39cd2c804, 0xbdd9ca94170f3e42),
	GOLDEN(854, 480, 896, 512, 123, 111902720, 110428160, 158436620, 348409, 6, 100,
	       0xaaac75596ebd079f, 0x3be0b6cca435e2bc, 0x06945617d3795a82,
	       0xa0b017d2a04de8a3, 0x3e3fa671a849925e, 0x44484c86cfacc6e4),
	GOLDEN(960, 540, 960, 576, 123, 136035584, 134545408, 159222220, 348465, 6, 100,
	       0x381099b6f430eca2, 0x968fbf57aa9ac7fb, 0xe7c70ba576a69f85,
	       0x3edaf20da30af3b0, 0x4b175d260ff59697, 0x26e8c39324741db5),
	GOLDEN(1001, 563, 1024, 576, 123, 144889088, 143392768, 159529740, 348465, 6, 100,
	       0xc9fb9da2297603df, 0x3664069be7665671, 0xac8cba61e19bd97f,
	       0x66e151ad666eb8d5, 0x5a4bdf60e83f5b62, 0xe539c275017d7244),
	GOLDEN(1152, 648, 1152, 704, 123, 198630400, 197099520, 161373804, 348521, 6, 100,
	       0xd48be3b1337fc1a5, 0xd32fb634cb07f8b9, 0x587e28820c33604b,
	       0x01f1ddd49336e51f, 0xebc1477459610c94, 0xd95616a70ea1e422),
	GOLDEN(1024, 768, 1088, 768, 123, 204532224, 202997760, 161578476, 348521, 6, 100,
	       0xa09070219a74e9ff, 0x3c4e15c670d32b28, 0x499020759066e94e,
	       0x4333b55449e60434, 0xb26d30d01531ebce, 0x1569dd392fa589f4),
	GOLDEN(1152, 720, 1152, 768, 123, 214571008, 213024768, 161988300, 348521, 6, 100,
	       0x25bf5c2564bd5183, 0x0b08f1d9ac8bac80, 0x68fcc2b2e4d723b6,
	       0x05f9d9530045eec9, 0x35e94f7226ce9a2e, 0x0fb2e499524fcdac),
	GOLDEN(1280, 720, 1344, 768, 123, 252245248, 250675200, 163217868, 348573, 6, 100,
	       0x2ed8dd694aa7d4ae, 0x9f08890b04ad1d28, 0x9f63e7a430a1cbfe,
	       0x4006e4a89e059758, 0xfa50bda18b15f553, 0x463b2d1602eaa82d),
	GOLDEN(1280, 800, 1280, 832, 123, 200674560, 199098368, 163491788, 348849, 6, 100,
	       0xe3c3acf2583929b5, 0xc6cb9729a3d868c0, 0xe55fefe1b31f61ae,
	       0xddaea6e67fe051df, 0x529ef4e186561230, 0x4e3e472b470d7ab6),
	GOLDEN(1366, 768, 1408, 768, 123, 201332992, 199753728, 163629740, 349085, 6, 100,
	       0x6b9f956c87d2da32, 0x3320b417e8281e10, 0xe1f57e1794a6320e,
	       0x3caec83201885480, 0x6560b642c8088fb7, 0x1b13c2fd55e80371),
	GOLDEN(1280, 1024, 1344, 1024, 123, 258743296, 257097728, 166105772, 353783, 6, 100,
	       0x6ff175906a28d441, 0xd10dc6d667efddb3, 0x6bf66b0c97953fb9,
	       0x94ca7bf838c1f8f0, 0xea3faba919430a54, 0x9ac8a68ab71e168e),
	GOLDEN(1600, 900, 1600, 960, 123, 288876032, 287211520, 167447628, 356346, 6, 100,
	       0x235f08b72d1774ff, 0x209fc2229e3a69b8, 0xe938de463e9396f6,
	       0xfc5a22ba254a7b9b, 0x2007c5b57979fe26, 0x6ce309612a406d3c),
	GOLDEN(1920, 804, 1920, 832, 123, 300358656, 298680320, 167964652, 357317, 6, 100,
	       0xbe56478f8dad1fc7, 0x8f28992475558749, 0x6f0fa300d13b22ff,
	       0xba97e75454f5b261, 0x89b6a04213ba1412, 0xa96bde52c687ef1c),
	GOLDEN(1440, 1080, 1472, 1088, 123, 301178112, 299499520, 167997516, 357391, 6, 100,
	       0x8201d2ded474bb50, 0x83efdcf44565ddcd, 0xf5fd245fdca6cc9b,
	       0xda3fc3535c060b8d, 0x3c3fc2c27e09b0e9, 0x1664670594cbdd33),
	GOLDEN(1707, 960, 1728, 960, 123, 309939456, 308248576, 168479676, 358302, 6, 100,
	       0x7fb70444a0678865, 0x7fab20085dd5bf38, 0xe7e9f56be5e96996,
	       0xda824cd32854e759, 0xf7dc249729243db0, 0x1b8bfd181d9f661e),
	GOLDEN(1680, 1050, 1728, 1088, 123, 353337856, 351600640, 170336556, 361818, 6, 100,
	       0x081527c00f8549b3, 0xcc68a1a4db8dec2d, 0x282116bef025e163,
	       0x92dae6258ab9771f, 0x1b25e539df770822, 0x1f05bfe25e6d908c),
	GOLDEN(1600, 1200, 1600, 1216, 123, 365541376, 363790336, 170886444, 362851, 6, 100,
	       0x860450d1f4de9a5c, 0x4b45cb8be1de8af1, 0xb0006185cf85f66f,
	       0x21ea8125c575fb4d, 0xc9d330536d56ab5d, 0x1af44bb847291f5f),
	GOLDEN(1919, 1079, 1920, 1088, 123, 392277248, 390496256, 172090828, 365117, 6, 100,
	       0xb25a3b2c89d5c003, 0x9b7c8db6572477fc, 0xc127cf793d036e32,
	       0xf161d160a6330aed, 0x84a31840953757ee, 0x73ff13faea0fa0f4),
	GOLDEN(1080, 1920, 1088, 1920, 123, 392277248, 390496256, 172090828, 365117, 6, 100,
	       0x9948bd38ebefab8b, 0xc66c181af4f3376d, 0x2dbb7da0fe59d51b,
	       0xf161d160a6330aed, 0xcca8a797e0723d46, 0x28a6a626e3cfad70),
	GOLDEN(1920, 1080, 1920, 1088, 123, 392277248, 390496256, 172090828, 365117, 6, 100,
	       0xb25a3b2c89d5c003, 0x0e77368975e0d934, 0x1734ee75a89782f2,
	       0xf161d160a6330aed, 0x84a31840953757ee, 0x73ff13faea0fa0f4),
	GOLDEN(1920, 1200, 1920, 1216, 123, 438040064, 436207616, 174153932, 369013, 6, 100,
	       0x52278671bb7a4891, 0x49cae4fbd16848b7, 0xfc6b1d1b266f02e5,
	       0xcb29e66fcd7cdb28, 0x04d64b1ac9150154, 0xe58f55f4db4ae356),
	GOLDEN(2560, 1080, 2560, 1088, 123, 522447872, 520519680, 177938444, 376187, 6, 100,
	       0x56c9206effdf7962, 0x6cccfc3eda776b97, 0xa77a84f86f11fe39,
	       0xcdc4561783fb376d, 0x2d24bc2b77d6b29f, 0x6ce6c3315f303b41),
	GOLDEN(2560, 1440, 2560, 1472, 123, 701895168, 699760640, 186189772, 391779, 6, 100,
	       0x7d0aa9148a223f9b, 0xcc305b8edbe08d17, 0x6a33a69889c389ad,
	       0x9606066a7e165f97, 0xe948c87a2abf746e, 0xb00097ff073a25ec),
	GOLDEN(3840, 2160, 3840, 2176, 123, 1545993728, 1543208960, 224419820, 395122, 6, 96,
	       0xd8773932e3b63714, 0x81270d9ddc925aa3, 0x734d768a9bc76ce9,
	       0x4b7752b8d303fce8, 0x5649c365ee6a9e89, 0x20fcabf9a0820abb),
	GOLDEN(5120, 2880, 5120, 2880, 123, 2757088000, 2753167360, 277874828, 452881, 6, 96,
	       0x26cd30bad5bf2c28, 0xffd956f0618d26c6, 0x7d350f0980f055b8,
	       0x25c685b892ab46dc, 0x1d57a895eb3bb0fd, 0x13013f43e908e3ff),
};

#undef GOLDEN

/** @brief The number of goldens. */
#define GOLDEN_COUNT (sizeof GOLDENS / sizeof *GOLDENS)

/** @brief The extents at which the synthetic model is packed: the smallest planned, which views the
 *         post block's input, small ones and odd ones, dlsslopd's 720p tier, a portrait one and 4K,
 *         where fewer steps are chained. */
static struct {
	uint32_t width;
	uint32_t height;
} const PACKED[] = {
	{17, 17}, {64, 64}, {1, 400}, {1152, 720}, {1280, 720}, {1707, 960}, {1080, 1920}, {3840, 2160},
};

/** @brief The golden of an extent, or nullptr. */
static struct golden const *
golden (uint32_t width,
        uint32_t height)
{
	for (size_t i = 0; i < GOLDEN_COUNT; ++i)
		if (GOLDENS[i].width == width && GOLDENS[i].height == height)
			return &GOLDENS[i];
	return nullptr;
}

/** @brief The Swin heads of the real model, the pack that linux/package/model-tools extracts from
 *         nvngx_dlssnr 310.8.0, that its weights free of the exponent's upper clamp, by block
 *         (upstream's audit in its build, g_nohi_heads). */
static struct vulkan_clamp_free const MODEL_CLAMP_FREE = {{
	0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x03, 0x03, 0x09, 0x06, 0x0a,
	0x01, 0x0f, 0x08, 0xf7, 0xa1, 0x6a, 0x21, 0x51, 0xb6, 0x3e, 0x01, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x08, 0x28, 0x01, 0x10, 0x10, 0x11, 0xa9, 0xf2, 0x04, 0x01, 0x00, 0x08,
	0x03, 0x09, 0x03, 0x00, 0x02, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
}};

/** @brief A kernel's SPIR-V stem. */
static char const *
stem (enum vulkan_kernel k)
{
	return VULKAN_PLAN_KERNELS[k].stem;
}

/** @brief The kernel whose twin without the upper clamp a kernel is, or the kernel. */
static enum vulkan_kernel
clamped (enum vulkan_kernel k)
{
	for (size_t t = 0; t < VULKAN_PLAN_UNCLAMPED_COUNT; ++t)
		if (VULKAN_PLAN_UNCLAMPED[t].twin == k)
			return VULKAN_PLAN_UNCLAMPED[t].kernel;
	return k;
}

/** @brief A copy of memory, which the caller frees. */
static void *
copy_of (void const *data,
         size_t      size)
{
	void *const copy = allocated(malloc(size ? size : 1));
	if (size)
		memcpy(copy, data, size);
	return copy;
}

/** @brief A copy of a plan, which vulkan_plan_fini() frees. */
static struct vulkan_plan
plan_copy (struct vulkan_plan const *p)
{
	struct vulkan_plan c = *p;
	c.steps = copy_of(p->steps, p->step_count * sizeof *p->steps);
	c.push = copy_of(p->push, p->push_count * sizeof *p->push);
	c.segments = copy_of(p->segments, p->segment_count * sizeof *p->segments);
	c.tables = copy_of(p->tables, p->table_count * sizeof *p->tables);
	c.values = copy_of(p->values, p->value_count * sizeof *p->values);
	c.timeouts = copy_of(p->timeouts, p->timeout_count * sizeof *p->timeouts);
	return c;
}

/** @brief A copy of a plan with a model's kernels without the upper clamp. */
static struct vulkan_plan
unclamped (struct vulkan_plan const       *p,
           struct vulkan_clamp_free const *heads)
{
	struct vulkan_plan c = plan_copy(p);
	vulkan_plan_unclamp(&c, heads);
	return c;
}

/** @brief A tile-counter record (vulkan_schedule.c's chain), of 7 words: the counters its step
 *         waits on, the units it needs of each (the frame counter's tick), its table, the counters
 *         it signals, the frame counter, the error word its waits set when they give up, and the
 *         magic word. */
enum record : size_t {
	RECORD_WAITS,
	RECORD_UNITS,
	RECORD_TABLE,
	RECORD_SIGNALS,
	RECORD_FRAME,
	RECORD_ERROR,
	RECORD_MAGIC,
	RECORD_WORDS,
};

/** @brief Where in a plan's tables a step's record starts, or SIZE_MAX: the table of a record's
 *         words at the last push word of a step.
 *
 * @param p The plan.
 * @param s The step.
 * @return  The record's first word in the tables.
 */
static size_t
record (struct vulkan_plan const *p,
        struct vulkan_step const *s)
{
	uint32_t const word = p->push[s->push + s->words - 1];
	if (word == UINT32_MAX)
		return SIZE_MAX;
	for (size_t i = 0; i < p->segment_count; ++i) {
		struct vulkan_segment const *const g = &p->segments[i];
		if (g->recipe == VULKAN_RECIPE_TABLE && g->offset == UINT64_C(4) * word &&
		    g->bytes == 4 * RECORD_WORDS)
			return p->tables[g->index + RECORD_MAGIC] == 0x54434852u ? g->index : SIZE_MAX;
	}
	return SIZE_MAX;
}

/** @brief Where in a plan's tables the records of its steps start.
 *
 * @param p     The plan.
 * @param count Receives their number.
 * @return      The records' first words, which the caller frees.
 */
static size_t *
records (struct vulkan_plan const *p,
         size_t                   *count)
{
	size_t *const at = allocated(malloc((p->step_count ? p->step_count : 1) * sizeof *at));
	size_t n = 0;
	for (size_t i = 0; i < p->step_count; ++i) {
		size_t const r = record(p, &p->steps[i]);
		if (r != SIZE_MAX)
			at[n++] = r;
	}
	*count = n;
	return at;
}

/** @brief A plan as upstream plans it.
 *
 * Each record that waits names the word before its counters as its error word, and one that only
 * signals none; a persistent run that neither waits nor signals names no record, where the plan's
 * name one that orders nothing, the blob's last segment.
 *
 * @param p The plan, which changes.
 */
static void
upstream (struct vulkan_plan *p)
{
	size_t count = 0;
	size_t *at = records(p, &count);
	for (size_t i = 0; i < count; ++i) {
		uint32_t *const r = &p->tables[at[i]];
		r[RECORD_ERROR] = r[RECORD_TABLE] != UINT32_MAX ? r[RECORD_WAITS] - 1 : 0;
	}
	free(at);
	at = nullptr;
	struct vulkan_segment const last = p->segments[p->segment_count - 1];
	if (last.recipe != VULKAN_RECIPE_TABLE || last.bytes != 4 * RECORD_WORDS)
		return;
	uint32_t const *const r = &p->tables[last.index];
	if (r[RECORD_MAGIC] != 0x54434852u || r[RECORD_WAITS] != UINT32_MAX || r[RECORD_UNITS] ||
	    r[RECORD_TABLE] != UINT32_MAX || r[RECORD_SIGNALS] != UINT32_MAX)
		return;
	for (size_t i = 0; i < p->step_count; ++i) {
		struct vulkan_step const *const s = &p->steps[i];
		uint32_t *const word = &p->push[s->push + s->words - 1];
		if (vulkan_plan_persistent(s->kernel) && UINT64_C(4) * *word == last.offset)
			*word = UINT32_MAX;
	}
	p->table_count = last.index;
	--p->segment_count;
	struct vulkan_segment const *const end = &p->segments[p->segment_count - 1];
	p->blob_bytes = end->offset + end->bytes;
}

/** @brief A copy of a plan as upstream plans it. */
static struct vulkan_plan
upstream_copy (struct vulkan_plan const *p)
{
	struct vulkan_plan c = plan_copy(p);
	upstream(&c);
	return c;
}

/** @brief Hashes formatted text on from a hash.
 *
 * @param hash The hash so far.
 * @param fmt  A printf format.
 * @param ...  The format's arguments.
 * @return     The hash with the text.
 */
[[gnu::format(printf, 2, 3)]]
static uint64_t
hash_text (uint64_t    hash,
           char const *fmt,
           ...)
{
	char text[256];
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(text, sizeof text, fmt, args);
	va_end(args);
	if (!expect(n >= 0 && n < (int)sizeof text, "a hashed line does not fit"))
		return hash;
	return vulkan_pack_fnv1a(hash, text, (size_t)n);
}

/** @brief The steps as the goldens hash them. */
static uint64_t
dispatch_hash (struct vulkan_plan const *p)
{
	uint64_t hash = VULKAN_PACK_FNV1A_BASIS;
	for (size_t i = 0; i < p->step_count; ++i) {
		struct vulkan_step const *const s = &p->steps[i];
		hash = hash_text(hash, "%s %u %u %u %s", stem(s->kernel), s->groups[0], s->groups[1],
		                 s->groups[2], s->after == VULKAN_AFTER_NOTHING ? "1" : "0");
		for (uint32_t w = 0; w < s->words; ++w)
			hash = hash_text(hash, " %u", p->push[s->push + w]);
		hash = vulkan_pack_fnv1a(hash, "\n", 1);
	}
	return hash;
}

/** @brief The bytes of a value's line, its null included. */
#define VALUE_LINE_BYTES 128

/** @brief A value as the goldens hash it.
 *
 * @param v    The value.
 * @param line Receives its line.
 * @return     The line's length.
 */
static size_t
value_line (struct vulkan_value const *v,
            char                       line[VALUE_LINE_BYTES])
{
	int const n = snprintf(line, VALUE_LINE_BYTES,
	                       "%" PRIu64 " b%" PRIu64 "/%" PRIu64 " off %" PRIu64 " size %" PRIu64 " overread %"
	                       PRIu64 "\n", v->key, v->key / 8, v->key % 8, v->offset, v->bytes, v->overread);
	return n < 0 ? 0 : n < VALUE_LINE_BYTES ? (size_t)n : VALUE_LINE_BYTES - 1;
}

/** @brief The values as the goldens hash them. */
static uint64_t
value_hash (struct vulkan_plan const *p)
{
	uint64_t hash = VULKAN_PACK_FNV1A_BASIS;
	for (size_t i = 0; i < p->value_count; ++i) {
		char line[VALUE_LINE_BYTES];
		size_t const length = value_line(&p->values[i], line);
		hash = vulkan_pack_fnv1a(hash, line, length);
	}
	return hash;
}

/** @brief What a plan must hold whatever its extent. */
static void
check_invariants (struct vulkan_plan const *p)
{
	char at[32];
	snprintf(at, sizeof at, "%ux%u", p->width, p->height);
	struct vulkan_step const *const steps = p->steps;
	size_t const count = p->step_count;
	expect(clamped(steps[0].kernel) == VULKAN_KERNEL_FSWIN_IMAGE_PREDS32 &&
	       steps[count - 1].kernel == VULKAN_KERNEL_FSWIN_IMAGE_POST32,
	       "%s: the steps do not start with the pre block and end with the post block", at);
	for (size_t i = 0; i < count; ++i) {
		struct vulkan_step const *const s = &steps[i];
		enum vulkan_kernel const kernel = clamped(s->kernel);
		uint32_t const *const w = &p->push[s->push];
		unsigned const range = VULKAN_PLAN_KERNELS[s->kernel].push;
		expect(s->words * 4u == range && s->push + s->words <= p->push_count,
		       "%s: step %zu, %s, pushes %u words into a range of %u bytes", at, i, stem(s->kernel),
		       s->words, range);
		expect((s->after == VULKAN_AFTER_FULL) == (i + 1 == count), "%s: step %zu is not followed as the %s",
		       at, i, i + 1 == count ? "last" : "others");
		// No step reads the arena where it writes: nothing orders one workgroup against another.
		// A persistent run's layers are in its records.
		if (vulkan_plan_persistent(kernel))
			continue;
		uint64_t in[2] = {w[0], UINT64_MAX};
		uint64_t out = w[1];
		if (kernel == VULKAN_KERNEL_FSWIN_FUSED_UP32 || kernel == VULKAN_KERNEL_FSWIN_IMAGE_POST32) {
			in[0] = w[sizeof (struct push_f_swin) / 4];
			in[1] = w[sizeof (struct push_f_swin) / 4 + 1];
		} else if (s->kernel >= VULKAN_KERNEL_GEMM_PROJC && s->kernel <= VULKAN_KERNEL_GEMM_VQKVS) {
			bool const residual = s->kernel != VULKAN_KERNEL_GEMM_NORES &&
			                      s->kernel != VULKAN_KERNEL_GEMM_VACT &&
			                      s->kernel != VULKAN_KERNEL_GEMM_VQKV_NORM &&
			                      s->kernel != VULKAN_KERNEL_GEMM_VQKV_NORMS &&
			                      s->kernel != VULKAN_KERNEL_GEMM_VQKVS;
			in[0] = w[1];
			in[1] = residual ? w[2] : UINT64_MAX;
			// gemmvqkvs writes binary16s, by their index.
			out = s->kernel == VULKAN_KERNEL_GEMM_VQKVS ? UINT64_C(2) * w[3] : w[3];
		} else if (s->kernel == VULKAN_KERNEL_DEC_UPS) {
			in[0] = UINT64_C(2) * w[0];
			in[1] = w[1];
			out = w[2];
		}
		expect(in[0] != out && in[1] != out,
		       "%s: step %zu, %s, reads the arena at %" PRIu64 " where it writes", at, i, stem(s->kernel),
		       out);
	}
	// fswinfusedup32nh leaves out the tests of its tiles: the fused upsample's grid is unshifted
	// and covers its tile raster whole.
	struct vulkan_step const *up = nullptr;
	for (size_t i = 0; i < count && !up; ++i)
		if (clamped(steps[i].kernel) == VULKAN_KERNEL_FSWIN_FUSED_UP32)
			up = &steps[i];
	struct push_f_swin f = {0};
	if (up)
		memcpy(&f, &p->push[up->push], sizeof f);
	expect(up && !f.shift && !f.shift_y && f.tiles_x == 2 * up->groups[0] && f.tiles_y == 2 * up->groups[1],
	       "%s: the fused upsample is missing, shifted, or not on a grid of its tile raster", at);
	// The segments in order, inside the blob.
	uint64_t end = 0;
	for (size_t i = 0; i < p->segment_count; ++i) {
		struct vulkan_segment const *const s = &p->segments[i];
		expect(s->offset >= end, "%s: the segment at %u overlaps the one before", at, s->offset);
		end = (uint64_t)s->offset + s->bytes;
		if (s->recipe == VULKAN_RECIPE_TABLE)
			expect(s->bytes % 4 == 0 && s->index + s->bytes / 4 <= p->table_count,
			       "%s: the table segment at %u is outside the tables", at, s->offset);
	}
	expect(end <= p->blob_bytes, "%s: the segments end past the blob", at);
	expect(p->segment_count && p->segments[0].recipe == VULKAN_RECIPE_ACTIVATIONS && !p->segments[0].offset,
	       "%s: the activation table is not at the blob's start", at);
	bool noise = false;
	for (size_t i = 0; i < p->segment_count && !noise; ++i) {
		struct vulkan_segment const *const s = &p->segments[i];
		noise = s->recipe == VULKAN_RECIPE_ZEROS && s->offset == 4 * p->noise.off &&
		        s->bytes == (uint64_t)p->noise.width * p->noise.height * 8;
	}
	expect(noise && p->noise.width == p->work_width && p->noise.height == p->work_height,
	       "%s: the noise field is not a zeroed region of the working extent", at);
	// The values below their end; the sync regions and counters after it.
	for (size_t i = 0; i < p->value_count; ++i) {
		struct vulkan_value const *const v = &p->values[i];
		uint64_t const reach = v->bytes < v->overread ? v->overread : v->bytes;
		expect(!v->bytes || v->offset + reach <= p->values_end,
		       "%s: value %" PRIu64 " reaches past the values' end", at, v->key);
	}
	expect(p->values_end % 256 == 0 && p->values_end + UINT64_C(4) * p->counter_words <= p->arena_bytes &&
	       p->arena_bytes <= UINT32_MAX,
	       "%s: the arena's regions do not fit it", at);
	// The words that waits set when they give up: one that every record names, after the frame
	// counter and before every counter, then each persistent run's, sync_off + 3, in its sync
	// region below the frame counter. Nothing else writes them. Every persistent run names a
	// record.
	uint32_t const *const t = p->timeouts;
	size_t const timeouts = p->timeout_count;
	uint32_t frame = UINT32_MAX;
	uint32_t counters = UINT32_MAX;
	uint32_t waits = 0;
	size_t record_count = 0;
	size_t *at_records = records(p, &record_count);
	for (size_t i = 0; i < record_count; ++i) {
		uint32_t const *const r = &p->tables[at_records[i]];
		frame = r[RECORD_FRAME];
		if (r[RECORD_WAITS] < counters)
			counters = r[RECORD_WAITS];
		if (r[RECORD_SIGNALS] < counters)
			counters = r[RECORD_SIGNALS];
		expect(timeouts && r[RECORD_ERROR] == t[0], "%s: a record names another error word than the plan's",
		       at);
		waits += r[RECORD_TABLE] != UINT32_MAX;
	}
	free(at_records);
	at_records = nullptr;
	size_t run_count = 0;
	bool runs_match = true;
	for (size_t i = 0; i < count; ++i) {
		struct vulkan_step const *const s = &steps[i];
		if (!vulkan_plan_persistent(s->kernel))
			continue;
		uint32_t const word = p->push[s->push + offsetof(struct push_persist, sync_off) / 4] + 3;
		runs_match = runs_match && 1 + run_count < timeouts && t[1 + run_count] == word;
		++run_count;
		expect(record(p, s) != SIZE_MAX, "%s: persistent run %s names no record", at, stem(s->kernel));
	}
	runs_match = runs_match && 1 + run_count == timeouts;
	expect(waits == p->chained && timeouts && t[0] == frame + 1 && t[0] < counters && runs_match,
	       "%s: the words that waits set are not the shared one, then each persistent run's", at);
	for (size_t i = 1; i < timeouts; ++i)
		expect(UINT64_C(4) * t[i] >= p->values_end && t[i] < frame && (i == 1 || t[i] > t[i - 1]),
		       "%s: a persistent run's word that its waits set is outside its own sync region", at);
}

/** @brief The plans against upstream's goldens. */
static void
check_goldens (void)
{
	for (size_t i = 0; i < GOLDEN_COUNT; ++i) {
		struct golden const *const g = &GOLDENS[i];
		struct vulkan_plan p;
		struct error e;
		enum error_code const code = vulkan_plan_init(&p, g->width, g->height, UINT64_MAX, &e);
		if (!expect(!code, "%ux%u: %s", g->width, g->height, code ? e.what : ""))
			continue;
		uint32_t runs = 0;
		for (size_t s = 0; s < p.step_count; ++s)
			runs += vulkan_plan_persistent(p.steps[s].kernel);
		expect(p.work_width == g->work_width && p.work_height == g->work_height,
		       "%ux%u: the working extent is %ux%u, upstream's %ux%u", g->width, g->height, p.work_width,
		       p.work_height, g->work_width, g->work_height);
		expect(p.step_count == g->steps && runs == g->runs && p.chained == g->chained,
		       "%ux%u: %zu steps, %u persistent runs and %u chained, upstream's %u, %u and %u", g->width,
		       g->height, p.step_count, runs, p.chained, g->steps, g->runs, g->chained);
		expect(p.arena_bytes == g->arena && p.values_end == g->values_end && p.counter_words == g->counters,
		       "%ux%u: an arena of %" PRIu64 " bytes, values to %" PRIu64 " and %u counter words, "
		       "upstream's %" PRIu64 ", %" PRIu64 " and %u",
		       g->width, g->height, p.arena_bytes, p.values_end, p.counter_words, g->arena, g->values_end,
		       g->counters);
		struct vulkan_plan free_plan = unclamped(&p, &MODEL_CLAMP_FREE);
		struct vulkan_plan theirs = upstream_copy(&free_plan);
		expect(theirs.blob_bytes == g->blob, "%ux%u: a weight blob of %u bytes, upstream's %u", g->width,
		       g->height, theirs.blob_bytes, g->blob);
		struct vulkan_plan ours = upstream_copy(&p);
		expect(dispatch_hash(&ours) == g->dispatches, "%ux%u: the steps differ from upstream's dispatches",
		       g->width, g->height);
		vulkan_plan_fini(&ours);
		expect(dispatch_hash(&theirs) == g->clamped,
		       "%ux%u: the steps with the real model's kernels differ from upstream's dispatches (--print)",
		       g->width, g->height);
		uint64_t const tables = vulkan_pack_fnv1a(VULKAN_PACK_FNV1A_BASIS, theirs.tables,
		                                          4 * theirs.table_count);
		expect(tables == g->real_tables,
		       "%ux%u: the tables with the real model's free heads differ from upstream's", g->width,
		       g->height);
		expect(value_hash(&p) == g->values, "%ux%u: the values differ from upstream's (--print)", g->width,
		       g->height);
		check_invariants(&p);
		check_invariants(&free_plan);
		vulkan_plan_fini(&theirs);
		vulkan_plan_fini(&free_plan);
		vulkan_plan_fini(&p);
	}
}

/** @brief Expects an extent to be rejected with given words after the network's refusal.
 *
 * @param width   The frames' width.
 * @param height  Their height.
 * @param why     The words after the refusal.
 * @param storage The device's storage buffers' bytes.
 */
static void
refused (uint32_t    width,
         uint32_t    height,
         char const *why,
         uint64_t    storage)
{
	char what[ERROR_WHAT_BYTES];
	snprintf(what, sizeof what, "the network does not take %ux%u frames: %s", width, height, why);
	struct vulkan_plan p;
	struct error e;
	enum error_code const code = vulkan_plan_init(&p, width, height, storage, &e);
	expect(code == ERROR_REJECTED && !strcmp(e.what, what), "%ux%u: expected rejected \"%s\", got \"%s\"",
	       width, height, what, code ? e.what : "a plan");
	vulkan_plan_fini(&p);
}

/** @brief The frames that the network cannot take. */
static void
check_rejections (void)
{
	// Upstream's build fails at working extents that are not multiples of 8, looking for kernels
	// its shaders are not built as.
	refused(1, 1, "its working extent 321x320 is not a multiple of 8", UINT64_MAX);
	refused(8, 8, "its working extent 322x320 is not a multiple of 8", UINT64_MAX);
	refused(16, 16, "its working extent 324x320 is not a multiple of 8", UINT64_MAX);
	refused(0, 720, "a side is 0 or above 16384", UINT64_MAX);
	refused(1280, 0, "a side is 0 or above 16384", UINT64_MAX);
	refused(16385, 720, "a side is 0 or above 16384", UINT64_MAX);
	// Upstream truncated the offsets past 4 GiB: at 7680x4320 its values end at 6291095552 bytes.
	refused(7680, 4320, "its activation arena of at least 6291095552 bytes overflows 32-bit offsets",
	        UINT64_MAX);
	// Nor did it check the device's storage buffers, which must each hold the arena or the weights
	// whole.
#define STORAGE " bytes or weights of 163217868 bytes exceed the device's storage buffers of "
	refused(1280, 720, "its activation arena of 252245248" STORAGE "252245247 bytes", 252245247);
	refused(1280, 720, "its activation arena of 252245248" STORAGE "163217867 bytes", 163217867);
#undef STORAGE
	struct vulkan_plan p;
	struct error e;
	expect(!vulkan_plan_init(&p, 1280, 720, 252245248, &e),
	       "1280x720: rejected on a device that holds its arena");
	vulkan_plan_fini(&p);
}

/** @brief A blob that grows to hold a plan's. */
struct blob {
	uint8_t *bytes; //!< The memory.
	size_t   size;  //!< Its bytes.
};

/** @brief Packs a plan's blob from a model, with the kernels without the upper clamp that the model
 *         frees, and hashes it.
 *
 * @param p     The plan, whose kernels change.
 * @param model The model.
 * @param blob  The blob.
 * @param heads Receives the heads that the model frees.
 * @param hash  Receives the blob's FNV-1a 64.
 * @param e     Receives the words for what stopped it, or nullptr.
 * @return      ERROR_NONE, or ERROR_FAILED.
 */
static enum error_code
blob_hash (struct vulkan_plan        *p,
           struct vulkan_model const *model,
           struct blob               *blob,
           struct vulkan_clamp_free  *heads,
           uint64_t                  *hash,
           struct error              *e)
{
	enum error_code code = vulkan_weights_clamp_free(p->segments, p->segment_count, model, heads, e);
	if (code)
		return code;
	vulkan_plan_unclamp(p, heads);
	if (blob->size < p->blob_bytes) {
		blob->bytes = allocated(realloc(blob->bytes, p->blob_bytes));
		blob->size = p->blob_bytes;
	}
	code = vulkan_weights_pack(p->segments, p->segment_count, p->tables, p->table_count, model, blob->bytes,
	                           p->blob_bytes, e);
	if (code)
		return code;
	*hash = vulkan_pack_fnv1a(VULKAN_PACK_FNV1A_BASIS, blob->bytes, p->blob_bytes);
	return ERROR_NONE;
}

/** @brief The blobs packed from the synthetic model against upstream's. */
static void
check_synthetic (void)
{
	struct vulkan_plan p;
	struct error e;
	enum error_code code = vulkan_plan_init(&p, 1280, 720, UINT64_MAX, &e);
	struct vulkan_pack pack;
	vulkan_pack_init(&pack);
	bool const made = !code && vulkan_pack_synthetic_model(&p, &pack, false);
	vulkan_plan_fini(&p);
	if (!expect(made, "cannot make the synthetic model")) {
		vulkan_pack_fini(&pack);
		return;
	}
	struct vulkan_model model;
	code = vulkan_model_open(&model, pack.path, &e);
	if (expect(!code, "the synthetic model: %s", code ? e.what : "")) {
		struct blob blob = {0};
		for (size_t i = 0; i < sizeof PACKED / sizeof *PACKED; ++i) {
			uint32_t const width = PACKED[i].width;
			uint32_t const height = PACKED[i].height;
			struct vulkan_plan q;
			struct vulkan_clamp_free heads = {0};
			uint64_t hash = 0;
			code = vulkan_plan_init(&q, width, height, UINT64_MAX, &e);
			bool const planned = !code;
			if (planned) {
				upstream(&q);
				code = blob_hash(&q, &model, &blob, &heads, &hash, &e);
			}
			expect(planned && !code && hash == golden(width, height)->synthetic_blob,
			       "%ux%u: the blob packed from the synthetic model differs from upstream's%s%s", width,
			       height, planned && code ? ": " : "", planned && code ? e.what : "");
			bool none = true;
			for (size_t b = 0; b < VULKAN_PLAN_BLOCKS; ++b)
				none = none && !heads.heads[b];
			expect(none, "%ux%u: the synthetic model frees a head of the upper clamp", width, height);
			vulkan_plan_fini(&q);
		}
		free(blob.bytes);
		blob.bytes = nullptr;
	}
	vulkan_model_fini(&model);
	vulkan_pack_fini(&pack);
}

/** @brief --model: the plan at each golden extent packed from the real model, against upstream's
 *         blob, the heads that it frees of the upper clamp against upstream's audit, and every entry
 *         the plans read of the size that vulkan_pack_model_bytes() gives it.
 *
 * @param path The real model.
 * @return     The exit status.
 */
static int
check_model (char const *path)
{
	struct vulkan_model model;
	struct error e;
	enum error_code code = vulkan_model_open(&model, path, &e);
	if (!expect(!code, "%s", code ? e.what : "")) {
		vulkan_model_fini(&model);
		return 1;
	}
	struct blob blob = {0};
	struct vulkan_scratch scratch = {0};
	for (size_t i = 0; i < GOLDEN_COUNT; ++i) {
		struct golden const *const g = &GOLDENS[i];
		struct vulkan_plan p;
		code = vulkan_plan_init(&p, g->width, g->height, UINT64_MAX, &e);
		if (!expect(!code, "%ux%u: %s", g->width, g->height, code ? e.what : ""))
			continue;
		if (!i) {
			size_t count = 0;
			struct vulkan_pack_source *read = allocated(vulkan_pack_plan_entries(&p, &count));
			for (size_t r = 0; r < count; ++r) {
				size_t size = 0;
				code = vulkan_model_read(&model, &read[r].source, &scratch, &size, &e);
				expect(!code && size == vulkan_pack_model_bytes(&read[r].source), "%s: %s",
				       read[r].name, !code ? "not the size the synthetic model gives it" : e.what);
			}
			free(read);
			read = nullptr;
		}
		struct vulkan_clamp_free heads = {0};
		uint64_t hash = 0;
		upstream(&p);
		code = blob_hash(&p, &model, &blob, &heads, &hash, &e);
		expect(code || !memcmp(heads.heads, MODEL_CLAMP_FREE.heads, sizeof heads.heads),
		       "%ux%u: %s frees other heads of the upper clamp than upstream's audit", g->width, g->height,
		       path);
		if (expect(!code && hash == g->real_blob,
		           "%ux%u: the blob packed from %s differs from upstream's%s%s", g->width, g->height, path,
		           code ? ": " : "", code ? e.what : ""))
			printf("%ux%u: blob %016" PRIx64 " as upstream's\n", g->width, g->height, hash);
		vulkan_plan_fini(&p);
	}
	vulkan_scratch_fini(&scratch);
	free(blob.bytes);
	blob.bytes = nullptr;
	vulkan_model_fini(&model);
	if (!failures)
		printf("vulkan-plan test: %s packs as upstream packed it\n", path);
	return failures ? 1 : 0;
}

/** @brief --print: the plan at WxH with the real model's kernels and upstream's records, as the
 *         dumps of upstream's build of the real model have it, its dispatches' "D" lines and its
 *         values' "V" lines.
 *
 * @param extent The extent, WxH.
 * @return       The exit status.
 */
static int
print (char const *extent)
{
	uint32_t width;
	uint32_t height;
	if (!support_extent(extent, &width, &height))
		return 2;
	struct vulkan_plan p;
	struct error e;
	if (vulkan_plan_init(&p, width, height, UINT64_MAX, &e)) {
		fprintf(stderr, "vulkan-plan test: %s\n", e.what);
		return 1;
	}
	vulkan_plan_unclamp(&p, &MODEL_CLAMP_FREE);
	upstream(&p);
	printf("source %ux%u plan %ux%u\nact_total %" PRIu64 " values_end %" PRIu64 "\nwblob %u\ndisp %zu\n", width,
	       height, p.work_width, p.work_height, p.arena_bytes, p.values_end, p.blob_bytes, p.step_count);
	for (size_t i = 0; i < p.step_count; ++i) {
		struct vulkan_step const *const s = &p.steps[i];
		printf("D %3zu %-18s b%03ul%u grid %u %u %u steps %u..%u tcnobar %d chainnobar 0 push %u :", i,
		       stem(s->kernel), s->block, s->layer, s->groups[0], s->groups[1], s->groups[2], s->first,
		       s->last, s->after == VULKAN_AFTER_NOTHING, s->words * 4);
		for (uint32_t w = 0; w < s->words; ++w)
			printf(" %u", p.push[s->push + w]);
		printf("\n");
	}
	for (size_t i = 0; i < p.value_count; ++i) {
		char line[VALUE_LINE_BYTES];
		value_line(&p.values[i], line);
		printf("V %s", line);
	}
	printf("tc_words %u\nnoise off %u %ux%u seed %u noise %g\n", p.counter_words, p.noise.off, p.noise.width,
	       p.noise.height, p.noise.seed, (double)p.noise.noise);
	vulkan_plan_fini(&p);
	return 0;
}

/** @brief --markers: the markers the plan expects beside the SPIR-V, each line of each file after
 *         the file's name.
 *
 * @return The exit status.
 */
static int
markers (void)
{
	printf("shader-constants.txt %s\n", VULKAN_PLAN_MANIFEST);
	for (size_t i = 0; i < VULKAN_PLAN_SHADER_CONSTANT_COUNT; ++i)
		printf("shader-constants.txt %s %u\n", VULKAN_PLAN_SHADER_CONSTANTS[i].key,
		       VULKAN_PLAN_SHADER_CONSTANTS[i].value);
	for (size_t i = 0; i < VULKAN_PLAN_MARKER_COUNT; ++i)
		printf("%s %s\n", VULKAN_PLAN_MARKERS[i].file, VULKAN_PLAN_MARKERS[i].line);
	return 0;
}

/** @brief --defines: what the plan assumes of the defines its pipelines are built with beyond the
 *         markers, "pipeline define value" a line.
 *
 * NR_ATTN_WPAIR 1 when the 720p plan packs attn's QKV matrix N-paired, and gemmvqkvnorms' tile of
 * VULKAN_PLAN_GEMM_QKVS_MT tokens (gemm1x1's NR_MTILE) by VULKAN_PLAN_GEMM_QKVS_NT outputs
 * (NR_NTILE).
 *
 * @return The exit status.
 */
static int
defines (void)
{
	struct vulkan_plan p;
	struct error e;
	if (vulkan_plan_init(&p, 1280, 720, UINT64_MAX, &e))
		return 1;
	int status = 1;
	for (size_t i = 0; i < p.step_count; ++i) {
		if (p.steps[i].kernel != VULKAN_KERNEL_ATTN)
			continue;
		struct push_attn a;
		memcpy(&a, &p.push[p.steps[i].push], sizeof a);
		for (size_t s = 0; s < p.segment_count; ++s)
			if (p.segments[s].offset == a.w_off) {
				bool const paired = p.segments[s].flags & VULKAN_SEGMENT_NPAIR;
				printf("attn NR_ATTN_WPAIR %d\n", paired ? 1 : 0);
				printf("gemmvqkvnorms NR_MTILE %u\ngemmvqkvnorms NR_NTILE %u\n",
				       VULKAN_PLAN_GEMM_QKVS_MT, VULKAN_PLAN_GEMM_QKVS_NT);
				status = 0;
				break;
			}
		break;
	}
	vulkan_plan_fini(&p);
	return status;
}

/** @brief Orders entries by name. */
static int
by_name (void const *a,
         void const *b)
{
	return strcmp(((struct vulkan_pack_source const *)a)->name, ((struct vulkan_pack_source const *)b)->name);
}

/** @brief --entries: the model entries the plans read at the golden extents.
 *
 * @return The exit status.
 */
static int
list_entries (void)
{
	struct vulkan_pack_source *all = nullptr;
	size_t count = 0;
	for (size_t i = 0; i < GOLDEN_COUNT; ++i) {
		struct vulkan_plan p;
		struct error e;
		if (vulkan_plan_init(&p, GOLDENS[i].width, GOLDENS[i].height, UINT64_MAX, &e))
			continue;
		size_t read_count = 0;
		struct vulkan_pack_source *read = allocated(vulkan_pack_plan_entries(&p, &read_count));
		vulkan_plan_fini(&p);
		all = allocated(realloc(all, (count + read_count + 1) * sizeof *all));
		memcpy(all + count, read, read_count * sizeof *read);
		count += read_count;
		free(read);
		read = nullptr;
	}
	if (count)
		qsort(all, count, sizeof *all, by_name);
	size_t listed = 0;
	for (size_t i = 0; i < count; ++i)
		if (!i || strcmp(all[i - 1].name, all[i].name)) {
			printf("%s\n", all[i].name);
			++listed;
		}
	free(all);
	all = nullptr;
	return listed ? 0 : 1;
}

/** @brief A binding of a SPIR-V module's set. */
struct binding {
	uint32_t slot; //!< Its binding number, or 1000 + its set for one of another set than 0.
	char     kind; //!< 'b' a storage buffer, 's' a storage image, 't' a sampled one, '?' anything else.
	bool     used; //!< Whether an instruction loads, stores or reaches into it.
};

/** @brief The most push-constant members and bindings an interface holds. */
#define INTERFACE_MOST 64

/** @brief A SPIR-V module's interface.
 *
 * Its push-constant block's members, a type each ('u' u32, 'i' i32, 'f' f32, '?' anything else) at
 * its offset, its descriptor bindings of set 0, and those of them that an instruction loads, stores
 * or reaches into.
 */
struct interface {
	uint32_t       offsets[INTERFACE_MOST];  //!< The members' offsets.
	struct binding bindings[INTERFACE_MOST]; //!< The bindings by slot.
	size_t         binding_count;            //!< Their number.
	size_t         push_count;               //!< The members.
	char           push[INTERFACE_MOST + 1]; //!< The members' types.
};

/** @brief A binding of an interface, added in slot order unless it is there.
 *
 * @param face The interface.
 * @param slot The binding's slot.
 * @return     The binding, or nullptr when the interface holds as many as it can.
 */
static struct binding *
binding_at (struct interface *face,
            uint32_t          slot)
{
	size_t at = 0;
	while (at < face->binding_count && face->bindings[at].slot < slot)
		++at;
	if (at < face->binding_count && face->bindings[at].slot == slot)
		return &face->bindings[at];
	if (face->binding_count == INTERFACE_MOST)
		return nullptr;
	memmove(&face->bindings[at + 1], &face->bindings[at], (face->binding_count - at) * sizeof *face->bindings);
	++face->binding_count;
	face->bindings[at] = (struct binding){slot, '?', false};
	return &face->bindings[at];
}

/** @brief Whether an interface's binding at a slot is used. */
static bool
used_at (struct interface const *face,
         size_t                  slot)
{
	for (size_t i = 0; i < face->binding_count; ++i)
		if (face->bindings[i].slot == slot)
			return face->bindings[i].used;
	return false;
}

/** @brief What a SPIR-V module says of each id. */
struct id_info {
	uint32_t binding;       //!< OpDecorate Binding.
	uint32_t set;           //!< OpDecorate DescriptorSet.
	uint32_t image;         //!< OpTypeImage's Sampled.
	uint32_t element;       //!< OpTypeArray's and OpTypeRuntimeArray's element.
	uint32_t storage;       //!< OpTypePointer's storage class.
	uint32_t pointee;       //!< Its type.
	uint32_t members;       //!< OpTypeStruct's first member type, as a word index.
	uint32_t member_count;  //!< Its members.
	char     scalar;        //!< OpTypeInt's and OpTypeFloat's kind.
	bool     has_binding;   //!< Whether it has a binding.
	bool     is_image;      //!< Whether it is an OpTypeImage.
	bool     is_element;    //!< Whether it is an array.
	bool     buffer_block;  //!< OpDecorate BufferBlock.
	bool     sampled_image; //!< OpTypeSampledImage.
	bool     reached;       //!< Loaded, stored or chained into.
};

/** @brief A member offset of a struct (OpMemberDecorate Offset). */
struct member_offset {
	uint32_t type;   //!< The struct.
	uint32_t member; //!< The member.
	uint32_t offset; //!< Its offset.
};

/** @brief Reads a SPIR-V module's interface.
 *
 * @param module The module's bytes.
 * @param size   Their number.
 * @param face   Receives the interface.
 * @return       true if the module is SPIR-V.
 */
static bool
interface_of (uint8_t const    *module,
              size_t            size,
              struct interface *face)
{
	*face = (struct interface){0};
	size_t const count = size / 4;
	if (count < 5)
		return false;
	uint32_t *w = allocated(malloc(count * sizeof *w));
	memcpy(w, module, count * 4);
	uint32_t const bound = w[3];
	if (w[0] != 0x07230203 || !bound) {
		free(w);
		w = nullptr;
		return false;
	}
	struct id_info *id = allocated(calloc(bound, sizeof *id));
	struct member_offset *offsets = allocated(malloc(count * sizeof *offsets));
	uint32_t *variables = allocated(malloc(count * sizeof *variables));
	size_t offset_count = 0;
	size_t variable_count = 0;
	bool ok = true;
	for (size_t i = 5; i < count && ok;) {
		uint32_t const op = w[i] & 0xffff;
		uint32_t const words = w[i] >> 16;
		if (!words || i + words > count) {
			ok = false;
			break;
		}
		uint32_t const *const a = &w[i + 1];
		// The operands that name ids, as the instructions below read them.
		uint32_t const first = words > 1 ? a[0] : 0;
		switch (op) {
		case 71: // OpDecorate: Binding, DescriptorSet, BufferBlock
			ok = first < bound;
			if (ok && a[1] == 33) {
				id[first].binding = a[2];
				id[first].has_binding = true;
			}
			if (ok && a[1] == 34)
				id[first].set = a[2];
			if (ok && a[1] == 3)
				id[first].buffer_block = true;
			break;
		case 72: // OpMemberDecorate: Offset
			if (a[2] == 35)
				offsets[offset_count++] = (struct member_offset){a[0], a[1], a[3]};
			break;
		case 21:
			ok = first < bound;
			if (ok)
				id[first].scalar = a[1] != 32 ? '?' : a[2] ? 'i' : 'u';
			break;
		case 22:
			ok = first < bound;
			if (ok)
				id[first].scalar = a[1] == 32 ? 'f' : '?';
			break;
		case 25: // OpTypeImage: sampled 1 or 2
			ok = first < bound;
			if (ok) {
				id[first].image = a[6];
				id[first].is_image = true;
			}
			break;
		case 27:
			ok = first < bound;
			if (ok)
				id[first].sampled_image = true;
			break;
		case 28: // OpTypeArray
		case 29: // OpTypeRuntimeArray
			ok = first < bound;
			if (ok) {
				id[first].element = a[1];
				id[first].is_element = true;
			}
			break;
		case 30:
			ok = first < bound;
			if (ok) {
				id[first].members = (uint32_t)(i + 2);
				id[first].member_count = words - 2;
			}
			break;
		case 32:
			ok = first < bound;
			if (ok) {
				id[first].storage = a[1];
				id[first].pointee = a[2];
			}
			break;
		case 59: // OpVariable
			variables[variable_count++] = (uint32_t)(i + 1);
			break;
		case 60: // OpImageTexelPointer
		case 61: // OpLoad
		case 65: // OpAccessChain
		case 66: // OpInBoundsAccessChain
		case 67: // OpPtrAccessChain
			ok = a[2] < bound;
			if (ok)
				id[a[2]].reached = true;
			break;
		case 62: // OpStore
		case 63: // OpCopyMemory
			ok = first < bound;
			if (ok)
				id[first].reached = true;
			break;
		}
		i += words;
	}
	for (size_t v = 0; v < variable_count && ok; ++v) {
		uint32_t const type = w[variables[v]];
		uint32_t const var = w[variables[v] + 1];
		ok = type < bound && var < bound;
		if (!ok)
			break;
		uint32_t const storage = id[type].storage;
		uint32_t const pointee = id[type].pointee;
		ok = pointee < bound;
		if (!ok)
			break;
		if (storage == 9) { // PushConstant
			for (uint32_t m = 0; m < id[pointee].member_count && face->push_count < INTERFACE_MOST; ++m) {
				uint32_t const member = w[id[pointee].members + m];
				bool const scalar = member < bound && id[member].scalar;
				face->push[face->push_count++] = scalar ? id[member].scalar : '?';
				uint32_t offset = 0;
				for (size_t o = 0; o < offset_count; ++o)
					if (offsets[o].type == pointee && offsets[o].member == m)
						offset = offsets[o].offset;
				face->offsets[m] = offset;
			}
			continue;
		}
		if (!id[var].has_binding || id[var].set) {
			if (id[var].has_binding) {
				struct binding *const b = binding_at(face, 1000 + id[var].set);
				if (b)
					b->kind = '?';
			}
			continue;
		}
		uint32_t t = pointee;
		while (t < bound && id[t].is_element)
			t = id[t].element;
		char kind = '?';
		if (t < bound) {
			if (storage == 12 || (storage == 2 && id[t].buffer_block))
				kind = 'b';
			else if (storage == 0 && id[t].sampled_image)
				kind = 't';
			else if (storage == 0 && id[t].is_image)
				kind = id[t].image == 2 ? 's' : 't';
		}
		struct binding *const b = binding_at(face, id[var].binding);
		if (b) {
			b->kind = kind;
			b->used = id[var].reached;
		}
	}
	free(variables);
	variables = nullptr;
	free(offsets);
	offsets = nullptr;
	free(id);
	id = nullptr;
	free(w);
	w = nullptr;
	return ok;
}

/** @brief Text and its length. */
struct part {
	char const *text;   //!< The text.
	size_t      length; //!< Its length.
};

/** @brief A string literal as a part. */
#define PART(literal) {"" literal, sizeof literal - 1}

/** @brief Writes each kernel's push block as the plan fills it, a type a word: its parts in a row,
 *         and the tile-counter word of the kernels that take one. A twin without the upper clamp
 *         takes its kernel's.
 *
 * @param k     The kernel.
 * @param types Receives the types and a null.
 * @return      The types' length.
 */
static size_t
push_types (enum vulkan_kernel k,
            char               types[INTERFACE_MOST + 1])
{
	static struct part const swin = PART("uuuuuuuuuuuuuuuiiuuu");
	static struct part const ds = PART("uuuuuuuuuuu");
	static struct part const ups = PART("uuuuuuuuuu");
	static struct part const image = PART("uuuufffffffu");
	static struct part const tail = PART("uf");
	static struct part const persist = PART("uuuuuuuu");
	static struct part const gemm = PART("uuuuuuuuuuuuuuuuuuuuu");
	static struct part const counter = PART("u");
	struct part parts[3] = {0};
	switch (clamped(k)) {
	case VULKAN_KERNEL_FSWIN32:
		parts[0] = swin, parts[1] = counter;
		break;
	case VULKAN_KERNEL_FSWIN_IMAGE_PREDS32:
		parts[0] = swin, parts[1] = image;
		break;
	case VULKAN_KERNEL_FSWIN_DSP32:
		parts[0] = swin, parts[1] = ds, parts[2] = counter;
		break;
	case VULKAN_KERNEL_FSWIN_FUSED_UP32:
		parts[0] = swin, parts[1] = ups, parts[2] = (struct part)PART("uu");
		break;
	case VULKAN_KERNEL_FSWIN_IMAGE_POST32:
		parts[0] = swin, parts[1] = ups, parts[2] = tail;
		break;
	case VULKAN_KERNEL_FFWD3:
	case VULKAN_KERNEL_FFWD3W:
		parts[0] = (struct part)PART("uuuuuuuu");
		break;
	case VULKAN_KERNEL_ATTN:
		parts[0] = (struct part)PART("uuuuuuuuuiiu");
		break;
	case VULKAN_KERNEL_GEMM_POOL:
	case VULKAN_KERNEL_GEMM_NORES:
	case VULKAN_KERNEL_GEMM_VQKVS:
		parts[0] = gemm;
		break;
	case VULKAN_KERNEL_VIT_ATTN:
		parts[0] = (struct part)PART("uuuuuu");
		break;
	case VULKAN_KERNEL_DEC_UPS:
		parts[0] = (struct part)PART("uuuuuuu");
		break;
	case VULKAN_KERNEL_UPS_VIEW:
		parts[0] = (struct part)PART("uuuuuuuu");
		break;
	case VULKAN_KERNEL_NOISE_FIELD:
		parts[0] = (struct part)PART("uuuuf");
		break;
	default:
		parts[0] = vulkan_plan_persistent(k) ? persist : gemm, parts[1] = counter;
		break;
	}
	size_t length = 0;
	for (size_t i = 0; i < 3 && parts[i].text; ++i) {
		memcpy(types + length, parts[i].text, parts[i].length);
		length += parts[i].length;
	}
	types[length] = '\0';
	return length;
}

#undef PART

/** @brief --spirv: each kernel's SPIR-V in a directory against the push block and bindings the plan
 *         gives it, and the post block's stores against the images the runtime binds.
 *
 * @param directory The network's SPIR-V.
 * @return          The exit status; 77 when the directory is not a built network.
 */
static int
check_spirv (char const *directory)
{
	size_t const directory_length = strlen(directory);
	char *constants = allocated(files_join(directory, directory_length, "shader-constants.txt",
	                                             sizeof "shader-constants.txt" - 1, nullptr));
	bool const built = files_is_regular_file(constants);
	free(constants);
	constants = nullptr;
	if (!built) {
		printf("vulkan-plan test: skipped: no network SPIR-V in %s\n", directory);
		return 77;
	}
	for (size_t k = 0; k < VULKAN_KERNEL_COUNT; ++k) {
		struct vulkan_kernel_info const *const info = &VULKAN_PLAN_KERNELS[k];
		char file[VULKAN_WEIGHTS_NAME_BYTES];
		int const n = snprintf(file, sizeof file, "g_%s.spv", info->stem);
		if (!expect(n > 0 && n < (int)sizeof file, "a kernel's file name does not fit"))
			continue;
		char *path = allocated(files_join(directory, directory_length, file, (size_t)n, nullptr));
		struct files_data module;
		struct error e;
		enum error_code const code = files_read(&module, path, &e);
		struct interface face;
		bool const read = !code && interface_of(module.bytes, module.size, &face) && face.binding_count;
		files_data_fini(&module);
		if (!expect(read, "%s: %s", path, !code ? "not SPIR-V with bindings" : e.what)) {
			free(path);
			path = nullptr;
			continue;
		}
		// The block is the plan's, but gemmpool's, gemmnores' and gemmvqkvs' end before remap, the
		// plan's last word.
		char types[INTERFACE_MOST + 1];
		size_t const type_count = push_types((enum vulkan_kernel)k, types);
		bool const short_gemm = k == VULKAN_KERNEL_GEMM_POOL || k == VULKAN_KERNEL_GEMM_NORES ||
		                        k == VULKAN_KERNEL_GEMM_VQKVS;
		size_t const member_count = face.push_count;
		bool packed = member_count == type_count - short_gemm && !memcmp(face.push, types, member_count) &&
		              type_count * 4 == info->push;
		for (size_t m = 0; packed && m < member_count; ++m)
			packed = face.offsets[m] == 4 * m;
		expect(packed, "%s: a push block of %s, not the plan's %s%s in %u bytes", path, face.push, types,
		       short_gemm ? " without its last word" : "", info->push);
		char kinds[INTERFACE_MOST + 1];
		size_t const buffers = strlen(info->buffers);
		for (size_t i = 0; i < buffers; ++i)
			kinds[i] = info->buffers[i] == 'a' || info->buffers[i] == 'w' ? 'b' : info->buffers[i];
		size_t kind_count = buffers;
		if (info->images == VULKAN_IMAGES_INPUT)
			kinds[kind_count++] = 't';
		if (info->images == VULKAN_IMAGES_OUTPUT) {
			memcpy(kinds + kind_count, "sst", 3);
			kind_count += 3;
		}
		for (size_t b = 0; b < face.binding_count; ++b) {
			uint32_t const slot = face.bindings[b].slot;
			char const kind = face.bindings[b].kind;
			expect(slot < kind_count && kinds[slot] == kind, "%s: binding %u is '%c', the plan's '%c'",
			       path, slot, kind, slot < kind_count ? kinds[slot] : '-');
		}
		// The post block stores the answer and never its second output, where the runtime binds a
		// 1x1 image unless the temporal post block writes it.
		size_t const answer = buffers;
		if (info->images == VULKAN_IMAGES_OUTPUT)
			expect(used_at(&face, answer) && !used_at(&face, answer + 1),
			       "%s: the answer is %sstored, the second output %sstored", path,
			       used_at(&face, answer) ? "" : "not ", used_at(&face, answer + 1) ? "" : "not ");
		free(path);
		path = nullptr;
	}
	if (!failures)
		puts("vulkan-plan test: every kernel's push block and bindings are the plan's");
	return failures ? 1 : 0;
}

/** @brief Every kernel's bindings within VULKAN_KERNEL_MOST_BINDINGS, which the runtime's binding
 *         arrays hold. */
static void
check_kernels (void)
{
	static unsigned const images[] = {[VULKAN_IMAGES_NONE] = 0, [VULKAN_IMAGES_INPUT] = 1,
	                                  [VULKAN_IMAGES_OUTPUT] = 3};
	for (size_t k = 0; k < VULKAN_KERNEL_COUNT; ++k) {
		struct vulkan_kernel_info const *const info = &VULKAN_PLAN_KERNELS[k];
		size_t const bindings = strlen(info->buffers) + images[info->images];
		expect(bindings <= VULKAN_KERNEL_MOST_BINDINGS, "%s takes %zu bindings, more than %d", info->stem,
		       bindings, VULKAN_KERNEL_MOST_BINDINGS);
	}
}

/** @brief What --help prints. */
static char const USAGE[] =
	"Usage: vulkan-plan-test [OPTION]...\n"
	"Checks the Vulkan network's plan against goldens of upstream's graph build. No GPU or model\n"
	"needed.\n"
	" -p, --print WxH    Instead, print the plan for WxH frames with the real model's kernels in\n"
	"                    the form of the dumps of upstream's build that the goldens come from\n"
	"                    (default: unset)\n"
	" -m, --model FILE   Instead, pack the weight blob from the model pack FILE at every golden\n"
	"                    extent and check it, and the heads that FILE frees of the exponent's\n"
	"                    upper clamp, against upstream's (default: unset)\n"
	" -s, --spirv DIR    Instead, check each kernel's SPIR-V in DIR against the push block and\n"
	"                    bindings the plan gives it, and that the post block stores the answer\n"
	"                    and not its second output; exit 77 when DIR holds no network\n"
	"                    (default: unset)\n"
	" -M, --markers      Instead, print the markers the plan expects beside the SPIR-V, each\n"
	"                    line after its file's name (default: off)\n"
	" -D, --defines      Instead, print what the plan assumes of its pipelines' defines beyond\n"
	"                    the markers, \"pipeline define value\" a line (default: off)\n"
	" -e, --entries      Instead, print the model entries the plan reads (default: off)\n"
	" -h, --help         Show help (default: off)\n";

int
main (int    argc,
      char **argv)
{
	char const *model = "";
	char const *spirv = "";
	char const *extent = "";
	bool list_markers = false;
	bool list_defines = false;
	bool list_read = false;
	static struct option const options[] = {
		{"print",   required_argument, nullptr, 'p'},
		{"model",   required_argument, nullptr, 'm'},
		{"spirv",   required_argument, nullptr, 's'},
		{"markers", no_argument,       nullptr, 'M'},
		{"defines", no_argument,       nullptr, 'D'},
		{"entries", no_argument,       nullptr, 'e'},
		{"help",    no_argument,       nullptr, 'h'},
		{},
	};
	for (int code; (code = getopt_long(argc, argv, "+p:m:s:MDeh", options, nullptr)) != -1;) {
		switch (code) {
		case 'p':
			extent = optarg;
			break;
		case 'm':
			model = optarg;
			break;
		case 's':
			spirv = optarg;
			break;
		case 'M':
			list_markers = true;
			break;
		case 'D':
			list_defines = true;
			break;
		case 'e':
			list_read = true;
			break;
		case 'h':
			fputs(USAGE, stdout);
			return support_written("vulkan-plan test", 0);
		default:
			return 2;
		}
	}
	if (optind != argc || !!*extent + !!*model + !!*spirv + list_markers + list_defines + list_read > 1)
		return 2;
	if (*extent)
		return support_written("vulkan-plan test", print(extent));
	if (*model)
		return support_written("vulkan-plan test", check_model(model));
	if (*spirv)
		return support_written("vulkan-plan test", check_spirv(spirv));
	if (list_markers)
		return support_written("vulkan-plan test", markers());
	if (list_defines)
		return support_written("vulkan-plan test", defines());
	if (list_read)
		return support_written("vulkan-plan test", list_entries());
	check_kernels();
	check_goldens();
	check_rejections();
	check_synthetic();
	if (!failures)
		puts("vulkan-plan test: every check passed");
	return support_written("vulkan-plan test", failures ? 1 : 0);
}
