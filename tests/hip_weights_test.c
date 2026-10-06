/** @file
 *
 * Host test of the HIP network's weight packing, without a GPU or a model. The encodings are
 * checked against their definitions, and they and every recipe against digests that the pinned
 * upstream packers (packed_weights.h and the weight loaders of hip_reference_network.h at c190831)
 * gave for the same inputs. --model packs a real model instead, for comparison with the uploads a
 * trace of upstream's network records.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "files.h"
#include "hip_plan.h"
#include "hip_weights.h"
#include "shm_protocol.h"
#include "support.h"

/** @brief The FNV-1a 64 offset basis. */
#define FNV1A_BASIS UINT64_C(0xcbf29ce484222325)

/** @brief The FNV-1a 64 of bytes, after others.
 *
 * @param data  The bytes.
 * @param bytes Their number.
 * @param hash  The hash of the bytes before them, or FNV1A_BASIS.
 * @return      The hash.
 */
static uint64_t
fnv1a (void const *data,
       size_t      bytes,
       uint64_t    hash)
{
	for (unsigned char const *at = data; bytes--; ++at)
		hash = (hash ^ *at) * 0x100000001b3u;
	return hash;
}

/** @brief Formats words into a buffer, or ends the test when they do not fit.
 *
 * @param dest Receives the words and a null.
 * @param size The buffer's bytes.
 * @param fmt  A printf format.
 * @param ...  The format's arguments.
 * @return     The words' length.
 */
[[gnu::format(printf, 3, 4)]]
static size_t
format (char       *dest,
        size_t      size,
        char const *fmt,
        ...)
{
	va_list args;
	va_start(args, fmt);
	int const length = vsnprintf(dest, size, fmt, args);
	va_end(args);
	if (length < 0 || (size_t)length >= size) {
		fprintf(stderr, "hip-weights test: \"%s\" does not fit %zu bytes\n", fmt, size);
		exit(1);
	}
	return (size_t)length;
}

/** @brief The bytes of a weight's key, its null included. */
#define KEY_BYTES (HIP_WEIGHTS_STEM_BYTES + 16)

/** @brief Writes a weight's key: STEM@SUFFIX, upstream's key without the extension; STEM alone
 *         for a raw weight.
 *
 * @param spec The weight.
 * @param key  Receives the key and a null.
 * @return     The key's length.
 */
static size_t
key (struct hip_weight_spec const *spec,
     char                          key[KEY_BYTES])
{
	char const *const suffix = HIP_WEIGHTS_RECIPE_SUFFIX[spec->recipe];
	return format(key, KEY_BYTES, "%s%s%s", spec->stem, *suffix ? "@" : "", suffix);
}

/** @brief splitmix64. */
struct random {
	uint64_t state; //!< Its state.
};

/** @brief The next number of a splitmix64. */
static uint64_t
random_next (struct random *r)
{
	uint64_t z = r->state += 0x9e3779b97f4a7c15u;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
	return z ^ (z >> 31);
}

/** @brief What a synthetic weight holds at an element: an exact E4M3 value, an exact binary16
 *         value, anything the rounding packers see (ties, subnormals, overflow, NaN), a scale or
 *         bias (kept as read, or split into E4M3 pieces), or a zero outside the grouped
 *         contraction. */
enum content : uint8_t {
	CONTENT_FP8,
	CONTENT_HALF,
	CONTENT_ROUNDED,
	CONTENT_FLOAT,
	CONTENT_ZERO,
};

/** @brief What content() needs to know of a weight, worked out once per weight. */
struct shape {
	size_t          c;         //!< Its channels.
	size_t          fp8_first; //!< The first E4M3 value of a C32 weight.
	size_t          fp8_end;   //!< The end of its E4M3 values.
	enum hip_recipe recipe;    //!< Its recipe.
};

/** @brief What content() needs to know of a weight.
 *
 * @param spec The weight.
 * @return     Its shape.
 */
static struct shape
shape (struct hip_weight_spec const *spec)
{
	// A C32 attention weight's E4M3 values come first, an FFN weight's after its first 512.
	static char const attention[] = "attention";
	size_t const length = spec->length, suffix = sizeof attention - 1;
	bool const is_attention = length >= suffix && !memcmp(spec->stem + length - suffix, attention, suffix);
	return (struct shape){
		.c = hip_weights_channels(spec),
		.fp8_first = is_attention ? 0 : 512,
		.fp8_end = is_attention ? 4096 : 8704,
		.recipe = spec->recipe,
	};
}

/** @brief What a synthetic weight holds at an element.
 *
 * @param w The weight's shape.
 * @param i The element.
 * @return  Its content.
 */
static enum content
content (struct shape const *w,
         size_t              i)
{
	size_t const c = w->c, cc = c * c;
	switch (w->recipe) {
	case HIP_RECIPE_RAW:
	case HIP_RECIPE_DS_CAST:
		return CONTENT_ROUNDED;
	case HIP_RECIPE_C32:
		return i >= w->fp8_first && i < w->fp8_end ? CONTENT_FP8 : CONTENT_FLOAT;
	case HIP_RECIPE_DS_FRAG:
		return CONTENT_HALF;
	case HIP_RECIPE_MH_FFN:
	case HIP_RECIPE_FFN_FRAG:
		if (i >= 4 * cc && i < 8 * cc && (i - 4 * cc) % (4 * c) / 128 != (i - 4 * cc) / (4 * c) / 32)
			return CONTENT_ZERO;
		return i < 9 * cc ? CONTENT_FP8 : CONTENT_FLOAT;
	case HIP_RECIPE_MH_ATTENTION:
	case HIP_RECIPE_MH_ATTENTION_DIAG:
	case HIP_RECIPE_QKV_FRAG_ONLY:
	case HIP_RECIPE_QKV_FRAG:
		return i < 4 * cc ? CONTENT_FP8 : CONTENT_FLOAT;
	case HIP_RECIPE_SPLIT_MIX_F16:
		return i < 262144 ? CONTENT_ROUNDED : i < 393216 ? CONTENT_HALF : CONTENT_FP8;
	case HIP_RECIPE_PROJ_FRAG:
		return i < 262144 ? CONTENT_FP8 : CONTENT_FLOAT;
	case HIP_RECIPE_VIT_FRAG:
		return i < 4194304 ? CONTENT_FP8 : CONTENT_FLOAT;
	case HIP_RECIPE_QKV_F16_FRAG:
		return i < 3145728 ? CONTENT_HALF : CONTENT_FLOAT;
	case HIP_RECIPE_VIT_PROJ_FRAG:
		return i < 1048576 ? CONTENT_FP8 : CONTENT_FLOAT;
	case HIP_RECIPE_DECODER_F16R:
		return i < 2 * cc ? CONTENT_ROUNDED : CONTENT_FLOAT;
	case HIP_RECIPE_COUNT:
		break;
	}
	return CONTENT_FLOAT;
}

/** @brief The value of an E4M3 code other than NaN. */
static float
fp8_value (unsigned code)
{
	unsigned const e = code >> 3 & 15, m = code & 7;
	float const magnitude = e ? ldexpf((float)(8 + m), (int)e - 10) : (float)m / 512;
	return code & 0x80 ? -magnitude : magnitude;
}

/** @brief The value of a finite binary16. */
static float
half_value (unsigned h)
{
	unsigned const e = h >> 10 & 31, m = h & 1023;
	float const magnitude = e ? ldexpf((float)(1024 + m), (int)e - 25) : ldexpf((float)m, -24);
	return h & 0x8000 ? -magnitude : magnitude;
}

/** @brief A value as binary16; it is zero or a normal binary16 value. */
static uint16_t
half_bits (float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4);
	uint32_t const sign = bits >> 16 & 0x8000;
	return (uint16_t)(bits & 0x7fffffff ? sign | ((bits >> 23 & 255) - 112) << 10 | (bits >> 13 & 1023) : sign);
}

/** @brief An E4M3 code other than NaN, from random bits. */
static unsigned
fp8_code (uint64_t x)
{
	return (x & 0x7f) == 0x7f ? (unsigned)(x & 0xff) ^ 1 : (unsigned)(x & 0xff);
}

/** @brief A finite binary16, from random bits. */
static unsigned
finite_half (uint64_t x)
{
	return (x & 0x7c00) == 0x7c00 ? (unsigned)(x & 0xffff) ^ 0x4000 : (unsigned)(x & 0xffff);
}

/** @brief A random f32 value of a content.
 *
 * @param c The content.
 * @param r The numbers.
 * @return  The value.
 */
static float
f32_value (enum content   c,
           struct random *r)
{
	uint64_t const x = random_next(r);
	uint32_t const low = (uint32_t)x, sign = low & 0x80000000u;
	unsigned const pick = (unsigned)(x >> 60), draw = (unsigned)(x >> 32 & 0xffff);
	uint32_t bits = low;
	switch (c) {
	case CONTENT_FP8:
		return fp8_value(fp8_code(x));
	case CONTENT_HALF:
		return half_value(finite_half(x));
	case CONTENT_ZERO:
		return x & 1 ? -0.f : 0.f;
	case CONTENT_ROUNDED:
		// Binary16's and E4M3's ranges and a little beyond; one in sixteen is any bit pattern:
		// NaN, infinity, the huge and the tiny.
		if (pick) {
			unsigned const exponent = 100 + draw % 45; // 2^-27 .. 2^17
			uint32_t mantissa = low & 0x7fffff;
			if (pick == 1)
				mantissa = (mantissa & ~0x1fffu) | 0x1000; // binary16 tie
			if (pick == 2)
				mantissa = (mantissa & ~0xfffffu) | 0x80000; // E4M3 tie
			if (pick == 3) { // binary16 subnormal tie
				unsigned const e = 102 + draw % 11, n = 126 - e; // 2^-25 .. 2^-15
				mantissa = (((mantissa | 0x800000) & ~((1u << n) - 1)) | 1u << (n - 1)) & 0x7fffff;
				bits = sign | e << 23 | mantissa;
				break;
			}
			bits = sign | exponent << 23 | mantissa;
		}
		break;
	case CONTENT_FLOAT:
		// Mostly the magnitudes of scales and biases; some below E4M3's normals, some beyond
		// its range, and a few of any bit pattern.
		if (pick) {
			unsigned const exponent = pick < 4 ? 97 + draw % 24
			                        : pick < 6 ? 135 + draw % 12
			                        : 115 + draw % 16;
			bits = sign | exponent << 23 | (low & 0x7fffff);
		}
		break;
	}
	float value;
	memcpy(&value, &bits, 4);
	return value;
}

/** @brief A random binary16 of a content.
 *
 * @param c The content.
 * @param r The numbers.
 * @return  The binary16 bits.
 */
static uint16_t
f16_value (enum content   c,
           struct random *r)
{
	uint64_t const x = random_next(r);
	switch (c) {
	case CONTENT_FP8:
		return half_bits(fp8_value(fp8_code(x)));
	case CONTENT_HALF:
		return (uint16_t)finite_half(x);
	case CONTENT_ZERO:
		return x & 1 ? 0x8000 : 0;
	default:
		return (uint16_t)x; // Any binary16: NaN, infinity, subnormals.
	}
}

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
	fputs("hip-weights test: ", stderr);
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
		fputs("hip-weights test: out of memory\n", stderr);
		exit(1);
	}
}

/** @brief Expects a call to have failed with given words.
 *
 * @param code What the call returned.
 * @param e    Its error.
 * @param fmt  A printf format for the words it should have put there.
 * @param ...  The format's arguments.
 */
[[gnu::format(printf, 3, 4)]]
static void
expect_error (enum error_code     code,
              struct error const *e,
              char const         *fmt,
              ...)
{
	va_list args;
	va_start(args, fmt);
	char *words = support_vformat(nullptr, fmt, args);
	va_end(args);
	allocated(words);
	char const *const what = code ? e->what : "no error";
	expect(!strcmp(what, words), "expected \"%s\", got \"%s\"", words, what);
	free(words);
	words = nullptr;
}

/** @brief A synthetic file of a weight: f32, or binary16.
 *
 * @param w        The weight's shape.
 * @param name     Its key.
 * @param length   The key's length.
 * @param elements Its values.
 * @param half     Whether binary16.
 * @param size     Receives the file's bytes.
 * @return         The file's bytes, which the caller frees.
 */
static uint8_t *
synthetic_file (struct shape const *w,
                char const         *name,
                size_t              length,
                size_t              elements,
                bool                half,
                size_t             *size)
{
	struct random r = {fnv1a(name, length, FNV1A_BASIS) + half};
	*size = elements << (half ? 1 : 2);
	uint8_t *const bytes = malloc(*size);
	allocated(bytes);
	for (size_t i = 0; i < elements; ++i) {
		if (half) {
			uint16_t const value = f16_value(content(w, i), &r);
			memcpy(bytes + i * 2, &value, 2);
		} else {
			float const value = f32_value(content(w, i), &r);
			memcpy(bytes + i * 4, &value, 4);
		}
	}
	return bytes;
}

/** @brief One weight per recipe and channel count that upstream's production path packs. */
static struct hip_weight_spec const cases[] = {
	HIP_WEIGHT_SPEC("post70-scales", HIP_RECIPE_RAW),
	HIP_WEIGHT_SPEC("post70-head", HIP_RECIPE_RAW),
	HIP_WEIGHT_SPEC("block23-ffwd", HIP_RECIPE_RAW),
	HIP_WEIGHT_SPEC("block0-ffn", HIP_RECIPE_C32),
	HIP_WEIGHT_SPEC("block0-attention", HIP_RECIPE_C32),
	HIP_WEIGHT_SPEC("block4-ds", HIP_RECIPE_DS_CAST),
	HIP_WEIGHT_SPEC("block8-ds", HIP_RECIPE_DS_FRAG),
	HIP_WEIGHT_SPEC("block14-ds", HIP_RECIPE_DS_FRAG),
	HIP_WEIGHT_SPEC("block22-ds", HIP_RECIPE_DS_FRAG),
	HIP_WEIGHT_SPEC("head-matrix", HIP_RECIPE_DS_FRAG),
	HIP_WEIGHT_SPEC("block5-ffn", HIP_RECIPE_MH_FFN),
	HIP_WEIGHT_SPEC("block9-ffn", HIP_RECIPE_MH_FFN),
	HIP_WEIGHT_SPEC("block5-attention", HIP_RECIPE_MH_ATTENTION),
	HIP_WEIGHT_SPEC("block9-attention", HIP_RECIPE_MH_ATTENTION),
	HIP_WEIGHT_SPEC("block15-attention", HIP_RECIPE_MH_ATTENTION),
	HIP_WEIGHT_SPEC("block23-attention", HIP_RECIPE_MH_ATTENTION),
	HIP_WEIGHT_SPEC("block5-attention", HIP_RECIPE_MH_ATTENTION_DIAG),
	HIP_WEIGHT_SPEC("block9-attention", HIP_RECIPE_MH_ATTENTION_DIAG),
	HIP_WEIGHT_SPEC("block15-attention", HIP_RECIPE_MH_ATTENTION_DIAG),
	HIP_WEIGHT_SPEC("block15-ffn", HIP_RECIPE_FFN_FRAG),
	HIP_WEIGHT_SPEC("block15-attention", HIP_RECIPE_QKV_FRAG_ONLY),
	HIP_WEIGHT_SPEC("block23-ffwd", HIP_RECIPE_SPLIT_MIX_F16),
	HIP_WEIGHT_SPEC("block23-ffwd-projection", HIP_RECIPE_PROJ_FRAG),
	HIP_WEIGHT_SPEC("block23-attention", HIP_RECIPE_QKV_FRAG),
	HIP_WEIGHT_SPEC("block31-expand", HIP_RECIPE_VIT_FRAG),
	HIP_WEIGHT_SPEC("block31-contract", HIP_RECIPE_VIT_FRAG),
	HIP_WEIGHT_SPEC("block31-qkv", HIP_RECIPE_QKV_F16_FRAG),
	HIP_WEIGHT_SPEC("block31-projection", HIP_RECIPE_VIT_PROJ_FRAG),
	HIP_WEIGHT_SPEC("decoder39-weights", HIP_RECIPE_DECODER_F16R),
	HIP_WEIGHT_SPEC("block48-weights", HIP_RECIPE_DECODER_F16R),
	HIP_WEIGHT_SPEC("block56-weights", HIP_RECIPE_DECODER_F16R),
	HIP_WEIGHT_SPEC("block62-weights", HIP_RECIPE_DECODER_F16R),
	HIP_WEIGHT_SPEC("block66-weights", HIP_RECIPE_DECODER_F16R),
};

/** @brief The number of f32 bit patterns in the sample. */
#define SAMPLE_COUNT (UINT32_C(1) << 21)

/** @brief A sample of f32 bit patterns: every sign, exponent and top 11 mantissa bits, with the
 *         low 12 bits zero and with them scrambled. */
static uint32_t
sample (uint32_t j)
{
	return (j >> 1) << 12 | (j & 1) * ((j * 0x9e3779b1u) >> 20);
}

/** @brief What upstream gave for the sample and for every binary16. */
static struct {
	uint64_t widen_half;
	uint64_t exact_fp8;
	uint64_t exact_half;
	uint64_t round_half;
	uint64_t saturate_fp8;
	uint64_t scale_piece;
} const upstream_primitives = {
	0xb0659868ec053145u, 0xc7f211df4383f3c9u, 0x92886aaababdea21u,
	0xe5692a3085370020u, 0xb34b0cb5c438f085u, 0xd121ea1c179fb205u,
};

/** @brief What upstream made of a case, from its synthetic input as f32 and as binary16: the inputs'
 *         hashes, the image's bytes and the images' hashes. */
struct case_digest {
	uint64_t input[2];  //!< The input's hash, f32 and binary16.
	uint64_t packed[2]; //!< The image's hash of each.
	size_t   bytes;     //!< The image's bytes.
};

/** @brief What upstream made of each case, in the order of cases[]. */
static struct case_digest const upstream_cases[] = {
	// post70-scales
	{{0xe2d6d4e40e98cf1du, 0xee46147e18c68760u}, {0xe2d6d4e40e98cf1du, 0x2b10b2017141567du}, 256},
	// post70-head
	{{0xdfd82875d596ca19u, 0x1c91e0f64fdfabdcu}, {0xdfd82875d596ca19u, 0xd4bfc8202e72fd6au}, 384},
	// block23-ffwd
	{{0x24b764c0257dd203u, 0xeb4a17b7dab08ff5u}, {0x24b764c0257dd203u, 0xb9be0717ea626b5bu}, 2097152},
	// block0-ffn@c32fp8
	{{0x98ee6f18e84fe688u, 0xb5c3e5b70f2f1e39u}, {0xa6f9e0c65c7de9b3u, 0x60720e754fc68466u}, 41088},
	// block0-attention@c32fp8
	{{0xe1f60a5bd1a958beu, 0x214e4cb7cd465213u}, {0xe792550614343a6bu, 0x75048295436b358du}, 32900},
	// block4-ds@ds-cast
	{{0x9f3b50a11bef0c0bu, 0x1a9f86d204abc565u}, {0x5d4396a4207ec368u, 0xc21ae67dc05d97afu}, 8192},
	// block8-ds@ds-frag
	{{0xd0311835a075b300u, 0xbb97eed0c9726cd5u}, {0x9388d2044df54e46u, 0xe1741fece8cb626du}, 16384},
	// block14-ds@ds-frag
	{{0x98d274b090362e6fu, 0x1214055e48c43fb5u}, {0xc2e5500b3706d0aau, 0x48dd15c11bd2d469u}, 65536},
	// block22-ds@ds-frag
	{{0x9870fe06e692a6d0u, 0x77924ebb70bf3543u}, {0x3ac30cc5da23023eu, 0x323716957a9b3ecbu}, 262144},
	// head-matrix@ds-frag
	{{0x913895cf14f63e03u, 0x9f4f7574e41fdd78u}, {0xa2dc2698a5a49053u, 0xb176949723e94244u}, 1048576},
	// block5-ffn@fp8-g128
	{{0x9da6e5dc69547e20u, 0xc489ba41a5432f70u}, {0x1883b6b248a77735u, 0xb53e550921c2952du}, 147712},
	// block9-ffn@fp8-g128
	{{0x7921b9ad45d92e24u, 0x03cea55d8f1e3cfau}, {0x9b350f1a9d6931a5u, 0xc6b73a120e5abee1u}, 590336},
	// block5-attention@fp8
	{{0xd0c4fe11ad594fedu, 0xaeeb9207c5bd76a5u}, {0x0401d10103f648c1u, 0x6c6cf21d1c4d2703u}, 98568},
	// block9-attention@fp8
	{{0xfff29ede2234e5f7u, 0xd3693c2590d4eca5u}, {0x6580cd05fe82b486u, 0x0ba9173edd754594u}, 328208},
	// block15-attention@fp8
	{{0x19dd262ff3c8fe07u, 0xa555693211e9bebbu}, {0xb639c6f675c94f60u, 0x3a447121ff40c5ecu}, 1180704},
	// block23-attention@fp8
	{{0xfece94947600d068u, 0x9710f7fa37c3e3afu}, {0xccbee78cf4a6dcdcu, 0x4c647e24a6230c8fu}, 4458560},
	// block5-attention@fp8-diag
	{{0xb1f913d69e82714au, 0x14a1a942cad0e494u}, {0x1c6a5520eb295128u, 0x6ce5eb3f1f021461u}, 104712},
	// block9-attention@fp8-diag
	{{0xa9cb0085eff87e4eu, 0x727ff631eb718281u}, {0x847d07836171ce95u, 0xab8c6e4040fbd2dbu}, 340496},
	// block15-attention@fp8-diag
	{{0x20b29c9771b0099fu, 0x4e2c849375b48526u}, {0x340e4ab15147a1d3u, 0xa92c121efd8f8e39u}, 1205280},
	// block15-ffn@ffn-frag
	{{0x89a11fdc731aa248u, 0x6f74eb34c3073162u}, {0xcbe47d666c4b2e39u, 0x76ce4652ac4cf123u}, 2360320},
	// block15-attention@qkv-frag-only
	{{0xa062a4dff3ef2a8fu, 0x14da9fbfd3e4c2ceu}, {0xac1616427285d24du, 0x72d0e72b11a26c0du}, 1180704},
	// block23-ffwd@split-mix-f16
	{{0xdbfe57bb11476621u, 0xba8c39dd55012b64u}, {0x7fa9b583bf969105u, 0x9286e10155be67ebu}, 2097152},
	// block23-ffwd-projection@proj-frag
	{{0x8f2dfb62181d5365u, 0x94d69f88ef2e6870u}, {0x199bac3494f9b773u, 0xe477efb58dcf740eu}, 1050624},
	// block23-attention@qkv-frag
	{{0x918bb1b563e89ccdu, 0xea64b9ed5bd7e041u}, {0xf4215534654bb99au, 0x37f527ae87d7b5e7u}, 4458560},
	// block31-expand@vit-frag
	{{0x4de4dced765ff09cu, 0xcb4ba2c439d4f459u}, {0x5deace420868adc7u, 0xd581d96e67fd667bu}, 16777216},
	// block31-contract@vit-frag
	{{0x7b4d69dba00a5273u, 0x44919f4bdd291084u}, {0x00c0d7b6991ed6dbu, 0xbb91b9be9a17cc2cu}, 16781312},
	// block31-qkv@qkv-f16-frag
	{{0xa160e16e92d8faa3u, 0xc3b81417b891200eu}, {0x35f21dcef561e10du, 0xfd8bbc759959fcc5u}, 6291584},
	// block31-projection@vit-proj-frag
	{{0x3b330bfa99861feau, 0xe3ef5a640ed5437au}, {0xecd2639c0e0f7214u, 0x857f95f9adfd3e2au}, 4198400},
	// decoder39-weights@decoder-f16r
	{{0x9b655c5bfb876328u, 0x3620c5db9c6c6c3au}, {0xae870dfdb135aa0du, 0x28198f8d7719ed93u}, 2099200},
	// block48-weights@decoder-f16r
	{{0xd6bcd09fbd07bdbdu, 0x002e28316818f9d8u}, {0x5b96caf0dd42656fu, 0xb060dd7304976f47u}, 525312},
	// block56-weights@decoder-f16r
	{{0xf3df075dbc0d50a4u, 0xf4234956f3ed6878u}, {0xa8b1e102042213f0u, 0xf674de6a2dfc46ddu}, 131584},
	// block62-weights@decoder-f16r
	{{0x573454f7c479b949u, 0x89f171abc31b6c86u}, {0x341334197e62fb3bu, 0x31b1433c8b961744u}, 33024},
	// block66-weights@decoder-f16r
	{{0x6c9cf9a25c850c5au, 0xff1429a00531a4acu}, {0xe7d56e119e9ebc18u, 0x4b957920d1f12634u}, 8320},
};

/** @brief A float's bits. */
static uint32_t
bits (float value)
{
	uint32_t b;
	memcpy(&b, &value, sizeof b);
	return b;
}

/** @brief The float of some bits. */
static float
from_bits (uint32_t b)
{
	float value;
	memcpy(&value, &b, sizeof value);
	return value;
}

/** @brief The FNV-1a 64 of a run of values. */
struct hash {
	uint64_t value; //!< The hash so far.
};

/** @brief Adds a value's bytes to a hash. */
static void
hash_add (struct hash *h,
          void const  *value,
          size_t       size)
{
	h->value = fnv1a(value, size, h->value);
}

/** @brief Checks the primitives against upstream's digests. */
static void
check_upstream_primitives (void)
{
	struct hash widen = {FNV1A_BASIS};
	for (uint32_t h = 0; h < 65536; ++h) {
		float const wide = hip_weights_widen_half((uint16_t)h);
		hash_add(&widen, &wide, sizeof wide);
	}
	struct hash fp8 = {FNV1A_BASIS}, half = {FNV1A_BASIS}, rounded = {FNV1A_BASIS};
	struct hash saturated = {FNV1A_BASIS}, piece = {FNV1A_BASIS};
	for (uint32_t j = 0; j < SAMPLE_COUNT; ++j) {
		float const x = from_bits(sample(j));
		// A flaw hashes as 256 or 65536 plus upstream's number for it, HIP_FLAW_NONE's less.
		uint8_t code;
		enum hip_flaw flaw = hip_weights_exact_fp8(x, &code);
		uint16_t const fp8_entry = flaw ? (uint16_t)(256 + flaw - 1) : code;
		hash_add(&fp8, &fp8_entry, sizeof fp8_entry);
		uint16_t h;
		flaw = hip_weights_exact_half(x, &h);
		uint32_t const half_entry = flaw ? 65536 + (uint32_t)flaw - 1 : h;
		hash_add(&half, &half_entry, sizeof half_entry);
		uint16_t const r = hip_weights_round_half(x);
		hash_add(&rounded, &r, sizeof r);
		float const s = hip_weights_saturate_fp8(x);
		hash_add(&saturated, &s, sizeof s);
		float const p = hip_weights_scale_piece(x);
		hash_add(&piece, &p, sizeof p);
	}
	expect(widen.value == upstream_primitives.widen_half, "widen_half differs from upstream's Half");
	expect(fp8.value == upstream_primitives.exact_fp8, "exact_fp8 differs from upstream's ExactWeightFp8");
	expect(half.value == upstream_primitives.exact_half, "exact_half differs from upstream's ExactWeightHalf");
	expect(rounded.value == upstream_primitives.round_half, "round_half differs from upstream's RoundWeightHalf");
	expect(saturated.value == upstream_primitives.saturate_fp8, "saturate_fp8 differs from upstream's HostF");
	expect(piece.value == upstream_primitives.scale_piece, "scale_piece differs from upstream's HostScalePiece");
}

/** @brief A value and why it has no exact encoding. */
struct rejected {
	float         value; //!< The value.
	enum hip_flaw flaw;  //!< Why.
};

/** @brief Checks the exact E4M3 encoding. */
static void
check_fp8 (void)
{
	for (unsigned code = 0; code < 256; ++code) {
		if ((code & 0x7f) == 0x7f)
			continue; // NaN
		uint8_t packed;
		enum hip_flaw const flaw = hip_weights_exact_fp8(fp8_value(code), &packed);
		if (!expect(!flaw && packed == code, "E4M3 code %#x does not round-trip", code))
			break;
	}
	struct rejected const rejected[] = {
		{NAN, HIP_FLAW_FP8_NONFINITE},
		{-INFINITY, HIP_FLAW_FP8_NONFINITE},
		{480.f, HIP_FLAW_FP8_INEXACT}, // Exponent 15, mantissa 7: NaN's pattern.
		{512.f, HIP_FLAW_FP8_INEXACT},
		{1.0625f, HIP_FLAW_FP8_INEXACT},
		{0x1p-10f, HIP_FLAW_FP8_SUBNORMAL},
		{0x1.8p-9f, HIP_FLAW_FP8_SUBNORMAL},
		{-0x1.2p-7f, HIP_FLAW_FP8_SUBNORMAL},
	};
	for (size_t i = 0; i < sizeof rejected / sizeof *rejected; ++i) {
		uint8_t packed;
		expect(hip_weights_exact_fp8(rejected[i].value, &packed) == rejected[i].flaw,
		       "exact_fp8(%a) does not fail as %s", (double)rejected[i].value,
		       HIP_WEIGHTS_FLAW_WORDS[rejected[i].flaw]);
	}
}

/** @brief Whether a value rounds to binary16 as a conversion to _Float16 does. */
static bool
rounds_as_float16 (float x)
{
	_Float16 const h = (_Float16)x;
	uint16_t b;
	memcpy(&b, &h, sizeof b);
	return hip_weights_round_half(x) == b;
}

/** @brief Checks the binary16 encodings. */
static void
check_halves (void)
{
	for (uint32_t h = 0; h < 65536; ++h) {
		float const wide = hip_weights_widen_half((uint16_t)h);
		uint16_t packed;
		enum hip_flaw const flaw = hip_weights_exact_half(wide, &packed);
		if ((h & 0x7c00) == 0x7c00) {
			// Infinity and NaN, payload kept: a signalling NaN stays one.
			if (!expect(bits(wide) == ((h & 0x8000u) << 16 | 0x7f800000 | (h & 1023) << 13) &&
			            flaw == HIP_FLAW_HALF_NONFINITE,
			            "binary16 %#x does not widen to its f32 NaN or infinity", (unsigned)h))
				break;
			continue;
		}
		if (!expect(bits(wide) == bits(half_value(h)) && !flaw && packed == h,
		            "binary16 %#x does not round-trip", (unsigned)h))
			break;
	}
	struct rejected const rejected[] = {
		{INFINITY, HIP_FLAW_HALF_NONFINITE},
		{65536.f, HIP_FLAW_HALF_OVERFLOW},
		{-0x1p20f, HIP_FLAW_HALF_OVERFLOW},
		{1 + 0x1p-11f, HIP_FLAW_HALF_INEXACT},
		{0x1p-25f, HIP_FLAW_HALF_SUBNORMAL},
		{0x1.8p-24f, HIP_FLAW_HALF_SUBNORMAL},
	};
	for (size_t i = 0; i < sizeof rejected / sizeof *rejected; ++i) {
		uint16_t packed;
		expect(hip_weights_exact_half(rejected[i].value, &packed) == rejected[i].flaw,
		       "exact_half(%a) does not fail as %s", (double)rejected[i].value,
		       HIP_WEIGHTS_FLAW_WORDS[rejected[i].flaw]);
	}
	// Every value but NaN rounds as a conversion to _Float16 does.
	float const edges[] = {
		2049.f, 2051.f, 65504.f, 65519.f, 65520.f, -0.f, 0x1p-25f, 0x1.000002p-25f, 0x1.8p-24f,
		0x1p-24f, 0x1.fffp-15f, 0x1.fff8p-15f, -INFINITY, 0x1p-149f,
	};
	for (size_t i = 0; i < sizeof edges / sizeof *edges; ++i)
		expect(rounds_as_float16(edges[i]), "round_half(%a) differs from _Float16", (double)edges[i]);
	for (uint32_t j = 0; j < SAMPLE_COUNT; ++j) {
		float const x = from_bits(sample(j));
		if (!isnan(x) && !expect(rounds_as_float16(x), "round_half(%a) differs from _Float16", (double)x))
			break;
	}
	expect(hip_weights_round_half(-NAN) == 0xfe00 && hip_weights_round_half(from_bits(0x7f800001)) == 0x7e00,
	       "round_half does not make NaN 0x7e00 with its sign");
}

/** @brief Checks the saturating E4M3 rounding and the scale pieces. */
static void
check_saturation (void)
{
	struct {
		float value, saturated, piece;
	} const table[] = {
		{0.f, 0.f, 0.f},
		{-0.f, 0.f, -0.f},
		{1000.f, 448.f, 448.f},
		{-INFINITY, -448.f, -448.f},
		{NAN, 448.f, 448.f},
		{464.f, 448.f, 448.f},
		{1.0625f, 1.f, 1.f},
		{1.1875f, 1.25f, 1.25f},
		{1.0703125f, 1.125f, 1.125f},
		{0.001f, 0x1p-9f, 0x1p-9f},
		{0x1.8p-9f, 0x1p-8f, 0x1p-8f},
		{0x1p-10f, 0.f, 0.f},
		{-0x1p-11f, -0.f, -0.f},
		{0.0155f, 0x1p-6f, 7 * 0x1p-9f}, // The piece stays subnormal.
		{-0.0155f, -0x1p-6f, -7 * 0x1p-9f},
	};
	for (size_t i = 0; i < sizeof table / sizeof *table; ++i) {
		float const saturated = hip_weights_saturate_fp8(table[i].value);
		float const piece = hip_weights_scale_piece(table[i].value);
		expect(bits(saturated) == bits(table[i].saturated), "saturate_fp8(%a) is %a, not %a",
		       (double)table[i].value, (double)saturated, (double)table[i].saturated);
		expect(bits(piece) == bits(table[i].piece), "scale_piece(%a) is %a, not %a", (double)table[i].value,
		       (double)piece, (double)table[i].piece);
	}
}

/** @brief Checks the fragment tiles' permutation. */
static void
check_fragment_tiles (void)
{
	struct {
		size_t rows, columns;
	} const shapes[] = {{16, 32}, {64, 32}, {32, 128}, {48, 64}};
	struct random r = {1};
	for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
		size_t const rows = shapes[s].rows, columns = shapes[s].columns, bytes = rows * columns;
		size_t const count = bytes / 4 + 3;
		float *values = malloc(count * sizeof *values);
		float *before = malloc(count * sizeof *before);
		unsigned *hits = calloc(bytes, sizeof *hits);
		allocated(values && before && hits);
		for (size_t i = 0; i < count; ++i)
			values[i] = from_bits((uint32_t)random_next(&r));
		memcpy(before, values, count * sizeof *values);
		struct error e;
		expect(!hip_weights_fragment_tiles(values, 1, rows, columns, &e), "fragment_tiles of %zux%zu: %s",
		       rows, columns, e.what);
		uint8_t const *const in = (uint8_t const *)(before + 1);
		uint8_t const *const out = (uint8_t const *)(values + 1);
		bool same = true;
		for (size_t n = 0; n < rows; ++n) {
			for (size_t k = 0; k < columns; ++k) {
				// FragmentPackedMatrix's permutation, as upstream writes it.
				size_t const at = ((n / 16) * (columns / 32) + k / 32) * 512 +
				                  (((k % 32) / 16 * 2 + (k % 16) / 8) * 16 + n % 16) * 8 + k % 8;
				same &= at < bytes && out[at] == in[n * columns + k];
				if (at < bytes)
					++hits[at];
			}
		}
		bool bijection = true;
		for (size_t i = 0; i < bytes; ++i)
			bijection &= hits[i] == 1;
		expect(same && bijection, "fragment_tiles of %zux%zu is not upstream's permutation", rows, columns);
		expect(bits(values[0]) == bits(before[0]) && !memcmp(&values[1 + bytes / 4], &before[1 + bytes / 4], 8),
		       "fragment_tiles of %zux%zu writes outside its matrix", rows, columns);
		free(hits);
		hits = nullptr;
		free(before);
		before = nullptr;
		free(values);
		values = nullptr;
	}
}

/** @brief Piece part of a scale, as upstream's diagonal loops chain them. */
static float
piece (float    scale,
       unsigned part)
{
	float remaining = scale, result = 0;
	for (unsigned j = 0; j <= part; ++j) {
		result = hip_weights_scale_piece(remaining);
		remaining -= result;
	}
	return result;
}

/** @brief Whether a C32 FFN weight's diagonal tiles hold its scales' pieces where upstream puts
 *         them: byte (gr*16 + rc)*8 + e of tile (part*2 + ci)*2 + kt is piece part of the scale of
 *         column c = ci*16 + rc where k = kt*16 + gr*8 + e is c, else 0.
 *
 * @param values The weight's floats and its tiles.
 * @return       true if they do.
 */
static bool
c32_diagonals_as_upstream (float const *values)
{
	uint8_t const *const tiles = (uint8_t const *)(values + HIP_WEIGHTS_C32_FFN_FLOATS);
	bool same = true;
	for (unsigned tile = 0; tile < 12; ++tile) {
		unsigned const part = tile / 4, ci = tile / 2 % 2, kt = tile % 2;
		for (unsigned byte = 0; byte < 256; ++byte) {
			unsigned const gr = byte / 128, rc = byte / 8 % 16, i = byte % 8;
			unsigned const c = ci * 16 + rc, k = kt * 16 + gr * 8 + i;
			uint8_t code = 0;
			bool const exact = !hip_weights_exact_fp8(piece(values[8704 + c], part), &code);
			same &= exact && tiles[tile * 512 + byte] == (k == c ? code : 0);
		}
	}
	return same;
}

/** @brief Whether an MH attention weight's diagonal tiles hold its scales' pieces where upstream
 *         puts them: byte (gr*16 + rc)*8 + e of tile part*(c/16) + ct is piece part of the scale of
 *         column ct*16 + rc where k = gr*8 + e is rc, else 0.
 *
 * @param values The weight's floats.
 * @param base   Their number: its tiles follow them.
 * @param scales Where its scales start.
 * @param c      Its channels.
 * @return       true if they do.
 */
static bool
mh_diagonals_as_upstream (float const *values,
                          size_t       base,
                          size_t       scales,
                          unsigned     c)
{
	uint8_t const *const tiles = (uint8_t const *)(values + base);
	bool same = true;
	for (unsigned tile = 0; tile < 3 * (c / 16); ++tile) {
		unsigned const part = tile / (c / 16), ct = tile % (c / 16);
		for (unsigned byte = 0; byte < 256; ++byte) {
			unsigned const gr = byte / 128, rc = byte / 8 % 16, i = byte % 8;
			uint8_t code = 0;
			bool const exact = !hip_weights_exact_fp8(piece(values[scales + ct * 16 + rc], part), &code);
			same &= exact && tiles[tile * 512 + byte] == (gr * 8 + i == rc ? code : 0);
		}
	}
	return same;
}

/** @brief Checks the diagonal tiles of the C32 FFN and MH attention weights. */
static void
check_diagonals (void)
{
	struct random r = {2};
	size_t const c32_floats = HIP_WEIGHTS_C32_FFN_FLOATS + HIP_WEIGHTS_C32_DIAGONAL_FLOATS;
	float *c32 = malloc(HIP_WEIGHTS_C32_FFN_FLOATS * sizeof *c32);
	float *packed = malloc(c32_floats * sizeof *packed);
	allocated(c32 && packed);
	for (size_t i = 0; i < HIP_WEIGHTS_C32_FFN_FLOATS; ++i)
		c32[i] = f32_value(CONTENT_FLOAT, &r);
	memcpy(packed, c32, HIP_WEIGHTS_C32_FFN_FLOATS * sizeof *packed);
	hip_weights_append_c32_diagonals(packed);
	expect(c32_floats == 10272 && !memcmp(packed, c32, HIP_WEIGHTS_C32_FFN_FLOATS * sizeof *c32) &&
	       c32_diagonals_as_upstream(packed),
	       "C32 diagonal tiles differ from upstream's layout");
	free(packed);
	packed = nullptr;
	free(c32);
	c32 = nullptr;

	unsigned const widths[] = {64, 128, 256, 512};
	for (size_t w = 0; w < sizeof widths / sizeof *widths; ++w) {
		unsigned const c = widths[w];
		size_t const heads = c / 32, scales = 4 * (size_t)c * c + heads * 4096 + heads, base = scales + c;
		float *mh = malloc(base * sizeof *mh);
		packed = malloc((base + 24 * (size_t)c) * sizeof *packed);
		allocated(mh && packed);
		for (size_t i = 0; i < base; ++i)
			mh[i] = f32_value(CONTENT_FLOAT, &r);
		memcpy(packed, mh, base * sizeof *packed);
		hip_weights_append_mh_diagonals(packed, base, c);
		expect(!memcmp(packed, mh, base * sizeof *mh) && mh_diagonals_as_upstream(packed, base, scales, c),
		       "MH diagonal tiles at c %u differ from upstream's layout", c);
		free(packed);
		packed = nullptr;
		free(mh);
		mh = nullptr;
	}
}

/** @brief Checks the grouped contraction's check. */
static void
check_grouped_contract (void)
{
	struct hip_weight_spec const specs[] = {
		HIP_WEIGHT_SPEC("block5-ffn", HIP_RECIPE_MH_FFN),
		HIP_WEIGHT_SPEC("block9-ffn", HIP_RECIPE_MH_FFN),
		HIP_WEIGHT_SPEC("block15-ffn", HIP_RECIPE_FFN_FRAG),
	};
	for (size_t s = 0; s < sizeof specs / sizeof *specs; ++s) {
		unsigned const c = hip_weights_channels(&specs[s]);
		size_t const cc = (size_t)c * c, count = 9 * cc + c;
		float *values = malloc(count * sizeof *values);
		allocated(values);
		struct random r = {c};
		struct shape const w = shape(&specs[s]);
		for (size_t i = 0; i < count; ++i)
			values[i] = f32_value(content(&w, i), &r);
		// -0 counts as zero; any other value outside the groups does not.
		values[4 * cc + 128] = -0.f;
		struct error e;
		expect(!hip_weights_check_grouped_contract(values, c, &e), "grouped contraction at c %u rejected", c);
		float const bad[] = {0x1p-149f, NAN};
		for (size_t b = 0; b < sizeof bad / sizeof *bad; ++b) {
			size_t const at = 4 * cc + (c - 1) * 4 * (size_t)c; // Row c-1, k 0.
			float const kept = values[at];
			values[at] = bad[b];
			expect_error(hip_weights_check_grouped_contract(values, c, &e), &e,
			             "element %zu: nonzero outside grouped contraction", at);
			values[at] = kept;
		}
		free(values);
		values = nullptr;
	}
}

/** @brief The most files a scratch directory holds. */
#define SCRATCH_FILES 16

/** @brief A directory for the test's files, removed with them. */
struct scratch {
	char  *directory;             //!< Its path.
	char  *files[SCRATCH_FILES];  //!< The files written since the last scratch_clear().
	size_t length;                //!< The path's length.
	size_t count;                 //!< The number of files.
};

/** @brief Removes the files written since the last call. */
static void
scratch_clear (struct scratch *s)
{
	for (size_t i = 0; i < s->count; ++i) {
		unlink(s->files[i]);
		free(s->files[i]);
		s->files[i] = nullptr;
	}
	s->count = 0;
}

/** @brief Writes a file in a scratch directory.
 *
 * @param s    The scratch.
 * @param file The file's name.
 * @param data Its bytes.
 * @param size Their number.
 */
static void
scratch_write (struct scratch *s,
               char const     *file,
               void const     *data,
               size_t          size)
{
	char *const path = files_join(s->directory, s->length, file, strlen(file), nullptr);
	allocated(path && s->count < SCRATCH_FILES);
	struct error e;
	expect(!files_write(path, data, size, &e), "cannot write %s: %s", path, e.what);
	s->files[s->count++] = path;
}

/** @brief Writes a file of zero bytes in a scratch directory.
 *
 * @param s    The scratch.
 * @param file The file's name.
 * @param size Its bytes.
 */
static void
scratch_zeros (struct scratch *s,
               char const     *file,
               size_t          size)
{
	void *zeros = calloc(size ? size : 1, 1);
	allocated(zeros);
	scratch_write(s, file, zeros, size);
	free(zeros);
	zeros = nullptr;
}

/** @brief Checks every recipe against upstream's digests.
 *
 * @param s The scratch.
 */
static void
check_recipes (struct scratch *s)
{
	static_assert(sizeof upstream_cases / sizeof *upstream_cases == sizeof cases / sizeof *cases,
	              "a digest per case");
	for (size_t d = 0; d < sizeof upstream_cases / sizeof *upstream_cases; ++d) {
		struct case_digest const *const digest = &upstream_cases[d];
		struct hip_weight_spec const *const spec = &cases[d];
		char weight[KEY_BYTES];
		size_t const length = key(spec, weight);
		struct shape const w = shape(spec);
		size_t const elements = hip_weights_file_elements(spec), packed = hip_weights_packed_bytes(spec);
		for (uint8_t half = 0; half < 2; ++half) {
			char name[KEY_BYTES + 16];
			format(name, sizeof name, "%s from %s", weight, half ? "binary16" : "f32");
			size_t size;
			uint8_t *input = synthetic_file(&w, weight, length, elements, half, &size);
			char file[HIP_WEIGHTS_STEM_BYTES + 4];
			format(file, sizeof file, "%s%s", spec->stem, half ? ".f16" : ".f32");
			scratch_write(s, file, input, size);
			struct error e;
			struct hip_weight_file weights;
			enum error_code const code = hip_weights_load(&weights, s->directory, s->length, spec, &e);
			scratch_clear(s);
			bool const same_input = fnv1a(input, size, FNV1A_BASIS) == digest->input[half];
			free(input);
			input = nullptr;
			if (expect(same_input, "%s: synthetic input differs", name) &&
			    expect(!code, "%s: %s", name, e.what)) {
				size_t const bytes = weights.count * 4;
				expect(bytes == digest->bytes && bytes == packed,
				       "%s: %zu bytes, upstream uploads %zu", name, bytes, digest->bytes);
				expect(fnv1a(weights.values, bytes, FNV1A_BASIS) == digest->packed[half],
				       "%s: packed image differs from upstream's", name);
			}
			hip_weight_file_fini(&weights);
		}
	}
}

/** @brief Loads a weight from a scratch directory and frees it.
 *
 * @param s    The scratch.
 * @param spec The weight.
 * @param e    Receives the words for what stopped it.
 * @return     What hip_weights_load() returned.
 */
static enum error_code
load_code (struct scratch const         *s,
           struct hip_weight_spec const *spec,
           struct error                 *e)
{
	struct hip_weight_file file;
	enum error_code const code = hip_weights_load(&file, s->directory, s->length, spec, e);
	hip_weight_file_fini(&file);
	return code;
}

/** @brief Packs a synthetic file named PATH of a weight, with one value replaced.
 *
 * @param spec    The weight.
 * @param element The value's index.
 * @param value   The value.
 * @param e       Receives the words for what stopped it.
 * @return        What hip_weights_pack() returned.
 */
static enum error_code
pack_code (struct hip_weight_spec const *spec,
           size_t                        element,
           float                         value,
           struct error                 *e)
{
	size_t const count = hip_weights_file_elements(spec);
	struct hip_weight_file file = {
		.path = strdup("PATH"),
		.values = malloc(count * sizeof *file.values),
		.count = count,
		.capacity = count,
	};
	allocated(file.path && file.values);
	struct random r = {3};
	struct shape const w = shape(spec);
	for (size_t i = 0; i < count; ++i)
		file.values[i] = f32_value(content(&w, i), &r);
	file.values[element] = value;
	enum error_code const code = hip_weights_pack(spec, &file, e);
	hip_weight_file_fini(&file);
	return code;
}

/** @brief A weight of a literal stem, as a pointer to a compound literal. */
#define SPEC(stem, recipe) (&(struct hip_weight_spec)HIP_WEIGHT_SPEC(stem, recipe))

/** @brief Checks the reader's and the packers' errors.
 *
 * @param s The scratch.
 */
static void
check_errors (struct scratch *s)
{
	char const *const dir = s->directory;
	struct hip_weight_spec const ffn = HIP_WEIGHT_SPEC("block5-ffn", HIP_RECIPE_MH_FFN);
	struct error e;
	expect_error(load_code(s, SPEC("block5-bogus", HIP_RECIPE_DS_FRAG), &e), &e,
	             "unknown weight block5-bogus");
	expect_error(load_code(s, SPEC("block10-ffn", HIP_RECIPE_SPLIT_MIX_F16), &e), &e,
	             "no split-mix-f16 packing of block10-ffn");
	expect_error(load_code(s, SPEC("block66-ffn", HIP_RECIPE_MH_FFN), &e), &e,
	             "no fp8-g128 packing of block66-ffn");
	expect_error(load_code(s, &ffn, &e), &e, "missing weight %s/block5-ffn (neither .f32 nor .f16)", dir);
	// A file that does not hold exactly its stem's values is rejected with both sizes.
	size_t const elements = hip_weights_file_elements(&ffn);
	scratch_write(s, "block5-ffn.f16", "", 0);
	expect_error(load_code(s, &ffn, &e), &e, "%s/block5-ffn.f16: 0 bytes, expected %zu", dir, 2 * elements);
	scratch_write(s, "block5-ffn.f16", "abc", 3);
	expect_error(load_code(s, &ffn, &e), &e, "%s/block5-ffn.f16: 3 bytes, expected %zu", dir, 2 * elements);
	scratch_zeros(s, "block5-ffn.f16", 2 * elements);
	expect(!load_code(s, &ffn, &e), "a zero block5-ffn.f16 rejected");
	// Whenever the f32 file exists, it is the one read.
	scratch_zeros(s, "block5-ffn.f32", 2 * elements);
	expect_error(load_code(s, &ffn, &e), &e, "%s/block5-ffn.f32: %zu bytes, expected %zu", dir, 2 * elements,
	             4 * elements);
	scratch_zeros(s, "block5-ffn.f32", 4 * elements);
	char *f32 = support_format(nullptr, "%s/block5-ffn.f32", dir);
	char *f16 = support_format(nullptr, "%s/block5-ffn.f16", dir);
	allocated(f32 && f16);
	struct hip_weight_file full;
	enum error_code const code = hip_weights_load(&full, dir, s->length, &ffn, &e);
	expect(!code && !strcmp(full.path, f32), "a zero block5-ffn.f32 not read");
	hip_weight_file_fini(&full);
	// An f32 file that does not open is an error, not a reason to read the f16 one. The check
	// needs a user whom the file's mode keeps out, not root.
	expect(!chmod(f32, 0), "cannot chmod %s", f32);
	if (access(f32, R_OK)) {
		expect_error(load_code(s, &ffn, &e), &e, "open %s: %s", f32, strerror(EACCES));
		// With no f32 file, an f16 file that does not open is an open error, not a missing
		// weight.
		expect(!unlink(f32), "cannot remove %s", f32);
		expect(!chmod(f16, 0), "cannot chmod %s", f16);
		expect_error(load_code(s, &ffn, &e), &e, "open %s: %s", f16, strerror(EACCES));
	}
	free(f16);
	f16 = nullptr;
	free(f32);
	f32 = nullptr;
	scratch_zeros(s, "post70-head.f32", 95 * 4);
	expect_error(load_code(s, SPEC("post70-head", HIP_RECIPE_RAW), &e), &e,
	             "%s/post70-head.f32: 380 bytes, expected 384", dir);
	// The ds-cast, ds-frag and decoder files as well, which upstream's loaders read at any size of
	// at least 2c² values.
	struct hip_weight_spec const sized[] = {
		HIP_WEIGHT_SPEC("block4-ds", HIP_RECIPE_DS_CAST),
		HIP_WEIGHT_SPEC("block8-ds", HIP_RECIPE_DS_FRAG),
		HIP_WEIGHT_SPEC("block62-weights", HIP_RECIPE_DECODER_F16R),
	};
	for (size_t i = 0; i < sizeof sized / sizeof *sized; ++i) {
		size_t const n = hip_weights_file_elements(&sized[i]);
		char name[HIP_WEIGHTS_STEM_BYTES + 4];
		format(name, sizeof name, "%s.f16", sized[i].stem);
		size_t const counts[] = {n - 1, n + 1};
		for (size_t j = 0; j < 2; ++j) {
			scratch_zeros(s, name, 2 * counts[j]);
			expect_error(load_code(s, &sized[i], &e), &e, "%s/%s: %zu bytes, expected %zu", dir, name,
			             2 * counts[j], 2 * n);
		}
	}
	// Encoding errors name the file read and the element.
	struct hip_weight_spec const attention = HIP_WEIGHT_SPEC("block0-attention", HIP_RECIPE_C32);
	size_t const size = 2 * hip_weights_file_elements(&attention);
	uint8_t *input = calloc(size, 1);
	allocated(input);
	input[2 * 7] = 0x55; // 0x3555, about a third in binary16, which E4M3 cannot hold.
	input[2 * 7 + 1] = 0x35;
	scratch_write(s, "block0-attention.f16", input, size);
	free(input);
	input = nullptr;
	struct hip_weight_file file;
	enum error_code const load = hip_weights_load(&file, dir, s->length, &attention, &e);
	hip_weight_file_fini(&file);
	expect_error(load, &e, "%s/block0-attention.f16: element 7: matrix weight not exact finite FP8", dir);
	scratch_clear(s);

	// So do the other encodings' errors, here of a file named PATH.
	expect_error(pack_code(&attention, 7, 0.3f, &e), &e, "PATH: element 7: matrix weight not exact finite FP8");
	expect_error(pack_code(SPEC("block0-ffn", HIP_RECIPE_C32), 600, INFINITY, &e), &e,
	             "PATH: element 600: nonfinite FP8 matrix weight");
	expect_error(pack_code(SPEC("block5-attention", HIP_RECIPE_MH_ATTENTION), 12288,
	                       0x1p-12f, &e), &e,
	             "PATH: element 12288: matrix weight not exact FP8 subnormal");
	expect_error(pack_code(&ffn, 4 * 4096 + 128, 1.f, &e), &e,
	             "PATH: element 16512: nonzero outside grouped contraction");
	expect_error(pack_code(SPEC("block23-ffwd", HIP_RECIPE_SPLIT_MIX_F16), 262149, 1.f / 3,
	                       &e), &e,
	             "PATH: element 262149: weight not exact half");
	expect_error(pack_code(SPEC("block31-qkv", HIP_RECIPE_QKV_F16_FRAG), 2097153, 0x1p16f,
	                       &e), &e,
	             "PATH: element 2097153: half weight overflow");
	expect_error(pack_code(SPEC("block22-ds", HIP_RECIPE_DS_FRAG), 70000, 0x1p-26f, &e), &e,
	             "PATH: element 70000: weight not exact half subnormal");
	expect_error(pack_code(SPEC("block31-projection", HIP_RECIPE_VIT_PROJ_FRAG), 1048575,
	                       NAN, &e), &e,
	             "PATH: element 1048575: nonfinite FP8 matrix weight");
	// Values outside the regions a recipe encodes are kept, whatever they are.
	expect(!pack_code(SPEC("decoder39-weights", HIP_RECIPE_DECODER_F16R), 524799, NAN, &e),
	       "a NaN scale rejected");
}

#undef SPEC

/** @brief The weights dlsslopd uploads, in upload order: those of the plan at the first tier.
 *         hip-plan checks that the plan at every tier lists the same.
 *
 * @param performance Whether with --performance.
 * @param plan        Receives the plan.
 * @param e           Receives the words for what stopped it.
 * @return            What hip_plan_init() returned.
 */
static enum error_code
production_weights (bool             performance,
                    struct hip_plan *plan,
                    struct error    *e)
{
	return hip_plan_init(plan, kNativeTiers[0].width, kNativeTiers[0].networkHeight, performance, e);
}

/** @brief Checks the weights dlsslopd uploads against the totals of upstream's traces. */
static void
check_production_list (void)
{
	for (uint8_t performance = 0; performance < 2; ++performance) {
		struct error e;
		struct hip_plan plan;
		if (!expect(!production_weights(performance, &plan, &e), "%s", e.what))
			continue;
		char (*keys)[KEY_BYTES] = malloc(plan.weight_count * sizeof *keys);
		allocated(keys);
		size_t bytes = 0;
		for (size_t w = 0; w < plan.weight_count; ++w) {
			struct hip_weight_spec const *const spec = &plan.weights[w];
			bytes += hip_weights_packed_bytes(spec);
			expect(hip_weights_file_elements(spec) != 0, "%s: unknown stem", spec->stem);
			key(spec, keys[w]);
			bool twice = false;
			for (size_t v = 0; v < w && !twice; ++v)
				twice = !strcmp(keys[w], keys[v]);
			expect(!twice, "%s listed twice", keys[w]);
		}
		free(keys);
		keys = nullptr;
		// The weights upstream uploads that a launch reads: its traces show these totals.
		size_t const count = performance ? 224 : 236, total = performance ? 608027816 : 644222504;
		expect(plan.weight_count == count && bytes == total,
		       "%zu weights%s of %zu bytes; upstream's launches read %zu of %zu", plan.weight_count,
		       performance ? " with --performance" : "", bytes, count, total);
		hip_plan_fini(&plan);
	}
}

/** @brief Packs weights from a model: prints the bytes, FNV-1a 64 and name of each.
 *
 * @param assets The model's directory.
 * @param plan   The plan whose weights are packed.
 * @return       0, or 1 when one fails.
 */
static int
pack_model (char const            *assets,
            struct hip_plan const *plan)
{
	size_t const length = strlen(assets);
	for (size_t w = 0; w < plan->weight_count; ++w) {
		struct hip_weight_spec const *const spec = &plan->weights[w];
		struct error e;
		struct hip_weight_file file;
		if (hip_weights_load(&file, assets, length, spec, &e)) {
			fprintf(stderr, "hip-weights-test: %s\n", e.what);
			return 1;
		}
		size_t const bytes = file.count * 4;
		char name[KEY_BYTES];
		key(spec, name);
		printf("%zu\t%016" PRIx64 "\t%s\n", bytes, fnv1a(file.values, bytes, FNV1A_BASIS), name);
		hip_weight_file_fini(&file);
	}
	return 0;
}

/** @brief The usage that --help prints. */
static char const USAGE[] =
	"Usage: hip-weights-test [OPTION]...\n"
	"Checks the HIP network's weight packing against upstream's packers. No GPU or model needed.\n"
	" -m, --model DIR    Instead, pack every weight dlsslopd uploads from the model in DIR\n"
	"                    and print each image's bytes, FNV-1a 64 and name, in upload order\n"
	"                    (default: unset)\n"
	" -l, --list         Instead, print the stem, file elements, packed bytes and name of\n"
	"                    every weight dlsslopd uploads, in upload order (default: off)\n"
	" -p, --performance  With --model or --list, leave out the blocks that dlsslopd\n"
	"                    --performance skips (default: off)\n"
	" -h, --help         Show help (default: off)\n";

int
main (int    argc,
      char **argv)
{
	char const *model = "";
	bool list = false;
	bool performance = false;
	static struct option const options[] = {
		{"model",       required_argument, nullptr, 'm'},
		{"list",        no_argument,       nullptr, 'l'},
		{"performance", no_argument,       nullptr, 'p'},
		{"help",        no_argument,       nullptr, 'h'},
		{},
	};
	for (int code; (code = getopt_long(argc, argv, "+m:lph", options, nullptr)) != -1;) {
		switch (code) {
		case 'm':
			model = optarg;
			break;
		case 'l':
			list = true;
			break;
		case 'p':
			performance = true;
			break;
		case 'h':
			fputs(USAGE, stdout);
			return support_written("hip-weights test", 0);
		default:
			return 2;
		}
	}
	if (optind != argc || (list && *model))
		return 2;
	if (list || *model) {
		struct error e;
		struct hip_plan plan;
		if (production_weights(performance, &plan, &e)) {
			fprintf(stderr, "hip-weights-test: %s\n", e.what);
			return 1;
		}
		int status = 0;
		if (*model) {
			status = pack_model(model, &plan);
		} else {
			for (size_t w = 0; w < plan.weight_count; ++w) {
				struct hip_weight_spec const *const spec = &plan.weights[w];
				char name[KEY_BYTES];
				key(spec, name);
				printf("%s\t%zu\t%zu\t%s\n", spec->stem, hip_weights_file_elements(spec),
				       hip_weights_packed_bytes(spec), name);
			}
		}
		hip_plan_fini(&plan);
		return support_written("hip-weights test", status);
	}
	struct scratch s = {.directory = support_temp_dir("/tmp", "dlsslop-amd-hip-weights", nullptr)};
	if (!expect(s.directory, "cannot create a directory in /tmp"))
		return 1;
	s.length = strlen(s.directory);
	check_upstream_primitives();
	check_fp8();
	check_halves();
	check_saturation();
	check_fragment_tiles();
	check_diagonals();
	check_grouped_contract();
	check_recipes(&s);
	check_errors(&s);
	check_production_list();
	scratch_clear(&s);
	expect(support_remove_tree(s.directory), "cannot remove %s", s.directory);
	free(s.directory);
	s.directory = nullptr;
	if (!failures)
		puts("hip-weights test: every check passed");
	return support_written("hip-weights test", failures ? 1 : 0);
}
