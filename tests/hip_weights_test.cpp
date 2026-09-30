// SPDX-License-Identifier: MIT
// Host test of the HIP network's weight packing, without a GPU or a model. The
// encodings are checked against their definitions, and they and every recipe
// against digests that the pinned upstream packers (packed_weights.h and the
// weight loaders of hip_reference_network.h at c190831) gave for the same
// inputs. --model packs a real model instead, for comparison with the uploads
// a trace of upstream's network records.
#include "../backend/files.h"
#include "../backend/hip_plan.h"
#include "../external/layer/common/shm_protocol.h"

#include <getopt.h>
#include <unistd.h>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {
using dlsslop::hip::Flaw;
using dlsslop::hip::Recipe;
using dlsslop::hip::WeightFile;
using dlsslop::hip::WeightSpec;
namespace hip = dlsslop::hip;

uint64_t fnv1a(const void* data, size_t bytes, uint64_t hash = 0xcbf29ce484222325u)
{
    for (auto* at = static_cast<const unsigned char*>(data); bytes--; ++at) hash = (hash ^ *at) * 0x100000001b3u;
    return hash;
}
// STEM@SUFFIX, upstream's key without the extension; STEM alone for raw weights.
std::string key(const WeightSpec& spec)
{
    const char* suffix = dlsslop::hip::kRecipeSuffix[size_t(spec.recipe)];
    return *suffix ? std::string(spec.stem) + "@" + suffix : std::string(spec.stem);
}

// splitmix64.
struct Random {
    uint64_t state;
    uint64_t next()
    {
        uint64_t z = state += 0x9e3779b97f4a7c15u;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
        return z ^ (z >> 31);
    }
};

// What a synthetic weight holds at an element: an exact E4M3 value, an exact
// binary16 value, anything the rounding packers see (ties, subnormals,
// overflow, NaN), a scale or bias (kept as read, or split into E4M3 pieces),
// or a zero outside the grouped contraction.
enum Content : uint8_t { kFp8, kHalf, kRounded, kFloat, kZero };

Content content(const WeightSpec& spec, size_t i)
{
    const size_t c = hip::channels(spec), cc = c * c;
    switch (spec.recipe) {
    case Recipe::kRaw:
    case Recipe::kDsCast: return kRounded;
    case Recipe::kC32:
        if (std::string_view(spec.stem).ends_with("attention")) return i < 4096 ? kFp8 : kFloat;
        return i >= 512 && i < 8704 ? kFp8 : kFloat;
    case Recipe::kDsFrag: return kHalf;
    case Recipe::kMhFfn:
    case Recipe::kFfnFrag:
        if (i >= 4 * cc && i < 8 * cc && (i - 4 * cc) % (4 * c) / 128 != (i - 4 * cc) / (4 * c) / 32) return kZero;
        return i < 9 * cc ? kFp8 : kFloat;
    case Recipe::kMhAttention:
    case Recipe::kMhAttentionDiag:
    case Recipe::kQkvFragOnly:
    case Recipe::kQkvFrag: return i < 4 * cc ? kFp8 : kFloat;
    case Recipe::kSplitMixF16: return i < 262144 ? kRounded : i < 393216 ? kHalf : kFp8;
    case Recipe::kProjFrag: return i < 262144 ? kFp8 : kFloat;
    case Recipe::kVitFrag: return i < 4194304 ? kFp8 : kFloat;
    case Recipe::kQkvF16Frag: return i < 3145728 ? kHalf : kFloat;
    case Recipe::kVitProjFrag: return i < 1048576 ? kFp8 : kFloat;
    case Recipe::kDecoderF16r: return i < 2 * cc ? kRounded : kFloat;
    }
    return kFloat;
}

// The value of an E4M3 code other than NaN.
float fp8_value(unsigned code)
{
    const unsigned e = code >> 3 & 15, m = code & 7;
    const float magnitude = e ? std::ldexp(float(8 + m), int(e) - 10) : float(m) / 512;
    return code & 0x80 ? -magnitude : magnitude;
}
// The value of a finite binary16.
float half_value(unsigned h)
{
    const unsigned e = h >> 10 & 31, m = h & 1023;
    const float magnitude = e ? std::ldexp(float(1024 + m), int(e) - 25) : std::ldexp(float(m), -24);
    return h & 0x8000 ? -magnitude : magnitude;
}
// VALUE as binary16; it is zero or a normal binary16 value.
uint16_t half_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const unsigned sign = bits >> 16 & 0x8000;
    return uint16_t(bits & 0x7fffffff ? sign | ((bits >> 23 & 255) - 112) << 10 | (bits >> 13 & 1023) : sign);
}
unsigned fp8_code(uint64_t x) { return (x & 0x7f) == 0x7f ? unsigned(x & 0xff) ^ 1 : unsigned(x & 0xff); }
unsigned finite_half(uint64_t x)
{
    return (x & 0x7c00) == 0x7c00 ? unsigned(x & 0xffff) ^ 0x4000 : unsigned(x & 0xffff);
}

float f32_value(Content content, Random& random)
{
    const uint64_t x = random.next();
    const uint32_t low = uint32_t(x), sign = low & 0x80000000u;
    const unsigned pick = unsigned(x >> 60), draw = unsigned(x >> 32 & 0xffff);
    uint32_t bits = low;
    switch (content) {
    case kFp8: return fp8_value(fp8_code(x));
    case kHalf: return half_value(finite_half(x));
    case kZero: return x & 1 ? -0.f : 0.f;
    case kRounded:
        // Binary16's and E4M3's ranges and a little beyond; one in sixteen
        // is any bit pattern: NaN, infinity, the huge and the tiny.
        if (pick) {
            const unsigned exponent = 100 + draw % 45; // 2^-27 .. 2^17
            uint32_t mantissa = low & 0x7fffff;
            if (pick == 1) mantissa = (mantissa & ~0x1fffu) | 0x1000; // binary16 tie
            if (pick == 2) mantissa = (mantissa & ~0xfffffu) | 0x80000; // E4M3 tie
            if (pick == 3) { // binary16 subnormal tie
                const unsigned e = 102 + draw % 11, n = 126 - e; // 2^-25 .. 2^-15
                mantissa = (((mantissa | 0x800000) & ~((1u << n) - 1)) | 1u << (n - 1)) & 0x7fffff;
                bits = sign | e << 23 | mantissa;
                break;
            }
            bits = sign | exponent << 23 | mantissa;
        }
        break;
    case kFloat:
        // Mostly the magnitudes of scales and biases; some below E4M3's
        // normals, some beyond its range, and a few of any bit pattern.
        if (pick) {
            const unsigned exponent = pick < 4 ? 97 + draw % 24 : pick < 6 ? 135 + draw % 12 : 115 + draw % 16;
            bits = sign | exponent << 23 | (low & 0x7fffff);
        }
        break;
    }
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
uint16_t f16_value(Content content, Random& random)
{
    const uint64_t x = random.next();
    switch (content) {
    case kFp8: return half_bits(fp8_value(fp8_code(x)));
    case kHalf: return uint16_t(finite_half(x));
    case kZero: return x & 1 ? 0x8000 : 0;
    default: return uint16_t(x); // Any binary16: NaN, infinity, subnormals.
    }
}
// A synthetic file for SPEC of ELEMENTS values: f32, or binary16 when HALF.
std::string synthetic_file(const WeightSpec& spec, size_t elements, bool half)
{
    const std::string name = key(spec);
    Random random{fnv1a(name.data(), name.size()) + half};
    std::string bytes(elements << (half ? 1 : 2), '\0');
    for (size_t i = 0; i < elements; ++i) {
        if (half) {
            const uint16_t value = f16_value(content(spec, i), random);
            std::memcpy(&bytes[i * 2], &value, 2);
        } else {
            const float value = f32_value(content(spec, i), random);
            std::memcpy(&bytes[i * 4], &value, 4);
        }
    }
    return bytes;
}

// One weight per recipe and channel count that production packs.
constexpr WeightSpec kCases[] = {
    {"post70-scales", Recipe::kRaw},
    {"post70-head", Recipe::kRaw},
    {"block23-ffwd", Recipe::kRaw},
    {"block0-ffn", Recipe::kC32},
    {"block0-attention", Recipe::kC32},
    {"block4-ds", Recipe::kDsCast},
    {"block8-ds", Recipe::kDsFrag},
    {"block14-ds", Recipe::kDsFrag},
    {"block22-ds", Recipe::kDsFrag},
    {"head-matrix", Recipe::kDsFrag},
    {"block5-ffn", Recipe::kMhFfn},
    {"block9-ffn", Recipe::kMhFfn},
    {"block5-attention", Recipe::kMhAttention},
    {"block9-attention", Recipe::kMhAttention},
    {"block15-attention", Recipe::kMhAttention},
    {"block23-attention", Recipe::kMhAttention},
    {"block5-attention", Recipe::kMhAttentionDiag},
    {"block9-attention", Recipe::kMhAttentionDiag},
    {"block15-attention", Recipe::kMhAttentionDiag},
    {"block15-ffn", Recipe::kFfnFrag},
    {"block15-attention", Recipe::kQkvFragOnly},
    {"block23-ffwd", Recipe::kSplitMixF16},
    {"block23-ffwd-projection", Recipe::kProjFrag},
    {"block23-attention", Recipe::kQkvFrag},
    {"block31-expand", Recipe::kVitFrag},
    {"block31-contract", Recipe::kVitFrag},
    {"block31-qkv", Recipe::kQkvF16Frag},
    {"block31-projection", Recipe::kVitProjFrag},
    {"decoder39-weights", Recipe::kDecoderF16r},
    {"block48-weights", Recipe::kDecoderF16r},
    {"block56-weights", Recipe::kDecoderF16r},
    {"block62-weights", Recipe::kDecoderF16r},
    {"block66-weights", Recipe::kDecoderF16r},
};

// A sample of f32 bit patterns: every sign, exponent and top 11 mantissa bits,
// with the low 12 bits zero and with them scrambled.
constexpr uint32_t kSampleCount = 1u << 21;
uint32_t sample(uint32_t j) { return (j >> 1) << 12 | (j & 1) * ((j * 0x9e3779b1u) >> 20); }

// The weights dlsslopd uploads, in upload order. hip-plan checks that the
// plan at every tier lists the same.
dlsslop::Result<std::vector<WeightSpec>> production_weights(bool performance)
{
    return hip::plan(kNativeTiers[0].width, kNativeTiers[0].networkHeight, performance)
        .transform([](hip::Plan&& plan) { return std::move(plan.weights); });
}

// What upstream gave for the sample and for every binary16.
struct PrimitiveDigests {
    uint64_t widen_half, exact_fp8, exact_half, round_half, saturate_fp8, scale_piece;
};
constexpr PrimitiveDigests kUpstreamPrimitives = {
    0xb0659868ec053145u, 0xc7f211df4383f3c9u, 0x92886aaababdea21u,
    0xe5692a3085370020u, 0xb34b0cb5c438f085u, 0xd121ea1c179fb205u,
};

// What upstream made of each case: its synthetic input, f32 or binary16, and
// the image it uploaded.
struct CaseDigest {
    size_t spec;
    bool half;
    uint64_t input;
    size_t bytes;
    uint64_t packed;
};
constexpr CaseDigest kUpstreamCases[] = {
    {0, false, 0xe2d6d4e40e98cf1du, 256, 0xe2d6d4e40e98cf1du}, // post70-scales
    {0, true, 0xee46147e18c68760u, 256, 0x2b10b2017141567du}, // post70-scales
    {1, false, 0xdfd82875d596ca19u, 384, 0xdfd82875d596ca19u}, // post70-head
    {1, true, 0x1c91e0f64fdfabdcu, 384, 0xd4bfc8202e72fd6au}, // post70-head
    {2, false, 0x24b764c0257dd203u, 2097152, 0x24b764c0257dd203u}, // block23-ffwd
    {2, true, 0xeb4a17b7dab08ff5u, 2097152, 0xb9be0717ea626b5bu}, // block23-ffwd
    {3, false, 0x98ee6f18e84fe688u, 41088, 0xa6f9e0c65c7de9b3u}, // block0-ffn@c32fp8
    {3, true, 0xb5c3e5b70f2f1e39u, 41088, 0x60720e754fc68466u}, // block0-ffn@c32fp8
    {4, false, 0xe1f60a5bd1a958beu, 32900, 0xe792550614343a6bu}, // block0-attention@c32fp8
    {4, true, 0x214e4cb7cd465213u, 32900, 0x75048295436b358du}, // block0-attention@c32fp8
    {5, false, 0x9f3b50a11bef0c0bu, 8192, 0x5d4396a4207ec368u}, // block4-ds@ds-cast
    {5, true, 0x1a9f86d204abc565u, 8192, 0xc21ae67dc05d97afu}, // block4-ds@ds-cast
    {6, false, 0xd0311835a075b300u, 16384, 0x9388d2044df54e46u}, // block8-ds@ds-frag
    {6, true, 0xbb97eed0c9726cd5u, 16384, 0xe1741fece8cb626du}, // block8-ds@ds-frag
    {7, false, 0x98d274b090362e6fu, 65536, 0xc2e5500b3706d0aau}, // block14-ds@ds-frag
    {7, true, 0x1214055e48c43fb5u, 65536, 0x48dd15c11bd2d469u}, // block14-ds@ds-frag
    {8, false, 0x9870fe06e692a6d0u, 262144, 0x3ac30cc5da23023eu}, // block22-ds@ds-frag
    {8, true, 0x77924ebb70bf3543u, 262144, 0x323716957a9b3ecbu}, // block22-ds@ds-frag
    {9, false, 0x913895cf14f63e03u, 1048576, 0xa2dc2698a5a49053u}, // head-matrix@ds-frag
    {9, true, 0x9f4f7574e41fdd78u, 1048576, 0xb176949723e94244u}, // head-matrix@ds-frag
    {10, false, 0x9da6e5dc69547e20u, 147712, 0x1883b6b248a77735u}, // block5-ffn@fp8-g128
    {10, true, 0xc489ba41a5432f70u, 147712, 0xb53e550921c2952du}, // block5-ffn@fp8-g128
    {11, false, 0x7921b9ad45d92e24u, 590336, 0x9b350f1a9d6931a5u}, // block9-ffn@fp8-g128
    {11, true, 0x03cea55d8f1e3cfau, 590336, 0xc6b73a120e5abee1u}, // block9-ffn@fp8-g128
    {12, false, 0xd0c4fe11ad594fedu, 98568, 0x0401d10103f648c1u}, // block5-attention@fp8
    {12, true, 0xaeeb9207c5bd76a5u, 98568, 0x6c6cf21d1c4d2703u}, // block5-attention@fp8
    {13, false, 0xfff29ede2234e5f7u, 328208, 0x6580cd05fe82b486u}, // block9-attention@fp8
    {13, true, 0xd3693c2590d4eca5u, 328208, 0x0ba9173edd754594u}, // block9-attention@fp8
    {14, false, 0x19dd262ff3c8fe07u, 1180704, 0xb639c6f675c94f60u}, // block15-attention@fp8
    {14, true, 0xa555693211e9bebbu, 1180704, 0x3a447121ff40c5ecu}, // block15-attention@fp8
    {15, false, 0xfece94947600d068u, 4458560, 0xccbee78cf4a6dcdcu}, // block23-attention@fp8
    {15, true, 0x9710f7fa37c3e3afu, 4458560, 0x4c647e24a6230c8fu}, // block23-attention@fp8
    {16, false, 0xb1f913d69e82714au, 104712, 0x1c6a5520eb295128u}, // block5-attention@fp8-diag
    {16, true, 0x14a1a942cad0e494u, 104712, 0x6ce5eb3f1f021461u}, // block5-attention@fp8-diag
    {17, false, 0xa9cb0085eff87e4eu, 340496, 0x847d07836171ce95u}, // block9-attention@fp8-diag
    {17, true, 0x727ff631eb718281u, 340496, 0xab8c6e4040fbd2dbu}, // block9-attention@fp8-diag
    {18, false, 0x20b29c9771b0099fu, 1205280, 0x340e4ab15147a1d3u}, // block15-attention@fp8-diag
    {18, true, 0x4e2c849375b48526u, 1205280, 0xa92c121efd8f8e39u}, // block15-attention@fp8-diag
    {19, false, 0x89a11fdc731aa248u, 2360320, 0xcbe47d666c4b2e39u}, // block15-ffn@ffn-frag
    {19, true, 0x6f74eb34c3073162u, 2360320, 0x76ce4652ac4cf123u}, // block15-ffn@ffn-frag
    {20, false, 0xa062a4dff3ef2a8fu, 1180704, 0xac1616427285d24du}, // block15-attention@qkv-frag-only
    {20, true, 0x14da9fbfd3e4c2ceu, 1180704, 0x72d0e72b11a26c0du}, // block15-attention@qkv-frag-only
    {21, false, 0xdbfe57bb11476621u, 2097152, 0x7fa9b583bf969105u}, // block23-ffwd@split-mix-f16
    {21, true, 0xba8c39dd55012b64u, 2097152, 0x9286e10155be67ebu}, // block23-ffwd@split-mix-f16
    {22, false, 0x8f2dfb62181d5365u, 1050624, 0x199bac3494f9b773u}, // block23-ffwd-projection@proj-frag
    {22, true, 0x94d69f88ef2e6870u, 1050624, 0xe477efb58dcf740eu}, // block23-ffwd-projection@proj-frag
    {23, false, 0x918bb1b563e89ccdu, 4458560, 0xf4215534654bb99au}, // block23-attention@qkv-frag
    {23, true, 0xea64b9ed5bd7e041u, 4458560, 0x37f527ae87d7b5e7u}, // block23-attention@qkv-frag
    {24, false, 0x4de4dced765ff09cu, 16777216, 0x5deace420868adc7u}, // block31-expand@vit-frag
    {24, true, 0xcb4ba2c439d4f459u, 16777216, 0xd581d96e67fd667bu}, // block31-expand@vit-frag
    {25, false, 0x7b4d69dba00a5273u, 16781312, 0x00c0d7b6991ed6dbu}, // block31-contract@vit-frag
    {25, true, 0x44919f4bdd291084u, 16781312, 0xbb91b9be9a17cc2cu}, // block31-contract@vit-frag
    {26, false, 0xa160e16e92d8faa3u, 6291584, 0x35f21dcef561e10du}, // block31-qkv@qkv-f16-frag
    {26, true, 0xc3b81417b891200eu, 6291584, 0xfd8bbc759959fcc5u}, // block31-qkv@qkv-f16-frag
    {27, false, 0x3b330bfa99861feau, 4198400, 0xecd2639c0e0f7214u}, // block31-projection@vit-proj-frag
    {27, true, 0xe3ef5a640ed5437au, 4198400, 0x857f95f9adfd3e2au}, // block31-projection@vit-proj-frag
    {28, false, 0x9b655c5bfb876328u, 2099200, 0xae870dfdb135aa0du}, // decoder39-weights@decoder-f16r
    {28, true, 0x3620c5db9c6c6c3au, 2099200, 0x28198f8d7719ed93u}, // decoder39-weights@decoder-f16r
    {29, false, 0xd6bcd09fbd07bdbdu, 525312, 0x5b96caf0dd42656fu}, // block48-weights@decoder-f16r
    {29, true, 0x002e28316818f9d8u, 525312, 0xb060dd7304976f47u}, // block48-weights@decoder-f16r
    {30, false, 0xf3df075dbc0d50a4u, 131584, 0xa8b1e102042213f0u}, // block56-weights@decoder-f16r
    {30, true, 0xf4234956f3ed6878u, 131584, 0xf674de6a2dfc46ddu}, // block56-weights@decoder-f16r
    {31, false, 0x573454f7c479b949u, 33024, 0x341334197e62fb3bu}, // block62-weights@decoder-f16r
    {31, true, 0x89f171abc31b6c86u, 33024, 0x31b1433c8b961744u}, // block62-weights@decoder-f16r
    {32, false, 0x6c9cf9a25c850c5au, 8320, 0xe7d56e119e9ebc18u}, // block66-weights@decoder-f16r
    {32, true, 0xff1429a00531a4acu, 8320, 0x4b957920d1f12634u}, // block66-weights@decoder-f16r
};

int failures = 0;
// Whether OK; if not, the test fails with the message.
[[gnu::format(printf, 2, 3)]] bool expect(bool ok, const char* format, ...)
{
    if (ok) return true;
    std::va_list args;
    va_start(args, format);
    std::fputs("hip-weights test: ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    ++failures;
    return false;
}
bool expect_error(const dlsslop::Result<void>& result, const std::string& error)
{
    const std::string what = result ? "no error" : result.error().what;
    return expect(what == error, "expected \"%s\", got \"%s\"", error.c_str(), what.c_str());
}

uint32_t bits(float value) { return std::bit_cast<uint32_t>(value); }
float from_bits(uint32_t bits) { return std::bit_cast<float>(bits); }

struct Hash {
    uint64_t value = 0xcbf29ce484222325u;
    template <class T>
    void add(T x)
    {
        value = fnv1a(&x, sizeof x, value);
    }
};

void check_upstream_primitives()
{
    Hash widen;
    for (unsigned h = 0; h < 65536; ++h) widen.add(hip::widen_half(uint16_t(h)));
    Hash fp8, half, rounded, saturated, piece;
    for (uint32_t j = 0; j < kSampleCount; ++j) {
        const float x = from_bits(sample(j));
        const auto code = hip::exact_fp8(x);
        fp8.add(uint16_t(code ? *code : 256 + unsigned(code.error())));
        const auto h = hip::exact_half(x);
        half.add(uint32_t(h ? *h : 65536 + unsigned(h.error())));
        rounded.add(hip::round_half(x));
        saturated.add(hip::saturate_fp8(x));
        piece.add(hip::scale_piece(x));
    }
    const PrimitiveDigests& u = kUpstreamPrimitives;
    expect(widen.value == u.widen_half, "widen_half differs from upstream's Half");
    expect(fp8.value == u.exact_fp8, "exact_fp8 differs from upstream's ExactWeightFp8");
    expect(half.value == u.exact_half, "exact_half differs from upstream's ExactWeightHalf");
    expect(rounded.value == u.round_half, "round_half differs from upstream's RoundWeightHalf");
    expect(saturated.value == u.saturate_fp8, "saturate_fp8 differs from upstream's HostF");
    expect(piece.value == u.scale_piece, "scale_piece differs from upstream's HostScalePiece");
}

void check_fp8()
{
    for (unsigned code = 0; code < 256; ++code) {
        if ((code & 0x7f) == 0x7f) continue; // NaN
        const auto packed = hip::exact_fp8(fp8_value(code));
        if (!expect(packed && *packed == code, "E4M3 code %#x does not round-trip", code)) break;
    }
    const struct {
        float value;
        Flaw flaw;
    } rejected[] = {
        {NAN, Flaw::kFp8Nonfinite},
        {-INFINITY, Flaw::kFp8Nonfinite},
        {480.f, Flaw::kFp8Inexact}, // Exponent 15, mantissa 7: NaN's pattern.
        {512.f, Flaw::kFp8Inexact},
        {1.0625f, Flaw::kFp8Inexact},
        {0x1p-10f, Flaw::kFp8Subnormal},
        {0x1.8p-9f, Flaw::kFp8Subnormal},
        {-0x1.2p-7f, Flaw::kFp8Subnormal},
    };
    for (const auto& r : rejected) {
        const auto packed = hip::exact_fp8(r.value);
        expect(!packed && packed.error() == r.flaw, "exact_fp8(%a) does not fail as %s", double(r.value),
               hip::kFlawWords[size_t(r.flaw)]);
    }
}

void check_halves()
{
    for (unsigned h = 0; h < 65536; ++h) {
        const float wide = hip::widen_half(uint16_t(h));
        if ((h & 0x7c00) == 0x7c00) {
            // Infinity and NaN, payload kept: a signalling NaN stays one.
            const auto packed = hip::exact_half(wide);
            if (!expect(bits(wide) == ((h & 0x8000u) << 16 | 0x7f800000 | (h & 1023) << 13) && !packed &&
                            packed.error() == Flaw::kHalfNonfinite,
                        "binary16 %#x does not widen to its f32 NaN or infinity", h))
                break;
            continue;
        }
        const auto packed = hip::exact_half(wide);
        if (!expect(bits(wide) == bits(half_value(h)) && packed && *packed == h,
                    "binary16 %#x does not round-trip", h))
            break;
    }
    const struct {
        float value;
        Flaw flaw;
    } rejected[] = {
        {INFINITY, Flaw::kHalfNonfinite},   {65536.f, Flaw::kHalfOverflow},  {-0x1p20f, Flaw::kHalfOverflow},
        {1 + 0x1p-11f, Flaw::kHalfInexact}, {0x1p-25f, Flaw::kHalfSubnormal}, {0x1.8p-24f, Flaw::kHalfSubnormal},
    };
    for (const auto& r : rejected) {
        const auto packed = hip::exact_half(r.value);
        expect(!packed && packed.error() == r.flaw, "exact_half(%a) does not fail as %s", double(r.value),
               hip::kFlawWords[size_t(r.flaw)]);
    }
    // Every value but NaN rounds as a conversion to _Float16 does.
    const float edges[] = {2049.f,          2051.f,     65504.f,  65519.f,      65520.f,       -0.f,      0x1p-25f,
                           0x1.000002p-25f, 0x1.8p-24f, 0x1p-24f, 0x1.fffp-15f, 0x1.fff8p-15f, -INFINITY, 0x1p-149f};
    auto matches = [](float x) { return hip::round_half(x) == std::bit_cast<uint16_t>(static_cast<_Float16>(x)); };
    for (float x : edges) expect(matches(x), "round_half(%a) differs from _Float16", double(x));
    for (uint32_t j = 0; j < kSampleCount; ++j) {
        const float x = from_bits(sample(j));
        if (!std::isnan(x) && !expect(matches(x), "round_half(%a) differs from _Float16", double(x))) break;
    }
    expect(hip::round_half(-NAN) == 0xfe00 && hip::round_half(from_bits(0x7f800001)) == 0x7e00,
           "round_half does not make NaN 0x7e00 with its sign");
}

void check_saturation()
{
    const struct {
        float value, saturated, piece;
    } table[] = {
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
    for (const auto& t : table) {
        expect(bits(hip::saturate_fp8(t.value)) == bits(t.saturated), "saturate_fp8(%a) is %a, not %a",
               double(t.value), double(hip::saturate_fp8(t.value)), double(t.saturated));
        expect(bits(hip::scale_piece(t.value)) == bits(t.piece), "scale_piece(%a) is %a, not %a", double(t.value),
               double(hip::scale_piece(t.value)), double(t.piece));
    }
}

void check_fragment_tiles()
{
    const struct {
        size_t rows, columns;
    } shapes[] = {{16, 32}, {64, 32}, {32, 128}, {48, 64}};
    Random random{1};
    for (const auto& s : shapes) {
        const size_t bytes = s.rows * s.columns;
        std::vector<float> values(bytes / 4 + 3);
        for (float& v : values) v = from_bits(uint32_t(random.next()));
        const std::vector<float> before = values;
        hip::fragment_tiles(values, 1, s.rows, s.columns);
        const auto* in = reinterpret_cast<const uint8_t*>(before.data() + 1);
        const auto* out = reinterpret_cast<const uint8_t*>(values.data() + 1);
        std::vector<unsigned> hits(bytes);
        bool same = true;
        for (size_t n = 0; n < s.rows; ++n)
            for (size_t k = 0; k < s.columns; ++k) {
                // FragmentPackedMatrix's permutation, as upstream writes it.
                const size_t at = ((n / 16) * (s.columns / 32) + k / 32) * 512 +
                                  (((k % 32) / 16 * 2 + (k % 16) / 8) * 16 + n % 16) * 8 + k % 8;
                same &= at < bytes && out[at] == in[n * s.columns + k];
                if (at < bytes) ++hits[at];
            }
        bool bijection = true;
        for (unsigned h : hits) bijection &= h == 1;
        expect(same && bijection, "fragment_tiles of %zux%zu is not upstream's permutation", s.rows, s.columns);
        expect(bits(values[0]) == bits(before[0]) && !std::memcmp(&values[1 + bytes / 4], &before[1 + bytes / 4], 8),
               "fragment_tiles of %zux%zu writes outside its matrix", s.rows, s.columns);
    }
}

// Piece PART of SCALE, as upstream's diagonal loops chain them.
float piece(float scale, unsigned part)
{
    float remaining = scale, result = 0;
    for (unsigned j = 0; j <= part; ++j) {
        result = hip::scale_piece(remaining);
        remaining -= result;
    }
    return result;
}

void check_diagonals()
{
    Random random{2};
    std::vector<float> c32(8736);
    for (float& v : c32) v = f32_value(kFloat, random);
    std::vector<float> packed = c32;
    hip::append_c32_diagonals(packed);
    bool same = packed.size() == 10272 && !std::memcmp(packed.data(), c32.data(), 8736 * 4);
    const auto* tiles = reinterpret_cast<const uint8_t*>(packed.data()) + 34944;
    for (unsigned part = 0; part < 3; ++part)
        for (unsigned ci = 0; ci < 2; ++ci)
            for (unsigned kt = 0; kt < 2; ++kt)
                for (unsigned gr = 0; gr < 2; ++gr)
                    for (unsigned rc = 0; rc < 16; ++rc)
                        for (unsigned e = 0; e < 8; ++e) {
                            const unsigned c = ci * 16 + rc, k = kt * 16 + gr * 8 + e;
                            const auto code = hip::exact_fp8(piece(c32[8704 + c], part));
                            const uint8_t expected = k == c ? code.value_or(0) : 0;
                            const size_t at = ((part * 2 + ci) * 2 + kt) * 512 + (gr * 16 + rc) * 8 + e;
                            same &= code && tiles[at] == expected;
                        }
    expect(same, "C32 diagonal tiles differ from upstream's layout");

    for (unsigned c : {64u, 128u, 256u, 512u}) {
        const size_t heads = c / 32, scales = 4 * size_t(c) * c + heads * 4096 + heads, base = scales + c;
        std::vector<float> mh(base);
        for (float& v : mh) v = f32_value(kFloat, random);
        packed = mh;
        hip::append_mh_diagonals(packed, c);
        same = packed.size() == base + 24 * c && !std::memcmp(packed.data(), mh.data(), base * 4);
        tiles = reinterpret_cast<const uint8_t*>(packed.data() + base);
        for (unsigned part = 0; part < 3; ++part)
            for (unsigned ct = 0; ct < c / 16; ++ct)
                for (unsigned gr = 0; gr < 2; ++gr)
                    for (unsigned rc = 0; rc < 16; ++rc)
                        for (unsigned e = 0; e < 8; ++e) {
                            const auto code = hip::exact_fp8(piece(mh[scales + ct * 16 + rc], part));
                            const uint8_t expected = gr * 8 + e == rc ? code.value_or(0) : 0;
                            const size_t at = (part * (c / 16) + ct) * 512 + (gr * 16 + rc) * 8 + e;
                            same &= code && tiles[at] == expected;
                        }
        expect(same, "MH diagonal tiles at c %u differ from upstream's layout", c);
    }
}

void check_grouped_contract()
{
    const WeightSpec specs[] = {{"block5-ffn", Recipe::kMhFfn}, {"block9-ffn", Recipe::kMhFfn},
                                {"block15-ffn", Recipe::kFfnFrag}};
    for (const WeightSpec& spec : specs) {
        const unsigned c = hip::channels(spec);
        const size_t cc = size_t(c) * c;
        std::vector<float> values(9 * cc + c);
        Random random{c};
        for (size_t i = 0; i < values.size(); ++i) values[i] = f32_value(content(spec, i), random);
        // -0 counts as zero; any other value outside the groups does not.
        values[4 * cc + 128] = -0.f;
        expect(bool(hip::check_grouped_contract(values, c)), "grouped contraction at c %u rejected", c);
        for (const float bad : {0x1p-149f, NAN}) {
            const size_t at = 4 * cc + (c - 1) * 4 * c; // Row c-1, k 0.
            const float kept = values[at];
            values[at] = bad;
            expect_error(hip::check_grouped_contract(values, c),
                         "element " + std::to_string(at) + ": nonzero outside grouped contraction");
            values[at] = kept;
        }
    }
}

// A directory for the test's files, removed with them.
struct Scratch {
    std::string directory;
    std::vector<std::string> files;
    Scratch()
    {
        char name[] = "/tmp/dlsslop-amd-hip-weights-XXXXXX";
        if (mkdtemp(name)) directory = name;
    }
    ~Scratch()
    {
        clear();
        if (!directory.empty()) rmdir(directory.c_str());
    }
    void clear()
    {
        for (const std::string& file : files) unlink(file.c_str());
        files.clear();
    }
    // FILE in the directory, holding DATA.
    std::string write(const std::string& file, std::string_view data)
    {
        std::string path = dlsslop::join(directory, file);
        expect(bool(dlsslop::write_file(path, data)), "cannot write %s", path.c_str());
        files.push_back(path);
        return path;
    }
};

void check_recipes(Scratch& scratch)
{
    for (const CaseDigest& d : kUpstreamCases) {
        const WeightSpec& spec = kCases[d.spec];
        const std::string name = key(spec) + (d.half ? " from binary16" : " from f32");
        const std::string input = synthetic_file(spec, hip::file_elements(spec), d.half);
        scratch.write(std::string(spec.stem) + (d.half ? ".f16" : ".f32"), input);
        auto file = hip::read_weights(scratch.directory, spec);
        const auto packed = file.and_then([&](WeightFile& f) { return hip::pack(spec, f); });
        scratch.clear();
        if (!expect(fnv1a(input.data(), input.size()) == d.input, "%s: synthetic input differs", name.c_str()) ||
            !expect(bool(packed), "%s: %s", name.c_str(), packed ? "" : packed.error().what.c_str()))
            continue;
        const size_t bytes = file->values.size() * 4;
        expect(bytes == d.bytes && bytes == hip::packed_bytes(spec), "%s: %zu bytes, upstream uploads %zu",
               name.c_str(), bytes, d.bytes);
        expect(fnv1a(file->values.data(), bytes) == d.packed, "%s: packed image differs from upstream's",
               name.c_str());
    }
}

void check_errors(Scratch& scratch)
{
    const std::string& dir = scratch.directory;
    const WeightSpec ffn{"block5-ffn", Recipe::kMhFfn};
    const std::string f16 = dir + "/block5-ffn.f16", f32 = dir + "/block5-ffn.f32";
    auto read = [&](const WeightSpec& spec) { return hip::read_weights(dir, spec).transform([](auto&&) {}); };
    expect_error(read({"block5-bogus", Recipe::kDsFrag}), "unknown weight block5-bogus");
    expect_error(read({"block10-ffn", Recipe::kSplitMixF16}), "no split-mix-f16 packing of block10-ffn");
    expect_error(read({"block66-ffn", Recipe::kMhFfn}), "no fp8-g128 packing of block66-ffn");
    expect_error(read(ffn), "missing " + f16);
    scratch.write("block5-ffn.f16", "");
    expect_error(read(ffn), "empty " + f16);
    scratch.write("block5-ffn.f16", "abc");
    expect_error(read(ffn), "weight size " + f16);
    scratch.write("block5-ffn.f16", std::string(2 * hip::file_elements(ffn), '\0'));
    expect(bool(read(ffn)), "a zero block5-ffn.f16 rejected");
    // Whenever the f32 file opens, it is the one read.
    scratch.write("block5-ffn.f32", "abcdef");
    expect_error(read(ffn), "weight size " + f32);
    scratch.write("block5-ffn.f32", std::string(8, '\0'));
    expect_error(read(ffn), "packed weight shape " + f32);
    scratch.write("post70-head.f32", std::string(95 * 4, '\0'));
    expect_error(read({"post70-head", Recipe::kRaw}), "weight shape " + dir + "/post70-head.f32: 95");
    // The ds-cast, ds-frag and decoder loaders check only a lower bound. The
    // ds-cast image is the whole file, the ds-frag one its first 2c² values.
    const WeightSpec cast{"block4-ds", Recipe::kDsCast}, frag{"block8-ds", Recipe::kDsFrag};
    scratch.write("block4-ds.f16", std::string(2 * (2 * 32 * 32 + 5), '\0'));
    auto ds = hip::read_weights(dir, cast);
    expect(ds && hip::pack(cast, *ds) && ds->values.size() == 2 * 32 * 32 + 5,
           "a longer block4-ds not packed whole");
    scratch.write("block8-ds.f16", std::string(2 * (2 * 64 * 64 + 5), '\0'));
    ds = hip::read_weights(dir, frag);
    expect(ds && hip::pack(frag, *ds) && ds->values.size() == 64 * 64, "a longer block8-ds not packed to c² floats");
    const WeightSpec decoder{"block62-weights", Recipe::kDecoderF16r};
    scratch.write("block62-weights.f16", std::string(2 * 2 * 64 * 64, '\0'));
    expect(bool(read(decoder)), "a block62-weights of 2c² values rejected");
    scratch.write("block62-weights.f16", std::string(2 * (2 * 64 * 64 - 1), '\0'));
    expect_error(read(decoder), "decoder weight shape " + dir + "/block62-weights.f16");
    scratch.clear();

    // Encoding errors name the file and the element.
    auto packed = [](const WeightSpec& spec, size_t element, float value) {
        WeightFile file{"PATH", std::vector<float>(hip::file_elements(spec))};
        Random random{3};
        for (size_t i = 0; i < file.values.size(); ++i) file.values[i] = f32_value(content(spec, i), random);
        file.values[element] = value;
        return hip::pack(spec, file);
    };
    expect_error(packed({"block0-attention", Recipe::kC32}, 7, 0.3f),
                 "PATH: element 7: matrix weight not exact finite FP8");
    expect_error(packed({"block0-ffn", Recipe::kC32}, 600, INFINITY),
                 "PATH: element 600: nonfinite FP8 matrix weight");
    expect_error(packed({"block5-attention", Recipe::kMhAttention}, 12288, 0x1p-12f),
                 "PATH: element 12288: matrix weight not exact FP8 subnormal");
    expect_error(packed(ffn, 4 * 4096 + 128, 1.f), "PATH: element 16512: nonzero outside grouped contraction");
    expect_error(packed({"block23-ffwd", Recipe::kSplitMixF16}, 262149, 1.f / 3),
                 "PATH: element 262149: weight not exact half");
    expect_error(packed({"block31-qkv", Recipe::kQkvF16Frag}, 2097153, 0x1p16f),
                 "PATH: element 2097153: half weight overflow");
    expect_error(packed({"block22-ds", Recipe::kDsFrag}, 70000, 0x1p-26f),
                 "PATH: element 70000: weight not exact half subnormal");
    expect_error(packed({"block31-projection", Recipe::kVitProjFrag}, 1048575, NAN),
                 "PATH: element 1048575: nonfinite FP8 matrix weight");
    // Values outside the regions a recipe encodes are kept, whatever they are.
    expect(bool(packed({"decoder39-weights", Recipe::kDecoderF16r}, 524799, NAN)), "a NaN scale rejected");
}

void check_production_list()
{
    for (const bool performance : {false, true}) {
        const auto list = production_weights(performance);
        if (!expect(bool(list), "%s", list ? "" : list.error().what.c_str())) continue;
        size_t bytes = 0;
        std::set<std::string> keys;
        for (const WeightSpec& spec : *list) {
            bytes += hip::packed_bytes(spec);
            expect(hip::file_elements(spec) != 0, "%s: unknown stem", spec.stem);
            expect(keys.insert(key(spec)).second, "%s listed twice", key(spec).c_str());
        }
        // The weights upstream uploads: its traces show these totals.
        const size_t count = performance ? 253 : 268, total = performance ? 654182056 : 696668200;
        expect(list->size() == count && bytes == total, "%zu weights%s of %zu bytes; upstream uploads %zu of %zu",
               list->size(), performance ? " with --performance" : "", bytes, count, total);
    }
}

// Packs WEIGHTS from ASSETS: bytes, FNV-1a 64 and name of each.
int pack_model(const std::string& assets, const std::vector<WeightSpec>& weights)
{
    for (const WeightSpec& spec : weights) {
        auto file = hip::read_weights(assets, spec);
        if (const auto packed = file.and_then([&](WeightFile& f) { return hip::pack(spec, f); }); !packed) {
            std::fprintf(stderr, "hip-weights-test: %s\n", packed.error().what.c_str());
            return 1;
        }
        const size_t bytes = file->values.size() * 4;
        std::printf("%zu\t%016" PRIx64 "\t%s\n", bytes, fnv1a(file->values.data(), bytes), key(spec).c_str());
    }
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    std::string model;
    bool list = false, performance = false;
    const option options[] = {{"model", required_argument, nullptr, 'm'}, {"list", no_argument, nullptr, 'l'},
                              {"performance", no_argument, nullptr, 'p'}, {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+m:lph", options, nullptr)) != -1;) {
        if (code == 'm') model = optarg;
        else if (code == 'l') list = true;
        else if (code == 'p') performance = true;
        else if (code == 'h') {
            std::puts("Usage: hip-weights-test [OPTION]...\n"
                      "Checks the HIP network's weight packing against upstream's packers. No GPU or model needed.\n"
                      " -m, --model DIR    Instead, pack every weight dlsslopd uploads from the model in DIR\n"
                      "                    and print each image's bytes, FNV-1a 64 and name, in upload order\n"
                      "                    (default: unset)\n"
                      " -l, --list         Instead, print the stem, file elements, packed bytes and name of\n"
                      "                    every weight dlsslopd uploads, in upload order (default: off)\n"
                      " -p, --performance  With --model or --list, leave out the blocks that dlsslopd\n"
                      "                    --performance skips (default: off)\n"
                      " -h, --help         Show help (default: off)");
            return 0;
        } else
            return 2;
    }
    if (optind != argc || (list && !model.empty())) return 2;
    if (list || !model.empty()) {
        const auto weights = production_weights(performance);
        if (!weights) {
            std::fprintf(stderr, "hip-weights-test: %s\n", weights.error().what.c_str());
            return 1;
        }
        if (!model.empty()) return pack_model(model, *weights);
        for (const WeightSpec& spec : *weights)
            std::printf("%s\t%zu\t%zu\t%s\n", spec.stem, hip::file_elements(spec), hip::packed_bytes(spec),
                        key(spec).c_str());
        return 0;
    }
    Scratch scratch;
    if (!expect(!scratch.directory.empty(), "cannot create a directory in /tmp")) return 1;
    check_upstream_primitives();
    check_fp8();
    check_halves();
    check_saturation();
    check_fragment_tiles();
    check_diagonals();
    check_grouped_contract();
    check_recipes(scratch);
    check_errors(scratch);
    check_production_list();
    if (!failures) std::puts("hip-weights test: every check passed");
    return failures ? 1 : 0;
}
