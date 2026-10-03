// SPDX-License-Identifier: MIT
#include "hip_weights.hpp"
#include "files.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

namespace dlsslop::hip {

namespace {
uint32_t bits(float value) { return std::bit_cast<uint32_t>(value); }
float from_bits(uint32_t bits) { return std::bit_cast<float>(bits); }

// A stem's type: what follows blockN- or post70-. head-matrix is a ds and
// decoder39-weights a weights; post70-scales and post70-head have their own.
enum Type : uint8_t {
    kDs, kWeights, kFfwd, kFfwdProjection, kFfn, kAttention, kExpand, kContract, kQkv, kProjection, kScales, kHead
};
// The names parse() looks up after blockN- or post70-.
constexpr std::string_view kTypeNames[] = {"ds", "weights", "ffwd", "ffwd-projection", "ffn", "attention",
    "expand", "contract", "qkv", "projection"};
static_assert(std::size(kTypeNames) == kScales);
constexpr uint16_t kAnyType = 0xffff;
// The channel counts a loader applies to, as bits.
enum Channels : uint8_t { kAt32 = 1, kAbove32 = 2, kAnyChannels = 3 };

// The stems upstream applies each recipe's loader to: their types and channel
// counts.
struct Domain {
    uint16_t types;
    Channels channels;
};
constexpr Domain kDomains[] = {
    {kAnyType, kAnyChannels},
    {1 << kFfn | 1 << kAttention, kAt32},
    {1 << kDs, kAt32},
    {1 << kDs, kAbove32},
    {1 << kFfn, kAbove32},
    {1 << kAttention, kAbove32},
    {1 << kAttention, kAbove32},
    {1 << kFfn, kAbove32},
    {1 << kAttention, kAbove32},
    {1 << kFfwd, kAbove32},
    {1 << kFfwdProjection, kAbove32},
    {1 << kAttention, kAbove32},
    {1 << kExpand | 1 << kContract, kAbove32},
    {1 << kQkv, kAbove32},
    {1 << kProjection, kAbove32},
    {1 << kWeights, kAnyChannels},
};
static_assert(std::size(kDomains) == std::size(kRecipeSuffix));

// The channel count of each block, by the last block of each stage (upstream:
// WeightElements). Later blocks, and post70 as block 70, have 32.
constexpr struct {
    uint8_t last;
    uint16_t c;
} kStages[] = {{4, 32}, {8, 64}, {14, 128}, {22, 256}, {30, 512}, {38, 1024}, {47, 512}, {55, 256}, {61, 128},
               {65, 64}};

// A stem's element count, channel count and type; the counts are 0 for a stem
// upstream does not know.
struct Stem {
    size_t elements;
    unsigned c;
    Type type;
};

// The values in a file of TYPE at C channels.
size_t type_elements(Type type, size_t c)
{
    const size_t cc = c * c;
    switch (type) {
    case kDs: return 2 * cc;
    case kWeights: return 2 * cc + c;
    case kFfwd: return 524288;
    case kFfwdProjection: return 262656;
    case kFfn: return c == 32 ? 8736 : 9 * cc + c;
    case kAttention: return c == 32 ? 8225 : 4 * cc + (c / 32) * 4096 + c / 32 + c;
    case kExpand: return 4194304;
    case kContract: return 4195328;
    case kQkv: return 3145760;
    case kProjection: return 1049600;
    case kScales:
    case kHead: break; // parse() knows these stems by name.
    }
    return 0;
}

// Upstream: WeightElements. Its stage rule also gives the channel count that
// upstream passes the stem's loaders.
Stem parse(std::string_view stem)
{
    if (stem == "head-matrix") return {524288, 512, kDs};
    if (stem == "decoder39-weights") return {524800, 512, kWeights};
    if (stem == "post70-scales") return {64, 32, kScales};
    if (stem == "post70-head") return {96, 32, kHead};
    unsigned block = 70;
    std::string_view name;
    if (stem.starts_with("block")) {
        const auto [dash, error] = std::from_chars(stem.data() + 5, stem.data() + stem.size(), block);
        if (error != std::errc() || dash == stem.data() + stem.size() || *dash != '-') return {};
        name = stem.substr(size_t(dash + 1 - stem.data()));
    } else if (stem.starts_with("post70-")) {
        name = stem.substr(7);
    } else {
        return {};
    }
    const auto* known = std::ranges::find(kTypeNames, name);
    if (known == std::end(kTypeNames)) return {};
    const Type type = Type(known - kTypeNames);
    unsigned c = 32;
    for (const auto& stage : kStages)
        if (block <= stage.last) {
            c = stage.c;
            break;
        }
    return {type_elements(type, c), c, type};
}

// A matrix of ROWS x COLUMNS bytes from float START on, for fragment_tiles().
struct Tiles {
    size_t start, rows, columns;
};
// PARTS matrices of ROWS x COLUMNS values as exact binary16 16x16 B-fragment
// tiles, which replace the values; the TAIL floats after the matrices follow.
struct HalfTiles {
    size_t parts, rows, columns, tail;
};
enum Diagonals : uint8_t { kNoDiagonals, kC32Diagonals, kMhDiagonals };
constexpr size_t kC32DiagonalFloats = 1536; // 12 tiles of 512 bytes.

// What a recipe does, in the order upstream's loaders do it: the grouped
// contraction check, the encodings, then the layouts of the encoded bytes.
// Each binary16 or E4M3 region is written from its first byte on.
struct Layout {
    bool grouped;
    Region rounded;   // Binary16, rounded to nearest even (upstream: HalfR).
    Region halves;    // Exact binary16 (upstream: HalfExact).
    Region saturated; // E4M3 of the values saturated to its range.
    std::array<Region, 3> fp8; // Exact E4M3 (upstream: Fp8).
    std::array<Tiles, 3> tiles;
    HalfTiles half_tiles;
    Diagonals diagonals;
};

Layout layout(Recipe recipe, const Stem& stem)
{
    const size_t c = stem.c, cc = c * c;
    const std::array<Region, 3> attention{{{0, 3 * cc}, {3 * cc, cc}}};
    const std::array<Region, 3> ffn{{{0, 4 * cc}, {4 * cc, 4 * cc}, {8 * cc, cc}}};
    switch (recipe) {
    case Recipe::kRaw: return {};
    case Recipe::kC32:
        if (stem.type == kAttention) return {.fp8 = {{{0, 4096}}}};
        return {.fp8 = {{{512, 4096}, {4608, 4096}}}, .diagonals = kC32Diagonals};
    // Upstream clamps the values to ±448 first, which saturate_fp8() does too.
    case Recipe::kDsCast: return {.saturated = {0, 2 * cc}};
    case Recipe::kDsFrag: return {.half_tiles = {1, 2 * c, c, 0}};
    case Recipe::kMhFfn: return {.grouped = true, .fp8 = ffn};
    case Recipe::kMhAttention: return {.fp8 = attention};
    case Recipe::kMhAttentionDiag: return {.fp8 = attention, .diagonals = kMhDiagonals};
    case Recipe::kFfnFrag:
        return {.grouped = true, .fp8 = ffn, .tiles = {{{0, 4 * c, c}, {4 * cc, c, 4 * c}, {8 * cc, c, c}}}};
    case Recipe::kQkvFragOnly: return {.fp8 = attention, .tiles = {{{0, 3 * c, c}}}};
    case Recipe::kSplitMixF16:
        return {.rounded = {0, 262144}, .halves = {262144, 131072}, .fp8 = {{{393216, 131072}}}};
    case Recipe::kProjFrag: return {.fp8 = {{{0, 262144}}}, .tiles = {{{0, 512, 512}}}};
    case Recipe::kQkvFrag: return {.fp8 = attention, .tiles = {{{0, 3 * c, c}, {3 * cc, c, c}}}};
    case Recipe::kVitFrag: {
        // The expansion is 4096 rows of 1024, the contraction 1024 of 4096.
        const size_t rows = stem.type == kExpand ? 4096 : 1024;
        return {.fp8 = {{{0, 4194304}}}, .tiles = {{{0, rows, 4194304 / rows}}}};
    }
    case Recipe::kQkvF16Frag: return {.half_tiles = {3, 1024, 1024, 32}};
    case Recipe::kVitProjFrag: return {.fp8 = {{{0, 1048576}}}, .tiles = {{{0, 1024, 1024}}}};
    case Recipe::kDecoderF16r: return {.rounded = {0, 2 * cc}};
    }
    return {};
}

// The E4M3 code of a value saturate_fp8() or scale_piece() made: always exact.
uint8_t fp8_code(float exact) { return exact_fp8(exact).value_or(0); }

// The error for the value at ELEMENT, made out of line so that the loops that
// find it build no words.
[[gnu::cold, gnu::noinline]] Result<void> bad_element(size_t element, Flaw flaw)
{
    return fail("element " + std::to_string(element) + ": " + kFlawWords[size_t(flaw)]);
}

// ACTION PATH and why the last system call failed, with errno read before the
// words are built.
[[gnu::cold, gnu::noinline]] Result<WeightFile> file_error(const char* action, const std::string& path)
{
    const int error = errno;
    return fail(std::string(action) + " " + path + ": " + std::strerror(error));
}

void round_halves(std::vector<float>& values, Region region)
{
    auto* halves = reinterpret_cast<uint8_t*>(values.data() + region.first);
    for (size_t i = 0; i < region.count; ++i) {
        const uint16_t half = round_half(values[region.first + i]);
        std::memcpy(halves + 2 * i, &half, 2);
    }
}

Result<void> exact_halves(std::vector<float>& values, Region region)
{
    auto* halves = reinterpret_cast<uint8_t*>(values.data() + region.first);
    for (size_t i = 0; i < region.count; ++i) {
        const auto half = exact_half(values[region.first + i]);
        if (!half) return bad_element(region.first + i, half.error());
        std::memcpy(halves + 2 * i, &*half, 2);
    }
    return {};
}

void saturate_fp8s(std::vector<float>& values, Region region)
{
    auto* codes = reinterpret_cast<uint8_t*>(values.data() + region.first);
    for (size_t i = 0; i < region.count; ++i) codes[i] = fp8_code(saturate_fp8(values[region.first + i]));
}

// Upstream: the loops of PackedDsWeightFrag and PackedVitQkvWeightFrag. Lane
// (g, rc) of tile (nt, kt) holds k = g*8.. g*8+7 of row nt*16+rc.
Result<void> half_tiles(std::vector<float>& values, const HalfTiles& t)
{
    const size_t matrix = t.rows * t.columns, halves = t.parts * matrix;
    std::vector<uint16_t> tiles(halves);
    for (size_t part = 0; part < t.parts; ++part)
        for (size_t nt = 0; nt < t.rows / 16; ++nt)
            for (size_t kt = 0; kt < t.columns / 16; ++kt)
                for (size_t g = 0; g < 2; ++g)
                    for (size_t rc = 0; rc < 16; ++rc)
                        for (size_t e = 0; e < 8; ++e) {
                            const size_t from = part * matrix + (nt * 16 + rc) * t.columns + kt * 16 + g * 8 + e;
                            const auto half = exact_half(values[from]);
                            if (!half) return bad_element(from, half.error());
                            tiles[part * matrix + (nt * (t.columns / 16) + kt) * 256 + (g * 16 + rc) * 8 + e] = *half;
                        }
    std::memmove(values.data() + halves / 2, values.data() + halves, t.tail * 4);
    values.resize(halves / 2 + t.tail);
    std::memcpy(values.data(), tiles.data(), halves * 2);
    return {};
}

Result<void> pack_values(const Layout& l, std::vector<float>& values, unsigned c)
{
    if (l.grouped) DLSSLOP_TRY(check_grouped_contract(values, c));
    round_halves(values, l.rounded);
    DLSSLOP_TRY(exact_halves(values, l.halves));
    saturate_fp8s(values, l.saturated);
    DLSSLOP_TRY(pack_fp8_regions(values, l.fp8));
    for (const Tiles& t : l.tiles) fragment_tiles(values, t.start, t.rows, t.columns);
    if (l.half_tiles.parts) DLSSLOP_TRY(half_tiles(values, l.half_tiles));
    if (l.diagonals == kC32Diagonals) append_c32_diagonals(values);
    if (l.diagonals == kMhDiagonals) append_mh_diagonals(values, c);
    return {};
}
} // namespace

size_t file_elements(const WeightSpec& spec) { return parse(spec.stem).elements; }
unsigned channels(const WeightSpec& spec) { return parse(spec.stem).c; }

size_t packed_bytes(const WeightSpec& spec)
{
    const Stem stem = parse(spec.stem);
    const Layout l = layout(spec.recipe, stem);
    const HalfTiles& t = l.half_tiles;
    if (t.parts) return 2 * t.parts * t.rows * t.columns + 4 * t.tail;
    const size_t diagonals = l.diagonals == kC32Diagonals ? kC32DiagonalFloats
                             : l.diagonals == kMhDiagonals ? 24 * size_t(stem.c)
                                                           : 0;
    return 4 * (stem.elements + diagonals);
}

Result<WeightFile> read_weights(std::string_view assets, const WeightSpec& spec)
{
    const Stem stem = parse(spec.stem);
    if (!stem.c) return fail("unknown weight " + std::string(spec.stem));
    const Domain& domain = kDomains[size_t(spec.recipe)];
    if (!(domain.types >> stem.type & 1) || !(domain.channels & (stem.c == 32 ? kAt32 : kAbove32)))
        return fail("no " + std::string(kRecipeSuffix[size_t(spec.recipe)]) + " packing of " + spec.stem);
    WeightFile file{join(assets, spec.stem) + ".f32", {}};
    Descriptor in(::open(file.path.c_str(), O_RDONLY | O_CLOEXEC));
    const bool full = in.fd >= 0;
    if (!full) {
        if (errno != ENOENT) return file_error("open", file.path);
        file.path.replace(file.path.size() - 2, 2, "16");
        in.fd = ::open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (in.fd < 0)
            return errno == ENOENT ? fail("missing weight " + file.path.substr(0, file.path.size() - 4) +
                                          " (neither .f32 nor .f16)")
                                   : file_error("open", file.path);
    }
    struct stat st{};
    if (fstat(in.fd, &st)) return file_error("read", file.path);
    const size_t bytes = stem.elements * (full ? 4 : 2);
    if (st.st_size != off_t(bytes))
        return fail(file.path + ": " + std::to_string(st.st_size) + " bytes, expected " + std::to_string(bytes));
    file.values.resize(stem.elements);
    // A binary16 file goes into the second half of the floats and is widened
    // from the front: each float overwrites only halves widened already.
    auto* data = reinterpret_cast<char*>(file.values.data()) + stem.elements * 4 - bytes;
    if (const auto got = read_all(in.fd, data, bytes); !got)
        return fail("read " + file.path + ": " + got.error().what);
    if (full) return file;
    for (size_t i = 0; i < stem.elements; ++i) {
        uint16_t half;
        std::memcpy(&half, data + 2 * i, 2);
        file.values[i] = widen_half(half);
    }
    return file;
}

Result<void> pack(const WeightSpec& spec, WeightFile& file)
{
    const Stem stem = parse(spec.stem);
    if (const auto packed = pack_values(layout(spec.recipe, stem), file.values, stem.c); !packed)
        return fail(file.path + ": " + packed.error().what);
    return {};
}

float widen_half(uint16_t half)
{
    const uint32_t sign = uint32_t(half & 0x8000) << 16, e = half >> 10 & 31, m = half & 1023;
    if (e) return from_bits(sign | (e == 31 ? 255 : e + 112) << 23 | m << 13);
    if (!m) return from_bits(sign);
    // A subnormal: shifted until bit 10 is set, the exponent lowered as often.
    const unsigned shift = unsigned(std::countl_zero(m)) - 21;
    return from_bits(sign | (113 - shift) << 23 | (m << shift & 1023) << 13);
}

std::expected<uint8_t, Flaw> exact_fp8(float value)
{
    const uint32_t a = bits(value) & 0x7fffffff;
    const uint8_t sign = uint8_t(bits(value) >> 24 & 128);
    if (!a) return sign;
    if (a >= 0x7f800000) return std::unexpected(Flaw::kFp8Nonfinite);
    if (a < 0x3c800000) {
        // Below 2^-6: an E4M3 subnormal, q * 2^-9 with q 1..7.
        const float q = from_bits(a) * 512.f;
        if (q < 1 || q > 7 || q != float(uint32_t(q))) return std::unexpected(Flaw::kFp8Subnormal);
        return uint8_t(sign | uint8_t(q));
    }
    const int exponent = int(a >> 23) - 127 + 7;
    const uint32_t mantissa = a >> 20 & 7;
    // Exponent 15 with mantissa 7 is NaN.
    if (exponent > 15 || a & 0xfffff || (exponent == 15 && mantissa == 7))
        return std::unexpected(Flaw::kFp8Inexact);
    return uint8_t(sign | exponent << 3 | mantissa);
}

std::expected<uint16_t, Flaw> exact_half(float value)
{
    const uint32_t a = bits(value) & 0x7fffffff;
    const uint16_t sign = uint16_t(bits(value) >> 16 & 0x8000);
    if (!a) return sign;
    if (a >= 0x7f800000) return std::unexpected(Flaw::kHalfNonfinite);
    const int e = int(a >> 23) - 127;
    if (e > 15) return std::unexpected(Flaw::kHalfOverflow);
    if (e >= -14) {
        if (a & 8191) return std::unexpected(Flaw::kHalfInexact);
        return uint16_t(sign | (e + 15) << 10 | (a >> 13 & 1023));
    }
    // A binary16 subnormal: q * 2^-24 with q 1..1023.
    const float q = from_bits(a) * 16777216.f;
    if (q < 1 || q > 1023 || q != float(uint32_t(q))) return std::unexpected(Flaw::kHalfSubnormal);
    return uint16_t(sign | uint16_t(q));
}

uint16_t round_half(float value)
{
    const uint32_t a = bits(value) & 0x7fffffff;
    const uint16_t sign = uint16_t(bits(value) >> 16 & 0x8000);
    const int e = int(a >> 23) - 127;
    if (a >= 0x7f800000) return uint16_t(sign | 0x7c00 | (a & 0x7fffff ? 0x200 : 0));
    if (e < -25) return sign;
    if (e < -14) {
        // A binary16 subnormal.
        const uint32_t m = (a & 0x7fffff) | 0x800000, n = uint32_t(-e - 1), rest = m & ((1u << n) - 1),
                       half_way = 1u << (n - 1);
        const uint32_t q = (m >> n) + (rest > half_way || (rest == half_way && m >> n & 1));
        return uint16_t(sign | q);
    }
    if (e > 15) return uint16_t(sign | 0x7c00);
    const uint32_t h = ((a + 0xfff + (a >> 13 & 1)) >> 13) - 0x1c000;
    return uint16_t(sign | (h >= 0x7c00 ? 0x7c00 : h));
}

float saturate_fp8(float value)
{
    const uint32_t a = bits(value) & 0x7fffffff, sign = bits(value) & 0x80000000;
    if (!a) return 0.f;
    if (a >= 0x43e00000) return from_bits(sign | 0x43e00000);
    if (a < 0x3c800000) {
        // An E4M3 subnormal, q * 2^-9, rounded to nearest even.
        const float scaled = from_bits(a) * 512.f;
        uint32_t q = uint32_t(scaled);
        const float rest = scaled - float(q);
        q += rest > .5f || (rest == .5f && q & 1);
        const float magnitude = float(q) / 512.f;
        return sign ? -magnitude : magnitude;
    }
    const uint32_t rounded = (a + 0x7ffff + (a >> 20 & 1)) & 0xfff00000;
    return from_bits(sign | (rounded > 0x43e00000 ? 0x43e00000 : rounded));
}

float scale_piece(float value)
{
    const float magnitude = std::fabs(value);
    if (!(magnitude < .015625f)) return saturate_fp8(value);
    const float scaled = magnitude * 512.f;
    uint32_t q = uint32_t(scaled);
    const float rest = scaled - float(q);
    q += rest > .5f || (rest == .5f && q & 1);
    // The device keeps a piece below 2^-6 subnormal too: 8 * 2^-9 becomes 7 * 2^-9.
    if (q > 7) q = 7;
    return from_bits(bits(float(q) / 512.f) | (bits(value) & 0x80000000));
}

Result<void> pack_fp8_regions(std::vector<float>& values, std::span<const Region> regions)
{
    auto* codes = reinterpret_cast<uint8_t*>(values.data());
    // Byte 4*first + i overwrites only values at or before first + i, which are
    // read already.
    for (const Region& r : regions)
        for (size_t i = 0; i < r.count; ++i) {
            const auto code = exact_fp8(values[r.first + i]);
            if (!code) return bad_element(r.first + i, code.error());
            codes[4 * r.first + i] = *code;
        }
    return {};
}

void fragment_tiles(std::vector<float>& values, size_t start, size_t rows, size_t columns)
{
    // Tile (n/16, k/32) is 512 bytes: runs of 8 k of one row, by k%32/8, then n%16.
    auto* tiles = reinterpret_cast<uint8_t*>(values.data() + start);
    const std::vector<uint8_t> matrix(tiles, tiles + rows * columns);
    for (size_t n = 0; n < rows; ++n)
        for (size_t k = 0; k < columns; k += 8)
            std::memcpy(tiles + ((n / 16) * (columns / 32) + k / 32) * 512 + (k % 32 / 8 * 16 + n % 16) * 8,
                        &matrix[n * columns + k], 8);
}

void append_c32_diagonals(std::vector<float>& values)
{
    // Byte (gr*16 + rc)*8 + e of tile (part*2 + ci)*2 + kt is piece PART of the
    // scale of column c = ci*16 + rc where k = kt*16 + gr*8 + e is c, else 0.
    values.resize(8736 + kC32DiagonalFloats);
    auto* tiles = reinterpret_cast<uint8_t*>(values.data() + 8736);
    for (unsigned c = 0; c < 32; ++c) {
        const unsigned ci = c / 16, rc = c % 16;
        float remaining = values[8704 + c];
        for (unsigned part = 0; part < 3; ++part) {
            const float piece = scale_piece(remaining);
            remaining -= piece;
            tiles[((part * 2 + ci) * 2 + ci) * 512 + (rc / 8 * 16 + rc) * 8 + rc % 8] = fp8_code(piece);
        }
    }
}

void append_mh_diagonals(std::vector<float>& values, unsigned c)
{
    // Byte (gr*16 + rc)*8 + e of tile part*(c/16) + ct is piece PART of the
    // scale of column ct*16 + rc where k = gr*8 + e is rc, else 0.
    const size_t cc = size_t(c) * c, heads = c / 32, scales = 4 * cc + heads * 4096 + heads, base = values.size();
    values.resize(base + 24 * size_t(c));
    auto* tiles = reinterpret_cast<uint8_t*>(values.data() + base);
    for (unsigned column = 0; column < c; ++column) {
        const unsigned ct = column / 16, rc = column % 16;
        float remaining = values[scales + column];
        for (unsigned part = 0; part < 3; ++part) {
            const float piece = scale_piece(remaining);
            remaining -= piece;
            tiles[(part * (c / 16) + ct) * 512 + (rc / 8 * 16 + rc) * 8 + rc % 8] = fp8_code(piece);
        }
    }
}

Result<void> check_grouped_contract(const std::vector<float>& values, unsigned c)
{
    // The contraction, c rows of 4c after the expansion: row r may be nonzero
    // only in k 128*(r/32) .. 128*(r/32)+127.
    const size_t matrix = 4 * size_t(c) * c;
    for (size_t row = 0; row < c; ++row)
        for (size_t k = 0; k < 4 * size_t(c); ++k)
            if (k / 128 != row / 32 && values[matrix + row * 4 * c + k] != 0.f)
                return bad_element(matrix + row * 4 * c + k, Flaw::kUngrouped);
    return {};
}

} // namespace dlsslop::hip
