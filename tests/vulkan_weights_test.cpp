// SPDX-License-Identifier: MIT
// Host test of the Vulkan network's weight packing, without a GPU or a model.
// The primitives are checked against upstream's definitions, transcribed from
// DLSSNR-AMD's tinlayout.hpp and nr_graph.cpp at 3dfdddc, and every recipe
// against digests of what upstream's NrSession::build packed from a synthetic
// model pack. The model reader is checked on packs in memory, good and
// damaged, and pack() on segments that do not fit.
#include "vulkan_pack.h"
#include "vulkan_weights.h"

#include <getopt.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {
using dlsslop::vulkan::Directory;
using dlsslop::vulkan::Recipe;
using dlsslop::vulkan::Segment;
using dlsslop::vulkan::Source;
using dlsslop::vulkan::Suffix;
using enum dlsslop::vulkan::Segment::Flag;
namespace vulkan = dlsslop::vulkan;
using vulkan_test::bytes_of;
using vulkan_test::fnv1a;
using vulkan_test::index_entry;
using vulkan_test::pack_bytes;
using vulkan_test::pack_header;
using vulkan_test::synthetic_entry;

int failures = 0;
// Whether OK; if not, the test fails with the message.
[[gnu::format(printf, 2, 3)]] bool expect(bool ok, const char* format, ...)
{
    if (ok) return true;
    std::va_list args;
    va_start(args, format);
    std::fputs("vulkan-weights test: ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    ++failures;
    return false;
}
template <class T>
bool expect_error(const dlsslop::Result<T>& result, const std::string& error)
{
    const std::string what = result ? "no error" : result.error().what;
    return expect(what == error, "expected \"%s\", got \"%s\"", error.c_str(), what.c_str());
}

// A model pack in memory, which must have been written.
struct Pack : vulkan_test::Pack {
    explicit Pack(std::string_view bytes) : vulkan_test::Pack(bytes) { expect(ok, "cannot make a model pack"); }
};

// Upstream's conversions, transcribed from tinlayout.hpp at 3dfdddc.
namespace tin {
float e4m3_to_f(uint8_t b)
{
    const float s = (b & 0x80) ? -1.0f : 1.0f;
    const int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0xF && m == 0x7) return NAN;
    if (e == 0) return s * float(m) * 0.001953125f;
    return s * (1.0f + float(m) / 8.0f) * std::ldexp(1.0f, e - 7);
}
uint8_t f_to_e4m3(float x)
{
    if (std::isnan(x)) return 0x7F;
    if (x > 448.0f) return 0x7E;
    if (x < -448.0f) return 0xFE;
    int best = 0;
    float bd = INFINITY;
    for (int i = 0; i < 256; ++i) {
        if (i == 0x7F || i == 0xFF) continue;
        const float d = std::fabs(e4m3_to_f(uint8_t(i)) - x);
        if (d < bd || (d == bd && (i & 1) == 0)) {
            bd = d;
            best = i;
        }
    }
    return uint8_t(best);
}
float f16_to_f(uint16_t h)
{
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    const int e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x3FFu;
    if (e == 0) {
        const float v = std::ldexp(float(m), -24);
        return s ? -v : v;
    }
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    float out;
    const uint32_t o = s | (uint32_t(e - 15 + 127) << 23) | (m << 13);
    std::memcpy(&out, &o, 4);
    return out;
}
uint16_t f_to_f16(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t sg = (u >> 16) & 0x8000u;
    const int32_t ex = int32_t((u >> 23) & 0xFF) - 127;
    const uint32_t mn = u & 0x7FFFFFu;
    if (ex == 128) return uint16_t(sg | 0x7C00u | (mn ? 0x200u : 0u));
    if (ex < -127) return uint16_t(sg);
    int32_t he = ex + 15;
    const uint32_t m24 = mn | 0x800000u;
    int shift = 13;
    if (he <= 0) {
        shift = 14 - he;
        he = 0;
        if (shift > 24) return uint16_t(sg);
    }
    uint32_t hi = m24 >> shift;
    const uint32_t lo = m24 & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (lo > half || (lo == half && (hi & 1u))) ++hi;
    if (he == 0) return uint16_t(hi >= 0x400u ? (sg | 0x400u) : (sg | hi));
    if (hi >= 0x800u) {
        hi >>= 1;
        ++he;
    }
    if (he >= 31) return uint16_t(sg | 0x7C00u);
    return uint16_t(sg | (uint32_t(he) << 10) | (hi & 0x3FFu));
}
} // namespace tin

void check_conversions()
{
    for (unsigned code = 0; code < 256; ++code) {
        const uint8_t expected = tin::f_to_e4m3(tin::e4m3_to_f(uint8_t(code)));
        expect(vulkan::requantise(uint8_t(code)) == expected, "requantise(%#x) is %#x, upstream's round trip %#x",
               code, vulkan::requantise(uint8_t(code)), expected);
    }
    for (unsigned h = 0; h < 65536; ++h) {
        const float wide = tin::f16_to_f(uint16_t(h));
        if (!expect(vulkan::widen_half(uint16_t(h)) == std::bit_cast<uint32_t>(wide),
                    "widen_half(%#x) differs from tin::f16_to_f", h) ||
            !expect(vulkan::recode_half(uint16_t(h)) == tin::f_to_f16(wide),
                    "recode_half(%#x) differs from tin::f_to_f16 of tin::f16_to_f", h))
            break;
    }
}

// Byte PLANE of each element's index in a ROWS x COLS matrix.
std::vector<uint8_t> index_plane(size_t rows, size_t cols, unsigned plane)
{
    std::vector<uint8_t> matrix(rows * cols);
    for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = uint8_t(i >> 8 * plane);
    return matrix;
}
// Which element of a ROWS x COLS matrix LAYOUT puts at each byte.
template <class Layout>
std::vector<uint32_t> element_at(size_t rows, size_t cols, Layout layout)
{
    std::vector<uint32_t> element(rows * cols, 0);
    std::vector<uint8_t> out(rows * cols);
    for (unsigned plane = 0; plane < 4; ++plane) {
        layout(index_plane(rows, cols, plane).data(), rows, cols, out.data());
        for (size_t i = 0; i < out.size(); ++i) element[i] |= uint32_t(out[i]) << 8 * plane;
    }
    return element;
}

void check_layouts()
{
    const struct {
        size_t rows, cols;
    } shapes[] = {{32, 16}, {64, 48}, {96, 32}, {32, 128}, {128, 32}};
    for (const auto& s : shapes) {
        const size_t bytes = s.rows * s.cols, ktiles = s.cols / 16;
        // tin::tile_blocked's order, and the weight layout 3 pass on it
        // (nr_graph.cpp:3270-3281).
        std::vector<uint32_t> tiled(bytes), paired(bytes);
        for (size_t r = 0; r < s.rows; ++r)
            for (size_t k = 0; k < s.cols; ++k)
                tiled[((r / 16) * ktiles + k / 16) * 256 + (r % 16) * 16 + (k % 16)] = uint32_t(r * s.cols + k);
        for (size_t n = 0; n < s.rows / 16; n += 2)
            for (size_t k = 0; k < ktiles; ++k)
                for (size_t lane = 0; lane < 32; ++lane)
                    for (size_t j = 0; j < 2; ++j)
                        for (size_t c = 0; c < 8; ++c)
                            paired[((n / 2) * ktiles + k) * 512 + lane * 16 + j * 8 + c] =
                                tiled[((n + j) * ktiles + k) * 256 + (lane % 16) * 16 + (lane / 16) * 8 + c];
        std::vector<bool> seen(bytes);
        for (uint32_t e : paired) seen[e] = true;
        expect(std::ranges::count(seen, true) == ptrdiff_t(bytes), "upstream's layouts of %zux%zu are not bijections",
               s.rows, s.cols);
        expect(element_at(s.rows, s.cols, vulkan::tile_blocked) == tiled,
               "tile_blocked of %zux%zu differs from upstream's order", s.rows, s.cols);
        expect(element_at(s.rows, s.cols, vulkan::npair_blocked) == paired,
               "npair_blocked of %zux%zu differs from upstream's order", s.rows, s.cols);
    }
}

// Whether BYTE(a, b, c) over A x B x C names each of the COUNT bytes from FIRST on once.
template <class Byte>
bool covers(size_t first, size_t count, size_t a, size_t b, size_t c, Byte byte, std::vector<unsigned>& hits)
{
    hits.assign(first + count, 0);
    for (size_t i = 0; i < a; ++i)
        for (size_t j = 0; j < b; ++j)
            for (size_t k = 0; k < c; ++k) {
                const size_t at = byte(i, j, k);
                if (at < first || at >= first + count) return false;
                ++hits[at];
            }
    return std::ranges::count(hits.begin() + ptrdiff_t(first), hits.end(), 1u) == ptrdiff_t(count);
}

void check_gathers()
{
    std::vector<unsigned> hits;
    auto a = [](size_t g, size_t r, size_t k) { return vulkan::ff_a(unsigned(g), unsigned(r), unsigned(k)); };
    auto q0 = [](size_t g, size_t j, size_t r) { return vulkan::ff_q0(unsigned(g), unsigned(j), unsigned(r)); };
    auto q2 = [](size_t g, size_t n, size_t j) { return vulkan::ff_q2(unsigned(g), unsigned(n), unsigned(j)); };
    expect(covers(0, 262144, 8, 64, 512, a, hits), "ff_a is not a bijection onto bytes 0..262143");
    expect(covers(262144, 131072, 8, 256, 64, q0, hits), "ff_q0 is not a bijection onto bytes 262144..393215");
    expect(covers(393216, 131072, 8, 64, 256, q2, hits), "ff_q2 is not a bijection onto bytes 393216..524287");
    expect(covers(128, 3145728, 3, 1024, 1024, vulkan::vit_qkv_weight_byte, hits),
           "vit_qkv_weight_byte is not a bijection onto bytes 128..3145855");
    auto bias = [](size_t, size_t i, size_t j) { return vulkan::bias_source(i, j); };
    expect(covers(0, 4096, 1, 64, 64, bias, hits), "bias_source is not a permutation of 4096 values");
}

// SHA-256 (FIPS 180-4) of DATA, in hexadecimal.
std::string sha256(const uint8_t* data, size_t bytes)
{
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> message(data, data + bytes);
    message.push_back(0x80);
    while (message.size() % 64 != 56) message.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) message.push_back(uint8_t(uint64_t(bytes) * 8 >> shift));
    for (size_t block = 0; block < message.size(); block += 64) {
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i)
            w[i] = uint32_t(message[block + 4 * i]) << 24 | uint32_t(message[block + 4 * i + 1]) << 16 |
                   uint32_t(message[block + 4 * i + 2]) << 8 | message[block + 4 * i + 3];
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t v[8];
        std::memcpy(v, h, sizeof v);
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = std::rotr(v[4], 6) ^ std::rotr(v[4], 11) ^ std::rotr(v[4], 25);
            const uint32_t t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
            const uint32_t s0 = std::rotr(v[0], 2) ^ std::rotr(v[0], 13) ^ std::rotr(v[0], 22);
            const uint32_t t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
            std::memmove(v + 1, v, 7 * sizeof *v);
            v[4] += t1;
            v[0] = t1 + t2;
        }
        for (size_t i = 0; i < 8; ++i) h[i] += v[i];
    }
    char hex[65];
    for (size_t i = 0; i < 8; ++i) std::snprintf(hex + 8 * i, 9, "%08x", h[i]);
    return hex;
}

void check_activation_table()
{
    const auto table = vulkan::activation_table();
    expect(sha256(reinterpret_cast<const uint8_t*>("abc"), 3) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "the test's SHA-256 is wrong");
    // The digest that nr_activation_lut.hpp states.
    expect(sha256(table.data(), table.size()) == "71eed43d50e6447a80db431d73f7d8000f0829b63c9f795f5019a037755558d5",
           "the activation table differs from activation_lut_v1");
    size_t runs = 1;
    for (size_t i = 1; i < table.size(); ++i) runs += table[i] != table[i - 1];
    expect(runs == 220, "the activation table has %zu runs, not 220", runs);
}

// The entries that the recipe cases read, with their sizes in the real model.
constexpr struct {
    Source source;
    uint32_t bytes;
} kEntries[] = {
    {{Directory::kUnpacked, 1, 0, Suffix::kMlpContract}, 4096},
    {{Directory::kUnpacked, 1, 0, Suffix::kResidualScale}, 96},
    {{Directory::kUnpacked, 1, 0, Suffix::kAttnResidualScale}, 80},
    {{Directory::kUnpacked, 1, 0, Suffix::kScalarsB}, 16},
    {{Directory::kUnpacked, 4, 0, Suffix::kResample}, 2048},
    {{Directory::kUnpacked, 5, 0, Suffix::kMlpMid}, 8192},
    {{Directory::kUnpacked, 9, 0, Suffix::kAttnOutProj}, 16384},
    {{Directory::kUnpacked, 15, 0, Suffix::kQkv}, 196608},
    {{Directory::kUnpacked, 15, 0, Suffix::kAttnPosBias}, 65536},
    {{Directory::kUnpacked, 22, 0, Suffix::kResample}, 131072},
    {{Directory::kUnpacked, 48, 0, Suffix::kResidualScale}, 544},
    {{Directory::kUnpacked, 48, 0, Suffix::kAttnResidualScale}, 528},
    {{Directory::kUnpacked, 48, 0, Suffix::kScalarsB}, 32},
    {{Directory::kUnpacked, 48, 0, Suffix::kResample}, 131072},
    {{Directory::kUnpacked, 48, 0, Suffix::kUpsampleGain}, 480},
    {{Directory::kUnpacked, 62, 0, Suffix::kResidualScale}, 160},
    {{Directory::kUnpacked, 62, 0, Suffix::kUpsampleGain}, 96},
    {{Directory::kUnpacked, 66, 0, Suffix::kResidualScale}, 96},
    {{Directory::kUnpacked, 66, 0, Suffix::kResample}, 2048},
    {{Directory::kUnpacked, 66, 0, Suffix::kUpsampleGain}, 64},
    {{Directory::kPreblock, 0, 0, Suffix::kMlpExpand}, 4096},
    {{Directory::kPreblock, 0, 0, Suffix::kAttnPosBias}, 8192},
    {{Directory::kPreblock, 0, 0, Suffix::kResidualScale}, 80},
    {{Directory::kPreblock, 0, 0, Suffix::kInputLift}, 1024},
    {{Directory::kPostblock, 70, 0, Suffix::kQkv}, 3072},
    {{Directory::kPostblock, 70, 0, Suffix::kResidualScale}, 64},
    {{Directory::kPostblock, 70, 0, Suffix::kOutProject}, 1024},
    {{Directory::kPostblock, 70, 0, Suffix::kSkipGain}, 64},
    {{Directory::kPostblock, 70, 0, Suffix::kMainGain}, 64},
    {{Directory::kSplitSwin, 23, 1, Suffix::kWeight}, 262144},
    {{Directory::kSplitSwin, 23, 1, Suffix::kSkipWeight}, 1024},
    {{Directory::kSplitSwin, 23, 2, Suffix::kQkv}, 786432},
    {{Directory::kSplitSwin, 23, 2, Suffix::kAttnPosBias}, 131072},
    {{Directory::kSplitSwin, 23, 2, Suffix::kTail}, 64},
    {{Directory::kSplitSwin, 30, 3, Suffix::kWeight}, 262144},
    {{Directory::kSplitSwin, 30, 4, Suffix::kWeight}, 524288},
    {{Directory::kSplitSwin, 39, 0, Suffix::kWeight}, 524288},
    {{Directory::kSplitSwin, 39, 0, Suffix::kSkipWeight}, 1024},
    {{Directory::kVit, 31, 0, Suffix::kWeight}, 4194304},
    {{Directory::kVit, 31, 1, Suffix::kWeight}, 4194304},
    {{Directory::kVit, 31, 1, Suffix::kSkipWeight}, 2048},
    {{Directory::kVit, 31, 4, Suffix::kWeight}, 1048576},
    {{Directory::kRecords, 23, 0, Suffix::kNone}, 524288},
    {{Directory::kRecords, 31, 2, Suffix::kNone}, 3145856},
};

// Segments as the network's plan makes them, each with the FNV-1a 64 of the
// bytes that upstream's NrSession::build (nr_graph.cpp at 3dfdddc, built with
// its rdna4.sh defines) packed for it from the synthetic pack of the real
// pack's names and sizes (synthetic_entry()). Offsets are the test's.
constexpr struct {
    Segment segment;
    uint64_t digest;
} kUpstreamCases[] = {
    {{0, 4096, Recipe::kActivations, 0, {}, 0, 0, 0},
     0x8802ec9dcae156c8u},
    {{0, 4096, Recipe::kMatrix, kRequantise | kNpair, {Directory::kPreblock, 0, 0, Suffix::kMlpExpand}, 128, 32, 0},
     0x4a1dd5bc7b7a043fu},
    {{0, 4096, Recipe::kMatrix, kRequantise | kNpair, {Directory::kUnpacked, 1, 0, Suffix::kMlpContract}, 32, 128, 0},
     0x9d7bdaef5f4f02dfu},
    {{0, 8192, Recipe::kMatrix, kRequantise | kNpair, {Directory::kUnpacked, 5, 0, Suffix::kMlpMid}, 64, 128, 0},
     0x97e5e1cb579c9af9u},
    {{0, 16384, Recipe::kMatrix, kRequantise | kNpair, {Directory::kUnpacked, 9, 0, Suffix::kAttnOutProj}, 128, 128, 0},
     0x3ab19af565304963u},
    {{0, 196608, Recipe::kMatrix, kRequantise | kNpair, {Directory::kUnpacked, 15, 0, Suffix::kQkv}, 768, 256, 0},
     0xbc91680bcd6a8cdbu},
    {{0, 3072, Recipe::kMatrix, kRequantise | kNpair, {Directory::kPostblock, 70, 0, Suffix::kQkv}, 96, 32, 0},
     0xfe5f8de627dacc64u},
    {{0, 262144, Recipe::kMatrix, kNpair, {Directory::kSplitSwin, 23, 1, Suffix::kWeight}, 512, 512, 0},
     0x7e6bf9d7e27d6949u},
    {{0, 4194304, Recipe::kMatrix, kNpair, {Directory::kVit, 31, 0, Suffix::kWeight}, 4096, 1024, 0},
     0xcf1abf13d93ba57fu},
    {{0, 4194304, Recipe::kMatrix, kNpair, {Directory::kVit, 31, 1, Suffix::kWeight}, 1024, 4096, 0},
     0xa763bab1010b8eedu},
    {{0, 1048576, Recipe::kMatrix, kNpair, {Directory::kVit, 31, 4, Suffix::kWeight}, 1024, 1024, 0},
     0x6864acfdddab50a4u},
    {{0, 524288, Recipe::kMatrix, kNpair, {Directory::kSplitSwin, 39, 0, Suffix::kWeight}, 512, 1024, 0},
     0x5dc371446e7d5da6u},
    {{0, 786432, Recipe::kMatrix, 0, {Directory::kSplitSwin, 23, 2, Suffix::kQkv}, 1536, 512, 0},
     0xef411e3648765299u},
    {{0, 2048, Recipe::kMatrix, 0, {Directory::kUnpacked, 4, 0, Suffix::kResample}, 64, 32, 0},
     0x2b72403c8fa72eebu},
    {{0, 131072, Recipe::kMatrix, 0, {Directory::kUnpacked, 22, 0, Suffix::kResample}, 512, 256, 0},
     0x8d6b636def36599cu},
    {{0, 131072, Recipe::kMatrix, 0, {Directory::kUnpacked, 48, 0, Suffix::kResample}, 256, 512, 0},
     0x1af391239f32bbccu},
    {{0, 2048, Recipe::kMatrix, 0, {Directory::kUnpacked, 66, 0, Suffix::kResample}, 32, 64, 0},
     0xf651d9728df2fb38u},
    {{0, 524288, Recipe::kMatrix, 0, {Directory::kSplitSwin, 30, 4, Suffix::kWeight}, 1024, 512, 0},
     0xbc8f88d92dd05d10u},
    {{0, 262144, Recipe::kMatrix, 0, {Directory::kSplitSwin, 30, 3, Suffix::kWeight}, 512, 512, 0},
     0x334ebdaaf7fc9b88u},
    {{0, 262144, Recipe::kFfwd, kNpair, {Directory::kRecords, 23, 0, Suffix::kNone}, 0, 0, 0},
     0x8a4b63b5a7d453dcu},
    {{0, 131072, Recipe::kFfwd, kNpair, {Directory::kRecords, 23, 0, Suffix::kNone}, 0, 0, 1},
     0x51be1a24e88a0d6au},
    {{0, 131072, Recipe::kFfwd, kNpair, {Directory::kRecords, 23, 0, Suffix::kNone}, 0, 0, 2},
     0xa961486978e0c959u},
    {{0, 3145728, Recipe::kVitQkv, kNpair, {Directory::kRecords, 31, 2, Suffix::kNone}, 3072, 1024, 0},
     0x10543654d8c5a0dbu},
    {{0, 16384, Recipe::kBias, kAffine, {Directory::kPreblock, 0, 0, Suffix::kAttnPosBias}, 0, 0, 0},
     0x29d00d9b91cd07ecu},
    {{0, 131072, Recipe::kBias, kAffine, {Directory::kUnpacked, 15, 0, Suffix::kAttnPosBias}, 0, 0, 0},
     0x6824effc03b242edu},
    {{0, 262144, Recipe::kBias, 0, {Directory::kSplitSwin, 23, 2, Suffix::kAttnPosBias}, 0, 0, 0},
     0x42e70e2381b2ca5bu},
    {{0, 128, Recipe::kScales, kPadded, {Directory::kUnpacked, 1, 0, Suffix::kResidualScale}, 0, 0, 0},
     0x566aaf25b9e292b4u},
    {{0, 1024, Recipe::kScales, kPadded, {Directory::kUnpacked, 48, 0, Suffix::kResidualScale}, 0, 0, 0},
     0x7b6555748f951705u},
    {{0, 128, Recipe::kScales, kPadded, {Directory::kPreblock, 0, 0, Suffix::kResidualScale}, 0, 0, 0},
     0x9c063e8c9f9d7a01u},
    {{0, 128, Recipe::kScales, kPadded, {Directory::kPostblock, 70, 0, Suffix::kResidualScale}, 0, 0, 0},
     0xd9b3365b18e4cfceu},
    {{0, 128, Recipe::kScales, 0, {Directory::kUnpacked, 1, 0, Suffix::kAttnResidualScale}, 0, 0, 0},
     0x397d969de084bff7u},
    {{0, 1024, Recipe::kHalf, 0, {Directory::kSplitSwin, 23, 1, Suffix::kSkipWeight}, 0, 0, 0},
     0xc53e7b0d685f0ac1u},
    {{0, 2048, Recipe::kHalf, 0, {Directory::kVit, 31, 1, Suffix::kSkipWeight}, 0, 0, 0},
     0x407ca1cfdef06e7cu},
    {{0, 1024, Recipe::kHalf, 0, {Directory::kSplitSwin, 39, 0, Suffix::kSkipWeight}, 0, 0, 0},
     0x6b03ceedce682fa6u},
    {{0, 64, Recipe::kHalf, kExact, {Directory::kPostblock, 70, 0, Suffix::kSkipGain}, 0, 0, 0},
     0x1aa798b71578e659u},
    {{0, 64, Recipe::kHalf, kExact, {Directory::kPostblock, 70, 0, Suffix::kMainGain}, 0, 0, 0},
     0xbb4ceca14b37fa79u},
    {{0, 512, Recipe::kHalf, kGainTail, {Directory::kUnpacked, 48, 0, Suffix::kUpsampleGain}, 0, 0, 0},
     0xa87dde9855554156u},
    {{0, 128, Recipe::kHalf, kGainTail, {Directory::kUnpacked, 62, 0, Suffix::kUpsampleGain}, 0, 0, 0},
     0x02f6426babd71be1u},
    {{0, 64, Recipe::kHalf, kGainTail, {Directory::kUnpacked, 66, 0, Suffix::kUpsampleGain}, 0, 0, 0},
     0x047542155cf3bf2cu},
    {{0, 1024, Recipe::kDiagonal, kPadded, {Directory::kUnpacked, 1, 0, Suffix::kResidualScale}, 0, 0, 0},
     0x1ccf2228fcda20f6u},
    {{0, 1024, Recipe::kDiagonal, kPadded, {Directory::kPostblock, 70, 0, Suffix::kResidualScale}, 0, 0, 0},
     0x5a01315ad67633f3u},
    {{0, 8192, Recipe::kDiagonal, 0, {Directory::kUnpacked, 48, 0, Suffix::kAttnResidualScale}, 0, 0, 0},
     0x58c442f0c6b64f76u},
    {{0, 16384, Recipe::kDiagonal, 0, {Directory::kSplitSwin, 23, 1, Suffix::kSkipWeight}, 0, 0, 0},
     0x06331066856cf841u},
    {{0, 32, Recipe::kBytes, kExact, {Directory::kUnpacked, 48, 0, Suffix::kScalarsB}, 0, 0, 0},
     0x9528a1d79173b478u},
    {{0, 16, Recipe::kBytes, kExact, {Directory::kUnpacked, 1, 0, Suffix::kScalarsB}, 0, 0, 0},
     0xbc7a39464f7bacb7u},
    {{0, 64, Recipe::kBytes, kExact, {Directory::kSplitSwin, 23, 2, Suffix::kTail}, 0, 0, 0},
     0xfff8d3891fe7d480u},
    {{0, 1024, Recipe::kBytes, kExact, {Directory::kPreblock, 0, 0, Suffix::kInputLift}, 0, 0, 0},
     0xb3355eeae2e0d1acu},
    {{0, 1024, Recipe::kBytes, kExact, {Directory::kPostblock, 70, 0, Suffix::kOutProject}, 0, 0, 0},
     0x474e79c7770e7da1u},
    {{0, 128, Recipe::kBytes, 0, {Directory::kRecords, 31, 2, Suffix::kNone}, 0, 0, 0},
     0xc492902d0a05fb75u},
    {{0, 1024, Recipe::kLift, kExact, {Directory::kPreblock, 0, 0, Suffix::kInputLift}, 0, 0, 0},
     0x93ddcf3b4ed94d70u},
};

// A pack of the case entries in the order of kEntries.
std::string synthetic_pack()
{
    std::vector<vulkan_test::Entry> entries;
    for (const auto& e : kEntries) {
        std::string name = vulkan::entry_name(e.source);
        entries.push_back({name, synthetic_entry(name, e.bytes)});
    }
    return pack_bytes(entries);
}

void check_recipes(const vulkan::Model& model)
{
    // Every case after the one before, with a gap of 16 bytes between them.
    std::vector<Segment> segments;
    size_t at = 0;
    for (const auto& c : kUpstreamCases) {
        segments.push_back(c.segment);
        segments.back().offset = uint32_t(at);
        at += c.segment.bytes + 16;
    }
    std::vector<uint8_t> blob(at + 7, 0xaa);
    const auto packed = vulkan::pack(segments, {}, model, blob);
    if (!expect(bool(packed), "packing the cases: %s", packed ? "" : packed.error().what.c_str())) return;
    for (size_t i = 0; i < segments.size(); ++i) {
        const Segment& s = segments[i];
        const std::string name =
            s.recipe == Recipe::kActivations ? "the activation table" : vulkan::entry_name(s.source);
        expect(fnv1a(blob.data() + s.offset, s.bytes) == kUpstreamCases[i].digest,
               "case %zu, %s: packed bytes differ from upstream's", i, name.c_str());
        const uint8_t* gap = blob.data() + s.offset + s.bytes;
        expect(std::all_of(gap, gap + 16, [](uint8_t b) { return b == 0; }), "case %zu: the gap after it is not zeroed",
               i);
    }
    expect(std::all_of(blob.end() - 7, blob.end(), [](uint8_t b) { return b == 0; }), "the blob's end is not zeroed");
}

void check_tables(const vulkan::Model& model)
{
    const uint32_t tables[] = {1, 2, 3, 0xdeadbeef, 5};
    const Segment segments[] = {{4, 8, Recipe::kTable, 0, {}, 0, 0, 2}, {16, 4, Recipe::kZeros, 0, {}, 0, 0, 0}};
    std::vector<uint8_t> blob(24, 0xaa);
    const auto packed = vulkan::pack(segments, tables, model, blob);
    uint32_t words[6];
    std::memcpy(words, blob.data(), sizeof words);
    expect(packed && words[0] == 0 && words[1] == 3 && words[2] == 0xdeadbeef && !words[3] && !words[4] && !words[5],
           "a table and zeros are not packed as words 0 3 0xdeadbeef 0 0 0");
}

// Pack errors: segments that overlap, leave the blob or do not hold what
// their recipe writes, and entries of the wrong size.
void check_pack_errors(const vulkan::Model& model)
{
    const uint32_t tables[4] = {};
    std::vector<uint8_t> blob(1 << 22);
    auto pack = [&](std::initializer_list<Segment> segments) {
        return vulkan::pack({segments.begin(), segments.size()}, tables, model, blob);
    };
    auto bad = [](uint32_t offset) { return "network plan: bad weight segment at " + std::to_string(offset); };
    const Segment zeros{16, 32, Recipe::kZeros, 0, {}, 0, 0, 0};
    expect_error(pack({zeros, {40, 8, Recipe::kZeros, 0, {}, 0, 0, 0}}), bad(40));
    expect_error(pack({zeros, {8, 8, Recipe::kZeros, 0, {}, 0, 0, 0}}), bad(8));
    expect_error(pack({{4194302, 4, Recipe::kZeros, 0, {}, 0, 0, 0}}), bad(4194302));
    expect_error(pack({{4194305, 0, Recipe::kZeros, 0, {}, 0, 0, 0}}), bad(4194305));
    expect_error(pack({{0, 12, Recipe::kTable, 0, {}, 0, 0, 2}}), bad(0));
    const Source expand{Directory::kPreblock, 0, 0, Suffix::kMlpExpand};
    expect_error(pack({{0, 4096, Recipe::kMatrix, kNpair, expand, 16, 256, 0}}), bad(0));
    expect_error(pack({{0, 4000, Recipe::kMatrix, 0, expand, 128, 32, 0}}), bad(0));
    const Source ffwd{Directory::kRecords, 23, 0, Suffix::kNone};
    expect_error(pack({{0, 256, Recipe::kFfwd, 0, ffwd, 0, 0, 3}}), bad(0));
    const Source scales{Directory::kUnpacked, 1, 0, Suffix::kResidualScale};
    expect_error(pack({{0, 100, Recipe::kDiagonal, 0, scales, 0, 0, 0}}), bad(0));

    // Short entries, and entries of the wrong size where it must be exact.
    const std::string& path = model.path();
    expect_error(pack({{0, 8192, Recipe::kMatrix, 0, expand, 256, 32, 0}}),
                 path + ": unpacked-preblock/block0.layer0.layer.mlp_expand.bin has 4096 bytes, 8192 needed");
    const Source scalars{Directory::kUnpacked, 1, 0, Suffix::kScalarsB};
    const Source scalars48{Directory::kUnpacked, 48, 0, Suffix::kScalarsB};
    expect_error(pack({{0, 32, Recipe::kBytes, kExact, scalars, 0, 0, 0}}),
                 path + ": unpacked/block1.layer0.layer.scalars_b.bin has 16 bytes, expected 32");
    expect_error(pack({{0, 16, Recipe::kBytes, kExact, scalars48, 0, 0, 0}}),
                 path + ": unpacked/block48.layer0.layer.scalars_b.bin has 32 bytes, expected 16");
    const Source bias{Directory::kPreblock, 0, 0, Suffix::kAttnPosBias};
    expect_error(pack({{0, 32768, Recipe::kBias, 0, bias, 0, 0, 0}}),
                 path + ": unpacked-preblock/block0.layer0.layer.attn_pos_bias.bin has 8192 bytes, 16384 needed");
    const Source weight{Directory::kSplitSwin, 23, 1, Suffix::kWeight};
    expect_error(pack({{0, 131072, Recipe::kFfwd, 0, weight, 0, 0, 1}}),
                 path + ": unpacked-splitswin/block23.layer1.layer.weight.bin has 262144 bytes, 524288 needed");
    expect_error(pack({{0, 3145728, Recipe::kVitQkv, 0, ffwd, 3072, 1024, 0}}),
                 path + ": inventory/weights/block23.layer0.layer.bin has 524288 bytes, 3145856 needed");
    // Block 1's residual scales start with eight zeros, so 41 of them are read
    // from the ninth on; its attention residual scales from the first.
    expect_error(pack({{0, 164, Recipe::kScales, kPadded, scales, 0, 0, 0}}),
                 path + ": unpacked/block1.layer0.layer.residual_scale.bin has 96 bytes, 98 needed");
    const Source attention{Directory::kUnpacked, 1, 0, Suffix::kAttnResidualScale};
    expect_error(pack({{0, 1536, Recipe::kDiagonal, kPadded, attention, 0, 0, 0}}),
                 path + ": unpacked/block1.layer0.layer.attn_residual_scale.bin has 80 bytes, 96 needed");
    // Block 62's upsample gain holds 48 values: for 272 it needs the last 224
    // of its 80 residual scales.
    const Source gain62{Directory::kUnpacked, 62, 0, Suffix::kUpsampleGain};
    const Source gain48{Directory::kUnpacked, 48, 0, Suffix::kUpsampleGain};
    expect_error(pack({{0, 544, Recipe::kHalf, kGainTail, gain62, 0, 0, 0}}),
                 path + ": unpacked/block62.layer0.layer.residual_scale.bin has 160 bytes, 448 needed");
    expect_error(pack({{0, 1024, Recipe::kHalf, 0, gain48, 0, 0, 0}}),
                 path + ": unpacked/block48.layer0.layer.upsample_gain.bin has 480 bytes, 1024 needed");
    const Source qkv{Directory::kPostblock, 70, 0, Suffix::kQkv};
    expect_error(pack({{0, 1024, Recipe::kLift, kExact, qkv, 0, 0, 0}}),
                 path + ": unpacked-postblock/block70.layer0.layer.qkv.bin has 3072 bytes, expected 1024");
    expect_error(pack({{0, 64, Recipe::kBytes, 0, {Directory::kVit, 31, 3, Suffix::kWeight}, 0, 0, 0}}),
                 path + ": missing unpacked-vit/block31.layer3.layer.weight.bin");
}

// Residual scales whose first eight values are -0: upstream compared each
// with 0.0f (nr_graph.cpp:1825-1826), so it read them from the ninth on too.
void check_negative_zeros()
{
    constexpr size_t kCount = 32;
    const Source scales{Directory::kUnpacked, 1, 0, Suffix::kResidualScale};
    uint16_t halves[8 + kCount];
    std::fill_n(halves, 8, uint16_t(0x8000));
    vulkan_test::Random random{0x5eed};
    for (size_t i = 8; i < std::size(halves); ++i) halves[i] = uint16_t(random.next());
    const Pack pack(pack_bytes({{vulkan::entry_name(scales), {reinterpret_cast<const char*>(halves), sizeof halves}}}));
    const auto model = vulkan::Model::open(pack.path);
    if (!expect(bool(model), "opening the pack of -0 scales: %s", model ? "" : model.error().what.c_str())) return;
    const Segment segments[] = {{0, 4 * kCount, Recipe::kScales, kPadded, scales, 0, 0, 0},
                                {4 * kCount, 32 * kCount, Recipe::kDiagonal, kPadded, scales, 0, 0, 0}};
    std::vector<uint8_t> blob(36 * kCount, 0xaa), expected(blob.size(), 0);
    for (size_t i = 0; i < kCount; ++i) {
        const float value = tin::f16_to_f(halves[8 + i]);
        const uint16_t half = tin::f_to_f16(value);
        std::memcpy(&expected[4 * i], &value, 4);
        // Block i / 16, row and column i % 16.
        std::memcpy(&expected[4 * kCount + 2 * (i / 16 * 256 + i % 16 * 17)], &half, 2);
    }
    const auto packed = vulkan::pack(segments, {}, *model, blob);
    expect(packed && blob == expected, "residual scales after eight -0 values are not read from the ninth on");
}

// The model reader on packs good and damaged.
void check_model()
{
    const Source tail{Directory::kSplitSwin, 23, 2, Suffix::kTail}, record{Directory::kRecords, 31, 2, Suffix::kNone};
    const std::string tail_name = vulkan::entry_name(tail), record_name = vulkan::entry_name(record);
    expect(tail_name == "unpacked-splitswin/block23.layer2.layer.tail.bin" &&
               record_name == "inventory/weights/block31.layer2.layer.bin",
           "entry names differ from upstream's: %s, %s", tail_name.c_str(), record_name.c_str());
    {
        const Pack pack(pack_bytes({{tail_name, "tail bytes"}, {record_name, "record"}}));
        const auto model = vulkan::Model::open(pack.path);
        if (!expect(bool(model), "a good pack is refused: %s", model ? "" : model.error().what.c_str())) return;
        std::vector<uint8_t> scratch(64, 0xaa);
        auto read = [&](const Source& source) {
            const auto bytes = model->read(source, scratch);
            return bytes ? std::string(bytes->begin(), bytes->end()) : bytes.error().what;
        };
        expect(read(record) == "record" && read(tail) == "tail bytes" && scratch.size() == 64,
               "a good pack's entries are not read into the start of the scratch");
        expect_error(model->read({Directory::kSplitSwin, 23, 2, Suffix::kQkv}, scratch),
                     pack.path + ": missing unpacked-splitswin/block23.layer2.layer.qkv.bin");
    }
    auto refused = [](const std::string& bytes, const std::string& error) {
        const Pack pack(bytes);
        expect_error(vulkan::Model::open(pack.path), pack.path + ": " + error);
    };
    const std::string entry = index_entry(tail_name, 16 + 4 + tail_name.size() + 16, 10);
    refused("NRMODEL", "not a model pack");
    refused("NRMODEL2" + bytes_of(uint64_t(0)), "not a model pack");
    refused(pack_header(2) + entry + "tail bytes", "index is damaged");
    refused(pack_header(1) + entry.substr(0, entry.size() - 1), "index is damaged");
    refused(pack_header(1) + bytes_of(uint32_t(4097)) + std::string(4097 + 16, 'a'), "index is damaged");
    refused(pack_header(1) + entry + "tail byte", "entry " + tail_name + " past the end");
    refused(pack_header(1) + index_entry(tail_name, ~uint64_t(0), 2), "entry " + tail_name + " past the end");
    refused(pack_bytes({{tail_name, "a"}, {record_name, "b"}, {tail_name, "c"}}),
            "entry " + tail_name + " listed twice");
    expect_error(vulkan::Model::open("/nonexistent/dlssnr.bin"),
                 std::string("cannot read /nonexistent/dlssnr.bin: ") + std::strerror(ENOENT));
}
} // namespace

int main(int argc, char** argv)
{
    const option options[] = {{"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+h", options, nullptr)) != -1;) {
        if (code != 'h') return 2;
        std::puts("Usage: vulkan-weights-test [OPTION]...\n"
                  "Checks the Vulkan network's weight packing against upstream's. No GPU or model needed.\n"
                  " -h, --help  Show help (default: off)");
        return 0;
    }
    if (optind != argc) return 2;
    check_conversions();
    check_layouts();
    check_gathers();
    check_activation_table();
    check_model();
    check_negative_zeros();
    const Pack pack(synthetic_pack());
    const auto model = vulkan::Model::open(pack.path);
    if (expect(bool(model), "opening the synthetic pack: %s", model ? "" : model.error().what.c_str())) {
        check_recipes(*model);
        check_tables(*model);
        check_pack_errors(*model);
    }
    if (!failures) std::puts("vulkan-weights test: every check passed");
    return failures ? 1 : 0;
}
