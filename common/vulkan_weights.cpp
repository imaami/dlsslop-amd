// SPDX-License-Identifier: MIT
#include "vulkan_weights.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

namespace dlsslop::vulkan {

static_assert(std::endian::native == std::endian::little, "the blob holds little-endian values");

namespace {
constexpr const char* kDirectoryNames[] = {"unpacked", "unpacked-preblock", "unpacked-postblock",
                                           "unpacked-splitswin", "unpacked-vit", "inventory/weights"};
static_assert(std::size(kDirectoryNames) == size_t(Directory::kRecords) + 1);
constexpr const char* kSuffixNames[] = {"", ".mlp_expand", ".mlp_contract", ".mlp_mid", ".qkv", ".attn_out_proj",
    ".attn_pos_bias", ".residual_scale", ".attn_residual_scale", ".scalars_b", ".resample", ".upsample_gain",
    ".input_lift", ".out_project", ".skip_gain", ".main_gain", ".weight", ".skip_weight", ".tail"};
static_assert(std::size(kSuffixNames) == size_t(Suffix::kTail) + 1);
// Room for the longest name of a Source and its terminator.
constexpr size_t kNameSize = 80;

// SOURCE's name in NAME; its length.
size_t format(const Source& s, char (&name)[kNameSize])
{
    return size_t(std::snprintf(name, kNameSize, "%s/block%u.layer%u.layer%s.bin",
                                kDirectoryNames[size_t(s.directory)], s.block, s.layer,
                                kSuffixNames[size_t(s.suffix)]));
}

// The activation table as runs of one value: its 4096 bytes in 220 runs.
constexpr struct {
    uint8_t value;
    uint16_t count;
} kActivationRuns[] = {
    {0x00, 328}, {0x01, 99}, {0x02, 46}, {0x03, 36}, {0x04, 19}, {0x05, 18}, {0x06, 17}, {0x07, 16}, {0x08, 8},
    {0x09, 9}, {0x0a, 9}, {0x0b, 9}, {0x0c, 8}, {0x0d, 9}, {0x0e, 9}, {0x0f, 4}, {0x10, 7}, {0x11, 8}, {0x12, 9},
    {0x13, 8}, {0x14, 9}, {0x15, 8}, {0x16, 9}, {0x17, 5}, {0x18, 6}, {0x19, 9}, {0x1a, 8}, {0x1b, 8}, {0x1c, 8},
    {0x1d, 8}, {0x1e, 8}, {0x1f, 7}, {0x20, 6}, {0x21, 8}, {0x22, 8}, {0x23, 7}, {0x24, 8}, {0x25, 7}, {0x26, 8},
    {0x27, 7}, {0x28, 7}, {0x29, 8}, {0x2a, 6}, {0x2b, 7}, {0x2c, 7}, {0x2d, 7}, {0x2e, 6}, {0x2f, 6}, {0x30, 10},
    {0x31, 8}, {0x32, 6}, {0x33, 6}, {0x34, 6}, {0x35, 5}, {0x36, 6}, {0x37, 5}, {0x38, 8}, {0x39, 10}, {0x3a, 9},
    {0x3b, 5}, {0x3c, 5}, {0x3d, 4}, {0x3e, 5}, {0x3f, 4}, {0x40, 7}, {0x41, 8}, {0x42, 9}, {0x43, 8}, {0x44, 8},
    {0x45, 4}, {0x46, 4}, {0x47, 4}, {0x48, 5}, {0x49, 8}, {0x4a, 8}, {0x4b, 8}, {0x4c, 8}, {0x4d, 8}, {0x4e, 8},
    {0x4f, 4}, {0x50, 7}, {0x51, 9}, {0x52, 9}, {0x53, 9}, {0x54, 9}, {0x55, 9}, {0x56, 8}, {0x57, 4}, {0x58, 7},
    {0x59, 9}, {0x5a, 9}, {0x5b, 9}, {0x5c, 9}, {0x5d, 9}, {0x5e, 8}, {0x5f, 4}, {0x60, 7}, {0x61, 9}, {0x62, 9},
    {0x63, 9}, {0x64, 9}, {0x65, 9}, {0x66, 8}, {0x67, 4}, {0x68, 7}, {0x69, 9}, {0x6a, 9}, {0x6b, 9}, {0x6c, 9},
    {0x6d, 9}, {0x6e, 8}, {0x6f, 4}, {0x70, 7}, {0x71, 9}, {0x72, 9}, {0x73, 9}, {0x74, 9}, {0x75, 9}, {0x76, 8},
    {0x77, 4}, {0x78, 7}, {0x79, 9}, {0x7a, 9}, {0x7b, 9}, {0x7c, 9}, {0x7d, 9}, {0x7e, 463}, {0x7f, 56},
    {0xff, 64}, {0x80, 328}, {0x81, 99}, {0x82, 47}, {0x83, 36}, {0x84, 19}, {0x85, 18}, {0x86, 18}, {0x87, 15},
    {0x88, 9}, {0x89, 9}, {0x8a, 9}, {0x8b, 9}, {0x8c, 9}, {0x8d, 10}, {0x8e, 7}, {0x8f, 4}, {0x90, 7}, {0x91, 10},
    {0x92, 9}, {0x93, 10}, {0x94, 9}, {0x95, 9}, {0x96, 7}, {0x97, 5}, {0x98, 7}, {0x99, 10}, {0x9a, 10}, {0x9b, 9},
    {0x9c, 11}, {0x9d, 9}, {0x9e, 5}, {0x9f, 6}, {0xa0, 8}, {0xa1, 10}, {0xa2, 11}, {0xa3, 11}, {0xa4, 12},
    {0xa5, 6}, {0xa6, 7}, {0xa7, 6}, {0xa8, 10}, {0xa9, 14}, {0xaa, 15}, {0xab, 11}, {0xac, 10}, {0xad, 10},
    {0xae, 12}, {0xaf, 16}, {0xb0, 64}, {0xaf, 10}, {0xae, 5}, {0xad, 4}, {0xac, 4}, {0xab, 4}, {0xaa, 3},
    {0xa9, 3}, {0xa8, 3}, {0xa7, 1}, {0xa6, 2}, {0xa5, 2}, {0xa4, 2}, {0xa3, 2}, {0xa2, 1}, {0xa1, 2}, {0xa0, 1},
    {0x9f, 2}, {0x9e, 1}, {0x9d, 1}, {0x9c, 1}, {0x9b, 1}, {0x9a, 1}, {0x99, 1}, {0x98, 1}, {0x97, 1}, {0x95, 1},
    {0x94, 1}, {0x92, 1}, {0x91, 1}, {0x90, 1}, {0x8d, 1}, {0x8c, 1}, {0x89, 1}, {0x87, 1}, {0x86, 1}, {0x85, 1},
    {0x83, 2}, {0x81, 1}, {0x80, 1}, {0x81, 1}, {0x80, 897}, {0xff, 64}};

// A C=512 FFN record, and its parts as matrices of eight groups of rows
// (upstream: A, Q0 and Q2 of the F_FFWD3 lowering, nr_graph.cpp:2706-2747).
constexpr size_t kFfwdRecord = 524288;
constexpr struct {
    uint16_t rows, cols;
} kFfwdParts[] = {{8 * 64, 512}, {8 * 256, 64}, {8 * 64, 256}};
// A ViT QKV record: 128 bytes of scales, then Q, K and V of 1024 x 1024.
constexpr size_t kQkvScales = 128, kQkvRows = 3072, kQkvCols = 1024;
// Math profile 3's exponent affine on a Swin position bias (nr_graph.cpp:1725-1726).
constexpr float kBiasScale = 0.044921875f, kBiasOffset = 1.30078125f;

uint16_t load_half(const uint8_t* p)
{
    uint16_t h;
    std::memcpy(&h, p, 2);
    return h;
}
void store_half(uint8_t* p, uint16_t h) { std::memcpy(p, &h, 2); }
void store_word(uint8_t* p, uint32_t w) { std::memcpy(p, &w, 4); }

[[gnu::cold, gnu::noinline]] Result<Model> unreadable(const std::string& path)
{
    const int error = errno;
    return fail("cannot read " + path + ": " + std::strerror(error));
}
// BYTES at OFFSET of the model at PATH, or why not.
Result<void> load(int fd, const std::string& path, uint64_t offset, void* data, size_t bytes)
{
    if (auto got = read_at(fd, offset, data, bytes); !got) return fail("cannot read " + path + ": " + got.error().what);
    return {};
}
[[gnu::cold, gnu::noinline]] Result<Model> damaged(const std::string& path)
{
    return fail(path + ": index is damaged");
}
[[gnu::cold, gnu::noinline]] Result<void> bad_size(const Model& model, const Source& source, size_t size,
                                                   size_t needed, bool exact)
{
    return fail(model.path() + ": " + entry_name(source) + " has " + std::to_string(size) + " bytes, " +
                (exact ? "expected " + std::to_string(needed) : std::to_string(needed) + " needed"));
}
[[gnu::cold, gnu::noinline]] Result<void> bad_segment(const Segment& s)
{
    return fail("network plan: bad weight segment at " + std::to_string(s.offset));
}

// Whether S holds what its recipe writes, inside TABLES words for kTable.
bool shaped(const Segment& s, size_t tables)
{
    switch (s.recipe) {
    case Recipe::kZeros:
    case Recipe::kBytes: return true;
    case Recipe::kTable: return s.bytes % 4 == 0 && s.index <= tables && s.bytes / 4 <= tables - s.index;
    case Recipe::kActivations: return s.bytes == 4096;
    case Recipe::kMatrix:
        return s.rows % (s.flags & Segment::kNpair ? 32 : 16) == 0 && s.cols % 16 == 0 &&
               s.bytes == size_t(s.rows) * s.cols;
    case Recipe::kFfwd:
        return s.index < std::size(kFfwdParts) &&
               s.bytes == size_t(kFfwdParts[s.index].rows) * kFfwdParts[s.index].cols;
    case Recipe::kVitQkv: return s.bytes == kQkvRows * kQkvCols;
    case Recipe::kBias: return s.bytes % 16384 == 0;
    case Recipe::kScales: return s.bytes % 4 == 0;
    case Recipe::kHalf: return s.bytes % 2 == 0;
    case Recipe::kDiagonal: return s.bytes % 512 == 0;
    case Recipe::kLift: return s.bytes == 1024;
    }
    return false;
}

// A matrix of GROUPS x ROWS rows of COLS codes, each code the byte of RECORD
// that BYTE(group, row, col) names.
template <auto byte>
void gather(const uint8_t* record, size_t groups, size_t rows, size_t cols, uint8_t* matrix)
{
    for (size_t g = 0; g < groups; ++g)
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) *matrix++ = record[byte(g, r, c)];
}

// COUNT binary16 values from IN through recode_half() to OUT.
void recode_halves(const uint8_t* in, size_t count, uint8_t* out)
{
    for (size_t i = 0; i < count; ++i) store_half(out + 2 * i, recode_half(load_half(in + 2 * i)));
}

// Whether the first eight binary16 values at P are zero.
bool zero_eight(const uint8_t* p)
{
    for (size_t i = 0; i < 8; ++i)
        if (load_half(p + 2 * i) & 0x7fff) return false;
    return true;
}

// Packs segment after segment, with scratch for what they read.
struct Packer {
    const Model& model;
    std::span<const uint32_t> tables;
    std::vector<uint8_t> entry, tail, matrix;

    Result<void> check(const Source& source, size_t size, size_t needed, bool exact) const
    {
        if (exact ? size == needed : size >= needed) return {};
        return bad_size(model, source, size, needed, exact);
    }
    // S's entry, when it holds at least NEEDED bytes, or exactly those with kExact.
    Result<std::span<uint8_t>> read(const Segment& s, size_t needed)
    {
        const auto bytes = DLSSLOP_TRY(model.read(s.source, entry));
        DLSSLOP_TRY(check(s.source, bytes.size(), needed, s.flags & Segment::kExact));
        return bytes;
    }
    // S's COUNT binary16 values (upstream: load_f16): the first ones, or with
    // kPadded those from the ninth on when the first eight are zero
    // (nr_graph.cpp:1733-1750).
    Result<const uint8_t*> halves(const Segment& s, size_t count)
    {
        const auto bytes = DLSSLOP_TRY(model.read(s.source, entry));
        const size_t first = s.flags & Segment::kPadded && bytes.size() >= 16 && zero_eight(bytes.data()) ? 8 : 0;
        DLSSLOP_TRY(check(s.source, bytes.size(), 2 * (first + count), s.flags & Segment::kExact));
        return bytes.data() + 2 * first;
    }
    Result<void> put(const Segment& s, uint8_t* out);
    Result<void> put_halves(const Segment& s, uint8_t* out);
};

// S's matrix of ROWS x COLS codes, tile-blocked, and N-paired with kNpair.
void blocked(const Segment& s, const uint8_t* matrix, size_t rows, size_t cols, uint8_t* out)
{
    if (s.flags & Segment::kNpair)
        npair_blocked(matrix, rows, cols, out);
    else
        tile_blocked(matrix, rows, cols, out);
}

Result<void> Packer::put(const Segment& s, uint8_t* out)
{
    switch (s.recipe) {
    case Recipe::kZeros: std::memset(out, 0, s.bytes); return {};
    case Recipe::kTable: std::memcpy(out, tables.data() + s.index, s.bytes); return {};
    case Recipe::kActivations:
        for (const auto& run : kActivationRuns) out = std::fill_n(out, run.count, run.value);
        return {};
    case Recipe::kMatrix: {
        const auto codes = DLSSLOP_TRY(read(s, s.bytes)).first(s.bytes);
        if (s.flags & Segment::kRequantise) std::ranges::transform(codes, codes.begin(), requantise);
        blocked(s, codes.data(), s.rows, s.cols, out);
        return {};
    }
    case Recipe::kFfwd: {
        const uint8_t* record = DLSSLOP_TRY(read(s, kFfwdRecord)).data();
        const auto [rows, cols] = kFfwdParts[s.index];
        if (matrix.size() < s.bytes) matrix.resize(s.bytes);
        if (s.index == 0)
            gather<ff_a>(record, 8, rows / 8, cols, matrix.data());
        else if (s.index == 1)
            gather<ff_q0>(record, 8, rows / 8, cols, matrix.data());
        else
            gather<ff_q2>(record, 8, rows / 8, cols, matrix.data());
        blocked(s, matrix.data(), rows, cols, out);
        return {};
    }
    case Recipe::kVitQkv: {
        const uint8_t* record = DLSSLOP_TRY(read(s, kQkvScales + s.bytes)).data();
        if (matrix.size() < s.bytes) matrix.resize(s.bytes);
        gather<vit_qkv_weight_byte>(record, 3, kQkvRows / 3, kQkvCols, matrix.data());
        blocked(s, matrix.data(), kQkvRows, kQkvCols, out);
        return {};
    }
    case Recipe::kBias: {
        // Each head's values reordered to [i][j] (upstream: deswizzle_bias),
        // widened, and with kAffine through math profile 3's affine. A NaN
        // stays the NaN that widening makes.
        const size_t heads = s.bytes / 16384;
        const uint8_t* biases = DLSSLOP_TRY(read(s, heads * 8192)).data();
        for (size_t h = 0; h < heads; ++h, biases += 8192)
            for (size_t i = 0; i < 64; ++i)
                for (size_t j = 0; j < 64; ++j, out += 4) {
                    const uint32_t wide = widen_half(load_half(biases + 2 * bias_source(i, j)));
                    const float value = std::bit_cast<float>(wide);
                    store_word(out, s.flags & Segment::kAffine && value == value
                                        ? std::bit_cast<uint32_t>(std::fma(value, kBiasScale, kBiasOffset))
                                        : wide);
                }
        return {};
    }
    case Recipe::kScales: {
        const uint8_t* values = DLSSLOP_TRY(halves(s, s.bytes / 4));
        for (size_t i = 0; i < s.bytes / 4; ++i) store_word(out + 4 * i, widen_half(load_half(values + 2 * i)));
        return {};
    }
    case Recipe::kHalf: return put_halves(s, out);
    case Recipe::kDiagonal: {
        // Block b, row r, column c holds value 16b + r where r is c, else zero.
        const uint8_t* values = DLSSLOP_TRY(halves(s, s.bytes / 32));
        for (size_t e = 0; e < s.bytes / 2; ++e) {
            const size_t row = e % 256 / 16;
            store_half(out + 2 * e, row == e % 16 ? recode_half(load_half(values + 2 * (e / 256 * 16 + row))) : 0);
        }
        return {};
    }
    case Recipe::kBytes: std::memcpy(out, DLSSLOP_TRY(read(s, s.bytes)).data(), s.bytes); return {};
    case Recipe::kLift: {
        // The lift's [channel][k] as the input kernel addressed its packed
        // operand (nr_graph.cpp:1988-1995).
        const uint8_t* lift = DLSSLOP_TRY(read(s, s.bytes)).data();
        for (size_t ch = 0; ch < 32; ++ch)
            for (size_t k = 0; k < 16; ++k, out += 2) {
                const size_t from =
                    ch / 16 * 256 + 32 * (ch % 8) + 8 * (k % 8 / 2) + 4 * (ch / 8 % 2) + k % 2 + 2 * (k / 8);
                std::memcpy(out, lift + 2 * from, 2);
            }
        return {};
    }
    }
    return {};
}

// With kGainTail, an entry of fewer values than the segment's is completed in
// front by the last values of its layer's residual_scale
// (nr_graph.cpp:2296-2331).
Result<void> Packer::put_halves(const Segment& s, uint8_t* out)
{
    const size_t count = s.bytes / 2;
    const auto gains = DLSSLOP_TRY(model.read(s.source, entry));
    const size_t have = gains.size() / 2;
    if (!(s.flags & Segment::kGainTail) || have >= count) {
        DLSSLOP_TRY(check(s.source, gains.size(), s.bytes, s.flags & Segment::kExact));
        recode_halves(gains.data(), count, out);
        return {};
    }
    const Source residual{s.source.directory, s.source.block, s.source.layer, Suffix::kResidualScale};
    const auto scales = DLSSLOP_TRY(model.read(residual, tail));
    const size_t missing = count - have;
    DLSSLOP_TRY(check(residual, scales.size(), 2 * missing, false));
    recode_halves(scales.data() + 2 * (scales.size() / 2 - missing), missing, out);
    recode_halves(gains.data(), have, out + 2 * missing);
    return {};
}
} // namespace

std::string entry_name(const Source& source)
{
    char name[kNameSize];
    return {name, format(source, name)};
}

Result<Model> Model::open(const std::string& path)
{
    Model model;
    model.path_ = path;
    model.fd_.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st{};
    if (model.fd_.fd < 0 || fstat(model.fd_.fd, &st)) return unreadable(path);
    const uint64_t size = uint64_t(st.st_size);
    const int fd = model.fd_.fd;
    char header[16];
    if (size < sizeof header) return fail(path + ": not a model pack");
    DLSSLOP_TRY(load(fd, path, 0, header, sizeof header));
    if (std::memcmp(header, "NRMODEL1", 8)) return fail(path + ": not a model pack");
    uint32_t count;
    std::memcpy(&count, header + 8, 4);
    // An entry takes at least 20 bytes of the index.
    model.entries_.reserve(std::min<uint64_t>(count, (size - sizeof header) / 20));
    char name[4096 + 16];
    for (uint64_t at = sizeof header; count--;) {
        uint32_t length;
        if (size - at < 4) return damaged(path);
        DLSSLOP_TRY(load(fd, path, at, &length, 4));
        at += 4;
        if (length > 4096 || size - at < length + 16u) return damaged(path);
        DLSSLOP_TRY(load(fd, path, at, name, length + 16));
        at += length + 16;
        Entry e{0, 0, model.names_.size(), length};
        std::memcpy(&e.offset, name + length, 8);
        std::memcpy(&e.size, name + length + 8, 8);
        if (e.offset > size || e.size > size - e.offset)
            return fail(path + ": entry " + std::string(name, length) + " past the end");
        model.names_.append(name, length);
        model.entries_.push_back(e);
    }
    auto by_name = [&model](const Entry& e) { return model.name(e); };
    std::ranges::sort(model.entries_, {}, by_name);
    if (const auto twice = std::ranges::adjacent_find(model.entries_, {}, by_name); twice != model.entries_.end())
        return fail(path + ": entry " + std::string(model.name(*twice)) + " listed twice");
    return model;
}

Result<std::span<uint8_t>> Model::read(const Source& source, std::vector<uint8_t>& scratch) const
{
    char text[kNameSize];
    const std::string_view wanted(text, format(source, text));
    const auto e = std::ranges::lower_bound(entries_, wanted, {}, [this](const Entry& e) { return name(e); });
    if (e == entries_.end() || name(*e) != wanted) return fail(path_ + ": missing " + std::string(wanted));
    if (scratch.size() < e->size) scratch.resize(e->size);
    DLSSLOP_TRY(load(fd_.fd, path_, e->offset, scratch.data(), e->size));
    return std::span(scratch.data(), e->size);
}

Result<void> pack(std::span<const Segment> segments, std::span<const uint32_t> tables, const Model& model,
                  std::span<uint8_t> blob)
{
    Packer packer{model, tables, {}, {}, {}};
    size_t at = 0;
    for (const Segment& s : segments) {
        if (s.offset < at || s.offset > blob.size() || s.bytes > blob.size() - s.offset || !shaped(s, tables.size()))
            return bad_segment(s);
        std::memset(blob.data() + at, 0, s.offset - at);
        DLSSLOP_TRY(packer.put(s, blob.data() + s.offset));
        at = s.offset + s.bytes;
    }
    std::memset(blob.data() + at, 0, blob.size() - at);
    return {};
}

uint8_t requantise(uint8_t code) { return code == 0 ? 0x80 : code == 0xff ? 0x7f : code; }

uint32_t widen_half(uint16_t half)
{
    const uint32_t sign = uint32_t(half & 0x8000) << 16, e = half >> 10 & 31, m = half & 1023;
    if (e == 31) return m ? 0x7fc00000 : sign | 0x7f800000;
    if (e) return sign | (e + 112) << 23 | m << 13;
    if (!m) return sign;
    // A subnormal: shifted until bit 10 is set, the exponent lowered as often.
    const unsigned shift = unsigned(std::countl_zero(m)) - 21;
    return sign | (113 - shift) << 23 | (m << shift & 1023) << 13;
}

uint16_t recode_half(uint16_t half) { return (half & 0x7c00) == 0x7c00 && half & 1023 ? 0x7e00 : half; }

void tile_blocked(const uint8_t* matrix, size_t rows, size_t cols, uint8_t* out)
{
    // Tile (r/16, c/16) is the 16-byte runs of its 16 rows, one after another.
    for (size_t r = 0; r < rows; r += 16)
        for (size_t c = 0; c < cols; c += 16)
            for (size_t row = r; row < r + 16; ++row, out += 16) std::memcpy(out, matrix + row * cols + c, 16);
}

void npair_blocked(const uint8_t* matrix, size_t rows, size_t cols, uint8_t* out)
{
    // Lane l of the pair of tiles (r/16, c/16) and (r/16 + 1, c/16) is 8
    // columns of row r + l%16 of each: c + 8*(l/16) on.
    for (size_t r = 0; r < rows; r += 32)
        for (size_t c = 0; c < cols; c += 16)
            for (size_t lane = 0; lane < 32; ++lane) {
                const uint8_t* run = matrix + (r + lane % 16) * cols + c + lane / 16 * 8;
                std::memcpy(out, run, 8);
                std::memcpy(out + 8, run + 16 * cols, 8);
                out += 16;
            }
}

size_t ff_a(unsigned g, unsigned row, unsigned k)
{
    return size_t((k & 1) << 0) | ((k >> 1 & 1) << 4) | ((k >> 2 & 1) << 5) | ((k >> 3 & 1) << 1) |
           ((k >> 4 & 1) << 2) | ((k >> 5 & 1) << 14) | ((k >> 6 & 1) << 15) | ((k >> 7 & 1) << 16) |
           ((k >> 8 & 1) << 17) | ((row & 1) << 3) | ((row >> 1 & 1) << 6) | ((row >> 2 & 1) << 7) |
           ((row >> 3 & 1) << 8) | ((row >> 4 & 1) << 9) | ((row >> 5 & 1) << 10) | ((g & 1) << 11) |
           ((g >> 1 & 1) << 12) | ((g >> 2 & 1) << 13);
}

size_t ff_q0(unsigned g, unsigned j, unsigned row)
{
    return size_t(262144) + size_t(g) * 16384 +
           (((row & 1) << 1) | ((row >> 1 & 1) << 0) | ((row >> 2 & 1) << 4) | ((row >> 3 & 1) << 5) |
            ((row >> 4 & 1) << 2) | ((row >> 5 & 1) << 13) | ((j & 1) << 6) | ((j >> 1 & 1) << 3) |
            ((j >> 2 & 1) << 9) | ((j >> 3 & 1) << 7) | ((j >> 4 & 1) << 8) | ((j >> 5 & 1) << 10) |
            ((j >> 6 & 1) << 11) | ((j >> 7 & 1) << 12));
}

size_t ff_q2(unsigned g, unsigned n, unsigned j)
{
    return size_t(393216) + size_t(g) * 16384 +
           (((j & 1) << 0) | ((j >> 1 & 1) << 1) | ((j >> 2 & 1) << 2) | ((j >> 3 & 1) << 4) |
            ((j >> 4 & 1) << 5) | ((j >> 5 & 1) << 11) | ((j >> 6 & 1) << 12) | ((j >> 7 & 1) << 13) |
            ((n & 1) << 6) | ((n >> 1 & 1) << 7) | ((n >> 2 & 1) << 8) | ((n >> 3 & 1) << 3) |
            ((n >> 4 & 1) << 9) | ((n >> 5 & 1) << 10));
}

size_t vit_qkv_weight_byte(size_t which, size_t row, size_t k)
{
    const size_t b = (k & 1) | ((k >> 1 & 1) << 4) | ((k >> 2 & 1) << 5) | ((k >> 3 & 1) << 1) |
                     ((k >> 4 & 1) << 2) | ((row & 1) << 6) | ((row >> 1 & 1) << 7) | ((row >> 2 & 1) << 8) |
                     ((row >> 3 & 1) << 3) | ((row >> 4 & 1) << 9);
    const size_t tile = (row >> 5) | ((k >> 5) << 5);
    return kQkvScales + (3 * tile + which) * 1024 + b;
}

size_t bias_source(size_t i, size_t j)
{
    const size_t lane = 4 * (i % 8) + j % 8 / 2, slot = (j % 2) | (i % 16 / 8) << 1 | (j % 16 / 8) << 2;
    return (4 * (i / 16) + j / 16) * 256 + lane * 8 + slot;
}

std::array<uint8_t, 4096> activation_table()
{
    std::array<uint8_t, 4096> table;
    uint8_t* at = table.data();
    for (const auto& run : kActivationRuns) at = std::fill_n(at, run.count, run.value);
    return table;
}

} // namespace dlsslop::vulkan
