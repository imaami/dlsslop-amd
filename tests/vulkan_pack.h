// Model packs for the Vulkan network's host tests, in memory: NRMODEL1 files
// of given entries, the synthetic entries that upstream's graph build packed
// for the tests' digests, and synthetic models of the entries a plan reads.
// SPDX-License-Identifier: MIT
#pragma once
#include "files.h"
#include "vulkan_weights.h"

#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace vulkan_test {

using dlsslop::vulkan::Directory;
using dlsslop::vulkan::Plan;
using dlsslop::vulkan::Recipe;
using dlsslop::vulkan::Segment;
using dlsslop::vulkan::Source;
using dlsslop::vulkan::Suffix;

inline uint64_t fnv1a(const void* data, size_t bytes, uint64_t hash = 0xcbf29ce484222325u)
{
    for (auto* at = static_cast<const unsigned char*>(data); bytes--; ++at) hash = (hash ^ *at) * 0x100000001b3u;
    return hash;
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

// A synthetic entry of BYTES: splitmix64 outputs, little-endian, seeded with
// the FNV-1a 64 of NAME. Every residual_scale in unpacked/ but those of blocks
// 48, 56 and 62 starts with eight zero values, as in the real model, so that
// both of upstream's readings of it are taken.
inline std::string synthetic_entry(const std::string& name, size_t bytes)
{
    std::string data(bytes, '\0');
    Random random{fnv1a(name.data(), name.size())};
    for (size_t at = 0; at < bytes; at += 8) {
        const uint64_t x = random.next();
        std::memcpy(&data[at], &x, std::min<size_t>(8, bytes - at));
    }
    if (name.starts_with("unpacked/") && name.ends_with(".residual_scale.bin") &&
        !name.starts_with("unpacked/block48.") && !name.starts_with("unpacked/block56.") &&
        !name.starts_with("unpacked/block62."))
        std::memset(data.data(), 0, std::min<size_t>(16, bytes));
    return data;
}

template <class T>
std::string bytes_of(T value)
{
    return {reinterpret_cast<const char*>(&value), sizeof value};
}
// The start of an NRMODEL1 file of COUNT entries.
inline std::string pack_header(uint32_t count) { return "NRMODEL1" + bytes_of(count) + bytes_of(uint32_t(0)); }
// An entry of the index: NAME's length, NAME, OFFSET and SIZE.
inline std::string index_entry(const std::string& name, uint64_t offset, uint64_t size)
{
    return bytes_of(uint32_t(name.size())) + name + bytes_of(offset) + bytes_of(size);
}

// A model pack in memory, which Model::open() reads through /proc. OK tells
// whether its bytes were written.
struct Pack {
    int fd = memfd_create("vulkan-test-pack", MFD_CLOEXEC);
    std::string path = "/proc/self/fd/" + std::to_string(fd);
    bool ok = fd >= 0;
    Pack() = default;
    explicit Pack(std::string_view bytes) { write(bytes); }
    Pack(const Pack&) = delete;
    ~Pack() { close(fd); }
    void write(std::string_view bytes) { ok = ok && dlsslop::write_all(fd, bytes.data(), bytes.size()); }
};
struct Entry {
    std::string name, data;
};
// The NRMODEL1 file of ENTRIES, their data after the index in their order.
inline std::string pack_bytes(const std::vector<Entry>& entries)
{
    std::string index = pack_header(uint32_t(entries.size())), data;
    uint64_t offset = index.size();
    for (const Entry& e : entries) offset += 4 + e.name.size() + 16;
    for (const Entry& e : entries) {
        index += index_entry(e.name, offset + data.size(), e.data.size());
        data += e.data;
    }
    return index + data;
}

// The size of an entry in the real model, the pack that linux/package/
// model-tools extracts from nvngx_dlssnr 310.8.0.
inline uint32_t model_bytes(const Source& s)
{
    using enum Suffix;
    const unsigned b = s.block;
    if (s.directory == Directory::kRecords) return b >= 31 && b <= 38 ? 3145856 : 524288;
    if (s.directory == Directory::kVit) return s.suffix == kSkipWeight ? 2048 : s.layer == 4 ? 1048576 : 4194304;
    if (s.directory == Directory::kSplitSwin) switch (s.suffix) {
        case kQkv: return 786432;
        case kAttnPosBias: return 131072;
        case kSkipWeight: return 1024;
        case kTail: return 64;
        default: return (b == 30 && s.layer == 4) || b == 39 ? 524288 : 262144;
        }
    // The Swin blocks: at C=32 the pre and post blocks and the first and last
    // levels, then 64, 128 and 256 down to block 22 and back from block 48.
    const bool edge = s.directory != Directory::kUnpacked;
    const unsigned c = edge || b <= 4 || b >= 66 ? 32 : b <= 8 || b >= 62 ? 64 : b <= 14 || b >= 56 ? 128 : 256,
                   heads = c / 32;
    switch (s.suffix) {
    case kMlpExpand: return 4 * c * c;
    case kMlpContract: return c * (heads > 1 ? c : 4 * c);
    case kMlpMid: return c * 128;
    case kQkv: return 3 * c * c;
    case kAttnOutProj: return c * c;
    case kAttnPosBias: return heads * 8192;
    case kResidualScale:
        return s.directory == Directory::kPreblock ? 80 : s.directory == Directory::kPostblock ? 64 : 2 * (c + 16);
    case kAttnResidualScale: return edge ? 80 : b == 4 || b == 8 || b == 14 || b == 22 ? 2 * c : 2 * (c + 8);
    case kScalarsB: return 4 * std::max(4u, heads);
    case kResample: return 2 * c * c;
    case kUpsampleGain: return c >= 64 ? 2 * (c - 16) : 2 * c;
    case kSkipGain:
    case kMainGain: return 64;
    default: return 1024; // input_lift, out_project
    }
}

// The entries a plan's segments read, each once, by name.
inline std::map<std::string, Source> entries(const Plan& p)
{
    std::map<std::string, Source> read;
    for (const Segment& s : p.segments) {
        if (s.recipe == Recipe::kZeros || s.recipe == Recipe::kTable || s.recipe == Recipe::kActivations) continue;
        read.emplace(dlsslop::vulkan::entry_name(s.source), s.source);
        // A short upsample gain takes the tail of its layer's residual scales.
        if (s.flags & Segment::kGainTail) {
            const Source scales{s.source.directory, s.source.block, s.source.layer, Suffix::kResidualScale};
            read.emplace(dlsslop::vulkan::entry_name(scales), scales);
        }
    }
    return read;
}

// A model pack of synthetic entries (synthetic_entry()), of the
// real model's names and sizes, holding every entry the plan reads: the pack
// that upstream's synthetic goldens came from, but for the entries no plan
// reads. Its weights free no Swin head of the exponent's upper clamp; with
// UNCLAMPED, every position bias and head scale is zero, which frees every
// head.
inline bool synthetic_model(const Plan& p, Pack& pack, bool unclamped = false)
{
    const auto read = entries(p);
    std::string index = pack_header(uint32_t(read.size()));
    uint64_t offset = index.size();
    for (const auto& [name, source] : read) offset += 4 + name.size() + 16;
    for (const auto& [name, source] : read) {
        index += index_entry(name, offset, model_bytes(source));
        offset += model_bytes(source);
    }
    pack.write(index);
    for (const auto& [name, source] : read)
        pack.write(unclamped && (source.suffix == Suffix::kAttnPosBias || source.suffix == Suffix::kScalarsB)
                       ? std::string(model_bytes(source), '\0')
                       : synthetic_entry(name, model_bytes(source)));
    return pack.ok;
}

} // namespace vulkan_test
