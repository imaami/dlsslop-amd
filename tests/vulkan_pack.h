// Model packs for the Vulkan network's host tests, in memory: NRMODEL1 files
// of given entries, and the synthetic entries that upstream's graph build
// packed for the tests' digests.
// SPDX-License-Identifier: MIT
#pragma once
#include "files.h"

#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace vulkan_test {

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

} // namespace vulkan_test
