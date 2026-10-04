// SPDX-License-Identifier: MIT
// The CPU references' goldens: deterministic fixtures, and the FNV-1a 64 of
// what a reference makes of them. A build with another compiler, or a port of
// a reference, must reproduce every golden bit for bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace golden {

// Value I of fixture SEED: splitmix64's finalizer over both.
inline std::uint64_t bits(std::uint32_t seed, std::uint64_t i)
{
    std::uint64_t z = ((std::uint64_t(seed) << 40) ^ i) + 0x9e3779b97f4a7c15u;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
    return z ^ (z >> 31);
}

// Value I of fixture SEED, uniform in [0, 1) with 24 bits.
inline float unit(std::uint32_t seed, std::uint64_t i)
{
    return float(bits(seed, i) >> 40) * 0x1p-24f;
}

// Value I of fixture SEED as a network's answer: half of the values in
// [-0.25, 1.25), the other half of either sign, from 2^-27 to below 2^15,
// with every fraction bit set at random. Each rounds to a finite binary16.
inline float neural(std::uint32_t seed, std::uint64_t i)
{
    const std::uint64_t v = bits(seed, i);
    if (v & 1) return float(v >> 40) * 0x1p-24f * 1.5f - 0.25f;
    const std::uint32_t word = std::uint32_t((v >> 32) & 0x80000000u) |
        (std::uint32_t(100 + ((v >> 1) & 0xffff) % 42) << 23) | std::uint32_t((v >> 17) & 0x7fffffu);
    float value;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

// Value I of fixture SEED as a binary16 of any sign, magnitude and fraction
// but a nonfinite one.
inline std::uint16_t half(std::uint32_t seed, std::uint64_t i)
{
    const std::uint16_t h = std::uint16_t(bits(seed, i));
    return (h & 0x7c00) == 0x7c00 ? std::uint16_t(h ^ 0x0400) : h;
}

inline std::uint64_t fnv1a64(const void* data, std::size_t bytes)
{
    std::uint64_t hash = 0xcbf29ce484222325u;
    for (auto* at = static_cast<const unsigned char*>(data); bytes--; ++at) hash = (hash ^ *at) * 0x100000001b3u;
    return hash;
}

// Whether BYTES of DATA hash to EXPECTED. When they do not, it says which
// golden moved and to what, so that a change shows every golden it moves.
inline bool check(const char* name, const void* data, std::size_t bytes, std::uint64_t expected)
{
    const std::uint64_t actual = fnv1a64(data, bytes);
    if (actual == expected) return true;
    std::fprintf(stderr, "golden %s: fnv1a64=%016llx, not %016llx\n", name, static_cast<unsigned long long>(actual),
                 static_cast<unsigned long long>(expected));
    return false;
}

} // namespace golden
