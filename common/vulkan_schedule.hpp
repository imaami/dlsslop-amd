// Inside the Vulkan network's plan: the dispatches the lowering of its layers
// makes, and the schedule that merges them into persistent runs, shares the
// activation arena among their values and chains them with tile counters.
// Shared by vulkan_plan.cpp and vulkan_schedule.cpp only.
// SPDX-License-Identifier: MIT
#pragma once
#include "vulkan_plan.hpp"

#include <cstring>
#include <vector>

namespace dlsslop::vulkan {

// A Swin body at C=64, 128 or 256 as the lowering leaves it, for a persistent
// run to take as a layer: a plain one (upstream: fswin<C>), one with the
// downsample fused (fswindsp<C>), or one with the upsample fused
// (fswinfusedup<C>).
enum class Body : uint8_t { kNone, kSwin, kSwinDs, kSwinUp };

// A dispatch (upstream: Disp): KERNEL, or a BODY of width C when KERNEL is
// kCount, over GROUPS with the push block of WORDS words. It runs layers
// FIRST to LAST, starting with BLOCK's layer LAYER; AFTER orders it before the
// next.
struct Dispatch {
    Kernel kernel;
    Body body;
    After after;
    uint16_t c;
    uint8_t words, block, layer, first, last;
    uint32_t groups[3];
    uint32_t push[32];
};
// The push block's part of type T from word AT on.
template <class T>
T load(const Dispatch& d, size_t at = 0)
{
    T part;
    std::memcpy(&part, d.push + at, sizeof part);
    return part;
}
template <class T>
void store(Dispatch& d, const T& part, size_t at = 0)
{
    std::memcpy(d.push + at, &part, sizeof part);
}

// The activation arena's values by key (see Value): their sizes, the bytes
// their readers reach past them, their offsets, and the first and last layers
// whose lowering names them, -1 when none does (upstream: vsize, overread,
// voff.m and voff.used).
inline constexpr int kKeys = 8 * kBlocks;
struct Values {
    uint64_t size[kKeys], overread[kKeys], offset[kKeys];
    int16_t first[kKeys], last[kKeys];
};

// The weight blob as it is put together (upstream: wblob and its put()).
struct Blob {
    std::vector<Segment>& segments;
    std::vector<uint32_t>& tables;
    uint64_t bytes;
    // SEGMENT at the next multiple of ALIGN; its offset.
    uint64_t put(Segment segment, uint32_t align);
    // COUNT words as a table at the next multiple of 16; its offset.
    uint64_t put_words(const uint32_t* words, size_t count);
    // Word INDEX of the blob, which a table holds (upstream: wword).
    Result<uint32_t> word(uint64_t index) const;
};

// LOWERED with its Swin bodies at C>=64 merged into persistent runs, their
// tables put in BLOB and their sync regions after the ARENA's end, which
// grows (upstream: nr_graph.cpp:3354-3637). EPOCH is set when there is a run
// at C=256, whose sync words count the frames that tile counters need.
Result<std::vector<Dispatch>> merge(const std::vector<Dispatch>& lowered, Blob& blob, uint64_t& arena, bool& epoch);
// The values' offsets where they share the arena by lifetime, from the plain
// ones in VALUES and the layers DISPATCHES run; the end of the values
// (upstream: the arena probe, nr_graph.cpp:3657-3799). CHAINS keeps a value a
// stretch of tile-counted steps reads until its end.
Result<uint64_t> share(const std::vector<Dispatch>& dispatches, bool chains, Values& values);
// DISPATCHES, which end with the post block, without the post block's window
// rows that start past the picture's HEIGHT, nor the window rows of the plain
// C=32 layers before it that only those read (upstream: the dead rows of
// NrSession::build, NR_DEAD_ROWS).
void trim(std::vector<Dispatch>& dispatches, uint32_t height);
// Tile counters in place of the barriers between DISPATCHES that they can
// replace: their tables and records put in BLOB, their counters after the
// ARENA's end, which grows, and AFTER kNothing for each dispatch they order
// (upstream: nr_graph.cpp:3844-4247); the counters' words. ERROR is the arena's
// u32 word that every record names, and that the waits set when they give up.
Result<uint32_t> chain(std::vector<Dispatch>& dispatches, Blob& blob, uint64_t& arena, uint32_t& error);
// The word of a persistent run's sync region, from its sync_off on, that the
// run's waits set when they run out (fswin_t.comp).
inline constexpr uint32_t kRunError = 3;

} // namespace dlsslop::vulkan
