// SPDX-License-Identifier: MIT
#include "vulkan_schedule.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace dlsslop::vulkan {

namespace {

uint64_t align(uint64_t n, uint64_t alignment) { return (n + alignment - 1) / alignment * alignment; }

bool downsampling_run(Kernel k) { return k >= Kernel::kFswinPds64 && k <= Kernel::kFswinPds256; }
bool upsampling_run(Kernel k) { return k >= Kernel::kFswinPup64 && k <= Kernel::kFswinPup256; }
bool gemm(Kernel k) { return k >= Kernel::kGemmProjc && k <= Kernel::kGemmVqkvs; }

// Whether a tile counter can order step P before step Q, by their kernels
// (upstream: tchain_pair): the C=512 levels' attention, projections and FFNs;
// the ViT's expansion, contraction and QKV, and its attention, projection and
// next expansion, but not its QKV and attention; the encoder's downsampling
// runs down to the first C=512 FFN; and the decoder from the last C=512
// projection up through its upsampling runs to the fused C=32 upsample.
bool counted_pair(Kernel p, Kernel q)
{
    using enum Kernel;
    auto c512 = [](Kernel k) { return k == kGemmProjc; };
    auto ffn = [](Kernel k) { return k == kFfwd3 || k == kFfwd3w; };
    auto projection = [](Kernel k) { return k == kGemmProj || k == kGemmProjw || k == kGemmProjt; };
    auto qkv = [](Kernel k) { return k == kGemmVqkvNorm || k == kGemmVqkvNorms; };
    return (p == kAttn && c512(q)) || (c512(p) && ffn(q)) || (ffn(p) && c512(q)) || (c512(p) && q == kAttn) ||
           (p == kGemmVact && projection(q)) || (projection(p) && qkv(q)) ||
           (p == kVitAttn && projection(q)) || (projection(p) && q == kGemmVact) ||
           (c512(p) && upsampling_run(q)) || (upsampling_run(p) && (upsampling_run(q) || q == kFswinFusedUp32)) ||
           (p == kFswinDsp32 && q == kFswinPds64) || (p == kFswinPds64 && q == kFswinPds128) ||
           (p == kFswinPds128 && q == kFswinPds256) || (p == kFswinPds256 && ffn(q));
}
// Whether a kernel's push block ends with the index of its tile-counter
// record (upstream: tchain_kern).
bool counted(Kernel k)
{
    using enum Kernel;
    return k == kAttn || k == kGemmProjc || k == kFfwd3 || k == kFfwd3w || k == kGemmVact || k == kGemmProj ||
           k == kGemmProjw || k == kGemmProjt || k == kGemmVqkvNorm || k == kGemmVqkvNorms || k == kVitAttn ||
           upsampling_run(k) || k == kFswinFusedUp32 || downsampling_run(k) || k == kFswinDsp32 || k == kFswin32;
}

// The pairs of the persistent runs, the fused C=32 downsample and upsample
// that tile counters order on big frames all the same (upstream:
// NR_TC_BIG_ALLOW's default): the decoder's C=128 -> C=64 run and C=64 run ->
// fused C=32 upsample, and the encoder's C=128 -> C=256 run and C=256 run ->
// first C=512 FFN.
constexpr std::pair<Kernel, Kernel> kBigFramePairs[] = {{Kernel::kFswinPup128, Kernel::kFswinPup64},
                                                        {Kernel::kFswinPup64, Kernel::kFswinFusedUp32},
                                                        {Kernel::kFswinPds128, Kernel::kFswinPds256},
                                                        {Kernel::kFswinPds256, Kernel::kFfwd3w}};

bool same_tiles(const PushFSwin& a, const PushFSwin& b) { return a.tiles_x == b.tiles_x && a.tiles_y == b.tiles_y; }
// Words of a PersistRec in the blob: its layer's tile raster and window grid.
static_assert(offsetof(PersistRec, p) == 0);
constexpr size_t kTilesX = offsetof(PushFSwin, tiles_x) / 4, kTilesY = offsetof(PushFSwin, tiles_y) / 4,
                 kGridX = offsetof(PersistRec, gx) / 4, kGridY = offsetof(PersistRec, gy) / 4;

// A persistent run's tables of dependencies between its items, a window of
// one of its layers each (upstream: the NR_PERSIST_DF block,
// nr_graph.cpp:3538-3602): each item's producers in the layer before, the
// up to four consumers each has, and the items no producer holds up; then, for the
// straggler queue, the layer of each item, and each layer's items that wait
// with the rank from which the oldest workgroups take them, a fraction
// STRAGGLE of them.
Result<std::vector<uint32_t>> dependencies(const std::vector<PersistRec>& rec, uint32_t windows, double straggle,
                                           uint32_t& free_items)
{
    std::vector<uint32_t> need(windows, 0), cons(size_t(windows) * 4, ~0u), init;
    for (size_t k = 1; k < rec.size(); ++k) {
        const PersistRec &r = rec[k], &q = rec[k - 1];
        for (uint32_t w = 0; w < r.windows; ++w) {
            // Tile (tx, ty) of layer k comes from window floor((tx - shift)
            // / 2), floor((ty - shift_y) / 2) of layer k - 1.
            const int wx = int(w % r.gx), wy = int(w / r.gx);
            uint32_t producers[4];
            size_t count = 0;
            for (int t = 0; t < 4; ++t) {
                const int tx = 2 * wx + r.p.shift + (t & 1), ty = 2 * wy + r.p.shift_y + (t >> 1);
                if (tx < 0 || ty < 0 || tx >= int(r.p.tiles_x) || ty >= int(r.p.tiles_y)) continue;
                const int ax = tx - q.p.shift, ay = ty - q.p.shift_y;
                const int px = ax >= 0 ? ax / 2 : -((1 - ax) / 2), py = ay >= 0 ? ay / 2 : -((1 - ay) / 2);
                if (px < 0 || py < 0 || px >= int(q.gx) || py >= int(q.gy)) continue;
                // In order, once each.
                const uint32_t producer = q.flag_base + uint32_t(py) * q.gx + uint32_t(px);
                if (std::find(producers, producers + count, producer) != producers + count) continue;
                size_t at = count++;
                for (; at && producers[at - 1] > producer; --at) producers[at] = producers[at - 1];
                producers[at] = producer;
            }
            const uint32_t item = r.flag_base + w;
            need[item] = uint32_t(count);
            for (size_t i = 0; i < count; ++i) {
                uint32_t* slots = &cons[size_t(producers[i]) * 4];
                uint32_t* slot = std::find(slots, slots + 4, ~0u);
                if (slot == slots + 4)
                    return fail("network plan: a persistent run's item has more than four consumers");
                *slot = item;
            }
        }
    }
    for (uint32_t item = 0; item < windows; ++item)
        if (!need[item]) init.push_back(item);
    std::vector<uint32_t> table = need;
    table.insert(table.end(), cons.begin(), cons.end());
    table.insert(table.end(), init.begin(), init.end());
    table.resize(size_t(6) * windows, 0);
    uint32_t queued[16] = {};
    for (size_t k = 0; k < rec.size(); ++k)
        for (uint32_t w = 0; w < rec[k].windows; ++w) {
            table.push_back(uint32_t(k));
            queued[k] += need[rec[k].flag_base + w] != 0;
        }
    uint32_t first = uint32_t(init.size()), late = 0;
    for (uint32_t q : queued) {
        const uint32_t threshold = uint32_t(std::ceil(straggle * double(q)));
        table.push_back(q);
        table.push_back(threshold);
        first += std::min(threshold, q);
        late += q - std::min(threshold, q);
    }
    table.push_back(first);
    table.push_back(late);
    free_items = uint32_t(init.size());
    return table;
}

// The words a dispatch reads from its predecessor and writes for its
// successor (upstream: io): a persistent run's input is its first layer's,
// or its upsampling table's, and its output its last layer's, or its
// downsampling table's; the fused C=32 upsample's input is its blend's.
Result<uint32_t> input(const Dispatch& d, const Blob& blob)
{
    if (gemm(d.kernel)) return load<PushGemm>(d).x_off;
    if (persistent(d.kernel)) {
        const auto p = load<PushPersist>(d);
        return blob.word(upsampling_run(d.kernel) && p.ds_off ? p.ds_off : p.layers_off);
    }
    if (d.kernel == Kernel::kFswinFusedUp32) return load<PushUps>(d, sizeof(PushFSwin) / 4).p_off;
    return d.push[0];
}
Result<uint32_t> output(const Dispatch& d, const Blob& blob)
{
    if (gemm(d.kernel)) return load<PushGemm>(d).o_off;
    if (persistent(d.kernel)) {
        const auto p = load<PushPersist>(d);
        return blob.word(downsampling_run(d.kernel) && p.ds_off
                             ? p.ds_off + 1
                             : p.layers_off + (p.n_layers - 1) * (sizeof(PersistRec) / 4) + 1);
    }
    if (d.kernel == Kernel::kFswinDsp32) return load<PushDsProj>(d, sizeof(PushFSwin) / 4).o_off;
    if (d.kernel == Kernel::kFswinFusedUp32) return 0xfffffffeu;
    return d.push[1];
}
// The tile raster in which an upsampling consumer gathers its lower-level
// input (upstream: ups_grid); false when it has none.
Result<bool> gather_raster(const Dispatch& d, const Blob& blob, uint32_t& x, uint32_t& y)
{
    if (persistent(d.kernel)) {
        const auto p = load<PushPersist>(d);
        if (!p.ds_off) return false;
        x = DLSSLOP_TRY(blob.word(p.ds_off + offsetof(PushUps, itiles_x) / 4));
        y = DLSSLOP_TRY(blob.word(p.ds_off + offsetof(PushUps, itiles_y) / 4));
    } else {
        const auto u = load<PushUps>(d, sizeof(PushFSwin) / 4);
        x = u.itiles_x;
        y = u.itiles_y;
    }
    return x && y;
}
// Word AT of a persistent run's last layer's record, or of its first with FIRST.
Result<uint32_t> layer_word(const Dispatch& d, const Blob& blob, size_t at, bool first = false)
{
    const auto p = load<PushPersist>(d);
    return blob.word(p.layers_off + (first ? 0 : (p.n_layers - 1) * (sizeof(PersistRec) / 4)) + at);
}

// A free block of the arena, and the last dispatch of what it held.
struct Block {
    uint64_t offset, size;
    int freed;
};
// A placed value's bytes, and its last dispatch.
struct Live {
    int last;
    uint64_t offset, size;
};
// L's bytes into FREE, merged with the blocks next to them; a free block that
// ends at TOP lowers TOP instead.
void release(const Live& l, std::vector<Block>& free, uint64_t& top)
{
    free.push_back({l.offset, l.size, l.last});
    std::sort(free.begin(), free.end(), [](const Block& a, const Block& b) { return a.offset < b.offset; });
    std::vector<Block> merged;
    for (const Block& b : free)
        if (!merged.empty() && merged.back().offset + merged.back().size == b.offset) {
            merged.back().size += b.size;
            merged.back().freed = std::max(merged.back().freed, b.freed);
        } else
            merged.push_back(b);
    free.swap(merged);
    if (!free.empty() && free.back().offset + free.back().size == top) {
        top = free.back().offset;
        free.pop_back();
    }
}

// TABLE's entry for TILE: COUNTER, and the units NEED of it that make the tile
// whole; whether both fit it.
bool set_entry(std::vector<uint32_t>& table, uint64_t tile, uint32_t counter, uint32_t need)
{
    if (tile >= table.size() || counter >= 1u << 20 || need >= 1u << 12) return false;
    table[tile] = counter | need << 20;
    return true;
}

} // namespace

Result<std::vector<Dispatch>> merge(const std::vector<Dispatch>& d, Blob& blob, uint64_t& arena, bool& epoch)
{
    constexpr uint32_t kSwinWords = sizeof(PushFSwin) / 4,
                       kFoldWords = (sizeof(PushFSwin) + sizeof(PushDsProj)) / 4; // Also a fused upsample's.
    static_assert(sizeof(PushFSwin) + sizeof(PushUps) + 4 == 4 * kFoldWords);
    auto swin = [](const Dispatch& d) { return load<PushFSwin>(d); };
    auto is = [](const Dispatch& d, Body body, unsigned c, uint32_t words) {
        return d.body == body && d.c == c && d.words == words;
    };
    std::vector<Dispatch> out;
    out.reserve(d.size());
    epoch = false;
    for (size_t i = 0; i < d.size();) {
        // A maximal run of plain bodies of one width at one tile raster, each
        // reading what the one before wrote.
        const unsigned c = d[i].body == Body::kSwin && d[i].words == kSwinWords ? d[i].c : 0;
        size_t j = i;
        while (c && j + 1 < d.size() && is(d[j + 1], Body::kSwin, c, kSwinWords) &&
               same_tiles(swin(d[j + 1]), swin(d[i])) && swin(d[j + 1]).x_off == swin(d[j]).o_off)
            ++j;
        if (j == i) {
            out.push_back(d[i++]);
            continue;
        }
        // The downsampling body after the run joins it as its last layer, or
        // else the upsampling one before it as its first.
        const bool ds_fold = j + 1 < d.size() && is(d[j + 1], Body::kSwinDs, c, kFoldWords) &&
                             same_tiles(swin(d[j + 1]), swin(d[i])) && swin(d[j + 1]).x_off == swin(d[j]).o_off;
        if (ds_fold) ++j;
        const bool up_fold = !ds_fold && !out.empty() && is(out.back(), Body::kSwinUp, c, kFoldWords) &&
                             same_tiles(swin(out.back()), swin(d[i])) && swin(out.back()).o_off == swin(d[i]).x_off;
        if (!ds_fold && !up_fold) return fail("network plan: a persistent run neither downsamples nor upsamples");
        std::vector<const Dispatch*> layers;
        const Dispatch up_body = up_fold ? out.back() : Dispatch{};
        if (up_fold) {
            out.pop_back();
            layers.push_back(&up_body);
        }
        for (size_t t = i; t <= j; ++t) layers.push_back(&d[t]);
        if (layers.size() > 16) return fail("network plan: a persistent run has more than 16 layers");
        std::vector<PersistRec> rec(layers.size());
        uint32_t windows = 0, most = 0;
        for (size_t k = 0; k < layers.size(); ++k) {
            rec[k] = {swin(*layers[k]), layers[k]->groups[0], layers[k]->groups[1],
                      layers[k]->groups[0] * layers[k]->groups[1], windows};
            windows += rec[k].windows;
            most = std::max(most, rec[k].windows);
        }
        // Workgroups by the width's occupancy. A downsampling or upsampling
        // run of C=128 or 256 of 65 to kPersistOneMax windows a layer takes
        // one an item, which leaves its straggler queue nothing.
        const uint32_t bit = c == 64 ? 1 : c == 128 ? 2 : 4, cap = c == 64 ? 512 : c == 128 ? 256 : 128;
        const bool one = (kPersistOneMask & bit) && most > 64 && most <= kPersistOneMax;
        const uint32_t groups = one ? windows : std::min(most, cap);
        PushPersist p{};
        std::vector<uint32_t> words(rec.size() * sizeof(PersistRec) / 4);
        std::memcpy(words.data(), rec.data(), words.size() * 4);
        p.layers_off = uint32_t(blob.put_words(words.data(), words.size()) / 4);
        const Dispatch& fold = ds_fold ? d[j] : up_body;
        p.ds_off = uint32_t(blob.put_words(fold.push + kSwinWords, kFoldWords - kSwinWords) / 4);
        const auto table =
            DLSSLOP_TRY(dependencies(rec, windows, one ? 1.0 : kStragglerPercent / 100.0, p.df_n0));
        p.df_off = uint32_t(blob.put_words(table.data(), table.size()) / 4);
        // The sync region, in the arena: counters, flags, broadcast slots, the
        // ready queue and the straggler queues.
        arena = align(arena, 256);
        p.sync_off = uint32_t(arena / 4);
        epoch = epoch || c == 256;
        arena += align(uint64_t(4 + windows + groups + windows + 1 + 18 + windows + 8) * 4, 256);
        p.n_layers = uint32_t(rec.size());
        p.spin_limit = kSpinLimit;
        p.total_windows = windows;
        Dispatch m{};
        const int width = c == 64 ? 0 : c == 128 ? 1 : 2;
        m.kernel = Kernel(int(ds_fold ? Kernel::kFswinPds64 : Kernel::kFswinPup64) + width);
        m.after = After::kInvalidate;
        m.block = d[i].block;
        m.layer = d[i].layer;
        m.first = layers.front()->first;
        m.last = layers.front()->last;
        for (const Dispatch* l : layers) {
            m.first = std::min(m.first, l->first);
            m.last = std::max(m.last, l->last);
        }
        m.groups[0] = groups;
        m.groups[1] = m.groups[2] = 1;
        store(m, p);
        m.words = sizeof p / 4;
        out.push_back(m);
        i = j + 1;
    }
    if (std::any_of(out.begin(), out.end(), [](const Dispatch& d) { return d.kernel == Kernel::kCount; }))
        return fail("network plan: a Swin layer at C>=64 is outside a persistent run");
    return out;
}

Result<uint64_t> share(const std::vector<Dispatch>& dispatches, bool chains, Values& v)
{
    const int count = int(dispatches.size());
    // Each value's dispatches: every one whose layers meet the layers that
    // name it, or the whole frame when none does.
    struct Span {
        int first, last;
    };
    Span span[kKeys] = {};
    for (int k = 0; k < kKeys; ++k) {
        if (v.first[k] < 0) continue;
        span[k] = {count, -1};
        for (int f = 0; f < count; ++f)
            if (dispatches[f].first <= v.last[k] && dispatches[f].last >= v.first[k])
                span[k] = {std::min(span[k].first, f), std::max(span[k].last, f)};
        if (span[k].last < 0) span[k] = {0, count};
    }
    // A key with no size aliases the value that its plain offset lies in,
    // which lives for it too (skip(30) and block 0's output).
    int owner[kKeys];
    uint64_t delta[kKeys] = {};
    std::fill(std::begin(owner), std::end(owner), -1);
    for (int k = 0; k < kKeys; ++k) {
        if (v.first[k] < 0 || v.size[k]) continue;
        for (int o = 0; o < kKeys; ++o)
            if (v.size[o] && v.offset[o] <= v.offset[k] && v.offset[k] < v.offset[o] + v.size[o]) owner[k] = o;
        if (owner[k] < 0) return fail("network plan: a value without a size aliases no value");
        delta[k] = v.offset[k] - v.offset[owner[k]];
        span[owner[k]] = {std::min(span[owner[k]].first, span[k].first), std::max(span[owner[k]].last, span[k].last)};
    }
    // Values read past their size live all frame, so the bytes past it stay
    // zero.
    for (int k = 0; k < kKeys; ++k)
        if (v.overread[k] && v.first[k] >= 0) span[k] = {0, count};
    std::vector<int> order;
    for (int k = 0; k < kKeys; ++k)
        if (v.first[k] >= 0 && v.size[k]) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (span[a].first != span[b].first) return span[a].first < span[b].first;
        if (v.size[a] != v.size[b]) return v.size[a] > v.size[b];
        return a < b;
    });
    // Steps that tile counters order overlap: a value read in such a stretch
    // stays until its end.
    if (chains) {
        std::vector<int> stretch_end(size_t(count) + 1);
        for (int f = count; f >= 0; --f)
            stretch_end[f] = f + 1 < count && counted_pair(dispatches[f].kernel, dispatches[f + 1].kernel)
                                 ? stretch_end[f + 1]
                                 : f;
        for (int k = 0; k < kKeys; ++k)
            if (v.first[k] >= 0 && span[k].last >= 0 && span[k].last < count) span[k].last = stretch_end[span[k].last];
    }
    // First fit by first use, the smallest block that holds a value; a block
    // is free once its last dispatch is behind a barrier. A value smaller than
    // kArenaColdMax takes only a block freed at least kArenaCold dispatches
    // before its first.
    std::vector<Block> free;
    std::vector<Live> live;
    uint64_t top = 0, high = 0;
    uint64_t offset[kKeys] = {};
    for (int k : order) {
        const int first = span[k].first;
        for (size_t l = 0; l < live.size();)
            if (live[l].last < first) {
                release(live[l], free, top);
                live.erase(live.begin() + ptrdiff_t(l));
            } else
                ++l;
        const uint64_t need = align(std::max(v.size[k], v.overread[k]), 256);
        const bool cold = need < kArenaColdMax;
        size_t best = free.size();
        for (size_t b = 0; b < free.size(); ++b)
            if (free[b].size >= need && !(cold && free[b].freed + int(kArenaCold) >= first) &&
                (best == free.size() || free[b].size < free[best].size))
                best = b;
        if (best < free.size()) {
            offset[k] = free[best].offset;
            if (free[best].size > need)
                free[best] = {free[best].offset + need, free[best].size - need, free[best].freed};
            else
                free.erase(free.begin() + ptrdiff_t(best));
        } else {
            offset[k] = top;
            top += need;
        }
        live.push_back({span[k].last, offset[k], need});
        high = std::max(high, top);
    }
    for (int k = 0; k < kKeys; ++k)
        if (owner[k] >= 0) offset[k] = offset[owner[k]] + delta[k];
    std::copy(std::begin(offset), std::end(offset), v.offset);
    return align(high, 256);
}

void trim(std::vector<Dispatch>& d, uint32_t height)
{
    // The post block's window rows that start inside the picture: 8 wy + 4
    // shift_y < HEIGHT.
    Dispatch& post = d.back();
    const auto f = load<PushFSwin>(post);
    post.groups[1] = std::min(post.groups[1], uint32_t(std::max(1, (int(height) - 4 * f.shift_y + 7) / 8)));
    // The rows of its input, at half its resolution, that those read; then,
    // back through the plain C=32 layers that write that input, each read by
    // the next alone, the window rows that start inside the rows read.
    int rows = (8 * int(post.groups[1]) + 4 * f.shift_y + 1) / 2;
    uint32_t input = load<PushUps>(post, sizeof f / 4).p_off;
    for (size_t i = d.size() - 1; i-- && d[i].kernel == Kernel::kFswin32;) {
        const auto g = load<PushFSwin>(d[i]);
        const uint32_t groups = std::min(d[i].groups[1], uint32_t(std::max(1, (rows - 4 * g.shift_y + 7) / 8)));
        if (g.o_off != input || groups == d[i].groups[1]) break;
        d[i].groups[1] = groups;
        rows = 8 * int(groups) + 4 * g.shift_y;
        input = g.x_off;
    }
}

Result<uint32_t> chain(std::vector<Dispatch>& d, Blob& blob, uint64_t& arena, uint32_t& error)
{
    arena = align(arena, 256);
    const uint64_t base = arena / 4;
    // Every consumer's wait sets one error word, the first pair's, so that one
    // word says whether a wait of the frame ran out; upstream's records name
    // each pair's own. Each pair keeps its word before its counters, which
    // stay where upstream puts them.
    error = uint32_t(base + 1);
    // The frame counter, which the first fswin32 ticks.
    const size_t tick =
        size_t(std::find_if(d.begin(), d.end(), [](const Dispatch& d) { return d.kernel == Kernel::kFswin32; }) -
               d.begin());
    uint64_t words = 1;
    // Each dispatch's record: the counters it waits on, the units it needs of
    // each, its table, the counters it signals, the frame counter, its error
    // word and the magic word.
    std::vector<std::array<uint32_t, 7>> rec(d.size(), {~0u, 0u, ~0u, ~0u, uint32_t(base), 0u, 0x54434852u});
    if (tick < d.size()) rec[tick][1] = 1;
    // On a big frame, with a persistent run of more than 4096 windows a layer,
    // the runs and the fused C=32 downsample and upsample keep their barriers
    // but between kBigFramePairs.
    bool big = false;
    for (const Dispatch& r : d)
        if (persistent(r.kernel)) {
            const auto p = load<PushPersist>(r);
            big = big || (p.n_layers && p.total_windows / p.n_layers > 4096);
        }
    auto run = [](Kernel k) { return persistent(k) || k == Kernel::kFswinFusedUp32 || k == Kernel::kFswinDsp32; };
    for (size_t i = 0; i < d.size() && tick < d.size(); ++i) {
        if (i + 1 == d.size() || i < tick) continue;
        Dispatch &p = d[i], &q = d[i + 1];
        const bool kept = std::ranges::find(kBigFramePairs, std::pair(p.kernel, q.kernel)) != std::end(kBigFramePairs);
        if (!counted_pair(p.kernel, q.kernel) || (big && !kept && (run(p.kernel) || run(q.kernel)))) continue;
        if (DLSSLOP_TRY(input(q, blob)) != DLSSLOP_TRY(output(p, blob))) continue;
        uint32_t itx = 0, ity = 0;
        const bool up = upsampling_run(q.kernel) || q.kernel == Kernel::kFswinFusedUp32;
        if (up && !DLSSLOP_TRY(gather_raster(q, blob, itx, ity))) continue;
        // A table entry per token tile the consumer reads: the counter of the
        // producer's unit that writes it, and the units that make it whole.
        uint32_t tiles = 4096;
        // A persistent run's tile raster: that of its first layer as a
        // consumer, of its last as a producer.
        if (persistent(q.kernel))
            tiles = std::max(tiles,
                             DLSSLOP_TRY(layer_word(q, blob, kTilesX, true)) * DLSSLOP_TRY(layer_word(q, blob, kTilesY, true)));
        if (persistent(p.kernel))
            tiles = std::max(tiles, DLSSLOP_TRY(layer_word(p, blob, kTilesX)) * DLSSLOP_TRY(layer_word(p, blob, kTilesY)));
        if (up) tiles = std::max(tiles, itx * ity);
        std::vector<uint32_t> table(tiles, ~0u);
        uint32_t counters = tiles;
        bool fits = true;
        if (p.kernel == Kernel::kAttn) {
            // A window of four tiles a workgroup.
            const auto a = load<PushAttn>(p);
            counters = p.groups[0] * p.groups[1];
            for (uint32_t wy = 0; wy < p.groups[1]; ++wy)
                for (uint32_t wx = 0; wx < p.groups[0]; ++wx)
                    for (int t = 0; t < 4; ++t) {
                        const int tx = 2 * int(wx) + a.shift + (t & 1), ty = 2 * int(wy) + a.shift_y + (t >> 1);
                        if (tx < 0 || ty < 0 || tx >= int(a.tiles_x) || ty >= int(a.tiles_y)) continue;
                        fits &= set_entry(table, uint64_t(ty) * a.tiles_x + uint64_t(tx), wy * p.groups[0] + wx,
                                          p.groups[2]);
                    }
        } else if (gemm(p.kernel)) {
            // Every M tile, signalled by each wave of each N tile.
            const bool wide = p.kernel == Kernel::kGemmVact || p.kernel == Kernel::kGemmVqkvNorm;
            const uint32_t mt = wide ? kGemmWideMt : p.kernel == Kernel::kGemmProjw ? kGemmProjwMt : kGemmProjMt,
                           waves = wide ? kGemmWideNt / 64
                                   : p.kernel == Kernel::kGemmProjw ? kGemmProjwNt / 64
                                                                     : kGemmProjNt / 32;
            for (uint32_t k = 0; k < p.groups[0] * (mt / 16); ++k) fits &= set_entry(table, k, k, p.groups[1] * waves);
        } else if (p.kernel == Kernel::kVitAttn) {
            // A query tile, by each head.
            for (uint32_t k = 0; k < (load<PushVAttn>(p).tokens + 15) / 16; ++k)
                fits &= set_entry(table, k, k, p.groups[1]);
        } else if (p.kernel == Kernel::kFswinDsp32 || downsampling_run(p.kernel)) {
            // A downsampling layer's windows, onto the next level's tiles, when
            // it writes them in the consumer's own raster.
            PushFSwin f;
            PushDsProj ds;
            uint32_t gx = p.groups[0], gy = p.groups[1];
            if (p.kernel == Kernel::kFswinDsp32) {
                f = load<PushFSwin>(p);
                ds = load<PushDsProj>(p, sizeof f / 4);
            } else {
                const auto r = load<PushPersist>(p);
                if (!r.ds_off) continue;
                uint32_t w[sizeof(PersistRec) / 4], x[sizeof ds / 4];
                for (size_t k = 0; k < std::size(w); ++k) w[k] = DLSSLOP_TRY(layer_word(p, blob, k));
                for (size_t k = 0; k < std::size(x); ++k) x[k] = DLSSLOP_TRY(blob.word(r.ds_off + k));
                std::memcpy(&f, w, sizeof f);
                std::memcpy(&ds, x, sizeof ds);
                gx = w[kGridX];
                gy = w[kGridY];
            }
            const uint32_t rows = ds.writer_rows ? ds.writer_rows : ds.rows;
            if (ds.mode == 2 || ds.raster != ds.crow || rows != ds.rows) continue;
            if (persistent(q.kernel) && DLSSLOP_TRY(layer_word(q, blob, kTilesX, true)) != ds.otx) continue;
            const int lx = std::min(std::min(int(f.pool_tiles_x * 4), int(ds.raster)), int(ds.otx * 4)) - 1,
                      ly = std::min(int(f.pool_tiles_y * 4), int(rows)) - 1;
            std::vector<uint32_t> need(tiles, 0);
            for (uint32_t wy = 0; wy < gy; ++wy)
                for (uint32_t wx = 0; wx < gx; ++wx) {
                    const int px = (2 * int(wx) + f.shift) * 2, py = (2 * int(wy) + f.shift_y) * 2;
                    const int x0 = std::max(px, 0), x1 = std::min(px + 3, lx), y0 = std::max(py, 0),
                              y1 = std::min(py + 3, ly);
                    if (x0 > x1 || y0 > y1) continue;
                    for (int ty = y0 / 4; ty <= y1 / 4; ++ty)
                        for (int tx = x0 / 4; tx <= x1 / 4; ++tx) {
                            const uint64_t tile = uint64_t(ty) * ds.otx + uint64_t(tx);
                            if (tile < tiles)
                                ++need[tile];
                            else
                                fits = false;
                        }
                }
            for (uint32_t k = 0; k < tiles; ++k)
                if (need[k]) fits &= set_entry(table, k, k, need[k]);
        } else if (persistent(p.kernel)) {
            // An upsampling run's last layer: a window a tile.
            const uint32_t tx = DLSSLOP_TRY(layer_word(p, blob, kTilesX)), ty = DLSSLOP_TRY(layer_word(p, blob, kTilesY));
            if (up && itx != tx) continue;
            for (uint32_t k = 0; k < tx * ty; ++k) fits &= set_entry(table, k, k, 1);
        } else {
            // The FFN: a wave a weight group and tile.
            for (uint32_t k = 0; k < (load<PushFfwd3>(p).M + 15) / 16; ++k) fits &= set_entry(table, k, k, 8);
        }
        if (!fits) return reject("its tile counters overflow their tables");
        const uint32_t counter = uint32_t(base + words + 1);
        words += 1 + uint64_t(counters);
        // A C=512 projection that waits takes its M tiles in order.
        if (q.kernel == Kernel::kGemmProjc) {
            auto g = load<PushGemm>(q);
            g.remap = 1;
            store(q, g);
        }
        rec[i][3] = counter;
        rec[i + 1][0] = counter;
        rec[i + 1][2] = uint32_t(blob.put_words(table.data(), table.size()) / 4);
        rec[i + 1][5] = error;
        p.after = After::kNothing;
    }
    for (size_t i = 0; i < d.size(); ++i) {
        if (!counted(d[i].kernel)) continue;
        const bool none = rec[i][1] == 0 && rec[i][2] == ~0u && rec[i][3] == ~0u;
        d[i].push[d[i].words++] = none ? ~0u : uint32_t(blob.put_words(rec[i].data(), rec[i].size()) / 4);
    }
    arena += align(words * 4, 256);
    if (words > UINT32_MAX) return reject("its tile counters overflow 32-bit indices");
    return uint32_t(words);
}

} // namespace dlsslop::vulkan
