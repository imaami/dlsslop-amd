// SPDX-License-Identifier: MIT
// Host test of the Vulkan network's plan, without a GPU or a model. At 40
// frame extents from 17x17 to 5120x2880, the plan's steps with their push
// constants and its activation arena's values and sizes are checked against
// goldens of upstream's own NrSession::build (DLSSNR-AMD's nr_graph.cpp at
// 3dfdddc, built with its rdna4.sh defines and stopped before the device), and
// the weight blob packed from a synthetic model pack against upstream's. The
// frames the network cannot take are rejected.
#include "vulkan_pack.h"
#include "vulkan_plan.h"
#include "vulkan_weights.h"

#include <getopt.h>
#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {
namespace vulkan = dlsslop::vulkan;
using vulkan::After;
using vulkan::Directory;
using vulkan::Kernel;
using vulkan::Plan;
using vulkan::Recipe;
using vulkan::Segment;
using vulkan::Source;
using vulkan::Step;
using vulkan::Suffix;
using vulkan_test::fnv1a;

int failures = 0;
// Whether OK; if not, the test fails with the message.
[[gnu::format(printf, 2, 3)]] bool expect(bool ok, const char* format, ...)
{
    if (ok) return true;
    std::va_list args;
    va_start(args, format);
    std::fputs("vulkan-plan test: ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    ++failures;
    return false;
}

// Upstream's plan of WIDTH x HEIGHT frames: its working extent, dispatches,
// activation arena and values' end, weight blob, tile-counter words,
// persistent runs and dispatches ordered by tile counters; the FNV-1a 64 of
// its dispatches, "kernel gx gy gz chained push-words...\n" each, and of its
// values, "key bBLOCK/SLOT off OFFSET size BYTES overread BYTES\n" each; and of
// the weight blob it packed from the synthetic pack and from the real model.
// Recorded from upstream's build by a probe that dumps them where the build
// meets the device.
struct Golden {
    uint32_t width, height, work_width, work_height, steps;
    uint64_t arena, values_end;
    uint32_t blob, counters, runs, chained;
    uint64_t dispatches, values, synthetic_blob, real_blob;
};
constexpr Golden kGoldens[] = {
    {17, 17, 328, 320, 124, 27948288, 26550272, 155469644, 344224, 6, 99,
     0x4725b01e932a14e3u, 0x7c39a7c135a6f2abu, 0xeb28c10f4091fed8u, 0x1139783aac0df130u},
    {1, 400, 320, 448, 123, 36382976, 34963456, 155804972, 348335, 6, 100,
     0xd1a0a133db128544u, 0x2cf2a0137495ca85u, 0xa550da0f6e50c1b8u, 0xb1c6ce4c696375f0u},
    {400, 1, 448, 320, 123, 36382976, 34963456, 155804972, 348337, 6, 100,
     0x2f67053509984e9bu, 0x2cf2a0137495ca85u, 0x503c3301549a4058u, 0x1c03a506ed52aa30u},
    {32, 32, 328, 320, 124, 27948288, 26550272, 155469644, 344224, 6, 99,
     0x988d942a1062d1f9u, 0x7c39a7c135a6f2abu, 0xeb28c10f4091fed8u, 0x1139783aac0df130u},
    {64, 64, 336, 320, 123, 28210432, 26796032, 155506540, 348321, 6, 100,
     0x48a9ac6af3e03f09u, 0xb494e00c5a47741fu, 0x8912ddc4c703e0f1u, 0xb699f14dcf167519u},
    {160, 120, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0x7d433140eae611c2u, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {200, 100, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0x4cbc70b50fed8fadu, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {256, 144, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0x00f1373ace19b41cu, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {128, 320, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0x7040a3cb76bb41acu, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {320, 128, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0x2fd5ef8b3388b3a4u, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {320, 129, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0xa6d6d9f5b5a57471u, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {320, 180, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100,
     0x1cd3cff2da4abbaeu, 0xc331ba7fb0e0bb1fu, 0x40b17b38087bc596u, 0xf5b0b1c3500232beu},
    {333, 333, 384, 384, 123, 36973824, 35553280, 155839068, 348321, 6, 100,
     0x2cc11d89a80e579eu, 0x6b33d951c0bcfbc7u, 0x3fecdadf92dac286u, 0x80cae43a7f9b17beu},
    {640, 360, 640, 384, 123, 60683520, 59244544, 156659772, 348351, 6, 100,
     0x279e3656777f30f6u, 0xd08ba5e9c77522d6u, 0x7b82ae26575a5673u, 0x65b0e14eb89142fbu},
    {640, 480, 640, 512, 123, 80357888, 78905344, 157343020, 348372, 6, 100,
     0x7e5046473f0b12f2u, 0x4d6a79d85edfccd5u, 0xde6fad53a135a3f2u, 0x8aab40ddd95a2ceau},
    {800, 450, 832, 512, 123, 104949760, 103481344, 158163244, 348409, 6, 100,
     0x60fedf5b82beda3fu, 0xc573bd1a9fee5da2u, 0x97b6744e1401a72eu, 0x58b66c5245e5e706u},
    {854, 480, 896, 512, 123, 111901696, 110428160, 158436620, 348409, 6, 100,
     0x274bf8a0db0e5f0eu, 0xa0b017d2a04de8a3u, 0x00aefdab011a0b37u, 0x0c2ea98a0964bcdfu},
    {960, 540, 960, 576, 123, 136034560, 134545408, 159222220, 348465, 6, 100,
     0xd247886a5323db1bu, 0x3edaf20da30af3b0u, 0x390312064ae96bd6u, 0xe9c89bf16d84710eu},
    {1001, 563, 1024, 576, 123, 144888064, 143392768, 159529740, 348465, 6, 100,
     0x9e2d886f9a16351eu, 0x66e151ad666eb8d5u, 0x9b85c33caf0c6a45u, 0xb322441c81c306fdu},
    {1152, 648, 1152, 704, 123, 198629376, 197099520, 161373804, 348521, 6, 100,
     0x36c2cb5fb49aa64eu, 0x01f1ddd49336e51fu, 0x548205088e16d335u, 0x6bbab5458a40a96du},
    {1024, 768, 1088, 768, 123, 204531200, 202997760, 161578476, 348521, 6, 100,
     0x5bd1f61f738dbabdu, 0x4333b55449e60434u, 0xcf6833b54b714da5u, 0x38e607fbb279972du},
    {1152, 720, 1152, 768, 123, 214569984, 213024768, 161988300, 348521, 6, 100,
     0xd55668499e67c8eeu, 0x05f9d9530045eec9u, 0x1a29d0e1b0dfba41u, 0x1a461c1ff76a97e9u},
    {1280, 720, 1344, 768, 123, 252244224, 250675200, 163217868, 348573, 6, 100,
     0x810a62ce2256b292u, 0x4006e4a89e059758u, 0xe52b760769a08fcbu, 0x0c5868919aa7b233u},
    {1280, 800, 1280, 832, 123, 200673536, 199098368, 163491788, 348849, 6, 100,
     0xf554b70922b31732u, 0xddaea6e67fe051dfu, 0xdc961d430e7ad94fu, 0xc3a88fed133290b7u},
    {1366, 768, 1408, 768, 123, 201331968, 199753728, 163629740, 349085, 6, 100,
     0xec54bc6993561461u, 0x3caec83201885480u, 0xf105e6fae2ffc580u, 0x0dd7969dfe892788u},
    {1280, 1024, 1344, 1024, 123, 258742272, 257097728, 166105772, 353783, 6, 100,
     0x165a156f3cbb947eu, 0x94ca7bf838c1f8f0u, 0xf937d3487aedd057u, 0x191edd0f5116933fu},
    {1600, 900, 1600, 960, 123, 288875008, 287211520, 167447628, 356346, 6, 100,
     0x5a335ceb3e36621bu, 0xfc5a22ba254a7b9bu, 0xbd834b7981b71ef3u, 0x98450805839f10ebu},
    {1920, 804, 1920, 832, 123, 300357632, 298680320, 167964652, 357317, 6, 100,
     0x283dca62a7916cf2u, 0xba97e75454f5b261u, 0xbf2b69b5068761fbu, 0x691354c0df5f7353u},
    {1440, 1080, 1472, 1088, 123, 301177088, 299499520, 167997516, 357391, 6, 100,
     0xbb919f0bbaaee2dbu, 0xda3fc3535c060b8du, 0x7f9f793dd7f54668u, 0xa8f624ff64c6ae90u},
    {1707, 960, 1728, 960, 123, 309938432, 308248576, 168479676, 358302, 6, 100,
     0xf42212778cca9f32u, 0xda824cd32854e759u, 0x6290f9bcb8955bebu, 0xe66674c63c8f6fe3u},
    {1680, 1050, 1728, 1088, 123, 353336832, 351600640, 170336556, 361818, 6, 100,
     0x119353c5e16603c2u, 0x92dae6258ab9771fu, 0x4668ab4d675b3481u, 0x3dbc7143a79f2259u},
    {1600, 1200, 1600, 1216, 123, 365540352, 363790336, 170886444, 362851, 6, 100,
     0x3c50e64e273caffeu, 0x21ea8125c575fb4du, 0x588ba58c42435516u, 0x4f3e75ab52cabc0eu},
    {1919, 1079, 1920, 1088, 123, 392276224, 390496256, 172090828, 365117, 6, 100,
     0x56d3a31775b98216u, 0xf161d160a6330aedu, 0x4b6d1ac03758a76eu, 0x08c8349fd4716606u},
    {1080, 1920, 1088, 1920, 123, 392276224, 390496256, 172090828, 365117, 6, 100,
     0x3069afc077709d3au, 0xf161d160a6330aedu, 0x890c57ba11f78c32u, 0x24ec339d330f553au},
    {1920, 1080, 1920, 1088, 123, 392276224, 390496256, 172090828, 365117, 6, 100,
     0x6a2c6e4d8945969eu, 0xf161d160a6330aedu, 0x4b6d1ac03758a76eu, 0x08c8349fd4716606u},
    {1920, 1200, 1920, 1216, 123, 438039040, 436207616, 174153932, 369013, 6, 100,
     0xb872d840cbb5a359u, 0xcb29e66fcd7cdb28u, 0x0a3484d0c0870dc7u, 0x838a13e1c5b6720fu},
    {2560, 1080, 2560, 1088, 123, 522446848, 520519680, 177938444, 376187, 6, 100,
     0xee324eac4500a9feu, 0xcdc4561783fb376du, 0x1faaffeccd2f417du, 0x39bb65ac3b47e675u},
    {2560, 1440, 2560, 1472, 123, 701894144, 699760640, 186189772, 391779, 6, 100,
     0x045eb36e81e4c5a7u, 0x9606066a7e165f97u, 0x8c670db3e6ec5cb9u, 0xfb9390c01c105941u},
    {3840, 2160, 3840, 2176, 123, 1545683712, 1543208960, 224109516, 317582, 6, 92,
     0x192456be83600b76u, 0x4b7752b8d303fce8u, 0x34319496ae9fbc34u, 0x56217f897345a98cu},
    {5120, 2880, 5120, 2880, 123, 2756553216, 2753167360, 277339884, 319181, 6, 92,
     0x14a73f78e985b843u, 0x25c685b892ab46dcu, 0xc1d0785dee3cbcd4u, 0xf7abc97f741ee2acu},
};
// The extents at which the synthetic model is packed: the smallest planned,
// which views the post block's input, small ones and odd ones, dlsslopd's 720p
// tier, a portrait one and 4K, where fewer steps are chained.
constexpr struct {
    uint32_t width, height;
} kPacked[] = {{17, 17}, {64, 64}, {1, 400}, {1152, 720}, {1280, 720}, {1707, 960}, {1080, 1920}, {3840, 2160}};

const Golden* golden(uint32_t width, uint32_t height)
{
    for (const Golden& g : kGoldens)
        if (g.width == width && g.height == height) return &g;
    return nullptr;
}

const char* stem(Kernel k) { return vulkan::kKernels[size_t(k)].stem; }
bool persistent(Kernel k) { return k >= Kernel::kFswinPds64 && k <= Kernel::kFswinPup256; }

// The steps as the goldens hash them.
uint64_t dispatch_hash(const Plan& p)
{
    std::string text;
    char word[16];
    for (const Step& s : p.steps) {
        text += stem(s.kernel);
        for (uint32_t g : s.groups) text += " " + std::to_string(g);
        text += s.after == After::kNothing ? " 1" : " 0";
        for (uint32_t w = 0; w < s.words; ++w) {
            std::snprintf(word, sizeof word, " %u", p.push[s.push + w]);
            text += word;
        }
        text += '\n';
    }
    return fnv1a(text.data(), text.size());
}
// The values as the goldens hash them.
std::string value_line(const vulkan::Value& v)
{
    char line[128];
    std::snprintf(line, sizeof line, "%u b%u/%u off %" PRIu64 " size %" PRIu64 " overread %" PRIu64 "\n", v.key,
                  v.key / 8, v.key % 8, v.offset, v.bytes, v.overread);
    return line;
}
uint64_t value_hash(const Plan& p)
{
    std::string text;
    for (const vulkan::Value& v : p.values) text += value_line(v);
    return fnv1a(text.data(), text.size());
}

// What a plan must hold whatever its extent.
void check_invariants(const Plan& p)
{
    char at[32];
    std::snprintf(at, sizeof at, "%ux%u", p.width, p.height);
    const std::vector<Step>& steps = p.steps;
    expect(steps.front().kernel == Kernel::kFswinImagePreds32 && steps.back().kernel == Kernel::kFswinImagePost32,
           "%s: the steps do not start with the pre block and end with the post block", at);
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step& s = steps[i];
        const uint32_t* w = &p.push[s.push];
        expect(s.words * 4u == vulkan::kKernels[size_t(s.kernel)].push && s.push + s.words <= p.push.size(),
               "%s: step %zu, %s, pushes %u words into a range of %u bytes", at, i, stem(s.kernel), s.words,
               vulkan::kKernels[size_t(s.kernel)].push);
        expect((s.after == After::kFull) == (i + 1 == steps.size()), "%s: step %zu is not followed as the %s", at, i,
               i + 1 == steps.size() ? "last" : "others");
        // No step reads the arena where it writes: nothing orders one
        // workgroup against another. A persistent run's layers are in its
        // records.
        if (persistent(s.kernel)) continue;
        uint64_t in[2] = {w[0], ~0ull}, out = w[1];
        if (s.kernel == Kernel::kFswinFusedUp32 || s.kernel == Kernel::kFswinImagePost32) {
            in[0] = w[sizeof(vulkan::PushFSwin) / 4];
            in[1] = w[sizeof(vulkan::PushFSwin) / 4 + 1];
        } else if (s.kernel >= Kernel::kGemmProjc && s.kernel <= Kernel::kGemmVqkvs) {
            const bool residual = s.kernel != Kernel::kGemmNores && s.kernel != Kernel::kGemmVact &&
                                  s.kernel != Kernel::kGemmVqkvNorm && s.kernel != Kernel::kGemmVqkvs;
            in[0] = w[1];
            in[1] = residual ? w[2] : ~0ull;
            // gemmvqkvs writes binary16s, by their index.
            out = s.kernel == Kernel::kGemmVqkvs ? 2ull * w[3] : w[3];
        } else if (s.kernel == Kernel::kDecUps) {
            in[0] = 2ull * w[0];
            in[1] = w[1];
            out = w[2];
        }
        expect(in[0] != out && in[1] != out, "%s: step %zu, %s, reads the arena at %" PRIu64 " where it writes", at,
               i, stem(s.kernel), out);
    }
    // The segments in order, inside the blob.
    uint64_t end = 0;
    for (const Segment& s : p.segments) {
        expect(s.offset >= end, "%s: the segment at %u overlaps the one before", at, s.offset);
        end = uint64_t(s.offset) + s.bytes;
        if (s.recipe == Recipe::kTable)
            expect(s.bytes % 4 == 0 && s.index + s.bytes / 4 <= p.tables.size(),
                   "%s: the table segment at %u is outside the tables", at, s.offset);
    }
    expect(end <= p.blob_bytes, "%s: the segments end past the blob", at);
    expect(!p.segments.empty() && p.segments[0].recipe == Recipe::kActivations && !p.segments[0].offset,
           "%s: the activation table is not at the blob's start", at);
    const bool noise = std::any_of(p.segments.begin(), p.segments.end(), [&](const Segment& s) {
        return s.recipe == Recipe::kZeros && s.offset == 4 * p.noise.off &&
               s.bytes == uint64_t(p.noise.width) * p.noise.height * 8;
    });
    expect(noise && p.noise.width == p.work_width && p.noise.height == p.work_height,
           "%s: the noise field is not a zeroed region of the working extent", at);
    // The values below their end; the sync regions and counters after it.
    for (const vulkan::Value& v : p.values)
        expect(!v.bytes || v.offset + std::max(v.bytes, v.overread) <= p.values_end,
               "%s: value %u reaches past the values' end", at, v.key);
    expect(p.values_end % 256 == 0 && p.values_end + 4ull * p.counter_words <= p.arena_bytes &&
               p.arena_bytes <= UINT32_MAX,
           "%s: the arena's regions do not fit it", at);
}

void check_goldens()
{
    for (const Golden& g : kGoldens) {
        const auto p = vulkan::plan(g.width, g.height);
        if (!expect(bool(p), "%ux%u: %s", g.width, g.height, p ? "" : p.error().what.c_str())) continue;
        uint32_t runs = 0;
        for (const Step& s : p->steps) runs += persistent(s.kernel);
        expect(p->work_width == g.work_width && p->work_height == g.work_height,
               "%ux%u: the working extent is %ux%u, upstream's %ux%u", g.width, g.height, p->work_width,
               p->work_height, g.work_width, g.work_height);
        expect(p->steps.size() == g.steps && runs == g.runs && p->chained == g.chained,
               "%ux%u: %zu steps, %u persistent runs and %u chained, upstream's %u, %u and %u", g.width, g.height,
               p->steps.size(), runs, p->chained, g.steps, g.runs, g.chained);
        expect(p->arena_bytes == g.arena && p->values_end == g.values_end && p->counter_words == g.counters,
               "%ux%u: an arena of %" PRIu64 " bytes, values to %" PRIu64 " and %u counter words, upstream's %" PRIu64
               ", %" PRIu64 " and %u",
               g.width, g.height, p->arena_bytes, p->values_end, p->counter_words, g.arena, g.values_end,
               g.counters);
        expect(p->blob_bytes == g.blob, "%ux%u: a weight blob of %u bytes, upstream's %u", g.width, g.height,
               p->blob_bytes, g.blob);
        expect(dispatch_hash(*p) == g.dispatches, "%ux%u: the steps differ from upstream's dispatches (--print)",
               g.width, g.height);
        expect(value_hash(*p) == g.values, "%ux%u: the values differ from upstream's (--print)", g.width, g.height);
        check_invariants(*p);
    }
}

void check_rejections()
{
    auto refused = [](uint32_t width, uint32_t height, const std::string& why, uint64_t storage = UINT64_MAX) {
        const std::string what = "the network does not take " + std::to_string(width) + "x" +
                                 std::to_string(height) + " frames: " + why;
        const auto p = vulkan::plan(width, height, storage);
        expect(!p && p.error().rejected && p.error().what == what, "%ux%u: expected rejected \"%s\", got \"%s\"",
               width, height, what.c_str(), p ? "a plan" : p.error().what.c_str());
    };
    // Upstream's build fails at working extents that are not multiples of 8,
    // looking for kernels its shaders are not built as.
    refused(1, 1, "its working extent 321x320 is not a multiple of 8");
    refused(8, 8, "its working extent 322x320 is not a multiple of 8");
    refused(16, 16, "its working extent 324x320 is not a multiple of 8");
    refused(0, 720, "a side is 0 or above 16384");
    refused(1280, 0, "a side is 0 or above 16384");
    refused(16385, 720, "a side is 0 or above 16384");
    // Upstream truncated the offsets past 4 GiB: at 7680x4320 its values end
    // at 6291095552 bytes.
    refused(7680, 4320, "its activation arena of at least 6291095552 bytes overflows 32-bit offsets");
    // Nor did it check the device's storage buffers, which must each hold the
    // arena or the weights whole.
    const std::string storage = " bytes or weights of 163217868 bytes exceed the device's storage buffers of ";
    refused(1280, 720, "its activation arena of 252244224" + storage + "252244223 bytes", 252244223);
    refused(1280, 720, "its activation arena of 252244224" + storage + "163217867 bytes", 163217867);
    expect(bool(vulkan::plan(1280, 720, 252244224)), "1280x720: rejected on a device that holds its arena");
}

// The size of an entry in the real model, the pack that linux/package/
// model-tools extracts from nvngx_dlssnr 310.8.0.
uint32_t model_bytes(const Source& s)
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
std::map<std::string, Source> entries(const Plan& p)
{
    std::map<std::string, Source> read;
    for (const Segment& s : p.segments) {
        if (s.recipe == Recipe::kZeros || s.recipe == Recipe::kTable || s.recipe == Recipe::kActivations) continue;
        read.emplace(vulkan::entry_name(s.source), s.source);
        // A short upsample gain takes the tail of its layer's residual scales.
        if (s.flags & Segment::kGainTail) {
            const Source scales{s.source.directory, s.source.block, s.source.layer, Suffix::kResidualScale};
            read.emplace(vulkan::entry_name(scales), scales);
        }
    }
    return read;
}

// A model pack of synthetic entries (vulkan_test::synthetic_entry), of the
// real model's names and sizes, holding every entry the plan reads: the pack
// that upstream's synthetic goldens came from, but for the entries no plan
// reads.
bool synthetic_model(const Plan& p, vulkan_test::Pack& pack)
{
    const auto read = entries(p);
    std::string index = vulkan_test::pack_header(uint32_t(read.size()));
    uint64_t offset = index.size();
    for (const auto& [name, source] : read) offset += 4 + name.size() + 16;
    for (const auto& [name, source] : read) {
        index += vulkan_test::index_entry(name, offset, model_bytes(source));
        offset += model_bytes(source);
    }
    pack.write(index);
    for (const auto& [name, source] : read) pack.write(vulkan_test::synthetic_entry(name, model_bytes(source)));
    return pack.ok;
}

// PLAN's blob packed from MODEL, and its FNV-1a 64.
dlsslop::Result<uint64_t> blob_hash(const Plan& p, const vulkan::Model& model, std::vector<uint8_t>& blob)
{
    blob.resize(p.blob_bytes);
    DLSSLOP_TRY(vulkan::pack(p.segments, p.tables, model, blob));
    return fnv1a(blob.data(), blob.size());
}

void check_synthetic()
{
    const auto p = vulkan::plan(1280, 720);
    vulkan_test::Pack pack;
    if (!expect(p && synthetic_model(*p, pack), "cannot make the synthetic model")) return;
    const auto model = vulkan::Model::open(pack.path);
    if (!expect(bool(model), "the synthetic model: %s", model ? "" : model.error().what.c_str())) return;
    std::vector<uint8_t> blob;
    for (const auto& e : kPacked) {
        const auto q = vulkan::plan(e.width, e.height);
        const auto hash = q ? blob_hash(*q, *model, blob) : dlsslop::Result<uint64_t>();
        expect(q && hash && *hash == golden(e.width, e.height)->synthetic_blob,
               "%ux%u: the blob packed from the synthetic model differs from upstream's%s%s", e.width, e.height,
               q && !hash ? ": " : "", q && !hash ? hash.error().what.c_str() : "");
    }
}

// --model: the plan at each golden extent packed from the real model at PATH,
// against upstream's blob, and every entry the plans read of the size
// model_bytes() gives it.
int check_model(const std::string& path)
{
    const auto model = vulkan::Model::open(path);
    if (!expect(bool(model), "%s", model ? "" : model.error().what.c_str())) return 1;
    std::vector<uint8_t> blob;
    for (const Golden& g : kGoldens) {
        const auto p = vulkan::plan(g.width, g.height);
        if (!expect(bool(p), "%ux%u: %s", g.width, g.height, p ? "" : p.error().what.c_str())) continue;
        if (&g == kGoldens)
            for (const auto& [name, source] : entries(*p)) {
                const auto bytes = model->read(source, blob);
                expect(bytes && bytes->size() == model_bytes(source), "%s: %s", name.c_str(),
                       bytes ? "not the size the synthetic model gives it" : bytes.error().what.c_str());
            }
        const auto hash = blob_hash(*p, *model, blob);
        if (expect(hash && *hash == g.real_blob, "%ux%u: the blob packed from %s differs from upstream's%s%s",
                   g.width, g.height, path.c_str(), hash ? "" : ": ", hash ? "" : hash.error().what.c_str()))
            std::printf("%ux%u: blob %016" PRIx64 " as upstream's\n", g.width, g.height, *hash);
    }
    if (!failures) std::printf("vulkan-plan test: %s packs as upstream packed it\n", path.c_str());
    return failures ? 1 : 0;
}

// --print: the plan at WxH as the dumps of upstream's build have it, its
// dispatches' "D" lines and its values' "V" lines.
int print(const std::string& extent)
{
    unsigned width, height;
    char end;
    if (std::sscanf(extent.c_str(), "%ux%u%c", &width, &height, &end) != 2) return 2;
    const auto p = vulkan::plan(width, height);
    if (!p) {
        std::fprintf(stderr, "vulkan-plan test: %s\n", p.error().what.c_str());
        return 1;
    }
    std::printf("source %ux%u plan %ux%u\nact_total %" PRIu64 " values_end %" PRIu64 "\nwblob %u\ndisp %zu\n", width,
                height, p->work_width, p->work_height, p->arena_bytes, p->values_end, p->blob_bytes, p->steps.size());
    for (size_t i = 0; i < p->steps.size(); ++i) {
        const Step& s = p->steps[i];
        std::printf("D %3zu %-18s b%03ul%u grid %u %u %u steps %u..%u tcnobar %d chainnobar 0 push %u :", i,
                    stem(s.kernel), s.block, s.layer, s.groups[0], s.groups[1], s.groups[2], s.first, s.last,
                    s.after == After::kNothing, s.words * 4);
        for (uint32_t w = 0; w < s.words; ++w) std::printf(" %u", p->push[s.push + w]);
        std::printf("\n");
    }
    for (const vulkan::Value& v : p->values) std::printf("V %s", value_line(v).c_str());
    std::printf("tc_words %u\nnoise off %u %ux%u seed %u noise %g\n", p->counter_words, p->noise.off,
                p->noise.width, p->noise.height, p->noise.seed, double(p->noise.noise));
    return 0;
}

// --markers: the markers the plan expects beside the SPIR-V, each line of
// each file after the file's name.
int markers()
{
    std::printf("shader-constants.txt %s\n", vulkan::kManifest);
    for (const auto& c : vulkan::kShaderConstants) std::printf("shader-constants.txt %s %u\n", c.key, c.value);
    for (const auto& m : vulkan::kMarkers) std::printf("%s %s\n", m.file, m.line);
    return 0;
}

// --entries: the model entries the plans read at the golden extents.
int list_entries()
{
    std::map<std::string, Source> all;
    for (const Golden& g : kGoldens)
        if (const auto p = vulkan::plan(g.width, g.height)) all.merge(entries(*p));
    for (const auto& [name, source] : all) std::printf("%s\n", name.c_str());
    return all.empty() ? 1 : 0;
}

// A SPIR-V module's interface: its push-constant block's members, a type
// each ('u' u32, 'i' i32, 'f' f32, '?' anything else) at its offset, and its
// descriptor bindings of set 0 ('b' a storage buffer, 's' a storage image,
// 't' a sampled one, '?' anything else).
struct Interface {
    std::string push;
    std::vector<uint32_t> offsets;
    std::map<uint32_t, char> bindings;
};
bool interface_of(const std::string& module, Interface& face)
{
    const size_t count = module.size() / 4;
    std::vector<uint32_t> w(count);
    std::memcpy(w.data(), module.data(), count * 4);
    if (count < 5 || w[0] != 0x07230203) return false;
    std::map<uint32_t, char> scalar;                              // OpTypeInt, OpTypeFloat
    std::map<uint32_t, uint32_t> image, element, binding, set;    // sampled field; array element
    std::map<uint32_t, std::vector<uint32_t>> members;            // OpTypeStruct
    std::map<uint32_t, std::map<uint32_t, uint32_t>> offset;      // OpMemberDecorate Offset
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> pointer;    // storage class, type
    std::vector<std::pair<uint32_t, uint32_t>> variables;         // id, pointer type
    std::map<uint32_t, bool> buffer_block, sampled_image;
    for (size_t i = 5; i < count;) {
        const uint32_t op = w[i] & 0xffff, words = w[i] >> 16;
        if (!words || i + words > count) return false;
        const uint32_t* a = &w[i + 1];
        switch (op) {
        case 71: // OpDecorate: Binding, DescriptorSet, BufferBlock
            if (a[1] == 33) binding[a[0]] = a[2];
            if (a[1] == 34) set[a[0]] = a[2];
            if (a[1] == 3) buffer_block[a[0]] = true;
            break;
        case 72: // OpMemberDecorate: Offset
            if (a[2] == 35) offset[a[0]][a[1]] = a[3];
            break;
        case 21: scalar[a[0]] = a[1] != 32 ? '?' : a[2] ? 'i' : 'u'; break;
        case 22: scalar[a[0]] = a[1] == 32 ? 'f' : '?'; break;
        case 25: image[a[0]] = a[6]; break; // OpTypeImage: sampled 1 or 2
        case 27: sampled_image[a[0]] = true; break;
        case 28:
        case 29: element[a[0]] = a[1]; break; // OpTypeArray, OpTypeRuntimeArray
        case 30: members[a[0]].assign(a + 1, a + words - 1); break;
        case 32: pointer[a[0]] = {a[1], a[2]}; break;
        case 59: variables.push_back({a[1], a[0]}); break; // OpVariable
        }
        i += words;
    }
    for (const auto& [id, type] : variables) {
        const auto [storage, pointee] = pointer[type];
        if (storage == 9) { // PushConstant
            for (size_t m = 0; m < members[pointee].size(); ++m) {
                const auto t = scalar.find(members[pointee][m]);
                face.push += t == scalar.end() ? '?' : t->second;
                face.offsets.push_back(offset[pointee][uint32_t(m)]);
            }
            continue;
        }
        if (!binding.count(id) || set[id]) {
            if (binding.count(id)) face.bindings[1000 + set[id]] = '?';
            continue;
        }
        uint32_t t = pointee;
        while (element.count(t)) t = element[t];
        char kind = '?';
        if (storage == 12 || (storage == 2 && buffer_block[t])) kind = 'b';
        else if (storage == 0 && sampled_image[t]) kind = 't';
        else if (storage == 0 && image.count(t)) kind = image[t] == 2 ? 's' : 't';
        face.bindings[binding[id]] = kind;
    }
    return true;
}

// Each kernel's push block as the plan fills it, a type a word: its parts in
// a row, and the tile-counter word of the kernels that take one.
std::string push_types(Kernel k)
{
    using enum Kernel;
    const std::string swin = "uuuuuuuuuuuuuuuiiuuu", ds = std::string(11, 'u'), ups = std::string(10, 'u'),
                      image = "uuuufffffffu", tail = "uf", persist = std::string(8, 'u'),
                      gemm = std::string(21, 'u');
    switch (k) {
    case kFswin32: return swin + "u";
    case kFswinImagePreds32: return swin + image;
    case kFswinDsp32: return swin + ds + "u";
    case kFswinFusedUp32: return swin + ups + "uu";
    case kFswinImagePost32: return swin + ups + tail;
    case kFfwd3:
    case kFfwd3w: return std::string(8, 'u');
    case kAttn: return "uuuuuuuuuiiu";
    case kGemmPool:
    case kGemmNores:
    case kGemmVqkvs: return gemm;
    case kVitAttn: return std::string(6, 'u');
    case kDecUps: return std::string(7, 'u');
    case kUpsView: return std::string(8, 'u');
    case kNoiseField: return "uuuuf";
    default: return persistent(k) ? persist + "u" : gemm + "u";
    }
}

// --spirv: each kernel's SPIR-V in DIR against the push block and bindings the
// plan gives it; 77 when DIR is not a built network.
int check_spirv(const std::string& directory)
{
    if (!dlsslop::is_regular_file(dlsslop::join(directory, "shader-constants.txt"))) {
        std::printf("vulkan-plan test: skipped: no network SPIR-V in %s\n", directory.c_str());
        return 77;
    }
    for (size_t k = 0; k < size_t(Kernel::kCount); ++k) {
        const vulkan::KernelInfo& info = vulkan::kKernels[k];
        const std::string path = dlsslop::join(directory, std::string("g_") + info.stem + ".spv");
        const auto module = dlsslop::read_file(path);
        Interface face;
        if (!expect(module && interface_of(*module, face) && !face.bindings.empty(), "%s: %s", path.c_str(),
                    module ? "not SPIR-V with bindings" : module.error().what.c_str()))
            continue;
        // The block is the plan's, but gemmpool's, gemmnores' and gemmvqkvs'
        // end before remap, the plan's last word.
        const std::string types = push_types(Kernel(k));
        const bool short_gemm = Kernel(k) == Kernel::kGemmPool || Kernel(k) == Kernel::kGemmNores ||
                                Kernel(k) == Kernel::kGemmVqkvs;
        bool packed = face.push == types.substr(0, types.size() - short_gemm) && types.size() * 4 == info.push;
        for (size_t m = 0; packed && m < face.offsets.size(); ++m) packed = face.offsets[m] == 4 * m;
        expect(packed, "%s: a push block of %s, not the plan's %s%s in %u bytes", path.c_str(), face.push.c_str(),
               types.c_str(), short_gemm ? " without its last word" : "", info.push);
        std::string kinds(info.buffers);
        std::replace_if(kinds.begin(), kinds.end(), [](char c) { return c == 'a' || c == 'w'; }, 'b');
        if (info.images == vulkan::Images::kInput) kinds += "t";
        if (info.images == vulkan::Images::kOutput) kinds += "sst";
        for (const auto& [slot, kind] : face.bindings)
            expect(slot < kinds.size() && kinds[slot] == kind, "%s: binding %u is '%c', the plan's '%c'",
                   path.c_str(), slot, kind, slot < kinds.size() ? kinds[slot] : '-');
    }
    if (!failures) std::puts("vulkan-plan test: every kernel's push block and bindings are the plan's");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string model, spirv, extent;
    bool list_markers = false, list_read = false;
    const option options[] = {{"print", required_argument, nullptr, 'p'},   {"model", required_argument, nullptr, 'm'},
                              {"spirv", required_argument, nullptr, 's'},   {"markers", no_argument, nullptr, 'M'},
                              {"entries", no_argument, nullptr, 'e'},       {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+p:m:s:Meh", options, nullptr)) != -1;) {
        if (code == 'p') extent = optarg;
        else if (code == 'm') model = optarg;
        else if (code == 's') spirv = optarg;
        else if (code == 'M') list_markers = true;
        else if (code == 'e') list_read = true;
        else if (code == 'h') {
            std::puts("Usage: vulkan-plan-test [OPTION]...\n"
                      "Checks the Vulkan network's plan against goldens of upstream's graph build. No GPU or model\n"
                      "needed.\n"
                      " -p, --print WxH    Instead, print the plan for WxH frames in the form of the dumps of\n"
                      "                    upstream's build that the goldens come from (default: unset)\n"
                      " -m, --model FILE   Instead, pack the weight blob from the model pack FILE at every golden\n"
                      "                    extent and check it against upstream's (default: unset)\n"
                      " -s, --spirv DIR    Instead, check each kernel's SPIR-V in DIR against the push block and\n"
                      "                    bindings the plan gives it; exit 77 when DIR holds no network\n"
                      "                    (default: unset)\n"
                      " -M, --markers      Instead, print the markers the plan expects beside the SPIR-V, each\n"
                      "                    line after its file's name (default: off)\n"
                      " -e, --entries      Instead, print the model entries the plan reads (default: off)\n"
                      " -h, --help         Show help (default: off)");
            return 0;
        } else
            return 2;
    }
    if (optind != argc || !extent.empty() + !model.empty() + !spirv.empty() + list_markers + list_read > 1) return 2;
    if (!extent.empty()) return print(extent);
    if (!model.empty()) return check_model(model);
    if (!spirv.empty()) return check_spirv(spirv);
    if (list_markers) return markers();
    if (list_read) return list_entries();
    check_goldens();
    check_rejections();
    check_synthetic();
    if (!failures) std::puts("vulkan-plan test: every check passed");
    return failures ? 1 : 0;
}
