// SPDX-License-Identifier: MIT
// Host test of the Vulkan network's plan, without a GPU or a model. At 40
// frame extents from 17x17 to 5120x2880, the plan's steps with their push
// constants, as planned and with the kernels without the exponent's upper
// clamp that the real model's weights allow, the tables of its weight blob
// with the heads that model frees of that clamp, and its activation arena's
// values and sizes are checked against goldens of upstream's own
// NrSession::build (DLSSNR-AMD's nr_graph.cpp at 82560c4, built with its
// rdna4.sh defines and stopped before the device), and the weight blob packed
// from a synthetic model pack against upstream's. Where upstream's
// tile-counter records that wait each name an error word of their own and
// those that only signal none, the plan's all name one they share, and the
// persistent runs that upstream gives no record share one more, after the
// others: the goldens are checked with upstream's words put back and that
// record taken out. The frames the network cannot take are rejected.
#include "vulkan_pack.hpp"
#include "vulkan_plan.hpp"
#include "vulkan_weights.hpp"

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
using vulkan::ClampFree;
using vulkan::Kernel;
using vulkan::Plan;
using vulkan::Recipe;
using vulkan::Segment;
using vulkan::Source;
using vulkan::Step;
using vulkan::persistent;
using vulkan_test::entries;
using vulkan_test::fnv1a;
using vulkan_test::model_bytes;
using vulkan_test::synthetic_model;

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
// the tables in its weight blob for the real model, whose persistent runs'
// records carry the heads that model frees of the exponent's upper clamp,
// REAL_TABLES; of its dispatches, "kernel gx gy gz chained push-words...\n"
// each, as it planned them for the synthetic pack, which frees no head of that
// clamp, and for the real model, CLAMPED; of its values, "key bBLOCK/SLOT off
// OFFSET size BYTES overread BYTES\n" each; and of the weight blob it packed
// from the synthetic pack and from the real model. Recorded from upstream's
// build by a probe that dumps them where the build meets the device, the
// tables from the weight blob where the plan puts them.
struct Golden {
    uint32_t width, height, work_width, work_height, steps;
    uint64_t arena, values_end;
    uint32_t blob, counters, runs, chained;
    uint64_t real_tables, dispatches, clamped, values, synthetic_blob, real_blob;
};
constexpr Golden kGoldens[] = {
    {17, 17, 328, 320, 124, 27948288, 26550272, 155469644, 344224, 6, 99, 0x4cf188352abd04adu,
     0xac36df2d91920ebfu, 0x3a78a163f2c62f99u, 0x7c39a7c135a6f2abu, 0xafae60b012a87738u, 0xbbb5ac1f3aee438eu},
    {1, 400, 320, 448, 123, 36382976, 34963456, 155804972, 348335, 6, 100, 0xaebd9fc79b1f33c8u,
     0xf6d036c2e947b105u, 0x0976322fb7cd4e7fu, 0x2cf2a0137495ca85u, 0xb446e89512b6f371u, 0x30ebc0897abc17d7u},
    {400, 1, 448, 320, 123, 36382976, 34963456, 155804972, 348337, 6, 100, 0xbb144751d798c970u,
     0x7f533198b8b8012bu, 0x06f07580efaf4d39u, 0x2cf2a0137495ca85u, 0xfb702c0473262c71u, 0xaa0d893a65db36cbu},
    {32, 32, 328, 320, 124, 27948288, 26550272, 155469644, 344224, 6, 99, 0x4cf188352abd04adu,
     0x869b9f39fefc592fu, 0x1fb32f4ab8a5fc9du, 0x7c39a7c135a6f2abu, 0xafae60b012a87738u, 0xbbb5ac1f3aee438eu},
    {64, 64, 336, 320, 123, 28210432, 26796032, 155506540, 348321, 6, 100, 0x86b7b010ceb4ca38u,
     0x2026062dedaeeef7u, 0xb087316315fb19f1u, 0xb494e00c5a47741fu, 0x6e830263ad05ae39u, 0x93fd711bff4bb9b3u},
    {160, 120, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0x1856adec0aa2f035u, 0xd3638c3ecc405a9bu, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {200, 100, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0x09d80048b4ed851au, 0xce7b64232426ada0u, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {256, 144, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0xca589d388cdd3cd9u, 0x9d98e096eb2ed17bu, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {128, 320, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0x947919b23ced93ecu, 0x2e152e7050faa166u, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {320, 128, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0x864937e722be9f78u, 0xdc2315105520ab1au, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {320, 129, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0xaee336ab0f2c3d81u, 0xa8a8238280b6b26bu, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {320, 180, 320, 320, 123, 26447616, 25034752, 155462844, 348321, 6, 100, 0x7500245e6afda6b6u,
     0x4c90c77f72d49464u, 0xbf79b370fb94a3a2u, 0xc331ba7fb0e0bb1fu, 0x8e2cf06f232d449fu, 0xdf6111acc943189du},
    {333, 333, 384, 384, 123, 36973824, 35553280, 155839068, 348321, 6, 100, 0xfb828ac519522972u,
     0x2139df0bb258c75bu, 0xcb6ffcb66120cde5u, 0x6b33d951c0bcfbc7u, 0xdfd3e82f50838b5bu, 0x315a6453da11cb99u},
    {640, 360, 640, 384, 123, 60683520, 59244544, 156659772, 348351, 6, 100, 0xe1a9e58807e3095fu,
     0x5911d840006ddb38u, 0xf9501a487a69041eu, 0xd08ba5e9c77522d6u, 0x43997cb9a9682362u, 0x298e1e72ff78de6cu},
    {640, 480, 640, 512, 123, 80357888, 78905344, 157343020, 348372, 6, 100, 0xae97cdf54e807756u,
     0xc97f2bd563a789eau, 0x7ca3cbff0a3c61b8u, 0x4d6a79d85edfccd5u, 0x05327c7118b5518fu, 0x4b45520c4aa31d35u},
    {800, 450, 832, 512, 123, 104950272, 103481344, 158163244, 348409, 6, 100, 0x74409f22d32a02bdu,
     0x687acb224f8f437eu, 0xec870c63f184a2fcu, 0xc573bd1a9fee5da2u, 0x93d20ac39cd2c804u, 0xbdd9ca94170f3e42u},
    {854, 480, 896, 512, 123, 111902720, 110428160, 158436620, 348409, 6, 100, 0xaaac75596ebd079fu,
     0x3be0b6cca435e2bcu, 0x06945617d3795a82u, 0xa0b017d2a04de8a3u, 0x3e3fa671a849925eu, 0x44484c86cfacc6e4u},
    {960, 540, 960, 576, 123, 136035584, 134545408, 159222220, 348465, 6, 100, 0x381099b6f430eca2u,
     0x968fbf57aa9ac7fbu, 0xe7c70ba576a69f85u, 0x3edaf20da30af3b0u, 0x4b175d260ff59697u, 0x26e8c39324741db5u},
    {1001, 563, 1024, 576, 123, 144889088, 143392768, 159529740, 348465, 6, 100, 0xc9fb9da2297603dfu,
     0x3664069be7665671u, 0xac8cba61e19bd97fu, 0x66e151ad666eb8d5u, 0x5a4bdf60e83f5b62u, 0xe539c275017d7244u},
    {1152, 648, 1152, 704, 123, 198630400, 197099520, 161373804, 348521, 6, 100, 0xd48be3b1337fc1a5u,
     0xd32fb634cb07f8b9u, 0x587e28820c33604bu, 0x01f1ddd49336e51fu, 0xebc1477459610c94u, 0xd95616a70ea1e422u},
    {1024, 768, 1088, 768, 123, 204532224, 202997760, 161578476, 348521, 6, 100, 0xa09070219a74e9ffu,
     0x3c4e15c670d32b28u, 0x499020759066e94eu, 0x4333b55449e60434u, 0xb26d30d01531ebceu, 0x1569dd392fa589f4u},
    {1152, 720, 1152, 768, 123, 214571008, 213024768, 161988300, 348521, 6, 100, 0x25bf5c2564bd5183u,
     0x0b08f1d9ac8bac80u, 0x68fcc2b2e4d723b6u, 0x05f9d9530045eec9u, 0x35e94f7226ce9a2eu, 0x0fb2e499524fcdacu},
    {1280, 720, 1344, 768, 123, 252245248, 250675200, 163217868, 348573, 6, 100, 0x2ed8dd694aa7d4aeu,
     0x9f08890b04ad1d28u, 0x9f63e7a430a1cbfeu, 0x4006e4a89e059758u, 0xfa50bda18b15f553u, 0x463b2d1602eaa82du},
    {1280, 800, 1280, 832, 123, 200674560, 199098368, 163491788, 348849, 6, 100, 0xe3c3acf2583929b5u,
     0xc6cb9729a3d868c0u, 0xe55fefe1b31f61aeu, 0xddaea6e67fe051dfu, 0x529ef4e186561230u, 0x4e3e472b470d7ab6u},
    {1366, 768, 1408, 768, 123, 201332992, 199753728, 163629740, 349085, 6, 100, 0x6b9f956c87d2da32u,
     0x3320b417e8281e10u, 0xe1f57e1794a6320eu, 0x3caec83201885480u, 0x6560b642c8088fb7u, 0x1b13c2fd55e80371u},
    {1280, 1024, 1344, 1024, 123, 258743296, 257097728, 166105772, 353783, 6, 100, 0x6ff175906a28d441u,
     0xd10dc6d667efddb3u, 0x6bf66b0c97953fb9u, 0x94ca7bf838c1f8f0u, 0xea3faba919430a54u, 0x9ac8a68ab71e168eu},
    {1600, 900, 1600, 960, 123, 288876032, 287211520, 167447628, 356346, 6, 100, 0x235f08b72d1774ffu,
     0x209fc2229e3a69b8u, 0xe938de463e9396f6u, 0xfc5a22ba254a7b9bu, 0x2007c5b57979fe26u, 0x6ce309612a406d3cu},
    {1920, 804, 1920, 832, 123, 300358656, 298680320, 167964652, 357317, 6, 100, 0xbe56478f8dad1fc7u,
     0x8f28992475558749u, 0x6f0fa300d13b22ffu, 0xba97e75454f5b261u, 0x89b6a04213ba1412u, 0xa96bde52c687ef1cu},
    {1440, 1080, 1472, 1088, 123, 301178112, 299499520, 167997516, 357391, 6, 100, 0x8201d2ded474bb50u,
     0x83efdcf44565ddcdu, 0xf5fd245fdca6cc9bu, 0xda3fc3535c060b8du, 0x3c3fc2c27e09b0e9u, 0x1664670594cbdd33u},
    {1707, 960, 1728, 960, 123, 309939456, 308248576, 168479676, 358302, 6, 100, 0x7fb70444a0678865u,
     0x7fab20085dd5bf38u, 0xe7e9f56be5e96996u, 0xda824cd32854e759u, 0xf7dc249729243db0u, 0x1b8bfd181d9f661eu},
    {1680, 1050, 1728, 1088, 123, 353337856, 351600640, 170336556, 361818, 6, 100, 0x081527c00f8549b3u,
     0xcc68a1a4db8dec2du, 0x282116bef025e163u, 0x92dae6258ab9771fu, 0x1b25e539df770822u, 0x1f05bfe25e6d908cu},
    {1600, 1200, 1600, 1216, 123, 365541376, 363790336, 170886444, 362851, 6, 100, 0x860450d1f4de9a5cu,
     0x4b45cb8be1de8af1u, 0xb0006185cf85f66fu, 0x21ea8125c575fb4du, 0xc9d330536d56ab5du, 0x1af44bb847291f5fu},
    {1919, 1079, 1920, 1088, 123, 392277248, 390496256, 172090828, 365117, 6, 100, 0xb25a3b2c89d5c003u,
     0x9b7c8db6572477fcu, 0xc127cf793d036e32u, 0xf161d160a6330aedu, 0x84a31840953757eeu, 0x73ff13faea0fa0f4u},
    {1080, 1920, 1088, 1920, 123, 392277248, 390496256, 172090828, 365117, 6, 100, 0x9948bd38ebefab8bu,
     0xc66c181af4f3376du, 0x2dbb7da0fe59d51bu, 0xf161d160a6330aedu, 0xcca8a797e0723d46u, 0x28a6a626e3cfad70u},
    {1920, 1080, 1920, 1088, 123, 392277248, 390496256, 172090828, 365117, 6, 100, 0xb25a3b2c89d5c003u,
     0x0e77368975e0d934u, 0x1734ee75a89782f2u, 0xf161d160a6330aedu, 0x84a31840953757eeu, 0x73ff13faea0fa0f4u},
    {1920, 1200, 1920, 1216, 123, 438040064, 436207616, 174153932, 369013, 6, 100, 0x52278671bb7a4891u,
     0x49cae4fbd16848b7u, 0xfc6b1d1b266f02e5u, 0xcb29e66fcd7cdb28u, 0x04d64b1ac9150154u, 0xe58f55f4db4ae356u},
    {2560, 1080, 2560, 1088, 123, 522447872, 520519680, 177938444, 376187, 6, 100, 0x56c9206effdf7962u,
     0x6cccfc3eda776b97u, 0xa77a84f86f11fe39u, 0xcdc4561783fb376du, 0x2d24bc2b77d6b29fu, 0x6ce6c3315f303b41u},
    {2560, 1440, 2560, 1472, 123, 701895168, 699760640, 186189772, 391779, 6, 100, 0x7d0aa9148a223f9bu,
     0xcc305b8edbe08d17u, 0x6a33a69889c389adu, 0x9606066a7e165f97u, 0xe948c87a2abf746eu, 0xb00097ff073a25ecu},
    {3840, 2160, 3840, 2176, 123, 1545993728, 1543208960, 224419820, 395122, 6, 96, 0xd8773932e3b63714u,
     0x81270d9ddc925aa3u, 0x734d768a9bc76ce9u, 0x4b7752b8d303fce8u, 0x5649c365ee6a9e89u, 0x20fcabf9a0820abbu},
    {5120, 2880, 5120, 2880, 123, 2757088000, 2753167360, 277874828, 452881, 6, 96, 0x26cd30bad5bf2c28u,
     0xffd956f0618d26c6u, 0x7d350f0980f055b8u, 0x25c685b892ab46dcu, 0x1d57a895eb3bb0fdu, 0x13013f43e908e3ffu},
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

// The Swin heads of the real model, the pack that linux/package/model-tools
// extracts from nvngx_dlssnr 310.8.0, that its weights free of the exponent's
// upper clamp, by block (upstream's audit in its build, g_nohi_heads).
constexpr ClampFree kModelClampFree{{
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x03, 0x03, 0x09, 0x06, 0x0a, 0x01, 0x0f, 0x08, 0xf7, 0xa1, 0x6a,
    0x21, 0x51, 0xb6, 0x3e, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x28, 0x01, 0x10, 0x10, 0x11,
    0xa9, 0xf2, 0x04, 0x01, 0x00, 0x08, 0x03, 0x09, 0x03, 0x00, 0x02, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
}};

const char* stem(Kernel k) { return vulkan::kKernels[size_t(k)].stem; }
// The kernel whose twin without the upper clamp K is, or K.
Kernel clamped(Kernel k)
{
    const auto twin = std::ranges::find(vulkan::kUnclamped, k, &std::pair<Kernel, Kernel>::second);
    return twin == std::end(vulkan::kUnclamped) ? k : twin->first;
}
// P with FREE's kernels.
Plan unclamped(Plan p, const ClampFree& free)
{
    vulkan::unclamp(p, free);
    return p;
}
bool operator==(const ClampFree& a, const ClampFree& b) { return std::ranges::equal(a.heads, b.heads); }

// A tile-counter record (vulkan_schedule.cpp's chain()): 7 words, the counters
// its step waits on, the units it needs of each (the frame counter's tick), its
// table, the counters it signals, the frame counter, the error word its waits
// set when they give up, and the magic word.
enum Record : size_t { kWaits, kUnits, kTable, kSignals, kFrame, kError, kMagic, kRecordWords };
// Where in P.tables the record of step S starts, or SIZE_MAX: the table of a
// record's words at the last push word of a step.
size_t record(const Plan& p, const Step& s)
{
    const uint32_t word = p.push[s.push + s.words - 1];
    const auto table = std::ranges::find_if(p.segments, [&](const Segment& g) {
        return g.recipe == Recipe::kTable && g.offset == 4ull * word && g.bytes == 4 * kRecordWords;
    });
    if (word == ~0u || table == p.segments.end() || p.tables[table->index + kMagic] != 0x54434852u) return SIZE_MAX;
    return table->index;
}
// Where in P.tables the records of P's steps start.
std::vector<size_t> records(const Plan& p)
{
    std::vector<size_t> at;
    for (const Step& s : p.steps)
        if (const size_t r = record(p, s); r != SIZE_MAX) at.push_back(r);
    return at;
}
// P as upstream plans it: each record that waits names the word before its
// counters as its error word, and one that only signals none; a persistent run
// that neither waits nor signals names no record, where the plan's name one
// that orders nothing, the blob's last segment.
Plan upstream(Plan p)
{
    for (const size_t r : records(p))
        p.tables[r + kError] = p.tables[r + kTable] != ~0u ? p.tables[r + kWaits] - 1 : 0;
    const Segment last = p.segments.back();
    if (last.recipe != Recipe::kTable || last.bytes != 4 * kRecordWords) return p;
    const uint32_t* r = &p.tables[last.index];
    if (r[kMagic] != 0x54434852u || r[kWaits] != ~0u || r[kUnits] || r[kTable] != ~0u || r[kSignals] != ~0u) return p;
    for (const Step& s : p.steps)
        if (uint32_t& word = p.push[s.push + s.words - 1]; persistent(s.kernel) && 4ull * word == last.offset)
            word = ~0u;
    p.tables.resize(last.index);
    p.segments.pop_back();
    p.blob_bytes = p.segments.back().offset + p.segments.back().bytes;
    return p;
}

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
    expect(clamped(steps.front().kernel) == Kernel::kFswinImagePreds32 &&
               steps.back().kernel == Kernel::kFswinImagePost32,
           "%s: the steps do not start with the pre block and end with the post block", at);
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step& s = steps[i];
        const Kernel kernel = clamped(s.kernel);
        const uint32_t* w = &p.push[s.push];
        expect(s.words * 4u == vulkan::kKernels[size_t(s.kernel)].push && s.push + s.words <= p.push.size(),
               "%s: step %zu, %s, pushes %u words into a range of %u bytes", at, i, stem(s.kernel), s.words,
               vulkan::kKernels[size_t(s.kernel)].push);
        expect((s.after == After::kFull) == (i + 1 == steps.size()), "%s: step %zu is not followed as the %s", at, i,
               i + 1 == steps.size() ? "last" : "others");
        // No step reads the arena where it writes: nothing orders one
        // workgroup against another. A persistent run's layers are in its
        // records.
        if (persistent(kernel)) continue;
        uint64_t in[2] = {w[0], ~0ull}, out = w[1];
        if (kernel == Kernel::kFswinFusedUp32 || kernel == Kernel::kFswinImagePost32) {
            in[0] = w[sizeof(vulkan::PushFSwin) / 4];
            in[1] = w[sizeof(vulkan::PushFSwin) / 4 + 1];
        } else if (s.kernel >= Kernel::kGemmProjc && s.kernel <= Kernel::kGemmVqkvs) {
            const bool residual = s.kernel != Kernel::kGemmNores && s.kernel != Kernel::kGemmVact &&
                                  s.kernel != Kernel::kGemmVqkvNorm && s.kernel != Kernel::kGemmVqkvNorms &&
                                  s.kernel != Kernel::kGemmVqkvs;
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
    // fswinfusedup32nh leaves out the tests of its tiles: the fused upsample's
    // grid is unshifted and covers its tile raster whole.
    const auto up = std::ranges::find(steps, Kernel::kFswinFusedUp32, [](const Step& s) { return clamped(s.kernel); });
    vulkan::PushFSwin f{};
    if (up != steps.end()) std::memcpy(&f, &p.push[up->push], sizeof f);
    expect(up != steps.end() && !f.shift && !f.shift_y && f.tiles_x == 2 * up->groups[0] &&
               f.tiles_y == 2 * up->groups[1],
           "%s: the fused upsample is missing, shifted, or not on a grid of its tile raster", at);
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
    // The words that waits set when they give up: one that every record
    // names, after the frame counter and before every counter, then each
    // persistent run's, sync_off + 3, in its sync region below the frame
    // counter. Nothing else writes them. Every persistent run names a record.
    const std::vector<uint32_t>& t = p.timeouts;
    uint32_t frame = ~0u, counters = ~0u, waits = 0;
    for (const size_t r : records(p)) {
        frame = p.tables[r + kFrame];
        for (const Record c : {kWaits, kSignals}) counters = std::min(counters, p.tables[r + c]);
        expect(!t.empty() && p.tables[r + kError] == t[0], "%s: a record names another error word than the plan's",
               at);
        waits += p.tables[r + kTable] != ~0u;
    }
    std::vector<uint32_t> runs;
    for (const Step& s : steps)
        if (persistent(s.kernel)) {
            runs.push_back(p.push[s.push + offsetof(vulkan::PushPersist, sync_off) / 4] + 3);
            expect(record(p, s) != SIZE_MAX, "%s: persistent run %s names no record", at, stem(s.kernel));
        }
    expect(waits == p.chained && !t.empty() && t[0] == frame + 1 && t[0] < counters &&
               std::ranges::equal(t.begin() + 1, t.end(), runs.begin(), runs.end()),
           "%s: the words that waits set are not the shared one, then each persistent run's", at);
    for (size_t i = 1; i < t.size(); ++i)
        expect(4ull * t[i] >= p.values_end && t[i] < frame && (i == 1 || t[i] > t[i - 1]),
               "%s: a persistent run's word that its waits set is outside its own sync region", at);
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
        const Plan free = unclamped(*p, kModelClampFree), theirs = upstream(free);
        expect(theirs.blob_bytes == g.blob, "%ux%u: a weight blob of %u bytes, upstream's %u", g.width, g.height,
               theirs.blob_bytes, g.blob);
        expect(dispatch_hash(upstream(*p)) == g.dispatches, "%ux%u: the steps differ from upstream's dispatches",
               g.width, g.height);
        expect(dispatch_hash(theirs) == g.clamped,
               "%ux%u: the steps with the real model's kernels differ from upstream's dispatches (--print)", g.width,
               g.height);
        expect(fnv1a(theirs.tables.data(), 4 * theirs.tables.size()) == g.real_tables,
               "%ux%u: the tables with the real model's free heads differ from upstream's", g.width, g.height);
        expect(value_hash(*p) == g.values, "%ux%u: the values differ from upstream's (--print)", g.width, g.height);
        check_invariants(*p);
        check_invariants(free);
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
    refused(1280, 720, "its activation arena of 252245248" + storage + "252245247 bytes", 252245247);
    refused(1280, 720, "its activation arena of 252245248" + storage + "163217867 bytes", 163217867);
    expect(bool(vulkan::plan(1280, 720, 252245248)), "1280x720: rejected on a device that holds its arena");
}

// PLAN's blob packed from MODEL, with the kernels without the upper clamp
// that MODEL frees, the heads in FREE, and its FNV-1a 64.
dlsslop::Result<uint64_t> blob_hash(Plan p, const vulkan::Model& model, std::vector<uint8_t>& blob, ClampFree& free)
{
    free = DLSSLOP_TRY(vulkan::clamp_free(p.segments, model));
    vulkan::unclamp(p, free);
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
        ClampFree free{};
        const auto hash = q ? blob_hash(upstream(*q), *model, blob, free) : dlsslop::Result<uint64_t>();
        expect(q && hash && *hash == golden(e.width, e.height)->synthetic_blob,
               "%ux%u: the blob packed from the synthetic model differs from upstream's%s%s", e.width, e.height,
               q && !hash ? ": " : "", q && !hash ? hash.error().what.c_str() : "");
        expect(free == ClampFree{}, "%ux%u: the synthetic model frees a head of the upper clamp", e.width, e.height);
    }
}

// --model: the plan at each golden extent packed from the real model at PATH,
// against upstream's blob, the heads it frees of the upper clamp against
// upstream's audit, and every entry the plans read of the size model_bytes()
// gives it.
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
        ClampFree free{};
        const auto hash = blob_hash(upstream(*p), *model, blob, free);
        expect(!hash || free == kModelClampFree, "%ux%u: %s frees other heads of the upper clamp than upstream's audit",
               g.width, g.height, path.c_str());
        if (expect(hash && *hash == g.real_blob, "%ux%u: the blob packed from %s differs from upstream's%s%s",
                   g.width, g.height, path.c_str(), hash ? "" : ": ", hash ? "" : hash.error().what.c_str()))
            std::printf("%ux%u: blob %016" PRIx64 " as upstream's\n", g.width, g.height, *hash);
    }
    if (!failures) std::printf("vulkan-plan test: %s packs as upstream packed it\n", path.c_str());
    return failures ? 1 : 0;
}

// --print: the plan at WxH with the real model's kernels and upstream's
// records, as the dumps of upstream's build of the real model have it, its
// dispatches' "D" lines and its values' "V" lines.
int print(const std::string& extent)
{
    unsigned width, height;
    char end;
    if (std::sscanf(extent.c_str(), "%ux%u%c", &width, &height, &end) != 2) return 2;
    auto p = vulkan::plan(width, height);
    if (!p) {
        std::fprintf(stderr, "vulkan-plan test: %s\n", p.error().what.c_str());
        return 1;
    }
    vulkan::unclamp(*p, kModelClampFree);
    *p = upstream(std::move(*p));
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

// --defines: what the plan assumes of the defines its pipelines are built
// with beyond the markers, "pipeline define value" a line: NR_ATTN_WPAIR 1
// when the 720p plan packs attn's QKV matrix N-paired, and gemmvqkvnorms'
// tile of kGemmQkvsMt tokens (gemm1x1's NR_MTILE) by kGemmQkvsNt outputs
// (NR_NTILE).
int defines()
{
    const auto p = vulkan::plan(1280, 720);
    if (!p) return 1;
    const auto attn = std::ranges::find(p->steps, Kernel::kAttn, &Step::kernel);
    if (attn == p->steps.end()) return 1;
    vulkan::PushAttn a;
    std::memcpy(&a, &p->push[attn->push], sizeof a);
    const auto weights = std::ranges::find(p->segments, a.w_off, &Segment::offset);
    if (weights == p->segments.end()) return 1;
    std::printf("attn NR_ATTN_WPAIR %d\n", weights->flags & Segment::kNpair ? 1 : 0);
    std::printf("gemmvqkvnorms NR_MTILE %u\ngemmvqkvnorms NR_NTILE %u\n", vulkan::kGemmQkvsMt, vulkan::kGemmQkvsNt);
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
// each ('u' u32, 'i' i32, 'f' f32, '?' anything else) at its offset, its
// descriptor bindings of set 0 ('b' a storage buffer, 's' a storage image,
// 't' a sampled one, '?' anything else), and those of them that an
// instruction loads, stores or reaches into.
struct Interface {
    std::string push;
    std::vector<uint32_t> offsets;
    std::map<uint32_t, char> bindings;
    std::map<uint32_t, bool> used;
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
    std::map<uint32_t, bool> buffer_block, sampled_image, reached; // reached: loaded, stored, chained into
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
        case 60: // OpImageTexelPointer
        case 61: // OpLoad
        case 65: // OpAccessChain
        case 66: // OpInBoundsAccessChain
        case 67: reached[a[2]] = true; break; // OpPtrAccessChain
        case 62: // OpStore
        case 63: reached[a[0]] = true; break; // OpCopyMemory
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
        face.used[binding[id]] = reached[id];
    }
    return true;
}

// Each kernel's push block as the plan fills it, a type a word: its parts in
// a row, and the tile-counter word of the kernels that take one. A twin
// without the upper clamp takes its kernel's.
std::string push_types(Kernel k)
{
    using enum Kernel;
    const std::string swin = "uuuuuuuuuuuuuuuiiuuu", ds = std::string(11, 'u'), ups = std::string(10, 'u'),
                      image = "uuuufffffffu", tail = "uf", persist = std::string(8, 'u'),
                      gemm = std::string(21, 'u');
    switch (clamped(k)) {
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
// plan gives it, and the post block's stores against the images the runtime
// binds; 77 when DIR is not a built network.
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
        // The post block stores the answer and never its second output, where
        // the runtime binds a 1x1 image unless the temporal post block writes it.
        const uint32_t answer = uint32_t(std::strlen(info.buffers));
        if (info.images == vulkan::Images::kOutput)
            expect(face.used[answer] && !face.used[answer + 1], "%s: the answer is %sstored, the second output %sstored",
                   path.c_str(), face.used[answer] ? "" : "not ", face.used[answer + 1] ? "" : "not ");
    }
    if (!failures) std::puts("vulkan-plan test: every kernel's push block and bindings are the plan's");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string model, spirv, extent;
    bool list_markers = false, list_defines = false, list_read = false;
    const option options[] = {{"print", required_argument, nullptr, 'p'},   {"model", required_argument, nullptr, 'm'},
                              {"spirv", required_argument, nullptr, 's'},   {"markers", no_argument, nullptr, 'M'},
                              {"defines", no_argument, nullptr, 'D'},       {"entries", no_argument, nullptr, 'e'},
                              {"help", no_argument, nullptr, 'h'},          {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+p:m:s:MDeh", options, nullptr)) != -1;) {
        if (code == 'p') extent = optarg;
        else if (code == 'm') model = optarg;
        else if (code == 's') spirv = optarg;
        else if (code == 'M') list_markers = true;
        else if (code == 'D') list_defines = true;
        else if (code == 'e') list_read = true;
        else if (code == 'h') {
            std::puts("Usage: vulkan-plan-test [OPTION]...\n"
                      "Checks the Vulkan network's plan against goldens of upstream's graph build. No GPU or model\n"
                      "needed.\n"
                      " -p, --print WxH    Instead, print the plan for WxH frames with the real model's kernels in\n"
                      "                    the form of the dumps of upstream's build that the goldens come from\n"
                      "                    (default: unset)\n"
                      " -m, --model FILE   Instead, pack the weight blob from the model pack FILE at every golden\n"
                      "                    extent and check it, and the heads that FILE frees of the exponent's\n"
                      "                    upper clamp, against upstream's (default: unset)\n"
                      " -s, --spirv DIR    Instead, check each kernel's SPIR-V in DIR against the push block and\n"
                      "                    bindings the plan gives it, and that the post block stores the answer\n"
                      "                    and not its second output; exit 77 when DIR holds no network\n"
                      "                    (default: unset)\n"
                      " -M, --markers      Instead, print the markers the plan expects beside the SPIR-V, each\n"
                      "                    line after its file's name (default: off)\n"
                      " -D, --defines      Instead, print what the plan assumes of its pipelines' defines beyond\n"
                      "                    the markers, \"pipeline define value\" a line (default: off)\n"
                      " -e, --entries      Instead, print the model entries the plan reads (default: off)\n"
                      " -h, --help         Show help (default: off)");
            return 0;
        } else
            return 2;
    }
    if (optind != argc ||
        !extent.empty() + !model.empty() + !spirv.empty() + list_markers + list_defines + list_read > 1)
        return 2;
    if (!extent.empty()) return print(extent);
    if (!model.empty()) return check_model(model);
    if (!spirv.empty()) return check_spirv(spirv);
    if (list_markers) return markers();
    if (list_defines) return defines();
    if (list_read) return list_entries();
    check_goldens();
    check_rejections();
    check_synthetic();
    if (!failures) std::puts("vulkan-plan test: every check passed");
    return failures ? 1 : 0;
}
