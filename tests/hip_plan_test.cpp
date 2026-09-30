// SPDX-License-Identifier: MIT
// Host test of the HIP network's launch plan, without a GPU or a model. At
// every tier and preset its launches, buffers and uploads are checked against
// traces that the tracing HIP runtime (tests/hiptrace) took of upstream's
// network (hip_reference_network.h at c190831) in dlsslopd: hashes of the
// launch lists with buffers named by first use, the pool's buffers, the
// weights and the gather maps each launch reads. tests/hiptrace/plan_hashes.py
// prints these values from a trace. --modules checks the launches against the
// built kernels' metadata instead.
#include "../backend/files.h"
#include "../backend/hip_plan.h"
#include "../external/layer/common/shm_protocol.h"

#include <elf.h>
#include <getopt.h>
#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
using dlsslop::hip::Arg;
using dlsslop::hip::Kernel;
using dlsslop::hip::Launch;
using dlsslop::hip::Plan;
namespace hip = dlsslop::hip;

// What the traces show at a tier and preset: launches and distinct kernels
// per frame, and the FNV-1a 64 of the canonical launch list of the first
// frame, of a later one, and of a later one with a history (0: not traced),
// and of the weight uploads that a launch reads (upload_text()).
struct Traced {
    unsigned tier;
    bool performance;
    size_t launches, kernels;
    uint64_t first, later, history, uploads;
};
constexpr Traced kTraced[] = {
    {720, false, 250, 43, 0xa2dae6593e416beeu, 0x1f7c6f29d1c28a84u, 0x240a88c7eb4b11a6u, 0x0ee01bec21d79f03u},
    {720, true, 229, 43, 0x834bc420b12694dcu, 0x892cd4d1d3db66a5u, 0, 0xfd99729c88d07861u},
    {900, false, 254, 41, 0x32d0cdfcc08b67e2u, 0x0c823eac9f698da2u, 0, 0xf023500cbbedf100u},
    {900, true, 233, 41, 0xff9623a799b34742u, 0x42c5410c76dcbbb9u, 0, 0x38d51bf165148771u},
    {1080, false, 254, 42, 0x18bff6041650c106u, 0xdffb8c9588486891u, 0x4f31a38dc45a9974u, 0xf023500cbbedf100u},
    {1080, true, 233, 42, 0x1f5cb0ebce0e78d8u, 0xfb73f9fd4c0be465u, 0, 0x38d51bf165148771u},
};
// Per tier, with either preset: the pool's buffers in creation order, ending
// with 0, and the FNV-1a 64 of the gather maps in the order the launches first
// read them.
struct TracedTier {
    unsigned tier;
    size_t buffers[16];
    uint64_t gather[2];
};
constexpr TracedTier kTracedTiers[] = {
    {720,
     {31457280, 31457280, 15728640, 15925248, 11796480, 3932160, 1966080, 7864320, 1966080, 491520, 1474560, 3145728,
      3145728, 3145728, 31457280},
     {0x9089c42dbe84bd25u, 0xa6bc7ecb30496925u}},
    {900,
     {49152000, 49152000, 24576000, 24821760, 18432000, 6144000, 3072000, 12288000, 3670016, 3670016, 917504, 2752512,
      4587520, 4587520, 49152000},
     {0x87ab221ca3c6a125u, 0xc8863327344d5b25u}},
    {1080,
     {70778880, 70778880, 35389440, 35684352, 26542080, 8847360, 4423680, 17694720, 5242880, 5242880, 1310720, 3932160,
      5242880, 70778880},
     {0xd4e847b0089b1b25u, 0xe7cf48cabb5e6325u}},
};
// Without and with --performance: the weights uploaded, their bytes, and the
// FNV-1a 64 of identity_text(). The traces hold only the payloads. The names
// are those under which the pinned upstream packers, run on the real model,
// made the same payloads in the same order; no two payloads are alike.
// Upstream also uploads 32 weights, 29 with --performance, that no launch
// reads, and dlsslopd leaves them out: these values and the upload hashes
// above are of the traced uploads without them. A weight that no launch reads
// fails the upload hash, as upload_text() writes "BYTES -" for it.
constexpr struct {
    size_t count, bytes;
    uint64_t identity;
} kTracedWeights[] = {{236, 644222504, 0x675dbc4f8229a0b1u}, {224, 608027816, 0xec181f70bf0130c9u}};

int failures = 0;
// Whether OK; if not, the test fails with the message.
[[gnu::format(printf, 2, 3)]] bool expect(bool ok, const char* format, ...)
{
    if (ok) return true;
    std::va_list args;
    va_start(args, format);
    std::fputs("hip-plan test: ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    ++failures;
    return false;
}

uint64_t fnv1a(const void* data, size_t bytes)
{
    uint64_t hash = 0xcbf29ce484222325u;
    for (auto* at = static_cast<const unsigned char*>(data); bytes--; ++at) hash = (hash ^ *at) * 0x100000001b3u;
    return hash;
}
uint64_t fnv1a(std::string_view text) { return fnv1a(text.data(), text.size()); }

const hip::KernelInfo& info(Kernel kernel) { return hip::kKernels[size_t(kernel)]; }
bool pointer(Arg arg) { return arg.kind != Arg::kU32 && arg.kind != Arg::kF32 && arg.kind != Arg::kTemporal; }

// The launch list as the trace tools write a traced one: a line per launch,
// "MODULE:KERNEL\tGRIDx1x1\tTHREADSx1x1\t0\tARGUMENTS", with each buffer
// named B<n>+0 in the order of first use. BUFFER_OF places the tensors; a
// frame without a HISTORY passes its input in its stead.
std::string canonical_text(const Plan& plan, const std::vector<uint16_t>& buffer_of, bool history)
{
    std::map<uint64_t, size_t> names;
    std::string text;
    for (const Launch& l : plan.launches) {
        const hip::KernelInfo& k = info(l.kernel);
        text += std::string(hip::kModuleFiles[size_t(k.module)]) + ":" + k.name + "\t" + std::to_string(l.grid) +
                "x1x1\t" + std::to_string(k.threads) + "x1x1\t0\t";
        for (unsigned i = 0; i < l.count; ++i) {
            const Arg a = l.args[i];
            if (i) text += ',';
            if (a.kind == Arg::kNull) {
                text += "null";
                continue;
            }
            if (!pointer(a)) {
                text += std::to_string(a.kind == Arg::kTemporal ? unsigned(history) : a.value);
                continue;
            }
            const Arg::Kind kind = a.kind == Arg::kHistory && !history ? Arg::kRgba : a.kind;
            const uint64_t buffer = uint64_t(kind) << 32 | (kind == Arg::kTensor ? buffer_of[a.value] : a.value);
            text += "B" + std::to_string(names.emplace(buffer, names.size()).first->second) + "+0";
        }
        text += '\n';
    }
    return text;
}

// The weight uploads as the trace tools list them: a line per weight, in
// upload order, of its bytes and the first launch and argument that read it
// in the first frame, "BYTES LAUNCH ARGUMENT", or "BYTES -" when none does.
std::string upload_text(const Plan& plan)
{
    std::string text;
    for (uint32_t w = 0; w < plan.weights.size(); ++w) {
        text += std::to_string(hip::packed_bytes(plan.weights[w]));
        std::string use = " -";
        for (size_t n = 0; n < plan.launches.size() && use == " -"; ++n)
            for (unsigned i = 0; i < plan.launches[n].count; ++i)
                if (plan.launches[n].args[i].kind == Arg::kWeight && plan.launches[n].args[i].value == w) {
                    use = " " + std::to_string(n) + " " + std::to_string(i);
                    break;
                }
        text += use + "\n";
    }
    return text;
}

// Which weight each upload is, a line per weight in upload order:
// "STEM@SUFFIX C", or "STEM C" for a raw one, with the channel count upstream
// passes its loader.
std::string identity_text(const Plan& plan)
{
    std::string text;
    for (const hip::WeightSpec& spec : plan.weights) {
        const char* suffix = hip::kRecipeSuffix[size_t(spec.recipe)];
        text += std::string(spec.stem) + (*suffix ? "@" : "") + suffix + " " + std::to_string(hip::channels(spec)) +
                "\n";
    }
    return text;
}

void check_plans()
{
    const auto unsupported = hip::plan(1280, 720, false);
    expect(!unsupported && unsupported.error().what == "unsupported processing geometry 1280x720",
           "a 1280x720 plan does not fail as unsupported");
    std::set<Kernel> launched;
    std::string weights[2];
    for (const NativeTier& tier : kNativeTiers)
        for (const bool performance : {false, true}) {
            const Traced* traced = nullptr;
            for (const Traced& t : kTraced)
                if (t.tier == tier.height && t.performance == performance) traced = &t;
            const TracedTier* pool = nullptr;
            for (const TracedTier& t : kTracedTiers)
                if (t.tier == tier.height) pool = &t;
            const char* preset = performance ? " --performance" : "";
            if (!expect(traced && pool, "tier %u%s has no traced values", tier.height, preset)) continue;
            const auto plan = hip::plan(tier.width, tier.networkHeight, performance);
            if (!expect(bool(plan), "tier %u%s: %s", tier.height, preset, plan ? "" : plan.error().what.c_str()))
                continue;
            std::set<Kernel> kernels;
            for (const Launch& l : plan->launches) kernels.insert(l.kernel);
            launched.insert(kernels.begin(), kernels.end());
            expect(plan->launches.size() == traced->launches && kernels.size() == traced->kernels,
                   "tier %u%s: %zu launches of %zu kernels, upstream's %zu of %zu", tier.height, preset,
                   plan->launches.size(), kernels.size(), traced->launches, traced->kernels);

            const auto placement = hip::place(*plan);
            if (!expect(bool(placement), "tier %u%s: %s", tier.height, preset,
                        placement ? "" : placement.error().what.c_str()))
                continue;
            size_t buffers = 0;
            while (pool->buffers[buffers]) ++buffers;
            expect(std::vector<size_t>(pool->buffers, pool->buffers + buffers) == placement->buffers,
                   "tier %u%s: the pool's buffers differ from upstream's", tier.height, preset);
            const struct {
                const char* frame;
                const std::vector<uint16_t>& buffer_of;
                bool history;
                uint64_t hash;
            } frames[] = {{"the first frame", placement->first, false, traced->first},
                          {"later frames", placement->later, false, traced->later},
                          {"frames with a history", placement->later, true, traced->history}};
            for (const auto& f : frames) {
                const uint64_t hash = fnv1a(canonical_text(*plan, f.buffer_of, f.history));
                expect(!f.hash || hash == f.hash,
                       "tier %u%s: the launches of %s hash to %016" PRIx64 ", upstream's to %016" PRIx64, tier.height,
                       preset, f.frame, hash, f.hash);
            }

            size_t bytes = 0;
            for (const hip::WeightSpec& spec : plan->weights) bytes += hip::packed_bytes(spec);
            const auto& w = kTracedWeights[performance];
            expect(plan->weights.size() == w.count && bytes == w.bytes,
                   "tier %u%s: %zu weights of %zu bytes; upstream's launches read %zu of %zu", tier.height, preset,
                   plan->weights.size(), bytes, w.count, w.bytes);
            const std::string identity = identity_text(*plan);
            expect(fnv1a(identity) == w.identity, "tier %u%s: other weights uploaded than upstream's launches read",
                   tier.height, preset);
            expect(fnv1a(upload_text(*plan)) == traced->uploads,
                   "tier %u%s: weights uploaded otherwise than upstream's launches read them", tier.height, preset);
            // One model serves every tier.
            if (weights[performance].empty()) weights[performance] = identity;
            expect(identity == weights[performance], "tier %u%s: weights differ from tier %u's", tier.height, preset,
                   kNativeTiers[0].height);
            // The gather maps, in the order the launches first read them.
            std::vector<uint32_t> maps;
            for (size_t n = 0; n < plan->launches.size(); ++n)
                for (unsigned i = 0; i < plan->launches[n].count; ++i) {
                    const Arg a = plan->launches[n].args[i];
                    if (a.kind != Arg::kGather || std::ranges::count(maps, a.value)) continue;
                    const std::vector<uint32_t> map = hip::gather_map(plan->tokens, bool(a.value));
                    expect(maps.size() < 2 && fnv1a(map.data(), map.size() * 4) == pool->gather[maps.size()],
                           "tier %u%s: the gather map launch %zu reads differs from upstream's", tier.height, preset,
                           n);
                    maps.push_back(a.value);
                }
            expect(maps.size() == 2, "tier %u%s: %zu gather maps read; upstream reads 2", tier.height, preset,
                   maps.size());
        }
    // At the tiers, the first block of the c64 and c128 levels is never mapped.
    for (size_t k = 0; k < size_t(Kernel::kCount); ++k) {
        const bool unused = Kernel(k) == Kernel::kFfnC64Mapped || Kernel(k) == Kernel::kFfnC128Mapped;
        expect(bool(launched.count(Kernel(k))) != unused, "%s is %slaunched", info(Kernel(k)).name,
               unused ? "" : "never ");
    }
}

// A msgpack reader, for as much of it as the AMDGPU metadata uses.
class MsgPack {
    const uint8_t* at_;
    const uint8_t* end_;
    bool ok_ = true;

    // The next N bytes, big-endian.
    uint64_t take(size_t n)
    {
        if (size_t(end_ - at_) < n) {
            ok_ = false;
            at_ = end_;
            return 0;
        }
        uint64_t value = 0;
        for (size_t i = 0; i < n; ++i) value = value << 8 | at_[i];
        at_ += n;
        return value;
    }

public:
    // A value, but for a container's items: a string's text, an unsigned
    // integer, or a container's length.
    struct Head {
        enum Type { kOther, kString, kNumber, kArray, kMap } type;
        uint64_t value;
        std::string_view text;
    };
    MsgPack(const void* data, size_t bytes) : at_(static_cast<const uint8_t*>(data)), end_(at_ + bytes) {}
    bool ok() const { return ok_; }

    Head head()
    {
        const unsigned c = unsigned(take(1));
        size_t bytes = 0;
        bool string = false;
        if (c <= 0x7f) return {Head::kNumber, c, {}};
        if ((c & 0xf0) == 0x80) return {Head::kMap, c & 15u, {}};
        if ((c & 0xf0) == 0x90) return {Head::kArray, c & 15u, {}};
        if ((c & 0xe0) == 0xa0) {
            bytes = c & 31u;
            string = true;
        } else if (c >= 0xe0 || c == 0xc0 || c == 0xc2 || c == 0xc3) {
            return {Head::kOther, 0, {}};
        } else if (c >= 0xcc && c <= 0xcf) {
            return {Head::kNumber, take(size_t(1) << (c - 0xcc)), {}};
        } else if (c >= 0xd9 && c <= 0xdb) {
            bytes = take(size_t(1) << (c - 0xd9));
            string = true;
        } else if (c == 0xdc || c == 0xdd) {
            return {Head::kArray, take(c == 0xdc ? 2 : 4), {}};
        } else if (c == 0xde || c == 0xdf) {
            return {Head::kMap, take(c == 0xde ? 2 : 4), {}};
        } else if (c >= 0xd0 && c <= 0xd3) {
            bytes = size_t(1) << (c - 0xd0); // Signed integers.
        } else if (c == 0xca || c == 0xcb) {
            bytes = c == 0xca ? 4 : 8;
        } else if (c >= 0xc4 && c <= 0xc6) {
            bytes = take(size_t(1) << (c - 0xc4)); // Binary.
        } else {
            ok_ = false; // Extension types.
            return {Head::kOther, 0, {}};
        }
        if (size_t(end_ - at_) < bytes) {
            ok_ = false;
            return {Head::kOther, 0, {}};
        }
        const std::string_view text(reinterpret_cast<const char*>(at_), bytes);
        at_ += bytes;
        return string ? Head{Head::kString, 0, text} : Head{Head::kOther, 0, {}};
    }
    // The items of the container whose head is HEAD.
    void skip(const Head& head)
    {
        const uint64_t items = head.type == Head::kMap ? 2 * head.value : head.type == Head::kArray ? head.value : 0;
        for (uint64_t i = 0; i < items && ok_; ++i) skip(this->head());
    }
};

// A kernel's explicit arguments, their kind and bytes, and its largest group.
struct KernelAbi {
    std::vector<std::pair<std::string, uint64_t>> args;
    uint64_t max_group = 0;
};

// The kernels that the NT_AMDGPU_METADATA note of the code object IMAGE
// describes; none when it has none.
std::map<std::string, KernelAbi> kernels_of(const std::string& image)
{
    std::map<std::string, KernelAbi> kernels;
    Elf64_Ehdr eh;
    if (image.size() < sizeof eh) return kernels;
    std::memcpy(&eh, image.data(), sizeof eh);
    if (std::memcmp(eh.e_ident, ELFMAG, SELFMAG) || eh.e_ident[EI_CLASS] != ELFCLASS64) return kernels;
    for (unsigned s = 0; s < eh.e_shnum; ++s) {
        Elf64_Shdr sh;
        const size_t at = eh.e_shoff + size_t(s) * eh.e_shentsize;
        if (at + sizeof sh > image.size()) break;
        std::memcpy(&sh, image.data() + at, sizeof sh);
        if (sh.sh_type != SHT_NOTE || sh.sh_offset + sh.sh_size > image.size()) continue;
        const char* notes = image.data() + sh.sh_offset;
        for (size_t note = 0; note + 12 <= sh.sh_size;) {
            uint32_t word[3]; // Name bytes, description bytes, type.
            std::memcpy(word, notes + note, 12);
            const size_t name = note + 12, description = name + ((word[0] + 3) & ~3u);
            note = description + ((word[1] + 3) & ~3u);
            if (note > sh.sh_size) break;
            if (word[2] != 32 || word[0] != 7 || std::memcmp(notes + name, "AMDGPU", 7)) continue;
            MsgPack m(notes + description, word[1]);
            const MsgPack::Head root = m.head();
            for (uint64_t i = 0; i < (root.type == MsgPack::Head::kMap ? root.value : 0) && m.ok(); ++i) {
                const MsgPack::Head key = m.head(), list = m.head();
                if (key.text != "amdhsa.kernels" || list.type != MsgPack::Head::kArray) {
                    m.skip(list);
                    continue;
                }
                for (uint64_t k = 0; k < list.value && m.ok(); ++k) {
                    const MsgPack::Head fields = m.head();
                    std::string_view kernel_name;
                    KernelAbi kernel;
                    for (uint64_t f = 0; f < (fields.type == MsgPack::Head::kMap ? fields.value : 0) && m.ok(); ++f) {
                        const MsgPack::Head field = m.head(), value = m.head();
                        if (field.text == ".name") kernel_name = value.text;
                        if (field.text == ".max_flat_workgroup_size") kernel.max_group = value.value;
                        if (field.text != ".args" || value.type != MsgPack::Head::kArray) {
                            m.skip(value);
                            continue;
                        }
                        for (uint64_t a = 0; a < value.value && m.ok(); ++a) {
                            const MsgPack::Head arg = m.head();
                            std::pair<std::string, uint64_t> described;
                            for (uint64_t p = 0; p < (arg.type == MsgPack::Head::kMap ? arg.value : 0) && m.ok(); ++p) {
                                const MsgPack::Head property = m.head(), setting = m.head();
                                if (property.text == ".value_kind") described.first = setting.text;
                                if (property.text == ".size") described.second = setting.value;
                                m.skip(setting);
                            }
                            if (!described.first.starts_with("hidden_")) kernel.args.push_back(described);
                        }
                    }
                    if (m.ok()) kernels.emplace(kernel_name, std::move(kernel));
                }
            }
        }
    }
    return kernels;
}

// Every planned launch against its kernel's metadata in the modules in
// DIRECTORY: its arguments, pointers or 4-byte values, and its group size.
int check_abi(const std::string& directory)
{
    std::map<std::string, KernelAbi> modules[size_t(hip::Module::kCount)];
    for (size_t m = 0; m < std::size(modules); ++m) {
        const std::string path = dlsslop::join(directory, hip::kModuleFiles[m]);
        if (!dlsslop::is_regular_file(path)) {
            std::printf("hip-plan test: skipped: no module %s\n", path.c_str());
            return 77;
        }
        const auto image = dlsslop::read_file(path);
        if (!expect(bool(image), "%s", image ? "" : image.error().what.c_str())) return 1;
        modules[m] = kernels_of(*image);
        expect(!modules[m].empty(), "%s: no kernel metadata", path.c_str());
    }
    for (size_t k = 0; k < size_t(Kernel::kCount); ++k) {
        const hip::KernelInfo& i = info(Kernel(k));
        expect(modules[size_t(i.module)].count(i.name), "%s has no kernel %s", hip::kModuleFiles[size_t(i.module)],
               i.name);
    }
    std::set<Kernel> reported;
    for (const NativeTier& tier : kNativeTiers)
        for (const bool performance : {false, true}) {
            const auto plan = hip::plan(tier.width, tier.networkHeight, performance);
            if (!expect(bool(plan), "tier %u: %s", tier.height, plan ? "" : plan.error().what.c_str())) continue;
            for (const Launch& l : plan->launches) {
                const hip::KernelInfo& i = info(l.kernel);
                const auto found = modules[size_t(i.module)].find(i.name);
                if (found == modules[size_t(i.module)].end() || reported.count(l.kernel)) continue;
                const KernelAbi& abi = found->second;
                bool same = abi.args.size() == l.count;
                for (unsigned a = 0; same && a < l.count; ++a)
                    same = pointer(l.args[a]) ? abi.args[a] == std::pair<std::string, uint64_t>("global_buffer", 8)
                                              : abi.args[a] == std::pair<std::string, uint64_t>("by_value", 4);
                if (!expect(same, "%s takes other arguments than the plan passes", i.name) ||
                    !expect(i.threads <= abi.max_group, "%s: groups of %u threads, at most %" PRIu64, i.name,
                            i.threads, abi.max_group))
                    reported.insert(l.kernel);
            }
        }
    if (!failures) std::puts("hip-plan test: every launch fits its kernel");
    return failures ? 1 : 0;
}

// Prints the canonical launch list the checks hash.
int print(const std::string& height, bool performance, bool first, bool history)
{
    const NativeTier* tier = nullptr;
    for (const NativeTier& t : kNativeTiers)
        if (std::to_string(t.height) == height) tier = &t;
    if (!tier) {
        std::fprintf(stderr, "hip-plan-test: no tier %s\n", height.c_str());
        return 2;
    }
    const auto plan = hip::plan(tier->width, tier->networkHeight, performance);
    const auto placement = plan.and_then([](const Plan& p) { return hip::place(p); });
    if (!placement) {
        std::fprintf(stderr, "hip-plan-test: %s\n", placement.error().what.c_str());
        return 1;
    }
    std::fputs(canonical_text(*plan, first ? placement->first : placement->later, history).c_str(), stdout);
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    std::string modules, tier;
    bool performance = false, first = false, history = false;
    const option options[] = {{"modules", required_argument, nullptr, 'm'}, {"print", required_argument, nullptr, 'P'},
                              {"performance", no_argument, nullptr, 'p'},   {"first", no_argument, nullptr, 'f'},
                              {"history", no_argument, nullptr, 'H'},       {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+m:P:pfHh", options, nullptr)) != -1;) {
        if (code == 'm') modules = optarg;
        else if (code == 'P') tier = optarg;
        else if (code == 'p') performance = true;
        else if (code == 'f') first = true;
        else if (code == 'H') history = true;
        else if (code == 'h') {
            std::puts("Usage: hip-plan-test [OPTION]...\n"
                      "Checks the HIP network's launch plan against traces of upstream's network. No GPU or model\n"
                      "needed.\n"
                      " -m, --modules DIR  Instead, check each planned launch's arguments and group size against\n"
                      "                    the kernels in the HIP modules in DIR; exit 77 when one is missing\n"
                      "                    (default: unset)\n"
                      " -P, --print TIER   Instead, print the launch list at TIER (720, 900 or 1080) as the checks\n"
                      "                    hash it, buffers named by first use (default: unset)\n"
                      " -p, --performance  With --print, leave out the blocks that dlsslopd --performance skips\n"
                      "                    (default: off)\n"
                      " -f, --first        With --print, the first frame's buffers, not a later frame's\n"
                      "                    (default: off)\n"
                      " -H, --history      With --print, a frame with a history (default: off)\n"
                      " -h, --help         Show help (default: off)");
            return 0;
        } else
            return 2;
    }
    if (optind != argc || (!tier.empty() && !modules.empty())) return 2;
    if (!tier.empty()) return print(tier, performance, first, history);
    if (!modules.empty()) return check_abi(modules);
    check_plans();
    if (!failures) std::puts("hip-plan test: every check passed");
    return failures ? 1 : 0;
}
