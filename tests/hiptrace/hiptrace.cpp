// SPDX-License-Identifier: MIT
// A tracing HIP runtime for dlsslopd: DLSSLOP_HIP_LIBRARY=libhiptrace.so
// (tests/hiptrace/trace.sh; VALIDATION.md describes the procedure).
//
// Exports every entry point dlsslopd's loader (backend/hip.h) resolves, and
// those upstream's host code resolved, so that builds from before the port
// trace too. Forwards each call to the real runtime and logs it to
// HIPTRACE_FILE (no file: forward only). Device and host pointers are logged
// as allocation id + byte offset, so traces of two runs, or of two
// implementations, compare with diff. Kernel arguments are decoded with the
// AMDGPU metadata note of the code object passed to hipModuleLoadData.
//
// Environment:
//   HIPTRACE_FILE          the log; unset or empty: no log
//   HIPTRACE_REAL          the real runtime; unset or empty: the first of
//                          dlsslopd's loader's candidates that loads
//   HIPTRACE_MODULES       directory whose *.hsaco name the loaded images
//                          (matched by FNV-1a 64 of the image); unset or
//                          empty: DLSSLOP_MODULES
//   HIPTRACE_DEEP          comma-separated ranges "A-B" and ordinals "A",
//                          such as "0-10,20": after launches A..B-1, or A
//                          (0-based launch ordinals), wait for the stream and
//                          log the FNV of every non-uploaded buffer argument,
//                          from the argument to the end of its allocation;
//                          "all": after every launch; unset or empty: none;
//                          anything else: none, with a message on stderr and
//                          deep=invalid in the log
//   HIPTRACE_DEEP_KERNELS  comma-separated kernel names deep-hashed always;
//                          unset or empty: none
#include "../../backend/hip.h"

#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace hiptrace {
namespace {

using Handle = void*;

constexpr uint64_t kFnvBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

uint64_t fnv(const void* data, size_t n, uint64_t h = kFnvBasis)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * kFnvPrime;
    return h;
}

// Every entry point that dlsslopd or upstream's host code resolves, as ROCm 7
// (and 6.4) declare them.
#define HIPTRACE_FUNCTIONS(X)                                                                                        \
    X(int, hipGetDevicePropertiesR0600, (struct hip_device_properties*, int))                                         \
    X(int, hipModuleLoadData, (Handle*, const void*))                                                                 \
    X(int, hipInit, (unsigned))                                                                                       \
    X(int, hipRuntimeGetVersion, (int*))                                                                              \
    X(int, hipGetDeviceCount, (int*))                                                                                 \
    X(int, hipDeviceGetName, (char*, int, int))                                                                       \
    X(int, hipSetDevice, (int))                                                                                       \
    X(int, hipSetDeviceFlags, (unsigned))                                                                             \
    X(int, hipMemGetInfo, (size_t*, size_t*))                                                                         \
    X(int, hipMalloc, (void**, size_t))                                                                               \
    X(int, hipHostMalloc, (void**, size_t, unsigned))                                                                 \
    X(int, hipHostFree, (void*))                                                                                      \
    X(int, hipHostRegister, (void*, size_t, unsigned))                                                                \
    X(int, hipHostUnregister, (void*))                                                                                \
    X(int, hipFree, (void*))                                                                                          \
    X(int, hipMemcpy, (void*, const void*, size_t, int))                                                              \
    X(int, hipMemcpyAsync, (void*, const void*, size_t, int, Handle))                                                 \
    X(int, hipMemsetAsync, (void*, int, size_t, Handle))                                                              \
    X(int, hipEventCreate, (Handle*))                                                                                 \
    X(int, hipEventRecord, (Handle, Handle))                                                                          \
    X(int, hipEventElapsedTime, (float*, Handle, Handle))                                                             \
    X(int, hipEventDestroy, (Handle))                                                                                 \
    X(int, hipEventSynchronize, (Handle))                                                                             \
    X(int, hipDeviceSynchronize, ())                                                                                  \
    X(int, hipStreamCreate, (Handle*))                                                                                \
    X(int, hipStreamSynchronize, (Handle))                                                                            \
    X(int, hipStreamDestroy, (Handle))                                                                                \
    X(int, hipImportExternalMemory, (Handle*, const struct hip_memory_desc*))                                         \
    X(int, hipExternalMemoryGetMappedBuffer, (void**, Handle, const struct hip_buffer_desc*))                         \
    X(int, hipDestroyExternalMemory, (Handle))                                                                        \
    X(int, hipImportExternalSemaphore, (Handle*, const void*))                                                        \
    X(int, hipSignalExternalSemaphoresAsync, (const Handle*, const void*, unsigned, Handle))                          \
    X(int, hipWaitExternalSemaphoresAsync, (const Handle*, const void*, unsigned, Handle))                            \
    X(int, hipDestroyExternalSemaphore, (Handle))                                                                     \
    X(int, hipStreamBeginCapture, (Handle, int))                                                                      \
    X(int, hipStreamEndCapture, (Handle, Handle*))                                                                    \
    X(int, hipGraphInstantiate, (Handle*, Handle, Handle*, char*, size_t))                                            \
    X(int, hipGraphLaunch, (Handle, Handle))                                                                          \
    X(int, hipGraphDestroy, (Handle))                                                                                 \
    X(int, hipGraphExecDestroy, (Handle))                                                                             \
    X(int, hipMemAddressReserve, (void**, size_t, size_t, void*, unsigned long long))                                \
    X(int, hipMemAddressFree, (void*, size_t))                                                                        \
    X(int, hipMemCreate, (void**, size_t, const void*, unsigned long long))                                           \
    X(int, hipMemRelease, (void*))                                                                                    \
    X(int, hipMemMap, (void*, size_t, size_t, void*, unsigned long long))                                             \
    X(int, hipMemUnmap, (void*, size_t))                                                                              \
    X(int, hipMemSetAccess, (void*, size_t, const void*, size_t))                                                     \
    X(int, hipMemGetAllocationGranularity, (size_t*, const void*, unsigned))                                          \
    X(int, hipModuleLoad, (Handle*, const char*))                                                                     \
    X(int, hipModuleGetFunction, (Handle*, Handle, const char*))                                                      \
    X(int, hipModuleLaunchKernel,                                                                                     \
      (Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Handle, void**, void**))         \
    X(int, hipExtModuleLaunchKernel,                                                                                  \
      (Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, size_t, Handle, void**, void**, Handle,    \
       Handle, unsigned))                                                                                             \
    X(int, hipModuleUnload, (Handle))                                                                                 \
    X(const char*, hipGetErrorName, (int))

struct Real {
    void* dll = nullptr;
    std::string path;
#define HIPTRACE_FIELD(ret, name, params) ret(*name) params = nullptr;
    HIPTRACE_FUNCTIONS(HIPTRACE_FIELD)
#undef HIPTRACE_FIELD
};

constexpr int kErrorNotFound = 500; // hipErrorNotFound, for an entry point the real runtime lacks.

// ---- msgpack, as much as the AMDGPU metadata note uses -------------------------------------------------

struct Msg {
    enum Type : unsigned char { kNil, kBool, kUint, kInt, kFloat, kStr, kBin, kArray, kMap } type = kNil;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0;
    std::string_view s;
    std::vector<Msg> items; // An array's elements, or a map's keys and values alternating.

    const Msg* get(std::string_view key) const
    {
        if (type != kMap) return nullptr;
        for (size_t k = 0; k + 1 < items.size(); k += 2)
            if (items[k].type == kStr && items[k].s == key) return &items[k + 1];
        return nullptr;
    }
    uint64_t number() const { return type == kUint ? u : type == kInt ? uint64_t(i) : 0; }
    uint64_t number(std::string_view key) const
    {
        const Msg* m = get(key);
        return m ? m->number() : 0;
    }
    std::string_view text(std::string_view key) const
    {
        const Msg* m = get(key);
        return m && m->type == kStr ? m->s : std::string_view();
    }
};

class MsgReader {
    const uint8_t* p_;
    const uint8_t* end_;

    bool need(size_t n) const { return size_t(end_ - p_) >= n; }
    uint64_t be(unsigned n)
    {
        uint64_t v = 0;
        for (unsigned k = 0; k < n; ++k) v = v << 8 | p_[k];
        p_ += n;
        return v;
    }
    bool bytes(Msg& out, Msg::Type type, size_t n)
    {
        if (!need(n)) return false;
        out.type = type;
        out.s = std::string_view(reinterpret_cast<const char*>(p_), n);
        p_ += n;
        return true;
    }
    bool sequence(Msg& out, Msg::Type type, size_t n, unsigned depth)
    {
        out.type = type;
        const size_t count = type == Msg::kMap ? n * 2 : n;
        if (count > size_t(end_ - p_)) return false; // Every item takes at least one byte.
        out.items.resize(count);
        for (Msg& item : out.items)
            if (!read(item, depth + 1)) return false;
        return true;
    }

public:
    MsgReader(const void* data, size_t n) : p_(static_cast<const uint8_t*>(data)), end_(p_ + n) {}

    bool read(Msg& out, unsigned depth = 0)
    {
        if (depth > 32 || !need(1)) return false;
        const uint8_t c = *p_++;
        if (c <= 0x7f) { out.type = Msg::kUint; out.u = c; return true; }
        if (c >= 0xe0) { out.type = Msg::kInt; out.i = int8_t(c); return true; }
        if ((c & 0xf0) == 0x80) return sequence(out, Msg::kMap, c & 15, depth);
        if ((c & 0xf0) == 0x90) return sequence(out, Msg::kArray, c & 15, depth);
        if ((c & 0xe0) == 0xa0) return bytes(out, Msg::kStr, c & 31);
        switch (c) {
        case 0xc0: out.type = Msg::kNil; return true;
        case 0xc2: case 0xc3: out.type = Msg::kBool; out.u = c & 1; return true;
        case 0xc4: case 0xc5: case 0xc6: {
            const unsigned w = 1u << (c - 0xc4);
            if (!need(w)) return false;
            return bytes(out, Msg::kBin, be(w));
        }
        case 0xca: {
            if (!need(4)) return false;
            const uint32_t b = uint32_t(be(4));
            float v;
            std::memcpy(&v, &b, 4);
            out.type = Msg::kFloat;
            out.f = v;
            return true;
        }
        case 0xcb: {
            if (!need(8)) return false;
            const uint64_t b = be(8);
            std::memcpy(&out.f, &b, 8);
            out.type = Msg::kFloat;
            return true;
        }
        case 0xcc: case 0xcd: case 0xce: case 0xcf: {
            const unsigned w = 1u << (c - 0xcc);
            if (!need(w)) return false;
            out.type = Msg::kUint;
            out.u = be(w);
            return true;
        }
        case 0xd0: case 0xd1: case 0xd2: case 0xd3: {
            const unsigned w = 1u << (c - 0xd0);
            if (!need(w)) return false;
            const uint64_t v = be(w);
            out.type = Msg::kInt;
            out.i = w == 8 ? int64_t(v) : int64_t(v << (64 - 8 * w)) >> (64 - 8 * w);
            return true;
        }
        case 0xd9: case 0xda: case 0xdb: {
            const unsigned w = 1u << (c - 0xd9);
            if (!need(w)) return false;
            return bytes(out, Msg::kStr, be(w));
        }
        case 0xdc: case 0xdd: {
            const unsigned w = c == 0xdc ? 2 : 4;
            if (!need(w)) return false;
            return sequence(out, Msg::kArray, be(w), depth);
        }
        case 0xde: case 0xdf: {
            const unsigned w = c == 0xde ? 2 : 4;
            if (!need(w)) return false;
            return sequence(out, Msg::kMap, be(w), depth);
        }
        default: return false; // ext types: not in AMDGPU metadata.
        }
    }
};

// ---- code objects --------------------------------------------------------------------------------------

struct Arg {
    uint32_t offset = 0, size = 0;
    std::string kind, name;
    bool hidden() const { return kind.rfind("hidden_", 0) == 0; }
    bool pointer() const { return kind == "global_buffer" || kind == "dynamic_shared_pointer"; }
};

struct Kernel {
    std::string name;
    std::vector<Arg> args; // Explicit ones first, as the metadata lists them.
    uint64_t kernarg = 0, lds = 0, scratch = 0, vgpr = 0, sgpr = 0, wave = 0, max_group = 0, vgpr_spill = 0,
             sgpr_spill = 0;
    unsigned explicit_args() const
    {
        unsigned n = 0;
        for (const Arg& a : args) n += !a.hidden();
        return n;
    }
};

// The image's extent from its headers, or 0 when it is not a 64-bit ELF.
size_t elf_size(const uint8_t* image)
{
    Elf64_Ehdr eh;
    std::memcpy(&eh, image, sizeof eh);
    if (std::memcmp(eh.e_ident, ELFMAG, SELFMAG) || eh.e_ident[EI_CLASS] != ELFCLASS64) return 0;
    size_t size = eh.e_ehsize;
    size = std::max<size_t>(size, eh.e_phoff + size_t(eh.e_phnum) * eh.e_phentsize);
    size = std::max<size_t>(size, eh.e_shoff + size_t(eh.e_shnum) * eh.e_shentsize);
    for (unsigned k = 0; k < eh.e_phnum; ++k) {
        Elf64_Phdr ph;
        std::memcpy(&ph, image + eh.e_phoff + size_t(k) * eh.e_phentsize, sizeof ph);
        size = std::max<size_t>(size, ph.p_offset + ph.p_filesz);
    }
    for (unsigned k = 0; k < eh.e_shnum; ++k) {
        Elf64_Shdr sh;
        std::memcpy(&sh, image + eh.e_shoff + size_t(k) * eh.e_shentsize, sizeof sh);
        if (sh.sh_type != SHT_NOBITS) size = std::max<size_t>(size, sh.sh_offset + sh.sh_size);
    }
    return size;
}

// The kernels the NT_AMDGPU_METADATA note describes; empty when it has none.
std::map<std::string, Kernel> kernels_of(const uint8_t* image)
{
    std::map<std::string, Kernel> kernels;
    Elf64_Ehdr eh;
    std::memcpy(&eh, image, sizeof eh);
    for (unsigned k = 0; k < eh.e_shnum; ++k) {
        Elf64_Shdr sh;
        std::memcpy(&sh, image + eh.e_shoff + size_t(k) * eh.e_shentsize, sizeof sh);
        if (sh.sh_type != SHT_NOTE) continue;
        for (size_t at = 0; at + 12 <= sh.sh_size;) {
            uint32_t word[3];
            std::memcpy(word, image + sh.sh_offset + at, 12);
            const size_t name_at = at + 12, desc_at = name_at + ((word[0] + 3) & ~3u);
            at = desc_at + ((word[1] + 3) & ~3u);
            if (at > sh.sh_size) break;
            if (word[2] != 32 || word[0] != 7 || std::memcmp(image + sh.sh_offset + name_at, "AMDGPU", 7)) continue;
            Msg root;
            MsgReader reader(image + sh.sh_offset + desc_at, word[1]);
            if (!reader.read(root)) continue;
            const Msg* list = root.get("amdhsa.kernels");
            if (!list || list->type != Msg::kArray) continue;
            for (const Msg& m : list->items) {
                Kernel kernel;
                kernel.name = m.text(".name");
                kernel.kernarg = m.number(".kernarg_segment_size");
                kernel.lds = m.number(".group_segment_fixed_size");
                kernel.scratch = m.number(".private_segment_fixed_size");
                kernel.vgpr = m.number(".vgpr_count");
                kernel.sgpr = m.number(".sgpr_count");
                kernel.vgpr_spill = m.number(".vgpr_spill_count");
                kernel.sgpr_spill = m.number(".sgpr_spill_count");
                kernel.wave = m.number(".wavefront_size");
                kernel.max_group = m.number(".max_flat_workgroup_size");
                if (const Msg* args = m.get(".args"); args && args->type == Msg::kArray)
                    for (const Msg& a : args->items) {
                        Arg arg;
                        arg.offset = uint32_t(a.number(".offset"));
                        arg.size = uint32_t(a.number(".size"));
                        arg.kind = a.text(".value_kind");
                        arg.name = a.text(".name");
                        kernel.args.push_back(std::move(arg));
                    }
                kernels.emplace(kernel.name, std::move(kernel));
            }
        }
    }
    return kernels;
}

// ---- state ---------------------------------------------------------------------------------------------

struct Range {
    char kind;       // d: hipMalloc, h: hipHostMalloc, r: hipHostRegister, x: external mapped buffer
    unsigned id;
    size_t size;
    bool uploaded;   // The destination of a host-to-device copy (weights, maps, constants).
};

struct Module {
    unsigned id = 0;
    std::string file;
    std::map<std::string, Kernel> kernels;
};

struct Function {
    unsigned id = 0;
    Handle module = nullptr;
    const Kernel* kernel = nullptr;
    std::string label; // file:kernel
};

struct State {
    std::mutex lock;
    Real real;
    FILE* log = nullptr;
    uint64_t sequence = 0, launches = 0;
    std::map<uintptr_t, Range> ranges;
    std::map<char, unsigned> counters;
    std::map<Handle, std::string> streams, events, memories;
    std::map<Handle, Module> modules;
    std::map<Handle, Function> functions;
    std::map<uint64_t, std::string> files; // FNV of each module file -> its name
    unsigned threads = 0;
    std::vector<std::pair<uint64_t, uint64_t>> deep_ranges; // [begin, end) launch ordinals
    std::set<std::string> deep_kernels;
};

State& state()
{
    static State* s = new State; // Never destroyed: runtime threads may call in during exit.
    return *s;
}

thread_local int thread_index = -1;

// Callers hold the lock.
void emit(State& s, const char* format, ...) __attribute__((format(printf, 2, 3)));
void emit(State& s, const char* format, ...)
{
    if (!s.log) return;
    if (thread_index < 0) thread_index = int(s.threads++);
    std::fprintf(s.log, "%" PRIu64 " t%d ", s.sequence++, thread_index);
    va_list args;
    va_start(args, format);
    std::vfprintf(s.log, format, args);
    va_end(args);
    std::fputc('\n', s.log);
}

std::string next_id(State& s, char kind)
{
    return std::string(1, kind) + std::to_string(++s.counters[kind]);
}

const Range* find(const State& s, uintptr_t address, uintptr_t* base)
{
    auto it = s.ranges.upper_bound(address);
    if (it == s.ranges.begin()) return nullptr;
    --it;
    if (address - it->first >= it->second.size) return nullptr;
    *base = it->first;
    return &it->second;
}

std::string where(const State& s, const void* pointer)
{
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    if (!address) return "null";
    uintptr_t base;
    if (const Range* r = find(s, address, &base))
        return std::string(1, r->kind) + std::to_string(r->id) + "+" + std::to_string(address - base);
    char text[32];
    std::snprintf(text, sizeof text, "?%" PRIxPTR, address);
    return text;
}

// A copy's side: unregistered host memory has no stable name.
std::string side(const State& s, const void* pointer)
{
    std::string name = where(s, pointer);
    return name[0] == '?' ? "host" : name;
}

std::string handle_name(const std::map<Handle, std::string>& names, Handle h, const char* null_name)
{
    if (!h) return null_name;
    auto it = names.find(h);
    return it == names.end() ? "?" : it->second;
}

std::string stream_name(const State& s, Handle stream) { return handle_name(s.streams, stream, "s0"); }

void add_range(State& s, char kind, void* pointer, size_t size, const char* call, unsigned flags = 0)
{
    const std::string id = next_id(s, kind);
    s.ranges[reinterpret_cast<uintptr_t>(pointer)] = Range{kind, s.counters[kind], size, false};
    emit(s, "%s %s bytes=%zu flags=%u", call, id.c_str(), size, flags);
}

void drop_range(State& s, void* pointer, const char* call)
{
    const std::string name = where(s, pointer);
    const auto it = s.ranges.find(reinterpret_cast<uintptr_t>(pointer));
    size_t size = 0;
    if (it != s.ranges.end()) {
        size = it->second.size;
        s.ranges.erase(it);
    }
    emit(s, "%s %s bytes=%zu", call, name.c_str(), size);
}

const char* kind_name(int kind)
{
    static const char* const names[] = {"H2H", "H2D", "D2H", "D2D", "DEFAULT"};
    return kind >= 0 && kind <= 4 ? names[kind] : "?";
}

// Whether the address is device memory this trace knows of.
bool device_memory(const State& s, const void* pointer)
{
    uintptr_t base;
    const Range* r = find(s, reinterpret_cast<uintptr_t>(pointer), &base);
    return r && (r->kind == 'd' || r->kind == 'x');
}

void mark_uploaded(State& s, const void* pointer)
{
    uintptr_t base;
    if (find(s, reinterpret_cast<uintptr_t>(pointer), &base)) s.ranges[base].uploaded = true;
}

// HIPTRACE_DEEP's launch ordinals as [begin, end) ranges; false when TEXT is
// neither "all" nor comma-separated "A-B" ranges and "A" ordinals.
bool parse_deep(const char* text, std::vector<std::pair<uint64_t, uint64_t>>& ranges)
{
    if (!std::strcmp(text, "all")) {
        ranges.emplace_back(0, UINT64_MAX);
        return true;
    }
    for (const char* at = text;; ++at) {
        char* end = nullptr;
        if (*at < '0' || *at > '9') return false;
        const uint64_t begin = std::strtoull(at, &end, 10);
        uint64_t stop = begin + 1;
        if (*end == '-') {
            at = end + 1;
            if (*at < '0' || *at > '9') return false;
            stop = std::strtoull(at, &end, 10);
        }
        ranges.emplace_back(begin, stop);
        if (!*end) return true;
        if (*end != ',') return false;
        at = end;
    }
}

void load_module_files(State& s, const char* directory)
{
    DIR* dir = directory && *directory ? opendir(directory) : nullptr;
    if (!dir) return;
    while (const dirent* entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.size() < 6 || name.compare(name.size() - 6, 6, ".hsaco")) continue;
        const std::string path = std::string(directory) + "/" + name;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) continue;
        std::vector<unsigned char> bytes;
        unsigned char buffer[65536];
        for (size_t n; (n = std::fread(buffer, 1, sizeof buffer, f));) bytes.insert(bytes.end(), buffer, buffer + n);
        std::fclose(f);
        s.files[fnv(bytes.data(), bytes.size())] = name;
    }
    closedir(dir);
}

__attribute__((constructor)) void initialize()
{
    State& s = state();
    // HIPTRACE_REAL, else backend/hip.c's candidates.
    const char* candidates[] = {std::getenv("HIPTRACE_REAL"), "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so",
                                "/opt/rocm/lib/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so.6",
                                "/opt/rocm/lib/libamdhip64.so"};
    for (const char* name : candidates) {
        if (!name || !*name) continue;
        if ((s.real.dll = dlopen(name, RTLD_NOW | RTLD_LOCAL))) {
            s.real.path = name;
            break;
        }
        if (name == candidates[0]) break;
    }
    std::string missing;
    if (s.real.dll) {
#define HIPTRACE_LOAD(ret, name, params)                                                                               \
    s.real.name = reinterpret_cast<decltype(s.real.name)>(dlsym(s.real.dll, #name));                                   \
    if (!s.real.name) missing += " " #name;
        HIPTRACE_FUNCTIONS(HIPTRACE_LOAD)
#undef HIPTRACE_LOAD
    } else {
        std::fprintf(stderr, "hiptrace: no real HIP runtime: %s\n", dlerror());
    }
    const char* modules = std::getenv("HIPTRACE_MODULES");
    if (!modules || !*modules) modules = std::getenv("DLSSLOP_MODULES");
    load_module_files(s, modules);
    bool deep_invalid = false;
    if (const char* deep = std::getenv("HIPTRACE_DEEP"); deep && *deep && !parse_deep(deep, s.deep_ranges)) {
        std::fprintf(stderr, "hiptrace: HIPTRACE_DEEP=%s is not all or comma-separated A-B ranges and A ordinals; "
                             "no launch is hashed by ordinal\n", deep);
        s.deep_ranges.clear();
        deep_invalid = true;
    }
    if (const char* list = std::getenv("HIPTRACE_DEEP_KERNELS"); list && *list) {
        std::string_view rest = list;
        while (!rest.empty()) {
            const size_t comma = rest.find(',');
            const std::string_view name = rest.substr(0, comma);
            if (!name.empty()) s.deep_kernels.emplace(name);
            if (comma == std::string_view::npos) break;
            rest.remove_prefix(comma + 1);
        }
    }
    const char* path = std::getenv("HIPTRACE_FILE");
    if (path && *path) {
        s.log = std::fopen(path, "w");
        if (!s.log) std::fprintf(stderr, "hiptrace: cannot write %s\n", path);
        else setvbuf(s.log, nullptr, _IOFBF, 1 << 20);
    }
    std::lock_guard guard(s.lock);
    std::string ranges;
    for (const auto& [begin, stop] : s.deep_ranges)
        ranges += (ranges.empty() ? "" : ",") + std::to_string(begin) + "-" + std::to_string(stop);
    emit(s, "trace real=%s module_files=%zu modules_dir=%s deep=%s deep_kernels=%zu missing=%s",
         s.real.path.empty() ? "none" : s.real.path.c_str(), s.files.size(), modules ? modules : "",
         deep_invalid ? "invalid" : ranges.empty() ? "none" : ranges.c_str(), s.deep_kernels.size(),
         missing.empty() ? "none" : missing.c_str());
    if (s.log) std::fflush(s.log);
}

__attribute__((destructor)) void finish()
{
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "exit launches=%" PRIu64, s.launches);
    if (s.log) std::fflush(s.log);
}

void flush(State& s)
{
    if (s.log) std::fflush(s.log);
}

// Each kernel argument as the runtime copies it: pointers normalised, other
// values in decimal (4 bytes) or hex.
std::string argument_text(const State& s, const Kernel& kernel, void** params)
{
    std::string text;
    unsigned index = 0;
    for (const Arg& arg : kernel.args) {
        if (arg.hidden()) continue;
        if (!text.empty()) text += ',';
        const auto* value = static_cast<const unsigned char*>(params[index++]);
        if (arg.pointer() && arg.size == 8) {
            void* pointer;
            std::memcpy(&pointer, value, 8);
            text += where(s, pointer);
        } else if (arg.size == 4) {
            uint32_t v;
            std::memcpy(&v, value, 4);
            text += std::to_string(v);
        } else {
            char hex[3];
            text += "0x";
            for (uint32_t k = arg.size; k--;) { // Little-endian bytes, most significant first.
                std::snprintf(hex, sizeof hex, "%02x", value[k]);
                text += hex;
            }
        }
    }
    return text;
}

// After a deep launch: the FNV of every buffer argument, from the argument to
// the end of its allocation. The lock is held; the real runtime is called directly.
std::string deep_hashes(State& s, const Kernel& kernel, void** params, Handle stream)
{
    std::string text;
    const int rc = s.real.hipStreamSynchronize(stream);
    if (rc) return "sync_rc=" + std::to_string(rc);
    unsigned index = 0;
    std::vector<unsigned char> copy;
    for (const Arg& arg : kernel.args) {
        if (arg.hidden()) continue;
        const unsigned position = index;
        const auto* value = static_cast<const unsigned char*>(params[index++]);
        if (!arg.pointer() || arg.size != 8) continue;
        void* pointer;
        std::memcpy(&pointer, value, 8);
        uintptr_t base;
        const Range* r = find(s, reinterpret_cast<uintptr_t>(pointer), &base);
        if (!r) continue;
        const size_t bytes = r->size - (reinterpret_cast<uintptr_t>(pointer) - base);
        char item[96];
        if (r->uploaded) {
            std::snprintf(item, sizeof item, " a%u=uploaded", position);
        } else {
            const void* data = pointer;
            if (r->kind == 'd' || r->kind == 'x') {
                copy.resize(bytes);
                if (const int e = s.real.hipMemcpy(copy.data(), pointer, bytes, 2)) {
                    std::snprintf(item, sizeof item, " a%u=copy_rc%d", position, e);
                    text += item;
                    continue;
                }
                data = copy.data();
            }
            std::snprintf(item, sizeof item, " a%u=%zu:%016" PRIx64, position, bytes, fnv(data, bytes));
        }
        text += item;
    }
    return text;
}

int launch(const char* call, Handle f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
           size_t shared, Handle stream, void** params, void** extra, int rc)
{
    State& s = state();
    std::lock_guard guard(s.lock);
    const uint64_t ordinal = s.launches++;
    const auto it = s.functions.find(f);
    const Function* fn = it == s.functions.end() ? nullptr : &it->second;
    std::string args = "?";
    if (fn && fn->kernel && params) {
        args = argument_text(s, *fn->kernel, params);
    } else if (extra) { // HIP_LAUNCH_PARAM_BUFFER_POINTER / SIZE / END
        const unsigned char* buffer = nullptr;
        size_t size = 0;
        for (void** e = extra; *e != reinterpret_cast<void*>(3); e += 2) {
            if (*e == reinterpret_cast<void*>(1)) buffer = static_cast<const unsigned char*>(e[1]);
            if (*e == reinterpret_cast<void*>(2)) size = *static_cast<const size_t*>(e[1]);
        }
        args = "buffer:";
        char hex[3];
        for (size_t k = 0; buffer && k < size; ++k) {
            std::snprintf(hex, sizeof hex, "%02x", buffer[k]);
            args += hex;
        }
    }
    const bool deep = !rc && fn && fn->kernel && params &&
                      (std::any_of(s.deep_ranges.begin(), s.deep_ranges.end(),
                                   [&](const auto& r) { return ordinal >= r.first && ordinal < r.second; }) ||
                       s.deep_kernels.count(fn->kernel->name));
    const std::string hashes = deep ? deep_hashes(s, *fn->kernel, params, stream) : std::string();
    emit(s, "%s n=%" PRIu64 " f=%s k=%s grid=%ux%ux%u block=%ux%ux%u lds=%zu s=%s args=%s rc=%d%s%s", call, ordinal,
         fn ? ("f" + std::to_string(fn->id)).c_str() : "?", fn ? fn->label.c_str() : "?", gx, gy, gz, bx, by, bz,
         shared, stream_name(s, stream).c_str(), args.c_str(), rc, deep ? " deep" : "", hashes.c_str());
    return rc;
}

} // namespace
} // namespace hiptrace

// ---- exports -------------------------------------------------------------------------------------------

using hiptrace::Handle;
using hiptrace::State;
using hiptrace::emit;
using hiptrace::state;

#define HIPTRACE_EXPORT extern "C" __attribute__((visibility("default")))
#define HIPTRACE_REAL(name, ...) (state().real.name ? state().real.name(__VA_ARGS__) : hiptrace::kErrorNotFound)

HIPTRACE_EXPORT const char* hipGetErrorName(int error)
{
    return state().real.hipGetErrorName ? state().real.hipGetErrorName(error) : "hipErrorNotFound";
}

HIPTRACE_EXPORT int hipInit(unsigned flags)
{
    const int rc = HIPTRACE_REAL(hipInit, flags);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipInit flags=%u rc=%d", flags, rc);
    return rc;
}

HIPTRACE_EXPORT int hipRuntimeGetVersion(int* version)
{
    const int rc = HIPTRACE_REAL(hipRuntimeGetVersion, version);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipRuntimeGetVersion version=%d rc=%d", rc ? 0 : *version, rc);
    return rc;
}

HIPTRACE_EXPORT int hipGetDeviceCount(int* count)
{
    const int rc = HIPTRACE_REAL(hipGetDeviceCount, count);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipGetDeviceCount count=%d rc=%d", rc ? 0 : *count, rc);
    return rc;
}

HIPTRACE_EXPORT int hipGetDevicePropertiesR0600(struct hip_device_properties* p, int device)
{
    const int rc = HIPTRACE_REAL(hipGetDevicePropertiesR0600, p, device);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipGetDevicePropertiesR0600 device=%d name=\"%s\" arch=%s rc=%d", device, rc ? "" : p->name,
         rc ? "" : p->gcnArchName, rc);
    return rc;
}

HIPTRACE_EXPORT int hipDeviceGetName(char* name, int length, int device)
{
    const int rc = HIPTRACE_REAL(hipDeviceGetName, name, length, device);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipDeviceGetName device=%d rc=%d", device, rc);
    return rc;
}

HIPTRACE_EXPORT int hipSetDevice(int device)
{
    const int rc = HIPTRACE_REAL(hipSetDevice, device);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipSetDevice device=%d rc=%d", device, rc);
    return rc;
}

HIPTRACE_EXPORT int hipSetDeviceFlags(unsigned flags)
{
    const int rc = HIPTRACE_REAL(hipSetDeviceFlags, flags);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipSetDeviceFlags flags=%u rc=%d", flags, rc);
    return rc;
}

HIPTRACE_EXPORT int hipMemGetInfo(size_t* free, size_t* total)
{
    const int rc = HIPTRACE_REAL(hipMemGetInfo, free, total);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipMemGetInfo rc=%d", rc);
    return rc;
}

HIPTRACE_EXPORT int hipMalloc(void** pointer, size_t size)
{
    const int rc = HIPTRACE_REAL(hipMalloc, pointer, size);
    State& s = state();
    std::lock_guard guard(s.lock);
    if (rc) emit(s, "hipMalloc bytes=%zu rc=%d", size, rc);
    else hiptrace::add_range(s, 'd', *pointer, size, "hipMalloc");
    return rc;
}

HIPTRACE_EXPORT int hipFree(void* pointer)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        hiptrace::drop_range(s, pointer, "hipFree");
    }
    return HIPTRACE_REAL(hipFree, pointer);
}

HIPTRACE_EXPORT int hipHostMalloc(void** pointer, size_t size, unsigned flags)
{
    const int rc = HIPTRACE_REAL(hipHostMalloc, pointer, size, flags);
    State& s = state();
    std::lock_guard guard(s.lock);
    if (rc) emit(s, "hipHostMalloc bytes=%zu flags=%u rc=%d", size, flags, rc);
    else hiptrace::add_range(s, 'h', *pointer, size, "hipHostMalloc", flags);
    return rc;
}

HIPTRACE_EXPORT int hipHostFree(void* pointer)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        hiptrace::drop_range(s, pointer, "hipHostFree");
    }
    return HIPTRACE_REAL(hipHostFree, pointer);
}

HIPTRACE_EXPORT int hipHostRegister(void* pointer, size_t size, unsigned flags)
{
    const int rc = HIPTRACE_REAL(hipHostRegister, pointer, size, flags);
    State& s = state();
    std::lock_guard guard(s.lock);
    if (rc) emit(s, "hipHostRegister bytes=%zu flags=%u rc=%d", size, flags, rc);
    else hiptrace::add_range(s, 'r', pointer, size, "hipHostRegister", flags);
    return rc;
}

HIPTRACE_EXPORT int hipHostUnregister(void* pointer)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        hiptrace::drop_range(s, pointer, "hipHostUnregister");
    }
    return HIPTRACE_REAL(hipHostUnregister, pointer);
}

HIPTRACE_EXPORT int hipMemcpy(void* dst, const void* src, size_t size, int kind)
{
    State& s = state();
    std::string where_dst, where_src;
    uint64_t payload = 0;
    bool from_host = false;
    {
        std::lock_guard guard(s.lock);
        where_dst = hiptrace::side(s, dst);
        where_src = hiptrace::side(s, src);
        from_host = kind == 1 || (kind == 4 && !hiptrace::device_memory(s, src) && hiptrace::device_memory(s, dst));
        if (from_host) {
            payload = hiptrace::fnv(src, size);
            hiptrace::mark_uploaded(s, dst);
        }
    }
    const int rc = HIPTRACE_REAL(hipMemcpy, dst, src, size, kind);
    std::lock_guard guard(s.lock);
    const bool to_host = kind == 2 || (kind == 4 && !hiptrace::device_memory(s, dst));
    if (!rc && to_host) payload = hiptrace::fnv(dst, size);
    emit(s, "hipMemcpy kind=%s dst=%s src=%s bytes=%zu %s=%016" PRIx64 " rc=%d", hiptrace::kind_name(kind),
         where_dst.c_str(), where_src.c_str(), size, from_host ? "src_fnv" : to_host ? "dst_fnv" : "fnv",
         (from_host || to_host) ? payload : 0, rc);
    hiptrace::flush(s);
    return rc;
}

HIPTRACE_EXPORT int hipMemcpyAsync(void* dst, const void* src, size_t size, int kind, Handle stream)
{
    State& s = state();
    std::string where_dst, where_src;
    uint64_t payload = 0;
    bool from_host = false;
    {
        std::lock_guard guard(s.lock);
        where_dst = hiptrace::side(s, dst);
        where_src = hiptrace::side(s, src);
        from_host = kind == 1 || (kind == 4 && !hiptrace::device_memory(s, src) && hiptrace::device_memory(s, dst));
        if (from_host) {
            payload = hiptrace::fnv(src, size);
            hiptrace::mark_uploaded(s, dst);
        }
    }
    const int rc = HIPTRACE_REAL(hipMemcpyAsync, dst, src, size, kind, stream);
    std::lock_guard guard(s.lock);
    if (from_host)
        emit(s, "hipMemcpyAsync kind=%s dst=%s src=%s bytes=%zu s=%s src_fnv=%016" PRIx64 " rc=%d",
             hiptrace::kind_name(kind), where_dst.c_str(), where_src.c_str(), size,
             hiptrace::stream_name(s, stream).c_str(), payload, rc);
    else
        emit(s, "hipMemcpyAsync kind=%s dst=%s src=%s bytes=%zu s=%s rc=%d", hiptrace::kind_name(kind),
             where_dst.c_str(), where_src.c_str(), size, hiptrace::stream_name(s, stream).c_str(), rc);
    return rc;
}

HIPTRACE_EXPORT int hipMemsetAsync(void* dst, int value, size_t size, Handle stream)
{
    const int rc = HIPTRACE_REAL(hipMemsetAsync, dst, value, size, stream);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipMemsetAsync dst=%s value=%d bytes=%zu s=%s rc=%d", hiptrace::where(s, dst).c_str(), value, size,
         hiptrace::stream_name(s, stream).c_str(), rc);
    return rc;
}

HIPTRACE_EXPORT int hipEventCreate(Handle* event)
{
    const int rc = HIPTRACE_REAL(hipEventCreate, event);
    State& s = state();
    std::lock_guard guard(s.lock);
    std::string id = "?";
    if (!rc) s.events[*event] = id = hiptrace::next_id(s, 'e');
    emit(s, "hipEventCreate %s rc=%d", id.c_str(), rc);
    return rc;
}

HIPTRACE_EXPORT int hipEventRecord(Handle event, Handle stream)
{
    const int rc = HIPTRACE_REAL(hipEventRecord, event, stream);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipEventRecord %s s=%s rc=%d", hiptrace::handle_name(s.events, event, "null").c_str(),
         hiptrace::stream_name(s, stream).c_str(), rc);
    return rc;
}

HIPTRACE_EXPORT int hipEventElapsedTime(float* ms, Handle begin, Handle end)
{
    const int rc = HIPTRACE_REAL(hipEventElapsedTime, ms, begin, end);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipEventElapsedTime %s %s rc=%d", hiptrace::handle_name(s.events, begin, "null").c_str(),
         hiptrace::handle_name(s.events, end, "null").c_str(), rc);
    return rc;
}

HIPTRACE_EXPORT int hipEventDestroy(Handle event)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        emit(s, "hipEventDestroy %s", hiptrace::handle_name(s.events, event, "null").c_str());
        s.events.erase(event);
    }
    return HIPTRACE_REAL(hipEventDestroy, event);
}

HIPTRACE_EXPORT int hipEventSynchronize(Handle event)
{
    const int rc = HIPTRACE_REAL(hipEventSynchronize, event);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipEventSynchronize %s rc=%d", hiptrace::handle_name(s.events, event, "null").c_str(), rc);
    hiptrace::flush(s);
    return rc;
}

HIPTRACE_EXPORT int hipDeviceSynchronize()
{
    const int rc = HIPTRACE_REAL(hipDeviceSynchronize);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipDeviceSynchronize rc=%d", rc);
    hiptrace::flush(s);
    return rc;
}

HIPTRACE_EXPORT int hipStreamCreate(Handle* stream)
{
    const int rc = HIPTRACE_REAL(hipStreamCreate, stream);
    State& s = state();
    std::lock_guard guard(s.lock);
    std::string id = "?";
    if (!rc) s.streams[*stream] = id = hiptrace::next_id(s, 's');
    emit(s, "hipStreamCreate %s rc=%d", id.c_str(), rc);
    return rc;
}

HIPTRACE_EXPORT int hipStreamSynchronize(Handle stream)
{
    const int rc = HIPTRACE_REAL(hipStreamSynchronize, stream);
    State& s = state();
    std::lock_guard guard(s.lock);
    emit(s, "hipStreamSynchronize %s rc=%d", hiptrace::stream_name(s, stream).c_str(), rc);
    hiptrace::flush(s);
    return rc;
}

HIPTRACE_EXPORT int hipStreamDestroy(Handle stream)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        emit(s, "hipStreamDestroy %s", hiptrace::stream_name(s, stream).c_str());
        s.streams.erase(stream);
    }
    return HIPTRACE_REAL(hipStreamDestroy, stream);
}

HIPTRACE_EXPORT int hipImportExternalMemory(Handle* memory, const struct hip_memory_desc* d)
{
    const int rc = HIPTRACE_REAL(hipImportExternalMemory, memory, d);
    State& s = state();
    std::lock_guard guard(s.lock);
    std::string id = "?";
    if (!rc) s.memories[*memory] = id = hiptrace::next_id(s, 'X');
    emit(s, "hipImportExternalMemory %s type=%d bytes=%llu flags=%u rc=%d", id.c_str(), d->type, d->size, d->flags,
         rc);
    return rc;
}

HIPTRACE_EXPORT int hipExternalMemoryGetMappedBuffer(void** pointer, Handle memory, const struct hip_buffer_desc* d)
{
    const int rc = HIPTRACE_REAL(hipExternalMemoryGetMappedBuffer, pointer, memory, d);
    State& s = state();
    std::lock_guard guard(s.lock);
    if (rc) {
        emit(s, "hipExternalMemoryGetMappedBuffer of=%s rc=%d", hiptrace::handle_name(s.memories, memory, "null").c_str(),
             rc);
    } else {
        const std::string call = "hipExternalMemoryGetMappedBuffer of=" +
                                 hiptrace::handle_name(s.memories, memory, "null") +
                                 " offset=" + std::to_string(d->offset);
        hiptrace::add_range(s, 'x', *pointer, d->size, call.c_str(), d->flags);
    }
    return rc;
}

HIPTRACE_EXPORT int hipDestroyExternalMemory(Handle memory)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        emit(s, "hipDestroyExternalMemory %s", hiptrace::handle_name(s.memories, memory, "null").c_str());
        s.memories.erase(memory);
    }
    return HIPTRACE_REAL(hipDestroyExternalMemory, memory);
}

// Upstream's host code resolved these but never called them in dlsslopd;
// forwarded with a log line.
#define HIPTRACE_PLAIN(name, params, args)                                                                           \
    HIPTRACE_EXPORT int name params                                                                                  \
    {                                                                                                                \
        const int rc = HIPTRACE_REAL(name, HIPTRACE_UNPAREN args);                                                   \
        State& s = state();                                                                                          \
        std::lock_guard guard(s.lock);                                                                               \
        emit(s, #name " rc=%d", rc);                                                                                 \
        return rc;                                                                                                   \
    }
#define HIPTRACE_UNPAREN(...) __VA_ARGS__
HIPTRACE_PLAIN(hipImportExternalSemaphore, (Handle * a, const void* b), (a, b))
HIPTRACE_PLAIN(hipSignalExternalSemaphoresAsync, (const Handle* a, const void* b, unsigned c, Handle d), (a, b, c, d))
HIPTRACE_PLAIN(hipWaitExternalSemaphoresAsync, (const Handle* a, const void* b, unsigned c, Handle d), (a, b, c, d))
HIPTRACE_PLAIN(hipDestroyExternalSemaphore, (Handle a), (a))
HIPTRACE_PLAIN(hipStreamBeginCapture, (Handle a, int b), (a, b))
HIPTRACE_PLAIN(hipStreamEndCapture, (Handle a, Handle* b), (a, b))
HIPTRACE_PLAIN(hipGraphInstantiate, (Handle * a, Handle b, Handle* c, char* d, size_t e), (a, b, c, d, e))
HIPTRACE_PLAIN(hipGraphLaunch, (Handle a, Handle b), (a, b))
HIPTRACE_PLAIN(hipGraphDestroy, (Handle a), (a))
HIPTRACE_PLAIN(hipGraphExecDestroy, (Handle a), (a))
HIPTRACE_PLAIN(hipMemAddressReserve, (void** a, size_t b, size_t c, void* d, unsigned long long e), (a, b, c, d, e))
HIPTRACE_PLAIN(hipMemAddressFree, (void* a, size_t b), (a, b))
HIPTRACE_PLAIN(hipMemCreate, (void** a, size_t b, const void* c, unsigned long long d), (a, b, c, d))
HIPTRACE_PLAIN(hipMemRelease, (void* a), (a))
HIPTRACE_PLAIN(hipMemMap, (void* a, size_t b, size_t c, void* d, unsigned long long e), (a, b, c, d, e))
HIPTRACE_PLAIN(hipMemUnmap, (void* a, size_t b), (a, b))
HIPTRACE_PLAIN(hipMemSetAccess, (void* a, size_t b, const void* c, size_t d), (a, b, c, d))
HIPTRACE_PLAIN(hipMemGetAllocationGranularity, (size_t * a, const void* b, unsigned c), (a, b, c))

namespace {
// The module's record: its file (by hash), its kernels; the lock is held.
void add_module(State& s, Handle module, const void* image, const char* call, const char* path)
{
    hiptrace::Module m;
    m.id = ++s.counters['m'];
    const auto* bytes = static_cast<const uint8_t*>(image);
    const size_t size = hiptrace::elf_size(bytes);
    const uint64_t hash = size ? hiptrace::fnv(bytes, size) : 0;
    const auto file = s.files.find(hash);
    m.file = file != s.files.end() ? file->second : path ? path : "?";
    if (size) m.kernels = hiptrace::kernels_of(bytes);
    emit(s, "%s m%u file=%s bytes=%zu fnv=%016" PRIx64 " kernels=%zu", call, m.id, m.file.c_str(), size, hash,
         m.kernels.size());
    s.modules[module] = std::move(m);
}
} // namespace

HIPTRACE_EXPORT int hipModuleLoadData(Handle* module, const void* image)
{
    const int rc = HIPTRACE_REAL(hipModuleLoadData, module, image);
    State& s = state();
    std::lock_guard guard(s.lock);
    if (rc) emit(s, "hipModuleLoadData rc=%d", rc);
    else add_module(s, *module, image, "hipModuleLoadData", nullptr);
    return rc;
}

HIPTRACE_EXPORT int hipModuleLoad(Handle* module, const char* path)
{
    const int rc = HIPTRACE_REAL(hipModuleLoad, module, path);
    State& s = state();
    std::lock_guard guard(s.lock);
    if (rc || !path) {
        emit(s, "hipModuleLoad path=%s rc=%d", path ? path : "null", rc);
        return rc;
    }
    FILE* f = std::fopen(path, "rb");
    std::vector<uint8_t> bytes;
    if (f) {
        uint8_t buffer[65536];
        for (size_t n; (n = std::fread(buffer, 1, sizeof buffer, f));) bytes.insert(bytes.end(), buffer, buffer + n);
        std::fclose(f);
    }
    bytes.resize(std::max<size_t>(bytes.size(), sizeof(Elf64_Ehdr)));
    add_module(s, *module, bytes.data(), "hipModuleLoad", path);
    return rc;
}

HIPTRACE_EXPORT int hipModuleGetFunction(Handle* function, Handle module, const char* name)
{
    const int rc = HIPTRACE_REAL(hipModuleGetFunction, function, module, name);
    State& s = state();
    std::lock_guard guard(s.lock);
    const auto m = s.modules.find(module);
    const std::string module_name = m == s.modules.end() ? "?" : "m" + std::to_string(m->second.id);
    const std::string file = m == s.modules.end() ? "?" : m->second.file;
    if (rc) {
        emit(s, "hipModuleGetFunction %s k=%s rc=%d", module_name.c_str(), name, rc);
        return rc;
    }
    hiptrace::Function f;
    f.module = module;
    f.label = file + ":" + name;
    if (m != s.modules.end()) {
        const auto k = m->second.kernels.find(name);
        if (k != m->second.kernels.end()) f.kernel = &k->second;
    }
    const auto known = s.functions.find(*function);
    f.id = known != s.functions.end() ? known->second.id : ++s.counters['f'];
    if (f.kernel) {
        const hiptrace::Kernel& k = *f.kernel;
        std::string layout;
        for (const hiptrace::Arg& a : k.args) {
            if (!layout.empty()) layout += ',';
            layout += (a.pointer() ? "p" : a.hidden() ? "h" : "v") + std::to_string(a.size) + "@" +
                      std::to_string(a.offset);
        }
        emit(s,
             "hipModuleGetFunction f%u %s k=%s args=%u kernarg=%" PRIu64 " lds=%" PRIu64 " scratch=%" PRIu64
             " vgpr=%" PRIu64 " sgpr=%" PRIu64 " vgpr_spill=%" PRIu64 " sgpr_spill=%" PRIu64 " wave=%" PRIu64
             " max_group=%" PRIu64 " layout=%s rc=0",
             f.id, module_name.c_str(), f.label.c_str(), k.explicit_args(), k.kernarg, k.lds, k.scratch, k.vgpr, k.sgpr,
             k.vgpr_spill, k.sgpr_spill, k.wave, k.max_group, layout.c_str());
    } else {
        emit(s, "hipModuleGetFunction f%u %s k=%s metadata=none rc=0", f.id, module_name.c_str(), f.label.c_str());
    }
    s.functions[*function] = std::move(f);
    return rc;
}

HIPTRACE_EXPORT int hipModuleLaunchKernel(Handle f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                          unsigned bz, unsigned shared, Handle stream, void** params, void** extra)
{
    const int rc = HIPTRACE_REAL(hipModuleLaunchKernel, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    return hiptrace::launch("launch", f, gx, gy, gz, bx, by, bz, shared, stream, params, extra, rc);
}

HIPTRACE_EXPORT int hipExtModuleLaunchKernel(Handle f, unsigned gx, unsigned gy, unsigned gz, unsigned lx, unsigned ly,
                                             unsigned lz, size_t shared, Handle stream, void** params, void** extra,
                                             Handle start, Handle stop, unsigned flags)
{
    const int rc = HIPTRACE_REAL(hipExtModuleLaunchKernel, f, gx, gy, gz, lx, ly, lz, shared, stream, params, extra,
                                 start, stop, flags);
    // Global sizes are threads: logged as groups, like hipModuleLaunchKernel.
    return hiptrace::launch(flags ? "ext_launch_anyorder" : "ext_launch", f, lx ? gx / lx : gx, ly ? gy / ly : gy,
                            lz ? gz / lz : gz, lx, ly, lz, shared, stream, params, extra, rc);
}

HIPTRACE_EXPORT int hipModuleUnload(Handle module)
{
    State& s = state();
    {
        std::lock_guard guard(s.lock);
        const auto m = s.modules.find(module);
        emit(s, "hipModuleUnload %s", m == s.modules.end() ? "?" : ("m" + std::to_string(m->second.id)).c_str());
        for (auto it = s.functions.begin(); it != s.functions.end();)
            it = it->second.module == module ? s.functions.erase(it) : std::next(it);
        if (m != s.modules.end()) s.modules.erase(m);
    }
    return HIPTRACE_REAL(hipModuleUnload, module);
}
