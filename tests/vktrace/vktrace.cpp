// SPDX-License-Identifier: MIT
// VK_LAYER_LOCAL_vktrace: an explicit Vulkan layer that logs what an
// application asks of a device, with every handle renamed by kind and creation
// order (buf12, img3, pipe40, ...), so that two runs, or two implementations,
// can be compared line by line or canonically (compare.py). trace.sh enables
// it for one dlsslopd run; VALIDATION.md describes the procedure.
//
// Logged: instance and device creation (enabled extensions and every enabled
// feature bit), queues, memory allocations (size, type and its property
// flags, dedicated/import/export), buffers and images (create info) and their
// binds (allocation + offset), views, samplers, shader modules (FNV-1a 64 of
// the SPIR-V and the file it matches in VKTRACE_SPIRV), descriptor set
// layouts, pipeline layouts (push ranges), compute pipelines (module, entry,
// required subgroup size, specialization map and data), descriptor pools and
// sets and every descriptor written or copied, under the binding it reaches,
// command pools and buffers, and each recorded command the network uses
// (binds, push constants with their bytes, dispatches, barriers v1/v2, copies,
// blits, fills, updates, clears, queries, timestamps), queue submits and
// waits.
//
// Host data: every copy out of mapped host-visible memory logs the FNV of its
// source bytes at submit ("Upload"), and every copy into host-visible memory
// logs the FNV of what arrived once its fence or queue is waited for
// ("Readback").
//
// Content hashes (VKTRACE_HASH, off by default): after a submit completes,
// the FNV of selected buffers and images, copied out through a staging
// buffer on the same queue ("Hash"). Selectors, comma-separated:
//   i2b        the byte range each vkCmdCopyImageToBuffer wrote
//   copydst    every copy/blit/fill/update/clear destination
//   storage    every storage buffer range and storage image a dispatch bound
//   dispatch=N the storage bindings of the submit's dispatch N (0-based)
//   buf12,img3 those resources, whole
//   all        every live buffer and image of the device
// VKTRACE_HASH_SUBMITS limits hashing to submits (per device, 1-based):
// "A-B,C", or "dispatch" for submits that dispatch anything. Default: all.
// Items that mean nothing are reported on standard error and ignored.
//
// Environment: VKTRACE_FILE (the log; %p is the pid; default
// /tmp/vktrace.%p.log), VKTRACE_SPIRV (':'-separated directories scanned
// recursively for *.spv), VKTRACE_HASH, VKTRACE_HASH_SUBMITS.
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <dirent.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "vkt_names.inc"

namespace {

constexpr uint64_t kFnvBasis = 14695981039346656037ull;
uint64_t fnv(const void* data, size_t n, uint64_t h = kFnvBasis)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// FNV of host-visible memory that may be uncached (write-combined VRAM through the
// BAR): read in 16-byte streaming loads into a cached chunk first. Byte loads from
// such memory each cross the bus, which made a 3.6 MB frame cost ~300 ms to hash.
__attribute__((target("sse4.1"))) uint64_t fnv_uncached(const void* data, size_t n)
{
    static thread_local std::vector<unsigned char> chunk(1 << 16);
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = kFnvBasis;
    const size_t head = std::min(n, size_t((16 - (reinterpret_cast<uintptr_t>(p) & 15)) & 15));
    h = fnv(p, head, h);
    p += head;
    n -= head;
    while (n >= 16) {
        const size_t take = std::min(n & ~size_t(15), chunk.size());
        for (size_t i = 0; i < take; i += 16) {
            const __m128i v = _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<unsigned char*>(p + i)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(chunk.data() + i), v);
        }
        h = fnv(chunk.data(), take, h);
        p += take;
        n -= take;
    }
    return fnv(p, n, h);
}

// ---------------------------------------------------------------------------
// Text

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...)
{
    char small[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(small, sizeof small, fmt, ap);
    va_end(ap);
    if (n < int(sizeof small)) return std::string(small, n > 0 ? size_t(n) : 0);
    std::string big(size_t(n) + 1, '\0');
    va_start(ap, fmt);
    std::vsnprintf(big.data(), big.size(), fmt, ap);
    va_end(ap);
    big.resize(size_t(n));
    return big;
}

std::string hex(uint64_t v) { return format("0x%" PRIx64, v); }

std::string bytes_hex(const void* data, size_t n)
{
    static const char digits[] = "0123456789abcdef";
    std::string s(n * 2, '0');
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        s[2 * i] = digits[p[i] >> 4];
        s[2 * i + 1] = digits[p[i] & 15];
    }
    return s;
}

// A name for a value from the generated tables, or the number.
template <class E> std::string enum_name(const char* (*table)(E), E v)
{
    if (const char* n = table(v)) return n;
    return std::to_string(int64_t(v));
}
std::string fmt_name(VkFormat f) { return enum_name(FormatName, f); }
std::string layout_name(VkImageLayout l) { return enum_name(ImageLayoutName, l); }
std::string dtype_name(VkDescriptorType t) { return enum_name(DescriptorTypeName, t); }
std::string stype_name(VkStructureType t) { return enum_name(StructureTypeName, t); }

std::string size_str(VkDeviceSize s) { return s == VK_WHOLE_SIZE ? std::string("WHOLE") : std::to_string(s); }

std::string family_str(uint32_t f)
{
    if (f == VK_QUEUE_FAMILY_IGNORED) return "ign";
    if (f == VK_QUEUE_FAMILY_EXTERNAL) return "ext";
    if (f == VK_QUEUE_FAMILY_FOREIGN_EXT) return "foreign";
    return std::to_string(f);
}

// Spaces would split a field; names from drivers and applications carry them.
std::string token(const char* s)
{
    std::string t = s ? s : "";
    for (char& c : t)
        if (c == ' ' || c == '\t' || c == '\n' || c == '=') c = '_';
    return t.empty() ? std::string("-") : t;
}

std::string range_str(const VkImageSubresourceRange& r)
{
    return format("%s/%u+%s/%u+%s", hex(r.aspectMask).c_str(), r.baseMipLevel,
                  r.levelCount == VK_REMAINING_MIP_LEVELS ? "all" : std::to_string(r.levelCount).c_str(),
                  r.baseArrayLayer,
                  r.layerCount == VK_REMAINING_ARRAY_LAYERS ? "all" : std::to_string(r.layerCount).c_str());
}
std::string layers_str(const VkImageSubresourceLayers& l)
{
    return format("%s/%u/%u+%u", hex(l.aspectMask).c_str(), l.mipLevel, l.baseArrayLayer, l.layerCount);
}

// ---------------------------------------------------------------------------
// Handles

enum Kind : unsigned {
    kInstance, kPhysical, kDevice, kQueue, kMemory, kBuffer, kImage, kView, kBufferView, kSampler, kShader,
    kCache, kSetLayout, kPipeLayout, kPipeline, kDescPool, kDescSet, kCmdPool, kCmd, kFence, kSemaphore,
    kQueryPool, kEvent, kSwapchain, kKinds
};
constexpr const char* kPrefix[kKinds] = {"inst", "pd", "dev", "q", "mem", "buf", "img", "view", "bview", "smp",
                                         "shm", "pc", "dsl", "pl", "pipe", "dp", "ds", "cp", "cb", "fence", "sem",
                                         "qp", "ev", "sc"};

template <class H> uint64_t key_of(H h) { return uint64_t(reinterpret_cast<uintptr_t>(h)); }

// The loader's dispatch table pointer, shared by a device and its queues and
// command buffers (and by an instance and its physical devices).
void* dispatch_key(const void* object) { return *static_cast<void* const*>(object); }

struct InstanceData;
struct DeviceData;

struct MemoryInfo {
    DeviceData* device = nullptr;
    VkDeviceSize size = 0;
    uint32_t type = 0;
    VkMemoryPropertyFlags flags = 0;
    void* mapped = nullptr;  // the application's mapping
    VkDeviceSize map_offset = 0, map_size = 0;
    bool external = false;
};
struct BufferInfo {
    DeviceData* device = nullptr;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    bool external = false;
    // The queue family the last submitted ownership transfer gave it to;
    // VK_QUEUE_FAMILY_IGNORED until one is seen.
    uint32_t owner = VK_QUEUE_FAMILY_IGNORED;
};
struct ImageInfo {
    DeviceData* device = nullptr;
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
    uint32_t mips = 1, layers = 1;
    VkImageUsageFlags usage = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;  // after the last submitted barrier
    bool swapchain = false, external = false;
};
struct ViewInfo {
    VkImage image = VK_NULL_HANDLE;
};
struct Desc {
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, range = 0;
    VkImageView view = VK_NULL_HANDLE;
};
// The descriptor count of each binding number of a set layout.
struct LayoutInfo {
    std::map<uint32_t, uint32_t> counts;
    bool variable = false;  // the last binding's count is given at allocation
};
struct SetInfo {
    std::map<uint64_t, Desc> slots;       // binding << 32 | element
    std::map<uint32_t, uint32_t> counts;  // of its layout; empty when unknown
};
// A resource region a hash selector names.
struct Target {
    bool image = false;
    uint64_t handle = 0;
    VkDeviceSize offset = 0, size = VK_WHOLE_SIZE;
    std::string label;
};
// Host-visible bytes to hash: at submit (uploads) or once complete (readbacks).
struct HostRange {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, size = 0;  // within the allocation
    std::string label;
};
struct CbState {
    DeviceData* device = nullptr;
    uint32_t index = 0;  // commands recorded
    VkPipeline compute = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets;  // bound for compute, by set number
    std::vector<std::pair<VkImage, VkImageLayout>> layouts;
    std::vector<std::pair<VkBuffer, uint32_t>> owners;  // ownership transfers, in order
    std::vector<HostRange> uploads, downloads;
    std::vector<Target> copy_dst, i2b_dst;
    std::vector<std::vector<Target>> dispatch_res;
    uint32_t dispatches = 0;
};
struct QueueInfo {
    DeviceData* device = nullptr;
    uint32_t family = 0;
};
struct Pending {
    VkQueue queue = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint64_t submit = 0;
    std::vector<HostRange> ranges;
};

#define INSTANCE_FUNCS(X)                                                                                              \
    X(DestroyInstance)                                                                                                 \
    X(EnumeratePhysicalDevices)                                                                                        \
    X(GetPhysicalDeviceProperties)                                                                                     \
    X(GetPhysicalDeviceMemoryProperties)                                                                               \
    X(GetPhysicalDeviceQueueFamilyProperties)

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr gipa = nullptr;
#define X(name) PFN_vk##name name = nullptr;
    INSTANCE_FUNCS(X)
#undef X
};

// The device functions the layer intercepts, and with DEVICE_FUNCS those it
// calls on the next layer: these and the one the hashing uses besides.
#define DEVICE_HOOKS(X)                                                                                                \
    X(GetDeviceProcAddr)                                                                                               \
    X(DestroyDevice)                                                                                                   \
    X(GetDeviceQueue)                                                                                                  \
    X(GetDeviceQueue2)                                                                                                 \
    X(QueueSubmit)                                                                                                     \
    X(QueueSubmit2)                                                                                                    \
    X(QueueSubmit2KHR)                                                                                                 \
    X(QueueWaitIdle)                                                                                                   \
    X(DeviceWaitIdle)                                                                                                  \
    X(AllocateMemory)                                                                                                  \
    X(FreeMemory)                                                                                                      \
    X(MapMemory)                                                                                                       \
    X(UnmapMemory)                                                                                                     \
    X(CreateBuffer)                                                                                                    \
    X(DestroyBuffer)                                                                                                   \
    X(GetBufferMemoryRequirements)                                                                                     \
    X(BindBufferMemory)                                                                                                \
    X(BindBufferMemory2)                                                                                               \
    X(CreateImage)                                                                                                     \
    X(DestroyImage)                                                                                                    \
    X(GetImageMemoryRequirements)                                                                                      \
    X(BindImageMemory)                                                                                                 \
    X(BindImageMemory2)                                                                                                \
    X(CreateImageView)                                                                                                 \
    X(DestroyImageView)                                                                                                \
    X(CreateSampler)                                                                                                   \
    X(DestroySampler)                                                                                                  \
    X(CreateShaderModule)                                                                                              \
    X(DestroyShaderModule)                                                                                             \
    X(CreatePipelineCache)                                                                                             \
    X(DestroyPipelineCache)                                                                                            \
    X(CreateDescriptorSetLayout)                                                                                       \
    X(DestroyDescriptorSetLayout)                                                                                      \
    X(CreatePipelineLayout)                                                                                            \
    X(DestroyPipelineLayout)                                                                                           \
    X(CreateComputePipelines)                                                                                          \
    X(CreateGraphicsPipelines)                                                                                         \
    X(DestroyPipeline)                                                                                                 \
    X(CreateDescriptorPool)                                                                                            \
    X(DestroyDescriptorPool)                                                                                           \
    X(ResetDescriptorPool)                                                                                             \
    X(AllocateDescriptorSets)                                                                                          \
    X(FreeDescriptorSets)                                                                                              \
    X(UpdateDescriptorSets)                                                                                            \
    X(CreateCommandPool)                                                                                               \
    X(DestroyCommandPool)                                                                                              \
    X(ResetCommandPool)                                                                                                \
    X(AllocateCommandBuffers)                                                                                          \
    X(FreeCommandBuffers)                                                                                              \
    X(BeginCommandBuffer)                                                                                              \
    X(EndCommandBuffer)                                                                                                \
    X(ResetCommandBuffer)                                                                                              \
    X(CmdBindPipeline)                                                                                                 \
    X(CmdBindDescriptorSets)                                                                                           \
    X(CmdPushConstants)                                                                                                \
    X(CmdDispatch)                                                                                                     \
    X(CmdDispatchBase)                                                                                                 \
    X(CmdDispatchIndirect)                                                                                             \
    X(CmdPipelineBarrier)                                                                                              \
    X(CmdPipelineBarrier2)                                                                                             \
    X(CmdPipelineBarrier2KHR)                                                                                          \
    X(CmdCopyBuffer)                                                                                                   \
    X(CmdCopyImage)                                                                                                    \
    X(CmdBlitImage)                                                                                                    \
    X(CmdCopyBufferToImage)                                                                                            \
    X(CmdCopyImageToBuffer)                                                                                            \
    X(CmdFillBuffer)                                                                                                   \
    X(CmdUpdateBuffer)                                                                                                 \
    X(CmdClearColorImage)                                                                                              \
    X(CmdWriteTimestamp)                                                                                               \
    X(CmdWriteTimestamp2)                                                                                              \
    X(CmdResetQueryPool)                                                                                               \
    X(CmdBeginQuery)                                                                                                   \
    X(CmdEndQuery)                                                                                                     \
    X(CmdExecuteCommands)                                                                                              \
    X(CreateQueryPool)                                                                                                 \
    X(DestroyQueryPool)                                                                                                \
    X(GetQueryPoolResults)                                                                                             \
    X(CreateFence)                                                                                                     \
    X(DestroyFence)                                                                                                    \
    X(ResetFences)                                                                                                     \
    X(WaitForFences)                                                                                                   \
    X(GetFenceStatus)                                                                                                  \
    X(CreateSemaphore)                                                                                                 \
    X(DestroySemaphore)                                                                                                \
    X(GetMemoryFdKHR)                                                                                                  \
    X(CreateSwapchainKHR)                                                                                              \
    X(DestroySwapchainKHR)                                                                                             \
    X(GetSwapchainImagesKHR)                                                                                           \
    X(QueuePresentKHR)
#define DEVICE_FUNCS(X) DEVICE_HOOKS(X) X(InvalidateMappedMemoryRanges)

struct DeviceData {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    InstanceData* instance = nullptr;
    PFN_vkSetDeviceLoaderData set_loader_data = nullptr;
    VkPhysicalDeviceMemoryProperties memory{};
    uint64_t submits = 0;
    std::vector<Pending> pending;
    // Content hashing: a command pool per queue family and a staging buffer,
    // which hash_lock gives to one submitting thread at a time.
    std::mutex hash_lock;
    std::map<uint32_t, std::pair<VkCommandPool, VkCommandBuffer>> pools;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkDeviceSize staging_size = 0;
    void* staging_mapped = nullptr;
    bool staging_coherent = true, staging_cached = true;
#define X(name) PFN_vk##name name = nullptr;
    DEVICE_FUNCS(X)
#undef X
};

// ---------------------------------------------------------------------------
// State. One lock guards all of it and the log, except each device's hashing
// objects, which its hash_lock guards.

std::mutex g_lock;
FILE* g_out = nullptr;
uint64_t g_seq = 0;
std::unordered_map<pid_t, unsigned> g_threads;
std::unordered_map<void*, InstanceData*> g_instances;
std::unordered_map<void*, DeviceData*> g_devices;
std::unordered_map<uint64_t, InstanceData*> g_physical;
std::unordered_map<uint64_t, uint32_t> g_names[kKinds];
uint32_t g_counts[kKinds];
std::unordered_map<uint64_t, MemoryInfo> g_mem;
std::unordered_map<uint64_t, BufferInfo> g_buf;
std::unordered_map<uint64_t, ImageInfo> g_img;
std::unordered_map<uint64_t, ViewInfo> g_view;
std::unordered_map<uint64_t, LayoutInfo> g_layout;
std::unordered_map<uint64_t, SetInfo> g_set;
std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> g_pool_sets;
std::unordered_map<uint64_t, CbState> g_cb;
std::unordered_map<uint64_t, std::vector<VkCommandBuffer>> g_cmdpool_cbs;
std::unordered_map<uint64_t, QueueInfo> g_queue;
std::unordered_map<uint64_t, std::vector<VkImage>> g_swapchain_images;

struct Config {
    bool loaded = false;
    std::string spirv_dirs;
    std::unordered_map<uint64_t, std::string> spirv;  // FNV -> file
    bool spirv_scanned = false;
    // Hash selectors.
    bool hash_i2b = false, hash_copydst = false, hash_storage = false, hash_all = false;
    std::vector<uint32_t> hash_dispatch;
    std::vector<std::string> hash_names;
    bool hash_any = false, hash_need_dispatch = false;
    std::vector<std::pair<uint64_t, uint64_t>> hash_submits;
    bool hash_dispatch_submits = false;
} g_cfg;

// Calls EACH with every nonempty SEPARATOR-separated item of TEXT.
template <class F> void items(std::string_view text, char separator, F each)
{
    while (!text.empty()) {
        const size_t at = text.find(separator);
        if (at) each(text.substr(0, at));
        text = at == std::string_view::npos ? std::string_view() : text.substr(at + 1);
    }
}

// The number TEXT when it is one; otherwise UINT64_MAX.
uint64_t number(std::string_view text)
{
    uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9' || value > (UINT64_MAX - 9) / 10) return UINT64_MAX;
        value = value * 10 + uint64_t(c - '0');
    }
    return text.empty() ? UINT64_MAX : value;
}

// Reports an item of VARIABLE that means nothing, which the layer ignores.
void ignored(const char* variable, std::string_view item)
{
    const int n = int(item.size());
    std::fprintf(stderr, "vktrace: ignoring %s item '%.*s'\n", variable, n, item.data());
    std::fprintf(g_out, "# ignored %s item %.*s\n", variable, n, item.data());
}

void load_config()
{
    if (g_cfg.loaded) return;
    g_cfg.loaded = true;
    std::string path = "/tmp/vktrace.%p.log";
    if (const char* f = std::getenv("VKTRACE_FILE"); f && *f) path = f;
    if (const size_t at = path.find("%p"); at != std::string::npos) path.replace(at, 2, std::to_string(getpid()));
    g_out = std::fopen(path.c_str(), "w");
    if (!g_out) {
        std::fprintf(stderr, "vktrace: cannot write %s (%s); logging to standard error\n", path.c_str(),
                     std::strerror(errno));
        g_out = stderr;
    }
    std::setvbuf(g_out, nullptr, _IOFBF, 1 << 20);
    // Unset and empty variables read alike: "-" in the header.
    const auto variable = [](const char* name) {
        const char* value = std::getenv(name);
        return std::string_view(value ? value : "");
    };
    const std::string_view hash = variable("VKTRACE_HASH"), submits = variable("VKTRACE_HASH_SUBMITS");
    g_cfg.spirv_dirs = variable("VKTRACE_SPIRV");
    std::fprintf(g_out, "# vktrace 1 pid=%d hash=%s hash_submits=%s spirv=%s\n", int(getpid()),
                 hash.empty() ? "-" : hash.data(), submits.empty() ? "-" : submits.data(),
                 g_cfg.spirv_dirs.empty() ? "-" : g_cfg.spirv_dirs.c_str());
    items(hash, ',', [](std::string_view item) {
        const bool dispatch = item.starts_with("dispatch="), named = item.starts_with("buf") || item.starts_with("img");
        const uint64_t n = dispatch || named ? number(item.substr(dispatch ? 9 : 3)) : UINT64_MAX;
        if (item == "i2b") g_cfg.hash_i2b = true;
        else if (item == "copydst") g_cfg.hash_copydst = true;
        else if (item == "storage") g_cfg.hash_storage = true;
        else if (item == "all") g_cfg.hash_all = true;
        else if (n > UINT32_MAX) return ignored("VKTRACE_HASH", item);
        else if (dispatch) g_cfg.hash_dispatch.push_back(uint32_t(n));
        else g_cfg.hash_names.emplace_back(item);
        g_cfg.hash_any = true;
    });
    g_cfg.hash_need_dispatch = g_cfg.hash_storage || !g_cfg.hash_dispatch.empty();
    items(submits, ',', [](std::string_view item) {
        const size_t dash = item.find('-');
        const uint64_t a = number(item.substr(0, dash));
        const uint64_t b = dash == std::string_view::npos ? a : number(item.substr(dash + 1));
        if (item == "dispatch") g_cfg.hash_dispatch_submits = true;
        else if (a > b || b == UINT64_MAX) ignored("VKTRACE_HASH_SUBMITS", item);
        else g_cfg.hash_submits.emplace_back(a, b);
    });
}

void scan_spirv(const std::string& dir, const std::string& rel)
{
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    std::vector<std::string> names;
    while (dirent* e = readdir(d)) names.emplace_back(e->d_name);
    closedir(d);
    std::sort(names.begin(), names.end());
    for (const std::string& n : names) {
        if (n == "." || n == "..") continue;
        const std::string full = dir + "/" + n, sub = rel.empty() ? n : rel + "/" + n;
        struct stat st{};
        if (stat(full.c_str(), &st)) continue;
        if (S_ISDIR(st.st_mode)) {
            scan_spirv(full, sub);
        } else if (n.size() > 4 && n.ends_with(".spv")) {
            FILE* f = std::fopen(full.c_str(), "rb");
            if (!f) continue;
            std::vector<unsigned char> data(size_t(st.st_size));
            const size_t got = std::fread(data.data(), 1, data.size(), f);
            std::fclose(f);
            if (got != data.size()) continue;
            const uint64_t h = fnv(data.data(), data.size());
            auto [it, fresh] = g_cfg.spirv.emplace(h, sub);
            if (!fresh && it->second.find(sub) == std::string::npos) it->second += "|" + sub;
        }
    }
}

void spirv_files()
{
    if (g_cfg.spirv_scanned) return;
    g_cfg.spirv_scanned = true;
    items(g_cfg.spirv_dirs, ':', [](std::string_view dir) { scan_spirv(std::string(dir), ""); });
    std::fprintf(g_out, "# spirv files=%zu\n", g_cfg.spirv.size());
}

unsigned thread_number()
{
    const pid_t tid = pid_t(syscall(SYS_gettid));
    auto [it, fresh] = g_threads.emplace(tid, unsigned(g_threads.size() + 1));
    return it->second;
}

// One log line: "SEQ tN CALL fields". Caller holds g_lock.
void emit(const char* call, const std::string& fields)
{
    std::fprintf(g_out, "%" PRIu64 " t%u %s %s\n", ++g_seq, thread_number(), call, fields.c_str());
}

template <class H> std::string name(Kind k, H h)
{
    const uint64_t v = key_of(h);
    if (!v) return "null";
    auto it = g_names[k].find(v);
    if (it == g_names[k].end()) return "?" + hex(v);
    return std::string(kPrefix[k]) + std::to_string(it->second);
}
template <class H> std::string fresh(Kind k, H h)
{
    const uint64_t v = key_of(h);
    if (!v) return "null";
    const uint32_t n = ++g_counts[k];
    g_names[k][v] = n;
    return std::string(kPrefix[k]) + std::to_string(n);
}
template <class H> std::string forget(Kind k, H h)
{
    std::string n = name(k, h);
    g_names[k].erase(key_of(h));
    return n;
}

InstanceData* instance_of(const void* dispatchable)
{
    auto it = g_instances.find(dispatch_key(dispatchable));
    return it == g_instances.end() ? nullptr : it->second;
}
DeviceData* device_of(const void* dispatchable)
{
    auto it = g_devices.find(dispatch_key(dispatchable));
    return it == g_devices.end() ? nullptr : it->second;
}
InstanceData* instance_of_physical(VkPhysicalDevice pd)
{
    if (auto it = g_physical.find(key_of(pd)); it != g_physical.end()) return it->second;
    return instance_of(pd);
}

std::string rc(VkResult r) { return std::to_string(int(r)); }

// ---------------------------------------------------------------------------
// Structures

// Walks a feature chain and returns "Struct:bit,bit;..." for every enabled bit.
std::string features_str(const VkPhysicalDeviceFeatures* core, const void* chain, std::string* others)
{
    std::string out;
    const auto bits = [&out](const char* sname, const void* s, const FeatureBitName* names, unsigned n) {
        std::string on;
        for (unsigned i = 0; i < n; ++i) {
            VkBool32 b;
            std::memcpy(&b, static_cast<const unsigned char*>(s) + names[i].offset, sizeof b);
            if (b) on += (on.empty() ? "" : ",") + std::string(names[i].name);
        }
        if (!on.empty()) out += (out.empty() ? "" : ";") + std::string(sname) + ":" + on;
    };
    if (core) bits("VkPhysicalDeviceFeatures", core, kCoreFeatureBits, std::size(kCoreFeatureBits));
    for (const void* p = chain; p;) {
        VkStructureType t;
        std::memcpy(&t, p, sizeof t);
        bool known = false;
        if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            bits("VkPhysicalDeviceFeatures", &static_cast<const VkPhysicalDeviceFeatures2*>(p)->features,
                 kCoreFeatureBits, std::size(kCoreFeatureBits));
            known = true;
        }
        for (const auto& fs : kFeatureStructs)
            if (fs.type == t) {
                bits(fs.name, p, fs.bits, fs.count);
                known = true;
            }
        if (!known && others) *others += (others->empty() ? "" : ",") + stype_name(t);
        const void* next;
        std::memcpy(&next, static_cast<const unsigned char*>(p) + offsetof(VkBaseInStructure, pNext), sizeof next);
        p = next;
    }
    return out;
}

template <class T> const T* find_in_chain(const void* chain, VkStructureType type)
{
    for (auto* p = static_cast<const VkBaseInStructure*>(chain); p; p = p->pNext)
        if (p->sType == type) return reinterpret_cast<const T*>(p);
    return nullptr;
}

std::string chain_types(const void* chain)
{
    std::string s;
    for (auto* p = static_cast<const VkBaseInStructure*>(chain); p; p = p->pNext)
        s += (s.empty() ? "" : ",") + stype_name(p->sType);
    return "[" + s + "]";
}

std::string version_str(uint32_t v)
{
    return format("%u.%u.%u", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
}

// ---------------------------------------------------------------------------
// Layer entry points: instance

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char* name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name);

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* alloc,
                                              VkInstance* out)
{
    VkLayerInstanceCreateInfo* link = nullptr;
    for (auto* p = static_cast<const VkBaseInStructure*>(ci->pNext); p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
            reinterpret_cast<const VkLayerInstanceCreateInfo*>(p)->function == VK_LAYER_LINK_INFO)
            link = const_cast<VkLayerInstanceCreateInfo*>(reinterpret_cast<const VkLayerInstanceCreateInfo*>(p));
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;
    const PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult r = create(ci, alloc, out);
    std::lock_guard lock(g_lock);
    load_config();
    std::string layers, exts;
    for (uint32_t i = 0; i < ci->enabledLayerCount; ++i) layers += (i ? "," : "") + token(ci->ppEnabledLayerNames[i]);
    for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i)
        exts += (i ? "," : "") + token(ci->ppEnabledExtensionNames[i]);
    const VkApplicationInfo* app = ci->pApplicationInfo;
    if (r != VK_SUCCESS) {
        emit("CreateInstance", "rc=" + rc(r));
        return r;
    }
    auto* data = new InstanceData;
    data->instance = *out;
    data->gipa = gipa;
#define X(fn) data->fn = reinterpret_cast<PFN_vk##fn>(gipa(*out, "vk" #fn));
    INSTANCE_FUNCS(X)
#undef X
    g_instances[dispatch_key(*out)] = data;
    emit("CreateInstance", format("inst=%s api=%s app=%s layers=[%s] exts=[%s] rc=0", fresh(kInstance, *out).c_str(),
                                  app ? version_str(app->apiVersion).c_str() : "-",
                                  app ? token(app->pApplicationName).c_str() : "-", layers.c_str(), exts.c_str()));
    std::fflush(g_out);
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks* alloc)
{
    InstanceData* data;
    {
        std::lock_guard lock(g_lock);
        data = instance_of(instance);
        emit("DestroyInstance", "inst=" + forget(kInstance, instance));
        std::fflush(g_out);
    }
    if (!data) return;
    data->DestroyInstance(instance, alloc);
    std::lock_guard lock(g_lock);
    g_instances.erase(dispatch_key(instance));
    for (auto it = g_physical.begin(); it != g_physical.end();)
        it = it->second == data ? g_physical.erase(it) : std::next(it);
    delete data;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance instance, uint32_t* count, VkPhysicalDevice* out)
{
    InstanceData* data;
    {
        std::lock_guard lock(g_lock);
        data = instance_of(instance);
    }
    const VkResult r = data->EnumeratePhysicalDevices(instance, count, out);
    if (!out || (r != VK_SUCCESS && r != VK_INCOMPLETE)) return r;
    std::lock_guard lock(g_lock);
    std::string list;
    for (uint32_t i = 0; i < *count; ++i) {
        g_physical[key_of(out[i])] = data;
        const bool known = g_names[kPhysical].count(key_of(out[i]));
        list += (i ? "," : "") + (known ? name(kPhysical, out[i]) : fresh(kPhysical, out[i]));
    }
    emit("EnumeratePhysicalDevices", format("inst=%s pds=[%s] rc=%d", name(kInstance, instance).c_str(), list.c_str(),
                                            int(r)));
    return r;
}

// ---------------------------------------------------------------------------
// Device

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* ci,
                                            const VkAllocationCallbacks* alloc, VkDevice* out)
{
    VkLayerDeviceCreateInfo* link = nullptr;
    PFN_vkSetDeviceLoaderData set_loader_data = nullptr;
    for (auto* p = static_cast<const VkBaseInStructure*>(ci->pNext); p; p = p->pNext) {
        if (p->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO) continue;
        auto* l = const_cast<VkLayerDeviceCreateInfo*>(reinterpret_cast<const VkLayerDeviceCreateInfo*>(p));
        if (l->function == VK_LAYER_LINK_INFO) link = l;
        else if (l->function == VK_LOADER_DATA_CALLBACK) set_loader_data = l->u.pfnSetDeviceLoaderData;
    }
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;
    const PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    InstanceData* inst;
    {
        std::lock_guard lock(g_lock);
        inst = instance_of_physical(physical);
    }
    if (!inst) return VK_ERROR_INITIALIZATION_FAILED;
    const auto create = reinterpret_cast<PFN_vkCreateDevice>(gipa(inst->instance, "vkCreateDevice"));
    const VkResult r = create(physical, ci, alloc, out);
    std::lock_guard lock(g_lock);
    if (r != VK_SUCCESS) {
        emit("CreateDevice", format("pd=%s rc=%d", name(kPhysical, physical).c_str(), int(r)));
        return r;
    }
    auto* d = new DeviceData;
    d->device = *out;
    d->physical = physical;
    d->instance = inst;
    d->set_loader_data = set_loader_data;
#define X(fn) d->fn = reinterpret_cast<PFN_vk##fn>(gdpa(*out, "vk" #fn));
    DEVICE_FUNCS(X)
#undef X
    d->GetDeviceProcAddr = gdpa;
    inst->GetPhysicalDeviceMemoryProperties(physical, &d->memory);
    g_devices[dispatch_key(*out)] = d;
    VkPhysicalDeviceProperties props{};
    inst->GetPhysicalDeviceProperties(physical, &props);
    std::string queues, exts, others;
    for (uint32_t i = 0; i < ci->queueCreateInfoCount; ++i)
        queues += format("%sf%u*%u", i ? "," : "", ci->pQueueCreateInfos[i].queueFamilyIndex,
                         ci->pQueueCreateInfos[i].queueCount);
    std::vector<std::string> sorted;
    for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) sorted.push_back(token(ci->ppEnabledExtensionNames[i]));
    std::sort(sorted.begin(), sorted.end());
    for (const auto& e : sorted) exts += (exts.empty() ? "" : ",") + e;
    const std::string features = features_str(ci->pEnabledFeatures, ci->pNext, &others);
    emit("CreateDevice",
         format("dev=%s pd=%s name=%s api=%s driver=0x%x vendor=0x%x device=0x%x queues=[%s] exts=[%s] rc=0",
                fresh(kDevice, *out).c_str(), name(kPhysical, physical).c_str(), token(props.deviceName).c_str(),
                version_str(props.apiVersion).c_str(), props.driverVersion, props.vendorID, props.deviceID,
                queues.c_str(), exts.c_str()));
    // One line per feature structure, in the order given.
    items(features, ';', [&](std::string_view item) {
        const size_t colon = item.find(':');
        emit("DeviceFeatures", format("dev=%s struct=%s on=[%s]", name(kDevice, *out).c_str(),
                                      std::string(item.substr(0, colon)).c_str(),
                                      std::string(item.substr(colon + 1)).c_str()));
    });
    if (!others.empty()) emit("DeviceChain", format("dev=%s other=[%s]", name(kDevice, *out).c_str(), others.c_str()));
    for (uint32_t i = 0; i < d->memory.memoryTypeCount; ++i)
        emit("MemoryType", format("dev=%s index=%u flags=%s heap=%u heapsize=%" PRIu64, name(kDevice, *out).c_str(), i,
                                  hex(d->memory.memoryTypes[i].propertyFlags).c_str(),
                                  d->memory.memoryTypes[i].heapIndex,
                                  uint64_t(d->memory.memoryHeaps[d->memory.memoryTypes[i].heapIndex].size)));
    std::fflush(g_out);
    return r;
}

void release_hashing(DeviceData* d)
{
    std::lock_guard hashing(d->hash_lock);
    for (auto& [family, pool] : d->pools) d->DestroyCommandPool(d->device, pool.first, nullptr);
    d->pools.clear();
    if (d->staging) d->DestroyBuffer(d->device, d->staging, nullptr);
    if (d->staging_memory) d->FreeMemory(d->device, d->staging_memory, nullptr);
    d->staging = VK_NULL_HANDLE;
    d->staging_memory = VK_NULL_HANDLE;
    d->staging_size = 0;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc)
{
    DeviceData* d;
    {
        std::lock_guard lock(g_lock);
        d = device_of(device);
        emit("DestroyDevice", "dev=" + forget(kDevice, device));
        std::fflush(g_out);
    }
    if (!d) return;
    release_hashing(d);
    d->DestroyDevice(device, alloc);
    std::lock_guard lock(g_lock);
    g_devices.erase(dispatch_key(device));
    delete d;
}

void note_queue(DeviceData* d, VkQueue q, uint32_t family, uint32_t index, const char* call)
{
    const bool known = g_names[kQueue].count(key_of(q));
    g_queue[key_of(q)] = {d, family};
    emit(call, format("dev=%s family=%u index=%u queue=%s", name(kDevice, d->device).c_str(), family, index,
                      known ? name(kQueue, q).c_str() : fresh(kQueue, q).c_str()));
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue* out)
{
    DeviceData* d;
    {
        std::lock_guard lock(g_lock);
        d = device_of(device);
    }
    d->GetDeviceQueue(device, family, index, out);
    std::lock_guard lock(g_lock);
    if (*out) note_queue(d, *out, family, index, "GetDeviceQueue");
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* info, VkQueue* out)
{
    DeviceData* d;
    {
        std::lock_guard lock(g_lock);
        d = device_of(device);
    }
    d->GetDeviceQueue2(device, info, out);
    std::lock_guard lock(g_lock);
    if (*out) note_queue(d, *out, info->queueFamilyIndex, info->queueIndex, "GetDeviceQueue2");
}

DeviceData* dev(const void* dispatchable)
{
    std::lock_guard lock(g_lock);
    return device_of(dispatchable);
}

// ---------------------------------------------------------------------------
// Memory and resources

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* ai,
                                              const VkAllocationCallbacks* alloc, VkDeviceMemory* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->AllocateMemory(device, ai, alloc, out);
    std::lock_guard lock(g_lock);
    std::string extra;
    bool external = false;
    if (auto* ded = find_in_chain<VkMemoryDedicatedAllocateInfo>(ai->pNext,
                                                                  VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO))
        extra += " dedicated=" + (ded->buffer ? name(kBuffer, ded->buffer) : name(kImage, ded->image));
    if (auto* imp = find_in_chain<VkImportMemoryFdInfoKHR>(ai->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR)) {
        extra += " import=fd:" + hex(imp->handleType);
        external = true;
    }
    if (auto* exp =
            find_in_chain<VkExportMemoryAllocateInfo>(ai->pNext, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)) {
        extra += " export=" + hex(exp->handleTypes);
        external = true;
    }
    if (auto* fl = find_in_chain<VkMemoryAllocateFlagsInfo>(ai->pNext, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO))
        extra += " allocflags=" + hex(fl->flags);
    const VkMemoryPropertyFlags flags =
        ai->memoryTypeIndex < d->memory.memoryTypeCount ? d->memory.memoryTypes[ai->memoryTypeIndex].propertyFlags : 0;
    if (r != VK_SUCCESS) {
        emit("AllocateMemory", format("size=%" PRIu64 " type=%u flags=%s%s rc=%d", uint64_t(ai->allocationSize),
                                      ai->memoryTypeIndex, hex(flags).c_str(), extra.c_str(), int(r)));
        return r;
    }
    MemoryInfo m;
    m.device = d;
    m.size = ai->allocationSize;
    m.type = ai->memoryTypeIndex;
    m.flags = flags;
    m.external = external;
    g_mem[key_of(*out)] = m;
    emit("AllocateMemory", format("mem=%s size=%" PRIu64 " type=%u flags=%s%s rc=0", fresh(kMemory, *out).c_str(),
                                  uint64_t(ai->allocationSize), ai->memoryTypeIndex, hex(flags).c_str(),
                                  extra.c_str()));
    return r;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (memory) {
            emit("FreeMemory", "mem=" + forget(kMemory, memory));
            g_mem.erase(key_of(memory));
        }
    }
    d->FreeMemory(device, memory, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                         VkDeviceSize size, VkMemoryMapFlags flags, void** out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->MapMemory(device, memory, offset, size, flags, out);
    std::lock_guard lock(g_lock);
    if (r == VK_SUCCESS)
        if (auto it = g_mem.find(key_of(memory)); it != g_mem.end()) {
            it->second.mapped = *out;
            it->second.map_offset = offset;
            it->second.map_size = size == VK_WHOLE_SIZE ? it->second.size - offset : size;
        }
    emit("MapMemory", format("mem=%s offset=%" PRIu64 " size=%s rc=%d", name(kMemory, memory).c_str(),
                             uint64_t(offset), size_str(size).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice device, VkDeviceMemory memory)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (auto it = g_mem.find(key_of(memory)); it != g_mem.end()) it->second.mapped = nullptr;
        emit("UnmapMemory", "mem=" + name(kMemory, memory));
    }
    d->UnmapMemory(device, memory);
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR* info, int* fd)
{
    DeviceData* d = dev(device);
    const VkResult r = d->GetMemoryFdKHR(device, info, fd);
    std::lock_guard lock(g_lock);
    emit("GetMemoryFdKHR", format("mem=%s type=%s rc=%d", name(kMemory, info->memory).c_str(),
                                  hex(info->handleType).c_str(), int(r)));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(VkDevice device, const VkBufferCreateInfo* ci,
                                            const VkAllocationCallbacks* alloc, VkBuffer* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateBuffer(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    std::string extra;
    bool external = false;
    if (auto* e = find_in_chain<VkExternalMemoryBufferCreateInfo>(
            ci->pNext, VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO)) {
        extra = " external=" + hex(e->handleTypes);
        external = true;
    }
    if (r == VK_SUCCESS) {
        BufferInfo b;
        b.device = d;
        b.size = ci->size;
        b.usage = ci->usage;
        b.external = external;
        g_buf[key_of(*out)] = b;
    }
    emit("CreateBuffer", format("buf=%s size=%" PRIu64 " usage=%s flags=%s sharing=%d%s rc=%d",
                                r == VK_SUCCESS ? fresh(kBuffer, *out).c_str() : "null", uint64_t(ci->size),
                                hex(ci->usage).c_str(), hex(ci->flags).c_str(), int(ci->sharingMode), extra.c_str(),
                                int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (buffer) {
            emit("DestroyBuffer", "buf=" + forget(kBuffer, buffer));
            g_buf.erase(key_of(buffer));
        }
    }
    d->DestroyBuffer(device, buffer, alloc);
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* req)
{
    DeviceData* d = dev(device);
    d->GetBufferMemoryRequirements(device, buffer, req);
    std::lock_guard lock(g_lock);
    emit("GetBufferMemoryRequirements", format("buf=%s size=%" PRIu64 " align=%" PRIu64 " types=%s",
                                               name(kBuffer, buffer).c_str(), uint64_t(req->size),
                                               uint64_t(req->alignment), hex(req->memoryTypeBits).c_str()));
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* req)
{
    DeviceData* d = dev(device);
    d->GetImageMemoryRequirements(device, image, req);
    std::lock_guard lock(g_lock);
    emit("GetImageMemoryRequirements", format("img=%s size=%" PRIu64 " align=%" PRIu64 " types=%s",
                                              name(kImage, image).c_str(), uint64_t(req->size),
                                              uint64_t(req->alignment), hex(req->memoryTypeBits).c_str()));
}

void bind_buffer(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset, VkResult r, const char* call)
{
    if (r == VK_SUCCESS)
        if (auto it = g_buf.find(key_of(buffer)); it != g_buf.end()) {
            it->second.memory = memory;
            it->second.offset = offset;
            if (auto m = g_mem.find(key_of(memory)); m != g_mem.end() && m->second.external) it->second.external = true;
        }
    emit(call, format("buf=%s mem=%s offset=%" PRIu64 " rc=%d", name(kBuffer, buffer).c_str(),
                      name(kMemory, memory).c_str(), uint64_t(offset), int(r)));
}

void bind_image(VkImage image, VkDeviceMemory memory, VkDeviceSize offset, VkResult r, const char* call)
{
    if (r == VK_SUCCESS)
        if (auto it = g_img.find(key_of(image)); it != g_img.end()) {
            it->second.memory = memory;
            it->second.offset = offset;
            if (auto m = g_mem.find(key_of(memory)); m != g_mem.end() && m->second.external) it->second.external = true;
        }
    emit(call, format("img=%s mem=%s offset=%" PRIu64 " rc=%d", name(kImage, image).c_str(),
                      name(kMemory, memory).c_str(), uint64_t(offset), int(r)));
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                                VkDeviceSize offset)
{
    DeviceData* d = dev(device);
    const VkResult r = d->BindBufferMemory(device, buffer, memory, offset);
    std::lock_guard lock(g_lock);
    bind_buffer(buffer, memory, offset, r, "BindBufferMemory");
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory2(VkDevice device, uint32_t n, const VkBindBufferMemoryInfo* infos)
{
    DeviceData* d = dev(device);
    const VkResult r = d->BindBufferMemory2(device, n, infos);
    std::lock_guard lock(g_lock);
    for (uint32_t i = 0; i < n; ++i)
        bind_buffer(infos[i].buffer, infos[i].memory, infos[i].memoryOffset, r, "BindBufferMemory2");
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice device, const VkImageCreateInfo* ci,
                                           const VkAllocationCallbacks* alloc, VkImage* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateImage(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    std::string extra;
    bool external = false;
    if (auto* e = find_in_chain<VkExternalMemoryImageCreateInfo>(ci->pNext,
                                                                 VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO)) {
        extra = " external=" + hex(e->handleTypes);
        external = true;
    }
    if (auto* l =
            find_in_chain<VkImageFormatListCreateInfo>(ci->pNext, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO)) {
        extra += " viewformats=[";
        for (uint32_t i = 0; i < l->viewFormatCount; ++i) extra += (i ? "," : "") + fmt_name(l->pViewFormats[i]);
        extra += "]";
    }
    if (r == VK_SUCCESS) {
        ImageInfo im;
        im.device = d;
        im.type = ci->imageType;
        im.format = ci->format;
        im.extent = ci->extent;
        im.mips = ci->mipLevels;
        im.layers = ci->arrayLayers;
        im.usage = ci->usage;
        im.layout = ci->initialLayout;
        im.external = external;
        g_img[key_of(*out)] = im;
    }
    emit("CreateImage",
         format("img=%s type=%s format=%s extent=%ux%ux%u mips=%u layers=%u samples=%u tiling=%s usage=%s flags=%s "
                "sharing=%d initial=%s%s rc=%d",
                r == VK_SUCCESS ? fresh(kImage, *out).c_str() : "null", enum_name(ImageTypeName, ci->imageType).c_str(),
                fmt_name(ci->format).c_str(), ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels,
                ci->arrayLayers, unsigned(ci->samples), enum_name(ImageTilingName, ci->tiling).c_str(),
                hex(ci->usage).c_str(), hex(ci->flags).c_str(), int(ci->sharingMode),
                layout_name(ci->initialLayout).c_str(), extra.c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (image) {
            emit("DestroyImage", "img=" + forget(kImage, image));
            g_img.erase(key_of(image));
        }
    }
    d->DestroyImage(device, image, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory,
                                               VkDeviceSize offset)
{
    DeviceData* d = dev(device);
    const VkResult r = d->BindImageMemory(device, image, memory, offset);
    std::lock_guard lock(g_lock);
    bind_image(image, memory, offset, r, "BindImageMemory");
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2(VkDevice device, uint32_t n, const VkBindImageMemoryInfo* infos)
{
    DeviceData* d = dev(device);
    const VkResult r = d->BindImageMemory2(device, n, infos);
    std::lock_guard lock(g_lock);
    for (uint32_t i = 0; i < n; ++i)
        bind_image(infos[i].image, infos[i].memory, infos[i].memoryOffset, r, "BindImageMemory2");
    return r;
}

std::string swizzle_str(const VkComponentMapping& c)
{
    const auto one = [](VkComponentSwizzle s) {
        switch (s) {
        case VK_COMPONENT_SWIZZLE_IDENTITY: return "i";
        case VK_COMPONENT_SWIZZLE_ZERO: return "0";
        case VK_COMPONENT_SWIZZLE_ONE: return "1";
        case VK_COMPONENT_SWIZZLE_R: return "r";
        case VK_COMPONENT_SWIZZLE_G: return "g";
        case VK_COMPONENT_SWIZZLE_B: return "b";
        case VK_COMPONENT_SWIZZLE_A: return "a";
        default: return "?";
        }
    };
    return format("%s%s%s%s", one(c.r), one(c.g), one(c.b), one(c.a));
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice device, const VkImageViewCreateInfo* ci,
                                               const VkAllocationCallbacks* alloc, VkImageView* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateImageView(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    std::string extra;
    if (auto* u = find_in_chain<VkImageViewUsageCreateInfo>(ci->pNext, VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO))
        extra = " viewusage=" + hex(u->usage);
    if (r == VK_SUCCESS) g_view[key_of(*out)] = {ci->image};
    emit("CreateImageView", format("view=%s img=%s type=%s format=%s range=%s swizzle=%s flags=%s%s rc=%d",
                                   r == VK_SUCCESS ? fresh(kView, *out).c_str() : "null",
                                   name(kImage, ci->image).c_str(), enum_name(ImageViewTypeName, ci->viewType).c_str(),
                                   fmt_name(ci->format).c_str(), range_str(ci->subresourceRange).c_str(),
                                   swizzle_str(ci->components).c_str(), hex(ci->flags).c_str(), extra.c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (view) {
            emit("DestroyImageView", "view=" + forget(kView, view));
            g_view.erase(key_of(view));
        }
    }
    d->DestroyImageView(device, view, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(VkDevice device, const VkSamplerCreateInfo* ci,
                                             const VkAllocationCallbacks* alloc, VkSampler* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateSampler(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateSampler",
         format("smp=%s mag=%s min=%s mip=%s u=%s v=%s w=%s lodbias=%g aniso=%u:%g compare=%u:%s lod=%g-%g border=%s "
                "unnormalized=%u flags=%s chain=%s rc=%d",
                r == VK_SUCCESS ? fresh(kSampler, *out).c_str() : "null", enum_name(FilterName, ci->magFilter).c_str(),
                enum_name(FilterName, ci->minFilter).c_str(), enum_name(SamplerMipmapModeName, ci->mipmapMode).c_str(),
                enum_name(SamplerAddressModeName, ci->addressModeU).c_str(),
                enum_name(SamplerAddressModeName, ci->addressModeV).c_str(),
                enum_name(SamplerAddressModeName, ci->addressModeW).c_str(), double(ci->mipLodBias),
                ci->anisotropyEnable, double(ci->maxAnisotropy), ci->compareEnable,
                enum_name(CompareOpName, ci->compareOp).c_str(), double(ci->minLod), double(ci->maxLod),
                enum_name(BorderColorName, ci->borderColor).c_str(), ci->unnormalizedCoordinates,
                hex(ci->flags).c_str(), chain_types(ci->pNext).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (sampler) emit("DestroySampler", "smp=" + forget(kSampler, sampler));
    }
    d->DestroySampler(device, sampler, alloc);
}

std::string spirv_str(const void* code, size_t bytes)
{
    spirv_files();
    const uint64_t h = fnv(code, bytes);
    auto it = g_cfg.spirv.find(h);
    return format("bytes=%zu fnv=%016" PRIx64 " file=%s", bytes, h, it == g_cfg.spirv.end() ? "?" : it->second.c_str());
}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* ci,
                                                  const VkAllocationCallbacks* alloc, VkShaderModule* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateShaderModule(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateShaderModule", format("shm=%s %s rc=%d", r == VK_SUCCESS ? fresh(kShader, *out).c_str() : "null",
                                      spirv_str(ci->pCode, ci->codeSize).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice device, VkShaderModule module,
                                               const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (module) emit("DestroyShaderModule", "shm=" + forget(kShader, module));
    }
    d->DestroyShaderModule(device, module, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(VkDevice device, const VkPipelineCacheCreateInfo* ci,
                                                   const VkAllocationCallbacks* alloc, VkPipelineCache* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreatePipelineCache(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreatePipelineCache", format("pc=%s initial=%zu flags=%s rc=%d",
                                       r == VK_SUCCESS ? fresh(kCache, *out).c_str() : "null", ci->initialDataSize,
                                       hex(ci->flags).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(VkDevice device, VkPipelineCache cache,
                                                const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (cache) emit("DestroyPipelineCache", "pc=" + forget(kCache, cache));
    }
    d->DestroyPipelineCache(device, cache, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* ci,
                                                         const VkAllocationCallbacks* alloc,
                                                         VkDescriptorSetLayout* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateDescriptorSetLayout(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    std::string bindings;
    const auto* flags = find_in_chain<VkDescriptorSetLayoutBindingFlagsCreateInfo>(
        ci->pNext, VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
    LayoutInfo info;
    for (uint32_t i = 0; i < ci->bindingCount; ++i) {
        const auto& b = ci->pBindings[i];
        info.counts[b.binding] = b.descriptorCount;
        if (flags && i < flags->bindingCount &&
            (flags->pBindingFlags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT))
            info.variable = true;
        std::string imm;
        if (b.pImmutableSamplers) {
            for (uint32_t k = 0; k < b.descriptorCount; ++k)
                imm += (k ? "|" : "") + name(kSampler, b.pImmutableSamplers[k]);
            imm = ":imm=" + imm;
        }
        bindings += format("%s%u:%s*%u@%s%s%s", i ? "," : "", b.binding, dtype_name(b.descriptorType).c_str(),
                           b.descriptorCount, hex(b.stageFlags).c_str(), imm.c_str(),
                           flags && i < flags->bindingCount ? (":f" + hex(flags->pBindingFlags[i])).c_str() : "");
    }
    if (r == VK_SUCCESS) g_layout[key_of(*out)] = std::move(info);
    emit("CreateDescriptorSetLayout", format("dsl=%s flags=%s bindings=[%s] rc=%d",
                                             r == VK_SUCCESS ? fresh(kSetLayout, *out).c_str() : "null",
                                             hex(ci->flags).c_str(), bindings.c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout,
                                                      const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (layout) emit("DestroyDescriptorSetLayout", "dsl=" + forget(kSetLayout, layout));
        g_layout.erase(key_of(layout));
    }
    d->DestroyDescriptorSetLayout(device, layout, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* ci,
                                                    const VkAllocationCallbacks* alloc, VkPipelineLayout* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreatePipelineLayout(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    std::string sets, push;
    for (uint32_t i = 0; i < ci->setLayoutCount; ++i) sets += (i ? "," : "") + name(kSetLayout, ci->pSetLayouts[i]);
    for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i)
        push += format("%s%s:%u+%u", i ? "," : "", hex(ci->pPushConstantRanges[i].stageFlags).c_str(),
                       ci->pPushConstantRanges[i].offset, ci->pPushConstantRanges[i].size);
    emit("CreatePipelineLayout", format("pl=%s sets=[%s] push=[%s] flags=%s rc=%d",
                                        r == VK_SUCCESS ? fresh(kPipeLayout, *out).c_str() : "null", sets.c_str(),
                                        push.c_str(), hex(ci->flags).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(VkDevice device, VkPipelineLayout layout,
                                                 const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (layout) emit("DestroyPipelineLayout", "pl=" + forget(kPipeLayout, layout));
    }
    d->DestroyPipelineLayout(device, layout, alloc);
}

std::string stage_str(const VkPipelineShaderStageCreateInfo& s)
{
    std::string out = format("stage=%s", hex(s.stage).c_str());
    if (s.module) {
        out += " module=" + name(kShader, s.module);
    } else if (auto* inl =
                   find_in_chain<VkShaderModuleCreateInfo>(s.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)) {
        out += " module=inline " + spirv_str(inl->pCode, inl->codeSize);
    } else {
        out += " module=null";
    }
    out += format(" entry=%s stageflags=%s", token(s.pName).c_str(), hex(s.flags).c_str());
    if (auto* sub = find_in_chain<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>(
            s.pNext, VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO))
        out += format(" subgroup=%u", sub->requiredSubgroupSize);
    std::string others;
    for (auto* p = static_cast<const VkBaseInStructure*>(s.pNext); p; p = p->pNext)
        if (p->sType != VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO &&
            p->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
            others += (others.empty() ? "" : ",") + stype_name(p->sType);
    if (!others.empty()) out += " stagechain=[" + others + "]";
    if (const VkSpecializationInfo* sp = s.pSpecializationInfo) {
        std::string map;
        for (uint32_t i = 0; i < sp->mapEntryCount; ++i)
            map += format("%s%u:%u+%zu", i ? "," : "", sp->pMapEntries[i].constantID, sp->pMapEntries[i].offset,
                          sp->pMapEntries[i].size);
        out += " spec=[" + map + "] specdata=" + bytes_hex(sp->pData, sp->dataSize);
    } else {
        out += " spec=none";
    }
    return out;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t n,
                                                      const VkComputePipelineCreateInfo* ci,
                                                      const VkAllocationCallbacks* alloc, VkPipeline* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateComputePipelines(device, cache, n, ci, alloc, out);
    std::lock_guard lock(g_lock);
    for (uint32_t i = 0; i < n; ++i)
        emit("CreateComputePipeline", format("pipe=%s cache=%s layout=%s flags=%s %s chain=%s rc=%d",
                                             out[i] ? fresh(kPipeline, out[i]).c_str() : "null",
                                             name(kCache, cache).c_str(), name(kPipeLayout, ci[i].layout).c_str(),
                                             hex(ci[i].flags).c_str(), stage_str(ci[i].stage).c_str(),
                                             chain_types(ci[i].pNext).c_str(), int(r)));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t n,
                                                       const VkGraphicsPipelineCreateInfo* ci,
                                                       const VkAllocationCallbacks* alloc, VkPipeline* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateGraphicsPipelines(device, cache, n, ci, alloc, out);
    std::lock_guard lock(g_lock);
    for (uint32_t i = 0; i < n; ++i) {
        std::string stages;
        for (uint32_t k = 0; k < ci[i].stageCount; ++k) {
            std::string one = stage_str(ci[i].pStages[k]);
            std::replace(one.begin(), one.end(), ' ', '/');
            stages += (k ? "|" : "") + one;
        }
        emit("CreateGraphicsPipeline", format("pipe=%s cache=%s layout=%s stages={%s} rc=%d",
                                              out[i] ? fresh(kPipeline, out[i]).c_str() : "null",
                                              name(kCache, cache).c_str(), name(kPipeLayout, ci[i].layout).c_str(),
                                              stages.c_str(), int(r)));
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (pipeline) emit("DestroyPipeline", "pipe=" + forget(kPipeline, pipeline));
    }
    d->DestroyPipeline(device, pipeline, alloc);
}

// ---------------------------------------------------------------------------
// Descriptors

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* ci,
                                                    const VkAllocationCallbacks* alloc, VkDescriptorPool* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateDescriptorPool(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    std::string sizes;
    for (uint32_t i = 0; i < ci->poolSizeCount; ++i)
        sizes += format("%s%s*%u", i ? "," : "", dtype_name(ci->pPoolSizes[i].type).c_str(),
                        ci->pPoolSizes[i].descriptorCount);
    emit("CreateDescriptorPool", format("dp=%s flags=%s maxsets=%u sizes=[%s] rc=%d",
                                        r == VK_SUCCESS ? fresh(kDescPool, *out).c_str() : "null",
                                        hex(ci->flags).c_str(), ci->maxSets, sizes.c_str(), int(r)));
    return r;
}

void drop_pool_sets(VkDescriptorPool pool)
{
    auto it = g_pool_sets.find(key_of(pool));
    if (it == g_pool_sets.end()) return;
    for (VkDescriptorSet s : it->second) {
        g_names[kDescSet].erase(key_of(s));
        g_set.erase(key_of(s));
    }
    g_pool_sets.erase(it);
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice device, VkDescriptorPool pool,
                                                 const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (pool) {
            emit("DestroyDescriptorPool", "dp=" + forget(kDescPool, pool));
            drop_pool_sets(pool);
        }
    }
    d->DestroyDescriptorPool(device, pool, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice device, VkDescriptorPool pool,
                                                   VkDescriptorPoolResetFlags flags)
{
    DeviceData* d = dev(device);
    const VkResult r = d->ResetDescriptorPool(device, pool, flags);
    std::lock_guard lock(g_lock);
    emit("ResetDescriptorPool", format("dp=%s rc=%d", name(kDescPool, pool).c_str(), int(r)));
    drop_pool_sets(pool);
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* ai,
                                                      VkDescriptorSet* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->AllocateDescriptorSets(device, ai, out);
    std::lock_guard lock(g_lock);
    std::string sets;
    const auto* variable = find_in_chain<VkDescriptorSetVariableDescriptorCountAllocateInfo>(
        ai->pNext, VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);
    for (uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
        const bool ok = r == VK_SUCCESS && out[i];
        if (ok) {
            SetInfo& set = g_set[key_of(out[i])] = {};
            if (auto l = g_layout.find(key_of(ai->pSetLayouts[i])); l != g_layout.end()) {
                set.counts = l->second.counts;
                // The variable-count binding is the last one; without a count given, it has none.
                if (l->second.variable && !set.counts.empty())
                    set.counts.rbegin()->second =
                        variable && i < variable->descriptorSetCount ? variable->pDescriptorCounts[i] : 0;
            }
            g_pool_sets[key_of(ai->descriptorPool)].push_back(out[i]);
        }
        sets += format("%s%s:%s", i ? "," : "", ok ? fresh(kDescSet, out[i]).c_str() : "null",
                       name(kSetLayout, ai->pSetLayouts[i]).c_str());
    }
    emit("AllocateDescriptorSets", format("dp=%s sets=[%s] chain=%s rc=%d", name(kDescPool, ai->descriptorPool).c_str(),
                                          sets.c_str(), chain_types(ai->pNext).c_str(), int(r)));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(VkDevice device, VkDescriptorPool pool, uint32_t n,
                                                  const VkDescriptorSet* sets)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        std::string list;
        auto& owned = g_pool_sets[key_of(pool)];
        for (uint32_t i = 0; i < n; ++i) {
            list += (i ? "," : "") + forget(kDescSet, sets[i]);
            g_set.erase(key_of(sets[i]));
            owned.erase(std::remove(owned.begin(), owned.end(), sets[i]), owned.end());
        }
        emit("FreeDescriptorSets", format("dp=%s sets=[%s]", name(kDescPool, pool).c_str(), list.c_str()));
    }
    return d->FreeDescriptorSets(device, pool, n, sets);
}

bool is_buffer_type(VkDescriptorType t)
{
    return t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
}
bool is_image_type(VkDescriptorType t)
{
    return t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
           t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT ||
           t == VK_DESCRIPTOR_TYPE_SAMPLER;
}

// Moves BINDING and ELEMENT past the end of a binding of SET to the next binding
// that has descriptors, from its first element, as consecutive binding updates
// do. Without the set's layout, or past its last binding, they stay.
void consecutive(const SetInfo& set, uint32_t& binding, uint32_t& element)
{
    uint32_t b = binding, e = element;
    for (auto it = set.counts.find(b); it != set.counts.end() && e >= it->second;) {
        e -= it->second;
        do ++it;
        while (it != set.counts.end() && !it->second);
        if (it == set.counts.end()) return;
        b = it->first;
    }
    binding = b;
    element = e;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t nw, const VkWriteDescriptorSet* w,
                                                uint32_t nc, const VkCopyDescriptorSet* c)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        emit("UpdateDescriptorSets", format("writes=%u copies=%u", nw, nc));
        for (uint32_t i = 0; i < nw; ++i) {
            const auto& wr = w[i];
            SetInfo& set = g_set[key_of(wr.dstSet)];
            uint32_t binding = wr.dstBinding, element = wr.dstArrayElement;
            for (uint32_t k = 0; k < wr.descriptorCount; ++k, ++element) {
                consecutive(set, binding, element);
                Desc desc;
                desc.type = wr.descriptorType;
                std::string what;
                if (is_buffer_type(wr.descriptorType)) {
                    const auto& b = wr.pBufferInfo[k];
                    desc.buffer = b.buffer;
                    desc.offset = b.offset;
                    desc.range = b.range;
                    what = format("buf=%s off=%" PRIu64 " range=%s", name(kBuffer, b.buffer).c_str(),
                                  uint64_t(b.offset), size_str(b.range).c_str());
                } else if (is_image_type(wr.descriptorType)) {
                    const auto& im = wr.pImageInfo[k];
                    desc.view = im.imageView;
                    what = format("view=%s layout=%s smp=%s", name(kView, im.imageView).c_str(),
                                  layout_name(im.imageLayout).c_str(), name(kSampler, im.sampler).c_str());
                } else if (wr.pTexelBufferView) {
                    what = format("bview=%s", name(kBufferView, wr.pTexelBufferView[k]).c_str());
                } else {
                    what = "chain=" + chain_types(wr.pNext);
                }
                set.slots[uint64_t(binding) << 32 | element] = desc;
                emit("DescriptorWrite",
                     format("set=%s binding=%u elem=%u type=%s %s", name(kDescSet, wr.dstSet).c_str(), binding, element,
                            dtype_name(wr.descriptorType).c_str(), what.c_str()));
            }
        }
        // One line per descriptor copied, with the bindings it came from and went to.
        for (uint32_t i = 0; i < nc; ++i) {
            const auto& cp = c[i];
            const SetInfo& src = g_set[key_of(cp.srcSet)];
            SetInfo& dst = g_set[key_of(cp.dstSet)];
            uint32_t sb = cp.srcBinding, se = cp.srcArrayElement, db = cp.dstBinding, de = cp.dstArrayElement;
            for (uint32_t k = 0; k < cp.descriptorCount; ++k, ++se, ++de) {
                consecutive(src, sb, se);
                consecutive(dst, db, de);
                if (auto it = src.slots.find(uint64_t(sb) << 32 | se); it != src.slots.end())
                    dst.slots[uint64_t(db) << 32 | de] = it->second;
                emit("DescriptorCopy", format("src=%s:%u:%u dst=%s:%u:%u", name(kDescSet, cp.srcSet).c_str(), sb, se,
                                              name(kDescSet, cp.dstSet).c_str(), db, de));
            }
        }
    }
    d->UpdateDescriptorSets(device, nw, w, nc, c);
}

// ---------------------------------------------------------------------------
// Command pools and buffers

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* ci,
                                                 const VkAllocationCallbacks* alloc, VkCommandPool* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateCommandPool(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateCommandPool", format("cp=%s family=%u flags=%s rc=%d",
                                     r == VK_SUCCESS ? fresh(kCmdPool, *out).c_str() : "null", ci->queueFamilyIndex,
                                     hex(ci->flags).c_str(), int(r)));
    return r;
}

void drop_cb(VkCommandBuffer cb)
{
    g_names[kCmd].erase(key_of(cb));
    g_cb.erase(key_of(cb));
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (pool) {
            emit("DestroyCommandPool", "cp=" + forget(kCmdPool, pool));
            for (VkCommandBuffer cb : g_cmdpool_cbs[key_of(pool)]) drop_cb(cb);
            g_cmdpool_cbs.erase(key_of(pool));
        }
    }
    d->DestroyCommandPool(device, pool, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags)
{
    DeviceData* d = dev(device);
    const VkResult r = d->ResetCommandPool(device, pool, flags);
    std::lock_guard lock(g_lock);
    emit("ResetCommandPool", format("cp=%s flags=%s rc=%d", name(kCmdPool, pool).c_str(), hex(flags).c_str(), int(r)));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* ai,
                                                      VkCommandBuffer* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->AllocateCommandBuffers(device, ai, out);
    std::lock_guard lock(g_lock);
    std::string list;
    for (uint32_t i = 0; r == VK_SUCCESS && i < ai->commandBufferCount; ++i) {
        list += (i ? "," : "") + fresh(kCmd, out[i]);
        g_cb[key_of(out[i])] = CbState{};
        g_cb[key_of(out[i])].device = d;
        g_cmdpool_cbs[key_of(ai->commandPool)].push_back(out[i]);
    }
    emit("AllocateCommandBuffers", format("cp=%s level=%s cbs=[%s] rc=%d", name(kCmdPool, ai->commandPool).c_str(),
                                          enum_name(CommandBufferLevelName, ai->level).c_str(), list.c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t n,
                                              const VkCommandBuffer* cbs)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        std::string list;
        auto& owned = g_cmdpool_cbs[key_of(pool)];
        for (uint32_t i = 0; i < n; ++i) {
            if (!cbs[i]) continue;
            list += (list.empty() ? "" : ",") + name(kCmd, cbs[i]);
            drop_cb(cbs[i]);
            owned.erase(std::remove(owned.begin(), owned.end(), cbs[i]), owned.end());
        }
        emit("FreeCommandBuffers", format("cp=%s cbs=[%s]", name(kCmdPool, pool).c_str(), list.c_str()));
    }
    d->FreeCommandBuffers(device, pool, n, cbs);
}

// The recording state of CB; caller holds g_lock.
CbState& cb_state(VkCommandBuffer cb) { return g_cb[key_of(cb)]; }

void reset_cb(CbState& s)
{
    DeviceData* d = s.device;
    s = CbState{};
    s.device = d;
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        reset_cb(cb_state(cb));
        emit("BeginCommandBuffer", format("cb=%s flags=%s", name(kCmd, cb).c_str(), hex(bi->flags).c_str()));
    }
    return d->BeginCommandBuffer(cb, bi);
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer cb)
{
    DeviceData* d = dev(cb);
    const VkResult r = d->EndCommandBuffer(cb);
    std::lock_guard lock(g_lock);
    const CbState& s = cb_state(cb);
    emit("EndCommandBuffer", format("cb=%s cmds=%u dispatches=%u rc=%d", name(kCmd, cb).c_str(), s.index,
                                    s.dispatches, int(r)));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags flags)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        reset_cb(cb_state(cb));
        emit("ResetCommandBuffer", format("cb=%s flags=%s", name(kCmd, cb).c_str(), hex(flags).c_str()));
    }
    return d->ResetCommandBuffer(cb, flags);
}

// A command line: "CALL cb=cbN i=K fields". Returns the command's index.
uint32_t cmd(const char* call, VkCommandBuffer cb, const std::string& fields)
{
    CbState& s = cb_state(cb);
    const uint32_t i = s.index++;
    emit(call, format("cb=%s i=%u ", name(kCmd, cb).c_str(), i) + fields);
    return i;
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) cb_state(cb).compute = pipeline;
        cmd("CmdBindPipeline", cb, format("bind=%s pipe=%s", enum_name(PipelineBindPointName, bp).c_str(),
                                          name(kPipeline, pipeline).c_str()));
    }
    d->CmdBindPipeline(cb, bp, pipeline);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                                 uint32_t first, uint32_t n, const VkDescriptorSet* sets,
                                                 uint32_t ndyn, const uint32_t* dyn)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string list, offsets;
        CbState& s = cb_state(cb);
        for (uint32_t i = 0; i < n; ++i) {
            list += (i ? "," : "") + name(kDescSet, sets[i]);
            if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) {
                if (s.sets.size() <= first + i) s.sets.resize(first + i + 1);
                s.sets[first + i] = sets[i];
            }
        }
        for (uint32_t i = 0; i < ndyn; ++i) offsets += format("%s%u", i ? "," : "", dyn[i]);
        cmd("CmdBindDescriptorSets", cb,
            format("bind=%s layout=%s first=%u sets=[%s] dyn=[%s]", enum_name(PipelineBindPointName, bp).c_str(),
                   name(kPipeLayout, layout).c_str(), first, list.c_str(), offsets.c_str()));
    }
    d->CmdBindDescriptorSets(cb, bp, layout, first, n, sets, ndyn, dyn);
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages,
                                            uint32_t offset, uint32_t size, const void* data)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        cmd("CmdPushConstants", cb, format("layout=%s stages=%s offset=%u size=%u data=%s",
                                           name(kPipeLayout, layout).c_str(), hex(stages).c_str(), offset, size,
                                           bytes_hex(data, size).c_str()));
    }
    d->CmdPushConstants(cb, layout, stages, offset, size, data);
}

// The storage resources the bound compute sets hold, for hash selectors.
void note_dispatch(CbState& s)
{
    ++s.dispatches;
    if (!g_cfg.hash_need_dispatch) return;
    std::vector<Target> res;
    for (size_t set = 0; set < s.sets.size(); ++set) {
        auto it = g_set.find(key_of(s.sets[set]));
        if (it == g_set.end()) continue;
        for (const auto& [slot, desc] : it->second.slots) {
            const std::string label = format("set%zu.b%u", set, unsigned(slot >> 32));
            if (desc.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && desc.buffer)
                res.push_back({false, key_of(desc.buffer), desc.offset, desc.range, label});
            else if (desc.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && desc.view)
                if (auto v = g_view.find(key_of(desc.view)); v != g_view.end())
                    res.push_back({true, key_of(v->second.image), 0, VK_WHOLE_SIZE, label});
        }
    }
    s.dispatch_res.push_back(std::move(res));
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        CbState& s = cb_state(cb);
        cmd("CmdDispatch", cb, format("n=%u pipe=%s x=%u y=%u z=%u", s.dispatches, name(kPipeline, s.compute).c_str(),
                                      x, y, z));
        note_dispatch(s);
    }
    d->CmdDispatch(cb, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                           uint32_t y, uint32_t z)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        CbState& s = cb_state(cb);
        cmd("CmdDispatchBase", cb, format("n=%u pipe=%s base=%u,%u,%u x=%u y=%u z=%u", s.dispatches,
                                          name(kPipeline, s.compute).c_str(), bx, by, bz, x, y, z));
        note_dispatch(s);
    }
    d->CmdDispatchBase(cb, bx, by, bz, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        CbState& s = cb_state(cb);
        cmd("CmdDispatchIndirect", cb, format("n=%u pipe=%s buf=%s offset=%" PRIu64, s.dispatches,
                                              name(kPipeline, s.compute).c_str(), name(kBuffer, buffer).c_str(),
                                              uint64_t(offset)));
        note_dispatch(s);
    }
    d->CmdDispatchIndirect(cb, buffer, offset);
}

std::string image_barrier_str(VkImage image, VkImageLayout from, VkImageLayout to, const std::string& access,
                              uint32_t sq, uint32_t dq, const VkImageSubresourceRange& r)
{
    return format("%s:%s>%s:%s:%s>%s:%s", name(kImage, image).c_str(), layout_name(from).c_str(),
                  layout_name(to).c_str(), access.c_str(), family_str(sq).c_str(), family_str(dq).c_str(),
                  range_str(r).c_str());
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                              VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier* mb,
                                              uint32_t nb, const VkBufferMemoryBarrier* bb, uint32_t ni,
                                              const VkImageMemoryBarrier* ib)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string m, b, im;
        for (uint32_t i = 0; i < nm; ++i)
            m += format("%s%s>%s", i ? "," : "", hex(mb[i].srcAccessMask).c_str(), hex(mb[i].dstAccessMask).c_str());
        for (uint32_t i = 0; i < nb; ++i)
            b += format("%s%s@%" PRIu64 "+%s:%s>%s:%s>%s", i ? "," : "", name(kBuffer, bb[i].buffer).c_str(),
                        uint64_t(bb[i].offset), size_str(bb[i].size).c_str(), hex(bb[i].srcAccessMask).c_str(),
                        hex(bb[i].dstAccessMask).c_str(), family_str(bb[i].srcQueueFamilyIndex).c_str(),
                        family_str(bb[i].dstQueueFamilyIndex).c_str());
        CbState& s = cb_state(cb);
        for (uint32_t i = 0; i < nb; ++i)
            if (bb[i].srcQueueFamilyIndex != bb[i].dstQueueFamilyIndex)
                s.owners.emplace_back(bb[i].buffer, bb[i].dstQueueFamilyIndex);
        for (uint32_t i = 0; i < ni; ++i) {
            im += (i ? "," : "") +
                  image_barrier_str(ib[i].image, ib[i].oldLayout, ib[i].newLayout,
                                    hex(ib[i].srcAccessMask) + ">" + hex(ib[i].dstAccessMask),
                                    ib[i].srcQueueFamilyIndex, ib[i].dstQueueFamilyIndex, ib[i].subresourceRange);
            s.layouts.emplace_back(ib[i].image, ib[i].newLayout);
        }
        cmd("CmdPipelineBarrier", cb, format("src=%s dst=%s dep=%s mem=[%s] buf=[%s] img=[%s]", hex(src).c_str(),
                                             hex(dst).c_str(), hex(dep).c_str(), m.c_str(), b.c_str(), im.c_str()));
    }
    d->CmdPipelineBarrier(cb, src, dst, dep, nm, mb, nb, bb, ni, ib);
}

void barrier2(const char* call, VkCommandBuffer cb, const VkDependencyInfo* di)
{
    std::string m, b, im;
    const auto sa = [](VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        return format("%s:%s>%s:%s", hex(ss).c_str(), hex(sa).c_str(), hex(ds).c_str(), hex(da).c_str());
    };
    for (uint32_t i = 0; i < di->memoryBarrierCount; ++i) {
        const auto& x = di->pMemoryBarriers[i];
        m += (i ? "," : "") + sa(x.srcStageMask, x.srcAccessMask, x.dstStageMask, x.dstAccessMask);
    }
    for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; ++i) {
        const auto& x = di->pBufferMemoryBarriers[i];
        b += format("%s%s@%" PRIu64 "+%s:%s:%s>%s", i ? "," : "", name(kBuffer, x.buffer).c_str(), uint64_t(x.offset),
                    size_str(x.size).c_str(),
                    sa(x.srcStageMask, x.srcAccessMask, x.dstStageMask, x.dstAccessMask).c_str(),
                    family_str(x.srcQueueFamilyIndex).c_str(), family_str(x.dstQueueFamilyIndex).c_str());
    }
    CbState& s = cb_state(cb);
    for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; ++i) {
        const auto& x = di->pBufferMemoryBarriers[i];
        if (x.srcQueueFamilyIndex != x.dstQueueFamilyIndex) s.owners.emplace_back(x.buffer, x.dstQueueFamilyIndex);
    }
    for (uint32_t i = 0; i < di->imageMemoryBarrierCount; ++i) {
        const auto& x = di->pImageMemoryBarriers[i];
        im += (i ? "," : "") + image_barrier_str(x.image, x.oldLayout, x.newLayout,
                                                 sa(x.srcStageMask, x.srcAccessMask, x.dstStageMask, x.dstAccessMask),
                                                 x.srcQueueFamilyIndex, x.dstQueueFamilyIndex, x.subresourceRange);
        s.layouts.emplace_back(x.image, x.newLayout);
    }
    cmd(call, cb, format("dep=%s mem=[%s] buf=[%s] img=[%s]", hex(di->dependencyFlags).c_str(), m.c_str(), b.c_str(),
                         im.c_str()));
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        barrier2("CmdPipelineBarrier2", cb, di);
    }
    d->CmdPipelineBarrier2(cb, di);
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* di)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        barrier2("CmdPipelineBarrier2", cb, di);
    }
    d->CmdPipelineBarrier2KHR(cb, di);
}

uint64_t host_fnv(const MemoryInfo& m, const void* data, size_t n)
{
    return (m.flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? fnv(data, n) : fnv_uncached(data, n);
}

bool host_visible(VkDeviceMemory memory)
{
    auto it = g_mem.find(key_of(memory));
    return it != g_mem.end() && (it->second.flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
}

// A host range in BUFFER's memory, if it is host-visible.
bool host_range(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::string label, HostRange& out)
{
    auto it = g_buf.find(key_of(buffer));
    if (it == g_buf.end() || !it->second.memory || !host_visible(it->second.memory)) return false;
    out = {it->second.memory, it->second.offset + offset, size, std::move(label)};
    return true;
}

// Bytes a buffer<->image region spans in the buffer.
VkDeviceSize region_bytes(const VkBufferImageCopy& r, VkFormat format)
{
    const VkDeviceSize texel = FormatBytes(format);
    const VkDeviceSize row = r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width;
    const VkDeviceSize height = r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height;
    const VkDeviceSize slices = VkDeviceSize(r.imageExtent.depth) * r.imageSubresource.layerCount;
    if (!texel || !r.imageExtent.width || !r.imageExtent.height || !slices) return 0;
    return texel * ((slices - 1) * row * height + (r.imageExtent.height - 1) * row + r.imageExtent.width);
}

std::string buffer_image_regions(uint32_t n, const VkBufferImageCopy* r)
{
    std::string s;
    for (uint32_t i = 0; i < n; ++i)
        s += format("%s%" PRIu64 ":%u:%u:%s:%d,%d,%d:%ux%ux%u", i ? "," : "", uint64_t(r[i].bufferOffset),
                    r[i].bufferRowLength, r[i].bufferImageHeight, layers_str(r[i].imageSubresource).c_str(),
                    r[i].imageOffset.x, r[i].imageOffset.y, r[i].imageOffset.z, r[i].imageExtent.width,
                    r[i].imageExtent.height, r[i].imageExtent.depth);
    return s;
}

VkFormat image_format(VkImage image)
{
    auto it = g_img.find(key_of(image));
    return it == g_img.end() ? VK_FORMAT_UNDEFINED : it->second.format;
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n,
                                         const VkBufferCopy* r)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string regions;
        for (uint32_t i = 0; i < n; ++i)
            regions += format("%s%" PRIu64 ">%" PRIu64 "+%" PRIu64, i ? "," : "", uint64_t(r[i].srcOffset),
                              uint64_t(r[i].dstOffset), uint64_t(r[i].size));
        const uint32_t at = cmd("CmdCopyBuffer", cb, format("src=%s dst=%s regions=[%s]", name(kBuffer, src).c_str(),
                                                            name(kBuffer, dst).c_str(), regions.c_str()));
        CbState& s = cb_state(cb);
        for (uint32_t i = 0; i < n; ++i) {
            HostRange h;
            const std::string label = format("%s#%u.%u", name(kCmd, cb).c_str(), at, i);
            if (host_range(src, r[i].srcOffset, r[i].size, label, h)) s.uploads.push_back(h);
            if (host_range(dst, r[i].dstOffset, r[i].size, label, h)) s.downloads.push_back(h);
            s.copy_dst.push_back({false, key_of(dst), r[i].dstOffset, r[i].size, label});
        }
    }
    d->CmdCopyBuffer(cb, src, dst, n, r);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                        VkImageLayout dl, uint32_t n, const VkImageCopy* r)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string regions;
        for (uint32_t i = 0; i < n; ++i)
            regions +=
                format("%s%s:%d,%d,%d>%s:%d,%d,%d:%ux%ux%u", i ? "," : "", layers_str(r[i].srcSubresource).c_str(),
                       r[i].srcOffset.x, r[i].srcOffset.y, r[i].srcOffset.z, layers_str(r[i].dstSubresource).c_str(),
                       r[i].dstOffset.x, r[i].dstOffset.y, r[i].dstOffset.z, r[i].extent.width, r[i].extent.height,
                       r[i].extent.depth);
        const uint32_t at =
            cmd("CmdCopyImage", cb,
                format("src=%s:%s dst=%s:%s regions=[%s]", name(kImage, src).c_str(), layout_name(sl).c_str(),
                       name(kImage, dst).c_str(), layout_name(dl).c_str(), regions.c_str()));
        cb_state(cb).copy_dst.push_back(
            {true, key_of(dst), 0, VK_WHOLE_SIZE, format("%s#%u", name(kCmd, cb).c_str(), at)});
    }
    d->CmdCopyImage(cb, src, sl, dst, dl, n, r);
}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                        VkImageLayout dl, uint32_t n, const VkImageBlit* r, VkFilter filter)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string regions;
        for (uint32_t i = 0; i < n; ++i)
            regions += format("%s%s:%d,%d,%d-%d,%d,%d>%s:%d,%d,%d-%d,%d,%d", i ? "," : "",
                              layers_str(r[i].srcSubresource).c_str(), r[i].srcOffsets[0].x, r[i].srcOffsets[0].y,
                              r[i].srcOffsets[0].z, r[i].srcOffsets[1].x, r[i].srcOffsets[1].y, r[i].srcOffsets[1].z,
                              layers_str(r[i].dstSubresource).c_str(), r[i].dstOffsets[0].x, r[i].dstOffsets[0].y,
                              r[i].dstOffsets[0].z, r[i].dstOffsets[1].x, r[i].dstOffsets[1].y, r[i].dstOffsets[1].z);
        const uint32_t at = cmd("CmdBlitImage", cb,
                                format("src=%s:%s dst=%s:%s filter=%s regions=[%s]", name(kImage, src).c_str(),
                                       layout_name(sl).c_str(), name(kImage, dst).c_str(), layout_name(dl).c_str(),
                                       enum_name(FilterName, filter).c_str(), regions.c_str()));
        cb_state(cb).copy_dst.push_back(
            {true, key_of(dst), 0, VK_WHOLE_SIZE, format("%s#%u", name(kCmd, cb).c_str(), at)});
    }
    d->CmdBlitImage(cb, src, sl, dst, dl, n, r, filter);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout dl,
                                                uint32_t n, const VkBufferImageCopy* r)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        const uint32_t at = cmd("CmdCopyBufferToImage", cb,
                                format("src=%s dst=%s:%s regions=[%s]", name(kBuffer, src).c_str(),
                                       name(kImage, dst).c_str(), layout_name(dl).c_str(),
                                       buffer_image_regions(n, r).c_str()));
        CbState& s = cb_state(cb);
        const VkFormat f = image_format(dst);
        for (uint32_t i = 0; i < n; ++i) {
            HostRange h;
            if (host_range(src, r[i].bufferOffset, region_bytes(r[i], f),
                           format("%s#%u.%u", name(kCmd, cb).c_str(), at, i), h))
                s.uploads.push_back(h);
        }
        s.copy_dst.push_back({true, key_of(dst), 0, VK_WHOLE_SIZE, format("%s#%u", name(kCmd, cb).c_str(), at)});
    }
    d->CmdCopyBufferToImage(cb, src, dst, dl, n, r);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkBuffer dst,
                                                uint32_t n, const VkBufferImageCopy* r)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        const uint32_t at = cmd("CmdCopyImageToBuffer", cb,
                                format("src=%s:%s dst=%s regions=[%s]", name(kImage, src).c_str(),
                                       layout_name(sl).c_str(), name(kBuffer, dst).c_str(),
                                       buffer_image_regions(n, r).c_str()));
        CbState& s = cb_state(cb);
        const VkFormat f = image_format(src);
        for (uint32_t i = 0; i < n; ++i) {
            const VkDeviceSize bytes = region_bytes(r[i], f);
            const std::string label = format("%s#%u.%u", name(kCmd, cb).c_str(), at, i);
            HostRange h;
            if (host_range(dst, r[i].bufferOffset, bytes, label, h)) s.downloads.push_back(h);
            s.i2b_dst.push_back({false, key_of(dst), r[i].bufferOffset, bytes, label});
            s.copy_dst.push_back({false, key_of(dst), r[i].bufferOffset, bytes, label});
        }
    }
    d->CmdCopyImageToBuffer(cb, src, sl, dst, n, r);
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                         uint32_t data)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        const uint32_t at = cmd("CmdFillBuffer", cb, format("buf=%s offset=%" PRIu64 " size=%s data=0x%08x",
                                                            name(kBuffer, buffer).c_str(), uint64_t(offset),
                                                            size_str(size).c_str(), data));
        cb_state(cb).copy_dst.push_back(
            {false, key_of(buffer), offset, size, format("%s#%u", name(kCmd, cb).c_str(), at)});
    }
    d->CmdFillBuffer(cb, buffer, offset, size, data);
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                           VkDeviceSize size, const void* data)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        const uint32_t at = cmd("CmdUpdateBuffer", cb,
                                format("buf=%s offset=%" PRIu64 " size=%" PRIu64 " fnv=%016" PRIx64 " data=%s",
                                       name(kBuffer, buffer).c_str(), uint64_t(offset), uint64_t(size),
                                       fnv(data, size_t(size)),
                                       size <= 1024 ? bytes_hex(data, size_t(size)).c_str() : "long"));
        cb_state(cb).copy_dst.push_back(
            {false, key_of(buffer), offset, size, format("%s#%u", name(kCmd, cb).c_str(), at)});
    }
    d->CmdUpdateBuffer(cb, buffer, offset, size, data);
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                                              const VkClearColorValue* color, uint32_t n,
                                              const VkImageSubresourceRange* ranges)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string list;
        for (uint32_t i = 0; i < n; ++i) list += (i ? "," : "") + range_str(ranges[i]);
        const uint32_t at =
            cmd("CmdClearColorImage", cb,
                format("img=%s:%s color=%s ranges=[%s]", name(kImage, image).c_str(), layout_name(layout).c_str(),
                       bytes_hex(color, sizeof *color).c_str(), list.c_str()));
        cb_state(cb).copy_dst.push_back(
            {true, key_of(image), 0, VK_WHOLE_SIZE, format("%s#%u", name(kCmd, cb).c_str(), at)});
    }
    d->CmdClearColorImage(cb, image, layout, color, n, ranges);
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits stage, VkQueryPool pool,
                                             uint32_t query)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        cmd("CmdWriteTimestamp", cb, format("stage=%s qp=%s query=%u", hex(stage).c_str(),
                                            name(kQueryPool, pool).c_str(), query));
    }
    d->CmdWriteTimestamp(cb, stage, pool, query);
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool,
                                              uint32_t query)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        cmd("CmdWriteTimestamp2", cb, format("stage=%s qp=%s query=%u", hex(stage).c_str(),
                                             name(kQueryPool, pool).c_str(), query));
    }
    d->CmdWriteTimestamp2(cb, stage, pool, query);
}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        cmd("CmdResetQueryPool", cb, format("qp=%s first=%u count=%u", name(kQueryPool, pool).c_str(), first, count));
    }
    d->CmdResetQueryPool(cb, pool, first, count);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query,
                                         VkQueryControlFlags flags)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        cmd("CmdBeginQuery", cb, format("qp=%s query=%u flags=%s", name(kQueryPool, pool).c_str(), query,
                                        hex(flags).c_str()));
    }
    d->CmdBeginQuery(cb, pool, query, flags);
}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        cmd("CmdEndQuery", cb, format("qp=%s query=%u", name(kQueryPool, pool).c_str(), query));
    }
    d->CmdEndQuery(cb, pool, query);
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* cbs)
{
    DeviceData* d = dev(cb);
    {
        std::lock_guard lock(g_lock);
        std::string list;
        for (uint32_t i = 0; i < n; ++i) list += (i ? "," : "") + name(kCmd, cbs[i]);
        cmd("CmdExecuteCommands", cb, "cbs=[" + list + "]");
    }
    d->CmdExecuteCommands(cb, n, cbs);
}

// ---------------------------------------------------------------------------
// Queries, fences, semaphores, swapchains

VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* ci,
                                               const VkAllocationCallbacks* alloc, VkQueryPool* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateQueryPool(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateQueryPool",
         format("qp=%s type=%s count=%u rc=%d", r == VK_SUCCESS ? fresh(kQueryPool, *out).c_str() : "null",
                enum_name(QueryTypeName, ci->queryType).c_str(), ci->queryCount, int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(VkDevice device, VkQueryPool pool, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (pool) emit("DestroyQueryPool", "qp=" + forget(kQueryPool, pool));
    }
    d->DestroyQueryPool(device, pool, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResults(VkDevice device, VkQueryPool pool, uint32_t first, uint32_t count,
                                                   size_t size, void* data, VkDeviceSize stride,
                                                   VkQueryResultFlags flags)
{
    DeviceData* d = dev(device);
    const VkResult r = d->GetQueryPoolResults(device, pool, first, count, size, data, stride, flags);
    std::lock_guard lock(g_lock);
    emit("GetQueryPoolResults", format("qp=%s first=%u count=%u flags=%s rc=%d", name(kQueryPool, pool).c_str(), first,
                                       count, hex(flags).c_str(), int(r)));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice device, const VkFenceCreateInfo* ci,
                                           const VkAllocationCallbacks* alloc, VkFence* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateFence(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateFence",
         format("fence=%s flags=%s chain=%s rc=%d", r == VK_SUCCESS ? fresh(kFence, *out).c_str() : "null",
                hex(ci->flags).c_str(), chain_types(ci->pNext).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (fence) emit("DestroyFence", "fence=" + forget(kFence, fence));
    }
    d->DestroyFence(device, fence, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice device, uint32_t n, const VkFence* fences)
{
    DeviceData* d = dev(device);
    const VkResult r = d->ResetFences(device, n, fences);
    std::lock_guard lock(g_lock);
    std::string list;
    for (uint32_t i = 0; i < n; ++i) list += (i ? "," : "") + name(kFence, fences[i]);
    emit("ResetFences", format("fences=[%s] rc=%d", list.c_str(), int(r)));
    return r;
}

// Hashes the host ranges of completed work; caller holds g_lock.
void hash_host(DeviceData* d, const Pending& p, const char* why)
{
    for (const HostRange& h : p.ranges) {
        auto it = g_mem.find(key_of(h.memory));
        if (it == g_mem.end()) continue;
        const MemoryInfo& m = it->second;
        const unsigned char* base = nullptr;
        bool mine = false;
        void* mapped = nullptr;
        if (m.mapped && h.offset >= m.map_offset && h.offset + h.size <= m.map_offset + m.map_size) {
            base = static_cast<const unsigned char*>(m.mapped) + (h.offset - m.map_offset);
            if (!(m.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, h.memory, m.map_offset,
                                          VK_WHOLE_SIZE};
                d->InvalidateMappedMemoryRanges(d->device, 1, &range);
            }
        } else if (!m.mapped && d->MapMemory(d->device, h.memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
            mine = true;
            base = static_cast<const unsigned char*>(mapped) + h.offset;
        }
        if (base)
            emit("Readback",
                 format("sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=%016" PRIx64 " via=%s",
                        p.submit, h.label.c_str(), name(kMemory, h.memory).c_str(), uint64_t(h.offset),
                        uint64_t(h.size), host_fnv(m, base, size_t(h.size)), why));
        else
            emit("Readback", format("sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=unmapped", p.submit,
                                    h.label.c_str(), name(kMemory, h.memory).c_str(), uint64_t(h.offset),
                                    uint64_t(h.size)));
        if (mine) d->UnmapMemory(d->device, h.memory);
    }
}

template <class Pred> void complete(DeviceData* d, Pred done, const char* why)
{
    for (auto it = d->pending.begin(); it != d->pending.end();) {
        if (done(*it)) {
            hash_host(d, *it, why);
            it = d->pending.erase(it);
        } else {
            ++it;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice device, uint32_t n, const VkFence* fences, VkBool32 all,
                                             uint64_t timeout)
{
    DeviceData* d = dev(device);
    const VkResult r = d->WaitForFences(device, n, fences, all, timeout);
    std::lock_guard lock(g_lock);
    std::string list;
    for (uint32_t i = 0; i < n; ++i) list += (i ? "," : "") + name(kFence, fences[i]);
    emit("WaitForFences", format("fences=[%s] all=%u timeout=%" PRIu64 " rc=%d", list.c_str(), all, timeout, int(r)));
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < n; ++i)
            if (all || n == 1 || d->GetFenceStatus(device, fences[i]) == VK_SUCCESS)
                complete(d, [&](const Pending& p) { return p.fence == fences[i]; }, "fence");
    std::fflush(g_out);
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(VkDevice device, VkFence fence)
{
    DeviceData* d = dev(device);
    const VkResult r = d->GetFenceStatus(device, fence);
    std::lock_guard lock(g_lock);
    emit("GetFenceStatus", format("fence=%s rc=%d", name(kFence, fence).c_str(), int(r)));
    if (r == VK_SUCCESS) complete(d, [&](const Pending& p) { return p.fence == fence; }, "fence");
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* ci,
                                               const VkAllocationCallbacks* alloc, VkSemaphore* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateSemaphore(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateSemaphore", format("sem=%s chain=%s rc=%d", r == VK_SUCCESS ? fresh(kSemaphore, *out).c_str() : "null",
                                   chain_types(ci->pNext).c_str(), int(r)));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroySemaphore(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (sem) emit("DestroySemaphore", "sem=" + forget(kSemaphore, sem));
    }
    d->DestroySemaphore(device, sem, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* ci,
                                                  const VkAllocationCallbacks* alloc, VkSwapchainKHR* out)
{
    DeviceData* d = dev(device);
    const VkResult r = d->CreateSwapchainKHR(device, ci, alloc, out);
    std::lock_guard lock(g_lock);
    emit("CreateSwapchainKHR",
         format("sc=%s format=%s colorspace=%d extent=%ux%u usage=%s images=%u present=%d old=%s rc=%d",
                r == VK_SUCCESS ? fresh(kSwapchain, *out).c_str() : "null", fmt_name(ci->imageFormat).c_str(),
                int(ci->imageColorSpace), ci->imageExtent.width, ci->imageExtent.height, hex(ci->imageUsage).c_str(),
                ci->minImageCount, int(ci->presentMode), name(kSwapchain, ci->oldSwapchain).c_str(), int(r)));
    if (r == VK_SUCCESS) {
        ImageInfo im;
        im.device = d;
        im.format = ci->imageFormat;
        im.extent = {ci->imageExtent.width, ci->imageExtent.height, 1};
        im.layers = ci->imageArrayLayers;
        im.usage = ci->imageUsage;
        im.swapchain = true;
        g_img[key_of(*out) ^ 1] = im;  // a template for the images, looked up below
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(VkDevice device, VkSwapchainKHR sc, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = dev(device);
    {
        std::lock_guard lock(g_lock);
        if (sc) {
            emit("DestroySwapchainKHR", "sc=" + forget(kSwapchain, sc));
            for (VkImage im : g_swapchain_images[key_of(sc)]) {
                g_names[kImage].erase(key_of(im));
                g_img.erase(key_of(im));
            }
            g_swapchain_images.erase(key_of(sc));
            g_img.erase(key_of(sc) ^ 1);
        }
    }
    d->DestroySwapchainKHR(device, sc, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR sc, uint32_t* count,
                                                     VkImage* images)
{
    DeviceData* d = dev(device);
    const VkResult r = d->GetSwapchainImagesKHR(device, sc, count, images);
    if (!images || (r != VK_SUCCESS && r != VK_INCOMPLETE)) return r;
    std::lock_guard lock(g_lock);
    std::string list;
    auto& owned = g_swapchain_images[key_of(sc)];
    for (uint32_t i = 0; i < *count; ++i) {
        const bool known = g_names[kImage].count(key_of(images[i]));
        list += (i ? "," : "") + (known ? name(kImage, images[i]) : fresh(kImage, images[i]));
        if (!known) {
            owned.push_back(images[i]);
            if (auto t = g_img.find(key_of(sc) ^ 1); t != g_img.end()) g_img[key_of(images[i])] = t->second;
        }
    }
    emit("GetSwapchainImagesKHR", format("sc=%s imgs=[%s] rc=%d", name(kSwapchain, sc).c_str(), list.c_str(), int(r)));
    return r;
}

// ---------------------------------------------------------------------------
// Submission and content hashes

bool hash_this_submit(uint64_t sub, bool dispatches)
{
    if (!g_cfg.hash_any) return false;
    if (g_cfg.hash_dispatch_submits && !dispatches) return false;
    if (g_cfg.hash_submits.empty()) return true;
    for (auto [a, b] : g_cfg.hash_submits)
        if (sub >= a && sub <= b) return true;
    return false;
}

// A command buffer of the device's own on FAMILY, recording; caller holds D's
// hash_lock and not g_lock.
VkCommandBuffer hash_cb(DeviceData* d, uint32_t family)
{
    auto it = d->pools.find(family);
    if (it == d->pools.end()) {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = family;
        VkCommandPool pool = VK_NULL_HANDLE;
        if (d->CreateCommandPool(d->device, &ci, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (d->AllocateCommandBuffers(d->device, &ai, &cb) != VK_SUCCESS) {
            d->DestroyCommandPool(d->device, pool, nullptr);
            return VK_NULL_HANDLE;
        }
        // A command buffer made below the loader carries no dispatch pointer yet.
        if (d->set_loader_data) d->set_loader_data(d->device, cb);
        else *reinterpret_cast<void**>(cb) = dispatch_key(d->device);
        it = d->pools.emplace(family, std::make_pair(pool, cb)).first;
    }
    VkCommandBuffer cb = it->second.second;
    d->ResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d->BeginCommandBuffer(cb, &bi);
    return cb;
}

// A host-visible staging buffer of at least BYTES; caller holds D's hash_lock.
bool staging(DeviceData* d, VkDeviceSize bytes)
{
    if (d->staging_size >= bytes) return true;
    if (d->staging) d->DestroyBuffer(d->device, d->staging, nullptr);
    if (d->staging_memory) d->FreeMemory(d->device, d->staging_memory, nullptr);
    d->staging = VK_NULL_HANDLE;
    d->staging_memory = VK_NULL_HANDLE;
    d->staging_size = 0;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (d->CreateBuffer(d->device, &bi, nullptr, &d->staging) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->device, d->staging, &req);
    uint32_t type = UINT32_MAX;
    // Cached host memory reads fast; any host-visible type does.
    // Cached system memory first; a device with nothing else (lavapipe) has only device-local types.
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const struct { VkMemoryPropertyFlags want, reject; } order[] = {
        {visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
        {visible, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
        {visible, 0}};
    for (const auto& o : order) {
        for (uint32_t i = 0; i < d->memory.memoryTypeCount && type == UINT32_MAX; ++i) {
            const VkMemoryPropertyFlags f = d->memory.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) && (f & o.want) == o.want && !(f & o.reject)) type = i;
        }
        if (type != UINT32_MAX) break;
    }
    if (type == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (d->AllocateMemory(d->device, &ai, nullptr, &d->staging_memory) != VK_SUCCESS) return false;
    if (d->BindBufferMemory(d->device, d->staging, d->staging_memory, 0) != VK_SUCCESS) return false;
    if (d->MapMemory(d->device, d->staging_memory, 0, VK_WHOLE_SIZE, 0, &d->staging_mapped) != VK_SUCCESS) return false;
    d->staging_coherent = d->memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    d->staging_cached = d->memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    d->staging_size = bytes;
    return true;
}

// What a target resolved to, for hashing without g_lock.
struct Job {
    Target target;
    std::string what;  // "buf12+off+size" or "img3"
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, bytes = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
    uint32_t layers = 1;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // A buffer another queue family owns (external memory the application released to
    // VK_QUEUE_FAMILY_EXTERNAL): acquired for the copy and released back to it.
    uint32_t owner = VK_QUEUE_FAMILY_IGNORED;
    std::string skip;
};

// Resolves TARGET against the tracked state; caller holds g_lock.
Job resolve(const Target& t)
{
    Job j;
    j.target = t;
    if (!t.image) {
        auto it = g_buf.find(t.handle);
        j.buffer = VkBuffer(uintptr_t(t.handle));
        j.what = name(kBuffer, j.buffer);
        if (it == g_buf.end()) {
            j.skip = "unknown";
            return j;
        }
        j.offset = t.offset;
        j.bytes = t.size == VK_WHOLE_SIZE ? it->second.size - std::min(t.offset, it->second.size) : t.size;
        j.what += format("+%" PRIu64 "+%" PRIu64, uint64_t(j.offset), uint64_t(j.bytes));
        if (!(it->second.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) j.skip = "no-transfer-src";
        else if (it->second.external) {
            // Memory shared with another device or process: only after a submitted transfer
            // says who holds it, and then through the same acquire and release.
            j.owner = it->second.owner;
            if (j.owner == VK_QUEUE_FAMILY_IGNORED) j.skip = "external";
        }
        return j;
    }
    auto it = g_img.find(t.handle);
    j.image = VkImage(uintptr_t(t.handle));
    j.what = name(kImage, j.image);
    if (it == g_img.end()) {
        j.skip = "unknown";
        return j;
    }
    const ImageInfo& im = it->second;
    j.format = im.format;
    j.extent = im.extent;
    j.layers = im.layers;
    j.layout = im.layout;
    j.bytes = VkDeviceSize(FormatBytes(im.format)) * im.extent.width * im.extent.height * im.extent.depth * im.layers;
    if (!(im.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) j.skip = "no-transfer-src";
    else if (!FormatBytes(im.format) || (im.format >= VK_FORMAT_D16_UNORM && im.format <= VK_FORMAT_D32_SFLOAT_S8_UINT))
        j.skip = "format";
    else if (im.external || im.swapchain) j.skip = "external";
    else if (im.layout == VK_IMAGE_LAYOUT_UNDEFINED || im.layout == VK_IMAGE_LAYOUT_PREINITIALIZED) j.skip = "layout";
    return j;
}

// Copies each job's bytes out on QUEUE and logs their FNV. Called without g_lock,
// on the thread that submitted, with the queue idle. The hashing objects are the
// device's, so threads that submit to its queues take turns.
void run_jobs(DeviceData* d, VkQueue queue, uint32_t family, uint64_t sub, std::vector<Job>& jobs)
{
    std::vector<std::string> lines;
    std::unique_lock hashing(d->hash_lock);
    for (Job& j : jobs) {
        if (j.skip.empty() && !j.bytes) j.skip = "empty";
        if (!j.skip.empty()) {
            lines.push_back(format("sub=%" PRIu64 " sel=%s res=%s skip=%s", sub, j.target.label.c_str(), j.what.c_str(),
                                   j.skip.c_str()));
            continue;
        }
        if (!staging(d, j.bytes)) {
            lines.push_back(
                format("sub=%" PRIu64 " sel=%s res=%s skip=staging", sub, j.target.label.c_str(), j.what.c_str()));
            continue;
        }
        VkCommandBuffer cb = hash_cb(d, family);
        if (!cb) break;
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT};
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0,
                              nullptr, 0, nullptr);
        if (j.buffer) {
            const bool foreign = j.owner != VK_QUEUE_FAMILY_IGNORED && j.owner != family;
            VkBufferMemoryBarrier own{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            own.srcAccessMask = 0;
            own.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            own.srcQueueFamilyIndex = j.owner;
            own.dstQueueFamilyIndex = family;
            own.buffer = j.buffer;
            own.offset = 0;
            own.size = VK_WHOLE_SIZE;
            if (foreign)
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                      nullptr, 1, &own, 0, nullptr);
            const VkBufferCopy region{j.offset, 0, j.bytes};
            d->CmdCopyBuffer(cb, j.buffer, d->staging, 1, &region);
            if (foreign) {
                std::swap(own.srcQueueFamilyIndex, own.dstQueueFamilyIndex);
                own.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                own.dstAccessMask = 0;
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                                      nullptr, 1, &own, 0, nullptr);
            }
        } else {
            const bool direct = j.layout == VK_IMAGE_LAYOUT_GENERAL || j.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            const VkImageLayout copy_layout = direct ? j.layout : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            VkImageMemoryBarrier to{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            to.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to.oldLayout = j.layout;
            to.newLayout = copy_layout;
            to.srcQueueFamilyIndex = to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.image = j.image;
            to.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, j.layers};
            if (!direct)
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                      nullptr, 0, nullptr, 1, &to);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, j.layers};
            region.imageExtent = j.extent;
            d->CmdCopyImageToBuffer(cb, j.image, copy_layout, d->staging, 1, &region);
            if (!direct) {
                std::swap(to.oldLayout, to.newLayout);
                to.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                to.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                      nullptr, 0, nullptr, 1, &to);
            }
        }
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_HOST_READ_BIT};
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                              0, nullptr);
        d->EndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        if (d->QueueSubmit(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS || d->QueueWaitIdle(queue) != VK_SUCCESS) {
            lines.push_back(
                format("sub=%" PRIu64 " sel=%s res=%s skip=submit", sub, j.target.label.c_str(), j.what.c_str()));
            break;
        }
        if (!d->staging_coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, d->staging_memory, 0,
                                      VK_WHOLE_SIZE};
            d->InvalidateMappedMemoryRanges(d->device, 1, &range);
        }
        lines.push_back(format("sub=%" PRIu64 " sel=%s res=%s bytes=%" PRIu64 " fnv=%016" PRIx64, sub,
                               j.target.label.c_str(), j.what.c_str(), uint64_t(j.bytes),
                               d->staging_cached ? fnv(d->staging_mapped, size_t(j.bytes))
                                                 : fnv_uncached(d->staging_mapped, size_t(j.bytes))));
    }
    hashing.unlock();
    std::lock_guard lock(g_lock);
    for (const auto& l : lines) emit("Hash", l);
    std::fflush(g_out);
}

// The targets the selectors name in the submitted command buffers; caller holds g_lock.
void select_targets(DeviceData* d, const std::vector<const CbState*>& cbs, std::vector<Target>& out)
{
    uint32_t dispatch = 0;
    for (const CbState* s : cbs) {
        if (g_cfg.hash_i2b)
            for (const Target& t : s->i2b_dst) out.push_back({t.image, t.handle, t.offset, t.size, "i2b:" + t.label});
        if (g_cfg.hash_copydst)
            for (const Target& t : s->copy_dst)
                out.push_back({t.image, t.handle, t.offset, t.size, "copydst:" + t.label});
        for (const auto& res : s->dispatch_res) {
            const bool wanted = g_cfg.hash_storage || std::find(g_cfg.hash_dispatch.begin(), g_cfg.hash_dispatch.end(),
                                                                dispatch) != g_cfg.hash_dispatch.end();
            if (wanted)
                for (const Target& t : res)
                    out.push_back(
                        {t.image, t.handle, t.offset, t.size, format("dispatch%u:%s", dispatch, t.label.c_str())});
            ++dispatch;
        }
    }
    for (const std::string& n : g_cfg.hash_names) {
        const bool image = n.starts_with("img");
        const Kind k = image ? kImage : kBuffer;
        for (const auto& [handle, number] : g_names[k])
            if (n == std::string(kPrefix[k]) + std::to_string(number))
                out.push_back({image, handle, 0, VK_WHOLE_SIZE, n});
    }
    if (g_cfg.hash_all) {
        std::vector<std::pair<uint32_t, Target>> all;
        for (const auto& [handle, info] : g_buf)
            if (info.device == d && g_names[kBuffer].count(handle))
                all.push_back({g_names[kBuffer][handle], {false, handle, 0, VK_WHOLE_SIZE, "all"}});
        for (const auto& [handle, info] : g_img)
            if (info.device == d && g_names[kImage].count(handle))
                all.push_back({g_names[kImage][handle] | 0x80000000u, {true, handle, 0, VK_WHOLE_SIZE, "all"}});
        std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& [n, t] : all) out.push_back(t);
    }
    // Several selectors may name one region; hash it once.
    std::vector<Target> unique;
    for (Target& t : out) {
        auto it = std::find_if(unique.begin(), unique.end(), [&](const Target& u) {
            return u.image == t.image && u.handle == t.handle && u.offset == t.offset && u.size == t.size;
        });
        if (it == unique.end()) unique.push_back(std::move(t));
        else it->label += "|" + t.label;
    }
    out = std::move(unique);
}

std::string submit_line(uint64_t sub, VkQueue queue, uint32_t batch, uint32_t ncb, const VkCommandBuffer* cbs,
                        const std::string& waits, const std::string& signals, VkFence fence)
{
    std::string list;
    for (uint32_t i = 0; i < ncb; ++i) list += (i ? "," : "") + name(kCmd, cbs[i]);
    return format("sub=%" PRIu64 " q=%s batch=%u cbs=[%s] wait=[%s] signal=[%s] fence=%s", sub,
                  name(kQueue, queue).c_str(), batch, list.c_str(), waits.c_str(), signals.c_str(),
                  name(kFence, fence).c_str());
}

// Logs one batch and does its host work; caller holds g_lock. Returns the
// hashing jobs it wants run once the submission completes.
void on_batch(DeviceData* d, VkQueue queue, VkFence fence, uint32_t batch, const VkCommandBuffer* cbs, uint32_t ncb,
              const std::string& waits, const std::string& signals, const char* call, std::vector<Job>& jobs,
              uint64_t& last_sub)
{
    const uint64_t sub = ++d->submits;
    last_sub = sub;
    emit(call, submit_line(sub, queue, batch, ncb, cbs, waits, signals, fence));
    Pending p;
    p.queue = queue;
    p.fence = fence;
    p.submit = sub;
    std::vector<const CbState*> states;
    bool dispatches = false;
    for (uint32_t i = 0; i < ncb; ++i) {
        auto it = g_cb.find(key_of(cbs[i]));
        if (it == g_cb.end()) continue;
        const CbState& s = it->second;
        states.push_back(&s);
        dispatches |= s.dispatches != 0;
        for (const auto& [buffer, owner] : s.owners)
            if (auto b = g_buf.find(key_of(buffer)); b != g_buf.end()) b->second.owner = owner;
        for (const auto& [image, layout] : s.layouts)
            if (auto im = g_img.find(key_of(image)); im != g_img.end()) im->second.layout = layout;
        for (const HostRange& h : s.uploads) {
            auto m = g_mem.find(key_of(h.memory));
            if (m == g_mem.end()) continue;
            const MemoryInfo& mi = m->second;
            if (mi.mapped && h.offset >= mi.map_offset && h.offset + h.size <= mi.map_offset + mi.map_size)
                emit("Upload", format("sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=%016" PRIx64, sub,
                                      h.label.c_str(), name(kMemory, h.memory).c_str(), uint64_t(h.offset),
                                      uint64_t(h.size),
                                      host_fnv(mi, static_cast<const unsigned char*>(mi.mapped) +
                                                       (h.offset - mi.map_offset), size_t(h.size))));
            else
                emit("Upload", format("sub=%" PRIu64 " at=%s mem=%s+%" PRIu64 " bytes=%" PRIu64 " fnv=unmapped", sub,
                                      h.label.c_str(), name(kMemory, h.memory).c_str(), uint64_t(h.offset),
                                      uint64_t(h.size)));
        }
        p.ranges.insert(p.ranges.end(), s.downloads.begin(), s.downloads.end());
    }
    if (!p.ranges.empty()) d->pending.push_back(std::move(p));
    if (hash_this_submit(sub, dispatches)) {
        std::vector<Target> targets;
        select_targets(d, states, targets);
        for (const Target& t : targets) {
            jobs.push_back(resolve(t));
            jobs.back().target.label = format("%s", t.label.c_str());
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo* s, VkFence fence)
{
    DeviceData* d = dev(queue);
    const VkResult r = d->QueueSubmit(queue, n, s, fence);
    std::vector<Job> jobs;
    uint64_t last = 0;
    uint32_t family = 0;
    {
        std::lock_guard lock(g_lock);
        family = g_queue[key_of(queue)].family;
        for (uint32_t b = 0; b < n; ++b) {
            std::string waits, signals;
            for (uint32_t i = 0; i < s[b].waitSemaphoreCount; ++i)
                waits += format("%s%s@%s", i ? "," : "", name(kSemaphore, s[b].pWaitSemaphores[i]).c_str(),
                                hex(s[b].pWaitDstStageMask[i]).c_str());
            for (uint32_t i = 0; i < s[b].signalSemaphoreCount; ++i)
                signals += (i ? "," : "") + name(kSemaphore, s[b].pSignalSemaphores[i]);
            if (auto* tl = find_in_chain<VkTimelineSemaphoreSubmitInfo>(
                    s[b].pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)) {
                for (uint32_t i = 0; i < tl->waitSemaphoreValueCount; ++i)
                    waits += format(":%" PRIu64, tl->pWaitSemaphoreValues[i]);
                for (uint32_t i = 0; i < tl->signalSemaphoreValueCount; ++i)
                    signals += format(":%" PRIu64, tl->pSignalSemaphoreValues[i]);
            }
            on_batch(d, queue, fence, b, s[b].pCommandBuffers, s[b].commandBufferCount,
                     waits, signals, "QueueSubmit", jobs, last);
        }
        if (!n)
            emit("QueueSubmit",
                 format("q=%s batches=0 fence=%s", name(kQueue, queue).c_str(), name(kFence, fence).c_str()));
        emit("QueueSubmitResult", format("q=%s last=%" PRIu64 " rc=%d", name(kQueue, queue).c_str(), last, int(r)));
        std::fflush(g_out);
    }
    if (r == VK_SUCCESS && !jobs.empty() && d->QueueWaitIdle(queue) == VK_SUCCESS)
        run_jobs(d, queue, family, last, jobs);
    return r;
}

VkResult submit2(PFN_vkQueueSubmit2 next, const char* call, VkQueue queue, uint32_t n, const VkSubmitInfo2* s,
                 VkFence fence)
{
    DeviceData* d = dev(queue);
    const VkResult r = next(queue, n, s, fence);
    std::vector<Job> jobs;
    uint64_t last = 0;
    uint32_t family = 0;
    {
        std::lock_guard lock(g_lock);
        family = g_queue[key_of(queue)].family;
        for (uint32_t b = 0; b < n; ++b) {
            std::string waits, signals;
            for (uint32_t i = 0; i < s[b].waitSemaphoreInfoCount; ++i) {
                const auto& w = s[b].pWaitSemaphoreInfos[i];
                waits += format("%s%s@%s:%" PRIu64, i ? "," : "", name(kSemaphore, w.semaphore).c_str(),
                                hex(w.stageMask).c_str(), w.value);
            }
            for (uint32_t i = 0; i < s[b].signalSemaphoreInfoCount; ++i) {
                const auto& g = s[b].pSignalSemaphoreInfos[i];
                signals += format("%s%s@%s:%" PRIu64, i ? "," : "", name(kSemaphore, g.semaphore).c_str(),
                                  hex(g.stageMask).c_str(), g.value);
            }
            std::vector<VkCommandBuffer> cbs(s[b].commandBufferInfoCount);
            for (uint32_t i = 0; i < s[b].commandBufferInfoCount; ++i)
                cbs[i] = s[b].pCommandBufferInfos[i].commandBuffer;
            on_batch(d, queue, fence, b, cbs.data(), uint32_t(cbs.size()), waits, signals,
                     call, jobs, last);
        }
        emit("QueueSubmitResult", format("q=%s last=%" PRIu64 " rc=%d", name(kQueue, queue).c_str(), last, int(r)));
        std::fflush(g_out);
    }
    if (r == VK_SUCCESS && !jobs.empty() && d->QueueWaitIdle(queue) == VK_SUCCESS)
        run_jobs(d, queue, family, last, jobs);
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2* s, VkFence fence)
{
    return submit2(dev(queue)->QueueSubmit2, "QueueSubmit2", queue, n, s, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2KHR(VkQueue queue, uint32_t n, const VkSubmitInfo2* s, VkFence fence)
{
    return submit2(dev(queue)->QueueSubmit2KHR, "QueueSubmit2", queue, n, s, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queue)
{
    DeviceData* d = dev(queue);
    const VkResult r = d->QueueWaitIdle(queue);
    std::lock_guard lock(g_lock);
    emit("QueueWaitIdle", format("q=%s rc=%d", name(kQueue, queue).c_str(), int(r)));
    if (r == VK_SUCCESS) complete(d, [&](const Pending& p) { return p.queue == queue; }, "queue");
    std::fflush(g_out);
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device)
{
    DeviceData* d = dev(device);
    const VkResult r = d->DeviceWaitIdle(device);
    std::lock_guard lock(g_lock);
    emit("DeviceWaitIdle", format("dev=%s rc=%d", name(kDevice, device).c_str(), int(r)));
    if (r == VK_SUCCESS) complete(d, [](const Pending&) { return true; }, "device");
    std::fflush(g_out);
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pi)
{
    DeviceData* d = dev(queue);
    {
        std::lock_guard lock(g_lock);
        std::string list, waits;
        for (uint32_t i = 0; i < pi->swapchainCount; ++i)
            list += format("%s%s:%u", i ? "," : "", name(kSwapchain, pi->pSwapchains[i]).c_str(), pi->pImageIndices[i]);
        for (uint32_t i = 0; i < pi->waitSemaphoreCount; ++i)
            waits += (i ? "," : "") + name(kSemaphore, pi->pWaitSemaphores[i]);
        emit("QueuePresentKHR", format("q=%s images=[%s] wait=[%s]", name(kQueue, queue).c_str(), list.c_str(),
                                       waits.c_str()));
    }
    const VkResult r = d->QueuePresentKHR(queue, pi);
    std::lock_guard lock(g_lock);
    emit("QueuePresentResult", format("q=%s rc=%d", name(kQueue, queue).c_str(), int(r)));
    std::fflush(g_out);
    return r;
}

// ---------------------------------------------------------------------------
// Lookup

struct Entry {
    const char* name;
    PFN_vkVoidFunction fn;
};

#define ENTRY(fn) {"vk" #fn, reinterpret_cast<PFN_vkVoidFunction>(fn)},
const Entry kInstanceEntries[] = {
    ENTRY(GetInstanceProcAddr) ENTRY(CreateInstance) ENTRY(DestroyInstance) ENTRY(EnumeratePhysicalDevices)
    ENTRY(CreateDevice)
};
const Entry kDeviceEntries[] = {DEVICE_HOOKS(ENTRY)};
#undef ENTRY

// Whether DEVICE's next layer provides NAME: an intercept for a function the
// device lacks must not be handed out.
PFN_vkVoidFunction device_entry(DeviceData* d, const char* name)
{
    for (const Entry& e : kDeviceEntries)
        if (!std::strcmp(e.name, name)) {
            if (d && !d->GetDeviceProcAddr(d->device, name)) return nullptr;
            return e.fn;
        }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name)
{
    DeviceData* d = dev(device);
    if (!d) return nullptr;
    if (PFN_vkVoidFunction f = device_entry(d, name)) return f;
    return d->GetDeviceProcAddr(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char* name)
{
    for (const Entry& e : kInstanceEntries)
        if (!std::strcmp(e.name, name)) return e.fn;
    if (!instance) return nullptr;
    InstanceData* data;
    {
        std::lock_guard lock(g_lock);
        data = instance_of(instance);
    }
    if (!data) return nullptr;
    // Device functions through the instance: ours, when the driver has them at all.
    if (PFN_vkVoidFunction f = device_entry(nullptr, name)) return data->gipa(instance, name) ? f : nullptr;
    return data->gipa(instance, name);
}

} // namespace

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v)
{
    if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
