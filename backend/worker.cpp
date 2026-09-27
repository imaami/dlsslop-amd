// Native Linux worker for DLSS5VKLayer's shared-memory transport.
// SPDX-License-Identifier: MIT
#include "codec.h"
#include "vendor/LmxxfProductionOptions.h"
#include "native_kernels.h"
#include "codec_gpu.h"
#include "tuning.h"
#include "temporal_gpu.h"
#include "control_selftest.h"
#include "trace.h"
#include "shm_protocol.h"
#include "vulkan_network.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <linux/futex.h>
#include <pwd.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
namespace selftest = dlsslop::control_selftest;
volatile sig_atomic_t stopping;
void stop_handler(int) { stopping = 1; }

std::string home_directory()
{
    const char* home = std::getenv("HOME");
    if (!home || !*home) {
        const struct passwd* user = getpwuid(getuid());
        home = user ? user->pw_dir : nullptr;
    }
    return home ? home : "";
}

// XDG_BASE (nonempty) or ~/FALLBACK, then NAME; empty without a home.
std::string xdg_path(const char* base, const char* fallback, const char* name)
{
    if (const char* root = std::getenv(base); root && *root) return (std::filesystem::path(root) / name).string();
    const std::string home = home_directory();
    return home.empty() ? home : (std::filesystem::path(home) / fallback / name).string();
}

std::string default_assets() { return xdg_path("XDG_DATA_HOME", ".local/share", "dlsslop-amd/model"); }
std::string default_vulkan_model() { return xdg_path("XDG_DATA_HOME", ".local/share", "dlsslop-amd/dlssnr.bin"); }
std::string default_config() { return xdg_path("XDG_CONFIG_HOME", ".config", "dlsslop-amd/dlsslopd.conf"); }

std::string executable_path()
{
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::string() : executable.string();
}

// The Vulkan network's SPIR-V: the installed copy, or a source build's beside
// the executable.
std::string vulkan_shaders()
{
    const std::filesystem::path executable = executable_path();
    const auto installed = executable.parent_path().parent_path() / "share/dlsslop-amd/vulkan";
    const auto development = executable.parent_path() / "vulkan-nr/network";
    std::error_code error;
    return (std::filesystem::is_directory(development, error) && !std::filesystem::is_directory(installed, error)
                ? development : installed).string();
}

std::string default_modules()
{
    if (const char* path = std::getenv("DLSSLOP_MODULES"); path && *path) return path;
    const std::filesystem::path executable = executable_path();
    if (executable.empty()) return {};
    const auto prefix = executable.parent_path().parent_path();
    std::error_code error;
    const auto installed = prefix / "share/dlsslop-amd/HIP/gfx1201";
    if (std::filesystem::is_directory(installed, error)) return installed.string();
    // A worker run directly from the source tree's build directory uses the
    // same modules as the kernel build script, without an installed launcher.
    const auto development = prefix / "assets/HIP/gfx1201";
    if (std::filesystem::is_directory(development, error)) return development.string();
    return installed.string();
}

struct Options {
    std::string assets, modules, shm = ShmNativeChannelPath();
    std::string input, output, trace_dir;
    std::string backend = "auto", vulkan_model;
    unsigned width = 0, height = 0, self_test_runs = 10;
    unsigned idle_exit = 0;
    // Unset, a serving worker keeps the channel's live values across restarts.
    std::optional<unsigned> tier, passes;
    int device = -1;
    bool diagnose = false, test_identity = false, once = false, self_test = false;
    bool cpu_compose = false, cpu_codec = false, performance = false;
};

Options default_options()
{
    Options o;
    o.assets = default_assets();
    o.modules = default_modules();
    o.vulkan_model = default_vulkan_model();
    return o;
}

struct ProcessingSettings {
    dlsslop::NativeTuning tuning;
    float color_preserve = 0;
    bool fp16 = false;
    bool precision16 = true;
    bool motion = false;
    unsigned motion_quality = kMVecBalanced;
    unsigned motion_grid = kMVecPixels4;
};

dlsslop::NativeTuning read_tuning(const ShmHeader* h)
{
    return {BitsToFloat(h->intensityBits.load()), BitsToFloat(h->localToneBits.load()),
            BitsToFloat(h->localStructureBits.load()), BitsToFloat(h->sharpnessBits.load())};
}

[[noreturn]] void system_error(const char* action)
{
    throw std::runtime_error(std::string(action) + ": " + std::strerror(errno));
}

unsigned number(const char* text, const char* name)
{
    char* end = nullptr;
    errno = 0;
    unsigned long n = std::strtoul(text, &end, 10);
    if (errno || !*text || *end || *text == '-' || n > 100000)
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<unsigned>(n);
}

bool flag(const char* value)
{
    if (!std::strcmp(value, "true")) return true;
    if (!std::strcmp(value, "false")) return false;
    throw std::runtime_error("expected true or false");
}

unsigned tier(const char* value)
{
    const unsigned t = number(value, "tier");
    if (!ShmNativeTier(t)) throw std::runtime_error("tier must be 720, 900, or 1080");
    return t;
}

// Every option once: its getopt spelling, whether the config file may set it
// (and whether that value is a path), and where a value lands. A flag gets
// "true" from the command line, or true or false from the config file.
struct Spec {
    const char* name;
    char letter;
    enum Kind : uint8_t { kFlag, kValue, kPath } kind;
    bool config;
    void (*apply)(Options&, const char*);
};

const Spec kSpecs[] = {
    {"config", 'f', Spec::kPath, false, nullptr}, // Read before the others apply.
    {"backend", 'b', Spec::kValue, true, [](Options& o, const char* v) {
        if (std::strcmp(v, "auto") && std::strcmp(v, "vulkan") && std::strcmp(v, "hip"))
            throw std::runtime_error("backend must be auto, vulkan or hip");
        o.backend = v;
    }},
    {"vulkan-model", 'M', Spec::kPath, true, [](Options& o, const char* v) { o.vulkan_model = v; }},
    {"assets", 'a', Spec::kPath, true, [](Options& o, const char* v) { o.assets = v; }},
    {"modules", 'm', Spec::kPath, true, [](Options& o, const char* v) { o.modules = v; }},
    // The channel pairs the worker with its launcher and socket unit.
    {"shm", 's', Spec::kPath, false, [](Options& o, const char* v) { o.shm = v; }},
    {"tier", 't', Spec::kValue, true, [](Options& o, const char* v) { o.tier = tier(v); }},
    {"passes", 'P', Spec::kValue, true, [](Options& o, const char* v) {
        o.passes = number(v, "passes");
        if (!*o.passes || *o.passes > kMaxPasses)
            throw std::runtime_error("--passes must be 1.." + std::to_string(kMaxPasses));
    }},
    {"device", 'd', Spec::kValue, true, [](Options& o, const char* v) { o.device = static_cast<int>(number(v, "device")); }},
    {"diagnose", 'D', Spec::kFlag, false, [](Options& o, const char*) { o.diagnose = true; }},
    {"self-test", 'S', Spec::kFlag, false, [](Options& o, const char*) { o.self_test = true; }},
    {"self-test-runs", 'r', Spec::kValue, false, [](Options& o, const char* v) { o.self_test_runs = number(v, "self-test runs"); }},
    {"input", 'i', Spec::kValue, false, [](Options& o, const char* v) { o.input = v; }},
    {"output", 'o', Spec::kValue, false, [](Options& o, const char* v) { o.output = v; }},
    {"width", 'W', Spec::kValue, false, [](Options& o, const char* v) { o.width = number(v, "width"); }},
    {"height", 'H', Spec::kValue, false, [](Options& o, const char* v) { o.height = number(v, "height"); }},
    {"cpu-compose", 'c', Spec::kFlag, false, [](Options& o, const char*) { o.cpu_compose = o.cpu_codec = true; }},
    {"cpu-codec", 'C', Spec::kFlag, false, [](Options& o, const char*) { o.cpu_codec = true; }},
    {"performance", 'p', Spec::kFlag, true, [](Options& o, const char* v) { o.performance = flag(v); }},
    {"once", '1', Spec::kFlag, false, [](Options& o, const char*) { o.once = true; }},
    {"idle-exit", 'x', Spec::kValue, true, [](Options& o, const char* v) { o.idle_exit = number(v, "idle-exit seconds"); }},
    {"trace-dir", 'R', Spec::kValue, false, [](Options& o, const char* v) {
        if (!*v) throw std::runtime_error("--trace-dir requires a nonempty directory");
        o.trace_dir = v;
    }},
    {"test-identity", 'T', Spec::kFlag, false, [](Options& o, const char*) { o.test_identity = true; }},
    {"help", 'h', Spec::kFlag, false, nullptr},
};

void usage(FILE* out)
{
    const Options defaults = default_options();
    const std::string config = default_config();
    std::string settable;
    for (const Spec& spec : kSpecs)
        if (spec.config) settable += (settable.empty() ? "" : ", ") + std::string(spec.name);
    std::fprintf(out,
        "Usage: dlsslopd [OPTIONS]\n"
        "  -f, --config FILE       Settings file of NAME = VALUE lines\n"
        "                          Default: %s\n"
        "                          Uses nonempty XDG_CONFIG_HOME/dlsslop-amd/dlsslopd.conf,\n"
        "                          otherwise ~/.config/dlsslop-amd/dlsslopd.conf;\n"
        "                          a missing default file is skipped. NAME is one of\n"
        "                          %s\n"
        "                          (performance = true or false); # starts a\n"
        "                          comment line; paths are absolute or start with\n"
        "                          ~/. Settings replace the defaults below and\n"
        "                          options override them\n"
        "  -b, --backend NAME      Where the network runs: vulkan, hip or auto\n"
        "                          Default: %s; auto takes Vulkan when its model is\n"
        "                          installed and a device supports it, else HIP.\n"
        "                          Vulkan evaluates NVIDIA's own intensity, local\n"
        "                          tone and local structure controls, sizes itself\n"
        "                          to each frame, ignores sharpness and color\n"
        "                          preservation and cannot trace. Its SPIR-V is in\n"
        "                          %s\n"
        "  -M, --vulkan-model FILE The Vulkan network's model (dlssnr.bin)\n"
        "                          Default: %s\n"
        "                          Uses nonempty XDG_DATA_HOME/dlsslop-amd/dlssnr.bin,\n"
        "                          otherwise ~/.local/share/dlsslop-amd/dlssnr.bin;\n"
        "                          dlsslop-setup --dll extracts it from your own\n"
        "                          nvngx_dlssnr.dll 310.8.0\n"
        "  -a, --assets DIR        HIP model weights (.f16/.f32)\n"
        "                          Default: %s\n"
        "                          Uses nonempty XDG_DATA_HOME/dlsslop-amd/model,\n"
        "                          otherwise ~/.local/share/dlsslop-amd/model\n"
        "  -m, --modules DIR       Native gfx1201 HIP modules (.hsaco)\n"
        "                          Default: %s\n"
        "                          Uses nonempty DLSSLOP_MODULES; otherwise\n"
        "                          <executable prefix>/share/dlsslop-amd/HIP/gfx1201\n"
        "                          (source build fallback: ../assets/HIP/gfx1201)\n"
        "  -s, --shm FILE          Layer transport in a private (0700) directory\n"
        "                          Default: %s\n"
        "                          From nonempty DLSSNR_SHM, otherwise %s\n"
        "  -t, --tier HEIGHT       Neural work raster: 720, 900, or 1080\n"
        "                          Default: %u on a new channel; game/display\n"
        "                          resolution unchanged. Unset here and in FILE, a\n"
        "                          starting worker keeps the channel's live tier,\n"
        "                          which dlsslopctl --tier changes while it runs\n"
        "  -P, --passes N          Chained neural evaluations per frame (1..%u)\n"
        "                          Default: %u on a new channel; each pass consumes\n"
        "                          the previous output. Unset here and in FILE, a\n"
        "                          starting worker keeps the channel's live count\n"
        "  -d, --device INDEX      HIP device, or with Vulkan physical device, index\n"
        "                          Default: auto, the first device that can run\n"
        "                          the network (HIP: gfx1201)\n"
        "  -D, --diagnose          Enumerate HIP devices, report the selection, exit\n"
        "                          Default: off; exit status 1 if none is usable\n"
        "  -S, --self-test         Real model test on a deterministic gradient\n"
        "                          Also checks tuning, motion history and FP16 codec\n"
        "                          Default: off\n"
        "  -r, --self-test-runs N  Identical-input runs, including baseline (2..1000)\n"
        "                          Default: %u; requires --self-test\n"
        "  -i, --input FILE        Offline tightly packed RGBA8 input\n"
        "                          Default: unset; serve shared-memory requests\n"
        "  -o, --output FILE       Offline RGBA8 output; self-test PPM output\n"
        "                          Default: unset; required with --input;\n"
        "                          --self-test writes no image unless specified\n"
        "  -W, --width PIXELS      Offline image width (1..7680)\n"
        "                          Default: %u (unset); required with --input\n"
        "  -H, --height PIXELS     Offline image height (1..4320)\n"
        "                          Default: %u (unset); required with --input\n"
        "  -c, --cpu-compose       Use CPU composition and codec (layer bypass)\n"
        "                          Default: off; Vulkan layer composes the result\n"
        "  -C, --cpu-codec         Slow CPU codec for numerical comparison\n"
        "                          Default: off; use the GPU codec\n"
        "  -p, --performance       Skip blocks 42,43,46, matching upstream preset\n"
        "                          Default: off; evaluate all 71 blocks\n"
        "  -1, --once              Answer one shared-memory request and exit,\n"
        "                          with status 1 when that request failed\n"
        "                          Default: off; run until stopped\n"
        "  -x, --idle-exit SECONDS Stop serving after SECONDS without a request\n"
        "                          Default: %u (never)\n"
        "  -R, --trace-dir DIR     Opt-in real-frame RGB float32 diagnostics\n"
        "                          Default: disabled; no readbacks or file checks\n"
        "                          DIR is created private (0700) or must be so\n"
        "                          To trace one frame, write a unique TOKEN (1-64\n"
        "                          of A-Z a-z 0-9 _ -, not 'request') to\n"
        "                          DIR/TOKEN.tmp, then ln it to DIR/request;\n"
        "                          DIR/TOKEN.done marks DIR/TOKEN/summary.json\n"
        "                          complete. PFM files hold each pass's input and\n"
        "                          raw output, plus -tuned/-color stages when\n"
        "                          active; serving real inference only\n"
        "  -T, --test-identity     DIAGNOSTIC ONLY: copy frames without inference\n"
        "                          Default: off; evaluate the real HIP network\n"
        "  -h, --help              Show this help and exit (default: off)\n",
        config.empty() ? "unset; no home directory" : config.c_str(), settable.c_str(),
        defaults.backend.c_str(), vulkan_shaders().c_str(),
        defaults.vulkan_model.empty() ? "unset; no home directory" : defaults.vulkan_model.c_str(),
        defaults.assets.empty() ? "unset; required for inference" : defaults.assets.c_str(),
        defaults.modules.empty() ? "unset; required for inference" : defaults.modules.c_str(),
        defaults.shm.c_str(), ShmNativeDefaultPath().c_str(), kNativeDefaultTier,
        kMaxPasses, kNativeDefaultPasses, defaults.self_test_runs,
        defaults.width, defaults.height, defaults.idle_exit);
}

std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t");
    return first == std::string::npos ? std::string() : text.substr(first, text.find_last_not_of(" \t") - first + 1);
}

// A config file's path is absolute or starts with ~/, never relative to
// wherever the worker was started.
std::string config_path(const std::string& value)
{
    if (value[0] == '/') return value;
    const std::string home = value.compare(0, 2, "~/") ? std::string() : home_directory();
    if (home.empty()) throw std::runtime_error("a path must be absolute or start with ~/");
    return home + value.substr(1);
}

// One NAME = VALUE line, NAME a config-settable long option.
void apply_setting(Options& o, const std::string& text)
{
    const auto equals = text.find('=');
    if (equals == std::string::npos) throw std::runtime_error("expected NAME = VALUE");
    const std::string name = trim(text.substr(0, equals));
    const auto spec = std::find_if(std::begin(kSpecs), std::end(kSpecs),
                                   [&name](const Spec& s) { return s.config && name == s.name; });
    if (spec == std::end(kSpecs)) throw std::runtime_error("'" + name + "' is not a config setting");
    const std::string value = trim(text.substr(equals + 1));
    spec->apply(o, (spec->kind == Spec::kPath ? config_path(value) : value).c_str());
}

// Settings, one per line; # starts a comment line. A default file that is
// missing is skipped; one that cannot be inspected or read is an error.
void read_config(Options& o, const std::string& path, bool given)
{
    std::error_code error;
    if (!given && (path.empty() || (!std::filesystem::exists(path, error) && !error))) return;
    std::ifstream file(path);
    std::string line;
    for (unsigned line_number = 1; std::getline(file, line); ++line_number) {
        const std::string text = trim(line);
        if (text.empty() || text[0] == '#') continue;
        try {
            apply_setting(o, text);
        } catch (const std::exception& e) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) + ": " + e.what());
        }
    }
    // Only a complete read ends at EOF: opening fails for a missing file, and
    // reading fails for a directory or on a disk error.
    if (!file.eof()) throw std::runtime_error("cannot read config file " + path + ": " + std::strerror(errno));
}

Options parse(int argc, char** argv)
{
    std::string letters;
    std::vector<option> options;
    for (const Spec& spec : kSpecs) {
        letters += spec.letter;
        if (spec.kind != Spec::kFlag) letters += ':';
        options.push_back({spec.name, spec.kind == Spec::kFlag ? no_argument : required_argument, nullptr, spec.letter});
    }
    options.push_back({});
    // Options override the config file, which one of them may name.
    std::vector<std::pair<const Spec*, const char*>> given;
    std::string config = default_config();
    bool config_given = false;
    for (int c; (c = getopt_long(argc, argv, letters.c_str(), options.data(), nullptr)) != -1;) {
        const auto spec = std::find_if(std::begin(kSpecs), std::end(kSpecs), [c](const Spec& s) { return s.letter == c; });
        if (spec == std::end(kSpecs)) {
            usage(stderr);
            throw std::runtime_error("invalid arguments");
        }
        if (c == 'h') {
            usage(stdout);
            std::exit(0);
        }
        if (c == 'f') {
            config = optarg;
            config_given = true;
            continue;
        }
        given.emplace_back(spec, optarg ? optarg : "true");
    }
    if (optind != argc) throw std::runtime_error("unexpected positional argument");
    Options o = default_options();
    read_config(o, config, config_given);
    for (const auto& [spec, value] : given) spec->apply(o, value);
    const auto on_command_line = [&given](char letter) {
        return std::any_of(given.begin(), given.end(), [letter](const auto& g) { return g.first->letter == letter; });
    };
    if (o.self_test_runs < 2 || o.self_test_runs > 1000)
        throw std::runtime_error("--self-test-runs must be 2..1000");
    if (on_command_line('r') && !o.self_test)
        throw std::runtime_error("--self-test-runs requires --self-test");
    if (!o.diagnose && !o.test_identity && (o.assets.empty() || o.modules.empty()))
        throw std::runtime_error("--assets and --modules are required for inference");
    if (o.self_test && (o.test_identity || !o.input.empty()))
        throw std::runtime_error("--self-test requires the real network and generates its own input");
    if (!o.self_test && (o.input.empty() != o.output.empty()))
        throw std::runtime_error("--input and --output must be supplied together");
    if (!o.input.empty() && (!o.width || !o.height || o.width > kMaxW || o.height > kMaxH))
        throw std::runtime_error("offline dimensions must be 1..7680 by 1..4320");
    if (!o.trace_dir.empty() && (o.self_test || !o.input.empty() || o.test_identity || o.diagnose))
        throw std::runtime_error("--trace-dir requires serving real shared-memory inference");
    // A configured idle exit is for serving; only an explicit one is an error.
    if (on_command_line('x') && o.idle_exit && (o.self_test || !o.input.empty() || o.diagnose))
        throw std::runtime_error("--idle-exit requires serving shared-memory requests");
    return o;
}

int select_device(int requested)
{
    hip_probe::Api api;
    api.Check(api.hipInit(0), "hipInit (check /dev/kfd permissions and ROCm userspace)");
    int count = 0, version = 0, selected = -1;
    api.Check(api.hipRuntimeGetVersion(&version), "HIP runtime version");
    api.Check(api.hipGetDeviceCount(&count), "HIP device count");
    std::fprintf(stderr, "HIP runtime=%d, visible devices=%d\n", version, count);
    for (int i = 0; i < count; ++i) {
        auto p = api.Properties(i);
        std::fprintf(stderr, "  device %d: %s; arch=%s; PCI=%04x:%02x:%02x; VRAM=%.0f MiB\n",
                     i, p.name, p.gcnArchName, p.pciDomainID, p.pciBusID, p.pciDeviceID,
                     p.totalGlobalMem / 1048576.0);
        const bool gfx1201 = !std::strncmp(p.gcnArchName, "gfx1201", 7) &&
                            (!p.gcnArchName[7] || p.gcnArchName[7] == ':');
        if (gfx1201 && ((requested < 0 && selected < 0) || requested == i)) selected = i;
    }
    if (selected < 0) throw std::runtime_error(requested < 0
        ? "no gfx1201 device found (RX 9070/9070 XT required); check HIP_VISIBLE_DEVICES"
        : "selected device is unavailable or is not gfx1201");
    return selected;
}

class Mapping {
    dlsslop::Descriptor file_;
public:
    ShmHeader* h = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;

    explicit Mapping(const std::string& name)
    {
        const auto parent = std::filesystem::path(name).parent_path();
        if (parent.empty()) throw std::runtime_error("--shm requires a path inside a private directory");
        dlsslop::private_directory(parent, "shared-memory");
        file_.fd = open(name.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (file_.fd < 0) system_error("open shared-memory file");
        if (flock(file_.fd, LOCK_EX | LOCK_NB))
            throw std::runtime_error("another worker owns this shared-memory file");
        struct stat st{};
        if (fstat(file_.fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid())
            throw std::runtime_error("shared-memory file must be regular and owned by the current user");
        if (fchmod(file_.fd, 0600)) system_error("make shared-memory file private");
        if (ftruncate(file_.fd, static_cast<off_t>(ShmTotalBytes()))) system_error("size shared-memory file");
        void* const mapping = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, file_.fd, 0);
        if (mapping == MAP_FAILED) system_error("map shared-memory file");
        h = static_cast<ShmHeader*>(mapping);
        if (h->magic.load() != kShmMagic || h->version.load() != kShmVersion) ShmInitNativeDefaults(h);
        input = static_cast<uint8_t*>(mapping) + kHeaderBytes;
        output = input + kMaxFrame;
        h->quit.store(0);
        h->modelUp.store(0);
        h->seq_ok.store(0);
        // Answer a request left by a previous worker as failed, so the layer
        // presents its own frame; requests made from here on are served.
        h->seq_resp.store(h->seq_req.load(std::memory_order_acquire), std::memory_order_release);
        syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAKE, 1, nullptr, nullptr, 0);
        h->helperState.store(kHelperStarting);
    }
    Mapping(const Mapping&) = delete;
    ~Mapping()
    {
        h->modelUp.store(0);
        if (h->helperState.load() != kHelperModelFailed) h->helperState.store(kHelperStopped);
        munmap(h, ShmTotalBytes());
    }
    void reason(const std::string& text)
    {
        ShmStoreString(h->helperReasonSeq, h->helperReason, kReasonBytes, text.c_str());
    }
};

// The network as the serving loop, the offline mode and the self-test see it.
class Backend {
public:
    virtual ~Backend() = default;
    // A request's frames: host memory, or an imported device-local pair of a
    // nonzero generation.
    struct Frames {
        const uint8_t* proxy = nullptr;
        uint8_t* answer = nullptr;
        uint32_t generation = 0;
    };
    virtual const char* name() const = 0;
    virtual unsigned tier() const = 0;
    // What the network runs at, for the log.
    virtual std::string processing() const = 0;
    // Takes a new tier without a new backend; false when it needs one.
    virtual bool retier(unsigned) { return false; }
    virtual void prepare() = 0;
    // False when a request of this shape needs a build first (reshape): seconds
    // of work the caller reports as a start, not a slow frame.
    virtual bool fits(unsigned, unsigned, unsigned, const ProcessingSettings&) const { return true; }
    virtual void reshape(unsigned, unsigned, unsigned, const ProcessingSettings&) {}
    // Serving only: DMA the channel's frame slots directly.
    virtual void pin(uint8_t*, uint8_t*, size_t) {}
    // Serving, between frames: imports an offered proxy/answer pair. The backend
    // owns each descriptor it imported; fds keeps the rest to close.
    virtual bool import(const ShmTransportOffer&, dlsslop::Descriptor (&)[2]) = 0;
    // The imported pair of a generation that holds bytes; generation 0 when none does.
    virtual Frames frames(uint32_t generation, size_t bytes) const = 0;
    // Diagnostics: copies host or device memory, such as an imported frame, to the host.
    virtual void read_back(void* host, const void* source, size_t bytes) = 0;
    // w * h RGBA8 frames, or RGBA16F with settings.fp16.
    virtual void infer(const Frames& io, unsigned w, unsigned h, unsigned passes,
                       const ProcessingSettings& settings = {}, dlsslop::FrameTrace* trace = nullptr) = 0;
    float upload_ms = 0, inference_ms = 0, readback_ms = 0;
};

class Engine : public Backend {
    Options options_;
    unsigned tier_;
    std::unique_ptr<hip_reference::Network> network_;
    std::optional<dlsslop::NativeKernels> kernels_;
    std::optional<dlsslop::GpuCodec> gpu_codec_;
    std::optional<dlsslop::GpuTemporal> temporal_;
    void* device_input_ = nullptr; // The encoded frame, unchanged until the next one.
    void* device_feedback_ = nullptr; // Later passes' input, allocated for multi-pass.
    void* device_output_ = nullptr;
    void* device_scratch_ = nullptr; // Tuning or colour: the other stage output.
    void* answer_ = nullptr; // The latest frame's final network answer.
    std::vector<float> encoded_, neural_, feedback_;
    ProcessingSettings previous_settings_;
    // Stream events: frame start, uploaded, evaluated, answered. Timing never
    // stalls the stream; the intervals are read once the answer is complete.
    hip_probe::Handle marks_[4]{};
    // Device-local frames the layer exported (ShmTransportOffer), one per
    // producer generation, the oldest replaced first.
    struct Imported {
        uint32_t generation = 0;
        size_t bytes = 0;
        hip_probe::Handle memory[2]{};
        void* frame[2]{}; // proxy, answer
    };
    std::array<Imported, 4> imported_{};
    unsigned next_import_ = 0;

    void release(Imported& slot)
    {
        auto& api = network_->Runtime();
        for (unsigned i = 0; i < 2; ++i) {
            if (slot.frame[i]) api.hipFree(slot.frame[i]);
            if (slot.memory[i]) api.hipDestroyExternalMemory(slot.memory[i]);
        }
        slot = {};
    }

    void mark(unsigned i)
    {
        auto& api = network_->Runtime();
        api.Check(api.hipEventRecord(marks_[i], network_->Stream()), "record timing event");
    }
public:
    Engine(Options o, unsigned tier) : options_(std::move(o)), tier_(tier) {}
    const char* name() const override { return "HIP"; }
    unsigned tier() const override { return tier_; }
    std::string processing() const override
    {
        const auto raster = dlsslop::geometry(1, 1, tier_);
        return "processing=" + std::to_string(raster.width) + "x" + std::to_string(raster.height);
    }
    // The members go next, in reverse order: the helpers before the network whose runtime they use.
    ~Engine()
    {
        if (!network_) return;
        auto& api = network_->Runtime();
        api.hipStreamSynchronize(network_->Stream());
        for (auto& slot : imported_) release(slot);
        for (auto event : marks_) api.hipEventDestroy(event);
        for (void* buffer : {device_input_, device_feedback_, device_output_, device_scratch_})
            if (buffer) api.hipFree(buffer);
    }
    void prepare() override
    {
        if (options_.test_identity) return;
        const auto raster = dlsslop::geometry(1, 1, tier_);
        auto opt = LmxxfProductionOptions(raster.width, raster.height, options_.modules, options_.assets);
        opt.device = static_cast<unsigned>(options_.device);
        if (!options_.performance) opt.skip_blocks.clear();
        // Upstream's shipped HIP configurations (scripts/hip-*-flags.txt) add
        // these bit-exact byte residual and fragment paths to the snapshot.
        opt.mh_feature_byte = opt.mh_proj_diag_fb = opt.mh_byte_stream = opt.decoder_byte = opt.mh_ffn_frag256 = true;
        // Production launches nothing from the WMMA, tiled, wave and fused-C32
        // modules these select; do not load them.
        opt.wmma = opt.tiled = opt.wave = opt.fused_c32 = false;
        network_ = std::make_unique<hip_reference::Network>(opt);
        network_->SetNoise({}); // Fast prefix uses procedural noise, not noise.f32.
        auto& api = network_->Runtime();
        for (auto& event : marks_) api.Check(api.hipEventCreate(&event), "create timing event");
        // Tuning, colour and motion use the module's kernels with the CPU codec too.
        kernels_.emplace(api, network_->Stream(), options_.modules + "/linux_native.hsaco");
        if (!options_.cpu_codec) gpu_codec_.emplace(*kernels_);
        temporal_.emplace(*kernels_);
        const size_t pixels = size_t(raster.width) * raster.height;
        api.Check(api.hipMalloc(&device_input_, pixels * 16), "allocate network input");
        api.Check(api.hipMalloc(&device_output_, pixels * 12), "allocate network output");
        // The host copy of the answer serves the CPU codec and the self-test's checks.
        if (!gpu_codec_ || options_.self_test) neural_.resize(pixels * 3);
        // Warm once before announcing readiness: upstream allocates weights and
        // scratch lazily, whatever the input; warmup has no temporal history.
        api.Check(api.hipMemsetAsync(device_input_, 0, pixels * 16, network_->Stream()), "warm input");
        network_->Enqueue(device_input_, nullptr, device_output_, 0);
        network_->Synchronize();
        network_->PrintMemory();
        if (options_.self_test) { // The kernels on synthetic inputs, independent of model weights.
            selftest::check_tuning(*kernels_);
            selftest::check_temporal(*kernels_);
            selftest::check_codec(*kernels_);
        }
    }
    // See GpuCodec::pin.
    void pin(uint8_t* input, uint8_t* output, size_t bytes) override
    {
        if (gpu_codec_) gpu_codec_->pin(input, output, bytes);
    }
    bool import(const ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2]) override
    {
        if (!gpu_codec_ || !offer.generation) return false;
        auto& api = network_->Runtime();
        Imported next{offer.generation, size_t(std::min(offer.size[0], offer.size[1]))};
        for (unsigned i = 0; i < 2; ++i) {
            hip_probe::MemoryDesc memory{};
            memory.type = 1; // hipExternalMemoryHandleTypeOpaqueFd
            memory.handle.fd = fds[i].fd;
            memory.size = offer.allocation[i];
            hip_probe::BufferDesc buffer{};
            buffer.size = offer.allocation[i];
            if (api.hipImportExternalMemory(&next.memory[i], &memory)) {
                release(next);
                return false;
            }
            fds[i].fd = -1;
            if (api.hipExternalMemoryGetMappedBuffer(&next.frame[i], next.memory[i], &buffer)) {
                release(next);
                return false;
            }
        }
        Imported* slot = nullptr;
        for (auto& imported : imported_)
            if (imported.generation == offer.generation) slot = &imported;
        if (!slot) slot = &imported_[next_import_++ % imported_.size()];
        release(*slot);
        *slot = next;
        return true;
    }
    Frames frames(uint32_t generation, size_t bytes) const override
    {
        for (const auto& imported : imported_)
            if (imported.generation == generation && imported.bytes >= bytes)
                return {static_cast<const uint8_t*>(imported.frame[0]), static_cast<uint8_t*>(imported.frame[1]), generation};
        return {};
    }
    void read_back(void* host, const void* source, size_t bytes) override
    {
        auto& api = network_->Runtime();
        api.Check(api.hipMemcpy(host, source, bytes, 4), "read diagnostic frame");
    }
    void infer(const Frames& io, unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings = {},
               dlsslop::FrameTrace* trace = nullptr) override
    {
        infer(io.proxy, w, h, io.answer, passes, settings, trace);
    }
    // Input and output are w * h RGBA8, or RGBA16F with settings.fp16. verify
    // checks the GPU codec against the CPU reference inside the timed frame.
    void infer(const uint8_t* input, unsigned w, unsigned h, uint8_t* output, unsigned passes,
               const ProcessingSettings& settings = {}, dlsslop::FrameTrace* trace = nullptr,
               bool verify = false)
    {
        if (options_.test_identity) {
            std::memcpy(output, input, size_t(w) * h * (settings.fp16 ? 8 : 4));
            return;
        }
        if (passes > 1 || options_.self_test || settings.motion) {
            // The optional upstream approximate cache has one history, not
            // one history per pass. Do not silently mix those states.
            const char* adaptive = std::getenv("DLSS5_VIT_ADAPTIVE");
            if (adaptive && std::strtoul(adaptive, nullptr, 10))
                throw std::range_error("multi-pass/motion/self-test requires DLSS5_VIT_ADAPTIVE=0 (uncached inference)");
        }
        const auto g = dlsslop::geometry(w, h, tier_);
        auto& api = network_->Runtime();
        const auto trace_image = [&](unsigned pass, const char* stage, const void* pointer, unsigned channels) {
            if (!trace) return;
            network_->Synchronize();
            std::vector<float> buffer(std::size_t(g.width) * g.height * channels);
            api.Check(api.hipMemcpy(buffer.data(), pointer, buffer.size() * sizeof(float), 2),
                      "read diagnostic neural stage");
            char name[32];
            std::snprintf(name, sizeof name, "pass-%02u-%s", pass + 1, stage);
            trace->image(name, buffer.data(), g, channels);
        };
        if (settings.fp16 && options_.cpu_compose)
            throw std::range_error("FP16 proxy transport requires Vulkan composition; disable --cpu-compose");
        if (!std::isfinite(settings.color_preserve) || settings.color_preserve < 0 || settings.color_preserve > 1)
            throw std::range_error("invalid color preservation strength");
        dlsslop::validate_native_tuning(settings.tuning);
        const bool tuned = !dlsslop::native_tuning_is_default(settings.tuning);
        const bool colored = settings.color_preserve > 0;
        if ((tuned || colored) && !device_scratch_)
            api.Check(api.hipMalloc(&device_scratch_, size_t(g.width) * g.height * 12), "allocate post-processing output");
        if (passes > 1 && !device_feedback_)
            api.Check(api.hipMalloc(&device_feedback_, size_t(g.width) * g.height * 16), "allocate inter-pass feedback");
        mark(0);
        if (gpu_codec_) {
            gpu_codec_->encode(input, g, device_input_, settings.fp16);
            if (verify) {
                std::vector<float> reference;
                dlsslop::encode_proxy(input, g, settings.fp16, reference);
                selftest::compare(selftest::Buffer::read_pointer(api, network_->Stream(), device_input_, reference.size()),
                                  reference, "GPU encoder");
                std::printf("GPU encode vs CPU reference: FP32 bit-identical\n");
                std::fflush(stdout);
            }
        } else {
            dlsslop::encode_proxy(input, g, settings.fp16, encoded_);
            api.Check(api.hipMemcpy(device_input_, encoded_.data(), encoded_.size() * sizeof(float), 1), "upload encoded frame");
        }
        mark(1);
        if (settings.motion) {
            // GpuTemporal itself drops the history for a new pass count, quality, grid or placement.
            const bool reset = options_.self_test || !previous_settings_.motion ||
                previous_settings_.fp16 != settings.fp16 || previous_settings_.precision16 != settings.precision16 ||
                !(previous_settings_.tuning == settings.tuning) || previous_settings_.color_preserve != settings.color_preserve;
            previous_settings_.motion = false; // Until the frame completes: a rejected one leaves no history.
            // Until end(), throw no std::range_error: the worker would serve on with the
            // history still pending, and the next begin() would fail.
            temporal_->begin(device_input_, g, settings.motion_quality, settings.motion_grid, passes, reset);
        } else {
            temporal_->reset();
        }
        void* answer = device_output_;
        for (unsigned pass = 0; pass < passes; ++pass) {
            void* const pass_input = pass ? device_feedback_ : device_input_;
            if (pass) {
                if (gpu_codec_) {
                    gpu_codec_->feedback(answer, pass_input, settings.precision16);
                    if (verify) {
                        dlsslop::feedback_neural_rgb(neural_.data(), g, feedback_, settings.precision16);
                        selftest::compare(selftest::Buffer::read_pointer(api, network_->Stream(), pass_input,
                                          feedback_.size()), feedback_, "GPU inter-pass feedback");
                        std::printf("GPU feedback for pass %u/%u vs CPU reference: FP32 bit-identical\n",
                                    pass + 1, passes);
                    }
                } else {
                    // Retain the initial encoded_ for final CPU composition.
                    dlsslop::feedback_neural_rgb(neural_.data(), g, feedback_, settings.precision16);
                    api.Check(api.hipMemcpy(pass_input, feedback_.data(), feedback_.size() * sizeof(float), 1),
                              "upload inter-pass feedback");
                }
            }
            trace_image(pass, "input", pass_input, 4);
            // Stages alternate between two buffers, since tuning and colour read
            // neighbours; with motion, the last writes the pass's history slot.
            void* history = nullptr;
            void* stages[] = {device_output_, device_scratch_, device_output_};
            if (settings.motion) {
                history = temporal_->history(pass, pass_input);
                stages[tuned + colored] = temporal_->target(pass);
            }
            // Graph replay (upstream o.graph, off) would need one stable rgb_output.
            network_->Enqueue(pass_input, history, stages[0], 0);
            trace_image(pass, "raw", stages[0], 3);
            if (tuned) {
                dlsslop::gpu_tune(*kernels_, g, pass_input, stages[0], stages[1], settings.tuning);
                trace_image(pass, "tuned", stages[1], 3);
            }
            if (colored) {
                dlsslop::gpu_preserve_color(*kernels_, g, device_input_, stages[tuned], stages[1 + tuned],
                                            settings.color_preserve);
                trace_image(pass, "color", stages[1 + tuned], 3);
            }
            answer = stages[tuned + colored];
            // The CPU codec, like the GPU one, rejects the nonfinite samples it reads.
            if (!gpu_codec_ || verify) {
                network_->Synchronize();
                api.Check(api.hipMemcpy(neural_.data(), answer, neural_.size() * sizeof(float), 2),
                          "read neural answer");
            }
        }
        answer_ = answer;
        if (settings.motion) temporal_->end();
        mark(2);
        if (gpu_codec_) {
            gpu_codec_->decode(answer, output);
            mark(3);
            gpu_codec_->finish();
            if (verify) {
                const size_t bpp = settings.fp16 ? 8 : 4;
                std::vector<uint8_t> reference(size_t(w) * h * bpp);
                dlsslop::decode_neural_proxy(input, g, settings.fp16, neural_.data(), reference.data());
                const size_t first = std::mismatch(reference.begin(), reference.end(), output).first - reference.begin();
                if (first < reference.size()) {
                    std::fprintf(stderr, "GPU decoder first mismatch: x=%zu y=%zu byte=%zu GPU=%u CPU=%u\n",
                                 (first / bpp) % w, (first / bpp) / w, first % bpp,
                                 unsigned(output[first]), unsigned(reference[first]));
                    throw std::runtime_error("GPU decoder disagrees with CPU reference");
                }
                std::printf("GPU decode vs CPU reference: bit-identical\n");
                std::fflush(stdout);
            }
        } else {
            if (options_.cpu_compose)
                dlsslop::decode_rgba8(input, g, encoded_.data(), neural_.data(), output);
            else
                dlsslop::decode_neural_proxy(input, g, settings.fp16, neural_.data(), output);
            mark(3);
        }
        previous_settings_ = settings;
        api.Check(api.hipEventSynchronize(marks_[3]), "timing event completion");
        api.Check(api.hipEventElapsedTime(&upload_ms, marks_[0], marks_[1]), "upload interval");
        api.Check(api.hipEventElapsedTime(&inference_ms, marks_[1], marks_[2]), "inference interval");
        api.Check(api.hipEventElapsedTime(&readback_ms, marks_[2], marks_[3]), "readback interval");
    }
    // The latest infer()'s raw network answer, read back outside its timing.
    const std::vector<float>& raw_result()
    {
        if (gpu_codec_) {
            auto& api = network_->Runtime();
            api.Check(api.hipMemcpy(neural_.data(), answer_, neural_.size() * sizeof(float), 2),
                      "read raw network answer");
        }
        return neural_;
    }
};

// DLSSNR-AMD's network on a Vulkan device of the daemon's own.
class VulkanEngine : public Backend {
    std::unique_ptr<dlsslop::VulkanNetwork> network_;
    unsigned tier_;
    bool warned_sharpness_ = false, warned_color_ = false;

    static dlsslop::VulkanFrame frame(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings)
    {
        dlsslop::VulkanFrame f;
        f.width = w;
        f.height = h;
        f.fp16 = settings.fp16;
        f.passes = passes;
        f.intensity = settings.tuning.intensity;
        f.local_tone = settings.tuning.tone;
        f.local_structure = settings.tuning.structure;
        f.motion = settings.motion;
        return f;
    }
public:
    VulkanEngine(std::unique_ptr<dlsslop::VulkanNetwork> network, unsigned tier)
        : network_(std::move(network)), tier_(tier) {}
    const char* name() const override { return "Vulkan"; }
    unsigned tier() const override { return tier_; }
    std::string processing() const override { return "Vulkan on " + network_->device_name() + " at each frame's extent"; }
    // The network sizes itself to each frame: a tier only changes the raster the layer targets.
    bool retier(unsigned tier) override
    {
        tier_ = tier;
        return true;
    }
    // Built before the daemon reports itself ready, for the raster's usual frame.
    void prepare() override
    {
        const auto raster = dlsslop::geometry(1, 1, tier_);
        network_->shape(frame(raster.width, tier_, 1, {}));
    }
    bool fits(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings) const override
    {
        return !network_->shape_differs(frame(w, h, passes, settings));
    }
    void reshape(unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings) override
    {
        network_->shape(frame(w, h, passes, settings));
    }
    bool import(const ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2]) override
    {
        int raw[2] = {fds[0].fd, fds[1].fd};
        const bool imported = network_->import(offer.generation, offer.allocation, offer.size, raw);
        fds[0].fd = raw[0];
        fds[1].fd = raw[1];
        return imported;
    }
    Frames frames(uint32_t generation, size_t bytes) const override
    {
        return network_->holds(generation, bytes) ? Frames{nullptr, nullptr, generation} : Frames{};
    }
    void read_back(void*, const void*, size_t) override
    {
        throw std::runtime_error("the Vulkan network cannot trace; use --backend hip");
    }
    void infer(const Frames& io, unsigned w, unsigned h, unsigned passes, const ProcessingSettings& settings = {},
               dlsslop::FrameTrace* trace = nullptr) override
    {
        if (trace) throw std::runtime_error("the Vulkan network cannot trace; use --backend hip");
        // Said once each: the HIP backend's own stages, which this network does not have.
        if (settings.tuning.sharpness != 0 && !std::exchange(warned_sharpness_, true))
            std::fprintf(stderr, "the Vulkan network ignores sharpness\n");
        if (settings.color_preserve != 0 && !std::exchange(warned_color_, true))
            std::fprintf(stderr, "the Vulkan network ignores color preservation\n");
        network_->infer(frame(w, h, passes, settings), io.generation, io.proxy, io.answer);
        upload_ms = network_->upload_ms;
        inference_ms = network_->inference_ms;
        readback_ms = network_->readback_ms;
    }
};

// Where the Vulkan network loads from; the pipeline cache is a convenience.
dlsslop::VulkanPaths vulkan_paths(const Options& o)
{
    std::string cache = xdg_path("XDG_CACHE_HOME", ".cache", "dlsslop-amd/vulkan-pipelines.cache");
    std::error_code error;
    if (!cache.empty() && !std::filesystem::create_directories(std::filesystem::path(cache).parent_path(), error) && error)
        cache.clear();
    return {o.vulkan_model, vulkan_shaders(), cache};
}

// The backend --backend selects. auto takes the Vulkan network when its model is
// there and a device can run it, and says why not before taking HIP.
std::unique_ptr<Backend> open_backend(Options& o, unsigned tier)
{
    if (!o.test_identity && o.backend != "hip") {
        try {
            std::error_code error;
            if (!std::filesystem::is_regular_file(o.vulkan_model, error))
                throw std::runtime_error("no model at " + o.vulkan_model +
                                         " (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)");
            auto network = std::make_unique<dlsslop::VulkanNetwork>(vulkan_paths(o), o.device);
            std::fprintf(stderr, "Vulkan network on %s\n", network->device_name().c_str());
            return std::make_unique<VulkanEngine>(std::move(network), tier);
        } catch (const std::exception& e) {
            if (o.backend == "vulkan") throw;
            std::fprintf(stderr, "Vulkan network unavailable (%s); using HIP\n", e.what());
        }
    }
    if (!o.test_identity) o.device = select_device(o.device);
    return std::make_unique<Engine>(o, tier);
}

// The Vulkan network on a deterministic gradient: finite, repeatable and changed.
void run_vulkan_self_test(const Options& o, Backend& engine)
{
    const unsigned passes = o.passes.value_or(kNativeDefaultPasses);
    const auto raster = dlsslop::geometry(1, 1, engine.tier());
    const unsigned w = raster.width, h = engine.tier();
    std::vector<uint8_t> input(size_t(w) * h * 4), output(input.size()), first;
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
            uint8_t* p = &input[(size_t(y) * w + x) * 4];
            p[0] = uint8_t(x * 255 / (w - 1));
            p[1] = uint8_t(y * 255 / (h - 1));
            p[2] = uint8_t(((x / 32 + y / 32) & 1) ? 200 : 60);
            p[3] = 255;
        }
    for (unsigned run = 0; run < o.self_test_runs; ++run) {
        engine.infer({input.data(), output.data(), 0}, w, h, passes);
        if (!run) first = output;
        else if (output != first) throw std::runtime_error("self-test run " + std::to_string(run + 1) + " differs from the first");
    }
    size_t changed = 0;
    for (size_t i = 0; i < input.size(); ++i) changed += (i % 4 != 3) && input[i] != output[i];
    if (!changed) throw std::runtime_error("self-test: the network left the input unchanged");
    if (!o.output.empty()) {
        std::ofstream ppm(o.output, std::ios::binary);
        ppm << "P6\n" << w << ' ' << h << "\n255\n";
        for (size_t i = 0; i < output.size(); i += 4) ppm.write(reinterpret_cast<const char*>(&output[i]), 3);
        if (!ppm) throw std::runtime_error("write " + o.output);
    }
    std::fprintf(stderr, "Vulkan self-test PASS: %u identical runs at %ux%u; changed_components=%zu\n"
                 "tier=%u; passes=%u; upload_ms=%.3f; network_ms=%.3f; readback_ms=%.3f\n",
                 o.self_test_runs, w, h, changed, engine.tier(), passes, engine.upload_ms, engine.inference_ms,
                 engine.readback_ms);
}

void run_self_test(const Options& o, Engine& engine)
{
    const unsigned passes = o.passes.value_or(kNativeDefaultPasses);
    constexpr unsigned w = 640, h = 360;
    std::vector<uint8_t> input(size_t(w) * h * 4), output(input.size());
    for (unsigned y = 0; y < h; ++y) for (unsigned x = 0; x < w; ++x) {
        const size_t p = (size_t(y) * w + x) * 4;
        input[p] = static_cast<uint8_t>(x * 255 / (w - 1));
        input[p + 1] = static_cast<uint8_t>(y * 255 / (h - 1));
        input[p + 2] = static_cast<uint8_t>((((x / 32) ^ (y / 32)) & 1) ? 192 : 64);
        input[p + 3] = 255;
    }
    const unsigned repeats = o.self_test_runs;
    const auto g = dlsslop::geometry(w, h, engine.tier());
    std::vector<float> first_raw;
    std::vector<uint8_t> first_output;
    // Only the first run checks the codec against the CPU reference, so the
    // later runs time the production path; each must reproduce the first.
    for (unsigned run = 0; run < repeats; ++run) {
        engine.infer(input.data(), w, h, output.data(), passes, {}, nullptr, !run);
        const auto& raw = engine.raw_result();
        if (!run) {
            first_raw = raw;
            first_output = output;
            float minimum = raw.front(), maximum = raw.front();
            size_t below_zero = 0, above_one = 0;
            for (float value : raw) {
                if (!std::isfinite(value)) throw std::runtime_error("network produced nonfinite values");
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                below_zero += value < 0.0f;
                above_one += value > 1.0f;
            }
            std::printf("raw network RGB range=%.9g..%.9g below_zero=%zu above_one=%zu samples=%zu\n",
                        double(minimum), double(maximum), below_zero, above_one, raw.size());
        } else if (std::memcmp(raw.data(), first_raw.data(), raw.size() * sizeof(float))) {
            size_t first = raw.size(), different = 0;
            float maximum = 0;
            uint32_t first_before = 0, first_after = 0;
            for (size_t i = 0; i < raw.size(); ++i) {
                uint32_t before, after;
                std::memcpy(&before, &first_raw[i], sizeof(before));
                std::memcpy(&after, &raw[i], sizeof(after));
                if (before == after) continue;
                if (first == raw.size()) {
                    first = i;
                    first_before = before;
                    first_after = after;
                }
                ++different;
                maximum = std::max(maximum, std::fabs(raw[i] - first_raw[i]));
            }
            std::fflush(stdout);
            std::fprintf(stderr,
                "network repeat %u/%u differs from first identical input: samples=%zu/%zu "
                "max_abs_error=%.9g; first x=%zu y=%zu channel=%zu "
                "first=%.9g (0x%08x) repeat=%.9g (0x%08x)\n",
                run + 1, repeats, different, raw.size(), double(maximum),
                (first / 3) % g.width, (first / 3) / g.width, first % 3,
                double(first_raw[first]), unsigned(first_before), double(raw[first]), unsigned(first_after));
            throw std::runtime_error("network is nondeterministic with identical input and fixed seed");
        } else if (const auto [a, b] = std::mismatch(first_output.begin(), first_output.end(), output.begin());
                   a != first_output.end()) {
            const size_t first = a - first_output.begin();
            std::fflush(stdout);
            std::fprintf(stderr,
                "network repeat %u/%u decoded output differs from the first run: "
                "first x=%zu y=%zu channel=%zu first=%u repeat=%u\n",
                run + 1, repeats, (first / 4) % w, (first / 4) / w, first % 4, unsigned(*a), unsigned(*b));
            throw std::runtime_error("decoded output differs from the verified first run");
        }
        std::printf("network repeat %u/%u: %s; passes=%u upload_ms=%.3f network_ms=%.3f readback_ms=%.3f\n",
                    run + 1, repeats, run ? "raw FP32 and output bit-identical" : "baseline", passes,
                    engine.upload_ms, engine.inference_ms, engine.readback_ms);
        std::fflush(stdout);
    }
    uint64_t hash = 14695981039346656037ull;
    unsigned low = 255, high = 0;
    size_t changed = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        hash = (hash ^ output[i]) * 1099511628211ull;
        if ((i & 3) == 3) continue;
        low = std::min(low, unsigned(output[i]));
        high = std::max(high, unsigned(output[i]));
        changed += output[i] != input[i];
    }
    if (high <= low || !changed)
        throw std::runtime_error("self-test returned constant or unchanged RGB output");
    if (!o.output.empty()) {
        std::ofstream file(o.output, std::ios::binary);
        file << "P6\n" << w << ' ' << h << "\n255\n";
        for (size_t p = 0; p < size_t(w) * h; ++p)
            file.write(reinterpret_cast<const char*>(output.data() + p * 4), 3);
        if (!file) throw std::runtime_error("write self-test PPM");
    }
    std::printf("real-network self-test PASS: finite output; %u identical-input runs bit-exact; RGB range=%u..%u; changed_components=%zu; fnv1a64=%016llx\n",
                repeats, low, high, changed, static_cast<unsigned long long>(hash));
    std::printf("tier=%u; passes=%u; blocks=%s; upload_ms=%.3f; network_ms=%.3f; readback_ms=%.3f\n",
                engine.tier(), passes, o.performance ? "upstream performance preset" : "all 71",
                engine.upload_ms, engine.inference_ms, engine.readback_ms);
}

void run_offline(const Options& o, Backend& engine)
{
    const unsigned passes = o.passes.value_or(kNativeDefaultPasses);
    const size_t bytes = size_t(o.width) * o.height * 4;
    std::ifstream in(o.input, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(bytes))
        throw std::runtime_error("offline input size must equal width * height * 4");
    std::vector<uint8_t> pixels(bytes), result(bytes);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("read offline input");
    engine.infer({pixels.data(), result.data(), 0}, o.width, o.height, passes);
    std::ofstream out(o.output, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("write offline output");
    std::fprintf(stderr, "offline: passes=%u upload=%.2f ms, network=%.2f ms, readback=%.2f ms\n",
                 passes, engine.upload_ms, engine.inference_ms, engine.readback_ms);
}

// The socket beside the channel file on which the layer offers its exported
// frames. systemd's socket unit hands it over already bound (LISTEN_FDS), and
// connecting to it is then how a client starts the worker; otherwise the
// worker binds it itself, only when the GPU codec can import frames.
struct TransportListener {
    dlsslop::Descriptor socket;
    std::string path;
    bool bound = false; // This worker created the pathname and removes it again.
    TransportListener(const std::string& channel, bool wanted) : path(ShmTransportPath(channel))
    {
        const char* pid = std::getenv("LISTEN_PID");
        const char* count = std::getenv("LISTEN_FDS");
        if (pid && count && std::strtol(pid, nullptr, 10) == getpid() && !std::strcmp(count, "1")) {
            socket.fd = 3; // SD_LISTEN_FDS_START
            fcntl(socket.fd, F_SETFL, fcntl(socket.fd, F_GETFL) | O_NONBLOCK);
            fcntl(socket.fd, F_SETFD, FD_CLOEXEC);
            return;
        }
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (!wanted) return;
        errno = ENAMETOOLONG;
        if (path.size() < sizeof address.sun_path) {
            std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
            socket.fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
            bound = socket.fd >= 0 && !bind(socket.fd, reinterpret_cast<const sockaddr*>(&address), sizeof address);
            // Never replace a socket someone else owns: systemd's, left listening by
            // an idle dlsslop.socket, would be gone for good once this worker exits.
            if (!bound && errno == EADDRINUSE)
                throw std::runtime_error(path + " exists: stop dlsslop.socket before starting dlsslopd "
                                         "by hand, or remove the file if nothing uses it");
            if (bound && !listen(socket.fd, 4))
                return;
        }
        std::fprintf(stderr, "device-local transport unavailable (%s): %s\n", path.c_str(), std::strerror(errno));
    }
    ~TransportListener() { if (bound) unlink(path.c_str()); }
};

// One offer: the layer sends it right after connecting.
bool receive_offer(int peer, ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2])
{
    pollfd ready{peer, POLLIN, 0};
    if (poll(&ready, 1, 100) != 1) return false;
    alignas(cmsghdr) char control[CMSG_SPACE(2 * sizeof(int))]{};
    iovec data{&offer, sizeof offer};
    msghdr message{};
    message.msg_iov = &data;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof control;
    const ssize_t got = recvmsg(peer, &message, MSG_CMSG_CLOEXEC);
    const cmsghdr* rights = got > 0 ? CMSG_FIRSTHDR(&message) : nullptr;
    if (!rights || rights->cmsg_level != SOL_SOCKET || rights->cmsg_type != SCM_RIGHTS) return false;
    const size_t count = std::min<size_t>(2, (rights->cmsg_len - CMSG_LEN(0)) / sizeof(int));
    for (size_t i = 0; i < count; ++i) std::memcpy(&fds[i].fd, CMSG_DATA(rights) + i * sizeof(int), sizeof(int));
    return size_t(got) == sizeof offer && count == 2 && offer.magic == kShmMagic &&
           !(message.msg_flags & (MSG_TRUNC | MSG_CTRUNC));
}

// Imports every pending offer and answers each on its own connection: one
// byte, nonzero when imported.
void accept_offers(const TransportListener& listener, Backend& engine)
{
    for (int peer; (peer = accept4(listener.socket.fd, nullptr, nullptr, SOCK_CLOEXEC)) >= 0; close(peer)) {
        ShmTransportOffer offer{};
        dlsslop::Descriptor fds[2];
        if (!receive_offer(peer, offer, fds)) continue; // A start or liveness probe sends nothing.
        const uint8_t imported = engine.import(offer, fds);
        if (!imported) std::fprintf(stderr, "device-local transport offer rejected\n");
        send(peer, &imported, 1, MSG_NOSIGNAL);
    }
}

// The layer must know the real neural raster before building its proxy.
// Otherwise it mistakes a worker-upscaled answer for native-resolution output
// and skips its detail-preserving composition branch. CPU composition and
// identity publish none: the mapping may retain a preceding neural worker's.
void publish_raster(const Options& o, ShmHeader* h, unsigned tier)
{
    const bool neural = !o.cpu_compose && !o.test_identity;
    h->nativeModelMaxWidth.store(neural ? dlsslop::geometry(1, 1, tier).width : 0);
    h->nativeModelMaxHeight.store(neural ? tier : 0);
}

// Between frames: rebuilds the network once for each tier a controller stores
// that differs from the active one. The layer presents its own frames while
// helperState reads Starting, then offers its device-local frames to the new
// engine. An unusable tier is overwritten with the active one. A failed
// rebuild ends the worker, leaving the active tier for the next one. True
// after a rebuild.
bool follow_tier(std::unique_ptr<Backend>& engine, const Options& o, Mapping& mapping, const char* ready)
{
    auto* h = mapping.h;
    const unsigned wanted = h->nativeTier.load(), active = engine->tier();
    if (wanted == active) return false;
    if (!ShmNativeTier(wanted)) {
        h->nativeTier.store(active);
        return false;
    }
    // A network that sizes itself to each frame only needs the layer to target the new raster.
    if (engine->retier(wanted)) {
        std::fprintf(stderr, "neural tier %u -> %u\n", active, wanted);
        publish_raster(o, h, wanted);
        return false;
    }
    h->helperState.store(kHelperStarting);
    h->modelUp.store(0);
    std::fprintf(stderr, "neural tier %u -> %u: rebuilding\n", active, wanted);
    mapping.reason("rebuilding for neural tier " + std::to_string(wanted));
    engine.reset(); // The old network's memory goes first.
    try {
        engine = std::make_unique<Engine>(o, wanted);
        engine->prepare();
    } catch (...) {
        h->nativeTier.store(active);
        throw;
    }
    publish_raster(o, h, wanted);
    mapping.reason(ready);
    h->modelUp.store(o.test_identity ? 0 : 1);
    h->helperState.store(kHelperRunning, std::memory_order_release);
    return true;
}

// Idle exit: stop taking requests before leaving. A layer that reads Stopped
// from now on sends none, and one whose request got in first is served: true
// when none did.
bool retire(ShmHeader* h, uint32_t request)
{
    h->helperState.store(kHelperStopped);
    if (h->seq_req.load() == request) return true;
    h->helperState.store(kHelperRunning, std::memory_order_release);
    return false;
}

void run_worker(Options o)
{
    Mapping mapping(o.shm);
    auto* h = mapping.h;
    std::unique_ptr<dlsslop::TraceRequests> traces;
    if (!o.trace_dir.empty())
        traces = std::make_unique<dlsslop::TraceRequests>(o.trace_dir, o.shm);
    std::unique_ptr<dlsslop::FrameTrace> pending_trace;
    // An explicit tier replaces the channel's; otherwise a usable live one stays.
    const unsigned live = h->nativeTier.load();
    const unsigned tier = o.tier.value_or(ShmNativeTier(live) ? live : kNativeDefaultTier);
    h->nativeTier.store(tier);
    if (o.passes) h->passes.store(*o.passes);
    h->compositionBypass.store(o.cpu_compose || o.test_identity ? 1 : 0);
    publish_raster(o, h, tier);
    // Linux futex wake is emitted by the patched Vulkan layer. Timeout maintains liveness
    // with old clients and permits signals/quit; no GPU polling is involved.
    std::atomic<bool> heartbeat_stop{false};
    std::thread heartbeat([&] {
        while (!heartbeat_stop.load()) {
            h->heartbeat.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    struct Stop {
        std::atomic<bool>& flag;
        std::thread& thread;
        ~Stop() { flag.store(true); thread.join(); }
    } stop_heartbeat{heartbeat_stop, heartbeat};
    try {
        mapping.reason(o.test_identity ? "IDENTITY TEST: inference disabled" : "initializing the network");
        const TransportListener transport(o.shm, !o.test_identity && !o.cpu_codec);
        std::unique_ptr<Backend> engine = open_backend(o, tier);
        if (traces && dynamic_cast<VulkanEngine*>(engine.get()))
            throw std::runtime_error("--trace-dir requires --backend hip");
        engine->prepare();
        // Only serving stops gracefully, from the ready announcement on.
        // Before it, and in every other mode, SIGINT and SIGTERM terminate.
        std::signal(SIGINT, stop_handler);
        std::signal(SIGTERM, stop_handler);
        h->modelUp.store(o.test_identity ? 0 : 1);
        h->helperState.store(kHelperRunning, std::memory_order_release);
        const std::string ready = o.test_identity ? "IDENTITY TEST: no neural rendering"
                                                  : std::string("native ") + engine->name() +
                                                        " ready; display-encoded RGBA8/FP16 proxy";
        mapping.reason(ready);
        std::fprintf(stderr, "worker ready: %s%s\n", o.shm.c_str(), o.test_identity ? " [IDENTITY TEST]" : "");
        if (!o.test_identity)
            std::fprintf(stderr, "neural tier=%u; %s; %s; live controls enabled\n", tier, engine->processing().c_str(),
                         o.cpu_compose ? "CPU composition" : "native-resolution Vulkan composition");
        uint64_t frames = 0;
        unsigned previous_passes = 0;
        uint32_t last = h->seq_resp.load(std::memory_order_acquire);
        // When the latest work ended: an answer, a rebuild, or readiness.
        auto active = std::chrono::steady_clock::now();
        std::string failure;
        while (!stopping && !h->quit.load(std::memory_order_relaxed)) {
            accept_offers(transport, *engine);
            if (traces && !pending_trace) {
                try {
                    pending_trace = traces->take();
                    if (pending_trace) h->controlSeq.fetch_add(1);
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "diagnostic request rejected: %s\n", error.what());
                }
            }
            const uint32_t request = h->seq_req.load(std::memory_order_acquire);
            if (request == last) {
                if (follow_tier(engine, o, mapping, ready.c_str())) active = std::chrono::steady_clock::now();
                const timespec timeout{0, 100000000};
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_req), FUTEX_WAIT,
                        request, &timeout, nullptr, 0);
                // A request that arrived during the wait is served, however late.
                if (o.idle_exit && h->seq_req.load(std::memory_order_acquire) == request &&
                    std::chrono::steady_clock::now() - active >= std::chrono::seconds(o.idle_exit) &&
                    retire(h, request)) {
                    std::fprintf(stderr, "no request for %u s; stopping\n", o.idle_exit);
                    break;
                }
                continue;
            }
            // Writers store a setting before bumping controlSeq, so sample the
            // generations after this loop's own bump (the trace claim) and
            // before reading any setting, the dimensions or the input; sample
            // again after inference.
            const uint32_t control_sequence = h->controlSeq.load();
            const uint32_t tuning_sequence = h->tuningSeq.load();
            const uint32_t held_input = pending_trace ? h->holdFrame.load() : 0;
            const unsigned w = h->width.load(), height = h->height.load();
            last = request;
            try {
                ProcessingSettings settings;
                settings.fp16 = h->hdrEncode.load() != 0;
                const bool hdr = h->hdrDetected.load() != kHdrNone && h->colourMode.load() != kColourDisplay;
                settings.precision16 = hdr || h->sdr16Multipass.load() != 0;
                settings.motion = h->mvecEnabled.load() != 0;
                settings.motion_quality = ShmMVecQuality(h);
                settings.motion_grid = ShmMVecPixelSize(h);
                settings.tuning = read_tuning(h);
                settings.color_preserve = BitsToFloat(h->colorPreserveBits.load());
                if (!w || !height || w > kMaxW || height > kMaxH || h->format.load() != 1)
                    throw std::range_error("unsupported request dimensions or proxy format");
                // Older/external clients must not silently enable unmapped
                // NVIDIA model controls that this fixed graph cannot honor.
                if (h->preset.load() || h->style.load() || h->autoMask.load() != 1 ||
                    BitsToFloat(h->skinStructureBits.load()) != -1.0f)
                    throw std::range_error("unmapped neural preset/style/skin-mask controls require their captured defaults");
                const size_t bytes = size_t(w) * height * (settings.fp16 ? 8 : 4);
                // A live control change takes effect on the next request;
                // never shorten or extend a chain partway through a frame.
                const unsigned passes = ShmPasses(h);
                if (!o.test_identity && passes != previous_passes) {
                    std::fprintf(stderr, "neural passes=%u; one final composition per frame\n", passes);
                    previous_passes = passes;
                }
                // The request's frames: an imported device-local pair, or the channel's slots.
                Backend::Frames io{mapping.input, mapping.output};
                if (const uint32_t generation = h->transportGen.load()) {
                    io = engine->frames(generation, bytes);
                    if (!io.generation) {
                        h->transportMiss.store(generation);
                        throw std::range_error("request names device-local frames this worker has not imported");
                    }
                } else {
                    engine->pin(mapping.input, mapping.output, bytes);
                }
                // A frame of a new shape needs a build: seconds in which the layer
                // presents its own frames rather than waiting for this one.
                if (!engine->fits(w, height, passes, settings)) {
                    h->helperState.store(kHelperStarting);
                    std::fprintf(stderr, "building the network for %ux%u%s\n", w, height, settings.fp16 ? " FP16" : "");
                    engine->reshape(w, height, passes, settings);
                    h->helperState.store(kHelperRunning, std::memory_order_release);
                }
                engine->infer(io, w, height, passes, settings, pending_trace.get());
                if (h->seq_req.load(std::memory_order_acquire) != request)
                    throw std::range_error("request changed during inference; old answer discarded");
                std::string trace_metadata;
                if (pending_trace) {
                    dlsslop::TraceFrameMetadata metadata;
                    metadata.frame_seq = request;
                    metadata.control_seq = control_sequence;
                    metadata.tuning_seq = tuning_sequence;
                    metadata.control_seq_end = h->controlSeq.load();
                    metadata.tuning_seq_end = h->tuningSeq.load();
                    metadata.held_input = held_input;
                    metadata.held_input_end = h->holdFrame.load();
                    std::vector<uint8_t> proxy(bytes);
                    engine->read_back(proxy.data(), io.proxy, bytes);
                    metadata.source_proxy_hash = 14695981039346656037ull;
                    for (uint8_t byte : proxy)
                        metadata.source_proxy_hash = (metadata.source_proxy_hash ^ byte) * 1099511628211ull;
                    metadata.passes = passes;
                    metadata.geometry = dlsslop::geometry(w, height, engine->tier());
                    metadata.fp16_proxy = settings.fp16;
                    metadata.fp16_feedback = settings.precision16;
                    metadata.motion = settings.motion;
                    metadata.intensity = settings.tuning.intensity;
                    metadata.local_tone = settings.tuning.tone;
                    metadata.local_structure = settings.tuning.structure;
                    metadata.sharpness = settings.tuning.sharpness;
                    metadata.color_preserve = settings.color_preserve;
                    trace_metadata = metadata.json();
                }
                if (!failure.empty()) { // Serving recovered: the failure is no longer current.
                    failure.clear();
                    mapping.reason(ready);
                }
                h->answeredW.store(w);
                h->answeredH.store(height);
                h->helperEvalMsBits.store(FloatToBits(engine->inference_ms));
                h->helperUploadMsBits.store(FloatToBits(engine->upload_ms));
                h->helperReadbackMsBits.store(FloatToBits(engine->readback_ms));
                ShmStore64(h->helperFramesLo, h->helperFramesHi, ++frames);
                h->seq_ok.store(request);
                h->seq_resp.store(request, std::memory_order_release);
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAKE,
                        1, nullptr, nullptr, 0);
                if (pending_trace) {
                    pending_trace->finish(trace_metadata);
                    pending_trace.reset();
                }
                if (o.once) break;
            } catch (const std::exception& e) {
                if (pending_trace) {
                    pending_trace->finish("{\"frame_seq\":" + std::to_string(request) + "}", e.what());
                    pending_trace.reset();
                }
                if (failure != e.what()) { // Report a persistent rejection once, not every frame.
                    mapping.reason(failure = e.what());
                    std::fprintf(stderr, "frame %u failed: %s\n", request, e.what());
                }
                h->answeredW.store(0);
                h->answeredH.store(0);
                h->seq_resp.store(request, std::memory_order_release);
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAKE,
                        1, nullptr, nullptr, 0);
                // A rejection (std::range_error) is answered as failed and serving goes
                // on. Any other failure ends the worker in every mode: a HIP or engine
                // fault is never silently replaced by fake output.
                if (o.once || !dynamic_cast<const std::range_error*>(&e)) throw;
            }
            active = std::chrono::steady_clock::now();
        }
        // Stopped before the teardown, which can take a while and unlinks a
        // socket this daemon bound: a layer that offers or presents from now on
        // leaves its frames for the next daemon instead of waiting on this one.
        h->helperState.store(kHelperStopped);
    } catch (const std::exception& e) {
        mapping.reason(e.what());
        h->helperState.store(kHelperModelFailed);
        throw;
    }
}
} // namespace

int main(int argc, char** argv)
{
    try {
        Options o = parse(argc, argv);
        const unsigned tier = o.tier.value_or(kNativeDefaultTier);
        if (o.diagnose) {
            const auto engine = open_backend(o, tier);
            if (!dynamic_cast<VulkanEngine*>(engine.get())) std::fprintf(stderr, "selected device %d\n", o.device);
            return 0;
        }
        if (o.test_identity)
            std::fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
        if (o.self_test || !o.input.empty()) {
            const auto engine = open_backend(o, tier);
            engine->prepare();
            if (!o.self_test) run_offline(o, *engine);
            else if (auto* hip = dynamic_cast<Engine*>(engine.get())) run_self_test(o, *hip);
            else run_vulkan_self_test(o, *engine);
        } else {
            run_worker(o);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dlsslopd: %s\n", e.what());
        return 1;
    }
}
