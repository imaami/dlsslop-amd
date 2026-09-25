// Native Linux worker for DLSS5VKLayer's shared-memory transport.
// SPDX-License-Identifier: MIT
#include "codec.h"
#include "vendor/LmxxfProductionOptions.h"
#include "codec_gpu.h"
#include "tuning.h"
#include "color_preserve.h"
#include "color_gpu.h"
#include "temporal_gpu.h"
#include "control_selftest.h"
#include "trace.h"
#include "shm_protocol.h"

#include <algorithm>
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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <linux/futex.h>
#include <pwd.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
namespace selftest = dlsslop::control_selftest;
volatile sig_atomic_t stopping;
void stop_handler(int) { stopping = 1; }

std::string default_assets()
{
    if (const char* data = std::getenv("XDG_DATA_HOME"); data && *data)
        return (std::filesystem::path(data) / "dlsslop-amd/model").string();
    const char* home = std::getenv("HOME");
    if (!home || !*home) {
        const struct passwd* user = getpwuid(getuid());
        home = user ? user->pw_dir : nullptr;
    }
    return home && *home ? (std::filesystem::path(home) / ".local/share/dlsslop-amd/model").string()
                         : std::string();
}

std::string default_modules()
{
    if (const char* path = std::getenv("DLSSLOP_MODULES"); path && *path) return path;
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) return {};
    const auto prefix = executable.parent_path().parent_path();
    const auto installed = prefix / "share/dlsslop-amd/HIP/gfx1201";
    if (std::filesystem::is_directory(installed, error)) return installed.string();
    // A worker run directly from the source tree's build directory uses the
    // same modules as the kernel build script, without an installed launcher.
    const auto development = prefix / "assets/HIP/gfx1201";
    if (std::filesystem::is_directory(development, error)) return development.string();
    return installed.string();
}

struct Options {
    std::string assets, modules, shm = ShmNativeDefaultPath();
    std::string input, output, trace_dir;
    unsigned tier = kNativeDefaultTier, width = 0, height = 0, self_test_runs = 10;
    unsigned passes = kNativeDefaultPasses;
    int device = -1;
    bool diagnose = false, test_identity = false, once = false, self_test = false;
    bool cpu_compose = false, cpu_codec = false, performance = false;
};

Options default_options()
{
    Options o;
    o.assets = default_assets();
    o.modules = default_modules();
    if (const char* path = std::getenv("DLSSNR_SHM"); path && *path) o.shm = path;
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
    unsigned motion_units = kMVecPixels;
};

dlsslop::NativeTuning read_tuning(const ShmHeader* h)
{
    return {BitsToFloat(h->intensityBits.load()), BitsToFloat(h->localToneBits.load()),
            BitsToFloat(h->localStructureBits.load()), BitsToFloat(h->sharpnessBits.load())};
}

bool same_tuning(const dlsslop::NativeTuning& a, const dlsslop::NativeTuning& b)
{
    return a.intensity == b.intensity && a.tone == b.tone &&
           a.structure == b.structure && a.sharpness == b.sharpness;
}

// Debounce expensive-looking slider drags without rebuilding fixed weights.
// The idle worker also commits pending changes and nudges the layer so a held
// frame updates after the settle interval, even when the game is not drawing.
class TuningLatch {
    uint32_t sequence_;
    dlsslop::NativeTuning current_, pending_;
    bool waiting_ = false;
    std::chrono::steady_clock::time_point changed_;
public:
    explicit TuningLatch(const ShmHeader* h)
        : sequence_(h->tuningSeq.load()), current_(read_tuning(h)), pending_(current_) {}
    dlsslop::NativeTuning update(ShmHeader* h)
    {
        const auto now = std::chrono::steady_clock::now();
        const auto sequence = h->tuningSeq.load();
        if (sequence != sequence_) {
            sequence_ = sequence;
            pending_ = read_tuning(h);
            changed_ = now;
            waiting_ = true;
        }
        const auto settle = std::chrono::milliseconds(std::min(h->rebuildSettleMs.load(), 5000u));
        if (waiting_ && now - changed_ >= settle) {
            current_ = pending_;
            waiting_ = false;
            h->controlSeq.fetch_add(1);
        }
        return current_;
    }
};

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

void usage(FILE* out)
{
    const Options defaults = default_options();
    std::fprintf(out,
        "Usage: dlsslopd [OPTIONS]\n"
        "  -a, --assets DIR        Model weights (.f16/.f32)\n"
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
        "                          Default: %u; game/display resolution unchanged\n"
        "  -P, --passes N          Chained neural evaluations per frame (1..%u)\n"
        "                          Default: %u; each pass consumes the previous output\n"
        "                          Sets startup count; live control may change it\n"
        "  -d, --device INDEX      HIP device index\n"
        "                          Default: auto, first visible gfx1201 device\n"
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
        defaults.assets.empty() ? "unset; required for inference" : defaults.assets.c_str(),
        defaults.modules.empty() ? "unset; required for inference" : defaults.modules.c_str(),
        defaults.shm.c_str(), ShmNativeDefaultPath().c_str(), defaults.tier,
        kMaxPasses, defaults.passes, defaults.self_test_runs,
        defaults.width, defaults.height);
}

Options parse(int argc, char** argv)
{
    Options o = default_options();
    bool self_test_runs_set = false;
    static const option opts[] = {
        {"assets", required_argument, nullptr, 'a'},
        {"modules", required_argument, nullptr, 'm'},
        {"shm", required_argument, nullptr, 's'},
        {"tier", required_argument, nullptr, 't'},
        {"passes", required_argument, nullptr, 'P'},
        {"device", required_argument, nullptr, 'd'},
        {"diagnose", no_argument, nullptr, 'D'},
        {"self-test", no_argument, nullptr, 'S'},
        {"self-test-runs", required_argument, nullptr, 'r'},
        {"input", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"width", required_argument, nullptr, 'W'},
        {"height", required_argument, nullptr, 'H'},
        {"cpu-compose", no_argument, nullptr, 'c'},
        {"cpu-codec", no_argument, nullptr, 'C'},
        {"performance", no_argument, nullptr, 'p'},
        {"once", no_argument, nullptr, '1'},
        {"trace-dir", required_argument, nullptr, 'R'},
        {"test-identity", no_argument, nullptr, 'T'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    for (int c; (c = getopt_long(argc, argv, "a:m:s:t:P:d:DSr:i:o:W:H:cCp1R:Th", opts, nullptr)) != -1;) {
        switch (c) {
        case 'a': o.assets = optarg; break;
        case 'm': o.modules = optarg; break;
        case 's': o.shm = optarg; break;
        case 't': o.tier = number(optarg, "tier"); break;
        case 'P': o.passes = number(optarg, "passes"); break;
        case 'd': o.device = static_cast<int>(number(optarg, "device")); break;
        case 'D': o.diagnose = true; break;
        case 'S': o.self_test = true; break;
        case 'r': o.self_test_runs = number(optarg, "self-test runs"); self_test_runs_set = true; break;
        case 'i': o.input = optarg; break;
        case 'o': o.output = optarg; break;
        case 'W': o.width = number(optarg, "width"); break;
        case 'H': o.height = number(optarg, "height"); break;
        case 'c': o.cpu_compose = o.cpu_codec = true; break;
        case 'C': o.cpu_codec = true; break;
        case 'p': o.performance = true; break;
        case '1': o.once = true; break;
        case 'R':
            if (!*optarg) throw std::runtime_error("--trace-dir requires a nonempty directory");
            o.trace_dir = optarg;
            break;
        case 'T': o.test_identity = true; break;
        case 'h': usage(stdout); std::exit(0);
        default: usage(stderr); throw std::runtime_error("invalid arguments");
        }
    }
    if (optind != argc) throw std::runtime_error("unexpected positional argument");
    if (o.tier != 720 && o.tier != 900 && o.tier != 1080)
        throw std::runtime_error("tier must be 720, 900, or 1080");
    if (!o.passes || o.passes > kMaxPasses)
        throw std::runtime_error("--passes must be 1.." + std::to_string(kMaxPasses));
    if (o.self_test_runs < 2 || o.self_test_runs > 1000)
        throw std::runtime_error("--self-test-runs must be 2..1000");
    if (self_test_runs_set && !o.self_test)
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
    int fd_ = -1;
    void* mapping_ = MAP_FAILED;
public:
    ShmHeader* h = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;

    explicit Mapping(const std::string& name)
    {
        const auto parent = std::filesystem::path(name).parent_path();
        if (parent.empty()) throw std::runtime_error("--shm requires a path inside a private directory");
        dlsslop::private_directory(parent, "shared-memory");
        fd_ = open(name.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) system_error("open shared-memory file");
        if (flock(fd_, LOCK_EX | LOCK_NB)) {
            close(fd_); fd_ = -1;
            throw std::runtime_error("another worker owns this shared-memory file");
        }
        struct stat st{};
        if (fstat(fd_, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid()) {
            close(fd_); fd_ = -1;
            throw std::runtime_error("shared-memory file must be regular and owned by the current user");
        }
        if (fchmod(fd_, 0600)) {
            const int error = errno;
            close(fd_); fd_ = -1;
            errno = error;
            system_error("make shared-memory file private");
        }
        if (ftruncate(fd_, static_cast<off_t>(ShmTotalBytes()))) {
            const int error = errno;
            close(fd_); fd_ = -1;
            errno = error;
            system_error("size shared-memory file");
        }
        mapping_ = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            const int error = errno;
            close(fd_); fd_ = -1;
            errno = error;
            system_error("map shared-memory file");
        }
        h = static_cast<ShmHeader*>(mapping_);
        if (h->magic.load() != kShmMagic || h->version.load() != kShmVersion) ShmInitNativeDefaults(h);
        input = static_cast<uint8_t*>(mapping_) + kHeaderBytes;
        output = input + kMaxFrame;
        h->quit.store(0);
        h->proxyExportSeq.store(0);
        h->answerExportSeq.store(0);
        h->layerProxySeq.store(0);
        h->layerAnswerSeq.store(0);
        h->helperPassCeiling.store(kMaxPasses);
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
        if (h) {
            h->modelUp.store(0);
            if (h->helperState.load() != kHelperModelFailed) h->helperState.store(kHelperStopped);
            munmap(mapping_, ShmTotalBytes());
        }
        if (fd_ >= 0) close(fd_);
    }
    void reason(const std::string& text)
    {
        ShmStoreString(h->helperReasonSeq, h->helperReason, kReasonBytes, text.c_str());
    }
};

class Engine {
    Options options_;
    std::unique_ptr<hip_reference::Network> network_;
    std::unique_ptr<dlsslop::GpuCodec> gpu_codec_;
    std::unique_ptr<dlsslop::GpuTuning> gpu_tuning_;
    std::unique_ptr<dlsslop::GpuColor> gpu_color_;
    bool color_backend_checked_ = false;
    std::unique_ptr<dlsslop::GpuTemporal> temporal_;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    void* device_tuned_ = nullptr;
    void* answer_ = nullptr; // The latest frame's final network answer.
    unsigned width_ = 0, height_ = 0;
    std::vector<float> encoded_, neural_, feedback_;
    std::vector<float> color_original_, color_raw_, color_result_;
    ProcessingSettings previous_settings_;
    unsigned previous_passes_ = 0;
    // Stream events: frame start, uploaded, evaluated, answered. Timing never
    // stalls the stream; the intervals are read once the answer is complete.
    hip_probe::Handle marks_[4]{};

    void mark(unsigned i)
    {
        auto& api = network_->Runtime();
        api.Check(api.hipEventRecord(marks_[i], network_->Stream()), "record timing event");
    }
public:
    const char* color_backend() const { return gpu_color_ ? "gpu" : color_backend_checked_ ? "cpu" : "off"; }
    float upload_ms = 0, inference_ms = 0, readback_ms = 0;
    explicit Engine(Options o) : options_(std::move(o)) {}
    ~Engine() { reset(); }
    void reset()
    {
        if (!network_) return;
        auto& api = network_->Runtime();
        api.hipStreamSynchronize(network_->Stream());
        for (auto event : marks_) api.hipEventDestroy(event);
        temporal_.reset();
        gpu_tuning_.reset();
        gpu_color_.reset();
        color_backend_checked_ = false;
        gpu_codec_.reset();
        if (device_input_) api.hipFree(device_input_);
        if (device_output_) api.hipFree(device_output_);
        if (device_tuned_) api.hipFree(device_tuned_);
        device_input_ = device_output_ = device_tuned_ = answer_ = nullptr;
        network_.reset();
    }
    void prepare()
    {
        if (options_.test_identity) return;
        const unsigned w = options_.tier == 720 ? 1280 : options_.tier == 900 ? 1600 : 1920;
        const unsigned h = options_.tier == 720 ? 768 : options_.tier == 900 ? 960 : 1152;
        auto opt = LmxxfProductionOptions(w, h, options_.modules, options_.assets);
        opt.device = static_cast<unsigned>(options_.device);
        if (!options_.performance) opt.skip_blocks.clear();
        network_ = std::make_unique<hip_reference::Network>(opt);
        network_->SetNoise({}); // Fast prefix uses procedural noise, not noise.f32.
        auto& api = network_->Runtime();
        for (auto& event : marks_) api.Check(api.hipEventCreate(&event), "create timing event");
        if (!options_.cpu_codec)
            gpu_codec_ = std::make_unique<dlsslop::GpuCodec>(api, network_->Stream(), options_.modules + "/linux_codec.hsaco");
        const size_t pixels = size_t(w) * h;
        api.Check(api.hipMalloc(&device_input_, pixels * 16), "allocate network input");
        api.Check(api.hipMalloc(&device_output_, pixels * 12), "allocate network output");
        width_ = w; height_ = h;
        encoded_.resize(pixels * 4);
        neural_.resize(pixels * 3);
        // Warm once before announcing readiness. Weights and scratch are lazy in upstream.
        // Keep alpha at one, as in the sRGB codec; warmup has no temporal history.
        for (size_t p = 0; p < pixels; ++p) encoded_[p * 4 + 3] = 1.0f;
        api.Check(api.hipMemcpy(device_input_, encoded_.data(), pixels * 16, 1), "warm input");
        network_->Enqueue(device_input_, nullptr, device_output_, 0);
        network_->Synchronize();
        network_->PrintMemory();
        if (options_.self_test)
            dlsslop::check_gpu_controls(api, network_->Stream(), options_.modules);
    }
    // Serving only: DMA the channel's frame slots directly (see GpuCodec::pin).
    void pin(uint8_t* input, uint8_t* output, size_t bytes)
    {
        if (gpu_codec_) gpu_codec_->pin(input, output, bytes);
    }
    // Input and output are w * h RGBA8, or RGBA16F with settings.fp16. verify
    // checks the GPU codec against the CPU reference inside the timed frame.
    void infer(const uint8_t* input, unsigned w, unsigned h, uint8_t* output, unsigned passes,
               const ProcessingSettings& settings = {}, dlsslop::FrameTrace* trace = nullptr,
               bool verify = false)
    {
        if (!passes || passes > kMaxPasses)
            throw std::invalid_argument("invalid neural pass count");
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
        const auto g = dlsslop::geometry(w, h, options_.tier);
        if (g.width != width_ || g.height != height_)
            throw std::runtime_error("codec/network geometry mismatch");
        auto& api = network_->Runtime();
        std::vector<float> trace_buffer;
        const auto trace_image = [&](const std::string& name, void* pointer, unsigned channels) {
            if (!trace) return;
            network_->Synchronize();
            trace_buffer.resize(std::size_t(g.width) * g.height * channels);
            api.Check(api.hipMemcpy(trace_buffer.data(), pointer,
                trace_buffer.size() * sizeof(float), 2), "read diagnostic neural stage");
            trace->image(name, trace_buffer.data(), g, channels);
        };
        if (settings.fp16 && options_.cpu_compose)
            throw std::range_error("FP16 proxy transport requires Vulkan composition; disable --cpu-compose");
        const bool tuned = !dlsslop::native_tuning_is_default(settings.tuning);
        if (tuned && !gpu_tuning_) {
            gpu_tuning_ = std::make_unique<dlsslop::GpuTuning>(api, network_->Stream(), options_.modules + "/linux_tuning.hsaco");
            api.Check(api.hipMalloc(&device_tuned_, size_t(width_) * height_ * 12), "allocate native tuning output");
        }
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
        if (!std::isfinite(settings.color_preserve) || settings.color_preserve < 0 || settings.color_preserve > 1)
            throw std::range_error("invalid color preservation strength");
        if (settings.color_preserve > 0) {
            if (!color_backend_checked_) {
                const auto path = options_.modules + "/linux_color.hsaco";
                if (std::filesystem::exists(path)) {
                    gpu_color_ = std::make_unique<dlsslop::GpuColor>(api, network_->Stream(), path, width_, height_);
                    std::fprintf(stderr, "color preservation backend: GPU (HIP)\n");
                } else {
                    std::fprintf(stderr, "color preservation backend: CPU fallback; linux_color.hsaco missing; significant per-pass cost\n");
                }
                color_backend_checked_ = true;
            }
            if (gpu_color_) {
                gpu_color_->begin(device_input_, g);
            } else {
                network_->Synchronize();
                color_original_.resize(size_t(width_) * height_ * 4);
                api.Check(api.hipMemcpy(color_original_.data(), device_input_, color_original_.size() * sizeof(float), 2),
                          "read original color reference");
                color_raw_.resize(size_t(width_) * height_ * 3);
            }
        }
        mark(1);
        if (settings.motion) {
            if (!temporal_)
                temporal_ = std::make_unique<dlsslop::GpuTemporal>(api, network_->Stream(), options_.modules + "/linux_temporal.hsaco");
            const bool reset = options_.self_test || previous_passes_ != passes ||
                !previous_settings_.motion || previous_settings_.fp16 != settings.fp16 ||
                previous_settings_.precision16 != settings.precision16 ||
                !same_tuning(previous_settings_.tuning, settings.tuning) ||
                previous_settings_.color_preserve != settings.color_preserve;
            previous_passes_ = 0; // Until the frame completes: a rejected one leaves no history.
            // Until end(), throw no std::range_error: the worker would serve on with the
            // history still pending, and the next begin() would fail.
            temporal_->begin(device_input_, g, settings.motion_quality, settings.motion_grid,
                             settings.motion_units, passes, reset);
        } else if (temporal_) {
            temporal_->reset();
        }
        void* answer = device_output_;
        for (unsigned pass = 0; pass < passes; ++pass) {
            if (pass) {
                if (gpu_codec_) {
                    gpu_codec_->feedback(g, answer, device_input_, settings.precision16);
                    if (verify) {
                        dlsslop::feedback_neural_rgb(neural_.data(), g, feedback_, settings.precision16);
                        selftest::compare(selftest::Buffer::read_pointer(api, network_->Stream(), device_input_,
                                          feedback_.size()), feedback_, "GPU inter-pass feedback");
                        std::printf("GPU feedback for pass %u/%u vs CPU reference: FP32 bit-identical\n",
                                    pass + 1, passes);
                    }
                } else {
                    // Retain the initial encoded_ for final CPU composition.
                    dlsslop::feedback_neural_rgb(neural_.data(), g, feedback_, settings.precision16);
                    api.Check(api.hipMemcpy(device_input_, feedback_.data(), feedback_.size() * sizeof(float), 1),
                              "upload inter-pass feedback");
                }
            }
            char stage[32]{};
            if (trace) {
                std::snprintf(stage, sizeof stage, "pass-%02u", pass + 1);
                trace_image(std::string(stage) + "-input", device_input_, 4);
            }
            void* history = settings.motion ? temporal_->history(pass, device_input_) : nullptr;
            network_->Enqueue(device_input_, history, device_output_, 0);
            answer = device_output_;
            if (trace) trace_image(std::string(stage) + "-raw", answer, 3);
            if (tuned) {
                gpu_tuning_->apply(g, device_input_, device_output_, device_tuned_, settings.tuning);
                answer = device_tuned_;
                if (trace) trace_image(std::string(stage) + "-tuned", answer, 3);
            }
            if (settings.color_preserve > 0) {
                if (gpu_color_) {
                    answer = gpu_color_->apply(answer, g, settings.color_preserve);
                } else {
                    network_->Synchronize();
                    api.Check(api.hipMemcpy(color_raw_.data(), answer, color_raw_.size() * sizeof(float), 2),
                              "read per-pass color input");
                    dlsslop::preserve_color(color_original_.data(), color_raw_.data(), g,
                                          settings.color_preserve, color_result_);
                    api.Check(api.hipMemcpy(answer, color_result_.data(), color_result_.size() * sizeof(float), 1),
                              "upload color-preserved answer");
                }
                if (trace) trace_image(std::string(stage) + "-color", answer, 3);
            }
            if (settings.motion) temporal_->finish_pass(pass, answer);
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
            gpu_codec_->decode(g, answer, output);
            mark(3);
            gpu_codec_->finish();
            if (verify) {
                std::vector<uint8_t> reference;
                dlsslop::decode_neural_proxy(input, g, settings.fp16, neural_.data(), reference);
                const size_t first = std::mismatch(reference.begin(), reference.end(), output).first - reference.begin();
                if (first < reference.size()) {
                    const size_t bpp = settings.fp16 ? 8 : 4;
                    std::fprintf(stderr, "GPU decoder first mismatch: x=%zu y=%zu byte=%zu GPU=%u CPU=%u\n",
                                 (first / bpp) % w, (first / bpp) / w, first % bpp,
                                 unsigned(output[first]), unsigned(reference[first]));
                    throw std::runtime_error("GPU decoder disagrees with CPU reference");
                }
                std::printf("GPU decode vs CPU reference: bit-identical\n");
                std::fflush(stdout);
            }
        } else {
            std::vector<uint8_t> result;
            if (options_.cpu_compose)
                dlsslop::decode_rgba8(input, g, encoded_.data(), neural_.data(), result);
            else
                dlsslop::decode_neural_proxy(input, g, settings.fp16, neural_.data(), result);
            std::memcpy(output, result.data(), result.size());
            mark(3);
        }
        previous_settings_ = settings;
        previous_passes_ = passes;
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

void run_self_test(const Options& o, Engine& engine)
{
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
    const auto g = dlsslop::geometry(w, h, o.tier);
    std::vector<float> first_raw;
    std::vector<uint8_t> first_output;
    // Only the first run checks the codec against the CPU reference, so the
    // later runs time the production path; each must reproduce the first.
    for (unsigned run = 0; run < repeats; ++run) {
        engine.infer(input.data(), w, h, output.data(), o.passes, {}, nullptr, !run);
        const auto& raw = engine.raw_result();
        if (raw.size() != size_t(g.width) * g.height * 3)
            throw std::runtime_error("self-test raw network output size mismatch");
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
                    run + 1, repeats, run ? "raw FP32 and output bit-identical" : "baseline", o.passes,
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
                o.tier, o.passes, o.performance ? "upstream performance preset" : "all 71",
                engine.upload_ms, engine.inference_ms, engine.readback_ms);
}

void run_offline(const Options& o, Engine& engine)
{
    const size_t bytes = size_t(o.width) * o.height * 4;
    std::ifstream in(o.input, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(bytes))
        throw std::runtime_error("offline input size must equal width * height * 4");
    std::vector<uint8_t> pixels(bytes), result(bytes);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("read offline input");
    engine.infer(pixels.data(), o.width, o.height, result.data(), o.passes);
    std::ofstream out(o.output, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("write offline output");
    std::fprintf(stderr, "offline: passes=%u upload=%.2f ms, network=%.2f ms, readback=%.2f ms\n",
                 o.passes, engine.upload_ms, engine.inference_ms, engine.readback_ms);
}

void run_worker(const Options& o)
{
    Mapping mapping(o.shm);
    auto* h = mapping.h;
    std::unique_ptr<dlsslop::TraceRequests> traces;
    if (!o.trace_dir.empty())
        traces = std::make_unique<dlsslop::TraceRequests>(o.trace_dir, o.shm, kShmVersion);
    std::unique_ptr<dlsslop::FrameTrace> pending_trace;
    h->passes.store(o.passes);
    h->compositionBypass.store(o.cpu_compose || o.test_identity ? 1 : 0);
    // The layer must know the real neural raster before building its proxy.
    // Otherwise it mistakes a worker-upscaled answer for native-resolution
    // output and skips its detail-preserving composition branch.
    if (!o.cpu_compose && !o.test_identity) {
        h->nativeModelMaxWidth.store(o.tier == 720 ? 1280 : o.tier == 900 ? 1600 : 1920);
        h->nativeModelMaxHeight.store(o.tier);
        h->transfer.store(2); // Native frame plus the edit measured at model resolution.
    } else {
        // The mapping may retain settings from a preceding neural worker.
        h->nativeModelMaxWidth.store(0);
        h->nativeModelMaxHeight.store(0);
    }
    // Linux futex wake is emitted by the patched Vulkan layer. Timeout maintains liveness
    // with old clients and permits signals/quit; no GPU polling is involved.
    std::atomic<bool> heartbeat_stop{false};
    std::thread heartbeat([&] {
        while (!heartbeat_stop.load()) {
            h->heartbeat.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    try {
        mapping.reason(o.test_identity ? "IDENTITY TEST: inference disabled" : "initializing native HIP model");
        Engine engine(o);
        engine.prepare();
        // Only serving stops gracefully, from the ready announcement on.
        // Before it, and in every other mode, SIGINT and SIGTERM terminate.
        std::signal(SIGINT, stop_handler);
        std::signal(SIGTERM, stop_handler);
        h->modelUp.store(o.test_identity ? 0 : 1);
        h->helperFeatures.store(o.test_identity ? 0 : 1);
        h->helperState.store(kHelperRunning, std::memory_order_release);
        const char* const ready = o.test_identity ? "IDENTITY TEST: no neural rendering"
                                                  : "native HIP ready; display-encoded RGBA8/FP16 proxy";
        mapping.reason(ready);
        std::fprintf(stderr, "worker ready: %s%s\n", o.shm.c_str(), o.test_identity ? " [IDENTITY TEST]" : "");
        if (!o.test_identity)
            std::fprintf(stderr, "neural tier=%u; processing=%ux%u; %s; live controls enabled\n",
                         o.tier, o.tier == 720 ? 1280 : o.tier == 900 ? 1600 : 1920,
                         o.tier == 720 ? 768 : o.tier == 900 ? 960 : 1152,
                         o.cpu_compose ? "CPU composition" : "native-resolution Vulkan composition");
        uint64_t frames = 0;
        unsigned previous_passes = 0;
        TuningLatch tuning(h);
        uint32_t last = h->seq_resp.load(std::memory_order_acquire);
        std::string failure;
        while (!stopping && !h->quit.load(std::memory_order_relaxed)) {
            if (traces && !pending_trace) {
                try {
                    pending_trace = traces->take();
                    if (pending_trace) h->controlSeq.fetch_add(1);
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "diagnostic request rejected: %s\n", error.what());
                }
            }
            const auto active_tuning = tuning.update(h);
            const uint32_t request = h->seq_req.load(std::memory_order_acquire);
            if (request == last) {
                const timespec timeout{0, 100000000};
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_req), FUTEX_WAIT,
                        request, &timeout, nullptr, 0);
                continue;
            }
            // Writers store a setting before bumping controlSeq, so sample the
            // generations after this loop's own bumps (trace claim, tuning
            // commit) and before reading any other setting, the dimensions or
            // the input; sample again after inference.
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
                settings.motion_units = ShmMVecScaleMode(h);
                settings.tuning = active_tuning;
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
                engine.pin(mapping.input, mapping.output, bytes);
                engine.infer(mapping.input, w, height, mapping.output, passes, settings, pending_trace.get());
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
                    metadata.source_proxy_hash = 14695981039346656037ull;
                    for (size_t i = 0; i < bytes; ++i)
                        metadata.source_proxy_hash = (metadata.source_proxy_hash ^ mapping.input[i]) * 1099511628211ull;
                    metadata.passes = passes;
                    metadata.geometry = dlsslop::geometry(w, height, o.tier);
                    metadata.fp16_proxy = settings.fp16;
                    metadata.fp16_feedback = settings.precision16;
                    metadata.motion = settings.motion;
                    metadata.intensity = settings.tuning.intensity;
                    metadata.local_tone = settings.tuning.tone;
                    metadata.local_structure = settings.tuning.structure;
                    metadata.sharpness = settings.tuning.sharpness;
                    metadata.color_preserve = settings.color_preserve;
                    metadata.color_backend = settings.color_preserve > 0 ? engine.color_backend() : "off";
                    trace_metadata = metadata.json();
                }
                if (!failure.empty()) { // Serving recovered: the failure is no longer current.
                    failure.clear();
                    mapping.reason(ready);
                }
                h->answeredW.store(w);
                h->answeredH.store(height);
                h->helperEvalMsBits.store(FloatToBits(engine.inference_ms));
                h->helperUploadMsBits.store(FloatToBits(engine.upload_ms));
                h->helperReadbackMsBits.store(FloatToBits(engine.readback_ms));
                ++frames;
                h->helperFramesLo.store(static_cast<uint32_t>(frames));
                h->helperFramesHi.store(static_cast<uint32_t>(frames >> 32));
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
                    pending_trace->fail(e.what());
                    pending_trace->finish("{\"frame_seq\":" + std::to_string(request) + "}");
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
        }
    } catch (const std::exception& e) {
        mapping.reason(e.what());
        heartbeat_stop.store(true);
        heartbeat.join();
        h->helperState.store(kHelperModelFailed);
        throw;
    } catch (...) {
        heartbeat_stop.store(true);
        heartbeat.join();
        h->helperState.store(kHelperModelFailed);
        throw;
    }
    heartbeat_stop.store(true);
    heartbeat.join();
}
} // namespace

int main(int argc, char** argv)
{
    try {
        Options o = parse(argc, argv);
        if (o.diagnose) { std::fprintf(stderr, "selected device %d\n", select_device(o.device)); return 0; }
        if (o.test_identity)
            std::fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
        else
            o.device = select_device(o.device);
        if (o.self_test || !o.input.empty()) {
            Engine engine(o);
            engine.prepare();
            if (o.self_test) run_self_test(o, engine);
            else run_offline(o, engine);
        } else {
            run_worker(o);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dlsslopd: %s\n", e.what());
        return 1;
    }
}
