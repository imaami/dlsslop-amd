// Native Linux controls. Shared-memory offsets and defaults come from the protocol.
#include "control_settings.h"

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <getopt.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using dlsslop_control::Setting;
using dlsslop_control::kSettings;

// --capture parses like an integer setting.
constexpr Setting kCapture{"capture", &ShmHeader::captureRequest, 0, 64};

constexpr std::size_t kCount = std::size(kSettings);
static_assert(kCount <= 64, "Options keeps one mask bit per setting");

uint64_t Bit(const Setting* setting) { return uint64_t(1) << (setting - kSettings); }

// "; steps of N" for a setting whose values skip, as help and errors print it.
std::string Steps(const Setting& s) {
    char text[32] = "";
    if (s.step != 1) std::snprintf(text, sizeof text, "; steps of %g", s.step);
    return text;
}

struct Options {
    std::string path;
    uint64_t assigned = 0;  // Bit(setting): values holds its new value
    uint64_t toggled = 0;   // Bit(setting): flip it once
    uint32_t values[kCount] = {};
    uint32_t capture = 0;
    bool help = false;
    bool status = false;
    bool settings = false;
    bool reset = false;
    bool quit = false;
    bool resume = false;
    bool captureSet = false;

    bool changesHeader() const {
        return assigned || toggled || reset || quit || resume || captureSet;
    }
};

void Usage() {
    ShmHeader defaults{};
    ShmInitNativeDefaults(&defaults, false);
    std::printf(
        "Usage: dlsslopctl [OPTION]...\n"
        "       dlssnr-shmctl [OPTION]...\n"
        "\n"
        "Omitted setting options leave current values unchanged. The setting defaults\n"
        "below are initial/reset values, not the current values. Use --settings to\n"
        "inspect both. With no action or setting option, --status is the default.\n"
        "All arguments are validated before opening or changing the channel.\n"
        "\n"
        "  -s, --shm PATH          Channel path (default: nonempty $DLSSNR_SHM, otherwise\n"
        "                         /tmp/dlsslop-amd-UID/shm.bin; effective default: %s)\n"
        "  -S, --status            Print live transport/status values (default: on if no action/setting)\n"
        "  -l, --settings          Print live settings and reset defaults (default: off)\n"
        "  -r, --reset             Restore supported settings before explicit changes (default: off)\n"
        "  -q, --quit              Request worker/layer stop (default: off; initial quit flag: 0)\n"
        "  -R, --resume            Clear stop request; does not restart the worker (default: off)\n"
        "  -c, --capture N         Request %g..%g matched before/after frames; 0 clears a pending request\n"
        "                         (default: unchanged; initial: 0; active captures continue)\n"
        "  -A, --toggle NAME       Toggle a boolean setting once, such as enabled from a key binding;\n"
        "                         repeat for different settings (default: none)\n"
        "  -h, --help              Print help without accessing the channel (default: off)\n"
        "\nSettings (all require a value):\n", ShmNativeChannelPath().c_str(), kCapture.minimum, kCapture.maximum);
    for (const auto& s : kSettings)
        std::printf("  -%c, --%-18s VALUE  %s\n"
                    "                                Range: %g..%g; default: %g%s\n",
                    s.shortName, s.name, s.help, s.minimum, s.maximum,
                    dlsslop_control::value(s, (defaults.*s.field).load()), Steps(s).c_str());
    std::printf(
        "\n"
        "Actions run after parsing: reset, explicit settings, toggle, stop/resume, capture;\n"
        "requested status/settings are printed last. Repeated settings use the last value.\n"
        "--quit and --resume conflict. A setting cannot be both assigned and toggled.\n"
        "Changes require an existing channel parent directory; the default private\n"
        "directory is created automatically. Settings persist in the shared-memory\n"
        "channel, not a configuration file; worker startup may reapply its own settings.\n"
        "\n"
        "dlsslopd supports SDR, HDR and successive neural passes; --passes changes the next\n"
        "frame's inference chain. Each added pass adds another network evaluation.\n"
        "On Vulkan, intensity, local tone/structure, style, skin structure and the automatic\n"
        "mask are the model's own controls. On HIP, intensity and local tone/structure are\n"
        "residual operations after each network pass, and style, skin structure and the\n"
        "automatic mask keep their defaults. Both sharpen and preserve color after each\n"
        "pass. --preset is fixed.\n"
        "White-point controls require linear-light input.\n"
        "--hdr-mode changes proxy precision; forcing 16-bit does not reinterpret SDR as HDR.\n"
        "HDR normalization follows --color-mode even with an 8-bit proxy.\n"
        "Motion estimates come from the presented frames, not engine motion vectors.\n"
        "--tier rebuilds the daemon's network for another raster, which takes under a second;\n"
        "the game presents its own frames meanwhile. Setting the active tier does nothing.\n"
        "Composition controls apply to the Vulkan composition path; --cpu-compose\n"
        "uses the worker's own composition. Layer environment overrides take precedence\n"
        "over --ratio-smooth (DLSSNR_RATIO_SMOOTH), --color-trust (DLSSNR_COLOUR_TRUST)\n"
        "and --toggle-key (DLSSNR_TOGGLE_KEY).\n"
        "\n"
        "Examples:\n"
        "  dlsslopctl --working-scale 1 --transfer 2\n"
        "  dlsslopctl --passes 2\n"
        "  dlsslopctl -e 0\n"
        "  dlsslopctl --tier 1080\n"
        "  dlsslopctl --reset --settings\n");
}

// strtod/strtoull skip leading whitespace, strtoull accepts signs and strtod
// parses nothing from an empty string. Reject all of them rather than silently
// coercing malformed arguments.
bool ParseValue(const char* text, const Setting& setting, uint32_t& result) {
    char* end = nullptr;
    errno = 0;
    if (setting.isFloat) {
        if (!*text || std::isspace(static_cast<unsigned char>(*text))) return false;
        const double value = std::strtod(text, &end);
        if (errno || *end || !dlsslop_control::inRange(setting, value)) return false;
        result = FloatToBits(static_cast<float>(value) + 0.0f);  // + 0.0f stores -0 as 0
    } else {
        if (*text < '0' || *text > '9') return false;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (errno || *end || !dlsslop_control::inRange(setting, value)) return false;
        result = static_cast<uint32_t>(value);
    }
    return true;
}

const Setting* FindSetting(const char* name) {
    for (const auto& setting : kSettings)
        if (!std::strcmp(name, setting.name)) return &setting;
    return nullptr;
}

bool ParseOptions(int argc, char** argv, Options& options) {
    options.path = ShmNativeChannelPath();
    std::vector<option> longOptions = {
        {"shm", required_argument, nullptr, 's'},
        {"status", no_argument, nullptr, 'S'},
        {"settings", no_argument, nullptr, 'l'},
        {"reset", no_argument, nullptr, 'r'},
        {"quit", no_argument, nullptr, 'q'},
        {"resume", no_argument, nullptr, 'R'},
        {"capture", required_argument, nullptr, 'c'},
        {"toggle", required_argument, nullptr, 'A'},
        {"help", no_argument, nullptr, 'h'},
    };
    std::string shortOptions = "+s:SlrqRc:A:h";
    for (const auto& s : kSettings) {
        longOptions.push_back({s.name, required_argument, nullptr, s.shortName});
        shortOptions += s.shortName;
        shortOptions += ':';
    }
    longOptions.push_back({nullptr, 0, nullptr, 0});
    int code;
    while ((code = getopt_long(argc, argv, shortOptions.c_str(), longOptions.data(), nullptr)) != -1) {
        switch (code) {
        case 's':
            if (!*optarg) {
                std::fprintf(stderr, "--shm requires a nonempty path\n");
                return false;
            }
            options.path = optarg;
            break;
        case 'S': options.status = true; break;
        case 'l': options.settings = true; break;
        case 'r': options.reset = true; break;
        case 'q': options.quit = true; break;
        case 'R': options.resume = true; break;
        // Repeating a toggle names the same action; it does not cancel itself out.
        case 'A': {
            const Setting* setting = FindSetting(optarg);
            if (!setting || setting->isFloat || setting->minimum != 0 || setting->maximum != 1) {
                std::fprintf(stderr, "--toggle requires a boolean setting name, got '%s'\n", optarg);
                return false;
            }
            options.toggled |= Bit(setting);
            break;
        }
        case 'h': options.help = true; break;
        case 'c':
            if (!ParseValue(optarg, kCapture, options.capture)) {
                std::fprintf(stderr, "--capture: expected an integer from %g through %g, got '%s'\n",
                             kCapture.minimum, kCapture.maximum, optarg);
                return false;
            }
            options.captureSet = true;
            break;
        case '?': return false;  // getopt has named the bad argument
        default: {  // any other code getopt returns is a setting's short name
            const Setting* setting = kSettings;
            while (setting->shortName != code) ++setting;
            uint32_t& value = options.values[setting - kSettings];
            if (!ParseValue(optarg, *setting, value)) {
                std::fprintf(stderr, "--%s: expected a finite %s in [%g, %g]%s, got '%s'\n",
                             setting->name, setting->isFloat ? "number" : "integer",
                             setting->minimum, setting->maximum, Steps(*setting).c_str(), optarg);
                return false;
            }
            options.assigned |= Bit(setting);
            break;
        }
        }
    }
    if (optind < argc) {
        std::fprintf(stderr, "unexpected positional argument '%s'; use options such as --status or --working-scale 1\n",
                     argv[optind]);
        return false;
    }
    if (options.quit && options.resume) {
        std::fprintf(stderr, "--quit and --resume cannot be combined\n");
        return false;
    }
    if (const uint64_t both = options.assigned & options.toggled) {
        const char* name = kSettings[__builtin_ctzll(both)].name;
        std::fprintf(stderr, "--toggle %s and --%s cannot be combined\n", name, name);
        return false;
    }
    if (!options.changesHeader() && !options.settings) options.status = true;
    return true;
}

bool Initialised(const ShmHeader* h) {
    return h->magic.load() == kShmMagic && h->version.load() == kShmVersion;
}

bool CreateDefaultDirectory(const std::string& path) {
    if (path != ShmNativeDefaultPath()) return true;
    const std::string directory = path.substr(0, path.rfind('/'));
    if (mkdir(directory.c_str(), 0700) && errno != EEXIST) return false;
    struct stat st {};
    if (lstat(directory.c_str(), &st)) return false;
    if (!S_ISDIR(st.st_mode) || st.st_uid != getuid() || (st.st_mode & 0077)) {
        errno = EACCES;
        return false;
    }
    return true;
}

// Readers never create, resize, initialise or write the channel. The mapping
// outlives the descriptor.
ShmHeader* MapHeader(const char* path, bool create, bool writable) {
    const int flags = (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0);
    const int fd = open(path, flags, 0600);
    if (fd < 0) return nullptr;
    const auto fail = [fd](int error) -> ShmHeader* { close(fd); errno = error; return nullptr; };
    struct stat st {};
    if (fstat(fd, &st)) return fail(errno);
    if (!S_ISREG(st.st_mode) || st.st_uid != getuid()) return fail(EACCES);
    if (writable) {
        // Never chmod, grow or initialise a file that is not a channel. A new or
        // grown file reads as zero, and initialisation clears the header before
        // it stores the magic, so a channel's header has the magic or is zero.
        uint32_t head[sizeof(ShmHeader) / 4] = {}, bits = 0;
        if (pread(fd, head, sizeof head, 0) < 0) return fail(errno);
        for (const uint32_t word : head) bits |= word;
        if (head[0] != kShmMagic && bits) {
            std::fprintf(stderr, "'%s' is not a dlsslop channel; refusing to modify it\n", path);
            return fail(EINVAL);
        }
        if (fchmod(fd, 0600)) return fail(errno);
    }
    if (st.st_size < static_cast<off_t>(ShmTotalBytes())) {
        if (!create) return fail(EINVAL);
        if (ftruncate(fd, static_cast<off_t>(ShmTotalBytes()))) return fail(errno);
    }
    void* mapping = mmap(nullptr, kHeaderBytes, PROT_READ | (writable ? PROT_WRITE : 0), MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) return fail(errno);
    close(fd);
    return static_cast<ShmHeader*>(mapping);
}

// Writers initialise a new channel and, like the worker and the layer,
// re-initialise one left by another protocol version. A process still on
// that version misreads the new layout and re-initialises it back when it
// next attaches, so say so.
bool Attach(ShmHeader* h, bool create, const char* path) {
    if (Initialised(h)) return true;
    if (!create) {
        std::fprintf(stderr, "channel '%s' is uninitialised or incompatible: magic %#x version %u, expected %#x version %u\n",
                     path, h->magic.load(), h->version.load(), kShmMagic, kShmVersion);
        return false;
    }
    if (h->magic.load() == kShmMagic)
        std::fprintf(stderr, "channel '%s' held protocol v%u and is re-initialised as v%u; restart any "
                     "dlsslopd, game or GUI still using it\n", path, h->version.load(), kShmVersion);
    ShmInitNativeDefaults(h, false);
    return true;
}

void PrintSettings(const ShmHeader* h) {
    ShmHeader defaults{};
    ShmInitNativeDefaults(&defaults, dlsslop_control::workerBypass(h));
    std::printf("# Live values; defaults are the initial/reset values for this worker mode.\n");
    // Nine digits round-trip every binary32 and ten print every uint32 exactly.
    for (const auto& s : kSettings) {
        const int digits = s.isFloat ? 9 : 10;
        std::printf("%s=%.*g default=%.*g\n", s.name, digits, dlsslop_control::value(s, (h->*s.field).load()),
                    digits, dlsslop_control::value(s, (defaults.*s.field).load()));
    }
}

void PrintStatus(const ShmHeader* h) {
    std::printf("initialised=%d\n", Initialised(h) ? 1 : 0);
    std::printf("magic=%#x\nversion=%u\n", h->magic.load(), h->version.load());
    if (!Initialised(h)) return;
    std::printf("seq_req=%u\nseq_resp=%u\n", h->seq_req.load(), h->seq_resp.load());
    std::printf("width=%u\nheight=%u\n", h->width.load(), h->height.load());
    std::printf("neural_max_width=%u\nneural_max_height=%u\n",
                h->nativeModelMaxWidth.load(), h->nativeModelMaxHeight.load());
    std::printf("passes=%u\npass_ceiling=%u\n", h->passes.load(), ShmPassCeiling(h));
    std::printf("display_width=%u\ndisplay_height=%u\n",
                h->layerWidth.load(), h->layerHeight.load());
    std::printf("quit=%u\nheartbeat=%u\ncontrol_seq=%u\ntuning_seq=%u\n", h->quit.load(), h->heartbeat.load(),
                h->controlSeq.load(), h->tuningSeq.load());
    std::printf("helper_state=%u\nmodel_up=%u\nhelper_frames=%llu\n", h->helperState.load(),
                h->modelUp.load(),
                (unsigned long long) ShmLoad64(&h->helperFramesLo, &h->helperFramesHi));
    std::printf("upload_ms=%.3f\nnetwork_ms=%.3f\nreadback_ms=%.3f\n",
                double(BitsToFloat(h->helperUploadMsBits.load())),
                double(BitsToFloat(h->helperEvalMsBits.load())),
                double(BitsToFloat(h->helperReadbackMsBits.load())));
    std::printf("layer_pid=%u\nlayer_composition_up=%u\nlayer_frames=%llu\nlayer_ms=%.2f\n",
                h->layerPid.load(),
                h->layerCompositionUp.load(),
                (unsigned long long) ShmLoad64(&h->layerFramesLo, &h->layerFramesHi),
                double(BitsToFloat(h->layerMsBits.load())));
    std::printf("measured_white_point=%g\n", double(BitsToFloat(h->layerMeasuredWhiteBits.load())));
    std::printf("hdr_mode=%u\nhdr_detected=%u\nhdr_active=%u\nproxy_format=%u\nhdr_encode=%u\n",
                h->hdrMode.load(), h->hdrDetected.load(), h->hdrActive.load(),
                h->proxyFormat.load(), h->hdrEncode.load());
    const std::string reason = ShmLoadString(h->helperReasonSeq, h->helperReason, kReasonBytes);
    if (!reason.empty()) std::printf("helper_reason=%s\n", reason.c_str());
    const std::string layer = ShmLoadString(h->layerReasonSeq, h->layerReason, kReasonBytes);
    if (!layer.empty()) std::printf("layer_reason=%s\n", layer.c_str());
}


} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!ParseOptions(argc, argv, options)) {
        std::fprintf(stderr, "Try --help for options, ranges and defaults.\n");
        return 2;
    }
    if (options.help) {
        Usage();
        return 0;
    }
    const bool writable = options.changesHeader();
    const bool create = writable && !options.quit;
    if (create && !CreateDefaultDirectory(options.path)) {
        std::fprintf(stderr, "cannot create private channel directory: %s\n", std::strerror(errno));
        return 1;
    }
    ShmHeader* h = MapHeader(options.path.c_str(), create, writable);
    if (!h) {
        if (options.quit && errno == ENOENT && !options.status && !options.settings &&
            !options.assigned && !options.toggled && !options.reset && !options.captureSet) return 0;
        std::fprintf(stderr, "cannot access channel '%s': %s\n", options.path.c_str(), std::strerror(errno));
        return 1;
    }
    if (!Attach(h, create, options.path.c_str())) return 1;
    // Store each setting once: readers never wait on controlSeq, so a second
    // store would show them a value nobody asked for. Reset copies controls
    // only and never memsets a live transport header.
    ShmHeader defaults{};
    if (options.reset) ShmInitNativeDefaults(&defaults, dlsslop_control::workerBypass(h));
    bool tuningChanged = options.reset;
    for (std::size_t i = 0; i < kCount; ++i) {
        const uint64_t bit = uint64_t(1) << i;
        const uint32_t flip = (options.toggled & bit) != 0;
        const Setting& s = kSettings[i];
        auto& field = h->*s.field;
        if (options.assigned & bit) {  // never also toggled
            field.store(options.values[i]);
            tuningChanged |= s.tuning;
        } else if (options.reset) field.store((defaults.*s.field).load() ^ flip);
        else if (flip) for (uint32_t value = field.load(); !field.compare_exchange_weak(value, !value);) {}
    }
    if (tuningChanged) h->tuningSeq.fetch_add(1);
    if (options.quit) h->quit.store(1);
    if (options.resume) h->quit.store(0);
    // Every change, a capture included, publishes a new control generation.
    const uint32_t publishedControl = writable ? h->controlSeq.fetch_add(1) + 1u : 0u;
    // Publish capture only after its settings and generation are visible.
    // The layer records this generation with the matched frame pair.
    if (options.captureSet) h->captureRequest.store(options.capture, std::memory_order_release);
    if (options.status) {
        PrintStatus(h);
        if (options.captureSet) std::printf("capture_control_seq=%u\n", publishedControl);
    }
    if (options.settings) PrintSettings(h);
    return 0;
}
