// SPDX-License-Identifier: MIT
#include "options.hpp"
#include "files.hpp"
#include "paths.hpp"
#include "shm_protocol.h"
#include "vulkan_network.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <sys/stat.h>
#include <vector>

namespace dlsslop {
const char* hip_only(const Options& o)
{
    // --cpu-compose sets cpu_codec too, so it goes first.
    const std::pair<bool, const char*> options[] = {
        {o.cpu_compose, "--cpu-compose"}, {o.cpu_codec, "--cpu-codec"}, {!o.trace_dir.empty(), "--trace-dir"}};
    for (const auto& [set, name] : options)
        if (set) return name;
    return nullptr;
}

Options default_options()
{
    Options o;
    o.assets = default_assets();
    o.modules = default_modules();
    o.vulkan_model = default_vulkan_model();
    return o;
}

namespace {
Result<unsigned> number(const char* text, const char* name)
{
    char* end = nullptr;
    errno = 0;
    unsigned long n = std::strtoul(text, &end, 10);
    if (errno || !*text || *end || *text == '-' || n > 100000) return fail(std::string("invalid ") + name);
    return static_cast<unsigned>(n);
}

// Every option once: its getopt spelling, whether the config file may set it
// (and whether that value is a path), and where a value lands. A flag gets
// "true" from the command line, or true or false from the config file.
struct Spec {
    const char* name;
    char letter;
    enum Kind : uint8_t { kFlag, kValue, kPath } kind;
    bool config;
    Result<void> (*apply)(Options&, const char*);
};

const Spec kSpecs[] = {
    {"config", 'f', Spec::kPath, false, nullptr}, // Read before the others apply.
    {"backend", 'b', Spec::kValue, true, [](Options& o, const char* v) -> Result<void> {
        if (std::strcmp(v, "auto") && std::strcmp(v, "vulkan") && std::strcmp(v, "hip"))
            return fail("backend must be auto, vulkan or hip");
        o.backend = v;
        return {};
    }},
    {"vulkan-model", 'M', Spec::kPath, true, [](Options& o, const char* v) -> Result<void> { o.vulkan_model = v; return {}; }},
    {"assets", 'a', Spec::kPath, true, [](Options& o, const char* v) -> Result<void> { o.assets = v; return {}; }},
    {"modules", 'm', Spec::kPath, true, [](Options& o, const char* v) -> Result<void> { o.modules = v; return {}; }},
    // The channel pairs the worker with its launcher and socket unit.
    {"shm", 's', Spec::kPath, false, [](Options& o, const char* v) -> Result<void> { o.shm = v; return {}; }},
    {"tier", 't', Spec::kValue, true, [](Options& o, const char* v) -> Result<void> {
        o.tier = DLSSLOP_TRY(number(v, "tier"));
        if (!ShmNativeTier(*o.tier)) return fail("tier must be 720, 900, or 1080");
        return {};
    }},
    {"passes", 'P', Spec::kValue, true, [](Options& o, const char* v) -> Result<void> {
        const unsigned passes = DLSSLOP_TRY(number(v, "passes"));
        if (!passes || passes > kMaxPasses) return fail("--passes must be 1.." + std::to_string(kMaxPasses));
        o.passes = passes;
        return {};
    }},
    {"device", 'd', Spec::kValue, true, [](Options& o, const char* v) -> Result<void> {
        o.device = static_cast<int>(DLSSLOP_TRY(number(v, "device")));
        return {};
    }},
    {"diagnose", 'D', Spec::kFlag, false, [](Options& o, const char*) -> Result<void> { o.diagnose = true; return {}; }},
    {"self-test", 'S', Spec::kFlag, false, [](Options& o, const char*) -> Result<void> { o.self_test = true; return {}; }},
    {"self-test-runs", 'r', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> {
        o.self_test_runs = DLSSLOP_TRY(number(v, "self-test runs"));
        return {};
    }},
    {"self-test-drops", 'X', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> {
        o.self_test_drops = DLSSLOP_TRY(number(v, "self-test drops"));
        return {};
    }},
    {"input", 'i', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> { o.input = v; return {}; }},
    {"output", 'o', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> { o.output = v; return {}; }},
    {"width", 'W', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> {
        o.width = DLSSLOP_TRY(number(v, "width"));
        return {};
    }},
    {"height", 'H', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> {
        o.height = DLSSLOP_TRY(number(v, "height"));
        return {};
    }},
    {"cpu-compose", 'c', Spec::kFlag, false, [](Options& o, const char*) -> Result<void> {
        o.cpu_compose = o.cpu_codec = true;
        return {};
    }},
    {"cpu-codec", 'C', Spec::kFlag, false, [](Options& o, const char*) -> Result<void> { o.cpu_codec = true; return {}; }},
    {"performance", 'p', Spec::kFlag, true, [](Options& o, const char* v) -> Result<void> {
        o.performance = !std::strcmp(v, "true");
        if (!o.performance && std::strcmp(v, "false")) return fail("expected true or false");
        return {};
    }},
    {"once", '1', Spec::kFlag, false, [](Options& o, const char*) -> Result<void> { o.once = true; return {}; }},
    {"idle-exit", 'x', Spec::kValue, true, [](Options& o, const char* v) -> Result<void> {
        o.idle_exit = DLSSLOP_TRY(number(v, "idle-exit seconds"));
        return {};
    }},
    {"trace-dir", 'R', Spec::kValue, false, [](Options& o, const char* v) -> Result<void> {
        if (!*v) return fail("--trace-dir requires a nonempty directory");
        o.trace_dir = v;
        return {};
    }},
    {"test-identity", 'T', Spec::kFlag, false, [](Options& o, const char*) -> Result<void> {
        o.test_identity = true;
        return {};
    }},
    {"help", 'h', Spec::kFlag, false, nullptr},
};
} // namespace

namespace {
const char* on_off(bool value) { return value ? "on" : "off"; }
} // namespace

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
        "                          Only HIP serves --cpu-compose, --cpu-codec and\n"
        "                          --trace-dir: auto takes HIP for them, vulkan\n"
        "                          refuses them. Vulkan evaluates NVIDIA's own\n"
        "                          intensity, local tone, local structure, style,\n"
        "                          skin structure and automatic mask controls and\n"
        "                          sizes itself to each frame. Its SPIR-V is in\n"
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
        "                          starting worker keeps the channel's live count.\n"
        "                          Vulkan runs at most %u and stores that count\n"
        "  -d, --device INDEX      HIP device, or with Vulkan physical device, index\n"
        "                          Default: auto, the first device that can run\n"
        "                          the network (HIP: gfx1201)\n"
        "  -D, --diagnose          Open the backend serving would use, report its\n"
        "                          device and exit; HIP also lists its devices\n"
        "                          Default: %s; exit status 1 if none is usable\n"
        "  -S, --self-test         Real model test on a deterministic gradient\n"
        "                          Also checks tuning, motion history and FP16 codec\n"
        "                          Default: %s\n"
        "  -r, --self-test-runs N  Identical-input runs, including baseline (2..1000)\n"
        "                          Default: %u; requires --self-test\n"
        "  -X, --self-test-drops N Most Vulkan runs to drop (0..999)\n"
        "                          Default: %u; requires --self-test. A run is\n"
        "                          dropped when a wait of the network ran out\n"
        "                          while other GPU work held the device\n"
        "  -i, --input FILE        Offline tightly packed RGBA8 input\n"
        "                          Default: unset; serve shared-memory requests\n"
        "  -o, --output FILE       Offline RGBA8 output; self-test PPM output\n"
        "                          Default: unset; required with --input;\n"
        "                          --self-test writes no image unless specified\n"
        "  -W, --width PIXELS      Offline image width (1..%u)\n"
        "                          Default: %u (unset); required with --input\n"
        "  -H, --height PIXELS     Offline image height (1..%u)\n"
        "                          Default: %u (unset); required with --input\n"
        "  -c, --cpu-compose       Use CPU composition and codec (layer bypass)\n"
        "                          Default: %s; Vulkan layer composes the result\n"
        "  -C, --cpu-codec         Slow CPU codec for numerical comparison\n"
        "                          Default: %s; use the GPU codec\n"
        "  -p, --performance       HIP: skip blocks 42,43,46, matching upstream preset\n"
        "                          Default: %s; evaluate all 71 blocks\n"
        "  -1, --once              Answer one shared-memory request and exit,\n"
        "                          with status 1 when that request failed\n"
        "                          Default: %s; run until stopped\n"
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
        "                          Default: %s; the backend's network runs\n"
        "  -h, --help              Show this help and exit (default: off)\n",
        config.empty() ? "unset; no home directory" : config.c_str(), settable.c_str(),
        defaults.backend.c_str(), vulkan_shaders().c_str(),
        defaults.vulkan_model.empty() ? "unset; no home directory" : defaults.vulkan_model.c_str(),
        defaults.assets.empty() ? "unset; required for inference" : defaults.assets.c_str(),
        defaults.modules.empty() ? "unset; required for inference" : defaults.modules.c_str(),
        defaults.shm.c_str(), ShmNativeDefaultPath().c_str(), kNativeDefaultTier,
        kMaxPasses, kNativeDefaultPasses, dlsslop::VulkanNetwork::kMaxPasses, on_off(defaults.diagnose),
        on_off(defaults.self_test), defaults.self_test_runs, defaults.self_test_drops, kMaxW, defaults.width,
        kMaxH, defaults.height,
        on_off(defaults.cpu_compose), on_off(defaults.cpu_codec), on_off(defaults.performance), on_off(defaults.once),
        defaults.idle_exit, on_off(defaults.test_identity));
}

namespace {
std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t");
    return first == std::string::npos ? std::string() : text.substr(first, text.find_last_not_of(" \t") - first + 1);
}

// A config file's path is absolute or starts with ~/, never relative to
// wherever the worker was started.
Result<std::string> config_path(const std::string& value)
{
    if (value[0] == '/') return value;
    const std::string home = value.compare(0, 2, "~/") ? std::string() : home_directory();
    if (home.empty()) return fail("a path must be absolute or start with ~/");
    return home + value.substr(1);
}

// One NAME = VALUE line, NAME a config-settable long option.
Result<void> apply_setting(Options& o, const std::string& text)
{
    const auto equals = text.find('=');
    if (equals == std::string::npos) return fail("expected NAME = VALUE");
    const std::string name = trim(text.substr(0, equals));
    const auto spec = std::find_if(std::begin(kSpecs), std::end(kSpecs),
                                   [&name](const Spec& s) { return s.config && name == s.name; });
    if (spec == std::end(kSpecs)) return fail("'" + name + "' is not a config setting");
    const std::string value = trim(text.substr(equals + 1));
    if (spec->kind != Spec::kPath) return spec->apply(o, value.c_str());
    return spec->apply(o, DLSSLOP_TRY(config_path(value)).c_str());
}

// Settings, one per line; # starts a comment line. A default file that is
// missing is skipped; one that cannot be inspected or read is an error.
Result<void> read_config(Options& o, const std::string& path, bool given)
{
    struct stat st{};
    if (!given && (path.empty() || (stat(path.c_str(), &st) && (errno == ENOENT || errno == ENOTDIR)))) return {};
    const auto text = read_file(path);
    if (!text) return fail("cannot read config file " + path + ": " + text.error().what);
    unsigned line_number = 0;
    for (size_t start = 0; start < text->size(); ++line_number) {
        const size_t end = std::min(text->find('\n', start), text->size());
        const std::string line = trim(text->substr(start, end - start));
        start = end + 1;
        if (line.empty() || line[0] == '#') continue;
        if (auto applied = apply_setting(o, line); !applied)
            return fail(path + ":" + std::to_string(line_number + 1) + ": " + applied.error().what);
    }
    return {};
}
} // namespace

Result<std::string> configured_vulkan_model()
{
    Options o = default_options();
    DLSSLOP_TRY(read_config(o, default_config(), false));
    return o.vulkan_model;
}

Result<Options> parse(int argc, char** argv)
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
            return fail("invalid arguments");
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
    if (optind != argc) return fail("unexpected positional argument");
    Options o = default_options();
    DLSSLOP_TRY(read_config(o, config, config_given));
    for (const auto& [spec, value] : given) DLSSLOP_TRY(spec->apply(o, value));
    const auto on_command_line = [&given](char letter) {
        return std::any_of(given.begin(), given.end(), [letter](const auto& g) { return g.first->letter == letter; });
    };
    if (o.self_test_runs < 2 || o.self_test_runs > 1000)
        return fail("--self-test-runs must be 2..1000");
    if (on_command_line('r') && !o.self_test)
        return fail("--self-test-runs requires --self-test");
    if (o.self_test_drops > 999)
        return fail("--self-test-drops must be 0..999");
    if (on_command_line('X') && !o.self_test)
        return fail("--self-test-drops requires --self-test");
    if (!o.diagnose && !o.test_identity && (o.assets.empty() || o.modules.empty()))
        return fail("--assets and --modules are required for inference");
    if (o.self_test && (o.test_identity || !o.input.empty()))
        return fail("--self-test requires the real network and generates its own input");
    if (!o.self_test && (o.input.empty() != o.output.empty()))
        return fail("--input and --output must be supplied together");
    if (!o.input.empty() && (!o.width || !o.height || o.width > kMaxW || o.height > kMaxH))
        return fail("offline dimensions must be 1..7680 by 1..4320");
    if (!o.trace_dir.empty() && (o.self_test || !o.input.empty() || o.test_identity || o.diagnose))
        return fail("--trace-dir requires serving real shared-memory inference");
    // A configured idle exit is for serving; only an explicit one is an error.
    if (on_command_line('x') && o.idle_exit && (o.self_test || !o.input.empty() || o.diagnose))
        return fail("--idle-exit requires serving shared-memory requests");
    if (const char* hip = hip_only(o); hip && o.backend == "vulkan")
        return fail(std::string(hip) + " requires --backend hip");
    return o;
}
} // namespace dlsslop
