// SPDX-License-Identifier: MIT
// The smallest fake layer, for tracing served frames (tests/hiptrace/trace.sh
// and tests/vktrace/trace.sh): creates a private channel, starts dlsslopd on
// it, publishes RGBA8 or FP16 frames the way the layer's ShmProcessFrame does,
// changes settings between frames as dlsslopctl does, and prints the FNV-1a
// 64 of every input and answer. No device-local transport is offered.
#include "control_settings.h"
#include "shm_channel.h"
#include "shm_protocol.hpp"

#include <fcntl.h>
#include <getopt.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

uint64_t fnv(const uint8_t* p, size_t n)
{
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

void wake(std::atomic<uint32_t>& word)
{
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

// The self-test's gradient (backend/self_test.cpp), varied per letter.
void gradient(uint8_t* rgba, unsigned w, unsigned h, unsigned letter)
{
    const uint8_t on = uint8_t(192 + 8 * letter), off = uint8_t(64 - 8 * letter);
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
            uint8_t* p = &rgba[(size_t(y) * w + x) * 4];
            p[0] = uint8_t((x + 16 * letter) * 255 / (w - 1 + 16 * letter));
            p[1] = uint8_t(y * 255 / (h - 1));
            p[2] = (((x + 8 * letter) / 32 ^ y / 32) & 1) ? on : off;
            p[3] = 255;
        }
}

// The flat frame f: (1, 0.25, 0.75, 1) as RGBA8 UNORM.
constexpr uint8_t kFlat[4] = {255, 64, 191, 255};

// The FP16 proxy of an RGBA8 frame: the same display-encoded values as binary16.
void widen(uint8_t* out, const uint8_t* rgba, size_t values)
{
    for (size_t i = 0; i < values; ++i) {
        const auto v = static_cast<_Float16>(float(rgba[i]) / 255.0f);
        std::memcpy(out + i * sizeof v, &v, sizeof v);
    }
}

// A setting's new value, stored before frame FRAME is sent (0: before the daemon starts).
struct Change {
    size_t frame;
    const control_setting* setting;
    uint32_t value;
};

constexpr const char* kUsage = "Usage: shmclient [OPTIONS] -- DLSSLOPD [ARGUMENTS...]\n";
constexpr const char* kDefaultFrames = "AAB";
constexpr const char* kDefaultLog = "/dev/null";
constexpr char kLetters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZf";

int usage(const char* problem)
{
    std::fprintf(stderr, "shmclient: %s\n%s(see --help)\n", problem, kUsage);
    return 2;
}

void help()
{
    std::array<uint32_t, CONTROL_SETTING_COUNT> defaults;
    control_settings_defaults(defaults.data(), false);
    std::printf("%s", kUsage);
    std::printf(
        "Create the channel FILE, start DLSSLOPD, which must serve FILE (--shm FILE),\n"
        "send it the frames of --frames and print the FNV-1a 64 of every input and\n"
        "answer. Exit status 0 when the daemon exits with 0 and answers every frame,\n"
        "2 on a usage error, otherwise 1.\n"
        "\n"
        "  -s, --shm FILE       Channel to create, in a private (0700) directory\n"
        "                       Default: unset; required\n"
        "  -W, --width PIXELS   Source width (2..%u)\n"
        "                       Default: unset; required\n"
        "  -H, --height PIXELS  Source height (2..%u)\n"
        "                       Default: unset; required\n"
        "  -f, --frames SPEC    One letter per frame naming its input: A to Z the\n"
        "                       self-test's gradient, shifted by the letter; f the\n"
        "                       flat colour (%u, %u, %u, %u)\n"
        "                       Default: %s\n"
        "  -F, --fp16           Send FP16 proxies of the same values (hdrEncode 1)\n"
        "                       Default: off; RGBA8\n"
        "  -d, --dump DIR       Write every answer to DIR/answer-N.rgba8, or\n"
        "                       DIR/answer-N.rgba16f with --fp16 (N from 0)\n"
        "                       Default: unset; no files\n"
        "  -l, --log FILE       The daemon's standard output and error\n"
        "                       Default: %s\n"
        "  -h, --help           Show this help and exit\n"
        "                       Default: off\n"
        "\n"
        "Settings, dlsslopctl's with their ranges and channel defaults, take\n"
        "[FRAME:]VALUE and may be repeated: VALUE is stored before frame FRAME (from\n"
        "0) is sent, and before the daemon starts when FRAME is 0 or omitted. The\n"
        "daemon follows a tier only between requests, so for a --tier from a later\n"
        "frame the client waits until the daemon publishes the new neural raster\n"
        "before it sends that frame; a daemon that publishes none (--test-identity)\n"
        "is refused as a usage error.\n",
        kMaxW, kMaxH, kFlat[0], kFlat[1], kFlat[2], kFlat[3], kDefaultFrames, kDefaultLog);
    for (size_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
        const control_setting& s = CONTROL_SETTINGS[i];
        std::printf("      --%s [FRAME:]VALUE\n"
                    "                       %s\n"
                    "                       Range: %g..%g; default: %g\n",
                    s.name, s.help, s.minimum, s.maximum, control_setting_value(&s, defaults[i]));
    }
}

// The exit status a shell reports for the wait status STATUS.
int exit_code(int status) { return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status); }

// The decimal number TEXT when it is one in [1, LIMIT]; otherwise 0.
unsigned number(const char* text, unsigned limit)
{
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return *text >= '0' && *text <= '9' && !*end && value <= limit ? unsigned(value) : 0;
}

// [FRAME:]VALUE of SETTING, parsed as dlsslopctl parses VALUE.
bool change(const char* text, const control_setting& setting, Change& out)
{
    char* end = nullptr;
    out = {0, &setting, 0};
    if (*text >= '0' && *text <= '9') {
        errno = 0;
        const unsigned long long frame = std::strtoull(text, &end, 10);
        if (errno) return false;
        if (*end == ':') {
            out.frame = frame;
            text = end + 1;
        }
    }
    if (setting.is_float) {
        if (!*text || std::isspace(static_cast<unsigned char>(*text))) return false;
        errno = 0;
        const double value = std::strtod(text, &end);
        if (errno || *end || !control_setting_in_range(&setting, value)) return false;
        out.value = FloatToBits(float(value) + 0.0f);
        return true;
    }
    if (*text < '0' || *text > '9') return false;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    out.value = uint32_t(value);
    return !errno && !*end && control_setting_in_range(&setting, double(value));
}

// Stores the changes for frame FRAME, publishes them as dlsslopctl does and
// prints them.
void apply(ShmHeader* h, const std::vector<Change>& changes, size_t frame)
{
    bool any = false, tuning = false;
    for (const Change& c : changes)
        if (c.frame == frame) {
            control_setting_store(h, c.setting, c.value);
            std::printf("%s%s=%g", any ? " " : "settings: ", c.setting->name,
                        control_setting_value(c.setting, c.value));
            any = true;
            tuning |= c.setting->tuning;
        }
    if (!any) return;
    std::printf(" before frame %zu\n", frame);
    if (tuning) h->tuningSeq.fetch_add(1);
    h->controlSeq.fetch_add(1);
}

// The tier stored before frame FRAME, or 0 when there is none.
uint32_t tier_at(const std::vector<Change>& changes, size_t frame)
{
    uint32_t tier = 0;
    for (const Change& c : changes)
        if (c.frame == frame && c.setting->offset == offsetof(ShmHeader, nativeTier)) tier = c.value;
    return tier;
}
} // namespace

int main(int argc, char** argv)
{
    std::string shm, frames = kDefaultFrames, log = kDefaultLog, dump;
    unsigned width = 0, height = 0;
    bool fp16 = false;
    std::vector<Change> changes;
    // The client's own options, then one long option per setting, whose
    // getopt code is 256 plus its index in CONTROL_SETTINGS.
    std::vector<option> options = {{"shm", required_argument, nullptr, 's'},
                                   {"width", required_argument, nullptr, 'W'},
                                   {"height", required_argument, nullptr, 'H'},
                                   {"frames", required_argument, nullptr, 'f'},
                                   {"fp16", no_argument, nullptr, 'F'},
                                   {"dump", required_argument, nullptr, 'd'},
                                   {"log", required_argument, nullptr, 'l'},
                                   {"help", no_argument, nullptr, 'h'}};
    for (const control_setting& s : CONTROL_SETTINGS)
        options.push_back({s.name, required_argument, nullptr, int(256 + (&s - CONTROL_SETTINGS))});
    options.push_back({nullptr, 0, nullptr, 0});
    for (int c; (c = getopt_long(argc, argv, "+s:W:H:f:Fd:l:h", options.data(), nullptr)) != -1;) {
        switch (c) {
        case 's': shm = optarg; break;
        case 'W': width = number(optarg, kMaxW); break;
        case 'H': height = number(optarg, kMaxH); break;
        case 'f': frames = optarg; break;
        case 'F': fp16 = true; break;
        case 'd': dump = optarg; break;
        case 'l': log = optarg; break;
        case 'h': help(); return 0;
        default: {
            const size_t index = size_t(c) - 256;
            if (index >= CONTROL_SETTING_COUNT) return usage("unknown option or missing value");
            const control_setting& s = CONTROL_SETTINGS[index];
            Change parsed;
            if (!change(optarg, s, parsed)) {
                std::fprintf(stderr, "shmclient: --%s: expected [FRAME:]VALUE with VALUE in [%g, %g], got '%s'\n",
                             s.name, s.minimum, s.maximum, optarg);
                return 2;
            }
            changes.push_back(parsed);
        }
        }
    }
    if (shm.empty()) return usage("--shm is required");
    if (width < 2 || height < 2) return usage("--width and --height are required and must be in range");
    if (frames.empty() || frames.find_first_not_of(kLetters) != std::string::npos)
        return usage("--frames must be letters from A to Z, or f");
    for (const Change& c : changes)
        if (c.frame >= frames.size()) return usage("a setting's FRAME is not one of --frames");
    if (optind >= argc) return usage("DLSSLOPD is required");

    // A new channel, created as every program creates one.
    unlink(shm.c_str());
    shm_channel channel{nullptr, 0, -1, 0};
    error e;
    if (shm_channel_open(&channel, shm.c_str(), shm.size(), ShmTotalBytes(), SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE,
                         &e)) {
        std::fprintf(stderr, "shmclient: %s\n", e.what);
        return 1;
    }
    // C created the header's objects before the file had a name; they live in the mapping for C++ too.
    auto* h = std::start_lifetime_as<ShmHeader>(channel.h);
    apply(h, changes, 0);
    uint8_t* const input = static_cast<uint8_t*>(static_cast<void*>(channel.h)) + kHeaderBytes;
    const uint8_t* const output = input + kMaxFrame;

    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        return 1;
    }
    if (!pid) {
        const int out = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (out < 0 || dup2(out, 1) < 0 || dup2(out, 2) < 0) _exit(127);
        execv(argv[optind], argv + optind);
        _exit(127);
    }
    int status = 0;
    const auto stop = [&](int code) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        return code;
    };
    // 0 once DONE holds; otherwise 1, with the daemon reaped or stopped.
    const auto await = [&](auto done, const char* what) {
        for (const auto deadline = Clock::now() + std::chrono::seconds(300); !done();) {
            if (waitpid(pid, &status, WNOHANG) == pid) {
                std::fprintf(stderr, "daemon exited before %s (status %d)\n", what, exit_code(status));
                return 1;
            }
            if (Clock::now() > deadline) {
                std::fprintf(stderr, "daemon timed out before %s\n", what);
                return stop(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return 0;
    };
    const auto running = [h] { return h->helperState.load(std::memory_order_acquire) == kHelperRunning; };
    if (await(running, "it was ready")) return 1;
    std::printf("ready tier=%u state=%u fp16=%d\n", h->nativeTier.load(), h->helperState.load(), int(fp16));
    // Without a published raster, nothing shows when the daemon has followed a tier.
    if (!h->nativeModelMaxHeight.load())
        for (size_t index = 1; index < frames.size(); ++index)
            if (tier_at(changes, index)) {
                std::fprintf(stderr,
                             "shmclient: --tier from frame %zu needs a daemon that publishes its neural raster, and "
                             "this one publishes none\n",
                             index);
                return stop(2);
            }

    const size_t pixels = size_t(width) * height, bytes = pixels * (fp16 ? 8 : 4);
    std::vector<uint8_t> rgba(pixels * 4), wide(fp16 ? bytes : 0);
    const uint8_t* const image = fp16 ? wide.data() : rgba.data();
    int failures = 0;
    for (size_t index = 0; index < frames.size(); ++index) {
        if (index) {
            apply(h, changes, index);
            // The daemon follows a tier only between requests: wake it and wait
            // until it has published the new raster, so that this frame is
            // served at that tier.
            if (const uint32_t tier = tier_at(changes, index)) {
                wake(h->seq_req);
                const auto followed = [&] { return h->nativeModelMaxHeight.load() == tier && running(); };
                if (await(followed, "it followed the tier")) return 1;
            }
        }
        if (frames[index] == 'f')
            for (size_t p = 0; p < pixels; ++p) std::memcpy(&rgba[p * 4], kFlat, 4);
        else
            gradient(rgba.data(), width, height, unsigned(frames[index] - 'A'));
        if (fp16) widen(wide.data(), rgba.data(), rgba.size());
        std::memcpy(input, image, bytes);
        h->width.store(width);
        h->height.store(height);
        h->hdrEncode.store(fp16);
        const uint32_t request = h->seq_req.load() + 1;
        const auto sent = Clock::now();
        std::atomic_thread_fence(std::memory_order_release);
        h->seq_req.store(request);
        wake(h->seq_req);
        const auto deadline = Clock::now() + std::chrono::seconds(180);
        for (uint32_t response; (response = h->seq_resp.load(std::memory_order_acquire)) != request;) {
            if (waitpid(pid, &status, WNOHANG) == pid) {
                std::fprintf(stderr, "daemon exited during request %u (status %d)\n", request, exit_code(status));
                return 1;
            }
            if (Clock::now() > deadline) {
                std::fprintf(stderr, "no answer to request %u\n", request);
                return stop(1);
            }
            const timespec timeout{0, 10000000};
            syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAIT, response, &timeout, nullptr, 0);
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sent).count();
        const bool ok = h->seq_ok.load() == request;
        const unsigned aw = h->answeredW.load(), ah = h->answeredH.load();
        const size_t answer = ok ? size_t(aw) * ah * (fp16 ? 8 : 4) : 0;
        failures += !ok;
        std::printf("frame %zu input=%c input_fnv=%016llx request=%u ok=%d answered=%ux%u answer_fnv=%016llx "
                    "ms=%.2f\n",
                    index, frames[index], static_cast<unsigned long long>(fnv(image, bytes)), request, ok, aw, ah,
                    static_cast<unsigned long long>(ok ? fnv(output, answer) : 0), ms);
        if (!ok) std::printf("reason: %s\n", ShmLoadString(h, SHM_TEXT_HELPER_REASON).c_str());
        if (ok && !dump.empty()) {
            const std::string path = dump + "/answer-" + std::to_string(index) + (fp16 ? ".rgba16f" : ".rgba8");
            FILE* file = std::fopen(path.c_str(), "wb");
            bool written = file && std::fwrite(output, 1, answer, file) == answer;
            if (file && std::fclose(file)) written = false;
            if (!written) {
                std::fprintf(stderr, "cannot write %s: %s\n", path.c_str(), std::strerror(errno));
                ++failures;
            }
        }
        std::fflush(stdout);
    }
    h->quit.store(1);
    wake(h->seq_req);
    for (const auto deadline = Clock::now() + std::chrono::seconds(60); waitpid(pid, &status, WNOHANG) != pid;) {
        if (Clock::now() > deadline) {
            std::fprintf(stderr, "daemon did not quit\n");
            return stop(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const int code = exit_code(status);
    std::printf("daemon exit=%d failures=%d\n", code, failures);
    shm_channel_fini(&channel);
    unlink(shm.c_str());
    return code || failures;
}
