// SPDX-License-Identifier: MIT
// The smallest fake layer, for tracing served frames (tests/hiptrace/trace.sh):
// creates a private channel, starts dlsslopd on it, publishes RGBA8 frames the
// way the layer's ShmProcessFrame does, and prints the FNV-1a 64 of every
// answer. No device-local transport is offered.
#include "shm_protocol.h"

#include <fcntl.h>
#include <getopt.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr const char* kUsage = "Usage: hiptrace-shmclient [OPTIONS] -- DLSSLOPD [ARGUMENTS...]\n";
constexpr const char* kDefaultFrames = "AAB";
constexpr const char* kDefaultLog = "/dev/null";
constexpr unsigned kDefaultPasses = 1;

int usage(const char* problem)
{
    std::fprintf(stderr, "hiptrace-shmclient: %s\n%s(see --help)\n", problem, kUsage);
    return 2;
}

void help()
{
    std::printf("%s", kUsage);
    std::printf(
        "Create the channel FILE, start DLSSLOPD, which must serve FILE (--shm FILE)\n"
        "at --tier, send it the frames of --frames and print the FNV-1a 64 of every\n"
        "input and answer. Exit status 0 when the daemon exits with 0 and answers\n"
        "every frame, 2 on a usage error, otherwise 1.\n"
        "\n"
        "  -s, --shm FILE       Channel to create, in a private (0700) directory\n"
        "                       Default: unset; required\n"
        "  -W, --width PIXELS   Source width (2..%u)\n"
        "                       Default: unset; required\n"
        "  -H, --height PIXELS  Source height (2..%u)\n"
        "                       Default: unset; required\n"
        "  -t, --tier HEIGHT    The channel's neural raster: 720, 900 or 1080\n"
        "                       Default: unset; required\n"
        "  -f, --frames SPEC    One letter from A to Z per frame, naming its input: the\n"
        "                       self-test's gradient, shifted by the letter\n"
        "                       Default: %s\n"
        "  -m, --motion         Motion history on\n"
        "                       Default: off\n"
        "  -P, --passes N       Chained evaluations per frame (1..%u)\n"
        "                       Default: %u\n"
        "  -l, --log FILE       The daemon's standard output and error\n"
        "                       Default: %s\n"
        "  -h, --help           Show this help and exit\n"
        "                       Default: off\n",
        kMaxW, kMaxH, kDefaultFrames, kMaxPasses, kDefaultPasses, kDefaultLog);
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
} // namespace

int main(int argc, char** argv)
{
    std::string shm, frames = kDefaultFrames, log = kDefaultLog;
    unsigned width = 0, height = 0, tier = 0, passes = kDefaultPasses;
    bool motion = false;
    static const option options[] = {{"shm", required_argument, nullptr, 's'},
                                      {"width", required_argument, nullptr, 'W'},
                                      {"height", required_argument, nullptr, 'H'},
                                      {"tier", required_argument, nullptr, 't'},
                                      {"frames", required_argument, nullptr, 'f'},
                                      {"motion", no_argument, nullptr, 'm'},
                                      {"passes", required_argument, nullptr, 'P'},
                                      {"log", required_argument, nullptr, 'l'},
                                      {"help", no_argument, nullptr, 'h'},
                                      {nullptr, 0, nullptr, 0}};
    for (int c; (c = getopt_long(argc, argv, "+s:W:H:t:f:mP:l:h", options, nullptr)) != -1;) {
        switch (c) {
        case 's': shm = optarg; break;
        case 'W': width = number(optarg, kMaxW); break;
        case 'H': height = number(optarg, kMaxH); break;
        case 't': tier = number(optarg, ~0u); break;
        case 'f': frames = optarg; break;
        case 'm': motion = true; break;
        case 'P': passes = number(optarg, kMaxPasses); break;
        case 'l': log = optarg; break;
        case 'h': help(); return 0;
        default: return usage("unknown option or missing value");
        }
    }
    if (shm.empty()) return usage("--shm is required");
    if (width < 2 || height < 2) return usage("--width and --height are required and must be in range");
    if (!ShmNativeTier(tier)) return usage("--tier must be 720, 900 or 1080");
    if (!passes) return usage("--passes is out of range");
    if (frames.empty() || frames.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos)
        return usage("--frames must be letters from A to Z");
    if (optind >= argc) return usage("DLSSLOPD is required");

    unlink(shm.c_str());
    const int fd = open(shm.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0 || ftruncate(fd, off_t(ShmTotalBytes()))) {
        std::perror("create channel");
        return 1;
    }
    void* mapping = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        std::perror("map channel");
        return 1;
    }
    auto* h = static_cast<ShmHeader*>(mapping);
    ShmInitNativeDefaults(h);
    h->nativeTier.store(tier);
    h->passes.store(passes);
    h->mvecEnabled.store(motion ? 1 : 0);
    uint8_t* const input = static_cast<uint8_t*>(mapping) + kHeaderBytes;
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
    for (const auto deadline = Clock::now() + std::chrono::seconds(300);
         h->helperState.load(std::memory_order_acquire) != kHelperRunning;) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            std::fprintf(stderr, "daemon exited before it was ready (status %d)\n", exit_code(status));
            return 1;
        }
        if (Clock::now() > deadline) {
            std::fprintf(stderr, "daemon not ready\n");
            return stop(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::printf("ready tier=%u state=%u\n", h->nativeTier.load(), h->helperState.load());

    const size_t bytes = size_t(width) * height * 4;
    std::vector<uint8_t> image(bytes);
    int failures = 0;
    for (size_t index = 0; index < frames.size(); ++index) {
        const unsigned letter = unsigned(frames[index] - 'A');
        gradient(image.data(), width, height, letter);
        std::memcpy(input, image.data(), bytes);
        h->width.store(width);
        h->height.store(height);
        h->format.store(1);
        h->hdrEncode.store(0);
        const uint32_t request = h->seq_req.load() + 1;
        std::atomic_thread_fence(std::memory_order_release);
        h->seq_req.store(request);
        wake(h->seq_req);
        const auto deadline = Clock::now() + std::chrono::seconds(120);
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
        const bool ok = h->seq_ok.load() == request;
        const unsigned aw = h->answeredW.load(), ah = h->answeredH.load();
        failures += !ok;
        std::printf("frame %zu input=%c input_fnv=%016llx request=%u ok=%d answered=%ux%u answer_fnv=%016llx\n",
                    index, frames[index], static_cast<unsigned long long>(fnv(image.data(), bytes)), request, ok,
                    aw, ah, static_cast<unsigned long long>(ok ? fnv(output, size_t(aw) * ah * 4) : 0));
        if (!ok) std::printf("reason: %s\n", ShmLoadString(h->helperReasonSeq, h->helperReason, kReasonBytes).c_str());
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
    munmap(mapping, ShmTotalBytes());
    close(fd);
    unlink(shm.c_str());
    return code || failures;
}
