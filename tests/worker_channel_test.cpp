// SPDX-License-Identifier: MIT
// Drives `dlsslopd --test-identity` over its shared-memory channel the way the
// layer's ShmProcessFrame does. Needs neither HIP nor model weights.
#include "shm_channel.h"
#include "shm_protocol.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
constexpr unsigned kWidth = 64, kHeight = 36;

// The worker a failed check stops; at most one runs at a time.
pid_t running = -1;
// The test's directory, removed however the test ends.
struct Scratch {
    std::filesystem::path directory;
    ~Scratch()
    {
        std::error_code ignored;
        if (!directory.empty()) std::filesystem::remove_all(directory, ignored);
    }
} scratch;

void require(bool condition, const std::string& message)
{
    if (condition) return;
    std::fprintf(stderr, "worker channel: %s\n", message.c_str());
    if (running > 0) {
        kill(running, SIGKILL);
        waitpid(running, nullptr, 0);
    }
    std::exit(1);
}

void wake(std::atomic<uint32_t>& word)
{
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

// The layer's side of the channel, created before any worker exists, as every program creates one.
class Channel {
    shm_channel channel_{nullptr, 0, -1, 0};
public:
    const std::string path;
    ShmHeader* h = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;

    explicit Channel(std::string name) : path(std::move(name))
    {
        error e;
        require(!shm_channel_open(&channel_, path.c_str(), path.size(), ShmTotalBytes(),
                                  SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, &e), "create channel: " + std::string(e.what));
        require(channel_.flags & SHM_CHANNEL_CREATED, "channel " + path + " existed");
        // C created the header's objects before the file had a name; they live in the mapping for C++ too.
        h = std::start_lifetime_as<ShmHeader>(channel_.h);
        input = static_cast<uint8_t*>(static_cast<void*>(channel_.h)) + kHeaderBytes;
        output = input + kMaxFrame;
    }
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    ~Channel() { shm_channel_fini(&channel_); }
    static size_t bytes(bool fp16) { return size_t(kWidth) * kHeight * (fp16 ? 8 : 4); }
    // Publish one request as the layer does and return its number. A WIDTH beyond kMaxW makes it
    // malformed.
    uint32_t publish(bool fp16, uint8_t seed, uint32_t width = kWidth)
    {
        for (size_t i = 0; i < bytes(fp16); ++i) input[i] = uint8_t(seed + i * 7);
        h->width.store(width);
        h->height.store(kHeight);
        h->hdrEncode.store(fp16 ? 1u : 0u);
        const uint32_t request = h->seq_req.load() + 1;
        std::atomic_thread_fence(std::memory_order_release);
        h->seq_req.store(request);
        wake(h->seq_req);
        return request;
    }
    // Wait for the answer; true when the worker delivered an identity frame.
    bool answered(uint32_t request, bool fp16)
    {
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        for (uint32_t response; (response = h->seq_resp.load(std::memory_order_acquire)) != request;) {
            require(Clock::now() < deadline, "no answer to request " + std::to_string(request));
            const timespec timeout{0, 10000000};
            syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAIT, response,
                    &timeout, nullptr, 0);
        }
        if (h->seq_ok.load() != request) return false;
        require(h->answeredW.load() == kWidth && h->answeredH.load() == kHeight,
                "answer dimensions differ");
        require(!std::memcmp(input, output, bytes(fp16)), "identity answer differs from its request");
        return true;
    }
    std::string reason() const { return ShmLoadString(h, SHM_TEXT_HELPER_REASON); }
};

class Worker {
    pid_t pid_ = -1;
    const std::string log_;
public:
    Worker(const char* executable, const Channel& channel, std::string log, bool once = false,
           const char* option = nullptr, const char* value = nullptr)
        : log_(std::move(log))
    {
        const char* args[] = {executable, "--test-identity", "--shm", channel.path.c_str(),
                              once ? "--once" : option, once ? option : value, once ? value : nullptr, nullptr};
        pid_ = fork();
        require(pid_ >= 0, "fork worker");
        running = pid_;
        if (!pid_) {
            const int fd = open(log_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (fd < 0 || dup2(fd, 1) < 0 || dup2(fd, 2) < 0) _exit(127);
            execv(executable, const_cast<char* const*>(args));
            _exit(127);
        }
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        while (channel.h->helperState.load(std::memory_order_acquire) != kHelperRunning) {
            require(waitpid(pid_, nullptr, WNOHANG) == 0, "worker exited before it was ready:\n" + text());
            require(Clock::now() < deadline, "worker did not become ready:\n" + text());
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    ~Worker()
    {
        if (pid_ <= 0) return;
        kill(pid_, SIGKILL);
        waitpid(pid_, nullptr, 0);
        running = -1;
    }
    std::string text() const
    {
        std::ifstream in(log_);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }
    // The exit status of a worker that exits by itself.
    int status()
    {
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        int status = 0;
        for (pid_t done; !(done = waitpid(pid_, &status, WNOHANG));) {
            require(Clock::now() < deadline, "worker did not exit:\n" + text());
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        pid_ = running = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    }
    int quit(Channel& channel)
    {
        channel.h->quit.store(1);
        wake(channel.h->seq_req);
        return status();
    }
};

// A worker stopped by a signal leaves the layer's request unanswered. Its
// successor fails that request at once (the layer presents its own frame),
// then serves every request made after it attached, in either precision.
void restart(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "restart.bin").string());
    auto* h = channel.h;
    h->seq_req.store(6);
    h->seq_resp.store(6);
    h->seq_ok.store(6);
    const uint32_t stale = channel.publish(true, 1);
    Worker worker(executable, channel, (directory / "restart.log").string());
    require(h->seq_resp.load() == stale && h->seq_ok.load() == 0,
            "a restarted worker did not fail the request left before it started");
    for (const bool fp16 : {true, false, true})
        require(channel.answered(channel.publish(fp16, uint8_t(fp16 + 2)), fp16),
                "a restarted worker failed a new request");
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    require(worker.text().find("failed") == std::string::npos,
            "a restarted worker reported a failed frame:\n" + worker.text());
    std::printf("PASS: a restarted worker failed the stale request and served new FP16 and RGBA8 ones\n");
}

size_t count(const std::string& text, const std::string& needle)
{
    size_t found = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++found;
    return found;
}

// A rejected request is answered as failed and the worker keeps serving. A
// rejection that repeats is logged and reported once; a served request makes
// the worker report itself ready again.
void rejection(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "rejection.bin").string());
    Worker worker(executable, channel, (directory / "rejection.log").string());
    const std::string ready = channel.reason();
    require(ready.find("IDENTITY") != std::string::npos, "an identity worker did not say so: " + ready);
    const auto reports = [&channel](const char* rejection) {
        require(channel.reason().rfind(rejection, 0) == 0, "the rejection is not the reason: " + channel.reason());
    };
    for (const bool fp16 : {false, true, false}) {
        for (const uint8_t seed : {4, 5})
            require(!channel.answered(channel.publish(fp16, seed, kMaxW + 1), fp16), "a malformed request succeeded");
        reports("unsupported request");
        require(channel.answered(channel.publish(fp16, 6), fp16), "the worker stopped serving after a rejection");
        require(channel.reason() == ready, "a served request left the rejection as the reason: " + channel.reason());
    }
    // A preset neither network has, and conditioning outside the model's range.
    channel.h->preset.store(1);
    require(!channel.answered(channel.publish(false, 7), false), "an unmapped preset was accepted");
    channel.h->preset.store(0);
    channel.h->style.store(3);
    require(!channel.answered(channel.publish(false, 8), false), "style 3 was accepted");
    reports("the preset must be 0");
    require(!channel.answered(channel.publish(false, 9, kMaxW + 1), false), "a malformed request succeeded");
    reports("unsupported request");
    channel.h->style.store(0);
    require(channel.answered(channel.publish(false, 10), false), "the worker stopped serving after a rejection");
    require(channel.reason() == ready, "a served request left the rejection as the reason: " + channel.reason());
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    // Once per run of the same rejection: three malformed runs, the preset and
    // style run and the malformed request that interrupted it.
    const std::string log = worker.text();
    require(count(log, " failed: unsupported request") == 4 && count(log, " failed: the preset must be 0") == 1,
            "each run of a rejection must be logged once:\n" + log);
    std::printf("PASS: rejected requests were answered as failed while serving continued\n");
}

// Native tuning applies on the next request. The worker only reads controls: it
// publishes no control generation of its own, however long after a change.
void controls(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "controls.bin").string());
    Worker worker(executable, channel, (directory / "controls.log").string());
    auto* h = channel.h;
    h->intensityBits.store(FloatToBits(0.5f));
    h->tuningSeq.fetch_add(1);
    const uint32_t control = h->controlSeq.fetch_add(1) + 1;
    require(channel.answered(channel.publish(false, 11), false), "a request after a tuning change failed");
    // Long enough after the change for a debounced rebuild to have published.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    require(channel.answered(channel.publish(false, 12), false), "a later request failed");
    require(h->controlSeq.load() == control, "the worker published a control generation");
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    std::printf("PASS: the worker published no control generation after a tuning change\n");
}

// A restarted worker keeps the channel's live pass count; only --passes, when
// given, replaces it.
void live_settings(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "live.bin").string());
    const std::string log = (directory / "live.log").string();
    {
        Worker worker(executable, channel, log);
        channel.h->passes.store(3);
        require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    }
    {
        Worker worker(executable, channel, log);
        require(channel.h->passes.load() == 3, "a restarted worker reset the live pass count");
        require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    }
    {
        Worker worker(executable, channel, log, false, "--passes", "2");
        require(channel.h->passes.load() == 2, "--passes did not replace the live pass count");
        require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    }
    // main points XDG_CONFIG_HOME at the test directory.
    const auto config = directory / "dlsslop-amd/dlsslopd.conf";
    std::error_code error;
    std::filesystem::create_directories(config.parent_path(), error);
    require(!error, "create the config directory");
    std::ofstream(config) << "# live settings\npasses = 4\n";
    {
        Worker worker(executable, channel, log);
        require(channel.h->passes.load() == 4, "the config file did not set the pass count");
        require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    }
    Worker worker(executable, channel, log, false, "--passes", "2");
    require(std::filesystem::remove(config, error), "remove the config file");
    require(channel.h->passes.load() == 2, "--passes did not override the config file");
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    std::printf("PASS: a restarted worker kept the live pass count; the config file and --passes replaced it\n");
}

// Waits for what the worker does by itself.
template <typename Condition>
void await(Condition done, const Worker& worker, const std::string& failure)
{
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!done()) {
        require(Clock::now() < deadline, failure + ":\n" + worker.text());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// A different tier rebuilds the worker once, between frames; the active tier
// again rebuilds nothing, and an unusable one is overwritten with the active
// one. A restarted worker keeps the live tier; only --tier replaces it.
void tiers(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "tiers.bin").string());
    auto* h = channel.h;
    const std::string log = (directory / "tiers.log").string();
    {
        Worker worker(executable, channel, log);
        require(h->nativeTier.load() == kNativeDefaultTier, "a worker changed a new channel's tier");
        const auto rebuilt = [&worker, h](size_t times) {
            return count(worker.text(), "rebuilding") == times && h->helperState.load() == kHelperRunning;
        };
        h->nativeTier.store(900);
        h->controlSeq.fetch_add(1);
        await([&rebuilt] { return rebuilt(1); }, worker, "the worker did not switch to tier 900");
        for (const uint32_t tier : {900u, 900u, 900u, 0u, 800u}) {
            h->nativeTier.store(tier);
            h->controlSeq.fetch_add(1);
            await([h] { return h->nativeTier.load() == 900; }, worker, "an unusable tier was not replaced");
        }
        require(rebuilt(1), "storing the active tier rebuilt the worker:\n" + worker.text());
        require(channel.answered(channel.publish(false, 13), false), "a request after a tier switch failed");
        h->nativeTier.store(1080);
        await([&rebuilt] { return rebuilt(2); }, worker, "the worker did not switch to tier 1080");
        require(!h->nativeModelMaxWidth.load() && !h->nativeModelMaxHeight.load(),
                "an identity worker published a neural raster");
        require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    }
    {
        Worker worker(executable, channel, log);
        require(h->nativeTier.load() == 1080, "a restarted worker reset the live tier");
        require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    }
    Worker worker(executable, channel, log, false, "--tier", "900");
    require(h->nativeTier.load() == 900, "--tier did not replace the live tier");
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    std::printf("PASS: each tier change rebuilt the worker once; repeats and unusable tiers did not\n");
}

// --once exits after its one answer, with status 1 when that answer failed.
void once(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "once.bin").string());
    for (const bool good : {false, true}) {
        Worker worker(executable, channel, (directory / "once.log").string(), true);
        const uint32_t request = channel.publish(true, 8, good ? kWidth : kMaxW + 1);
        require(channel.answered(request, true) == good, "--once answered its request wrongly");
        require(worker.status() == (good ? 0 : 1), std::string("--once exit status after a ") +
                (good ? "successful" : "failed") + " request:\n" + worker.text());
        require(channel.h->helperState.load() == (good ? kHelperStopped : kHelperModelFailed),
                "--once left the wrong helper state");
        require(channel.h->seq_req.load() == request, "--once consumed another request");
    }
    std::printf("PASS: --once exited after one answer, reporting a failed one\n");
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: worker-channel-test DLSSLOPD\n");
        return 2;
    }
    // mkdtemp makes the private (0700) directory the worker requires.
    std::error_code error;
    std::string buffer = (std::filesystem::temp_directory_path(error) / "dlsslop-worker-channel-XXXXXX").string();
    if (error || !mkdtemp(buffer.data())) {
        std::perror("mkdtemp");
        return 1;
    }
    scratch.directory = buffer;
    const std::filesystem::path& directory = scratch.directory;
    // Workers read no config file but the test's own.
    setenv("XDG_CONFIG_HOME", buffer.c_str(), 1);
    restart(argv[1], directory);
    rejection(argv[1], directory);
    controls(argv[1], directory);
    once(argv[1], directory);
    live_settings(argv[1], directory);
    tiers(argv[1], directory);
    return 0;
}
