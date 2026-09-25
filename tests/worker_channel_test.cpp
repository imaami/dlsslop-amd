// SPDX-License-Identifier: MIT
// Drives `dlsslopd --test-identity` over its shared-memory channel the way the
// layer's ShmProcessFrame does. Needs neither HIP nor model weights.
#include "shm_protocol.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
constexpr unsigned kWidth = 64, kHeight = 36;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void wake(std::atomic<uint32_t>& word)
{
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

// The layer's side of the channel, created before any worker exists.
class Channel {
    int fd_ = -1;
    void* mapping_ = MAP_FAILED;
public:
    const std::string path;
    ShmHeader* h = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;

    explicit Channel(std::string name) : path(std::move(name))
    {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        require(fd_ >= 0 && !ftruncate(fd_, off_t(ShmTotalBytes())), "create channel " + path);
        mapping_ = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        require(mapping_ != MAP_FAILED, "map channel");
        h = static_cast<ShmHeader*>(mapping_);
        ShmInitNativeDefaults(h);
        input = static_cast<uint8_t*>(mapping_) + kHeaderBytes;
        output = input + kMaxFrame;
    }
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    ~Channel()
    {
        if (mapping_ != MAP_FAILED) munmap(mapping_, ShmTotalBytes());
        if (fd_ >= 0) close(fd_);
    }
    static size_t bytes(bool fp16) { return size_t(kWidth) * kHeight * (fp16 ? 8 : 4); }
    // Publish one request as the layer does and return its number.
    uint32_t publish(bool fp16, uint8_t seed, uint32_t format = 1)
    {
        for (size_t i = 0; i < bytes(fp16); ++i) input[i] = uint8_t(seed + i * 7);
        h->width.store(kWidth);
        h->height.store(kHeight);
        h->format.store(format);
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
    std::string reason() const { return ShmLoadString(h->helperReasonSeq, h->helperReason, kReasonBytes); }
};

class Worker {
    pid_t pid_ = -1;
    const std::string log_;
public:
    Worker(const char* executable, const Channel& channel, std::string log, bool once = false)
        : log_(std::move(log))
    {
        const char* args[] = {executable, "--test-identity", "--shm", channel.path.c_str(),
                              once ? "--once" : nullptr, nullptr};
        pid_ = fork();
        require(pid_ >= 0, "fork worker");
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
        pid_ = -1;
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
            require(!channel.answered(channel.publish(fp16, seed, 2), fp16), "a malformed request succeeded");
        reports("unsupported request");
        require(channel.answered(channel.publish(fp16, 6), fp16), "the worker stopped serving after a rejection");
        require(channel.reason() == ready, "a served request left the rejection as the reason: " + channel.reason());
    }
    channel.h->preset.store(1);
    for (const uint8_t seed : {7, 8})
        require(!channel.answered(channel.publish(false, seed), false), "an unmapped preset was accepted");
    reports("unmapped");
    require(!channel.answered(channel.publish(false, 9, 2), false), "a malformed request succeeded");
    reports("unsupported request");
    channel.h->preset.store(0);
    require(channel.answered(channel.publish(false, 10), false), "the worker stopped serving after a rejection");
    require(channel.reason() == ready, "a served request left the rejection as the reason: " + channel.reason());
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    // Once per run of the same rejection: three malformed runs, the preset run
    // and the malformed request that interrupted it.
    const std::string log = worker.text();
    require(count(log, " failed: unsupported request") == 4 && count(log, " failed: unmapped") == 1,
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
    // Past the settle time the protocol still carries for the layer.
    std::this_thread::sleep_for(std::chrono::milliseconds(h->rebuildSettleMs.load() + 200));
    require(channel.answered(channel.publish(false, 12), false), "a later request failed");
    require(h->controlSeq.load() == control, "the worker published a control generation");
    require(worker.quit(channel) == 0, "worker did not quit cleanly:\n" + worker.text());
    std::printf("PASS: the worker published no control generation after a tuning change\n");
}

// --once exits after its one answer, with status 1 when that answer failed.
void once(const char* executable, const std::filesystem::path& directory)
{
    Channel channel((directory / "once.bin").string());
    for (const bool good : {false, true}) {
        Worker worker(executable, channel, (directory / "once.log").string(), true);
        const uint32_t request = channel.publish(true, 8, good ? 1 : 2);
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
    std::string buffer = (std::filesystem::temp_directory_path() / "dlsslop-worker-channel-XXXXXX").string();
    if (!mkdtemp(buffer.data())) {
        std::perror("mkdtemp");
        return 1;
    }
    const std::filesystem::path directory = buffer;
    int result = 0;
    try {
        restart(argv[1], directory);
        rejection(argv[1], directory);
        controls(argv[1], directory);
        once(argv[1], directory);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "worker channel: %s\n", e.what());
        result = 1;
    }
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    return result;
}
