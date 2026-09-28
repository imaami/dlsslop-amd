// The daemon's side of the shared-memory channel.
// SPDX-License-Identifier: MIT
#pragma once
#include "shm_protocol.h"
#include "trace.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <linux/futex.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace dlsslop {
[[noreturn]] inline void system_error(const char* action)
{
    throw std::runtime_error(std::string(action) + ": " + std::strerror(errno));
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
} // namespace dlsslop
