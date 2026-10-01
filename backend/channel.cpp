// SPDX-License-Identifier: MIT
#include "channel.h"

#include <fcntl.h>
#include <linux/futex.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace dlsslop {

Result<Mapping> Mapping::open(const std::string& name)
{
    const std::string parent = parent_path(name);
    if (parent.empty()) return fail("--shm requires a path inside a private directory");
    DLSSLOP_TRY(private_directory(parent, "shared-memory"));
    Descriptor file(::open(name.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (file.fd < 0) return fail_errno("open shared-memory file");
    if (flock(file.fd, LOCK_EX | LOCK_NB)) return fail("another worker owns this shared-memory file");
    struct stat st{};
    if (fstat(file.fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid())
        return fail("shared-memory file must be regular and owned by the current user");
    if (fchmod(file.fd, 0600)) return fail_errno("make shared-memory file private");
    if (ftruncate(file.fd, static_cast<off_t>(ShmTotalBytes()))) return fail_errno("size shared-memory file");
    void* const mapping = mmap(nullptr, ShmTotalBytes(), PROT_READ | PROT_WRITE, MAP_SHARED, file.fd, 0);
    if (mapping == MAP_FAILED) return fail_errno("map shared-memory file");
    return Mapping(std::move(file), mapping);
}

Mapping::Mapping(Descriptor file, void* mapping)
    : file_(std::move(file)), h(static_cast<ShmHeader*>(mapping)), input(static_cast<uint8_t*>(mapping) + kHeaderBytes),
      output(input + kMaxFrame)
{
    if (h->magic.load() != kShmMagic || h->version.load() != kShmVersion) ShmInitNativeDefaults(h, false);
    h->quit.store(0);
    h->modelUp.store(0);
    h->seq_ok.store(0);
    // Answer a request left by a previous worker as failed, so the layer
    // presents its own frame; requests made from here on are served.
    h->seq_resp.store(h->seq_req.load(std::memory_order_acquire), std::memory_order_release);
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAKE, 1, nullptr, nullptr, 0);
    h->helperState.store(kHelperStarting);
}

Mapping::Mapping(Mapping&& other) noexcept
    : file_(std::move(other.file_)), h(std::exchange(other.h, nullptr)), input(other.input), output(other.output)
{
}

Mapping::~Mapping()
{
    if (!h) return;
    h->modelUp.store(0);
    if (h->helperState.load() != kHelperModelFailed) h->helperState.store(kHelperStopped);
    munmap(h, ShmTotalBytes());
}

} // namespace dlsslop
