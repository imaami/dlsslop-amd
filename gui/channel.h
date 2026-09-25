// SPDX-License-Identifier: MIT
#pragma once
#include "../common/control_settings.h"
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dlsslop_gui {

// Short-lived mappings avoid retaining a channel replaced by a restarted worker.
// A controller never creates, truncates, initializes or unlinks the channel.
class Channel {
    int fd_ = -1;
    ShmHeader* header_ = nullptr;
    struct stat stat_ {};
public:
    Channel(const std::string& path, bool writable)
    {
        fd_ = open(path.c_str(), (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW);
        if (fd_ < 0) throw std::runtime_error(std::strerror(errno));
        try {
            if (fstat(fd_, &stat_)) throw std::runtime_error(std::strerror(errno));
            if (!S_ISREG(stat_.st_mode) || stat_.st_uid != getuid() ||
                stat_.st_size < static_cast<off_t>(ShmTotalBytes()))
                throw std::runtime_error("Channel must be an owned, full-size regular file");
            void* base = mmap(nullptr, kHeaderBytes, PROT_READ | (writable ? PROT_WRITE : 0),
                              MAP_SHARED, fd_, 0);
            if (base == MAP_FAILED) throw std::runtime_error(std::strerror(errno));
            header_ = static_cast<ShmHeader*>(base);
            if (header_->magic.load() != kShmMagic || header_->version.load() != kShmVersion)
                throw std::runtime_error("Channel protocol mismatch; use the matching controller version");
        } catch (...) {
            if (header_) munmap(header_, kHeaderBytes);
            close(fd_);
            throw;
        }
    }
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    ~Channel() { munmap(header_, kHeaderBytes); close(fd_); }
    ShmHeader* header() const { return header_; }
    dev_t device() const { return stat_.st_dev; }
    ino_t inode() const { return stat_.st_ino; }

    static double value(const dlsslop_control::Setting& s, uint32_t raw)
    { return s.isFloat ? static_cast<double>(BitsToFloat(raw)) : static_cast<double>(raw); }

    // Validate the complete batch before writing any field. Only edited fields
    // are written, preserving unrelated edits made by the CLI or another GUI.
    void write(const std::map<std::size_t, double>& changes)
    {
        ShmHeader defaults{};
        ShmInitNativeDefaults(&defaults);
        for (const auto& [index, number] : changes) {
            if (index >= std::size(dlsslop_control::kSettings))
                throw std::invalid_argument("Unknown setting");
            const auto& s = dlsslop_control::kSettings[index];
            if (!std::isfinite(number) || number < s.minimum || number > s.maximum ||
                (!s.isFloat && std::trunc(number) != number))
                throw std::invalid_argument("Setting outside supported range");
            if (dlsslop_control::fixed(s) && number != value(s, (defaults.*s.field).load()))
                throw std::invalid_argument("This captured model fixes that setting");
        }
        bool tuningChanged = false;
        for (const auto& [index, number] : changes) {
            const auto& s = dlsslop_control::kSettings[index];
            (header_->*s.field).store(s.isFloat ? FloatToBits(static_cast<float>(number)) :
                                               static_cast<uint32_t>(number));
            tuningChanged |= dlsslop_control::tuning(s);
        }
        if (changes.empty()) return;
        if (tuningChanged) header_->tuningSeq.fetch_add(1);
        header_->controlSeq.fetch_add(1);
    }
    void reset()
    {
        ShmHeader defaults{};
        const bool bypass = header_->heartbeat.load() && !header_->nativeModelMaxWidth.load() &&
                            !header_->nativeModelMaxHeight.load();
        ShmInitNativeDefaults(&defaults, bypass);
        for (const auto& s : dlsslop_control::kSettings)
            (header_->*s.field).store((defaults.*s.field).load());
        header_->tuningSeq.fetch_add(1);
        header_->controlSeq.fetch_add(1);
    }
    void capture(unsigned count)
    {
        if (count > 64) throw std::invalid_argument("Capture count must be 0..64");
        header_->controlSeq.fetch_add(1);
        header_->captureRequest.store(count, std::memory_order_release);
    }
    void stop(bool requested)
    {
        header_->quit.store(requested ? 1u : 0u);
        header_->controlSeq.fetch_add(1);
    }
};
} // namespace dlsslop_gui
