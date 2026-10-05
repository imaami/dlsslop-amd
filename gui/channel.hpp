// SPDX-License-Identifier: MIT
#pragma once
#include "../common/control_settings.h"
#include "../common/result.hpp"
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace dlsslop_gui {
using dlsslop::Result;

// Short-lived mappings avoid retaining a channel replaced by a restarted worker.
// A controller never creates, truncates, initializes or unlinks the channel.
class Channel {
    int fd_ = -1;
    ShmHeader* header_ = nullptr;
    struct stat stat_ {};

    explicit Channel(int fd) : fd_(fd) {}
public:
    static Result<Channel> open(const std::string& path, bool writable)
    {
        Channel channel(::open(path.c_str(), (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW));
        if (channel.fd_ < 0 || fstat(channel.fd_, &channel.stat_)) return dlsslop::fail(std::strerror(errno));
        if (!S_ISREG(channel.stat_.st_mode) || channel.stat_.st_uid != getuid() ||
            channel.stat_.st_size < static_cast<off_t>(ShmTotalBytes()))
            return dlsslop::fail("Channel must be an owned, full-size regular file");
        void* base = mmap(nullptr, kHeaderBytes, PROT_READ | (writable ? PROT_WRITE : 0), MAP_SHARED, channel.fd_, 0);
        if (base == MAP_FAILED) return dlsslop::fail(std::strerror(errno));
        channel.header_ = static_cast<ShmHeader*>(base);
        if (channel.header_->magic.load() != kShmMagic || channel.header_->version.load() != kShmVersion)
            return dlsslop::fail("Channel protocol mismatch; use the matching controller version");
        return channel;
    }
    Channel(Channel&& other) noexcept
        : fd_(std::exchange(other.fd_, -1)), header_(std::exchange(other.header_, nullptr)), stat_(other.stat_)
    {
    }
    ~Channel()
    {
        if (header_) munmap(header_, kHeaderBytes);
        if (fd_ >= 0) close(fd_);
    }
    ShmHeader* header() const { return header_; }
    dev_t device() const { return stat_.st_dev; }
    ino_t inode() const { return stat_.st_ino; }

    // Validate the complete batch before writing any field. Only edited fields
    // are written, preserving unrelated edits made by the CLI or another GUI.
    Result<void> write(const std::map<std::size_t, double>& changes)
    {
        for (const auto& [index, number] : changes) {
            if (index >= CONTROL_SETTING_COUNT) return dlsslop::fail("Unknown setting");
            const auto& s = CONTROL_SETTINGS[index];
            if (!control_setting_in_range(&s, number)) return dlsslop::fail("Setting outside supported range");
        }
        bool tuningChanged = false;
        for (const auto& [index, number] : changes) {
            const auto& s = CONTROL_SETTINGS[index];
            // + 0.0f stores -0 as 0.
            control_setting_store(header_, &s, s.is_float ? FloatToBits(static_cast<float>(number) + 0.0f) :
                                                            static_cast<uint32_t>(number));
            tuningChanged |= s.tuning;
        }
        if (changes.empty()) return {};
        if (tuningChanged) header_->tuningSeq.fetch_add(1);
        header_->controlSeq.fetch_add(1);
        return {};
    }
    void reset()
    {
        std::array<std::uint32_t, CONTROL_SETTING_COUNT> defaults;
        control_settings_defaults(defaults.data(), control_settings_worker_bypass(header_));
        for (std::size_t i = 0; i < CONTROL_SETTING_COUNT; ++i)
            control_setting_store(header_, &CONTROL_SETTINGS[i], defaults[i]);
        header_->tuningSeq.fetch_add(1);
        header_->controlSeq.fetch_add(1);
    }
    Result<void> capture(unsigned count)
    {
        if (count > 64) return dlsslop::fail("Capture count must be 0..64");
        header_->controlSeq.fetch_add(1);
        header_->captureRequest.store(count, std::memory_order_release);
        return {};
    }
    void stop(bool requested)
    {
        header_->quit.store(requested ? 1u : 0u);
        header_->controlSeq.fetch_add(1);
    }
};
} // namespace dlsslop_gui
