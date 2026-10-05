// SPDX-License-Identifier: MIT
#pragma once
#include "../common/control_settings.h"
#include "../common/shm_channel.h"
#include "result.hpp"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <utility>

namespace dlsslop_gui {
using dlsslop::Result;

// The channel, opened as every program opens it (common/shm_channel.h): a file
// of another protocol is refused. Short-lived mappings avoid retaining a
// channel replaced by a restarted worker. A controller never creates,
// truncates, initializes or unlinks the channel.
class Channel {
    shm_channel channel_{nullptr, 0, -1, 0};
    ShmHeader* header_ = nullptr;
    struct stat stat_ {};

    Channel() = default;
public:
    static Result<Channel> open(const std::string& path, bool writable)
    {
        Channel channel;
        error e;
        if (shm_channel_open(&channel.channel_, path.c_str(), path.size(), kHeaderBytes,
                             writable ? SHM_CHANNEL_WRITE : shm_channel_flags{}, &e))
            return dlsslop::fail(e.what);
        if (fstat(channel.channel_.fd, &channel.stat_)) return dlsslop::fail_errno("inspect the channel");
        // C created the header's objects before the file had a name; they live in the mapping for C++ too.
        channel.header_ = std::start_lifetime_as<ShmHeader>(channel.channel_.h);
        return channel;
    }
    Channel(Channel&& other) noexcept
        : channel_(std::exchange(other.channel_, shm_channel{nullptr, 0, -1, 0})),
          header_(std::exchange(other.header_, nullptr)), stat_(other.stat_)
    {
    }
    ~Channel() { shm_channel_fini(&channel_); }
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
