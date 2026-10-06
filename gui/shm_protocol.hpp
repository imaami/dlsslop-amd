// The channel for the GUI's C++ (../common/shm_protocol.h): its words and
// layout as C++ sees them, pinned, and the functions that write a string into
// the caller's buffer as functions that return a std::string.
#pragma once
#include "../common/shm_protocol.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

// The channel's words, counts and layout as C++ sees them; shm_protocol.c pins them in C. A word is
// a lock-free std::atomic<uint32_t> of four bytes here, as it is a lock-free _Atomic(uint32_t) there,
// and a frame count a lock-free std::atomic<uint64_t> of eight bytes aligned to eight.
static_assert(sizeof(std::atomic<std::uint32_t>) == 4 && alignof(std::atomic<std::uint32_t>) == 4,
              "a word of the channel is not four bytes aligned to four");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free, "a word of the channel is not lock-free");
static_assert(sizeof(std::atomic<std::uint64_t>) == 8 && alignof(std::atomic<std::uint64_t>) == 8,
              "a count of the channel is not eight bytes aligned to eight");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "a count of the channel is not lock-free");
static_assert(sizeof(ShmHeader) == 1912 && alignof(ShmHeader) == 8 && sizeof(PassControl) == 36,
              "the header's layout differs from shm_protocol.c's");
static_assert(offsetof(ShmHeader, helperFrames) == 8 && offsetof(ShmHeader, layerFrames) == 16 &&
                  offsetof(ShmHeader, enabled) == 56 && offsetof(ShmHeader, transferStrengthBits) == 96 &&
                  offsetof(ShmHeader, helperState) == 180 && offsetof(ShmHeader, helperReason) == 228 &&
                  offsetof(ShmHeader, layerReason) == 424 && offsetof(ShmHeader, gameName) == 620 &&
                  offsetof(ShmHeader, pass) == 748 && offsetof(ShmHeader, mvecEnabled) == 1828 &&
                  offsetof(ShmHeader, seq_ok) == 1836 && offsetof(ShmHeader, hdrEncode) == 1868 &&
                  offsetof(ShmHeader, transportGen) == 1900 && offsetof(ShmHeader, nativeTier) == 1908,
              "the header's layout differs from shm_protocol.c's");
static_assert(sizeof(ShmTransportOffer) == 72, "the transport offer's layout differs from shm_protocol.c's");

// ShmNativeChannelPath(), or an empty string if it cannot be formatted.
inline std::string ShmNativeChannelPath()
{
    const char* channel = std::getenv("DLSSNR_SHM");
    const int length = shm_native_channel_path(nullptr, 0, channel);
    if (length < 0) return {};
    std::string path(static_cast<std::size_t>(length), '\0');
    shm_native_channel_path(path.data(), path.size() + 1, channel);
    return path;
}

// ShmLoadString() of the text field FIELD of H, or an empty string if the
// field changed during every copy.
inline std::string ShmLoadString(const ShmHeader* h, shm_text field)
{
    char text[kReasonBytes > kNameBytes ? kReasonBytes : kNameBytes];
    ShmLoadString(h, field, text, sizeof text);
    return text;
}
