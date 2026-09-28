// The network as the serving loop, the offline mode and the self-test see it.
// SPDX-License-Identifier: MIT
#pragma once
#include "options.h"
#include "result.h"
#include "shm_protocol.h"
#include "trace.h"
#include "tuning_math.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

namespace dlsslop {
struct ProcessingSettings {
    dlsslop::NativeTuning tuning;
    // The Vulkan model's own conditioning; the HIP network has only the defaults.
    unsigned style = 0;
    float skin_structure = -1;
    bool auto_mask = true;
    float color_preserve = 0;
    bool fp16 = false;
    bool precision16 = true;
    bool motion = false;
    unsigned motion_quality = kMVecBalanced;
    unsigned motion_grid = kMVecPixels4;
};

// The network as the serving loop, the offline mode and the self-test see it.
class Backend {
public:
    virtual ~Backend() = default;
    // A request's frames: host memory, or the device-local pair in an import slot.
    struct Frames {
        const uint8_t* proxy = nullptr;
        uint8_t* answer = nullptr;
        int slot = -1;
    };
    virtual const char* name() const = 0;
    // The device it runs on, as --device names it, for --diagnose.
    virtual std::string device() const = 0;
    virtual unsigned tier() const = 0;
    // What the network runs at, for the log.
    virtual std::string processing() const = 0;
    // Takes a new tier without a new backend; false when it needs one.
    virtual bool retier(unsigned) { return false; }
    virtual Result<void> prepare() = 0;
    // False when a request of this shape needs a build first (reshape): seconds
    // of work the caller reports as a start, not a slow frame.
    virtual bool fits(unsigned, unsigned, unsigned, const ProcessingSettings&) const { return true; }
    virtual Result<void> reshape(unsigned, unsigned, unsigned, const ProcessingSettings&) { return {}; }
    // Serving only: DMA the channel's frame slots directly.
    virtual void pin(uint8_t*, uint8_t*, size_t) {}
    // Serving, between frames: imports an offered proxy/answer pair into the slot of
    // its generation, or else the oldest one. The backend owns each descriptor it
    // imported; fds keeps the rest to close.
    bool import(const ShmTransportOffer& offer, dlsslop::Descriptor (&fds)[2])
    {
        if (!offer.generation) return false;
        unsigned slot = 0;
        while (slot < kSlots && held_[slot].generation != offer.generation) ++slot;
        if (slot == kSlots) slot = next_slot_++ % kSlots;
        if (!import_into(slot, offer, fds)) return false;
        held_[slot] = {offer.generation, size_t(std::min(offer.size[0], offer.size[1]))};
        return true;
    }
    // The imported pair of a generation that holds bytes, or none (slot -1).
    Frames frames(uint32_t generation, size_t bytes) const
    {
        for (unsigned slot = 0; slot < kSlots; ++slot)
            if (held_[slot].generation == generation && held_[slot].bytes >= bytes) return frames_of(slot);
        return {};
    }
    // Diagnostics: copies host or device memory, such as an imported frame, to the host.
    virtual Result<void> read_back(void* host, const void* source, size_t bytes) = 0;
    // w * h RGBA8 frames, or RGBA16F with settings.fp16.
    virtual Result<void> infer(const Frames& io, unsigned w, unsigned h, unsigned passes,
                               const ProcessingSettings& settings = {}, dlsslop::FrameTrace* trace = nullptr) = 0;
    virtual Result<void> self_test(const Options& o) = 0;
    float upload_ms = 0, inference_ms = 0, readback_ms = 0;
    unsigned max_passes = kMaxPasses;
protected:
    // Device-local frames the layer exported (ShmTransportOffer): one pair per producer
    // generation, in as many slots, the oldest replaced first.
    static constexpr unsigned kSlots = 4;
    // Imports an offer into a slot, releasing the pair it held once the new one is in.
    virtual bool import_into(unsigned slot, const ShmTransportOffer&, dlsslop::Descriptor (&)[2]) = 0;
    // A slot's frames; the slot alone for a backend that finds them itself.
    virtual Frames frames_of(unsigned slot) const { return {nullptr, nullptr, int(slot)}; }
private:
    struct Held {
        uint32_t generation = 0;
        size_t bytes = 0;
    };
    std::array<Held, kSlots> held_{};
    unsigned next_slot_ = 0;
};

class HipEngine;
Result<void> run_self_test(const Options& o, HipEngine& engine);
Result<void> run_vulkan_self_test(const Options& o, Backend& engine);
} // namespace dlsslop
