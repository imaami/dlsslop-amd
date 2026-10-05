// The network as serving, the offline mode and the self-test see it: one of the
// engines, chosen once at startup, each compiled into what it serves. None is
// dispatched at run time.
// SPDX-License-Identifier: MIT
#pragma once
#include "options.hpp"
#include "processing.hpp"
#include "result.hpp"
#include "shm_protocol.h"
#include "trace.h"
#include "tuning_math.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <string>

namespace dlsslop {

// A request's frames: host memory, or the device-local pair in an import slot.
struct Frames {
    const uint8_t* proxy = nullptr;
    uint8_t* answer = nullptr;
    int slot = -1;
};

// What every engine shares: the device-local frames the layer exported
// (ShmTransportOffer), one pair per producer generation in as many slots, the
// oldest replaced first; the latest frame's timing; and doing nothing where an
// engine has nothing to do. An engine imports an offer into the slot it is given
// (import_into) and may say where a slot's frames are (frames_of).
template <class Derived>
class EngineBase {
public:
    static constexpr unsigned kSlots = 4;

private:
    struct Held {
        uint32_t generation = 0;
        size_t bytes = 0;
    };
    std::array<Held, kSlots> held_{};
    unsigned next_slot_ = 0;

    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

public:
    float upload_ms = 0, inference_ms = 0, readback_ms = 0;

    // Serving, between frames: imports an offered proxy/answer pair into the slot of
    // its generation, or else the oldest one. The engine owns each descriptor it
    // imported and sets it to -1; the caller closes the rest.
    bool import(const ShmTransportOffer& offer, int fds[2])
    {
        if (!offer.generation) return false;
        unsigned slot = 0;
        while (slot < kSlots && held_[slot].generation != offer.generation) ++slot;
        if (slot == kSlots) slot = next_slot_++ % kSlots;
        if (!self().import_into(slot, offer, fds)) return false;
        held_[slot] = {offer.generation, size_t(std::min(offer.size[0], offer.size[1]))};
        return true;
    }
    // The imported pair of a generation that holds bytes, or none (slot -1).
    Frames frames(uint32_t generation, size_t bytes) const
    {
        for (unsigned slot = 0; slot < kSlots; ++slot)
            if (held_[slot].generation == generation && held_[slot].bytes >= bytes) return self().frames_of(slot);
        return {};
    }

    // Defaults an engine replaces where it has something to do.
    bool import_into(unsigned, const ShmTransportOffer&, int[2]) { return false; }
    // A slot's frames: the slot alone, for an engine that finds them itself.
    Frames frames_of(unsigned slot) const { return {nullptr, nullptr, int(slot)}; }
    // False when a request of this shape needs a build first (reshape): work of
    // milliseconds to seconds that the caller reports as a start, not a slow
    // frame. admit, called before reshape, rejects a shape the engine does not
    // take; it builds nothing.
    bool fits(unsigned, unsigned, unsigned, const ProcessingSettings&) const { return true; }
    Result<void> admit(unsigned, unsigned, unsigned, const ProcessingSettings&) { return {}; }
    Result<void> reshape(unsigned, unsigned, unsigned, const ProcessingSettings&) { return {}; }
    // Serving only: DMA the channel's frame slots directly.
    void pin(uint8_t*, uint8_t*, size_t) {}
    // Diagnostics: copies host or device memory, such as an imported frame, to the host.
    Result<void> read_back(void*, const void*, size_t) { return fail("this network cannot trace"); }

protected:
    // Imports held before a rebuild are gone: the layer offers them again.
    void forget_imports() { held_ = {}; }
};

// What serving, the offline mode and --diagnose need of an engine.
template <class E>
concept Engine = std::derived_from<E, EngineBase<E>> && requires(E& engine, const E& view, unsigned n,
                                                                 const ProcessingSettings& settings, struct frame_trace* trace) {
    { E::max_passes } -> std::convertible_to<unsigned>;
    // Whether a tier change rebuilds it, with the layer presenting its own frames meanwhile.
    { E::rebuilds_for_tier } -> std::convertible_to<bool>;
    { view.name() } -> std::convertible_to<const char*>;
    // The device it runs on, as --device names it, for --diagnose.
    { view.device() } -> std::convertible_to<std::string>;
    { view.tier() } -> std::convertible_to<unsigned>;
    // What the network runs at, for the log.
    { view.processing() } -> std::convertible_to<std::string>;
    { engine.retier(n) } -> std::same_as<Result<void>>;
    { engine.prepare() } -> std::same_as<Result<void>>;
    // w * h RGBA8 frames, or RGBA16F with settings.fp16.
    { engine.infer(Frames{}, n, n, n, settings, trace) } -> std::same_as<Result<void>>;
};

class HipEngine;
class VulkanEngine;
Result<void> run_self_test(const Options& o, HipEngine& engine);
Result<void> run_self_test(const Options& o, VulkanEngine& engine);

} // namespace dlsslop
