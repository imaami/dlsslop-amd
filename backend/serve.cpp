// SPDX-License-Identifier: MIT
#include "serve.h"
#include "channel.h"
#include "engine.h"
#include "hip_engine.h"
#include "open.h"
#include "transport.h"
#include "unwrap.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <linux/futex.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace dlsslop {
namespace {
volatile sig_atomic_t stopping;
void stop_handler(int) { stopping = 1; }

dlsslop::NativeTuning read_tuning(const ShmHeader* h)
{
    return {BitsToFloat(h->intensityBits.load()), BitsToFloat(h->localToneBits.load()),
            BitsToFloat(h->localStructureBits.load()), BitsToFloat(h->sharpnessBits.load())};
}

// The layer must know the real neural raster before building its proxy.
// Otherwise it mistakes a worker-upscaled answer for native-resolution output
// and skips its detail-preserving composition branch. CPU composition and
// identity publish none: the mapping may retain a preceding neural worker's.
void publish_raster(const Options& o, ShmHeader* h, unsigned tier)
{
    const bool neural = !o.cpu_compose && !o.test_identity;
    h->nativeModelMaxWidth.store(neural ? ShmNativeTier(tier)->width : 0);
    h->nativeModelMaxHeight.store(neural ? tier : 0);
}

// Between frames: rebuilds the network once for each tier a controller stores
// that differs from the active one. The layer presents its own frames while
// helperState reads Starting, then offers its device-local frames to the new
// engine. An unusable tier is overwritten with the active one. A failed
// rebuild ends the worker, leaving the active tier for the next one. True
// after a rebuild.
bool follow_tier(std::unique_ptr<Backend>& engine, const Options& o, Mapping& mapping, const char* ready)
{
    auto* h = mapping.h;
    const unsigned wanted = h->nativeTier.load(), active = engine->tier();
    if (wanted == active) return false;
    if (!ShmNativeTier(wanted)) {
        h->nativeTier.store(active);
        return false;
    }
    // A network that sizes itself to each frame only needs the layer to target the new raster.
    if (engine->retier(wanted)) {
        std::fprintf(stderr, "neural tier %u -> %u\n", active, wanted);
        publish_raster(o, h, wanted);
        return false;
    }
    h->helperState.store(kHelperStarting);
    h->modelUp.store(0);
    std::fprintf(stderr, "neural tier %u -> %u: rebuilding\n", active, wanted);
    mapping.reason("rebuilding for neural tier " + std::to_string(wanted));
    engine.reset(); // The old network's memory goes first.
    try {
        engine = std::make_unique<Engine>(o, wanted);
        engine->prepare();
    } catch (...) {
        h->nativeTier.store(active);
        throw;
    }
    publish_raster(o, h, wanted);
    mapping.reason(ready);
    h->modelUp.store(o.test_identity ? 0 : 1);
    h->helperState.store(kHelperRunning, std::memory_order_release);
    return true;
}

// Idle exit: stop taking requests before leaving. A layer that reads Stopped
// from now on sends none, and one whose request got in first is served: true
// when none did.
bool retire(ShmHeader* h, uint32_t request)
{
    h->helperState.store(kHelperStopped);
    if (h->seq_req.load() == request) return true;
    h->helperState.store(kHelperRunning, std::memory_order_release);
    return false;
}
} // namespace

void run_worker(Options o)
{
    auto opened = Mapping::open(o.shm);
    if (!opened) throw std::runtime_error(opened.error().what);
    Mapping& mapping = *opened;
    auto* h = mapping.h;
    std::optional<TraceRequests> traces;
    if (!o.trace_dir.empty()) {
        auto requests = TraceRequests::open(o.trace_dir, o.shm);
        if (!requests) throw std::runtime_error(requests.error().what);
        traces.emplace(std::move(*requests));
    }
    std::unique_ptr<dlsslop::FrameTrace> pending_trace;
    // An explicit tier replaces the channel's; otherwise a usable live one stays.
    const unsigned live = h->nativeTier.load();
    const unsigned tier = o.tier.value_or(ShmNativeTier(live) ? live : kNativeDefaultTier);
    h->nativeTier.store(tier);
    if (o.passes) h->passes.store(*o.passes);
    h->compositionBypass.store(o.cpu_compose || o.test_identity ? 1 : 0);
    publish_raster(o, h, tier);
    // Linux futex wake is emitted by the patched Vulkan layer. Timeout maintains liveness
    // with old clients and permits signals/quit; no GPU polling is involved.
    std::atomic<bool> heartbeat_stop{false};
    std::thread heartbeat([&] {
        while (!heartbeat_stop.load()) {
            h->heartbeat.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    struct Stop {
        std::atomic<bool>& flag;
        std::thread& thread;
        ~Stop() { flag.store(true); thread.join(); }
    } stop_heartbeat{heartbeat_stop, heartbeat};
    try {
        mapping.reason(o.test_identity ? "IDENTITY TEST: inference disabled" : "initializing the network");
        auto listener = TransportListener::open(o.shm, !o.test_identity && !o.cpu_codec);
        if (!listener) throw std::runtime_error(listener.error().what);
        const TransportListener& transport = *listener;
        std::unique_ptr<Backend> engine = open_backend(o, tier);
        engine->prepare();
        // Only serving stops gracefully, from the ready announcement on.
        // Before it, and in every other mode, SIGINT and SIGTERM terminate.
        std::signal(SIGINT, stop_handler);
        std::signal(SIGTERM, stop_handler);
        h->modelUp.store(o.test_identity ? 0 : 1);
        h->helperState.store(kHelperRunning, std::memory_order_release);
        const std::string ready = o.test_identity ? "IDENTITY TEST: no neural rendering"
                                                  : std::string("native ") + engine->name() +
                                                        " ready; display-encoded RGBA8/FP16 proxy";
        mapping.reason(ready);
        std::fprintf(stderr, "worker ready: %s%s\n", o.shm.c_str(), o.test_identity ? " [IDENTITY TEST]" : "");
        if (!o.test_identity)
            std::fprintf(stderr, "neural tier=%u; %s; %s; live controls enabled\n", tier, engine->processing().c_str(),
                         o.cpu_compose ? "CPU composition" : "native-resolution Vulkan composition");
        uint64_t frames = 0;
        unsigned previous_passes = 0;
        uint32_t last = h->seq_resp.load(std::memory_order_acquire);
        // When the latest work ended: an answer, a rebuild, or readiness.
        auto active = std::chrono::steady_clock::now();
        std::string failure;
        while (!stopping && !h->quit.load(std::memory_order_relaxed)) {
            accept_offers(transport, *engine);
            if (traces && !pending_trace) {
                if (auto taken = traces->take(); !taken)
                    std::fprintf(stderr, "diagnostic request rejected: %s\n", taken.error().what.c_str());
                else if ((pending_trace = std::move(*taken)))
                    h->controlSeq.fetch_add(1);
            }
            const uint32_t request = h->seq_req.load(std::memory_order_acquire);
            if (request == last) {
                if (follow_tier(engine, o, mapping, ready.c_str())) active = std::chrono::steady_clock::now();
                const timespec timeout{0, 100000000};
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_req), FUTEX_WAIT,
                        request, &timeout, nullptr, 0);
                // A request that arrived during the wait is served, however late.
                if (o.idle_exit && h->seq_req.load(std::memory_order_acquire) == request &&
                    std::chrono::steady_clock::now() - active >= std::chrono::seconds(o.idle_exit) &&
                    retire(h, request)) {
                    std::fprintf(stderr, "no request for %u s; stopping\n", o.idle_exit);
                    break;
                }
                continue;
            }
            // Writers store a setting before bumping controlSeq, so sample the
            // generations after this loop's own bump (the trace claim) and
            // before reading any setting, the dimensions or the input; sample
            // again after inference.
            const uint32_t control_sequence = h->controlSeq.load();
            const uint32_t tuning_sequence = h->tuningSeq.load();
            const uint32_t held_input = pending_trace ? h->holdFrame.load() : 0;
            const unsigned w = h->width.load(), height = h->height.load();
            last = request;
            try {
                ProcessingSettings settings;
                settings.fp16 = h->hdrEncode.load() != 0;
                const bool hdr = h->hdrDetected.load() != kHdrNone && h->colourMode.load() != kColourDisplay;
                settings.precision16 = hdr || h->sdr16Multipass.load() != 0;
                settings.motion = h->mvecEnabled.load() != 0;
                settings.motion_quality = ShmMVecQuality(h);
                settings.motion_grid = ShmMVecPixelSize(h);
                settings.tuning = read_tuning(h);
                settings.color_preserve = BitsToFloat(h->colorPreserveBits.load());
                if (!w || !height || w > kMaxW || height > kMaxH || h->format.load() != 1)
                    throw std::range_error("unsupported request dimensions or proxy format");
                settings.style = h->style.load();
                settings.skin_structure = BitsToFloat(h->skinStructureBits.load());
                const uint32_t mask = h->autoMask.load();
                settings.auto_mask = mask != 0;
                // Older/external clients must not enable an NVIDIA preset neither
                // network has, nor controls outside their ranges.
                if (h->preset.load() || settings.style > 2 || mask > 1 ||
                    !(settings.skin_structure >= -1 && settings.skin_structure <= 2))
                    throw std::range_error("the preset must be 0, style 0..2, auto-mask 0 or 1 and skin structure -1..2");
                if (!(settings.color_preserve >= 0 && settings.color_preserve <= 1))
                    throw std::range_error("invalid color preservation strength");
                unwrap(dlsslop::validate_native_tuning(settings.tuning));
                const size_t bytes = size_t(w) * height * (settings.fp16 ? 8 : 4);
                // A live control change takes effect on the next request;
                // never shorten or extend a chain partway through a frame.
                unsigned passes = ShmPasses(h);
                // A count the backend cannot run is replaced with the most it can, as an unusable tier is.
                if (passes > engine->max_passes) h->passes.store(passes = engine->max_passes);
                if (!o.test_identity && passes != previous_passes) {
                    std::fprintf(stderr, "neural passes=%u; one final composition per frame\n", passes);
                    previous_passes = passes;
                }
                // The request's frames: an imported device-local pair, or the channel's slots.
                Backend::Frames io{mapping.input, mapping.output};
                if (const uint32_t generation = h->transportGen.load()) {
                    io = engine->frames(generation, bytes);
                    if (io.slot < 0) {
                        h->transportMiss.store(generation);
                        throw std::range_error("request names device-local frames this worker has not imported");
                    }
                } else {
                    engine->pin(mapping.input, mapping.output, bytes);
                }
                // A frame of a new shape needs a build: seconds in which the layer
                // presents its own frames rather than waiting for this one.
                if (!engine->fits(w, height, passes, settings)) {
                    h->helperState.store(kHelperStarting);
                    std::fprintf(stderr, "building the network for %ux%u%s\n", w, height, settings.fp16 ? " FP16" : "");
                    engine->reshape(w, height, passes, settings);
                    h->helperState.store(kHelperRunning, std::memory_order_release);
                }
                engine->infer(io, w, height, passes, settings, pending_trace.get());
                if (h->seq_req.load(std::memory_order_acquire) != request)
                    throw std::range_error("request changed during inference; old answer discarded");
                std::string trace_metadata;
                if (pending_trace) {
                    dlsslop::TraceFrameMetadata metadata;
                    metadata.frame_seq = request;
                    metadata.control_seq = control_sequence;
                    metadata.tuning_seq = tuning_sequence;
                    metadata.control_seq_end = h->controlSeq.load();
                    metadata.tuning_seq_end = h->tuningSeq.load();
                    metadata.held_input = held_input;
                    metadata.held_input_end = h->holdFrame.load();
                    std::vector<uint8_t> proxy(bytes);
                    engine->read_back(proxy.data(), io.proxy, bytes);
                    metadata.source_proxy_hash = 14695981039346656037ull;
                    for (uint8_t byte : proxy)
                        metadata.source_proxy_hash = (metadata.source_proxy_hash ^ byte) * 1099511628211ull;
                    metadata.passes = passes;
                    metadata.geometry = unwrap(dlsslop::geometry(w, height, engine->tier()));
                    metadata.fp16_proxy = settings.fp16;
                    metadata.fp16_feedback = settings.precision16;
                    metadata.motion = settings.motion;
                    metadata.intensity = settings.tuning.intensity;
                    metadata.local_tone = settings.tuning.tone;
                    metadata.local_structure = settings.tuning.structure;
                    metadata.sharpness = settings.tuning.sharpness;
                    metadata.color_preserve = settings.color_preserve;
                    trace_metadata = metadata.json();
                }
                if (!failure.empty()) { // Serving recovered: the failure is no longer current.
                    failure.clear();
                    mapping.reason(ready);
                }
                h->answeredW.store(w);
                h->answeredH.store(height);
                h->helperEvalMsBits.store(FloatToBits(engine->inference_ms));
                h->helperUploadMsBits.store(FloatToBits(engine->upload_ms));
                h->helperReadbackMsBits.store(FloatToBits(engine->readback_ms));
                ShmStore64(h->helperFramesLo, h->helperFramesHi, ++frames);
                h->seq_ok.store(request);
                h->seq_resp.store(request, std::memory_order_release);
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAKE,
                        1, nullptr, nullptr, 0);
                if (pending_trace) {
                    pending_trace->finish(trace_metadata);
                    pending_trace.reset();
                }
                if (o.once) break;
            } catch (const std::exception& e) {
                if (pending_trace) {
                    pending_trace->finish("{\"frame_seq\":" + std::to_string(request) + "}", e.what());
                    pending_trace.reset();
                }
                if (failure != e.what()) { // Report a persistent rejection once, not every frame.
                    mapping.reason(failure = e.what());
                    std::fprintf(stderr, "frame %u failed: %s\n", request, e.what());
                }
                h->answeredW.store(0);
                h->answeredH.store(0);
                h->seq_resp.store(request, std::memory_order_release);
                syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_resp), FUTEX_WAKE,
                        1, nullptr, nullptr, 0);
                // A rejection (std::range_error) is answered as failed and serving goes
                // on. Any other failure ends the worker in every mode: a HIP or engine
                // fault is never silently replaced by fake output.
                if (o.once || !dynamic_cast<const std::range_error*>(&e)) throw;
            }
            active = std::chrono::steady_clock::now();
        }
        // Stopped before the teardown, which can take a while and unlinks a
        // socket this daemon bound: a layer that offers or presents from now on
        // leaves its frames for the next daemon instead of waiting on this one.
        h->helperState.store(kHelperStopped);
    } catch (const std::exception& e) {
        mapping.reason(e.what());
        h->helperState.store(kHelperModelFailed);
        throw;
    }
}
} // namespace dlsslop
