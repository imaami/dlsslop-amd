// SPDX-License-Identifier: MIT
#include "serve.hpp"
#include "channel.h"
#include "geometry.h"
#include "open.hpp"
#include "processing.h"
#include "trace.h"
#include "transport.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <linux/futex.h>
#include <pthread.h>
#include <string>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace dlsslop {
namespace {
volatile sig_atomic_t stopping;
void stop_handler(int) { stopping = 1; }

using Clock = std::chrono::steady_clock;

void wake(std::atomic<uint32_t>& word)
{
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

// The layer must know the real neural raster before building its proxy.
// Otherwise it mistakes a worker-upscaled answer for native-resolution output
// and skips its detail-preserving composition branch. The identity mode
// publishes none: the mapping may retain a preceding neural worker's.
void publish_raster(const struct options& o, ShmHeader* h, unsigned tier)
{
    const bool neural = !o.test_identity;
    h->nativeModelMaxWidth.store(neural ? ShmNativeTier(tier)->width : 0);
    h->nativeModelMaxHeight.store(neural ? tier : 0);
}

// Counts the channel's heartbeat every 100 ms from start() until destroyed, so
// the layer sees a live daemon even while it builds a network.
class Heartbeat {
    ShmHeader* const h_;
    std::atomic<bool> stopping_ = false;
    pthread_t thread_{};
    bool running_ = false;

    static void* beat(void* self)
    {
        const auto& heartbeat = *static_cast<const Heartbeat*>(self);
        while (!heartbeat.stopping_.load(std::memory_order_relaxed)) {
            heartbeat.h_->heartbeat.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return nullptr;
    }

public:
    explicit Heartbeat(ShmHeader* h) : h_(h) {}
    Heartbeat(const Heartbeat&) = delete;
    ~Heartbeat()
    {
        if (!running_) return;
        stopping_.store(true, std::memory_order_relaxed);
        pthread_join(thread_, nullptr);
    }
    Result<void> start()
    {
        if (const int error = pthread_create(&thread_, nullptr, beat, this))
            return fail(std::string("start the heartbeat: ") + std::strerror(error));
        running_ = true;
        return {};
    }
};

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

// Between frames: rebuilds the network once for each tier a controller stores
// that differs from the active one. The layer presents its own frames while
// helperState reads Starting, then offers its device-local frames to the
// rebuilt engine. An unusable tier is overwritten with the active one. A failed
// rebuild ends the worker, leaving the active tier for the next one. True after
// a rebuild.
template <Engine E>
Result<bool> follow_tier(E& engine, const struct options& o, struct mapping& mapping, const std::string& ready)
{
    auto* h = mapping.channel.h;
    const unsigned wanted = h->nativeTier.load(), active = engine.tier();
    if (wanted == active) return false;
    if (!ShmNativeTier(wanted)) {
        h->nativeTier.store(active);
        return false;
    }
    if constexpr (!E::rebuilds_for_tier) {
        DLSSLOP_TRY(engine.retier(wanted));
        std::fprintf(stderr, "neural tier %u -> %u\n", active, wanted);
        publish_raster(o, h, wanted);
        return false;
    }
    h->helperState.store(kHelperStarting);
    h->modelUp.store(0);
    std::fprintf(stderr, "neural tier %u -> %u: rebuilding\n", active, wanted);
    mapping_reason(&mapping, ("rebuilding for neural tier " + std::to_string(wanted)).c_str());
    if (auto rebuilt = engine.retier(wanted); !rebuilt) {
        h->nativeTier.store(active);
        return std::unexpected(std::move(rebuilt).error());
    }
    publish_raster(o, h, wanted);
    mapping_reason(&mapping, ready.c_str());
    h->modelUp.store(o.test_identity ? 0 : 1);
    h->helperState.store(kHelperRunning, std::memory_order_release);
    return true;
}

// One request as the channel states it, checked.
struct Request {
    uint32_t number;
    // The generations sampled before any setting, for a trace.
    uint32_t control_sequence, tuning_sequence, held_input;
    unsigned width, height, passes;
    size_t bytes;
    struct processing_settings settings = processing_settings();
};

// The request's settings, or its rejection. A pass count beyond what the engine
// runs is replaced in the channel with the most it can, as an unusable tier is.
Result<Request> read_request(const struct options& o, ShmHeader* h, Request request, unsigned max_passes,
                             unsigned& previous_passes)
{
    const unsigned w = request.width, height = request.height;
    if (!w || !height || w > kMaxW || height > kMaxH) return reject("unsupported request dimensions");
    struct error e;
    if (const enum error_code code = processing_read(h, &request.settings, &e)) return forward_c(code, e);
    const struct processing_settings& settings = request.settings;
    request.bytes = size_t(w) * height * (settings.fp16 ? 8 : 4);
    // A live control change takes effect on the next request;
    // never shorten or extend a chain partway through a frame.
    request.passes = ShmPasses(h);
    if (request.passes > max_passes) h->passes.store(request.passes = max_passes);
    if (!o.test_identity && request.passes != previous_passes) {
        std::fprintf(stderr, "neural passes=%u; one final composition per frame\n", request.passes);
        previous_passes = request.passes;
    }
    return request;
}

// The request's answer written, or why not; with a trace, its metadata.
template <Engine E>
Result<std::string> process(const Request& r, E& engine, struct mapping& mapping, struct frame_trace* trace)
{
    auto* h = mapping.channel.h;
    const struct processing_settings& settings = r.settings;
    // The request's frames: an imported device-local pair, or the channel's slots.
    Frames io{mapping.input, mapping.output};
    if (const uint32_t generation = h->transportGen.load()) {
        io = engine.frames(generation, r.bytes);
        if (io.slot < 0) {
            h->transportMiss.store(generation);
            return reject("request names device-local frames this worker has not imported");
        }
    } else {
        engine.pin(mapping.input, mapping.output, r.bytes);
    }
    // A frame of a new shape needs a build first: seconds for a new extent,
    // in which the layer presents its own frames rather than waiting for this
    // one, or milliseconds for a reshape that keeps the weights.
    if (!engine.fits(r.width, r.height, r.passes, settings)) {
        // A shape the engine does not take is rejected before a build: serving goes on as it was.
        DLSSLOP_TRY(engine.admit(r.width, r.height, r.passes, settings));
        h->helperState.store(kHelperStarting);
        std::fprintf(stderr, "building the network for %ux%u%s\n", r.width, r.height, settings.fp16 ? " FP16" : "");
        DLSSLOP_TRY(engine.reshape(r.width, r.height, r.passes, settings));
        h->helperState.store(kHelperRunning, std::memory_order_release);
    }
    DLSSLOP_TRY(engine.infer(io, r.width, r.height, r.passes, settings, trace));
    if (h->seq_req.load(std::memory_order_acquire) != r.number)
        return reject("request changed during inference; old answer discarded");
    if (!trace) return std::string();
    struct trace_metadata metadata{};
    metadata.frame_seq = r.number;
    metadata.control_seq = r.control_sequence;
    metadata.tuning_seq = r.tuning_sequence;
    metadata.control_seq_end = h->controlSeq.load();
    metadata.tuning_seq_end = h->tuningSeq.load();
    metadata.held_input = r.held_input;
    metadata.held_input_end = h->holdFrame.load();
    std::vector<uint8_t> proxy(r.bytes);
    DLSSLOP_TRY(engine.read_back(proxy.data(), io.proxy, r.bytes));
    metadata.source_proxy_hash = 14695981039346656037ull;
    for (uint8_t byte : proxy) metadata.source_proxy_hash = (metadata.source_proxy_hash ^ byte) * 1099511628211ull;
    metadata.passes = r.passes;
    struct error e;
    if (const enum error_code code = geometry_init(&metadata.geometry, r.width, r.height, engine.tier(), &e))
        return forward_c(code, e);
    metadata.fp16_proxy = settings.fp16;
    metadata.fp16_feedback = settings.precision16;
    metadata.motion = settings.motion;
    metadata.intensity = settings.tuning.intensity;
    metadata.local_tone = settings.tuning.tone;
    metadata.local_structure = settings.tuning.structure;
    metadata.sharpness = settings.tuning.sharpness;
    metadata.color_preserve = settings.color_preserve;
    char text[TRACE_METADATA_BYTES];
    if (const enum error_code code = trace_metadata_json(&metadata, text, &e)) return forward_c(code, e);
    return std::string(text);
}

// Imports every pending offer and answers each on its own connection.
template <Engine E>
void accept_offers(const struct transport_listener& listener, E& engine)
{
    if (listener.socket < 0) return;
    for (int peer; (peer = accept4(listener.socket, nullptr, nullptr, SOCK_CLOEXEC)) >= 0; close(peer)) {
        ShmTransportOffer offer{};
        int fds[2];
        if (!transport_receive_offer(peer, &offer, fds)) continue; // A start or liveness probe sends nothing.
        transport_answer(peer, engine.import(offer, fds));
        // What the engine did not import.
        for (const int fd : fds)
            if (fd >= 0) close(fd);
    }
}

// Answers requests until stopped, told to quit, idle, or failed. A rejected
// request is answered as failed and serving goes on; any other failure ends
// it, as does any failure with --once: a HIP or engine fault is never silently
// replaced by fake output. A request claimed for a trace is pending.
template <Engine E>
Result<void> serve_frames(const struct options& o, struct mapping& mapping, const struct transport_listener& transport,
                          struct trace_requests* traces, E& engine, struct frame_trace& pending)
{
    auto* h = mapping.channel.h;
    DLSSLOP_TRY(engine.prepare());
    // Only serving stops gracefully, from the ready announcement on.
    // Before it, and in every other mode, SIGINT and SIGTERM terminate.
    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    h->modelUp.store(o.test_identity ? 0 : 1);
    h->helperState.store(kHelperRunning, std::memory_order_release);
    const std::string ready = o.test_identity ? "IDENTITY TEST: no neural rendering"
                                              : std::string("native ") + engine.name() +
                                                    " ready; display-encoded RGBA8/FP16 proxy";
    mapping_reason(&mapping, ready.c_str());
    std::fprintf(stderr, "worker ready: %s%s\n", o.shm, o.test_identity ? " [IDENTITY TEST]" : "");
    if (!o.test_identity)
        std::fprintf(stderr, "neural tier=%u; %s; native-resolution Vulkan composition; live controls enabled\n",
                     engine.tier(), engine.processing().c_str());
    uint64_t frames = 0;
    unsigned previous_passes = 0;
    uint32_t last = h->seq_resp.load(std::memory_order_acquire);
    // When the latest work ended: an answer, a rebuild, or readiness.
    auto active = Clock::now();
    std::string failure;
    while (!stopping && !h->quit.load(std::memory_order_relaxed)) {
        accept_offers(transport, engine);
        if (traces && !pending.path) {
            struct error e;
            if (trace_requests_take(traces, &pending, &e))
                std::fprintf(stderr, "diagnostic request rejected: %s\n", e.what);
            else if (pending.path)
                h->controlSeq.fetch_add(1);
        }
        const uint32_t number = h->seq_req.load(std::memory_order_acquire);
        if (number == last) {
            if (DLSSLOP_TRY(follow_tier(engine, o, mapping, ready))) active = Clock::now();
            // Linux futex wake is emitted by the patched Vulkan layer. Timeout maintains liveness
            // with old clients and permits signals/quit; no GPU polling is involved.
            const timespec timeout{0, 100000000};
            syscall(SYS_futex, reinterpret_cast<uint32_t*>(&h->seq_req), FUTEX_WAIT, number, &timeout, nullptr, 0);
            // A request that arrived during the wait is served, however late.
            if (o.idle_exit && h->seq_req.load(std::memory_order_acquire) == number &&
                Clock::now() - active >= std::chrono::seconds(o.idle_exit) && retire(h, number)) {
                std::fprintf(stderr, "no request for %u s; stopping\n", o.idle_exit);
                break;
            }
            continue;
        }
        // Writers store a setting before bumping controlSeq, so sample the
        // generations after this loop's own bump (the trace claim) and
        // before reading any setting, the dimensions or the input; sample
        // again after inference. The members are initialized in this order.
        Request request{number, h->controlSeq.load(), h->tuningSeq.load(), pending.path ? h->holdFrame.load() : 0,
                        h->width.load(), h->height.load()};
        last = number;
        const auto answered = read_request(o, h, request, E::max_passes, previous_passes).and_then([&](const Request& r) {
            return process(r, engine, mapping, pending.path ? &pending : nullptr);
        });
        if (answered) {
            if (!failure.empty()) { // Serving recovered: the failure is no longer current.
                failure.clear();
                mapping_reason(&mapping, ready.c_str());
            }
            h->answeredW.store(request.width);
            h->answeredH.store(request.height);
            h->helperEvalMsBits.store(FloatToBits(engine.inference_ms));
            h->helperUploadMsBits.store(FloatToBits(engine.upload_ms));
            h->helperReadbackMsBits.store(FloatToBits(engine.readback_ms));
            h->helperFrames.store(++frames);
            h->seq_ok.store(number);
            h->seq_resp.store(number, std::memory_order_release);
            wake(h->seq_resp);
            if (pending.path) frame_trace_finish(&pending, answered->c_str(), nullptr);
            if (o.once) break;
        } else {
            const Error& error = answered.error();
            if (pending.path)
                frame_trace_finish(&pending, ("{\"frame_seq\":" + std::to_string(number) + "}").c_str(),
                                   error.what.c_str());
            // Report a persistent rejection once, not every frame. A frame that the Vulkan network
            // dropped is not serving's failure: the network logs drops, at most every 10 s.
            if (failure != error.what && error.what != VULKAN_NETWORK_DROPPED) {
                failure = error.what;
                mapping_reason(&mapping, failure.c_str());
                std::fprintf(stderr, "frame %u failed: %s\n", number, error.what.c_str());
            }
            h->answeredW.store(0);
            h->answeredH.store(0);
            h->seq_resp.store(number, std::memory_order_release);
            wake(h->seq_resp);
            if (o.once || !error.rejected) return std::unexpected(error);
        }
        active = Clock::now();
    }
    // Stopped before the teardown, which can take a while and unlinks a
    // socket this daemon bound: a layer that offers or presents from now on
    // leaves its frames for the next daemon instead of waiting on this one.
    h->helperState.store(kHelperStopped);
    return {};
}

// serve_frames(), with a request it claimed for a trace published however serving ends.
template <Engine E>
Result<void> serve(const struct options& o, struct mapping& mapping, const struct transport_listener& transport,
                   struct trace_requests* traces, E& engine)
{
    struct frame_trace pending{};
    auto served = serve_frames(o, mapping, transport, traces, engine, pending);
    frame_trace_fini(&pending);
    return served;
}

// serve() with the engine --backend selects, on the transport's socket.
Result<void> serve_engine(struct options& o, unsigned tier, struct mapping& mapping, struct trace_requests* traces)
{
    struct transport_listener transport;
    struct error e;
    if (const enum error_code code = transport_listener_init(&transport, o.shm, o.shm_length,
                                                             !o.test_identity && !o.cpu_codec, &e))
        return forward_c(code, e);
    auto served = with_engine(o, tier, [&](auto& engine) { return serve(o, mapping, transport, traces, engine); });
    transport_listener_fini(&transport);
    return served;
}

// run_worker() with the channel mapped and the trace directory, if any, taken.
Result<void> run_traced(struct options& o, struct mapping& mapping, struct trace_requests* traces)
{
    auto* h = mapping.channel.h;
    // An explicit tier replaces the channel's; otherwise a usable live one stays.
    const unsigned live = h->nativeTier.load();
    const unsigned tier = o.tier ? o.tier : (ShmNativeTier(live) ? live : kNativeDefaultTier);
    h->nativeTier.store(tier);
    if (o.passes) h->passes.store(o.passes);
    h->compositionBypass.store(o.test_identity ? 1 : 0);
    publish_raster(o, h, tier);
    Heartbeat heartbeat(h);
    DLSSLOP_TRY(heartbeat.start());
    mapping_reason(&mapping, o.test_identity ? "IDENTITY TEST: inference disabled" : "initializing the network");
    auto served = serve_engine(o, tier, mapping, traces);
    if (!served) {
        mapping_reason(&mapping, served.error().what.c_str());
        h->helperState.store(kHelperModelFailed);
    }
    return served;
}

// run_worker() with the channel mapped.
Result<void> run_mapped(struct options& o, struct mapping& mapping)
{
    if (!o.trace_dir_length) return run_traced(o, mapping, nullptr);
    struct trace_requests traces;
    struct error e;
    if (const enum error_code code = trace_requests_init(&traces, o.trace_dir, o.trace_dir_length, o.shm,
                                                         o.shm_length, &e))
        return forward_c(code, e);
    auto served = run_traced(o, mapping, &traces);
    trace_requests_fini(&traces);
    return served;
}
} // namespace

Result<void> run_worker(struct options& o)
{
    struct mapping mapping;
    struct error e;
    if (const enum error_code code = mapping_init(&mapping, o.shm, o.shm_length, &e)) return forward_c(code, e);
    auto served = run_mapped(o, mapping);
    mapping_fini(&mapping);
    return served;
}
} // namespace dlsslop
