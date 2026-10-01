// VK_LAYER_NV_dlssnr — Linux-side Vulkan layer (loaded by the host loader,
// including the one inside Wine/winevulkan). Hooks the swapchain lifecycle;
// on present, ships the frame to the selected helper over shared
// memory and presents the neural-processed result.
//
// Enabled implicitly via enable_environment DLSSNR_ENABLE=1 (winevulkan
// rejects explicitly-named layers, so implicit enable is required under Wine).
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/shm_protocol.h"
#include "composition.h"
#include "hotkey.h"
#include "log.h"
#include "vk_table.h"
#include "device_features.h"
#include "network_module.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>

// The layer's name has to differ per architecture.
//
// The loader keys implicit layers by name, so two manifests claiming the same name are one layer to
// it: it keeps whichever it read first and then rejects it for the process's word size, reporting
// only "Requested layer VK_LAYER_NV_dlssnr was wrong bit-type" -- with the manifest that would have
// worked sitting unread beside it. Steam hit the same wall and answered it the same way, which is why
// its overlay is VK_LAYER_VALVE_steam_overlay_32 next to _64 rather than one name twice.
#ifdef DLSSLOP_LAYER_NAME
#define VK_LAYER_NAME DLSSLOP_LAYER_NAME
#elif defined(DLSSNR_LAYER_32)
#define VK_LAYER_NAME "VK_LAYER_NV_dlssnr_32"
#else
#define VK_LAYER_NAME "VK_LAYER_NV_dlssnr"
#endif

// ---------------------------------------------------------------------------
// Shared memory transport
// ---------------------------------------------------------------------------
struct ShmMap {
    // Three mappings rather than one.
    //
    // The file is a header followed by two pixel regions each large enough for the biggest frame the
    // protocol allows, which is a quarter of a gigabyte in total. Mapping all of it was fine on
    // 64-bit and is not on 32-bit: a 32-bit game has about 3 GB of address space and would be handing
    // a tenth of it to a reservation it never touches. Both region offsets are page-aligned by
    // construction, so each can be mapped on its own at the size actually in use -- a few megabytes
    // for a real frame instead of 265.
    int fd = -1;
    ShmHeader* hdr = nullptr;
    uint8_t* inPixels = nullptr;
    uint8_t* outPixels = nullptr;
    size_t mappedFrameBytes = 0;
    uint32_t seq = 0;
    uint32_t timeouts = 0;

    // Liveness, so a game is never made to wait on a helper that is not there.
    uint32_t firstHeartbeat = 0;
    bool everAnswered = false;
    double retryAfterMs = 0.0;
    double startAfterMs = 0.0; // StartWorker's rate limit.
    uint32_t lastControlSeq = 0;
    uint32_t lastHeartbeat = 0;
    bool dead = false;
    // The file path, kept so the transport socket and the producer lock can be named beside it.
    std::string path;
    int producerFd = -1;

    ~ShmMap() {
        if (inPixels) munmap(inPixels, mappedFrameBytes);
        if (outPixels) munmap(outPixels, mappedFrameBytes);
        if (hdr) munmap(hdr, kHeaderBytes);
        if (fd >= 0) close(fd);
        if (producerFd >= 0) close(producerFd);
    }
};

// The channel has one pixel slot. A game can have a separate launcher or helper
// process with its own Vulkan device, so the in-process primary-swapchain election
// is not sufficient. Contending processes present untouched while another owns the
// slot. This is a separate file because the worker locks the SHM file for its lifetime.
// Only this user can create it: ShmOpen's EnsureParentDir admits nothing but a private
// directory. A failed open is retried on the next frame.
class NativeFrameGuard {
    int fd_ = -1;
public:
    explicit NativeFrameGuard(ShmMap& s) {
        if (!s.hdr) return;
        if (s.producerFd < 0)
            s.producerFd = open((s.path + ".producer.lock").c_str(),
                                O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (!flock(s.producerFd, LOCK_EX | LOCK_NB)) fd_ = s.producerFd;
    }
    ~NativeFrameGuard() { if (fd_ >= 0) flock(fd_, LOCK_UN); }
    NativeFrameGuard(const NativeFrameGuard&) = delete;
    NativeFrameGuard& operator=(const NativeFrameGuard&) = delete;
    explicit operator bool() const { return fd_ >= 0; }
};

// The directory now lives under /tmp, which is world-writable, so it is worth checking that what we
// are about to open really is ours: a directory, owned by this uid, with nothing granted to anyone
// else. Anything else and we refuse rather than create the file inside it.
static bool EnsureParentDir(const std::string& path) {
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    std::string dir = path.substr(0, slash);
    size_t pos = 1;
    while ((pos = dir.find('/', pos)) != std::string::npos) {
        mkdir(dir.substr(0, pos).c_str(), 0700);
        pos += 1;
    }
    mkdir(dir.c_str(), 0700);

    struct stat st{};
    if (lstat(dir.c_str(), &st) != 0) { log_printf("[shm] %s is missing", dir.c_str()); return false; }
    if (!S_ISDIR(st.st_mode) || st.st_uid != getuid() || (st.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        log_printf("[shm] refusing %s: it is not a private directory owned by this user", dir.c_str());
        return false;
    }
    return true;
}

// Maps the two pixel regions at the size this frame needs, remapping when the size changes.
static bool ShmMapFrames(ShmMap& s, size_t bytes) {
    if (s.mappedFrameBytes >= bytes && s.inPixels && s.outPixels) return true;
    if (bytes > kMaxFrame) return false;

    // Round up to 64 KiB so a small change in resolution does not remap every frame.
    const size_t want = (bytes + 65535) / 65536 * 65536;

    if (s.inPixels) munmap(s.inPixels, s.mappedFrameBytes);
    if (s.outPixels) munmap(s.outPixels, s.mappedFrameBytes);
    s.inPixels = s.outPixels = nullptr;
    s.mappedFrameBytes = 0;

    void* in = mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, s.fd, (off_t) kHeaderBytes);
    if (in == MAP_FAILED) { log_printf("[shm] could not map the input region (%zu bytes)", want); return false; }

    void* out = mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, s.fd, (off_t) (kHeaderBytes + kMaxFrame));
    if (out == MAP_FAILED) {
        munmap(in, want);
        log_printf("[shm] could not map the output region (%zu bytes)", want);
        return false;
    }

    s.inPixels = (uint8_t*) in;
    s.outPixels = (uint8_t*) out;
    s.mappedFrameBytes = want;
    return true;
}

static bool ShmOpen(ShmMap& s) {
    if (s.hdr) return true;
    const char* path = getenv("DLSSNR_SHM");
    std::string p = (path && *path) ? path : ShmDefaultPath();
    if (!EnsureParentDir(p)) return false;
    int fd = open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) { log_printf("[shm] open %s failed", p.c_str()); return false; }
    // The file still spans the whole protocol -- the offsets are fixed and both sides agree on them --
    // but it is sparse, so the size on disk is what has actually been written.
    size_t total = ShmTotalBytes();
    struct stat st{};
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < total) {
        if (ftruncate(fd, (off_t)total) != 0) { close(fd); return false; }
    }

    void* m = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { log_printf("[shm] mmap of the header failed"); close(fd); return false; }
    s.fd = fd;
    s.hdr = (ShmHeader*)m;
    // A mapping left by an older build has a different magic, a different version, or a header
    // laid out differently; re-initialising is the only safe reading of any of those.
    //
    // But say so, loudly. A live process on the other side of the mismatch keeps re-initialising the
    // other way, and the two then silently reset each other's settings forever -- the layer keeps
    // composing with its old field set and every setting the newer side writes is invisible. That is
    // indistinguishable from "the new feature does nothing", which is how a stale layer reads until
    // someone checks the log.
    if (s.hdr->magic.load() != kShmMagic || s.hdr->version.load() != kShmVersion ||
        s.hdr->passes.load() == 0) {
        if (s.hdr->magic.load() == kShmMagic && s.hdr->version.load() != kShmVersion)
            log_printf("[shm] header is version %u but this layer is v%u -- another process is out of date, "
                       "re-initialising it; update the layer, the helper and the GUI together",
                       s.hdr->version.load(), kShmVersion);
        ShmInitDefaults(s.hdr);
    }
    s.lastHeartbeat = s.hdr->heartbeat.load();
    s.firstHeartbeat = s.lastHeartbeat;
    s.path = p;
    log_printf("[shm] attached %s seq_req=%u seq_resp=%u", p.c_str(),
               s.hdr->seq_req.load(), s.hdr->seq_resp.load());
    return true;
}

static bool ShmNeuralEnabled(ShmMap& s) {
    if (!ShmOpen(s)) return true;
    if (s.hdr->quit.load()) { s.dead = true; return false; }
    const uint32_t ctrl = s.hdr->controlSeq.load();
    if (ctrl != s.lastControlSeq) {
        s.lastControlSeq = ctrl;
        if (s.dead && ::ShmNeuralEnabled(s.hdr)) {
            s.dead = false;
            s.timeouts = 0;
            log_printf("[shm] control changed, re-enabling");
        }
    }
    const uint32_t hb = s.hdr->heartbeat.load();
    if (hb != s.lastHeartbeat) {
        s.lastHeartbeat = hb;
        // A heartbeat alone is not a reason to try again immediately. The helper ticks it while it
        // sits idle, so a helper that is up but not answering used to re-enable the layer the moment
        // it had given up -- which cost the game another round of full-length waits, over and over.
        // That is the stutter: recover, stall, give up, recover.
        if (s.dead && log_now_ms() >= s.retryAfterMs && ::ShmNeuralEnabled(s.hdr)) {
            s.dead = false;
            s.timeouts = 0;
            log_printf("[shm] helper heartbeat, trying again");
        }
    }
    if (s.dead) return false;
    return ::ShmNeuralEnabled(s.hdr);
}

// One round trip: publish the proxy, wait for the model's answer, and copy it back unless it
// crossed in the device-local transport.
//
// What crosses is the proxy at the model's own resolution: R8G8B8A8_UNORM or, with hdrEncode,
// R16G16B16A16_SFLOAT, sRGB-encoded at either precision because the encode has already done that
// work on the GPU. The helper therefore never has to know what format the game presents in, and
// the working scale reduces this copy quadratically.
static bool ShmProcessFrame(ShmMap& s, uint32_t w, uint32_t h, size_t bytes, const void* proxy,
                          void* modelOut, bool hdrEncode, uint32_t transportGen) {
    if (s.dead) return false;
    if (!ShmOpen(s)) { s.dead = true; return false; }
    if (w > kMaxW || h > kMaxH || w < kMinW || h < kMinH) return false;
    if (bytes != size_t(w) * h * 4 && bytes != size_t(w) * h * 8) return false;
    if (s.hdr->quit.load()) { s.dead = true; return false; }

    const bool time = log_time_enabled();
    const double t0 = log_now_ms();
    if (!ShmMapFrames(s, bytes)) { s.dead = true; return false; }
    // A request that names a device-local transport generation crosses in the exported buffers:
    // the GPU already wrote the proxy where the daemon reads it, and there is nothing to copy.
    if (!transportGen) std::memcpy(s.inPixels, proxy, bytes);
    const double tCopy = log_now_ms();
    s.hdr->width.store(w);
    s.hdr->height.store(h);
    s.hdr->format.store(1u);  // RGBA byte order either way; the float path keeps the same swizzle
    // Say what the bytes ARE before announcing them: the helper sizes its read by this, never by
    // what it hopes the layer has switched to. The release fence below covers it like the pixels.
    s.hdr->hdrEncode.store(hdrEncode ? 1u : 0u);
    s.hdr->transportGen.store(transportGen);
    uint32_t req = s.hdr->seq_req.load() + 1;
    // The release pairs with the helper's acquire on seq_resp: everything this process wrote --
    // the proxy, whether by the GPU into the exported buffer or by the memcpy above -- is visible
    // to the helper before it sees the new request number. (The GPU's own write is fenced earlier,
    // by leg 1's vkWaitForFences; this fence covers the host-visible ordering across processes.)
    std::atomic_thread_fence(std::memory_order_release);
    s.hdr->seq_req.store(req);
    syscall(SYS_futex, &s.hdr->seq_req, FUTEX_WAKE, 1, nullptr, nullptr, 0);

    // How long this frame may wait, which is a question about whether anyone is listening.
    //
    // A live helper needs real time: the model is milliseconds of work and building its feature on the
    // first frame is far more than that. A helper that is not running needs none at all, and the old
    // fixed second-per-frame budget meant a game whose helper was simply not started froze for eight
    // seconds before the layer gave up. That is what this is for.
    // Is anything listening? The helper says so itself, from the moment it attaches until it exits,
    // which is the only signal that stays true while it is busy. Heartbeats do not: it stops ticking
    // them precisely while it is building the model's feature.
    const bool helperPresent = s.hdr->helperState.load() != kHelperStopped;

    // The first frame of a size is not like the others. It makes the helper load the model and build
    // a feature -- measured at 194 ms for a small frame and more for a large one -- against about 4 ms
    // once it is warm. Timing that out and giving up is how a working helper gets abandoned before it
    // has answered once.
    const bool warmingUp = !s.everAnswered;
    // HIP passes share one model and run sequentially. Allow a longer chain
    // under game GPU contention without changing the single-pass deadline.
    const double frameBudgetMs = std::min(10000.0, 1000.0 * double(ShmPasses(s.hdr)));
    const double budgetMs = !helperPresent ? 20.0 : (warmingUp ? 10000.0 : frameBudgetMs);

    // Wait for the helper (fail-open: present the original frame on timeout).
    const double tSignal = log_now_ms();
    uint32_t beat = s.hdr->heartbeat.load();
    double beatAt = 0.0;
    for (;;) {
        const uint32_t response = s.hdr->seq_resp.load(std::memory_order_acquire);
        if (response == req) {
            s.timeouts = 0;
            s.everAnswered = true;
            // The helper's GPU wrote the answer into this region (or the memcpy below reads the
            // staging copy of it); the acquire pairs with the helper's release before seq_resp.
            std::atomic_thread_fence(std::memory_order_acquire);
            // The helper answers even when it could not use the frame. seq_ok says whether the
            // answer is worth composing; when it is not, the game's own frame is what to present.
            // The echo says the answer was made for this raster: another swapchain (the Steam
            // overlay, or this one's predecessor mid-resize) may have had its request answered in
            // the meantime, and seq_resp only counts. Composing that answer here would copy a
            // different number of bytes into these surfaces -- the row-shifted colour garbage this
            // check exists to refuse.
            const bool ok = s.hdr->seq_ok.load() == req && s.hdr->answeredW.load() == w &&
                            s.hdr->answeredH.load() == h;
            if (!ok) log_printf("[shm] helper could not use frame %u (ok=%u)", req, s.hdr->seq_ok.load());
            if (ok && !transportGen) std::memcpy(modelOut, s.outPixels, bytes);
            if (time) {
                static int frameNo = 0;
                if (++frameNo % log_time_interval() == 0) {
                    const double tDone = log_now_ms();
                    log_printf("[time] shm copy=%.2f signal=%.2f wait=%.2f total=%.2f ms",
                               tCopy - t0, tSignal - tCopy, tDone - tSignal, tDone - t0);
                }
            }
            return ok;
        }
        if (s.hdr->quit.load()) { s.dead = true; return false; }
        const double elapsed = log_now_ms() - tSignal;
        if (elapsed >= budgetMs) break;
        // The worker ticks its heartbeat every 100 ms, also mid-frame. One that
        // exits or dies (even by SIGKILL) will not answer; stop waiting for it.
        if (s.hdr->helperState.load() != kHelperRunning) break;
        if (const uint32_t b = s.hdr->heartbeat.load(); b != beat) {
            beat = b;
            beatAt = elapsed;
        } else if (elapsed - beatAt > 500.0) {
            break;
        }
        // Shared futexes work across the two MAP_SHARED mappings. Comparing the
        // observed response value in the kernel closes the check-to-sleep race;
        // the deadline (under 50 ms) also lets a dead worker fail open and quit
        // remain responsive.
        const timespec timeout{0, long(std::min(50.0, budgetMs - elapsed) * 1000000.0)};
        syscall(SYS_futex, &s.hdr->seq_resp, FUTEX_WAIT, response, &timeout, nullptr, 0);
    }
    log_printf("[shm] worker did not answer frame %u in %.0f ms (state=%u heartbeat=%u); "
               "presenting original frames until it answers", req, log_now_ms() - tSignal,
               s.hdr->helperState.load(), s.hdr->heartbeat.load());
    // Four rather than eight, and with a pause before the next attempt, so giving up costs a
    // fraction of a second and retrying costs that again only every few seconds.
    if (++s.timeouts >= 4) {
        s.dead = true;
        s.retryAfterMs = log_now_ms() + 5000.0;
        log_printf("[shm] no answer in %.0f ms x4 (helper %s); passing frames through, retrying in 5s "
                   "(seq_req=%u seq_resp=%u heartbeat=%u)",
                   budgetMs, helperPresent ? "is present but silent" : "not running",
                   s.hdr->seq_req.load(), s.hdr->seq_resp.load(), s.hdr->heartbeat.load());
    }
    return false;
}

// A worker that stopped (idle, crashed) is started again by its systemd socket unit, for which a
// connection is enough. At most every two seconds, and never waiting: this is the present path.
static void StartWorker(ShmMap& s) {
    const double now = log_now_ms();
    if (now < s.startAfterMs) return;
    s.startAfterMs = now + 2000.0;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string path = ShmTransportPath(s.path);
    if (path.size() >= sizeof address.sun_path) return;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    const int sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    connect(sock, reinterpret_cast<const sockaddr*>(&address), sizeof address);
    close(sock);
}

// What became of a device-local transport offer: the daemon imported it; refused it, or it could
// not be sent; has not answered yet; or stopped before it took the offer, so the pair goes to the
// next daemon that serves.
enum class Offer { kReady, kDeclined, kWaiting, kLater };

// Hands the composition's exported frames to the daemon under a fresh generation and wakes it.
// The connection the answer comes on, or -1.
static int SendOffer(ShmMap& s, dlssnr::Composition& comp) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string path = ShmTransportPath(s.path);
    if (path.size() >= sizeof address.sun_path || !ShmOpen(s)) return -1;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    ShmTransportOffer offer{};
    offer.magic = kShmMagic;
    int fds[2];
    if (!comp.ExportTransport(fds, offer)) return -1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof fds)]{};
    iovec data{&offer, sizeof offer};
    msghdr message{};
    message.msg_iov = &data;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof control;
    cmsghdr* rights = CMSG_FIRSTHDR(&message);
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof fds);
    std::memcpy(CMSG_DATA(rights), fds, sizeof fds);
    const int sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    const bool sent = sock >= 0 && !connect(sock, reinterpret_cast<const sockaddr*>(&address), sizeof address) &&
                      sendmsg(sock, &message, MSG_NOSIGNAL) == ssize_t(sizeof offer);
    close(fds[0]);
    close(fds[1]);
    if (!sent) {
        if (sock >= 0) close(sock);
        return -1;
    }
    // The daemon takes offers between requests.
    syscall(SYS_futex, &s.hdr->seq_req, FUTEX_WAKE, 1, nullptr, nullptr, 0);
    return sock;
}

// The daemon's answer to the composition's offer: the first look sends it and waits a moment,
// later ones only look. No answer within expires counts as a refusal.
static Offer OfferTransport(ShmMap& s, dlssnr::Composition& comp, double& expires) {
    int wait = 0;
    if (comp.OfferConnection() < 0) {
        const int connection = SendOffer(s, comp);
        if (connection < 0)
            return s.hdr->helperState.load(std::memory_order_acquire) == kHelperRunning ? Offer::kDeclined
                                                                                    : Offer::kLater;
        comp.AwaitAnswer(connection);
        expires = log_now_ms() + 5000.0;
        wait = 250;
    }
    pollfd answer{comp.OfferConnection(), POLLIN, 0};
    if (poll(&answer, 1, wait) != 1) return log_now_ms() >= expires ? Offer::kDeclined : Offer::kWaiting;
    uint8_t imported = 0;
    if (recv(answer.fd, &imported, 1, 0) != 1) return Offer::kLater; // Closed unanswered.
    return imported ? Offer::kReady : Offer::kDeclined;
}

// ---------------------------------------------------------------------------
// Dispatch chains
// ---------------------------------------------------------------------------
struct InstanceChain {
    VkInstance self = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;
    // The game's own: device features beyond it need extensions of their own.
    uint32_t apiVersion = VK_API_VERSION_1_0;

    // The instance-level entry points the layer and the composition need, resolved once. Kept here
    // rather than on the device chain because this is where the VkInstance handle is in scope.
    dlssnr::InstanceTable table;
};

struct SwapchainState {
    std::vector<VkImage> images;
    // The present queue's family: the composition's, and the in-layer network's, which converts
    // formats with blits and so needs a graphics family.
    uint32_t family = 0;
    bool graphics = false;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // HdrKind: what this swapchain's format and colour space say the frame carries. The float
    // swapchain holds linear light; a 10-bit one with a PQ colour space holds ST 2084 code.
    uint32_t hdrKind = kHdrNone;
    uint32_t width = 0, height = 0;
    // Two fences, not one. Leg 1's must be waited on before the proxy is handed to the helper --
    // the sequence number is the helper's only ordering signal, and it may not be bumped ahead of
    // the write it announces. Leg 2's needs no wait at all in its own frame: the present waits on
    // the image's semaphore, so the GPU orders them without the CPU. The wait moves to the start of
    // the next present, where the command buffer and the composed surfaces are reused, which takes
    // a full GPU stall out of the frame it belongs to.
    VkFence fenceLeg1 = VK_NULL_HANDLE;
    VkFence fenceLeg2 = VK_NULL_HANDLE;
    bool leg2Pending = false;
    // When the composition's outstanding transport offer counts as refused (log_now_ms).
    double offerExpires = 0.0;
    // Per image: leg 2 signals it and the present waits on it. Queue order alone does not order a
    // present behind earlier work, and an image is acquired again only after its present waited.
    std::vector<VkSemaphore> leg2Done;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    // Reused by the present path; resizing only occurs when an unusual caller supplies more waits.
    std::vector<VkPipelineStageFlags> waitStages;
    bool ready = false;
    bool passThrough = false;

    // The pass. Owns every surface it needs, including the transport pair -- exported device-local
    // memory when the daemon imports it, host-visible staging when it does not.
    std::unique_ptr<dlssnr::Composition> comp;
};

struct DeviceChain {
    InstanceChain* instance = nullptr;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice self = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr next_dpa = nullptr;

    // The device-level entry points the layer and the composition need, resolved once.
    dlssnr::DeviceTable table;

    // The loader's hook for installing a dispatch table on a dispatchable object a layer creates.
    // Handed to every layer in its own VkLayerDeviceCreateInfo node; see Hook_CreateDevice.
    PFN_vkSetDeviceLoaderData setDeviceLoaderData = nullptr;
    std::atomic<bool> inert{false};
    std::mutex lock;
    // Held by the in-layer network's build around each of its submits, and by the layer's own
    // device waits, which run outside the lock: a device wait may not overlap a submit.
    std::mutex networkSubmit;

    bool exportMemory = false;       // VK_KHR_external_memory_fd was enabled
    // The ledger: the in-layer network was asked for and its features and
    // extensions were enabled.
    bool network = false;
    // The in-layer network, opened on the first frame; off once it failed. Its memory belongs to
    // the queue family it opened on.
    DlsslopNetwork* inLayer = nullptr;
    bool inLayerOff = false;
    uint32_t inLayerFamily = 0;
    // The swapchain whose frame last reached the network. A build frees what that frame used.
    SwapchainState* inLayerLast = nullptr;
    // What the channel's layer reason last said of it.
    std::string networkReason;
    std::unordered_map<VkSwapchainKHR, SwapchainState> swapchains;
    std::unordered_map<VkQueue, uint32_t> queueFamilies;
    ShmMap shm;
    uint64_t framesComposed = 0;
    uint64_t framesPassedThrough = 0;
};

static std::unordered_map<VkInstance, InstanceChain> g_instances;
static std::unordered_map<VkPhysicalDevice, InstanceChain*> g_phys;
static std::unordered_map<VkDevice, DeviceChain*> g_devices;
static std::mutex g_stateMutex;
// The in-layer network's module (layer/network_module.h), beside the layer,
// loaded once for the first device that enabled the network.
static dlssnr::NetworkModule g_network;

// The one swapchain allowed to drive the neural round trip, chosen as the largest in the process.
//
// The shared-memory channel carries a single raster at a time, but a process can present more than
// one swapchain -- the game window and the Steam overlay, or, mid-resize, the old and new windows at
// once. Feeding all of them through one channel makes the helper rebuild its model on every size
// switch and lets one swapchain be handed another's answer. The largest is the game; the rest present
// raw. The record is global rather than per-device because the overlay builds its own VkDevice.
struct PrimarySwap {
    VkDevice device = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint64_t area = 0;
};
static PrimarySwap g_primary;
// Its own mutex, never nested with dc->lock or g_stateMutex, so the lock order in the present hook
// cannot invert against the device hooks.
static std::mutex g_primaryMutex;

// Adopts a larger swapchain; a present from anything else passes through untouched.
static bool ClaimPrimary(VkDevice device, VkSwapchainKHR swapchain, uint32_t w, uint32_t h) {
    std::lock_guard<std::mutex> lk(g_primaryMutex);
    const uint64_t area = uint64_t(w) * h;
    if (g_primary.swapchain == swapchain && g_primary.device == device) return true;
    if (g_primary.swapchain != VK_NULL_HANDLE && area <= g_primary.area) return false;
    g_primary.device = device;
    g_primary.swapchain = swapchain;
    g_primary.area = area;
    return true;
}

static void ReleasePrimary(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lk(g_primaryMutex);
    if (g_primary.swapchain == swapchain && g_primary.device == device) g_primary = PrimarySwap{};
}

// Where this copy of the layer was loaded from, for the duplicate check below.
static std::string LayerObjectPath() {
    Dl_info info{};
    if (dladdr((const void*)&LayerObjectPath, &info) && info.dli_fname && *info.dli_fname)
        return info.dli_fname;
    return std::string();
}

// True when a *different* copy of this layer is already in the chain.
//
// Local packaging installs an implicit-layer manifest pointing at the build tree while install.sh installs
// another pointing at the install prefix, and the loader honours both: two copies of the layer, two
// present hooks, two full round trips, and a single shared-memory file with two writers racing on
// one sequence number. Only the first copy stays live; the rest declare themselves inert and pass
// everything through, which turns a corrupted picture or a hang into one warning line.
//
// The claim is the object's own path rather than a bare flag, so a second call into the same copy --
// which is legal, the loader may negotiate more than once -- is told apart from a second copy.
static bool DuplicateLayerCopy() {
    static const bool dup = [] {
        const std::string self = LayerObjectPath();
        const char* claimed = getenv("DLSSNR_LAYER_OBJECT");
        if (claimed && *claimed) {
            if (self.empty() || self == claimed) return false;
            log_printf("[layer] another copy is already loaded from %s; this copy (%s) stays inert. "
                       "Remove one of the implicit-layer manifests.", claimed, self.c_str());
            return true;
        }
        if (!self.empty()) setenv("DLSSNR_LAYER_OBJECT", self.c_str(), 0);
        return false;
    }();
    return dup;
}

// One set of keyboards for the process, however many devices the game creates.
static dlssnr::Hotkeys g_hotkeys;

// The key to watch, from the header if the interface has set one and from the environment otherwise,
// so it can be bound in a launch option without the interface being involved.
static uint32_t ToggleKey(const ShmHeader* hdr) {
    static const uint32_t fromEnv = [] {
        const char* v = getenv("DLSSNR_TOGGLE_KEY");
        return v && *v ? dlssnr::KeyCodeFromName(v) : 0u;
    }();
    if (fromEnv) return fromEnv;
    return hdr ? hdr->toggleKey.load() : 0u;
}

// Polled before anything asks whether the pass is enabled, because asking first would make turning it
// off a one-way door: the early return would skip the very code that reads the key to turn it back on.
static void PollHotkeys(DeviceChain* dc) {
    if (!ShmOpen(dc->shm) || !dc->shm.hdr) return;
    const uint32_t key = ToggleKey(dc->shm.hdr);
    if (!key || !g_hotkeys.Pressed(key)) return;

    const bool wasOn = dc->shm.hdr->enabled.load() != 0;
    dc->shm.hdr->enabled.store(wasOn ? 0u : 1u);
    dc->shm.hdr->controlSeq.fetch_add(1);
    log_printf("[hotkey] %s -> neural rendering %s", dlssnr::KeyNameFromCode(key), wasOn ? "off" : "on");
}

static bool LayerEnabled() {
    static const bool e = [] {
        if (DuplicateLayerCopy()) return false;
        const char* v = getenv("VKLayer_DLSS5");
        const char* upper = getenv("VKLAYER_DLSS5");
        if ((v && v[0] == '1') || (upper && upper[0] == '1')) return true;
        const char* o = getenv("DLSSNR_ENABLE");
        return o && o[0] == '1';
    }();
    return e;
}

// ---------------------------------------------------------------------------
// Instance hooks
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
    auto* link = const_cast<VkLayerInstanceCreateInfo*>((const VkLayerInstanceCreateInfo*)pCreateInfo->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerInstanceCreateInfo*)link->pNext;
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto create = (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;

    // Documented pattern: keep the link node in pNext (layers below need it)
    // and advance u.pLayerInfo so the next layer resolves its own chain entry.
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    VkResult res = create(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    InstanceChain chain{};
    chain.self = *pInstance;
    chain.next_gipa = next_gipa;
    if (pCreateInfo->pApplicationInfo && pCreateInfo->pApplicationInfo->apiVersion)
        chain.apiVersion = pCreateInfo->pApplicationInfo->apiVersion;

    chain.table.next_gipa = next_gipa;
    chain.table.Load(*pInstance);

    std::lock_guard<std::mutex> lk(g_stateMutex);
    g_instances[*pInstance] = chain;
    log_printf("[layer] vkCreateInstance -> %p", (void*)*pInstance);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyInstance(VkInstance instance,
                                                       const VkAllocationCallbacks* pAllocator) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    auto it = g_instances.find(instance);
    if (it == g_instances.end()) return;
    auto destroy = it->second.table.vkDestroyInstance;
    InstanceChain* chain = &it->second;
    g_instances.erase(it);
    for (auto pit = g_phys.begin(); pit != g_phys.end();)
        pit = (pit->second == chain) ? g_phys.erase(pit) : std::next(pit);
    if (destroy) destroy(instance, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumeratePhysicalDevices(
    VkInstance instance, uint32_t* pCount, VkPhysicalDevice* pPhysicalDevices) {
    InstanceChain* chain = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_instances.find(instance);
        if (it != g_instances.end()) chain = &it->second;
    }
    if (!chain || !chain->table.vkEnumeratePhysicalDevices) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = chain->table.vkEnumeratePhysicalDevices(instance, pCount, pPhysicalDevices);
    if (res == VK_SUCCESS && pPhysicalDevices) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        for (uint32_t i = 0; i < *pCount; ++i) g_phys[pPhysicalDevices[i]] = chain;
    }
    return res;
}

// ---------------------------------------------------------------------------
// Device hooks
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    auto* link = const_cast<VkLayerDeviceCreateInfo*>((const VkLayerDeviceCreateInfo*)pCreateInfo->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerDeviceCreateInfo*)link->pNext;
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_dpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    auto create = (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;

    // A second node in the same chain carries vkSetDeviceLoaderData. Every dispatchable object this
    // layer allocates has to be passed through it; see SetLoaderData below for why.
    PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
    for (const auto* n = (const VkLayerDeviceCreateInfo*)pCreateInfo->pNext; n;
         n = (const VkLayerDeviceCreateInfo*)n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            n->function == VK_LOADER_DATA_CALLBACK) {
            setLoaderData = n->u.pfnSetDeviceLoaderData;
            break;
        }
    }

    InstanceChain* ic = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_phys.find(physicalDevice);
        if (it != g_phys.end()) ic = it->second;
    }

    // VK_KHR_external_memory_fd is what vkGetMemoryFdKHR needs to export the device-local
    // transport, so the proxy and the model's answer never pass through host memory. It is a device
    // extension and the application decides what the device enables, but a layer may add to that
    // list on the way down -- and does, when the pass is on, the device offers it, and the app did
    // not already enable it. If any of that is false the composition stages frames through host
    // memory.
    static const char* const kWantExts[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME };
    constexpr size_t kWantCount = sizeof(kWantExts) / sizeof(kWantExts[0]);
    const VkDeviceCreateInfo* effective = pCreateInfo;
    VkDeviceCreateInfo modified = *pCreateInfo;
    std::vector<const char*> enabledExts;
    if (LayerEnabled() && ic && ic->table.vkEnumerateDeviceExtensionProperties) {
        bool available[kWantCount] = {};
        bool enabled[kWantCount] = {};
        uint32_t n = 0;
        ic->table.vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> avail(n);
        if (n && ic->table.vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n,
                                                                avail.data()) == VK_SUCCESS) {
            for (uint32_t i = 0; i < n; ++i)
                for (size_t k = 0; k < kWantCount; ++k)
                    if (!std::strcmp(avail[i].extensionName, kWantExts[k])) available[k] = true;
        }
        for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
            for (size_t k = 0; k < kWantCount; ++k)
                    if (!std::strcmp(pCreateInfo->ppEnabledExtensionNames[i], kWantExts[k])) enabled[k] = true;
        for (size_t k = 0; k < kWantCount; ++k) {
            if (!available[k] || enabled[k]) continue;
            if (enabledExts.empty()) {
                enabledExts.reserve(pCreateInfo->enabledExtensionCount + kWantCount);
                for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
                    enabledExts.push_back(pCreateInfo->ppEnabledExtensionNames[i]);
            }
            enabledExts.push_back(kWantExts[k]);
        }
        if (!enabledExts.empty()) {
            modified.enabledExtensionCount = uint32_t(enabledExts.size());
            modified.ppEnabledExtensionNames = enabledExts.data();
            effective = &modified;
        }
    }
    // The in-layer network, on request, where the game's instance and device allow it.
    const char* networkOff = nullptr;
    const bool network = LayerEnabled() && ic && dlssnr::NetworkRequested() &&
                         !(networkOff = dlssnr::NetworkUnavailable(
                               physicalDevice, ic->apiVersion, ic->table.vkGetPhysicalDeviceProperties2,
                               ic->table.vkGetPhysicalDeviceFeatures2, ic->table.vkEnumerateDeviceExtensionProperties,
                               ic->table.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR));
    if (networkOff) log_printf("[layer] in-layer network unavailable: needs %s", networkOff);
    if (network && dlssnr::AddNetworkExtensions(modified, enabledExts)) effective = &modified;

    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto* const nextLayerInfo = link->u.pLayerInfo;
    // The features the layer adds. Our write-only storage shaders must accept RGBA8/BGRA8/FP16
    // views, an optional capability requested explicitly, and the in-layer network, on request,
    // needs its own. The request owns private copies of the game's structures it changes; behind
    // a structure it cannot copy it declines, which disables composition safely.
    dlssnr::DeviceFeatureRequest features;
    if (LayerEnabled()) {
        VkPhysicalDeviceFeatures supported{};
        auto query = ic ? ic->table.vkGetPhysicalDeviceFeatures : nullptr;
        if (query) query(physicalDevice, &supported);
        if (supported.shaderStorageImageWriteWithoutFormat) {
            if (features.Enable(modified, network)) {
                effective = &modified;
            } else if (network && features.Enable(modified, false)) {
                effective = &modified;
                log_printf("[layer] in-layer network unavailable: cannot safely copy the game's feature chain");
            } else {
                log_printf("[layer] cannot safely clone the Features2 prefix; compositor disabled");
            }
        } else {
            log_printf("[layer] formatless storage writes unsupported; compositor disabled");
        }
    }
    VkResult res = create(physicalDevice, effective, pAllocator, pDevice);
    if (res != VK_SUCCESS && effective != pCreateInfo) {
        // Nothing added here is worth failing a device creation over.
        log_printf("[layer] vkCreateDevice refused the layer's additions (%d); retrying with the game's list",
                   (int) res);
        link->u.pLayerInfo = nextLayerInfo;
        effective = pCreateInfo;
        res = create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    }
    if (res != VK_SUCCESS) return res;

    DeviceChain* dc = new DeviceChain();
    dc->instance = ic;
    dc->physical = physicalDevice;
    dc->self = *pDevice;
    dc->next_dpa = next_dpa;
    dc->setDeviceLoaderData = setLoaderData;
    dc->table.next_dpa = next_dpa;
    dc->table.Load(*pDevice);
    for (uint32_t i = 0; i < effective->enabledExtensionCount; ++i)
        if (!std::strcmp(effective->ppEnabledExtensionNames[i], VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME))
            dc->exportMemory = dc->table.vkGetMemoryFdKHR != nullptr;
    if (network) {
        dc->network = dlssnr::NetworkEnabled(*effective);
        log_printf("[layer] in-layer network features %s", dc->network ? "enabled" : "not enabled");
    }
    if (!dc->table.vkQueuePresentKHR || !dc->table.vkCreateSwapchainKHR || !ic) dc->inert = true;
    if (LayerEnabled() && !dlssnr::HasFormatlessStorageWrites(*effective)) {
        dc->inert = true;
        log_printf("[layer] shaderStorageImageWriteWithoutFormat not enabled; presenting untouched");
    }

    // The native HIP worker runs on an AMD GPU, so on anything else there is nothing for this
    // layer to do but cost a round trip.
    // Hybrid machines are the case that matters: an implicit layer is loaded for every device the
    // loader builds, including the integrated one a game may well be running on.
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = "?";
    if (ic && ic->table.vkGetPhysicalDeviceProperties) {
        VkPhysicalDeviceProperties props{};
        ic->table.vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::snprintf(deviceName, sizeof(deviceName), "%s", props.deviceName);
        bool vendorSupported = props.vendorID == 0x1002u;
#ifdef DLSSLOP_TEST_LAVAPIPE
        // Test builds only: exercise the real Vulkan capture/composition/IPC route
        // with Mesa's CPU driver. Production binaries have no vendor override.
        vendorSupported = vendorSupported || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
#endif
        if (!vendorSupported) {
            dc->inert = true;
            log_printf("[layer] inert on non-AMD device (vendor %#x): %s", props.vendorID, deviceName);
        }
    }

    std::lock_guard<std::mutex> lk(g_stateMutex);
    g_devices[*pDevice] = dc;
    log_printf("[layer] vkCreateDevice -> %p on %s (inert=%d enabled=%d)", (void*)*pDevice, deviceName,
               (int) dc->inert.load(), (int) LayerEnabled());
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDevice(VkDevice device,
                                                     const VkAllocationCallbacks* pAllocator) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc) return;
    {
        std::lock_guard<std::mutex> lk(dc->lock);
        for (auto& kv : dc->swapchains) ReleasePrimary(device, kv.first);
    }
    // Say the layer has gone. A reader that finds a pid here checks it is alive, so a crash is
    // caught too, but an orderly exit should not need anyone to go looking.
    if (dc->shm.hdr && dc->shm.hdr->layerPid.load() == uint32_t(getpid())) {
        dc->shm.hdr->layerPid.store(0);
        dc->shm.hdr->layerCompositionUp.store(0);
        // The in-layer network's state goes with it.
        if (!dc->networkReason.empty())
            ShmStoreString(dc->shm.hdr->layerReasonSeq, dc->shm.hdr->layerReason, kReasonBytes, "");
    }
    if (dc->table.vkDeviceWaitIdle) {
        std::lock_guard<std::mutex> network(dc->networkSubmit);
        dc->table.vkDeviceWaitIdle(device);
    }
    // While the device is still found by its queues: the network's build submits through them.
    if (dc->inLayer) g_network.close(dc->inLayer);
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        g_devices.erase(device);
    }
    {
        std::lock_guard<std::mutex> lk(dc->lock);
        for (auto& kv : dc->swapchains) {
            SwapchainState& sc = kv.second;
            sc.comp.reset();
            if (sc.fenceLeg1) dc->table.vkDestroyFence(device, sc.fenceLeg1, nullptr);
            if (sc.fenceLeg2) dc->table.vkDestroyFence(device, sc.fenceLeg2, nullptr);
            for (VkSemaphore semaphore : sc.leg2Done) dc->table.vkDestroySemaphore(device, semaphore, nullptr);
            if (sc.pool) dc->table.vkDestroyCommandPool(device, sc.pool, nullptr);
        }
        dc->swapchains.clear();
    }
    if (dc->table.vkDestroyDevice) dc->table.vkDestroyDevice(device, pAllocator);
    delete dc;
}

static DeviceChain* FindDevice(VkDevice device) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    auto it = g_devices.find(device);
    return it == g_devices.end() ? nullptr : it->second;
}

static void RememberQueue(DeviceChain* dc, VkQueue queue, uint32_t family) {
    if (!queue) return;
    std::lock_guard<std::mutex> lk(dc->lock);
    dc->queueFamilies[queue] = family;
}

static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue(VkDevice device, uint32_t family,
                                                      uint32_t index, VkQueue* pQueue) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->table.vkGetDeviceQueue) return;
    dc->table.vkGetDeviceQueue(device, family, index, pQueue);
    RememberQueue(dc, *pQueue, family);
}

// The 1.1 way of asking for a queue, and the only way to reach one created with
// VkDeviceQueueCreateFlags. A game that uses it never registered its queue through the hook above,
// so the present path could not tell which family the queue belonged to and fell back to family
// zero -- which is the family the command pool was then created on, and need not be the queue's.
static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue2(VkDevice device,
                                                       const VkDeviceQueueInfo2* pQueueInfo,
                                                       VkQueue* pQueue) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->table.vkGetDeviceQueue2) return;
    dc->table.vkGetDeviceQueue2(device, pQueueInfo, pQueue);
    if (pQueueInfo) RememberQueue(dc, *pQueue, pQueueInfo->queueFamilyIndex);
}

// ---------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------
// Which swapchain formats the pass can work in. Every one of them has a UNORM twin the composition
// uses internally; the ten-bit and float entries are new here, and are what lets an HDR game reach
// the model at all -- the encode is exactly the step that turns open-ended light into the kind of
// picture the model was trained on.
static bool SupportedFormat(VkFormat f) {
    return dlssnr::CompositionFormat(f) != VK_FORMAT_UNDEFINED;
}

// What a swapchain's format and colour space together say about the light in the frame.
//
// A float swapchain is the easy case: games hand over linear light and the HDR path divides it by
// the white point and hands the model the result. The ten-bit formats are the ones worth the colour
// space: on a desktop set to HDR10 they carry ST 2084 code -- absolute nits, which is why the old
// display-referred reading of them (tone map as if it were SDR) banding-crushed them to eight bits
// on the way to the model. A ten-bit swapchain in an SDR colour space is just a bit more precision
// on a tone-mapped frame, and stays on the SDR path.
static uint32_t DetectHdrKind(VkFormat f, VkColorSpaceKHR cs) {
    if (f == VK_FORMAT_R16G16B16A16_SFLOAT) return kHdrLinearFp16;
    const bool tenBit = f == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
                        f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
                        f == VkFormat(1000452000) /* R12G12B12A16_UNORM_PACK32 */;
    const bool pq = cs == VK_COLOR_SPACE_HDR10_ST2084_EXT ||
                    cs == VkColorSpaceKHR(1000459000) /* HDR10_ST2084_COMPATIBLE */;
    if (tenBit && pq) return kHdrPq10;
    // A float swapchain in a linear BT.2020 space is still linear light; nothing else here is HDR.
    return kHdrNone;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc || !dc->table.vkCreateSwapchainKHR) return VK_ERROR_INITIALIZATION_FAILED;

    // Native HIP supports display-referred SDR, linear scRGB/BT.709 FP16,
    // and PQ/BT.2020 HDR10. Other transfer/primary combinations need their
    // own color conversion and remain pass-through.
    const bool unsupportedHdr = !(
        pCreateInfo->imageColorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
        (pCreateInfo->imageFormat == VK_FORMAT_R16G16B16A16_SFLOAT &&
         pCreateInfo->imageColorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) ||
        DetectHdrKind(pCreateInfo->imageFormat, pCreateInfo->imageColorSpace) == kHdrPq10);
    bool unsupportedTransfer = false;
    if (!dc->inert && LayerEnabled()) {
        const VkImageUsageFlags required = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkSurfaceCapabilitiesKHR capabilities{};
        auto query = dc->instance ? dc->instance->table.vkGetPhysicalDeviceSurfaceCapabilitiesKHR : nullptr;
        unsupportedTransfer = !query ||
            query(dc->physical, pCreateInfo->surface, &capabilities) != VK_SUCCESS ||
            (capabilities.supportedUsageFlags & required) != required;
    }
    VkSwapchainCreateInfoKHR m = *pCreateInfo;
    if (!dc->inert && LayerEnabled() && !unsupportedHdr && !unsupportedTransfer &&
        SupportedFormat(pCreateInfo->imageFormat))
        m.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkResult res = dc->table.vkCreateSwapchainKHR(device, &m, pAllocator, pSwapchain);
    if (res != VK_SUCCESS || dc->inert || !LayerEnabled()) return res;

    uint32_t count = 0;
    dc->table.vkGetSwapchainImagesKHR(device, *pSwapchain, &count, nullptr);
    std::vector<VkImage> images(count);
    dc->table.vkGetSwapchainImagesKHR(device, *pSwapchain, &count, images.data());

    SwapchainState sc{};
    sc.images = std::move(images);
    sc.format = pCreateInfo->imageFormat;
    sc.hdrKind = DetectHdrKind(sc.format, pCreateInfo->imageColorSpace);
    sc.width = pCreateInfo->imageExtent.width;
    sc.height = pCreateInfo->imageExtent.height;
    const bool tooSmall = sc.width < kMinW || sc.height < kMinH;
    sc.passThrough = unsupportedHdr || unsupportedTransfer || !SupportedFormat(sc.format) || sc.width > kMaxW ||
                     sc.height > kMaxH || tooSmall;

    std::lock_guard<std::mutex> lk(dc->lock);
    log_printf("[layer] swapchain %p %ux%u fmt=%d hdr=%u passThrough=%d%s", (void*)*pSwapchain,
               pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
               (int)pCreateInfo->imageFormat, sc.hdrKind, (int)sc.passThrough,
               sc.passThrough ? (unsupportedHdr ? " (unsupported color space for native HIP)"
                                 : unsupportedTransfer ? " (surface cannot transfer frames)"
                                 : !SupportedFormat(sc.format) ? " (unsupported format)"
                                 : tooSmall ? " (too small)" : " (too large)") : "");
    dc->swapchains[*pSwapchain] = std::move(sc);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroySwapchainKHR(VkDevice device,
    VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc) return;
    ReleasePrimary(device, swapchain);
    std::unique_lock<std::mutex> lk(dc->lock);
    auto it = dc->swapchains.find(swapchain);
    if (it != dc->swapchains.end()) {
        lk.unlock();
        if (dc->table.vkDeviceWaitIdle) {
            std::lock_guard<std::mutex> network(dc->networkSubmit);
            dc->table.vkDeviceWaitIdle(device);
        }
        lk.lock();
        SwapchainState& sc = it->second;
        sc.comp.reset();
        if (sc.fenceLeg1) dc->table.vkDestroyFence(device, sc.fenceLeg1, nullptr);
        if (sc.fenceLeg2) dc->table.vkDestroyFence(device, sc.fenceLeg2, nullptr);
        for (VkSemaphore semaphore : sc.leg2Done) dc->table.vkDestroySemaphore(device, semaphore, nullptr);
        if (sc.pool) dc->table.vkDestroyCommandPool(device, sc.pool, nullptr);
        if (dc->inLayerLast == &sc) dc->inLayerLast = nullptr;
        dc->swapchains.erase(it);
    }
    lk.unlock();
    if (dc->table.vkDestroySwapchainKHR) dc->table.vkDestroySwapchainKHR(device, swapchain, pAllocator);
}

// ---------------------------------------------------------------------------
// Present-time neural round-trip
// ---------------------------------------------------------------------------
// Give a dispatchable object this layer allocated the dispatch table the loader expects on it.
//
// VkCommandBuffer and VkQueue are dispatchable: their first word points at a dispatch table, and
// every layer below reads it to find its own state for that object. The loader fills that word in
// for objects the application allocates through the trampoline -- but a layer that allocates one by
// calling straight down the chain bypasses the trampoline, so the loader never sees it and the word
// keeps whatever the ICD left there. The loader hands each layer vkSetDeviceLoaderData precisely so
// it can do that fill-in itself, and calling it is mandatory, not advisory.
//
// Skipping it is invisible with no other layer present: the next call goes straight to the driver,
// which does not read the word. Add any second layer -- Steam's overlay, MangoHud, validation -- and
// that layer reads the word, finds the ICD's loader magic instead of a table, and aborts. Validation
// says so out loud: 'The VkDevice dispatch handle was not found and Validation will crash.'
static bool SetLoaderData(DeviceChain* dc, void* object) {
    if (!dc->setDeviceLoaderData) return true;  // no loader in the chain; nothing to fill in
    return dc->setDeviceLoaderData(dc->self, object) == VK_SUCCESS;
}

static bool CreateResources(DeviceChain* dc, SwapchainState& sc, uint32_t family) {
    VkDevice d = dc->self;

    // Present-only queues cannot execute the composition's compute dispatches.
    // Proton normally presents on its graphics queue; unusual applications can
    // still present untouched instead of issuing invalid commands.
    VkQueueFamilyProperties families[32];
    uint32_t count = 32;
    dc->instance->table.vkGetPhysicalDeviceQueueFamilyProperties(dc->physical, &count, families);
    if (family >= count || !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        log_printf("[layer] present queue family %u cannot execute the compositor", family);
        return false;
    }

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = family;
    if (dc->table.vkCreateCommandPool(d, &cpci, nullptr, &sc.pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = sc.pool; cbai.commandBufferCount = 1;
    if (dc->table.vkAllocateCommandBuffers(d, &cbai, &sc.cb) != VK_SUCCESS) return false;
    if (!SetLoaderData(dc, sc.cb)) {
        log_printf("[layer] vkSetDeviceLoaderData failed for the present command buffer");
        return false;
    }
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (dc->table.vkCreateFence(d, &fci, nullptr, &sc.fenceLeg1) != VK_SUCCESS) return false;
    if (dc->table.vkCreateFence(d, &fci, nullptr, &sc.fenceLeg2) != VK_SUCCESS) return false;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sc.leg2Done.resize(sc.images.size());
    for (VkSemaphore& semaphore : sc.leg2Done)
        if (dc->table.vkCreateSemaphore(d, &sci, nullptr, &semaphore) != VK_SUCCESS) return false;

    if (!dc->instance) return false;
    sc.comp = std::make_unique<dlssnr::Composition>(&dc->table, &dc->instance->table, d, dc->physical);
    if (!sc.comp->Usable()) {
        log_printf("[layer] composition unavailable: %s", sc.comp->Reason());
        sc.comp.reset();
        return false;
    }
    if (dc->exportMemory) sc.comp->EnableExport(family);
    sc.family = family;
    sc.graphics = families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT;
    return true;
}

// Every Vulkan result on the present path, looked at rather than collapsed into a bool.
//
// A failure here was previously indistinguishable from 'nothing to do': the call returned false, the
// caller presented the original frame, and the next frame tried exactly the same thing again. That is
// the right answer for a transient failure and the wrong one for VK_ERROR_DEVICE_LOST, where the
// device is gone, every subsequent call will fail the same way, and the log fills with it.
//
// Losing the device also latches the layer inert, because after that point the fail-open path is the
// only correct one and it costs nothing to take it directly.
static bool NoteVk(DeviceChain* dc, VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    if (r == VK_ERROR_DEVICE_LOST) {
        if (!dc->inert.exchange(true)) log_printf("[layer] %s -> DEVICE_LOST; layer inert for this device", what);
        return false;
    }
    static std::atomic<uint32_t> reported{0};
    if (reported.fetch_add(1) < 8) log_printf("[layer] %s -> %d", what, (int) r);
    return false;
}

// A leg of the present path: its command buffer begun, then ended and submitted, and its fence
// waited on and reset.
static bool BeginLeg(DeviceChain* dc, VkCommandBuffer cb) {
    static constexpr VkCommandBufferBeginInfo kOnce{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                                     VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    return NoteVk(dc, dc->table.vkBeginCommandBuffer(cb, &kOnce), "vkBeginCommandBuffer");
}

// A fence wait does not make device writes visible to the host, and the host reads what the legs
// copy out: the proxy, the meter mirror and the capture pair.
static bool SubmitLeg(DeviceChain* dc, VkQueue queue, const VkSubmitInfo& si, VkFence fence, bool& waitsConsumed) {
    static constexpr VkMemoryBarrier kToHost{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
    VkCommandBuffer cb = si.pCommandBuffers[0];
    dc->table.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &kToHost,
                                   0, nullptr, 0, nullptr);
    if (!NoteVk(dc, dc->table.vkEndCommandBuffer(cb), "vkEndCommandBuffer")) return false;
    if (!NoteVk(dc, dc->table.vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit")) return false;
    if (si.waitSemaphoreCount != 0) waitsConsumed = true;
    return true;
}

static bool WaitLeg(DeviceChain* dc, VkFence fence) {
    if (!NoteVk(dc, dc->table.vkWaitForFences(dc->self, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences"))
        return false;
    dc->table.vkResetFences(dc->self, 1, &fence);
    return true;
}

// Leg 2 done: the command buffer and the composed surfaces are free again, and the capture pair it
// copied out is written.
static bool CollectLeg2(DeviceChain* dc, SwapchainState& sc) {
    if (!WaitLeg(dc, sc.fenceLeg2)) return false;
    sc.leg2Pending = false;
    sc.comp->WriteCapturedFrame();
    return true;
}

// A composed frame, for the controllers: MS is the time the layer took over it.
static void PublishFrame(DeviceChain* dc, const SwapchainState& sc, double ms) {
    ShmHeader* const hdr = dc->shm.hdr;
    if (!hdr) return;
    ShmStore64(hdr->layerFramesLo, hdr->layerFramesHi, ++dc->framesComposed);
    hdr->layerWidth.store(sc.width);
    hdr->layerHeight.store(sc.height);
    hdr->layerFormat.store(uint32_t(sc.format));
    hdr->layerCompositionUp.store(1);
    hdr->layerPid.store(uint32_t(getpid()));
    hdr->layerMsBits.store(FloatToBits(float(ms)));
    hdr->layerMeasuredWhiteBits.store(FloatToBits(sc.comp->MeasuredWhitePoint()));
    hdr->layerHeartbeat.fetch_add(1);
}

static void NetworkLog(const char* line) { log_printf("[network] %s", line); }

// The in-layer network's state in the channel's layer reason, for the controllers: WHAT, DETAIL and
// AFTER, stored when it changes and logged when it is news. Unchanged, it allocates nothing.
static void NetworkReason(DeviceChain* dc, const char* what, const char* detail = "", const char* after = "") {
    std::string& reason = dc->networkReason;
    const size_t a = std::strlen(what), b = std::strlen(detail);
    if (reason.size() == a + b + std::strlen(after) && !reason.compare(0, a, what) &&
        !reason.compare(a, b, detail) && !reason.compare(a + b, std::string::npos, after))
        return;
    reason.assign(what).append(detail).append(after);
    log_printf("[layer] %s", reason.c_str());
    if (dc->shm.hdr) ShmStoreString(dc->shm.hdr->layerReasonSeq, dc->shm.hdr->layerReason, kReasonBytes, reason.c_str());
}

// The network failed: frames go to dlsslopd from now on.
static bool NetworkFailed(DeviceChain* dc) {
    NetworkReason(dc, "in-layer network off: ", g_network.error(dc->inLayer), "; frames go to dlsslopd");
    dc->inLayerOff = true;
    return false;
}

// The network on DC's device, building through QUEUE, SC's present queue; null, with the reason
// published, when it does not open.
static DlsslopNetwork* OpenNetwork(DeviceChain* dc, const SwapchainState& sc, VkQueue queue) {
    static const bool loaded = [] {
        const std::string self = LayerObjectPath();
        return g_network.Load((self.substr(0, self.rfind('/') + 1) + "libdlsslop-network.so").c_str());
    }();
    if (!loaded) {
        NetworkReason(dc, "in-layer network off, module unavailable: ", g_network.failure.c_str(),
                      "; frames go to dlsslopd");
        return nullptr;
    }
    DlsslopNetworkDevice device{};
    device.instance = dc->instance->self;
    device.physical = dc->physical;
    device.device = dc->self;
    device.queue = queue;
    device.family = sc.family;
    device.lockQueue = [](void* context) { static_cast<DeviceChain*>(context)->networkSubmit.lock(); };
    device.unlockQueue = [](void* context) { static_cast<DeviceChain*>(context)->networkSubmit.unlock(); };
    device.context = dc;
    device.physicalDispatch = dc->instance->next_gipa;
    dc->instance->table.vkGetPhysicalDeviceMemoryProperties(dc->physical, &device.memory);
    device.log = NetworkLog;
    DlsslopNetwork* network = g_network.open(&device);
    if (network)
        log_printf("[layer] in-layer network on the game's device: frames skip dlsslopd");
    else
        NetworkReason(dc, "in-layer network off: out of memory; frames go to dlsslopd");
    return network;
}

// The swapchain presents where the network cannot run: frames go to dlsslopd, and the network
// waits for one that presents where it can.
static bool NetworkIdle(DeviceChain* dc) {
    NetworkReason(dc, "in-layer network idle: the swapchain's queue family cannot run it; frames go to dlsslopd");
    return false;
}

// Whether this frame goes through the in-layer network instead of dlsslopd: the device enabled it,
// it has not failed, and SC presents on a graphics family, the one it opened on once it has. It
// opens once, for the first such swapchain.
static bool UseInLayerNetwork(DeviceChain* dc, const SwapchainState& sc, VkQueue queue) {
    if (!dc->network || dc->inLayerOff) return false;
    if (dc->inLayer) return sc.family == dc->inLayerFamily || NetworkIdle(dc);
    if (!sc.graphics) return NetworkIdle(dc);
    dc->inLayer = OpenNetwork(dc, sc, queue);
    dc->inLayerOff = !dc->inLayer;
    dc->inLayerFamily = sc.family;
    return dc->inLayer;
}

// The frame through the in-layer network: capture, network and composition in one submission, SI
// with the game's waits, and no CPU wait between them. False presents the game's own frame: while
// the network builds, for a rejected setting, or once it fails, after which frames go to dlsslopd.
static bool ProcessInLayer(DeviceChain* dc, SwapchainState& sc, VkQueue queue, uint32_t index,
                           const dlssnr::FrameSettings& fs, VkSubmitInfo& si, bool& waitsConsumed, double t0) {
    // The previous frame's meter, whose fence the frame waited for.
    sc.comp->ConsumeMeter();
    // A swapchain that takes over, a resized one, has not waited for its predecessor's last frame,
    // which may still run the network that a build or a reshape at the new shape frees.
    SwapchainState* const last = dc->inLayerLast;
    if (last && last != &sc && last->leg2Pending && !CollectLeg2(dc, *last)) return false;
    dc->inLayerLast = &sc;
    switch (g_network.prepare(dc->inLayer, dc->shm.hdr, sc.comp->ModelWidth(), sc.comp->ModelHeight(),
                              sc.comp->HdrProxyActive())) {
    case kDlsslopNetworkReady:
        NetworkReason(dc, "in-layer network running");
        break;
    case kDlsslopNetworkBuilding:
        NetworkReason(dc, "in-layer network building; the game presents its own frames");
        return false;
    case kDlsslopNetworkRejected:
        NetworkReason(dc, "in-layer network: ", g_network.error(dc->inLayer));
        return false;
    default:
        return NetworkFailed(dc);
    }
    VkCommandBuffer cb = sc.cb;
    if (!BeginLeg(dc, cb)) return false;
    if (!sc.comp->RecordCapture(cb, sc.images[index], fs)) {
        dc->table.vkEndCommandBuffer(cb);
        return false;
    }
    // Past the capture, a failure submits what was recorded, as leg 1 alone would be: the
    // composition's state moved with it, and the image is back in PRESENT_SRC_KHR. With NETWORK,
    // the network's frame is among what was recorded, and the network is told it was submitted.
    const auto salvage = [&](bool network) {
        if (SubmitLeg(dc, queue, si, sc.fenceLeg1, waitsConsumed)) {
            if (network) g_network.submitted(dc->inLayer);
            WaitLeg(dc, sc.fenceLeg1);
        }
        return false;
    };
    if (g_network.record(dc->inLayer, cb, sc.comp->ProxyBuffer(), sc.comp->AnswerBuffer(), sc.family,
                         sc.comp->TransportExported()) != kDlsslopNetworkReady) {
        NetworkFailed(dc);
        return salvage(false);
    }
    if (!sc.comp->RecordCompose(cb, sc.images[index], fs)) return salvage(true);
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sc.leg2Done[index];
    if (!SubmitLeg(dc, queue, si, sc.fenceLeg2, waitsConsumed)) return false;
    // The network's motion history takes in only the frames that reached the queue.
    g_network.submitted(dc->inLayer);
    sc.leg2Pending = true;
    if (sc.comp->CaptureRecorded()) CollectLeg2(dc, sc);
    PublishFrame(dc, sc, log_now_ms() - t0);
    return true;
}

// Returns true if the swapchain image will hold the composed frame: leg 2 was submitted and signals
// sc.leg2Done[index], which the present must wait on.
//
// Three steps around one round trip. The pass encodes a proxy of the frame on the GPU, that proxy
// crosses to the helper and comes back as the model's answer, and the pass composes the answer onto
// the frame. The proxy's crossing is fenced on the CPU because the model is in another process and
// there is nothing to wait on but a sequence number; the answer's return is not -- the present waits
// on leg 2's semaphore, and the fence that covers leg 2 is only waited on at the start of the NEXT
// frame, where the command buffer and the composed surfaces are reused.
//
// The caller's present semaphores are consumed by the first submit, because that submit is the first
// thing to touch the image. They are therefore unsignalled by the time this returns and must not be
// handed to vkQueuePresentKHR again; the caller presents with the composed image's semaphore or none.
// The submit that takes them sets waitsConsumed, which nothing here clears.
//
// Every path out leaves the swapchain image in PRESENT_SRC_KHR, including the ones that give up.
static bool ProcessPresent(DeviceChain* dc, SwapchainState& sc, VkQueue queue,
                            uint32_t index, uint32_t waitCount,
                            const VkSemaphore* waitSemaphores, bool& waitsConsumed) {
    if (!sc.comp) return false;
    NativeFrameGuard producer(dc->shm);
    if (!producer) return false;
    // A timed-out request still owns the sole input/output slot. Do this BEFORE
    // recording the GPU download, not just before publishing seq_req: the helper may
    // still be uploading those pixels or returning its result. While it finishes,
    // present the game's original image without blocking or overwriting the slot.
    // Do the same while no worker serves (starting, failed or stopped): a request
    // then would only stall this present and outlive the worker.
    ShmHeader* const hdr = dc->shm.hdr;
    const bool inLayer = UseInLayerNetwork(dc, sc, queue);
    if (inLayer) {
        // The raster the network runs at, which dlsslopd would publish.
        if (const NativeTier* tier = ShmNativeTier(hdr->nativeTier.load())) {
            hdr->nativeModelMaxWidth.store(tier->width);
            hdr->nativeModelMaxHeight.store(tier->height);
        }
    } else if (hdr->helperState.load(std::memory_order_acquire) != kHelperRunning) {
        // A daemon that serves again gets a fresh offer; its predecessor's imports are gone.
        sc.comp->WithdrawOffer();
        StartWorker(dc->shm);
        return false;
    } else if (hdr->seq_req.load(std::memory_order_acquire) != hdr->seq_resp.load(std::memory_order_acquire)) {
        return false;
    }
    VkCommandBuffer cb = sc.cb;
    const bool time = log_time_enabled();
    const double t0 = log_now_ms();

    // The previous frame's compose, if it is still running, must finish before anything here
    // touches the surfaces it reads or the command buffer it was recorded into. Waiting here rather
    // than at the end of that frame keeps the game thread out of the GPU's way for the whole of the
    // helper's round trip. The capture pair that compose recorded lands with it.
    if (sc.leg2Pending && !CollectLeg2(dc, sc)) return false;

    dlssnr::FrameSettings fs = dlssnr::FrameSettings::Read(dc->shm.hdr);

    // The HDR decision, made once per frame before anything is sized or encoded.
    //
    // hdrActive is what this process intends; proxyFormat is what the helper actually built, and the
    // float encode is only taken when both agree -- a model that refused the float input leaves the
    // frame on the 8-bit path it has always used. hdrActive doubles as the echo the helper reads, so
    // it builds the float images only for a layer that has said it will feed them.
    const uint32_t hdrMode = dc->shm.hdr ? dc->shm.hdr->hdrMode.load() : kHdrAuto;
    const uint32_t colorMode = dc->shm.hdr ? dc->shm.hdr->colourMode.load() : kColourAuto;
    const bool hdrActive = hdrMode != kHdrOff && (hdrMode == kHdrForce || sc.hdrKind != kHdrNone);
    // HIP consumes an encoded picture in either precision; unlike NGX, format support does not
    // require rebuilding its model, so the float proxy follows the intent alone. The per-request
    // hdrEncode field publishes the selected transport before seq_req.
    const bool hdrProxy = hdrActive;
    const bool linearHdr = colorMode == kColourLinearHdr ||
                           (colorMode == kColourAuto && sc.hdrKind != kHdrNone);
    const uint32_t hdrTransfer = linearHdr && sc.hdrKind == kHdrPq10 ? 1u : 0u;

    if (!sc.comp->Prepare(sc.width, sc.height, sc.format, fs, linearHdr, hdrProxy, hdrTransfer)) {
        log_printf("[layer] composition cannot run here: %s", sc.comp->Reason());
        return false;
    }

    if (dc->shm.hdr) {
        dc->shm.hdr->hdrDetected.store(sc.hdrKind);
        // The intent, not the format-gated decision: the helper builds the float images only for a
        // layer that has said it will feed them, and that handshake has to start while the proxy is
        // still 8-bit. Publishing HdrProxyActive() here would wait on proxyFormat, which waits on
        // this field, and neither would ever move.
        dc->shm.hdr->hdrActive.store(hdrActive ? 1u : 0u);
    }

    // A capture is asked for by writing a frame count into the header; taking it clears the request,
    // so one press produces one run rather than one per frame for as long as nobody clears it.
    // A request published after this frame's settings snapshot belongs to the next frame.
    if (dc->shm.hdr && dc->shm.hdr->controlSeq.load() == fs.controlSeq) {
        if (const uint32_t frames = dc->shm.hdr->captureRequest.exchange(0); frames > 0)
            sc.comp->RequestCapture(std::min<uint32_t>(frames, 64), fs.controlSeq);
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    // Only the first submit waits: the later one is ordered behind it on the same queue and fenced
    // besides, and a binary semaphore may be waited on once per signal.
    if (sc.waitStages.size() < waitCount)
        sc.waitStages.resize(waitCount, VK_PIPELINE_STAGE_TRANSFER_BIT);
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waitCount ? waitSemaphores : nullptr;
    si.pWaitDstStageMask = waitCount ? sc.waitStages.data() : nullptr;
    const auto dropWaits = [&] {
        si.waitSemaphoreCount = 0;
        si.pWaitSemaphores = nullptr;
        si.pWaitDstStageMask = nullptr;
    };

    if (inLayer) return ProcessInLayer(dc, sc, queue, index, fs, si, waitsConsumed, t0);
    // The daemon imports an exported pair once, before any frame names it. No frame runs on a
    // pending offer: until the answer comes, present the game's own. A refusal stages this
    // swapchain through host memory from then on; a daemon that stops serving first leaves the
    // pair for the next one.
    if (sc.comp->TransportPending()) {
        switch (OfferTransport(dc->shm, *sc.comp, sc.offerExpires)) {
        case Offer::kReady:
            sc.comp->SetTransportReady(true);
            log_printf("[shm] device-local transport ready: frames stay in video memory");
            break;
        case Offer::kDeclined:
            log_printf("[shm] device-local transport declined; staging frames through host memory");
            sc.comp->DisableExport();
            break;
        case Offer::kWaiting:
            return false;
        case Offer::kLater:
            sc.comp->WithdrawOffer();
            log_printf("[shm] daemon stopped before it took the transport offer; offering it again later");
            return false;
        }
    }
    const uint32_t transportGen = sc.comp->TransportGeneration();

    // ---- leg 1: the frame the model is shown ----
    if (!BeginLeg(dc, cb)) return false;
    if (!sc.comp->RecordCapture(cb, sc.images[index], fs)) {
        dc->table.vkEndCommandBuffer(cb);
        return false;
    }
    // This one fence is real: the proxy the helper is about to read is written by these commands,
    // and the sequence number must not outrun the pixels it announces.
    if (!SubmitLeg(dc, queue, si, sc.fenceLeg1, waitsConsumed)) return false;
    if (!WaitLeg(dc, sc.fenceLeg1)) return false;
    dropWaits();
    sc.comp->ConsumeMeter();
    const double tCapture = time ? log_now_ms() : 0.0;

    // ---- the round trip ----
    if (!ShmProcessFrame(dc->shm, sc.comp->ModelWidth(), sc.comp->ModelHeight(), sc.comp->ModelBytes(),
                         sc.comp->ProxyPixels(), sc.comp->ModelPixels(), sc.comp->HdrProxyActive(),
                         transportGen)) {
        // A restarted worker holds no imports: offer the pair again on the next frame.
        if (transportGen && dc->shm.hdr->transportMiss.load() == transportGen) sc.comp->SetTransportReady(false);
        // Fail-open. Leg 1 already put the image back in PRESENT_SRC_KHR, so the original frame is
        // what gets presented and nothing else is owed.
        return false;
    }
    sc.comp->SetCaptureInference(dc->shm.hdr->seq_req.load());
    const double tHelper = time ? log_now_ms() : 0.0;

    // ---- leg 2: the answer, composed back ----
    if (!BeginLeg(dc, cb)) return false;
    if (!sc.comp->RecordCompose(cb, sc.images[index], fs)) {
        dc->table.vkEndCommandBuffer(cb);
        return false;
    }
    // No CPU wait. The present waits on this semaphore, so the image is composed before it is shown
    // without the CPU ever parking here; the fence is collected at the top of the next frame, where
    // the reused surfaces actually need it.
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sc.leg2Done[index];
    if (!SubmitLeg(dc, queue, si, sc.fenceLeg2, waitsConsumed)) return false;
    sc.leg2Pending = true;
    // A diagnostic request must finish even if a paused application presents
    // no subsequent frame. Frames without a recorded pair keep the asynchronous
    // leg-2 path, including every frame of a capture whose buffer failed.
    // The semaphore is signalled either way, so a failed wait still presents.
    if (sc.comp->CaptureRecorded()) CollectLeg2(dc, sc);
    const double tReturn = log_now_ms();
    PublishFrame(dc, sc, tReturn - t0);

    if (time) {
        static int frameNo = 0;
        if (++frameNo % log_time_interval() == 0) {
            log_printf("[time] encode=%.2f helper=%.2f resolve=%.2f total=%.2f ms (model %ux%u)",
                       tCapture - t0, tHelper - tCapture, tReturn - tHelper, tReturn - t0,
                       sc.comp->ModelWidth(), sc.comp->ModelHeight());
        }
    }
    return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueuePresentKHR(VkQueue queue,
                                                            const VkPresentInfoKHR* pPresentInfo) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        if (g_devices.size() == 1) dc = g_devices.begin()->second;
        else {
            for (auto& kv : g_devices) {
                std::lock_guard<std::mutex> dl(kv.second->lock);
                if (kv.second->queueFamilies.count(queue)) { dc = kv.second; break; }
            }
        }
    }
    if (!dc || !dc->table.vkQueuePresentKHR) return VK_ERROR_INITIALIZATION_FAILED;

    // Whether this call's wait semaphores have already been consumed by a submit of ours. They are
    // handed to the first swapchain we actually process; every path after that presents without them,
    // because a semaphore signalled once may only be waited on once. Presenting with them a second
    // time is a wait that never completes -- which is what a second layer in the chain, Steam's
    // overlay among them, turns from a latent bug into a hang.
    bool waitsConsumed = false;
    // Signalled by the composition's last submit; the present waits on it.
    VkSemaphore composedSem = VK_NULL_HANDLE;

    // Held from here to the present at the end, alongside the queue hooks. Vulkan requires external
    // synchronization for every operation on a queue; the in-layer network's build submits to the
    // application's queue from a thread of its own, so the present must not overlap it.
    std::unique_lock<std::mutex> lk;
    if (!dc->inert && LayerEnabled()) {
        lk = std::unique_lock<std::mutex>(dc->lock);
        PollHotkeys(dc);
        if (!ShmNeuralEnabled(dc->shm)) return dc->table.vkQueuePresentKHR(queue, pPresentInfo);
        uint32_t family = 0;
        auto qit = dc->queueFamilies.find(queue);
        if (qit != dc->queueFamilies.end()) family = qit->second;
        for (uint32_t i = 0; i < pPresentInfo->swapchainCount; ++i) {
            auto sit = dc->swapchains.find(pPresentInfo->pSwapchains[i]);
            if (sit == dc->swapchains.end()) continue;
            SwapchainState& sc = sit->second;
            if (sc.passThrough || pPresentInfo->pImageIndices[i] >= sc.images.size()) continue;
            // One swapchain drives the channel; the rest present raw. See ClaimPrimary.
            if (!ClaimPrimary(dc->self, pPresentInfo->pSwapchains[i], sc.width, sc.height)) {
                static std::atomic<uint32_t> n{0};
                const uint32_t k = n.fetch_add(1);
                if (k < 3) log_printf("[layer] present: swapchain %p is not primary", (void*)pPresentInfo->pSwapchains[i]);
                continue;
            }
            // One composition, and one semaphore, per present. A larger swapchain that has just taken
            // the primary role over composes from its next present.
            if (composedSem) continue;
            if (!sc.ready && !dc->shm.dead) {
                if (!CreateResources(dc, sc, family)) {
                    log_printf("[layer] staging resources failed for swapchain %p (%ux%u, family %u); "
                               "passing this swapchain through",
                               (void*)pPresentInfo->pSwapchains[i], sc.width, sc.height, family);
                    sc.passThrough = true;
                    // This swapchain claimed the primary role and just gave it up. Without the
                    // release the claim would sit on a swapchain that never drives the channel,
                    // and no peer of equal or smaller area could take it over.
                    ReleasePrimary(dc->self, pPresentInfo->pSwapchains[i]);
                    continue;
                }
                sc.ready = true;
            }
            if (!sc.ready || dc->shm.dead) {
                ReleasePrimary(dc->self, pPresentInfo->pSwapchains[i]);
                continue;
            }
            const uint32_t waitCount = waitsConsumed ? 0u : pPresentInfo->waitSemaphoreCount;
            const bool composed = ProcessPresent(dc, sc, queue, pPresentInfo->pImageIndices[i],
                                                 waitCount, pPresentInfo->pWaitSemaphores, waitsConsumed);
            if (composed) composedSem = sc.leg2Done[pPresentInfo->pImageIndices[i]];
            else ++dc->framesPassedThrough;
            if (log_verbose()) {
                log_printf("[present] swapchain=%p image=%u seq=%u composed=%d",
                           (void*)pPresentInfo->pSwapchains[i], pPresentInfo->pImageIndices[i],
                           dc->shm.hdr ? dc->shm.hdr->seq_req.load() : 0u, int(composed));
            }
            // On failure before the capture submit, leave the application's waits attached to the
            // original present. Once our first submit accepted them they have been consumed.
        }
        if (log_time_enabled()) {
            static int frameNo = 0;
            if (++frameNo % log_time_interval() == 0) {
                log_printf("[layer] composed=%llu passed through=%llu",
                           (unsigned long long)dc->framesComposed,
                           (unsigned long long)dc->framesPassedThrough);
            }
        }
    }

    if (!waitsConsumed && !composedSem) return dc->table.vkQueuePresentKHR(queue, pPresentInfo);

    // pNext is carried through untouched: present ids, present timing and the rest belong to the
    // caller and none of them are about semaphores.
    VkPresentInfoKHR pi = *pPresentInfo;
    pi.waitSemaphoreCount = composedSem ? 1u : 0u;
    pi.pWaitSemaphores = &composedSem;
    return dc->table.vkQueuePresentKHR(queue, &pi);
}

static DeviceChain* DeviceForQueue(VkQueue queue) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    if (g_devices.size() == 1) return g_devices.begin()->second;
    for (auto& kv : g_devices) {
        std::lock_guard<std::mutex> dl(kv.second->lock);
        if (kv.second->queueFamilies.count(queue)) return kv.second;
    }
    return nullptr;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit(VkQueue queue, uint32_t submitCount,
                                                        const VkSubmitInfo* pSubmits, VkFence fence) {
    DeviceChain* dc = DeviceForQueue(queue);
    if (!dc || !dc->table.vkQueueSubmit) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lk(dc->lock);
    return dc->table.vkQueueSubmit(queue, submitCount, pSubmits, fence);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit2(VkQueue queue, uint32_t submitCount,
                                                         const VkSubmitInfo2* pSubmits, VkFence fence) {
    DeviceChain* dc = DeviceForQueue(queue);
    if (!dc || !dc->table.vkQueueSubmit2) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lk(dc->lock);
    return dc->table.vkQueueSubmit2(queue, submitCount, pSubmits, fence);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueWaitIdle(VkQueue queue) {
    DeviceChain* dc = DeviceForQueue(queue);
    if (!dc || !dc->table.vkQueueWaitIdle) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lk(dc->lock);
    return dc->table.vkQueueWaitIdle(queue);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit2KHR(VkQueue queue, uint32_t submitCount,
                                                            const VkSubmitInfo2* pSubmits, VkFence fence) {
    DeviceChain* dc = DeviceForQueue(queue);
    if (!dc || !dc->table.vkQueueSubmit2KHR) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lk(dc->lock);
    return dc->table.vkQueueSubmit2KHR(queue, submitCount, pSubmits, fence);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueBindSparse(VkQueue queue, uint32_t bindInfoCount,
                                                            const VkBindSparseInfo* pBindInfo, VkFence fence) {
    DeviceChain* dc = DeviceForQueue(queue);
    if (!dc || !dc->table.vkQueueBindSparse) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lk(dc->lock);
    return dc->table.vkQueueBindSparse(queue, bindInfoCount, pBindInfo, fence);
}

// A device wait may not overlap a submit to any of its queues, and the network's build submits
// under the lock.
static VKAPI_ATTR VkResult VKAPI_CALL Hook_DeviceWaitIdle(VkDevice device) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->table.vkDeviceWaitIdle) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lk(dc->lock);
    return dc->table.vkDeviceWaitIdle(device);
}

// ---------------------------------------------------------------------------
// Loader entry points
// ---------------------------------------------------------------------------
static PFN_vkVoidFunction LookupHook(const char* n) {
    if (!std::strcmp(n, "vkCreateInstance")) return (PFN_vkVoidFunction)Hook_CreateInstance;
    if (!std::strcmp(n, "vkDestroyInstance")) return (PFN_vkVoidFunction)Hook_DestroyInstance;
    if (!std::strcmp(n, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)Hook_EnumeratePhysicalDevices;
    if (!std::strcmp(n, "vkCreateDevice")) return (PFN_vkVoidFunction)Hook_CreateDevice;
    if (!std::strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)Hook_DestroyDevice;
    if (!std::strcmp(n, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue;
    if (!std::strcmp(n, "vkGetDeviceQueue2")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue2;
    if (!std::strcmp(n, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)Hook_CreateSwapchainKHR;
    if (!std::strcmp(n, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)Hook_DestroySwapchainKHR;
    if (!std::strcmp(n, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)Hook_QueuePresentKHR;
    // The in-layer network's build submits to the game's queue from a thread of its own: the queue
    // hooks serialize it with the game's queue operations and device waits.
    if (!dlssnr::NetworkRequested()) return nullptr;
    if (!std::strcmp(n, "vkQueueSubmit")) return (PFN_vkVoidFunction)Hook_QueueSubmit;
    if (!std::strcmp(n, "vkQueueSubmit2")) return (PFN_vkVoidFunction)Hook_QueueSubmit2;
    if (!std::strcmp(n, "vkQueueWaitIdle")) return (PFN_vkVoidFunction)Hook_QueueWaitIdle;
    if (!std::strcmp(n, "vkQueueSubmit2KHR")) return (PFN_vkVoidFunction)Hook_QueueSubmit2KHR;
    if (!std::strcmp(n, "vkQueueBindSparse")) return (PFN_vkVoidFunction)Hook_QueueBindSparse;
    if (!std::strcmp(n, "vkDeviceWaitIdle")) return (PFN_vkVoidFunction)Hook_DeviceWaitIdle;
    return nullptr;
}

static PFN_vkVoidFunction LookupDeviceHook(const char* n) {
    if (!std::strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)Hook_DestroyDevice;
    if (!std::strcmp(n, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue;
    if (!std::strcmp(n, "vkGetDeviceQueue2")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue2;
    if (!std::strcmp(n, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)Hook_CreateSwapchainKHR;
    if (!std::strcmp(n, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)Hook_DestroySwapchainKHR;
    if (!std::strcmp(n, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)Hook_QueuePresentKHR;
    // The in-layer network's build submits to the game's queue from a thread of its own: the queue
    // hooks serialize it with the game's queue operations and device waits.
    if (!dlssnr::NetworkRequested()) return nullptr;
    if (!std::strcmp(n, "vkQueueSubmit")) return (PFN_vkVoidFunction)Hook_QueueSubmit;
    if (!std::strcmp(n, "vkQueueSubmit2")) return (PFN_vkVoidFunction)Hook_QueueSubmit2;
    if (!std::strcmp(n, "vkQueueWaitIdle")) return (PFN_vkVoidFunction)Hook_QueueWaitIdle;
    if (!std::strcmp(n, "vkQueueSubmit2KHR")) return (PFN_vkVoidFunction)Hook_QueueSubmit2KHR;
    if (!std::strcmp(n, "vkQueueBindSparse")) return (PFN_vkVoidFunction)Hook_QueueBindSparse;
    if (!std::strcmp(n, "vkDeviceWaitIdle")) return (PFN_vkVoidFunction)Hook_DeviceWaitIdle;
    return nullptr;
}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName);

VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
    if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (v->loaderLayerInterfaceVersion > 7) v->loaderLayerInterfaceVersion = 7;
    if (v->loaderLayerInterfaceVersion >= 2) {
        v->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        v->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        v->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    // Once per process, not once per negotiate.
    //
    // The loader re-enumerates the implicit layer directory many times during a single instance
    // creation -- 628 times for one 32-bit vkCreateInstance here, and the same for every other
    // manifest in the directory -- and loads this library on each pass. That is the loader's
    // business, but announcing it each time turned one line into 627 in the user's log. Every other
    // layer stays quiet because none of them log from here.
    static std::once_flag announced;
    if (LayerEnabled()) std::call_once(announced, [] {
        const char* mixed = getenv("VKLayer_DLSS5");
        const char* upper = getenv("VKLAYER_DLSS5");
        const char* value = mixed ? mixed : upper;
        log_printf("=== %s loaded (VKLayer_DLSS5=%s) ===", VK_LAYER_NAME, value ? value : "(unset)");
    });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* pCount,
                                                                  VkLayerProperties* pProperties) {
    if (!pCount) return VK_SUCCESS;
    if (!pProperties) { *pCount = 1; return VK_SUCCESS; }
    if (*pCount < 1) { *pCount = 1; return VK_INCOMPLETE; }
    std::memset(pProperties, 0, sizeof(*pProperties));
    std::strncpy(pProperties->layerName, VK_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    std::strncpy(pProperties->description, "DLSS Linux Open Proxy for AMD",
                 VK_MAX_DESCRIPTION_SIZE - 1);
    pProperties->specVersion = VK_MAKE_VERSION(1, 3, 0);
    pProperties->implementationVersion = 1;
    *pCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char*, uint32_t* pCount,
                                                                      VkExtensionProperties*) {
    if (pCount) *pCount = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!pName) return nullptr;
    if (!std::strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!std::strcmp(pName, "vkNegotiateLoaderLayerInterfaceVersion"))
        return (PFN_vkVoidFunction)vkNegotiateLoaderLayerInterfaceVersion;
    if (!std::strcmp(pName, "vkEnumerateInstanceLayerProperties"))
        return (PFN_vkVoidFunction)vkEnumerateInstanceLayerProperties;
    if (!std::strcmp(pName, "vkEnumerateInstanceExtensionProperties"))
        return (PFN_vkVoidFunction)vkEnumerateInstanceExtensionProperties;
    if (auto fn = LookupHook(pName)) return fn;
    if (instance) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_instances.find(instance);
        if (it != g_instances.end() && it->second.next_gipa) return it->second.next_gipa(instance, pName);
    }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName) return nullptr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (auto fn = LookupDeviceHook(pName)) return fn;
    if (device) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end() && it->second->next_dpa) return it->second->next_dpa(device, pName);
    }
    return nullptr;
}

}  // extern "C"
