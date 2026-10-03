/** @file
 *
 * VK_LAYER_NV_dlssnr — Linux-side Vulkan layer (loaded by the host loader, including the one inside
 * Wine/winevulkan). Hooks the swapchain lifecycle; on present, ships the frame to the selected helper
 * over shared memory and presents the neural-processed result.
 *
 * Enabled implicitly via enable_environment DLSSNR_ENABLE=1 (winevulkan rejects explicitly-named
 * layers, so implicit enable is required under Wine).
 */
#ifndef VK_NO_PROTOTYPES
# define VK_NO_PROTOTYPES
#endif
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/futex.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include "../common/list.h"
#include "../common/shm_protocol.h"
#include "composition.h"
#include "device_features.h"
#include "hotkey.h"
#include "log.h"
#include "network_module.h"
#include "vk_table.h"

/** @brief The layer's name, which has to differ per architecture.
 *
 * The loader keys implicit layers by name, so two manifests claiming the same name are one layer to
 * it: it keeps whichever it read first and then rejects it for the process's word size, reporting
 * only "Requested layer VK_LAYER_NV_dlssnr was wrong bit-type" -- with the manifest that would have
 * worked sitting unread beside it. Steam hit the same wall and answered it the same way, which is
 * why its overlay is VK_LAYER_VALVE_steam_overlay_32 next to _64 rather than one name twice.
 */
#ifdef DLSSLOP_LAYER_NAME
# define VK_LAYER_NAME DLSSLOP_LAYER_NAME
#elif defined(DLSSNR_LAYER_32)
# define VK_LAYER_NAME "VK_LAYER_NV_dlssnr_32"
#else
# define VK_LAYER_NAME "VK_LAYER_NV_dlssnr"
#endif

/** @brief The smaller of two numbers, as std::min compares them: @a a unless @a b is less. */
static double
min_d (double a,
       double b)
{
	return b < a ? b : a;
}

// ---------------------------------------------------------------------------
// Shared memory transport
// ---------------------------------------------------------------------------

/** @brief The state that struct shm_map records in its flags. */
enum shm_map_flags : uint32_t {
	SHM_MAP_ANSWERED = 1 << 0, //!< The helper has answered a frame.
	SHM_MAP_DEAD     = 1 << 1  //!< Frames pass through until the helper is back.
};

/** @brief The layer's mappings of the channel, and what it knows of the helper behind it.
 *
 * Three mappings rather than one.
 *
 * The file is a header followed by two pixel regions each large enough for the biggest frame the
 * protocol allows, which is a quarter of a gigabyte in total. Mapping all of it was fine on 64-bit
 * and is not on 32-bit: a 32-bit game has about 3 GB of address space and would be handing a tenth of
 * it to a reservation it never touches. Both region offsets are page-aligned by construction, so each
 * can be mapped on its own at the size actually in use -- a few megabytes for a real frame instead of
 * 265.
 *
 * The rest is liveness, so a game is never made to wait on a helper that is not there.
 *
 * shm_map() makes an empty map, whose descriptors are -1. shm_map_open() maps the header, and
 * shm_map_fini() unmaps and closes what the map holds, and leaves it empty.
 */
struct shm_map {
	struct ShmHeader *hdr;                //!< The header, mapped; nullptr until shm_map_open().
	uint8_t          *in_pixels;          //!< The input region, mapped.
	uint8_t          *out_pixels;         //!< The output region, mapped.
	char             *path;               //!< The file's path: the socket and the lock are beside it.
	size_t            mapped_frame_bytes; //!< The size of each region's mapping.
	size_t            path_length;        //!< The bytes of path.
	double            retry_after_ms;     //!< When a heartbeat may end SHM_MAP_DEAD (log_now_ms()).
	double            start_after_ms;     //!< start_worker()'s rate limit (log_now_ms()).
	double            open_after_ms;      //!< shm_map_open()'s rate limit (log_now_ms()).
	uint32_t          timeouts;           //!< The requests in a row that the helper did not answer.
	uint32_t          last_control_seq;   //!< The header's controlSeq, as last seen.
	uint32_t          last_heartbeat;     //!< The header's heartbeat, as last seen.
	uint32_t          flags;              //!< enum shm_map_flags.
	int               fd;                 //!< The channel's descriptor, or -1.
	int               producer_fd;        //!< The producer lock's descriptor, or -1.
};

/** @brief An empty map: nothing mapped, and no descriptor.
 *
 * @return The map.
 */
static struct shm_map
shm_map (void)
{
	return (struct shm_map){ .fd = -1, .producer_fd = -1 };
}

/** @brief Unmaps the channel and closes the descriptors that the map holds, then leaves it empty.
 *
 * @param dest The map, or nullptr.
 */
static void
shm_map_fini (struct shm_map *dest)
{
	if (!dest)
		return;

	if (dest->in_pixels)
		munmap(dest->in_pixels, dest->mapped_frame_bytes);
	if (dest->out_pixels)
		munmap(dest->out_pixels, dest->mapped_frame_bytes);
	if (dest->hdr)
		munmap(dest->hdr, kHeaderBytes);
	if (dest->fd >= 0) {
		close(dest->fd);
		dest->fd = -1;
	}
	if (dest->producer_fd >= 0) {
		close(dest->producer_fd);
		dest->producer_fd = -1;
	}
	free(dest->path);
	dest->path = nullptr;
	*dest = shm_map();
}

/** @brief Takes the channel's pixel slot for this process, without waiting.
 *
 * The channel has one pixel slot. A game can have a separate launcher or helper process with its own
 * Vulkan device, so the in-process primary-swapchain election is not sufficient. Contending processes
 * present untouched while another owns the slot. This is a separate file because the worker locks
 * the SHM file for its lifetime. Only this user can create it: shm_map_open()'s ensure_parent_dir()
 * admits nothing but a private directory. A failed open is retried on the next frame.
 *
 * @param s The map.
 * @return  true if this process now holds the slot; the caller releases it with
 *          flock(s->producer_fd, LOCK_UN).
 */
static bool
shm_map_lock_producer (struct shm_map *s)
{
	if (!s->hdr)
		return false;

	if (s->producer_fd < 0) {
		static constexpr char suffix[] = ".producer.lock";
		char *lock = malloc(s->path_length + sizeof suffix);
		if (!lock)
			return false;
		memcpy(lock, s->path, s->path_length);
		memcpy(lock + s->path_length, suffix, sizeof suffix);
		int const fd = open(lock, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
		free(lock);
		lock = nullptr;
		if (fd < 0)
			return false;
		s->producer_fd = fd;
	}
	return !flock(s->producer_fd, LOCK_EX | LOCK_NB);
}

/** @brief Creates a directory unless it exists.
 *
 * @param dir The directory.
 * @return    true if it exists now; otherwise why it could not be created is logged.
 */
static bool
make_dir (char const *dir)
{
	if (!mkdir(dir, 0700) || errno == EEXIST)
		return true;
	log_printf("[shm] cannot create %s: %s", dir, strerror(errno));
	return false;
}

/** @brief Creates the directory of a file, and checks that it is this user's alone.
 *
 * The directory now lives under /tmp, which is world-writable, so it is worth checking that what we
 * are about to open really is ours: a directory, owned by this uid, with nothing granted to anyone
 * else. Anything else and we refuse rather than create the file inside it.
 *
 * @param path   The file's path, which the function cuts at each slash in turn and restores.
 * @param length The path's length.
 * @return       true if the file may be created there.
 */
static bool
ensure_parent_dir (char   *path,
                   size_t  length)
{
	char *const slash = memrchr(path, '/', length);
	if (!slash || slash == path)
		return true;

	// The directory: the path up to the file's name.
	*slash = '\0';
	// Each ancestor, then the directory itself, up to the first that cannot be created.
	bool made = true;
	for (char *end = strchr(path + 1, '/'); made && end; end = strchr(end + 1, '/')) {
		*end = '\0';
		made = make_dir(path);
		*end = '/';
	}
	made = made && make_dir(path);

	struct stat st;
	bool const exists = made && lstat(path, &st) == 0;
	bool const ours = exists && S_ISDIR(st.st_mode) && st.st_uid == getuid()
	                  && (st.st_mode & (S_IRWXG | S_IRWXO)) == 0;
	if (made && !exists)
		log_printf("[shm] %s is missing", path);
	else if (exists && !ours)
		log_printf("[shm] refusing %s: it is not a private directory owned by this user", path);
	*slash = '/';
	return ours;
}

/** @brief Maps the two pixel regions at the size this frame needs, remapping when the size changes.
 *
 * @param s     The map, whose header is mapped.
 * @param bytes The frame's size.
 * @return      true if both regions hold at least @a bytes.
 */
static bool
shm_map_frames (struct shm_map *s,
                size_t          bytes)
{
	if (s->mapped_frame_bytes >= bytes && s->in_pixels && s->out_pixels)
		return true;
	if (bytes > kMaxFrame)
		return false;

	// Round up to 64 KiB so a small change in resolution does not remap every frame.
	size_t const want = (bytes + 65535) / 65536 * 65536;

	if (s->in_pixels)
		munmap(s->in_pixels, s->mapped_frame_bytes);
	if (s->out_pixels)
		munmap(s->out_pixels, s->mapped_frame_bytes);
	s->in_pixels = nullptr;
	s->out_pixels = nullptr;
	s->mapped_frame_bytes = 0;

	void *const in = mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, s->fd, (off_t)kHeaderBytes);
	if (in == MAP_FAILED) {
		log_printf("[shm] could not map the input region (%zu bytes)", want);
		return false;
	}

	void *const out = mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, s->fd,
	                       (off_t)(kHeaderBytes + kMaxFrame));
	if (out == MAP_FAILED) {
		munmap(in, want);
		log_printf("[shm] could not map the output region (%zu bytes)", want);
		return false;
	}

	s->in_pixels = in;
	s->out_pixels = out;
	s->mapped_frame_bytes = want;
	return true;
}

/** @brief The channel's path: DLSSNR_SHM if it is set and not empty, otherwise ShmDefaultPath().
 *
 * @param length Receives the path's length.
 * @return       The path, which the caller frees, or nullptr without memory or when the default
 *               path cannot be formatted.
 */
static char *
channel_path (size_t *length)
{
	char const *const env = getenv("DLSSNR_SHM");
	if (env && *env) {
		size_t const n = strlen(env);
		char *const path = malloc(n + 1);
		if (!path)
			return nullptr;
		memcpy(path, env, n + 1);
		*length = n;
		return path;
	}

	int const n = ShmDefaultPath(nullptr, 0);
	if (n < 0)
		return nullptr;
	size_t const size = (size_t)n + 1;
	char *path = malloc(size);
	// DLSSNR_UID can change between the two calls, and a path cut to fit would name another file.
	if (path && ShmDefaultPath(path, size) != n) {
		free(path);
		path = nullptr;
	}
	if (path)
		*length = size - 1;
	return path;
}

/** @brief Maps the channel's header, creating the file if it is not there.
 *
 * The channel is channel_path()'s, which the map keeps once the header is mapped. Every present asks
 * for it, so a failed attempt is not repeated, nor logged again, for two seconds.
 *
 * @param s The map.
 * @return  true if the header is mapped.
 */
static bool
shm_map_open (struct shm_map *s)
{
	if (s->hdr)
		return true;
	double const now = log_now_ms();
	if (now < s->open_after_ms)
		return false;
	s->open_after_ms = now + 2000.0;

	size_t length;
	char *p = channel_path(&length);
	if (!p) {
		log_printf("[shm] cannot make the channel's path");
		return false;
	}
	if (!ensure_parent_dir(p, length))
		goto fail;
	int fd = open(p, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
	if (fd < 0) {
		log_printf("[shm] open %s failed", p);
		goto fail;
	}
	// The file still spans the whole protocol -- the offsets are fixed and both sides agree on them --
	// but it is sparse, so the size on disk is what has actually been written.
	off_t const total = (off_t)ShmTotalBytes();
	struct stat st;
	if ((fstat(fd, &st) != 0 || st.st_size < total) && ftruncate(fd, total) != 0) {
		close(fd);
		fd = -1;
		goto fail;
	}

	void *const m = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (m == MAP_FAILED) {
		log_printf("[shm] mmap of the header failed");
		close(fd);
		fd = -1;
		goto fail;
	}
	s->fd = fd;
	s->hdr = m;
	s->path = p;
	s->path_length = length;
	// A mapping left by an older build has a different magic, a different version, or a header laid
	// out differently; re-initialising is the only safe reading of any of those.
	//
	// But say so, loudly. A live process on the other side of the mismatch keeps re-initialising the
	// other way, and the two then silently reset each other's settings forever -- the layer keeps
	// composing with its old field set and every setting the newer side writes is invisible. That is
	// indistinguishable from "the new feature does nothing", which is how a stale layer reads until
	// someone checks the log.
	// Each word read once: another process may be rewriting them.
	uint32_t const magic = atomic_load(&s->hdr->magic);
	uint32_t const version = atomic_load(&s->hdr->version);
	if (magic != kShmMagic || version != kShmVersion || atomic_load(&s->hdr->passes) == 0) {
		if (magic == kShmMagic && version != kShmVersion)
			log_printf("[shm] header is version %u but this layer is v%u -- another process is out of "
			           "date, re-initialising it; update the layer, the helper and the GUI together",
			           version, kShmVersion);
		ShmInitDefaults(s->hdr);
	}
	s->last_heartbeat = atomic_load(&s->hdr->heartbeat);
	log_printf("[shm] attached %s seq_req=%u seq_resp=%u", s->path, atomic_load(&s->hdr->seq_req),
	           atomic_load(&s->hdr->seq_resp));
	return true;

fail:
	free(p);
	p = nullptr;
	return false;
}

/** @brief Whether frames go to the helper: the channel asks for neural rendering, and the helper is
 *         not given up on.
 *
 * @param s The map.
 * @return  true if the frame goes to the helper, and true if the channel does not open, which leaves
 *          the frame untouched later.
 */
static bool
shm_map_neural_enabled (struct shm_map *s)
{
	if (!shm_map_open(s))
		return true;
	if (atomic_load(&s->hdr->quit)) {
		s->flags |= SHM_MAP_DEAD;
		return false;
	}
	uint32_t const ctrl = atomic_load(&s->hdr->controlSeq);
	if (ctrl != s->last_control_seq) {
		s->last_control_seq = ctrl;
		if ((s->flags & SHM_MAP_DEAD) && ShmNeuralEnabled(s->hdr)) {
			s->flags &= ~SHM_MAP_DEAD;
			s->timeouts = 0;
			log_printf("[shm] control changed, re-enabling");
		}
	}
	uint32_t const hb = atomic_load(&s->hdr->heartbeat);
	if (hb != s->last_heartbeat) {
		s->last_heartbeat = hb;
		// A heartbeat alone is not a reason to try again immediately. The helper ticks it while it
		// sits idle, so a helper that is up but not answering used to re-enable the layer the moment
		// it had given up -- which cost the game another round of full-length waits, over and over.
		// That is the stutter: recover, stall, give up, recover.
		if ((s->flags & SHM_MAP_DEAD) && log_now_ms() >= s->retry_after_ms && ShmNeuralEnabled(s->hdr)) {
			s->flags &= ~SHM_MAP_DEAD;
			s->timeouts = 0;
			log_printf("[shm] helper heartbeat, trying again");
		}
	}
	if (s->flags & SHM_MAP_DEAD)
		return false;
	return ShmNeuralEnabled(s->hdr);
}

/** @brief Gives up waiting for the helper's answer to a request.
 *
 * Four requests in a row rather than eight, and with a pause before the next attempt, so giving up
 * costs a fraction of a second and retrying costs that again only every few seconds.
 *
 * @param s              The map.
 * @param req            The request's number.
 * @param t_signal       When the request was published (log_now_ms()).
 * @param budget_ms      How long the request could wait.
 * @param helper_present Whether the helper said it was attached.
 * @return               false: the frame presents untouched.
 */
static bool
shm_map_time_out (struct shm_map *s,
                  uint32_t        req,
                  double          t_signal,
                  double          budget_ms,
                  bool            helper_present)
{
	log_printf("[shm] worker did not answer frame %u in %.0f ms (state=%u heartbeat=%u); "
	           "presenting original frames until it answers", req, log_now_ms() - t_signal,
	           atomic_load(&s->hdr->helperState), atomic_load(&s->hdr->heartbeat));
	if (++s->timeouts < 4)
		return false;

	s->flags |= SHM_MAP_DEAD;
	s->retry_after_ms = log_now_ms() + 5000.0;
	log_printf("[shm] no answer in %.0f ms x4 (helper %s); passing frames through, retrying in 5s "
	           "(seq_req=%u seq_resp=%u heartbeat=%u)",
	           budget_ms, helper_present ? "is present but silent" : "not running",
	           atomic_load(&s->hdr->seq_req), atomic_load(&s->hdr->seq_resp),
	           atomic_load(&s->hdr->heartbeat));
	return false;
}

/** @brief One round trip: publishes the proxy, waits for the model's answer, and copies it back
 *         unless it crossed in the device-local transport.
 *
 * What crosses is the proxy at the model's own resolution: R8G8B8A8_UNORM or, with hdrEncode,
 * R16G16B16A16_SFLOAT, sRGB-encoded at either precision because the encode has already done that
 * work on the GPU. The helper therefore never has to know what format the game presents in, and the
 * working scale reduces this copy quadratically.
 *
 * @param s             The map.
 * @param w             The proxy's width.
 * @param h             The proxy's height.
 * @param bytes         The proxy's size: 4 or 8 bytes a pixel.
 * @param proxy         The proxy, for the staging copy.
 * @param model_out     Receives the answer, for the staging copy.
 * @param hdr_encode    Whether the proxy is float16.
 * @param transport_gen The device-local transport's generation, or 0 to stage through the channel.
 * @return              true if the helper answered with a frame to compose.
 */
static bool
shm_map_process_frame (struct shm_map *s,
                       uint32_t        w,
                       uint32_t        h,
                       size_t          bytes,
                       void const     *proxy,
                       void           *model_out,
                       bool            hdr_encode,
                       uint32_t        transport_gen)
{
	if (s->flags & SHM_MAP_DEAD)
		return false;
	if (!shm_map_open(s)) {
		s->flags |= SHM_MAP_DEAD;
		return false;
	}
	if (w > kMaxW || h > kMaxH || w < kMinW || h < kMinH)
		return false;
	if (bytes != (size_t)w * h * 4 && bytes != (size_t)w * h * 8)
		return false;
	if (atomic_load(&s->hdr->quit)) {
		s->flags |= SHM_MAP_DEAD;
		return false;
	}

	bool const time = log_time_enabled();
	double const t0 = log_now_ms();
	if (!shm_map_frames(s, bytes)) {
		s->flags |= SHM_MAP_DEAD;
		return false;
	}
	// A request that names a device-local transport generation crosses in the exported buffers: the
	// GPU already wrote the proxy where the daemon reads it, and there is nothing to copy.
	if (!transport_gen)
		memcpy(s->in_pixels, proxy, bytes);
	double const t_copy = log_now_ms();
	atomic_store(&s->hdr->width, w);
	atomic_store(&s->hdr->height, h);
	// RGBA byte order either way; the float path keeps the same swizzle
	atomic_store(&s->hdr->format, 1u);
	// Say what the bytes ARE before announcing them: the helper sizes its read by this, never by what
	// it hopes the layer has switched to. The release fence below covers it like the pixels.
	atomic_store(&s->hdr->hdrEncode, hdr_encode ? 1u : 0u);
	atomic_store(&s->hdr->transportGen, transport_gen);
	uint32_t const req = atomic_load(&s->hdr->seq_req) + 1;
	// The release pairs with the helper's acquire on seq_resp: everything this process wrote -- the
	// proxy, whether by the GPU into the exported buffer or by the memcpy above -- is visible to the
	// helper before it sees the new request number. (The GPU's own write is fenced earlier, by leg
	// 1's vkWaitForFences; this fence covers the host-visible ordering across processes.)
	atomic_thread_fence(memory_order_release);
	atomic_store(&s->hdr->seq_req, req);
	syscall(SYS_futex, &s->hdr->seq_req, FUTEX_WAKE, 1, nullptr, nullptr, 0);

	// How long this frame may wait, which is a question about whether anyone is listening.
	//
	// A live helper needs real time: the model is milliseconds of work and building its feature on
	// the first frame is far more than that. A helper that is not running needs none at all, and the
	// old fixed second-per-frame budget meant a game whose helper was simply not started froze for
	// eight seconds before the layer gave up. That is what this is for.
	// Is anything listening? The helper says so itself, from the moment it attaches until it exits,
	// which is the only signal that stays true while it is busy. Heartbeats do not: it stops ticking
	// them precisely while it is building the model's feature.
	bool const helper_present = atomic_load(&s->hdr->helperState) != kHelperStopped;

	// The first frame of a size is not like the others. It makes the helper load the model and build
	// a feature -- measured at 194 ms for a small frame and more for a large one -- against about 4 ms
	// once it is warm. Timing that out and giving up is how a working helper gets abandoned before it
	// has answered once.
	bool const warming_up = !(s->flags & SHM_MAP_ANSWERED);
	// HIP passes share one model and run sequentially. Allow a longer chain under game GPU contention
	// without changing the single-pass deadline.
	double const frame_budget_ms = min_d(10000.0, 1000.0 * (double)ShmPasses(s->hdr));
	double const budget_ms = !helper_present ? 20.0 : (warming_up ? 10000.0 : frame_budget_ms);

	// Wait for the helper (fail-open: present the original frame on timeout).
	double const t_signal = log_now_ms();
	uint32_t beat = atomic_load(&s->hdr->heartbeat);
	double beat_at = 0.0;
	for (;;) {
		uint32_t const response = atomic_load_explicit(&s->hdr->seq_resp, memory_order_acquire);
		if (response == req)
			break;
		if (atomic_load(&s->hdr->quit)) {
			s->flags |= SHM_MAP_DEAD;
			return false;
		}
		double const elapsed = log_now_ms() - t_signal;
		// The worker ticks its heartbeat every 100 ms, also mid-frame. One that exits or dies (even
		// by SIGKILL) will not answer; stop waiting for it.
		if (elapsed >= budget_ms || atomic_load(&s->hdr->helperState) != kHelperRunning)
			return shm_map_time_out(s, req, t_signal, budget_ms, helper_present);
		uint32_t const b = atomic_load(&s->hdr->heartbeat);
		if (b != beat) {
			beat = b;
			beat_at = elapsed;
		} else if (elapsed - beat_at > 500.0) {
			return shm_map_time_out(s, req, t_signal, budget_ms, helper_present);
		}
		// Shared futexes work across the two MAP_SHARED mappings. Comparing the observed response
		// value in the kernel closes the check-to-sleep race; the deadline (under 50 ms) also lets a
		// dead worker fail open and quit remain responsive.
		struct timespec const timeout = { 0, (long)(min_d(50.0, budget_ms - elapsed) * 1000000.0) };
		syscall(SYS_futex, &s->hdr->seq_resp, FUTEX_WAIT, response, &timeout, nullptr, 0);
	}

	s->timeouts = 0;
	s->flags |= SHM_MAP_ANSWERED;
	// The helper's GPU wrote the answer into this region (or the memcpy below reads the staging copy
	// of it); the acquire pairs with the helper's release before seq_resp.
	atomic_thread_fence(memory_order_acquire);
	// The helper answers even when it could not use the frame. seq_ok says whether the answer is
	// worth composing; when it is not, the game's own frame is what to present. The echo says the
	// answer was made for this raster: another swapchain (the Steam overlay, or this one's
	// predecessor mid-resize) may have had its request answered in the meantime, and seq_resp only
	// counts. Composing that answer here would copy a different number of bytes into these surfaces
	// -- the row-shifted colour garbage this check exists to refuse.
	bool const ok = atomic_load(&s->hdr->seq_ok) == req && atomic_load(&s->hdr->answeredW) == w
	                && atomic_load(&s->hdr->answeredH) == h;
	if (!ok)
		log_printf("[shm] helper could not use frame %u (ok=%u)", req, atomic_load(&s->hdr->seq_ok));
	if (ok && !transport_gen)
		memcpy(model_out, s->out_pixels, bytes);
	// Every device's frames, which presents on other threads count too.
	static _Atomic(uint32_t) frame_no;
	if (time && (atomic_fetch_add(&frame_no, 1) + 1) % log_time_interval() == 0) {
		double const t_done = log_now_ms();
		log_printf("[time] shm copy=%.2f signal=%.2f wait=%.2f total=%.2f ms",
		           t_copy - t0, t_signal - t_copy, t_done - t_signal, t_done - t0);
	}
	return ok;
}

/** @brief The address of the daemon's transport socket beside the channel, as ShmTransportPath()
 *         names it.
 *
 * @param s       The map, whose header is mapped.
 * @param address Receives the address.
 * @return        false if the socket's path does not fit in an address: a path cut to fit would name
 *                another socket.
 */
static bool
transport_address (struct shm_map const *s,
                   struct sockaddr_un   *address)
{
	if (s->path_length + sizeof kShmTransportSuffix > sizeof address->sun_path)
		return false;

	*address = (struct sockaddr_un){ .sun_family = AF_UNIX };
	memcpy(address->sun_path, s->path, s->path_length);
	memcpy(address->sun_path + s->path_length, kShmTransportSuffix, sizeof kShmTransportSuffix);
	return true;
}

/** @brief Starts a worker that stopped (idle, crashed) through its systemd socket unit, for which a
 *         connection is enough.
 *
 * At most every two seconds, and never waiting: this is the present path.
 *
 * @param s The map, whose header is mapped.
 */
static void
start_worker (struct shm_map *s)
{
	double const now = log_now_ms();
	if (now < s->start_after_ms)
		return;

	s->start_after_ms = now + 2000.0;
	struct sockaddr_un address;
	if (!transport_address(s, &address))
		return;
	int sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	if (sock < 0)
		return;
	connect(sock, (struct sockaddr const *)&address, sizeof address);
	close(sock);
	sock = -1;
}

/** @brief What became of a device-local transport offer. */
enum offer {
	OFFER_READY,    //!< The daemon imported it.
	OFFER_DECLINED, //!< The daemon refused it, or it could not be sent.
	OFFER_WAITING,  //!< The daemon has not answered yet.
	OFFER_LATER     //!< The daemon stopped before it took it: the pair waits for the next one.
};

/** @brief Hands the composition's exported frames to the daemon under a fresh generation and wakes
 *         it.
 *
 * @param s    The map, whose header is mapped.
 * @param comp The composition.
 * @return     The connection the answer comes on, or -1.
 */
static int
send_offer (struct shm_map     *s,
            struct composition *comp)
{
	struct sockaddr_un address;
	if (!transport_address(s, &address))
		return -1;
	struct ShmTransportOffer offer = { .magic = kShmMagic };
	int fds[2] = { -1, -1 };
	if (!composition_export_transport(comp, fds, &offer))
		return -1;
	alignas(struct cmsghdr) char control[CMSG_SPACE(sizeof fds)] = {0};
	struct iovec data = { .iov_base = &offer, .iov_len = sizeof offer };
	struct msghdr const message = {
		.msg_iov        = &data,
		.msg_iovlen     = 1,
		.msg_control    = control,
		.msg_controllen = sizeof control
	};
	struct cmsghdr *const rights = CMSG_FIRSTHDR(&message);
	rights->cmsg_level = SOL_SOCKET;
	rights->cmsg_type = SCM_RIGHTS;
	rights->cmsg_len = CMSG_LEN(sizeof fds);
	memcpy(CMSG_DATA(rights), fds, sizeof fds);
	int sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
	bool const sent = sock >= 0 && !connect(sock, (struct sockaddr const *)&address, sizeof address)
	                  && sendmsg(sock, &message, MSG_NOSIGNAL) == (ssize_t)sizeof offer;
	close(fds[0]);
	fds[0] = -1;
	close(fds[1]);
	fds[1] = -1;
	if (!sent) {
		if (sock >= 0) {
			close(sock);
			sock = -1;
		}
		return -1;
	}
	// The daemon takes offers between requests.
	syscall(SYS_futex, &s->hdr->seq_req, FUTEX_WAKE, 1, nullptr, nullptr, 0);
	return sock;
}

/** @brief The daemon's answer to the composition's offer.
 *
 * The first look sends the offer and waits a moment, later ones only look. No answer by the time
 * @a expires holds counts as a refusal.
 *
 * @param s       The map, whose header is mapped.
 * @param comp    The composition.
 * @param expires When an outstanding offer counts as refused (log_now_ms()); set when it is sent.
 * @return        What became of the offer.
 */
static enum offer
offer_transport (struct shm_map     *s,
                 struct composition *comp,
                 double             *expires)
{
	int wait = 0;
	if (composition_offer_connection(comp) < 0) {
		int const connection = send_offer(s, comp);
		if (connection < 0)
			return atomic_load_explicit(&s->hdr->helperState, memory_order_acquire) == kHelperRunning
			       ? OFFER_DECLINED : OFFER_LATER;
		composition_await_answer(comp, connection);
		*expires = log_now_ms() + 5000.0;
		wait = 250;
	}
	struct pollfd answer = { .fd = composition_offer_connection(comp), .events = POLLIN };
	if (poll(&answer, 1, wait) != 1)
		return log_now_ms() >= *expires ? OFFER_DECLINED : OFFER_WAITING;
	uint8_t imported;
	if (recv(answer.fd, &imported, 1, 0) != 1)
		return OFFER_LATER; // Closed unanswered.
	return imported ? OFFER_READY : OFFER_DECLINED;
}

// ---------------------------------------------------------------------------
// Dispatch chains
// ---------------------------------------------------------------------------

/** @brief An instance the layer knows.
 *
 * The instance-level entry points the layer and the composition need are resolved once. They are
 * kept here rather than on the device chain because this is where the VkInstance handle is in
 * scope. instance_chain_create() makes a chain, and instance_chain_destroy() frees it.
 */
struct instance_chain {
	struct list           node;           //!< The chain's hook in g_instances.
	struct instance_table table;          //!< The next layer's entry points for the instance.
	VkInstance            self;           //!< The instance.
	VkPhysicalDevice     *physical;       //!< What vkEnumeratePhysicalDevices returned for it.
	uint32_t              physical_count; //!< The entries of physical.
	uint32_t              api_version;    //!< The game's: features beyond it need their own extensions.
};

/** @brief Allocates the chain of an instance that is about to be created.
 *
 * @param next_gipa The next layer's vkGetInstanceProcAddr.
 * @param app       The game's application info, or nullptr.
 * @return          The chain, unlinked and without an instance, or nullptr without memory.
 */
static struct instance_chain *
instance_chain_create (PFN_vkGetInstanceProcAddr  next_gipa,
                       VkApplicationInfo const   *app)
{
	struct instance_chain *const ret = calloc(1, sizeof *ret);
	if (!ret)
		return nullptr;

	ret->table.next_gipa = next_gipa;
	ret->api_version = app && app->apiVersion ? app->apiVersion : VK_API_VERSION_1_0;
	return ret;
}

/** @brief Frees what an instance chain owns, then leaves it empty.
 *
 * @param dest The chain, unlinked, or nullptr.
 */
static void
instance_chain_fini (struct instance_chain *dest)
{
	if (!dest)
		return;

	free(dest->physical);
	dest->physical = nullptr;
	*dest = (struct instance_chain){0};
}

/** @brief Frees an instance chain.
 *
 * @param p_dest The chain, unlinked; the pointer is nullptr afterwards. May be nullptr.
 */
static void
instance_chain_destroy (struct instance_chain **p_dest)
{
	if (!p_dest || !*p_dest)
		return;

	struct instance_chain *ptr = *p_dest;
	*p_dest = nullptr;
	instance_chain_fini(ptr);
	free(ptr);
	ptr = nullptr;
}

/** @brief A queue that the game took from its device: the present path needs its family. */
struct device_queue {
	struct list node;   //!< The queue's hook in its device chain's queue_families.
	VkQueue     queue;  //!< The queue.
	uint32_t    family; //!< Its family.
};

/** @brief The state that struct swapchain_state records in its flags. */
enum swapchain_state_flags : uint32_t {
	SWAPCHAIN_STATE_GRAPHICS     = 1 << 0, //!< The present queue's family runs graphics.
	SWAPCHAIN_STATE_LEG2_PENDING = 1 << 1, //!< Leg 2 is submitted and its fence not yet waited on.
	SWAPCHAIN_STATE_READY        = 1 << 2, //!< create_resources() succeeded.
	SWAPCHAIN_STATE_PASS_THROUGH = 1 << 3  //!< Its frames present untouched.
};

/** @brief A swapchain the layer tracks, and what the layer made for it.
 *
 * Two fences, not one. Leg 1's must be waited on before the proxy is handed to the helper -- the
 * sequence number is the helper's only ordering signal, and it may not be bumped ahead of the write
 * it announces. Leg 2's needs no wait at all in its own frame: the present waits on the image's
 * semaphore, so the GPU orders them without the CPU. The wait moves to the start of the next present,
 * where the command buffer and the composed surfaces are reused, which takes a full GPU stall out of
 * the frame it belongs to.
 *
 * Per image, leg 2 signals a semaphore and the present waits on it. Queue order alone does not order
 * a present behind earlier work, and an image is acquired again only after its present waited.
 *
 * The family is the present queue's: the composition's, and the in-layer network's, which converts
 * formats with blits and so needs a graphics family. hdr_kind is what the swapchain's format and
 * colour space say the frame carries: the float swapchain holds linear light, and a 10-bit one with a
 * PQ colour space holds ST 2084 code. The composition owns every surface it needs, including the
 * transport pair -- exported device-local memory when the daemon imports it, host-visible staging
 * when it does not. It is empty until create_resources() builds it.
 *
 * swapchain_state_create() makes a state, create_resources() its Vulkan objects, and
 * swapchain_state_destroy() frees it. A handle is stored only once the call that made it succeeded.
 */
struct swapchain_state {
	struct composition         comp;             //!< The pass and its surfaces.
	struct list                node;             //!< The state's hook in its device chain's swapchains.
	struct device_table const *vk;               //!< The device's next-layer entry points.
	VkDevice                   device;           //!< The device.
	VkSwapchainKHR             handle;           //!< The swapchain.
	VkImage                   *images;           //!< The swapchain's images.
	VkSemaphore               *leg2_done;        //!< Per image, what leg 2 signals.
	VkPipelineStageFlags      *wait_stages;      //!< The first submit's wait stages, reused.
	VkFence                    fence_leg1;       //!< Leg 1's fence.
	VkFence                    fence_leg2;       //!< Leg 2's fence.
	VkCommandPool              pool;             //!< The pool of cb.
	VkCommandBuffer            cb;               //!< Both legs' command buffer.
	double                     offer_expires;    //!< When the offer counts as refused (log_now_ms()).
	uint32_t                   image_count;      //!< The entries of images and leg2_done.
	uint32_t                   family;           //!< The present queue's family.
	uint32_t                   hdr_kind;         //!< enum HdrKind of its format and colour space.
	uint32_t                   width;            //!< The swapchain's width.
	uint32_t                   height;           //!< The swapchain's height.
	uint32_t                   wait_stage_count; //!< The entries of wait_stages.
	VkFormat                   format;           //!< The swapchain's format.
	uint32_t                   flags;            //!< enum swapchain_state_flags.
};

/** @brief Allocates the state of a swapchain.
 *
 * @param vk          The device's next-layer entry points.
 * @param device      The device.
 * @param handle      The swapchain.
 * @param image_count The number of its images.
 * @return            The state, unlinked and with room for the images, or nullptr without memory.
 */
static struct swapchain_state *
swapchain_state_create (struct device_table const *vk,
                        VkDevice                   device,
                        VkSwapchainKHR             handle,
                        uint32_t                   image_count)
{
	struct swapchain_state *ret = calloc(1, sizeof *ret);
	VkImage *images = calloc(image_count, sizeof *images);
	VkSemaphore *leg2_done = calloc(image_count, sizeof *leg2_done);
	if (!ret || !images || !leg2_done) {
		free(ret);
		ret = nullptr;
		free(images);
		images = nullptr;
		free(leg2_done);
		leg2_done = nullptr;
		return nullptr;
	}

	ret->comp = composition_empty();
	ret->vk = vk;
	ret->device = device;
	ret->handle = handle;
	ret->images = images;
	ret->leg2_done = leg2_done;
	ret->image_count = image_count;
	return ret;
}

/** @brief Destroys what the layer made for a swapchain and frees what its state owns, then leaves
 *         the state empty.
 *
 * @param dest The state, unlinked, or nullptr.
 */
static void
swapchain_state_fini (struct swapchain_state *dest)
{
	if (!dest)
		return;

	composition_fini(&dest->comp);
	struct device_table const *const vk = dest->vk;
	if (dest->fence_leg1)
		vk->vkDestroyFence(dest->device, dest->fence_leg1, nullptr);
	if (dest->fence_leg2)
		vk->vkDestroyFence(dest->device, dest->fence_leg2, nullptr);
	for (uint32_t i = 0; i < dest->image_count; ++i)
		if (dest->leg2_done[i])
			vk->vkDestroySemaphore(dest->device, dest->leg2_done[i], nullptr);
	if (dest->pool)
		vk->vkDestroyCommandPool(dest->device, dest->pool, nullptr);
	free(dest->images);
	dest->images = nullptr;
	free(dest->leg2_done);
	dest->leg2_done = nullptr;
	free(dest->wait_stages);
	dest->wait_stages = nullptr;
	*dest = (struct swapchain_state){ .comp = composition_empty() };
}

/** @brief Destroys what the layer made for a swapchain and frees its state.
 *
 * @param p_dest The state, unlinked; the pointer is nullptr afterwards. May be nullptr.
 */
static void
swapchain_state_destroy (struct swapchain_state **p_dest)
{
	if (!p_dest || !*p_dest)
		return;

	struct swapchain_state *ptr = *p_dest;
	*p_dest = nullptr;
	swapchain_state_fini(ptr);
	free(ptr);
	ptr = nullptr;
}

/** @brief The state that struct device_chain records in its flags. */
enum device_chain_flags : uint32_t {
	DEVICE_CHAIN_EXPORT_MEMORY = 1 << 0, //!< VK_KHR_external_memory_fd and vkGetMemoryFdKHR are there.
	DEVICE_CHAIN_NETWORK       = 1 << 1, //!< The ledger enabled the in-layer network's requirements.
	DEVICE_CHAIN_IN_LAYER_OFF  = 1 << 2  //!< The in-layer network failed or did not open.
};

/** @brief A device the layer knows.
 *
 * The loader hands every layer vkSetDeviceLoaderData in its own VkLayerDeviceCreateInfo node, the
 * hook for installing a dispatch table on a dispatchable object a layer creates; see
 * set_loader_data(). network_submit is held by the in-layer network's build around each of its
 * submits, and by the layer's own device waits, which run outside lock: a device wait may not overlap
 * a submit. The in-layer network is opened on the first frame, and is off once it failed. Its memory
 * belongs to the queue family it opened on. A build of the network frees what the frame of
 * in_layer_last used.
 *
 * device_chain_create() makes a chain in place, with its mutexes and an empty channel map, and
 * device_chain_destroy() frees it once the swapchains' states are destroyed and the network is
 * closed.
 */
struct device_chain {
	struct list                node;                   //!< The chain's hook in g_devices.
	struct device_table        table;                  //!< The next layer's device entry points.
	struct shm_map             shm;                    //!< The channel.
	struct list                swapchains;             //!< The swapchain_states the layer tracks.
	struct list                queue_families;         //!< The device_queues of the game's queues.
	pthread_mutex_t            lock;                   //!< Guards swapchains, queues and presents.
	pthread_mutex_t            network_submit;         //!< Held around the network build's submits.
	struct instance_chain     *instance;               //!< The instance's chain, or nullptr.
	VkPhysicalDevice           physical;               //!< The physical device.
	VkDevice                   self;                   //!< The device.
	PFN_vkSetDeviceLoaderData  set_device_loader_data; //!< The loader's, or nullptr.
	struct DlsslopNetwork     *in_layer;               //!< The in-layer network, once it opened.
	struct swapchain_state    *in_layer_last;          //!< The swapchain whose frame last reached it.
	struct device_queue       *queue_store;            //!< A device_queue per queue of the device.
	uint64_t                   frames_composed;        //!< The frames composed.
	uint64_t                   frames_passed_through;  //!< The frames presented untouched.
	uint32_t                   queue_store_count;      //!< The entries of queue_store.
	uint32_t                   queue_store_used;       //!< The entries of queue_store in use.
	uint32_t                   in_layer_family;        //!< The queue family the network opened on.
	uint32_t                   flags;                  //!< enum device_chain_flags.
	uint32_t                   pid;                    //!< The process, as the channel's layerPid holds it.
	_Atomic(bool)              inert;                  //!< The layer leaves the device alone.
	char                       network_reason[2048];   //!< The network's last layer reason, as logged.
};

/** @brief Allocates and initializes the chain of a device that is about to be created.
 *
 * A device_queue entry for each queue the device is created with is allocated with the chain,
 * because vkGetDeviceQueue cannot fail. Both mutexes are initialized here, in place, and the
 * channel's map is shm_map()'s, whose descriptors are -1.
 *
 * @param queue_count The number of queues the device is created with.
 * @return            The chain, unlinked and without a device, or nullptr without memory or a
 *                    mutex.
 */
static struct device_chain *
device_chain_create (uint32_t queue_count)
{
	struct device_chain *ret = calloc(1, sizeof *ret);
	struct device_queue *queues = calloc(queue_count, sizeof *queues);
	if (!ret || !queues || pthread_mutex_init(&ret->lock, nullptr))
		goto fail;
	if (pthread_mutex_init(&ret->network_submit, nullptr)) {
		pthread_mutex_destroy(&ret->lock);
		goto fail;
	}

	ret->shm = shm_map();
	ret->pid = (uint32_t)getpid();
	ret->queue_store = queues;
	ret->queue_store_count = queue_count;
	list_init(&ret->swapchains);
	list_init(&ret->queue_families);
	atomic_init(&ret->inert, false);
	return ret;

fail:
	free(ret);
	ret = nullptr;
	free(queues);
	queues = nullptr;
	return nullptr;
}

/** @brief Frees what a device chain owns and destroys its mutexes, then leaves it empty.
 *
 * @param dest The chain, unlinked, without swapchain states and with the network closed, or
 *             nullptr.
 */
static void
device_chain_fini (struct device_chain *dest)
{
	if (!dest)
		return;

	free(dest->queue_store);
	dest->queue_store = nullptr;
	pthread_mutex_destroy(&dest->network_submit);
	pthread_mutex_destroy(&dest->lock);
	shm_map_fini(&dest->shm);
	*dest = (struct device_chain){ .shm = shm_map() };
}

/** @brief Frees a device chain.
 *
 * @param p_dest The chain, as device_chain_fini() takes it; the pointer is nullptr afterwards. May
 *               be nullptr.
 */
static void
device_chain_destroy (struct device_chain **p_dest)
{
	if (!p_dest || !*p_dest)
		return;

	struct device_chain *ptr = *p_dest;
	*p_dest = nullptr;
	device_chain_fini(ptr);
	free(ptr);
	ptr = nullptr;
}

static struct list     g_instances   = LIST_INIT(g_instances);   //!< Under g_state_mutex.
static struct list     g_devices     = LIST_INIT(g_devices);     //!< Under g_state_mutex.
static pthread_mutex_t g_state_mutex = PTHREAD_MUTEX_INITIALIZER; //!< Guards both lists.

/** @brief The in-layer network's module (layer/network_module.h), beside the layer, loaded once for
 *         the first device that enabled the network.
 */
static struct network_module g_network;

/** @brief The chain of an instance; the caller holds g_state_mutex.
 *
 * @param instance The instance.
 * @return         Its chain, or nullptr.
 */
static struct instance_chain *
find_instance (VkInstance instance)
{
	struct instance_chain *ic;
	list_foreach(ic, &g_instances, struct instance_chain, node)
		if (ic->self == instance)
			return ic;
	return nullptr;
}

/** @brief Whether an instance enumerated a physical device; the caller holds g_state_mutex.
 *
 * @param ic       The instance's chain.
 * @param physical The physical device.
 * @return         true if it did.
 */
static bool
has_physical (struct instance_chain const *ic,
              VkPhysicalDevice             physical)
{
	for (uint32_t i = 0; i < ic->physical_count; ++i)
		if (ic->physical[i] == physical)
			return true;
	return false;
}

/** @brief The chain of the instance that enumerated a physical device; the caller holds
 *         g_state_mutex.
 *
 * @param physical The physical device.
 * @return         The instance's chain, or nullptr.
 */
static struct instance_chain *
instance_for_physical (VkPhysicalDevice physical)
{
	struct instance_chain *ic;
	list_foreach(ic, &g_instances, struct instance_chain, node)
		if (has_physical(ic, physical))
			return ic;
	return nullptr;
}

/** @brief Adds the physical devices that an instance chain does not hold yet; the caller holds
 *         g_state_mutex.
 *
 * So that vkCreateDevice finds the instance of its physical device.
 *
 * @param ic      The instance's chain.
 * @param devices What vkEnumeratePhysicalDevices returned.
 * @param count   The number of @a devices.
 * @return        false, adding none, when out of memory.
 */
static bool
remember_physical (struct instance_chain  *ic,
                   VkPhysicalDevice const *devices,
                   uint32_t                count)
{
	// The array grows only when a device is new to IC, by the devices from the first new one on.
	uint32_t i = 0;
	while (i < count && has_physical(ic, devices[i]))
		++i;
	if (i == count)
		return true;
	VkPhysicalDevice *const grown = realloc(ic->physical,
	                                        (ic->physical_count + (size_t)(count - i)) * sizeof *grown);
	if (!grown)
		return false;
	ic->physical = grown;
	ic->physical[ic->physical_count++] = devices[i];
	while (++i < count)
		if (!has_physical(ic, devices[i]))
			ic->physical[ic->physical_count++] = devices[i];
	return true;
}

/** @brief find_device() without the lock; the caller holds g_state_mutex.
 *
 * @param device The device.
 * @return       Its chain, or nullptr.
 */
static struct device_chain *
find_device_ (VkDevice device)
{
	struct device_chain *dc;
	list_foreach(dc, &g_devices, struct device_chain, node)
		if (dc->self == device)
			return dc;
	return nullptr;
}

/** @brief The chain of a device.
 *
 * @param device The device.
 * @return       Its chain, or nullptr.
 */
static struct device_chain *
find_device (VkDevice device)
{
	pthread_mutex_lock(&g_state_mutex);
	struct device_chain *const dc = find_device_(device);
	pthread_mutex_unlock(&g_state_mutex);
	return dc;
}

/** @brief A device's entry of a queue; the caller holds dc->lock.
 *
 * @param dc    The device's chain.
 * @param queue The queue.
 * @return      The entry, or nullptr.
 */
static struct device_queue *
find_queue (struct device_chain *dc,
            VkQueue              queue)
{
	struct device_queue *q;
	list_foreach(q, &dc->queue_families, struct device_queue, node)
		if (q->queue == queue)
			return q;
	return nullptr;
}

/** @brief A device's state of a swapchain; the caller holds dc->lock.
 *
 * @param dc        The device's chain.
 * @param swapchain The swapchain.
 * @return          The state, or nullptr.
 */
static struct swapchain_state *
find_swapchain (struct device_chain *dc,
                VkSwapchainKHR       swapchain)
{
	struct swapchain_state *sc;
	list_foreach(sc, &dc->swapchains, struct swapchain_state, node)
		if (sc->handle == swapchain)
			return sc;
	return nullptr;
}

/** @brief device_for_queue() without the lock; the caller holds g_state_mutex.
 *
 * @param queue The queue.
 * @return      The chain of its device, or nullptr.
 */
static struct device_chain *
device_for_queue_ (VkQueue queue)
{
	struct list *const only = list_only(&g_devices);
	if (only)
		return container_of(only, struct device_chain, node);

	struct device_chain *dc;
	list_foreach(dc, &g_devices, struct device_chain, node) {
		pthread_mutex_lock(&dc->lock);
		bool const found = find_queue(dc, queue) != nullptr;
		pthread_mutex_unlock(&dc->lock);
		if (found)
			return dc;
	}
	return nullptr;
}

/** @brief The chain of a queue's device: the only device if there is one, otherwise the device
 *         that the game took the queue from.
 *
 * @param queue The queue.
 * @return      The chain, or nullptr.
 */
static struct device_chain *
device_for_queue (VkQueue queue)
{
	pthread_mutex_lock(&g_state_mutex);
	struct device_chain *const dc = device_for_queue_(queue);
	pthread_mutex_unlock(&g_state_mutex);
	return dc;
}

/** @brief The layer's own device wait.
 *
 * It runs outside dc->lock, but under network_submit, because a device wait may not overlap the
 * in-layer network's submits.
 *
 * @param dc The device's chain.
 */
static void
wait_device_idle (struct device_chain *dc)
{
	if (!dc->table.vkDeviceWaitIdle)
		return;

	pthread_mutex_lock(&dc->network_submit);
	dc->table.vkDeviceWaitIdle(dc->self);
	pthread_mutex_unlock(&dc->network_submit);
}

/** @brief The one swapchain allowed to drive the neural round trip, chosen as the largest in the
 *         process.
 *
 * The shared-memory channel carries a single raster at a time, but a process can present more than
 * one swapchain -- the game window and the Steam overlay, or, mid-resize, the old and new windows at
 * once. Feeding all of them through one channel makes the helper rebuild its model on every size
 * switch and lets one swapchain be handed another's answer. The largest is the game; the rest present
 * raw. The record is global rather than per-device because the overlay builds its own VkDevice.
 */
struct primary_swap {
	VkDevice       device;    //!< The swapchain's device.
	VkSwapchainKHR swapchain; //!< The swapchain, or VK_NULL_HANDLE.
	uint64_t       area;      //!< Its width times its height.
};

static struct primary_swap g_primary; //!< Under g_primary_mutex.

/** @brief Guards g_primary. A leaf: taken under dc->lock (the present hook's swapchains and
 *         hook_destroy_device()) and never the other way round, and never with g_state_mutex, so the
 *         lock order in the present hook cannot invert against the device hooks.
 */
static pthread_mutex_t g_primary_mutex = PTHREAD_MUTEX_INITIALIZER;

/** @brief claim_primary() without the lock; the caller holds g_primary_mutex. */
static bool
claim_primary_ (VkDevice       device,
                VkSwapchainKHR swapchain,
                uint32_t       w,
                uint32_t       h)
{
	uint64_t const area = (uint64_t)w * h;
	if (g_primary.swapchain == swapchain && g_primary.device == device)
		return true;
	if (g_primary.swapchain != VK_NULL_HANDLE && area <= g_primary.area)
		return false;
	g_primary.device = device;
	g_primary.swapchain = swapchain;
	g_primary.area = area;
	return true;
}

/** @brief Adopts a larger swapchain; a present from anything else passes through untouched.
 *
 * @param device    The swapchain's device.
 * @param swapchain The swapchain.
 * @param w         Its width.
 * @param h         Its height.
 * @return          true if the swapchain drives the channel.
 */
static bool
claim_primary (VkDevice       device,
               VkSwapchainKHR swapchain,
               uint32_t       w,
               uint32_t       h)
{
	pthread_mutex_lock(&g_primary_mutex);
	bool const primary = claim_primary_(device, swapchain, w, h);
	pthread_mutex_unlock(&g_primary_mutex);
	return primary;
}

/** @brief Gives up a swapchain's claim, if it holds it.
 *
 * @param device    The swapchain's device.
 * @param swapchain The swapchain.
 */
static void
release_primary (VkDevice       device,
                 VkSwapchainKHR swapchain)
{
	pthread_mutex_lock(&g_primary_mutex);
	if (g_primary.swapchain == swapchain && g_primary.device == device)
		g_primary = (struct primary_swap){0};
	pthread_mutex_unlock(&g_primary_mutex);
}

/** @brief Where this copy of the layer was loaded from, for the duplicate check below.
 *
 * @return The dynamic linker's record of the path (dladdr()'s dli_fname), which lasts as long as the
 *         layer stays loaded; empty if dladdr() cannot tell.
 */
static char const *
layer_object_path (void)
{
	Dl_info info;
	if (dladdr((void const *)&layer_object_path, &info) && info.dli_fname)
		return info.dli_fname;
	return "";
}

/** @brief Whether a *different* copy of this layer is already in the chain.
 *
 * Local packaging installs an implicit-layer manifest pointing at the build tree while install.sh
 * installs another pointing at the install prefix, and the loader honours both: two copies of the
 * layer, two present hooks, two full round trips, and a single shared-memory file with two writers
 * racing on one sequence number. Only the first copy stays live; the rest declare themselves inert
 * and pass everything through, which turns a corrupted picture or a hang into one warning line.
 *
 * The claim is the object's own path rather than a bare flag, so a second call into the same copy --
 * which is legal, the loader may negotiate more than once -- is told apart from a second copy. Asked
 * once, by read_layer_enabled().
 *
 * @return true if another copy claimed DLSSNR_LAYER_OBJECT; otherwise this copy claims it.
 */
static bool
duplicate_layer_copy (void)
{
	char const *const self = layer_object_path();
	char const *const claimed = getenv("DLSSNR_LAYER_OBJECT");
	if (claimed && *claimed) {
		if (!*self || !strcmp(self, claimed))
			return false;
		log_printf("[layer] another copy is already loaded from %s; this copy (%s) stays inert. "
		           "Remove one of the implicit-layer manifests.", claimed, self);
		return true;
	}
	if (*self && setenv("DLSSNR_LAYER_OBJECT", self, 0))
		log_printf("[layer] cannot claim DLSSNR_LAYER_OBJECT (%s): a second copy of the layer would not "
		           "stand aside", strerror(errno));
	return false;
}

/** @brief One set of keyboards for the process, however many devices the game creates.
 *
 * Zeroed, it has opened nothing. Each device polls it under its own lock, and two devices can present
 * at the same time: one present's rescan could move the node table while the other present reads it,
 * as on upstream's Hotkeys. g_hotkeys_mutex keeps them apart.
 */
static struct hotkeys g_hotkeys;

/** @brief Guards g_hotkeys. A leaf: taken under dc->lock, and nothing but the log's lock under it. */
static pthread_mutex_t g_hotkeys_mutex = PTHREAD_MUTEX_INITIALIZER;

/** @brief Closes the keyboards that the hotkeys opened and unloads libX11 and libXi, so that a layer
 *         loaded again starts with nothing open.
 *
 * Runs at unload, which the loader does when the last instance is destroyed, and at exit. Leaves the
 * hotkeys alone if their mutex is held: at exit another thread can be polling them, and in a child
 * that fork() made while a thread polled, the mutex stays held. The process closes what they opened
 * when it ends.
 */
[[gnu::destructor]]
static void
unload_layer (void)
{
	if (pthread_mutex_trylock(&g_hotkeys_mutex))
		return;

	hotkeys_fini(&g_hotkeys);
	pthread_mutex_unlock(&g_hotkeys_mutex);
}

static pthread_once_t g_toggle_key_once = PTHREAD_ONCE_INIT; //!< Runs read_toggle_key().
static uint32_t       g_toggle_key_from_env;                 //!< DLSSNR_TOGGLE_KEY's key; 0 if none.

/** @brief Reads DLSSNR_TOGGLE_KEY into g_toggle_key_from_env. */
static void
read_toggle_key (void)
{
	g_toggle_key_from_env = hotkey_key_code_from_name(getenv("DLSSNR_TOGGLE_KEY"));
}

/** @brief The key to watch.
 *
 * From the environment if it names one, so it can be bound in a launch option without the
 * interface being involved, and from the header otherwise.
 *
 * @param hdr The header, or nullptr.
 * @return    A Linux KEY_* code; 0 if none is bound.
 */
static uint32_t
toggle_key (struct ShmHeader const *hdr)
{
	pthread_once(&g_toggle_key_once, read_toggle_key);
	if (g_toggle_key_from_env)
		return g_toggle_key_from_env;
	return hdr ? atomic_load(&hdr->toggleKey) : 0u;
}

/** @brief Toggles neural rendering when the key was pressed.
 *
 * Polled before anything asks whether the pass is enabled, because asking first would make turning
 * it off a one-way door: the early return would skip the very code that reads the key to turn it
 * back on.
 *
 * @param dc The device's chain.
 */
static void
poll_hotkeys (struct device_chain *dc)
{
	if (!shm_map_open(&dc->shm))
		return;
	uint32_t const key = toggle_key(dc->shm.hdr);
	// A present that finds another device polling skips its poll rather than wait: that one answers
	// the press.
	if (pthread_mutex_trylock(&g_hotkeys_mutex))
		return;
	bool const pressed = hotkeys_pressed(&g_hotkeys, key);
	pthread_mutex_unlock(&g_hotkeys_mutex);
	if (!pressed)
		return;

	// One read-modify-write: the GUI and dlsslopctl write the same word from other processes.
	uint32_t was_on = atomic_load(&dc->shm.hdr->enabled);
	while (!atomic_compare_exchange_weak(&dc->shm.hdr->enabled, &was_on, was_on ? 0u : 1u))
		continue;
	atomic_fetch_add(&dc->shm.hdr->controlSeq, 1);
	log_printf("[hotkey] %s -> neural rendering %s", hotkey_key_name_from_code(key),
	           was_on ? "off" : "on");
}

static pthread_once_t g_layer_enabled_once = PTHREAD_ONCE_INIT; //!< Runs read_layer_enabled().
static bool           g_layer_enabled;                          //!< Whether this copy works.

/** @brief Reads whether this copy of the layer works into g_layer_enabled. */
static void
read_layer_enabled (void)
{
	if (duplicate_layer_copy())
		return;
	char const *const v = getenv("VKLayer_DLSS5");
	char const *const upper = getenv("VKLAYER_DLSS5");
	char const *const o = getenv("DLSSNR_ENABLE");
	g_layer_enabled = (v && v[0] == '1') || (upper && upper[0] == '1') || (o && o[0] == '1');
}

/** @brief Whether this copy of the layer works.
 *
 * @return true if no other copy is loaded and VKLayer_DLSS5, VKLAYER_DLSS5 or DLSSNR_ENABLE starts
 *         with 1.
 */
static bool
layer_enabled (void)
{
	pthread_once(&g_layer_enabled_once, read_layer_enabled);
	return g_layer_enabled;
}

static pthread_once_t g_network_requested_once = PTHREAD_ONCE_INIT; //!< Runs read_network_requested().
static bool           g_network_requested;                          //!< Whether the network is asked for.

/** @brief Reads whether the in-layer network is asked for into g_network_requested. */
static void
read_network_requested (void)
{
	g_network_requested = device_features_network_requested();
}

/** @brief Whether the in-layer network is asked for, read once, so that a device that enabled it
 *         and the lookups of the queue hooks that serialize its build agree.
 *
 * @return device_features_network_requested() as it was at the first call.
 */
static bool
network_requested (void)
{
	pthread_once(&g_network_requested_once, read_network_requested);
	return g_network_requested;
}

// ---------------------------------------------------------------------------
// Instance hooks
// ---------------------------------------------------------------------------

/** @brief vkCreateInstance: tracks the instance. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_create_instance (VkInstanceCreateInfo const  *pCreateInfo,
                      VkAllocationCallbacks const *pAllocator,
                      VkInstance                  *pInstance)
{
	VkLayerInstanceCreateInfo *link = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
	while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
	                 && link->function == VK_LAYER_LINK_INFO))
		link = (VkLayerInstanceCreateInfo *)link->pNext;
	if (!link || !link->u.pLayerInfo)
		return VK_ERROR_INITIALIZATION_FAILED;

	PFN_vkGetInstanceProcAddr const next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	PFN_vkCreateInstance const create = (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE,
	                                                                    "vkCreateInstance");
	if (!create)
		return VK_ERROR_INITIALIZATION_FAILED;
	// Before the instance, so that running out leaves nothing to destroy.
	struct instance_chain *chain = instance_chain_create(next_gipa, pCreateInfo->pApplicationInfo);
	if (!chain) {
		log_printf("[layer] vkCreateInstance: out of host memory for the layer's state");
		return VK_ERROR_OUT_OF_HOST_MEMORY;
	}

	// Documented pattern: keep the link node in pNext (layers below need it) and advance
	// u.pLayerInfo so the next layer resolves its own chain entry.
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	VkResult const res = create(pCreateInfo, pAllocator, pInstance);
	if (res != VK_SUCCESS) {
		instance_chain_destroy(&chain);
		return res;
	}

	chain->self = *pInstance;
	instance_table_load(&chain->table, *pInstance);

	pthread_mutex_lock(&g_state_mutex);
	list_append(&g_instances, &chain->node);
	log_printf("[layer] vkCreateInstance -> %p", (void *)*pInstance);
	pthread_mutex_unlock(&g_state_mutex);
	return VK_SUCCESS;
}

/** @brief hook_destroy_instance() without the lock; the caller holds g_state_mutex. */
static void
hook_destroy_instance_ (VkInstance                   instance,
                        VkAllocationCallbacks const *pAllocator)
{
	struct instance_chain *chain = find_instance(instance);
	if (!chain)
		return;

	PFN_vkDestroyInstance const destroy = chain->table.vkDestroyInstance;
	list_del(&chain->node);
	instance_chain_destroy(&chain);
	if (destroy)
		destroy(instance, pAllocator);
}

/** @brief vkDestroyInstance: forgets the instance and its physical devices. */
static VKAPI_ATTR void VKAPI_CALL
hook_destroy_instance (VkInstance                   instance,
                       VkAllocationCallbacks const *pAllocator)
{
	pthread_mutex_lock(&g_state_mutex);
	hook_destroy_instance_(instance, pAllocator);
	pthread_mutex_unlock(&g_state_mutex);
}

/** @brief vkEnumeratePhysicalDevices: records the physical devices in their instance's chain. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_enumerate_physical_devices (VkInstance        instance,
                                 uint32_t         *pCount,
                                 VkPhysicalDevice *pPhysicalDevices)
{
	pthread_mutex_lock(&g_state_mutex);
	struct instance_chain *const chain = find_instance(instance);
	pthread_mutex_unlock(&g_state_mutex);
	if (!chain || !chain->table.vkEnumeratePhysicalDevices)
		return VK_ERROR_INITIALIZATION_FAILED;
	VkResult const res = chain->table.vkEnumeratePhysicalDevices(instance, pCount, pPhysicalDevices);
	// VK_INCOMPLETE still wrote *pCount devices, which a game that asks for fewer than there are uses.
	if ((res != VK_SUCCESS && res != VK_INCOMPLETE) || !pPhysicalDevices)
		return res;
	pthread_mutex_lock(&g_state_mutex);
	bool const remembered = remember_physical(chain, pPhysicalDevices, *pCount);
	pthread_mutex_unlock(&g_state_mutex);
	if (!remembered)
		log_printf("[layer] vkEnumeratePhysicalDevices: out of host memory; a device created on these "
		           "physical devices before a later enumeration records them presents untouched");
	return res;
}

// ---------------------------------------------------------------------------
// Device hooks
// ---------------------------------------------------------------------------

/** @brief The loader's vkSetDeviceLoaderData in a device's create info.
 *
 * A second node in the same chain as the layer's link carries it. Every dispatchable object this
 * layer allocates has to be passed through it; see set_loader_data() for why.
 *
 * @param info The game's create info.
 * @return     The function, or nullptr.
 */
static PFN_vkSetDeviceLoaderData
loader_data_callback (VkDeviceCreateInfo const *info)
{
	for (VkLayerDeviceCreateInfo const *n = info->pNext; n; n = n->pNext)
		if (n->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
		    && n->function == VK_LOADER_DATA_CALLBACK)
			return n->u.pfnSetDeviceLoaderData;
	return nullptr;
}

/** @brief The device extensions that the layer adds when the device offers them and the game did not
 *         enable them.
 *
 * VK_KHR_external_memory_fd is what vkGetMemoryFdKHR needs to export the device-local transport, so
 * the proxy and the model's answer never pass through host memory. It is a device extension and the
 * application decides what the device enables, but a layer may add to that list on the way down --
 * and does, when the pass is on, the device offers it, and the app did not already enable it. If any
 * of that is false the composition stages frames through host memory.
 */
static char const *const WANT_EXTS[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME };

/** @brief The number of WANT_EXTS. */
static constexpr size_t WANT_COUNT = sizeof WANT_EXTS / sizeof *WANT_EXTS;

/** @brief Which of WANT_EXTS an extension is.
 *
 * @param name The extension's name.
 * @return     Its index in WANT_EXTS, or WANT_COUNT if it is none of them.
 */
static size_t
wanted_extension (char const *name)
{
	size_t k = 0;
	while (k < WANT_COUNT && strcmp(name, WANT_EXTS[k]))
		++k;
	return k;
}

/** @brief Adds the WANT_EXTS that the device offers and the game did not enable to a device's
 *         extensions.
 *
 * The added ones go behind the game's, which are copied in front once there are any.
 *
 * @param ic         The instance's chain.
 * @param physical   The physical device.
 * @param info       The game's create info.
 * @param modified   A copy of @a info, which names @a extensions if any were added.
 * @param extensions Room for the game's extensions and WANT_COUNT more.
 * @return           true if any were added.
 */
static bool
add_wanted_extensions (struct instance_chain const *ic,
                       VkPhysicalDevice             physical,
                       VkDeviceCreateInfo const    *info,
                       VkDeviceCreateInfo          *modified,
                       char const                 **extensions)
{
	struct instance_table const *const t = &ic->table;
	if (!t->vkEnumerateDeviceExtensionProperties)
		return false;

	// Indexed by wanted_extension(), whose WANT_COUNT is every other extension.
	bool available[WANT_COUNT + 1] = {0};
	bool enabled[WANT_COUNT + 1] = {0};
	// A failed call leaves the count undefined: nothing is added then.
	uint32_t n = 0;
	if (t->vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, nullptr) != VK_SUCCESS)
		return false;
	// One more than counted, so that a device that offers none does not read as running out.
	VkExtensionProperties *avail = calloc((size_t)n + 1, sizeof *avail);
	if (!avail) {
		log_printf("[layer] vkCreateDevice: out of host memory to list the device's extensions; "
		           "adding none for the transport");
	} else if (n && t->vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, avail) == VK_SUCCESS) {
		for (uint32_t i = 0; i < n; ++i)
			available[wanted_extension(avail[i].extensionName)] = true;
	}
	free(avail);
	avail = nullptr;
	for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
		enabled[wanted_extension(info->ppEnabledExtensionNames[i])] = true;

	uint32_t count = info->enabledExtensionCount;
	for (size_t k = 0; k < WANT_COUNT; ++k)
		if (available[k] && !enabled[k])
			extensions[count++] = WANT_EXTS[k];
	if (count == info->enabledExtensionCount)
		return false;

	for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
		extensions[i] = info->ppEnabledExtensionNames[i];
	modified->enabledExtensionCount = count;
	modified->ppEnabledExtensionNames = extensions;
	return true;
}

/** @brief What the in-layer network needs that the game's instance or device lacks.
 *
 * @param ic       The instance's chain.
 * @param physical The physical device.
 * @return         nullptr if the device can be created with the network's requirements.
 */
static char const *
network_unavailable (struct instance_chain const *ic,
                     VkPhysicalDevice             physical)
{
	struct instance_table const *const t = &ic->table;
	return device_features_network_unavailable(physical, ic->api_version,
	                                           t->vkGetPhysicalDeviceProperties2,
	                                           t->vkGetPhysicalDeviceFeatures2,
	                                           t->vkEnumerateDeviceExtensionProperties,
	                                           t->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR);
}

/** @brief Whether the in-layer network was asked for and the game's instance and device allow it.
 *
 * @param ic       The instance's chain.
 * @param physical The physical device.
 * @return         true if the device can be created with the network's requirements; otherwise
 *                 what they need is logged.
 */
static bool
network_available (struct instance_chain const *ic,
                   VkPhysicalDevice             physical)
{
	if (!network_requested())
		return false;

	char const *const off = network_unavailable(ic, physical);
	if (!off)
		return true;
	log_printf("[layer] in-layer network unavailable: needs %s", off);
	return false;
}

/** @brief Adds the features the layer needs to a device's create info.
 *
 * Our write-only storage shaders must accept RGBA8/BGRA8/FP16 views, an optional capability
 * requested explicitly, and the in-layer network, on request, needs its own. The request owns private
 * copies of the game's structures it changes; behind a structure it cannot copy it declines, which
 * disables composition safely.
 *
 * @param ic       The instance's chain, or nullptr.
 * @param physical The physical device.
 * @param features The request, empty.
 * @param modified The create info that the request changes.
 * @param network  Whether to add the in-layer network's features too.
 * @return         true if @a modified asks for formatless storage writes.
 */
static bool
add_features (struct instance_chain const *ic,
              VkPhysicalDevice             physical,
              struct device_features      *features,
              VkDeviceCreateInfo          *modified,
              bool                         network)
{
	VkPhysicalDeviceFeatures supported = {0};
	PFN_vkGetPhysicalDeviceFeatures const query = ic ? ic->table.vkGetPhysicalDeviceFeatures : nullptr;
	if (query)
		query(physical, &supported);
	if (!supported.shaderStorageImageWriteWithoutFormat) {
		log_printf("[layer] formatless storage writes unsupported; compositor disabled");
		return false;
	}
	if (device_features_enable(features, modified, network))
		return true;
	if (network && device_features_enable(features, modified, false)) {
		log_printf("[layer] in-layer network unavailable: cannot safely copy the game's feature chain");
		return true;
	}
	log_printf("[layer] cannot safely clone the Features2 prefix; compositor disabled");
	return false;
}

/** @brief Whether a device's create info enables an extension.
 *
 * @param info The create info.
 * @param name The extension's name.
 * @return     true if it does.
 */
static bool
enables_extension (VkDeviceCreateInfo const *info,
                   char const               *name)
{
	for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
		if (!strcmp(info->ppEnabledExtensionNames[i], name))
			return true;
	return false;
}

/** @brief vkCreateDevice: adds what the layer needs to the device, then tracks it. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_create_device (VkPhysicalDevice             physicalDevice,
                    VkDeviceCreateInfo const    *pCreateInfo,
                    VkAllocationCallbacks const *pAllocator,
                    VkDevice                    *pDevice)
{
	VkLayerDeviceCreateInfo *link = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
	while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
	                 && link->function == VK_LAYER_LINK_INFO))
		link = (VkLayerDeviceCreateInfo *)link->pNext;
	if (!link || !link->u.pLayerInfo)
		return VK_ERROR_INITIALIZATION_FAILED;

	PFN_vkGetInstanceProcAddr const next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	PFN_vkGetDeviceProcAddr const next_dpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
	PFN_vkCreateDevice const create = (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
	if (!create)
		return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkSetDeviceLoaderData const set_loader_data = loader_data_callback(pCreateInfo);

	// The layer's state, before the device, so that running out leaves nothing to destroy: the
	// chain, with an entry for each queue the device is created with, and a list with room for the
	// game's extensions and the ones the layer adds, WANT_EXTS and the network's.
	uint32_t queue_count = 0;
	for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i)
		queue_count += pCreateInfo->pQueueCreateInfos[i].queueCount;
	struct device_chain *dc = device_chain_create(queue_count);
	char const **extensions = calloc((size_t)pCreateInfo->enabledExtensionCount + WANT_COUNT
	                                 + NETWORK_FEATURE_COUNT, sizeof *extensions);
	if (!dc || !extensions) {
		device_chain_destroy(&dc);
		free(extensions);
		extensions = nullptr;
		log_printf("[layer] vkCreateDevice: out of host memory for the layer's state");
		return VK_ERROR_OUT_OF_HOST_MEMORY;
	}

	pthread_mutex_lock(&g_state_mutex);
	struct instance_chain *const ic = instance_for_physical(physicalDevice);
	pthread_mutex_unlock(&g_state_mutex);

	bool const enabled = layer_enabled();
	VkDeviceCreateInfo const *effective = pCreateInfo;
	VkDeviceCreateInfo modified = *pCreateInfo;
	if (enabled && ic && add_wanted_extensions(ic, physicalDevice, pCreateInfo, &modified, extensions))
		effective = &modified;
	// The in-layer network, on request, where the game's instance and device allow it.
	bool const network = enabled && ic && network_available(ic, physicalDevice);
	// The request's extensions and the network's, in the layer's list.
	if (network && device_features_add_network_extensions(&modified, extensions))
		effective = &modified;

	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	VkLayerDeviceLink *const next_layer_info = link->u.pLayerInfo;
	// After the link moved on: the request may copy the loader's node.
	struct device_features features = {0};
	if (enabled && add_features(ic, physicalDevice, &features, &modified, network))
		effective = &modified;
	VkResult res = create(physicalDevice, effective, pAllocator, pDevice);
	if (res != VK_SUCCESS && effective != pCreateInfo) {
		// Nothing added here is worth failing a device creation over.
		log_printf("[layer] vkCreateDevice refused the layer's additions (%d); "
		           "retrying with the game's list", (int)res);
		link->u.pLayerInfo = next_layer_info;
		effective = pCreateInfo;
		res = create(physicalDevice, pCreateInfo, pAllocator, pDevice);
	}
	if (res != VK_SUCCESS) {
		device_chain_destroy(&dc);
		free(extensions);
		extensions = nullptr;
		return res;
	}

	dc->instance = ic;
	dc->physical = physicalDevice;
	dc->self = *pDevice;
	dc->set_device_loader_data = set_loader_data;
	dc->table.next_dpa = next_dpa;
	device_table_load(&dc->table, *pDevice);
	if (dc->table.vkGetMemoryFdKHR
	    && enables_extension(effective, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME))
		dc->flags |= DEVICE_CHAIN_EXPORT_MEMORY;
	if (network) {
		bool const ledger = device_features_network_enabled(effective);
		dc->flags |= ledger ? DEVICE_CHAIN_NETWORK : 0;
		log_printf("[layer] in-layer network features %s", ledger ? "enabled" : "not enabled");
	}
	if (!dc->table.vkQueuePresentKHR || !dc->table.vkCreateSwapchainKHR || !ic)
		atomic_store(&dc->inert, true);
	if (enabled && !device_features_has_formatless_storage_writes(effective)) {
		atomic_store(&dc->inert, true);
		log_printf("[layer] shaderStorageImageWriteWithoutFormat not enabled; presenting untouched");
	}
	// The request's last reader is above.
	free(extensions);
	extensions = nullptr;

	// The native HIP worker runs on an AMD GPU, so on anything else there is nothing for this layer
	// to do but cost a round trip.
	// Hybrid machines are the case that matters: an implicit layer is loaded for every device the
	// loader builds, including the integrated one a game may well be running on.
	char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = "?";
	if (ic && ic->table.vkGetPhysicalDeviceProperties) {
		VkPhysicalDeviceProperties props;
		ic->table.vkGetPhysicalDeviceProperties(physicalDevice, &props);
		snprintf(device_name, sizeof device_name, "%s", props.deviceName);
		bool vendor_supported = props.vendorID == 0x1002u;
#ifdef DLSSLOP_TEST_LAVAPIPE
		// Test builds only: exercise the real Vulkan capture/composition/IPC route with Mesa's CPU
		// driver. Production binaries have no vendor override.
		vendor_supported = vendor_supported || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
#endif
		if (!vendor_supported) {
			atomic_store(&dc->inert, true);
			log_printf("[layer] inert on non-AMD device (vendor %#x): %s", props.vendorID, device_name);
		}
	}

	pthread_mutex_lock(&g_state_mutex);
	list_append(&g_devices, &dc->node);
	log_printf("[layer] vkCreateDevice -> %p on %s (inert=%d enabled=%d)", (void *)*pDevice, device_name,
	           (int)atomic_load(&dc->inert), (int)enabled);
	pthread_mutex_unlock(&g_state_mutex);
	return VK_SUCCESS;
}

/** @brief Unlinks a swapchain's state from its device and destroys it; the caller holds dc->lock.
 *
 * @param dc The device's chain.
 * @param sc The state.
 */
static void
drop_swapchain (struct device_chain    *dc,
                struct swapchain_state *sc)
{
	if (dc->in_layer_last == sc)
		dc->in_layer_last = nullptr;
	list_del(&sc->node);
	swapchain_state_destroy(&sc);
}

/** @brief vkDestroyDevice: says the layer has gone, then destroys what it made for the device. */
static VKAPI_ATTR void VKAPI_CALL
hook_destroy_device (VkDevice                     device,
                     VkAllocationCallbacks const *pAllocator)
{
	struct device_chain *dc = find_device(device);
	if (!dc)
		return;

	struct swapchain_state *sc;
	pthread_mutex_lock(&dc->lock);
	list_foreach(sc, &dc->swapchains, struct swapchain_state, node)
		release_primary(device, sc->handle);
	pthread_mutex_unlock(&dc->lock);
	// Say the layer has gone. A reader that finds a pid here checks it is alive, so a crash is caught
	// too, but an orderly exit should not need anyone to go looking.
	if (dc->shm.hdr && atomic_load(&dc->shm.hdr->layerPid) == dc->pid) {
		atomic_store(&dc->shm.hdr->layerPid, 0);
		atomic_store(&dc->shm.hdr->layerCompositionUp, 0);
		// The in-layer network's state goes with it.
		if (dc->network_reason[0])
			ShmStoreString(&dc->shm.hdr->layerReasonSeq, dc->shm.hdr->layerReason, kReasonBytes, "");
	}
	wait_device_idle(dc);
	// While the device is still found by its queues: the network's build submits through them.
	if (dc->in_layer) {
		g_network.close(dc->in_layer);
		dc->in_layer = nullptr;
	}
	pthread_mutex_lock(&g_state_mutex);
	list_del(&dc->node);
	pthread_mutex_unlock(&g_state_mutex);
	pthread_mutex_lock(&dc->lock);
	list_foreach(sc, &dc->swapchains, struct swapchain_state, node)
		drop_swapchain(dc, sc);
	pthread_mutex_unlock(&dc->lock);
	if (dc->table.vkDestroyDevice)
		dc->table.vkDestroyDevice(device, pAllocator);
	device_chain_destroy(&dc);
}

/** @brief remember_queue() without the lock; the caller holds dc->lock.
 *
 * A queue beyond the ones the device was created with, which a valid game cannot take, stays
 * unknown.
 */
static void
remember_queue_ (struct device_chain *dc,
                 VkQueue              queue,
                 uint32_t             family)
{
	struct device_queue *const known = find_queue(dc, queue);
	if (known) {
		known->family = family;
		return;
	}
	if (dc->queue_store_used == dc->queue_store_count)
		return;

	struct device_queue *const q = &dc->queue_store[dc->queue_store_used++];
	q->queue = queue;
	q->family = family;
	list_append(&dc->queue_families, &q->node);
}

/** @brief Records the family of a queue the game took.
 *
 * @param dc     The device's chain.
 * @param queue  The queue, or VK_NULL_HANDLE.
 * @param family Its family.
 */
static void
remember_queue (struct device_chain *dc,
                VkQueue              queue,
                uint32_t             family)
{
	if (!queue)
		return;

	pthread_mutex_lock(&dc->lock);
	remember_queue_(dc, queue, family);
	pthread_mutex_unlock(&dc->lock);
}

/** @brief vkGetDeviceQueue: records the queue's family. */
static VKAPI_ATTR void VKAPI_CALL
hook_get_device_queue (VkDevice  device,
                       uint32_t  family,
                       uint32_t  index,
                       VkQueue  *pQueue)
{
	struct device_chain *const dc = find_device(device);
	if (!dc || !dc->table.vkGetDeviceQueue)
		return;

	dc->table.vkGetDeviceQueue(device, family, index, pQueue);
	remember_queue(dc, *pQueue, family);
}

/** @brief vkGetDeviceQueue2: records the queue's family.
 *
 * The 1.1 way of asking for a queue, and the only way to reach one created with
 * VkDeviceQueueCreateFlags. A game that uses it never registered its queue through the hook above,
 * so the present path could not tell which family the queue belonged to and fell back to family
 * zero -- which is the family the command pool was then created on, and need not be the queue's.
 */
static VKAPI_ATTR void VKAPI_CALL
hook_get_device_queue2 (VkDevice                  device,
                        VkDeviceQueueInfo2 const *pQueueInfo,
                        VkQueue                  *pQueue)
{
	struct device_chain *const dc = find_device(device);
	if (!dc || !dc->table.vkGetDeviceQueue2)
		return;

	dc->table.vkGetDeviceQueue2(device, pQueueInfo, pQueue);
	if (pQueueInfo)
		remember_queue(dc, *pQueue, pQueueInfo->queueFamilyIndex);
}

// ---------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------

/** @brief Whether the pass can work in a swapchain format.
 *
 * Every one of them has a UNORM twin the composition uses internally; the ten-bit and float entries
 * are new here, and are what lets an HDR game reach the model at all -- the encode is exactly the
 * step that turns open-ended light into the kind of picture the model was trained on.
 *
 * @param f The format.
 * @return  true if composition_format() has a twin for it.
 */
static bool
supported_format (VkFormat f)
{
	return composition_format(f) != VK_FORMAT_UNDEFINED;
}

/** @brief What a swapchain's format and colour space together say about the light in the frame.
 *
 * A float swapchain is the easy case: games hand over linear light and the HDR path divides it by the
 * white point and hands the model the result. The ten-bit formats are the ones worth the colour
 * space: on a desktop set to HDR10 they carry ST 2084 code -- absolute nits, which is why the old
 * display-referred reading of them (tone map as if it were SDR) banding-crushed them to eight bits on
 * the way to the model. A ten-bit swapchain in an SDR colour space is just a bit more precision on a
 * tone-mapped frame, and stays on the SDR path.
 *
 * @param f  The format.
 * @param cs The colour space.
 * @return   An enum HdrKind.
 */
static uint32_t
detect_hdr_kind (VkFormat        f,
                 VkColorSpaceKHR cs)
{
	if (f == VK_FORMAT_R16G16B16A16_SFLOAT)
		return kHdrLinearFp16;
	bool const ten_bit = f == VK_FORMAT_A2R10G10B10_UNORM_PACK32 || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32
	                     || f == (VkFormat)1000452000 /* R12G12B12A16_UNORM_PACK32 */;
	bool const pq = cs == VK_COLOR_SPACE_HDR10_ST2084_EXT
	                || cs == (VkColorSpaceKHR)1000459000 /* HDR10_ST2084_COMPATIBLE */;
	if (ten_bit && pq)
		return kHdrPq10;
	// A float swapchain in a linear BT.2020 space is still linear light; nothing else here is HDR.
	return kHdrNone;
}

/** @brief Whether the images of a surface's swapchains can be transfer sources and destinations.
 *
 * @param dc      The device's chain.
 * @param surface The surface.
 * @return        true if the surface's capabilities say so.
 */
static bool
surface_transfers (struct device_chain const *dc,
                   VkSurfaceKHR               surface)
{
	if (!dc->instance || !dc->instance->table.vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
		return false;

	VkImageUsageFlags const required = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	VkSurfaceCapabilitiesKHR capabilities;
	return dc->instance->table.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dc->physical, surface,
	                                                                     &capabilities) == VK_SUCCESS
	       && (capabilities.supportedUsageFlags & required) == required;
}

/** @brief vkCreateSwapchainKHR: asks for the transfers the pass needs, then tracks the swapchain. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_create_swapchain_khr (VkDevice                        device,
                           VkSwapchainCreateInfoKHR const *pCreateInfo,
                           VkAllocationCallbacks const    *pAllocator,
                           VkSwapchainKHR                 *pSwapchain)
{
	struct device_chain *const dc = find_device(device);
	if (!dc || !dc->table.vkCreateSwapchainKHR)
		return VK_ERROR_INITIALIZATION_FAILED;

	// Native HIP supports display-referred SDR, linear scRGB/BT.709 FP16, and PQ/BT.2020 HDR10. Other
	// transfer/primary combinations need their own color conversion and remain pass-through.
	VkFormat const format = pCreateInfo->imageFormat;
	VkColorSpaceKHR const space = pCreateInfo->imageColorSpace;
	uint32_t const hdr_kind = detect_hdr_kind(format, space);
	bool const format_supported = supported_format(format);
	bool const unsupported_hdr = !(space == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
	                               || (format == VK_FORMAT_R16G16B16A16_SFLOAT
	                                   && space == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
	                               || hdr_kind == kHdrPq10);
	bool const active = !atomic_load(&dc->inert) && layer_enabled();
	bool const unsupported_transfer = active && !surface_transfers(dc, pCreateInfo->surface);
	VkSwapchainCreateInfoKHR m = *pCreateInfo;
	if (active && !unsupported_hdr && !unsupported_transfer && format_supported)
		m.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	VkResult const res = dc->table.vkCreateSwapchainKHR(device, &m, pAllocator, pSwapchain);
	if (res != VK_SUCCESS || !active || atomic_load(&dc->inert))
		return res;
	// A failed call leaves the count undefined, and a failed listing the images.
	uint32_t count = 0;
	if (!dc->table.vkGetSwapchainImagesKHR
	    || dc->table.vkGetSwapchainImagesKHR(device, *pSwapchain, &count, nullptr) != VK_SUCCESS) {
		log_printf("[layer] swapchain %#" PRIx64 ": its images cannot be listed; presenting it "
		           "untouched", (uint64_t)*pSwapchain);
		return VK_SUCCESS;
	}
	struct swapchain_state *sc = swapchain_state_create(&dc->table, device, *pSwapchain, count);
	if (!sc) {
		log_printf("[layer] swapchain %#" PRIx64 ": out of host memory for the layer's state; "
		           "presenting it untouched", (uint64_t)*pSwapchain);
		return VK_SUCCESS;
	}
	if (dc->table.vkGetSwapchainImagesKHR(device, *pSwapchain, &count, sc->images) != VK_SUCCESS) {
		log_printf("[layer] swapchain %#" PRIx64 ": its images cannot be listed; presenting it "
		           "untouched", (uint64_t)*pSwapchain);
		swapchain_state_destroy(&sc);
		return VK_SUCCESS;
	}

	sc->format = format;
	sc->hdr_kind = hdr_kind;
	sc->width = pCreateInfo->imageExtent.width;
	sc->height = pCreateInfo->imageExtent.height;
	// Why its frames present untouched, the first reason that holds; empty if they do not.
	char const *const pass_through = unsupported_hdr ? " (unsupported color space for native HIP)"
	                                 : unsupported_transfer ? " (surface cannot transfer frames)"
	                                 : !format_supported ? " (unsupported format)"
	                                 : sc->width < kMinW || sc->height < kMinH ? " (too small)"
	                                 : sc->width > kMaxW || sc->height > kMaxH ? " (too large)"
	                                 : "";
	if (*pass_through)
		sc->flags |= SWAPCHAIN_STATE_PASS_THROUGH;

	pthread_mutex_lock(&dc->lock);
	log_printf("[layer] swapchain %#" PRIx64 " %ux%u fmt=%d hdr=%u passThrough=%d%s",
	           (uint64_t)*pSwapchain, pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
	           (int)format, sc->hdr_kind, (int)(*pass_through != '\0'), pass_through);
	list_append(&dc->swapchains, &sc->node);
	pthread_mutex_unlock(&dc->lock);
	return VK_SUCCESS;
}

/** @brief vkDestroySwapchainKHR: destroys what the layer made for the swapchain, once the device is
 *         idle.
 */
static VKAPI_ATTR void VKAPI_CALL
hook_destroy_swapchain_khr (VkDevice                     device,
                            VkSwapchainKHR               swapchain,
                            VkAllocationCallbacks const *pAllocator)
{
	struct device_chain *const dc = find_device(device);
	if (!dc)
		return;

	release_primary(device, swapchain);
	pthread_mutex_lock(&dc->lock);
	// SC stays valid while the lock is released for the wait: an entry never moves when others are
	// linked or unlinked, and only this hook and hook_destroy_device() free one, which the game may
	// not call for this swapchain or its device while this call runs.
	struct swapchain_state *const sc = find_swapchain(dc, swapchain);
	if (sc) {
		pthread_mutex_unlock(&dc->lock);
		wait_device_idle(dc);
		pthread_mutex_lock(&dc->lock);
		drop_swapchain(dc, sc);
	}
	pthread_mutex_unlock(&dc->lock);
	if (dc->table.vkDestroySwapchainKHR)
		dc->table.vkDestroySwapchainKHR(device, swapchain, pAllocator);
}

// ---------------------------------------------------------------------------
// Present-time neural round-trip
// ---------------------------------------------------------------------------

/** @brief Gives a dispatchable object this layer allocated the dispatch table the loader expects on
 *         it.
 *
 * VkCommandBuffer and VkQueue are dispatchable: their first word points at a dispatch table, and
 * every layer below reads it to find its own state for that object. The loader fills that word in
 * for objects the application allocates through the trampoline -- but a layer that allocates one by
 * calling straight down the chain bypasses the trampoline, so the loader never sees it and the word
 * keeps whatever the ICD left there. The loader hands each layer vkSetDeviceLoaderData precisely so
 * it can do that fill-in itself, and calling it is mandatory, not advisory.
 *
 * Skipping it is invisible with no other layer present: the next call goes straight to the driver,
 * which does not read the word. Add any second layer -- Steam's overlay, MangoHud, validation -- and
 * that layer reads the word, finds the ICD's loader magic instead of a table, and aborts. Validation
 * says so out loud: 'The VkDevice dispatch handle was not found and Validation will crash.'
 *
 * @param dc     The device's chain.
 * @param object The object.
 * @return       true if the object has its table, or there is no loader in the chain to fill it in.
 */
static bool
set_loader_data (struct device_chain *dc,
                 void                *object)
{
	if (!dc->set_device_loader_data)
		return true; // no loader in the chain; nothing to fill in
	return dc->set_device_loader_data(dc->self, object) == VK_SUCCESS;
}

/** @brief Makes the command buffer, the fences, the semaphores and the composition of a swapchain.
 *
 * @param dc     The device's chain.
 * @param sc     The swapchain's state.
 * @param family The present queue's family.
 * @return       true if everything was made; otherwise what was made stays in @a sc until it is
 *               destroyed.
 */
static bool
create_resources (struct device_chain    *dc,
                  struct swapchain_state *sc,
                  uint32_t                family)
{
	VkDevice const d = dc->self;

	// Present-only queues cannot execute the composition's compute dispatches. Proton normally
	// presents on its graphics queue; unusual applications can still present untouched instead of
	// issuing invalid commands.
	VkQueueFamilyProperties families[32];
	uint32_t count = sizeof families / sizeof *families;
	dc->instance->table.vkGetPhysicalDeviceQueueFamilyProperties(dc->physical, &count, families);
	if (family >= count || !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
		log_printf("[layer] present queue family %u cannot execute the compositor", family);
		return false;
	}

	VkCommandPoolCreateInfo const cpci = {
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = family
	};
	VkCommandPool pool;
	if (dc->table.vkCreateCommandPool(d, &cpci, nullptr, &pool) != VK_SUCCESS)
		return false;
	sc->pool = pool;
	VkCommandBufferAllocateInfo const cbai = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = pool,
		.commandBufferCount = 1
	};
	VkCommandBuffer cb;
	if (dc->table.vkAllocateCommandBuffers(d, &cbai, &cb) != VK_SUCCESS)
		return false;
	sc->cb = cb;
	if (!set_loader_data(dc, cb)) {
		log_printf("[layer] vkSetDeviceLoaderData failed for the present command buffer");
		return false;
	}
	VkFenceCreateInfo const fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence;
	if (dc->table.vkCreateFence(d, &fci, nullptr, &fence) != VK_SUCCESS)
		return false;
	sc->fence_leg1 = fence;
	if (dc->table.vkCreateFence(d, &fci, nullptr, &fence) != VK_SUCCESS)
		return false;
	sc->fence_leg2 = fence;
	VkSemaphoreCreateInfo const sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	for (uint32_t i = 0; i < sc->image_count; ++i) {
		VkSemaphore semaphore;
		if (dc->table.vkCreateSemaphore(d, &sci, nullptr, &semaphore) != VK_SUCCESS)
			return false;
		sc->leg2_done[i] = semaphore;
	}

	if (composition_init(&sc->comp, &dc->table, &dc->instance->table, d, dc->physical) != VK_SUCCESS) {
		log_printf("[layer] composition unavailable: %s", composition_reason(&sc->comp));
		composition_fini(&sc->comp);
		return false;
	}
	if (dc->flags & DEVICE_CHAIN_EXPORT_MEMORY)
		composition_enable_export(&sc->comp, family);
	sc->family = family;
	if (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		sc->flags |= SWAPCHAIN_STATE_GRAPHICS;
	return true;
}

/** @brief Looks at a Vulkan result on the present path rather than collapsing it into a bool.
 *
 * A failure here was previously indistinguishable from 'nothing to do': the call returned false, the
 * caller presented the original frame, and the next frame tried exactly the same thing again. That is
 * the right answer for a transient failure and the wrong one for VK_ERROR_DEVICE_LOST, where the
 * device is gone, every subsequent call will fail the same way, and the log fills with it.
 *
 * Losing the device also latches the layer inert, because after that point the fail-open path is the
 * only correct one and it costs nothing to take it directly.
 *
 * @param dc   The device's chain.
 * @param r    The result.
 * @param what The call that returned it.
 * @return     true for VK_SUCCESS.
 */
static bool
note_vk (struct device_chain *dc,
         VkResult             r,
         char const          *what)
{
	if (r == VK_SUCCESS)
		return true;
	if (r == VK_ERROR_DEVICE_LOST) {
		if (!atomic_exchange(&dc->inert, true))
			log_printf("[layer] %s -> DEVICE_LOST; layer inert for this device", what);
		return false;
	}
	static _Atomic(uint32_t) reported;
	if (atomic_fetch_add(&reported, 1) < 8)
		log_printf("[layer] %s -> %d", what, (int)r);
	return false;
}

/** @brief Begins a leg of the present path: its command buffer.
 *
 * A leg's command buffer is begun, then ended and submitted (submit_leg()), and its fence waited on
 * and reset (wait_leg()).
 *
 * @param dc The device's chain.
 * @param cb The command buffer.
 * @return   true if it began.
 */
static bool
begin_leg (struct device_chain *dc,
           VkCommandBuffer      cb)
{
	static constexpr VkCommandBufferBeginInfo ONCE = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	return note_vk(dc, dc->table.vkBeginCommandBuffer(cb, &ONCE), "vkBeginCommandBuffer");
}

/** @brief Ends a leg's command buffer and submits it.
 *
 * @param dc             The device's chain.
 * @param queue          The queue.
 * @param si             The submission, whose first command buffer is the leg's.
 * @param fence          The leg's fence.
 * @param waits_consumed Set if the submission took wait semaphores.
 * @return               true if it was submitted.
 */
static bool
submit_leg (struct device_chain *dc,
            VkQueue              queue,
            VkSubmitInfo const  *si,
            VkFence              fence,
            bool                *waits_consumed)
{
	// A fence wait does not make device writes visible to the host, and the host reads what the legs
	// copy out: the proxy, the meter mirror and the capture pair.
	static constexpr VkMemoryBarrier TO_HOST = {
		.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT
	};
	VkCommandBuffer const cb = si->pCommandBuffers[0];
	dc->table.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
	                               &TO_HOST, 0, nullptr, 0, nullptr);
	if (!note_vk(dc, dc->table.vkEndCommandBuffer(cb), "vkEndCommandBuffer"))
		return false;
	if (!note_vk(dc, dc->table.vkQueueSubmit(queue, 1, si, fence), "vkQueueSubmit"))
		return false;
	if (si->waitSemaphoreCount != 0)
		*waits_consumed = true;
	return true;
}

/** @brief Waits on a leg's fence and resets it.
 *
 * @param dc    The device's chain.
 * @param fence The fence.
 * @return      true if the wait and the reset succeeded.
 */
static bool
wait_leg (struct device_chain *dc,
          VkFence              fence)
{
	VkResult const waited = dc->table.vkWaitForFences(dc->self, 1, &fence, VK_TRUE, UINT64_MAX);
	if (!note_vk(dc, waited, "vkWaitForFences"))
		return false;
	// A fence that is not reset stays signalled, which the leg's next submit may not take.
	return note_vk(dc, dc->table.vkResetFences(dc->self, 1, &fence), "vkResetFences");
}

/** @brief Waits for leg 2: the command buffer and the composed surfaces are free again, and the
 *         capture pair it copied out is written.
 *
 * @param dc The device's chain.
 * @param sc The swapchain's state.
 * @return   true if the wait succeeded.
 */
static bool
collect_leg2 (struct device_chain    *dc,
              struct swapchain_state *sc)
{
	if (!wait_leg(dc, sc->fence_leg2))
		return false;
	sc->flags &= ~SWAPCHAIN_STATE_LEG2_PENDING;
	composition_write_captured_frame(&sc->comp);
	return true;
}

/** @brief Publishes a composed frame for the controllers.
 *
 * @param dc The device's chain.
 * @param sc The swapchain's state.
 * @param ms The time the layer took over the frame.
 */
static void
publish_frame (struct device_chain          *dc,
               struct swapchain_state const *sc,
               double                        ms)
{
	struct ShmHeader *const hdr = dc->shm.hdr;
	if (!hdr)
		return;

	ShmStore64(&hdr->layerFramesLo, &hdr->layerFramesHi, ++dc->frames_composed);
	atomic_store(&hdr->layerWidth, sc->width);
	atomic_store(&hdr->layerHeight, sc->height);
	atomic_store(&hdr->layerFormat, (uint32_t)sc->format);
	atomic_store(&hdr->layerCompositionUp, 1);
	atomic_store(&hdr->layerPid, dc->pid);
	atomic_store(&hdr->layerMsBits, FloatToBits((float)ms));
	atomic_store(&hdr->layerMeasuredWhiteBits, FloatToBits(composition_measured_white_point(&sc->comp)));
	atomic_fetch_add(&hdr->layerHeartbeat, 1);
}

/** @brief Logs one of the in-layer network's lines. */
static void
network_log (char const *line)
{
	log_printf("[network] %s", line);
}

/** @brief The length of the start of a piece that a reason holds.
 *
 * @param reason The reason.
 * @param piece  The piece.
 * @return       The bytes of @a piece, up to its end or the first that differs from @a reason's.
 */
static size_t
common_length (char const *reason,
               char const *piece)
{
	size_t n = 0;
	while (piece[n] && reason[n] == piece[n])
		++n;
	return n;
}

/** @brief Whether a reason holds three pieces one after the other, cut to fit as snprintf() cuts
 *         them.
 *
 * Each byte is read once, up to the first difference; one at the last byte means the reason holds
 * the cut text.
 *
 * @param reason A buffer whose last byte is NUL (snprintf() writes nothing else there).
 * @param size   The buffer's size.
 * @param what   The first piece.
 * @param detail The second piece.
 * @param after  The third piece.
 * @return       true if it holds them.
 */
static bool
reason_is (char const *reason,
           size_t      size,
           char const *what,
           char const *detail,
           char const *after)
{
	char const *const pieces[] = { what, detail, after };
	char const *const end = reason + size - 1;
	for (size_t i = 0; i < sizeof pieces / sizeof *pieces; ++i) {
		size_t const n = common_length(reason, pieces[i]);
		reason += n;
		if (pieces[i][n])
			return reason == end;
	}
	return !*reason;
}

/** @brief Says what the in-layer network is doing, in the channel's layer reason for the controllers.
 *
 * The reason is stored when it changes and logged when it is news. Unchanged, it copies nothing.
 *
 * @param dc     The device's chain.
 * @param what   The reason's first piece.
 * @param detail Its second piece.
 * @param after  Its third piece; the three are cut to the size of dc->network_reason.
 */
static void
network_reason (struct device_chain *dc,
                char const          *what,
                char const          *detail,
                char const          *after)
{
	char *const reason = dc->network_reason;
	if (reason_is(reason, sizeof dc->network_reason, what, detail, after))
		return;

	snprintf(reason, sizeof dc->network_reason, "%s%s%s", what, detail, after);
	log_printf("[layer] %s", reason);
	if (dc->shm.hdr)
		ShmStoreString(&dc->shm.hdr->layerReasonSeq, dc->shm.hdr->layerReason, kReasonBytes, reason);
}

/** @brief Turns the in-layer network off after it failed: frames go to dlsslopd from now on.
 *
 * @param dc The device's chain.
 * @return   false.
 */
static bool
network_failed (struct device_chain *dc)
{
	network_reason(dc, "in-layer network off: ", g_network.error(dc->in_layer),
	               "; frames go to dlsslopd");
	dc->flags |= DEVICE_CHAIN_IN_LAYER_OFF;
	return false;
}

static pthread_once_t g_network_once = PTHREAD_ONCE_INIT; //!< Runs load_network_module().

/** @brief Loads g_network from beside the layer. */
static void
load_network_module (void)
{
	static constexpr char name[] = "libdlsslop-network.so";
	char const *const self = layer_object_path();
	char const *const slash = strrchr(self, '/');
	size_t const directory = slash ? (size_t)(slash - self) + 1 : 0;
	char *path = malloc(directory + sizeof name);
	if (!path) {
		snprintf(g_network.failure, sizeof g_network.failure, "%s: no memory for the module's path", self);
		return;
	}
	memcpy(path, self, directory);
	memcpy(path + directory, name, sizeof name);
	network_module_load(&g_network, path);
	free(path);
	path = nullptr;
}

/** @brief Takes the device's network_submit before a submit of the network's build. */
static void
lock_queue (void *context)
{
	struct device_chain *const dc = context;
	pthread_mutex_lock(&dc->network_submit);
}

/** @brief Releases the device's network_submit after a submit of the network's build. */
static void
unlock_queue (void *context)
{
	struct device_chain *const dc = context;
	pthread_mutex_unlock(&dc->network_submit);
}

/** @brief Opens the in-layer network on a device.
 *
 * @param dc    The device's chain.
 * @param sc    The state of the swapchain whose present queue the network builds through.
 * @param queue The present queue.
 * @return      The network, or nullptr, with the reason published, when it does not open.
 */
static struct DlsslopNetwork *
open_network (struct device_chain          *dc,
              struct swapchain_state const *sc,
              VkQueue                       queue)
{
	pthread_once(&g_network_once, load_network_module);
	if (!g_network.library) {
		network_reason(dc, "in-layer network off, module unavailable: ", g_network.failure,
		               "; frames go to dlsslopd");
		return nullptr;
	}
	struct dlsslop_network_device device = {
		.instance          = dc->instance->self,
		.physical          = dc->physical,
		.device            = dc->self,
		.queue             = queue,
		.family            = sc->family,
		.lock_queue        = lock_queue,
		.unlock_queue      = unlock_queue,
		.context           = dc,
		.physical_dispatch = dc->instance->table.next_gipa,
		.log               = network_log
	};
	dc->instance->table.vkGetPhysicalDeviceMemoryProperties(dc->physical, &device.memory);
	struct DlsslopNetwork *const network = g_network.open(&device);
	if (network)
		log_printf("[layer] in-layer network on the game's device: frames skip dlsslopd");
	else
		network_reason(dc, "in-layer network off: out of memory; frames go to dlsslopd", "", "");
	return network;
}

/** @brief Says that the swapchain presents where the network cannot run: frames go to dlsslopd, and
 *         the network waits for one that presents where it can.
 *
 * @param dc The device's chain.
 * @return   false.
 */
static bool
network_idle (struct device_chain *dc)
{
	network_reason(dc, "in-layer network idle: the swapchain's queue family cannot run it; "
	               "frames go to dlsslopd", "", "");
	return false;
}

/** @brief Whether this frame goes through the in-layer network instead of dlsslopd.
 *
 * It does if the device enabled the network, the network has not failed, and the swapchain presents
 * on a graphics family, the one it opened on once it has. It opens once, for the first such
 * swapchain.
 *
 * @param dc    The device's chain.
 * @param sc    The swapchain's state.
 * @param queue The present queue.
 * @return      true if the frame goes through the network.
 */
static bool
use_in_layer_network (struct device_chain          *dc,
                      struct swapchain_state const *sc,
                      VkQueue                       queue)
{
	if ((dc->flags & (DEVICE_CHAIN_NETWORK | DEVICE_CHAIN_IN_LAYER_OFF)) != DEVICE_CHAIN_NETWORK)
		return false;
	if (dc->in_layer)
		return sc->family == dc->in_layer_family || network_idle(dc);
	if (!(sc->flags & SWAPCHAIN_STATE_GRAPHICS))
		return network_idle(dc);

	dc->in_layer = open_network(dc, sc, queue);
	dc->in_layer_family = sc->family;
	if (dc->in_layer)
		return true;
	dc->flags |= DEVICE_CHAIN_IN_LAYER_OFF;
	return false;
}

/** @brief Passes a swapchain through for good once a leg could not be submitted or waited for.
 *
 * What the composition tracks then need not be what the device ran, and the command buffer and leg
 * 1's fence may still be in use, so nothing records into them again.
 *
 * @param dc The device's chain.
 * @param sc The swapchain's state.
 * @return   false: the game's own frame is presented.
 */
static bool
abandon_swapchain (struct device_chain    *dc,
                   struct swapchain_state *sc)
{
	log_printf("[layer] swapchain %#" PRIx64 ": a leg could not be submitted or waited for; passing this "
	           "swapchain through", (uint64_t)sc->handle);
	sc->flags |= SWAPCHAIN_STATE_PASS_THROUGH;
	release_primary(dc->self, sc->handle);
	return false;
}

/** @brief Submits what a leg recorded before one of its steps failed, and waits for it.
 *
 * What was recorded is submitted as leg 1 alone would be: the composition's state moved with it, and
 * the image is back in PRESENT_SRC_KHR. A submit or a wait that fails passes the swapchain through
 * (abandon_swapchain()).
 *
 * @param dc             The device's chain.
 * @param sc             The swapchain's state.
 * @param queue          The present queue.
 * @param si             The submission.
 * @param waits_consumed Set if the submission took wait semaphores.
 * @param network        Whether the network's frame is among what was recorded; the network is then
 *                       told it was submitted.
 * @return               false: the game's own frame is presented.
 */
static bool
salvage (struct device_chain    *dc,
         struct swapchain_state *sc,
         VkQueue                 queue,
         VkSubmitInfo const     *si,
         bool                   *waits_consumed,
         bool                    network)
{
	if (!submit_leg(dc, queue, si, sc->fence_leg1, waits_consumed))
		return abandon_swapchain(dc, sc);
	if (network)
		g_network.submitted(dc->in_layer);
	if (!wait_leg(dc, sc->fence_leg1))
		return abandon_swapchain(dc, sc);
	return false;
}

/** @brief The frame through the in-layer network: capture, network and composition in one
 *         submission, with the game's waits, and no CPU wait between them.
 *
 * @param dc             The device's chain.
 * @param sc             The swapchain's state.
 * @param queue          The present queue.
 * @param index          The image's index.
 * @param fs             The frame's settings.
 * @param si             The submission, with the game's waits.
 * @param waits_consumed Set if a submission took the waits.
 * @param t0             When the present started (log_now_ms()).
 * @return               true if the image will hold the composed frame. false presents the game's
 *                       own frame: while the network builds, for a rejected setting, or once it
 *                       fails, after which frames go to dlsslopd.
 */
static bool
process_in_layer (struct device_chain                     *dc,
                  struct swapchain_state                  *sc,
                  VkQueue                                  queue,
                  uint32_t                                 index,
                  struct composition_frame_settings const *fs,
                  VkSubmitInfo                            *si,
                  bool                                    *waits_consumed,
                  double                                   t0)
{
	// The previous frame's meter, whose fence the frame waited for.
	composition_consume_meter(&sc->comp);
	// A swapchain that takes over, a resized one, has not waited for its predecessor's last frame,
	// which may still run the network that a build or a reshape at the new shape frees.
	struct swapchain_state *const last = dc->in_layer_last;
	if (last && last != sc && (last->flags & SWAPCHAIN_STATE_LEG2_PENDING) && !collect_leg2(dc, last))
		return false;
	dc->in_layer_last = sc;
	enum dlsslop_network_state const state = g_network.prepare(dc->in_layer, dc->shm.hdr,
	                                                           composition_model_width(&sc->comp),
	                                                           composition_model_height(&sc->comp),
	                                                           composition_hdr_proxy_active(&sc->comp));
	switch (state) {
	case DLSSLOP_NETWORK_READY:
		network_reason(dc, "in-layer network running", "", "");
		break;
	case DLSSLOP_NETWORK_BUILDING:
		network_reason(dc, "in-layer network building; the game presents its own frames", "", "");
		return false;
	case DLSSLOP_NETWORK_REJECTED:
		network_reason(dc, "in-layer network: ", g_network.error(dc->in_layer), "");
		return false;
	default:
		return network_failed(dc);
	}
	VkCommandBuffer const cb = sc->cb;
	if (!begin_leg(dc, cb))
		return false;
	// A failure submits what was recorded (salvage()): the composition's state moved with it.
	if (!composition_record_capture(&sc->comp, cb, sc->images[index], fs))
		return salvage(dc, sc, queue, si, waits_consumed, false);
	if (g_network.record(dc->in_layer, cb, composition_proxy_buffer(&sc->comp),
	                     composition_answer_buffer(&sc->comp), sc->family,
	                     composition_transport_exported(&sc->comp)) != DLSSLOP_NETWORK_READY) {
		network_failed(dc);
		return salvage(dc, sc, queue, si, waits_consumed, false);
	}
	if (!composition_record_compose(&sc->comp, cb, sc->images[index], fs))
		return salvage(dc, sc, queue, si, waits_consumed, true);
	si->signalSemaphoreCount = 1;
	si->pSignalSemaphores = &sc->leg2_done[index];
	if (!submit_leg(dc, queue, si, sc->fence_leg2, waits_consumed))
		return abandon_swapchain(dc, sc);
	// The network's motion history takes in only the frames that reached the queue.
	g_network.submitted(dc->in_layer);
	sc->flags |= SWAPCHAIN_STATE_LEG2_PENDING;
	if (composition_capture_recorded(&sc->comp))
		collect_leg2(dc, sc);
	publish_frame(dc, sc, log_now_ms() - t0);
	return true;
}

/** @brief Makes room for the stage masks of the first submit's waits in a swapchain's wait_stages.
 *
 * The present path reuses the array; it grows only when an unusual caller supplies more waits.
 *
 * @param sc    The swapchain's state.
 * @param count The number of waits.
 * @return      false, logged, when out of host memory.
 */
static bool
reserve_wait_stages (struct swapchain_state *sc,
                     uint32_t                count)
{
	if (sc->wait_stage_count >= count)
		return true;

	VkPipelineStageFlags *const grown = realloc(sc->wait_stages, count * sizeof *grown);
	if (!grown) {
		log_printf("[layer] present: out of host memory for %u wait stages; presenting untouched", count);
		return false;
	}
	for (uint32_t i = sc->wait_stage_count; i < count; ++i)
		grown[i] = VK_PIPELINE_STAGE_TRANSFER_BIT;
	sc->wait_stages = grown;
	sc->wait_stage_count = count;
	return true;
}

/** @brief process_present() without the lock; the caller holds the channel's producer lock.
 *
 * Three steps around one round trip. The pass encodes a proxy of the frame on the GPU, that proxy
 * crosses to the helper and comes back as the model's answer, and the pass composes the answer onto
 * the frame. The proxy's crossing is fenced on the CPU because the model is in another process and
 * there is nothing to wait on but a sequence number; the answer's return is not -- the present waits
 * on leg 2's semaphore, and the fence that covers leg 2 is only waited on at the start of the NEXT
 * frame, where the command buffer and the composed surfaces are reused.
 *
 * The caller's present semaphores are consumed by the first submit, because that submit is the first
 * thing to touch the image. They are therefore unsignalled by the time this returns and must not be
 * handed to vkQueuePresentKHR again; the caller presents with the composed image's semaphore or none.
 * The submit that takes them sets @a waits_consumed, which nothing here clears.
 *
 * Every path out leaves the swapchain image in PRESENT_SRC_KHR, including the ones that give up.
 *
 * @param dc              The device's chain, whose header is mapped.
 * @param sc              The swapchain's state.
 * @param queue           The present queue.
 * @param index           The image's index.
 * @param wait_count      The number of @a wait_semaphores.
 * @param wait_semaphores The present's wait semaphores.
 * @param waits_consumed  Set if a submission took the waits.
 * @return                true if the swapchain image will hold the composed frame: leg 2 was
 *                        submitted and signals sc->leg2_done[index], which the present must wait on.
 */
static bool
process_present_ (struct device_chain    *dc,
                  struct swapchain_state *sc,
                  VkQueue                 queue,
                  uint32_t                index,
                  uint32_t                wait_count,
                  VkSemaphore const      *wait_semaphores,
                  bool                   *waits_consumed)
{
	// A timed-out request still owns the sole input/output slot. Do this BEFORE recording the GPU
	// download, not just before publishing seq_req: the helper may still be uploading those pixels or
	// returning its result. While it finishes, present the game's original image without blocking or
	// overwriting the slot. Do the same while no worker serves (starting, failed or stopped): a
	// request then would only stall this present and outlive the worker.
	struct ShmHeader *const hdr = dc->shm.hdr;
	bool const in_layer = use_in_layer_network(dc, sc, queue);
	if (in_layer) {
		// The raster the network runs at, which dlsslopd would publish.
		struct NativeTier const *const tier = ShmNativeTier(atomic_load(&hdr->nativeTier));
		if (tier) {
			atomic_store(&hdr->nativeModelMaxWidth, tier->width);
			atomic_store(&hdr->nativeModelMaxHeight, tier->height);
		}
	} else if (atomic_load_explicit(&hdr->helperState, memory_order_acquire) != kHelperRunning) {
		// A daemon that serves again gets a fresh offer; its predecessor's imports are gone.
		composition_withdraw_offer(&sc->comp);
		start_worker(&dc->shm);
		return false;
	} else if (atomic_load_explicit(&hdr->seq_req, memory_order_acquire)
	           != atomic_load_explicit(&hdr->seq_resp, memory_order_acquire)) {
		return false;
	}
	VkCommandBuffer const cb = sc->cb;
	bool const time = log_time_enabled();
	double const t0 = log_now_ms();

	// The previous frame's compose, if it is still running, must finish before anything here touches
	// the surfaces it reads or the command buffer it was recorded into. Waiting here rather than at
	// the end of that frame keeps the game thread out of the GPU's way for the whole of the helper's
	// round trip. The capture pair that compose recorded lands with it.
	if ((sc->flags & SWAPCHAIN_STATE_LEG2_PENDING) && !collect_leg2(dc, sc))
		return false;

	struct composition_frame_settings const fs = composition_frame_settings_read(hdr);

	// The HDR decision, made once per frame before anything is sized or encoded.
	//
	// hdr_active is what this process intends; proxyFormat is what the helper actually built, and the
	// float encode is only taken when both agree -- a model that refused the float input leaves the
	// frame on the 8-bit path it has always used. hdr_active doubles as the echo the helper reads, so
	// it builds the float images only for a layer that has said it will feed them.
	uint32_t const hdr_mode = atomic_load(&hdr->hdrMode);
	uint32_t const color_mode = atomic_load(&hdr->colourMode);
	bool const hdr_active = hdr_mode != kHdrOff && (hdr_mode == kHdrForce || sc->hdr_kind != kHdrNone);
	// HIP consumes an encoded picture in either precision; unlike NGX, format support does not
	// require rebuilding its model, so the float proxy follows the intent alone. The per-request
	// hdrEncode field publishes the selected transport before seq_req.
	bool const hdr_proxy = hdr_active;
	bool const linear_hdr = color_mode == kColourLinearHdr
	                        || (color_mode == kColourAuto && sc->hdr_kind != kHdrNone);
	uint32_t const hdr_transfer = linear_hdr && sc->hdr_kind == kHdrPq10 ? 1u : 0u;

	if (!composition_prepare(&sc->comp, sc->width, sc->height, sc->format, &fs, linear_hdr, hdr_proxy,
	                         hdr_transfer)) {
		log_printf("[layer] composition cannot run here: %s", composition_reason(&sc->comp));
		return false;
	}

	atomic_store(&hdr->hdrDetected, sc->hdr_kind);
	// The intent, not the format-gated decision: the helper builds the float images only for a layer
	// that has said it will feed them, and that handshake has to start while the proxy is still 8-bit.
	// Publishing composition_hdr_proxy_active() here would wait on proxyFormat, which waits on this
	// field, and neither would ever move.
	atomic_store(&hdr->hdrActive, hdr_active ? 1u : 0u);

	// A capture is asked for by writing a frame count into the header; taking it clears the request,
	// so one press produces one run rather than one per frame for as long as nobody clears it. A
	// request published after this frame's settings snapshot belongs to the next frame.
	if (atomic_load(&hdr->controlSeq) == fs.control_seq) {
		uint32_t const frames = atomic_exchange(&hdr->captureRequest, 0);
		if (frames > 0)
			composition_request_capture(&sc->comp, frames, fs.control_seq);
	}

	VkSubmitInfo si = {
		.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers    = &cb
	};

	// Only the first submit waits: the later one is ordered behind it on the same queue and fenced
	// besides, and a binary semaphore may be waited on once per signal.
	if (!reserve_wait_stages(sc, wait_count))
		return false;
	si.waitSemaphoreCount = wait_count;
	si.pWaitSemaphores = wait_count ? wait_semaphores : nullptr;
	si.pWaitDstStageMask = wait_count ? sc->wait_stages : nullptr;

	if (in_layer)
		return process_in_layer(dc, sc, queue, index, &fs, &si, waits_consumed, t0);
	// The daemon imports an exported pair once, before any frame names it. No frame runs on a pending
	// offer: until the answer comes, present the game's own. A refusal stages this swapchain through
	// host memory from then on; a daemon that stops serving first leaves the pair for the next one.
	if (composition_transport_pending(&sc->comp)) {
		switch (offer_transport(&dc->shm, &sc->comp, &sc->offer_expires)) {
		case OFFER_READY:
			composition_set_transport_ready(&sc->comp, true);
			log_printf("[shm] device-local transport ready: frames stay in video memory");
			break;
		case OFFER_DECLINED:
			log_printf("[shm] device-local transport declined; staging frames through host memory");
			composition_disable_export(&sc->comp);
			break;
		case OFFER_WAITING:
			return false;
		case OFFER_LATER:
			composition_withdraw_offer(&sc->comp);
			log_printf("[shm] daemon stopped before it took the transport offer; offering it again later");
			return false;
		}
	}
	uint32_t const transport_gen = composition_transport_generation(&sc->comp);

	// ---- leg 1: the frame the model is shown ----
	if (!begin_leg(dc, cb))
		return false;
	// A failure submits what was recorded (salvage()): the composition's state moved with it.
	if (!composition_record_capture(&sc->comp, cb, sc->images[index], &fs))
		return salvage(dc, sc, queue, &si, waits_consumed, false);
	// This one fence is real: the proxy the helper is about to read is written by these commands, and
	// the sequence number must not outrun the pixels it announces.
	if (!submit_leg(dc, queue, &si, sc->fence_leg1, waits_consumed) || !wait_leg(dc, sc->fence_leg1))
		return abandon_swapchain(dc, sc);
	si.waitSemaphoreCount = 0;
	si.pWaitSemaphores = nullptr;
	si.pWaitDstStageMask = nullptr;
	composition_consume_meter(&sc->comp);
	double const t_capture = time ? log_now_ms() : 0.0;

	// ---- the round trip ----
	if (!shm_map_process_frame(&dc->shm, composition_model_width(&sc->comp),
	                           composition_model_height(&sc->comp), composition_model_bytes(&sc->comp),
	                           composition_proxy_pixels(&sc->comp), composition_model_pixels(&sc->comp),
	                           composition_hdr_proxy_active(&sc->comp), transport_gen)) {
		// A restarted worker holds no imports: offer the pair again on the next frame.
		if (transport_gen && atomic_load(&hdr->transportMiss) == transport_gen)
			composition_set_transport_ready(&sc->comp, false);
		// Fail-open. Leg 1 already put the image back in PRESENT_SRC_KHR, so the original frame is
		// what gets presented and nothing else is owed.
		return false;
	}
	composition_set_capture_inference(&sc->comp, atomic_load(&hdr->seq_req));
	double const t_helper = time ? log_now_ms() : 0.0;

	// ---- leg 2: the answer, composed back ----
	if (!begin_leg(dc, cb))
		return false;
	// The submission holds no waits or signal by now: leg 1 took the waits.
	if (!composition_record_compose(&sc->comp, cb, sc->images[index], &fs))
		return salvage(dc, sc, queue, &si, waits_consumed, false);
	// No CPU wait. The present waits on this semaphore, so the image is composed before it is shown
	// without the CPU ever parking here; the fence is collected at the top of the next frame, where
	// the reused surfaces actually need it.
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores = &sc->leg2_done[index];
	if (!submit_leg(dc, queue, &si, sc->fence_leg2, waits_consumed))
		return abandon_swapchain(dc, sc);
	sc->flags |= SWAPCHAIN_STATE_LEG2_PENDING;
	// A diagnostic request must finish even if a paused application presents no subsequent frame.
	// Frames without a recorded pair keep the asynchronous leg-2 path, including every frame of a
	// capture whose buffer failed. The semaphore is signalled either way, so a failed wait still
	// presents.
	if (composition_capture_recorded(&sc->comp))
		collect_leg2(dc, sc);
	double const t_return = log_now_ms();
	publish_frame(dc, sc, t_return - t0);

	// Every device's frames, which presents on other threads count too.
	static _Atomic(uint32_t) frame_no;
	if (time && (atomic_fetch_add(&frame_no, 1) + 1) % log_time_interval() == 0)
		log_printf("[time] encode=%.2f helper=%.2f resolve=%.2f total=%.2f ms (model %ux%u)",
		           t_capture - t0, t_helper - t_capture, t_return - t_helper, t_return - t0,
		           composition_model_width(&sc->comp), composition_model_height(&sc->comp));
	return true;
}

/** @brief process_present_() under the channel's producer lock, which the frame holds from the first
 *         look at the channel to the last.
 *
 * @return false, presenting the game's own frame, while another process holds the lock; otherwise
 *         what process_present_() returns.
 */
static bool
process_present (struct device_chain    *dc,
                 struct swapchain_state *sc,
                 VkQueue                 queue,
                 uint32_t                index,
                 uint32_t                wait_count,
                 VkSemaphore const      *wait_semaphores,
                 bool                   *waits_consumed)
{
	if (!shm_map_lock_producer(&dc->shm))
		return false;

	bool const composed = process_present_(dc, sc, queue, index, wait_count, wait_semaphores,
	                                       waits_consumed);
	if (flock(dc->shm.producer_fd, LOCK_UN)) {
		// Closing the descriptor drops the lock; the next frame opens it again.
		log_printf("[shm] cannot release the producer lock (%s); closing it", strerror(errno));
		close(dc->shm.producer_fd);
		dc->shm.producer_fd = -1;
	}
	return composed;
}

/** @brief Makes a swapchain's resources, or passes the swapchain through for good if they cannot be
 *         made.
 *
 * @param dc     The device's chain.
 * @param sc     The swapchain's state.
 * @param family The present queue's family.
 * @return       true if the swapchain is ready to compose.
 */
static bool
ready_swapchain (struct device_chain    *dc,
                 struct swapchain_state *sc,
                 uint32_t                family)
{
	if (create_resources(dc, sc, family)) {
		sc->flags |= SWAPCHAIN_STATE_READY;
		return true;
	}
	log_printf("[layer] staging resources failed for swapchain %#" PRIx64 " (%ux%u, family %u); "
	           "passing this swapchain through", (uint64_t)sc->handle, sc->width, sc->height, family);
	sc->flags |= SWAPCHAIN_STATE_PASS_THROUGH;
	// This swapchain claimed the primary role and just gave it up. Without the release the claim
	// would sit on a swapchain that never drives the channel, and no peer of equal or smaller area
	// could take it over.
	release_primary(dc->self, sc->handle);
	return false;
}

/** @brief One swapchain of a present: composes its image if it drives the channel.
 *
 * @param dc             The device's chain; the caller holds dc->lock.
 * @param queue          The present queue.
 * @param info           The present.
 * @param i              The swapchain's index in @a info.
 * @param family         The present queue's family.
 * @param composed_sem   The semaphore that a composed image of the present signals, or
 *                       VK_NULL_HANDLE; set if this swapchain's image is composed.
 * @param waits_consumed Set if a submission took the present's waits.
 */
static void
present_swapchain (struct device_chain    *dc,
                   VkQueue                 queue,
                   VkPresentInfoKHR const *info,
                   uint32_t                i,
                   uint32_t                family,
                   VkSemaphore            *composed_sem,
                   bool                   *waits_consumed)
{
	struct swapchain_state *const sc = find_swapchain(dc, info->pSwapchains[i]);
	uint32_t const index = info->pImageIndices[i];
	if (!sc || (sc->flags & SWAPCHAIN_STATE_PASS_THROUGH) || index >= sc->image_count)
		return;
	// One swapchain drives the channel; the rest present raw. See claim_primary().
	if (!claim_primary(dc->self, sc->handle, sc->width, sc->height)) {
		static _Atomic(uint32_t) n;
		if (atomic_fetch_add(&n, 1) < 3)
			log_printf("[layer] present: swapchain %#" PRIx64 " is not primary",
			           (uint64_t)sc->handle);
		return;
	}
	// One composition, and one semaphore, per present. A larger swapchain that has just taken the
	// primary role over composes from its next present.
	if (*composed_sem)
		return;
	if (dc->shm.flags & SHM_MAP_DEAD) {
		release_primary(dc->self, sc->handle);
		return;
	}
	if (!(sc->flags & SWAPCHAIN_STATE_READY) && !ready_swapchain(dc, sc, family))
		return;

	uint32_t const wait_count = *waits_consumed ? 0u : info->waitSemaphoreCount;
	bool const composed = process_present(dc, sc, queue, index, wait_count, info->pWaitSemaphores,
	                                      waits_consumed);
	if (log_verbose())
		log_printf("[present] swapchain=%#" PRIx64 " image=%u seq=%u composed=%d", (uint64_t)sc->handle,
		           index, dc->shm.hdr ? atomic_load(&dc->shm.hdr->seq_req) : 0u, (int)composed);
	// On failure before the capture submit, the application's waits stay attached to the original
	// present. Once our first submit accepted them they have been consumed.
	if (!composed) {
		++dc->frames_passed_through;
		return;
	}
	*composed_sem = sc->leg2_done[index];
}

/** @brief hook_queue_present_khr() without the lock; the caller holds dc->lock. */
static VkResult
hook_queue_present_khr_ (struct device_chain    *dc,
                         VkQueue                 queue,
                         VkPresentInfoKHR const *pPresentInfo)
{
	poll_hotkeys(dc);
	if (!shm_map_neural_enabled(&dc->shm))
		return dc->table.vkQueuePresentKHR(queue, pPresentInfo);

	// Whether this call's wait semaphores have already been consumed by a submit of ours. They are
	// handed to the first swapchain we actually process; every path after that presents without
	// them, because a semaphore signalled once may only be waited on once. Presenting with them a
	// second time is a wait that never completes -- which is what a second layer in the chain, Steam's
	// overlay among them, turns from a latent bug into a hang.
	bool waits_consumed = false;
	// Signalled by the composition's last submit; the present waits on it.
	VkSemaphore composed_sem = VK_NULL_HANDLE;

	struct device_queue const *const present_queue = find_queue(dc, queue);
	uint32_t const family = present_queue ? present_queue->family : 0;
	for (uint32_t i = 0; i < pPresentInfo->swapchainCount; ++i)
		present_swapchain(dc, queue, pPresentInfo, i, family, &composed_sem, &waits_consumed);
	// Every device's presents, which other threads make too.
	static _Atomic(uint32_t) frame_no;
	if (log_time_enabled() && (atomic_fetch_add(&frame_no, 1) + 1) % log_time_interval() == 0)
		log_printf("[layer] composed=%llu passed through=%llu", (unsigned long long)dc->frames_composed,
		           (unsigned long long)dc->frames_passed_through);

	if (!waits_consumed && !composed_sem)
		return dc->table.vkQueuePresentKHR(queue, pPresentInfo);

	// pNext is carried through untouched: present ids, present timing and the rest belong to the
	// caller and none of them are about semaphores.
	VkPresentInfoKHR pi = *pPresentInfo;
	pi.waitSemaphoreCount = composed_sem ? 1u : 0u;
	pi.pWaitSemaphores = &composed_sem;
	return dc->table.vkQueuePresentKHR(queue, &pi);
}

/** @brief vkQueuePresentKHR: composes the frame through the neural round trip, then presents it.
 *
 * dc->lock is held from the first look at the swapchains to the present at the end, alongside the
 * queue hooks. Vulkan requires external synchronization for every operation on a queue; the in-layer
 * network's build submits to the application's queue from a thread of its own, so the present must
 * not overlap it.
 */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_queue_present_khr (VkQueue                 queue,
                        VkPresentInfoKHR const *pPresentInfo)
{
	struct device_chain *const dc = device_for_queue(queue);
	if (!dc || !dc->table.vkQueuePresentKHR)
		return VK_ERROR_INITIALIZATION_FAILED;
	if (atomic_load(&dc->inert) || !layer_enabled())
		return dc->table.vkQueuePresentKHR(queue, pPresentInfo);

	pthread_mutex_lock(&dc->lock);
	VkResult const res = hook_queue_present_khr_(dc, queue, pPresentInfo);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

/** @brief vkQueueSubmit, under dc->lock. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_queue_submit (VkQueue             queue,
                   uint32_t            submitCount,
                   VkSubmitInfo const *pSubmits,
                   VkFence             fence)
{
	struct device_chain *const dc = device_for_queue(queue);
	if (!dc || !dc->table.vkQueueSubmit)
		return VK_ERROR_INITIALIZATION_FAILED;

	pthread_mutex_lock(&dc->lock);
	VkResult const res = dc->table.vkQueueSubmit(queue, submitCount, pSubmits, fence);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

/** @brief vkQueueSubmit2, under dc->lock. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_queue_submit2 (VkQueue              queue,
                    uint32_t             submitCount,
                    VkSubmitInfo2 const *pSubmits,
                    VkFence              fence)
{
	struct device_chain *const dc = device_for_queue(queue);
	if (!dc || !dc->table.vkQueueSubmit2)
		return VK_ERROR_INITIALIZATION_FAILED;

	pthread_mutex_lock(&dc->lock);
	VkResult const res = dc->table.vkQueueSubmit2(queue, submitCount, pSubmits, fence);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

/** @brief vkQueueWaitIdle, under dc->lock. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_queue_wait_idle (VkQueue queue)
{
	struct device_chain *const dc = device_for_queue(queue);
	if (!dc || !dc->table.vkQueueWaitIdle)
		return VK_ERROR_INITIALIZATION_FAILED;

	pthread_mutex_lock(&dc->lock);
	VkResult const res = dc->table.vkQueueWaitIdle(queue);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

/** @brief vkQueueSubmit2KHR, under dc->lock. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_queue_submit2_khr (VkQueue              queue,
                        uint32_t             submitCount,
                        VkSubmitInfo2 const *pSubmits,
                        VkFence              fence)
{
	struct device_chain *const dc = device_for_queue(queue);
	if (!dc || !dc->table.vkQueueSubmit2KHR)
		return VK_ERROR_INITIALIZATION_FAILED;

	pthread_mutex_lock(&dc->lock);
	VkResult const res = dc->table.vkQueueSubmit2KHR(queue, submitCount, pSubmits, fence);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

/** @brief vkQueueBindSparse, under dc->lock. */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_queue_bind_sparse (VkQueue                 queue,
                        uint32_t                bindInfoCount,
                        VkBindSparseInfo const *pBindInfo,
                        VkFence                 fence)
{
	struct device_chain *const dc = device_for_queue(queue);
	if (!dc || !dc->table.vkQueueBindSparse)
		return VK_ERROR_INITIALIZATION_FAILED;

	pthread_mutex_lock(&dc->lock);
	VkResult const res = dc->table.vkQueueBindSparse(queue, bindInfoCount, pBindInfo, fence);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

/** @brief vkDeviceWaitIdle, under dc->lock.
 *
 * A device wait may not overlap a submit to any of its queues, and the network's build submits under
 * the lock.
 */
static VKAPI_ATTR VkResult VKAPI_CALL
hook_device_wait_idle (VkDevice device)
{
	struct device_chain *const dc = find_device(device);
	if (!dc || !dc->table.vkDeviceWaitIdle)
		return VK_ERROR_INITIALIZATION_FAILED;

	pthread_mutex_lock(&dc->lock);
	VkResult const res = dc->table.vkDeviceWaitIdle(device);
	pthread_mutex_unlock(&dc->lock);
	return res;
}

// ---------------------------------------------------------------------------
// Loader entry points
// ---------------------------------------------------------------------------

// vkNegotiateLoaderLayerInterfaceVersion() hands these two to the loader; layer/dlssnr.map exports
// all five entry points.
extern VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr (VkInstance  instance,
                       char const *pName);

extern VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr (VkDevice    device,
                     char const *pName);

/** @brief Agrees on the loader's layer interface, at most version 7, and hands the loader the
 *         layer's vkGetInstanceProcAddr and vkGetDeviceProcAddr.
 *
 * Announces the layer in the log once per process, not once per negotiate. The loader re-enumerates
 * the implicit layer directory many times during a single instance creation -- 628 times for one
 * 32-bit vkCreateInstance here, and the same for every other manifest in the directory -- and loads
 * this library on each pass. That is the loader's business, but announcing it each time turned one
 * line into 627 in the user's log. Every other layer stays quiet because none of them log from here.
 *
 * @param v The negotiation.
 * @return  VK_SUCCESS, or VK_ERROR_INITIALIZATION_FAILED for a structure that is not one.
 */
VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion (VkNegotiateLayerInterface *v)
{
	if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
		return VK_ERROR_INITIALIZATION_FAILED;
	if (v->loaderLayerInterfaceVersion > 7)
		v->loaderLayerInterfaceVersion = 7;
	if (v->loaderLayerInterfaceVersion >= 2) {
		v->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
		v->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
		v->pfnGetPhysicalDeviceProcAddr = nullptr;
	}
	static atomic_flag announced = ATOMIC_FLAG_INIT;
	if (layer_enabled() && !atomic_flag_test_and_set(&announced)) {
		char const *const mixed = getenv("VKLayer_DLSS5");
		char const *const upper = getenv("VKLAYER_DLSS5");
		char const *const value = mixed ? mixed : upper;
		log_printf("=== %s loaded (VKLayer_DLSS5=%s) ===", VK_LAYER_NAME, value ? value : "(unset)");
	}
	return VK_SUCCESS;
}

/** @brief The layer's one layer: its name, "DLSS Linux Open Proxy for AMD", spec version 1.3.0 and
 *         implementation version 1.
 *
 * @param pCount      Receives the layers written: 1, or 0 without room in @a pProperties; 1 if
 *                    @a pProperties is nullptr. On input, the room in @a pProperties.
 * @param pProperties Receives the layer, or nullptr.
 * @return            VK_SUCCESS, or VK_INCOMPLETE without room.
 */
VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceLayerProperties (uint32_t          *pCount,
                                    VkLayerProperties *pProperties)
{
	if (!pCount)
		return VK_SUCCESS;
	if (!pProperties) {
		*pCount = 1;
		return VK_SUCCESS;
	}
	// No room: none was written, which *pCount already says.
	if (*pCount < 1)
		return VK_INCOMPLETE;
	memset(pProperties, 0, sizeof *pProperties);
	strncpy(pProperties->layerName, VK_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
	strncpy(pProperties->description, "DLSS Linux Open Proxy for AMD", VK_MAX_DESCRIPTION_SIZE - 1);
	pProperties->specVersion = VK_MAKE_VERSION(1, 3, 0);
	pProperties->implementationVersion = 1;
	*pCount = 1;
	return VK_SUCCESS;
}

#undef VK_LAYER_NAME

/** @brief The layer's instance extensions: none.
 *
 * @param pLayerName  Ignored.
 * @param pCount      Receives 0, or nullptr.
 * @param pProperties Ignored.
 * @return            VK_SUCCESS.
 */
VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceExtensionProperties (char const            *pLayerName,
                                        uint32_t              *pCount,
                                        VkExtensionProperties *pProperties)
{
	if (pCount)
		*pCount = 0;
	return VK_SUCCESS;
}

/** @brief Which queries answer a hook. */
enum hook_scope {
	HOOK_INSTANCE, //!< vkGetInstanceProcAddr.
	HOOK_DEVICE,   //!< vkGetInstanceProcAddr and vkGetDeviceProcAddr.
	HOOK_QUEUE     //!< Both, while the in-layer network is requested.
};

/** @brief A function the layer answers for. */
struct hook {
	char const         *name;  //!< The function's name.
	PFN_vkVoidFunction  fn;    //!< The layer's function.
	uint64_t            scope; //!< An enum hook_scope.
};

/** @brief The functions the layer answers for.
 *
 * vkGetInstanceProcAddr answers every row, and vkGetDeviceProcAddr the rows from HOOK_DEVICE on. The
 * queue rows are answered only while the in-layer network is requested: its build submits to the
 * game's queue from a thread of its own, and the queue hooks serialize that with the game's queue
 * operations and device waits.
 */
static struct hook const HOOKS[] = {
	{ "vkGetInstanceProcAddr", (PFN_vkVoidFunction)vkGetInstanceProcAddr, HOOK_INSTANCE },
	{ "vkNegotiateLoaderLayerInterfaceVersion", (PFN_vkVoidFunction)vkNegotiateLoaderLayerInterfaceVersion,
	  HOOK_INSTANCE },
	{ "vkEnumerateInstanceLayerProperties", (PFN_vkVoidFunction)vkEnumerateInstanceLayerProperties,
	  HOOK_INSTANCE },
	{ "vkEnumerateInstanceExtensionProperties", (PFN_vkVoidFunction)vkEnumerateInstanceExtensionProperties,
	  HOOK_INSTANCE },
	{ "vkCreateInstance", (PFN_vkVoidFunction)hook_create_instance, HOOK_INSTANCE },
	{ "vkDestroyInstance", (PFN_vkVoidFunction)hook_destroy_instance, HOOK_INSTANCE },
	{ "vkEnumeratePhysicalDevices", (PFN_vkVoidFunction)hook_enumerate_physical_devices, HOOK_INSTANCE },
	{ "vkCreateDevice", (PFN_vkVoidFunction)hook_create_device, HOOK_INSTANCE },
	{ "vkGetDeviceProcAddr", (PFN_vkVoidFunction)vkGetDeviceProcAddr, HOOK_DEVICE },
	{ "vkDestroyDevice", (PFN_vkVoidFunction)hook_destroy_device, HOOK_DEVICE },
	{ "vkGetDeviceQueue", (PFN_vkVoidFunction)hook_get_device_queue, HOOK_DEVICE },
	{ "vkGetDeviceQueue2", (PFN_vkVoidFunction)hook_get_device_queue2, HOOK_DEVICE },
	{ "vkCreateSwapchainKHR", (PFN_vkVoidFunction)hook_create_swapchain_khr, HOOK_DEVICE },
	{ "vkDestroySwapchainKHR", (PFN_vkVoidFunction)hook_destroy_swapchain_khr, HOOK_DEVICE },
	{ "vkQueuePresentKHR", (PFN_vkVoidFunction)hook_queue_present_khr, HOOK_DEVICE },
	{ "vkQueueSubmit", (PFN_vkVoidFunction)hook_queue_submit, HOOK_QUEUE },
	{ "vkQueueSubmit2", (PFN_vkVoidFunction)hook_queue_submit2, HOOK_QUEUE },
	{ "vkQueueWaitIdle", (PFN_vkVoidFunction)hook_queue_wait_idle, HOOK_QUEUE },
	{ "vkQueueSubmit2KHR", (PFN_vkVoidFunction)hook_queue_submit2_khr, HOOK_QUEUE },
	{ "vkQueueBindSparse", (PFN_vkVoidFunction)hook_queue_bind_sparse, HOOK_QUEUE },
	{ "vkDeviceWaitIdle", (PFN_vkVoidFunction)hook_device_wait_idle, HOOK_QUEUE }
};

/** @brief The number of HOOKS. */
static constexpr size_t HOOK_COUNT = sizeof HOOKS / sizeof *HOOKS;

/** @brief The layer's function of a name, for a query that answers the hooks from a scope on.
 *
 * @param name  The name.
 * @param first The first scope that the query answers.
 * @return      The layer's function, or nullptr where the next layer answers.
 */
static PFN_vkVoidFunction
lookup_hook (char const      *name,
             enum hook_scope  first)
{
	for (size_t i = 0; i < HOOK_COUNT; ++i) {
		if (HOOKS[i].scope < first || strcmp(HOOKS[i].name, name))
			continue;
		if (HOOKS[i].scope == HOOK_QUEUE && !network_requested())
			return nullptr;
		return HOOKS[i].fn;
	}
	return nullptr;
}

/** @brief The next layer's function of a name for an instance; the caller holds g_state_mutex.
 *
 * @param instance The instance.
 * @param name     The name.
 * @return         The function, or nullptr for an instance the layer does not know.
 */
static PFN_vkVoidFunction
next_instance_proc_addr (VkInstance  instance,
                         char const *name)
{
	struct instance_chain const *const ic = find_instance(instance);
	if (!ic || !ic->table.next_gipa)
		return nullptr;
	return ic->table.next_gipa(instance, name);
}

/** @brief The next layer's function of a name for a device; the caller holds g_state_mutex.
 *
 * @param device The device.
 * @param name   The name.
 * @return       The function, or nullptr for a device the layer does not know.
 */
static PFN_vkVoidFunction
next_device_proc_addr (VkDevice    device,
                       char const *name)
{
	struct device_chain const *const dc = find_device_(device);
	if (!dc || !dc->table.next_dpa)
		return nullptr;
	return dc->table.next_dpa(device, name);
}

/** @brief The layer's function of a name, or the next layer's for the instance.
 *
 * @param instance The instance, or VK_NULL_HANDLE.
 * @param pName    The name.
 * @return         The function, or nullptr.
 */
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr (VkInstance  instance,
                       char const *pName)
{
	if (!pName)
		return nullptr;
	PFN_vkVoidFunction const hook = lookup_hook(pName, HOOK_INSTANCE);
	if (hook)
		return hook;
	if (!instance)
		return nullptr;

	pthread_mutex_lock(&g_state_mutex);
	PFN_vkVoidFunction const next = next_instance_proc_addr(instance, pName);
	pthread_mutex_unlock(&g_state_mutex);
	return next;
}

/** @brief The layer's device-level function of a name, or the next layer's for the device.
 *
 * @param device The device, or VK_NULL_HANDLE.
 * @param pName  The name.
 * @return       The function, or nullptr.
 */
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr (VkDevice    device,
                     char const *pName)
{
	if (!pName)
		return nullptr;
	PFN_vkVoidFunction const hook = lookup_hook(pName, HOOK_DEVICE);
	if (hook)
		return hook;
	if (!device)
		return nullptr;

	pthread_mutex_lock(&g_state_mutex);
	PFN_vkVoidFunction const next = next_device_proc_addr(device, pName);
	pthread_mutex_unlock(&g_state_mutex);
	return next;
}
