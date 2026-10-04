/** @file
 *
 * The channel's functions and its table of native neural rasters, and the layout of its header and
 * of a transport offer, pinned.
 */
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "shm_protocol.h"

static_assert(sizeof (struct ShmHeader) <= kHeaderBytes, "ShmHeader outgrew its region");

// The words, pinned. shm_protocol.hpp pins them and the layout below in C++ too.
//
// A word is shared by processes that may have been built from different languages, so its size and
// alignment must not depend on the language. And it must be lock-free: a word that is not goes
// through a lock table local to each process, and is then not atomic across processes. uint32_t has
// the size of unsigned int, for which ATOMIC_INT_LOCK_FREE answers.
static_assert(sizeof (_Atomic(uint32_t)) == 4, "a word of the channel is not four bytes");
static_assert(alignof (_Atomic(uint32_t)) == 4, "a word of the channel is not aligned to four bytes");
static_assert(sizeof (uint32_t) == sizeof (unsigned) && ATOMIC_INT_LOCK_FREE == 2,
              "a word of the channel is not lock-free");

// The frame counts, pinned likewise. A count is one 64-bit atomic, so a reader takes it whole. Its
// alignment of eight bytes starts it at a multiple of eight in the header (184 and 2000 below), as an
// atomic access of eight bytes needs, and it must be lock-free for the reason a word must. uint64_t
// has the size of unsigned long long, for which ATOMIC_LLONG_LOCK_FREE answers.
static_assert(sizeof (_Atomic(uint64_t)) == 8, "a count of the channel is not eight bytes");
static_assert(alignof (_Atomic(uint64_t)) == 8, "a count of the channel is not aligned to eight bytes");
static_assert(sizeof (uint64_t) == sizeof (unsigned long long) && ATOMIC_LLONG_LOCK_FREE == 2,
              "a count of the channel is not lock-free");

// The layout, pinned.
//
// Every process that maps this file agrees on where each field is only because they were compiled
// from the same header. A field inserted anywhere but the end silently moves everything after it, and
// a build that has not caught up then reads its neighbour's value -- which is not a crash, it is a
// status display quietly reporting 4861 for a flag that is 0 or 1, and it took a nonsensical number
// on screen to notice.
//
// The version check already existed to prevent exactly that; what was missing was anything to make
// someone remember to use it. If these fire, the layout changed: bump kShmVersion in the same commit,
// then update these numbers.
//
// Every member is a word, a count, a char array of a multiple of four bytes, or the pass array, and
// the counts start at multiples of eight bytes, so there is no padding: with the words and the
// counts pinned above, these hold in C and C++ alike.
static_assert(sizeof (struct ShmHeader) == 1912, "the header layout changed -- bump kShmVersion");
static_assert(alignof (struct ShmHeader) == 8, "layout changed -- bump kShmVersion");
static_assert(sizeof (struct PassControl) == 36, "layout changed -- bump kShmVersion");

static_assert(offsetof(struct ShmHeader, helperFrames) == 8, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, layerFrames) == 16, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, enabled) == 56, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, transferStrengthBits) == 96, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, helperState) == 180, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, helperReason) == 228, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, layerReason) == 424, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, gameName) == 620, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, pass) == 748, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, mvecEnabled) == 1828, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, seq_ok) == 1836, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, hdrEncode) == 1868, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, transportGen) == 1900, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, nativeTier) == 1908, "layout changed -- bump kShmVersion");

static_assert(sizeof (struct ShmTransportOffer) == 72, "the transport offer's layout changed");

struct NativeTier const kNativeTiers[] = {
	{kNativeTierMin, 1280, 768},
	{kNativeTierMin + kNativeTierStep, 1600, 960},
	{kNativeTierMin + 2 * kNativeTierStep, 1920, 1152},
};

int
shm_runtime_path (char       *buf,
                  size_t      size,
                  char const *uid,
                  char const *file)
{
	if (uid && *uid)
		return snprintf(buf, size, "/tmp/dlssnr-%s%s", uid, file);
	return snprintf(buf, size, "/tmp/dlssnr-%u%s", (unsigned)getuid(), file);
}

int
ShmRuntimeDir (char   *buf,
               size_t  size)
{
	return shm_runtime_path(buf, size, getenv("DLSSNR_UID"), "");
}

int
ShmDefaultPath (char   *buf,
                size_t  size)
{
	return shm_runtime_path(buf, size, getenv("DLSSNR_UID"), "/shm.bin");
}

int
ShmNativeDefaultPath (char   *buf,
                      size_t  size)
{
	return snprintf(buf, size, "/tmp/dlsslop-amd-%u/shm.bin", (unsigned)getuid());
}

int
shm_native_channel_path (char       *buf,
                         size_t      size,
                         char const *channel)
{
	if (channel && *channel)
		return snprintf(buf, size, "%s", channel);
	return ShmNativeDefaultPath(buf, size);
}

int
ShmNativeChannelPath (char   *buf,
                      size_t  size)
{
	return shm_native_channel_path(buf, size, getenv("DLSSNR_SHM"));
}

size_t
ShmTotalBytes (void)
{
	return kHeaderBytes + kMaxFrame * 2;
}

int
ShmTransportPath (char       *buf,
                  size_t      size,
                  char const *channel)
{
	return snprintf(buf, size, "%s%s", channel, kShmTransportSuffix);
}

uint32_t
FloatToBits (float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof u);
	return u;
}

float
BitsToFloat (uint32_t u)
{
	float f;
	memcpy(&f, &u, sizeof f);
	return f;
}

/** @brief Where a text field of the header and its sequence number are. */
struct shm_text_field {
	uint32_t seq;  //!< The sequence number's offset.
	uint32_t text; //!< The field's offset.
	uint32_t size; //!< The field's size.
};

/** @brief The text fields of enum shm_text. */
static struct shm_text_field const shm_text_fields[] = {
#define SHM_TEXT_FIELD(name) \
	{offsetof(struct ShmHeader, name##Seq), offsetof(struct ShmHeader, name), \
	 sizeof ((struct ShmHeader *)nullptr)->name}
	[SHM_TEXT_HELPER_REASON] = SHM_TEXT_FIELD(helperReason),
	[SHM_TEXT_LAYER_REASON]  = SHM_TEXT_FIELD(layerReason),
	[SHM_TEXT_GAME_NAME]     = SHM_TEXT_FIELD(gameName),
#undef SHM_TEXT_FIELD
};

void
ShmStoreString (struct ShmHeader *h,
                enum shm_text     field,
                char const       *src)
{
	struct shm_text_field const *const f = &shm_text_fields[field];
	char *const base = (char *)h;
	memset(base + f->text, 0, f->size);
	if (src)
		strncpy(base + f->text, src, f->size - 1);
	atomic_fetch_add((_Atomic(uint32_t) *)(void *)(base + f->seq), 1);
}

bool
ShmLoadString (struct ShmHeader const *h,
               enum shm_text           field,
               char                   *out,
               size_t                  size)
{
	struct shm_text_field const *const f = &shm_text_fields[field];
	char const *const base = (char const *)h;
	_Atomic(uint32_t) const *const seq = (_Atomic(uint32_t) const *)(void const *)(base + f->seq);
	size_t const length = size < f->size ? size : f->size;
	for (uint32_t attempt = 0; attempt < 4; ++attempt) {
		uint32_t const before = atomic_load(seq);
		memcpy(out, base + f->text, length);
		out[length - 1] = '\0';
		if (atomic_load(seq) == before)
			return true;
	}
	*out = '\0';
	return false;
}

void
ShmInitDefaults (struct ShmHeader *h)
{
	memset(h, 0, sizeof *h);
	atomic_store(&h->magic, kShmMagic);
	atomic_store(&h->version, kShmVersion);
	atomic_store(&h->helperState, kHelperStopped);
	atomic_store(&h->passes, 1);
	atomic_store(&h->enabled, 1);
	atomic_store(&h->autoMask, 1);
	atomic_store(&h->intensityBits, FloatToBits(1.0f));
	atomic_store(&h->localToneBits, FloatToBits(1.0f));
	atomic_store(&h->localStructureBits, FloatToBits(1.0f));
	atomic_store(&h->skinStructureBits, FloatToBits(-1.0f));
	atomic_store(&h->sharpnessBits, FloatToBits(0.0f));

	atomic_store(&h->transferStrengthBits, FloatToBits(1.0f));
	atomic_store(&h->colourStrengthBits, FloatToBits(1.0f));
	atomic_store(&h->maxRatioBits, FloatToBits(2.0f));
	atomic_store(&h->transfer, 1);
	atomic_store(&h->debugScaleBits, FloatToBits(1.0f));
	atomic_store(&h->whitePointBits, FloatToBits(1.0f));
	atomic_store(&h->whitePointScaleBits, FloatToBits(1.0f));
	atomic_store(&h->whitePointTrimBits, FloatToBits(1.0f));
	atomic_store(&h->whitePointSource, kWhitePointManual);
	atomic_store(&h->workingScaleBits, FloatToBits(1.0f));
	atomic_store(&h->compareSplitBits, FloatToBits(0.5f));
	atomic_store(&h->compareZoomBits, FloatToBits(1.0f));
	atomic_store(&h->colourMode, kColourAuto);
	atomic_store(&h->reversibleMode, kReversibleKnee);
	atomic_store(&h->applyModel, 1);
	atomic_store(&h->holdFrame, 0);

	atomic_store(&h->hdrMode, kHdrAuto);
	atomic_store(&h->hdrDetected, kHdrNone);
	atomic_store(&h->hdrActive, 0);
	atomic_store(&h->proxyFormat, kProxyRgba8);

	atomic_store(&h->mvecEnabled, 1);
	atomic_store(&h->mvecQuality, kMVecBalanced);
	atomic_store(&h->mvecPixelSize, kMVecPixels4);
	atomic_store(&h->seq_ok, 0);
	atomic_store(&h->compositionBypass, 1);
	atomic_store(&h->colourTrustPercent, 200);

	atomic_store(&h->ratioSmoothPercent, 100);
	atomic_store(&h->sdr16Multipass, 0);

	for (uint32_t i = 0; i < kMaxPasses; ++i) {
		struct PassControl *pass = &h->pass[i];
		atomic_store(&pass->overrideMask, 0);
		atomic_store(&pass->intensityBits, FloatToBits(1.0f));
		atomic_store(&pass->localToneBits, FloatToBits(1.0f));
		atomic_store(&pass->localStructureBits, FloatToBits(1.0f));
		atomic_store(&pass->skinStructureBits, FloatToBits(-1.0f));
		atomic_store(&pass->sharpnessBits, FloatToBits(0.0f));
		// Inert until overrideMask names them, but initialised to the global defaults so a pass that
		// is switched on later starts from what the rest of the frame is already doing.
		atomic_store(&pass->style, 0);
		atomic_store(&pass->preset, 0);
		atomic_store(&pass->autoMask, 1);
	}
}

void
ShmInitNativeDefaults (struct ShmHeader *h,
                       bool              bypass)
{
	ShmInitDefaults(h);
	atomic_store(&h->hdrMode, kHdrOff);
	atomic_store(&h->mvecEnabled, 0);
	atomic_store(&h->sdr16Multipass, 1);
	atomic_store(&h->passes, kNativeDefaultPasses);
	atomic_store(&h->nativeTier, kNativeDefaultTier);
	atomic_store(&h->transfer, 2);
	atomic_store(&h->compositionBypass, bypass);
}

struct NativeTier const *
ShmNativeTier (uint32_t height)
{
	for (size_t i = 0; i < sizeof kNativeTiers / sizeof *kNativeTiers; ++i)
		if (kNativeTiers[i].height == height)
			return &kNativeTiers[i];
	return nullptr;
}

uint32_t
ShmPassCeiling (struct ShmHeader const *h)
{
	(void)h;
	return kMaxPasses;
}

uint32_t
ShmPasses (struct ShmHeader const *h)
{
	uint32_t p = atomic_load(&h->passes);
	uint32_t ceiling = ShmPassCeiling(h);
	if (p == 0)
		return 1;
	return p > ceiling ? ceiling : p;
}

bool
ShmNeuralEnabled (struct ShmHeader const *h)
{
	return atomic_load(&h->enabled) != 0;
}

uint32_t
ShmMVecQuality (struct ShmHeader const *h)
{
	uint32_t q = atomic_load(&h->mvecQuality);
	return q <= kMVecQuality ? q : kMVecBalanced;
}

uint32_t
ShmMVecPixelSize (struct ShmHeader const *h)
{
	uint32_t size = atomic_load(&h->mvecPixelSize);
	return size <= kMVecPixels8 ? size : kMVecPixels4;
}

