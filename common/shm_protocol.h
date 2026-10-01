/** @file
 *
 * Shared-memory contract between the three processes that make up DLSSNR:
 *
 *   the Linux Vulkan layer   captures the frame, runs the composition, presents the result
 *   the Windows helper       owns nvngx_dlssnr.dll and runs the model
 *   the Qt GUI               writes settings and reads status
 *
 * Everything here is plain atomics in a file mapping, so no side needs the others' toolchain and a
 * process dying leaves the others reading a consistent -- if stale -- picture.
 *
 * Layout of the mapping:
 *
 *   [0, kHeaderBytes)                       ShmHeader
 *   [kHeaderBytes, +kMaxFrame)              the proxy the layer encoded, for the model
 *   [kHeaderBytes + kMaxFrame, +kMaxFrame)  the model's answer, for the composition
 *
 * The proxy is RGBA8 or RGBA16F. The legacy Windows helper publishes its supported
 * format; its HDR proxy carries white-point-normalized linear light. Native HIP
 * uses display-encoded proxies in either precision, including HDR normalized by
 * the layer; hdrEncode is the per-request precision flag. `format` remains 1 for
 * compatibility.
 *
 * C and C++ share this header. A word of the channel is an _Atomic(uint32_t),
 * which <stdatomic.h> makes a std::atomic<uint32_t> in C++, so C++ code reads
 * and writes the words with their load() and store() members. C code, and
 * the functions here, use the atomic_* functions of <stdatomic.h>, which have
 * the same orders. For the functions that return a string in C++, C has
 * versions that write it into the caller's buffer.
 */
#ifndef DLSSLOP_AMD_COMMON_SHM_PROTOCOL_H_
#define DLSSLOP_AMD_COMMON_SHM_PROTOCOL_H_

#ifndef _WIN32
# include <unistd.h>
#endif

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
# include <cstdio>
# include <cstdlib>
# include <cstring>
# include <stdatomic.h>
# include <string>
# define SHM_STD(x) std::x
// Inline, so that every translation unit shares one definition: inline C++ code in other headers
// calls these functions and reads the tier table, and the one-definition rule asks for that.
# define SHM_INLINE inline
# define SHM_CONSTEXPR inline constexpr
extern "C" {
#else
# include <stdatomic.h>
# include <stddef.h>
# include <stdint.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# define SHM_STD(x) x
// Static: C has no inline objects, and an inline function with external linkage would need an
// external definition in one translation unit.
# define SHM_INLINE static inline
# define SHM_CONSTEXPR static constexpr
#endif

/** @brief 'GNR2'.
 *
 * Bumped from the v1 magic on purpose: a stale v1 mapping left in XDG_RUNTIME_DIR must be
 * re-initialised rather than half-read, because the header grew and every offset moved.
 */
static constexpr SHM_STD(uint32_t) kShmMagic = 0x32524E47;

/** @brief The version of the header's layout and of the protocol.
 *
 * v14: this branch and upstream both grew the header, so neither side's number describes it.
 * From upstream: the pixel regions are eight bytes a pixel for the float16 HDR proxy, the header is
 * 64 KiB because VK_EXT_external_memory_host demands the imported pointer meet
 * minImportedHostPointerAlignment and NVIDIA answers 64 KiB, and the dma-buf exchange and HDR
 * and round-trip attribution. A stale mapping of either lineage must be re-created, not half-read.
 * v21-v23: native HIP raster bounds, request-published proxy precision and color anchoring.
 * v24: native device-local transport generations.
 * v25: live native tier requests.
 * v26: offers are answered on their own connection; transportAck is retired.
 * v27: offers carry their buffers' sizes.
 * v28: offers name the exporting device and driver.
 */
static constexpr SHM_STD(uint32_t) kShmVersion = 28;

static constexpr SHM_STD(uint32_t) kMaxW = 7680; //!< The widest frame.
static constexpr SHM_STD(uint32_t) kMaxH = 4320; //!< The tallest frame.

/** @brief The narrowest frame.
 *
 * A game that presents a 1x1 probe swapchain -- NWN:EE does, behind its real window --
 * used to have the model built at that size, and the first submit hung the GPU channel outright
 * (Xid 109, CTX SWITCH TIMEOUT), taking the game down with it. 300x300 is known good; nothing below
 * this is a frame worth composing anyway.
 */
static constexpr SHM_STD(uint32_t) kMinW = 64;
static constexpr SHM_STD(uint32_t) kMinH = 64; //!< The shortest frame. See kMinW.

/** @brief A native neural raster.
 *
 * A tier is the height of the picture the network sees, inside an input padded to networkHeight
 * rows.
 */
struct NativeTier {
	SHM_STD(uint32_t) height;
	SHM_STD(uint32_t) width;
	SHM_STD(uint32_t) networkHeight;
};

/** @brief The native neural rasters, smallest first. */
SHM_CONSTEXPR struct NativeTier kNativeTiers[] = {
	{720, 1280, 768}, {900, 1600, 960}, {1080, 1920, 1152}
};

static constexpr SHM_STD(uint32_t) kNativeDefaultTier = 720; //!< The height of the default tier.
static constexpr SHM_STD(uint32_t) kNativeDefaultPasses = 1; //!< The default pass count.

/** @brief The size of a pixel region.
 *
 * Eight bytes a pixel: the float16 proxy needs them, and the 8-bit path simply uses the first half of
 * each region. The mapping is file-backed and sparse, so an SDR session never commits the second half.
 */
static constexpr SHM_STD(size_t) kMaxFrame = (SHM_STD(size_t))kMaxW * kMaxH * 8;
static constexpr SHM_STD(size_t) kHeaderBytes = 65536; //!< The size of the header's region.

/** @brief The ceiling on how many times the model runs over one frame.
 *
 * It is OptiScaler's DlssNr::kMaxPasses and sizes the per-pass arrays below.
 */
static constexpr SHM_STD(uint32_t) kMaxPasses = 30;

static constexpr SHM_STD(size_t) kReasonBytes = 192; //!< The size of a reason string's field.
static constexpr SHM_STD(size_t) kNameBytes = 128;   //!< The size of the game name's field.

/** @brief Which fields a per-pass entry actually overrides.
 *
 * Sparse by design: a pass with no entry, or an entry that does not name a field, follows the global
 * setting -- which is what makes "pass 3 is gentler" expressible without restating everything else
 * about pass 3.
 */
enum PassOverrideBit : SHM_STD(uint32_t) {
	kOverrideIntensity = 1u << 0,
	kOverrideLocalStructure = 1u << 1,
	kOverrideLocalTone = 1u << 2,
	kOverrideSkinStructure = 1u << 3,
	kOverrideStyle = 1u << 4,
	kOverridePreset = 1u << 5,
	kOverrideAutoMask = 1u << 6,
	kOverrideSharpness = 1u << 7,
};

/** @brief Where the white point comes from.
 *
 * The layer has no game exposure texture to read -- it sees a finished swapchain image and nothing
 * else -- so OptiScaler's source 1 has no counterpart here and the choice is between the slider and
 * the frame's own measurement.
 */
enum WhitePointSource : SHM_STD(uint32_t) {
	kWhitePointManual = 0,   //!< the slider, and nothing else
	kWhitePointMeasured = 1, //!< the calibration grid, measured off the untouched copy
};

/** @brief What the swapchain holds.
 *
 * Getting this wrong encodes an encoded frame a second time, which looks washed out and banded --
 * so the default is to decide it from the format rather than to guess.
 */
enum ColourMode : SHM_STD(uint32_t) {
	kColourAuto = 0,      //!< 8-bit: display-referred. 10-bit and float: linear HDR.
	kColourDisplay = 1,   //!< force display-referred; the encode becomes a pass-through
	kColourLinearHdr = 2, //!< force linear HDR; the encode scales by the white point and sRGB-encodes
};

/** @brief The RenoDX reversible proxy.
 *
 * Which encode the model is shown, and how its answer comes back.
 *
 *   0  soft knee + our composition            the shipped behaviour, and byte-identical to it
 *   1  unclipped Neutwo proxy + composition
 *   2  Neutwo proxy + pure-inverse replace    the model's answer straight back, no composition
 *   3  hybrid proxy + composition             identity midtones, unclipped highlights
 *   4  hybrid proxy + replace
 *
 * From Dagherbou/OptiScaler_DLSSNR; the proxy itself is RenoDX's (clshortfuse). Modes 2 and 4 are
 * noted upstream as flashing on bright lights, which is why 0 is the default rather than a taste.
 */
enum ReversibleMode : SHM_STD(uint32_t) {
	kReversibleKnee = 0,
	kReversibleNeutwo = 1,
	kReversibleNeutwoReplace = 2,
	kReversibleHybrid = 3,
	kReversibleHybridReplace = 4,
	kReversibleModeCount = 5,
};

/** @brief The filter that brings the model's answer back down when it ran above native resolution.
 *
 * Only consulted when the working scale is above 1.0.
 *
 * These are OptiScaler's own Scaler numbers, kept identical so a value copied from an OptiScaler
 * profile means the same thing here. FSR1 keeps slot 0 for that reason even though this pass cannot
 * use it -- it wants a different constant block, and it is an upscaler rather than the averaging
 * filter the down-leg needs. A header asking for it falls back to Lanczos3.
 */
enum Downscaler : SHM_STD(uint32_t) {
	kDownscaleFsr1 = 0,  //!< unsupported here; reserved so the numbering matches upstream
	kDownscaleBicubic = 1,
	kDownscaleCatmullRom = 2,
	kDownscaleLanczos2 = 3,
	kDownscaleLanczos3 = 4,  //!< upstream's default: the sharp one
	kDownscaleKaiser2 = 5,
	kDownscaleKaiser3 = 6,
	kDownscaleMagic = 7,
	kDownscalerCount = 8,
};

/** @brief What the helper has managed to do, for the GUI and for the layer's fail-open decision.
 *
 * The layer reads this to decide whether anything is listening, so "nobody" has to be the value a
 * freshly initialised header holds -- not a state that also means "starting".
 */
enum HelperState : SHM_STD(uint32_t) {
	kHelperStarting = 0,
	kHelperNoVulkan = 1,   //!< no NVIDIA device with the NVX extensions
	kHelperNoBinaries = 2, //!< nvngx_dlssnr.dll not found
	kHelperModelFailed = 3,
	kHelperRunning = 4,
	kHelperStopped = 5,
};

/** @brief Proxy precision policy.
 *
 * The legacy Windows helper must also confirm float16
 * support. Native HIP supports both precisions; colourMode independently decides
 * whether the layer normalizes linear-light/HDR content before proxy encoding.
 */
enum HdrMode : SHM_STD(uint32_t) {
	kHdrAuto = 0,   //!< HDR when the swapchain is HDR
	kHdrOff = 1,    //!< always the 8-bit proxy, whatever the swapchain
	kHdrForce = 2,  //!< float16 proxy even for an SDR swapchain (an A/B tool, not a preference)
};

/** @brief What the swapchain's format and colour space hold. */
enum HdrKind : SHM_STD(uint32_t) {
	kHdrNone = 0,       //!< 8-bit swapchain: already tone mapped
	kHdrLinearFp16 = 1, //!< R16G16B16A16_SFLOAT: linear light, open range
	kHdrPq10 = 2,       //!< 10-bit with a PQ/BT.2020 colour space: ST 2084 code
};

/** @brief What the crossing images actually are, published by the helper.
 *
 * The model decides, and the layer encodes to match rather than to hope.
 */
enum ProxyFormat : SHM_STD(uint32_t) {
	kProxyUnknown = 0,
	kProxyRgba8 = 1,
	kProxyRgba16F = 2,
};

/** @brief How the motion field the helper hands the model is scaled.
 *
 * From bmitch87's motion-vector work.
 */
enum MVecScaleMode : SHM_STD(uint32_t) {
	kMVecNormalized = 0,
	kMVecPixels = 1,
	kMVecUv01 = 2,
};

/** @brief What the optical-flow engine is asked for.
 *
 * Higher costs more of the frame's budget.
 */
enum MVecQuality : SHM_STD(uint32_t) {
	kMVecFast = 0,
	kMVecBalanced = 1,
	kMVecQuality = 2,
};

/** @brief The motion grid's spacing. */
enum MVecPixelSize : SHM_STD(uint32_t) {
	kMVecPixels1 = 0,
	kMVecPixels2 = 1,
	kMVecPixels4 = 2,
	kMVecPixels8 = 3,
};

/** @brief Writes a path in the shared runtime directory: the directory, then @a file.
 *
 * @param buf  Receives the path; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @param uid  The value of DLSSNR_UID, or nullptr if it is not set.
 * @param file What follows the directory.
 * @return     What snprintf() returns.
 */
SHM_INLINE int
shm_runtime_path (char            *buf,
                  SHM_STD(size_t)  size,
                  char const      *uid,
                  char const      *file)
{
	if (uid && *uid)
		return SHM_STD(snprintf)(buf, size, "/tmp/dlssnr-%s%s", uid, file);
#ifdef _WIN32
	// The helper is always handed DLSSNR_SHM by the launcher, so this is only ever a last resort.
	return SHM_STD(snprintf)(buf, size, "/tmp/dlssnr%s", file);
#else
	return SHM_STD(snprintf)(buf, size, "/tmp/dlssnr-%u%s", (unsigned)getuid(), file);
#endif
}

/** @brief Where the mapping lives.
 *
 * It has to name the same file in every process that touches it, and a Steam game does not share a
 * mount namespace with the helper: pressure-vessel gives the container a private tmpfs at
 * $XDG_RUNTIME_DIR, so a mapping put there is simply absent inside the game. The layer then creates
 * its own empty one at a path that reads identically in the log and waits forever for a helper that
 * is answering on the other file -- the "attached ... seq_req=0 / helper not running" case. /tmp is
 * bind-mounted from the host into the container, so both sides land on one file; it is also what a
 * Wine prefix exposes as Z:\tmp\..., which is how the helper opens it.
 *
 * The directory is /tmp/dlssnr-UID, where UID is DLSSNR_UID if it is set and not empty, otherwise
 * the real user ID.
 *
 * @param buf  Receives the directory; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the directory's length, which is @a size or more if @a buf
 *             holds only the start of it.
 */
SHM_INLINE int
ShmRuntimeDir (char            *buf,
               SHM_STD(size_t)  size)
{
	return shm_runtime_path(buf, size, SHM_STD(getenv)("DLSSNR_UID"), "");
}

/** @brief The shared default channel: shm.bin in ShmRuntimeDir().
 *
 * @param buf  Receives the path; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the path's length, which is @a size or more if @a buf holds
 *             only the start of it.
 */
SHM_INLINE int
ShmDefaultPath (char            *buf,
                SHM_STD(size_t)  size)
{
	return shm_runtime_path(buf, size, SHM_STD(getenv)("DLSSNR_UID"), "/shm.bin");
}

#ifndef _WIN32
/** @brief The native tools' default channel, /tmp/dlsslop-amd-UID/shm.bin with the real user ID.
 *
 * Keep the native HIP tools on their own channel, including direct invocations
 * that do not pass through the installed Bash wrappers.
 *
 * @param buf  Receives the path; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the path's length, which is @a size or more if @a buf holds
 *             only the start of it.
 */
SHM_INLINE int
ShmNativeDefaultPath (char            *buf,
                      SHM_STD(size_t)  size)
{
	return SHM_STD(snprintf)(buf, size, "/tmp/dlsslop-amd-%u/shm.bin", (unsigned)getuid());
}

/** @brief Writes the native tools' channel: @a channel, or the native default if it is empty.
 *
 * @param buf     Receives the path; may be nullptr if @a size is 0.
 * @param size    The size of @a buf.
 * @param channel The value of DLSSNR_SHM, or nullptr if it is not set.
 * @return        What snprintf() returns.
 */
SHM_INLINE int
shm_native_channel_path (char            *buf,
                         SHM_STD(size_t)  size,
                         char const      *channel)
{
	if (channel && *channel)
		return SHM_STD(snprintf)(buf, size, "%s", channel);
	return ShmNativeDefaultPath(buf, size);
}

/** @brief The native tools' channel: nonempty DLSSNR_SHM, otherwise the native default.
 *
 * @param buf  Receives the path; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the path's length, which is @a size or more if @a buf holds
 *             only the start of it.
 */
SHM_INLINE int
ShmNativeChannelPath (char            *buf,
                      SHM_STD(size_t)  size)
{
	return shm_native_channel_path(buf, size, SHM_STD(getenv)("DLSSNR_SHM"));
}
#endif

/** @brief The size of the mapping.
 *
 * @return The header's region and both pixel regions, in bytes.
 */
SHM_INLINE SHM_STD(size_t)
ShmTotalBytes (void)
{
	return kHeaderBytes + kMaxFrame * 2;
}

/** @brief One pass's overrides.
 *
 * Every field is present; `overrideMask` says which of them mean anything.
 */
struct PassControl {
	_Atomic(SHM_STD(uint32_t)) overrideMask;
	_Atomic(SHM_STD(uint32_t)) intensityBits;
	_Atomic(SHM_STD(uint32_t)) localToneBits;
	_Atomic(SHM_STD(uint32_t)) localStructureBits;
	_Atomic(SHM_STD(uint32_t)) skinStructureBits;
	_Atomic(SHM_STD(uint32_t)) sharpnessBits;
	_Atomic(SHM_STD(uint32_t)) style;
	_Atomic(SHM_STD(uint32_t)) preset;
	_Atomic(SHM_STD(uint32_t)) autoMask;
};

/** @brief The header at the start of the mapping. */
struct ShmHeader {
	_Atomic(SHM_STD(uint32_t)) magic;
	_Atomic(SHM_STD(uint32_t)) version;

	// The frame handshake. The layer bumps seq_req after writing a proxy; the helper answers by
	// storing the same number into seq_resp once the model's answer is in the output region.
	_Atomic(SHM_STD(uint32_t)) seq_req;
	_Atomic(SHM_STD(uint32_t)) seq_resp;
	_Atomic(SHM_STD(uint32_t)) width;
	_Atomic(SHM_STD(uint32_t)) height;
	_Atomic(SHM_STD(uint32_t)) format; //!< always 1 (RGBA byte order); kept so a v1 helper is not silently wrong
	_Atomic(SHM_STD(uint32_t)) quit;
	_Atomic(SHM_STD(uint32_t)) heartbeat;

	// Bumped by whoever writes a setting. The layer and the helper watch it rather than re-reading
	// thirty values every frame.
	_Atomic(SHM_STD(uint32_t)) controlSeq;

	// Bumped only when something the model latches at feature creation changes. The helper rebuilds
	// its features on this and debounces the rebuild; bumping it every frame exhausts the driver's
	// latches and the model stops responding until the process restarts.
	_Atomic(SHM_STD(uint32_t)) tuningSeq;

	// --- the model ---------------------------------------------------------------------------
	_Atomic(SHM_STD(uint32_t)) enabled;
	_Atomic(SHM_STD(uint32_t)) passes;
	_Atomic(SHM_STD(uint32_t)) unlockPasses;
	_Atomic(SHM_STD(uint32_t)) preset;
	_Atomic(SHM_STD(uint32_t)) style;
	_Atomic(SHM_STD(uint32_t)) autoMask;
	_Atomic(SHM_STD(uint32_t)) intensityBits;
	_Atomic(SHM_STD(uint32_t)) localToneBits;
	_Atomic(SHM_STD(uint32_t)) localStructureBits;
	_Atomic(SHM_STD(uint32_t)) skinStructureBits;
	_Atomic(SHM_STD(uint32_t)) sharpnessBits;

	// --- the composition ---------------------------------------------------------------------
	// How much of the model's edit reaches the frame, and how much of it is allowed to be colour
	// rather than luminance. Separating the two is what keeps saturated highlights from shifting hue.
	_Atomic(SHM_STD(uint32_t)) transferStrengthBits;
	_Atomic(SHM_STD(uint32_t)) colourStrengthBits;
	// The most the pass may multiply or divide a pixel by. The transfer is a ratio, and a ratio
	// against a near-black proxy pixel is unbounded without one.
	_Atomic(SHM_STD(uint32_t)) maxRatioBits;
	// How a model that worked below the frame's size is brought back. 0 classic, 1 matched residual,
	// 2 native + edit -- the frame's own pixels with only the model's difference added, so what the
	// model left alone never passes through the enlargement.
	_Atomic(SHM_STD(uint32_t)) transfer;
	// 0 off, 1 the picture the model was shown, 2 its raw answer, 3 what it changed, amplified.
	_Atomic(SHM_STD(uint32_t)) debugView;
	_Atomic(SHM_STD(uint32_t)) debugScaleBits;
	_Atomic(SHM_STD(uint32_t)) whitePointBits;
	_Atomic(SHM_STD(uint32_t)) whitePointScaleBits;
	_Atomic(SHM_STD(uint32_t)) whitePointSource;
	_Atomic(SHM_STD(uint32_t)) whitePointTrimBits;
	// What fraction of the frame's resolution the model works at. The frame itself is never reduced:
	// only the model's contribution is computed at this scale and resized, so the picture underneath
	// is untouched whatever this is.
	//
	// Above 1.0 is supersampling -- the model runs above native and its answer is brought back down
	// by scalingDownscaler. Upstream allows up to 2.0. Below 1.0 it also cuts what crosses the shared
	// memory, quadratically, which on this transport matters more than it does upstream.
	_Atomic(SHM_STD(uint32_t)) workingScaleBits;
	// 0 off, 1 side by side, 2 a wipe.
	_Atomic(SHM_STD(uint32_t)) compareMode;
	_Atomic(SHM_STD(uint32_t)) compareSplitBits;
	_Atomic(SHM_STD(uint32_t)) compareZoomBits;
	_Atomic(SHM_STD(uint32_t)) compareSwap;
	_Atomic(SHM_STD(uint32_t)) colourMode;
	// Writes one set of matched before/after frames per session when the layer next presents.
	_Atomic(SHM_STD(uint32_t)) captureRequest;

	// A Linux KEY_ code the layer watches to toggle the pass, or 0 for none. Unbound by default,
	// because a key that does something unexpected is worse than a key that does nothing.
	//
	// Only useful where the layer can read the keyboard at all: an X11 or XWayland session, inside
	// gamescope, or anywhere the user is in the 'input' group. A game presenting through winewayland
	// is a Wayland client whose keys never reach this process, and keyboards get no uaccess ACL, so
	// there the answer is a desktop shortcut bound to 'dlsslopctl --toggle enabled' instead.
	_Atomic(SHM_STD(uint32_t)) toggleKey;

	// Which proxy the model is shown, and whether its answer is composed or substituted. See
	// ReversibleMode. Default 0 keeps the picture identical to the pre-import behaviour.
	_Atomic(SHM_STD(uint32_t)) reversibleMode;

	// Whether the model's edit is applied at all. Off keeps the whole pass running -- the capture,
	// the round trip, the encode -- and simply presents the clean frame, which is what makes an
	// honest A/B possible: the cost is unchanged, so only the picture differs.
	_Atomic(SHM_STD(uint32_t)) applyModel;

	// Freeze the frame the pass works on, so changing a setting re-runs the composition over the
	// SAME picture instead of over whatever the game has drawn since. The only clean way to compare
	// two settings, and a live testing control rather than a saved preference.
	//
	// In this architecture it is cheaper than upstream: the layer already holds the captured proxy
	// and the model's last answer, so holding means not re-capturing rather than keeping a frame
	// alive somewhere it would not otherwise be.
	_Atomic(SHM_STD(uint32_t)) holdFrame;

	// The filter for the supersampling down-leg. See Downscaler; only read when workingScale > 1.
	_Atomic(SHM_STD(uint32_t)) scalingDownscaler;

	// --- status, written by the helper --------------------------------------------------------
	_Atomic(SHM_STD(uint32_t)) helperState;
	_Atomic(SHM_STD(uint32_t)) modelUp;
	_Atomic(SHM_STD(uint32_t)) helperFramesLo;
	_Atomic(SHM_STD(uint32_t)) helperFramesHi;
	_Atomic(SHM_STD(uint32_t)) helperEvalMsBits;
	_Atomic(SHM_STD(uint32_t)) helperUploadMsBits;
	_Atomic(SHM_STD(uint32_t)) helperReadbackMsBits;
	_Atomic(SHM_STD(uint32_t)) helperVramMB;
	_Atomic(SHM_STD(uint32_t)) helperFeatures;    //!< how many NGX features are actually built
	_Atomic(SHM_STD(uint32_t)) helperPassCeiling; //!< what the VRAM budget currently allows

	// --- status, written by the layer ---------------------------------------------------------
	// The pid of the process the layer is loaded into, or 0 when no layer is attached.
	//
	// This field was declared and never written, and that gap is why a paused video read as nothing
	// running at all: the only evidence of a layer was its frame counter, so "is a layer attached"
	// and "is it drawing right now" were the same question. They are not. A video that is paused, a
	// window that is occluded and a game that has exited all present no frames, and only the last of
	// them means there is nothing there.
	//
	// A pid rather than a flag, because a flag cannot survive the process that set it. Nothing clears
	// this when a game crashes, so a reader checks the pid is still alive rather than trusting the
	// value. Same width as the flag it replaces, so the layout and the protocol version are
	// unchanged; an older layer simply leaves it zero, which reads as "no layer" exactly as before.
	_Atomic(SHM_STD(uint32_t)) layerPid;
	_Atomic(SHM_STD(uint32_t)) layerFramesLo;
	_Atomic(SHM_STD(uint32_t)) layerFramesHi;
	_Atomic(SHM_STD(uint32_t)) layerWidth;
	_Atomic(SHM_STD(uint32_t)) layerHeight;
	_Atomic(SHM_STD(uint32_t)) layerFormat;
	_Atomic(SHM_STD(uint32_t)) layerCompositionUp;
	_Atomic(SHM_STD(uint32_t)) layerMsBits;
	_Atomic(SHM_STD(uint32_t)) layerMeasuredWhiteBits;
	_Atomic(SHM_STD(uint32_t)) layerHeartbeat;

	// Free text, each guarded by its own sequence number: bumped after the bytes are written, so a
	// reader that sees an unchanged number is looking at a whole string.
	_Atomic(SHM_STD(uint32_t)) helperReasonSeq;
	char                       helperReason[kReasonBytes];
	_Atomic(SHM_STD(uint32_t)) layerReasonSeq;
	char                       layerReason[kReasonBytes];
	_Atomic(SHM_STD(uint32_t)) gameNameSeq;
	char                       gameName[kNameBytes];

	struct PassControl         pass[kMaxPasses];

	// Appended after the pass array on purpose: everything before it has a pinned offset, and a new
	// field inserted higher up would move all of them. Motion vectors, from bmitch87's work.
	_Atomic(SHM_STD(uint32_t)) mvecEnabled;
	_Atomic(SHM_STD(uint32_t)) mvecScaleMode;
	_Atomic(SHM_STD(uint32_t)) mvecQuality;
	// How far the helper has answered *successfully*. seq_resp says a frame came back; this says it
	// was worth using, so the layer can present the game's own frame when it was not.
	_Atomic(SHM_STD(uint32_t)) seq_ok;

	// 0: the composition blends the model's edit onto the frame under the strength and guard limits.
	// 1: no composition at all -- the model's raw answer IS the presented frame, and the limits,
	// enlargement and compare overlays are moot. Default 1: the composition is off until the user
	// turns it on.
	_Atomic(SHM_STD(uint32_t)) compositionBypass;

	// Wall-clock milliseconds the helper waits after the last tuning change before it rebuilds a
	// feature, and between one rebuild and the next. NGX creation is expensive and back-to-back
	// creation was seen to exhaust the driver's latches on some setups, so the default spaces
	// rebuilds rather than firing them at once; 0 means no spacing -- build the moment the change
	// settles and chain the remaining builds back to back. It is time rather than frames because a
	// frame-counted wait crawls on a 30 fps game and races on a 144 fps one. dlsslopd, the layer and
	// the tools do not read it; it stays so that the header's layout does not change.
	_Atomic(SHM_STD(uint32_t)) rebuildSettleMs;

	// The raster the helper actually answered, echoed before seq_resp. More than one swapchain can
	// share this channel -- a game and the Steam overlay, or a game mid-resize with its old and new
	// swapchains both presenting -- and seq_resp only says *a* frame came back. Without the echo a
	// swapchain waiting on its own request can be satisfied by another's answer and copy the wrong
	// number of bytes, which is the row-shifted colour garbage this field exists to refuse.
	_Atomic(SHM_STD(uint32_t)) answeredW;
	_Atomic(SHM_STD(uint32_t)) answeredH;

	// Retired: phase 5's dma-buf exchange. The helper named its exported proxy and answer images
	// here, and the layer echoed the export sequences it had imported. Nothing uses these fields any
	// more; they keep their slots so that the layout stays the same.
	_Atomic(SHM_STD(uint32_t)) proxyExportSeq;
	_Atomic(SHM_STD(uint32_t)) proxyPid;
	_Atomic(SHM_STD(uint32_t)) proxyFd;
	_Atomic(SHM_STD(uint32_t)) proxyGen;
	_Atomic(SHM_STD(uint32_t)) answerExportSeq;
	_Atomic(SHM_STD(uint32_t)) answerPid;
	_Atomic(SHM_STD(uint32_t)) answerFd;
	_Atomic(SHM_STD(uint32_t)) answerGen;
	_Atomic(SHM_STD(uint32_t)) layerProxySeq;
	_Atomic(SHM_STD(uint32_t)) layerAnswerSeq;

	// The HDR input path. hdrMode is the user's choice (see HdrMode); hdrDetected and hdrKind are the
	// layer's reading of the primary swapchain's format and colour space; hdrActive is the decision
	// the layer actually encoded this frame under, and proxyFormat is what the helper built the
	// crossing images as. The layer uses the float16 path only while both agree it exists -- if the
	// model refused the float input, proxyFormat stays 8-bit and the encode tone maps as before.
	_Atomic(SHM_STD(uint32_t)) hdrMode;
	_Atomic(SHM_STD(uint32_t)) hdrDetected;
	_Atomic(SHM_STD(uint32_t)) hdrActive;
	_Atomic(SHM_STD(uint32_t)) proxyFormat;
	// What the proxy bytes in the shared region actually are for the request being made: the layer
	// writes this immediately before seq_req. Native HIP uses it directly as the payload
	// precision, without proxyFormat. Legacy Windows transport may lag hdrActive while
	// crossing images are rebuilt; mismatched frames are refused, never misread.
	_Atomic(SHM_STD(uint32_t)) hdrEncode;

	// How much of the chroma-agreement gate to apply, in hundredths. 100 is the gate as written; 0
	// switches it off and takes the model's colour everywhere.
	//
	// The gate exists because the model's colour disagrees with the frame's most at edges -- an edge
	// being precisely what it was asked to re-decide -- and taking that hue whole put one colour on
	// one side of an edge and its complement on the other. Measured, the pass moved colour balance
	// three to five times more at edges than on flat pixels.
	//
	// The cost of it is that where the gate closes, the composed pixel falls back to the frame's own
	// colour scaled by one luminance ratio -- so on a detailed frame the model's chroma detail is
	// dropped exactly where the model had most to say. That is a real part of "composition only ever
	// removes detail", and it was never separable from the colour strength control, because the gate
	// multiplies that control rather than being one.
	_Atomic(SHM_STD(uint32_t)) colourTrustPercent;

	// How much of the relighting ratio is taken from the pixel's neighbourhood instead of the pixel,
	// in hundredths. 0 is the behaviour that shipped before it existed.
	//
	// The composition rebuilds the frame as its own pixel times one per-pixel number. On detailed
	// content that number varies sharply, because the model's answer differs sharply there, and the
	// highlight guard is all that holds it -- so raising the guard lets the variation through as
	// blown and black pixels wearing whatever colour the texture had. Carrying a ratio at full
	// spatial frequency is the mistake: what the model knows at this scale is how much light belongs
	// here, not which pixel is brighter than its neighbour, and the frame already knows that.
	_Atomic(SHM_STD(uint32_t)) ratioSmoothPercent;

	// SDR uses 8-bit ping-pong images by default. Enable 16-bit UNORM to avoid quantising between
	// passes at higher memory and bandwidth cost.
	_Atomic(SHM_STD(uint32_t)) sdr16Multipass;
	_Atomic(SHM_STD(uint32_t)) mvecPixelSize;

	// Native worker capability, not a user scaling setting. Zero keeps the
	// original transport extent (CPU composition and identity diagnostic modes).
	_Atomic(SHM_STD(uint32_t)) nativeModelMaxWidth;
	_Atomic(SHM_STD(uint32_t)) nativeModelMaxHeight;
	_Atomic(SHM_STD(uint32_t)) colorPreserveBits; //!< Native per-pass color anchoring, 0..1.

	// Native device-local transport. The layer offers its exported proxy and answer buffers on
	// ShmTransportPath() under a nonzero generation, and the daemon answers on the offer's own
	// connection (ShmTransportOffer). A request naming a generation in transportGen is read from
	// and answered into those buffers instead of the pixel regions (zero: the regions). A worker
	// that does not hold the named generation stores it in transportMiss and fails the frame, so
	// the layer offers again.
	_Atomic(SHM_STD(uint32_t)) transportGen;
	// Protocols 24 and 25 acknowledged offers here. Unused since, the slot keeps transportMiss
	// where a layer attached before an upgrade still reads it, so that layer offers again.
	_Atomic(SHM_STD(uint32_t)) retiredTransportAck;
	_Atomic(SHM_STD(uint32_t)) transportMiss;

	// Native neural raster height: 720, 900 or 1080. A controller stores the tier it wants; between
	// frames the worker rebuilds its network for a different one (helperState reads Starting
	// meanwhile) and publishes the new raster in nativeModelMaxWidth/Height. A value it cannot use
	// is overwritten with the active tier. Storing the active tier again changes nothing.
	_Atomic(SHM_STD(uint32_t)) nativeTier;
};

static_assert(sizeof (struct ShmHeader) <= kHeaderBytes, "ShmHeader outgrew its region");

// The words, pinned in C and C++ alike.
//
// A word is shared by processes that may have been built from different languages, so its size and
// alignment must not depend on the language. And it must be lock-free: a word that is not goes
// through a lock table local to each process, and is then not atomic across processes. uint32_t has
// the size of unsigned int, for which ATOMIC_INT_LOCK_FREE answers.
static_assert(sizeof (_Atomic(SHM_STD(uint32_t))) == 4, "a word of the channel is not four bytes");
static_assert(alignof (_Atomic(SHM_STD(uint32_t))) == 4, "a word of the channel is not aligned to four bytes");
static_assert(sizeof (SHM_STD(uint32_t)) == sizeof (unsigned) && ATOMIC_INT_LOCK_FREE == 2,
              "a word of the channel is not lock-free");

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
// Every member is a word, a char array of a multiple of four bytes, or the pass array, so there is
// no padding: with the words pinned above, these hold in C and C++ alike.
static_assert(sizeof (struct ShmHeader) == 1996, "the header layout changed -- bump kShmVersion");
static_assert(alignof (struct ShmHeader) == 4, "layout changed -- bump kShmVersion");
static_assert(sizeof (struct PassControl) == 36, "layout changed -- bump kShmVersion");

static_assert(offsetof(struct ShmHeader, enabled) == 44, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, transferStrengthBits) == 88, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, helperState) == 176, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, helperReason) == 260, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, layerReason) == 456, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, gameName) == 652, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, pass) == 780, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, mvecEnabled) == 1860, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, seq_ok) == 1872, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, hdrEncode) == 1948, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, transportGen) == 1980, "layout changed -- bump kShmVersion");
static_assert(offsetof(struct ShmHeader, nativeTier) == 1992, "layout changed -- bump kShmVersion");

/** @brief One device-local transport offer.
 *
 * A single SOCK_SEQPACKET message on ShmTransportPath() whose two SCM_RIGHTS descriptors are the
 * exported proxy and answer memory, in that order. The daemon answers on the same connection with
 * one byte, nonzero when it imported the pair. A connection it closes unanswered was never taken:
 * the daemon stopped first.
 *
 * Each memory is a dedicated allocation of one buffer: exportable as an opaque fd, exclusive, for
 * transfers both ways, of the offered size. A Vulkan importer binds it to a buffer created exactly
 * so, as dedicated imports require, and only on a device and driver of the offered UUIDs.
 */
struct ShmTransportOffer {
	SHM_STD(uint32_t) magic;          //!< kShmMagic
	SHM_STD(uint32_t) generation;     //!< nonzero; what requests name in transportGen
	SHM_STD(uint64_t) allocation[2];  //!< the proxy and answer memory sizes
	SHM_STD(uint64_t) size[2];        //!< their buffers' sizes, 1..allocation; frames up to the smaller fit
	SHM_STD(uint8_t)  deviceUuid[16]; //!< the exporter's VkPhysicalDeviceIDProperties
	SHM_STD(uint8_t)  driverUuid[16];
};

static_assert(sizeof (struct ShmTransportOffer) == 72, "the transport offer's layout changed");

/** @brief The socket on which the layer offers its transport.
 *
 * @param buf     Receives the path; may be nullptr if @a size is 0.
 * @param size    The size of @a buf.
 * @param channel The channel's path.
 * @return        What snprintf() returns: the path's length, which is @a size or more if @a buf
 *                holds only the start of it.
 */
SHM_INLINE int
ShmTransportPath (char            *buf,
                  SHM_STD(size_t)  size,
                  char const      *channel)
{
	return SHM_STD(snprintf)(buf, size, "%s.sock", channel);
}

/** @brief A float's bits.
 *
 * @param f The float.
 * @return  Its bits, as a word of the channel holds them.
 */
SHM_INLINE SHM_STD(uint32_t)
FloatToBits (float f)
{
	SHM_STD(uint32_t) u = 0;
	SHM_STD(memcpy)(&u, &f, sizeof u);
	return u;
}

/** @brief The float that bits hold.
 *
 * @param u The bits, as a word of the channel holds them.
 * @return  The float.
 */
SHM_INLINE float
BitsToFloat (SHM_STD(uint32_t) u)
{
	float f = 0.0f;
	SHM_STD(memcpy)(&f, &u, sizeof f);
	return f;
}

/** @brief Publishes a string in one of the header's text fields.
 *
 * Writes the string, cut to @a cap - 1 bytes and padded with zeros, then bumps @a seq.
 *
 * @param seq The field's sequence number.
 * @param dst The field.
 * @param cap The field's size; at least 1.
 * @param src The string, or nullptr for an empty one.
 */
SHM_INLINE void
ShmStoreString (_Atomic(SHM_STD(uint32_t)) *seq,
                char                       *dst,
                SHM_STD(size_t)             cap,
                char const                 *src)
{
	SHM_STD(memset)(dst, 0, cap);
	if (src)
		SHM_STD(strncpy)(dst, src, cap - 1);
	atomic_fetch_add(seq, 1);
}

/** @brief Reads a string that ShmStoreString() published.
 *
 * Copies the field and keeps the copy if @a seq did not change meanwhile, trying four times.
 *
 * @param seq The field's sequence number.
 * @param src The field.
 * @param cap The field's size, and @a out's; at least 1.
 * @param out Receives the string, or an empty one if every copy was torn.
 * @return    true if @a out holds a whole string.
 */
SHM_INLINE bool
ShmLoadString (_Atomic(SHM_STD(uint32_t)) const *seq,
               char const                       *src,
               SHM_STD(size_t)                   cap,
               char                             *out)
{
	for (SHM_STD(uint32_t) attempt = 0; attempt < 4; ++attempt) {
		SHM_STD(uint32_t) before = atomic_load(seq);
		SHM_STD(memcpy)(out, src, cap);
		out[cap - 1] = '\0';
		if (atomic_load(seq) == before)
			return true;
	}
	*out = '\0';
	return false;
}

/** @brief Resets a header to the defaults of the shared channel.
 *
 * @param h The header; every byte of it is written.
 */
SHM_INLINE void
ShmInitDefaults (struct ShmHeader *h)
{
	// The cast tells g++ that clearing a struct of atomics is meant.
	SHM_STD(memset)((void *)h, 0, sizeof *h);
	atomic_store(&h->magic, kShmMagic);
	atomic_store(&h->version, kShmVersion);
	atomic_store(&h->helperState, kHelperStopped);
	atomic_store(&h->format, 1);
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
	atomic_store(&h->scalingDownscaler, kDownscaleLanczos3);

	atomic_store(&h->hdrMode, kHdrAuto);
	atomic_store(&h->hdrDetected, kHdrNone);
	atomic_store(&h->hdrActive, 0);
	atomic_store(&h->proxyFormat, kProxyRgba8);

	atomic_store(&h->mvecEnabled, 1);
	atomic_store(&h->mvecScaleMode, kMVecPixels);
	atomic_store(&h->mvecQuality, kMVecBalanced);
	atomic_store(&h->mvecPixelSize, kMVecPixels4);
	atomic_store(&h->seq_ok, 0);
	atomic_store(&h->compositionBypass, 1);
	atomic_store(&h->rebuildSettleMs, 100);
	atomic_store(&h->colourTrustPercent, 200);

	atomic_store(&h->ratioSmoothPercent, 100);
	atomic_store(&h->sdr16Multipass, 0);

	for (SHM_STD(uint32_t) i = 0; i < kMaxPasses; ++i) {
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

/** @brief Resets a header to the defaults of the native channel.
 *
 * This is the single source of startup/reset/help defaults for the Linux HIP
 * backend. The CPU-composition and identity worker modes already return a final
 * image, so their compositor bypass default is one.
 *
 * @param h      The header; every byte of it is written.
 * @param bypass The default of compositionBypass: true for the worker modes that return a final
 *               image, otherwise false.
 */
SHM_INLINE void
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

/** @brief The tier of a height.
 *
 * @param height A tier's height.
 * @return       The tier of that height, or nullptr if no tier has it.
 */
SHM_INLINE struct NativeTier const *
ShmNativeTier (SHM_STD(uint32_t) height)
{
	for (SHM_STD(size_t) i = 0; i < sizeof kNativeTiers / sizeof *kNativeTiers; ++i)
		if (kNativeTiers[i].height == height)
			return &kNativeTiers[i];
	return nullptr;
}

/** @brief The most passes a header may ask for.
 *
 * @param h The header.
 * @return  kMaxPasses.
 */
SHM_INLINE SHM_STD(uint32_t)
ShmPassCeiling (struct ShmHeader const *h)
{
	(void)h;
	return kMaxPasses;
}

/** @brief The passes a header asks for.
 *
 * @param h The header.
 * @return  passes, between 1 and ShmPassCeiling(); 0 counts as 1.
 */
SHM_INLINE SHM_STD(uint32_t)
ShmPasses (struct ShmHeader const *h)
{
	SHM_STD(uint32_t) p = atomic_load(&h->passes);
	SHM_STD(uint32_t) ceiling = ShmPassCeiling(h);
	if (p == 0)
		return 1;
	return p > ceiling ? ceiling : p;
}

/** @brief Whether a header asks for neural rendering.
 *
 * @param h The header.
 * @return  true if enabled is not 0.
 */
SHM_INLINE bool
ShmNeuralEnabled (struct ShmHeader const *h)
{
	return atomic_load(&h->enabled) != 0;
}

/** @brief Reads a 64-bit count that two words hold.
 *
 * @param lo The low word.
 * @param hi The high word.
 * @return   The count.
 */
SHM_INLINE SHM_STD(uint64_t)
ShmLoad64 (_Atomic(SHM_STD(uint32_t)) const *lo,
           _Atomic(SHM_STD(uint32_t)) const *hi)
{
	return ((SHM_STD(uint64_t))atomic_load(hi) << 32) | (SHM_STD(uint64_t))atomic_load(lo);
}

/** @brief Stores a 64-bit count in two words, the high word first.
 *
 * @param lo The low word.
 * @param hi The high word.
 * @param v  The count.
 */
SHM_INLINE void
ShmStore64 (_Atomic(SHM_STD(uint32_t)) *lo,
            _Atomic(SHM_STD(uint32_t)) *hi,
            SHM_STD(uint64_t)           v)
{
	atomic_store(hi, (SHM_STD(uint32_t))(v >> 32));
	atomic_store(lo, (SHM_STD(uint32_t))(v & 0xFFFFFFFFu));
}

/** @brief The motion search quality that a header asks for.
 *
 * @param h The header.
 * @return  mvecQuality, or kMVecBalanced if it names no quality.
 */
SHM_INLINE SHM_STD(uint32_t)
ShmMVecQuality (struct ShmHeader const *h)
{
	SHM_STD(uint32_t) q = atomic_load(&h->mvecQuality);
	return q <= kMVecQuality ? q : kMVecBalanced;
}

/** @brief The motion grid's spacing that a header asks for.
 *
 * @param h The header.
 * @return  mvecPixelSize, or kMVecPixels4 if it names no spacing.
 */
SHM_INLINE SHM_STD(uint32_t)
ShmMVecPixelSize (struct ShmHeader const *h)
{
	SHM_STD(uint32_t) size = atomic_load(&h->mvecPixelSize);
	return size <= kMVecPixels8 ? size : kMVecPixels4;
}

#ifdef __cplusplus
} /* extern "C" */

/** @brief ShmDefaultPath() as a string.
 *
 * @return The path.
 */
inline std::string
ShmDefaultPath ()
{
	char const *uid = std::getenv("DLSSNR_UID");
	std::string path(static_cast<std::size_t>(shm_runtime_path(nullptr, 0, uid, "/shm.bin")), '\0');
	shm_runtime_path(path.data(), path.size() + 1, uid, "/shm.bin");
	return path;
}

# ifndef _WIN32
/** @brief ShmNativeDefaultPath() as a string.
 *
 * @return The path.
 */
inline std::string
ShmNativeDefaultPath ()
{
	std::string path(static_cast<std::size_t>(ShmNativeDefaultPath(nullptr, 0)), '\0');
	ShmNativeDefaultPath(path.data(), path.size() + 1);
	return path;
}

/** @brief ShmNativeChannelPath() as a string.
 *
 * @return The path.
 */
inline std::string
ShmNativeChannelPath ()
{
	char const *channel = std::getenv("DLSSNR_SHM");
	std::string path(static_cast<std::size_t>(shm_native_channel_path(nullptr, 0, channel)), '\0');
	shm_native_channel_path(path.data(), path.size() + 1, channel);
	return path;
}
# endif

/** @brief ShmTransportPath() as a string.
 *
 * @param channel The channel's path.
 * @return        The path.
 */
inline std::string
ShmTransportPath (std::string const &channel)
{
	std::string path(static_cast<std::size_t>(ShmTransportPath(nullptr, 0, channel.c_str())), '\0');
	ShmTransportPath(path.data(), path.size() + 1, channel.c_str());
	return path;
}

/** @brief ShmLoadString() as a string.
 *
 * @param seq The field's sequence number.
 * @param src The field.
 * @param cap The field's size; at least 1. At most the larger of kReasonBytes and kNameBytes is
 *            read.
 * @return    The string, or an empty one if every copy was torn.
 */
inline std::string
ShmLoadString (std::atomic<std::uint32_t> const &seq,
               char const                       *src,
               std::size_t                       cap)
{
	char text[kReasonBytes > kNameBytes ? kReasonBytes : kNameBytes];
	ShmLoadString(&seq, src, cap < sizeof text ? cap : sizeof text, text);
	return text;
}
#endif

#undef SHM_CONSTEXPR
#undef SHM_INLINE
#undef SHM_STD

#endif /* DLSSLOP_AMD_COMMON_SHM_PROTOCOL_H_ */
