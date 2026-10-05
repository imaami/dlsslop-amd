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
 * C and C++ share this header, which declares the channel; shm_protocol.c defines its functions
 * and kNativeTiers, and both languages call them. A word of the channel is an _Atomic(uint32_t) and
 * a frame count an _Atomic(uint64_t), which <stdatomic.h> makes a std::atomic<uint32_t> and a
 * std::atomic<uint64_t> in C++, so C++ code reads and writes them with their load() and store()
 * members. C code uses the atomic_* functions of <stdatomic.h>, which have the same orders. The
 * functions name a text field by the header and an enum shm_text, not by its words, as a C++ word
 * is another type than a C one.
 * shm_protocol.hpp has versions that return a std::string of those that write a string into the
 * caller's buffer.
 */
#ifndef DLSSLOP_AMD_COMMON_SHM_PROTOCOL_H_
#define DLSSLOP_AMD_COMMON_SHM_PROTOCOL_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
#else
# include <stddef.h>
# include <stdint.h>
#endif

#include <stdatomic.h>

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief The channel's identity. */
enum : STD(uint32_t) {
	/** @brief 'GNR2'.
	 *
	 * Bumped from the v1 magic on purpose: a stale v1 mapping left in XDG_RUNTIME_DIR must be
	 * re-initialised rather than half-read, because the header grew and every offset moved.
	 */
	kShmMagic = 0x32524E47,

	/** @brief The version of the header's layout and of the protocol.
	 *
	 * v14: this branch and upstream both grew the header, so neither side's number describes it.
	 * From upstream: the pixel regions are eight bytes a pixel for the float16 HDR proxy, the header
	 * is 64 KiB because VK_EXT_external_memory_host demands the imported pointer meet
	 * minImportedHostPointerAlignment and NVIDIA answers 64 KiB, and the dma-buf exchange and HDR
	 * and round-trip attribution. A stale mapping of either lineage must be re-created, not
	 * half-read.
	 * v21-v23: native HIP raster bounds, request-published proxy precision and color anchoring.
	 * v24: native device-local transport generations.
	 * v25: live native tier requests.
	 * v26: offers are answered on their own connection; transportAck is retired.
	 * v27: offers carry their buffers' sizes.
	 * v28: offers name the exporting device and driver.
	 * v29: each frame count is one 64-bit atomic, and layerFrames moved to the end.
	 * v30: the working scale is at most 1, and scalingDownscaler is retired.
	 * v31: the retired and unused fields are gone, and the frame counts lead the header.
	 */
	kShmVersion = 31,
};

/** @brief The frames that the channel carries. */
enum : STD(uint32_t) {
	kMaxW = 7680, //!< The widest frame.
	kMaxH = 4320, //!< The tallest frame.

	/** @brief The narrowest frame.
	 *
	 * A game that presents a 1x1 probe swapchain -- NWN:EE does, behind its real window -- used to
	 * have the model built at that size, and the first submit hung the GPU channel outright (Xid
	 * 109, CTX SWITCH TIMEOUT), taking the game down with it. 300x300 is known good; nothing below
	 * this is a frame worth composing anyway.
	 */
	kMinW = 64,
	kMinH = 64, //!< The shortest frame. See kMinW.
};

/** @brief A native neural raster.
 *
 * A tier is the height of the picture the network sees, inside an input padded to networkHeight
 * rows.
 */
struct NativeTier {
	STD(uint32_t) height;
	STD(uint32_t) width;
	STD(uint32_t) networkHeight;
};

/** @brief The native neural rasters' heights. */
enum : STD(uint32_t) {
	kNativeTierMin = 720,  //!< The height of the smallest native neural raster.
	kNativeTierStep = 180, //!< From one raster's height to the next.
	kNativeTierCount = 3,  //!< The rasters in kNativeTiers.

	/** @brief The height of the largest native neural raster. */
	kNativeTierMax = kNativeTierMin + (kNativeTierCount - 1) * kNativeTierStep,

	kNativeDefaultTier = 720, //!< The height of the default tier.
};

/** @brief The native neural rasters, smallest first, every kNativeTierStep rows from kNativeTierMin. */
extern struct NativeTier const kNativeTiers[kNativeTierCount];

/** @brief The pass counts. */
enum : STD(uint32_t) {
	/** @brief The ceiling on how many times the model runs over one frame.
	 *
	 * It is OptiScaler's DlssNr::kMaxPasses and sizes the per-pass arrays below.
	 */
	kMaxPasses = 30,

	kNativeDefaultPasses = 1, //!< The default pass count.
};

/** @brief The regions of the mapping. */
enum : STD(size_t) {
	/** @brief The size of a pixel region.
	 *
	 * Eight bytes a pixel: the float16 proxy needs them, and the 8-bit path simply uses the first
	 * half of each region. The mapping is file-backed and sparse, so an SDR session never commits
	 * the second half.
	 */
	kMaxFrame = (STD(size_t))kMaxW * kMaxH * 8,

	kHeaderBytes = 65536, //!< The size of the header's region.
};

/** @brief The sizes of the header's text fields. */
enum : STD(size_t) {
	kReasonBytes = 192, //!< The size of a reason string's field.
	kNameBytes = 128,   //!< The size of the game name's field.
};

/** @brief Which fields a per-pass entry actually overrides.
 *
 * Sparse by design: a pass with no entry, or an entry that does not name a field, follows the global
 * setting -- which is what makes "pass 3 is gentler" expressible without restating everything else
 * about pass 3.
 */
enum PassOverrideBit : STD(uint32_t) {
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
enum WhitePointSource : STD(uint32_t) {
	kWhitePointManual = 0,   //!< the slider, and nothing else
	kWhitePointMeasured = 1, //!< the calibration grid, measured off the untouched copy
};

/** @brief What the swapchain holds.
 *
 * Getting this wrong encodes an encoded frame a second time, which looks washed out and banded --
 * so the default is to decide it from the format rather than to guess.
 */
enum ColourMode : STD(uint32_t) {
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
enum ReversibleMode : STD(uint32_t) {
	kReversibleKnee = 0,
	kReversibleNeutwo = 1,
	kReversibleNeutwoReplace = 2,
	kReversibleHybrid = 3,
	kReversibleHybridReplace = 4,
	kReversibleModeCount = 5,
};

/** @brief What the helper has managed to do, for the GUI and for the layer's fail-open decision.
 *
 * The layer reads this to decide whether anything is listening, so "nobody" has to be the value a
 * freshly initialised header holds -- not a state that also means "starting".
 */
enum HelperState : STD(uint32_t) {
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
enum HdrMode : STD(uint32_t) {
	kHdrAuto = 0,   //!< HDR when the swapchain is HDR
	kHdrOff = 1,    //!< always the 8-bit proxy, whatever the swapchain
	kHdrForce = 2,  //!< float16 proxy even for an SDR swapchain (an A/B tool, not a preference)
};

/** @brief What the swapchain's format and colour space hold. */
enum HdrKind : STD(uint32_t) {
	kHdrNone = 0,       //!< 8-bit swapchain: already tone mapped
	kHdrLinearFp16 = 1, //!< R16G16B16A16_SFLOAT: linear light, open range
	kHdrPq10 = 2,       //!< 10-bit with a PQ/BT.2020 colour space: ST 2084 code
};

/** @brief What the crossing images actually are, published by the helper.
 *
 * The model decides, and the layer encodes to match rather than to hope.
 */
enum ProxyFormat : STD(uint32_t) {
	kProxyUnknown = 0,
	kProxyRgba8 = 1,
	kProxyRgba16F = 2,
};

/** @brief What the optical-flow engine is asked for.
 *
 * Higher costs more of the frame's budget.
 */
enum MVecQuality : STD(uint32_t) {
	kMVecFast = 0,
	kMVecBalanced = 1,
	kMVecQuality = 2,
};

/** @brief The motion grid's spacing. */
enum MVecPixelSize : STD(uint32_t) {
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
extern int
shm_runtime_path (char        *buf,
                  STD(size_t)  size,
                  char const  *uid,
                  char const  *file);

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
extern int
ShmRuntimeDir (char        *buf,
               STD(size_t)  size);

/** @brief The shared default channel: shm.bin in ShmRuntimeDir().
 *
 * @param buf  Receives the path; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the path's length, which is @a size or more if @a buf holds
 *             only the start of it.
 */
extern int
ShmDefaultPath (char        *buf,
                STD(size_t)  size);

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
extern int
ShmNativeDefaultPath (char        *buf,
                      STD(size_t)  size);

/** @brief Writes the native tools' channel: @a channel, or the native default if it is empty.
 *
 * @param buf     Receives the path; may be nullptr if @a size is 0.
 * @param size    The size of @a buf.
 * @param channel The value of DLSSNR_SHM, or nullptr if it is not set.
 * @return        What snprintf() returns.
 */
extern int
shm_native_channel_path (char        *buf,
                         STD(size_t)  size,
                         char const  *channel);

/** @brief The native tools' channel: nonempty DLSSNR_SHM, otherwise the native default.
 *
 * @param buf  Receives the path; may be nullptr if @a size is 0.
 * @param size The size of @a buf.
 * @return     What snprintf() returns: the path's length, which is @a size or more if @a buf holds
 *             only the start of it.
 */
extern int
ShmNativeChannelPath (char        *buf,
                      STD(size_t)  size);

/** @brief The size of the mapping.
 *
 * @return The header's region and both pixel regions, in bytes.
 */
extern STD(size_t)
ShmTotalBytes (void);

/** @brief One pass's overrides.
 *
 * Every field is present; `overrideMask` says which of them mean anything.
 */
struct PassControl {
	_Atomic(STD(uint32_t)) overrideMask;
	_Atomic(STD(uint32_t)) intensityBits;
	_Atomic(STD(uint32_t)) localToneBits;
	_Atomic(STD(uint32_t)) localStructureBits;
	_Atomic(STD(uint32_t)) skinStructureBits;
	_Atomic(STD(uint32_t)) sharpnessBits;
	_Atomic(STD(uint32_t)) style;
	_Atomic(STD(uint32_t)) preset;
	_Atomic(STD(uint32_t)) autoMask;
};

/** @brief The header at the start of the mapping. */
struct ShmHeader {
	_Atomic(STD(uint32_t)) magic;
	_Atomic(STD(uint32_t)) version;

	// The frame counts, first so that they start at multiples of eight bytes.
	_Atomic(STD(uint64_t)) helperFrames; //!< the frames the helper answered
	_Atomic(STD(uint64_t)) layerFrames;  //!< the frames the layer composed

	// The frame handshake. The layer bumps seq_req after writing a proxy; the helper answers by
	// storing the same number into seq_resp once the model's answer is in the output region.
	_Atomic(STD(uint32_t)) seq_req;
	_Atomic(STD(uint32_t)) seq_resp;
	_Atomic(STD(uint32_t)) width;
	_Atomic(STD(uint32_t)) height;
	_Atomic(STD(uint32_t)) quit;
	_Atomic(STD(uint32_t)) heartbeat;

	// Bumped by whoever writes a setting. The layer and the helper watch it rather than re-reading
	// thirty values every frame.
	_Atomic(STD(uint32_t)) controlSeq;

	// Bumped only when something the model latches at feature creation changes. The helper rebuilds
	// its features on this and debounces the rebuild; bumping it every frame exhausts the driver's
	// latches and the model stops responding until the process restarts.
	_Atomic(STD(uint32_t)) tuningSeq;

	// --- the model ---------------------------------------------------------------------------
	_Atomic(STD(uint32_t)) enabled;
	_Atomic(STD(uint32_t)) passes;
	_Atomic(STD(uint32_t)) preset;
	_Atomic(STD(uint32_t)) style;
	_Atomic(STD(uint32_t)) autoMask;
	_Atomic(STD(uint32_t)) intensityBits;
	_Atomic(STD(uint32_t)) localToneBits;
	_Atomic(STD(uint32_t)) localStructureBits;
	_Atomic(STD(uint32_t)) skinStructureBits;
	_Atomic(STD(uint32_t)) sharpnessBits;

	// --- the composition ---------------------------------------------------------------------
	// How much of the model's edit reaches the frame, and how much of it is allowed to be colour
	// rather than luminance. Separating the two is what keeps saturated highlights from shifting hue.
	_Atomic(STD(uint32_t)) transferStrengthBits;
	_Atomic(STD(uint32_t)) colourStrengthBits;
	// The most the pass may multiply or divide a pixel by. The transfer is a ratio, and a ratio
	// against a near-black proxy pixel is unbounded without one.
	_Atomic(STD(uint32_t)) maxRatioBits;
	// How a model that worked below the frame's size is brought back. 0 classic, 1 matched residual,
	// 2 native + edit -- the frame's own pixels with only the model's difference added, so what the
	// model left alone never passes through the enlargement.
	_Atomic(STD(uint32_t)) transfer;
	// 0 off, 1 the picture the model was shown, 2 its raw answer, 3 what it changed, amplified.
	_Atomic(STD(uint32_t)) debugView;
	_Atomic(STD(uint32_t)) debugScaleBits;
	_Atomic(STD(uint32_t)) whitePointBits;
	_Atomic(STD(uint32_t)) whitePointScaleBits;
	_Atomic(STD(uint32_t)) whitePointSource;
	_Atomic(STD(uint32_t)) whitePointTrimBits;
	// What fraction of the frame's resolution the model works at, 0.25..1. The frame itself is never
	// reduced: only the model's contribution is computed at this scale and resized, so the picture
	// underneath is untouched whatever this is.
	//
	// Below 1.0 it also cuts what crosses the shared memory, quadratically, which on this transport
	// matters more than it does upstream. Upstream goes up to 2.0, where the model supersamples; the
	// layer stops at 1.0.
	_Atomic(STD(uint32_t)) workingScaleBits;
	// 0 off, 1 side by side, 2 a wipe.
	_Atomic(STD(uint32_t)) compareMode;
	_Atomic(STD(uint32_t)) compareSplitBits;
	_Atomic(STD(uint32_t)) compareZoomBits;
	_Atomic(STD(uint32_t)) compareSwap;
	_Atomic(STD(uint32_t)) colourMode;
	// Writes one set of matched before/after frames per session when the layer next presents.
	_Atomic(STD(uint32_t)) captureRequest;

	// A Linux KEY_ code the layer watches to toggle the pass, or 0 for none. Unbound by default,
	// because a key that does something unexpected is worse than a key that does nothing.
	//
	// Only useful where the layer can read the keyboard at all: an X11 or XWayland session, inside
	// gamescope, or anywhere the user is in the 'input' group. A game presenting through winewayland
	// is a Wayland client whose keys never reach this process, and keyboards get no uaccess ACL, so
	// there the answer is a desktop shortcut bound to 'dlsslopctl --toggle enabled' instead.
	_Atomic(STD(uint32_t)) toggleKey;

	// Which proxy the model is shown, and whether its answer is composed or substituted. See
	// ReversibleMode. Default 0 keeps the picture identical to the pre-import behaviour.
	_Atomic(STD(uint32_t)) reversibleMode;

	// Whether the model's edit is applied at all. Off keeps the whole pass running -- the capture,
	// the round trip, the encode -- and simply presents the clean frame, which is what makes an
	// honest A/B possible: the cost is unchanged, so only the picture differs.
	_Atomic(STD(uint32_t)) applyModel;

	// Freeze the frame the pass works on, so changing a setting re-runs the composition over the
	// SAME picture instead of over whatever the game has drawn since. The only clean way to compare
	// two settings, and a live testing control rather than a saved preference.
	//
	// In this architecture it is cheaper than upstream: the layer already holds the captured proxy
	// and the model's last answer, so holding means not re-capturing rather than keeping a frame
	// alive somewhere it would not otherwise be.
	_Atomic(STD(uint32_t)) holdFrame;

	// --- status, written by the helper --------------------------------------------------------
	_Atomic(STD(uint32_t)) helperState;
	_Atomic(STD(uint32_t)) modelUp;
	_Atomic(STD(uint32_t)) helperEvalMsBits;
	_Atomic(STD(uint32_t)) helperUploadMsBits;
	_Atomic(STD(uint32_t)) helperReadbackMsBits;

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
	_Atomic(STD(uint32_t)) layerPid;
	_Atomic(STD(uint32_t)) layerWidth;
	_Atomic(STD(uint32_t)) layerHeight;
	_Atomic(STD(uint32_t)) layerCompositionUp;
	_Atomic(STD(uint32_t)) layerMsBits;
	_Atomic(STD(uint32_t)) layerMeasuredWhiteBits;

	// Free text, each guarded by its own sequence number, which ShmStoreString() bumps after it
	// writes the bytes. Nothing marks a field while it is written, so a reader that copies then can
	// take a torn or empty string; closing that race would change the protocol.
	_Atomic(STD(uint32_t)) helperReasonSeq;
	char                   helperReason[kReasonBytes];
	_Atomic(STD(uint32_t)) layerReasonSeq;
	char                   layerReason[kReasonBytes];
	_Atomic(STD(uint32_t)) gameNameSeq;
	char                   gameName[kNameBytes];

	struct PassControl     pass[kMaxPasses];

	// Appended after the pass array on purpose: everything before it has a pinned offset, and a new
	// field inserted higher up would move all of them. Motion vectors, from bmitch87's work.
	_Atomic(STD(uint32_t)) mvecEnabled;
	_Atomic(STD(uint32_t)) mvecQuality;
	// How far the helper has answered *successfully*. seq_resp says a frame came back; this says it
	// was worth using, so the layer can present the game's own frame when it was not.
	_Atomic(STD(uint32_t)) seq_ok;

	// 0: the composition blends the model's edit onto the frame under the strength and guard limits.
	// 1: no composition at all -- the model's raw answer IS the presented frame, and the limits,
	// enlargement and compare overlays are moot. Default 1: the composition is off until the user
	// turns it on.
	_Atomic(STD(uint32_t)) compositionBypass;

	// The raster the helper actually answered, echoed before seq_resp. More than one swapchain can
	// share this channel -- a game and the Steam overlay, or a game mid-resize with its old and new
	// swapchains both presenting -- and seq_resp only says *a* frame came back. Without the echo a
	// swapchain waiting on its own request can be satisfied by another's answer and copy the wrong
	// number of bytes, which is the row-shifted colour garbage this field exists to refuse.
	_Atomic(STD(uint32_t)) answeredW;
	_Atomic(STD(uint32_t)) answeredH;

	// The HDR input path. hdrMode is the user's choice (see HdrMode); hdrDetected and hdrKind are the
	// layer's reading of the primary swapchain's format and colour space; hdrActive is the decision
	// the layer actually encoded this frame under, and proxyFormat is what the helper built the
	// crossing images as. The layer uses the float16 path only while both agree it exists -- if the
	// model refused the float input, proxyFormat stays 8-bit and the encode tone maps as before.
	_Atomic(STD(uint32_t)) hdrMode;
	_Atomic(STD(uint32_t)) hdrDetected;
	_Atomic(STD(uint32_t)) hdrActive;
	_Atomic(STD(uint32_t)) proxyFormat;
	// What the proxy bytes in the shared region actually are for the request being made: the layer
	// writes this immediately before seq_req. Native HIP uses it directly as the payload
	// precision, without proxyFormat. Legacy Windows transport may lag hdrActive while
	// crossing images are rebuilt; mismatched frames are refused, never misread.
	_Atomic(STD(uint32_t)) hdrEncode;

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
	_Atomic(STD(uint32_t)) colourTrustPercent;

	// How much of the relighting ratio is taken from the pixel's neighbourhood instead of the pixel,
	// in hundredths. 0 is the behaviour that shipped before it existed.
	//
	// The composition rebuilds the frame as its own pixel times one per-pixel number. On detailed
	// content that number varies sharply, because the model's answer differs sharply there, and the
	// highlight guard is all that holds it -- so raising the guard lets the variation through as
	// blown and black pixels wearing whatever colour the texture had. Carrying a ratio at full
	// spatial frequency is the mistake: what the model knows at this scale is how much light belongs
	// here, not which pixel is brighter than its neighbour, and the frame already knows that.
	_Atomic(STD(uint32_t)) ratioSmoothPercent;

	// SDR uses 8-bit ping-pong images by default. Enable 16-bit UNORM to avoid quantising between
	// passes at higher memory and bandwidth cost.
	_Atomic(STD(uint32_t)) sdr16Multipass;
	_Atomic(STD(uint32_t)) mvecPixelSize;

	// Native worker capability, not a user scaling setting. Zero keeps the
	// original transport extent (the identity diagnostic mode).
	_Atomic(STD(uint32_t)) nativeModelMaxWidth;
	_Atomic(STD(uint32_t)) nativeModelMaxHeight;
	_Atomic(STD(uint32_t)) colorPreserveBits; //!< Native per-pass color anchoring, 0..1.

	// Native device-local transport. The layer offers its exported proxy and answer buffers on
	// ShmTransportPath() under a nonzero generation, and the daemon answers on the offer's own
	// connection (ShmTransportOffer). A request naming a generation in transportGen is read from
	// and answered into those buffers instead of the pixel regions (zero: the regions). A worker
	// that does not hold the named generation stores it in transportMiss and fails the frame, so
	// the layer offers again.
	_Atomic(STD(uint32_t)) transportGen;
	_Atomic(STD(uint32_t)) transportMiss;

	// Native neural raster height: 720, 900 or 1080. A controller stores the tier it wants; between
	// frames the worker rebuilds its network for a different one (helperState reads Starting
	// meanwhile) and publishes the new raster in nativeModelMaxWidth/Height. A value it cannot use
	// is overwritten with the active tier. Storing the active tier again changes nothing.
	_Atomic(STD(uint32_t)) nativeTier;
};

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
	STD(uint32_t) magic;          //!< kShmMagic
	STD(uint32_t) generation;     //!< nonzero; what requests name in transportGen
	STD(uint64_t) allocation[2];  //!< the proxy and answer memory sizes
	STD(uint64_t) size[2];        //!< their buffers' sizes, 1..allocation; frames up to the smaller fit
	STD(uint8_t)  deviceUuid[16]; //!< the exporter's VkPhysicalDeviceIDProperties
	STD(uint8_t)  driverUuid[16];
};

/** @brief What the transport's socket adds to the channel's path (ShmTransportPath()). */
#define kShmTransportSuffix ".sock"

/** @brief The socket on which the layer offers its transport.
 *
 * @param buf     Receives the path; may be nullptr if @a size is 0.
 * @param size    The size of @a buf.
 * @param channel The channel's path.
 * @return        What snprintf() returns: the path's length, which is @a size or more if @a buf
 *                holds only the start of it.
 */
extern int
ShmTransportPath (char        *buf,
                  STD(size_t)  size,
                  char const  *channel);

/** @brief A float's bits.
 *
 * @param f The float.
 * @return  Its bits, as a word of the channel holds them.
 */
extern STD(uint32_t)
FloatToBits (float f);

/** @brief The float that bits hold.
 *
 * @param u The bits, as a word of the channel holds them.
 * @return  The float.
 */
extern float
BitsToFloat (STD(uint32_t) u);

/** @brief Initializes a header with the defaults: those of a new channel, which shm_channel_open()
 *         gives it whichever process creates it.
 *
 * Gives each atomic object of the header its default with atomic_init(), and zeroes the text fields.
 * The atomic objects of a new file, of allocated storage or of an automatic header declared without
 * an initializer have no valid state until they are initialized (C23 7.17.2), and they are
 * initialized once: a channel is never initialized again, and a reset of its settings stores the
 * defaults of a header initialized here. No process can open a channel before its initialization
 * has finished: shm_channel_open() names the file only then.
 *
 * This is the single source of startup/reset/help defaults for the Linux HIP
 * backend. The CPU-composition and identity worker modes already return a final
 * image, so their compositor bypass default is one.
 *
 * @param h      The header, which holds no channel; every byte of it is written.
 * @param bypass The default of compositionBypass: true for the worker modes that return a final
 *               image, otherwise false.
 */
extern void
ShmInitNativeDefaults (struct ShmHeader *h,
                       bool              bypass);

/** @brief The tier of a height.
 *
 * @param height A tier's height.
 * @return       The tier of that height, or nullptr if no tier has it.
 */
extern struct NativeTier const *
ShmNativeTier (STD(uint32_t) height);

/** @brief The most passes a header may ask for.
 *
 * @param h The header.
 * @return  kMaxPasses.
 */
extern STD(uint32_t)
ShmPassCeiling (struct ShmHeader const *h);

/** @brief The passes a header asks for.
 *
 * @param h The header.
 * @return  passes, between 1 and ShmPassCeiling(); 0 counts as 1.
 */
extern STD(uint32_t)
ShmPasses (struct ShmHeader const *h);

/** @brief Whether a header asks for neural rendering.
 *
 * @param h The header.
 * @return  true if enabled is not 0.
 */
extern bool
ShmNeuralEnabled (struct ShmHeader const *h);

/** @brief The motion search quality that a header asks for.
 *
 * @param h The header.
 * @return  mvecQuality, or kMVecBalanced if it names no quality.
 */
extern STD(uint32_t)
ShmMVecQuality (struct ShmHeader const *h);

/** @brief The motion grid's spacing that a header asks for.
 *
 * @param h The header.
 * @return  mvecPixelSize, or kMVecPixels4 if it names no spacing.
 */
extern STD(uint32_t)
ShmMVecPixelSize (struct ShmHeader const *h);

/** @brief The header's text fields, each guarded by its own sequence number. */
enum shm_text : STD(uint32_t) {
	SHM_TEXT_HELPER_REASON, //!< helperReason, guarded by helperReasonSeq.
	SHM_TEXT_LAYER_REASON,  //!< layerReason, guarded by layerReasonSeq.
	SHM_TEXT_GAME_NAME,     //!< gameName, guarded by gameNameSeq.
};

/** @brief Publishes a string in one of the header's text fields.
 *
 * Writes the string, cut to the field's size less one byte and padded with zeros, then bumps the
 * field's sequence number. Nothing marks the field while the bytes are written, so a reader can take
 * a torn or empty string (see ShmLoadString()).
 *
 * @param h     The header.
 * @param field The field.
 * @param src   The string, or nullptr for an empty one.
 */
extern void
ShmStoreString (struct ShmHeader *h,
                enum shm_text     field,
                char const       *src);

/** @brief Reads a string that ShmStoreString() published.
 *
 * Copies the field and keeps the copy if its sequence number did not change meanwhile, trying four
 * times. A copy made while a writer writes the bytes passes, torn or empty, when the writer bumps
 * the sequence number only after the second load.
 *
 * @param h     The header.
 * @param field The field.
 * @param out   Receives the string, cut to @a size - 1 bytes, or an empty one if the sequence number
 *              changed during every copy.
 * @param size  The size of @a out; at least 1.
 * @return      true if the sequence number did not change during the copy that @a out holds.
 */
extern bool
ShmLoadString (struct ShmHeader const *h,
               enum shm_text           field,
               char                   *out,
               STD(size_t)             size);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_COMMON_SHM_PROTOCOL_H_ */
