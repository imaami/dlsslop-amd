#pragma once
// The DLSS-NR pass, arranged for a present-time layer.
//
// OptiScaler runs this immediately after the game's upscaler, on surfaces it already holds. Here the
// only thing available is a finished swapchain image, and the model lives in another process behind a
// shared-memory round trip, so the same pass has to be split in two around that round trip:
//
//   leg 1   swapchain -> frame -> ENCODE -> proxy, unless a display-referred UNORM frame is its own
//                              -> DOWNSAMPLE -> work, when the model runs below the frame or
//                                 reads the frame directly
//                              -> host buffer
//     ...   the helper runs the model on those pixels and answers
//   leg 2   host buffer -> model image -> RESOLVE -> composed -> swapchain
//
// What crosses the shared memory is the proxy, not the frame: it is display-referred and 8-bit by
// construction, so the helper never has to know what format the game presents in, and the working
// scale reduces it quadratically.
//
// The composition itself -- what the resolve does with the model's answer -- is entirely the vendored
// shader's. Everything in this file is plumbing: which image is bound where, in what layout, and what
// goes in the constant block.
#include "capture.h"
#include "dlssnr_pass.h"
#include "scaler_vk.h"
#include "vk_table.h"

#include "../common/shm_protocol.h"

#include <memory>
#include <string>

namespace dlssnr {

// One frame's worth of settings, read from the shared header once so that a control changed
// mid-frame cannot make the encode and the resolve disagree about what they are doing.
struct FrameSettings {
    uint32_t controlSeq = 0, tuningSeq = 0, passes = 0;
    float transferStrength = 1.0f;
    float colourStrength = 1.0f;
    float maxRatio = 2.0f;
    float debugScale = 1.0f;
    // The white point is three numbers, not one: where it comes from, the multiplier that says what
    // the model should consider white, and the trim that belongs to a measured reading rather than to
    // the slider. Keeping them apart is upstream's fix for a real bug -- sharing one stored value
    // meant touching the slider in one mode silently destroyed the number found in the other.
    float whitePointManual = 1.0f;
    float whitePointScale = 1.0f;
    float whitePointTrim = 1.0f;
    uint32_t whitePointSource = kWhitePointManual;
    float compareSplit = 0.5f;
    float compareZoom = 1.0f;
    float workingScale = 1.0f;
    uint32_t nativeModelMaxWidth = 0;
    uint32_t nativeModelMaxHeight = 0;
    uint32_t transfer = 1;
    uint32_t debugView = 0;
    uint32_t compareMode = 0;
    uint32_t compareSwap = 0;
    uint32_t reversibleMode = kReversibleKnee;
    uint32_t applyModel = 1;
    uint32_t holdFrame = 0;
    uint32_t downscaler = kScalerLanczos3;
    // 1: present the model's raw answer as the frame -- no blend, no guard, no compare.
    uint32_t compositionBypass = 0;

    // How much of the way toward a newly arrived answer the running pair moves each frame, 0..1.

    float ghostSlack = 0.5f;

    // Radius splitting the stale edit's safe half from the half that can ghost, in uv.
    float editBlur = 0.04f;

    float motionSmooth = 1.0f;

    // How much of the chroma-agreement gate to apply. See colourTrustPercent.
    float colourTrust = 1.0f;

    // How much of the relighting ratio comes from the neighbourhood. See ratioSmoothPercent.
    float ratioSmooth = 0.0f;


    static FrameSettings Read(const ShmHeader* h);
};

// Whether a swapchain format can be composed at all, and what this pass works in when it can.
//
// Every internal surface uses the format's UNORM twin rather than the swapchain's own: sampling a
// _SRGB view would decode to linear on the way in and re-encode on the way out, and the composition
// wants exactly the display-referred numbers the game already wrote. Copies between the two are
// byte-for-byte, which is why the final result goes back with vkCmdCopyImage and not a blit -- a blit
// into an _SRGB image would apply the encode a second time.
VkFormat CompositionFormat(VkFormat swapchainFormat);

// Whether the frame the game presents holds light or a picture. 8-bit and 10-bit formats are already
// display-referred -- the game tone mapped before it got here -- and only a float swapchain is linear.
bool ColourIsLinearHdr(VkFormat swapchainFormat, uint32_t colourMode);

class Composition {
  public:
    Composition(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                VkPhysicalDevice physicalDevice);
    ~Composition();

    Composition(const Composition&) = delete;
    Composition& operator=(const Composition&) = delete;

    bool Usable() const { return _usable; }

    // Why not, when not. Empty while it is working.
    const char* Reason() const { return _reason.c_str(); }

    // Build or rebuild everything sized to this frame and this model resolution. Cheap and a no-op
    // when nothing has changed, so it is safe to call every present.
    // hdrProxy asks for the float16 HDR proxy: the crossing surfaces become R16G16B16A16_SFLOAT and
    // the encode writes normalised linear light instead of an sRGB picture. hdrTransfer = 1 says the
    // swapchain itself carries PQ. Both are honoured only while the device can hold a float16 proxy;
    // Prepare reports the decision it actually made through HdrProxyActive().
    // Native HIP is the exception: its network cannot consume the NGX linear-light
    // float proxy, so the proxy stays sRGB-encoded at either precision and
    // hdrTransfer is honoured with an RGBA8 proxy too.
    bool Prepare(uint32_t width, uint32_t height, VkFormat swapchainFormat, const FrameSettings& s,
                 bool linearHdr, bool hdrProxy = false, uint32_t hdrTransfer = 0);

    // The raster the model will work at, for a caller that has to size a transport before Prepare
    // runs. Same arithmetic Prepare uses, in one place so the two cannot disagree.
    static void ModelExtent(uint32_t width, uint32_t height, const FrameSettings& s,
                            uint32_t& modelW, uint32_t& modelH);

    uint32_t ModelWidth() const { return _modelW; }
    uint32_t ModelHeight() const { return _modelH; }
    // Eight bytes a pixel while the float16 proxy is on, four when it is not -- every transport size
    // in the layer derives from this one number so the two sides cannot disagree.
    size_t ModelBytes() const { return size_t(_modelW) * _modelH * (_hdrProxy ? 8 : 4); }
    bool HdrProxyActive() const { return _hdrProxy; }
    uint32_t HdrTransfer() const { return _hdrProxy ? _hdrTransfer : 0; }

    // Native HIP: build the transport pair as exportable device-local memory the worker imports,
    // so the proxy and the answer never leave VRAM. family is the queue family both legs run on.
    void EnableExport(uint32_t family) { _export = true; _exportFamily = family; }
    // The pair is exported memory the worker has not acknowledged: no frame may use it before
    // an offer succeeds (ExportTransport, SetTransportReady) or DisableExport replaces it.
    bool TransportPending() const { return _download.allocation && !_transportReady; }
    // The acknowledged generation requests name, or zero for the host transport.
    uint32_t TransportGeneration() const { return _transportReady ? _transportGen : 0; }
    // The connection an offer of the pair awaits the daemon's answer on, or -1 (AwaitAnswer,
    // WithdrawOffer). The composition closes it.
    int OfferConnection() const { return _offer; }
    void AwaitAnswer(int connection) { WithdrawOffer(); _offer = connection; }
    void WithdrawOffer();
    // True when the daemon imported the pair; false when it holds none, so the pair is offered
    // again. Either ends an outstanding offer.
    void SetTransportReady(bool ready) { _transportReady = ready; WithdrawOffer(); }
    // Two new descriptors for the proxy and answer memory, which the caller owns, and the rest of
    // their offer, under a fresh generation number.
    bool ExportTransport(int fds[2], ShmTransportOffer& offer);
    // The worker cannot import the pair: carry frames through host staging from now on.
    void DisableExport();

    // Leg 1. Leaves the proxy the model should see in the download buffer, and the swapchain image
    // back in PRESENT_SRC_KHR so a caller that gives up after this still presents something valid.
    bool RecordCapture(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s);

    // The pixels leg 1 produced, and where the model's answer goes before leg 2.
    const void* ProxyPixels() const { return _download.mapped; }
    // The transfer pair, for a network recorded on this device: capture writes the proxy and
    // compose reads the answer. Exported, both belong to VK_QUEUE_FAMILY_EXTERNAL between uses.
    VkBuffer ProxyBuffer() const { return _download.buffer; }
    VkBuffer AnswerBuffer() const { return _upload.buffer; }
    bool TransportExported() const { return _download.allocation && _upload.allocation; }
    void* ModelPixels() { return _upload.mapped; }

    // Leg 2. Composes and leaves the swapchain image holding the result, in PRESENT_SRC_KHR.
    bool RecordCompose(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s);

    // Whether a frame has ever been captured from the swapchain.
    //
    // Asked by the idle repaint before it takes an image. Composing again means re-running what sits
    // downstream of the captured frame, and if nothing was ever captured there is nothing to re-run
    // it over.
    bool HasCapturedFrame() const { return _frameCaptured; }

    // Put the captured frame back on the screen, untouched.
    //
    // For the case where the effect is switched off while the picture is still. The composed result
    // is what is sitting in the swapchain, and nothing will overwrite it until the application draws
    // again -- so switching off leaves the edit on screen indefinitely. This is the frame as it was
    // read, so it restores exactly what the application drew.
    bool RecordRestore(VkCommandBuffer cb, VkImage swapchainImage);

    bool HasModelFrame() const { return _haveModel; }
    void MarkModelFrame() { _haveModel = true; }

    // Write this many matched before/after pairs, starting with the next composed frame.
    void RequestCapture(uint32_t frames, uint32_t controlSeq) { _capture.Begin(frames, controlSeq); }
    void SetCaptureInference(uint32_t seq) { _captureMetadata.inferenceSeq = seq; }
    bool CaptureActive() const { return _capture.Active(); }
    // This frame's compose recorded a pair; an active capture without its host buffer records none.
    bool CaptureRecorded() const { return _captureRecorded; }

    // Called after leg 2's fence, when the readback the compose recorded has landed.
    void WriteCapturedFrame();

    // Called after leg 1's fence: turns the tile grid the meter wrote into a white point.
    void ConsumeMeter();

    // What the meter settled on, or 0 when it has not taken a usable reading. For the interface, so
    // the number in use is visible rather than inferred.
    float MeasuredWhitePoint() const { return _measuredWhitePoint; }

  private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0, height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct HostBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        size_t size = 0;
        // Nonzero, the allocation size, when the buffer is exported device-local memory.
        VkDeviceSize allocation = 0;
    };

    bool MakeImage(Image& img, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage);
    void DropImage(Image& img);
    bool MakeHostBuffer(HostBuffer& buf, size_t bytes, VkBufferUsageFlags usage);
    bool MakeExportBuffer(HostBuffer& buf, size_t bytes);
    void ExternalOwnership(VkCommandBuffer cb, const HostBuffer& buf, uint32_t from, uint32_t to,
                           VkAccessFlags access);
    void DropHostBuffer(HostBuffer& buf);
    void EnsureTransport();
    void DropAll();

    bool BuildMeterPipeline();
    void DropMeterObjects();
    bool BuildMeterDescriptors();
    bool MakeMeterState();
    void DropMeterState();

    void Transition(VkCommandBuffer cb, Image& img, VkImageLayout to);
    void TransitionSwapchain(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to);
    bool FormatSupportsBlit(VkFormat format) const;
    void CopyWholeImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst,
                        VkImageLayout dstLayout, uint32_t w, uint32_t h);

    bool FormatSupportsStorage(VkFormat format) const;
    DlssNrConstants BaseConstants(const FrameSettings& s) const;
    float ResolvedWhitePoint(const FrameSettings& s) const;

    const DeviceTable* _vk = nullptr;
    const InstanceTable* _instance = nullptr;
    VkDevice _device = VK_NULL_HANDLE;
    VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;

    std::unique_ptr<DlssNrPass> _pass;
    bool _usable = false;
    std::string _reason;

    uint32_t _width = 0, _height = 0;
    uint32_t _modelW = 0, _modelH = 0;
    VkFormat _swapchainFormat = VK_FORMAT_UNDEFINED;
    VkFormat _workFormat = VK_FORMAT_UNDEFINED;  // usually the swapchain's UNORM twin; see Prepare
    bool _blitSwapchain = false;                 // set when _workFormat is not the swapchain's twin
    bool _linearHdr = false;
    bool _hdrProxy = false;
    uint32_t _hdrTransfer = 0;
    bool _haveModel = false;

    Image _frame{}, _proxy{}, _work{}, _model{}, _composed{};

    // Supersampling: the model works above the frame, so the proxy is enlarged on the way in and the
    // answer averaged back on the way out. _modelNative holds that average; without it the resolve
    // would read the larger answer through a bilinear sampler and alias, which is the reason upstream
    // gave this its own filter rather than reusing the resolve's.
    Image _modelNative{};
    std::unique_ptr<ScalerVk> _superUp, _superDown;
    bool _superSample = false;
    uint32_t _scalerFilter = kScalerLanczos3;

    // The white point meter: a grid of tile peak luminances measured off the captured frame, and the
    // percentile taken across it -- on the GPU. The reduce pass keeps the percentile, the history
    // and the resolved value in a device-local state buffer; the resolve reads the resolved value
    // through a four-byte copy into its own constant block, so no tensor crosses to the host. The
    // mirror is a 128-byte copy of the state the CPU reads after leg 1's fence, for the frame-hold
    // snapshot and the status field only. See ConsumeMeter.
    Image _meter{};
    bool _meterGpu = false;
    bool _meterStateCleared = false;
    VkBuffer _meterState = VK_NULL_HANDLE;
    VkDeviceMemory _meterStateMemory = VK_NULL_HANDLE;
    HostBuffer _meterMirror{};
    VkPipeline _meterPipeline = VK_NULL_HANDLE;
    VkPipelineLayout _meterPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout _meterDescriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool _meterDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet _meterDescriptorSet = VK_NULL_HANDLE;
    VkSampler _meterSampler = VK_NULL_HANDLE;
    float _measuredWhitePoint = 0.0f;
    float _meterSteadiness = 0.0f;
    HostBuffer _download{}, _upload{}, _captureBuf{};

    // The exported native transport (EnableExport).
    bool _export = false;
    bool _transportReady = false;
    int _offer = -1;
    uint32_t _exportFamily = 0;
    uint32_t _transportGen = 0;

    CaptureWriter _capture;
    CaptureMetadata _captureMetadata{};
    bool _captureRecorded = false;

    // Frame hold. The freeze point is the raw colour the encode reads, not the proxy: the proxy is
    // derived from it and the resolve reads it, and freezing further down would stop a setting
    // change from re-encoding, which is the whole point of holding.
    bool _holding = false;
    bool _frameCaptured = false;
    float _heldWhitePoint = 1.0f;
};

}  // namespace dlssnr
