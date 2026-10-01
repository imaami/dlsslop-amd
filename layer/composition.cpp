#include "composition.h"
#include "log.h"
#include "shaders/meter_reduce_spv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <unistd.h>
#include <sys/random.h>

namespace dlssnr {

// The meter state buffer and its host mirror share one layout, pinned to match
// the MeterState block in shaders/meter_reduce.comp.
namespace {
constexpr size_t kMeterStateBytes = 128;
constexpr VkDeviceSize kMeterResolvedOffset = 8;

struct MeterPush {
    float manual;
    float scale;
    float trim;
    float holdValue;
    uint32_t source;
    uint32_t hold;
};
static_assert(sizeof(MeterPush) == 24, "must match the push_constant block in meter_reduce.comp");
}  // namespace

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------
VkFormat CompositionFormat(VkFormat swapchainFormat) {
    switch (swapchainFormat) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        default:
            return VK_FORMAT_UNDEFINED;
    }
}

bool ColourIsLinearHdr(VkFormat swapchainFormat, uint32_t colourMode) {
    if (colourMode == kColourDisplay) return false;
    if (colourMode == kColourLinearHdr) return true;
    // Auto. An 8-bit frame has been tone mapped or there would be nothing to see, and HDR10's
    // ten-bit formats carry PQ, which is display-referred as well. Only a float swapchain is light.
    return swapchainFormat == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
FrameSettings FrameSettings::Read(const ShmHeader* h) {
    FrameSettings s;
    if (!h) return s;
    s.controlSeq = h->controlSeq.load();
    s.tuningSeq = h->tuningSeq.load();
    s.passes = h->passes.load();
    s.transferStrength = BitsToFloat(h->transferStrengthBits.load());
    s.colourStrength = BitsToFloat(h->colourStrengthBits.load());
    s.maxRatio = BitsToFloat(h->maxRatioBits.load());
    s.debugScale = BitsToFloat(h->debugScaleBits.load());
    s.compareSplit = BitsToFloat(h->compareSplitBits.load());
    s.compareZoom = BitsToFloat(h->compareZoomBits.load());
    s.workingScale = BitsToFloat(h->workingScaleBits.load());
    s.nativeModelMaxWidth = h->nativeModelMaxWidth.load();
    s.nativeModelMaxHeight = h->nativeModelMaxHeight.load();
    if (s.nativeModelMaxWidth < kMinW || s.nativeModelMaxWidth > kMaxW ||
        s.nativeModelMaxHeight < kMinH || s.nativeModelMaxHeight > kMaxH) {
        s.nativeModelMaxWidth = 0;
        s.nativeModelMaxHeight = 0;
    }
    s.transfer = h->transfer.load();
    s.debugView = h->debugView.load();
    s.compareMode = h->compareMode.load();
    s.compareSwap = h->compareSwap.load();
    s.reversibleMode = h->reversibleMode.load();
    s.applyModel = h->applyModel.load();
    s.holdFrame = h->holdFrame.load();
    s.downscaler = h->scalingDownscaler.load();
    s.compositionBypass = h->compositionBypass.load();
    {
        static const int forced = [] {
            const char* v = getenv("DLSSNR_GHOST_SLACK");
            return v && *v ? atoi(v) : -1;
        }();
        if (forced >= 0) s.ghostSlack = float(forced) / 100.0f;
        if (!std::isfinite(s.ghostSlack) || s.ghostSlack < 0.0f) s.ghostSlack = 0.5f;
    }
    {
        s.ratioSmooth = float(h->ratioSmoothPercent.load()) / 100.0f;
        static const int forcedRs = [] {
            const char* v = getenv("DLSSNR_RATIO_SMOOTH");
            return v && *v ? atoi(v) : -1;
        }();
        if (forcedRs >= 0) s.ratioSmooth = float(forcedRs) / 100.0f;
        if (!std::isfinite(s.ratioSmooth) || s.ratioSmooth < 0.0f) s.ratioSmooth = 0.0f;
        if (s.ratioSmooth > 1.0f) s.ratioSmooth = 1.0f;

        s.colourTrust = float(h->colourTrustPercent.load()) / 100.0f;
        static const int forcedCt = [] {
            const char* v = getenv("DLSSNR_COLOUR_TRUST");
            return v && *v ? atoi(v) : -1;
        }();
        if (forcedCt >= 0) s.colourTrust = float(forcedCt) / 100.0f;
        if (!std::isfinite(s.colourTrust) || s.colourTrust < 0.0f) s.colourTrust = 1.0f;
        if (s.colourTrust > 8.0f) s.colourTrust = 8.0f;

        static const int forced = [] {
            const char* v = getenv("DLSSNR_MOTION_SMOOTH");
            return v && *v ? atoi(v) : -1;
        }();
        if (forced >= 0) s.motionSmooth = float(forced) / 100.0f;
        if (!std::isfinite(s.motionSmooth) || s.motionSmooth < 0.0f) s.motionSmooth = 0.0f;
        if (s.motionSmooth > 1.0f) s.motionSmooth = 1.0f;
    }
    {
        static const int forced = [] {
            const char* v = getenv("DLSSNR_EDIT_BLUR");
            return v && *v ? atoi(v) : -1;
        }();
        if (forced >= 0) s.editBlur = float(forced) / 1000.0f;
        if (!std::isfinite(s.editBlur) || s.editBlur < 0.0f) s.editBlur = 0.0f;
        if (s.editBlur > 0.25f) s.editBlur = 0.25f;
    }


    s.whitePointManual = BitsToFloat(h->whitePointBits.load());
    s.whitePointScale = BitsToFloat(h->whitePointScaleBits.load());
    s.whitePointTrim = BitsToFloat(h->whitePointTrimBits.load());
    s.whitePointSource = h->whitePointSource.load();

    // Clamped here rather than trusted, because these come from a file any process can write.
    const auto clamp = [](float v, float lo, float hi, float fallback) {
        if (!std::isfinite(v)) return fallback;
        return std::min(std::max(v, lo), hi);
    };
    s.transferStrength = clamp(s.transferStrength, 0.0f, 4.0f, 1.0f);
    s.colourStrength = clamp(s.colourStrength, 0.0f, 4.0f, 1.0f);
    s.maxRatio = clamp(s.maxRatio, 1.0f, float(kMaxPasses), 2.0f);
    s.debugScale = clamp(s.debugScale, 0.01f, 100.0f, 1.0f);
    s.whitePointManual = clamp(s.whitePointManual, 1e-4f, 2000.0f, 1.0f);
    s.whitePointScale = clamp(s.whitePointScale, 0.01f, 100.0f, 1.0f);
    s.whitePointTrim = clamp(s.whitePointTrim, 0.01f, 100.0f, 1.0f);
    if (s.whitePointSource > kWhitePointMeasured) s.whitePointSource = kWhitePointManual;
    s.compareSplit = clamp(s.compareSplit, 0.0f, 1.0f, 0.5f);
    s.compareZoom = clamp(s.compareZoom, 1.0f, 2.0f, 1.0f);

    // Above 1.0 the model supersamples, up to upstream's 2x ceiling.
    s.workingScale = clamp(s.workingScale, 0.25f, 2.0f, 1.0f);
    if (s.downscaler >= kScalerCount || s.downscaler == kScalerFsr1) s.downscaler = kScalerLanczos3;

    // Native + edit is mode 2; the clamp used to stop at 1 and silently killed it.
    if (s.transfer > 2) s.transfer = 2;
    // 4 and 5 are the two views of the colour bound. This clamp is why they did nothing when they
    // were added: the shader grew the cases and the validation did not, so the GUI offered them, the
    // header carried them, and the layer quietly rewrote them to 0 on the way past.
    if (s.debugView > 5) s.debugView = 0;
    if (s.compareMode > 2) s.compareMode = 0;
    if (s.reversibleMode >= kReversibleModeCount) s.reversibleMode = kReversibleKnee;
    return s;
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------
Composition::Composition(const device_table* vk, const instance_table* instance, VkDevice device,
                         VkPhysicalDevice physicalDevice)
    : _vk(vk), _instance(instance), _device(device), _physicalDevice(physicalDevice) {
    if (!device_table_complete(vk)) {
        _reason = "the device does not expose everything a compute pass needs";
        log_printf("[comp] %s", _reason.c_str());
        return;
    }

    _pass = std::make_unique<DlssNrPass>(vk, instance, device, physicalDevice);
    if (!_pass->CanRender()) {
        _reason = "the composition pipeline could not be built";
        _pass.reset();
        return;
    }

    _usable = true;
}

Composition::~Composition() {
    DropAll();
    DropMeterObjects();
    _pass.reset();
}

void Composition::DropAll() {
    DropHostBuffer(_captureBuf);
    DropImage(_frame);
    DropImage(_proxy);
    DropImage(_work);
    DropImage(_model);
    DropImage(_composed);
    DropImage(_modelNative);
    DropImage(_meter);
    DropMeterState();
    DropHostBuffer(_download);
    DropHostBuffer(_upload);
    WithdrawOffer();
    _superUp.reset();
    _superDown.reset();
    _superSample = false;
    _width = _height = _modelW = _modelH = 0;
    _frameCaptured = false;
    _captureRecorded = false;
    _measuredWhitePoint = 0.0f;
}

bool Composition::FormatSupportsStorage(VkFormat format) const {
    VkFormatProperties props{};
    _instance->vkGetPhysicalDeviceFormatProperties(_physicalDevice, format, &props);
    return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

// Both directions, because the swapchain is the source of the capture and the destination of the
// composition, and a blit needs the format to allow each end it is used at.
bool Composition::FormatSupportsBlit(VkFormat format) const {
    VkFormatProperties props{};
    _instance->vkGetPhysicalDeviceFormatProperties(_physicalDevice, format, &props);
    const VkFormatFeatureFlags both = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    return (props.optimalTilingFeatures & both) == both;
}

bool Composition::MakeImage(Image& img, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage) {
    DropImage(img);

    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (_vk->vkCreateImage(_device, &ci, nullptr, &img.image) != VK_SUCCESS) {
        log_printf("[comp] vkCreateImage %ux%u fmt=%d failed", w, h, (int) format);
        return false;
    }

    VkMemoryRequirements req{};
    _vk->vkGetImageMemoryRequirements(_device, img.image, &req);

    VkPhysicalDeviceMemoryProperties mp{};
    _instance->vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) { DropImage(img); return false; }

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (_vk->vkAllocateMemory(_device, &mai, nullptr, &img.memory) != VK_SUCCESS) {
        log_printf("[comp] out of device memory for a %ux%u surface", w, h);
        DropImage(img);
        return false;
    }
    if (_vk->vkBindImageMemory(_device, img.image, img.memory, 0) != VK_SUCCESS) {
        DropImage(img);
        return false;
    }

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (_vk->vkCreateImageView(_device, &vi, nullptr, &img.view) != VK_SUCCESS) {
        DropImage(img);
        return false;
    }

    img.format = format;
    img.width = w;
    img.height = h;
    img.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void Composition::DropImage(Image& img) {
    if (img.view) _vk->vkDestroyImageView(_device, img.view, nullptr);
    if (img.image) _vk->vkDestroyImage(_device, img.image, nullptr);
    if (img.memory) _vk->vkFreeMemory(_device, img.memory, nullptr);
    img = Image{};
}

bool Composition::MakeHostBuffer(HostBuffer& buf, size_t bytes, VkBufferUsageFlags usage) {
    DropHostBuffer(buf);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (_vk->vkCreateBuffer(_device, &bci, nullptr, &buf.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    _vk->vkGetBufferMemoryRequirements(_device, buf.buffer, &req);

    VkPhysicalDeviceMemoryProperties mp{};
    _instance->vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &mp);

    // Host visible and coherent is the requirement; cached is a large win on the readback and
    // harmless on the upload, so it is preferred rather than demanded.
    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int best = -1, bestScore = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & required) != required) continue;
        int score = 0;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += 100;
        if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) score -= 50;
        if (score > bestScore) { bestScore = score; best = (int) i; }
    }
    if (best < 0) { DropHostBuffer(buf); return false; }

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = uint32_t(best);
    if (_vk->vkAllocateMemory(_device, &mai, nullptr, &buf.memory) != VK_SUCCESS) {
        DropHostBuffer(buf);
        return false;
    }
    if (_vk->vkBindBufferMemory(_device, buf.buffer, buf.memory, 0) != VK_SUCCESS ||
        _vk->vkMapMemory(_device, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped) != VK_SUCCESS) {
        DropHostBuffer(buf);
        return false;
    }

    buf.size = bytes;
    return true;
}

void Composition::DropHostBuffer(HostBuffer& buf) {
    if (buf.mapped) _vk->vkUnmapMemory(_device, buf.memory);
    if (buf.buffer) _vk->vkDestroyBuffer(_device, buf.buffer, nullptr);
    if (buf.memory) _vk->vkFreeMemory(_device, buf.memory, nullptr);
    buf = HostBuffer{};
}

// Native transport: device-local memory exported as an opaque fd the daemon imports, so neither
// side copies the frame through host memory. Both buffers are made as ShmTransportOffer states, for
// a Vulkan importer to repeat.
bool Composition::MakeExportBuffer(HostBuffer& buf, size_t bytes) {
    DropHostBuffer(buf);
    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (_vk->vkCreateBuffer(_device, &bci, nullptr, &buf.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    _vk->vkGetBufferMemoryRequirements(_device, buf.buffer, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    _instance->vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &mp);
    uint32_t type = 0;
    while (type < mp.memoryTypeCount && (!(req.memoryTypeBits & (1u << type)) ||
           !(mp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
        ++type;
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = buf.buffer;
    VkExportMemoryAllocateInfo exp{};
    exp.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exp.pNext = &dedicated;
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &exp;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (type == mp.memoryTypeCount ||
        _vk->vkAllocateMemory(_device, &mai, nullptr, &buf.memory) != VK_SUCCESS ||
        _vk->vkBindBufferMemory(_device, buf.buffer, buf.memory, 0) != VK_SUCCESS) {
        DropHostBuffer(buf);
        return false;
    }
    buf.size = bytes;
    buf.allocation = req.size;
    return true;
}

bool Composition::ExportTransport(int fds[2], ShmTransportOffer& offer) {
    // Opaque fds import only on the device and driver that made them: the offer names both.
    if (!_instance->vkGetPhysicalDeviceProperties2) return false;
    VkPhysicalDeviceIDProperties ids{};
    ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &ids;
    _instance->vkGetPhysicalDeviceProperties2(_physicalDevice, &properties);
    std::memcpy(offer.deviceUuid, ids.deviceUUID, sizeof offer.deviceUuid);
    std::memcpy(offer.driverUuid, ids.driverUUID, sizeof offer.driverUuid);
    const HostBuffer* pair[2] = { &_download, &_upload };
    for (int i = 0; i < 2; ++i) {
        VkMemoryGetFdInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        info.memory = pair[i]->memory;
        info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        if (_vk->vkGetMemoryFdKHR(_device, &info, &fds[i]) != VK_SUCCESS) {
            if (i) close(fds[0]);
            return false;
        }
        offer.allocation[i] = pair[i]->allocation;
        offer.size[i] = pair[i]->size;
    }
    // A fresh number per offer: a restarted worker's acknowledgement can never be an old one.
    // No random source means no offer; the low bit keeps the number nonzero (zero is the host
    // transport).
    if (getrandom(&_transportGen, sizeof _transportGen, 0) != sizeof _transportGen) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    offer.generation = _transportGen |= 1u;
    return true;
}

void Composition::WithdrawOffer() {
    if (_offer >= 0) close(_offer);
    _offer = -1;
}

void Composition::DisableExport() {
    _export = false;
    DropHostBuffer(_download);
    DropHostBuffer(_upload);
    EnsureTransport();
}

// Exported memory changes hands with the worker at every leg: acquired before a copy touches it
// and released after. The worker's own accesses are ordered by the fences and futexes around them.
void Composition::ExternalOwnership(VkCommandBuffer cb, const HostBuffer& buf, uint32_t from, uint32_t to,
                                    VkAccessFlags access) {
    if (!buf.allocation) return;
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = access;
    b.dstAccessMask = access;
    b.srcQueueFamilyIndex = from;
    b.dstQueueFamilyIndex = to;
    b.buffer = buf.buffer;
    b.size = VK_WHOLE_SIZE;
    _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 1, &b, 0, nullptr);
}

// Build the transport pair at the model size: exported device-local memory while the export is
// on, private host-visible staging otherwise. Called from Prepare (which knows the model size)
// and from DisableExport, after each has dropped the old pair.
void Composition::EnsureTransport() {
    if (!_frame.image) return;
    const size_t bytes = ModelBytes();
    _transportReady = false;
    WithdrawOffer(); // New buffers: any offer was of the old ones.
    if (_export && MakeExportBuffer(_download, bytes) && MakeExportBuffer(_upload, bytes)) return;
    // MakeHostBuffer drops an exported buffer that a failed pair left behind.
    // Both ways: the in-layer network reads the proxy and writes the answer between the copies.
    constexpr VkBufferUsageFlags kTransfer = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    MakeHostBuffer(_download, bytes, kTransfer);
    MakeHostBuffer(_upload, bytes, kTransfer);
}

// ---------------------------------------------------------------------------
// The white-point meter, on the GPU
// ---------------------------------------------------------------------------
// The reduce pipeline is size-independent and survives rebuilds; only the state buffer, the mirror
// and the descriptor's image binding follow the meter image.
bool Composition::BuildMeterPipeline() {
    if (_meterPipeline) return true;
    DropMeterObjects();  // whatever a failed attempt left behind
    if (!_vk->vkCreateShaderModule || !_vk->vkCmdPushConstants || !_vk->vkCmdFillBuffer) return false;

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = kMeterReduceSpvLen * sizeof(uint32_t);
    smci.pCode = kMeterReduceSpv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (_vk->vkCreateShaderModule(_device, &smci, nullptr, &module) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 2;
    dli.pBindings = bindings;
    if (_vk->vkCreateDescriptorSetLayout(_device, &dli, nullptr, &_meterDescriptorLayout) != VK_SUCCESS) {
        _vk->vkDestroyShaderModule(_device, module, nullptr);
        return false;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(MeterPush);
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &_meterDescriptorLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (_vk->vkCreatePipelineLayout(_device, &pli, nullptr, &_meterPipelineLayout) != VK_SUCCESS) {
        _vk->vkDestroyShaderModule(_device, module, nullptr);
        return false;
    }

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (_vk->vkCreateSampler(_device, &sci, nullptr, &_meterSampler) != VK_SUCCESS) {
        _vk->vkDestroyShaderModule(_device, module, nullptr);
        return false;
    }

    const VkDescriptorPoolSize poolSizes[] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
                                               { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 } };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    if (_vk->vkCreateDescriptorPool(_device, &dpci, nullptr, &_meterDescriptorPool) != VK_SUCCESS) {
        _vk->vkDestroyShaderModule(_device, module, nullptr);
        return false;
    }
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = _meterDescriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &_meterDescriptorLayout;
    if (_vk->vkAllocateDescriptorSets(_device, &dsai, &_meterDescriptorSet) != VK_SUCCESS) {
        _vk->vkDestroyShaderModule(_device, module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.layout = _meterPipelineLayout;
    const VkResult built = _vk->vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpi, nullptr,
                                                         &_meterPipeline);
    _vk->vkDestroyShaderModule(_device, module, nullptr);
    return built == VK_SUCCESS && _meterPipeline != VK_NULL_HANDLE;
}

void Composition::DropMeterObjects() {
    if (_meterPipeline && _vk->vkDestroyPipeline) _vk->vkDestroyPipeline(_device, _meterPipeline, nullptr);
    if (_meterPipelineLayout) _vk->vkDestroyPipelineLayout(_device, _meterPipelineLayout, nullptr);
    if (_meterDescriptorLayout) _vk->vkDestroyDescriptorSetLayout(_device, _meterDescriptorLayout, nullptr);
    if (_meterDescriptorPool) _vk->vkDestroyDescriptorPool(_device, _meterDescriptorPool, nullptr);
    if (_meterSampler) _vk->vkDestroySampler(_device, _meterSampler, nullptr);
    _meterPipeline = VK_NULL_HANDLE;
    _meterPipelineLayout = VK_NULL_HANDLE;
    _meterDescriptorLayout = VK_NULL_HANDLE;
    _meterDescriptorPool = VK_NULL_HANDLE;
    _meterDescriptorSet = VK_NULL_HANDLE;
    _meterSampler = VK_NULL_HANDLE;
}

bool Composition::BuildMeterDescriptors() {
    if (!_meterDescriptorSet || !_meter.view || !_meterState) return false;
    VkDescriptorImageInfo grid{ _meterSampler, _meter.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo state{ _meterState, 0, VK_WHOLE_SIZE };
    const VkWriteDescriptorSet writes[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, _meterDescriptorSet, 0, 0, 1,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &grid, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, _meterDescriptorSet, 1, 0, 1,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &state, nullptr },
    };
    _vk->vkUpdateDescriptorSets(_device, 2, writes, 0, nullptr);
    return true;
}

// The meter's device-local state (percentile, history, resolved value) plus the 128-byte host
// mirror the CPU reads after leg 1's fence -- for the frame-hold snapshot and the status field.
bool Composition::MakeMeterState() {
    DropMeterState();
    if (!BuildMeterPipeline()) return false;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = kMeterStateBytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;  // vkCmdFillBuffer clears it
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (_vk->vkCreateBuffer(_device, &bci, nullptr, &_meterState) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    _vk->vkGetBufferMemoryRequirements(_device, _meterState, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    _instance->vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        _vk->vkDestroyBuffer(_device, _meterState, nullptr);
        _meterState = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (_vk->vkAllocateMemory(_device, &mai, nullptr, &_meterStateMemory) != VK_SUCCESS ||
        _vk->vkBindBufferMemory(_device, _meterState, _meterStateMemory, 0) != VK_SUCCESS) {
        DropMeterState();
        return false;
    }
    if (!MakeHostBuffer(_meterMirror, kMeterStateBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        DropMeterState();
        return false;
    }
    if (!BuildMeterDescriptors()) {
        DropMeterState();
        return false;
    }
    _meterStateCleared = false;
    _meterGpu = true;
    return true;
}

void Composition::DropMeterState() {
    _meterGpu = false;
    DropHostBuffer(_meterMirror);
    if (_meterState) _vk->vkDestroyBuffer(_device, _meterState, nullptr);
    if (_meterStateMemory) _vk->vkFreeMemory(_device, _meterStateMemory, nullptr);
    _meterState = VK_NULL_HANDLE;
    _meterStateMemory = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------
// The model works at this fraction of the frame.
//
// No rounding to a workgroup multiple: every dispatch here covers a partial group and the shader
// bounds-checks against gWidth/gHeight, so alignment buys nothing -- and rounding *up* was worse
// than nothing, because at a scale of exactly 1.0 it pushed a 500-pixel frame to 504 and quietly
// engaged supersampling on a setting that means "leave it alone". A floor of 64 only stops a
// pathologically small window from producing a degenerate raster.
void Composition::ModelExtent(uint32_t width, uint32_t height, const FrameSettings& s,
                              uint32_t& modelW, uint32_t& modelH) {
    if (s.nativeModelMaxWidth && s.nativeModelMaxHeight && width && height) {
        // Keep the low-resolution source and answer together. Upscaling the
        // answer inside the worker hid its true resolution from the resolve
        // shader and bypassed the native-detail-preserving transfer branch.
        const double scale = std::min({double(s.workingScale), 1.0,
            double(s.nativeModelMaxWidth) / width,
            double(s.nativeModelMaxHeight) / height});
        modelW = std::max<uint32_t>(kMinW, uint32_t(std::lround(width * scale)));
        modelH = std::max<uint32_t>(kMinH, uint32_t(std::lround(height * scale)));
        return;
    }
    const auto scaled = [&](uint32_t v) {
        if (s.workingScale == 1.0f) return v;
        return std::max<uint32_t>(64, uint32_t(std::lround(double(v) * double(s.workingScale))));
    };
    modelW = scaled(width);
    modelH = scaled(height);
}

bool Composition::Prepare(uint32_t width, uint32_t height, VkFormat swapchainFormat, const FrameSettings& s,
                          bool linearHdr, bool hdrProxy, uint32_t hdrTransfer) {
    if (!_usable) return false;

    VkFormat work = CompositionFormat(swapchainFormat);
    if (work == VK_FORMAT_UNDEFINED) {
        _reason = "unsupported swapchain format";
        return false;
    }

    // The composed surface is written as a storage image and then handed back to the swapchain. When
    // the swapchain's own UNORM twin can be written that way, that is what everything internal uses:
    // it shares the swapchain's bit layout, so both ends are a byte-for-byte copy and nothing is
    // reinterpreted.
    //
    // Not every presentable format can be written as a storage image, though. NVIDIA does not expose
    // A2R10G10B10_UNORM_PACK32 that way, and that is exactly what a 10-bit desktop hands most games
    // -- so the pass used to switch itself off, for the whole run, on the machines it was written
    // for. Compose in half float in that case and blit at both ends instead: the blit converts, and
    // sixteen bits a channel hold more than the ten the swapchain can show, so nothing is lost that
    // the display could have displayed.
    bool blit = false;
    if (!FormatSupportsStorage(work)) {
        const VkFormat wide = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (FormatSupportsStorage(wide) && FormatSupportsBlit(wide) && FormatSupportsBlit(swapchainFormat)) {
            // Prepare runs every frame; this is only news when the swapchain changed under it.
            if (swapchainFormat != _swapchainFormat)
                log_printf("[comp] format %d cannot be written as a storage image here; composing in half float",
                           (int) work);
            work = wide;
            blit = true;
        } else {
            _reason = "this device cannot write the swapchain's format as a storage image";
            log_printf("[comp] %s (format %d)", _reason.c_str(), (int) work);
            _usable = false;
            return false;
        }
    }

    // The build declares write-only storage images formatless and the device enables
    // shaderStorageImageWriteWithoutFormat, so each declaration matches the bound view's
    // actual format (RGBA8, BGRA8 or float) without changing shader arithmetic.
    uint32_t modelW = 0, modelH = 0;
    ModelExtent(width, height, s, modelW, modelH);
    const bool superSample = modelW > width || modelH > height;

    // The float16 proxy needs a surface the shader can write and sample as float. Where the device
    // says it cannot, the request quietly becomes the 8-bit arrangement that shipped before -- the
    // same shape as every other capability step in this file.
    if (hdrProxy) {
        VkFormatProperties fp16{};
        _instance->vkGetPhysicalDeviceFormatProperties(_physicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT, &fp16);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((fp16.optimalTilingFeatures & need) != need) {
            if (!_hdrProxy) log_printf("[comp] float16 proxy requested but not supported here; staying 8-bit");
            hdrProxy = false;
        }
    }

    if (_width == width && _height == height && _swapchainFormat == swapchainFormat &&
        _modelW == modelW && _modelH == modelH && _linearHdr == linearHdr && _hdrProxy == hdrProxy &&
        _hdrTransfer == hdrTransfer && _scalerFilter == s.downscaler && _frame.image)
        return true;

    log_printf("[comp] building %ux%u, model %ux%u, %s%s%s", width, height, modelW, modelH,
               linearHdr ? "linear HDR" : "display-referred",
               hdrProxy ? (hdrTransfer ? ", float16 proxy (PQ in)" : ", float16 proxy") : "",
               superSample ? " (supersampling)" : "");

    if (superSample) {
        // Said out loud because it is the transport, not the GPU, that decides whether this is
        // usable: the proxy and the answer both cross shared memory at the model's raster, so the
        // per-frame copy grows with the square of the scale.
        const double mb = double(modelW) * modelH * (hdrProxy ? 8.0 : 4.0) / (1024.0 * 1024.0);
        log_printf("[comp] supersampling to %ux%u means %.0f MB across shared memory each way, every frame",
                   modelW, modelH, mb);
    }

    // Keep the captured frame across a rebuild that does not change its shape.
    //
    // Almost everything here is rebuilt because the *model's* raster changed -- passes, model
    // resolution, the down-leg filter -- while the captured frame stays the swapchain's size in the
    // swapchain's working format. Dropping it anyway costs nothing while the frame is not held,
    // because the next frame captures another one. A held frame would be lost: RecordCapture
    // unfreezes the moment _frameCaptured goes false, so the next composition would read the
    // swapchain image instead of the held picture.
    const bool keepFrame = _frameCaptured && _frame.image && _width == width && _height == height &&
                           _workFormat == work;
    Image savedFrame{};
    if (keepFrame) {
        savedFrame = _frame;
        _frame = Image{};   // detached, so DropAll leaves it alone
    }

    DropAll();

    if (keepFrame) {
        _frame = savedFrame;
        _frameCaptured = true;
    }

    _swapchainFormat = swapchainFormat;
    _workFormat = work;
    _blitSwapchain = blit;
    _linearHdr = linearHdr;
    _hdrProxy = hdrProxy;
    _hdrTransfer = hdrTransfer;
    const VkFormat proxyFormat = _hdrProxy ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

    // A display-referred UNORM frame is its own proxy: the encode would only copy it, so there is no
    // proxy and the downsample reads the frame, converting it into the model's format even at equal
    // size. A float frame still goes through the encode, which clamps its negative channels.
    const bool direct = !linearHdr && work != VK_FORMAT_R16G16B16A16_SFLOAT;

    const VkImageUsageFlags sampled = VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags storage = VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImageUsageFlags src = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageUsageFlags dst = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    const bool ok =
        (keepFrame || MakeImage(_frame, width, height, work, sampled | src | dst)) &&
        (direct || MakeImage(_proxy, width, height, proxyFormat, sampled | storage | src)) &&
        MakeImage(_model, modelW, modelH, proxyFormat, sampled | src | dst) &&
        MakeImage(_composed, width, height, work, storage | src);

    // The transport pair is sized to the model raster: exported device-local memory while the
    // export is on, staging otherwise.
    _modelW = modelW;
    _modelH = modelH;
    EnsureTransport();
    const bool okTransport = _download.buffer && _upload.buffer;


    // The meter is a fixed 64x64 grid whatever the frame is, and is only built when there is
    // something to measure: on a frame the game already tone mapped there is no white point to find,
    // so the dispatch and its reduction are skipped entirely rather than run and ignored. The
    // reduction -- percentile, gates, history, resolve -- runs on the GPU; only a 128-byte mirror
    // of its answer ever reaches the CPU.
    const bool okMeter =
        !linearHdr ||
        (MakeImage(_meter, DLSS_NR_METER_GRID, DLSS_NR_METER_GRID, VK_FORMAT_R32_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) &&
         MakeMeterState());

    const bool needWork = direct || modelW != width || modelH != height;
    const bool okWork = !needWork || MakeImage(_work, modelW, modelH, proxyFormat,
                                               sampled | storage | src);

    // The averaged answer, and the two filters that get there. Built only when supersampling.
    bool okSuper = true;
    if (superSample) {
        okSuper = MakeImage(_modelNative, width, height, proxyFormat, sampled | storage);
        _superUp = std::make_unique<ScalerVk>(_vk, _instance, _device, _physicalDevice, true, s.downscaler);
        _superDown = std::make_unique<ScalerVk>(_vk, _instance, _device, _physicalDevice, false, s.downscaler);
        if (!_superUp->CanRender() || !_superDown->CanRender()) {
            log_printf("[comp] the resampling filters could not be built; supersampling is unavailable");
            _superUp.reset();
            _superDown.reset();
            okSuper = false;
        }
    } else {
        _superUp.reset();
        _superDown.reset();
    }

    if (!ok || !okTransport || !okWork || !okMeter || !okSuper) {
        _reason = "could not allocate the composition surfaces";
        DropAll();
        return false;
    }

    _width = width;
    _height = height;
    _modelW = modelW;
    _modelH = modelH;
    _superSample = superSample;
    _scalerFilter = s.downscaler;
    _reason.clear();
    return true;
}

void Composition::Transition(VkCommandBuffer cb, Image& img, VkImageLayout to) {
    if (img.layout == to) return;
    VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    _pass->SetImageLayout(cb, img.image, img.layout, to, range);
    img.layout = to;
}

void Composition::TransitionSwapchain(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    // A transfer layout carries its transfer access. PRESENT_SRC needs only an execution
    // dependency: BOTTOM_OF_PIPE is every prior command as a source and nothing as a destination.
    const auto access = [](VkImageLayout l) -> VkAccessFlags {
        return l == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ? VK_ACCESS_TRANSFER_READ_BIT
             : l == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ? VK_ACCESS_TRANSFER_WRITE_BIT : 0;
    };
    const auto stage = [](VkAccessFlags a) -> VkPipelineStageFlags {
        return a ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    };
    b.srcAccessMask = access(from);
    b.dstAccessMask = access(to);
    _vk->vkCmdPipelineBarrier(cb, stage(b.srcAccessMask), stage(b.dstAccessMask), 0, 0,
                              nullptr, 0, nullptr, 1, &b);
}

// One step in or out of the swapchain. A copy when the two formats share a bit layout, which is the
// usual case and moves the bytes untouched; a blit when the working format had to differ, which
// converts between them. Same extent either way -- this never resamples.
void Composition::CopyWholeImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst,
                                 VkImageLayout dstLayout, uint32_t w, uint32_t h) {
    if (_blitSwapchain) {
        VkImageBlit b{};
        b.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        b.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        b.srcOffsets[1] = { (int32_t) w, (int32_t) h, 1 };
        b.dstOffsets[1] = { (int32_t) w, (int32_t) h, 1 };
        _vk->vkCmdBlitImage(cb, src, srcLayout, dst, dstLayout, 1, &b, VK_FILTER_NEAREST);
        return;
    }
    VkImageCopy copy{};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.extent = { w, h, 1 };
    _vk->vkCmdCopyImage(cb, src, srcLayout, dst, dstLayout, 1, &copy);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// Where the white point comes from, in one place.
//
// The measured reading is only taken when there is one: the meter needs a lit scene and a linear
// frame to say anything, and until it has spoken the slider is the answer rather than zero. The scale
// applies to whichever was chosen, because it is the user saying what the model should treat as white
// rather than a property of the measurement.
float Composition::ResolvedWhitePoint(const FrameSettings& s) const {
    float base = s.whitePointManual;
    if (s.whitePointSource == kWhitePointMeasured && _measuredWhitePoint > 0.0f)
        base = _measuredWhitePoint * s.whitePointTrim;
    const float wp = base * s.whitePointScale;
    return std::min(std::max(wp, 1e-4f), 2000.0f);
}

dlss_nr_constants Composition::BaseConstants(const FrameSettings& s) const {
    dlss_nr_constants c{};

    c.white_point = ResolvedWhitePoint(s);
    c.transfer_strength = s.transferStrength;
    c.colour_strength = s.colourStrength;
    c.max_ratio = s.maxRatio;
    c.debug_view = s.debugView;
    c.debug_scale = s.debugScale;
    c.transfer = s.transfer;
    c.compare_mode = s.compareMode;
    c.compare_split = s.compareSplit;
    c.compare_zoom = s.compareZoom;
    c.compare_swap = s.compareSwap;
    c.reversible_mode = s.reversibleMode;
    c.apply_model = s.applyModel;

    // The model's answer IS the frame. The raw-answer debug path returns the model's picture ahead of
    // every step of the composition -- no ratio, no guard, no blend, no compare -- and it returns
    // before the normalisation step, so the scale that step would have applied has to come from here.
    //
    // That early return is also why compare did nothing under a bypass: the overlay lives at the tail
    // of the resolve, past every return, so the raw path showed one picture with a divider across it
    // and nothing to compare. The replace modes are the only route that presents the model's answer
    // without returning early -- pure inverse of the encode, none of the composition -- so comparing
    // under a bypass borrows them: soft knee and Neutwo pair with NeutwoDecode, Hybrid with
    // HybridDecode. The encode reads these same constants, so the pair stays an exact inverse. The
    // price is that the proxy shown to the model while comparing is the replace mode's own rather
    // than the user's, which is acceptable for a diagnostic view and why this is not done when the
    // composition is on, where the overlay already runs on the composed picture.
    if (s.compositionBypass) {
        if (s.compareMode != 0) {
            c.reversible_mode = s.reversibleMode >= 3 ? 4u : 2u;
        } else {
            c.debug_view = 2;
            c.debug_scale = 1.0f;  // resolve applies the white-point/transfer scale once
        }
    }

    // A frame the game already tone mapped goes through the encode untouched, and the composition
    // works in its units rather than normalising by a white point that means nothing here.
    c.passthrough = _linearHdr ? 0u : 1u;

    // No motion vectors and no exposure reach a present-time layer. The guides are declared at the
    // frame's own size so nothing downstream scales by a ratio that does not exist.
    c.mv_scale_x = 1.0f;
    c.mv_scale_y = 1.0f;
    c.guide_width = _width;
    c.guide_height = _height;
    c.use_game_exposure = 0;
    c.exposure_pre_mul = 1.0f;
    // Surface precision and model color domain are independent for native HIP.
    c.hdr_proxy = _hdrProxy ? 2u : 0u;
    c.hdr_transfer = _hdrTransfer;
    c.colour_trust = s.colourTrust;
    c.ratio_smooth = s.ratioSmooth;
    return c;
}

// ---------------------------------------------------------------------------
// Leg 1: the frame the model is shown
// ---------------------------------------------------------------------------
bool Composition::RecordCapture(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s) {
    if (!_usable || !_frame.image) return false;

    // Frame hold, on the edge rather than the level, so the white point is snapshotted once at the
    // moment it comes on rather than re-read every frame it stays on. The snapshot comes from the
    // meter's mirror -- the resolved value the GPU settled on for the frame just captured -- so a
    // hold freezes the same number the resolve has been using, not a host-side approximation.
    if (s.holdFrame && !_holding) {
        _holding = true;
        const float* mirror = _meterGpu && _meterMirror.mapped ? (const float*) _meterMirror.mapped : nullptr;
        _heldWhitePoint = mirror && mirror[2] > 0.0f ? mirror[2] : ResolvedWhitePoint(s);
        log_printf("[comp] frame held (white point %.3f)", double(_heldWhitePoint));
    } else if (!s.holdFrame && _holding) {
        _holding = false;
        log_printf("[comp] frame released");
    }

    // While held the frame is not re-read, but everything downstream of it still runs: the encode
    // re-encodes, the model re-evaluates and the resolve re-composes, so a setting changed now is
    // answered on the same picture. Freezing the proxy instead would be wrong -- settings must still
    // re-encode -- and freezing it would also desynchronise it from the frame the resolve reads.
    const bool freeze = _holding && _frameCaptured;

    if (!freeze) {
        TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Transition(cb, _frame, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CopyWholeImage(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _frame.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _width, _height);

        // Straight back, so that every path out of this leg -- including the ones that give up --
        // leaves the image in the layout the present engine requires.
        TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        _frameCaptured = true;
    }

    Transition(cb, _frame, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Image* source = &_frame;
    if (_proxy.image) {
        Transition(cb, _proxy, VK_IMAGE_LAYOUT_GENERAL);
        dlss_nr_constants enc = BaseConstants(s);
        enc.mode = DLSS_NR_MODE_ENCODE;
        enc.width = _width;
        enc.height = _height;
        if (!_pass->Dispatch(cb, enc, _width, _height, _frame.view, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                             _proxy.view, VK_NULL_HANDLE))
            return false;
        source = &_proxy;
    }

    // The meter, measured off the captured frame and never off anything this pass writes. That
    // distinction is the whole reason it is safe: an earlier white point meter upstream read its own
    // output and chased it, walking one session from 0.010 to 97.910. There is no path from what this
    // pass writes back into what this reads.
    //
    // The grid lands in a device-local state buffer and the reduce pass turns it into the resolved
    // white point without the bytes ever crossing to the host; the resolve reads the answer through
    // a four-byte copy into its own constant block (RecordCompose). The 128-byte mirror the transfer
    // copies at the end is for the CPU's frame-hold snapshot and status field only.
    if (_meter.image && _meterGpu) {
        Transition(cb, _meter, VK_IMAGE_LAYOUT_GENERAL);

        dlss_nr_constants meter = BaseConstants(s);
        meter.mode = DLSS_NR_MODE_CALIBRATE;
        meter.width = DLSS_NR_METER_GRID;
        meter.height = DLSS_NR_METER_GRID;
        if (_pass->Dispatch(cb, meter, DLSS_NR_METER_GRID, DLSS_NR_METER_GRID, _frame.view, VK_NULL_HANDLE,
                            VK_NULL_HANDLE, VK_NULL_HANDLE, _meter.view, VK_NULL_HANDLE)) {
            if (!_meterStateCleared) {
                _vk->vkCmdFillBuffer(cb, _meterState, 0, VK_WHOLE_SIZE, 0);
                // The reduce reads and writes what the fill cleared.
                const VkMemoryBarrier cleared{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          0, 1, &cleared, 0, nullptr, 0, nullptr);
                _meterStateCleared = true;
            }
            MeterPush pc{};
            pc.manual = s.whitePointManual;
            pc.scale = s.whitePointScale;
            pc.trim = s.whitePointTrim;
            pc.holdValue = _holding ? ResolvedWhitePoint(s) : 0.0f;
            pc.source = s.whitePointSource;
            pc.hold = _holding && _frameCaptured ? 1u : 0u;
            _vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _meterPipeline);
            _vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _meterPipelineLayout, 0, 1,
                                         &_meterDescriptorSet, 0, nullptr);
            _vk->vkCmdPushConstants(cb, _meterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(pc), &pc);
            _vk->vkCmdDispatch(cb, 1, 1, 1);

            // The mirror must show this frame's answer, so the transfer read waits on the reduce's
            // write. The state buffer stays in its default layout; buffers need no transition.
            VkBufferMemoryBarrier toMirror{};
            toMirror.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            toMirror.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            toMirror.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            toMirror.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toMirror.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toMirror.buffer = _meterState;
            toMirror.offset = 0;
            toMirror.size = VK_WHOLE_SIZE;
            _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 1, &toMirror, 0, nullptr);
            const VkBufferCopy mirror{ 0, 0, kMeterStateBytes };
            _vk->vkCmdCopyBuffer(cb, _meterState, _meterMirror.buffer, 1, &mirror);
        }
    }

    // What the model is actually handed: the full-resolution proxy, or a reduction of it.
    if (_work.image) {
        Transition(cb, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cb, _work, VK_IMAGE_LAYOUT_GENERAL);

        if (_superSample) {
            // Enlarge, so the model has more pixels to synthesise into than the frame has.
            if (!_superUp->Dispatch(cb, source->view, _work.view, _width, _height, _modelW, _modelH))
                return false;
        } else {
            // Reduce, with the module's own area filter -- the model then works on fewer pixels and
            // less crosses the shared memory.
            dlss_nr_constants down = BaseConstants(s);
            down.mode = DLSS_NR_MODE_DOWNSAMPLE;
            down.width = _modelW;
            down.height = _modelH;
            if (!_pass->Dispatch(cb, down, _modelW, _modelH, source->view, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                 VK_NULL_HANDLE, _work.view, VK_NULL_HANDLE))
                return false;
        }
        source = &_work;
    }

    Transition(cb, *source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { _modelW, _modelH, 1 };

    ExternalOwnership(cb, _download, VK_QUEUE_FAMILY_EXTERNAL, _exportFamily, VK_ACCESS_TRANSFER_WRITE_BIT);
    _vk->vkCmdCopyImageToBuffer(cb, source->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _download.buffer, 1,
                                &region);
    ExternalOwnership(cb, _download, _exportFamily, VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_WRITE_BIT);
    return true;
}

// ---------------------------------------------------------------------------
// Leg 2: the model's answer, composed back
// ---------------------------------------------------------------------------
bool Composition::RecordCompose(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s) {
    if (!_usable || !_composed.image) return false;

    const bool rawCopy = s.compositionBypass != 0 && s.compareMode == 0 && !capture_writer_active(&_capture) &&
                         !_superSample && !_hdrProxy && !_linearHdr &&
                         _modelW == _width && _modelH == _height && _model.format == _workFormat;
    Transition(cb, _model, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { _modelW, _modelH, 1 };
    ExternalOwnership(cb, _upload, VK_QUEUE_FAMILY_EXTERNAL, _exportFamily, VK_ACCESS_TRANSFER_READ_BIT);
    _vk->vkCmdCopyBufferToImage(cb, _upload.buffer, _model.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                &region);
    ExternalOwnership(cb, _upload, _exportFamily, VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_READ_BIT);
    Transition(cb, _model, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // When the model worked above the frame its answer is averaged back to native first, and the
    // composition then sees a native proxy against a native answer -- which is what it should see,
    // because from its point of view the model effectively ran at the frame's own resolution.
    Image* answer = &_model;
    Image* source = _work.image ? &_work : &_proxy;
    if (rawCopy) {
        Transition(cb, *answer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CopyWholeImage(cb, answer->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _width, _height);
        TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        return true;
    }
    if (_superSample) {
        Transition(cb, _modelNative, VK_IMAGE_LAYOUT_GENERAL);
        if (!_superDown->Dispatch(cb, answer->view, _modelNative.view, _modelW, _modelH, _width, _height))
            return false;
        Transition(cb, _modelNative, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        answer = &_modelNative;
        source = _proxy.image ? &_proxy : &_frame;
    }

    Transition(cb, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cb, _frame, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cb, _composed, VK_IMAGE_LAYOUT_GENERAL);

    dlss_nr_constants res = BaseConstants(s);
    res.mode = DLSS_NR_MODE_RESOLVE;
    res.width = _width;
    res.height = _height;

    // When the meter feeds the white point, the resolve must see the GPU's resolved value, not the
    // host's one-frame-stale mirror of it: copy the four bytes straight from the meter state into
    // the constant slot this dispatch is about to take, over the placeholder the host wrote. The
    // host memcpy ran before submit, so this transfer write is the last writer ahead of the
    // dispatch's uniform read, and the barrier states that.
    if (_meterGpu && s.whitePointSource == kWhitePointMeasured) {
        const VkDeviceSize slotOff = _pass->ConstantSlotStride() * _pass->NextConstantSlot() +
                                     offsetof(struct dlss_nr_constants, white_point);
        const VkBufferCopy patch{ kMeterResolvedOffset, slotOff, sizeof(float) };
        _vk->vkCmdCopyBuffer(cb, _meterState, _pass->ConstantBuffer(), 1, &patch);
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = _pass->ConstantBuffer();
        b.offset = slotOff;
        b.size = sizeof(float);
        _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 0, nullptr, 1, &b, 0, nullptr);
    }

    if (!_pass->Dispatch(cb, res, _width, _height, source->view, answer->view, _frame.view, VK_NULL_HANDLE,
                         _composed.view, VK_NULL_HANDLE))
        return false;

    Transition(cb, _composed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CopyWholeImage(cb, _composed.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _width, _height);
    TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // The pair, taken here because this is the one place that holds both the frame as the game
    // presented it and the frame the model edited, for the same frame.
    _captureRecorded = false;
    if (capture_writer_active(&_capture)) {
        _captureMetadata.frame_control_seq = s.controlSeq;
        _captureMetadata.tuning_seq = s.tuningSeq;
        _captureMetadata.passes = s.passes;
        _captureMetadata.debug_view = s.debugView;
        _captureMetadata.apply_model = s.applyModel;
        _captureMetadata.bypass = s.compositionBypass;
        _captureMetadata.hold = s.holdFrame;
        _captureMetadata.compare = s.compareMode;
        _captureMetadata.transfer = s.transfer;
        _captureMetadata.detail = s.transferStrength;
        _captureMetadata.color = s.colourStrength;
        _captureMetadata.debug_scale = s.debugScale;
        _captureMetadata.model_width = _modelW;
        _captureMetadata.model_height = _modelH;
        _captureMetadata.hdr_proxy = _hdrProxy;
        _captureMetadata.linear_hdr = _linearHdr;
        _captureMetadata.hdr_transfer = _hdrTransfer;
        const size_t bytes = size_t(_width) * _height * (_workFormat == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4);
        if (_captureBuf.buffer || MakeHostBuffer(_captureBuf, bytes * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
            Transition(cb, _frame, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            Transition(cb, _composed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy r{};
            r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            r.imageExtent = { _width, _height, 1 };
            _vk->vkCmdCopyImageToBuffer(cb, _frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                        _captureBuf.buffer, 1, &r);
            r.bufferOffset = bytes;
            _vk->vkCmdCopyImageToBuffer(cb, _composed.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                        _captureBuf.buffer, 1, &r);
            _captureRecorded = true;
        }
    }
    return true;
}

void Composition::WriteCapturedFrame() {
    if (!_captureRecorded || !_captureBuf.mapped) return;
    _captureRecorded = false;
    const size_t bytes = size_t(_width) * _height * (_workFormat == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4);
    const uint8_t* base = (const uint8_t*) _captureBuf.mapped;
    capture_writer_write_frame(&_capture, base, base + bytes, _width, _height, uint32_t(_workFormat),
                               &_captureMetadata);
}


// The percentile, the gates and the history now run on the GPU (shaders/meter_reduce.comp); this
// reads the 128-byte mirror of that state, recorded in leg 1 and landed by leg 1's fence. The
// reasoning the shader reproduces: the 90th percentile of tile peaks, not the maximum (a sun or a
// specular hit would normalise the whole picture into the dark) and not the mean (scene brightness
// says nothing about the buffer's scale); offered only when enough of the frame carries light
// against its own brightest tile, since the units are the game's and there is no absolute scale.
// A rejected reading carries the previous answer forward on the GPU, so the value here never needs
// to be zeroed by a dark or torn frame.
void Composition::ConsumeMeter() {
    if (!_meterGpu || !_meterMirror.mapped) return;
    const float* mirror = (const float*) _meterMirror.mapped;
    _measuredWhitePoint = mirror[1];
}

}  // namespace dlssnr
