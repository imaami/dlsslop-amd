#include "capture.h"
#include "log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO_STDLIB_DEFINED
#include "../third_party/stb/stb_image_write.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <cinttypes>

namespace dlssnr {
namespace {

void MakeDirs(const std::string& path) {
    size_t pos = 1;
    while ((pos = path.find('/', pos)) != std::string::npos) {
        mkdir(path.substr(0, pos).c_str(), 0700);
        ++pos;
    }
    mkdir(path.c_str(), 0700);
}

// True when the format is four 8-bit channels, which is the only shape PNG can hold here.
bool IsEightBitRgba(uint32_t f) {
    switch (f) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
            return true;
        default:
            return false;
    }
}

// PNG wants RGBA byte order; a B8G8R8A8 surface has red and blue the other way round.
bool NeedsChannelSwap(uint32_t f) { return f == VK_FORMAT_B8G8R8A8_UNORM; }

size_t BytesPerPixel(uint32_t f) {
    return f == VK_FORMAT_R16G16B16A16_SFLOAT ? 8u : 4u;
}

bool WritePng(const std::string& path, const void* pixels, uint32_t w, uint32_t h, bool swap) {
    const size_t px = size_t(w) * h;
    std::vector<uint8_t> rgba(px * 4);
    std::memcpy(rgba.data(), pixels, px * 4);
    if (swap) {
        for (size_t i = 0; i < px; ++i) std::swap(rgba[i * 4 + 0], rgba[i * 4 + 2]);
    }
    if (!stbi_write_png(path.c_str(), int(w), int(h), 4, rgba.data(), int(w) * 4)) {
        Log("[capture] could not write %s", path.c_str());
        return false;
    }
    return true;
}

bool WriteRaw(const std::string& path, const void* pixels, size_t bytes) {
    if (FILE* f = fopen(path.c_str(), "wb")) {
        const bool wrote = fwrite(pixels, 1, bytes, f) == bytes;
        const bool closed = fclose(f) == 0;
        if (wrote && closed) return true;
    }
    Log("[capture] could not write %s", path.c_str());
    return false;
}

// FNV-1a over 8-byte words, then the tail bytes: only equality matters.
uint64_t HashBytes(const void* pixels, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(pixels);
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t i = 0;
    for (uint64_t word; i + 8 <= bytes; i += 8) {
        std::memcpy(&word, p + i, 8);
        hash = (hash ^ word) * UINT64_C(1099511628211);
    }
    for (; i < bytes; ++i) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

void WriteMetadata(FILE* f, const std::string& prefix, const CaptureMetadata& m) {
#define CAPTURE_U(name, field) std::fprintf(f, "%s" name " %u\n", prefix.c_str(), m.field)
#define CAPTURE_F(name, field) std::fprintf(f, "%s" name " %.9g\n", prefix.c_str(), double(m.field))
    CAPTURE_U("frame_control_seq", frameControlSeq);
    CAPTURE_U("tuning_seq", tuningSeq);
    CAPTURE_U("inference_seq", inferenceSeq);
    CAPTURE_U("passes", passes);
    CAPTURE_U("debug_view", debugView);
    CAPTURE_U("apply_model", applyModel);
    CAPTURE_U("bypass", bypass);
    CAPTURE_U("hold", hold);
    CAPTURE_U("compare", compare);
    CAPTURE_U("transfer", transfer);
    CAPTURE_U("model_width", modelWidth);
    CAPTURE_U("model_height", modelHeight);
    CAPTURE_U("hdr_proxy", hdrProxy);
    CAPTURE_U("linear_hdr", linearHdr);
    CAPTURE_U("hdr_transfer", hdrTransfer);
    CAPTURE_F("detail", detail);
    CAPTURE_F("color", color);
    CAPTURE_F("debug_scale", debugScale);
#undef CAPTURE_U
#undef CAPTURE_F
    std::fprintf(f, "%sbefore_hash %016" PRIx64 "\n", prefix.c_str(), m.beforeHash);
}

}  // namespace

std::string CaptureWriter::Directory() {
    if (const char* state = getenv("XDG_STATE_HOME"); state && *state)
        return std::string(state) + "/dlssnr/captures";
    if (const char* home = getenv("HOME"); home && *home)
        return std::string(home) + "/.local/state/dlssnr/captures";
    return "/tmp/dlssnr-captures";
}

void CaptureWriter::Begin(uint32_t frames, uint32_t controlSeq) {
    if (frames == 0) return;
    const std::string dir = Directory();
    MakeDirs(dir);
    _batchDir = dir + "/capture-" + std::to_string(getpid()) + "-XXXXXX";
    if (!mkdtemp(_batchDir.data())) {
        Log("[capture] cannot create batch directory in %s", dir.c_str());
        _remaining = 0;
        return;
    }
    _batchName = uint32_t(dir.size() + 1);
    _controlSeq = controlSeq;
    _metadata.clear();
    _remaining = frames;
    _index = 0;
    Log("[capture] capturing %u frames, control %u, to %s", frames, controlSeq, _batchDir.c_str());
}

void CaptureWriter::WriteFrame(const void* before, const void* after, uint32_t width, uint32_t height,
                               uint32_t vkFormat, const CaptureMetadata& metadata) {
    if (_remaining == 0) return;

    const std::string& dir = _batchDir;
    const bool png = IsEightBitRgba(vkFormat);
    const bool swap = NeedsChannelSwap(vkFormat);
    const size_t bytes = size_t(width) * height * BytesPerPixel(vkFormat);

    char name[64];
    bool beforeWritten = false, afterWritten = false;
    if (png) {
        std::snprintf(name, sizeof(name), "/before_%02u.png", _index);
        beforeWritten = WritePng(dir + name, before, width, height, swap);
        std::snprintf(name, sizeof(name), "/after_%02u.png", _index);
        afterWritten = WritePng(dir + name, after, width, height, swap);
    } else {
        std::snprintf(name, sizeof(name), "/before_%02u.raw", _index);
        beforeWritten = WriteRaw(dir + name, before, bytes);
        std::snprintf(name, sizeof(name), "/after_%02u.raw", _index);
        afterWritten = WriteRaw(dir + name, after, bytes);
    }
    if (!beforeWritten || !afterWritten) {
        _remaining = 0;
        Log("[capture] batch failed; completion manifest was not published");
        return;
    }
    auto frameMetadata = metadata;
    frameMetadata.beforeHash = HashBytes(before, bytes);
    _metadata.push_back(frameMetadata);

    ++_index;
    --_remaining;

    if (_remaining == 0) {
        if (WriteManifest(width, height, vkFormat, png))
            Log("[capture] wrote %u pairs to %s", _index, dir.c_str());
        else
            Log("[capture] could not publish completion manifest for %s", dir.c_str());
    }
}

bool CaptureWriter::WriteManifest(uint32_t width, uint32_t height, uint32_t vkFormat, bool png) const {
    const std::string path = _batchDir + "/manifest.txt";
    FILE* f = fopen(path.c_str(), "wt");
    if (!f) return false;
    std::fprintf(f, "capture_metadata_version 2\n");
    std::fprintf(f, "capture_control_seq %u\n", _controlSeq);
    std::fprintf(f, "batch_dir %s\n", _batchDir.c_str() + _batchName);
    std::fprintf(f, "frames %u\n", _index);
    std::fprintf(f, "width %u\nheight %u\n", width, height);
    std::fprintf(f, "vk_format %u\n", vkFormat);
    std::fprintf(f, "encoding %s\n", png ? "png" : "raw");
    std::fprintf(f, "bytes_per_pixel %zu\n", BytesPerPixel(vkFormat));
    std::fprintf(f, "row_pitch %zu\n", BytesPerPixel(vkFormat) * width);
    if (!_metadata.empty()) WriteMetadata(f, "", _metadata.front());
    for (size_t i = 0; i < _metadata.size(); ++i)
        WriteMetadata(f, "frame_" + std::to_string(i) + "_", _metadata[i]);
    std::fprintf(f,
                 "\n"
                 "before_NN is the frame as the game presented it; after_NN is the same frame with\n"
                 "the model's edit composed onto it. Same frame, same run, one variable.\n");
    const bool written = ferror(f) == 0;
    if (fclose(f) != 0 || !written) return false;

    // The immutable batch is complete before the public completion pointer is
    // replaced. A hard link plus rename avoids partial manifests and never
    // overwrites any previous batch's images.
    const std::string pending = _batchDir + "/published.tmp";
    if (link(path.c_str(), pending.c_str()) != 0) return false;
    if (rename(pending.c_str(), (_batchDir.substr(0, _batchName) + "manifest.txt").c_str()) != 0) {
        unlink(pending.c_str());
        return false;
    }
    return true;
}

}  // namespace dlssnr
