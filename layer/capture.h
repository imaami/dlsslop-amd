#pragma once
// Matched before/after frames, written so questions about this pass get settled by measurement.
//
// Upstream's reasoning, which is worth keeping: every comparison of a detail pass tends to be two
// separate video captures -- different camera path, different exposure, and a codec in between
// throwing away exactly the high-frequency detail the argument is about. What is wanted instead is
// the same frames twice. The pass already holds the frame as the game presented it and the frame
// after the model's edit, so both are written for one run of consecutive frames: a control with
// nothing varying but the thing under test.
//
// Upstream writes raw because a codec is the confound. PNG is lossless, so it is not a confound, and
// a file you can open is worth a great deal more than one you cannot -- so an 8-bit frame is written
// as PNG. A 10-bit or float frame has no PNG that can hold it and is written raw, as upstream does,
// with a manifest saying how to read it. The manifest is written either way.
#include <cstdint>
#include <string>
#include <vector>

namespace dlssnr {

// Renderer settings come from the frame snapshot. inferenceSeq is the worker
// request that answered the frame; passes is the layer's observed requested count.
struct CaptureMetadata {
    uint32_t frameControlSeq = 0, tuningSeq = 0;
    uint32_t inferenceSeq = 0, passes = 0, debugView = 0, applyModel = 0, bypass = 0, hold = 0;
    uint32_t compare = 0, transfer = 0;
    uint32_t modelWidth = 0, modelHeight = 0, hdrProxy = 0, linearHdr = 0, hdrTransfer = 0;
    float detail = 0, color = 0, debugScale = 1;
    uint64_t beforeHash = 0;
};

class CaptureWriter {
  public:
    // Where captures go: $XDG_STATE_HOME/dlssnr/captures, or ~/.local/state/dlssnr/captures.
    // Each request uses a unique batch directory. Prior captures are preserved.
    static std::string Directory();

    void Begin(uint32_t frames, uint32_t controlSeq);
    bool Active() const { return _remaining > 0; }
    uint32_t Remaining() const { return _remaining; }

    // One frame's pair. `format` is the Vulkan format both surfaces are in; the writer decides
    // between PNG and raw from it, and reports what it chose in the manifest.
    void WriteFrame(const void* before, const void* after, uint32_t width, uint32_t height,
                    uint32_t vkFormat, const CaptureMetadata& metadata);

  private:
    bool WriteManifest(uint32_t width, uint32_t height, uint32_t vkFormat, bool png) const;

    uint32_t _remaining = 0;
    uint32_t _index = 0;
    uint32_t _controlSeq = 0;
    uint32_t _batchName = 0;  // where the batch's own name starts in _batchDir
    std::string _batchDir;
    std::vector<CaptureMetadata> _metadata;
};

}  // namespace dlssnr
