// SPDX-License-Identifier: MIT
#include "processing.hpp"
#include "tuning.hpp"

namespace dlsslop {

Result<ProcessingSettings> read_settings(const ShmHeader* h)
{
    ProcessingSettings settings;
    settings.fp16 = h->hdrEncode.load() != 0;
    const bool hdr = h->hdrDetected.load() != kHdrNone && h->colourMode.load() != kColourDisplay;
    settings.precision16 = hdr || h->sdr16Multipass.load() != 0;
    settings.motion = h->mvecEnabled.load() != 0;
    settings.motion_quality = ShmMVecQuality(h);
    settings.motion_grid = ShmMVecPixelSize(h);
    settings.tuning = {BitsToFloat(h->intensityBits.load()), BitsToFloat(h->localToneBits.load()),
                       BitsToFloat(h->localStructureBits.load()), BitsToFloat(h->sharpnessBits.load())};
    settings.color_preserve = BitsToFloat(h->colorPreserveBits.load());
    settings.style = h->style.load();
    settings.skin_structure = BitsToFloat(h->skinStructureBits.load());
    const uint32_t mask = h->autoMask.load();
    settings.auto_mask = mask != 0;
    if (h->preset.load() || settings.style > 2 || mask > 1 ||
        !(settings.skin_structure >= -1 && settings.skin_structure <= 2))
        return reject("the preset must be 0, style 0..2, auto-mask 0 or 1 and skin structure -1..2");
    if (!(settings.color_preserve >= 0 && settings.color_preserve <= 1))
        return reject("invalid color preservation strength");
    DLSSLOP_TRY(validate_native_tuning(settings.tuning));
    return settings;
}

}  // namespace dlsslop
