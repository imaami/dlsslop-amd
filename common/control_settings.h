// SPDX-License-Identifier: MIT
#pragma once
#include "../upstream-layer/common/shm_protocol.h"
#include <linux/input-event-codes.h>

namespace dlsslop_control {
struct Setting {
    const char* name;
    char shortName;
    std::atomic<uint32_t> ShmHeader::*field;
    bool isFloat;
    double minimum;
    double maximum;
    const char* help;
};

// All controls from the original interface. The captured network configuration
// is fixed for four fields; parsing rejects unsupported changes before mapping.
inline constexpr Setting kSettings[] = {
    {"color-preserve", 'L', &ShmHeader::colorPreserveBits, true, 0, 1, "Per-pass original-frame chroma anchoring (HIP when installed, otherwise CPU; 0 disables)"},
    {"enabled", 'e', &ShmHeader::enabled, false, 0, 1, "Enable neural rendering (0/1)"},
    {"hdr-mode", 'E', &ShmHeader::hdrMode, false, 0, 2, "Proxy precision: 0 auto (16-bit for HDR), 1 force 8-bit, 2 force 16-bit"},
    {"sdr16-multipass", 'B', &ShmHeader::sdr16Multipass, false, 0, 1, "SDR between-pass precision: 0 quantized 8-bit, 1 binary16; HDR stays binary16"},
    {"preset", 'N', &ShmHeader::preset, false, 0, 15, "Captured network preset; only 0 is supported"},
    {"style", 'y', &ShmHeader::style, false, 0, 2, "Captured network style; only 0 is supported"},
    {"auto-mask", 'M', &ShmHeader::autoMask, false, 0, 1, "Captured automatic-mask configuration; only 1 is supported"},
    {"intensity", 'i', &ShmHeader::intensityBits, true, 0, 4, "Native per-pass residual intensity"},
    {"local-tone", 'o', &ShmHeader::localToneBits, true, 0, 4, "Native per-pass low-frequency residual strength"},
    {"local-structure", 'j', &ShmHeader::localStructureBits, true, 0, 4, "Native per-pass high-frequency residual strength"},
    {"skin-structure", 'K', &ShmHeader::skinStructureBits, true, -1, 4, "Captured skin configuration; only -1 (follow structure) is supported"},
    {"sharpness", 'n', &ShmHeader::sharpnessBits, true, 0, 1, "Native per-pass sharpening strength"},
    {"rebuild-ms", 'J', &ShmHeader::rebuildSettleMs, false, 0, 5000, "Debounce native per-pass tuning updates for this many milliseconds"},
    {"mvec", 'V', &ShmHeader::mvecEnabled, false, 0, 1, "Estimate motion and reproject previous neural output (0/1)"},
    {"mvec-quality", 'Q', &ShmHeader::mvecQuality, false, 0, 2, "Motion search quality: 0 fast, 1 balanced, 2 quality"},
    {"mvec-units", 'U', &ShmHeader::mvecScaleMode, false, 0, 2, "Motion field units: 0 normalized, 1 pixels, 2 UV"},
    {"mvec-pixels", 'F', &ShmHeader::mvecPixelSize, false, 0, 3, "Motion grid spacing: 0=1px, 1=2px, 2=4px, 3=8px"},
    {"white-point", 'W', &ShmHeader::whitePointBits, true, 0.0001, 2000, "Manual paper white for linear-light input"},
    {"white-point-scale", 'G', &ShmHeader::whitePointScaleBits, true, 0.01, 100, "Multiplier on manual or measured white point"},
    {"white-point-source", 'O', &ShmHeader::whitePointSource, false, 0, 1, "0 manual paper white, 1 measured from frame"},
    {"white-point-trim", 'I', &ShmHeader::whitePointTrimBits, true, 0.01, 100, "Multiplier on measured white point only"},
    {"color-mode", 'Y', &ShmHeader::colourMode, false, 0, 2, "0 auto, 1 display-referred, 2 linear HDR"},
    {"reversible", 'Z', &ShmHeader::reversibleMode, false, 0, 4, "Proxy: 0 knee, 1 Neutwo, 2 Neutwo replace, 3 hybrid, 4 hybrid replace"},
    {"passes", 'P', &ShmHeader::passes, false, 1, kMaxPasses, "Successive neural evaluations per frame; each consumes the previous result"},
    {"detail", 'd', &ShmHeader::transferStrengthBits, true, 0, 4, "Strength of the neural edit"},
    {"color", 'C', &ShmHeader::colourStrengthBits, true, 0, 4, "Color contribution of the edit"},
    {"guard", 'g', &ShmHeader::maxRatioBits, true, 1, 30, "Maximum per-pixel gain or reciprocal gain"},
    {"transfer", 't', &ShmHeader::transfer, false, 0, 2, "0 classic, 1 matched residual, 2 native frame + edit"},
    {"bypass", 'b', &ShmHeader::compositionBypass, false, 0, 1, "0 compose the edit, 1 present the raw model result"},
    {"ratio-smooth", 'a', &ShmHeader::ratioSmoothPercent, false, 0, 100, "Neighbourhood contribution to relighting ratio (%)"},
    {"color-trust", 'u', &ShmHeader::colourTrustPercent, false, 0, 800, "Allowed color displacement, in hundredths"},
    {"debug-view", 'v', &ShmHeader::debugView, false, 0, 5, "0 off, 1 proxy, 2 model, 3 edit, 4/5 color-bound views"},
    {"debug-scale", 'D', &ShmHeader::debugScaleBits, true, 0.01, 100, "Debug-view intensity multiplier"},
    {"working-scale", 'w', &ShmHeader::workingScaleBits, true, 0.25, 2, "Proxy scale relative to game resolution; dlsslopd caps it at 1 and at its tier raster unless run with --cpu-compose or --test-identity"},
    {"downscaler", 'f', &ShmHeader::scalingDownscaler, false, 1, 7, "Supersampling down-leg filter, used only above working-scale 1 with dlsslopd --cpu-compose or --test-identity: 1 bicubic, 2 catmull, 3 lanczos2, 4 lanczos3, 5 kaiser2, 6 kaiser3, 7 magic"},
    {"compare", 'p', &ShmHeader::compareMode, false, 0, 2, "0 off, 1 side by side, 2 wipe"},
    {"compare-split", 'x', &ShmHeader::compareSplitBits, true, 0, 1, "Comparison split position"},
    {"compare-zoom", 'z', &ShmHeader::compareZoomBits, true, 1, 2, "Side-by-side comparison magnification"},
    {"compare-swap", 'X', &ShmHeader::compareSwap, false, 0, 1, "Swap comparison sides (0/1)"},
    {"apply-model", 'm', &ShmHeader::applyModel, false, 0, 1, "Apply the edit; 0 shows the clean frame but retains inference cost"},
    {"hold", 'H', &ShmHeader::holdFrame, false, 0, 1, "Freeze the input frame (0/1)"},
    {"toggle-key", 'k', &ShmHeader::toggleKey, false, 0, KEY_MAX, "Linux KEY_ code for the layer hotkey; 0 disables it"},
};

// Readers see float settings as binary32, which rounds some minimums below
// themselves, so compare with the bounds as stored; integer bounds are exact
// there. NaN fails.
inline bool inRange(const Setting& s, double v)
{
    return v >= static_cast<float>(s.minimum) && v <= static_cast<float>(s.maximum);
}
inline bool fixed(const Setting& s)
{
    return s.field == &ShmHeader::preset || s.field == &ShmHeader::style ||
        s.field == &ShmHeader::autoMask || s.field == &ShmHeader::skinStructureBits;
}
inline bool tuning(const Setting& s)
{
    return s.field == &ShmHeader::colorPreserveBits || s.field == &ShmHeader::intensityBits || s.field == &ShmHeader::localToneBits ||
        s.field == &ShmHeader::localStructureBits || s.field == &ShmHeader::sharpnessBits ||
        s.field == &ShmHeader::rebuildSettleMs;
}
} // namespace dlsslop_control
