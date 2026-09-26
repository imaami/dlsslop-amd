// SPDX-License-Identifier: MIT
#pragma once
#include "../upstream-layer/common/shm_protocol.h"
#include <linux/input-event-codes.h>

namespace dlsslop_control {
// Controller pages, in their order.
enum Section : uint8_t { kNeural, kComposition, kImage, kMotion, kInspect, kModel };

struct Setting {
    const char* name;
    std::atomic<uint32_t> ShmHeader::*field;
    double minimum;
    double maximum;
    const char* help;
    char shortName;
    bool isFloat;
    Section section;
    bool tuning;                   // a native tuning value: changing it bumps tuningSeq
    const char* choices = nullptr; // '|'-separated labels of minimum, minimum + 1, ...
};

// All controls from the original interface. The captured network supports one
// configuration, so the four kModel settings have a one-value range.
inline constexpr Setting kSettings[] = {
    {"color-preserve", &ShmHeader::colorPreserveBits, 0, 1, "Per-pass original-frame chroma anchoring (0 disables)", 'L', true, kNeural, false},
    {"enabled", &ShmHeader::enabled, 0, 1, "Enable neural rendering (0/1)", 'e', false, kNeural, false},
    {"hdr-mode", &ShmHeader::hdrMode, 0, 2, "Proxy precision: 0 auto (16-bit for HDR), 1 force 8-bit, 2 force 16-bit", 'E', false, kImage, false, "Automatic|8-bit proxy|16-bit proxy"},
    {"sdr16-multipass", &ShmHeader::sdr16Multipass, 0, 1, "SDR between-pass precision: 0 quantized 8-bit, 1 binary16; HDR stays binary16", 'B', false, kImage, false},
    {"preset", &ShmHeader::preset, 0, 0, "Captured network preset; only 0 is supported", 'N', false, kModel, false},
    {"style", &ShmHeader::style, 0, 0, "Captured network style; only 0 is supported", 'y', false, kModel, false},
    {"auto-mask", &ShmHeader::autoMask, 1, 1, "Captured automatic-mask configuration; only 1 is supported", 'M', false, kModel, false},
    {"intensity", &ShmHeader::intensityBits, 0, 4, "Native per-pass residual intensity", 'i', true, kNeural, true},
    {"local-tone", &ShmHeader::localToneBits, 0, 4, "Native per-pass low-frequency residual strength", 'o', true, kNeural, true},
    {"local-structure", &ShmHeader::localStructureBits, 0, 4, "Native per-pass high-frequency residual strength", 'j', true, kNeural, true},
    {"skin-structure", &ShmHeader::skinStructureBits, -1, -1, "Captured skin configuration; only -1 (follow structure) is supported", 'K', true, kModel, false},
    {"sharpness", &ShmHeader::sharpnessBits, 0, 1, "Native per-pass sharpening strength", 'n', true, kNeural, true},
    {"mvec", &ShmHeader::mvecEnabled, 0, 1, "Estimate motion and reproject previous neural output (0/1)", 'V', false, kMotion, false},
    {"mvec-quality", &ShmHeader::mvecQuality, 0, 2, "Motion search quality: 0 fast, 1 balanced, 2 quality", 'Q', false, kMotion, false, "Fast|Balanced|Quality"},
    {"mvec-pixels", &ShmHeader::mvecPixelSize, 0, 3, "Motion grid spacing: 0=1px, 1=2px, 2=4px, 3=8px", 'F', false, kMotion, false, "1 px|2 px|4 px|8 px"},
    {"white-point", &ShmHeader::whitePointBits, 0.0001, 2000, "Manual paper white for linear-light input", 'W', true, kImage, false},
    {"white-point-scale", &ShmHeader::whitePointScaleBits, 0.01, 100, "Multiplier on manual or measured white point", 'G', true, kImage, false},
    {"white-point-source", &ShmHeader::whitePointSource, 0, 1, "0 manual paper white, 1 measured from frame", 'O', false, kImage, false, "Manual|Measured"},
    {"white-point-trim", &ShmHeader::whitePointTrimBits, 0.01, 100, "Multiplier on measured white point only", 'I', true, kImage, false},
    {"color-mode", &ShmHeader::colourMode, 0, 2, "0 auto, 1 display-referred, 2 linear HDR", 'Y', false, kImage, false, "Automatic|Display-referred|Linear HDR"},
    {"reversible", &ShmHeader::reversibleMode, 0, 4, "Proxy: 0 knee, 1 Neutwo, 2 Neutwo replace, 3 hybrid, 4 hybrid replace", 'Z', false, kImage, false, "Knee|Neutwo|Neutwo replace|Hybrid|Hybrid replace"},
    {"passes", &ShmHeader::passes, 1, kMaxPasses, "Successive neural evaluations per frame; each consumes the previous result", 'P', false, kNeural, false},
    {"detail", &ShmHeader::transferStrengthBits, 0, 4, "Strength of the neural edit", 'd', true, kComposition, false},
    {"color", &ShmHeader::colourStrengthBits, 0, 4, "Color contribution of the edit", 'C', true, kComposition, false},
    {"guard", &ShmHeader::maxRatioBits, 1, 30, "Maximum per-pixel gain or reciprocal gain", 'g', true, kComposition, false},
    {"transfer", &ShmHeader::transfer, 0, 2, "0 classic, 1 matched residual, 2 native frame + edit", 't', false, kComposition, false, "Classic ratio|Matched residual|Native frame + edit"},
    {"bypass", &ShmHeader::compositionBypass, 0, 1, "0 compose the edit, 1 present the raw model result; default 1 with dlsslopd --cpu-compose or --test-identity", 'b', false, kComposition, false},
    {"ratio-smooth", &ShmHeader::ratioSmoothPercent, 0, 100, "Neighbourhood contribution to relighting ratio (%)", 'a', false, kComposition, false},
    {"color-trust", &ShmHeader::colourTrustPercent, 0, 800, "Allowed color displacement, in hundredths", 'u', false, kComposition, false},
    {"debug-view", &ShmHeader::debugView, 0, 5, "0 off, 1 proxy, 2 model, 3 edit, 4/5 color-bound views", 'v', false, kInspect, false, "Off|Input proxy|Model output|Edit|Color bound 1|Color bound 2"},
    {"debug-scale", &ShmHeader::debugScaleBits, 0.01, 100, "Debug-view intensity multiplier", 'D', true, kInspect, false},
    {"working-scale", &ShmHeader::workingScaleBits, 0.25, 2, "Proxy scale relative to game resolution; dlsslopd caps it at 1 and at its tier raster unless run with --cpu-compose or --test-identity", 'w', true, kImage, false},
    {"downscaler", &ShmHeader::scalingDownscaler, 1, 7, "Supersampling down-leg filter, used only above working-scale 1 with dlsslopd --cpu-compose or --test-identity: 1 bicubic, 2 catmull, 3 lanczos2, 4 lanczos3, 5 kaiser2, 6 kaiser3, 7 magic", 'f', false, kImage, false, "Bicubic|Catmull|Lanczos 2|Lanczos 3|Kaiser 2|Kaiser 3|Magic"},
    {"compare", &ShmHeader::compareMode, 0, 2, "0 off, 1 side by side, 2 wipe", 'p', false, kInspect, false, "Off|Side by side|Wipe"},
    {"compare-split", &ShmHeader::compareSplitBits, 0, 1, "Comparison split position", 'x', true, kInspect, false},
    {"compare-zoom", &ShmHeader::compareZoomBits, 1, 2, "Side-by-side comparison magnification", 'z', true, kInspect, false},
    {"compare-swap", &ShmHeader::compareSwap, 0, 1, "Swap comparison sides (0/1)", 'X', false, kInspect, false},
    {"apply-model", &ShmHeader::applyModel, 0, 1, "Apply the edit; 0 shows the clean frame but retains inference cost", 'm', false, kNeural, false},
    {"hold", &ShmHeader::holdFrame, 0, 1, "Freeze the input frame (0/1)", 'H', false, kNeural, false},
    {"toggle-key", &ShmHeader::toggleKey, 0, KEY_MAX, "Linux KEY_ code for the layer hotkey; 0 disables it", 'k', false, kInspect, false},
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
    return s.section == kModel;
}
inline double value(const Setting& s, uint32_t raw)
{
    return s.isFloat ? double(BitsToFloat(raw)) : double(raw);
}
// The bypass default for the worker mode: a started worker that publishes no
// neural raster (--cpu-compose, --test-identity) returns final images. A
// never-started channel keeps the native-composition default.
inline bool workerBypass(const ShmHeader* h)
{
    return h->heartbeat.load() && !h->nativeModelMaxWidth.load() && !h->nativeModelMaxHeight.load();
}
} // namespace dlsslop_control
