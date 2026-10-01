// The layer's pure functions on the host: the settings snapshot a frame composes with, the model
// raster, the composition's format table, the scaler names and the toggle key's names. The values
// are the C++ layer's, written out, so that a port has to reproduce them.
#include "composition.h"
#include "hotkey.h"
#include "shm_protocol.h"

#include <linux/input-event-codes.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace {

using dlssnr::FrameSettings;

[[noreturn]] void Fail(const std::string& message) {
    std::fprintf(stderr, "layer-units-test: %s\n", message.c_str());
    std::exit(1);
}

void Require(bool condition, const std::string& message) {
    if (!condition) Fail(message);
}

// Every field of FrameSettings, with its initial value.
struct RealField {
    const char* name;
    float FrameSettings::*field;
    float initial;
};

struct WordField {
    const char* name;
    uint32_t FrameSettings::*field;
    uint32_t initial;
};

constexpr RealField kReals[] = {
    {"transferStrength", &FrameSettings::transferStrength, 1.0f},
    {"colourStrength", &FrameSettings::colourStrength, 1.0f},
    {"maxRatio", &FrameSettings::maxRatio, 2.0f},
    {"debugScale", &FrameSettings::debugScale, 1.0f},
    {"whitePointManual", &FrameSettings::whitePointManual, 1.0f},
    {"whitePointScale", &FrameSettings::whitePointScale, 1.0f},
    {"whitePointTrim", &FrameSettings::whitePointTrim, 1.0f},
    {"compareSplit", &FrameSettings::compareSplit, 0.5f},
    {"compareZoom", &FrameSettings::compareZoom, 1.0f},
    {"workingScale", &FrameSettings::workingScale, 1.0f},
    {"ghostSlack", &FrameSettings::ghostSlack, 0.5f},
    {"editBlur", &FrameSettings::editBlur, 0.04f},
    {"motionSmooth", &FrameSettings::motionSmooth, 1.0f},
    {"colourTrust", &FrameSettings::colourTrust, 1.0f},
    {"ratioSmooth", &FrameSettings::ratioSmooth, 0.0f},
};

constexpr WordField kWords[] = {
    {"controlSeq", &FrameSettings::controlSeq, 0},
    {"tuningSeq", &FrameSettings::tuningSeq, 0},
    {"passes", &FrameSettings::passes, 0},
    {"whitePointSource", &FrameSettings::whitePointSource, kWhitePointManual},
    {"nativeModelMaxWidth", &FrameSettings::nativeModelMaxWidth, 0},
    {"nativeModelMaxHeight", &FrameSettings::nativeModelMaxHeight, 0},
    {"transfer", &FrameSettings::transfer, 1},
    {"debugView", &FrameSettings::debugView, 0},
    {"compareMode", &FrameSettings::compareMode, 0},
    {"compareSwap", &FrameSettings::compareSwap, 0},
    {"reversibleMode", &FrameSettings::reversibleMode, kReversibleKnee},
    {"applyModel", &FrameSettings::applyModel, 1},
    {"holdFrame", &FrameSettings::holdFrame, 0},
    {"downscaler", &FrameSettings::downscaler, dlssnr::kScalerLanczos3},
    {"compositionBypass", &FrameSettings::compositionBypass, 0},
};

static_assert(std::size(kReals) + std::size(kWords) == sizeof(FrameSettings) / 4,
              "every FrameSettings field needs a row");

FrameSettings Initial() {
    FrameSettings s;
    for (const RealField& f : kReals) s.*f.field = f.initial;
    for (const WordField& f : kWords) s.*f.field = f.initial;
    return s;
}

// Field by field, and bit for bit for the reals: -0.0 is not 0.0 here.
void Expect(const FrameSettings& got, const FrameSettings& want, const std::string& label) {
    for (const RealField& f : kReals) {
        if (FloatToBits(got.*f.field) == FloatToBits(want.*f.field)) continue;
        char text[160];
        std::snprintf(text, sizeof text, "%s: %s is %.9g, not %.9g", label.c_str(), f.name,
                      double(got.*f.field), double(want.*f.field));
        Fail(text);
    }
    for (const WordField& f : kWords)
        Require(got.*f.field == want.*f.field, label + ": " + f.name + " is " + std::to_string(got.*f.field) +
                ", not " + std::to_string(want.*f.field));
}

// A header as the daemon creates it, and what the layer reads from that.
struct Header {
    ShmHeader h{};
    Header() { ShmInitNativeDefaults(&h); }
};

FrameSettings NativeDefaults() {
    FrameSettings s = Initial();
    s.passes = kNativeDefaultPasses;
    s.transfer = 2;
    s.ratioSmooth = 1.0f;
    s.colourTrust = 2.0f;
    return s;
}

// The five overrides FrameSettings::Read takes from the environment, read once per process: each
// row runs in a child of its own, forked before this process reads any settings.
struct Override {
    const char* variable;
    const char* value;
    float FrameSettings::*field;
    float want;
};

const Override kOverrides[] = {
    // Percent; no bound above, a negative number or none keeps the default.
    {"DLSSNR_GHOST_SLACK", "25", &FrameSettings::ghostSlack, 25 / 100.0f},
    {"DLSSNR_GHOST_SLACK", "150", &FrameSettings::ghostSlack, 150 / 100.0f},
    {"DLSSNR_GHOST_SLACK", "0", &FrameSettings::ghostSlack, 0.0f},
    {"DLSSNR_GHOST_SLACK", "", &FrameSettings::ghostSlack, 0.5f},
    {"DLSSNR_GHOST_SLACK", "-5", &FrameSettings::ghostSlack, 0.5f},
    // atoi: a prefix of digits counts, and no digits are 0.
    {"DLSSNR_GHOST_SLACK", "12abc", &FrameSettings::ghostSlack, 12 / 100.0f},
    {"DLSSNR_GHOST_SLACK", "abc", &FrameSettings::ghostSlack, 0.0f},
    // Percent over the header's ratioSmoothPercent, up to 1.
    {"DLSSNR_RATIO_SMOOTH", "50", &FrameSettings::ratioSmooth, 50 / 100.0f},
    {"DLSSNR_RATIO_SMOOTH", "150", &FrameSettings::ratioSmooth, 1.0f},
    {"DLSSNR_RATIO_SMOOTH", "0", &FrameSettings::ratioSmooth, 0.0f},
    {"DLSSNR_RATIO_SMOOTH", "", &FrameSettings::ratioSmooth, 1.0f},
    {"DLSSNR_RATIO_SMOOTH", "-5", &FrameSettings::ratioSmooth, 1.0f},
    // Percent over the header's colourTrustPercent, up to 8.
    {"DLSSNR_COLOUR_TRUST", "50", &FrameSettings::colourTrust, 50 / 100.0f},
    {"DLSSNR_COLOUR_TRUST", "900", &FrameSettings::colourTrust, 8.0f},
    {"DLSSNR_COLOUR_TRUST", "", &FrameSettings::colourTrust, 2.0f},
    {"DLSSNR_COLOUR_TRUST", "-1", &FrameSettings::colourTrust, 2.0f},
    // Percent, up to 1; the header has no field for it.
    {"DLSSNR_MOTION_SMOOTH", "40", &FrameSettings::motionSmooth, 40 / 100.0f},
    {"DLSSNR_MOTION_SMOOTH", "150", &FrameSettings::motionSmooth, 1.0f},
    {"DLSSNR_MOTION_SMOOTH", "0", &FrameSettings::motionSmooth, 0.0f},
    {"DLSSNR_MOTION_SMOOTH", "", &FrameSettings::motionSmooth, 1.0f},
    {"DLSSNR_MOTION_SMOOTH", "-1", &FrameSettings::motionSmooth, 1.0f},
    // Per mille, up to 0.25.
    {"DLSSNR_EDIT_BLUR", "100", &FrameSettings::editBlur, 100 / 1000.0f},
    {"DLSSNR_EDIT_BLUR", "300", &FrameSettings::editBlur, 0.25f},
    {"DLSSNR_EDIT_BLUR", "0", &FrameSettings::editBlur, 0.0f},
    {"DLSSNR_EDIT_BLUR", "", &FrameSettings::editBlur, 0.04f},
    {"DLSSNR_EDIT_BLUR", "-1", &FrameSettings::editBlur, 0.04f},
};

void CheckOverrides() {
    for (const Override& o : kOverrides) {
        const std::string label = std::string(o.variable) + "=\"" + o.value + "\"";
        std::fflush(nullptr);
        const pid_t child = fork();
        Require(child >= 0, "fork failed");
        if (!child) {
            if (setenv(o.variable, o.value, 1)) _exit(2);
            const Header header;
            FrameSettings want = NativeDefaults();
            want.*o.field = o.want;
            Expect(FrameSettings::Read(&header.h), want, label);
            std::fflush(nullptr);
            _exit(0);
        }
        int status = 0;
        while (waitpid(child, &status, 0) < 0)
            Require(errno == EINTR, "waitpid failed");
        Require(WIFEXITED(status) && !WEXITSTATUS(status), label + " gave other settings");
    }
}

// The header's floats, and what Read makes of one that is not a number or infinite: the default,
// not a bound.
struct HeaderReal {
    std::atomic<uint32_t> ShmHeader::*bits;
    float FrameSettings::*field;
    float fallback;
};

const HeaderReal kHeaderReals[] = {
    {&ShmHeader::transferStrengthBits, &FrameSettings::transferStrength, 1.0f},
    {&ShmHeader::colourStrengthBits, &FrameSettings::colourStrength, 1.0f},
    {&ShmHeader::maxRatioBits, &FrameSettings::maxRatio, 2.0f},
    {&ShmHeader::debugScaleBits, &FrameSettings::debugScale, 1.0f},
    {&ShmHeader::whitePointBits, &FrameSettings::whitePointManual, 1.0f},
    {&ShmHeader::whitePointScaleBits, &FrameSettings::whitePointScale, 1.0f},
    {&ShmHeader::whitePointTrimBits, &FrameSettings::whitePointTrim, 1.0f},
    {&ShmHeader::compareSplitBits, &FrameSettings::compareSplit, 0.5f},
    {&ShmHeader::compareZoomBits, &FrameSettings::compareZoom, 1.0f},
    {&ShmHeader::workingScaleBits, &FrameSettings::workingScale, 1.0f},
};

// A finite header float and what Read makes of it: clamped to the nearer bound, or unchanged
// within them. -0.0 stays -0.0, as std::max keeps its first argument when both compare equal.
struct RealCase {
    std::atomic<uint32_t> ShmHeader::*bits;
    float FrameSettings::*field;
    float value, want;
};

const RealCase kRealCases[] = {
    {&ShmHeader::transferStrengthBits, &FrameSettings::transferStrength, -1.0f, 0.0f},
    {&ShmHeader::transferStrengthBits, &FrameSettings::transferStrength, -0.0f, -0.0f},
    {&ShmHeader::transferStrengthBits, &FrameSettings::transferStrength, 0.0f, 0.0f},
    {&ShmHeader::transferStrengthBits, &FrameSettings::transferStrength, 5.0f, 4.0f},
    {&ShmHeader::colourStrengthBits, &FrameSettings::colourStrength, -2.0f, 0.0f},
    {&ShmHeader::colourStrengthBits, &FrameSettings::colourStrength, 3.5f, 3.5f},
    {&ShmHeader::colourStrengthBits, &FrameSettings::colourStrength, 4.5f, 4.0f},
    {&ShmHeader::maxRatioBits, &FrameSettings::maxRatio, 0.5f, 1.0f},
    {&ShmHeader::maxRatioBits, &FrameSettings::maxRatio, 7.25f, 7.25f},
    {&ShmHeader::maxRatioBits, &FrameSettings::maxRatio, 100.0f, float(kMaxPasses)},
    {&ShmHeader::debugScaleBits, &FrameSettings::debugScale, 0.0f, 0.01f},
    {&ShmHeader::debugScaleBits, &FrameSettings::debugScale, 0.01f, 0.01f},
    {&ShmHeader::debugScaleBits, &FrameSettings::debugScale, 1000.0f, 100.0f},
    {&ShmHeader::whitePointBits, &FrameSettings::whitePointManual, 0.0f, 1e-4f},
    {&ShmHeader::whitePointBits, &FrameSettings::whitePointManual, 2000.0f, 2000.0f},
    {&ShmHeader::whitePointBits, &FrameSettings::whitePointManual, 1e6f, 2000.0f},
    {&ShmHeader::whitePointScaleBits, &FrameSettings::whitePointScale, -3.0f, 0.01f},
    {&ShmHeader::whitePointScaleBits, &FrameSettings::whitePointScale, 0.3f, 0.3f},
    {&ShmHeader::whitePointScaleBits, &FrameSettings::whitePointScale, 101.0f, 100.0f},
    {&ShmHeader::whitePointTrimBits, &FrameSettings::whitePointTrim, 0.001f, 0.01f},
    {&ShmHeader::whitePointTrimBits, &FrameSettings::whitePointTrim, 100.0f, 100.0f},
    {&ShmHeader::whitePointTrimBits, &FrameSettings::whitePointTrim, 1e30f, 100.0f},
    {&ShmHeader::compareSplitBits, &FrameSettings::compareSplit, -0.5f, 0.0f},
    {&ShmHeader::compareSplitBits, &FrameSettings::compareSplit, 1.0f, 1.0f},
    {&ShmHeader::compareSplitBits, &FrameSettings::compareSplit, 2.0f, 1.0f},
    {&ShmHeader::compareZoomBits, &FrameSettings::compareZoom, 0.5f, 1.0f},
    {&ShmHeader::compareZoomBits, &FrameSettings::compareZoom, 1.5f, 1.5f},
    {&ShmHeader::compareZoomBits, &FrameSettings::compareZoom, 3.0f, 2.0f},
    {&ShmHeader::workingScaleBits, &FrameSettings::workingScale, 0.1f, 0.25f},
    {&ShmHeader::workingScaleBits, &FrameSettings::workingScale, 0.25f, 0.25f},
    {&ShmHeader::workingScaleBits, &FrameSettings::workingScale, 4.0f, 2.0f},
};

// Reads a header that differs from the daemon's defaults in one float.
void CheckReal(std::atomic<uint32_t> ShmHeader::*bits, float FrameSettings::*field, float value, float want) {
    Header header;
    (header.h.*bits).store(FloatToBits(value));
    FrameSettings expected = NativeDefaults();
    expected.*field = want;
    Expect(FrameSettings::Read(&header.h), expected, "a header value of " + std::to_string(value));
}

void CheckRead() {
    Expect(FrameSettings{}, Initial(), "the initializers");
    Expect(FrameSettings::Read(nullptr), Initial(), "no header");
    {
        const Header header;
        Expect(FrameSettings::Read(&header.h), NativeDefaults(), "the daemon's defaults");
    }
    {
        // What the layer writes when it creates the channel.
        ShmHeader h{};
        ShmInitDefaults(&h);
        FrameSettings want = NativeDefaults();
        want.transfer = 1;
        want.compositionBypass = 1;
        Expect(FrameSettings::Read(&h), want, "the layer's defaults");
    }
    for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()})
        for (const HeaderReal& r : kHeaderReals) CheckReal(r.bits, r.field, value, r.fallback);
    for (const RealCase& c : kRealCases) CheckReal(c.bits, c.field, c.value, c.want);

    // The words: some clamp to a bound, some fall back to a default, the rest pass as they are.
    struct Word {
        std::atomic<uint32_t> ShmHeader::*source;
        uint32_t value;
        uint32_t FrameSettings::*field;
        uint32_t want;
    };
    static const Word kWordCases[] = {
        {&ShmHeader::whitePointSource, kWhitePointMeasured, &FrameSettings::whitePointSource, kWhitePointMeasured},
        {&ShmHeader::whitePointSource, kWhitePointMeasured + 1, &FrameSettings::whitePointSource, kWhitePointManual},
        {&ShmHeader::transfer, 0, &FrameSettings::transfer, 0},
        {&ShmHeader::transfer, 3, &FrameSettings::transfer, 2},
        {&ShmHeader::transfer, UINT32_MAX, &FrameSettings::transfer, 2},
        {&ShmHeader::debugView, 5, &FrameSettings::debugView, 5},
        {&ShmHeader::debugView, 6, &FrameSettings::debugView, 0},
        {&ShmHeader::compareMode, 2, &FrameSettings::compareMode, 2},
        {&ShmHeader::compareMode, 3, &FrameSettings::compareMode, 0},
        {&ShmHeader::reversibleMode, kReversibleModeCount - 1, &FrameSettings::reversibleMode,
         kReversibleModeCount - 1},
        {&ShmHeader::reversibleMode, kReversibleModeCount, &FrameSettings::reversibleMode, kReversibleKnee},
        {&ShmHeader::scalingDownscaler, dlssnr::kScalerBicubic, &FrameSettings::downscaler, dlssnr::kScalerBicubic},
        {&ShmHeader::scalingDownscaler, dlssnr::kScalerMagic, &FrameSettings::downscaler, dlssnr::kScalerMagic},
        {&ShmHeader::scalingDownscaler, dlssnr::kScalerFsr1, &FrameSettings::downscaler, dlssnr::kScalerLanczos3},
        {&ShmHeader::scalingDownscaler, dlssnr::kScalerCount, &FrameSettings::downscaler, dlssnr::kScalerLanczos3},
        {&ShmHeader::controlSeq, 77, &FrameSettings::controlSeq, 77},
        {&ShmHeader::tuningSeq, 78, &FrameSettings::tuningSeq, 78},
        {&ShmHeader::passes, 99, &FrameSettings::passes, 99},
        {&ShmHeader::compareSwap, 7, &FrameSettings::compareSwap, 7},
        {&ShmHeader::applyModel, 0, &FrameSettings::applyModel, 0},
        {&ShmHeader::holdFrame, 3, &FrameSettings::holdFrame, 3},
        {&ShmHeader::compositionBypass, 2, &FrameSettings::compositionBypass, 2},
    };
    for (const Word& w : kWordCases) {
        Header header;
        (header.h.*w.source).store(w.value);
        FrameSettings want = NativeDefaults();
        want.*w.field = w.want;
        Expect(FrameSettings::Read(&header.h), want, "word " + std::to_string(w.value));
    }

    // Percentages from the header: ratio smoothing up to 1, colour trust up to 8.
    struct Percent {
        std::atomic<uint32_t> ShmHeader::*source;
        uint32_t value;
        float FrameSettings::*field;
        float want;
    };
    static const Percent kPercents[] = {
        {&ShmHeader::ratioSmoothPercent, 0, &FrameSettings::ratioSmooth, 0.0f},
        {&ShmHeader::ratioSmoothPercent, 50, &FrameSettings::ratioSmooth, 50 / 100.0f},
        {&ShmHeader::ratioSmoothPercent, 150, &FrameSettings::ratioSmooth, 1.0f},
        {&ShmHeader::colourTrustPercent, 0, &FrameSettings::colourTrust, 0.0f},
        {&ShmHeader::colourTrustPercent, 350, &FrameSettings::colourTrust, 350 / 100.0f},
        {&ShmHeader::colourTrustPercent, 1000, &FrameSettings::colourTrust, 8.0f},
    };
    for (const Percent& p : kPercents) {
        Header header;
        (header.h.*p.source).store(p.value);
        FrameSettings want = NativeDefaults();
        want.*p.field = p.want;
        Expect(FrameSettings::Read(&header.h), want, "percent " + std::to_string(p.value));
    }

    // The native model's maximum raster: a pair within kMinW..kMaxW x kMinH..kMaxH, or none.
    struct Maxima {
        uint32_t width, height, wantWidth, wantHeight;
    };
    static const Maxima kMaxima[] = {
        {kMinW, kMinH, kMinW, kMinH},
        {kMaxW, kMaxH, kMaxW, kMaxH},
        {1280, 720, 1280, 720},
        {kMaxW + 1, kMaxH, 0, 0},
        {kMaxW, kMaxH + 1, 0, 0},
        {kMinW - 1, 720, 0, 0},
        {1280, kMinH - 1, 0, 0},
        {1280, 0, 0, 0},
        {0, 0, 0, 0},
    };
    for (const Maxima& m : kMaxima) {
        Header header;
        header.h.nativeModelMaxWidth.store(m.width);
        header.h.nativeModelMaxHeight.store(m.height);
        FrameSettings want = NativeDefaults();
        want.nativeModelMaxWidth = m.wantWidth;
        want.nativeModelMaxHeight = m.wantHeight;
        Expect(FrameSettings::Read(&header.h), want,
               "maxima " + std::to_string(m.width) + "x" + std::to_string(m.height));
    }
}

void CheckModelExtent() {
    struct Case {
        uint32_t width, height;
        float workingScale;
        uint32_t maxWidth, maxHeight;
        uint32_t modelWidth, modelHeight;
    };
    static const Case kCases[] = {
        // Without a native maximum: the working scale, rounded half away from zero, and at least 64
        // unless the scale is exactly 1.
        {1920, 1080, 1.0f, 0, 0, 1920, 1080},
        {32, 16, 1.0f, 0, 0, 32, 16},
        {32, 16, 0.999f, 0, 0, 64, 64},
        {640, 360, 0.999f, 0, 0, 639, 360},
        {1920, 1080, 0.5f, 0, 0, 960, 540},
        {1001, 333, 0.5f, 0, 0, 501, 167},
        {1280, 720, 0.75f, 0, 0, 960, 540},
        {200, 100, 0.25f, 0, 0, 64, 64},
        {1280, 720, 2.0f, 0, 0, 2560, 1440},
        // Half a maximum, or a frame side of zero, is no maximum.
        {1920, 1080, 0.5f, 960, 0, 960, 540},
        {1920, 1080, 0.5f, 0, 540, 960, 540},
        {0, 100, 1.0f, 960, 540, 0, 100},
        // With one: the least of the working scale, 1 and each side's ratio to the maximum, and at
        // least kMinW x kMinH. It never supersamples.
        {1920, 1080, 1.0f, 960, 540, 960, 540},
        {1920, 1080, 1.0f, 1280, 1280, 1280, 720},
        {320, 192, 1.0f, 160, 96, 160, 96},
        {640, 360, 1.0f, 1920, 1080, 640, 360},
        {640, 360, 2.0f, 1920, 1080, 640, 360},
        {1920, 1080, 0.25f, 3840, 2160, 480, 270},
        {1001, 333, 0.5f, 3840, 2160, 501, 167},
        {100, 80, 1.0f, 64, 64, 64, 64},
        {32, 32, 1.0f, 64, 64, 64, 64},
        // Both paths take the product in double: 105 * 0.7f is 73.4999987 in double and rounds to 73,
        // while the product in float is 73.5 and rounds to 74.
        {150, 105, 0.7f, 0, 0, 105, 73},
        {105, 105, 0.7f, 3840, 2160, 73, 73},
    };
    for (const Case& c : kCases) {
        FrameSettings s = Initial();
        s.workingScale = c.workingScale;
        s.nativeModelMaxWidth = c.maxWidth;
        s.nativeModelMaxHeight = c.maxHeight;
        uint32_t width = 0, height = 0;
        dlssnr::Composition::ModelExtent(c.width, c.height, s, width, height);
        char label[160];
        std::snprintf(label, sizeof label, "ModelExtent(%ux%u, scale %g, maximum %ux%u) is %ux%u, not %ux%u",
                      c.width, c.height, double(c.workingScale), c.maxWidth, c.maxHeight, width, height,
                      c.modelWidth, c.modelHeight);
        Require(width == c.modelWidth && height == c.modelHeight, label);
    }
}

void CheckFormats() {
    struct Format {
        VkFormat swapchain, composition;
    };
    static const Format kFormats[] = {
        {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM},
        {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM},
        {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
        {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM},
        {VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_A8B8G8R8_UNORM_PACK32},
        {VK_FORMAT_A8B8G8R8_SRGB_PACK32, VK_FORMAT_A8B8G8R8_UNORM_PACK32},
        {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
        {VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32},
        {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
        {VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_UNDEFINED},
        {VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_UNDEFINED},
        {VkFormat(1000452000), VK_FORMAT_UNDEFINED},
    };
    for (const Format& f : kFormats)
        Require(dlssnr::CompositionFormat(f.swapchain) == f.composition,
                "CompositionFormat(" + std::to_string(f.swapchain) + ") is " +
                std::to_string(dlssnr::CompositionFormat(f.swapchain)));

    // Display-referred and linear HDR are forced; auto, and any other mode, takes a float
    // swapchain for light and everything else for a picture.
    for (const Format& f : kFormats) {
        const bool fp16 = f.swapchain == VK_FORMAT_R16G16B16A16_SFLOAT;
        for (const auto& [mode, linear] : {std::pair{uint32_t(kColourAuto), fp16},
                                           std::pair{uint32_t(kColourDisplay), false},
                                           std::pair{uint32_t(kColourLinearHdr), true},
                                           std::pair{3u, fp16}, std::pair{UINT32_MAX, fp16}})
            Require(dlssnr::ColourIsLinearHdr(f.swapchain, mode) == linear,
                    "ColourIsLinearHdr(" + std::to_string(f.swapchain) + ", " + std::to_string(mode) + ")");
    }
}

void CheckScalerNames() {
    static const char* const kNames[] = {"lanczos3", "bicubic", "catmull-rom", "lanczos2",
                                         "lanczos3", "kaiser2", "kaiser3", "magic"};
    static_assert(std::size(kNames) == dlssnr::kScalerCount);
    for (uint32_t filter = 0; filter != dlssnr::kScalerCount; ++filter)
        Require(!std::strcmp(dlssnr::ScalerFilterName(filter), kNames[filter]),
                "ScalerFilterName(" + std::to_string(filter) + ")");
    for (const uint32_t filter : {uint32_t(dlssnr::kScalerCount), 100u, UINT32_MAX})
        Require(!std::strcmp(dlssnr::ScalerFilterName(filter), "lanczos3"),
                "ScalerFilterName(" + std::to_string(filter) + ")");
}

void CheckKeyNames() {
    struct Key {
        const char* name;
        uint32_t code;
    };
    static const Key kKeys[] = {
        {"", 0},
        {"F10", KEY_F10},
        {"f10", KEY_F10},
        {"KEY_HOME", KEY_HOME},
        {"key_home", KEY_HOME},
        {"Home", KEY_HOME},
        {"123", 123},
        {"nonsense", 0},
        // A bare number is a key code, also where the table names a digit key.
        {"1", 1},
        {"KEY_1", KEY_1},
        {"0", 0},
        {"KEY_0", KEY_0},
        // The code wraps to 32 bits.
        {"4294967297", 1},
        {"12a", 0},
        {" F10", 0},
        {"F10 ", 0},
        {"KEY_", 0},
        {"KEY_KEY_F10", 0},
        {"z", KEY_Z},
        {"grave", KEY_GRAVE},
        {"SysRq", KEY_SYSRQ},
    };
    for (const Key& k : kKeys)
        Require(dlssnr::KeyCodeFromName(k.name) == k.code,
                std::string("KeyCodeFromName(\"") + k.name + "\") is " +
                std::to_string(dlssnr::KeyCodeFromName(k.name)) + ", not " + std::to_string(k.code));

    static const Key kCodes[] = {
        {"F1", KEY_F1}, {"F10", KEY_F10}, {"F12", KEY_F12}, {"HOME", KEY_HOME}, {"END", KEY_END},
        {"INSERT", KEY_INSERT}, {"DELETE", KEY_DELETE}, {"PAGEUP", KEY_PAGEUP}, {"PAGEDOWN", KEY_PAGEDOWN},
        {"PAUSE", KEY_PAUSE}, {"SCROLLLOCK", KEY_SCROLLLOCK}, {"SYSRQ", KEY_SYSRQ}, {"GRAVE", KEY_GRAVE},
        {"A", KEY_A}, {"Z", KEY_Z}, {"0", KEY_0}, {"1", KEY_1}, {"9", KEY_9},
        {"?", 0}, {"?", KEY_ESC}, {"?", KEY_F13}, {"?", KEY_KPPLUSMINUS}, {"?", UINT32_MAX},
    };
    for (const Key& k : kCodes)
        Require(!std::strcmp(dlssnr::KeyNameFromCode(k.code), k.name),
                "KeyNameFromCode(" + std::to_string(k.code) + ") is " + dlssnr::KeyNameFromCode(k.code));

    // Every name the table has reads back as its code, except the digits, which read as numbers.
    // 12 function keys, 10 others, 26 letters and 10 digits.
    uint32_t named = 0;
    for (uint32_t code = 0; code <= KEY_MAX; ++code) {
        const char* name = dlssnr::KeyNameFromCode(code);
        if (!std::strcmp(name, "?")) continue;
        ++named;
        const bool digit = name[0] >= '0' && name[0] <= '9' && !name[1];
        const uint32_t want = digit ? uint32_t(name[0] - '0') : code;
        Require(dlssnr::KeyCodeFromName(name) == want, std::string("the name ") + name + " does not read back");
    }
    Require(named == 58, "the key table has " + std::to_string(named) + " names, not 58");
}

}  // namespace

int main() {
    // The overrides first: Read latches them in this process at its first call, and the children
    // forked below must each make that first call themselves.
    for (const char* variable : {"DLSSNR_GHOST_SLACK", "DLSSNR_RATIO_SMOOTH", "DLSSNR_COLOUR_TRUST",
                                 "DLSSNR_MOTION_SMOOTH", "DLSSNR_EDIT_BLUR"})
        Require(!unsetenv(variable), "unsetenv failed");
    CheckOverrides();
    CheckRead();
    CheckModelExtent();
    CheckFormats();
    CheckScalerNames();
    CheckKeyNames();
    std::printf("layer-units-test: settings, model extents, formats, scaler and key names hold\n");
    return 0;
}
