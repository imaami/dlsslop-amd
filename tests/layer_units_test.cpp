// The layer's pure functions on the host: the settings snapshot a frame composes with, the model
// raster, the composition's format table, the scaler names, the toggle key's names, and the log's
// switches, lines and clock. The values are the C++ layer's, written out, so that a port has to
// reproduce them.
#include "composition.h"
#include "hotkey.h"
#include "log.h"
#include "shm_protocol.h"

#include <linux/input-event-codes.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// The settings, under a name that does not hide behind the function composition_frame_settings().
using Settings = struct composition_frame_settings;

[[noreturn]] void Fail(const std::string& message) {
    std::fprintf(stderr, "layer-units-test: %s\n", message.c_str());
    std::exit(1);
}

void Require(bool condition, const std::string& message) {
    if (!condition) Fail(message);
}

// Every field of the settings, with its initial value.
struct RealField {
    const char* name;
    float Settings::*field;
    float initial;
};

struct WordField {
    const char* name;
    uint32_t Settings::*field;
    uint32_t initial;
};

constexpr RealField kReals[] = {
    {"transfer_strength", &Settings::transfer_strength, 1.0f},
    {"colour_strength", &Settings::colour_strength, 1.0f},
    {"max_ratio", &Settings::max_ratio, 2.0f},
    {"debug_scale", &Settings::debug_scale, 1.0f},
    {"white_point_manual", &Settings::white_point_manual, 1.0f},
    {"white_point_scale", &Settings::white_point_scale, 1.0f},
    {"white_point_trim", &Settings::white_point_trim, 1.0f},
    {"compare_split", &Settings::compare_split, 0.5f},
    {"compare_zoom", &Settings::compare_zoom, 1.0f},
    {"working_scale", &Settings::working_scale, 1.0f},
    {"ghost_slack", &Settings::ghost_slack, 0.5f},
    {"edit_blur", &Settings::edit_blur, 0.04f},
    {"motion_smooth", &Settings::motion_smooth, 1.0f},
    {"colour_trust", &Settings::colour_trust, 1.0f},
    {"ratio_smooth", &Settings::ratio_smooth, 0.0f},
};

constexpr WordField kWords[] = {
    {"control_seq", &Settings::control_seq, 0},
    {"tuning_seq", &Settings::tuning_seq, 0},
    {"passes", &Settings::passes, 0},
    {"white_point_source", &Settings::white_point_source, kWhitePointManual},
    {"native_model_max_width", &Settings::native_model_max_width, 0},
    {"native_model_max_height", &Settings::native_model_max_height, 0},
    {"transfer", &Settings::transfer, 1},
    {"debug_view", &Settings::debug_view, 0},
    {"compare_mode", &Settings::compare_mode, 0},
    {"compare_swap", &Settings::compare_swap, 0},
    {"reversible_mode", &Settings::reversible_mode, kReversibleKnee},
    {"apply_model", &Settings::apply_model, 1},
    {"hold_frame", &Settings::hold_frame, 0},
    {"downscaler", &Settings::downscaler, SCALER_VK_LANCZOS3},
    {"composition_bypass", &Settings::composition_bypass, 0},
};

static_assert(std::size(kReals) + std::size(kWords) == sizeof(Settings) / 4, "every settings field needs a row");

Settings Initial() {
    Settings s{};
    for (const RealField& f : kReals) s.*f.field = f.initial;
    for (const WordField& f : kWords) s.*f.field = f.initial;
    return s;
}

// Field by field, and bit for bit for the reals: -0.0 is not 0.0 here.
void Expect(const Settings& got, const Settings& want, const std::string& label) {
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
    Header() { ShmInitNativeDefaults(&h, false); }
};

Settings NativeDefaults() {
    Settings s = Initial();
    s.passes = kNativeDefaultPasses;
    s.transfer = 2;
    s.ratio_smooth = 1.0f;
    s.colour_trust = 2.0f;
    return s;
}

// The five overrides composition_frame_settings_read() takes from the environment, read once per
// process: each row runs in a child of its own, forked before this process reads any settings.
struct Override {
    const char* variable;
    const char* value;
    float Settings::*field;
    float want;
};

const Override kOverrides[] = {
    // Percent; no bound above, a negative number or none keeps the default.
    {"DLSSNR_GHOST_SLACK", "25", &Settings::ghost_slack, 25 / 100.0f},
    {"DLSSNR_GHOST_SLACK", "150", &Settings::ghost_slack, 150 / 100.0f},
    {"DLSSNR_GHOST_SLACK", "0", &Settings::ghost_slack, 0.0f},
    {"DLSSNR_GHOST_SLACK", "", &Settings::ghost_slack, 0.5f},
    {"DLSSNR_GHOST_SLACK", "-5", &Settings::ghost_slack, 0.5f},
    // atoi: a prefix of digits counts, and no digits are 0.
    {"DLSSNR_GHOST_SLACK", "12abc", &Settings::ghost_slack, 12 / 100.0f},
    {"DLSSNR_GHOST_SLACK", "abc", &Settings::ghost_slack, 0.0f},
    // Percent over the header's ratioSmoothPercent, up to 1.
    {"DLSSNR_RATIO_SMOOTH", "50", &Settings::ratio_smooth, 50 / 100.0f},
    {"DLSSNR_RATIO_SMOOTH", "150", &Settings::ratio_smooth, 1.0f},
    {"DLSSNR_RATIO_SMOOTH", "0", &Settings::ratio_smooth, 0.0f},
    {"DLSSNR_RATIO_SMOOTH", "", &Settings::ratio_smooth, 1.0f},
    {"DLSSNR_RATIO_SMOOTH", "-5", &Settings::ratio_smooth, 1.0f},
    // Percent over the header's colourTrustPercent, up to 8.
    {"DLSSNR_COLOUR_TRUST", "50", &Settings::colour_trust, 50 / 100.0f},
    {"DLSSNR_COLOUR_TRUST", "900", &Settings::colour_trust, 8.0f},
    {"DLSSNR_COLOUR_TRUST", "", &Settings::colour_trust, 2.0f},
    {"DLSSNR_COLOUR_TRUST", "-1", &Settings::colour_trust, 2.0f},
    // Percent, up to 1; the header has no field for it.
    {"DLSSNR_MOTION_SMOOTH", "40", &Settings::motion_smooth, 40 / 100.0f},
    {"DLSSNR_MOTION_SMOOTH", "150", &Settings::motion_smooth, 1.0f},
    {"DLSSNR_MOTION_SMOOTH", "0", &Settings::motion_smooth, 0.0f},
    {"DLSSNR_MOTION_SMOOTH", "", &Settings::motion_smooth, 1.0f},
    {"DLSSNR_MOTION_SMOOTH", "-1", &Settings::motion_smooth, 1.0f},
    // Per mille, up to 0.25.
    {"DLSSNR_EDIT_BLUR", "100", &Settings::edit_blur, 100 / 1000.0f},
    {"DLSSNR_EDIT_BLUR", "300", &Settings::edit_blur, 0.25f},
    {"DLSSNR_EDIT_BLUR", "0", &Settings::edit_blur, 0.0f},
    {"DLSSNR_EDIT_BLUR", "", &Settings::edit_blur, 0.04f},
    {"DLSSNR_EDIT_BLUR", "-1", &Settings::edit_blur, 0.04f},
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
            Settings want = NativeDefaults();
            want.*o.field = o.want;
            Expect(composition_frame_settings_read(&header.h), want, label);
            std::fflush(nullptr);
            _exit(0);
        }
        int status = 0;
        while (waitpid(child, &status, 0) < 0)
            Require(errno == EINTR, "waitpid failed");
        Require(WIFEXITED(status) && !WEXITSTATUS(status), label + " gave other settings");
    }
}

// The header's floats, and what composition_frame_settings_read() makes of one that is not a number
// or infinite: the default, not a bound.
struct HeaderReal {
    std::atomic<uint32_t> ShmHeader::*bits;
    float Settings::*field;
    float fallback;
};

const HeaderReal kHeaderReals[] = {
    {&ShmHeader::transferStrengthBits, &Settings::transfer_strength, 1.0f},
    {&ShmHeader::colourStrengthBits, &Settings::colour_strength, 1.0f},
    {&ShmHeader::maxRatioBits, &Settings::max_ratio, 2.0f},
    {&ShmHeader::debugScaleBits, &Settings::debug_scale, 1.0f},
    {&ShmHeader::whitePointBits, &Settings::white_point_manual, 1.0f},
    {&ShmHeader::whitePointScaleBits, &Settings::white_point_scale, 1.0f},
    {&ShmHeader::whitePointTrimBits, &Settings::white_point_trim, 1.0f},
    {&ShmHeader::compareSplitBits, &Settings::compare_split, 0.5f},
    {&ShmHeader::compareZoomBits, &Settings::compare_zoom, 1.0f},
    {&ShmHeader::workingScaleBits, &Settings::working_scale, 1.0f},
};

// A finite header float and what composition_frame_settings_read() makes of it: clamped to the
// nearer bound, or unchanged within them. -0.0 stays -0.0, as std::max keeps its first argument when
// both compare equal.
struct RealCase {
    std::atomic<uint32_t> ShmHeader::*bits;
    float Settings::*field;
    float value, want;
};

const RealCase kRealCases[] = {
    {&ShmHeader::transferStrengthBits, &Settings::transfer_strength, -1.0f, 0.0f},
    {&ShmHeader::transferStrengthBits, &Settings::transfer_strength, -0.0f, -0.0f},
    {&ShmHeader::transferStrengthBits, &Settings::transfer_strength, 0.0f, 0.0f},
    {&ShmHeader::transferStrengthBits, &Settings::transfer_strength, 5.0f, 4.0f},
    {&ShmHeader::colourStrengthBits, &Settings::colour_strength, -2.0f, 0.0f},
    {&ShmHeader::colourStrengthBits, &Settings::colour_strength, 3.5f, 3.5f},
    {&ShmHeader::colourStrengthBits, &Settings::colour_strength, 4.5f, 4.0f},
    {&ShmHeader::maxRatioBits, &Settings::max_ratio, 0.5f, 1.0f},
    {&ShmHeader::maxRatioBits, &Settings::max_ratio, 7.25f, 7.25f},
    {&ShmHeader::maxRatioBits, &Settings::max_ratio, 100.0f, float(kMaxPasses)},
    {&ShmHeader::debugScaleBits, &Settings::debug_scale, 0.0f, 0.01f},
    {&ShmHeader::debugScaleBits, &Settings::debug_scale, 0.01f, 0.01f},
    {&ShmHeader::debugScaleBits, &Settings::debug_scale, 1000.0f, 100.0f},
    {&ShmHeader::whitePointBits, &Settings::white_point_manual, 0.0f, 1e-4f},
    {&ShmHeader::whitePointBits, &Settings::white_point_manual, 2000.0f, 2000.0f},
    {&ShmHeader::whitePointBits, &Settings::white_point_manual, 1e6f, 2000.0f},
    {&ShmHeader::whitePointScaleBits, &Settings::white_point_scale, -3.0f, 0.01f},
    {&ShmHeader::whitePointScaleBits, &Settings::white_point_scale, 0.3f, 0.3f},
    {&ShmHeader::whitePointScaleBits, &Settings::white_point_scale, 101.0f, 100.0f},
    {&ShmHeader::whitePointTrimBits, &Settings::white_point_trim, 0.001f, 0.01f},
    {&ShmHeader::whitePointTrimBits, &Settings::white_point_trim, 100.0f, 100.0f},
    {&ShmHeader::whitePointTrimBits, &Settings::white_point_trim, 1e30f, 100.0f},
    {&ShmHeader::compareSplitBits, &Settings::compare_split, -0.5f, 0.0f},
    {&ShmHeader::compareSplitBits, &Settings::compare_split, 1.0f, 1.0f},
    {&ShmHeader::compareSplitBits, &Settings::compare_split, 2.0f, 1.0f},
    {&ShmHeader::compareZoomBits, &Settings::compare_zoom, 0.5f, 1.0f},
    {&ShmHeader::compareZoomBits, &Settings::compare_zoom, 1.5f, 1.5f},
    {&ShmHeader::compareZoomBits, &Settings::compare_zoom, 3.0f, 2.0f},
    {&ShmHeader::workingScaleBits, &Settings::working_scale, 0.1f, 0.25f},
    {&ShmHeader::workingScaleBits, &Settings::working_scale, 0.25f, 0.25f},
    {&ShmHeader::workingScaleBits, &Settings::working_scale, 4.0f, 2.0f},
};

// Reads a header that differs from the daemon's defaults in one float.
void CheckReal(std::atomic<uint32_t> ShmHeader::*bits, float Settings::*field, float value, float want) {
    Header header;
    (header.h.*bits).store(FloatToBits(value));
    Settings expected = NativeDefaults();
    expected.*field = want;
    Expect(composition_frame_settings_read(&header.h), expected,
           "a header value of " + std::to_string(value));
}

void CheckRead() {
    Expect(composition_frame_settings(), Initial(), "the defaults");
    Expect(composition_frame_settings_read(nullptr), Initial(), "no header");
    {
        const Header header;
        Expect(composition_frame_settings_read(&header.h), NativeDefaults(), "the daemon's defaults");
    }
    {
        // What the layer writes when it creates the channel.
        ShmHeader h{};
        ShmInitDefaults(&h);
        Settings want = NativeDefaults();
        want.transfer = 1;
        want.composition_bypass = 1;
        Expect(composition_frame_settings_read(&h), want, "the layer's defaults");
    }
    for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()})
        for (const HeaderReal& r : kHeaderReals) CheckReal(r.bits, r.field, value, r.fallback);
    for (const RealCase& c : kRealCases) CheckReal(c.bits, c.field, c.value, c.want);

    // The words: some clamp to a bound, some fall back to a default, the rest pass as they are.
    struct Word {
        std::atomic<uint32_t> ShmHeader::*source;
        uint32_t value;
        uint32_t Settings::*field;
        uint32_t want;
    };
    static const Word kWordCases[] = {
        {&ShmHeader::whitePointSource, kWhitePointMeasured, &Settings::white_point_source, kWhitePointMeasured},
        {&ShmHeader::whitePointSource, kWhitePointMeasured + 1, &Settings::white_point_source, kWhitePointManual},
        {&ShmHeader::transfer, 0, &Settings::transfer, 0},
        {&ShmHeader::transfer, 3, &Settings::transfer, 2},
        {&ShmHeader::transfer, UINT32_MAX, &Settings::transfer, 2},
        {&ShmHeader::debugView, 5, &Settings::debug_view, 5},
        {&ShmHeader::debugView, 6, &Settings::debug_view, 0},
        {&ShmHeader::compareMode, 2, &Settings::compare_mode, 2},
        {&ShmHeader::compareMode, 3, &Settings::compare_mode, 0},
        {&ShmHeader::reversibleMode, kReversibleModeCount - 1, &Settings::reversible_mode,
         kReversibleModeCount - 1},
        {&ShmHeader::reversibleMode, kReversibleModeCount, &Settings::reversible_mode, kReversibleKnee},
        {&ShmHeader::scalingDownscaler, SCALER_VK_BICUBIC, &Settings::downscaler, SCALER_VK_BICUBIC},
        {&ShmHeader::scalingDownscaler, SCALER_VK_MAGIC, &Settings::downscaler, SCALER_VK_MAGIC},
        {&ShmHeader::scalingDownscaler, SCALER_VK_FSR1, &Settings::downscaler, SCALER_VK_LANCZOS3},
        {&ShmHeader::scalingDownscaler, SCALER_VK_COUNT, &Settings::downscaler, SCALER_VK_LANCZOS3},
        {&ShmHeader::controlSeq, 77, &Settings::control_seq, 77},
        {&ShmHeader::tuningSeq, 78, &Settings::tuning_seq, 78},
        {&ShmHeader::passes, 99, &Settings::passes, 99},
        {&ShmHeader::compareSwap, 7, &Settings::compare_swap, 7},
        {&ShmHeader::applyModel, 0, &Settings::apply_model, 0},
        {&ShmHeader::holdFrame, 3, &Settings::hold_frame, 3},
        {&ShmHeader::compositionBypass, 2, &Settings::composition_bypass, 2},
    };
    for (const Word& w : kWordCases) {
        Header header;
        (header.h.*w.source).store(w.value);
        Settings want = NativeDefaults();
        want.*w.field = w.want;
        Expect(composition_frame_settings_read(&header.h), want, "word " + std::to_string(w.value));
    }

    // Percentages from the header: ratio smoothing up to 1, colour trust up to 8.
    struct Percent {
        std::atomic<uint32_t> ShmHeader::*source;
        uint32_t value;
        float Settings::*field;
        float want;
    };
    static const Percent kPercents[] = {
        {&ShmHeader::ratioSmoothPercent, 0, &Settings::ratio_smooth, 0.0f},
        {&ShmHeader::ratioSmoothPercent, 50, &Settings::ratio_smooth, 50 / 100.0f},
        {&ShmHeader::ratioSmoothPercent, 150, &Settings::ratio_smooth, 1.0f},
        {&ShmHeader::colourTrustPercent, 0, &Settings::colour_trust, 0.0f},
        {&ShmHeader::colourTrustPercent, 350, &Settings::colour_trust, 350 / 100.0f},
        {&ShmHeader::colourTrustPercent, 1000, &Settings::colour_trust, 8.0f},
    };
    for (const Percent& p : kPercents) {
        Header header;
        (header.h.*p.source).store(p.value);
        Settings want = NativeDefaults();
        want.*p.field = p.want;
        Expect(composition_frame_settings_read(&header.h), want, "percent " + std::to_string(p.value));
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
        Settings want = NativeDefaults();
        want.native_model_max_width = m.wantWidth;
        want.native_model_max_height = m.wantHeight;
        Expect(composition_frame_settings_read(&header.h), want,
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
        Settings s = Initial();
        s.working_scale = c.workingScale;
        s.native_model_max_width = c.maxWidth;
        s.native_model_max_height = c.maxHeight;
        const VkExtent2D model = composition_model_extent(c.width, c.height, &s);
        char label[176];
        std::snprintf(label, sizeof label,
                      "composition_model_extent(%ux%u, scale %g, maximum %ux%u) is %ux%u, not %ux%u", c.width,
                      c.height, double(c.workingScale), c.maxWidth, c.maxHeight, model.width, model.height,
                      c.modelWidth, c.modelHeight);
        Require(model.width == c.modelWidth && model.height == c.modelHeight, label);
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
        Require(composition_format(f.swapchain) == f.composition,
                "composition_format(" + std::to_string(f.swapchain) + ") is " +
                std::to_string(composition_format(f.swapchain)));

    // Display-referred and linear HDR are forced; auto, and any other mode, takes a float
    // swapchain for light and everything else for a picture.
    for (const Format& f : kFormats) {
        const bool fp16 = f.swapchain == VK_FORMAT_R16G16B16A16_SFLOAT;
        for (const auto& [mode, linear] : {std::pair{uint32_t(kColourAuto), fp16},
                                           std::pair{uint32_t(kColourDisplay), false},
                                           std::pair{uint32_t(kColourLinearHdr), true},
                                           std::pair{3u, fp16}, std::pair{UINT32_MAX, fp16}})
            Require(composition_colour_is_linear_hdr(f.swapchain, mode) == linear,
                    "composition_colour_is_linear_hdr(" + std::to_string(f.swapchain) + ", " + std::to_string(mode) +
                    ")");
    }
}

void CheckScalerNames() {
    static const char* const kNames[] = {"lanczos3", "bicubic", "catmull-rom", "lanczos2",
                                         "lanczos3", "kaiser2", "kaiser3", "magic"};
    static_assert(std::size(kNames) == SCALER_VK_COUNT);
    for (uint32_t filter = 0; filter != SCALER_VK_COUNT; ++filter)
        Require(!std::strcmp(scaler_vk_filter_name(filter), kNames[filter]),
                "scaler_vk_filter_name(" + std::to_string(filter) + ")");
    for (const uint32_t filter : {uint32_t(SCALER_VK_COUNT), 100u, UINT32_MAX})
        Require(!std::strcmp(scaler_vk_filter_name(filter), "lanczos3"),
                "scaler_vk_filter_name(" + std::to_string(filter) + ")");
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
        // The code wraps to 32 bits, up to the largest unsigned long; beyond it, nothing is bound.
        {"4294967297", 1},
        {"18446744073709551615", UINT32_MAX},
        {"018446744073709551615", UINT32_MAX},
        {"18446744073709551616", 0},
        {"99999999999999999999999999", 0},
        {"12a", 0},
        {" F10", 0},
        {"F10 ", 0},
        {"KEY_", 0},
        {"KEY_KEY_F10", 0},
        {"z", KEY_Z},
        {"grave", KEY_GRAVE},
        {"SysRq", KEY_SYSRQ},
        // The longest name, and longer ones, which no key has.
        {"KEY_SCROLLLOCK", KEY_SCROLLLOCK},
        {"key_ScrollLock", KEY_SCROLLLOCK},
        {"KEY_SCROLLLOCK1", 0},
        {"KEY_SCROLLLOCK12", 0},
        {"SCROLLLOCK123456", 0},
    };
    Require(hotkey_key_code_from_name(nullptr) == 0, "hotkey_key_code_from_name(nullptr) is not 0");
    Require(hotkey_key_code_from_name(("F10" + std::string(4096, ' ')).c_str()) == 0,
            "a name of 4099 bytes bound a key");
    for (const Key& k : kKeys)
        Require(hotkey_key_code_from_name(k.name) == k.code,
                std::string("hotkey_key_code_from_name(\"") + k.name + "\") is " +
                std::to_string(hotkey_key_code_from_name(k.name)) + ", not " + std::to_string(k.code));

    static const Key kCodes[] = {
        {"F1", KEY_F1}, {"F10", KEY_F10}, {"F12", KEY_F12}, {"HOME", KEY_HOME}, {"END", KEY_END},
        {"INSERT", KEY_INSERT}, {"DELETE", KEY_DELETE}, {"PAGEUP", KEY_PAGEUP}, {"PAGEDOWN", KEY_PAGEDOWN},
        {"PAUSE", KEY_PAUSE}, {"SCROLLLOCK", KEY_SCROLLLOCK}, {"SYSRQ", KEY_SYSRQ}, {"GRAVE", KEY_GRAVE},
        {"A", KEY_A}, {"Z", KEY_Z}, {"0", KEY_0}, {"1", KEY_1}, {"9", KEY_9},
        {"?", 0}, {"?", KEY_ESC}, {"?", KEY_F13}, {"?", KEY_KPPLUSMINUS}, {"?", UINT32_MAX},
    };
    for (const Key& k : kCodes)
        Require(!std::strcmp(hotkey_key_name_from_code(k.code), k.name),
                "hotkey_key_name_from_code(" + std::to_string(k.code) + ") is " + hotkey_key_name_from_code(k.code));

    // Every name the table has reads back as its code, except the digits, which read as numbers.
    // 12 function keys, 10 others, 26 letters and 10 digits.
    uint32_t named = 0;
    for (uint32_t code = 0; code <= KEY_MAX; ++code) {
        const char* name = hotkey_key_name_from_code(code);
        if (!std::strcmp(name, "?")) continue;
        ++named;
        const bool digit = name[0] >= '0' && name[0] <= '9' && !name[1];
        const uint32_t want = digit ? uint32_t(name[0] - '0') : code;
        Require(hotkey_key_code_from_name(name) == want, std::string("the name ") + name + " does not read back");
    }
    Require(named == 58, "the key table has " + std::to_string(named) + " names, not 58");
}

// The log reads its variables once per process, at the first call of a log.h function: each case
// runs in a child of its own, forked before this process calls one. The child leaves through
// exit(), so the log's destructor runs.
const char* const kLogVariables[] = {"VKLayer_DLSS5", "VKLAYER_DLSS5", "DLSSNR_ENABLE", "DLSSNR_LOG",
                                     "DLSSNR_VERBOSE", "DLSSNR_TIME", "DLSSNR_TIME_EVERY"};

// A variable for a child to set; a null value leaves it unset.
struct Env {
    const char* variable;
    const char* value;
};

template <typename Check>
void InLogChild(const std::string& label, const std::vector<Env>& env, Check check) {
    std::fflush(nullptr);
    const pid_t child = fork();
    Require(child >= 0, "fork failed");
    if (!child) {
        for (const Env& e : env)
            if (e.value && setenv(e.variable, e.value, 1)) _exit(2);
        check();
        std::fflush(nullptr);
        std::exit(0);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0)
        Require(errno == EINTR, "waitpid failed");
    Require(WIFEXITED(status) && !WEXITSTATUS(status),
            label + ": the child exited with " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
}

// The switches, which do not depend on whether the layer was asked for.
struct SwitchCase {
    const char* value;
    bool on;
};

const SwitchCase kSwitchCases[] = {
    {nullptr, false}, {"1", true}, {"10", true}, {"1x", true}, {"0", false}, {"", false}, {" 1", false},
    {"yes", false}, {"true", false},
};

// atoi: a prefix of digits counts, leading blanks and a sign are skipped, and anything not positive
// is the default.
struct IntervalCase {
    const char* value;
    uint32_t interval;
};

const IntervalCase kIntervalCases[] = {
    {nullptr, 30}, {"", 30}, {"0", 30}, {"-5", 30}, {"abc", 30}, {"1", 1}, {"45", 45}, {"7x", 7},
    {" 12", 12}, {"+9", 9}, {"2147483647", 2147483647},
};

void CheckLogSwitches() {
    for (const SwitchCase& c : kSwitchCases) {
        const std::string value = c.value ? std::string("\"") + c.value + "\"" : "unset";
        InLogChild("DLSSNR_VERBOSE " + value, {{"DLSSNR_VERBOSE", c.value}}, [&] {
            Require(log_verbose() == c.on, "DLSSNR_VERBOSE " + value + ": log_verbose() is wrong");
            Require(!log_time_enabled(), "DLSSNR_VERBOSE " + value + ": log_time_enabled() is on");
        });
        InLogChild("DLSSNR_TIME " + value, {{"DLSSNR_TIME", c.value}}, [&] {
            Require(log_time_enabled() == c.on, "DLSSNR_TIME " + value + ": log_time_enabled() is wrong");
            Require(!log_verbose(), "DLSSNR_TIME " + value + ": log_verbose() is on");
        });
    }
    for (const IntervalCase& c : kIntervalCases) {
        const std::string label = std::string("DLSSNR_TIME_EVERY ") +
                                  (c.value ? std::string("\"") + c.value + "\"" : "unset");
        InLogChild(label, {{"DLSSNR_TIME_EVERY", c.value}}, [&] {
            const uint32_t got = log_time_interval();
            Require(got == c.interval, label + ": log_time_interval() is " + std::to_string(got) + ", not " +
                    std::to_string(c.interval));
        });
    }
}

std::string ReadFile(const std::string& path) {
    std::string text;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return text;
    char buffer[4096];
    for (size_t n; (n = std::fread(buffer, 1, sizeof buffer, f)) > 0;) text.append(buffer, n);
    std::fclose(f);
    return text;
}

bool Exists(const std::string& path) {
    struct stat st;
    return !lstat(path.c_str(), &st);
}

void WriteFile(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "wb");
    Require(f && std::fwrite(text.data(), 1, text.size(), f) == text.size() && !std::fclose(f),
            "cannot write " + path);
}

// Sends the child's standard error to a file.
void RedirectStderr(const std::string& path) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0 || dup2(fd, 2) != 2) _exit(2);
    close(fd);
}

// The descriptor through which this process holds path open, -1 if there is none, or -2 if
// /proc/self/fd cannot be read.
int DescriptorOf(const char* path) {
    DIR* fds = opendir("/proc/self/fd");
    if (!fds) return -2;
    int found = -1;
    for (dirent* entry; found < 0 && (entry = readdir(fds));) {
        char target[PATH_MAX];
        const std::string link = std::string("/proc/self/fd/") + entry->d_name;
        const ssize_t n = readlink(link.c_str(), target, sizeof target - 1);
        if (n > 0 && size_t(n) == std::strlen(path) && !std::memcmp(target, path, size_t(n)))
            found = std::atoi(entry->d_name);
    }
    closedir(fds);
    return found;
}

// After the log's destructor, which runs before this one: the child's log file is no longer open,
// and its standard descriptors still are. A failure ends the child with 3. Static objects with
// destructors are gone by then, so the path is a plain array.
char g_closedAtExit[PATH_MAX];
bool g_checkAtExit = false;

[[gnu::destructor(101)]] void CheckAfterLogClose() {
    if (!g_checkAtExit) return;
    for (int fd = 0; fd < 3; ++fd)
        if (fcntl(fd, F_GETFD) < 0) _exit(3);
    if (g_closedAtExit[0] && DescriptorOf(g_closedAtExit) != -1) _exit(3);
}

void CheckLogLines(const std::string& dir) {
    // Each variable that asks for the layer, appended to what the file held.
    const Env kEnables[] = {{"DLSSNR_ENABLE", "1"}, {"VKLayer_DLSS5", "1"}, {"VKLAYER_DLSS5", "1x"}};
    for (const Env& enable : kEnables) {
        const std::string label = std::string(enable.variable) + "=" + enable.value;
        const std::string log = dir + "/" + enable.variable + ".log";
        const std::string err = dir + "/" + enable.variable + ".err";
        WriteFile(log, "an earlier line\n");
        InLogChild(label, {enable, {"DLSSNR_LOG", log.c_str()}}, [&] {
            RedirectStderr(err);
            log_printf("[test] %s %u %.2f %p", "one", 2u, 3.0, nullptr);
            log_printf("%s", "");
            std::snprintf(g_closedAtExit, sizeof g_closedAtExit, "%s", log.c_str());
            g_checkAtExit = true;
        });
        const std::string want = "an earlier line\n[dlssnr-layer] [test] one 2 3.00 (nil)\n[dlssnr-layer] \n";
        Require(ReadFile(log) == want, label + ": the log holds \"" + ReadFile(log) + "\"");
        Require(ReadFile(err).empty(), label + ": the log wrote to stderr");
    }

    // The programs that the process executes do not inherit the file.
    const std::string execLog = dir + "/exec.log";
    InLogChild("close on exec", {{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", execLog.c_str()}}, [&] {
        log_printf("[test] %s", "exec");
        const int fd = DescriptorOf(execLog.c_str());
        Require(fd >= 0, "close on exec: the log's file is not open");
        Require(fcntl(fd, F_GETFD) & FD_CLOEXEC, "close on exec: the log's file stays open in executed programs");
    });

    // Not asked for: nothing is written, and DLSSNR_LOG is not created.
    const std::string offLog = dir + "/off.log";
    const std::vector<Env> kNotEnabled[] = {
        {{"DLSSNR_LOG", offLog.c_str()}},
        {{"DLSSNR_ENABLE", "0"}, {"VKLayer_DLSS5", ""}, {"VKLAYER_DLSS5", "yes"}, {"DLSSNR_LOG", offLog.c_str()}},
        {{"DLSSNR_ENABLE", " 1"}, {"DLSSNR_VERBOSE", "1"}, {"DLSSNR_TIME", "1"}, {"DLSSNR_LOG", offLog.c_str()}},
    };
    int row = 0;
    for (const std::vector<Env>& env : kNotEnabled) {
        const std::string label = "not enabled, row " + std::to_string(row++);
        const std::string err = dir + "/off.err";
        InLogChild(label, env, [&] {
            RedirectStderr(err);
            log_printf("[test] %s", "dropped");
            g_checkAtExit = true;
        });
        Require(!Exists(offLog), label + ": DLSSNR_LOG was created");
        Require(ReadFile(err).empty(), label + ": the log wrote to stderr");
    }

    // No usable DLSSNR_LOG: standard error.
    const std::string missing = dir + "/missing/layer.log";
    const char* const kStderrLogs[] = {nullptr, "", missing.c_str(), dir.c_str()};
    for (const char* path : kStderrLogs) {
        const std::string label =
            std::string("DLSSNR_LOG ") + (path ? std::string("\"") + path + "\"" : "unset");
        const std::string err = dir + "/stderr.err";
        InLogChild(label, {{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", path}}, [&] {
            RedirectStderr(err);
            log_printf("[test] to %s", "stderr");
            g_checkAtExit = true;
        });
        Require(ReadFile(err) == "[dlssnr-layer] [test] to stderr\n",
                label + ": stderr holds \"" + ReadFile(err) + "\"");
        Require(!Exists(dir + "/missing"), label + ": a directory was created");
    }

    // The text is cut to 2047 bytes, the prefix and the newline are not.
    const std::string longLog = dir + "/long.log";
    InLogChild("long lines", {{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", longLog.c_str()}}, [&] {
        for (size_t length : {2046, 2047, 2048, 5000}) log_printf("%s", std::string(length, 'x').c_str());
    });
    std::string want;
    for (size_t length : {2046, 2047, 2047, 2047}) want += "[dlssnr-layer] " + std::string(length, 'x') + "\n";
    Require(ReadFile(longLog) == want, "long lines are not cut to 2047 bytes");

    // Lines from several threads at once stay whole, and each thread's stay in order.
    constexpr uint32_t kThreads = 8, kLines = 500;
    const std::string threadLog = dir + "/threads.log";
    InLogChild("threads", {{"DLSSNR_ENABLE", "1"}, {"DLSSNR_LOG", threadLog.c_str()}}, [&] {
        std::vector<std::thread> threads;
        for (uint32_t t = 0; t < kThreads; ++t)
            threads.emplace_back([t] {
                const std::string tail(200, char('a' + t));
                for (uint32_t i = 0; i < kLines; ++i) log_printf("thread %u line %u %s", t, i, tail.c_str());
            });
        for (std::thread& thread : threads) thread.join();
    });
    const std::string lines = ReadFile(threadLog);
    uint32_t next[kThreads] = {};
    size_t start = 0, count = 0;
    for (size_t end; (end = lines.find('\n', start)) != std::string::npos; start = end + 1, ++count) {
        unsigned t = 0, i = 0;
        char tail[256] = {};
        const std::string line = lines.substr(start, end - start);
        Require(std::sscanf(line.c_str(), "[dlssnr-layer] thread %u line %u %255s", &t, &i, tail) == 3 &&
                t < kThreads, "a thread's line is broken: " + line);
        Require(i == next[t]++ && std::string(tail) == std::string(200, char('a' + t)),
                "a thread's line is out of order or broken: " + line);
    }
    Require(start == lines.size() && count == kThreads * kLines,
            "the threads' log has " + std::to_string(count) + " lines");
}

// Whether a child exits with 0 within five seconds. A child that does not is killed.
bool ExitsInTime(pid_t child) {
    for (uint32_t ms = 0; ms < 5000; ++ms) {
        int status = 0;
        const pid_t done = waitpid(child, &status, WNOHANG);
        if (done == child) return WIFEXITED(status) && !WEXITSTATUS(status);
        Require(done == 0 || errno == EINTR, "waitpid failed");
        usleep(1000);
    }
    kill(child, SIGKILL);
    while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    return false;
}

// A child that fork() makes while a thread writes a line has the log's lock held by a thread it does
// not have. Its exit() runs the log's destructor, which must not wait for that lock. The writer logs
// to standard error, sent to /dev/null until the check ends, and allocates nothing. The children
// therefore hold no descriptor that valgrind's --track-fds reports and no memory that LeakSanitizer
// reports.
void CheckLogForkExit() {
    constexpr uint32_t kForks = 50;
    InLogChild("fork while a thread logs", {{"DLSSNR_ENABLE", "1"}}, [&] {
        const int err = dup(2);
        if (err < 0) _exit(2);
        RedirectStderr("/dev/null");
        std::atomic<bool> stop{false};
        std::thread writer([&stop] {
            while (!stop.load(std::memory_order_relaxed)) log_printf("[test] %0200u", 0u);
        });
        uint32_t exited = 0;
        bool forked = true;
        while (exited < kForks) {
            const pid_t child = fork();
            if (!child) {
                close(err);
                std::exit(0);
            }
            forked = child > 0;
            if (!forked || !ExitsInTime(child)) break;
            ++exited;
        }
        stop.store(true, std::memory_order_relaxed);
        writer.join();
        if (dup2(err, 2) != 2) _exit(2);
        close(err);
        Require(forked, "fork failed");
        Require(exited == kForks, "a child forked while a thread logged did not exit, after " +
                                      std::to_string(exited) + " that did");
    });
}

// log_now_ms reads the clock that std::chrono::steady_clock reads, in milliseconds.
void CheckClock() {
    const auto ms = [] {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::milli>(now).count();
    };
    for (int i = 0; i < 1000; ++i) {
        const double before = ms();
        const double now = log_now_ms();
        const double after = ms();
        Require(before <= now && now <= after, "log_now_ms() is not between two steady_clock readings");
    }
}

void CheckLog() {
    const char* tmp = std::getenv("TMPDIR");
    std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/dlsslop-amd-log-XXXXXX";
    Require(mkdtemp(pattern.data()), "mkdtemp failed");
    char dir[PATH_MAX];
    Require(realpath(pattern.c_str(), dir), "realpath failed");
    CheckLogSwitches();
    CheckLogLines(dir);
    CheckLogForkExit();
    CheckClock();
    DIR* entries = opendir(dir);
    Require(entries, "cannot list " + std::string(dir));
    for (dirent* entry; (entry = readdir(entries));)
        if (std::strcmp(entry->d_name, ".") && std::strcmp(entry->d_name, ".."))
            Require(!unlink((std::string(dir) + "/" + entry->d_name).c_str()), "unlink failed");
    closedir(entries);
    Require(!rmdir(dir), "rmdir failed");
}

}  // namespace

int main() {
    // The log and the overrides first: the log reads its variables and Read its overrides in this
    // process at their first call, and the children forked below must each make that first call
    // themselves.
    for (const char* variable : kLogVariables) Require(!unsetenv(variable), "unsetenv failed");
    CheckLog();
    for (const char* variable : {"DLSSNR_GHOST_SLACK", "DLSSNR_RATIO_SMOOTH", "DLSSNR_COLOUR_TRUST",
                                 "DLSSNR_MOTION_SMOOTH", "DLSSNR_EDIT_BLUR"})
        Require(!unsetenv(variable), "unsetenv failed");
    CheckOverrides();
    CheckRead();
    CheckModelExtent();
    CheckFormats();
    CheckScalerNames();
    CheckKeyNames();
    std::printf("layer-units-test: the log, settings, model extents, formats, scaler and key names hold\n");
    return 0;
}
