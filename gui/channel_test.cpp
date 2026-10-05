// SPDX-License-Identifier: MIT
#include "channel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {
// The fixture's files, removed however the test ends.
struct Fixture {
    char directory[33] = "/tmp/dlsslop-amd-gui-test-XXXXXX";
    std::string path, link, zeros;
    shm_channel channel{nullptr, 0, -1, 0};
    ~Fixture()
    {
        shm_channel_fini(&channel);
        if (path.empty()) return;
        unlink(zeros.c_str());
        unlink(link.c_str());
        unlink(path.c_str());
        rmdir(directory);
    }
} fixture;

[[noreturn]] void failed(const char* message)
{
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}
void require(bool condition, const char* message)
{
    if (!condition) failed(message);
}
template <class T> T accepted(dlsslop::Result<T> result)
{
    if (!result) failed(result.error().what.c_str());
    return *std::move(result);
}
template <class T> void rejects(const dlsslop::Result<T>& result) { require(!result, "invalid operation was accepted"); }
} // namespace

int main()
{
    if (!mkdtemp(fixture.directory)) return 1;
    fixture.path = std::string(fixture.directory) + "/channel";
    fixture.link = std::string(fixture.directory) + "/link";
    fixture.zeros = std::string(fixture.directory) + "/zeros";
    const std::string& path = fixture.path;
    const std::string& link = fixture.link;
    // A controller creates no channel, and initializes no file of a channel's size that holds none.
    rejects(dlsslop_gui::Channel::open(path, false));
    rejects(dlsslop_gui::Channel::open(path, true));
    require(access(path.c_str(), F_OK) != 0, "controller created a channel");
    {
        const int fd = open(fixture.zeros.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
        require(fd >= 0 && !ftruncate(fd, static_cast<off_t>(ShmTotalBytes())), "create zeros");
        rejects(dlsslop_gui::Channel::open(fixture.zeros, true));
        char head[4096];
        require(pread(fd, head, sizeof head, 0) == static_cast<ssize_t>(sizeof head) &&
                    std::all_of(head, head + sizeof head, [](char c) { return !c; }) && !close(fd),
                "a refused file was initialized");
    }
    // The channel as dlsslopd or the layer creates it, with the defaults.
    struct error e;
    if (shm_channel_open(&fixture.channel, path.c_str(), path.size(), kHeaderBytes,
                         SHM_CHANNEL_CREATE | SHM_CHANNEL_WRITE, &e))
        failed(e.what);
    ShmHeader* const header = std::start_lifetime_as<ShmHeader>(fixture.channel.h);
    std::array<std::uint32_t, CONTROL_SETTING_COUNT> defaults;
    control_settings_defaults(defaults.data(), false);
    for (std::size_t i = 0; i < CONTROL_SETTING_COUNT; ++i)
        require(control_setting_load(header, &CONTROL_SETTINGS[i]) == defaults[i], "a created channel lacks a default");
    // Fixed settings admit only their reset default, only the native tuning
    // values are marked as such, and each label list names its range once.
    for (const auto& s : CONTROL_SETTINGS) {
        const std::string name = s.name;
        const double initial = control_setting_value(&s, control_setting_load(header, &s));
        require(control_setting_in_range(&s, initial), "a default is outside its range");
        require(control_setting_fixed(&s) == (name == "preset"), "wrong fixed settings");
        require(!control_setting_fixed(&s) || (s.minimum == initial && s.maximum == initial),
                "a fixed setting admits another value");
        require(control_setting_fixed(&s) == (s.minimum == s.maximum), "a one-value range is not fixed");
        require(s.tuning == (name == "intensity" || name == "local-tone" || name == "local-structure" ||
                             name == "sharpness"), "wrong native tuning settings");
        if (!s.choices) continue;
        const auto labels = 1 + std::count(s.choices, s.choices + std::strlen(s.choices), '|');
        require(!s.is_float && labels == (s.maximum - s.minimum) / s.step + 1, "choice labels do not match the range");
    }
    std::size_t intensity = 0, color = 0, preset = 0, colorPreserve = 0;
    for (std::size_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
        const std::string name = CONTROL_SETTINGS[i].name;
        if (name == "intensity") intensity = i;
        if (name == "color") color = i;
        if (name == "preset") preset = i;
        if (name == "color-preserve") colorPreserve = i;
    }
    const uint32_t control = header->controlSeq.load(), tuning = header->tuningSeq.load();
    header->holdFrame.store(1);
    header->quit.store(1);
    {
        auto channel = accepted(dlsslop_gui::Channel::open(path, true));
        accepted(channel.write({{intensity, 0.312345}}));
        require(std::fabs(BitsToFloat(header->intensityBits.load()) - 0.312345) < 1e-7, "fractional precision");
        require(header->controlSeq.load() == control + 1 && header->tuningSeq.load() == tuning + 1, "tuning generations");
        require(header->holdFrame.load() == 1 && header->quit.load() == 1, "unrelated settings changed");
        accepted(channel.write({{color, 0.25}}));
        require(header->tuningSeq.load() == tuning + 1, "composition bumped tuning");
        accepted(channel.write({{colorPreserve, 0.5}}));
        require(header->tuningSeq.load() == tuning + 1, "color preservation bumped tuning");
        const auto seq = header->controlSeq.load();
        rejects(channel.write({{intensity, 0.5}, {preset, 1}}));
        require(header->controlSeq.load() == seq && BitsToFloat(header->intensityBits.load()) < 0.32f,
                "invalid batch partially applied");
        rejects(channel.write({{intensity, std::nan("")}}));
        rejects(channel.write({{intensity, HUGE_VAL}}));
        accepted(channel.write({{intensity, -0.0}}));
        require(header->intensityBits.load() == 0, "negative zero stored");
        // Every advertised bound is accepted and reads back in range, although
        // binary32 stores some float minimums below themselves.
        for (std::size_t i = 0; i < CONTROL_SETTING_COUNT; ++i) {
            const auto& s = CONTROL_SETTINGS[i];
            const double below = std::nextafter(static_cast<double>(static_cast<float>(s.minimum)), -HUGE_VAL);
            const double above = std::nextafter(static_cast<double>(static_cast<float>(s.maximum)), HUGE_VAL);
            require(!control_setting_in_range(&s, below) && !control_setting_in_range(&s, above) &&
                    !control_setting_in_range(&s, std::nan("")), "range admits an outside value");
            for (const double bound : {s.minimum, s.maximum}) {
                accepted(channel.write({{i, bound}}));
                require(control_setting_in_range(&s, control_setting_value(&s, control_setting_load(header, &s))),
                        "a written bound reads back out of range");
            }
        }
        accepted(channel.capture(3));
        require(header->captureRequest.load() == 3, "capture not published");
        rejects(channel.capture(65));
        channel.reset();
        require(header->quit.load() == 1, "reset changed stop request");
        require(BitsToFloat(header->intensityBits.load()) == 1, "reset missed intensity");
        require(header->compositionBypass.load() == 0, "a never-started channel reset bypass on");
        // A running worker that publishes no neural raster returns a final
        // image, so its reset default presents it; a neural raster composes.
        header->heartbeat.store(1);
        channel.reset();
        require(header->compositionBypass.load() == 1, "worker-mode reset left bypass off");
        header->nativeModelMaxWidth.store(1280);
        header->nativeModelMaxHeight.store(720);
        channel.reset();
        require(header->compositionBypass.load() == 0, "neural-raster reset left bypass on");
        channel.stop(false);
        require(!header->quit.load(), "resume failed");
        channel.stop(true);
    }
    require(header->quit.load() == 1, "destructor changed stop state");
    require(!symlink(path.c_str(), link.c_str()), "symlink fixture");
    rejects(dlsslop_gui::Channel::open(link, true));
    header->version.store(kShmVersion + 1);
    rejects(dlsslop_gui::Channel::open(path, true));
    require(header->version.load() == kShmVersion + 1, "mismatched channel reinitialized");
    std::puts("GUI channel tests passed: no creation, precision, bounds, batching, generations, isolation, reset, "
              "actions, protocol, foreign file and symlink checks");
    return 0;
}
