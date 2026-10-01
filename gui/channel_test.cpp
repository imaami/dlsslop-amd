// SPDX-License-Identifier: MIT
#include "channel.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
// The fixture's files, removed however the test ends.
struct Fixture {
    char directory[33] = "/tmp/dlsslop-amd-gui-test-XXXXXX";
    std::string path, link;
    int fd = -1;
    ShmHeader* header = nullptr;
    ~Fixture()
    {
        if (header) munmap(header, kHeaderBytes);
        if (fd >= 0) close(fd);
        if (path.empty()) return;
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
    const std::string& path = fixture.path;
    const std::string& link = fixture.link;
    int& fd = fixture.fd;
    ShmHeader*& header = fixture.header;
    rejects(dlsslop_gui::Channel::open(path, true));
    require(access(path.c_str(), F_OK) != 0, "controller created a channel");
    fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    require(fd >= 0, "create fixture");
    require(!ftruncate(fd, static_cast<off_t>(ShmTotalBytes())), "size fixture");
    void* memory = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    require(memory != MAP_FAILED, "map fixture");
    header = static_cast<ShmHeader*>(memory);
    rejects(dlsslop_gui::Channel::open(path, true));
    ShmInitNativeDefaults(header, false);
    // Fixed settings admit only their reset default, only the native tuning
    // values are marked as such, and each label list names its range once.
    for (const auto& s : dlsslop_control::kSettings) {
        const std::string name = s.name;
        const double initial = dlsslop_control::value(s, (header->*s.field).load());
        require(dlsslop_control::inRange(s, initial), "a default is outside its range");
        require(dlsslop_control::fixed(s) == (name == "preset"), "wrong fixed settings");
        require(!dlsslop_control::fixed(s) || (s.minimum == initial && s.maximum == initial),
                "a fixed setting admits another value");
        require(dlsslop_control::fixed(s) == (s.minimum == s.maximum), "a one-value range is not fixed");
        require(s.tuning == (name == "intensity" || name == "local-tone" || name == "local-structure" ||
                             name == "sharpness"), "wrong native tuning settings");
        if (!s.choices) continue;
        const auto labels = 1 + std::count(s.choices, s.choices + std::strlen(s.choices), '|');
        require(!s.isFloat && labels == (s.maximum - s.minimum) / s.step + 1, "choice labels do not match the range");
    }
    std::size_t intensity = 0, color = 0, preset = 0, colorPreserve = 0;
    for (std::size_t i = 0; i < std::size(dlsslop_control::kSettings); ++i) {
        const std::string name = dlsslop_control::kSettings[i].name;
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
        for (std::size_t i = 0; i < std::size(dlsslop_control::kSettings); ++i) {
            const auto& s = dlsslop_control::kSettings[i];
            const double below = std::nextafter(static_cast<double>(static_cast<float>(s.minimum)), -HUGE_VAL);
            const double above = std::nextafter(static_cast<double>(static_cast<float>(s.maximum)), HUGE_VAL);
            require(!dlsslop_control::inRange(s, below) && !dlsslop_control::inRange(s, above) &&
                    !dlsslop_control::inRange(s, std::nan("")), "range admits an outside value");
            for (const double bound : {s.minimum, s.maximum}) {
                accepted(channel.write({{i, bound}}));
                require(dlsslop_control::inRange(s, dlsslop_control::value(s, (header->*s.field).load())),
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
    std::puts("GUI channel tests passed: precision, bounds, batching, generations, isolation, reset, actions, protocol and symlink checks");
    return 0;
}
