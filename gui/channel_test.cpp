// SPDX-License-Identifier: MIT
#include "channel.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

static void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
static void rejects(const std::function<void()>& operation)
{
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid operation was accepted");
}
int main()
{
    char directory[] = "/tmp/dlsslop-amd-gui-test-XXXXXX";
    if (!mkdtemp(directory)) return 1;
    const std::string path = std::string(directory) + "/channel";
    const std::string link = std::string(directory) + "/link";
    int fd = -1;
    ShmHeader* header = nullptr;
    int result = 0;
    try {
        rejects([&] { dlsslop_gui::Channel channel(path, true); });
        require(access(path.c_str(), F_OK) != 0, "controller created a channel");
        fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        require(fd >= 0, "create fixture");
        require(!ftruncate(fd, static_cast<off_t>(ShmTotalBytes())), "size fixture");
        void* memory = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        require(memory != MAP_FAILED, "map fixture");
        header = static_cast<ShmHeader*>(memory);
        rejects([&] { dlsslop_gui::Channel channel(path, true); });
        ShmInitNativeDefaults(header);
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
            dlsslop_gui::Channel channel(path, true);
            channel.write({{intensity, 0.312345}});
            require(std::fabs(BitsToFloat(header->intensityBits.load()) - 0.312345) < 1e-7, "fractional precision");
            require(header->controlSeq.load() == control + 1 && header->tuningSeq.load() == tuning + 1, "tuning generations");
            require(header->holdFrame.load() == 1 && header->quit.load() == 1, "unrelated settings changed");
            channel.write({{color, 0.25}});
            require(header->tuningSeq.load() == tuning + 1, "composition bumped tuning");
            channel.write({{colorPreserve, 0.5}});
            require(header->tuningSeq.load() == tuning + 1, "color preservation bumped tuning");
            const auto seq = header->controlSeq.load();
            rejects([&] { channel.write({{intensity, 0.5}, {preset, 1}}); });
            require(header->controlSeq.load() == seq && BitsToFloat(header->intensityBits.load()) < 0.32f,
                    "invalid batch partially applied");
            rejects([&] { channel.write({{intensity, std::nan("")}}); });
            rejects([&] { channel.write({{intensity, HUGE_VAL}}); });
            channel.write({{intensity, -0.0}});
            require(header->intensityBits.load() == 0, "negative zero stored");
            // Every advertised bound is accepted and reads back in range, although
            // binary32 stores some float minimums below themselves.
            for (std::size_t i = 0; i < std::size(dlsslop_control::kSettings); ++i) {
                const auto& s = dlsslop_control::kSettings[i];
                const double below = std::nextafter(static_cast<double>(static_cast<float>(s.minimum)), -HUGE_VAL);
                const double above = std::nextafter(static_cast<double>(static_cast<float>(s.maximum)), HUGE_VAL);
                require(!dlsslop_control::inRange(s, below) && !dlsslop_control::inRange(s, above) &&
                        !dlsslop_control::inRange(s, std::nan("")), "range admits an outside value");
                if (dlsslop_control::fixed(s)) continue;
                for (const double bound : {s.minimum, s.maximum}) {
                    channel.write({{i, bound}});
                    require(dlsslop_control::inRange(s, dlsslop_gui::Channel::value(s, (header->*s.field).load())),
                            "a written bound reads back out of range");
                }
            }
            channel.capture(3);
            require(header->captureRequest.load() == 3, "capture not published");
            rejects([&] { channel.capture(65); });
            channel.reset();
            require(header->quit.load() == 1, "reset changed stop request");
            require(BitsToFloat(header->intensityBits.load()) == 1, "reset missed intensity");
            channel.stop(false);
            require(!header->quit.load(), "resume failed");
            channel.stop(true);
        }
        require(header->quit.load() == 1, "destructor changed stop state");
        require(!symlink(path.c_str(), link.c_str()), "symlink fixture");
        rejects([&] { dlsslop_gui::Channel channel(link, true); });
        header->version.store(kShmVersion + 1);
        rejects([&] { dlsslop_gui::Channel channel(path, true); });
        require(header->version.load() == kShmVersion + 1, "mismatched channel reinitialized");
        std::puts("GUI channel tests passed: precision, bounds, batching, generations, isolation, reset, actions, protocol and symlink checks");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    if (header) munmap(header, kHeaderBytes);
    if (fd >= 0) close(fd);
    unlink(link.c_str()); unlink(path.c_str()); rmdir(directory);
    return result;
}
