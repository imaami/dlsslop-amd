#include "capture.h"
#include "../upstream-layer/common/shm_protocol.h"
#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "cannot read expected capture file");
    return {std::istreambuf_iterator<char>(file), {}};
}

std::string field(const std::string& manifest, const std::string& key) {
    const std::string needle = key + " ";
    const auto found = manifest.find(needle);
    require(found != std::string::npos && (!found || manifest[found - 1] == '\n'),
            "missing manifest field");
    const auto begin = found + needle.size();
    return manifest.substr(begin, manifest.find('\n', begin) - begin);
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        char pattern[] = "/tmp/dlsslop-amd-capture-test-XXXXXX";
        const char* name = mkdtemp(pattern);
        require(name, "mkdtemp failed");
        path = name;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

std::set<std::filesystem::path> directories(const std::filesystem::path& root) {
    std::set<std::filesystem::path> result;
    for (const auto& item : std::filesystem::directory_iterator(root))
        if (item.is_directory()) result.insert(item.path());
    return result;
}

void check_channel_paths() {
    const char* inherited = std::getenv("DLSSNR_UID");
    const bool hadOverride = inherited != nullptr;
    const std::string originalOverride = inherited ? inherited : "";
    const std::string uid = std::to_string(static_cast<unsigned>(getuid()));
    const std::string nativePath = "/tmp/dlsslop-amd-" + uid + "/shm.bin";

    require(unsetenv("DLSSNR_UID") == 0, "unsetenv failed");
    require(ShmRuntimeDir() == "/tmp/dlssnr-" + uid, "shared runtime default changed");
    require(ShmDefaultPath() == "/tmp/dlssnr-" + uid + "/shm.bin",
            "shared channel default changed");
    require(ShmNativeDefaultPath() == nativePath, "native channel default changed");
    require(setenv("DLSSNR_UID", "12345", 1) == 0, "setenv failed");
    require(ShmRuntimeDir() == "/tmp/dlssnr-12345", "shared runtime ignored DLSSNR_UID");
    require(ShmDefaultPath() == "/tmp/dlssnr-12345/shm.bin", "shared channel ignored DLSSNR_UID");
    require(ShmNativeDefaultPath() == nativePath, "DLSSNR_UID changed the native channel");
    require(setenv("DLSSNR_UID", "", 1) == 0, "setenv failed");
    require(ShmRuntimeDir() == "/tmp/dlssnr-" + uid, "empty DLSSNR_UID changed the default");
    require((hadOverride ? setenv("DLSSNR_UID", originalOverride.c_str(), 1)
                         : unsetenv("DLSSNR_UID")) == 0, "restoring DLSSNR_UID failed");
}

} // namespace

int main() {
    try {
        check_channel_paths();
        TemporaryDirectory temporary;
        require(unsetenv("XDG_STATE_HOME") == 0, "unsetenv failed");
        require(setenv("HOME", temporary.path.c_str(), 1) == 0, "setenv failed");
        require(dlssnr::CaptureWriter::Directory() == temporary.path / ".local/state/dlssnr/captures",
                "wrong home capture directory");
        require(unsetenv("HOME") == 0, "unsetenv failed");
        require(dlssnr::CaptureWriter::Directory() == "/tmp/dlssnr-captures",
                "wrong fallback capture directory");
        require(setenv("XDG_STATE_HOME", temporary.path.c_str(), 1) == 0, "setenv failed");
        const std::filesystem::path root = dlssnr::CaptureWriter::Directory();
        require(root == temporary.path / "dlssnr/captures", "wrong state capture directory");
        std::filesystem::create_directories(root);
        // Legacy unbatched files and unrelated user content must survive a new capture.
        std::ofstream(root / "before_00.png") << "old capture";
        std::ofstream(root / "notes.txt") << "keep";

        dlssnr::CaptureWriter writer;
        dlssnr::CaptureMetadata metadata;
        metadata.frameControlSeq = 123;
        metadata.inferenceSeq = 55;
        metadata.hold = 1;
        metadata.passes = 1;
        metadata.debugView = 2;
        metadata.color = 0.25f;
        metadata.detail = 0.5f;
        const unsigned char bgra[] = {17, 32, 64, 255};
        writer.Begin(2, 123);
        writer.WriteFrame(bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, metadata);
        require(!std::filesystem::exists(root / "manifest.txt"), "published an incomplete batch");
        require(writer.Remaining() == 1, "wrong remaining frame count");
        writer.WriteFrame(bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, metadata);
        require(!writer.Active(), "completed batch remains active");
        const auto first = read(root / "manifest.txt");
        require(field(first, "capture_metadata_version") == "2", "wrong metadata version");
        require(field(first, "capture_control_seq") == "123", "wrong capture token");
        require(field(first, "frame_1_before_hash") == "c2de31fd48ac5f39", "wrong input hash");
        require(field(first, "frame_1_inference_seq") == "55", "wrong inference provenance");
        require(field(first, "debug_view") == "2", "wrong renderer view");
        require(field(first, "color") == "0.25", "wrong renderer color");
        const auto batch = root / field(first, "batch_dir");
        require(read(batch / "manifest.txt") == first, "batch and published manifests differ");
        int width = 0, height = 0, channels = 0;
        unsigned char* png = stbi_load((batch / "before_00.png").c_str(), &width, &height, &channels, 4);
        require(png, "cannot decode captured PNG");
        const bool correct = width == 1 && height == 1 && png[0] == 64 && png[1] == 32 &&
                             png[2] == 17 && png[3] == 255;
        stbi_image_free(png);
        require(correct, "BGRA capture channels were not converted to RGBA PNG");
        require(read(root / "before_00.png") == "old capture", "overwrote an existing capture");
        require(read(root / "notes.txt") == "keep", "removed unrelated content");

        writer.Begin(1, 124);
        require(read(root / "manifest.txt") == first, "changed completion before new batch finished");
        writer.WriteFrame(bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, metadata);
        const auto second = read(root / "manifest.txt");
        require(field(second, "capture_control_seq") == "124", "new completion not published");
        require(field(second, "batch_dir") != field(first, "batch_dir"), "reused batch directory");
        require(read(batch / "manifest.txt") == first, "changed prior batch metadata");

        writer.Begin(1, 125);
        const unsigned short fp16[] = {0x3800, 0x3800, 0x3800, 0x3c00};
        writer.WriteFrame(fp16, fp16, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, metadata);
        const auto third = read(root / "manifest.txt");
        require(field(third, "encoding") == "raw", "FP16 capture was not raw");
        require(field(third, "bytes_per_pixel") == "8", "FP16 capture byte count incorrect");
        // Eight bytes take the hash's word step; the BGRA pixel above takes its byte tail.
        require(field(third, "before_hash") == "736955abadf41fdf", "wrong FP16 input hash");
        const auto raw = read(root / field(third, "batch_dir") / "before_00.raw");
        require(raw == std::string(reinterpret_cast<const char*>(fp16), sizeof fp16), "FP16 bytes changed");

        // A failed image write must leave the last completed batch visible.
        const auto priorDirectories = directories(root);
        writer.Begin(1, 126);
        for (const auto& directory : directories(root))
            if (!priorDirectories.count(directory)) std::filesystem::remove(directory);
        writer.WriteFrame(bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, metadata);
        require(!writer.Active(), "failed batch still active");
        require(read(root / "manifest.txt") == third, "published failed batch as complete");
        std::printf("PASS: runtime and capture paths, capture publication, provenance, preserved batches, BGRA PNG, FP16 raw, write failure\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "capture-writer-test: %s\n", error.what());
        return 1;
    }
}
