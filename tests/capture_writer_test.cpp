#include "capture.h"
#include "shm_protocol.h"
#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <climits>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "capture-writer-test: %s\n", message);
    std::exit(1);
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
    std::error_code error;
    for (std::filesystem::directory_iterator item(root, error), end; !error && item != end; item.increment(error))
        if (item->is_directory(error)) result.insert(item->path());
    require(!error, "cannot list the capture directory");
    return result;
}

// ShmRuntimeDir(), which has only the C form.
std::string runtime_dir() {
    char dir[64];
    const int length = ShmRuntimeDir(dir, sizeof(dir));
    require(length >= 0 && size_t(length) < sizeof(dir), "the shared runtime directory did not fit");
    return dir;
}

// A C form writes what fits and returns the length of the whole path, as snprintf does.
void check_truncation(int length, const char (&buffer)[8], const std::string& path, const char* message) {
    require(length == int(path.size()) && path.substr(0, sizeof(buffer) - 1) == buffer, message);
}

void check_channel_paths() {
    const char* inherited = std::getenv("DLSSNR_UID");
    const bool hadOverride = inherited != nullptr;
    const std::string originalOverride = inherited ? inherited : "";
    const char* inheritedChannel = std::getenv("DLSSNR_SHM");
    const bool hadChannel = inheritedChannel != nullptr;
    const std::string originalChannel = inheritedChannel ? inheritedChannel : "";
    const std::string uid = std::to_string(static_cast<unsigned>(getuid()));
    const std::string nativePath = "/tmp/dlsslop-amd-" + uid + "/shm.bin";
    char small[8];

    require(unsetenv("DLSSNR_UID") == 0, "unsetenv failed");
    require(runtime_dir() == "/tmp/dlssnr-" + uid, "shared runtime default changed");
    require(ShmDefaultPath() == "/tmp/dlssnr-" + uid + "/shm.bin",
            "shared channel default changed");
    require(ShmNativeDefaultPath() == nativePath, "native channel default changed");
    require(setenv("DLSSNR_UID", "12345", 1) == 0, "setenv failed");
    require(runtime_dir() == "/tmp/dlssnr-12345", "shared runtime ignored DLSSNR_UID");
    require(ShmDefaultPath() == "/tmp/dlssnr-12345/shm.bin", "shared channel ignored DLSSNR_UID");
    check_truncation(ShmDefaultPath(small, sizeof(small)), small, "/tmp/dlssnr-12345/shm.bin",
                     "the shared channel's C form did not truncate as snprintf does");
    require(ShmNativeDefaultPath() == nativePath, "DLSSNR_UID changed the native channel");
    require(setenv("DLSSNR_UID", "", 1) == 0, "setenv failed");
    require(runtime_dir() == "/tmp/dlssnr-" + uid, "empty DLSSNR_UID changed the default");
    require((hadOverride ? setenv("DLSSNR_UID", originalOverride.c_str(), 1)
                         : unsetenv("DLSSNR_UID")) == 0, "restoring DLSSNR_UID failed");

    // The native channel: nonempty DLSSNR_SHM, at any length, otherwise the native default.
    const std::string longPath = "/tmp/" + std::string(5000, 'x') + "/shm.bin";
    require(setenv("DLSSNR_SHM", longPath.c_str(), 1) == 0, "setenv failed");
    require(ShmNativeChannelPath() == longPath, "the native channel ignored a long DLSSNR_SHM");
    check_truncation(ShmNativeChannelPath(small, sizeof(small)), small, longPath,
                     "the native channel's C form did not truncate as snprintf does");
    require(ShmTransportPath(longPath) == longPath + ".sock", "the transport path changed");
    check_truncation(ShmTransportPath(small, sizeof(small), longPath.c_str()), small, longPath + ".sock",
                     "the transport path's C form did not truncate as snprintf does");
    require(setenv("DLSSNR_SHM", "", 1) == 0, "setenv failed");
    require(ShmNativeChannelPath() == nativePath, "an empty DLSSNR_SHM changed the native channel");
    require((hadChannel ? setenv("DLSSNR_SHM", originalChannel.c_str(), 1)
                        : unsetenv("DLSSNR_SHM")) == 0, "restoring DLSSNR_SHM failed");
}

// A text field's sequence number is bumped after each write, and a load takes the string: one cut
// to the field's size, and an empty one for a null string.
void check_text_fields() {
    static ShmHeader header;
    ShmStoreString(&header.layerReasonSeq, header.layerReason, kReasonBytes, "first");
    require(header.layerReasonSeq.load() == 1 &&
                ShmLoadString(header.layerReasonSeq, header.layerReason, kReasonBytes) == "first",
            "a text field was not published");
    const std::string tooLong(kReasonBytes, 'x');
    ShmStoreString(&header.layerReasonSeq, header.layerReason, kReasonBytes, tooLong.c_str());
    require(header.layerReasonSeq.load() == 2 &&
                ShmLoadString(header.layerReasonSeq, header.layerReason, kReasonBytes) ==
                    tooLong.substr(0, kReasonBytes - 1),
            "a text field did not cut a string to its size");
    ShmStoreString(&header.layerReasonSeq, header.layerReason, kReasonBytes, nullptr);
    require(header.layerReasonSeq.load() == 3 &&
                ShmLoadString(header.layerReasonSeq, header.layerReason, kReasonBytes).empty() &&
                !header.layerReason[0],
            "a null string did not empty a text field");
}

// capture_writer_directory() as a string. The C form writes what fits and returns the length of
// the whole path, as snprintf does.
std::string capture_directory() {
    char dir[CAPTURE_WRITER_DIR_SIZE];
    const int length = capture_writer_directory(dir, sizeof dir);
    require(length >= 0 && size_t(length) < sizeof dir && size_t(length) == std::strlen(dir),
            "the capture directory did not fit");
    char small[8];
    check_truncation(capture_writer_directory(small, sizeof small), small, dir,
                     "the capture directory's C form did not truncate as snprintf does");
    require(capture_writer_directory(nullptr, 0) == length, "the capture directory's length changed");
    return dir;
}

// What the published manifest of the first batch below holds, with its batch directory.
std::string first_manifest(const std::string& batch) {
    std::string frame0, frame1;
    const char* const kFrame =
        "frame_control_seq 123\ntuning_seq 0\ninference_seq 55\npasses 1\ndebug_view 2\napply_model 0\n"
        "bypass 0\nhold 1\ncompare 0\ntransfer 0\nmodel_width 0\nmodel_height 0\nhdr_proxy 0\nlinear_hdr 0\n"
        "hdr_transfer 0\ndetail 0.5\ncolor 0.25\ndebug_scale 1\nbefore_hash c2de31fd48ac5f39\n";
    std::string frames = kFrame;
    for (const char* prefix : {"frame_0_", "frame_1_"})
        for (const char* line = kFrame; *line;) {
            const char* end = std::strchr(line, '\n') + 1;
            frames += prefix + std::string(line, end);
            line = end;
        }
    return "capture_metadata_version 2\ncapture_control_seq 123\nbatch_dir " + batch +
           "\nframes 2\nwidth 1\nheight 1\nvk_format 44\nencoding png\nbytes_per_pixel 4\nrow_pitch 4\n" + frames +
           "\nbefore_NN is the frame as the game presented it; after_NN is the same frame with\n"
           "the model's edit composed onto it. Same frame, same run, one variable.\n";
}

// Whether the log holds the line, after the log's prefix.
bool logged(const std::string& log, const std::string& line) {
    return ("\n" + log).find("\n[dlssnr-layer] " + line + "\n") != std::string::npos;
}

} // namespace

int main() {
    check_channel_paths();
    check_text_fields();
    // Static, so a failed check that exits removes it too.
    static const TemporaryDirectory temporary;
    // The log reads its variables at its first line, which comes below.
    const std::string logPath = (temporary.path / "layer.log").string();
    require(setenv("DLSSNR_ENABLE", "1", 1) == 0 && setenv("DLSSNR_LOG", logPath.c_str(), 1) == 0, "setenv failed");
    require(unsetenv("XDG_STATE_HOME") == 0, "unsetenv failed");
    require(setenv("HOME", temporary.path.c_str(), 1) == 0, "setenv failed");
    require(capture_directory() == temporary.path / ".local/state/dlssnr/captures",
            "wrong home capture directory");
    require(unsetenv("HOME") == 0, "unsetenv failed");
    require(capture_directory() == "/tmp/dlssnr-captures", "wrong fallback capture directory");
    require(setenv("XDG_STATE_HOME", temporary.path.c_str(), 1) == 0, "setenv failed");
    const std::filesystem::path root = capture_directory();
    require(root == temporary.path / "dlssnr/captures", "wrong state capture directory");
    std::error_code error;
    std::filesystem::create_directories(root, error);
    require(!error, "cannot create the capture directory");
    // Legacy unbatched files and unrelated user content must survive a new capture.
    std::ofstream(root / "before_00.png") << "old capture";
    std::ofstream(root / "notes.txt") << "keep";

    // Static: the writer holds its metadata and directory in place.
    static struct capture_writer writer = {};
    require(!capture_writer_active(&writer), "a zeroed writer is active");
    struct capture_metadata metadata = capture_metadata();
    require(metadata.debug_scale == 1.0f, "the metadata's debug scale does not start at 1");
    metadata.frame_control_seq = 123;
    metadata.inference_seq = 55;
    metadata.hold = 1;
    metadata.passes = 1;
    metadata.debug_view = 2;
    metadata.color = 0.25f;
    metadata.detail = 0.5f;
    const unsigned char bgra[] = {17, 32, 64, 255};
    capture_writer_begin(&writer, 2, 123);
    capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    require(!std::filesystem::exists(root / "manifest.txt", error) && !error, "published an incomplete batch");
    require(writer.remaining == 1, "wrong remaining frame count");
    // A missing writer, image or record changes nothing.
    capture_writer_begin(nullptr, 1, 1);
    capture_writer_write_frame(nullptr, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    capture_writer_write_frame(&writer, nullptr, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    capture_writer_write_frame(&writer, bgra, nullptr, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, nullptr);
    require(!capture_writer_active(nullptr) && writer.remaining == 1, "a null pointer changed the batch");
    capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    require(!capture_writer_active(&writer), "completed batch remains active");
    const auto first = read(root / "manifest.txt");
    require(field(first, "capture_metadata_version") == "2", "wrong metadata version");
    require(field(first, "capture_control_seq") == "123", "wrong capture token");
    require(field(first, "frame_1_before_hash") == "c2de31fd48ac5f39", "wrong input hash");
    require(field(first, "frame_1_inference_seq") == "55", "wrong inference provenance");
    require(field(first, "debug_view") == "2", "wrong renderer view");
    require(field(first, "color") == "0.25", "wrong renderer color");
    // The whole manifest, as the C++ writer wrote it.
    require(first == first_manifest(field(first, "batch_dir")), "the manifest's text changed");
    const auto batch = root / field(first, "batch_dir");
    const std::string batchName = "capture-" + std::to_string(getpid()) + "-";
    require(field(first, "batch_dir").rfind(batchName, 0) == 0 &&
            field(first, "batch_dir").size() == batchName.size() + 6, "wrong batch name");
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

    capture_writer_begin(&writer, 1, 124);
    require(read(root / "manifest.txt") == first, "changed completion before new batch finished");
    capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    const auto second = read(root / "manifest.txt");
    require(field(second, "capture_control_seq") == "124", "new completion not published");
    require(field(second, "batch_dir") != field(first, "batch_dir"), "reused batch directory");
    require(read(batch / "manifest.txt") == first, "changed prior batch metadata");

    capture_writer_begin(&writer, 1, 125);
    const unsigned short fp16[] = {0x3800, 0x3800, 0x3800, 0x3c00};
    capture_writer_write_frame(&writer, fp16, fp16, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, &metadata);
    const auto third = read(root / "manifest.txt");
    require(field(third, "encoding") == "raw", "FP16 capture was not raw");
    require(field(third, "bytes_per_pixel") == "8", "FP16 capture byte count incorrect");
    // Eight bytes take the hash's word step; the BGRA pixel above takes its byte tail.
    require(field(third, "before_hash") == "736955abadf41fdf", "wrong FP16 input hash");
    const auto raw = read(root / field(third, "batch_dir") / "before_00.raw");
    require(raw == std::string(reinterpret_cast<const char*>(fp16), sizeof fp16), "FP16 bytes changed");

    // A failed image write must leave the last completed batch visible.
    const auto priorDirectories = directories(root);
    capture_writer_begin(&writer, 1, 126);
    for (const auto& directory : directories(root))
        if (!priorDirectories.count(directory)) require(std::filesystem::remove(directory, error), "cannot remove a batch");
    capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    require(!capture_writer_active(&writer), "failed batch still active");
    require(read(root / "manifest.txt") == third, "published failed batch as complete");

    // A request for more frames than a batch holds writes as many as it holds. A float is written
    // with nine significant digits.
    metadata.detail = 0.1f;
    capture_writer_begin(&writer, 100, 128);
    require(writer.remaining == CAPTURE_WRITER_FRAMES, "a batch is not cut to CAPTURE_WRITER_FRAMES");
    for (uint32_t i = 0; i < CAPTURE_WRITER_FRAMES; ++i) {
        metadata.inference_seq = i;
        capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    }
    const auto full = read(root / "manifest.txt");
    require(!capture_writer_active(&writer) && field(full, "frames") == "64" &&
            field(full, "frame_63_inference_seq") == "63" && field(full, "frame_63_detail") == "0.100000001",
            "a full batch was not written");
    metadata.inference_seq = 55;
    metadata.detail = 0.5f;

    // A batch directory that cannot be created: the writer stays idle and abandons the batch it was
    // writing. With a capture directory that does not fit in CAPTURE_WRITER_DIR_SIZE bytes itself,
    // nothing is created, even when only its terminating null does not fit; with one that fits when
    // its batch does not, the directories are created and only the batch's is not.
    capture_writer_begin(&writer, 2, 129);
    require(capture_writer_active(&writer), "the batch before the failing directories did not start");
    // Components shorter than NAME_MAX, so that only the whole path is too long.
    std::string tooLong = temporary.path.string(), almost = tooLong, exact = tooLong;
    constexpr size_t size = CAPTURE_WRITER_DIR_SIZE;
    while (tooLong.size() < size) tooLong += "/" + std::string(200, 'x');
    while (almost.size() + 201 < size - 30) almost += "/" + std::string(200, 'y');
    almost += "/" + std::string(size - 31 - almost.size(), 'y');
    // A capture directory of exactly CAPTURE_WRITER_DIR_SIZE bytes, "/dlssnr/captures" included.
    while (exact.size() + 201 < size - 16) exact += "/" + std::string(200, 'z');
    require(exact.size() + 18 <= size, "the temporary directory leaves no room for the exact path");
    exact += "/" + std::string(size - 17 - exact.size(), 'z');
    // A capture directory under a file, which mkdir() cannot create.
    const std::string underFile = (temporary.path / "notes").string();
    std::ofstream(underFile) << "a file";
    for (const std::string& state : {tooLong, exact, almost, underFile}) {
        require(setenv("XDG_STATE_HOME", state.c_str(), 1) == 0, "setenv failed");
        capture_writer_begin(&writer, 1, 130);
        require(!capture_writer_active(&writer), "a batch began in a directory that cannot be created");
        capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
        require(!capture_writer_active(&writer), "an idle writer wrote a frame");
    }
    require(!std::filesystem::exists(temporary.path / std::string(200, 'x'), error),
            "a directory longer than CAPTURE_WRITER_DIR_SIZE was created in part");
    require(!std::filesystem::exists(temporary.path / std::string(200, 'z'), error),
            "a directory of exactly CAPTURE_WRITER_DIR_SIZE bytes was created in part");
    const std::string almostCaptures = almost + "/dlssnr/captures";
    require(std::filesystem::is_directory(almostCaptures, error) && std::filesystem::is_empty(almostCaptures, error),
            "the capture directory that fits was not created, or holds a batch");
    require(setenv("XDG_STATE_HOME", temporary.path.c_str(), 1) == 0, "setenv failed");

    // A batch publishes where it began, even if the environment moves before it completes.
    capture_writer_begin(&writer, 1, 127);
    require(setenv("XDG_STATE_HOME", (temporary.path / "moved").c_str(), 1) == 0, "setenv failed");
    capture_writer_write_frame(&writer, bgra, bgra, 1, 1, VK_FORMAT_B8G8R8A8_UNORM, &metadata);
    const auto fourth = read(root / "manifest.txt");
    require(field(fourth, "capture_control_seq") == "127", "batch did not publish where it began");
    require(read(root / field(fourth, "batch_dir") / "manifest.txt") == fourth, "wrong batch name");

    // The log's lines, which bench and test scripts read.
    const std::string log = read(logPath);
    const std::string rootText = root.string();
    require(logged(log, "[capture] capturing 2 frames, control 123, to " + rootText + "/" + field(first, "batch_dir")),
            "the first batch's start was not logged");
    require(logged(log, "[capture] wrote 2 pairs to " + rootText + "/" + field(first, "batch_dir")),
            "the first batch's end was not logged");
    require(logged(log, "[capture] capturing 64 frames, control 128, to " + rootText + "/" + field(full, "batch_dir")),
            "the cut batch's start was not logged");
    require(log.find("[capture] could not write " + rootText + "/capture-") != std::string::npos &&
            logged(log, "[capture] batch failed; completion manifest was not published"),
            "the failed batch was not logged");
    // The log cuts a line's text to 2047 bytes.
    const std::string failed = "[capture] cannot create batch directory in ";
    require(logged(log, (failed + almostCaptures).substr(0, 2047)), "the batch directory that did not fit was not logged");
    require(logged(log, (failed + tooLong).substr(0, 2047)),
            "the capture directory longer than CAPTURE_WRITER_DIR_SIZE was not logged");
    require(logged(log, (failed + exact).substr(0, 2047)),
            "the capture directory of CAPTURE_WRITER_DIR_SIZE bytes was not logged");
    require(logged(log, failed + underFile + "/dlssnr/captures"), "the capture directory under a file was not logged");
    std::printf("PASS: runtime and capture paths, text fields, capture publication, provenance, manifest text, preserved batches, BGRA PNG, FP16 raw, write failure, full batch, failed directories, moved environment, log lines\n");
    return 0;
}
