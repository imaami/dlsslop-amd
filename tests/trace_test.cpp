// SPDX-License-Identifier: MIT
#include "../backend/trace.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <initializer_list>
#include <iterator>

namespace {
// The test's directory, removed however the test ends.
struct Scratch {
    std::filesystem::path directory;
    ~Scratch() {
        std::error_code ignored;
        if (!directory.empty()) std::filesystem::remove_all(directory, ignored);
    }
} scratch;

void require(bool yes, const char* message) {
    if (yes) return;
    std::fprintf(stderr, "trace test: %s\n", message);
    std::exit(1);
}
// Whether FILE exists; an error reads as no.
bool present(const std::filesystem::path& file) {
    std::error_code ignored;
    return std::filesystem::exists(file, ignored);
}
std::string read_text(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
// The next claimed trace; failing to claim one fails the test.
std::unique_ptr<dlsslop::FrameTrace> taken(dlsslop::TraceRequests& requests) {
    auto trace = requests.take();
    require(bool(trace), "claim a trace request");
    return std::move(*trace);
}
void write_text(const std::filesystem::path& file, const std::string& text) {
    require(bool(dlsslop::trace_write_text(file.string(), text)), "write a text file");
}
// The client parses every file with Python's json module; nan, inf and a
// missing separator must fail here, not in a diagnostic run.
bool strict_json(const std::string& python, std::initializer_list<std::filesystem::path> files) {
    std::string command = "'" + python + "' -c 'import json, sys\n"
        "def constant(name): raise ValueError(name)\n"
        "for name in sys.argv[1:]: json.load(open(name), parse_constant=constant)'";
    for (const auto& file : files) command += " '" + file.string() + "'";
    return std::system(command.c_str()) == 0;
}
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: trace-test PYTHON\n");
        return 2;
    }
    char temporary[] = "/tmp/dlsslop-amd-trace-test-XXXXXX";
    if (!mkdtemp(temporary)) return 1;
    scratch.directory = temporary;
    const std::filesystem::path& directory = scratch.directory;
    std::error_code error;
    dlsslop::Geometry g{2, 2, 4, 4, 4, 1, 1, 2, 2};
    std::vector<float> rgba(4 * 4 * 4, 99.0f);
    for (unsigned y = 1; y <= 2; ++y) for (unsigned x = 1; x <= 2; ++x) {
        const unsigned p = (y * 4 + x) * 4;
        rgba[p] = float(y); rgba[p + 1] = float(x); rgba[p + 2] = -0.5f;
    }
    require(bool(dlsslop::trace_write_pfm((directory / "test.pfm").string(), rgba.data(), g, 4)), "write a PFM");
    std::ifstream pfm(directory / "test.pfm", std::ios::binary);
    std::string line;
    std::getline(pfm, line); require(line == "PF", "RGB PFM magic");
    std::getline(pfm, line); require(line == "2 2", "PFM fitted viewport dimensions");
    std::getline(pfm, line);
    const std::uint16_t endian = 1;
    require(line == (*reinterpret_cast<const unsigned char*>(&endian) ? "-1.0" : "1.0"), "PFM byte order");
    std::array<float, 12> actual{};
    pfm.read(reinterpret_cast<char*>(actual.data()), sizeof actual);
    const std::array<float, 12> expected{2,1,-.5f, 2,2,-.5f, 1,1,-.5f, 1,2,-.5f};
    require(actual == expected, "PFM rows bottom-up, channels RGB, padding and alpha excluded, "
            "extended values retained");
    require(dlsslop::trace_valid_token("stage_01-A") && !dlsslop::trace_valid_token("../escape") &&
            !dlsslop::trace_valid_token("") && !dlsslop::trace_valid_token(std::string(65, 'a')) &&
            dlsslop::trace_valid_token(std::string(64, 'a')) && !dlsslop::trace_valid_token("request") &&
            dlsslop::trace_valid_token("requests"), "safe bounded request tokens");
    // Other users must not be able to plant files (such as a symlinked
    // owner.json.tmp) in the root, so only a private one is accepted.
    const auto rejected = [&](const std::filesystem::path& root) {
        return !dlsslop::TraceRequests::open(root.string(), "/tmp/example-shm") &&
               !present(root / ".worker-lock") && !present(root / "owner.json");
    };
    for (const auto mode : {std::filesystem::perms::all, std::filesystem::perms::owner_all |
                            std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
                            std::filesystem::perms::others_read | std::filesystem::perms::others_exec}) {
        std::filesystem::permissions(directory, mode, error);
        require(!error && rejected(directory), "reject a trace root other users can read or write");
    }
    std::filesystem::permissions(directory, std::filesystem::perms::owner_all, error);
    require(!error, "restore the trace root's permissions");
    std::filesystem::create_directory_symlink(directory, directory / "link", error);
    require(!error && rejected(directory / "link"), "reject a symlinked trace root");
    require(std::filesystem::remove(directory / "link", error), "remove the symlinked root");
    {
        const auto created = dlsslop::TraceRequests::open((directory / "new/nested").string(), "/tmp/example-shm");
        require(bool(created) && (std::filesystem::status(directory / "new/nested", error).permissions() & std::filesystem::perms::all) ==
                std::filesystem::perms::owner_all, "a created trace root is private");
    }
    std::filesystem::remove_all(directory / "new", error);
    require(!error, "remove the created trace root");
    {
        auto opened = dlsslop::TraceRequests::open(directory.string(), "/tmp/example-shm");
        require(bool(opened), "open a private trace root");
        dlsslop::TraceRequests& requests = *opened;
        const auto owner = read_text(directory / "owner.json");
        require(owner.find("\"pid\":" + std::to_string(getpid()) + ",") != std::string::npos &&
                owner.find("\"shm\":\"/tmp/example-shm\"") != std::string::npos, "owner discovery metadata");
        require(read_text(directory / "owner.json").find("\"trace_metadata_schema\":2") != std::string::npos,
                "owner must advertise frozen-input evidence before capture");
        require(!dlsslop::TraceRequests::open(directory.string(), "/tmp/other"), "one worker per trace directory");
        require(!taken(requests), "no request means no trace");
        write_text(directory / "request", "frame_1\n");
        auto frame = taken(requests);
        require(frame && !taken(requests), "request consumed exactly once");
        require((std::filesystem::status(directory / "frame_1", error).permissions() & std::filesystem::perms::all) ==
                std::filesystem::perms::owner_all, "a token directory is private");
        frame->image("pass-01-input", rgba.data(), g, 4);
        frame->image("pass-01-raw", rgba.data(), g, 4);
        require(!present(directory / "frame_1/summary.json"), "summary must complete last");
        require(!present(directory / "frame_1.done"), "no early root completion marker");
        // Exercise the serializer used by the worker, not a hand-written
        // summary that could silently omit fields required by the client.
        dlsslop::TraceFrameMetadata metadata;
        metadata.frame_seq = 123;
        metadata.control_seq = 11;
        metadata.tuning_seq = 7;
        metadata.control_seq_end = 12;
        metadata.tuning_seq_end = 8;
        metadata.held_input = 1;
        metadata.held_input_end = 0;
        metadata.source_proxy_hash = 0x1234abcd;
        metadata.passes = 3;
        metadata.geometry = g;
        metadata.fp16_proxy = true;
        metadata.fp16_feedback = true;
        metadata.intensity = 0.5f;
        metadata.local_tone = 1;
        metadata.local_structure = 1;
        frame->finish(metadata.json());
        require(read_text(directory / "frame_1.done") == "frame_1/summary.json\n", "root marker follows summary commit");
        const auto summary = read_text(directory / "frame_1/summary.json");
        require(summary.find("\"status\":\"complete\"") != std::string::npos &&
                summary.find("\"frame_seq\":123") != std::string::npos, "summary completion and metadata");
        require(summary.find("\"schema\":1") != std::string::npos,
                "metadata changes preserve the outer summary file format");
        for (const char* field : {
                "\"trace_metadata_schema\":2", "\"control_seq\":11", "\"tuning_seq\":7",
                "\"control_seq_end\":12", "\"tuning_seq_end\":8",
                "\"held_input\":1", "\"held_input_end\":0",
                "\"source_proxy_hash\":\"000000001234abcd\"", "\"passes\":3",
                "\"source_width\":2", "\"source_height\":2",
                "\"network_width\":4", "\"network_height\":4",
                "\"fit_x\":1", "\"fit_y\":1", "\"fit_width\":2", "\"fit_height\":2",
                "\"fp16_proxy\":1", "\"fp16_feedback\":1", "\"motion\":0",
                "\"intensity\":0.5", "\"local_tone\":1", "\"local_structure\":1",
                "\"sharpness\":0",
                "\"file\":\"pass-01-input.pfm\"", "\"file\":\"pass-01-raw.pfm\""})
            require(summary.find(field) != std::string::npos, field);
        write_text(directory / "request", "frame_1\n");
        const bool duplicate = !requests.take();
        require(duplicate && read_text(directory / "frame_1/summary.json") == summary,
                "duplicate token must never overwrite evidence");
        write_text(directory / "request", "../escape\n");
        const bool traversal = !requests.take();
        require(traversal, "reject path traversal");
        // The protocol file's own name would make the evidence directory
        // DIR/request, which the next claim would take as a request.
        write_text(directory / "request", "request\n");
        const bool reserved = !requests.take();
        require(reserved && !present(directory / "request"), "reject the reserved token");
        require(std::filesystem::create_directory(directory / "request", error), "create a stray request directory");
        const bool stray = !requests.take();
        require(stray && !present(directory / "request") &&
                !present(directory / (".request-" + std::to_string(getpid()))),
                "a stray empty request directory is cleared, not left to block claims");
        // A failed frame still publishes its summary and root marker, with
        // its error escaped and no stage recorded after a failed stage.
        write_text(directory / "request", "frame_2\n");
        auto failed = taken(requests);
        require(bool(failed), "later requests are claimed");
        failed->image("pass-01-input", rgba.data(), g, 4);
        failed->image("pass-01-tuned", rgba.data(), g, 2);
        failed->image("pass-01-raw", rgba.data(), g, 4);
        failed->finish("{\"frame_seq\":124}", "bad \"quote\"\n\x01\\");
        require(read_text(directory / "frame_2.done") == "frame_2/summary.json\n", "failed trace marker");
        const auto failure = read_text(directory / "frame_2/summary.json");
        for (const char* field : {"\"status\":\"failed\"", "\"metadata\":{\"frame_seq\":124}",
                                  "\"error\":\"bad \\\"quote\\\"\\u000a\\u0001\\\\\"",
                                  "\"file\":\"pass-01-input.pfm\""})
            require(failure.find(field) != std::string::npos, field);
        require(failure.find("pass-01-raw") == std::string::npos &&
                failure.find("pass-01-tuned") == std::string::npos &&
                !present(directory / "frame_2/pass-01-raw.pfm") &&
                !present(directory / "frame_2/pass-01-tuned.pfm"),
                "no stage from or after a failed stage");
        // A trace dropped unfinished, as when the worker stops after the
        // claim, still publishes a failed summary and its root marker.
        write_text(directory / "request", "frame_3\n");
        taken(requests)->image("pass-01-input", rgba.data(), g, 4);
        require(read_text(directory / "frame_3.done") == "frame_3/summary.json\n", "abandoned trace marker");
        const auto abandoned = read_text(directory / "frame_3/summary.json");
        for (const char* field : {"\"status\":\"failed\"", "\"metadata\":{}",
                                  "\"error\":\"worker stopped before a traced frame completed\"",
                                  "\"file\":\"pass-01-input.pfm\""})
            require(abandoned.find(field) != std::string::npos, field);
        // A stage failure recorded but not finished keeps its own error.
        write_text(directory / "request", "frame_4\n");
        taken(requests)->image("pass-01-input", rgba.data(), g, 2);
        const auto unfinished = read_text(directory / "frame_4/summary.json");
        require(unfinished.find("\"status\":\"failed\"") != std::string::npos &&
                unfinished.find("\"error\":\"invalid diagnostic image geometry\"") != std::string::npos,
                "an unfinished failure keeps its error");
        // A finished trace is published once, not again on destruction.
        require(std::filesystem::remove(directory / "frame_2.done", error), "remove a completion marker");
        failed.reset();
        require(!present(directory / "frame_2.done"), "finish publishes once");
        require(strict_json(argv[1], {directory / "owner.json", directory / "frame_1/summary.json",
                                      directory / "frame_2/summary.json", directory / "frame_3/summary.json",
                                      directory / "frame_4/summary.json"}), "summaries are strict JSON");
    }
    require(!present(directory / "owner.json"), "remove stale owner on shutdown");
    std::puts("trace: fitted RGB PFM, safe requests, ownership, failure, completion and JSON passed");
    return 0;
}
