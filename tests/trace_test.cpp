// SPDX-License-Identifier: MIT
#include "../backend/trace.h"
#include <cstdlib>
#include <initializer_list>
#include <iterator>

namespace {
void require(bool yes, const char* message) {
    if (!yes) throw std::runtime_error(message);
}
std::string read_text(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
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
    const char* made = mkdtemp(temporary);
    if (!made) return 1;
    const std::filesystem::path directory(made);
    try {
        dlsslop::Geometry g{2, 2, 4, 4, 4, 4, 1, 1, 2, 2};
        std::vector<float> rgba(4 * 4 * 4, 99.0f);
        for (unsigned y = 1; y <= 2; ++y) for (unsigned x = 1; x <= 2; ++x) {
            const unsigned p = (y * 4 + x) * 4;
            rgba[p] = float(y); rgba[p + 1] = float(x); rgba[p + 2] = -0.5f;
        }
        const auto stats = dlsslop::trace_write_pfm(directory / "test.pfm", rgba.data(), g, 4);
        require(stats.pixels == 4 && stats.sum[0] == 6 && stats.sum[1] == 6 && stats.sum[2] == -2,
                "stats must exclude padding and alpha");
        require(stats.at_one[0] == 4 && stats.at_zero[2] == 4, "extended values must be retained");
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
        require(actual == expected, "PFM rows bottom-up and channels RGB");
        float nonfinite[] = {std::numeric_limits<float>::quiet_NaN(), 0, 1};
        dlsslop::TraceStats invalid; invalid.add(nonfinite);
        require(invalid.nonfinite[0] == 1 && invalid.json().find("\"mean\":null") != std::string::npos,
                "nonfinite statistics must remain valid JSON");
        require(dlsslop::trace_valid_token("stage_01-A") && !dlsslop::trace_valid_token("../escape") &&
                !dlsslop::trace_valid_token("") && !dlsslop::trace_valid_token(std::string(65, 'a')) &&
                dlsslop::trace_valid_token(std::string(64, 'a')) && !dlsslop::trace_valid_token("request") &&
                dlsslop::trace_valid_token("requests"), "safe bounded request tokens");
        // Other users must not be able to plant files (such as a symlinked
        // owner.json.tmp) in the root, so only a private one is accepted.
        const auto rejected = [&](const std::filesystem::path& root) {
            try { dlsslop::TraceRequests loose(root.string(), "/tmp/example-shm", 22); }
            catch (const std::runtime_error&) {
                return !std::filesystem::exists(root / ".worker-lock") && !std::filesystem::exists(root / "owner.json");
            }
            return false;
        };
        for (const auto mode : {std::filesystem::perms::all, std::filesystem::perms::owner_all |
                                std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
                                std::filesystem::perms::others_read | std::filesystem::perms::others_exec}) {
            std::filesystem::permissions(directory, mode);
            require(rejected(directory), "reject a trace root other users can read or write");
        }
        std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
        std::filesystem::create_directory_symlink(directory, directory / "link");
        require(rejected(directory / "link"), "reject a symlinked trace root");
        std::filesystem::remove(directory / "link");
        {
            dlsslop::TraceRequests created((directory / "new/nested").string(), "/tmp/example-shm", 22);
            require((std::filesystem::status(directory / "new/nested").permissions() & std::filesystem::perms::all) ==
                    std::filesystem::perms::owner_all, "a created trace root is private");
        }
        std::filesystem::remove_all(directory / "new");
        {
            dlsslop::TraceRequests requests(directory.string(), "/tmp/example-shm", 22);
            require(read_text(directory / "owner.json").find("\"protocol_version\":22") != std::string::npos,
                    "owner discovery metadata");
            require(read_text(directory / "owner.json").find("\"trace_metadata_schema\":2") != std::string::npos,
                    "owner must advertise frozen-input evidence before capture");
            bool locked = false;
            try { dlsslop::TraceRequests duplicate(directory.string(), "/tmp/other", 22); }
            catch (const std::runtime_error&) { locked = true; }
            require(locked, "one worker per trace directory");
            require(!requests.take(), "no request means no trace");
            dlsslop::trace_write_text(directory / "request", "frame_1\n");
            auto frame = requests.take();
            require(frame && !requests.take(), "request consumed exactly once");
            frame->image("pass-01-input", rgba.data(), g, 4);
            frame->image("pass-01-raw", rgba.data(), g, 4);
            require(!std::filesystem::exists(directory / "frame_1/summary.json"), "summary must complete last");
            require(!std::filesystem::exists(directory / "frame_1.done"), "no early root completion marker");
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
            metadata.detail = 0.75f;
            metadata.color = 0.25f;
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
                    "\"sharpness\":0", "\"detail\":0.75", "\"color\":0.25",
                    "\"file\":\"pass-01-input.pfm\"", "\"file\":\"pass-01-raw.pfm\""})
                require(summary.find(field) != std::string::npos, field);
            dlsslop::trace_write_text(directory / "request", "frame_1\n");
            bool duplicate = false;
            try { requests.take(); } catch (const std::runtime_error&) { duplicate = true; }
            require(duplicate && read_text(directory / "frame_1/summary.json") == summary,
                    "duplicate token must never overwrite evidence");
            dlsslop::trace_write_text(directory / "request", "../escape\n");
            bool traversal = false;
            try { requests.take(); } catch (const std::runtime_error&) { traversal = true; }
            require(traversal, "reject path traversal");
            // The protocol file's own name would make the evidence directory
            // DIR/request, which the next claim would take as a request.
            dlsslop::trace_write_text(directory / "request", "request\n");
            bool reserved = false;
            try { requests.take(); } catch (const std::runtime_error&) { reserved = true; }
            require(reserved && !std::filesystem::exists(directory / "request"), "reject the reserved token");
            std::filesystem::create_directory(directory / "request");
            bool stray = false;
            try { requests.take(); } catch (const std::runtime_error&) { stray = true; }
            require(stray && !std::filesystem::exists(directory / "request") &&
                    !std::filesystem::exists(directory / (".request-" + std::to_string(getpid()))),
                    "a stray empty request directory is cleared, not left to block claims");
            // A failed frame still publishes its summary and root marker, with
            // the error escaped and no stage recorded after the failure.
            dlsslop::trace_write_text(directory / "request", "frame_2\n");
            auto failed = requests.take();
            require(bool(failed), "later requests are claimed");
            failed->image("pass-01-input", rgba.data(), g, 4);
            failed->fail("bad \"quote\"\n\x01\\");
            failed->image("pass-01-raw", rgba.data(), g, 4);
            failed->finish("{\"frame_seq\":124}");
            require(read_text(directory / "frame_2.done") == "frame_2/summary.json\n", "failed trace marker");
            const auto failure = read_text(directory / "frame_2/summary.json");
            for (const char* field : {"\"status\":\"failed\"", "\"metadata\":{\"frame_seq\":124}",
                                      "\"error\":\"bad \\\"quote\\\"\\u000a\\u0001\\\\\"",
                                      "\"file\":\"pass-01-input.pfm\""})
                require(failure.find(field) != std::string::npos, field);
            require(failure.find("pass-01-raw") == std::string::npos &&
                    !std::filesystem::exists(directory / "frame_2/pass-01-raw.pfm"),
                    "no stage after a failure");
            require(strict_json(argv[1], {directory / "owner.json", directory / "frame_1/summary.json",
                                          directory / "frame_2/summary.json"}), "summaries are strict JSON");
        }
        require(!std::filesystem::exists(directory / "owner.json"), "remove stale owner on shutdown");
        std::filesystem::remove_all(directory);
        std::puts("trace: fitted RGB PFM, statistics, safe requests, ownership, failure, completion and JSON passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "trace test: %s\n", error.what());
        std::filesystem::remove_all(directory);
        return 1;
    }
}
