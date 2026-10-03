// SPDX-License-Identifier: MIT
// Host test of replace_file(), through which dlsslopd and every game's in-layer
// network save the one pipeline cache they share. Two writers replace a file at
// once while a reader reads it: every read must find the file, holding one
// writer's data whole, and no temporary file may stay behind. A replacement
// that fails must leave the file and its directory as they were.
#include "../backend/files.hpp"

#include <dirent.h>
#include <ftw.h>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;
// Whether OK; if not, the test fails with the message.
[[gnu::format(printf, 2, 3)]] bool expect(bool ok, const char* format, ...)
{
    if (ok) return true;
    std::va_list args;
    va_start(args, format);
    std::fputs("files test: ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    ++failures;
    return false;
}
bool expect_error(const dlsslop::Result<void>& result, const char* error)
{
    const char* what = result ? "no error" : result.error().what.c_str();
    return expect(!std::strcmp(what, error), "expected \"%s\", got \"%s\"", error, what);
}

// The names in DIRECTORY but . and .., sorted and joined by spaces.
std::string listing(const std::string& directory)
{
    std::vector<std::string> names;
    if (DIR* d = opendir(directory.c_str())) {
        while (const dirent* entry = readdir(d))
            if (std::strcmp(entry->d_name, ".") && std::strcmp(entry->d_name, "..")) names.emplace_back(entry->d_name);
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    std::string text;
    for (const std::string& name : names) text += (text.empty() ? "" : " ") + name;
    return text;
}

// BYTES that SEED sets apart from another writer's.
std::string pattern(size_t bytes, uint32_t seed)
{
    std::string data(bytes, '\0');
    for (char& c : data) c = char((seed = seed * 1664525u + 1013904223u) >> 24);
    return data;
}

// A replaced file holds the new data, and only its owner may read it.
void check_replace(const std::string& directory)
{
    const std::string file = dlsslop::join(directory, "file");
    for (const std::string& data : {pattern(1000, 1), std::string()}) {
        if (!expect(bool(dlsslop::replace_file(file, data)), "cannot replace %s", file.c_str())) return;
        const auto read = dlsslop::read_file(file);
        expect(read && *read == data, "%s does not hold the %zu bytes written", file.c_str(), data.size());
    }
    struct stat st{};
    const unsigned mode = stat(file.c_str(), &st) ? 0 : st.st_mode & 0777;
    expect(mode == 0600, "%s has mode %o, not 600", file.c_str(), mode);
    expect(listing(directory) == "file", "left in the directory: %s", listing(directory).c_str());
}

// Replacements that fail leave nothing behind: in a missing directory, and in
// place of a directory, which rename(2) does not replace with a file.
void check_failures(const std::string& directory)
{
    expect_error(dlsslop::replace_file(dlsslop::join(directory, "missing/file"), "data"), "No such file or directory");
    const std::string occupied = dlsslop::join(directory, "occupied");
    if (!expect(!mkdir(occupied.c_str(), 0700), "cannot create %s", occupied.c_str())) return;
    expect_error(dlsslop::replace_file(occupied, "data"), "Is a directory");
    expect(dlsslop::is_directory(occupied), "%s is no longer a directory", occupied.c_str());
    expect(listing(directory) == "occupied", "left in the directory: %s", listing(directory).c_str());
}

// Two writers replace one file ROUNDS times each, one with 256 KiB and the
// other with 384 KiB, while this thread reads it.
void check_concurrent(const std::string& directory, long rounds)
{
    const std::string file = dlsslop::join(directory, "vulkan-pipelines.cache");
    const std::string data[2] = {pattern(256 << 10, 1), pattern(384 << 10, 2)};
    if (!expect(bool(dlsslop::replace_file(file, data[0])), "cannot write %s", file.c_str())) return;
    std::string errors[2];
    std::atomic<int> writing{2};
    auto writer = [&](int w) {
        for (long round = 0; round < rounds; ++round)
            if (auto replaced = dlsslop::replace_file(file, data[w]); !replaced && errors[w].empty())
                errors[w] = replaced.error().what;
        writing.fetch_sub(1, std::memory_order_release);
    };
    std::thread first(writer, 0), second(writer, 1);
    size_t reads = 0, missing = 0, partial = 0, partial_bytes = 0;
    std::string missing_error;
    while (writing.load(std::memory_order_acquire)) {
        ++reads;
        const auto read = dlsslop::read_file(file);
        if (!read) {
            if (!missing++) missing_error = read.error().what;
        } else if (*read != data[0] && *read != data[1]) {
            if (!partial++) partial_bytes = read->size();
        }
    }
    first.join();
    second.join();
    for (int w = 0; w < 2; ++w) expect(errors[w].empty(), "writer %d failed: %s", w + 1, errors[w].c_str());
    expect(!missing, "%zu of %zu reads found no file, the first with \"%s\"", missing, reads, missing_error.c_str());
    expect(!partial, "%zu of %zu reads found neither writer's data whole, the first %zu bytes", partial, reads,
           partial_bytes);
    const auto last = dlsslop::read_file(file);
    expect(last && (*last == data[0] || *last == data[1]), "the last file is neither writer's data whole");
    expect(listing(directory) == "vulkan-pipelines.cache", "left in the directory: %s", listing(directory).c_str());
    std::printf("files test: %zu reads while 2 writers replaced the file %ld times each\n", reads, rounds);
}

} // namespace

int main(int argc, char** argv)
{
    const char* tmpdir = std::getenv("TMPDIR");
    std::string parent = tmpdir && *tmpdir ? tmpdir : "/tmp";
    long rounds = 1000;
    const option options[] = {{"directory", required_argument, nullptr, 'd'},
                              {"rounds", required_argument, nullptr, 'r'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, 0, nullptr, 0}};
    for (int code; (code = getopt_long(argc, argv, "+d:r:h", options, nullptr)) != -1;) {
        char* end = nullptr;
        switch (code) {
        case 'd':
            parent = optarg;
            break;
        case 'r':
            rounds = std::strtol(optarg, &end, 10);
            if (*optarg && !*end && rounds > 0) break;
            std::fprintf(stderr, "files-test: --rounds wants a positive count, not \"%s\"\n", optarg);
            return 2;
        case 'h':
            std::puts("Usage: files-test [OPTION]...\n"
                      "Checks that replace_file() replaces a file whole while other writers replace it too.\n"
                      " -d, --directory DIR  Work in a new directory in DIR (default: $TMPDIR, or /tmp without it)\n"
                      " -r, --rounds N       Replacements by each of the 2 writers (default: 1000)\n"
                      " -h, --help           Show help (default: off)");
            return 0;
        default:
            return 2;
        }
    }
    if (optind != argc) return 2;
    std::string root = dlsslop::join(parent, "files-test-XXXXXX");
    if (!mkdtemp(root.data())) {
        std::fprintf(stderr, "files test: cannot create a directory in %s: %s\n", parent.c_str(), std::strerror(errno));
        return 1;
    }
    const std::string replace = dlsslop::join(root, "replace"), fail = dlsslop::join(root, "fail"),
                      concurrent = dlsslop::join(root, "concurrent");
    if (expect(!mkdir(replace.c_str(), 0700) && !mkdir(fail.c_str(), 0700) && !mkdir(concurrent.c_str(), 0700),
               "cannot create directories in %s", root.c_str())) {
        check_replace(replace);
        check_failures(fail);
        check_concurrent(concurrent, rounds);
    }
    nftw(root.c_str(), [](const char* path, const struct stat*, int, FTW*) { return remove(path); }, 8,
         FTW_DEPTH | FTW_PHYS);
    if (!failures) std::puts("files test: every check passed");
    return failures ? 1 : 0;
}
