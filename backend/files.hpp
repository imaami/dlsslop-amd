// Paths and whole files through POSIX calls, without std::filesystem or iostreams.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <unistd.h>

namespace dlsslop {

// Owns a descriptor, and with it any flock on the file.
struct Descriptor {
    int fd = -1;
    Descriptor() = default;
    explicit Descriptor(int fd) : fd(fd) {}
    Descriptor(Descriptor&& other) noexcept : fd(std::exchange(other.fd, -1)) {}
    Descriptor& operator=(Descriptor&& other) noexcept
    {
        std::swap(fd, other.fd);
        return *this;
    }
    ~Descriptor()
    {
        if (fd >= 0) close(fd);
    }
};

bool is_directory(const std::string& path);
bool is_regular_file(const std::string& path);
// PATH without its last component: empty for a bare name, "/" for one at the root.
std::string parent_path(std::string_view path);
// A and B joined by one slash; B alone when A is empty.
std::string join(std::string_view a, std::string_view b);
// PATH, relative to the working directory unless it is absolute.
std::string absolute(const std::string& path);
// The whole file, or strerror of what stopped it.
Result<std::string> read_file(const std::string& path);
// BYTES from FD into DATA: a file that ends first is an error.
Result<void> read_all(int fd, void* data, size_t bytes);
// BYTES from FD at OFFSET into DATA, leaving FD's position as it was: a file
// that ends first is an error.
Result<void> read_at(int fd, uint64_t offset, void* data, size_t bytes);
// All of DATA to FD.
Result<void> write_all(int fd, const void* data, size_t bytes);
// DATA as FILE, created or truncated; or strerror of what stopped it.
Result<void> write_file(const std::string& file, std::string_view data);
// DATA as FILE, written to a new file beside it that rename(2) then puts in
// FILE's place: a reader finds the old FILE or the new one, whole, and
// concurrent writers each write a file of their own. The new FILE has mode
// 0600, as mkstemp(3) creates it. Or strerror of what stopped it, with FILE
// unchanged.
Result<void> replace_file(const std::string& file, std::string_view data);
// Creates DIR and any missing parents: true when DIR itself was created.
Result<bool> make_directories(const std::string& directory);
// Creates DIR, with any missing parents, as 0700, or accepts an existing real
// directory the current user owns with exactly that mode, so no other user can
// plant or swap files in it. WHAT names it in errors.
Result<void> private_directory(const std::string& directory, const char* what);

} // namespace dlsslop
