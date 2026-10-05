// Paths and whole files for the C++ that calls them: files.h's functions with
// std::string and Results.
// SPDX-License-Identifier: MIT
#pragma once
#include "files.h"
#include "result.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace dlsslop {

// TEXT, a path of LENGTH bytes that files.h made on the heap, as a string;
// TEXT is freed. Without memory for TEXT the program ends, as it does without
// memory for a string.
inline std::string take_path(char* text, size_t length)
{
    if (!text) std::abort();
    std::string path(text, length);
    std::free(text);
    return path;
}

inline bool is_directory(const std::string& path) { return files_is_directory(path.c_str()); }
inline bool is_regular_file(const std::string& path) { return files_is_regular_file(path.c_str()); }
// PATH without its last component: empty for a bare name, "/" for one at the root.
inline std::string parent_path(std::string_view path)
{
    return std::string(path.substr(0, files_parent(path.data(), path.size())));
}
// A and B joined by one slash; B alone when A is empty.
inline std::string join(std::string_view a, std::string_view b)
{
    size_t length = 0;
    char* const joined = files_join(a.data(), a.size(), b.data(), b.size(), &length);
    return take_path(joined, length);
}
// PATH, relative to the working directory unless it is absolute.
inline std::string absolute(const std::string& path)
{
    size_t length = 0;
    char* const made = files_absolute(path.c_str(), path.size(), &length);
    return take_path(made, length);
}
// The whole file, or the words for what stopped it.
inline Result<std::string> read_file(const std::string& path)
{
    struct files_data data;
    struct error e;
    if (const enum error_code code = files_read(&data, path.c_str(), &e)) return forward_c(code, e);
    std::string text(reinterpret_cast<const char*>(data.bytes), data.size);
    files_data_fini(&data);
    return text;
}
// BYTES from FD into DATA: a file that ends first is an error.
inline Result<void> read_all(int fd, void* data, size_t bytes)
{
    struct error e;
    if (const enum error_code code = files_read_all(fd, data, bytes, &e)) return forward_c(code, e);
    return {};
}
// BYTES from FD at OFFSET into DATA, leaving FD's position as it was: a file
// that ends first is an error.
inline Result<void> read_at(int fd, uint64_t offset, void* data, size_t bytes)
{
    struct error e;
    if (const enum error_code code = files_read_at(fd, offset, data, bytes, &e)) return forward_c(code, e);
    return {};
}
// All of DATA to FD.
inline Result<void> write_all(int fd, const void* data, size_t bytes)
{
    struct error e;
    if (const enum error_code code = files_write_all(fd, data, bytes, &e)) return forward_c(code, e);
    return {};
}
// DATA as FILE, created or truncated; or strerror of what stopped it.
inline Result<void> write_file(const std::string& file, std::string_view data)
{
    struct error e;
    if (const enum error_code code = files_write(file.c_str(), data.data(), data.size(), &e))
        return forward_c(code, e);
    return {};
}
// DATA as FILE, written to a new file beside it that rename(2) then puts in
// FILE's place: a reader finds the old FILE or the new one, whole, and
// concurrent writers each write a file of their own. The new FILE has mode
// 0600, as mkstemp(3) creates it. Or the words for what stopped it, with FILE
// unchanged.
inline Result<void> replace_file(const std::string& file, std::string_view data)
{
    struct error e;
    if (const enum error_code code = files_replace(file.c_str(), file.size(), data.data(), data.size(), &e))
        return forward_c(code, e);
    return {};
}
// Creates DIR and any missing parents: true when DIR itself was created.
inline Result<bool> make_directories(const std::string& directory)
{
    bool created = false;
    struct error e;
    if (const enum error_code code = files_make_directories(directory.c_str(), directory.size(), &created, &e))
        return forward_c(code, e);
    return created;
}
// Creates DIR, with any missing parents, as 0700, or accepts an existing real
// directory the current user owns with exactly that mode, so no other user can
// plant or swap files in it. WHAT names it in errors.
inline Result<void> private_directory(const std::string& directory, const char* what)
{
    struct error e;
    if (const enum error_code code = files_private_directory(directory.c_str(), directory.size(), what, &e))
        return forward_c(code, e);
    return {};
}

} // namespace dlsslop
