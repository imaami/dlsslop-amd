// SPDX-License-Identifier: MIT
#include "files.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dlsslop {

namespace {
mode_t file_type(const std::string& path)
{
    struct stat st{};
    return stat(path.c_str(), &st) ? 0 : st.st_mode & S_IFMT;
}
} // namespace

bool is_directory(const std::string& path) { return file_type(path) == S_IFDIR; }
bool is_regular_file(const std::string& path) { return file_type(path) == S_IFREG; }

std::string parent_path(std::string_view path)
{
    const size_t slash = path.rfind('/');
    return slash == std::string_view::npos ? std::string() : std::string(path.substr(0, slash ? slash : 1));
}

std::string join(std::string_view a, std::string_view b)
{
    std::string joined(a);
    if (!joined.empty() && joined.back() != '/') joined += '/';
    return joined += b;
}

Result<std::string> read_file(const std::string& path)
{
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail(std::strerror(errno));
    std::string text;
    char buffer[65536];
    ssize_t got;
    while ((got = read(fd, buffer, sizeof buffer)) > 0) text.append(buffer, size_t(got));
    const int error = errno;
    close(fd);
    if (got < 0) return fail(std::strerror(error));
    return text;
}

} // namespace dlsslop
