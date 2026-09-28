// SPDX-License-Identifier: MIT
#include "files.h"

#include <climits>
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

std::string absolute(const std::string& path)
{
    if (path.starts_with('/')) return path;
    char cwd[PATH_MAX];
    return getcwd(cwd, sizeof cwd) ? join(cwd, path) : path;
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

Result<void> write_all(int fd, const void* data, size_t bytes)
{
    for (auto* at = static_cast<const char*>(data); bytes;) {
        const ssize_t wrote = write(fd, at, bytes);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return fail(std::strerror(wrote ? errno : EIO));
        at += wrote;
        bytes -= size_t(wrote);
    }
    return {};
}

Result<void> write_file(const std::string& file, std::string_view data)
{
    const Descriptor out(open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666));
    if (out.fd < 0) return fail(std::strerror(errno));
    return write_all(out.fd, data.data(), data.size());
}

Result<bool> make_directories(const std::string& path)
{
    const std::string directory = path.substr(0, path.find_last_not_of('/') + 1);
    bool created = false;
    for (size_t end = directory.find('/', 1);; end = directory.find('/', end + 1)) {
        created = !mkdir(directory.substr(0, end).c_str(), 0777);
        if (!created && errno != EEXIST) return fail(std::strerror(errno));
        if (end == std::string::npos) return created;
    }
}

Result<void> private_directory(const std::string& path, const char* what)
{
    const std::string directory = path.substr(0, path.find_last_not_of('/') + 1);
    const auto made = make_directories(directory);
    if (!made) return fail(std::string("create ") + what + " directory: " + made.error().what);
    const bool created = *made;
    struct stat st{};
    if (lstat(directory.c_str(), &st) || !S_ISDIR(st.st_mode) || st.st_uid != getuid())
        return fail(std::string(what) + " directory must be owned by the current user and not a symlink");
    if (created ? chmod(directory.c_str(), 0700) != 0 : (st.st_mode & 0777) != 0700)
        return fail(std::string(what) + " directory must be private (mode 0700): " + directory);
    return {};
}

} // namespace dlsslop
