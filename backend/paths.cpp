// SPDX-License-Identifier: MIT
#include "paths.h"
#include "files.h"

#include <climits>
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>

namespace dlsslop {

Result<void> require_vulkan_model(const std::string& model)
{
    if (is_regular_file(model)) return {};
    return fail("no model at " + model + " (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)");
}

std::string home_directory()
{
    const char* home = std::getenv("HOME");
    if (!home || !*home) {
        const struct passwd* user = getpwuid(getuid());
        home = user ? user->pw_dir : nullptr;
    }
    return home ? home : "";
}

std::string xdg_path(const char* base, const char* fallback, const char* name)
{
    if (const char* root = std::getenv(base); root && *root) return join(root, name);
    const std::string home = home_directory();
    return home.empty() ? home : join(join(home, fallback), name);
}

std::string default_assets() { return xdg_path("XDG_DATA_HOME", ".local/share", "dlsslop-amd/model"); }
std::string default_vulkan_model() { return xdg_path("XDG_DATA_HOME", ".local/share", "dlsslop-amd/dlssnr.bin"); }
std::string default_config() { return xdg_path("XDG_CONFIG_HOME", ".config", "dlsslop-amd/dlsslopd.conf"); }

std::string executable_path()
{
    char path[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", path, sizeof path);
    return length > 0 && size_t(length) < sizeof path ? std::string(path, size_t(length)) : std::string();
}

std::string vulkan_shaders(const std::string& prefix, const std::string& build)
{
    const std::string installed = join(prefix, "share/dlsslop-amd/vulkan");
    const std::string development = join(build, "vulkan-nr/network");
    return is_directory(development) && !is_directory(installed) ? development : installed;
}

std::string vulkan_shaders()
{
    const std::string bin = parent_path(executable_path());
    return vulkan_shaders(parent_path(bin), bin);
}

std::string vulkan_cache()
{
    const std::string cache = xdg_path("XDG_CACHE_HOME", ".cache", "dlsslop-amd/vulkan-pipelines.cache");
    return !cache.empty() && make_directories(parent_path(cache)) ? cache : std::string();
}

std::string default_modules()
{
    if (const char* path = std::getenv("DLSSLOP_MODULES"); path && *path) return path;
    const std::string executable = executable_path();
    if (executable.empty()) return {};
    const std::string prefix = parent_path(parent_path(executable));
    const std::string installed = join(prefix, "share/dlsslop-amd/HIP/gfx1201");
    if (is_directory(installed)) return installed;
    // A worker run directly from the source tree's build directory uses the
    // same modules as the kernel build script, without an installed launcher.
    const std::string development = join(prefix, "assets/HIP/gfx1201");
    return is_directory(development) ? development : installed;
}

} // namespace dlsslop
