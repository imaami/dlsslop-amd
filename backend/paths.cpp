// SPDX-License-Identifier: MIT
#include "paths.h"

#include <cstdlib>
#include <filesystem>
#include <pwd.h>
#include <unistd.h>

namespace dlsslop {
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
    if (const char* root = std::getenv(base); root && *root) return (std::filesystem::path(root) / name).string();
    const std::string home = home_directory();
    return home.empty() ? home : (std::filesystem::path(home) / fallback / name).string();
}

std::string default_assets() { return xdg_path("XDG_DATA_HOME", ".local/share", "dlsslop-amd/model"); }
std::string default_vulkan_model() { return xdg_path("XDG_DATA_HOME", ".local/share", "dlsslop-amd/dlssnr.bin"); }
std::string default_config() { return xdg_path("XDG_CONFIG_HOME", ".config", "dlsslop-amd/dlsslopd.conf"); }

std::string executable_path()
{
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::string() : executable.string();
}

std::string vulkan_shaders()
{
    const std::filesystem::path executable = executable_path();
    const auto installed = executable.parent_path().parent_path() / "share/dlsslop-amd/vulkan";
    const auto development = executable.parent_path() / "vulkan-nr/network";
    std::error_code error;
    return (std::filesystem::is_directory(development, error) && !std::filesystem::is_directory(installed, error)
                ? development : installed).string();
}

std::string default_modules()
{
    if (const char* path = std::getenv("DLSSLOP_MODULES"); path && *path) return path;
    const std::filesystem::path executable = executable_path();
    if (executable.empty()) return {};
    const auto prefix = executable.parent_path().parent_path();
    std::error_code error;
    const auto installed = prefix / "share/dlsslop-amd/HIP/gfx1201";
    if (std::filesystem::is_directory(installed, error)) return installed.string();
    // A worker run directly from the source tree's build directory uses the
    // same modules as the kernel build script, without an installed launcher.
    const auto development = prefix / "assets/HIP/gfx1201";
    if (std::filesystem::is_directory(development, error)) return development.string();
    return installed.string();
}
} // namespace dlsslop
