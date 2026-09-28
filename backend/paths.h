// Where dlsslopd finds its model, modules, shaders and settings.
// SPDX-License-Identifier: MIT
#pragma once
#include <string>

namespace dlsslop {
std::string home_directory();
// XDG_BASE (nonempty) or ~/FALLBACK, then NAME; empty without a home.
std::string xdg_path(const char* base, const char* fallback, const char* name);
std::string default_assets();
std::string default_vulkan_model();
std::string default_config();
std::string executable_path();
// The Vulkan network's SPIR-V for a binary installed under PREFIX, or a
// source build's in BUILD, where the binary is.
std::string vulkan_shaders(const std::string& prefix, const std::string& build);
// dlsslopd's: the installed copy, or a source build's beside the executable.
std::string vulkan_shaders();
// The Vulkan network's writable pipeline cache; empty without a home or when
// its directory cannot be made.
std::string vulkan_cache();
std::string default_modules();
} // namespace dlsslop
