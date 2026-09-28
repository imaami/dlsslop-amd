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
// The Vulkan network's SPIR-V: the installed copy, or a source build's beside
// the executable.
std::string vulkan_shaders();
std::string default_modules();
} // namespace dlsslop
