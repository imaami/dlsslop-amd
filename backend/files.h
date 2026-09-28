// Paths and whole files through POSIX calls, without std::filesystem or iostreams.
// SPDX-License-Identifier: MIT
#pragma once
#include "result.h"

#include <string>
#include <string_view>

namespace dlsslop {

bool is_directory(const std::string& path);
bool is_regular_file(const std::string& path);
// PATH without its last component: empty for a bare name, "/" for one at the root.
std::string parent_path(std::string_view path);
// A and B joined by one slash; B alone when A is empty.
std::string join(std::string_view a, std::string_view b);
// The whole file, or strerror of what stopped it.
Result<std::string> read_file(const std::string& path);

} // namespace dlsslop
