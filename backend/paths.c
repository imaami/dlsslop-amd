/** @file
 *
 * Where dlsslopd and the in-layer network find their files: paths.h.
 */
// SPDX-License-Identifier: MIT
#include <pwd.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "files.h"
#include "paths.h"

/** @brief Gives the caller no path.
 *
 * @param dest   Receives nullptr.
 * @param length Receives 0.
 * @return       ERROR_NONE.
 */
static enum error_code
none (char   **dest,
      size_t  *length)
{
	*dest = nullptr;
	*length = 0;
	return ERROR_NONE;
}

/** @brief Gives the caller no path, as there was no memory for one.
 *
 * @param dest   Receives nullptr.
 * @param length Receives 0.
 * @param e      Receives the words, or nullptr.
 * @return       ERROR_FAILED.
 */
[[gnu::cold]]
static enum error_code
no_memory (char         **dest,
           size_t        *length,
           struct error  *e)
{
	none(dest, length);
	return error_fail(e, "out of memory");
}

struct paths_home
paths_home (void)
{
	char const *path = getenv("HOME");
	if (!path || !*path) {
		struct passwd const *const user = getpwuid(getuid());
		if (!user || !user->pw_dir || !*user->pw_dir)
			return (struct paths_home){};
		path = user->pw_dir;
	}
	return (struct paths_home){path, strlen(path)};
}

/** @brief A path below an XDG base directory: nonempty $BASE/NAME, otherwise ~/FALLBACK/NAME.
 *
 * @param dest              Receives the path; nullptr without a home or on a failure.
 * @param length            Receives its length; 0 without a path.
 * @param base              The base directory's environment variable.
 * @param name              NAME.
 * @param name_length       Its length.
 * @param below_home        FALLBACK/NAME.
 * @param below_home_length Its length.
 * @param home              The home, for ~/FALLBACK/NAME.
 * @param e                 Receives the words for what stopped it, or nullptr.
 * @return                  ERROR_NONE, or ERROR_FAILED without memory.
 */
static enum error_code
xdg (char                    **dest,
     size_t                   *length,
     char const               *base,
     char const               *name,
     size_t                    name_length,
     char const               *below_home,
     size_t                    below_home_length,
     struct paths_home const  *home,
     struct error             *e)
{
	char const *const root = getenv(base);
	if (root && *root)
		*dest = files_join(root, strlen(root), name, name_length, length);
	else if (home->path)
		*dest = files_join(home->path, home->length, below_home, below_home_length, length);
	else
		return none(dest, length);
	return *dest ? ERROR_NONE : no_memory(dest, length, e);
}

/** @brief xdg() of the string literals FALLBACK and NAME. */
#define XDG(dest, length, base, fallback, name, home, e) \
	xdg(dest, length, base, name, sizeof name - 1, fallback "/" name, sizeof fallback "/" name - 1, home, e)

enum error_code
paths_default_assets (char                    **dest,
                      size_t                   *length,
                      struct paths_home const  *home,
                      struct error             *e)
{
	return XDG(dest, length, "XDG_DATA_HOME", ".local/share", "dlsslop-amd/model", home, e);
}

enum error_code
paths_default_vulkan_model (char                    **dest,
                            size_t                   *length,
                            struct paths_home const  *home,
                            struct error             *e)
{
	return XDG(dest, length, "XDG_DATA_HOME", ".local/share", "dlsslop-amd/dlssnr.bin", home, e);
}

enum error_code
paths_default_config (char                    **dest,
                      size_t                   *length,
                      struct paths_home const  *home,
                      struct error             *e)
{
	return XDG(dest, length, "XDG_CONFIG_HOME", ".config", "dlsslop-amd/dlsslopd.conf", home, e);
}

enum error_code
paths_vulkan_cache (char                    **dest,
                    size_t                   *length,
                    struct paths_home const  *home,
                    struct error             *e)
{
	enum error_code const code = XDG(dest, length, "XDG_CACHE_HOME", ".cache",
	                                 "dlsslop-amd/vulkan-pipelines.cache", home, e);
	if (code || !*dest)
		return code;

	// The cache is a convenience: one whose directory cannot be made is none.
	bool created;
	if (files_make_directories(*dest, files_parent(*dest, *length), &created, nullptr)) {
		free(*dest);
		*dest = nullptr;
		return none(dest, length);
	}
	return ERROR_NONE;
}

#undef XDG

enum error_code
paths_executable (char         **dest,
                  size_t        *length,
                  struct error  *e)
{
	// A path that fills the buffer may have been cut: the next buffer is twice as large.
	for (size_t size = 256;; size *= 2) {
		char *path = malloc(size);
		if (!path)
			return no_memory(dest, length, e);
		ssize_t const n = readlink("/proc/self/exe", path, size);
		if (n <= 0) {
			free(path);
			path = nullptr;
			return none(dest, length);
		}
		size_t const used = (size_t)n;
		if (used < size) {
			path[used] = '\0';
			*dest = path;
			*length = used;
			return ERROR_NONE;
		}
		free(path);
		path = nullptr;
	}
}

/** @brief Of an installed directory and a source build's, keeps the installed one unless only the
 *         build's is there, and frees the other.
 *
 * @param dest         Holds the installed directory; receives the one kept.
 * @param length       Holds its length; receives the kept one's.
 * @param built        The build's directory.
 * @param built_length Its length.
 */
static void
prefer_installed (char   **dest,
                  size_t  *length,
                  char    *built,
                  size_t   built_length)
{
	if (files_is_directory(built) && !files_is_directory(*dest)) {
		free(*dest);
		*dest = built;
		*length = built_length;
	} else {
		free(built);
		built = nullptr;
	}
}

enum error_code
paths_default_modules (char         **dest,
                       size_t        *length,
                       char const    *binary,
                       size_t         binary_length,
                       struct error  *e)
{
	char const *const variable = getenv("DLSSLOP_MODULES");
	if (variable && *variable) {
		size_t const size = strlen(variable) + 1;
		*dest = malloc(size);
		if (!*dest)
			return no_memory(dest, length, e);
		memcpy(*dest, variable, size);
		*length = size - 1;
		return ERROR_NONE;
	}

	if (!binary)
		return none(dest, length);

	// The installed modules, or for a daemon run from the source tree's build directory, without
	// an installed launcher, those that the kernel build script made there.
	static char const installed[] = "share/dlsslop-amd/HIP/gfx1201";
	static char const development[] = "assets/HIP/gfx1201";
	size_t const prefix = files_parent(binary, files_parent(binary, binary_length));
	*dest = files_join(binary, prefix, installed, sizeof installed - 1, length);
	size_t built_length;
	char *built = files_join(binary, prefix, development, sizeof development - 1, &built_length);
	if (!*dest || !built) {
		free(built);
		built = nullptr;
		free(*dest);
		*dest = nullptr;
		return no_memory(dest, length, e);
	}
	prefer_installed(dest, length, built, built_length);
	return ERROR_NONE;
}

enum error_code
paths_vulkan_shaders_at (char         **dest,
                         size_t        *length,
                         char const    *prefix,
                         size_t         prefix_length,
                         char const    *build,
                         size_t         build_length,
                         struct error  *e)
{
	static char const installed[] = "share/dlsslop-amd/vulkan";
	static char const development[] = "vulkan-nr/network";
	*dest = files_join(prefix, prefix_length, installed, sizeof installed - 1, length);
	size_t built_length;
	char *built = files_join(build, build_length, development, sizeof development - 1, &built_length);
	if (!*dest || !built) {
		free(built);
		built = nullptr;
		free(*dest);
		*dest = nullptr;
		return no_memory(dest, length, e);
	}
	prefer_installed(dest, length, built, built_length);
	return ERROR_NONE;
}

enum error_code
paths_vulkan_shaders (char         **dest,
                      size_t        *length,
                      char const    *binary,
                      size_t         binary_length,
                      struct error  *e)
{
	// The binary's directory and the prefix above it; without the binary's path, both are empty,
	// and the shaders' paths are relative to the working directory.
	char const *const path = binary ? binary : "";
	size_t const directory = files_parent(path, binary_length);
	return paths_vulkan_shaders_at(dest, length, path, files_parent(path, directory), path, directory, e);
}

enum error_code
paths_require_vulkan_model (char const   *model,
                            struct error *e)
{
	if (model && files_is_regular_file(model))
		return ERROR_NONE;
	return error_fail(e, "no model at %s (dlsslop-setup --dll extracts it from nvngx_dlssnr.dll 310.8.0)",
	                  model ? model : "");
}
