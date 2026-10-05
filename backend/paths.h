/** @file
 *
 * Where dlsslopd and the in-layer network find the model, the HIP modules, the Vulkan network's
 * shaders and pipeline cache, and dlsslopd's settings. A path is a heap string, which the caller
 * frees. paths.c defines the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_BACKEND_PATHS_H_
#define DLSSLOP_AMD_BACKEND_PATHS_H_

#ifdef __cplusplus
# include <cstddef>
#else
# include <stddef.h>
#endif

#include "error.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief The user's home directory, which the default paths below it share. */
struct paths_home {
	char const  *path;   //!< Nonempty HOME, otherwise the account's; nullptr if there is none.
	STD(size_t)  length; //!< Its length; 0 without a home.
};

/** @brief Finds the user's home directory, once for every path that a caller makes below it.
 *
 * @return The home, whose path is not the caller's to free; it holds until the environment
 *         changes or the next getpwuid() call.
 */
extern struct paths_home
paths_home (void);

/** @brief The HIP model weights' default directory: nonempty XDG_DATA_HOME/dlsslop-amd/model,
 *         otherwise ~/.local/share/dlsslop-amd/model.
 *
 * @param dest   Receives the path; nullptr without a home or on a failure.
 * @param length Receives its length; 0 without a path.
 * @param home   The home, from paths_home().
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_default_assets (char                    **dest,
                      STD(size_t)              *length,
                      struct paths_home const  *home,
                      struct error             *e);

/** @brief The Vulkan network's default model: nonempty XDG_DATA_HOME/dlsslop-amd/dlssnr.bin,
 *         otherwise ~/.local/share/dlsslop-amd/dlssnr.bin.
 *
 * @param dest   Receives the path; nullptr without a home or on a failure.
 * @param length Receives its length; 0 without a path.
 * @param home   The home, from paths_home().
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_default_vulkan_model (char                    **dest,
                            STD(size_t)              *length,
                            struct paths_home const  *home,
                            struct error             *e);

/** @brief dlsslopd's default config file: nonempty XDG_CONFIG_HOME/dlsslop-amd/dlsslopd.conf,
 *         otherwise ~/.config/dlsslop-amd/dlsslopd.conf.
 *
 * @param dest   Receives the path; nullptr without a home or on a failure.
 * @param length Receives its length; 0 without a path.
 * @param home   The home, from paths_home().
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_default_config (char                    **dest,
                      STD(size_t)              *length,
                      struct paths_home const  *home,
                      struct error             *e);

/** @brief The running binary's path, which its default directories are found from.
 *
 * @param dest   Receives the path, which the caller frees; nullptr if it cannot be read or on a
 *               failure.
 * @param length Receives its length; 0 without a path.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_executable (char         **dest,
                  STD(size_t)   *length,
                  struct error  *e);

/** @brief The native gfx1201 HIP modules' default directory.
 *
 * Nonempty DLSSLOP_MODULES; otherwise share/dlsslop-amd/HIP/gfx1201 below the prefix the binary is
 * installed in, or the source tree's assets/HIP/gfx1201 for a binary in its build directory, which
 * the kernel build script fills.
 *
 * @param dest          Receives the path; nullptr if the binary's own path is unknown or on a
 *                      failure.
 * @param length        Receives its length; 0 without a path.
 * @param binary        The binary's path, from paths_executable(), or nullptr if it is unknown.
 * @param binary_length Its length.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_default_modules (char         **dest,
                       STD(size_t)   *length,
                       char const    *binary,
                       STD(size_t)    binary_length,
                       struct error  *e);

/** @brief The Vulkan network's SPIR-V for a binary installed under a prefix, or a source build's in
 *         its build directory: the installed one, unless only the build's is there.
 *
 * @param dest          Receives the path, prefix/share/dlsslop-amd/vulkan or
 *                      build/vulkan-nr/network; nullptr on a failure.
 * @param length        Receives its length.
 * @param prefix        The prefix; it need not be null-terminated.
 * @param prefix_length Its length.
 * @param build         The build directory; it need not be null-terminated.
 * @param build_length  Its length.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_vulkan_shaders_at (char         **dest,
                         STD(size_t)   *length,
                         char const    *prefix,
                         STD(size_t)    prefix_length,
                         char const    *build,
                         STD(size_t)    build_length,
                         struct error  *e);

/** @brief dlsslopd's Vulkan network SPIR-V: paths_vulkan_shaders_at() of the prefix its binary is
 *         installed in and the directory the binary is in.
 *
 * @param dest          Receives the path; nullptr on a failure.
 * @param length        Receives its length.
 * @param binary        The binary's path, from paths_executable(), or nullptr if it is unknown:
 *                      then the path is relative to the working directory.
 * @param binary_length Its length.
 * @param e             Receives the words for what stopped it, or nullptr.
 * @return              ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_vulkan_shaders (char         **dest,
                      STD(size_t)   *length,
                      char const    *binary,
                      STD(size_t)    binary_length,
                      struct error  *e);

/** @brief The Vulkan network's writable pipeline cache: nonempty
 *         XDG_CACHE_HOME/dlsslop-amd/vulkan-pipelines.cache, otherwise
 *         ~/.cache/dlsslop-amd/vulkan-pipelines.cache, whose directory is created.
 *
 * @param dest   Receives the path; nullptr without a home, when its directory cannot be made or
 *               on a failure.
 * @param length Receives its length; 0 without a path.
 * @param home   The home, from paths_home().
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED without memory.
 */
extern enum error_code
paths_vulkan_cache (char                    **dest,
                    STD(size_t)              *length,
                    struct paths_home const  *home,
                    struct error             *e);

/** @brief Checks that the Vulkan network's model is there, and says how to get one if not.
 *
 * @param model The model's path, or nullptr for none.
 * @param e     Receives the words for what is missing, or nullptr.
 * @return      ERROR_NONE if @a model is a regular file, otherwise ERROR_FAILED.
 */
extern enum error_code
paths_require_vulkan_model (char const   *model,
                            struct error *e);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_BACKEND_PATHS_H_ */
