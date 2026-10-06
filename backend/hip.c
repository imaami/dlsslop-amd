/** @file
 *
 * The HIP runtime, loaded with dlopen(), and code objects loaded from files: hip.h.
 */
// SPDX-License-Identifier: MIT
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "files.h"
#include "hip.h"

/** @brief Each entry point's place in struct hip_api. */
enum entry : size_t {
#define INDEX(name, required) ENTRY_##name,
	HIP_API_ENTRIES(INDEX)
#undef INDEX
	ENTRY_COUNT //!< The number of entry points.
};

// HIP_API_ENTRIES lists every member of struct hip_api, once and in order.
#define AT(name, required) \
	static_assert(offsetof(struct hip_api, name) == ENTRY_##name * sizeof (void *), \
	              #name " in struct hip_api's order");
HIP_API_ENTRIES(AT)
#undef AT
static_assert(sizeof (struct hip_api) == ENTRY_COUNT * sizeof (void *), "a pointer per entry point");

/** @brief Each entry point's name, in struct hip_api's order. */
static char const *const entry_names[] = {
#define NAME(name, required) #name,
	HIP_API_ENTRIES(NAME)
#undef NAME
};

/** @brief Whether the runtime must export each entry point, in struct hip_api's order. */
static bool const entry_required[] = {
#define REQUIRED(name, required) required,
	HIP_API_ENTRIES(REQUIRED)
#undef REQUIRED
};

enum error_code
hip_load (struct hip_api *dest,
          struct error   *e)
{
	static char const *const libraries[] = {
		"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so.7",
		"/opt/rocm/lib/libamdhip64.so.6", "/opt/rocm/lib/libamdhip64.so",
	};
	// A set DLSSLOP_HIP_LIBRARY is the only candidate; an empty one is unset.
	char const *const library = getenv("DLSSLOP_HIP_LIBRARY");
	char const *const *candidates = libraries;
	size_t count = sizeof libraries / sizeof *libraries;
	if (library && *library) {
		candidates = &library;
		count = 1;
	}

	// Each candidate's error, so that a runtime which is installed but cannot load is not blamed
	// on an absent one. A list that does not fit is cut, and so are the words that end in it.
	char errors[ERROR_WHAT_BYTES];
	errors[0] = '\0';
	size_t used = 0;
	void *dll = nullptr;
	for (size_t i = 0; i < count; ++i) {
		dll = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
		if (dll)
			break;
		char const *const why = dlerror();
		int const n = snprintf(errors + used, sizeof errors - used, "\n  %s", why ? why : "unknown error");
		if (n > 0)
			used += (size_t)n < sizeof errors - used ? (size_t)n : sizeof errors - 1 - used;
	}
	if (!dll)
		return error_fail(e, "Linux HIP runtime not found; install gfx1201-capable ROCm userspace "
		                  "(libamdhip64), or set DLSSLOP_HIP_LIBRARY:%s", errors);

	struct hip_api api;
	for (size_t i = 0; i < ENTRY_COUNT; ++i) {
		void *const symbol = dlsym(dll, entry_names[i]);
		if (!symbol && entry_required[i]) {
			// Nothing holds the library: struct hip_api keeps only its entry points.
			dlclose(dll);
			dll = nullptr;
			return error_fail(e, "missing HIP export %s", entry_names[i]);
		}
		// POSIX makes the address dlsym() gives for a function a valid pointer to it.
		memcpy((char *)&api + i * sizeof symbol, &symbol, sizeof symbol);
	}
	*dest = api;
	return ERROR_NONE;
}

enum error_code
hip_load_module (struct hip_api const  *api,
                 char const            *path,
                 void                 **module,
                 struct error          *e)
{
	struct files_data image;
	enum error_code const code = files_read(&image, path, e);
	if (code) {
		error_wrap(e, "cannot read module %s: ", path);
		return code;
	}
	if (!image.size) {
		files_data_fini(&image);
		return error_fail(e, "empty module %s", path);
	}

	void *loaded = nullptr;
	int const result = api->hipModuleLoadData(&loaded, image.bytes);
	files_data_fini(&image);
	if (result)
		return hip_fail(e, api, result, "load module %s", path);
	*module = loaded;
	return ERROR_NONE;
}

enum error_code
hip_fail (struct error         *e,
          struct hip_api const *api,
          int                   result,
          char const           *fmt,
          ...)
{
	if (!e)
		return ERROR_FAILED;
	char what[ERROR_WHAT_BYTES];
	va_list args;
	va_start(args, fmt);
	int const n = vsnprintf(what, sizeof what, fmt, args);
	va_end(args);
	return error_fail(e, "%s: %s (%d)", n < 0 ? "a HIP call" : what, api->hipGetErrorName(result), result);
}
