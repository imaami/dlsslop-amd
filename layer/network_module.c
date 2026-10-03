/** @file
 *
 * The layer's loader of the in-layer network's module.
 */
// SPDX-License-Identifier: MIT
#include <dlfcn.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "network_module.h"

/** @brief Records why network_module_load() failed, before anything else can reset dlerror().
 *
 * @param m    The module.
 * @param path The module's path, the reason when dlerror() has none.
 * @return     false.
 */
static bool
network_module_refuse (struct network_module *m,
                       char const            *path)
{
	char const *const why = dlerror();
	snprintf(m->failure, sizeof m->failure, "%s", why ? why : path);
	return false;
}

/** @brief Records why network_module_load() failed, as network_module_refuse() does, and closes the
 *         library it opened.
 *
 * @param m       The module.
 * @param library The library.
 * @param path    The library's path.
 * @return        false.
 */
static bool
network_module_drop (struct network_module *m,
                     void                  *library,
                     char const            *path)
{
	network_module_refuse(m, path);
	dlclose(library);
	return false;
}

bool
network_module_load (struct network_module *m,
                     char const            *path)
{
	if (!m)
		return false;
	// Looked up in a local first: a refused module leaves no pointer into it in m.
	struct network_module found = {.library = dlopen(path, RTLD_NOW | RTLD_LOCAL)};
	if (!found.library)
		return network_module_refuse(m, path);
	void const *const exported = dlsym(found.library, "dlsslop_network_interface");
	if (!exported)
		return network_module_drop(m, found.library, path);
	uint64_t const interface = *(uint64_t const *)exported;
	if (interface != DLSSLOP_NETWORK_INTERFACE) {
		snprintf(m->failure, sizeof m->failure, "%s: interface %#" PRIx64 ", the layer's is %#" PRIx64,
		         path, interface, DLSSLOP_NETWORK_INTERFACE);
		dlclose(found.library);
		return false;
	}
	// The lookups stop at the first that fails, whose dlerror() a later lookup that succeeds would
	// clear.
#define NETWORK_MODULE_FIND(name) \
	(found.name = (dlsslop_network_##name##_fn *)dlsym(found.library, "dlsslop_network_" #name))
	if (NETWORK_MODULE_FIND(open) && NETWORK_MODULE_FIND(prepare) && NETWORK_MODULE_FIND(record)
	    && NETWORK_MODULE_FIND(submitted) && NETWORK_MODULE_FIND(error) && NETWORK_MODULE_FIND(close)) {
		*m = found;
		return true;
	}
#undef NETWORK_MODULE_FIND
	return network_module_drop(m, found.library, path);
}
