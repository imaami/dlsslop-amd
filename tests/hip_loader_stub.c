/** @file
 *
 * HIP runtime stand-ins for the worker's loader and device-selection tests.
 * libhip-loader-stub.so needs libhip-loader-stub-dependency.so, which the test leaves unreachable.
 * libhip-fake-runtime.so loads and reports one device per architecture that HIP_FAKE_ARCHS lists
 * (comma separated; unset or empty: none); every call the device selection does not make fails.
 * libhip-fake-incomplete.so is the same without hipModuleUnload.
 */
// SPDX-License-Identifier: MIT
#if defined(HIP_LOADER_STUB_DEPENDENCY)

/** @brief Nothing: the stub's dependency. */
int
hip_loader_stub_dependency (void)
{
	return 0;
}

#elif defined(HIP_FAKE_RUNTIME)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../backend/hip.h"

/** @brief The architectures, as HIP_FAKE_ARCHS lists them.
 *
 * @return The list; empty when the variable is unset.
 */
static char const *
archs (void)
{
	char const *const text = getenv("HIP_FAKE_ARCHS");
	return text ? text : "";
}

/** @brief The number of architectures in a list: one before each comma, and the last one unless it
 *         is empty.
 *
 * @param list The list.
 * @return     The number.
 */
static int
arch_count (char const *list)
{
	int count = 0;
	char const *comma;
	while ((comma = strchr(list, ',')) != nullptr) {
		++count;
		list = comma + 1;
	}
	return count + (*list != '\0');
}

int
hipInit (unsigned flags)
{
	return 0;
}

int
hipRuntimeGetVersion (int *version)
{
	*version = 60443000;
	return 0;
}

int
hipGetDeviceCount (int *count)
{
	*count = arch_count(archs());
	return 0;
}

int
hipGetDevicePropertiesR0600 (struct hip_device_properties *p,
                             int                           device)
{
	if (device < 0)
		return 1;
	char const *arch = archs();
	for (int i = 0; i < device; ++i) {
		char const *const comma = strchr(arch, ',');
		if (!comma)
			return 1;
		arch = comma + 1;
	}
	// An empty last architecture is no device, as arch_count() counts.
	if (!*arch)
		return 1;
	int const length = (int)strcspn(arch, ",");
	snprintf(p->name, sizeof p->name, "Fake device %d", device);
	snprintf(p->gcnArchName, sizeof p->gcnArchName, "%.*s", length, arch);
	return 0;
}

char const *
hipGetErrorName (int result)
{
	return "hipErrorInvalidValue";
}

/** @brief Defines an entry point that dlsslopd requires when it loads the runtime, which fails. */
#define HIP_FAKE_FAILS(name) \
	int \
	name (void) \
	{ \
		return 1; \
	}

HIP_FAKE_FAILS(hipModuleLoadData)
HIP_FAKE_FAILS(hipEventCreate)
HIP_FAKE_FAILS(hipEventRecord)
HIP_FAKE_FAILS(hipEventElapsedTime)
HIP_FAKE_FAILS(hipEventDestroy)
HIP_FAKE_FAILS(hipEventSynchronize)
HIP_FAKE_FAILS(hipHostMalloc)
HIP_FAKE_FAILS(hipSetDevice)
HIP_FAKE_FAILS(hipMemGetInfo)
HIP_FAKE_FAILS(hipMalloc)
HIP_FAKE_FAILS(hipFree)
HIP_FAKE_FAILS(hipMemcpy)
HIP_FAKE_FAILS(hipMemcpyAsync)
HIP_FAKE_FAILS(hipMemsetAsync)
HIP_FAKE_FAILS(hipStreamCreate)
HIP_FAKE_FAILS(hipStreamSynchronize)
HIP_FAKE_FAILS(hipStreamDestroy)
HIP_FAKE_FAILS(hipImportExternalMemory)
HIP_FAKE_FAILS(hipExternalMemoryGetMappedBuffer)
HIP_FAKE_FAILS(hipDestroyExternalMemory)
HIP_FAKE_FAILS(hipModuleGetFunction)
HIP_FAKE_FAILS(hipModuleLaunchKernel)
#if !defined(HIP_FAKE_INCOMPLETE)
HIP_FAKE_FAILS(hipModuleUnload)
#endif

#undef HIP_FAKE_FAILS

#else

extern int
hip_loader_stub_dependency (void);

/** @brief Calls the dependency, which the stub cannot load without. */
int
hip_loader_stub (void)
{
	return hip_loader_stub_dependency();
}

#endif
