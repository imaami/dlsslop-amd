/** @file
 *
 * The in-layer network's module through its C functions, as the layer loads it, without a GPU: a
 * missing module, a library without the module's functions, or a module of another interface, is
 * refused, naming why, and the loader keeps no pointer into it; the module opens no network for a
 * device of another interface; images of another format or of no generation fail the network,
 * naming why; a frame whose settings are out of range is rejected and says which, a missing model
 * fails the network for good, naming the model's path and how to get it, and a frame of an extent
 * the network does not take, by its working extent or by the device's storage buffers, is rejected,
 * naming it, without a build, and so is another shape of that extent, which no network was built for
 * to reshape. The model is the one dlsslopd's config file names, and a config file dlsslopd refuses
 * fails the network. Takes the module's path, that of a library with all its functions but
 * dlsslop_network_close(), and that of a library with all of them and another interface.
 */
// SPDX-License-Identifier: MIT
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "network_module.h"
#include "shm_protocol.h"
#include "support.h"

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("network-module: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief The directory that the model and the pipeline cache resolve under, removed with what it
 *         holds when the test ends.
 */
static char *directory;

/** @brief Removes the test's directory, at exit; a failure ends the test with 1. */
static void
remove_directory (void)
{
	if (!directory)
		return;
	if (!support_remove_tree(directory)) {
		fprintf(stderr, "network-module: cannot remove %s\n", directory);
		_exit(1);
	}
	free(directory);
	directory = nullptr;
}

/** @brief The next layer's vkGetInstanceProcAddr of a device that has no functions. */
static PFN_vkVoidFunction VKAPI_PTR
no_functions (VkInstance  instance,
              char const *name)
{
	return nullptr;
}

/** @brief vkGetPhysicalDeviceProperties2 of a device whose storage buffers hold 1 MiB, less than any
 *         extent's arena.
 */
static void VKAPI_PTR
small_properties (VkPhysicalDevice             physical,
                  VkPhysicalDeviceProperties2 *properties)
{
	properties->properties.limits.maxStorageBufferRange = 1 << 20;
	for (VkBaseOutStructure *s = properties->pNext; s; s = s->pNext)
		if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
			((VkPhysicalDeviceMaintenance3Properties *)s)->maxMemoryAllocationSize = 1 << 20;
}

/** @brief The next layer's vkGetInstanceProcAddr of that device. */
static PFN_vkVoidFunction VKAPI_PTR
small_storage (VkInstance  instance,
               char const *name)
{
	return strcmp(name, "vkGetPhysicalDeviceProperties2") ? nullptr : (PFN_vkVoidFunction)small_properties;
}

/** @brief Images of a raster and a format, of the first generation; no handle: nothing here reaches
 *         Vulkan.
 *
 * @param width  Their width.
 * @param height Their height.
 * @param format Their format.
 * @return       The images.
 */
static struct dlsslop_network_images const *
images (uint32_t width,
        uint32_t height,
        VkFormat format)
{
	static struct dlsslop_network_images i;
	i = (struct dlsslop_network_images){
		.generation = 1,
		.format = format,
		.width = width,
		.height = height,
	};
	return &i;
}

/** @brief Writes a file, which must succeed.
 *
 * @param path The file.
 * @param text What it holds.
 */
static void
write_file (char const *path,
            char const *text)
{
	FILE *const file = fopen(path, "we");
	require(file, "cannot write the config file");
	bool const written = fputs(text, file) >= 0;
	require(!fclose(file) && written, "cannot write the config file");
}

/** @brief The first frame's failure, of a network opened while dlsslopd's config file holds a text.
 *
 * @param module The module.
 * @param header The channel.
 * @param config The config file's path.
 * @param text   What it holds.
 * @return       The failure, which the caller frees; empty if the frame did not fail.
 */
static char *
failure_with (struct network_module const *module,
              struct ShmHeader const      *header,
              char const                  *config,
              char const                  *text)
{
	write_file(config, text);
	struct dlsslop_network_device const device = {
		.interface = DLSSLOP_NETWORK_INTERFACE,
		.physical_dispatch = no_functions,
	};
	struct DlsslopNetwork *network = module->open(&device);
	require(network, "the module did not open");
	bool const failed = module->prepare(network, header, images(1280, 720, VK_FORMAT_R8G8B8A8_UNORM))
	                    == DLSSLOP_NETWORK_FAILED;
	char *const error = strdup(failed ? module->error(network) : "");
	require(error, "out of memory");
	module->close(network);
	network = nullptr;
	return error;
}

int
main (int    argc,
      char **argv)
{
	require(argc == 4, "usage: network-module-test MODULE INCOMPLETE OTHER");
	// A module that is not there, and a library without the module's functions, load nothing and
	// say why; the latter is unloaded again.
	struct network_module absent = {};
	require(!network_module_load(&absent, "/nonexistent/libdlsslop-network.so") && !absent.library
	        && strstr(absent.failure, "/nonexistent/libdlsslop-network.so"),
	        "a missing module was loaded, or its failure does not name it");
	struct network_module foreign = {};
	require(!network_module_load(&foreign, "libm.so.6") && !foreign.library
	        && strstr(foreign.failure, "dlsslop_network_"),
	        "a library without the module's functions was kept, or its failure does not name a function");
	// A library with all but one of the functions is unloaded too, and the loader keeps none of them.
	struct network_module incomplete = {};
	require(!network_module_load(&incomplete, argv[2]) && !incomplete.library
	        && strstr(incomplete.failure, "dlsslop_network_close") && !incomplete.open && !incomplete.prepare
	        && !incomplete.record && !incomplete.submitted && !incomplete.error && !incomplete.close,
	        "a library without one of the module's functions was kept, or pointers into it were");
	// A module of another interface is unloaded before any of its functions is looked up, and its
	// failure names both interfaces.
	struct network_module other = {};
	char interfaces[128];
	int const n = snprintf(interfaces, sizeof interfaces, ": interface %#" PRIx64 ", the layer's is %#" PRIx64,
	                       DLSSLOP_NETWORK_INTERFACE + 0x100, DLSSLOP_NETWORK_INTERFACE);
	require(n >= 0 && n < (int)sizeof interfaces, "the interfaces do not fit");
	require(!network_module_load(&other, argv[3]) && !other.library && strstr(other.failure, argv[3])
	        && strstr(other.failure, interfaces) && !other.open && !other.prepare && !other.record
	        && !other.submitted && !other.error && !other.close,
	        "a module of another interface was kept, or its failure does not name both interfaces");
	require(!network_module_load(nullptr, argv[1]), "a module was loaded into no loader");
	struct network_module module = {};
	bool const loaded = network_module_load(&module, argv[1]);
	require(loaded, "%s", module.failure);
	directory = support_temp_dir("/tmp", "dlsslop-network", nullptr);
	require(directory, "cannot make a temporary directory");
	require(!atexit(remove_directory), "atexit failed");
	require(!setenv("XDG_DATA_HOME", directory, 1) && !setenv("XDG_CACHE_HOME", directory, 1)
	        && !setenv("XDG_CONFIG_HOME", directory, 1), "setenv failed");
	// A device of another interface opens no network: an older layer's begins with its VkInstance.
	struct dlsslop_network_device device = {.physical_dispatch = no_functions};
	require(!module.open(&device), "the module opened a network for a device of another interface");
	// No device: nothing here reaches Vulkan.
	device.interface = DLSSLOP_NETWORK_INTERFACE;
	struct DlsslopNetwork *network = module.open(&device);
	require(network, "the module did not open");
	void *memory = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	require(memory != MAP_FAILED, "cannot map a channel header");
	struct ShmHeader *const header = memory;
	ShmInitNativeDefaults(header, false);

	// Images of another format, or of no generation, fail the network, naming why.
	struct DlsslopNetwork *other_format = module.open(&device);
	require(other_format, "the module did not open");
	require(module.prepare(other_format, header, images(1280, 720, VK_FORMAT_B8G8R8A8_UNORM))
	        == DLSSLOP_NETWORK_FAILED
	        && strstr(module.error(other_format), "not of format 44")
	        && module.prepare(other_format, header, images(1280, 720, VK_FORMAT_R8G8B8A8_UNORM))
	           == DLSSLOP_NETWORK_FAILED,
	        "images of another format did not fail the network for good, naming the format");
	module.close(other_format);
	other_format = nullptr;
	struct DlsslopNetwork *no_generation = module.open(&device);
	require(no_generation, "the module did not open");
	struct dlsslop_network_images none = *images(1280, 720, VK_FORMAT_R8G8B8A8_UNORM);
	none.generation = 0;
	require(module.prepare(no_generation, header, &none) == DLSSLOP_NETWORK_FAILED
	        && strstr(module.error(no_generation), "no generation"),
	        "images of no generation did not fail the network, naming why");
	module.close(no_generation);
	no_generation = nullptr;
	atomic_store(&header->style, 3);
	require(module.prepare(network, header, images(1280, 720, VK_FORMAT_R8G8B8A8_UNORM))
	        == DLSSLOP_NETWORK_REJECTED
	        && strstr(module.error(network), "style 0..2"),
	        "a setting out of range was not rejected, naming it");
	atomic_store(&header->style, 0);
	char *model = support_format(nullptr, "%s/dlsslop-amd/dlssnr.bin", directory);
	require(model, "out of memory");
	for (uint32_t frame = 0; frame < 2; ++frame)
		require(module.prepare(network, header, images(1280, 720, VK_FORMAT_R8G8B8A8_UNORM))
		        == DLSSLOP_NETWORK_FAILED
		        && strstr(module.error(network), model) && strstr(module.error(network), "dlsslop-setup --dll"),
		        "a missing model did not fail the network for good, naming it");
	module.close(network);
	network = nullptr;

	char *config = support_format(nullptr, "%s/dlsslop-amd/dlsslopd.conf", directory);
	char *custom = support_format(nullptr, "%s/custom.bin", directory);
	char *data = support_format(nullptr, "%s/dlsslop-amd", directory);
	require(config && custom && data, "out of memory");
	require(!mkdir(data, 0700) || errno == EEXIST, "cannot make %s", data);
	free(data);
	data = nullptr;
	FILE *const file = fopen(model, "we");
	require(file && !fclose(file), "cannot write a model file");
	network = module.open(&device);
	require(network, "the module did not open");
	for (uint32_t passes = 1; passes <= 2; ++passes) {
		atomic_store(&header->passes, passes);
		require(module.prepare(network, header, images(16, 16, VK_FORMAT_R8G8B8A8_UNORM))
		        == DLSSLOP_NETWORK_REJECTED
		        && strstr(module.error(network), "does not take 16x16 frames"),
		        "a frame of an extent the network does not take, or another shape of it, was not rejected, "
		        "naming it");
	}
	atomic_store(&header->passes, 1);
	module.close(network);
	network = nullptr;
	device.physical_dispatch = small_storage;
	network = module.open(&device);
	require(network, "the module did not open");
	for (uint32_t frame = 0; frame < 2; ++frame)
		require(module.prepare(network, header, images(1280, 720, VK_FORMAT_R8G8B8A8_UNORM))
		        == DLSSLOP_NETWORK_REJECTED
		        && strstr(module.error(network), "does not take 1280x720 frames")
		        && strstr(module.error(network), "exceed the device's storage buffers of 1048576 bytes"),
		        "a frame whose arena exceeds the device's storage buffers was not rejected, naming it");
	module.close(network);
	network = nullptr;
	require(!unlink(model), "cannot remove %s", model);
	char *text = support_format(nullptr, "vulkan-model = %s\n", custom);
	char *want = support_format(nullptr, "no model at %s", custom);
	require(text && want, "out of memory");
	char *failure = failure_with(&module, header, config, text);
	require(strstr(failure, want), "the network looked for another model than dlsslopd's config file names");
	free(failure);
	failure = nullptr;
	failure = failure_with(&module, header, config, "vulkan-model = relative.bin\n");
	require(strstr(failure, config), "a config file dlsslopd refuses did not fail the network, naming it");
	free(failure);
	failure = nullptr;
	free(want);
	want = nullptr;
	free(text);
	text = nullptr;
	require(!munmap(memory, kHeaderBytes), "cannot unmap the channel header");
	memory = nullptr;
	free(custom);
	custom = nullptr;
	free(config);
	config = nullptr;
	free(model);
	model = nullptr;
	if (puts("network module: a missing module, one without the functions or one of another interface is "
	         "refused, naming why, and so is a device of another interface; images of another format or no "
	         "generation fail the network; out-of-range settings rejected, a missing model fails for good, "
	         "extents the network or the device's storage buffers do not take rejected in any shape, and the "
	         "model is the one dlsslopd's config file names") == EOF)
		return 1;
	return 0;
}
