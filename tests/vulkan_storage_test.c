/** @file
 *
 * What the layer asks of a device, on the host: the shaders' storage images, which must be formatless
 * and only written, and the device features that the layer enables for them and for the in-layer
 * network, in the game's chain or in private copies of it; the network's requirements of a device
 * that stubs answer for; and DLSSLOP_LAYER_NETWORK.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "device_features.h"

// The SPIR-V that the layer loads.
#include "dlssnr/DlssNr_Shader_Vk.h"
#include "scaling/bcds_bicubic_Shader_Vk.h"
#include "scaling/bcds_catmull_Shader_Vk.h"
#include "scaling/bcds_kaiser2_Shader_Vk.h"
#include "scaling/bcds_kaiser3_Shader_Vk.h"
#include "scaling/bcds_lanczos2_Shader_Vk.h"
#include "scaling/bcds_lanczos3_Shader_Vk.h"
#include "scaling/bcds_magc_Shader_Vk.h"
#include "scaling/bcus_Shader_Vk.h"

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
check (bool        condition,
       char const *fmt,
       ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief Checks a shader as the layer loads it, which enables only
 *         shaderStorageImageWriteWithoutFormat. Every storage image must be a formatless 2D image
 *         that is only written: a storage read or atomic would need
 *         shaderStorageImageReadWithoutFormat or a typed format too.
 *
 * @param blob  The SPIR-V.
 * @param bytes Its size.
 * @param name  The shader's name.
 */
static void
shader (unsigned char const *blob,
        size_t               bytes,
        char const          *name)
{
	enum {
		OP_CAPABILITY          = 17,
		OP_TYPE_IMAGE          = 25,
		OP_IMAGE_TEXEL_POINTER = 60,
		OP_IMAGE_READ          = 98,
		OP_IMAGE_WRITE         = 99,
		OP_IMAGE_SPARSE_READ   = 320,
		READ_WITHOUT_FORMAT    = 55,
		WRITE_WITHOUT_FORMAT   = 56
	};
	check(bytes >= 5 * sizeof (uint32_t) && bytes % sizeof (uint32_t) == 0, "invalid SPIR-V byte count");
	size_t const count = bytes / sizeof (uint32_t);
	uint32_t *words = malloc(bytes);
	check(words, "out of memory");
	memcpy(words, blob, bytes);
	check(words[0] == 0x07230203 && words[4] == 0, "invalid SPIR-V header");
	uint32_t write_capabilities = 0;
	uint32_t storage = 0;
	uint32_t writes = 0;
	for (size_t i = 5; i < count; i += words[i] >> 16) {
		uint32_t const length = words[i] >> 16;
		uint32_t const op = words[i] & 0xffffu;
		check(length && length <= count - i, "truncated SPIR-V instruction");
		if (op == OP_CAPABILITY) {
			check(words[i + 1] != READ_WITHOUT_FORMAT, "storage reads without format are not enabled");
			write_capabilities += words[i + 1] == WRITE_WITHOUT_FORMAT;
		} else if (op == OP_TYPE_IMAGE) {
			check(length >= 9, "invalid OpTypeImage");
			if (words[i + 7] != 2)
				continue;
			// Dim2D, depth absent or unspecified, not arrayed or multisampled.
			check(words[i + 3] == 1 && words[i + 4] != 1 && words[i + 4] <= 2 && !words[i + 5]
			      && !words[i + 6], "unexpected storage image type");
			check(words[i + 8] == 0, "typed storage image");
			++storage;
		} else {
			check(op != OP_IMAGE_READ && op != OP_IMAGE_SPARSE_READ && op != OP_IMAGE_TEXEL_POINTER,
			      "storage read or atomic");
			writes += op == OP_IMAGE_WRITE;
		}
	}
	free(words);
	words = nullptr;
	check(write_capabilities == 1, "missing StorageImageWriteWithoutFormat capability");
	check(storage && writes, "expected a write-only storage image shader");
	printf("%s: write-only formatless storage verified\n", name);
}

/** @brief Formatless storage writes, in pEnabledFeatures and in a VkPhysicalDeviceFeatures2 chain. */
static void
features (void)
{
	VkPhysicalDeviceFeatures original = {.robustBufferAccess = VK_TRUE};
	VkDeviceCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pEnabledFeatures = &original,
	};
	static struct device_features legacy;
	check(device_features_enable(&legacy, &info, false), "cannot enable legacy features");
	check(info.pEnabledFeatures != &original && info.pEnabledFeatures->robustBufferAccess
	      && device_features_has_formatless_storage_writes(&info) && !original.shaderStorageImageWriteWithoutFormat,
	      "legacy features were not preserved/copied");

	VkBaseOutStructure unknown = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO};
	VkPhysicalDeviceFeatures2 core = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &unknown,
		.features.robustBufferAccess = VK_TRUE,
	};
	VkPhysicalDeviceVulkan12Features vulkan12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &core,
		.timelineSemaphore = VK_TRUE,
	};
	VkLayerDeviceCreateInfo loader = {
		.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO,
		.pNext = &vulkan12,
		.function = VK_LOADER_DATA_CALLBACK,
	};
	info.pNext = &loader;
	info.pEnabledFeatures = nullptr;
	static struct device_features chained;
	check(device_features_enable(&chained, &info, false), "cannot enable known Features2 chain");
	VkLayerDeviceCreateInfo const *const first = info.pNext;
	VkPhysicalDeviceVulkan12Features const *const second = first->pNext;
	VkPhysicalDeviceFeatures2 const *const third = second->pNext;
	check(first != &loader && second != &vulkan12 && third != &core && third->pNext == &unknown,
	      "incorrect Features2 prefix/suffix ownership");
	check(first->function == loader.function && second->timelineSemaphore && third->features.robustBufferAccess
	      && device_features_has_formatless_storage_writes(&info),
	      "caller settings lost in prefix copy");
	check(loader.pNext == &vulkan12 && vulkan12.pNext == &core && core.pNext == &unknown
	      && !core.features.shaderStorageImageWriteWithoutFormat, "caller chain was mutated");

	unknown.pNext = (VkBaseOutStructure *)&core;
	core.pNext = nullptr;
	info.pNext = &unknown;
	static struct device_features unsupported;
	check(!device_features_enable(&unsupported, &info, false) && info.pNext == &unknown
	      && !core.features.shaderStorageImageWriteWithoutFormat, "unknown prefix was not rejected intact");
	core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	check(device_features_enable(&unsupported, &info, false) && info.pNext == &unknown,
	      "already enabled unknown prefix was rejected");
	// Without a VkPhysicalDeviceFeatures2: pEnabledFeatures that enable formatless storage writes are
	// kept, and a request without pEnabledFeatures is given a copy of its own.
	VkPhysicalDeviceFeatures const enough = {.shaderStorageImageWriteWithoutFormat = VK_TRUE};
	VkDeviceCreateInfo kept = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pEnabledFeatures = &enough};
	static struct device_features unchanged;
	check(device_features_enable(&unchanged, &kept, false) && kept.pEnabledFeatures == &enough && !kept.pNext,
	      "pEnabledFeatures that enable formatless storage writes were replaced");
	VkDeviceCreateInfo bare = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	static struct device_features added;
	check(device_features_enable(&added, &bare, false) && bare.pEnabledFeatures == &added.legacy && !bare.pNext
	      && device_features_has_formatless_storage_writes(&bare),
	      "a request without pEnabledFeatures was not given formatless storage writes");
	puts("device features: private legacy/Features2 copies and unknown-prefix fallback verified");
}

/** @brief Whether vkCreateDevice accepts a chain: no structure twice, and no core
 *         VkPhysicalDeviceVulkan1xFeatures beside a structure it replaces.
 *
 * @param info The request.
 * @return     true if it does.
 */
static bool
valid (VkDeviceCreateInfo const *info)
{
	for (void const *node = info->pNext; node; node = vk_chain_next(node))
		for (void const *earlier = info->pNext; earlier != node; earlier = vk_chain_next(earlier))
			if (vk_chain_type(earlier) == vk_chain_type(node))
				return false;
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		struct network_feature const *const f = &NETWORK_FEATURES[i];
		if (f->core != NETWORK_FEATURE_NO_CORE && vk_chain_find(info->pNext, f->core)
		    && vk_chain_find(info->pNext, f->type))
			return false;
	}
	return true;
}

/** @brief Whether a chain enables every network feature where it enables it.
 *
 * @param info The request.
 * @return     true if it does.
 */
static bool
all_network_bits (VkDeviceCreateInfo const *info)
{
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		struct device_features_place const place = device_features_network_feature_in(info->pNext,
		                                                                              &NETWORK_FEATURES[i]);
		if (!place.node || !*vk_chain_bit(place.node, place.offset))
			return false;
	}
	return true;
}

/** @brief The network's extensions, in table order: explicit workgroup layout last.
 *
 * @param names Receives them.
 * @return      Their number.
 */
static uint32_t
network_extensions (char const *names[static NETWORK_FEATURE_COUNT])
{
	uint32_t count = 0;
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		char const *const extension = NETWORK_FEATURES[i].extension;
		if (extension && (!count || strcmp(names[count - 1], extension)))
			names[count++] = extension;
	}
	return count;
}

/** @brief The network's features in a game's chains: set in the game's structures or added, never
 *         twice, the game's chain unwritten, uncopyable prefixes declined, and the ledger.
 */
static void
network_features (void)
{
	// A DXVK-like chain: the core structures, one bit already on, an unknown suffix.
	VkBaseOutStructure unknown = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO};
	VkPhysicalDeviceVulkan13Features v13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &unknown,
		.dynamicRendering = VK_TRUE,
	};
	VkPhysicalDeviceVulkan12Features v12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &v13,
		.vulkanMemoryModel = VK_TRUE,
	};
	VkPhysicalDeviceVulkan11Features v11 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = &v12,
	};
	VkPhysicalDeviceFeatures2 core = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &v11,
		.features.robustBufferAccess = VK_TRUE,
	};
	VkLayerDeviceCreateInfo loader = {
		.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO,
		.pNext = &core,
		.function = VK_LAYER_LINK_INFO,
	};
	VkDeviceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &loader};
	unsigned char loader_bytes[sizeof loader];
	memcpy(loader_bytes, &loader, sizeof loader);
	VkBool32 const before[] = {core.features.shaderStorageImageWriteWithoutFormat, v11.storageBuffer16BitAccess,
	                           v12.shaderInt8, v13.subgroupSizeControl};
	static struct device_features request;
	check(device_features_enable(&request, &info, true), "cannot add the network to a core-structure chain");
	check(valid(&info) && all_network_bits(&info) && device_features_has_formatless_storage_writes(&info),
	      "network features missing from a core-structure chain");
	check(vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO) == &unknown,
	      "the unknown suffix was not left shared");
	check(!vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES),
	      "a core feature was given its own structure beside Vulkan12Features");
	check(vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
	      && vk_chain_find(info.pNext,
	                       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR),
	      "extension structures were not added");
	VkPhysicalDeviceVulkan13Features const *const copied13 =
		vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
	VkPhysicalDeviceFeatures2 const *const copied2 = device_features_core_features2(&info);
	check(copied13 != &v13 && copied13->dynamicRendering && copied2 != &core
	      && copied2->features.robustBufferAccess, "the game's own settings were lost in the copies");
	VkBool32 const after[] = {core.features.shaderStorageImageWriteWithoutFormat, v11.storageBuffer16BitAccess,
	                          v12.shaderInt8, v13.subgroupSizeControl};
	check(!memcmp(before, after, sizeof before) && !memcmp(loader_bytes, &loader, sizeof loader)
	      && v12.vulkanMemoryModel && core.pNext == &v11 && v13.pNext == &unknown,
	      "the game's chain was written");

	// The ledger reads what vkCreateDevice accepted: the extensions too.
	check(!device_features_network_enabled(&info), "ledger ignored the missing extensions");
	char const *extensions[NETWORK_FEATURE_COUNT];
	info.enabledExtensionCount = network_extensions(extensions);
	info.ppEnabledExtensionNames = extensions;
	check(device_features_network_enabled(&info), "ledger missed an enabled network");
	// The network requires explicit workgroup layout too (network_requirements.h).
	--info.enabledExtensionCount;
	check(!device_features_network_enabled(&info), "ledger ignored the missing explicit workgroup layout");

	// Structures of their own: one bit set in the game's, the rest added, none twice.
	VkPhysicalDeviceShaderFloat16Int8Features float16 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
		.shaderFloat16 = VK_TRUE,
	};
	VkPhysicalDeviceFeatures const legacy = {};
	VkDeviceCreateInfo own = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &float16,
		.pEnabledFeatures = &legacy,
	};
	static struct device_features separate;
	check(device_features_enable(&separate, &own, true) && valid(&own) && all_network_bits(&own)
	      && device_features_has_formatless_storage_writes(&own) && own.pEnabledFeatures != &legacy,
	      "cannot add the network beside the game's own structures");
	check(!float16.shaderInt8 && !legacy.shaderStorageImageWriteWithoutFormat
	      && vk_chain_find(own.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) != &float16,
	      "the game's structures were written");

	// A structure to change behind one that cannot be copied declines, untouched.
	VkPhysicalDeviceVulkan12Features late = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkBaseOutStructure opaque = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = (VkBaseOutStructure *)&late};
	VkDeviceCreateInfo hidden = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &opaque};
	static struct device_features declined;
	check(!device_features_enable(&declined, &hidden, true) && hidden.pNext == &opaque && !late.shaderInt8
	      && !hidden.pEnabledFeatures, "an uncopyable prefix was not declined intact");
	// With nothing to change behind it, the network's own structures go in front.
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i)
		if (NETWORK_FEATURES[i].core == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
			*vk_chain_bit(&late, NETWORK_FEATURES[i].core_offset) = VK_TRUE;
	static struct device_features ahead;
	check(device_features_enable(&ahead, &hidden, true) && valid(&hidden) && all_network_bits(&hidden)
	      && vk_chain_find(hidden.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO) == &opaque,
	      "the network's structures were not put ahead of an uncopyable chain");
	puts("device features: network features set in the game's structures or added, never twice, "
	     "game chains unwritten, uncopyable prefixes declined, and the ledger");
}

/** @brief The cooperative-matrix configuration that the network needs. */
static VkCooperativeMatrixPropertiesKHR const NETWORK_MATRIX = {
	.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
	.MSize = 16,
	.NSize = 16,
	.KSize = 16,
	.AType = VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT,
	.BType = VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT,
	.CType = VK_COMPONENT_TYPE_FLOAT32_KHR,
	.ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR,
	.saturatingAccumulation = VK_FALSE,
	.scope = VK_SCOPE_SUBGROUP_KHR,
};

/** @brief A device as network_requirements_unsupported() sees it through the next layer's
 *         functions.
 */
static struct {
	char const                       *extensions[NETWORK_FEATURE_COUNT]; //!< The extensions it offers.
	char const                       *unsupported;      //!< A feature it lacks, or nullptr.
	VkCooperativeMatrixPropertiesKHR  matrices[3];      //!< The cooperative-matrix configurations it lists.
	uint32_t                          extension_count;  //!< The entries of extensions.
	uint32_t                          matrix_count;     //!< The entries of matrices.
	uint32_t                          late;             //!< The last matrices, listed only after the count.
	uint32_t                          api;              //!< Its Vulkan version.
	uint32_t                          minimum_subgroup; //!< Its smallest subgroup.
	uint32_t                          maximum_subgroup; //!< Its largest subgroup.
	VkResult                          filled;           //!< What filling the extension list returns.
	VkResult                          counted;          //!< What counting the matrices returns.
} stub = {
	.api = VK_API_VERSION_1_3,
	.minimum_subgroup = 32,
	.maximum_subgroup = 64,
};

/** @brief vkGetPhysicalDeviceProperties2 of the stub device. */
static void VKAPI_CALL
stub_properties2 (VkPhysicalDevice             physical,
                  VkPhysicalDeviceProperties2 *properties)
{
	properties->properties.apiVersion = stub.api;
	VkPhysicalDeviceSubgroupSizeControlProperties *const subgroup =
		vk_chain_find(properties->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
	if (!subgroup)
		return;
	subgroup->minSubgroupSize = stub.minimum_subgroup;
	subgroup->maxSubgroupSize = stub.maximum_subgroup;
	subgroup->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
}

/** @brief vkGetPhysicalDeviceFeatures2 of the stub device: every network feature but the one it
 *         lacks.
 */
static void VKAPI_CALL
stub_features2 (VkPhysicalDevice           physical,
                VkPhysicalDeviceFeatures2 *features)
{
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		struct network_feature const *const f = &NETWORK_FEATURES[i];
		void *const node = vk_chain_find(features->pNext, f->type);
		if (node)
			*vk_chain_bit(node, f->offset) = !stub.unsupported || strcmp(stub.unsupported, f->name);
	}
}

/** @brief vkEnumerateDeviceExtensionProperties of the stub device; an error writes nothing. */
static VkResult VKAPI_CALL
stub_extensions (VkPhysicalDevice       physical,
                 char const            *layer,
                 uint32_t              *count,
                 VkExtensionProperties *properties)
{
	if (properties) {
		if (stub.filled != VK_SUCCESS)
			return stub.filled;
		for (uint32_t i = 0; i < *count && i < stub.extension_count; ++i) {
			int const n = snprintf(properties[i].extensionName, sizeof properties[i].extensionName, "%s",
			                       stub.extensions[i]);
			check(n >= 0 && n < (int)sizeof properties[i].extensionName,
			      "an extension's name does not fit");
		}
	}
	*count = stub.extension_count;
	return VK_SUCCESS;
}

/** @brief vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR of the stub device. */
static VkResult VKAPI_CALL
stub_matrices (VkPhysicalDevice                  physical,
               uint32_t                         *count,
               VkCooperativeMatrixPropertiesKHR *properties)
{
	if (!properties) {
		*count = stub.matrix_count - stub.late;
		return stub.counted;
	}
	if (*count > stub.matrix_count)
		*count = stub.matrix_count;
	memcpy(properties, stub.matrices, *count * sizeof *properties);
	return *count < stub.matrix_count ? VK_INCOMPLETE : VK_SUCCESS;
}

/** @brief What keeps the network off the stub device.
 *
 * @param instance The instance's Vulkan version.
 * @return         What device_features_network_unavailable() names, or nullptr.
 */
static char const *
stub_unavailable (uint32_t instance)
{
	return device_features_network_unavailable(VK_NULL_HANDLE, instance, stub_properties2, stub_features2,
	                                           stub_extensions, stub_matrices);
}

/** @brief Sets the cooperative-matrix configurations that the stub device lists.
 *
 * @param matrices The configurations.
 * @param count    Their number, at most 3.
 */
static void
stub_list (VkCooperativeMatrixPropertiesKHR const *matrices,
           uint32_t                                count)
{
	for (uint32_t i = 0; i < count; ++i)
		stub.matrices[i] = matrices[i];
	stub.matrix_count = count;
}

/** @brief Whether two strings are the same; nullptr is no string.
 *
 * @param a The first string, or nullptr.
 * @param b The second, or nullptr.
 * @return  true if both are strings and the same.
 */
static bool
same (char const *a,
      char const *b)
{
	return a && b && !strcmp(a, b);
}

/** @brief The number of fields of a configuration that network_matrices() changes, one at a time. */
enum { MATRIX_CHANGES = 9 };

/** @brief Changes one field of a cooperative-matrix configuration.
 *
 * @param m     The configuration.
 * @param which The field, below MATRIX_CHANGES.
 */
static void
change_matrix (VkCooperativeMatrixPropertiesKHR *m,
               uint32_t                          which)
{
	switch (which) {
	case 0: m->MSize = 8; break;
	case 1: m->NSize = 8; break;
	case 2: m->KSize = 32; break;
	case 3: m->AType = VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT; break;
	case 4: m->BType = VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT; break;
	case 5: m->CType = VK_COMPONENT_TYPE_FLOAT16_KHR; break;
	case 6: m->ResultType = VK_COMPONENT_TYPE_FLOAT16_KHR; break;
	case 7: m->saturatingAccumulation = VK_TRUE; break;
	case 8: m->scope = VK_SCOPE_WORKGROUP_KHR; break;
	}
}

/** @brief A device that lists the network's FP8 configuration among others is given the network,
 *         also when it counts fewer configurations than it lists and returns VK_INCOMPLETE. One that
 *         lists it with any one field changed, or that cannot be asked, is refused, naming what it
 *         lacks.
 */
static void
network_matrices (void)
{
	VkCooperativeMatrixPropertiesKHR fp16 = NETWORK_MATRIX;
	fp16.AType = fp16.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
	VkCooperativeMatrixPropertiesKHR fp16_only = fp16;
	fp16_only.CType = fp16_only.ResultType = VK_COMPONENT_TYPE_FLOAT16_KHR;
	stub_list((VkCooperativeMatrixPropertiesKHR const[]){fp16_only, fp16, NETWORK_MATRIX}, 3);
	check(!stub_unavailable(VK_API_VERSION_1_3), "a device that lists the network's FP8 matrices was refused");
	stub_list((VkCooperativeMatrixPropertiesKHR const[]){NETWORK_MATRIX, fp16, fp16_only}, 3);
	stub.late = 1;
	check(!stub_unavailable(VK_API_VERSION_1_3), "a device that counted fewer matrices than it lists was refused");
	stub.late = 0;
	for (uint32_t which = 0; which < MATRIX_CHANGES; ++which) {
		stub_list((VkCooperativeMatrixPropertiesKHR const[]){fp16_only, fp16, NETWORK_MATRIX}, 3);
		change_matrix(&stub.matrices[2], which);
		check(same(stub_unavailable(VK_API_VERSION_1_3), "e4m3 16x16x16 cooperative matrices"),
		      "a device without the network's FP8 matrices was given the network");
	}
	stub_list(nullptr, 0);
	check(same(stub_unavailable(VK_API_VERSION_1_3), "e4m3 16x16x16 cooperative matrices"),
	      "a device that lists no cooperative matrices was given the network");
	stub_list(&NETWORK_MATRIX, 1);
	stub.counted = VK_ERROR_OUT_OF_HOST_MEMORY;
	check(same(stub_unavailable(VK_API_VERSION_1_3), "e4m3 16x16x16 cooperative matrices"),
	      "a device whose cooperative matrices cannot be counted was given the network");
	stub.counted = VK_SUCCESS;
	check(same(device_features_network_unavailable(VK_NULL_HANDLE, VK_API_VERSION_1_3, stub_properties2,
	                                               stub_features2, stub_extensions, nullptr),
	           "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"),
	      "a device was given the network without a cooperative-matrix query");
}

/** @brief The network is refused for each missing requirement, its FP8 matrices included, its
 *         extensions are added once, and a declined request falls back.
 */
static void
network_availability (void)
{
	char const *all[NETWORK_FEATURE_COUNT];
	uint32_t const all_count = network_extensions(all);
	memcpy(stub.extensions, all, all_count * sizeof *all);
	stub.extension_count = all_count;
	stub_list(&NETWORK_MATRIX, 1);
	check(!stub_unavailable(VK_API_VERSION_1_3), "a capable device was refused the network");
	check(same(stub_unavailable(VK_API_VERSION_1_2), "a Vulkan 1.3 instance")
	      && same(device_features_network_unavailable(VK_NULL_HANDLE, VK_API_VERSION_1_3, nullptr, stub_features2,
	                                                  stub_extensions, stub_matrices),
	              "a Vulkan 1.3 instance"),
	      "a Vulkan 1.2 instance, or one without the 1.1 queries, was given the network");
	stub.api = VK_API_VERSION_1_2;
	check(same(stub_unavailable(VK_API_VERSION_1_3), "Vulkan 1.3"), "a Vulkan 1.2 device was given the network");
	stub.api = VK_API_VERSION_1_3;
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		stub.unsupported = NETWORK_FEATURES[i].name;
		check(same(stub_unavailable(VK_API_VERSION_1_3), NETWORK_FEATURES[i].name),
		      "a missing feature was not named");
	}
	stub.unsupported = nullptr;
	for (uint32_t i = 0; i < all_count; ++i) {
		// Every extension but the i-th.
		uint32_t n = 0;
		for (uint32_t e = 0; e < all_count; ++e)
			if (e != i)
				stub.extensions[n++] = all[e];
		stub.extension_count = n;
		check(same(stub_unavailable(VK_API_VERSION_1_3), all[i]), "a missing extension was not named");
	}
	stub.extension_count = 0;
	check(same(stub_unavailable(VK_API_VERSION_1_3), all[0])
	      && same(network_requirements_unsupported(VK_NULL_HANDLE, stub_properties2, stub_features2,
	                                               stub_extensions, stub_matrices,
	                                               VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
	              VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
	      "a device that offers no extensions was not refused, naming the first one needed");
	memcpy(stub.extensions, all, all_count * sizeof *all);
	stub.extension_count = all_count;
	stub.filled = VK_ERROR_OUT_OF_HOST_MEMORY;
	check(same(stub_unavailable(VK_API_VERSION_1_3), NETWORK_REQUIREMENTS_NO_MEMORY),
	      "a device whose extensions could not be listed was not refused for host memory");
	stub.filled = VK_SUCCESS;
	check(same(network_requirements_unsupported(VK_NULL_HANDLE, stub_properties2, stub_features2, stub_extensions,
	                                            stub_matrices, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
	           VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
	      "the extension dlsslopd needs besides was not named");
	stub.minimum_subgroup = 64;
	check(same(stub_unavailable(VK_API_VERSION_1_3), "32-lane compute subgroups"),
	      "a device without 32-lane subgroups was given the network");
	stub.minimum_subgroup = 32;
	network_matrices();

	// The network's extensions join a list once each, in the table's order, behind its own.
	char const *names[2 + NETWORK_FEATURE_COUNT] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, all[1]};
	uint32_t const named = network_requirements_append_extensions(names, 2);
	check(named == 1 + all_count && same(names[0], VK_KHR_SWAPCHAIN_EXTENSION_NAME) && same(names[1], all[1])
	      && same(names[2], all[0]) && same(names[3], all[2])
	      && network_requirements_append_extensions(names, named) == named,
	      "the network's extensions were not appended once each, in order, behind the list's");

	// The network's extensions join the list once each, behind the game's.
	char const *game[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
	VkDeviceCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.enabledExtensionCount = 2,
		.ppEnabledExtensionNames = game,
	};
	char const *list[2 + NETWORK_FEATURE_COUNT] = {};
	check(device_features_add_network_extensions(&info, list) && info.ppEnabledExtensionNames == list
	      && info.enabledExtensionCount == 1 + all_count && list[0] == game[0] && list[1] == game[1]
	      && same(list[2], all[1]) && same(list[3], all[2]),
	      "the network's extensions were not added once each behind the game's");
	// The request's own list, which already names them all.
	check(!device_features_add_network_extensions(&info, list) && info.ppEnabledExtensionNames == list
	      && info.enabledExtensionCount == 1 + all_count && list[0] == game[0],
	      "extensions already listed were added again");
	// A request without extensions.
	VkDeviceCreateInfo bare = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	char const *only[NETWORK_FEATURE_COUNT] = {};
	check(device_features_add_network_extensions(&bare, only) && bare.ppEnabledExtensionNames == only
	      && bare.enabledExtensionCount == all_count && same(only[0], all[0]),
	      "the network's extensions were not added to a request without extensions");
	check(!device_features_add_network_extensions(nullptr, only)
	      && !device_features_add_network_extensions(&bare, nullptr),
	      "extensions were added to no request or no list");

	// A request that declined the network still enables formatless storage writes.
	VkPhysicalDeviceVulkan12Features late = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkBaseOutStructure opaque = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pNext = (VkBaseOutStructure *)&late};
	VkPhysicalDeviceFeatures const legacy = {};
	VkDeviceCreateInfo hidden = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &opaque,
		.pEnabledFeatures = &legacy,
	};
	static struct device_features request;
	check(!device_features_enable(&request, &hidden, true) && device_features_enable(&request, &hidden, false)
	      && device_features_has_formatless_storage_writes(&hidden) && hidden.pNext == &opaque && !late.shaderInt8,
	      "a request that declined the network did not fall back to formatless storage alone");
	puts("device features: the network is refused for each missing requirement, its FP8 matrices included, "
	     "its extensions are added once, and a declined request falls back");
}

/** @brief The chain that asks a device for the network's features, and that dlsslopd enables: one
 *         zeroed structure of each type, in this order behind VkPhysicalDeviceFeatures2.
 */
static void
feature_chain (void)
{
	static VkStructureType const order[] = {
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
	};
	struct network_feature_chain chain;
	memset(&chain, 0xa5, sizeof chain);
	network_feature_chain_init(&chain);
	void const *node = &chain.head;
	struct network_feature_chain rest = chain;
	for (uint32_t i = 0; i < sizeof order / sizeof *order; ++i) {
		check(node && vk_chain_type(node) == order[i]
		      && network_feature_chain_structure(&chain, order[i]) == node,
		      "the feature chain's structures are not linked in order");
		size_t const offset = (size_t)((unsigned char const *)node - (unsigned char const *)&chain);
		memset((unsigned char *)&rest + offset, 0, sizeof (VkBaseInStructure));
		node = vk_chain_next(node);
	}
	check(!node, "the feature chain does not end after its last structure");
	unsigned char const *const bytes = (unsigned char const *)&rest;
	bool zeroed = true;
	for (size_t i = 0; zeroed && i < sizeof rest; ++i)
		zeroed = !bytes[i];
	check(zeroed, "the feature chain was not zeroed besides its types and links");
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		struct network_feature const *const f = &NETWORK_FEATURES[i];
		VkBool32 const *const bit = vk_chain_bit(vk_chain_find(&chain.head, f->type), f->offset);
		check(network_feature_chain_bit(&chain, f) == bit && !*network_feature_chain_bit(&chain, f),
		      "a feature's bit is not in its structure in the chain");
	}
	check(!network_feature_chain_structure(&chain, VK_STRUCTURE_TYPE_APPLICATION_INFO)
	      && !network_feature_chain_structure(nullptr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
	      && !network_feature_chain_bit(nullptr, &NETWORK_FEATURES[0]),
	      "a feature chain answered for a structure it does not have");
	network_feature_chain_init(nullptr);
	puts("network requirements: the feature chain is zeroed and linked in order");
}

/** @brief The copies' bound: a structure to change behind DEVICE_FEATURES_COPIES - 1 that need copies
 *         is changed in the last of DEVICE_FEATURES_COPIES copies; behind one more, the request
 *         declines, untouched. A valid chain repeats no structure but the loader's; this one is
 *         copyable all the same.
 */
static void
copy_bound (void)
{
	for (uint32_t i = 0; i < DEVICE_FEATURES_COPYABLE_COUNT; ++i) {
		uint32_t listed = 0;
		for (uint32_t o = 0; o < DEVICE_FEATURES_COPYABLE_COUNT; ++o)
			listed += DEVICE_FEATURES_COPYABLE[o].type == DEVICE_FEATURES_COPYABLE[i].type;
		check(DEVICE_FEATURES_COPYABLE[i].size <= sizeof (union device_features_copy) && listed == 1,
		      "a copyable structure does not fit a copy or is listed twice");
	}
	VkBaseOutStructure tail = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO};
	VkPhysicalDeviceFeatures2 core = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &tail};
	static VkPhysicalDeviceVulkan11Features ahead[DEVICE_FEATURES_COPIES];
	for (uint32_t i = 0; i < DEVICE_FEATURES_COPIES; ++i)
		ahead[i] = (VkPhysicalDeviceVulkan11Features){
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
			.pNext = i + 1 < DEVICE_FEATURES_COPIES ? (void *)&ahead[i + 1] : (void *)&core,
		};
	VkDeviceCreateInfo fits = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &ahead[1]};
	static struct device_features request;
	check(device_features_enable(&request, &fits, false) && device_features_has_formatless_storage_writes(&fits),
	      "a chain that needs DEVICE_FEATURES_COPIES copies was declined");
	// The structures in the request, by address: pointers to other objects have no order in C.
	uint32_t copies = 0;
	void const *node = fits.pNext;
	uintptr_t const first = (uintptr_t)&request;
	for (; node && node != &tail; node = vk_chain_next(node))
		copies += (uintptr_t)node - first < sizeof request;
	check(node == &tail && copies == DEVICE_FEATURES_COPIES && !core.features.shaderStorageImageWriteWithoutFormat,
	      "a chain of DEVICE_FEATURES_COPIES copies was not copied up to its rest");
	VkDeviceCreateInfo over = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &ahead[0]};
	static struct device_features declined;
	check(!device_features_enable(&declined, &over, false) && over.pNext == &ahead[0] && !over.pEnabledFeatures
	      && !core.features.shaderStorageImageWriteWithoutFormat,
	      "a chain that needs more than DEVICE_FEATURES_COPIES copies was not declined intact");
	// Nothing to change: no copy, and the chain stays the game's.
	core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	check(device_features_enable(&declined, &over, false) && over.pNext == &ahead[0],
	      "a chain with nothing to change was copied");
	puts("device features: at most DEVICE_FEATURES_COPIES copies, and a longer prefix declined intact");
}

/** @brief The request asks for the in-layer network only on DLSSLOP_LAYER_NETWORK=1, and its
 *         functions answer nothing for no request.
 */
static void
requests (void)
{
	check(!unsetenv("DLSSLOP_LAYER_NETWORK"), "unsetenv failed");
	bool const unset = device_features_network_requested();
	check(!setenv("DLSSLOP_LAYER_NETWORK", "1", 1), "setenv failed");
	bool const one = device_features_network_requested();
	check(!setenv("DLSSLOP_LAYER_NETWORK", "0", 1), "setenv failed");
	bool const zero = device_features_network_requested();
	check(!setenv("DLSSLOP_LAYER_NETWORK", "11", 1), "setenv failed");
	bool const eleven = device_features_network_requested();
	check(!unsetenv("DLSSLOP_LAYER_NETWORK"), "unsetenv failed");
	check(!unset && one && !zero && !eleven, "the in-layer network was asked for by another value than 1");
	VkDeviceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	static struct device_features request;
	check(!device_features_enable(nullptr, &info, false) && !device_features_enable(&request, nullptr, false)
	      && !device_features_network_enabled(nullptr) && !device_features_has_formatless_storage_writes(nullptr),
	      "a request's functions answered for no request");
	puts("device features: DLSSLOP_LAYER_NETWORK=1 asks for the network, and no request has no features");
}

int
main (void)
{
	feature_chain();
	copy_bound();
	requests();
	features();
	network_features();
	network_availability();
#define SHADER(name) shader(name##_spv, sizeof name##_spv, #name)
	SHADER(dlssnr);
	SHADER(bcus);
	SHADER(bcds_bicubic);
	SHADER(bcds_catmull);
	SHADER(bcds_lanczos2);
	SHADER(bcds_lanczos3);
	SHADER(bcds_kaiser2);
	SHADER(bcds_kaiser3);
	SHADER(bcds_magc);
#undef SHADER
	return 0;
}
