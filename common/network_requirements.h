/** @file
 *
 * What DLSSNR-AMD's Vulkan network needs of a device: one table for dlsslopd's own device and for a
 * game's device the layer adds it to.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_NETWORK_REQUIREMENTS_H_
#define DLSSLOP_AMD_COMMON_NETWORK_REQUIREMENTS_H_

#ifdef __cplusplus
# include <cstddef>
# include <cstdint>
# include <cstdlib>
# include <cstring>
# define NETWORK_STD(x) std::x
#else
# include <stddef.h>
# include <stdint.h>
# include <stdlib.h>
# include <string.h>
# define NETWORK_STD(x) x
#endif

#include <vulkan/vulkan.h>

#include "vk_chain.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A feature bit that the network's SPIR-V uses.
 *
 * Upstream listed them in DLSSNR-AMD's linux/src/core/nrvk.hpp, Context::create and adopt.
 */
struct network_feature {
	char const            *name;        //!< The bit's name.
	char const            *extension;   //!< The extension that provides it; nullptr in core Vulkan 1.3.
	VkStructureType        type;        //!< The structure that carries the bit alone.
	NETWORK_STD(uint32_t)  offset;      //!< The bit's offset in that structure.
	VkStructureType        core;        //!< The VkPhysicalDeviceVulkan1xFeatures that carries it too.
	NETWORK_STD(uint32_t)  core_offset; //!< The bit's offset in that structure.
};

/** @brief The core type of a bit that no VkPhysicalDeviceVulkan1xFeatures carries. */
static constexpr VkStructureType NETWORK_FEATURE_NO_CORE = VK_STRUCTURE_TYPE_MAX_ENUM;

#define NETWORK_FEATURE_ALONE(Struct, type, field) \
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##type, offsetof(Struct, field)
#define NETWORK_FEATURE_CORE(digits, version, field) \
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_##version##_FEATURES, \
	offsetof(VkPhysicalDeviceVulkan##digits##Features, field)
#define NETWORK_FEATURE_NONE NETWORK_FEATURE_NO_CORE, 0

/** @brief The feature bits that the network's SPIR-V uses. */
static struct network_feature const NETWORK_FEATURES[] = {
	{"cooperativeMatrix", VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, COOPERATIVE_MATRIX_FEATURES_KHR,
	                       cooperativeMatrix),
	 NETWORK_FEATURE_NONE},
	{"shaderFloat8", VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT,
	                       shaderFloat8),
	 NETWORK_FEATURE_NONE},
	{"shaderFloat8CooperativeMatrix", VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT,
	                       shaderFloat8CooperativeMatrix),
	 NETWORK_FEATURE_NONE},
	{"storageBuffer16BitAccess", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDevice16BitStorageFeatures, 16BIT_STORAGE_FEATURES,
	                       storageBuffer16BitAccess),
	 NETWORK_FEATURE_CORE(11, 1_1, storageBuffer16BitAccess)},
	{"storageBuffer8BitAccess", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDevice8BitStorageFeatures, 8BIT_STORAGE_FEATURES,
	                       storageBuffer8BitAccess),
	 NETWORK_FEATURE_CORE(12, 1_2, storageBuffer8BitAccess)},
	{"shaderFloat16", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES,
	                       shaderFloat16),
	 NETWORK_FEATURE_CORE(12, 1_2, shaderFloat16)},
	{"shaderInt8", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES,
	                       shaderInt8),
	 NETWORK_FEATURE_CORE(12, 1_2, shaderInt8)},
	{"vulkanMemoryModel", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures, VULKAN_MEMORY_MODEL_FEATURES,
	                       vulkanMemoryModel),
	 NETWORK_FEATURE_CORE(12, 1_2, vulkanMemoryModel)},
	// Not the network's own need: with the memory model on and this off, no shader on the device,
	// the game's or the composition's, may use Device scope (VUID-RuntimeSpirv-vulkanMemoryModel-06265).
	{"vulkanMemoryModelDeviceScope", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures, VULKAN_MEMORY_MODEL_FEATURES,
	                       vulkanMemoryModelDeviceScope),
	 NETWORK_FEATURE_CORE(12, 1_2, vulkanMemoryModelDeviceScope)},
	{"subgroupSizeControl", nullptr,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceSubgroupSizeControlFeatures, SUBGROUP_SIZE_CONTROL_FEATURES,
	                       subgroupSizeControl),
	 NETWORK_FEATURE_CORE(13, 1_3, subgroupSizeControl)},
	{"workgroupMemoryExplicitLayout", VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
	                       WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
	                       workgroupMemoryExplicitLayout),
	 NETWORK_FEATURE_NONE},
	{"workgroupMemoryExplicitLayout8BitAccess", VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
	                       WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
	                       workgroupMemoryExplicitLayout8BitAccess),
	 NETWORK_FEATURE_NONE},
	{"workgroupMemoryExplicitLayout16BitAccess", VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
	 NETWORK_FEATURE_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
	                       WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
	                       workgroupMemoryExplicitLayout16BitAccess),
	 NETWORK_FEATURE_NONE},
};

#undef NETWORK_FEATURE_ALONE
#undef NETWORK_FEATURE_CORE
#undef NETWORK_FEATURE_NONE

/** @brief The number of NETWORK_FEATURES. */
static constexpr NETWORK_STD(uint32_t) NETWORK_FEATURE_COUNT =
	sizeof NETWORK_FEATURES / sizeof *NETWORK_FEATURES;

/** @brief One structure of each type that NETWORK_FEATURES names, zeroed and chained behind a
 *         VkPhysicalDeviceFeatures2: for a support query or a vkCreateDevice.
 *
 * Its structures point at each other, so it is only ever initialized in place, by
 * network_feature_chain_init().
 */
struct network_feature_chain {
	VkPhysicalDeviceFeatures2                                head;      //!< The chain's first structure.
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR             coop;      //!< Cooperative matrices.
	VkPhysicalDeviceShaderFloat8FeaturesEXT                  fp8;       //!< FP8.
	VkPhysicalDevice16BitStorageFeatures                     storage16; //!< 16-bit storage.
	VkPhysicalDevice8BitStorageFeatures                      storage8;  //!< 8-bit storage.
	VkPhysicalDeviceShaderFloat16Int8Features                float16;   //!< FP16 and Int8 arithmetic.
	VkPhysicalDeviceVulkanMemoryModelFeatures                memory;    //!< The memory model.
	VkPhysicalDeviceSubgroupSizeControlFeatures              subgroup;  //!< Subgroup size control.
	VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR layout;    //!< Explicit workgroup layout.
};

/** @brief A structure of struct network_feature_chain. */
struct network_feature_chain_member {
	VkStructureType       type;   //!< The structure's type.
	NETWORK_STD(uint32_t) offset; //!< Its offset in the chain.
};

#define NETWORK_FEATURE_CHAIN_MEMBER(member, type) \
	{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##type, offsetof(struct network_feature_chain, member)}

/** @brief The structures of struct network_feature_chain, in the chain's order. */
static struct network_feature_chain_member const NETWORK_FEATURE_CHAIN_MEMBERS[] = {
	NETWORK_FEATURE_CHAIN_MEMBER(head, FEATURES_2),
	NETWORK_FEATURE_CHAIN_MEMBER(coop, COOPERATIVE_MATRIX_FEATURES_KHR),
	NETWORK_FEATURE_CHAIN_MEMBER(fp8, SHADER_FLOAT8_FEATURES_EXT),
	NETWORK_FEATURE_CHAIN_MEMBER(storage16, 16BIT_STORAGE_FEATURES),
	NETWORK_FEATURE_CHAIN_MEMBER(storage8, 8BIT_STORAGE_FEATURES),
	NETWORK_FEATURE_CHAIN_MEMBER(float16, SHADER_FLOAT16_INT8_FEATURES),
	NETWORK_FEATURE_CHAIN_MEMBER(memory, VULKAN_MEMORY_MODEL_FEATURES),
	NETWORK_FEATURE_CHAIN_MEMBER(subgroup, SUBGROUP_SIZE_CONTROL_FEATURES),
	NETWORK_FEATURE_CHAIN_MEMBER(layout, WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR),
};

#undef NETWORK_FEATURE_CHAIN_MEMBER

/** @brief The number of NETWORK_FEATURE_CHAIN_MEMBERS. */
static constexpr NETWORK_STD(uint32_t) NETWORK_FEATURE_CHAIN_MEMBER_COUNT =
	sizeof NETWORK_FEATURE_CHAIN_MEMBERS / sizeof *NETWORK_FEATURE_CHAIN_MEMBERS;

/** @brief Zeroes a chain's structures, sets their types and links them in the chain's order.
 *
 * @param dest The chain, or nullptr.
 */
static inline void
network_feature_chain_init (struct network_feature_chain *dest)
{
	if (!dest)
		return;
	NETWORK_STD(memset)(dest, 0, sizeof *dest);
	void const *next = nullptr;
	for (NETWORK_STD(uint32_t) i = NETWORK_FEATURE_CHAIN_MEMBER_COUNT; i--;) {
		struct network_feature_chain_member const *const m = &NETWORK_FEATURE_CHAIN_MEMBERS[i];
		unsigned char *const structure = (unsigned char *)dest + m->offset;
		NETWORK_STD(memcpy)(structure, &m->type, sizeof m->type);
		vk_chain_link(structure, next);
		next = structure;
	}
}

/** @brief A chain's structure of a type, whatever a caller has since linked it to.
 *
 * @param c    The chain, or nullptr.
 * @param type The type.
 * @return     The structure, or nullptr if the chain has none of @a type.
 */
static inline void *
network_feature_chain_structure (struct network_feature_chain *c,
                                 VkStructureType               type)
{
	if (!c)
		return nullptr;
	for (NETWORK_STD(uint32_t) i = 0; i < NETWORK_FEATURE_CHAIN_MEMBER_COUNT; ++i) {
		if (NETWORK_FEATURE_CHAIN_MEMBERS[i].type == type)
			return (unsigned char *)c + NETWORK_FEATURE_CHAIN_MEMBERS[i].offset;
	}
	return nullptr;
}

/** @brief A chain's bit of a feature.
 *
 * @param c The chain, or nullptr.
 * @param f The feature.
 * @return  The bit in the structure that carries it alone, or nullptr if the chain has none.
 */
static inline VkBool32 *
network_feature_chain_bit (struct network_feature_chain *c,
                           struct network_feature const *f)
{
	void *const structure = network_feature_chain_structure(c, f->type);
	return structure ? vk_chain_bit(structure, f->offset) : nullptr;
}

/** @brief What stops the network when the host's memory runs out, named like the other requirements. */
static constexpr char NETWORK_REQUIREMENTS_NO_MEMORY[] = "host memory";

/** @brief What stops the network on a device that lists no FP8 matrices of the network's shape. */
static constexpr char NETWORK_REQUIREMENTS_NO_MATRICES[] = "e4m3 16x16x16 cooperative matrices";

/** @brief Whether a list holds a name.
 *
 * @param list  The list.
 * @param count The names in it.
 * @param name  The name.
 * @return      true if @a list holds @a name.
 */
static inline bool
network_requirements_listed (char const *const     *list,
                             NETWORK_STD(uint32_t)  count,
                             char const            *name)
{
	for (NETWORK_STD(uint32_t) i = 0; i < count; ++i) {
		if (!NETWORK_STD(strcmp)(list[i], name))
			return true;
	}
	return false;
}

/** @brief Appends the network's extensions that a list lacks.
 *
 * @param list  The list, with room for NETWORK_FEATURE_COUNT more names.
 * @param count The names in it.
 * @return      The names in it now: more than @a count if any was appended.
 */
static inline NETWORK_STD(uint32_t)
network_requirements_append_extensions (char const            **list,
                                        NETWORK_STD(uint32_t)   count)
{
	for (NETWORK_STD(uint32_t) i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		char const *const extension = NETWORK_FEATURES[i].extension;
		if (extension && !network_requirements_listed(list, count, extension))
			list[count++] = extension;
	}
	return count;
}

/** @brief Whether a device's extensions hold one.
 *
 * @param offered The device's extensions.
 * @param count   The extensions in @a offered.
 * @param name    The extension.
 * @return        true if @a offered holds @a name.
 */
static inline bool
network_requirements_offered (VkExtensionProperties const *offered,
                              NETWORK_STD(uint32_t)        count,
                              char const                  *name)
{
	for (NETWORK_STD(uint32_t) i = 0; i < count; ++i) {
		if (!NETWORK_STD(strcmp)(offered[i].extensionName, name))
			return true;
	}
	return false;
}

/** @brief The first extension that a device's extensions lack: @a also, then the network's.
 *
 * @param offered The device's extensions.
 * @param count   The extensions in @a offered.
 * @param also    An extension the caller needs besides, or nullptr.
 * @return        The extension's name, or nullptr if @a offered holds them all.
 */
static inline char const *
network_requirements_unoffered (VkExtensionProperties const *offered,
                                NETWORK_STD(uint32_t)        count,
                                char const                  *also)
{
	if (also && !network_requirements_offered(offered, count, also))
		return also;
	for (NETWORK_STD(uint32_t) i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		char const *const extension = NETWORK_FEATURES[i].extension;
		if (extension && !network_requirements_offered(offered, count, extension))
			return extension;
	}
	return nullptr;
}

/** @brief The first extension that a device lacks: @a also, then the network's.
 *
 * @param physical   The device.
 * @param extensions The caller's vkEnumerateDeviceExtensionProperties.
 * @param also       An extension the caller needs besides, or nullptr.
 * @return           The extension's name, NETWORK_REQUIREMENTS_NO_MEMORY, or nullptr if the device
 *                   offers them all.
 */
static inline char const *
network_requirements_missing_extension (VkPhysicalDevice                         physical,
                                        PFN_vkEnumerateDeviceExtensionProperties extensions,
                                        char const                              *also)
{
	NETWORK_STD(uint32_t) count = 0;
	extensions(physical, nullptr, &count, nullptr);
	// One more than counted, so that a device that offers none is no special case. Zeroed, so that a
	// list the second call fails to fill holds empty names.
	VkExtensionProperties *const offered =
		(VkExtensionProperties *)NETWORK_STD(calloc)(1, sizeof *offered * count + sizeof *offered);
	if (!offered)
		return NETWORK_REQUIREMENTS_NO_MEMORY;
	extensions(physical, nullptr, &count, offered);
	char const *const missing = network_requirements_unoffered(offered, count, also);
	NETWORK_STD(free)(offered);
	return missing;
}

/** @brief Whether a list of cooperative-matrix configurations holds the network's.
 *
 * Every network kernel multiplies e4m3 times e4m3 into FP32, 16x16x16, in a subgroup (upstream:
 * nrvk's require_matrix_config). The shaders also declare an e4m3 accumulator that RADV lists in no
 * configuration and accepts, so it is not required.
 *
 * @param listed The configurations.
 * @param count  The configurations in @a listed.
 * @return       true if @a listed holds the network's.
 */
static inline bool
network_requirements_lists_matrix (VkCooperativeMatrixPropertiesKHR const *listed,
                                   NETWORK_STD(uint32_t)                   count)
{
	for (NETWORK_STD(uint32_t) i = 0; i < count; ++i) {
		VkCooperativeMatrixPropertiesKHR const *const m = &listed[i];
		if (m->MSize == 16 && m->NSize == 16 && m->KSize == 16 && m->scope == VK_SCOPE_SUBGROUP_KHR &&
		    m->AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
		    m->BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
		    m->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
		    m->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
		    !m->saturatingAccumulation)
			return true;
	}
	return false;
}

/** @brief Whether a device lists the network's cooperative-matrix configuration.
 *
 * @param physical The device.
 * @param matrices The caller's vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
 * @return         nullptr if it does, otherwise NETWORK_REQUIREMENTS_NO_MATRICES, or
 *                 NETWORK_REQUIREMENTS_NO_MEMORY.
 */
static inline char const *
network_requirements_missing_matrix (VkPhysicalDevice                                      physical,
                                     PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrices)
{
	NETWORK_STD(uint32_t) count = 0;
	if (matrices(physical, &count, nullptr) != VK_SUCCESS || !count)
		return NETWORK_REQUIREMENTS_NO_MATRICES;
	VkCooperativeMatrixPropertiesKHR *const listed =
		(VkCooperativeMatrixPropertiesKHR *)NETWORK_STD(malloc)(sizeof *listed * count);
	if (!listed)
		return NETWORK_REQUIREMENTS_NO_MEMORY;
	VkCooperativeMatrixPropertiesKHR const empty = {
		.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR
	};
	for (NETWORK_STD(uint32_t) i = 0; i < count; ++i)
		listed[i] = empty;
	// VK_INCOMPLETE leaves out configurations beyond those counted.
	VkResult const result = matrices(physical, &count, listed);
	bool const found = (result == VK_SUCCESS || result == VK_INCOMPLETE) &&
	                   network_requirements_lists_matrix(listed, count);
	NETWORK_STD(free)(listed);
	return found ? nullptr : NETWORK_REQUIREMENTS_NO_MATRICES;
}

/** @brief What stops the network running on a device.
 *
 * The functions are the caller's route to the device: the loader's, or the next layer's.
 *
 * @param physical    The device.
 * @param properties2 vkGetPhysicalDeviceProperties2.
 * @param features2   vkGetPhysicalDeviceFeatures2.
 * @param extensions  vkEnumerateDeviceExtensionProperties.
 * @param matrices    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, or nullptr.
 * @param also        An extension the caller needs besides, or nullptr.
 * @return            nullptr if nothing does. Otherwise "Vulkan 1.3", an extension's or a feature's
 *                    name, "32-lane compute subgroups", the name of @a matrices if it is nullptr,
 *                    NETWORK_REQUIREMENTS_NO_MATRICES, or NETWORK_REQUIREMENTS_NO_MEMORY.
 */
static inline char const *
network_requirements_unsupported (VkPhysicalDevice                                      physical,
                                  PFN_vkGetPhysicalDeviceProperties2                    properties2,
                                  PFN_vkGetPhysicalDeviceFeatures2                      features2,
                                  PFN_vkEnumerateDeviceExtensionProperties              extensions,
                                  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrices,
                                  char const                                           *also)
{
	VkPhysicalDeviceSubgroupSizeControlProperties subgroup = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
	};
	VkPhysicalDeviceProperties2 properties = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &subgroup
	};
	properties2(physical, &properties);
	if (properties.properties.apiVersion < VK_API_VERSION_1_3)
		return "Vulkan 1.3";
	char const *const missing = network_requirements_missing_extension(physical, extensions, also);
	if (missing)
		return missing;
	struct network_feature_chain supported;
	network_feature_chain_init(&supported);
	features2(physical, &supported.head);
	for (NETWORK_STD(uint32_t) i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		if (!*network_feature_chain_bit(&supported, &NETWORK_FEATURES[i]))
			return NETWORK_FEATURES[i].name;
	}
	// The cooperative-matrix fragments are laid out for 32-lane subgroups.
	if (subgroup.minSubgroupSize > 32 || subgroup.maxSubgroupSize < 32 ||
	    !(subgroup.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
		return "32-lane compute subgroups";
	if (!matrices)
		return "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR";
	return network_requirements_missing_matrix(physical, matrices);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef NETWORK_STD

#endif /* DLSSLOP_AMD_COMMON_NETWORK_REQUIREMENTS_H_ */
