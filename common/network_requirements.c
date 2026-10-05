/** @file
 *
 * What DLSSNR-AMD's Vulkan network needs of a device: the table and the functions.
 */
// SPDX-License-Identifier: MIT
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "network_requirements.h"
#include "vk_chain.h"

#define NETWORK_FEATURE_ALONE(Struct, type, field) \
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##type, offsetof(Struct, field)
#define NETWORK_FEATURE_CORE(digits, version, field) \
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_##version##_FEATURES, \
	offsetof(VkPhysicalDeviceVulkan##digits##Features, field)
#define NETWORK_FEATURE_NONE NETWORK_FEATURE_NO_CORE, 0

// The rows of NETWORK_FEATURES, each X(name, extension, alone, core): the bit's name and extension,
// and the type and offset of the structure that carries it alone and of the core one that carries it.
#define NETWORK_FEATURES_ROWS(X) \
	X("cooperativeMatrix", VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, COOPERATIVE_MATRIX_FEATURES_KHR, \
	                        cooperativeMatrix), \
	  NETWORK_FEATURE_NONE) \
	X("shaderFloat8", VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT, \
	                        shaderFloat8), \
	  NETWORK_FEATURE_NONE) \
	X("shaderFloat8CooperativeMatrix", VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat8FeaturesEXT, SHADER_FLOAT8_FEATURES_EXT, \
	                        shaderFloat8CooperativeMatrix), \
	  NETWORK_FEATURE_NONE) \
	X("storageBuffer16BitAccess", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDevice16BitStorageFeatures, 16BIT_STORAGE_FEATURES, \
	                        storageBuffer16BitAccess), \
	  NETWORK_FEATURE_CORE(11, 1_1, storageBuffer16BitAccess)) \
	X("storageBuffer8BitAccess", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDevice8BitStorageFeatures, 8BIT_STORAGE_FEATURES, \
	                        storageBuffer8BitAccess), \
	  NETWORK_FEATURE_CORE(12, 1_2, storageBuffer8BitAccess)) \
	X("shaderFloat16", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES, \
	                        shaderFloat16), \
	  NETWORK_FEATURE_CORE(12, 1_2, shaderFloat16)) \
	X("shaderInt8", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceShaderFloat16Int8Features, SHADER_FLOAT16_INT8_FEATURES, \
	                        shaderInt8), \
	  NETWORK_FEATURE_CORE(12, 1_2, shaderInt8)) \
	X("vulkanMemoryModel", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures, VULKAN_MEMORY_MODEL_FEATURES, \
	                        vulkanMemoryModel), \
	  NETWORK_FEATURE_CORE(12, 1_2, vulkanMemoryModel)) \
	/* Not the network's own need: with the memory model on and this off, no shader on the device, \
	 * the game's or the composition's, may use Device scope \
	 * (VUID-RuntimeSpirv-vulkanMemoryModel-06265). */ \
	X("vulkanMemoryModelDeviceScope", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceVulkanMemoryModelFeatures, VULKAN_MEMORY_MODEL_FEATURES, \
	                        vulkanMemoryModelDeviceScope), \
	  NETWORK_FEATURE_CORE(12, 1_2, vulkanMemoryModelDeviceScope)) \
	X("subgroupSizeControl", nullptr, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceSubgroupSizeControlFeatures, SUBGROUP_SIZE_CONTROL_FEATURES, \
	                        subgroupSizeControl), \
	  NETWORK_FEATURE_CORE(13, 1_3, subgroupSizeControl)) \
	X("workgroupMemoryExplicitLayout", VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, \
	                        WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, \
	                        workgroupMemoryExplicitLayout), \
	  NETWORK_FEATURE_NONE) \
	X("workgroupMemoryExplicitLayout8BitAccess", VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, \
	                        WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, \
	                        workgroupMemoryExplicitLayout8BitAccess), \
	  NETWORK_FEATURE_NONE) \
	X("workgroupMemoryExplicitLayout16BitAccess", VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, \
	  NETWORK_FEATURE_ALONE(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, \
	                        WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, \
	                        workgroupMemoryExplicitLayout16BitAccess), \
	  NETWORK_FEATURE_NONE)

#define NETWORK_FEATURE(name, extension, alone, core) {name, extension, alone, core},

struct network_feature const NETWORK_FEATURES[] = {
	NETWORK_FEATURES_ROWS(NETWORK_FEATURE)
};

static_assert(sizeof NETWORK_FEATURES / sizeof *NETWORK_FEATURES == NETWORK_FEATURE_COUNT,
              "NETWORK_FEATURE_COUNT is not the number of rows of NETWORK_FEATURES");

#undef NETWORK_FEATURE
#undef NETWORK_FEATURE_ALONE
#undef NETWORK_FEATURE_CORE
#undef NETWORK_FEATURE_NONE
#undef NETWORK_FEATURES_ROWS

/** @brief A structure of struct network_feature_chain. */
struct network_feature_chain_member {
	VkStructureType type;   //!< The structure's type.
	uint32_t        offset; //!< Its offset in the chain.
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
enum : uint32_t {
	NETWORK_FEATURE_CHAIN_MEMBER_COUNT =
		sizeof NETWORK_FEATURE_CHAIN_MEMBERS / sizeof *NETWORK_FEATURE_CHAIN_MEMBERS,
};

void
network_feature_chain_init (struct network_feature_chain *dest)
{
	if (!dest)
		return;
	memset(dest, 0, sizeof *dest);
	void const *next = nullptr;
	for (uint32_t i = NETWORK_FEATURE_CHAIN_MEMBER_COUNT; i--;) {
		struct network_feature_chain_member const *const m = &NETWORK_FEATURE_CHAIN_MEMBERS[i];
		unsigned char *const structure = (unsigned char *)dest + m->offset;
		memcpy(structure, &m->type, sizeof m->type);
		vk_chain_link(structure, next);
		next = structure;
	}
}

void *
network_feature_chain_structure (struct network_feature_chain *c,
                                 VkStructureType               type)
{
	if (!c)
		return nullptr;
	for (uint32_t i = 0; i < NETWORK_FEATURE_CHAIN_MEMBER_COUNT; ++i) {
		if (NETWORK_FEATURE_CHAIN_MEMBERS[i].type == type)
			return (unsigned char *)c + NETWORK_FEATURE_CHAIN_MEMBERS[i].offset;
	}
	return nullptr;
}

VkBool32 *
network_feature_chain_bit (struct network_feature_chain *c,
                           struct network_feature const *f)
{
	void *const structure = network_feature_chain_structure(c, f->type);
	return structure ? vk_chain_bit(structure, f->offset) : nullptr;
}

bool
network_requirements_listed (char const *const *list,
                             uint32_t           count,
                             char const        *name)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (!strcmp(list[i], name))
			return true;
	}
	return false;
}

uint32_t
network_requirements_append_extensions (char const **list,
                                        uint32_t     count)
{
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
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
static bool
network_requirements_offered (VkExtensionProperties const *offered,
                              uint32_t                     count,
                              char const                  *name)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (!strcmp(offered[i].extensionName, name))
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
static char const *
network_requirements_unoffered (VkExtensionProperties const *offered,
                                uint32_t                     count,
                                char const                  *also)
{
	if (also && !network_requirements_offered(offered, count, also))
		return also;
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
		char const *const extension = NETWORK_FEATURES[i].extension;
		if (extension && !network_requirements_offered(offered, count, extension))
			return extension;
	}
	return nullptr;
}

/** @brief The first extension that a device lacks: @a also, then the network's.
 *
 * Without a layer name, listing the extensions fails only when memory runs out.
 *
 * @param physical   The device.
 * @param extensions The caller's vkEnumerateDeviceExtensionProperties.
 * @param also       An extension the caller needs besides, or nullptr.
 * @return           The extension's name, NETWORK_REQUIREMENTS_NO_MEMORY, or nullptr if the device
 *                   offers them all.
 */
static char const *
network_requirements_missing_extension (VkPhysicalDevice                          physical,
                                        PFN_vkEnumerateDeviceExtensionProperties  extensions,
                                        char const                               *also)
{
	// A failed call leaves the count undefined.
	uint32_t count = 0;
	if (extensions(physical, nullptr, &count, nullptr) != VK_SUCCESS)
		return NETWORK_REQUIREMENTS_NO_MEMORY;
	// One more than counted, so that a device that offers none is no special case.
	VkExtensionProperties *offered = calloc((size_t)count + 1, sizeof *offered);
	if (!offered)
		return NETWORK_REQUIREMENTS_NO_MEMORY;
	// VK_INCOMPLETE leaves out extensions beyond those counted, and says how many it wrote.
	VkResult const listed = extensions(physical, nullptr, &count, offered);
	char const *const missing = listed == VK_SUCCESS || listed == VK_INCOMPLETE
	                            ? network_requirements_unoffered(offered, count, also)
	                            : NETWORK_REQUIREMENTS_NO_MEMORY;
	free(offered);
	offered = nullptr;
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
static bool
network_requirements_lists_matrix (VkCooperativeMatrixPropertiesKHR const *listed,
                                   uint32_t                                count)
{
	for (uint32_t i = 0; i < count; ++i) {
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
static char const *
network_requirements_missing_matrix (VkPhysicalDevice                                      physical,
                                     PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR matrices)
{
	uint32_t count = 0;
	if (matrices(physical, &count, nullptr) != VK_SUCCESS || !count)
		return NETWORK_REQUIREMENTS_NO_MATRICES;
	VkCooperativeMatrixPropertiesKHR *listed = malloc(sizeof *listed * count);
	if (!listed)
		return NETWORK_REQUIREMENTS_NO_MEMORY;
	VkCooperativeMatrixPropertiesKHR const empty = {
		.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR
	};
	for (uint32_t i = 0; i < count; ++i)
		listed[i] = empty;
	// VK_INCOMPLETE leaves out configurations beyond those counted.
	VkResult const result = matrices(physical, &count, listed);
	bool const found = (result == VK_SUCCESS || result == VK_INCOMPLETE) &&
	                   network_requirements_lists_matrix(listed, count);
	free(listed);
	listed = nullptr;
	return found ? nullptr : NETWORK_REQUIREMENTS_NO_MATRICES;
}

char const *
network_requirements_unsupported (VkPhysicalDevice                                       physical,
                                  PFN_vkGetPhysicalDeviceProperties2                     properties2,
                                  PFN_vkGetPhysicalDeviceFeatures2                       features2,
                                  PFN_vkEnumerateDeviceExtensionProperties               extensions,
                                  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR  matrices,
                                  char const                                            *also)
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
	for (uint32_t i = 0; i < NETWORK_FEATURE_COUNT; ++i) {
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
