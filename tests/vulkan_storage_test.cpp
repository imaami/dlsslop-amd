#include "device_features.h"
#include "dlssnr/DlssNr_Shader_Vk.h"
#include "scaling/bcus_Shader_Vk.h"
#include "scaling/bcds_bicubic_Shader_Vk.h"
#include "scaling/bcds_catmull_Shader_Vk.h"
#include "scaling/bcds_lanczos2_Shader_Vk.h"
#include "scaling/bcds_lanczos3_Shader_Vk.h"
#include "scaling/bcds_kaiser2_Shader_Vk.h"
#include "scaling/bcds_kaiser3_Shader_Vk.h"
#include "scaling/bcds_magc_Shader_Vk.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

static void Check(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

// The layer loads these modules as they are built and enables only
// shaderStorageImageWriteWithoutFormat. Every storage image must be a
// formatless 2D image that is only written: a storage read or atomic would
// need shaderStorageImageReadWithoutFormat or a typed format too.
template<size_t N> static void Shader(const unsigned char (&blob)[N], const char* name) {
    constexpr uint32_t kOpCapability = 17, kOpTypeImage = 25, kOpImageTexelPointer = 60;
    constexpr uint32_t kOpImageRead = 98, kOpImageWrite = 99, kOpImageSparseRead = 320;
    constexpr uint32_t kReadWithoutFormat = 55, kWriteWithoutFormat = 56;
    Check(N >= 5 * sizeof(uint32_t) && N % sizeof(uint32_t) == 0, "invalid SPIR-V byte count");
    std::vector<uint32_t> words(N / sizeof(uint32_t));
    std::memcpy(words.data(), blob, N);
    Check(words[0] == 0x07230203 && words[4] == 0, "invalid SPIR-V header");
    unsigned writeCapabilities = 0, storage = 0, writes = 0;
    for (size_t i = 5; i < words.size(); i += words[i] >> 16) {
        const uint32_t count = words[i] >> 16, op = words[i] & 0xffffu;
        Check(count && count <= words.size() - i, "truncated SPIR-V instruction");
        if (op == kOpCapability) {
            Check(words[i + 1] != kReadWithoutFormat, "storage reads without format are not enabled");
            writeCapabilities += words[i + 1] == kWriteWithoutFormat;
        } else if (op == kOpTypeImage) {
            Check(count >= 9, "invalid OpTypeImage");
            if (words[i + 7] != 2) continue;
            // Dim2D, depth absent or unspecified, not arrayed or multisampled.
            Check(words[i + 3] == 1 && words[i + 4] != 1 && words[i + 4] <= 2 && !words[i + 5] &&
                  !words[i + 6], "unexpected storage image type");
            Check(words[i + 8] == 0, "typed storage image");
            ++storage;
        } else {
            Check(op != kOpImageRead && op != kOpImageSparseRead && op != kOpImageTexelPointer,
                  "storage read or atomic");
            writes += op == kOpImageWrite;
        }
    }
    Check(writeCapabilities == 1, "missing StorageImageWriteWithoutFormat capability");
    Check(storage && writes, "expected a write-only storage image shader");
    std::printf("%s: write-only formatless storage verified\n", name);
}

static void Features() {
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    VkPhysicalDeviceFeatures original{};
    original.robustBufferAccess = VK_TRUE;
    info.pEnabledFeatures = &original;
    device_features legacy{};
    Check(device_features_enable(&legacy, &info, false), "cannot enable legacy features");
    Check(info.pEnabledFeatures != &original && info.pEnabledFeatures->robustBufferAccess &&
          device_features_has_formatless_storage_writes(&info) && !original.shaderStorageImageWriteWithoutFormat,
          "legacy features were not preserved/copied");

    VkBaseOutStructure unknown{};
    unknown.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    VkPhysicalDeviceFeatures2 core{};
    core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    core.features.robustBufferAccess = VK_TRUE;
    core.pNext = &unknown;
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.pNext = &core;
    vulkan12.timelineSemaphore = VK_TRUE;
    VkLayerDeviceCreateInfo loader{};
    loader.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO;
    loader.function = VK_LOADER_DATA_CALLBACK;
    loader.pNext = &vulkan12;
    info.pNext = &loader;
    info.pEnabledFeatures = nullptr;
    device_features chained{};
    Check(device_features_enable(&chained, &info, false), "cannot enable known Features2 chain");
    auto* first = static_cast<const VkLayerDeviceCreateInfo*>(info.pNext);
    auto* second = static_cast<const VkPhysicalDeviceVulkan12Features*>(first->pNext);
    auto* third = static_cast<const VkPhysicalDeviceFeatures2*>(second->pNext);
    Check(first != &loader && second != &vulkan12 && third != &core && third->pNext == &unknown,
          "incorrect Features2 prefix/suffix ownership");
    Check(first->function == loader.function && second->timelineSemaphore &&
          third->features.robustBufferAccess && device_features_has_formatless_storage_writes(&info),
          "caller settings lost in prefix copy");
    Check(loader.pNext == &vulkan12 && vulkan12.pNext == &core && core.pNext == &unknown &&
          !core.features.shaderStorageImageWriteWithoutFormat, "caller chain was mutated");

    unknown.pNext = reinterpret_cast<VkBaseOutStructure*>(&core);
    core.pNext = nullptr;
    info.pNext = &unknown;
    device_features unsupported{};
    Check(!device_features_enable(&unsupported, &info, false) && info.pNext == &unknown &&
          !core.features.shaderStorageImageWriteWithoutFormat, "unknown prefix was not rejected intact");
    core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    Check(device_features_enable(&unsupported, &info, false) && info.pNext == &unknown,
          "already enabled unknown prefix was rejected");
    // Without a VkPhysicalDeviceFeatures2: pEnabledFeatures that enable formatless storage writes are
    // kept, and a request without pEnabledFeatures is given a copy of its own.
    VkPhysicalDeviceFeatures enough{};
    enough.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    VkDeviceCreateInfo kept{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    kept.pEnabledFeatures = &enough;
    device_features unchanged{};
    Check(device_features_enable(&unchanged, &kept, false) && kept.pEnabledFeatures == &enough && !kept.pNext,
          "pEnabledFeatures that enable formatless storage writes were replaced");
    VkDeviceCreateInfo bare{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_features added{};
    Check(device_features_enable(&added, &bare, false) && bare.pEnabledFeatures == &added.legacy && !bare.pNext &&
              device_features_has_formatless_storage_writes(&bare),
          "a request without pEnabledFeatures was not given formatless storage writes");
    std::puts("device features: private legacy/Features2 copies and unknown-prefix fallback verified");
}

// A chain vkCreateDevice accepts: no structure twice, and no core
// VkPhysicalDeviceVulkan1xFeatures beside a structure it replaces.
static bool Valid(const VkDeviceCreateInfo& info) {
    std::vector<VkStructureType> seen;
    for (const void* node = info.pNext; node; node = vk_chain_next(node)) {
        for (VkStructureType type : seen)
            if (type == vk_chain_type(node)) return false;
        seen.push_back(vk_chain_type(node));
    }
    for (const auto& f : NETWORK_FEATURES)
        if (f.core != NETWORK_FEATURE_NO_CORE && vk_chain_find(info.pNext, f.core) &&
            vk_chain_find(info.pNext, f.type))
            return false;
    return true;
}

// Every network feature set where the chain enables it.
static bool AllNetworkBits(const VkDeviceCreateInfo& info) {
    for (const auto& f : NETWORK_FEATURES) {
        const auto place = device_features_network_feature_in(info.pNext, &f);
        if (!place.node || !*vk_chain_bit(place.node, place.offset))
            return false;
    }
    return true;
}

// The network's extensions, in table order: explicit workgroup layout last.
static std::vector<const char*> NetworkExtensions() {
    std::vector<const char*> names;
    for (const auto& f : NETWORK_FEATURES)
        if (f.extension && (names.empty() || std::strcmp(names.back(), f.extension)))
            names.push_back(f.extension);
    return names;
}

static void NetworkFeatures() {
    // A DXVK-like chain: the core structures, one bit already on, an unknown suffix.
    VkBaseOutStructure unknown{};
    unknown.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &unknown};
    v13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    v12.vulkanMemoryModel = VK_TRUE;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    core.features.robustBufferAccess = VK_TRUE;
    VkLayerDeviceCreateInfo loader{};
    loader.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO;
    loader.function = VK_LAYER_LINK_INFO;
    loader.pNext = &core;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &loader};
    const auto chain = [&] { return std::vector<unsigned char>(reinterpret_cast<unsigned char*>(&loader),
                                                                reinterpret_cast<unsigned char*>(&loader + 1)); };
    const auto before = std::make_tuple(core.features.shaderStorageImageWriteWithoutFormat, v11.storageBuffer16BitAccess,
                                        v12.shaderInt8, v13.subgroupSizeControl);
    const auto loaderBytes = chain();
    device_features request{};
    Check(device_features_enable(&request, &info, true), "cannot add the network to a core-structure chain");
    Check(Valid(info) && AllNetworkBits(info) && device_features_has_formatless_storage_writes(&info),
          "network features missing from a core-structure chain");
    Check(vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO) == &unknown,
          "the unknown suffix was not left shared");
    Check(!vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES),
          "a core feature was given its own structure beside Vulkan12Features");
    Check(vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR) &&
          vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR),
          "extension structures were not added");
    const auto* copied13 = static_cast<const VkPhysicalDeviceVulkan13Features*>(
        vk_chain_find(info.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES));
    const auto* copied2 = device_features_core_features2(&info);
    Check(copied13 != &v13 && copied13->dynamicRendering && copied2 != &core && copied2->features.robustBufferAccess,
          "the game's own settings were lost in the copies");
    Check(before == std::make_tuple(core.features.shaderStorageImageWriteWithoutFormat, v11.storageBuffer16BitAccess,
                                    v12.shaderInt8, v13.subgroupSizeControl) &&
          loaderBytes == chain() && v12.vulkanMemoryModel && core.pNext == &v11 && v13.pNext == &unknown,
          "the game's chain was written");

    // The ledger reads what vkCreateDevice accepted: the extensions too.
    Check(!device_features_network_enabled(&info), "ledger ignored the missing extensions");
    const auto extensions = NetworkExtensions();
    info.enabledExtensionCount = uint32_t(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    Check(device_features_network_enabled(&info), "ledger missed an enabled network");
    // The network requires explicit workgroup layout too (network_requirements.h).
    --info.enabledExtensionCount;
    Check(!device_features_network_enabled(&info), "ledger ignored the missing explicit workgroup layout");

    // Structures of their own: one bit set in the game's, the rest added, none twice.
    VkPhysicalDeviceShaderFloat16Int8Features float16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    float16.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceFeatures legacy{};
    VkDeviceCreateInfo own{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &float16};
    own.pEnabledFeatures = &legacy;
    device_features separate{};
    Check(device_features_enable(&separate, &own, true) && Valid(own) && AllNetworkBits(own) &&
          device_features_has_formatless_storage_writes(&own) && own.pEnabledFeatures != &legacy,
          "cannot add the network beside the game's own structures");
    Check(!float16.shaderInt8 && !legacy.shaderStorageImageWriteWithoutFormat &&
          vk_chain_find(own.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) !=
              &float16,
          "the game's structures were written");

    // A structure to change behind one that cannot be copied declines, untouched.
    VkPhysicalDeviceVulkan12Features late{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkBaseOutStructure opaque{VK_STRUCTURE_TYPE_APPLICATION_INFO, reinterpret_cast<VkBaseOutStructure*>(&late)};
    VkDeviceCreateInfo hidden{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &opaque};
    device_features declined{};
    Check(!device_features_enable(&declined, &hidden, true) && hidden.pNext == &opaque && !late.shaderInt8 &&
          !hidden.pEnabledFeatures, "an uncopyable prefix was not declined intact");
    // With nothing to change behind it, the network's own structures go in front.
    for (const auto& f : NETWORK_FEATURES)
        if (f.core == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) *vk_chain_bit(&late, f.core_offset) = VK_TRUE;
    device_features ahead{};
    Check(device_features_enable(&ahead, &hidden, true) && Valid(hidden) && AllNetworkBits(hidden) &&
          vk_chain_find(hidden.pNext, VK_STRUCTURE_TYPE_APPLICATION_INFO) == &opaque,
          "the network's structures were not put ahead of an uncopyable chain");
    std::puts("device features: network features set in the game's structures or added, never twice, "
              "game chains unwritten, uncopyable prefixes declined, and the ledger");
}

// A device as network_requirements_unsupported sees it through the next layer's functions.
namespace stub {
uint32_t api = VK_API_VERSION_1_3;
uint32_t minimumSubgroup = 32, maximumSubgroup = 64;
std::vector<std::string> extensions;
VkResult filled = VK_SUCCESS;  // what the call that fills the extension list returns; an error writes nothing
const char* unsupported = nullptr;  // a feature the device lacks
// The cooperative-matrix configurations it lists, the last LATE of them only
// after the count was taken.
constexpr VkCooperativeMatrixPropertiesKHR kNetworkMatrix{
    VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR, nullptr, 16, 16, 16, VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT,
    VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_FALSE,
    VK_SCOPE_SUBGROUP_KHR};
std::vector<VkCooperativeMatrixPropertiesKHR> matrices;
uint32_t late = 0;
VkResult counted = VK_SUCCESS;  // what the count query returns

void VKAPI_CALL Properties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* properties) {
    properties->properties.apiVersion = api;
    auto* subgroup = static_cast<VkPhysicalDeviceSubgroupSizeControlProperties*>(
        vk_chain_find(properties->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES));
    if (!subgroup) return;
    subgroup->minSubgroupSize = minimumSubgroup;
    subgroup->maxSubgroupSize = maximumSubgroup;
    subgroup->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
}

void VKAPI_CALL Features2(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    for (const auto& f : NETWORK_FEATURES)
        if (void* node = vk_chain_find(features->pNext, f.type))
            *vk_chain_bit(node, f.offset) = !unsupported || std::strcmp(unsupported, f.name);
}

VkResult VKAPI_CALL Extensions(VkPhysicalDevice, const char*, uint32_t* count, VkExtensionProperties* properties) {
    if (properties) {
        if (filled != VK_SUCCESS) return filled;
        for (uint32_t i = 0; i < *count && i < extensions.size(); ++i)
            std::snprintf(properties[i].extensionName, sizeof properties[i].extensionName, "%s", extensions[i].c_str());
    }
    *count = uint32_t(extensions.size());
    return VK_SUCCESS;
}

VkResult VKAPI_CALL Matrices(VkPhysicalDevice, uint32_t* count, VkCooperativeMatrixPropertiesKHR* properties) {
    if (!properties) {
        *count = uint32_t(matrices.size()) - late;
        return counted;
    }
    *count = std::min(*count, uint32_t(matrices.size()));
    std::copy_n(matrices.begin(), *count, properties);
    return *count < matrices.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

const char* Unavailable(uint32_t instance) {
    return device_features_network_unavailable(VK_NULL_HANDLE, instance, Properties2, Features2, Extensions, Matrices);
}
}  // namespace stub

static bool Same(const char* a, const char* b) { return a && b && !std::strcmp(a, b); }

// A device that lists the network's FP8 configuration among others is given the
// network, also when it counts fewer configurations than it lists and returns
// VK_INCOMPLETE. One that lists it with any one field changed, or that cannot
// be asked, is refused, naming what it lacks.
static void NetworkMatrices() {
    using Matrix = VkCooperativeMatrixPropertiesKHR;
    Matrix fp16 = stub::kNetworkMatrix;
    fp16.AType = fp16.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    Matrix fp16Only = fp16;
    fp16Only.CType = fp16Only.ResultType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    stub::matrices = {fp16Only, fp16, stub::kNetworkMatrix};
    Check(!stub::Unavailable(VK_API_VERSION_1_3), "a device that lists the network's FP8 matrices was refused");
    stub::matrices = {stub::kNetworkMatrix, fp16, fp16Only};
    stub::late = 1;
    Check(!stub::Unavailable(VK_API_VERSION_1_3), "a device that counted fewer matrices than it lists was refused");
    stub::late = 0;
    void (*const changes[])(Matrix&) = {
        [](Matrix& m) { m.MSize = 8; },
        [](Matrix& m) { m.NSize = 8; },
        [](Matrix& m) { m.KSize = 32; },
        [](Matrix& m) { m.AType = VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT; },
        [](Matrix& m) { m.BType = VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT; },
        [](Matrix& m) { m.CType = VK_COMPONENT_TYPE_FLOAT16_KHR; },
        [](Matrix& m) { m.ResultType = VK_COMPONENT_TYPE_FLOAT16_KHR; },
        [](Matrix& m) { m.saturatingAccumulation = VK_TRUE; },
        [](Matrix& m) { m.scope = VK_SCOPE_WORKGROUP_KHR; },
    };
    for (const auto change : changes) {
        stub::matrices = {fp16Only, fp16, stub::kNetworkMatrix};
        change(stub::matrices.back());
        Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "e4m3 16x16x16 cooperative matrices"),
              "a device without the network's FP8 matrices was given the network");
    }
    stub::matrices = {};
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "e4m3 16x16x16 cooperative matrices"),
          "a device that lists no cooperative matrices was given the network");
    stub::matrices = {stub::kNetworkMatrix};
    stub::counted = VK_ERROR_OUT_OF_HOST_MEMORY;
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "e4m3 16x16x16 cooperative matrices"),
          "a device whose cooperative matrices cannot be counted was given the network");
    stub::counted = VK_SUCCESS;
    Check(Same(device_features_network_unavailable(VK_NULL_HANDLE, VK_API_VERSION_1_3, stub::Properties2,
                                                   stub::Features2, stub::Extensions, nullptr),
               "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"),
          "a device was given the network without a cooperative-matrix query");
}

static void NetworkAvailability() {
    const auto all = NetworkExtensions();
    stub::extensions.assign(all.begin(), all.end());
    stub::matrices = {stub::kNetworkMatrix};
    Check(!stub::Unavailable(VK_API_VERSION_1_3), "a capable device was refused the network");
    Check(Same(stub::Unavailable(VK_API_VERSION_1_2), "a Vulkan 1.3 instance") &&
          Same(device_features_network_unavailable(VK_NULL_HANDLE, VK_API_VERSION_1_3, nullptr, stub::Features2,
                                                   stub::Extensions, stub::Matrices), "a Vulkan 1.3 instance"),
          "a Vulkan 1.2 instance, or one without the 1.1 queries, was given the network");
    stub::api = VK_API_VERSION_1_2;
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "Vulkan 1.3"), "a Vulkan 1.2 device was given the network");
    stub::api = VK_API_VERSION_1_3;
    for (const auto& f : NETWORK_FEATURES) {
        stub::unsupported = f.name;
        Check(Same(stub::Unavailable(VK_API_VERSION_1_3), f.name), "a missing feature was not named");
    }
    stub::unsupported = nullptr;
    for (size_t i = 0; i < all.size(); ++i) {
        stub::extensions.assign(all.begin(), all.end());
        stub::extensions.erase(stub::extensions.begin() + i);
        Check(Same(stub::Unavailable(VK_API_VERSION_1_3), all[i]), "a missing extension was not named");
    }
    stub::extensions.clear();
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), all[0]) &&
              Same(network_requirements_unsupported(VK_NULL_HANDLE, stub::Properties2, stub::Features2,
                                                    stub::Extensions, stub::Matrices,
                                                    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
                   VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
          "a device that offers no extensions was not refused, naming the first one needed");
    stub::extensions.assign(all.begin(), all.end());
    stub::filled = VK_ERROR_OUT_OF_HOST_MEMORY;
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), NETWORK_REQUIREMENTS_NO_MEMORY),
          "a device whose extensions could not be listed was not refused for host memory");
    stub::filled = VK_SUCCESS;
    Check(Same(network_requirements_unsupported(VK_NULL_HANDLE, stub::Properties2, stub::Features2, stub::Extensions,
                                                stub::Matrices, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
               VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME),
          "the extension dlsslopd needs besides was not named");
    stub::minimumSubgroup = 64;
    Check(Same(stub::Unavailable(VK_API_VERSION_1_3), "32-lane compute subgroups"),
          "a device without 32-lane subgroups was given the network");
    stub::minimumSubgroup = 32;
    NetworkMatrices();

    // The network's extensions join a list once each, in the table's order, behind its own.
    const char* names[2 + NETWORK_FEATURE_COUNT] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, all[1]};
    const uint32_t named = network_requirements_append_extensions(names, 2);
    Check(named == 1 + all.size() && Same(names[0], VK_KHR_SWAPCHAIN_EXTENSION_NAME) && Same(names[1], all[1]) &&
              Same(names[2], all[0]) && Same(names[3], all[2]) &&
              network_requirements_append_extensions(names, named) == named,
          "the network's extensions were not appended once each, in order, behind the list's");

    // The network's extensions join the list once each, behind the game's.
    const char* game[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.enabledExtensionCount = 2;
    info.ppEnabledExtensionNames = game;
    const char* list[2 + NETWORK_FEATURE_COUNT] = {};
    Check(device_features_add_network_extensions(&info, list) && info.ppEnabledExtensionNames == list &&
          info.enabledExtensionCount == 1 + all.size() && list[0] == game[0] && list[1] == game[1] &&
          Same(list[2], all[1]) && Same(list[3], all[2]),
          "the network's extensions were not added once each behind the game's");
    // The request's own list, which already names them all.
    Check(!device_features_add_network_extensions(&info, list) && info.ppEnabledExtensionNames == list &&
          info.enabledExtensionCount == 1 + all.size() && list[0] == game[0],
          "extensions already listed were added again");
    // A request without extensions.
    VkDeviceCreateInfo bare{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    const char* only[NETWORK_FEATURE_COUNT] = {};
    Check(device_features_add_network_extensions(&bare, only) && bare.ppEnabledExtensionNames == only &&
          bare.enabledExtensionCount == all.size() && Same(only[0], all[0]),
          "the network's extensions were not added to a request without extensions");
    Check(!device_features_add_network_extensions(nullptr, only) && !device_features_add_network_extensions(&bare, nullptr),
          "extensions were added to no request or no list");

    // A request that declined the network still enables formatless storage writes.
    VkPhysicalDeviceVulkan12Features late{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkBaseOutStructure opaque{VK_STRUCTURE_TYPE_APPLICATION_INFO, reinterpret_cast<VkBaseOutStructure*>(&late)};
    VkPhysicalDeviceFeatures legacy{};
    VkDeviceCreateInfo hidden{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &opaque};
    hidden.pEnabledFeatures = &legacy;
    device_features request{};
    Check(!device_features_enable(&request, &hidden, true) && device_features_enable(&request, &hidden, false) &&
          device_features_has_formatless_storage_writes(&hidden) && hidden.pNext == &opaque && !late.shaderInt8,
          "a request that declined the network did not fall back to formatless storage alone");
    std::puts("device features: the network is refused for each missing requirement, its FP8 matrices included, "
              "its extensions are added once, and a declined request falls back");
}

// The chain that asks a device for the network's features, and that dlsslopd enables: one zeroed
// structure of each type, in this order behind VkPhysicalDeviceFeatures2.
static void FeatureChain() {
    const VkStructureType order[] = {
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
    network_feature_chain chain;
    std::memset(&chain, 0xa5, sizeof chain);
    network_feature_chain_init(&chain);
    const void* node = &chain.head;
    network_feature_chain rest = chain;
    for (VkStructureType type : order) {
        Check(node && vk_chain_type(node) == type && network_feature_chain_structure(&chain, type) == node,
              "the feature chain's structures are not linked in order");
        const auto offset = static_cast<const unsigned char*>(node) - reinterpret_cast<const unsigned char*>(&chain);
        std::memset(reinterpret_cast<unsigned char*>(&rest) + offset, 0, sizeof(VkBaseInStructure));
        node = vk_chain_next(node);
    }
    Check(!node, "the feature chain does not end after its last structure");
    const auto* bytes = reinterpret_cast<const unsigned char*>(&rest);
    Check(std::all_of(bytes, bytes + sizeof rest, [](unsigned char b) { return !b; }),
          "the feature chain was not zeroed besides its types and links");
    for (const auto& f : NETWORK_FEATURES)
        Check(network_feature_chain_bit(&chain, &f) == vk_chain_bit(vk_chain_find(&chain.head, f.type), f.offset) &&
                  !*network_feature_chain_bit(&chain, &f),
              "a feature's bit is not in its structure in the chain");
    Check(!network_feature_chain_structure(&chain, VK_STRUCTURE_TYPE_APPLICATION_INFO) &&
              !network_feature_chain_structure(nullptr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) &&
              !network_feature_chain_bit(nullptr, &NETWORK_FEATURES[0]),
          "a feature chain answered for a structure it does not have");
    network_feature_chain_init(nullptr);
    std::puts("network requirements: the feature chain is zeroed and linked in order");
}

// The copies' bound: a structure to change behind DEVICE_FEATURES_COPIES - 1 that need copies is
// changed in the last of DEVICE_FEATURES_COPIES copies; behind one more, the request declines,
// untouched. A valid chain repeats no structure but the loader's; this one is copyable all the same.
static void CopyBound() {
    for (const auto& c : DEVICE_FEATURES_COPYABLE)
        Check(c.size <= sizeof(device_features_copy) &&
                  std::count_if(std::begin(DEVICE_FEATURES_COPYABLE), std::end(DEVICE_FEATURES_COPYABLE),
                                [&c](const device_features_copyable& o) { return o.type == c.type; }) == 1,
              "a copyable structure does not fit a copy or is listed twice");
    VkBaseOutStructure tail{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    VkPhysicalDeviceFeatures2 core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &tail};
    std::vector<VkPhysicalDeviceVulkan11Features> ahead(DEVICE_FEATURES_COPIES,
                                                         {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES});
    for (size_t i = 0; i < ahead.size(); ++i) ahead[i].pNext = i + 1 < ahead.size() ? &ahead[i + 1] : (void*)&core;
    VkDeviceCreateInfo fits{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &ahead[1]};
    device_features request{};
    Check(device_features_enable(&request, &fits, false) && device_features_has_formatless_storage_writes(&fits),
          "a chain that needs DEVICE_FEATURES_COPIES copies was declined");
    uint32_t copies = 0;
    const void* node = fits.pNext;
    for (; node && node != &tail; node = vk_chain_next(node)) {
        const auto* bytes = static_cast<const unsigned char*>(node);
        const auto* first = reinterpret_cast<const unsigned char*>(&request);
        copies += bytes >= first && bytes < first + sizeof request;
    }
    Check(node == &tail && copies == DEVICE_FEATURES_COPIES && !core.features.shaderStorageImageWriteWithoutFormat,
          "a chain of DEVICE_FEATURES_COPIES copies was not copied up to its rest");
    VkDeviceCreateInfo over{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &ahead[0]};
    device_features declined{};
    Check(!device_features_enable(&declined, &over, false) && over.pNext == &ahead[0] && !over.pEnabledFeatures &&
              !core.features.shaderStorageImageWriteWithoutFormat,
          "a chain that needs more than DEVICE_FEATURES_COPIES copies was not declined intact");
    // Nothing to change: no copy, and the chain stays the game's.
    core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    Check(device_features_enable(&declined, &over, false) && over.pNext == &ahead[0],
          "a chain with nothing to change was copied");
    std::puts("device features: at most DEVICE_FEATURES_COPIES copies, and a longer prefix declined intact");
}

// The request asks for the in-layer network only on DLSSLOP_LAYER_NETWORK=1, and its functions
// answer nothing for no request.
static void Requests() {
    unsetenv("DLSSLOP_LAYER_NETWORK");
    const bool unset = device_features_network_requested();
    setenv("DLSSLOP_LAYER_NETWORK", "1", 1);
    const bool one = device_features_network_requested();
    setenv("DLSSLOP_LAYER_NETWORK", "0", 1);
    const bool zero = device_features_network_requested();
    setenv("DLSSLOP_LAYER_NETWORK", "11", 1);
    const bool eleven = device_features_network_requested();
    unsetenv("DLSSLOP_LAYER_NETWORK");
    Check(!unset && one && !zero && !eleven, "the in-layer network was asked for by another value than 1");
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_features request{};
    Check(!device_features_enable(nullptr, &info, false) && !device_features_enable(&request, nullptr, false) &&
              !device_features_network_enabled(nullptr) && !device_features_has_formatless_storage_writes(nullptr),
          "a request's functions answered for no request");
    std::puts("device features: DLSSLOP_LAYER_NETWORK=1 asks for the network, and no request has no features");
}

int main() {
    FeatureChain();
    CopyBound();
    Requests();
    Features();
    NetworkFeatures();
    NetworkAvailability();
#define SHADER(name) Shader(name##_spv, #name)
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
}
