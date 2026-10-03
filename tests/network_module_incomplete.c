/** @file
 *
 * A library with every function of the in-layer network's module but dlsslop_network_close(), for
 * network-module-test: the loader refuses it and keeps no pointer into it.
 */
#include "network_module.h"

/** @brief Opens no network. */
struct DlsslopNetwork *
dlsslop_network_open (struct dlsslop_network_device const *)
{
	return nullptr;
}

/** @brief Fails every frame. */
enum dlsslop_network_state
dlsslop_network_prepare (struct DlsslopNetwork *, struct ShmHeader const *, uint32_t, uint32_t, bool)
{
	return DLSSLOP_NETWORK_FAILED;
}

/** @brief Records no frame. */
enum dlsslop_network_state
dlsslop_network_record (struct DlsslopNetwork *, VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, bool)
{
	return DLSSLOP_NETWORK_FAILED;
}

/** @brief Ignores a submit. */
void
dlsslop_network_submitted (struct DlsslopNetwork *)
{
}

/** @brief Says why every frame failed. */
char const *
dlsslop_network_error (struct DlsslopNetwork const *)
{
	return "incomplete module";
}
