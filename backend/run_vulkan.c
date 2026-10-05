/** @file
 *
 * What the Vulkan engine runs: run_engine.inc for vulkan_engine.h, which defines
 * run_vulkan_engine() (open.h).
 */
// SPDX-License-Identifier: MIT
#include "vulkan_engine.h"

#define ENGINE                   vulkan_engine
#define ENGINE_NAME              VULKAN_ENGINE_NAME
#define ENGINE_MAX_PASSES        VULKAN_ENGINE_MAX_PASSES
#define ENGINE_REBUILDS_FOR_TIER VULKAN_ENGINE_REBUILDS_FOR_TIER
#include "run_engine.inc"
