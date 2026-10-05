/** @file
 *
 * What the HIP engine runs: run_engine.inc for hip_engine.h, which defines
 * run_hip_engine() (open.h).
 */
// SPDX-License-Identifier: MIT
#include "hip_engine.h"

#define ENGINE                   hip_engine
#define ENGINE_NAME              HIP_ENGINE_NAME
#define ENGINE_MAX_PASSES        HIP_ENGINE_MAX_PASSES
#define ENGINE_REBUILDS_FOR_TIER HIP_ENGINE_REBUILDS_FOR_TIER
#include "run_engine.inc"
