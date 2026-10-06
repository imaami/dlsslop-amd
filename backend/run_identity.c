/** @file
 *
 * What the identity engine runs: run_engine.inc for identity_engine.h, which defines
 * run_identity_engine() (open.h).
 */
// SPDX-License-Identifier: MIT
#include "identity_engine.h"

#define ENGINE                   identity_engine
#define ENGINE_NAME              IDENTITY_ENGINE_NAME
#define ENGINE_MAX_PASSES        IDENTITY_ENGINE_MAX_PASSES
#define ENGINE_REBUILDS_FOR_TIER IDENTITY_ENGINE_REBUILDS_FOR_TIER
#define ENGINE_NEURAL            IDENTITY_ENGINE_NEURAL
#include "run_engine.inc"
