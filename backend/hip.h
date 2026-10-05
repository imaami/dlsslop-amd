/** @file
 *
 * The HIP runtime as dlsslopd calls it: the few ROCm 7 types it passes, its entry points, loaded
 * with dlopen(), and code objects loaded from files. The build needs no ROCm headers or GPU; static
 * assertions pin the layouts. hip.c defines the functions.
 *
 * Plain C API, consumable from C++.
 */
// SPDX-License-Identifier: MIT
//
// The type layouts follow include/hip/hip_runtime_api.h of ROCm/HIP
// rocm-7.1.1 (https://github.com/ROCm/HIP), which carries this notice:
//
// Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef DLSSLOP_AMD_BACKEND_HIP_H_
#define DLSSLOP_AMD_BACKEND_HIP_H_

#ifdef __cplusplus
# include <cstddef>
#else
# include <stddef.h>
#endif

#include "error.h"

#ifdef __cplusplus
# define STD(x) std::x
extern "C" {
#else
# define STD(x) x
#endif

/** @brief hipDeviceProp_tR0600, as ROCm 6 and 7 lay it out, with only the fields dlsslopd reads
 *         named; the runtime writes all of it. */
struct hip_device_properties {
	char          name[256];        //!< The device's name.
	unsigned char reserved0[32];    //!< Fields dlsslopd does not read.
	STD(size_t)   totalGlobalMem;   //!< The device's memory, in bytes.
	unsigned char reserved1[288];   //!< Fields dlsslopd does not read.
	int           pciBusID;         //!< The PCI bus.
	int           pciDeviceID;      //!< The PCI device.
	int           pciDomainID;      //!< The PCI domain.
	unsigned char reserved2[564];   //!< Fields dlsslopd does not read.
	char          gcnArchName[256]; //!< The architecture, such as "gfx1201:sramecc+:xnack-".
	unsigned char reserved3[56];    //!< Fields dlsslopd does not read.
};

static_assert(offsetof(struct hip_device_properties, totalGlobalMem) == 288 &&
              offsetof(struct hip_device_properties, pciBusID) == 584 &&
              offsetof(struct hip_device_properties, pciDeviceID) == 588 &&
              offsetof(struct hip_device_properties, pciDomainID) == 592 &&
              offsetof(struct hip_device_properties, gcnArchName) == 1160 &&
              sizeof (struct hip_device_properties) == 1472,
              "HIP R0600 x64 ABI");

/** @brief hipExternalMemoryHandleDesc, as ROCm 7 lays it out. */
struct hip_memory_desc {
	int type; //!< hipExternalMemoryHandleType; 1 is hipExternalMemoryHandleTypeOpaqueFd.
	/** @brief The handle, as the type says. */
	union {
		int fd; //!< A descriptor, which the runtime takes on a successful import.
		/** @brief A Windows handle. */
		struct {
			void       *handle; //!< The handle.
			void const *name;   //!< Its name.
		} win32;
		void const *object; //!< An object.
	} handle;
	unsigned long long size;         //!< The memory's bytes.
	unsigned           flags;        //!< Flags.
	unsigned           reserved[16]; //!< Zeroes.
};

/** @brief hipExternalMemoryBufferDesc, as ROCm 7 lays it out. */
struct hip_buffer_desc {
	unsigned long long offset;       //!< Where the buffer starts in the memory.
	unsigned long long size;         //!< Its bytes.
	unsigned           flags;        //!< Flags.
	unsigned           reserved[16]; //!< Zeroes.
};

static_assert(sizeof (struct hip_memory_desc) == 104 && sizeof (struct hip_buffer_desc) == 88,
              "HIP external memory ABI");

/** @brief The loaded runtime: the entry points dlsslopd calls, each under its own name, null for an
 *         optional one that the runtime does not export. Every handle is a void *. The runtime
 *         stays loaded for the process: the driver's threads may outlive every user. */
struct hip_api {
	int          (*hipGetDevicePropertiesR0600)      (struct hip_device_properties *, int);
	int          (*hipInit)                          (unsigned);
	int          (*hipRuntimeGetVersion)             (int *);
	int          (*hipGetDeviceCount)                (int *);
	int          (*hipSetDevice)                     (int);
	int          (*hipSetDeviceFlags)                (unsigned);
	int          (*hipMalloc)                        (void **, STD(size_t));
	int          (*hipFree)                          (void *);
	int          (*hipHostMalloc)                    (void **, STD(size_t), unsigned);
	int          (*hipHostFree)                      (void *);
	int          (*hipHostRegister)                  (void *, STD(size_t), unsigned);
	int          (*hipHostUnregister)                (void *);
	int          (*hipMemcpy)                        (void *, void const *, STD(size_t), int);
	int          (*hipMemcpyAsync)                   (void *, void const *, STD(size_t), int, void *);
	int          (*hipMemsetAsync)                   (void *, int, STD(size_t), void *);
	int          (*hipMemGetInfo)                    (STD(size_t) *, STD(size_t) *);
	int          (*hipEventCreate)                   (void **);
	int          (*hipEventRecord)                   (void *, void *);
	int          (*hipEventElapsedTime)              (float *, void *, void *);
	int          (*hipEventDestroy)                  (void *);
	int          (*hipEventSynchronize)              (void *);
	int          (*hipStreamCreate)                  (void **);
	int          (*hipStreamSynchronize)             (void *);
	int          (*hipStreamDestroy)                 (void *);
	int          (*hipImportExternalMemory)          (void **, struct hip_memory_desc const *);
	int          (*hipExternalMemoryGetMappedBuffer) (void **, void *, struct hip_buffer_desc const *);
	int          (*hipDestroyExternalMemory)         (void *);
	int          (*hipModuleLoadData)                (void **, void const *);
	int          (*hipModuleGetFunction)             (void **, void *, char const *);
	int          (*hipModuleLaunchKernel)            (void *, unsigned, unsigned, unsigned, unsigned, unsigned,
	                                                  unsigned, unsigned, void *, void **, void **);
	int          (*hipModuleUnload)                  (void *);
	char const * (*hipGetErrorName)                  (int);
};

/** @brief struct hip_api's members in their order, as X(name, required). The runtime must export
 *         every required one. */
#define HIP_API_ENTRIES(X) \
	X(hipGetDevicePropertiesR0600, true) \
	X(hipInit, true) \
	X(hipRuntimeGetVersion, true) \
	X(hipGetDeviceCount, true) \
	X(hipSetDevice, true) \
	/* Without it, waiting threads spin. */ \
	X(hipSetDeviceFlags, false) \
	X(hipMalloc, true) \
	X(hipFree, true) \
	X(hipHostMalloc, true) \
	/* The GPU codec needs it. */ \
	X(hipHostFree, false) \
	/* Without them, shared memory stays pageable. */ \
	X(hipHostRegister, false) \
	X(hipHostUnregister, false) \
	X(hipMemcpy, true) \
	X(hipMemcpyAsync, true) \
	X(hipMemsetAsync, true) \
	/* For the network's memory report. */ \
	X(hipMemGetInfo, true) \
	X(hipEventCreate, true) \
	X(hipEventRecord, true) \
	X(hipEventElapsedTime, true) \
	X(hipEventDestroy, true) \
	X(hipEventSynchronize, true) \
	X(hipStreamCreate, true) \
	X(hipStreamSynchronize, true) \
	X(hipStreamDestroy, true) \
	X(hipImportExternalMemory, true) \
	X(hipExternalMemoryGetMappedBuffer, true) \
	X(hipDestroyExternalMemory, true) \
	X(hipModuleLoadData, true) \
	X(hipModuleGetFunction, true) \
	X(hipModuleLaunchKernel, true) \
	X(hipModuleUnload, true) \
	X(hipGetErrorName, true)

/** @brief Loads the runtime: DLSSLOP_HIP_LIBRARY when it is set and not empty, else the
 *         libamdhip64 sonames, then /opt/rocm/lib's.
 *
 * @param dest Receives the runtime; unchanged on a failure.
 * @param e    Receives every candidate's loader error, or the missing export, or nullptr.
 * @return     ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_load (struct hip_api *dest,
          struct error   *e);

/** @brief Loads a code object from a file into the current device's context.
 *
 * @param api    The runtime.
 * @param path   The file.
 * @param module Receives the module, which hipModuleUnload() frees; unchanged on a failure.
 * @param e      Receives the words for what stopped it, or nullptr.
 * @return       ERROR_NONE, or ERROR_FAILED.
 */
extern enum error_code
hip_load_module (struct hip_api const  *api,
                 char const            *path,
                 void                 **module,
                 struct error          *e);

/** @brief Puts the words of a failed runtime call in an error: "WHAT: NAME (RESULT)", with the
 *         name that hipGetErrorName() gives the result.
 *
 * A caller tests the result itself and calls this only for an error, so that the frame path keeps
 * its test and only the words are out of line.
 *
 * @param e      The error, or nullptr where no caller needs the words.
 * @param api    The runtime.
 * @param result The call's nonzero result.
 * @param fmt    A printf format for what the call did.
 * @param ...    The format's arguments.
 * @return       ERROR_FAILED.
 */
[[gnu::cold, gnu::format(printf, 4, 5)]]
extern enum error_code
hip_fail (struct error         *e,
          struct hip_api const *api,
          int                   result,
          char const           *fmt,
          ...);

#undef STD

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLSSLOP_AMD_BACKEND_HIP_H_ */
