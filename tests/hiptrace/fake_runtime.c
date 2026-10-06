/** @file
 *
 * A HIP runtime stand-in for the tracing runtime's offline test (hiptrace-test.py). Device and pinned
 * host memory are zeroed host memory, so copies, memsets and the tracing runtime's hashes work on
 * them; modules, functions, streams and events are distinct addresses; every call it exports
 * succeeds. It exports no graph entry points: the tracing runtime reports them missing and fails
 * their calls.
 */
// SPDX-License-Identifier: MIT
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/** @brief hipErrorOutOfMemory. */
static constexpr int OUT_OF_MEMORY = 2;

/** @brief The bytes whose addresses are the handles. */
static char handles[256];

/** @brief The handles given so far. */
static size_t next_handle;

/** @brief A new handle: the next byte's address in handles, from the first again after the last.
 *
 * @return The handle.
 */
static void *
handle (void)
{
	return &handles[next_handle++ % sizeof handles];
}

/** @brief Gives a new module. */
int
hipModuleLoadData (void       **module,
                   void const  *image)
{
	*module = handle();
	return 0;
}

/** @brief Gives a new function. */
int
hipModuleGetFunction (void       **function,
                      void        *module,
                      char const  *name)
{
	*function = handle();
	return 0;
}

/** @brief Launches nothing. */
int
hipModuleLaunchKernel (void      *f,
                       unsigned   gx,
                       unsigned   gy,
                       unsigned   gz,
                       unsigned   bx,
                       unsigned   by,
                       unsigned   bz,
                       unsigned   shared,
                       void      *stream,
                       void     **params,
                       void     **extra)
{
	return 0;
}

/** @brief Launches nothing. */
int
hipExtModuleLaunchKernel (void      *f,
                          unsigned   gx,
                          unsigned   gy,
                          unsigned   gz,
                          unsigned   lx,
                          unsigned   ly,
                          unsigned   lz,
                          size_t     shared,
                          void      *stream,
                          void     **params,
                          void     **extra,
                          void      *start,
                          void      *stop,
                          unsigned   flags)
{
	return 0;
}

/** @brief Gives a new stream. */
int
hipStreamCreate (void **stream)
{
	*stream = handle();
	return 0;
}

/** @brief Waits for nothing. */
int
hipStreamSynchronize (void *stream)
{
	return 0;
}

/** @brief Gives a new event. */
int
hipEventCreate (void **event)
{
	*event = handle();
	return 0;
}

/** @brief Records nothing. */
int
hipEventRecord (void *event,
                void *stream)
{
	return 0;
}

/** @brief Gives 0 ms. */
int
hipEventElapsedTime (float *ms,
                     void  *begin,
                     void  *end)
{
	*ms = 0;
	return 0;
}

/** @brief Destroys nothing. */
int
hipEventDestroy (void *event)
{
	return 0;
}

/** @brief Gives zeroed host memory as device memory. */
int
hipMalloc (void   **pointer,
           size_t   bytes)
{
	*pointer = calloc(1, bytes);
	return *pointer ? 0 : OUT_OF_MEMORY;
}

/** @brief Frees what hipMalloc() gave. */
int
hipFree (void *pointer)
{
	free(pointer);
	pointer = nullptr;
	return 0;
}

/** @brief Gives zeroed host memory. */
int
hipHostMalloc (void     **pointer,
               size_t     bytes,
               unsigned   flags)
{
	*pointer = calloc(1, bytes);
	return *pointer ? 0 : OUT_OF_MEMORY;
}

/** @brief Frees what hipHostMalloc() gave. */
int
hipHostFree (void *pointer)
{
	free(pointer);
	pointer = nullptr;
	return 0;
}

/** @brief Registers nothing. */
int
hipHostRegister (void     *pointer,
                 size_t    bytes,
                 unsigned  flags)
{
	return 0;
}

/** @brief Unregisters nothing. */
int
hipHostUnregister (void *pointer)
{
	return 0;
}

/** @brief Copies on the host. */
int
hipMemcpy (void       *dst,
           void const *src,
           size_t      bytes,
           int         kind)
{
	memcpy(dst, src, bytes);
	return 0;
}

/** @brief Copies on the host, at once. */
int
hipMemcpyAsync (void       *dst,
                void const *src,
                size_t      bytes,
                int         kind,
                void       *stream)
{
	memcpy(dst, src, bytes);
	return 0;
}

/** @brief Fills on the host, at once. */
int
hipMemsetAsync (void   *dst,
                int     value,
                size_t  bytes,
                void   *stream)
{
	memset(dst, value, bytes);
	return 0;
}

/** @brief Names every error alike. */
char const *
hipGetErrorName (int error)
{
	return "hipErrorFake";
}
