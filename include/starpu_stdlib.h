/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_STDLIB_H__
#define __STARPU_STDLIB_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Standard_Memory_Library Standard Memory Library
   @{
*/

/**
   Value passed to the function starpu_malloc_flags() to indicate the
   memory allocation should be pinned.
*/
#define STARPU_MALLOC_PINNED	((1ULL)<<1)

/**
   Value passed to the function starpu_malloc_flags() to indicate the
   memory allocation should be in the limit defined by the environment
   variables \ref STARPU_LIMIT_CUDA_devid_MEM, \ref
   STARPU_LIMIT_CUDA_MEM, \ref STARPU_LIMIT_OPENCL_devid_MEM, \ref
   STARPU_LIMIT_OPENCL_MEM and \ref STARPU_LIMIT_CPU_MEM (see Section
   \ref HowToLimitMemoryPerNode).
   If no memory is available, it tries to reclaim memory from StarPU.
   Memory allocated this way needs to be freed by calling the function
   starpu_free_flags() with the same flag.
*/
#define STARPU_MALLOC_COUNT	((1ULL)<<2)

/**
   Value passed to the function starpu_malloc_flags() along
   ::STARPU_MALLOC_COUNT to indicate that while the memory allocation
   should be kept in the limits defined for ::STARPU_MALLOC_COUNT, no
   reclaiming should be performed by starpu_malloc_flags() itself,
   thus potentially overflowing the memory node a bit. StarPU will
   reclaim memory after next task termination, according to the \ref
   STARPU_MINIMUM_AVAILABLE_MEM, \ref STARPU_TARGET_AVAILABLE_MEM,
   \ref STARPU_MINIMUM_CLEAN_BUFFERS, and \ref
   STARPU_TARGET_CLEAN_BUFFERS environment variables. If
   ::STARPU_MEMORY_WAIT is set, no overflowing will happen,
   starpu_malloc_flags() will wait for other eviction mechanisms to
   release enough memory.
*/
#define STARPU_MALLOC_NORECLAIM	((1ULL)<<3)

/**
   Value passed to starpu_memory_allocate() to specify that the
   function should wait for the requested amount of memory to become
   available, and atomically allocate it.
*/
#define STARPU_MEMORY_WAIT	((1ULL)<<4)

/**
   Value passed to starpu_memory_allocate() to specify that the
   function should allocate the amount of memory, even if that means
   overflowing the total size of the memory node.
*/
#define STARPU_MEMORY_OVERFLOW	((1ULL)<<5)

/**
   Value passed to the function starpu_malloc_flags() to indicate that
   when StarPU is using simgrid, the allocation can be "folded", i.e.
   a memory area is allocated, but its content is actually a replicate
   of the same memory area, to avoid having to actually allocate that
   much memory . This thus allows to have a memory area that does not
   actually consumes memory, to which one can read from and write to
   normally, but get bogus values.
*/
#define STARPU_MALLOC_SIMULATION_FOLDED	((1ULL)<<6)

/**
   @deprecated
   Equivalent to starpu_malloc(). This macro is provided to avoid
   breaking old codes.
*/
#define starpu_data_malloc_pinned_if_possible	starpu_malloc

/**
   @deprecated
   Equivalent to starpu_free(). This macro is provided to avoid
   breaking old codes.
*/
#define starpu_data_free_pinned_if_possible	starpu_free

/**
   Set an alignment constraints for starpu_malloc() allocations. \p
   align must be a power of two. This is for instance called
   automatically by the OpenCL driver to specify its own alignment
   constraints.
*/
void starpu_malloc_set_align(size_t align);

/**
   Allocate data of the given size \p dim in main memory, and return
   the pointer to the allocated data through \p A. It will also try to
   pin it in CUDA or OpenCL, so that data transfers from this buffer
   can be asynchronous, and thus permit data transfer and computation
   overlapping. The allocated buffer must be freed thanks to the
   starpu_free() function.
*/
int starpu_malloc(void **A, size_t dim);

/**
   Free memory which has previously been allocated with
   starpu_malloc().
*/
int starpu_free(void *A);

/**
   Perform a memory allocation based on the constraints defined by the
   given flag.
*/
int starpu_malloc_flags(void **A, size_t dim, int flags);

/**
   Free memory by specifying its size. The given flags should be
   consistent with the ones given to starpu_malloc_flags() when
   allocating the memory.
*/
int starpu_free_flags(void *A, size_t dim, int flags);

typedef int (*starpu_malloc_hook)(unsigned dst_node, void **A, size_t dim, int flags);
typedef int (*starpu_free_hook)(unsigned dst_node, void *A, size_t dim, int flags);

/**
   Set allocation functions to be used by StarPU. By default, StarPU
   will use \c malloc() (or \c cudaHostAlloc() if CUDA GPUs are used)
   for all its data handle allocations. The application can specify
   another allocation primitive by calling this. The malloc_hook
   should pass the allocated pointer through the \c A parameter, and
   return 0 on success. On allocation failure, it should return
   -ENOMEM. The \c flags parameter contains ::STARPU_MALLOC_PINNED if
   the memory should be pinned by the hook for GPU transfer
   efficiency. The hook can use starpu_memory_pin() to achieve this.
   The \c dst_node parameter is the starpu memory node, one can
   convert it to an hwloc logical id with
   starpu_memory_nodes_numa_id_to_hwloclogid() or to an OS NUMA number
   with starpu_memory_nodes_numa_devid_to_id().
*/
void starpu_malloc_set_hooks(starpu_malloc_hook malloc_hook, starpu_free_hook free_hook);

/**
   Pin the given memory area, so that CPU-GPU transfers can be done
   asynchronously with DMAs. The memory must be unpinned with
   starpu_memory_unpin() before being freed. Return 0 on success, -1
   on error.
*/
int starpu_memory_pin(void *addr, size_t size);

/**
   Unpin the given memory area previously pinned with
   starpu_memory_pin(). Return 0 on success, -1 on error.
*/
int starpu_memory_unpin(void *addr, size_t size);

/**
   If a memory limit is defined on the given node (see Section \ref
   HowToLimitMemoryPerNode), return the amount of total memory on the
   node. Otherwise return -1.
*/
starpu_ssize_t starpu_memory_get_total(unsigned node);

/**
   If a memory limit is defined on the given node (see Section \ref
   HowToLimitMemoryPerNode), return the amount of available memory on
   the node. Otherwise return -1.
*/
starpu_ssize_t starpu_memory_get_available(unsigned node);

/**
   Return the amount of total memory on all memory nodes for whose a
   memory limit is defined (see Section \ref HowToLimitMemoryPerNode).
*/
starpu_ssize_t starpu_memory_get_total_all_nodes(void);

/**
   Return the amount of available memory on all memory nodes for whose
   a memory limit is defined (see Section \ref
   HowToLimitMemoryPerNode).
*/
starpu_ssize_t starpu_memory_get_available_all_nodes(void);

/**
   If a memory limit is defined on the given node (see Section \ref
   HowToLimitMemoryPerNode), try to allocate some of it. This does not
   actually allocate memory, but only accounts for it. This can be
   useful when the application allocates data another way, but want
   StarPU to be aware of the allocation size e.g. for memory
   reclaiming.
   By default, return <c>-ENOMEM</c> if there is not enough room on
   the given node. \p flags can be either ::STARPU_MEMORY_WAIT or
   ::STARPU_MEMORY_OVERFLOW to change this.
*/
int starpu_memory_allocate(unsigned node, size_t size, int flags);

/**
   If a memory limit is defined on the given node (see Section \ref
   HowToLimitMemoryPerNode), free some of it. This does not actually
   free memory, but only accounts for it, like
   starpu_memory_allocate(). The amount does not have to be exactly
   the same as what was passed to starpu_memory_allocate(), only the
   eventual amount needs to be the same, i.e. one call to
   starpu_memory_allocate() can be followed by several calls to
   starpu_memory_deallocate() to declare the deallocation piece by
   piece.
*/
void starpu_memory_deallocate(unsigned node, size_t size);

/**
   If a memory limit is defined on the given node (see Section \ref
   HowToLimitMemoryPerNode), this will wait for \p size bytes to
   become available on \p node. Of course, since another thread may be
   allocating memory concurrently, this does not necessarily mean that
   this amount will be actually available, just that it was reached.
   To atomically wait for some amount of memory and reserve it,
   starpu_memory_allocate() should be used with the
   ::STARPU_MEMORY_WAIT flag.
*/
void starpu_memory_wait_available(unsigned node, size_t size);

void starpu_sleep(float nb_sec);
void starpu_usleep(float nb_micro_sec);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_STDLIB_H__ */
