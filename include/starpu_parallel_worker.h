/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_PARALLEL_WORKERS_UTIL_H__
#define __STARPU_PARALLEL_WORKERS_UTIL_H__

#include <starpu_config.h>

#ifdef STARPU_PARALLEL_WORKER
#ifdef STARPU_HAVE_HWLOC

#include <hwloc.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Parallel_Worker Using Parallel Workers
   @{
 */

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_MIN_NB (1 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_MAX_NB (2 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_NB (3 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_PREFERE_MIN (4 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_KEEP_HOMOGENEOUS (5 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_POLICY_NAME (6 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_POLICY_STRUCT (7 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_CREATE_FUNC (8 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_CREATE_FUNC_ARG (9 << STARPU_MODE_SHIFT)
/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_TYPE (10 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_AWAKE_WORKERS (11 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_PARTITION_ONE (12 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_NEW (13 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_parallel_worker_init()
 */
#define STARPU_PARALLEL_WORKER_NCORES (14 << STARPU_MODE_SHIFT)

/**
   These represent the default available functions to enforce parallel_worker
   use by the sub-runtime
*/
enum starpu_parallel_worker_types
{
	STARPU_PARALLEL_WORKER_OPENMP,		 /**< todo */
	STARPU_PARALLEL_WORKER_INTEL_OPENMP_MKL, /**< todo */
#ifdef STARPU_MKL
	STARPU_PARALLEL_WORKER_GNU_OPENMP_MKL, /**< todo */
#endif
};

/**
   Parallel_Worker configuration
 */
struct starpu_parallel_worker_config;

/**
   Create parallel_workers on the machine with the given parameters
 */
struct starpu_parallel_worker_config *starpu_parallel_worker_init(hwloc_obj_type_t parallel_worker_level, ...);

/**
   Delete the given parallel_workers configuration
 */
int starpu_parallel_worker_shutdown(struct starpu_parallel_worker_config *parallel_workers);

/**
   Print the given parallel_workers configuration
 */
int starpu_parallel_worker_print(struct starpu_parallel_worker_config *parallel_workers);

/** Prologue functions */
void starpu_openmp_prologue(void *);
#define starpu_intel_openmp_mkl_prologue starpu_openmp_prologue
#ifdef STARPU_MKL
void starpu_gnu_openmp_mkl_prologue(void *);
#endif /* STARPU_MKL */

/** @} */

#ifdef __cplusplus
}
#endif
#endif
#endif

#endif /* __STARPU_PARALLEL_WORKERS_UTIL_H__ */
