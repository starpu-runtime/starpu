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
   Create parallel_workers on the machine with the given parameters.
   See \ref CreatingParallel for more details.
 */
struct starpu_parallel_worker_config *starpu_parallel_worker_init(hwloc_obj_type_t parallel_worker_level, ...);

/**
   Delete the given parallel_workers configuration
 */
int starpu_parallel_worker_shutdown(struct starpu_parallel_worker_config *parallel_workers);

/**
   Print the given parallel_workers configuration.
   See \ref CreatingParallel for more details.
 */
int starpu_parallel_worker_print(struct starpu_parallel_worker_config *parallel_workers);

/** Prologue functions */
void starpu_openmp_prologue(void *);
#define starpu_intel_openmp_mkl_prologue starpu_openmp_prologue
#ifdef STARPU_MKL
void starpu_gnu_openmp_mkl_prologue(void *);
#endif /* STARPU_MKL */

#define STARPU_CLUSTER_MIN_NB           STARPU_PARALLEL_WORKER_MIN_NB           /**< @deprecated Use ::STARPU_PARALLEL_WORKER_MIN_NB */
#define STARPU_CLUSTER_MAX_NB           STARPU_PARALLEL_WORKER_MAX_NB           /**< @deprecated Use ::STARPU_PARALLEL_WORKER_MAX_NB */
#define STARPU_CLUSTER_NB               STARPU_PARALLEL_WORKER_NB               /**< @deprecated Use ::STARPU_PARALLEL_WORKER_NB */
#define STARPU_CLUSTER_PREFERE_MIN      STARPU_PARALLEL_WORKER_PREFERE_MIN      /**< @deprecated Use ::STARPU_PARALLEL_WORKER_PREFERE_MIN */
#define STARPU_CLUSTER_KEEP_HOMOGENEOUS STARPU_PARALLEL_WORKER_KEEP_HOMOGENEOUS /**< @deprecated Use ::STARPU_PARALLEL_WORKER_KEEP_HOMOGENEOUS */
#define STARPU_CLUSTER_POLICY_NAME      STARPU_PARALLEL_WORKER_POLICY_NAME      /**< @deprecated Use ::STARPU_PARALLEL_WORKER_POLICY_NAME */
#define STARPU_CLUSTER_POLICY_STRUCT    STARPU_PARALLEL_WORKER_POLICY_STRUCT    /**< @deprecated Use ::STARPU_PARALLEL_WORKER_POLICY_STRUCT */
#define STARPU_CLUSTER_CREATE_FUNC      STARPU_PARALLEL_WORKER_CREATE_FUNC      /**< @deprecated Use ::STARPU_PARALLEL_WORKER_CREATE_FUNC */
#define STARPU_CLUSTER_CREATE_FUNC_ARG  STARPU_PARALLEL_WORKER_CREATE_FUNC_ARG  /**< @deprecated Use ::STARPU_PARALLEL_WORKER_CREATE_FUNC_ARG */
#define STARPU_CLUSTER_TYPE             STARPU_PARALLEL_WORKER_TYPE             /**< @deprecated Use ::STARPU_PARALLEL_WORKER_TYPE */
#define STARPU_CLUSTER_AWAKE_WORKERS    STARPU_PARALLEL_WORKER_AWAKE_WORKERS    /**< @deprecated Use ::STARPU_PARALLEL_WORKER_AWAKE_WORKERS */
#define STARPU_CLUSTER_PARTITION_ONE    STARPU_PARALLEL_WORKER_PARTITION_ONE    /**< @deprecated Use ::STARPU_PARALLEL_WORKER_PARTITION_ONE */
#define STARPU_CLUSTER_NEW              STARPU_PARALLEL_WORKER_NEW              /**< @deprecated Use ::STARPU_PARALLEL_WORKER_NEW */
#define STARPU_CLUSTER_NCORES           STARPU_PARALLEL_WORKER_NCORES           /**< @deprecated Use ::STARPU_PARALLEL_WORKER_NCORES */

/** @deprecated Use ::starpu_parallel_worker_types */
enum starpu_cluster_types
{
	STARPU_CLUSTER_OPENMP,		 /**< deprecated */
	STARPU_CLUSTER_INTEL_OPENMP_MKL, /**< deprecated */
#ifdef STARPU_MKL
	STARPU_CLUSTER_GNU_OPENMP_MKL,   /**< deprecated */
#endif
};
/** @deprecated Use starpu_parallel_worker_config */
struct starpu_cluster_machine;
/** @deprecated Use starpu_parallel_worker_init() */
struct starpu_cluster_machine *starpu_cluster_machine(hwloc_obj_type_t cluster_level, ...) STARPU_DEPRECATED;
/** @deprecated Use starpu_parallel_worker_shutdown() */
int starpu_uncluster_machine(struct starpu_cluster_machine *clusters) STARPU_DEPRECATED;
/** @deprecated Use starpu_parallel_worker_print() */
int starpu_cluster_print(struct starpu_cluster_machine *clusters) STARPU_DEPRECATED;

/** @} */

#ifdef __cplusplus
}
#endif
#endif
#endif

#endif /* __STARPU_PARALLEL_WORKERS_UTIL_H__ */
