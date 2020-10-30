/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_CLUSTERS_UTIL_H__
#define __STARPU_CLUSTERS_UTIL_H__

#include <starpu_config.h>

#ifdef STARPU_CLUSTER
#ifdef STARPU_HAVE_HWLOC

#include <hwloc.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Clustering_Machine Clustering Machine
   @{
 */

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_MIN_NB			(1<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_MAX_NB			(2<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_NB			(3<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_PREFERE_MIN		(4<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_KEEP_HOMOGENEOUS		(5<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_POLICY_NAME		(6<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_POLICY_STRUCT		(7<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_CREATE_FUNC		(8<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_CREATE_FUNC_ARG		(9<<STARPU_MODE_SHIFT)
/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_TYPE			(10<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_AWAKE_WORKERS		(11<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_PARTITION_ONE		(12<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_NEW			(13<<STARPU_MODE_SHIFT)

/**
   Used when calling starpu_cluster_machine
 */
#define STARPU_CLUSTER_NCORES			(14<<STARPU_MODE_SHIFT)

/**
   These represent the default available functions to enforce cluster
   use by the sub-runtime
*/
enum starpu_cluster_types
{
	STARPU_CLUSTER_OPENMP, /**< todo */
	STARPU_CLUSTER_INTEL_OPENMP_MKL,  /**< todo */
#ifdef STARPU_MKL
	STARPU_CLUSTER_GNU_OPENMP_MKL,  /**< todo */
#endif
};

struct starpu_cluster_machine;

struct starpu_cluster_machine* starpu_cluster_machine(hwloc_obj_type_t cluster_level, ...);
int starpu_uncluster_machine(struct starpu_cluster_machine* clusters);
int starpu_cluster_print(struct starpu_cluster_machine* clusters);

/** Prologue functions */
void starpu_openmp_prologue(void*);
#define starpu_intel_openmp_mkl_prologue starpu_openmp_prologue
#ifdef STARPU_MKL
void starpu_gnu_openmp_mkl_prologue(void*);
#endif /* STARPU_MKL */

/** @} */

#ifdef __cplusplus
}
#endif
#endif
#endif

#endif /* __STARPU_CLUSTERS_UTIL_H__ */
