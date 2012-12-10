/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_WORKER_H__
#define __STARPU_WORKER_H__

#include <stdlib.h>
#include <starpu_config.h>

#ifdef __cplusplus
extern "C"
{
#endif

enum starpu_archtype
{
#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	STARPU_ANY_WORKER, /* any worker, used in the hypervisor */
#endif
	STARPU_CPU_WORKER,    /* CPU core */
	STARPU_CUDA_WORKER,   /* NVIDIA CUDA device */
	STARPU_OPENCL_WORKER, /* OpenCL device */
	STARPU_GORDON_WORKER  /* Cell SPU */
};

/* This function returns the number of workers (ie. processing units executing
 * StarPU tasks). The returned value should be at most STARPU_NMAXWORKERS. */
unsigned starpu_worker_get_count(void);
unsigned starpu_combined_worker_get_count(void);
unsigned starpu_worker_is_combined_worker(int id);

unsigned starpu_cpu_worker_get_count(void);
unsigned starpu_cuda_worker_get_count(void);
unsigned starpu_spu_worker_get_count(void);
unsigned starpu_opencl_worker_get_count(void);

/* Return the identifier of the thread in case this is associated to a worker.
 * This will return -1 if this function is called directly from the application
 * or if it is some SPU worker where a single thread controls different SPUs. */
int starpu_worker_get_id(void);

int starpu_combined_worker_get_id(void);
int starpu_combined_worker_get_size(void);
int starpu_combined_worker_get_rank(void);


/* This function returns the type of worker associated to an identifier (as
 * returned by the starpu_worker_get_id function). The returned value indicates
 * the architecture of the worker: STARPU_CPU_WORKER for a CPU core,
 * STARPU_CUDA_WORKER for a CUDA device, and STARPU_GORDON_WORKER for a Cell
 * SPU. The value returned for an invalid identifier is unspecified.  */
enum starpu_archtype starpu_worker_get_type(int id);

/* Returns the number of workers of the type indicated by the argument. A
 * positive (or null) value is returned in case of success, -EINVAL indicates
 * that the type is not valid otherwise. */
int starpu_worker_get_count_by_type(enum starpu_archtype type);

/* Fill the workerids array with the identifiers of the workers that have the
 * type indicated in the first argument. The maxsize argument indicates the
 * size of the workids array. The returned value gives the number of
 * identifiers that were put in the array. -ERANGE is returned is maxsize is
 * lower than the number of workers with the appropriate type: in that case,
 * the array is filled with the maxsize first elements. To avoid such
 * overflows, the value of maxsize can be chosen by the means of the
 * starpu_worker_get_count_by_type function, or by passing a value greater or
 * equal to STARPU_NMAXWORKERS. */
int starpu_worker_get_ids_by_type(enum starpu_archtype type, int *workerids, int maxsize);

/* StarPU associates a unique human readable string to each processing unit.
 * This function copies at most the "maxlen" first bytes of the unique
 * string associated to a worker identified by its identifier "id" into
 * the "dst" buffer. The caller is responsible for ensuring that the
 * "dst" is a valid pointer to a buffer of "maxlen" bytes at least.
 * Calling this function on an invalid identifier results in an unspecified
 * behaviour. */
void starpu_worker_get_name(int id, char *dst, size_t maxlen);

/* This functions returns the device id of the worker associated to an
 *  identifier (as returned by the starpu_worker_get_id() function)
 */
int starpu_worker_get_devid(int id);

#ifdef __cplusplus
}
#endif

#ifdef STARPU_DEVEL
#warning do we really need both starpu_worker_set_sched_condition and starpu_worker_init_sched_condition functions
#endif

void starpu_worker_set_sched_condition(unsigned sched_ctx_id, int workerid, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond);

void starpu_worker_get_sched_condition(unsigned sched_ctx_id, int workerid, pthread_mutex_t **sched_mutex, pthread_cond_t **sched_cond);

void starpu_worker_init_sched_condition(unsigned sched_ctx_id, int workerid);

void starpu_worker_deinit_sched_condition(unsigned sched_ctx_id, int workerid);

#endif /* __STARPU_WORKER_H__ */

