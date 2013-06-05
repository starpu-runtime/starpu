/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
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

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif


#ifdef __cplusplus
extern "C"
{
#endif

enum starpu_worker_archtype
{
	STARPU_ANY_WORKER,    /* any worker, used in the hypervisor */
	STARPU_CPU_WORKER,    /* CPU core */
	STARPU_CUDA_WORKER,   /* NVIDIA CUDA device */
	STARPU_OPENCL_WORKER  /* OpenCL device */
};

struct starpu_sched_ctx_iterator
{
	int cursor;
};

/* types of structures the worker collection can implement */
enum starpu_worker_collection_type
{
	STARPU_WORKER_LIST
};

/* generic structure used by the scheduling contexts to iterate the workers */
struct starpu_worker_collection
{
	/* hidden data structure used to memorize the workers */
	void *workerids;
	/* the number of workers in the collection */
	unsigned nworkers;
	/* the type of structure */
	enum starpu_worker_collection_type type;
	/* checks if there is another element in collection */
	unsigned (*has_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	/* return the next element in the collection */
	int (*get_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	/* add a new element in the collection */
	int (*add)(struct starpu_worker_collection *workers, int worker);
	/* remove an element from the collection */
	int (*remove)(struct starpu_worker_collection *workers, int worker);
	/* initialize the structure */
	void (*init)(struct starpu_worker_collection *workers);
	/* free the structure */
	void (*deinit)(struct starpu_worker_collection *workers);
	/* initialize the cursor if there is one */
	void (*init_iterator)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
};

/* This function returns the number of workers (ie. processing units executing
 * StarPU tasks). The returned value should be at most STARPU_NMAXWORKERS. */
unsigned starpu_worker_get_count(void);
unsigned starpu_combined_worker_get_count(void);
unsigned starpu_worker_is_combined_worker(int id);

unsigned starpu_cpu_worker_get_count(void);
unsigned starpu_cuda_worker_get_count(void);
unsigned starpu_opencl_worker_get_count(void);

/* Return the identifier of the thread in case this is associated to a worker.
 * This will return -1 if this function is called directly from the application
 * or if it is a worker in which a single thread controls different devices. */
int starpu_worker_get_id(void);

int starpu_combined_worker_get_id(void);
int starpu_combined_worker_get_size(void);
int starpu_combined_worker_get_rank(void);

/* This function returns the type of worker associated to an identifier (as
 * returned by the starpu_worker_get_id function). The returned value indicates
 * the architecture of the worker: STARPU_CPU_WORKER for a CPU core,
 * STARPU_CUDA_WORKER for a CUDA device. The value returned for an
 * invalid identifier is unspecified.  */
enum starpu_worker_archtype starpu_worker_get_type(int id);

/* Returns the number of workers of the type indicated by the argument. A
 * positive (or null) value is returned in case of success, -EINVAL indicates
 * that the type is not valid otherwise. */
int starpu_worker_get_count_by_type(enum starpu_worker_archtype type);

/* Fill the workerids array with the identifiers of the workers that have the
 * type indicated in the first argument. The maxsize argument indicates the
 * size of the workids array. The returned value gives the number of
 * identifiers that were put in the array. -ERANGE is returned is maxsize is
 * lower than the number of workers with the appropriate type: in that case,
 * the array is filled with the maxsize first elements. To avoid such
 * overflows, the value of maxsize can be chosen by the means of the
 * starpu_worker_get_count_by_type function, or by passing a value greater or
 * equal to STARPU_NMAXWORKERS. */
int starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

/* Return the identifier of the n-th worker of a specific type */
int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);

/* Return the identifier of the worker devid of a specific type */
int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);

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

#endif /* __STARPU_WORKER_H__ */

