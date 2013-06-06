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
	STARPU_OPENCL_WORKER, /* OpenCL device */
	STARPU_MIC_WORKER,    /* Intel MIC device */
	STARPU_SCC_WORKER     /* Intel SCC device */
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
unsigned starpu_mic_worker_get_count(void);
unsigned starpu_scc_worker_get_count(void);

unsigned starpu_mic_device_get_count(void);

int starpu_worker_get_id(void);

int starpu_combined_worker_get_id(void);
int starpu_combined_worker_get_size(void);
int starpu_combined_worker_get_rank(void);

enum starpu_worker_archtype starpu_worker_get_type(int id);

int starpu_worker_get_count_by_type(enum starpu_worker_archtype type);

int starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);

int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);

void starpu_worker_get_name(int id, char *dst, size_t maxlen);

int starpu_worker_get_devid(int id);

int starpu_worker_get_mp_nodeid(int id);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_WORKER_H__ */

