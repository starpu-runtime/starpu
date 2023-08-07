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

#ifndef __STARPU_PARALLEL_WORKERS_CREATE_H__
#define __STARPU_PARALLEL_WORKERS_CREATE_H__

/** @file */

#include <starpu.h>
#include <core/workers.h>
#include <common/list.h>
#include <string.h>
#include <omp.h>
#ifdef STARPU_MKL
#include <mkl_service.h>
#endif

#ifdef STARPU_PARALLEL_WORKER

#ifdef __cplusplus
extern
#endif

#pragma GCC visibility push(hidden)

struct starpu_parallel_worker_config
{
	unsigned id;
	hwloc_topology_t topology;
	unsigned nparallel_workers;
	unsigned ngroups;
	struct _starpu_parallel_worker_group_list *groups;
	struct _starpu_parallel_worker_parameters *params;
};

struct _starpu_parallel_worker_parameters
{
	int min_nb;
	int max_nb;
	int nb;
	char *sched_policy_name;
	struct starpu_sched_policy *sched_policy_struct;
	unsigned keep_homogeneous;
	unsigned prefere_min;
	void (*create_func)(void*);
	void *create_func_arg;
	int type;
	unsigned awake_workers;
};

LIST_TYPE(_starpu_parallel_worker_group,
	unsigned id;
	hwloc_obj_t group_obj;
	int nparallel_workers;
	struct _starpu_parallel_worker_list *parallel_workers;
	struct starpu_parallel_worker_config *father;
	struct _starpu_parallel_worker_parameters *params;
)

LIST_TYPE(_starpu_parallel_worker,
	unsigned id;
	hwloc_cpuset_t cpuset;
	int ncores;
	int *cores;
	int *workerids;
	struct _starpu_parallel_worker_group *father;
	struct _starpu_parallel_worker_parameters *params;
)

/** Machine discovery and parallel_worker creation main functions */
int _starpu_parallel_worker_config(hwloc_obj_type_t parallel_worker_level, struct starpu_parallel_worker_config *machine);
int _starpu_parallel_worker_topology(hwloc_obj_type_t parallel_worker_level, struct starpu_parallel_worker_config *machine);
void _starpu_parallel_worker_group(hwloc_obj_type_t parallel_worker_level, struct starpu_parallel_worker_config *machine);
void _starpu_parallel_worker(struct _starpu_parallel_worker_group *group);

/** Parameter functions */
void _starpu_parallel_worker_init_parameters(struct _starpu_parallel_worker_parameters *globals);
void _starpu_parallel_worker_copy_parameters(struct _starpu_parallel_worker_parameters *src, struct _starpu_parallel_worker_parameters *dst);
int _starpu_parallel_worker_analyze_parameters(struct _starpu_parallel_worker_parameters *params, int npus);

/** Parallel_Worker helper functions */
void _starpu_parallel_worker_init(struct _starpu_parallel_worker *parallel_worker, struct _starpu_parallel_worker_group *father);
int _starpu_parallel_worker_create(struct _starpu_parallel_worker *parallel_worker);

int _starpu_parallel_worker_bind(struct _starpu_parallel_worker *parallel_worker);
int _starpu_parallel_worker_remove(struct _starpu_parallel_worker_list *parallel_worker_list, struct _starpu_parallel_worker *parallel_worker);

/** Parallel_Worker group helper function */
void _starpu_parallel_worker_group_init(struct _starpu_parallel_worker_group *group, struct starpu_parallel_worker_config *father);
int _starpu_parallel_worker_group_create(struct _starpu_parallel_worker_group *group);
int _starpu_parallel_worker_group_remove(struct _starpu_parallel_worker_group_list *group_list, struct _starpu_parallel_worker_group *group);

/** Binding helpers */
void _starpu_parallel_worker_noop(void *buffers[], void *cl_arg)
{
	(void) buffers;
	(void) cl_arg;
}

static struct starpu_codelet _starpu_parallel_worker_bind_cl=
{
	.cpu_funcs = {_starpu_parallel_worker_noop},
	.nbuffers = 0,
	.name = "parallel_worker_internal_runtime_init"
};

typedef void (*starpu_binding_function)(void*);
starpu_binding_function _starpu_parallel_worker_type_get_func(enum starpu_parallel_worker_types type);

#pragma GCC visibility pop

#endif
#endif /* __STARPU_PARALLEL_WORKERS_CREATE_H__ */
