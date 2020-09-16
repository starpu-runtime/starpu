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

#ifndef __STARPU_CLUSTERS_CREATE_H__
#define __STARPU_CLUSTERS_CREATE_H__

/** @file */

#include <starpu.h>
#include <core/workers.h>
#include <common/list.h>
#include <string.h>
#include <omp.h>
#ifdef STARPU_MKL
#include <mkl_service.h>
#endif

#ifdef STARPU_CLUSTER

#ifdef __cplusplus
extern
#endif

struct starpu_cluster_machine
{
	unsigned id;
	hwloc_topology_t topology;
	unsigned nclusters;
	unsigned ngroups;
	struct _starpu_cluster_group_list* groups;
	struct _starpu_cluster_parameters* params;
};

struct _starpu_cluster_parameters
{
	int min_nb;
	int max_nb;
	int nb;
	char* sched_policy_name;
	struct starpu_sched_policy* sched_policy_struct;
	unsigned keep_homogeneous;
	unsigned prefere_min;
	void (*create_func)(void*);
	void* create_func_arg;
	int type;
	unsigned awake_workers;
};

LIST_TYPE(_starpu_cluster_group,
	unsigned id;
	hwloc_obj_t group_obj;
	int nclusters;
	struct _starpu_cluster_list* clusters;
	struct starpu_cluster_machine* father;
	struct _starpu_cluster_parameters* params;
)

LIST_TYPE(_starpu_cluster,
	unsigned id;
	hwloc_cpuset_t cpuset;
	int ncores;
	int* cores;
	int* workerids;
	struct _starpu_cluster_group* father;
	struct _starpu_cluster_parameters* params;
)


/** Machine discovery and cluster creation main funcitons */
int _starpu_cluster_machine(hwloc_obj_type_t cluster_level,
			     struct starpu_cluster_machine* machine);
int _starpu_cluster_topology(hwloc_obj_type_t cluster_level,
			      struct starpu_cluster_machine* machine);
void _starpu_cluster_group(hwloc_obj_type_t cluster_level,
			   struct starpu_cluster_machine* machine);
void _starpu_cluster(struct _starpu_cluster_group* group);

/** Parameter functions */
void _starpu_cluster_init_parameters(struct _starpu_cluster_parameters* globals);
void _starpu_cluster_copy_parameters(struct _starpu_cluster_parameters* src,
				     struct _starpu_cluster_parameters* dst);
int _starpu_cluster_analyze_parameters(struct _starpu_cluster_parameters* params, int npus);

/** Cluster helper functions */
void _starpu_cluster_init(struct _starpu_cluster* cluster, struct _starpu_cluster_group* father);
void _starpu_cluster_create(struct _starpu_cluster* cluster);

int _starpu_cluster_bind(struct _starpu_cluster* cluster);
int _starpu_cluster_remove(struct _starpu_cluster_list* cluster_list,
			   struct _starpu_cluster* cluster);

/** Cluster group helper function */
void _starpu_cluster_group_init(struct _starpu_cluster_group* group,
				struct starpu_cluster_machine* father);
void _starpu_cluster_group_create(struct _starpu_cluster_group* group);
int _starpu_cluster_group_remove(struct _starpu_cluster_group_list* group_list,
				 struct _starpu_cluster_group* group);

/** Binding helpers */
void _starpu_cluster_noop(void* buffers[], void* cl_arg)
{
	(void) buffers;
	(void) cl_arg;
}

static struct starpu_codelet _starpu_cluster_bind_cl=
{
	.cpu_funcs = {_starpu_cluster_noop},
	.nbuffers = 0,
	.name = "cluster_internal_runtime_init"
};

typedef void (*starpu_binding_function)(void*);
starpu_binding_function _starpu_cluster_type_get_func(enum starpu_cluster_types type);

#endif
#endif /* __STARPU_CLUSTERS_CREATE_H__ */
