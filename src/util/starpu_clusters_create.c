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

/* This file creates an interface to manage clustering resources and make use
 * of parallel tasks. It entirely depends on the hwloc software. */

#include <util/starpu_clusters_create.h>

#ifdef STARPU_CLUSTER

starpu_binding_function _starpu_cluster_type_get_func(enum starpu_cluster_types type)
{
	starpu_binding_function prologue_func;

	switch (type)
	{
	case STARPU_CLUSTER_OPENMP:
		prologue_func = &starpu_openmp_prologue;
		break;
	case STARPU_CLUSTER_INTEL_OPENMP_MKL:
		prologue_func = &starpu_intel_openmp_mkl_prologue;
		break;
#ifdef STARPU_MKL
	case STARPU_CLUSTER_GNU_OPENMP_MKL:
		prologue_func = &starpu_gnu_openmp_mkl_prologue;
		break;
#endif
	default:
		prologue_func = NULL;
	}

	return prologue_func;
}

void starpu_openmp_prologue(void* arg)
{
	(void) arg;
	int workerid = starpu_worker_get_id_check();

	if (starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
	{
		struct starpu_task *task = starpu_task_get_current();
		int sched_ctx = task->sched_ctx;
		struct _starpu_sched_ctx *ctx_struct = _starpu_get_sched_ctx_struct(sched_ctx);
		/* If the view of the worker doesn't correspond to the view of the task,
			 adapt the thread team */
		if (ctx_struct->parallel_view != task->possibly_parallel)
		{
			int *cpuids = NULL;
			int ncpuids = 0;

			starpu_sched_ctx_get_available_cpuids(sched_ctx, &cpuids, &ncpuids);
			if (!task->possibly_parallel)
				ncpuids=1;
			omp_set_num_threads(ncpuids);
#pragma omp parallel
			{
				starpu_sched_ctx_bind_current_thread_to_cpuid(cpuids[omp_get_thread_num()]);
			}
			free(cpuids);
			ctx_struct->parallel_view = !ctx_struct->parallel_view;
		}
	}
	return;
}

#ifdef STARPU_MKL
void starpu_gnu_openmp_mkl_prologue(void* arg)
{
	int workerid = starpu_worker_get_id();

	if (starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
	{
		struct starpu_task *task = starpu_task_get_current();
		int sched_ctx = task->sched_ctx;
		struct _starpu_sched_ctx *ctx_struct = _starpu_get_sched_ctx_struct(sched_ctx);
		/* If the view of the worker doesn't correspond to the view of the task,
			 adapt the thread team */
		if (ctx_struct->parallel_view != task->possibly_parallel)
		{
			int *cpuids = NULL;
			int ncpuids = 0;

			starpu_sched_ctx_get_available_cpuids(sched_ctx, &cpuids, &ncpuids);
			if (!task->possibly_parallel)
				ncpuids=1;
			omp_set_num_threads(ncpuids);
			mkl_set_num_threads(ncpuids);
			mkl_set_dynamic(0);
#pragma omp parallel
			{
				starpu_sched_ctx_bind_current_thread_to_cpuid(cpuids[omp_get_thread_num()]);
			}
			free(cpuids);
			ctx_struct->parallel_view = !ctx_struct->parallel_view;
		}
	}
	return;
}
#endif

/* Main interface function to create a cluster view of the machine.
 * Its job is to capture what the user wants and store it in a standard view. */
struct starpu_cluster_machine *starpu_cluster_machine(hwloc_obj_type_t cluster_level, ...)
{
	va_list varg_list;
	int arg_type;
	struct _starpu_cluster_parameters *params;
	struct starpu_cluster_machine *machine;
	_STARPU_MALLOC(machine, sizeof(struct starpu_cluster_machine));

	_STARPU_MALLOC(machine->params, sizeof(struct _starpu_cluster_parameters));
	machine->id = STARPU_NMAX_SCHED_CTXS;
	machine->groups = _starpu_cluster_group_list_new();
	machine->nclusters = 0;
	machine->ngroups = 0;
	machine->topology = NULL;

	_starpu_cluster_init_parameters(machine->params);
	params = machine->params;

	va_start(varg_list, cluster_level);
	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type == STARPU_CLUSTER_MIN_NB)
		{
			params->min_nb = va_arg(varg_list, int);
			if (params->min_nb <= 0)
				_STARPU_DISP("Caution min number of contexts shouldn't be negative or null\n");
		}
		else if (arg_type == STARPU_CLUSTER_MAX_NB)
		{
			params->max_nb = va_arg(varg_list, int);
			if (params->max_nb <= 0)
				_STARPU_DISP("Caution max number of contexts shouldn't be negative or null\n");
		}
		else if (arg_type == STARPU_CLUSTER_NB)
		{
			params->nb = va_arg(varg_list, int);
			if (params->nb <= 0)
				_STARPU_DISP("Caution number of contexts shouldn't be negative or null\n");
		}
		else if (arg_type == STARPU_CLUSTER_POLICY_NAME)
		{
			params->sched_policy_name = va_arg(varg_list, char*);
		}
		else if (arg_type == STARPU_CLUSTER_POLICY_STRUCT)
		{
			params->sched_policy_struct = va_arg(varg_list,
							     struct starpu_sched_policy*);
		}
		else if (arg_type == STARPU_CLUSTER_KEEP_HOMOGENEOUS)
		{
			params->keep_homogeneous = va_arg(varg_list, int); /* 0=off, other=on */
		}
		else if (arg_type == STARPU_CLUSTER_PREFERE_MIN)
		{
			params->prefere_min = va_arg(varg_list, int); /* 0=off, other=on */
		}
		else if (arg_type == STARPU_CLUSTER_CREATE_FUNC)
		{
			params->create_func = va_arg(varg_list, void (*)(void*));
		}
		else if (arg_type == STARPU_CLUSTER_CREATE_FUNC_ARG)
		{
			params->create_func_arg = va_arg(varg_list, void*);
		}
		else if (arg_type == STARPU_CLUSTER_TYPE)
		{
			params->type = va_arg(varg_list, enum starpu_cluster_types);
		}
		else if (arg_type == STARPU_CLUSTER_AWAKE_WORKERS)
		{
			params->awake_workers = va_arg(varg_list, unsigned);
		}
		else if (arg_type == STARPU_CLUSTER_PARTITION_ONE)
		{
			struct _starpu_cluster_group *group = _starpu_cluster_group_new();
			_starpu_cluster_group_init(group, machine);
			_starpu_cluster_group_list_push_back(machine->groups, group);
			params = group->params;
		}
		else if (arg_type == STARPU_CLUSTER_NEW)
		{
			struct _starpu_cluster *cluster = _starpu_cluster_new();
			struct _starpu_cluster_group *group = _starpu_cluster_group_list_back(machine->groups);
			if (group == NULL)
			{
				group = _starpu_cluster_group_new();
				_starpu_cluster_group_init(group, machine);
				_starpu_cluster_group_list_push_back(machine->groups, group);
			}
			_starpu_cluster_init(cluster, group);
			_starpu_cluster_list_push_back(group->clusters, cluster);
			params = cluster->params;
		}
		else if (arg_type == STARPU_CLUSTER_NCORES)
		{
			struct _starpu_cluster_group *group = _starpu_cluster_group_list_back(machine->groups);
			if (group == NULL)
			{
				group = _starpu_cluster_group_new();
				_starpu_cluster_group_init(group, machine);
				_starpu_cluster_group_list_push_back(machine->groups, group);
			}
			struct _starpu_cluster *cluster =_starpu_cluster_list_back(group->clusters);
			cluster->ncores = va_arg(varg_list, unsigned);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
	}
	va_end(varg_list);

	switch(cluster_level)
	{
		case HWLOC_OBJ_MISC:
		case HWLOC_OBJ_BRIDGE:
		case HWLOC_OBJ_PCI_DEVICE:
		case HWLOC_OBJ_OS_DEVICE:
			STARPU_ABORT_MSG("Cluster aggregation isn't supported for level %s\n",
					 hwloc_obj_type_string(cluster_level));
			break;
		default: /* others can pass */
			break;
	}

	if (_starpu_cluster_machine(cluster_level, machine) == -ENODEV)
	{
		starpu_uncluster_machine(machine);
		machine = NULL;
	}

	return machine;
}

int starpu_uncluster_machine(struct starpu_cluster_machine *machine)
{
	if (machine == NULL)
		return -1;
	struct _starpu_cluster_group *g;
	struct _starpu_cluster_group_list *group_list = machine->groups;

	if (machine->id != STARPU_NMAX_SCHED_CTXS)
		starpu_sched_ctx_delete(machine->id);
	g = _starpu_cluster_group_list_begin(group_list);
	while (g != _starpu_cluster_group_list_end(group_list))
	{
		struct _starpu_cluster_group *tmp = g;
		g = _starpu_cluster_group_list_next(g);
		_starpu_cluster_group_remove(group_list, tmp);
	}
	_starpu_cluster_group_list_delete(group_list);
	if (machine->topology != NULL)
		hwloc_topology_destroy(machine->topology);
	free(machine->params);
	free(machine);
	starpu_sched_ctx_set_context(0);

	return 0;
}

int starpu_cluster_print(struct starpu_cluster_machine *clusters)
{
	if (clusters == NULL)
		return -1;

	int cnt, w;
	struct _starpu_cluster_group *group;
	struct _starpu_cluster *cluster;

	printf("Number of clusters created: %u\n", clusters->nclusters);
	cnt=0;
	if (clusters->nclusters)
	{
		for (group = _starpu_cluster_group_list_begin(clusters->groups);
		     group != _starpu_cluster_group_list_end(clusters->groups);
		     group = _starpu_cluster_group_list_next(group))
		{
			for (cluster = _starpu_cluster_list_begin(group->clusters);
			     cluster != _starpu_cluster_list_end(group->clusters);
			     cluster = _starpu_cluster_list_next(cluster))
			{
				printf("Cluster %d contains the following logical indexes:\n\t", cnt);
				for (w=0; w < cluster->ncores; w++)
					printf("%d ", cluster->cores[w]);
				printf("\n");
				cnt++;
			}
		}
	}
	return 0;
}

void _starpu_cluster_create(struct _starpu_cluster *cluster)
{
	if (cluster->params->awake_workers)
		cluster->id = starpu_sched_ctx_create(cluster->workerids, cluster->ncores,
		                                      "clusters",
		                                      STARPU_SCHED_CTX_AWAKE_WORKERS, 0);
	else
		cluster->id = starpu_sched_ctx_create(cluster->workerids, cluster->ncores,
		                                      "clusters", 0);

	/* cluster priority can be the lowest, so let's enforce it */
	starpu_sched_ctx_set_priority(cluster->workerids, cluster->ncores, cluster->id, 0);
	return;
}

void _starpu_cluster_group_create(struct _starpu_cluster_group *group)
{
	struct _starpu_cluster *c;
	for (c = _starpu_cluster_list_begin(group->clusters) ;
	     c != _starpu_cluster_list_end(group->clusters) ;
	     c = _starpu_cluster_list_next(c))
	{
		if (c->ncores == 0)
			continue;
		_starpu_cluster_create(c);
		if (!c->params->awake_workers)
			_starpu_cluster_bind(c);
	}

	return;
}

void _starpu_clusters_set_nesting(struct starpu_cluster_machine *m)
{
	struct _starpu_cluster_group *g;
	struct _starpu_cluster *c;

	for (g = _starpu_cluster_group_list_begin(m->groups) ;
	     g != _starpu_cluster_group_list_end(m->groups) ;
	     g = _starpu_cluster_group_list_next(g))
	{
		for (c = _starpu_cluster_list_begin(g->clusters) ;
		     c != _starpu_cluster_list_end(g->clusters) ;
		     c = _starpu_cluster_list_next(c))
			_starpu_get_sched_ctx_struct(c->id)->nesting_sched_ctx = m->id;
	}
}

int _starpu_cluster_bind(struct _starpu_cluster *cluster)
{
	starpu_binding_function func;
	void *func_arg;
	if (cluster->params->create_func)
	{
		func = cluster->params->create_func;
		func_arg = (void*) cluster->params->create_func_arg;
	}
	else
	{
		func = _starpu_cluster_type_get_func(cluster->params->type);
		func_arg = NULL;
	}

	return starpu_task_insert(&_starpu_cluster_bind_cl,
				  STARPU_SCHED_CTX, cluster->id,
				  STARPU_POSSIBLY_PARALLEL, 1,
				  STARPU_PROLOGUE_CALLBACK_POP, func,
				  STARPU_PROLOGUE_CALLBACK_POP_ARG, func_arg,
				  0);
}

void _starpu_cluster_group_init(struct _starpu_cluster_group *group,
				struct starpu_cluster_machine *father)
{
	group->id = 0;
	group->nclusters = 0;
	group->clusters = _starpu_cluster_list_new();
	group->father = father;
	_STARPU_MALLOC(group->params, sizeof(struct _starpu_cluster_parameters));
	_starpu_cluster_copy_parameters(father->params, group->params);
	return;
}

void _starpu_cluster_init(struct _starpu_cluster *cluster,
			  struct _starpu_cluster_group *father)
{
	cluster->id = STARPU_NMAX_SCHED_CTXS;
	cluster->cpuset = hwloc_bitmap_alloc();
	cluster->ncores = 0;
	cluster->cores = NULL;
	cluster->workerids = NULL;
	cluster->father = father;
	_STARPU_MALLOC(cluster->params, sizeof(struct _starpu_cluster_parameters));
	_starpu_cluster_copy_parameters(father->params, cluster->params);
}

int _starpu_cluster_remove(struct _starpu_cluster_list *cluster_list,
			   struct _starpu_cluster *cluster)
{
	if (cluster && cluster->id != STARPU_NMAX_SCHED_CTXS)
		starpu_sched_ctx_delete(cluster->id);
	else
		return -1;

	if (cluster->cores != NULL)
		free(cluster->cores);
	if (cluster->workerids != NULL)
		free(cluster->workerids);
	hwloc_bitmap_free(cluster->cpuset);
	free(cluster->params);
	_starpu_cluster_list_erase(cluster_list, cluster);
	_starpu_cluster_delete(cluster);

	return 0;
}

int _starpu_cluster_group_remove(struct _starpu_cluster_group_list *group_list,
				 struct _starpu_cluster_group *group)
{
	struct _starpu_cluster_list *cluster_list = group->clusters;
	struct _starpu_cluster *c = _starpu_cluster_list_begin(cluster_list);
	while (c != _starpu_cluster_list_end(cluster_list))
	{
		struct _starpu_cluster *tmp = c;
		c = _starpu_cluster_list_next(c);
		_starpu_cluster_remove(cluster_list, tmp);
	}
	_starpu_cluster_list_delete(cluster_list);

	free(group->params);
	_starpu_cluster_group_list_erase(group_list, group);
	_starpu_cluster_group_delete(group);

	return 0;
}

void _starpu_cluster_init_parameters(struct _starpu_cluster_parameters *params)
{
	params->min_nb = 0;
	params->max_nb = 0;
	params->nb = 0;
	params->sched_policy_name = NULL;
	params->sched_policy_struct = NULL;
	params->keep_homogeneous = 0;
	params->prefere_min = 0;
	params->create_func = NULL;
	params->create_func_arg = NULL;
	params->type = STARPU_CLUSTER_OPENMP;
	params->awake_workers = 0;

	return;
}

void _starpu_cluster_copy_parameters(struct _starpu_cluster_parameters *src, struct _starpu_cluster_parameters *dst)
{
	dst->min_nb = src->min_nb;
	dst->max_nb = src->max_nb;
	dst->nb = src->nb;
	dst->sched_policy_name = src->sched_policy_name;
	dst->sched_policy_struct = src->sched_policy_struct;
	dst->keep_homogeneous = src->keep_homogeneous;
	dst->prefere_min = src->prefere_min;
	dst->create_func = src->create_func;
	dst->create_func_arg = src->create_func_arg;
	dst->type = src->type;
	dst->awake_workers = src->awake_workers;

	return;
}

/* Considering the resources and parameters, how many clusters should we take? */
int _starpu_cluster_analyze_parameters(struct _starpu_cluster_parameters *params, int npus)
{
	int nb_clusters = 1, j;
	if (params->nb)
	{
		nb_clusters = params->nb <= npus?params->nb : npus;
	}
	else if (params->min_nb && params->max_nb)
	{
		if (!params->keep_homogeneous)
		{
			if (params->prefere_min)
				nb_clusters = params->min_nb <= npus? params->min_nb : npus;
			else
				nb_clusters = params->max_nb <= npus? params->max_nb : npus;
		}
		else
		{
			int begin = params->prefere_min? params->min_nb:params->max_nb;
			int end = params->prefere_min? params->max_nb+1:params->min_nb-1;
			j=begin;
			int best = 0, second_best = 0, cpu_loss = INT_MAX;
			while (j != end)
			{
				if (npus%j == 0)
				{
					best = j;
					break;
				}
				if (npus%j < cpu_loss)
				{
					cpu_loss = npus%j;
					second_best = j;
				}
				j = params->prefere_min? j+1:j-1;
			}

			if (best)
				nb_clusters = best;
			else if (second_best)
				nb_clusters = second_best;
		}
	}

	return nb_clusters;
}

int _starpu_cluster_machine(hwloc_obj_type_t cluster_level,
			     struct starpu_cluster_machine *machine)
{
	struct _starpu_cluster_group *g;
	int ret;

	ret = _starpu_cluster_topology(cluster_level, machine);
	if (ret)
		return ret;

	for (g = _starpu_cluster_group_list_begin(machine->groups) ;
	     g != _starpu_cluster_group_list_end(machine->groups) ;
	     g = _starpu_cluster_group_list_next(g))
		_starpu_cluster_group_create(g);

	starpu_task_wait_for_all();

	/* Create containing context */
	if (machine->params->sched_policy_struct != NULL)
	{
		machine->id = starpu_sched_ctx_create(NULL, -1, "main sched ctx",
		                                      STARPU_SCHED_CTX_POLICY_STRUCT,
		                                      machine->params->sched_policy_struct,
		                                      0);
	}
	else if (machine->params->sched_policy_name != NULL)
	{
		machine->id = starpu_sched_ctx_create(NULL, -1, "main sched ctx",
		                                      STARPU_SCHED_CTX_POLICY_NAME,
		                                      machine->params->sched_policy_name,
		                                      0);
	}
	else
	{
		struct starpu_sched_policy *sched_policy;
		struct _starpu_sched_ctx *global_ctx =_starpu_get_sched_ctx_struct(STARPU_GLOBAL_SCHED_CTX);
		sched_policy = _starpu_get_sched_policy(global_ctx);
		machine->id = starpu_sched_ctx_create(NULL, -1, "main sched ctx",
		                                      STARPU_SCHED_CTX_POLICY_STRUCT,
		                                      sched_policy, 0);
	}

	_starpu_clusters_set_nesting(machine);
	starpu_sched_ctx_set_context(&machine->id);

	return ret;
}

int _starpu_cluster_topology(hwloc_obj_type_t cluster_level,
			      struct starpu_cluster_machine *machine)
{
	int w;
	hwloc_topology_t topology;
	hwloc_cpuset_t avail_cpus;

	int nworkers = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	if (nworkers == 0)
		return -ENODEV;
	int *workers;
	_STARPU_MALLOC(workers, sizeof(int) * nworkers);
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, workers, nworkers);

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	STARPU_ASSERT_MSG(config->topology.hwtopology != NULL, "STARPU_CLUSTER: You "
			  "need to call starpu_init() or make sure to activate hwloc.");
	hwloc_topology_dup(&topology, config->topology.hwtopology);

	avail_cpus = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(avail_cpus);

	for (w = 0; w < nworkers ; w++)
	{
		struct _starpu_worker *worker_str = _starpu_get_worker_struct(workers[w]);
		hwloc_bitmap_or(avail_cpus, avail_cpus, worker_str->hwloc_cpu_set);
	}

	hwloc_topology_restrict(topology, avail_cpus, 0);
	free(workers);

	/* Use new topology to fill in the cluster list */
	machine->topology = topology;
	_starpu_cluster_group(cluster_level, machine);

	hwloc_bitmap_free(avail_cpus);

	return 0;
}

void _starpu_cluster_group(hwloc_obj_type_t cluster_level,
			   struct starpu_cluster_machine *machine)
{
	int nb_objects;
	int i;
	struct _starpu_cluster_group *group = NULL;

	if (machine->groups == NULL)
		machine->groups = _starpu_cluster_group_list_new();

	nb_objects = hwloc_get_nbobjs_by_type(machine->topology, cluster_level);
	if (nb_objects <= 0)
		return;
	/* XXX: handle nb_objects == -1 */

	group = _starpu_cluster_group_list_begin(machine->groups);
	for (i = 0 ; i < nb_objects ; i++)
	{
		hwloc_obj_t cluster_obj = hwloc_get_obj_by_type(machine->topology, cluster_level, i);

		if (group == NULL)
		{
			group = _starpu_cluster_group_new();
			_starpu_cluster_group_init(group, machine);
			_starpu_cluster_group_list_push_back(machine->groups, group);
		}

		group->group_obj = cluster_obj;

		_starpu_cluster(group);
		machine->ngroups++;
		machine->nclusters += group->nclusters;
		group = _starpu_cluster_group_list_next(group);
	}

	return;
}

void _starpu_cluster(struct _starpu_cluster_group *group)
{
	int i, avail_pus, npus, npreset=0;
	struct _starpu_cluster *cluster;
	npus = hwloc_get_nbobjs_inside_cpuset_by_type(group->father->topology,
						      group->group_obj->cpuset,
						      HWLOC_OBJ_PU);

	/* Preset clusters */
	avail_pus = npus;
	for (cluster=_starpu_cluster_list_begin(group->clusters);
	     cluster!=_starpu_cluster_list_end(group->clusters);
	     cluster=_starpu_cluster_list_next(cluster))
	{
		if (cluster->ncores > avail_pus)
			cluster->ncores = avail_pus;
		else if (avail_pus == 0)
			cluster->ncores = 0;

		if (cluster->ncores > 0)
		{
			_STARPU_MALLOC(cluster->cores, sizeof(int)*cluster->ncores);
			_STARPU_MALLOC(cluster->workerids, sizeof(int)*cluster->ncores);
			avail_pus -= cluster->ncores;
			npreset++;
		}
	}

	/* Automatic clusters */
	group->nclusters = _starpu_cluster_analyze_parameters(group->params, avail_pus);
	for (i=0 ; i<group->nclusters && avail_pus>0 ; i++)
	{
		if (cluster == NULL)
		{
			cluster = _starpu_cluster_new();
			_starpu_cluster_init(cluster, group);
			_starpu_cluster_list_push_back(group->clusters, cluster);
		}

		if (cluster->ncores != 0 && cluster->ncores > avail_pus)
		{
			cluster->ncores = avail_pus;
		}
		else
		{
			if (cluster->params->keep_homogeneous)
				cluster->ncores = avail_pus/(group->nclusters-i);
			else
				cluster->ncores = i==group->nclusters-1?
					avail_pus:
					avail_pus/(group->nclusters-i);
		}
		avail_pus -= cluster->ncores;
		_STARPU_MALLOC(cluster->cores, sizeof(int)*cluster->ncores);
		_STARPU_MALLOC(cluster->workerids, sizeof(int)*cluster->ncores);

		cluster = _starpu_cluster_list_next(cluster);
	}
	group->nclusters += npreset;

	cluster = _starpu_cluster_list_begin(group->clusters);
	int count = 0;
	static int starpu_cluster_warned = 0;

	for (i=0 ; i<npus ; i++)
	{
		hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_type(group->father->topology,
								     group->group_obj->cpuset,
								     HWLOC_OBJ_PU, i);

		/* If we have more than one worker on this resource, let's add them too --
		   even if it's bad (they'll all be boud on the same PU) */
		int size = 0, j;
		struct _starpu_hwloc_userdata *data = pu->userdata;
		struct _starpu_worker_list *list = data->worker_list;
		struct _starpu_worker *worker_str;
		for (worker_str = _starpu_worker_list_begin(list);
		     worker_str != _starpu_worker_list_end(list);
		     worker_str = _starpu_worker_list_next(worker_str))
		{
			if (worker_str->arch == STARPU_CPU_WORKER)
				size++;
		}

		if (size > 1)
		{
			if (!starpu_cluster_warned)
			{
				_STARPU_DISP("STARPU CLUSTERS: Caution! It seems that you have"
					     " multiple workers bound to the same PU. If you have"
					     " multithreading on your cores it is greatly adviced"
					     " to export STARPU_NTHREADS_PER_CORE=nb.");
				starpu_cluster_warned = 1;
			}
			cluster->ncores += size-1;
			_STARPU_REALLOC(cluster->cores, sizeof(int)*cluster->ncores);
			_STARPU_REALLOC(cluster->workerids, sizeof(int)*cluster->ncores);
		}

		/* grab workerid list and return first cpu */
		worker_str = _starpu_worker_list_begin(list);
		if (worker_str)
			hwloc_bitmap_or(cluster->cpuset, cluster->cpuset,
					worker_str->hwloc_cpu_set);
		j = 0;
		while (worker_str != _starpu_worker_list_end(list))
		{
			if (worker_str->arch == STARPU_CPU_WORKER)
			{
				cluster->cores[count+j] = worker_str->bindid;
				cluster->workerids[count+j] = worker_str->workerid;
				j++;
			}
			worker_str = _starpu_worker_list_next(worker_str);
		}

		count+=size;
		if (cluster->ncores == count)
		{
			count = 0;
			cluster = _starpu_cluster_list_next(cluster);
		}
	}

	return;
}

#endif
