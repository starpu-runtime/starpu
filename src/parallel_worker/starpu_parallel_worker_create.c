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

/* This file creates an interface to manage resources within parallel
 * workers and make use of parallel tasks. It entirely depends on the
 * hwloc software.
 */

#include <parallel_worker/starpu_parallel_worker_create.h>

#ifdef STARPU_PARALLEL_WORKER

starpu_binding_function _starpu_parallel_worker_type_get_func(enum starpu_parallel_worker_types type)
{
	starpu_binding_function prologue_func;

	switch (type)
	{
	case STARPU_PARALLEL_WORKER_OPENMP:
		prologue_func = &starpu_parallel_worker_openmp_prologue;
		break;
	case STARPU_PARALLEL_WORKER_INTEL_OPENMP_MKL:
		prologue_func = &starpu_parallel_worker_intel_openmp_mkl_prologue;
		break;
	case STARPU_PARALLEL_WORKER_GNU_OPENMP_MKL:
#ifdef STARPU_MKL
		prologue_func = &starpu_parallel_worker_gnu_openmp_mkl_prologue;
#else
		_STARPU_MSG("Warning: MKL support is not available, using STARPU_PARALLEL_WORKER_INTEL_OPENMP_MKL instead\n");
		prologue_func = &starpu_parallel_worker_intel_openmp_mkl_prologue;
#endif
		break;
	default:
		prologue_func = NULL;
	}

	return prologue_func;
}

void starpu_parallel_worker_openmp_prologue(void *arg)
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
void starpu_parallel_worker_gnu_openmp_mkl_prologue(void *arg)
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
			mkl_set_num_threads_local(ncpuids);
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

/* Main interface function to create a parallel worker view of the machine.
 * Its job is to capture what the user wants and store it in a standard view. */
struct starpu_parallel_worker_config *_starpu_parallel_worker_init_varg(hwloc_obj_type_t parallel_worker_level, va_list varg_list)
{
	int arg_type;
	struct starpu_parallel_worker_config *machine;

	_STARPU_CALLOC(machine, 1, sizeof(struct starpu_parallel_worker_config));
	_STARPU_CALLOC(machine->params, 1, sizeof(struct _starpu_parallel_worker_parameters));
	machine->id = STARPU_NMAX_SCHED_CTXS;
	machine->groups = _starpu_parallel_worker_group_list_new();
	machine->nparallel_workers = 0;
	machine->ngroups = 0;
	machine->topology = NULL;
	_starpu_parallel_worker_init_parameters(machine->params);

	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type == STARPU_PARALLEL_WORKER_MIN_NB)
		{
			machine->params->min_nb = va_arg(varg_list, int);
			if (machine->params->min_nb <= 0)
				_STARPU_DISP("Caution min number of contexts shouldn't be negative or null\n");
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_MAX_NB)
		{
			machine->params->max_nb = va_arg(varg_list, int);
			if (machine->params->max_nb <= 0)
				_STARPU_DISP("Caution max number of contexts shouldn't be negative or null\n");
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_NB)
		{
			machine->params->nb = va_arg(varg_list, int);
			if (machine->params->nb <= 0)
				_STARPU_DISP("Caution number of contexts shouldn't be negative or null\n");
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_POLICY_NAME)
		{
			machine->params->sched_policy_name = va_arg(varg_list, char*);
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_POLICY_STRUCT)
		{
			machine->params->sched_policy_struct = va_arg(varg_list, struct starpu_sched_policy*);
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_KEEP_HOMOGENEOUS)
		{
			machine->params->keep_homogeneous = va_arg(varg_list, int); /* 0=off, other=on */
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_PREFERE_MIN)
		{
			machine->params->prefere_min = va_arg(varg_list, int); /* 0=off, other=on */
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_CREATE_FUNC)
		{
			machine->params->create_func = va_arg(varg_list, void (*)(void*));
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_CREATE_FUNC_ARG)
		{
			machine->params->create_func_arg = va_arg(varg_list, void*);
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_TYPE)
		{
			machine->params->type = va_arg(varg_list, enum starpu_parallel_worker_types);
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_AWAKE_WORKERS)
		{
			machine->params->awake_workers = va_arg(varg_list, unsigned);
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_PARTITION_ONE)
		{
			struct _starpu_parallel_worker_group *group = _starpu_parallel_worker_group_new();
			_starpu_parallel_worker_group_init(group, machine);
			_starpu_parallel_worker_group_list_push_back(machine->groups, group);
			machine->params = group->params;
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_NEW)
		{
			struct _starpu_parallel_worker *parallel_worker = _starpu_parallel_worker_new();
			struct _starpu_parallel_worker_group *group = _starpu_parallel_worker_group_list_back(machine->groups);
			if (group == NULL)
			{
				group = _starpu_parallel_worker_group_new();
				_starpu_parallel_worker_group_init(group, machine);
				_starpu_parallel_worker_group_list_push_back(machine->groups, group);
			}
			_starpu_parallel_worker_init(parallel_worker, group);
			_starpu_parallel_worker_list_push_back(group->parallel_workers, parallel_worker);
			machine->params = parallel_worker->params;
		}
		else if (arg_type == STARPU_PARALLEL_WORKER_NCORES)
		{
			struct _starpu_parallel_worker_group *group = _starpu_parallel_worker_group_list_back(machine->groups);
			if (group == NULL)
			{
				group = _starpu_parallel_worker_group_new();
				_starpu_parallel_worker_group_init(group, machine);
				_starpu_parallel_worker_group_list_push_back(machine->groups, group);
			}
			struct _starpu_parallel_worker *parallel_worker =_starpu_parallel_worker_list_back(group->parallel_workers);
			parallel_worker->ncores = va_arg(varg_list, unsigned);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
	}
	va_end(varg_list);

	switch(parallel_worker_level)
	{
		case HWLOC_OBJ_MISC:
		case HWLOC_OBJ_BRIDGE:
		case HWLOC_OBJ_PCI_DEVICE:
		case HWLOC_OBJ_OS_DEVICE:
			STARPU_ABORT_MSG("Parallel_Worker aggregation isn't supported for level %s\n",
					 hwloc_obj_type_string(parallel_worker_level));
			break;
		default: /* others can pass */
			break;
	}

	if (_starpu_parallel_worker_config(parallel_worker_level, machine) == -ENODEV)
	{
		starpu_parallel_worker_shutdown(machine);
		machine = NULL;
	}

	return machine;
}

struct starpu_parallel_worker_config *starpu_parallel_worker_init(hwloc_obj_type_t parallel_worker_level, ...)
{
	struct starpu_parallel_worker_config *config;
	va_list varg_list;
	va_start(varg_list, parallel_worker_level);
	config = _starpu_parallel_worker_init_varg(parallel_worker_level, varg_list);
	va_end(varg_list);
	return config;
}

int starpu_parallel_worker_shutdown(struct starpu_parallel_worker_config *machine)
{
	if (machine == NULL)
		return -1;
	struct _starpu_parallel_worker_group *g;
	struct _starpu_parallel_worker_group_list *group_list = machine->groups;

	if (machine->id != STARPU_NMAX_SCHED_CTXS)
		starpu_sched_ctx_delete(machine->id);

	g = _starpu_parallel_worker_group_list_begin(group_list);
	while (g != _starpu_parallel_worker_group_list_end(group_list))
	{
		struct _starpu_parallel_worker_group *tmp = g;
		g = _starpu_parallel_worker_group_list_next(g);
		_starpu_parallel_worker_group_remove(group_list, tmp);
	}
	_starpu_parallel_worker_group_list_delete(group_list);

	if (machine->topology != NULL)
		hwloc_topology_destroy(machine->topology);
	free(machine->params);
	free(machine);
	starpu_sched_ctx_set_context(0);

	return 0;
}

int starpu_parallel_worker_print(struct starpu_parallel_worker_config *parallel_workers)
{
	if (parallel_workers == NULL)
		return -1;

	int cnt, w;
	struct _starpu_parallel_worker_group *group;
	struct _starpu_parallel_worker *parallel_worker;

	printf("Number of parallel workers created: %u\n", parallel_workers->nparallel_workers);
	cnt=0;
	if (parallel_workers->nparallel_workers)
	{
		for (group = _starpu_parallel_worker_group_list_begin(parallel_workers->groups);
		     group != _starpu_parallel_worker_group_list_end(parallel_workers->groups);
		     group = _starpu_parallel_worker_group_list_next(group))
		{
			for (parallel_worker = _starpu_parallel_worker_list_begin(group->parallel_workers);
			     parallel_worker != _starpu_parallel_worker_list_end(group->parallel_workers);
			     parallel_worker = _starpu_parallel_worker_list_next(parallel_worker))
			{
				printf("Parallel worker %d contains the following logical indexes:\n\t", cnt);
				for (w=0; w < parallel_worker->ncores; w++)
					printf("%d ", parallel_worker->cores[w]);
				printf("\n");
				cnt++;
			}
		}
	}
	return 0;
}

int _starpu_parallel_worker_create(struct _starpu_parallel_worker *parallel_worker)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	if (config->topology.nsched_ctxs == STARPU_NMAX_SCHED_CTXS)
		/* Too many contexts already :/ */
		return 0;

	if (parallel_worker->params->awake_workers)
		parallel_worker->id = starpu_sched_ctx_create(parallel_worker->workerids, parallel_worker->ncores,
							      "parallel_workers",
							      STARPU_SCHED_CTX_AWAKE_WORKERS, 0);
	else
		parallel_worker->id = starpu_sched_ctx_create(parallel_worker->workerids, parallel_worker->ncores,
							      "parallel_workers", 0);

	/* parallel_worker priority can be the lowest, so let's enforce it */
	starpu_sched_ctx_set_priority(parallel_worker->workerids, parallel_worker->ncores, parallel_worker->id, 0);
	return 1;
}

int _starpu_parallel_worker_group_create(struct _starpu_parallel_worker_group *group)
{
	struct _starpu_parallel_worker *c;
	for (c = _starpu_parallel_worker_list_begin(group->parallel_workers) ;
	     c != _starpu_parallel_worker_list_end(group->parallel_workers) ;
	     c = _starpu_parallel_worker_list_next(c))
	{
		if (c->ncores == 0)
			continue;
		if (_starpu_parallel_worker_create(c) == 0)
			return 0;
		if (!c->params->awake_workers)
			_starpu_parallel_worker_bind(c);
	}

	return 1;
}

void _starpu_parallel_workers_set_nesting(struct starpu_parallel_worker_config *m)
{
	struct _starpu_parallel_worker_group *g;
	struct _starpu_parallel_worker *c;

	for (g = _starpu_parallel_worker_group_list_begin(m->groups) ;
	     g != _starpu_parallel_worker_group_list_end(m->groups) ;
	     g = _starpu_parallel_worker_group_list_next(g))
	{
		for (c = _starpu_parallel_worker_list_begin(g->parallel_workers) ;
		     c != _starpu_parallel_worker_list_end(g->parallel_workers) ;
		     c = _starpu_parallel_worker_list_next(c))
			_starpu_get_sched_ctx_struct(c->id)->nesting_sched_ctx = m->id;
	}
}

int _starpu_parallel_worker_bind(struct _starpu_parallel_worker *parallel_worker)
{
	starpu_binding_function func;
	void *func_arg;
	if (parallel_worker->params->create_func)
	{
		func = parallel_worker->params->create_func;
		func_arg = (void*) parallel_worker->params->create_func_arg;
	}
	else
	{
		func = _starpu_parallel_worker_type_get_func(parallel_worker->params->type);
		func_arg = NULL;
	}

	return starpu_task_insert(&_starpu_parallel_worker_bind_cl,
				  STARPU_SCHED_CTX, parallel_worker->id,
				  STARPU_POSSIBLY_PARALLEL, 1,
				  STARPU_PROLOGUE_CALLBACK_POP, func,
				  STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE, func_arg,
				  0);
}

void _starpu_parallel_worker_group_init(struct _starpu_parallel_worker_group *group, struct starpu_parallel_worker_config *father)
{
	group->id = 0;
	group->nparallel_workers = 0;
	group->parallel_workers = _starpu_parallel_worker_list_new();
	group->father = father;
	_STARPU_MALLOC(group->params, sizeof(struct _starpu_parallel_worker_parameters));
	_starpu_parallel_worker_copy_parameters(father->params, group->params);
	return;
}

void _starpu_parallel_worker_init(struct _starpu_parallel_worker *parallel_worker, struct _starpu_parallel_worker_group *father)
{
	parallel_worker->id = STARPU_NMAX_SCHED_CTXS;
	parallel_worker->cpuset = hwloc_bitmap_alloc();
	parallel_worker->ncores = 0;
	parallel_worker->cores = NULL;
	parallel_worker->workerids = NULL;
	parallel_worker->father = father;
	_STARPU_MALLOC(parallel_worker->params, sizeof(struct _starpu_parallel_worker_parameters));
	_starpu_parallel_worker_copy_parameters(father->params, parallel_worker->params);
}

int _starpu_parallel_worker_remove(struct _starpu_parallel_worker_list *parallel_worker_list, struct _starpu_parallel_worker *parallel_worker)
{
	if (parallel_worker && parallel_worker->id != STARPU_NMAX_SCHED_CTXS)
		starpu_sched_ctx_delete(parallel_worker->id);
	else
		return -1;

	if (parallel_worker->cores != NULL)
		free(parallel_worker->cores);
	if (parallel_worker->workerids != NULL)
		free(parallel_worker->workerids);

	hwloc_bitmap_free(parallel_worker->cpuset);
	_starpu_parallel_worker_list_erase(parallel_worker_list, parallel_worker);
	_starpu_parallel_worker_delete(parallel_worker);

	return 0;
}

int _starpu_parallel_worker_group_remove(struct _starpu_parallel_worker_group_list *group_list, struct _starpu_parallel_worker_group *group)
{
	struct _starpu_parallel_worker_list *parallel_worker_list = group->parallel_workers;
	struct _starpu_parallel_worker *c = _starpu_parallel_worker_list_begin(parallel_worker_list);
	while (c != _starpu_parallel_worker_list_end(parallel_worker_list))
	{
		struct _starpu_parallel_worker *tmp = c;
		c = _starpu_parallel_worker_list_next(c);
		_starpu_parallel_worker_remove(parallel_worker_list, tmp);
	}
	_starpu_parallel_worker_list_delete(parallel_worker_list);

	free(group->params);
	_starpu_parallel_worker_group_list_erase(group_list, group);
	_starpu_parallel_worker_group_delete(group);

	return 0;
}

void _starpu_parallel_worker_init_parameters(struct _starpu_parallel_worker_parameters *params)
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
	params->type = STARPU_PARALLEL_WORKER_OPENMP;
	params->awake_workers = 0;

	return;
}

void _starpu_parallel_worker_copy_parameters(struct _starpu_parallel_worker_parameters *src, struct _starpu_parallel_worker_parameters *dst)
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

/* Considering the resources and parameters, how many parallel_workers should we take? */
int _starpu_parallel_worker_analyze_parameters(struct _starpu_parallel_worker_parameters *params, int npus)
{
	int nb_parallel_workers = 1, j;
	if (params->nb)
	{
		nb_parallel_workers = params->nb <= npus?params->nb : npus;
	}
	else if (params->min_nb && params->max_nb)
	{
		if (!params->keep_homogeneous)
		{
			if (params->prefere_min)
				nb_parallel_workers = params->min_nb <= npus? params->min_nb : npus;
			else
				nb_parallel_workers = params->max_nb <= npus? params->max_nb : npus;
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
				nb_parallel_workers = best;
			else if (second_best)
				nb_parallel_workers = second_best;
		}
	}

	return nb_parallel_workers;
}

int _starpu_parallel_worker_config(hwloc_obj_type_t parallel_worker_level, struct starpu_parallel_worker_config *machine)
{
	struct _starpu_parallel_worker_group *g;
	int ret;

	ret = _starpu_parallel_worker_topology(parallel_worker_level, machine);
	if (ret)
		return ret;

	for (g = _starpu_parallel_worker_group_list_begin(machine->groups) ;
	     g != _starpu_parallel_worker_group_list_end(machine->groups) ;
	     g = _starpu_parallel_worker_group_list_next(g))
		if (_starpu_parallel_worker_group_create(g) == 0)
			return -ENODEV;

	starpu_task_wait_for_all();

	struct _starpu_machine_config *config = _starpu_get_machine_config();

	if (config->topology.nsched_ctxs == STARPU_NMAX_SCHED_CTXS)
		/* Too many contexts already :/ */
		return -ENODEV;

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

	_starpu_parallel_workers_set_nesting(machine);
	starpu_sched_ctx_set_context(&machine->id);

	return 0;
}

int _starpu_parallel_worker_topology(hwloc_obj_type_t parallel_worker_level, struct starpu_parallel_worker_config *machine)
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
	STARPU_ASSERT_MSG(config->topology.hwtopology != NULL, "STARPU_PARALLEL_WORKER: You "
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

	/* Use new topology to fill in the parallel_worker list */
	machine->topology = topology;
	_starpu_parallel_worker_group(parallel_worker_level, machine);

	hwloc_bitmap_free(avail_cpus);

	return 0;
}

void _starpu_parallel_worker_group(hwloc_obj_type_t parallel_worker_level, struct starpu_parallel_worker_config *machine)
{
	int nb_objects;
	int i;
	struct _starpu_parallel_worker_group *group = NULL;

	if (machine->groups == NULL)
		machine->groups = _starpu_parallel_worker_group_list_new();

	nb_objects = hwloc_get_nbobjs_by_type(machine->topology, parallel_worker_level);
	if (nb_objects <= 0)
		return;
	/* XXX: handle nb_objects == -1 */

	group = _starpu_parallel_worker_group_list_begin(machine->groups);
	for (i = 0 ; i < nb_objects ; i++)
	{
		hwloc_obj_t parallel_worker_obj = hwloc_get_obj_by_type(machine->topology, parallel_worker_level, i);

		if (group == NULL)
		{
			group = _starpu_parallel_worker_group_new();
			_starpu_parallel_worker_group_init(group, machine);
			_starpu_parallel_worker_group_list_push_back(machine->groups, group);
		}

		group->group_obj = parallel_worker_obj;

		_starpu_parallel_worker(group);
		machine->ngroups++;
		machine->nparallel_workers += group->nparallel_workers;
		group = _starpu_parallel_worker_group_list_next(group);
	}

	return;
}

void _starpu_parallel_worker(struct _starpu_parallel_worker_group *group)
{
	int i, avail_pus, npus, npreset=0;
	struct _starpu_parallel_worker *parallel_worker;
	npus = hwloc_get_nbobjs_inside_cpuset_by_type(group->father->topology,
						      group->group_obj->cpuset,
						      HWLOC_OBJ_PU);

	/* Preset parallel_workers */
	avail_pus = npus;
	for (parallel_worker=_starpu_parallel_worker_list_begin(group->parallel_workers);
	     parallel_worker!=_starpu_parallel_worker_list_end(group->parallel_workers);
	     parallel_worker=_starpu_parallel_worker_list_next(parallel_worker))
	{
		if (parallel_worker->ncores > avail_pus)
			parallel_worker->ncores = avail_pus;
		else if (avail_pus == 0)
			parallel_worker->ncores = 0;

		if (parallel_worker->ncores > 0)
		{
			_STARPU_MALLOC(parallel_worker->cores, sizeof(int)*parallel_worker->ncores);
			_STARPU_MALLOC(parallel_worker->workerids, sizeof(int)*parallel_worker->ncores);
			avail_pus -= parallel_worker->ncores;
			npreset++;
		}
	}

	/* Automatic parallel_workers */
	group->nparallel_workers = _starpu_parallel_worker_analyze_parameters(group->params, avail_pus);
	for (i=0 ; i<group->nparallel_workers && avail_pus>0 ; i++)
	{
		if (parallel_worker == NULL)
		{
			parallel_worker = _starpu_parallel_worker_new();
			_starpu_parallel_worker_init(parallel_worker, group);
			_starpu_parallel_worker_list_push_back(group->parallel_workers, parallel_worker);
		}

		if (parallel_worker->ncores != 0 && parallel_worker->ncores > avail_pus)
		{
			parallel_worker->ncores = avail_pus;
		}
		else
		{
			if (parallel_worker->params->keep_homogeneous)
				parallel_worker->ncores = avail_pus/(group->nparallel_workers-i);
			else
				parallel_worker->ncores = i==group->nparallel_workers-1?
					avail_pus:
					avail_pus/(group->nparallel_workers-i);
		}
		avail_pus -= parallel_worker->ncores;
		_STARPU_MALLOC(parallel_worker->cores, sizeof(int)*parallel_worker->ncores);
		_STARPU_MALLOC(parallel_worker->workerids, sizeof(int)*parallel_worker->ncores);

		parallel_worker = _starpu_parallel_worker_list_next(parallel_worker);
	}
	group->nparallel_workers += npreset;

	parallel_worker = _starpu_parallel_worker_list_begin(group->parallel_workers);
	int count = 0;
	static int starpu_parallel_worker_warned = 0;

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
			STARPU_HG_DISABLE_CHECKING(starpu_parallel_worker_warned);
			if (!starpu_parallel_worker_warned)
			{
				_STARPU_DISP("STARPU PARALLEL_WORKERS: Caution! It seems that you have"
					     " multiple workers bound to the same PU. If you have"
					     " multithreading on your cores it is greatly advised"
					     " to export STARPU_NTHREADS_PER_CORE=nb.\n");
				starpu_parallel_worker_warned = 1;
			}
			parallel_worker->ncores += size-1;
			_STARPU_REALLOC(parallel_worker->cores, sizeof(int)*parallel_worker->ncores);
			_STARPU_REALLOC(parallel_worker->workerids, sizeof(int)*parallel_worker->ncores);
		}

		/* grab workerid list and return first cpu */
		worker_str = _starpu_worker_list_begin(list);
		if (worker_str)
			hwloc_bitmap_or(parallel_worker->cpuset, parallel_worker->cpuset,
					worker_str->hwloc_cpu_set);
		j = 0;
		while (worker_str != _starpu_worker_list_end(list))
		{
			if (worker_str->arch == STARPU_CPU_WORKER)
			{
				parallel_worker->cores[count+j] = worker_str->bindid;
				parallel_worker->workerids[count+j] = worker_str->workerid;
				j++;
			}
			worker_str = _starpu_worker_list_next(worker_str);
		}

		count+=size;
		if (parallel_worker->ncores == count)
		{
			count = 0;
			parallel_worker = _starpu_parallel_worker_list_next(parallel_worker);
		}
	}

	return;
}

struct starpu_cluster_machine STARPU_DEPRECATED
{
	unsigned id;
	hwloc_topology_t topology;
	unsigned nparallel_workers;
	unsigned ngroups;
	struct _starpu_parallel_worker_group_list *groups;
	struct _starpu_parallel_worker_parameters *params;
};

struct starpu_cluster_machine *starpu_cluster_machine(hwloc_obj_type_t cluster_level, ...)
{
	struct starpu_parallel_worker_config *config;
	va_list varg_list;
	va_start(varg_list, cluster_level);
	config = _starpu_parallel_worker_init_varg(cluster_level, varg_list);
	va_end(varg_list);
	return (struct starpu_cluster_machine *)config;
}

int starpu_uncluster_machine(struct starpu_cluster_machine *clusters)
{
	struct starpu_parallel_worker_config *c = (struct starpu_parallel_worker_config *)clusters;
	return starpu_parallel_worker_shutdown(c);
}

int starpu_cluster_print(struct starpu_cluster_machine *clusters)
{
	struct starpu_parallel_worker_config *c = (struct starpu_parallel_worker_config *)clusters;
	return starpu_parallel_worker_print(c);
}


#endif
