/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
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

#include <common/config.h>
#include <starpu.h>
#include <common/utils.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef STARPU_HAVE_HWLOC
/* This function returns 1 the subtree induced by obj only contains CPU
 * workers, otherwise 0 is returned. This function registers all valid worker
 * combination below obj. The id of the CPU workers are put in the worker_array
 * and their count is put in the worker_cnt pointer. */
static int find_combinations_with_hwloc_rec(hwloc_obj_t obj, int *worker_array, int *worker_cnt)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	/* Is this a leaf ? (eg. a PU for hwloc) */
	int is_leaf = !hwloc_compare_types(config->cpu_depth, obj->depth);

	if (is_leaf)
	{
		struct starpu_worker_s *worker = obj->userdata;

		/* If this is a CPU worker, append its id at the end of the
		 * list */
		if (worker && worker->arch == STARPU_CPU_WORKER)
		{
			worker_array[*worker_cnt] = worker->workerid;
			*worker_cnt = *worker_cnt + 1;
		}

		/* We cannot create a combined worker only if there is a CPU
		 * worker. */
		return (!worker || worker->arch == STARPU_CPU_WORKER);
	}

	/* If there is only one child, we go to the next level directly */
	if (obj->arity == 1)
		return find_combinations_with_hwloc_rec(obj->children[0], worker_array, worker_cnt);

	/* We recursively go from the root to the leaves of the tree to find
	 * subtrees that only have CPUs as leaves. */
	unsigned cpu_children_cnt = 0;

	int worker_array_rec[STARPU_NMAXWORKERS];
	int worker_cnt_rec = 0;
	memset(worker_array_rec, 0, sizeof(worker_array_rec));

	unsigned i;
	for (i = 0; i < obj->arity; i++)
	{
		int valid_subtree = find_combinations_with_hwloc_rec(obj->children[i],
						worker_array_rec, &worker_cnt_rec);
		if (valid_subtree)
			cpu_children_cnt++;
	}

	int child;

	if (cpu_children_cnt == obj->arity)
	for (child = 0; child < worker_cnt_rec; child++)
	{
		worker_array[*worker_cnt] = worker_array_rec[child];
		*worker_cnt = *worker_cnt + 1;
	}
	
	/* If there is at least 2 children that are valid, we combined them. */
	int maxsize = starpu_get_env_number("STARPU_MAX_WORKERSIZE");
	int minsize = starpu_get_env_number("STARPU_MIN_WORKERSIZE");

	if (cpu_children_cnt > 1 && worker_cnt_rec > 0 && worker_cnt_rec <= maxsize && worker_cnt_rec >= minsize)
		starpu_combined_worker_assign_workerid(worker_cnt_rec, worker_array_rec);

	return (cpu_children_cnt == obj->arity);
}

static void find_combinations_with_hwloc(struct starpu_machine_topology_s *topology)
{
	/* We don't care about the result */
	int worker_array[STARPU_NMAXWORKERS];
	int worker_cnt = 0;

	/* We recursively go from the root to the leaves of the tree to find
	 * subtrees that only have CPUs as leaves. */
	hwloc_obj_t root;
	root = hwloc_get_obj_by_depth(topology->hwtopology, HWLOC_OBJ_SYSTEM, 0); 
	find_combinations_with_hwloc_rec(root, worker_array, &worker_cnt);
}

#else

static void find_combinations_without_hwloc(struct starpu_machine_topology_s *topology)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	/* We put the id of all CPU workers in this array */
	int cpu_workers[STARPU_NMAXWORKERS];
	unsigned ncpus = 0;

	unsigned i;
	for (i = 0; i < topology->nworkers; i++)
	{
		if (config->workers[i].perf_arch == STARPU_CPU_DEFAULT)
			cpu_workers[ncpus++] = i;
	}
	
	unsigned size;
	for (size = 2; size <= ncpus; size *= 2)
	{
		unsigned first_cpu;
		for (first_cpu = 0; first_cpu < ncpus; first_cpu += size)
		{
			if (first_cpu + size <= ncpus)
			{
				int workerids[size];

				for (i = 0; i < size; i++)
					workerids[i] = cpu_workers[first_cpu + i];

				/* We register this combination */
				int ret;
				ret = starpu_combined_worker_assign_workerid(size, workerids); 
				STARPU_ASSERT(ret >= 0);
			}
		}
	}
}
#endif

static void combine_all_cpu_workers(struct starpu_machine_topology_s *topology)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	int cpu_workers[STARPU_NMAXWORKERS];
	unsigned ncpus = 0;

	unsigned i;
	for (i = 0; i < topology->nworkers; i++)
	{
		if (config->workers[i].perf_arch == STARPU_CPU_DEFAULT)
			cpu_workers[ncpus++] = i;
	}

	if (ncpus > 0)
	{
		int ret;
		ret = starpu_combined_worker_assign_workerid(ncpus, cpu_workers);
		STARPU_ASSERT(ret >= 0);
	}
}

void _starpu_sched_find_worker_combinations(struct starpu_machine_topology_s *topology)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	if (config->user_conf && config->user_conf->single_combined_worker)
		combine_all_cpu_workers(topology);
	else {
#ifdef STARPU_HAVE_HWLOC
		find_combinations_with_hwloc(topology);
		//find_combinations_without_hwloc(topology);
#else
		find_combinations_without_hwloc(topology);
#endif
	}
}
