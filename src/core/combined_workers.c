/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2013       Thibaut Lambert
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

#include <stdlib.h> // for qsort

#include <starpu.h>
#include <common/config.h>
#include <core/workers.h>

#ifdef __GLIBC__
#include <sched.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#endif

static int compar_int(const void *pa, const void *pb)
{
	int a = *((int *)pa);
	int b = *((int *)pb);

	return a - b;
}

static void sort_workerid_array(int nworkers, int workerid_array[])
{
	qsort(workerid_array, nworkers, sizeof(int), compar_int);
}

/* Create a new worker id for a combination of workers. This method should
 * typically be called at the initialization of the scheduling policy. This
 * worker should be the combination of the list of id's contained in the
 * workerid_array array which has nworkers entries. This function returns
 * the identifier of the combined worker in case of success, a negative value
 * is returned otherwise. */
int starpu_combined_worker_assign_workerid(int nworkers, int workerid_array[])
{
	int new_workerid;

	/* Return the number of actual workers. */
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	int basic_worker_count = (int)config->topology.nworkers;
	int combined_worker_id = (int)config->topology.ncombinedworkers;

	/* We sort the ids */
	sort_workerid_array(nworkers, workerid_array);

	/* Test that all workers are not combined workers already. */
	int i;
	for (i = 0; i < nworkers; i++)
	{
		int id = workerid_array[i];

		/* We only combine valid "basic" workers */
		if ((id < 0) || (id >= basic_worker_count))
			return -EINVAL;

#ifdef STARPU_USE_MIC
		STARPU_ASSERT(config->workers[id].arch == STARPU_CPU_WORKER || config->workers[id].arch == STARPU_MIC_WORKER);
		STARPU_ASSERT(config->workers[id].worker_mask == STARPU_CPU || config->workers[id].worker_mask == STARPU_MIC);
#else/* STARPU_USE_MIC */
		/* We only combine CPUs */
		STARPU_ASSERT(config->workers[id].arch == STARPU_CPU_WORKER);
		STARPU_ASSERT(config->workers[id].worker_mask == STARPU_CPU);
#endif /* STARPU_USE_MIC */
	}

	/* Get an id for that combined worker. Note that this is not thread
	 * safe because this method should only be called when the scheduler
	 * is being initialized. */
	new_workerid = basic_worker_count + combined_worker_id;
	STARPU_ASSERT_MSG(new_workerid < STARPU_NMAXWORKERS, "Too many combined workers for parallel task execution. Please use configure option --enable-maxcpus to increase it beyond the current value %d", STARPU_MAXCPUS);
	config->topology.ncombinedworkers++;

//	fprintf(stderr, "COMBINED WORKERS ");
//	for (i = 0; i < nworkers; i++)
//	{
//		fprintf(stderr, "%d ", workerid_array[i]);
//	}
//	fprintf(stderr, "into worker %d\n", new_workerid);

	for(i = 0; i < nworkers; i++)
		_starpu_get_worker_struct(workerid_array[i])->combined_workerid = new_workerid;

	struct _starpu_combined_worker *combined_worker =
		&config->combined_workers[combined_worker_id];

	combined_worker->worker_size = nworkers;
	_STARPU_MALLOC(combined_worker->perf_arch.devices, sizeof(struct starpu_perfmodel_device));
	combined_worker->perf_arch.ndevices = 1;
	combined_worker->perf_arch.devices[0].type = config->workers[workerid_array[0]].perf_arch.devices[0].type;
	combined_worker->perf_arch.devices[0].devid = config->workers[workerid_array[0]].perf_arch.devices[0].devid;
	combined_worker->perf_arch.devices[0].ncores = nworkers;
	combined_worker->worker_mask = config->workers[workerid_array[0]].worker_mask;

#ifdef STARPU_USE_MP
	combined_worker->count = nworkers -1;
	STARPU_PTHREAD_MUTEX_INIT(&combined_worker->count_mutex,NULL);
#endif

	/* We assume that the memory node should either be that of the first
	 * entry, and it is very likely that every worker in the combination
	 * should be on the same memory node.*/
	int first_id = workerid_array[0];
	combined_worker->memory_node = config->workers[first_id].memory_node;

	/* Save the list of combined workers */
	memcpy(&combined_worker->combined_workerid, workerid_array, nworkers*sizeof(int));

	/* Note that we maintain both the cpu_set and the hwloc_cpu_set so that
	 * the application is not forced to use hwloc when it is available. */
#ifdef __GLIBC__
	CPU_ZERO(&combined_worker->cpu_set);
#endif /* __GLIBC__ */

#ifdef STARPU_HAVE_HWLOC
	combined_worker->hwloc_cpu_set = hwloc_bitmap_alloc();
#endif

	for (i = 0; i < nworkers; i++)
	{
#if defined(__GLIBC__) || defined(STARPU_HAVE_HWLOC)
		int id = workerid_array[i];
#ifdef __GLIBC__
#ifdef CPU_OR
		CPU_OR(&combined_worker->cpu_set,
			&combined_worker->cpu_set,
			&config->workers[id].cpu_set);
#else
		int j;
		for (j = 0; j < CPU_SETSIZE; j++)
		{
			if (CPU_ISSET(j, &config->workers[id].cpu_set))
				CPU_SET(j, &combined_worker->cpu_set);
		}
#endif
#endif /* __GLIBC__ */

#ifdef STARPU_HAVE_HWLOC
		hwloc_bitmap_or(combined_worker->hwloc_cpu_set,
				combined_worker->hwloc_cpu_set,
				config->workers[id].hwloc_cpu_set);
#endif
#endif
	}

	starpu_sched_ctx_add_combined_workers(&new_workerid, 1, STARPU_GLOBAL_SCHED_CTX);

	return new_workerid;
}

int starpu_combined_worker_get_description(int workerid, int *worker_size, int **combined_workerid)
{
	/* Check that this is the id of a combined worker */
	struct _starpu_combined_worker *worker;
	worker = _starpu_get_combined_worker_struct(workerid);
	STARPU_ASSERT(worker);

	if (worker_size)
		*worker_size = worker->worker_size;

	if (combined_workerid)
		*combined_workerid = worker->combined_workerid;

	return 0;
}
