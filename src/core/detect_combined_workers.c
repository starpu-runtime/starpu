/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/utils.h>
#include <core/workers.h>
#include <math.h>
#include <core/detect_combined_workers.h>

int _starpu_initialized_combined_workers;

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>

static void find_workers(hwloc_obj_t obj, int cpu_workers[STARPU_NMAXWORKERS], unsigned *n)
{
	struct _starpu_hwloc_userdata *data = obj->userdata;
	if (!data->worker_list)
		/* Not something we run something on, don't care */
		return;
	if (data->worker_list == (void*) -1)
	{
		/* Intra node, recurse */
		unsigned i;
		for (i = 0; i < obj->arity; i++)
			find_workers(obj->children[i], cpu_workers, n);
		return;
	}

	/* Got to a PU leaf */
	struct _starpu_worker_list *workers = data->worker_list;
	struct _starpu_worker *worker;
	for(worker = _starpu_worker_list_begin(workers); worker != _starpu_worker_list_end(workers); worker = _starpu_worker_list_next(worker))
	{
		/* is it a CPU worker? */
		if (worker->perf_arch.devices[0].type == STARPU_CPU_WORKER && worker->perf_arch.devices[0].ncores == 1)
		{
			_STARPU_DEBUG("worker %d is part of it\n", worker->workerid);
			/* Add it to the combined worker */
			cpu_workers[(*n)++] = worker->workerid;
		}
	}
}

static void synthesize_intermediate_workers(hwloc_obj_t *children, unsigned min, unsigned max, unsigned arity, unsigned n, unsigned synthesize_arity)
{
	unsigned nworkers, i, j;
	unsigned chunk_size = (n + synthesize_arity-1) / synthesize_arity;
	unsigned chunk_start;
	int cpu_workers[STARPU_NMAXWORKERS];
	int ret;

	if (n <= synthesize_arity)
		/* Not too many children, do not synthesize */
		return;

	_STARPU_DEBUG("%u children > %u, synthesizing intermediate combined workers of size %u\n", n, synthesize_arity, chunk_size);

	n = 0;
	j = 0;
	nworkers = 0;
	chunk_start = 0;
	for (i = 0 ; i < arity; i++)
	{
		if (((struct _starpu_hwloc_userdata*)children[i]->userdata)->worker_list)
		{
			n++;
			_STARPU_DEBUG("child %u\n", i);
			find_workers(children[i], cpu_workers, &nworkers);
			j++;
		}
		/* Completed a chunk, or last bit (but not if it's just 1 subobject) */
		if (j == chunk_size || (i == arity-1 && j > 1))
		{
			if (nworkers >= min && nworkers <= max)
			{
				unsigned sched_ctx_id  = starpu_sched_ctx_get_context();
				if(sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
					sched_ctx_id = 0;
				struct starpu_worker_collection* workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

				_STARPU_DEBUG("Adding it\n");
				ret = starpu_combined_worker_assign_workerid(nworkers, cpu_workers);
				STARPU_ASSERT(ret >= 0);
				workers->add(workers,ret);
			}
			/* Recurse there */
			synthesize_intermediate_workers(children+chunk_start, min, max, i - chunk_start, n, synthesize_arity);
			/* And restart another one */
			n = 0;
			j = 0;
			nworkers = 0;
			chunk_start = i+1;
		}
	}
}

static void find_and_assign_combinations(hwloc_obj_t obj, unsigned min, unsigned max, unsigned synthesize_arity)
{
	char name[64];
	unsigned i, n, nworkers;
	int cpu_workers[STARPU_NMAXWORKERS];

#if HWLOC_API_VERSION >= 0x10000
	hwloc_obj_attr_snprintf(name, sizeof(name), obj, "#", 0);
#else
	hwloc_obj_snprintf(name, sizeof(name), _starpu_get_machine_config()->topology.hwtopology, obj, "#", 0);
#endif
	_STARPU_DEBUG("Looking at %s\n", name);

	for (n = 0, i = 0; i < obj->arity; i++)
		if (((struct _starpu_hwloc_userdata *)obj->children[i]->userdata)->worker_list)
			/* it has a CPU worker */
			n++;

	if (n == 1)
	{
		/* If there is only one child, we go to the next level right away */
		find_and_assign_combinations(obj->children[0], min, max, synthesize_arity);
		return;
	}

	/* Add this object */
	nworkers = 0;
	find_workers(obj, cpu_workers, &nworkers);

	if (nworkers >= min && nworkers <= max)
	{
		_STARPU_DEBUG("Adding it\n");
		unsigned sched_ctx_id  = starpu_sched_ctx_get_context();
		if(sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
			sched_ctx_id = 0;

		struct starpu_worker_collection* workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

		int newworkerid = starpu_combined_worker_assign_workerid(nworkers, cpu_workers);
		STARPU_ASSERT(newworkerid >= 0);
		workers->add(workers,newworkerid);
	}

	/* Add artificial intermediate objects recursively */
	synthesize_intermediate_workers(obj->children, min, max, obj->arity, n, synthesize_arity);

	/* And recurse */
	for (i = 0; i < obj->arity; i++)
		if (((struct _starpu_hwloc_userdata*) obj->children[i]->userdata)->worker_list == (void*) -1)
			find_and_assign_combinations(obj->children[i], min, max, synthesize_arity);
}

static void find_and_assign_combinations_with_hwloc(int *workerids, int nworkers)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	int synthesize_arity = starpu_get_env_number("STARPU_SYNTHESIZE_ARITY_COMBINED_WORKER");

	int min = starpu_get_env_number("STARPU_MIN_WORKERSIZE");
	if (min < 2)
		min = 2;
	int max = starpu_get_env_number("STARPU_MAX_WORKERSIZE");
	if (max == -1)
		max = INT_MAX;

	if (synthesize_arity == -1)
		synthesize_arity = 2;

	STARPU_ASSERT_MSG(synthesize_arity > 0, "STARPU_SYNTHESIZE_ARITY_COMBINED_WORKER must be greater than 0");

	/* First, mark nodes which contain CPU workers, simply by setting their userdata field */
	int i;
	for (i = 0; i < nworkers; i++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerids[i]);
		if (worker->perf_arch.devices[0].type == STARPU_CPU_WORKER && worker->perf_arch.devices[0].ncores == 1)
		{
			hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, config->pu_depth, worker->bindid);
			obj = obj->parent;
			while (obj)
			{
				((struct _starpu_hwloc_userdata*) obj->userdata)->worker_list = (void*) -1;
				obj = obj->parent;
			}
		}
	}
	find_and_assign_combinations(hwloc_get_root_obj(topology->hwtopology), min, max, synthesize_arity);
}

#else /* STARPU_HAVE_HWLOC */

static void assign_combinations_without_hwloc(struct starpu_worker_collection* worker_collection, int* workers, unsigned n, int min, int max)
{

	int size,i;
	//if the maximun number of worker is already reached
	if(worker_collection->nworkers >= STARPU_NMAXWORKERS - 1)
		return;

	for (size = min; size <= max; size *= 2)
	{
		unsigned first;
		for (first = 0; first < n; first += size)
		{
			if (first + size <= n)
			{
				int found_workerids[size];

				for (i = 0; i < size; i++)
					found_workerids[i] = workers[first + i];

				/* We register this combination */
				int newworkerid;
				newworkerid = starpu_combined_worker_assign_workerid(size, found_workerids);
				STARPU_ASSERT(newworkerid >= 0);
				worker_collection->add(worker_collection, newworkerid);
				//if the maximun number of worker is reached, then return
				if(worker_collection->nworkers >= STARPU_NMAXWORKERS - 1)
					return;
			}
		}
	}
}


static void find_and_assign_combinations_without_hwloc(int *workerids, int nworkers)
{
	int i;
	unsigned sched_ctx_id  = starpu_sched_ctx_get_context();
	if(sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
		sched_ctx_id = 0;
	int min, max;
#ifdef STARPU_USE_MIC
	unsigned j;
	int mic_min, mic_max;
#endif

	struct starpu_worker_collection* workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	/* We put the id of all CPU workers in this array */
	int cpu_workers[STARPU_NMAXWORKERS];
	unsigned ncpus = 0;
#ifdef STARPU_USE_MIC
	unsigned nb_mics = _starpu_get_machine_config()->topology.nmicdevices;
	unsigned * nmics_table;
	int * mic_id;
	int ** mic_workers;
	_STARPU_MALLOC(mic_id, sizeof(int)*nb_mics);
	_STARPU_MALLOC(nmics_table, sizeof(unsigned)*nb_mics);
	_STARPU_MALLOC(mic_workers, sizeof(int*)*nb_mics);
	for(j=0; j<nb_mics; j++)
	{
		mic_id[j] = -1;
		nmics_table[j] = 0;
		_STARPU_MALLOC(mic_workers[j], sizeof(int)*STARPU_NMAXWORKERS);
	}
#endif /* STARPU_USE_MIC */

	for (i = 0; i < nworkers; i++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerids[i]);
		if (worker->arch == STARPU_CPU_WORKER)
			cpu_workers[ncpus++] = i;
#ifdef STARPU_USE_MIC
		else if(worker->arch == STARPU_MIC_WORKER)
		{
			for(j=0; j<nb_mics && mic_id[j] != worker->devid && mic_id[j] != -1; j++);
			if(j<nb_mics)
			{
				if(mic_id[j] == -1)
				{
					mic_id[j] = worker->devid;
				}
				mic_workers[j][nmics_table[j]++] = i;
			}
		}
#endif /* STARPU_USE_MIC */

	}


	min = starpu_get_env_number("STARPU_MIN_WORKERSIZE");
	if (min < 2)
		min = 2;
	max = starpu_get_env_number("STARPU_MAX_WORKERSIZE");
	if (max == -1 || max > (int) ncpus)
		max = ncpus;

	assign_combinations_without_hwloc(workers,cpu_workers,ncpus,min,max);
#ifdef STARPU_USE_MIC
	mic_min = starpu_get_env_number("STARPU_MIN_WORKERSIZE");
	mic_max = starpu_get_env_number("STARPU_MAX_WORKERSIZE");
	if (mic_min < 2)
		mic_min = 2;
	for(j=0; j<nb_mics; j++)
	{
		int _mic_max = mic_max;
		if (_mic_max == -1 || _mic_max > (int) nmics_table[j])
			_mic_max = nmics_table[j];
		assign_combinations_without_hwloc(workers,mic_workers[j],nmics_table[j],mic_min,_mic_max);
		free(mic_workers[j]);
	}
	free(mic_id);
	free(nmics_table);
	free(mic_workers);
#endif /* STARPU_USE_MIC */
}

#endif /* STARPU_HAVE_HWLOC */

static void combine_all_cpu_workers(int *workerids, int nworkers)
{
	unsigned sched_ctx_id  = starpu_sched_ctx_get_context();
	if(sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
		sched_ctx_id = 0;
	struct starpu_worker_collection* workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	int cpu_workers[STARPU_NMAXWORKERS];
	int ncpus = 0;
	int i;
	int min;
	int max;

	for (i = 0; i < nworkers; i++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerids[i]);

		if (worker->arch == STARPU_CPU_WORKER)
			cpu_workers[ncpus++] = workerids[i];
	}

	min = starpu_get_env_number("STARPU_MIN_WORKERSIZE");
	if (min < 1)
		min = 1;
	max = starpu_get_env_number("STARPU_MAX_WORKERSIZE");
	if (max == -1 || max > ncpus)
		max = ncpus;

	for (i = min; i <= max; i++)
	{
		int newworkerid = starpu_combined_worker_assign_workerid(i, cpu_workers);
		STARPU_ASSERT(newworkerid >= 0);
		workers->add(workers, newworkerid);
	}
}

void _starpu_sched_find_worker_combinations(int *workerids, int nworkers)
{
	/* FIXME: this seems to be lacking shutdown support? */

	if (_starpu_initialized_combined_workers)
		return;
	_starpu_initialized_combined_workers = 1;

	struct _starpu_machine_config *config = _starpu_get_machine_config();

	if (config->conf.single_combined_worker > 0)
		combine_all_cpu_workers(workerids, nworkers);
	else
	{
#ifdef STARPU_HAVE_HWLOC
		find_and_assign_combinations_with_hwloc(workerids, nworkers);
#else
		find_and_assign_combinations_without_hwloc(workerids, nworkers);
#endif
	}
}

void starpu_sched_find_all_worker_combinations(void)
{
	const unsigned nbasic_workers = starpu_worker_get_count();
	int basic_workerids[nbasic_workers];
	unsigned i;
	for(i = 0; i < nbasic_workers; i++)
	{
		basic_workerids[i] = i;
	}

	_starpu_sched_find_worker_combinations(basic_workerids, nbasic_workers);
}
