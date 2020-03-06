/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/sched_policy.h>
#include <core/sched_ctx.h>
#include <common/utils.h>
#include <stdarg.h>

starpu_pthread_rwlock_t changing_ctx_mutex[STARPU_NMAX_SCHED_CTXS];

extern struct starpu_worker_collection worker_list;
static starpu_pthread_mutex_t sched_ctx_manag = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_mutex_t finished_submit_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static struct starpu_task stop_submission_task = STARPU_TASK_INITIALIZER;
starpu_pthread_key_t sched_ctx_key;
static unsigned with_hypervisor = 0;
static double hyp_start_sample[STARPU_NMAX_SCHED_CTXS];
static double hyp_start_allow_sample[STARPU_NMAX_SCHED_CTXS];
static double flops[STARPU_NMAX_SCHED_CTXS][STARPU_NMAXWORKERS];
static size_t data_size[STARPU_NMAX_SCHED_CTXS][STARPU_NMAXWORKERS];

static unsigned _starpu_get_first_free_sched_ctx(struct _starpu_machine_config *config);

static void _starpu_worker_gets_into_ctx(unsigned sched_ctx_id, struct _starpu_worker *worker)
{
	unsigned ret_sched_ctx = _starpu_sched_ctx_list_get_sched_ctx(worker->sched_ctx_list, sched_ctx_id);
	/* the worker was planning to go away in another ctx but finally he changed his mind & 
	   he's staying */
	if (ret_sched_ctx == STARPU_NMAX_SCHED_CTXS)
	{
		/* add context to worker */
		_starpu_sched_ctx_list_add(&worker->sched_ctx_list, sched_ctx_id);
		worker->nsched_ctxs++;
	}
	worker->removed_from_ctx[sched_ctx_id] = 0;
	return;
}

void _starpu_worker_gets_out_of_ctx(unsigned sched_ctx_id, struct _starpu_worker *worker)
{
	unsigned ret_sched_ctx = _starpu_sched_ctx_list_get_sched_ctx(worker->sched_ctx_list, sched_ctx_id);
	/* remove context from worker */
	if(ret_sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
		if(sched_ctx && sched_ctx->sched_policy && sched_ctx->sched_policy->remove_workers)
		{
			_STARPU_TRACE_WORKER_SCHEDULING_PUSH;
			sched_ctx->sched_policy->remove_workers(sched_ctx_id, &worker->workerid, 1);
			_STARPU_TRACE_WORKER_SCHEDULING_POP;
		}
		_starpu_sched_ctx_list_remove(&worker->sched_ctx_list, sched_ctx_id);
		worker->nsched_ctxs--;
	}
	return;
}

static void _starpu_update_workers_with_ctx(int *workerids, int nworkers, int sched_ctx_id)
{
	int i;
	struct _starpu_worker *worker = NULL;
 	struct _starpu_worker *curr_worker = _starpu_get_local_worker_key();
	
	for(i = 0; i < nworkers; i++)
	{
		worker = _starpu_get_worker_struct(workerids[i]);

		/* if the current thread requires resize it's no need
		   to lock it in order to change its  sched_ctx info */
		if(curr_worker && curr_worker == worker)
			_starpu_worker_gets_into_ctx(sched_ctx_id, worker);
		else
		{
			STARPU_PTHREAD_MUTEX_LOCK(&worker->sched_mutex);
			_starpu_worker_gets_into_ctx(sched_ctx_id, worker);
			STARPU_PTHREAD_MUTEX_UNLOCK(&worker->sched_mutex);
		}
	}

	return;
}

static void _starpu_update_workers_without_ctx(int *workerids, int nworkers, int sched_ctx_id, unsigned now)
{
	int i;
	struct _starpu_worker *worker = NULL;
 	struct _starpu_worker *curr_worker = _starpu_get_local_worker_key();
	
	for(i = 0; i < nworkers; i++)
	{
		worker = _starpu_get_worker_struct(workerids[i]);
		if(now)
		{
			if(curr_worker && curr_worker == worker)
				_starpu_worker_gets_out_of_ctx(sched_ctx_id, worker);
			else
			{
					STARPU_PTHREAD_MUTEX_LOCK(&worker->sched_mutex);
					_starpu_worker_gets_out_of_ctx(sched_ctx_id, worker);
					STARPU_PTHREAD_MUTEX_UNLOCK(&worker->sched_mutex);
			}
		}
		else
		{
			if(curr_worker && curr_worker == worker)
				worker->removed_from_ctx[sched_ctx_id] = 1;
			else
			{
				STARPU_PTHREAD_MUTEX_LOCK(&worker->sched_mutex);
				worker->removed_from_ctx[sched_ctx_id] = 1;
				STARPU_PTHREAD_MUTEX_UNLOCK(&worker->sched_mutex);
			}
		}
	}
	return;
}

void starpu_sched_ctx_stop_task_submission()
{
	_starpu_exclude_task_from_dag(&stop_submission_task);
	_starpu_task_submit_internally(&stop_submission_task);
}

void starpu_sched_ctx_worker_shares_tasks_lists(int workerid, int sched_ctx_id)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int curr_workerid = starpu_worker_get_id();
	/* if is the initial sched_ctx no point in taking the mutex, the workers are
	   not launched yet, or if the current worker is calling this */
	if(!sched_ctx->is_initial_sched && workerid != curr_workerid)
		STARPU_PTHREAD_MUTEX_LOCK(&worker->sched_mutex);

	worker->shares_tasks_lists[sched_ctx_id] = 1;

	if(!sched_ctx->is_initial_sched && workerid != curr_workerid)
		STARPU_PTHREAD_MUTEX_UNLOCK(&worker->sched_mutex);
}

static void _starpu_add_workers_to_sched_ctx(struct _starpu_sched_ctx *sched_ctx, int *workerids, int nworkers,
				       int *added_workers, int *n_added_workers)
{
	struct starpu_worker_collection *workers = sched_ctx->workers;
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();

	int nworkers_to_add = nworkers == -1 ? (int)config->topology.nworkers : nworkers;
	if (!nworkers_to_add)
		return;
	int workers_to_add[nworkers_to_add];

	int i = 0;
	for(i = 0; i < nworkers_to_add; i++)
	{
		/* added_workers is NULL for the call of this func at the creation of the context*/
		/* if the function is called at the creation of the context it's no need to do this verif */
		if(added_workers)
		{
			int worker = workers->add(workers, (workerids == NULL ? i : workerids[i]));
			if(worker >= 0)
				added_workers[(*n_added_workers)++] = worker;
			else
			{
				int curr_workerid = starpu_worker_get_id();
				struct _starpu_worker *worker_str = _starpu_get_worker_struct(workerids[i]);
				if(curr_workerid != workerids[i])
					STARPU_PTHREAD_MUTEX_LOCK(&worker_str->sched_mutex);

				worker_str->removed_from_ctx[sched_ctx->id] = 0;

				if(curr_workerid != workerids[i])
					STARPU_PTHREAD_MUTEX_UNLOCK(&worker_str->sched_mutex);
			}
		}
		else
		{
			int worker = (workerids == NULL ? i : workerids[i]);
			workers->add(workers, worker);
			workers_to_add[i] = worker;
		}
	}

	if(sched_ctx->sched_policy->add_workers)
	{
		_STARPU_TRACE_WORKER_SCHEDULING_PUSH;
		if(added_workers)
		{
			if(*n_added_workers > 0)
				sched_ctx->sched_policy->add_workers(sched_ctx->id, added_workers, *n_added_workers);
		}
		else
			sched_ctx->sched_policy->add_workers(sched_ctx->id, workers_to_add, nworkers_to_add);
		_STARPU_TRACE_WORKER_SCHEDULING_POP;
	}
	return;
}

static void _starpu_remove_workers_from_sched_ctx(struct _starpu_sched_ctx *sched_ctx, int *workerids,
						  int nworkers, int *removed_workers, int *n_removed_workers)
{
	struct starpu_worker_collection *workers = sched_ctx->workers;

	int i = 0;

	for(i = 0; i < nworkers; i++)
	{
		if(workers->nworkers > 0)
		{
			if(_starpu_worker_belongs_to_a_sched_ctx(workerids[i], sched_ctx->id))
			{
				int worker = workers->remove(workers, workerids[i]);
				if(worker >= 0)
					removed_workers[(*n_removed_workers)++] = worker;
			}
		}
	}

	return;
}

static void _starpu_sched_ctx_free_scheduling_data(struct _starpu_sched_ctx *sched_ctx)
{
	int *workerids = NULL;

	unsigned nworkers_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &workerids);

	if(nworkers_ctx > 0 && sched_ctx->sched_policy->remove_workers)
	{
		_STARPU_TRACE_WORKER_SCHEDULING_PUSH;
		sched_ctx->sched_policy->remove_workers(sched_ctx->id, workerids, nworkers_ctx);
		_STARPU_TRACE_WORKER_SCHEDULING_POP;
	}

	free(workerids);
	return;

}

struct _starpu_sched_ctx* _starpu_create_sched_ctx(struct starpu_sched_policy *policy, int *workerids,
						   int nworkers_ctx, unsigned is_initial_sched,
						   const char *sched_ctx_name,
						   int min_prio_set, int min_prio,
						   int max_prio_set, int max_prio)
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();

	STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	STARPU_ASSERT(config->topology.nsched_ctxs < STARPU_NMAX_SCHED_CTXS);

	unsigned id = _starpu_get_first_free_sched_ctx(config);

	struct _starpu_sched_ctx *sched_ctx = &config->sched_ctxs[id];
	sched_ctx->id = id;

	config->topology.nsched_ctxs++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);

	int nworkers = config->topology.nworkers;

	STARPU_ASSERT(nworkers_ctx <= nworkers);

	STARPU_PTHREAD_MUTEX_INIT(&sched_ctx->empty_ctx_mutex, NULL);

	starpu_task_list_init(&sched_ctx->empty_ctx_tasks);

	sched_ctx->sched_policy = (struct starpu_sched_policy*)malloc(sizeof(struct starpu_sched_policy));
	sched_ctx->is_initial_sched = is_initial_sched;
	sched_ctx->name = sched_ctx_name;
	sched_ctx->inheritor = STARPU_NMAX_SCHED_CTXS;
	sched_ctx->finished_submit = 0;
	sched_ctx->min_priority_is_set = min_prio_set;
	if (sched_ctx->min_priority_is_set) sched_ctx->min_priority = min_prio;
	sched_ctx->max_priority_is_set = max_prio_set;
	if (sched_ctx->max_priority_is_set) sched_ctx->max_priority = max_prio;

	_starpu_barrier_counter_init(&sched_ctx->tasks_barrier, 0);
	_starpu_barrier_counter_init(&sched_ctx->ready_tasks_barrier, 0);

	sched_ctx->ready_flops = 0.0;
	/*init the strategy structs and the worker_collection of the ressources of the context */
	_starpu_init_sched_policy(config, sched_ctx, policy);

	/* construct the collection of workers(list/tree/etc.) */
	sched_ctx->workers->init(sched_ctx->workers);

	if(is_initial_sched)
	{
		int i;
		/*initialize the mutexes for all contexts */
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		  {
			STARPU_PTHREAD_RWLOCK_INIT(&changing_ctx_mutex[i], NULL);
		  }
	}

	/* after having an worker_collection on the ressources add them */
	_starpu_add_workers_to_sched_ctx(sched_ctx, workerids, nworkers_ctx, NULL, NULL);


	/* if we create the initial big sched ctx we can update workers' status here
	   because they haven't been launched yet */
	if(is_initial_sched)
	{
		int i;
		for(i = 0; i < nworkers; i++)
		{
			struct _starpu_worker *worker = _starpu_get_worker_struct(i);
			worker->sched_ctx_list = (struct _starpu_sched_ctx_list*)malloc(sizeof(struct _starpu_sched_ctx_list));
                       _starpu_sched_ctx_list_init(worker->sched_ctx_list);
		       _starpu_sched_ctx_list_add(&worker->sched_ctx_list, sched_ctx->id);
			worker->nsched_ctxs++;
		}
	}

	return sched_ctx;
}

static void _get_workers(int min, int max, int *workers, int *nw, enum starpu_worker_archtype arch, unsigned allow_overlap)
{
	int pus[max];
	int npus = 0;
	int i;
	int n = 0;

	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	if(config->topology.nsched_ctxs == 1)
	{
		/*we have all available resources */
		npus = starpu_worker_get_nids_by_type(arch, pus, max);
/*TODO: hierarchical ctxs: get max good workers: close one to another */
		for(i = 0; i < npus; i++)
			workers[(*nw)++] = pus[i];
	}
	else
	{
		unsigned enough_ressources = 0;
		npus = starpu_worker_get_nids_ctx_free_by_type(arch, pus, max);

		for(i = 0; i < npus; i++)
			workers[(*nw)++] = pus[i];

		if(npus == max)
			/*we have enough available resources */
			enough_ressources = 1;

		if(!enough_ressources && npus >= min)
			/*we have enough available resources */
			enough_ressources = 1;

		if(!enough_ressources)
		{
			/* try to get ressources from ctx who have more than the min of workers they need */
			int s;
			for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
				{
					int _npus = 0;
					int _pus[STARPU_NMAXWORKERS];
					_npus = starpu_get_workers_of_sched_ctx(config->sched_ctxs[s].id, _pus, arch);
					int ctx_min = arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;
					if(_npus > ctx_min)
					{
						if(npus < min)
						{
							n = (_npus - ctx_min) > (min - npus) ? min - npus : (_npus - ctx_min);
							npus += n;
						}
/*TODO: hierarchical ctxs: get n good workers: close to the other ones I already assigned to the ctx */
						for(i = 0; i < n; i++)
							workers[(*nw)++] = _pus[i];
						starpu_sched_ctx_remove_workers(_pus, n, config->sched_ctxs[s].id);
					}
				}
			}

			if(npus >= min)
				enough_ressources = 1;
		}

		if(!enough_ressources)
		{
			/* if there is no available workers to satisfy the  minimum required
			 give them workers proportional to their requirements*/
			int global_npus = starpu_worker_get_count_by_type(arch);

			int req_npus = 0;

			int s;
			for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
				if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
					req_npus += arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;

			req_npus += min;

			for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
				{
					int ctx_min = arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;
					double needed_npus = ((double)ctx_min * (double)global_npus) / (double)req_npus;

					int _npus = 0;
					int _pus[STARPU_NMAXWORKERS];

					_npus = starpu_get_workers_of_sched_ctx(config->sched_ctxs[s].id, _pus, arch);
					if(needed_npus < (double)_npus)
					{
						double npus_to_rem = (double)_npus - needed_npus;
						int x = floor(npus_to_rem);
						double x_double = (double)x;
						double diff = npus_to_rem - x_double;
						int npus_to_remove = diff >= 0.5 ? x+1 : x;

						int pus_to_remove[npus_to_remove];
						int c = 0;

/*TODO: hierarchical ctxs: get npus_to_remove good workers: close to the other ones I already assigned to the ctx */
						for(i = _npus-1; i >= (_npus - npus_to_remove); i--)
						{
							workers[(*nw)++] = _pus[i];
							pus_to_remove[c++] = _pus[i];
						}
						if(!allow_overlap)
							starpu_sched_ctx_remove_workers(pus_to_remove, npus_to_remove, config->sched_ctxs[s].id);
					}

				}
			}
		}
	}
}

unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_ctx_name,
						 int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus,
						 unsigned allow_overlap)
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	struct starpu_sched_policy *selected_policy = _starpu_select_sched_policy(config, policy_name);

	struct _starpu_sched_ctx *sched_ctx = NULL;
	int workers[max_ncpus + max_ngpus];
	int nw = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	_get_workers(min_ncpus, max_ncpus, workers, &nw, STARPU_CPU_WORKER, allow_overlap);
	_get_workers(min_ngpus, max_ngpus, workers, &nw, STARPU_CUDA_WORKER, allow_overlap);
	STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);
	int i;
	printf("%d: ", nw);
	for(i = 0; i < nw; i++)
		printf("%d ", workers[i]);
	printf("\n");
	sched_ctx = _starpu_create_sched_ctx(selected_policy, workers, nw, 0, sched_ctx_name, 0, 0, 0, 0);
	sched_ctx->min_ncpus = min_ncpus;
	sched_ctx->max_ncpus = max_ncpus;
	sched_ctx->min_ngpus = min_ngpus;
	sched_ctx->max_ngpus = max_ngpus;

	_starpu_update_workers_without_ctx(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, sched_ctx->id, 0);
#ifdef STARPU_USE_SC_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return sched_ctx->id;

}

unsigned starpu_sched_ctx_create(int *workerids, int nworkers, const char *sched_ctx_name, ...)
{
	va_list varg_list;
	int arg_type;
	int min_prio_set = 0;
	int max_prio_set = 0;
	int min_prio = 0;
	int max_prio = 0;
	struct starpu_sched_policy *sched_policy = NULL;

	va_start(varg_list, sched_ctx_name);
	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type == STARPU_SCHED_CTX_POLICY_NAME)
		{
			char *policy_name = va_arg(varg_list, char *);
			struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
			sched_policy = _starpu_select_sched_policy(config, policy_name);
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_STRUCT)
		{
			sched_policy = va_arg(varg_list, struct starpu_sched_policy *);
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_MIN_PRIO)
		{
			min_prio = va_arg(varg_list, int);
			min_prio_set = 1;
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_MAX_PRIO)
		{
			max_prio = va_arg(varg_list, int);
			max_prio_set = 1;
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}

	}
	va_end(varg_list);

	struct _starpu_sched_ctx *sched_ctx = NULL;
	sched_ctx = _starpu_create_sched_ctx(sched_policy, workerids, nworkers, 0, sched_ctx_name, min_prio_set, min_prio, max_prio_set, max_prio);

	_starpu_update_workers_with_ctx(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, sched_ctx->id);
#ifdef STARPU_USE_SC_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return sched_ctx->id;
}

void starpu_sched_ctx_register_close_callback(unsigned sched_ctx_id, void (*close_callback)(unsigned sched_ctx_id, void* args), void *args)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->close_callback = close_callback;
	sched_ctx->close_args = args;
	return;
}

#ifdef STARPU_USE_SC_HYPERVISOR
void starpu_sched_ctx_set_perf_counters(unsigned sched_ctx_id, void* perf_counters)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->perf_counters = (struct starpu_sched_ctx_performance_counters *)perf_counters;
	return;
}
#endif

/* free all structures for the context */
static void _starpu_delete_sched_ctx(struct _starpu_sched_ctx *sched_ctx)
{
	STARPU_ASSERT(sched_ctx->id != STARPU_NMAX_SCHED_CTXS);
	_starpu_deinit_sched_policy(sched_ctx);
	free(sched_ctx->sched_policy);
	sched_ctx->sched_policy = NULL;

	STARPU_PTHREAD_MUTEX_DESTROY(&sched_ctx->empty_ctx_mutex);
	sched_ctx->id = STARPU_NMAX_SCHED_CTXS;

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	config->topology.nsched_ctxs--;
	STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);
}

void starpu_sched_ctx_delete(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
#ifdef STARPU_USE_SC_HYPERVISOR
	if(sched_ctx != NULL && sched_ctx_id != 0 && sched_ctx_id != STARPU_NMAX_SCHED_CTXS
	   && sched_ctx->perf_counters != NULL)
	{
		_STARPU_TRACE_HYPERVISOR_BEGIN();
		sched_ctx->perf_counters->notify_delete_context(sched_ctx_id);
		_STARPU_TRACE_HYPERVISOR_END();
	}
#endif //STARPU_USE_SC_HYPERVISOR

	unsigned inheritor_sched_ctx_id = sched_ctx->inheritor;
	struct _starpu_sched_ctx *inheritor_sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx->inheritor);

	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx_id]);
	STARPU_ASSERT(sched_ctx->id != STARPU_NMAX_SCHED_CTXS);

	int *workerids;
	unsigned nworkers_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &workerids);
	
	/*if both of them have all the ressources is pointless*/
	/*trying to transfer ressources from one ctx to the other*/
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	unsigned nworkers = config->topology.nworkers;

	if(nworkers_ctx > 0 && inheritor_sched_ctx && inheritor_sched_ctx->id != STARPU_NMAX_SCHED_CTXS && 
	   !(nworkers_ctx == nworkers && nworkers_ctx == inheritor_sched_ctx->workers->nworkers))
	{
		starpu_sched_ctx_add_workers(workerids, nworkers_ctx, inheritor_sched_ctx_id);
	}

	if(!_starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx_id))
	{
		/*if btw the mutex release & the mutex lock the context has changed take care to free all
		  scheduling data before deleting the context */
		_starpu_update_workers_without_ctx(workerids, nworkers_ctx, sched_ctx_id, 1);
//		_starpu_sched_ctx_free_scheduling_data(sched_ctx);
		_starpu_delete_sched_ctx(sched_ctx);

	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);
	/* workerids is malloc-ed in starpu_sched_ctx_get_workers_list, don't forget to free it when
	   you don't use it anymore */
	free(workerids);
	return;
}

/* called after the workers are terminated so we don't have anything else to do but free the memory*/
void _starpu_delete_all_sched_ctxs()
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(i);
		STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[i]);
		if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
		{
			_starpu_sched_ctx_free_scheduling_data(sched_ctx);
			_starpu_barrier_counter_destroy(&sched_ctx->tasks_barrier);
			_starpu_barrier_counter_destroy(&sched_ctx->ready_tasks_barrier);
			_starpu_delete_sched_ctx(sched_ctx);
		}
		STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[i]);

		STARPU_PTHREAD_RWLOCK_DESTROY(&changing_ctx_mutex[i]);
	}

	STARPU_PTHREAD_KEY_DELETE(sched_ctx_key);
	return;
}

static void _starpu_check_workers(int *workerids, int nworkers)
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	int nworkers_conf = config->topology.nworkers;

	int i;
	for(i = 0; i < nworkers; i++)
	{
		/* take care the user does not ask for a resource that does not exist */
		STARPU_ASSERT_MSG(workerids[i] >= 0 &&  workerids[i] <= nworkers_conf, "requested to add workerid = %d, but that is beyond the range 0 to %d", workerids[i], nworkers_conf);
	}
}

void _starpu_fetch_tasks_from_empty_ctx_list(struct _starpu_sched_ctx *sched_ctx)
{
	unsigned unlocked = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->empty_ctx_mutex);

	if(starpu_task_list_empty(&sched_ctx->empty_ctx_tasks))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->empty_ctx_mutex);
		return;
	}
	else
                /* you're not suppose to get here if you deleted the context
		   so no point in having the mutex locked */
		STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx->id]);
	
	while(!starpu_task_list_empty(&sched_ctx->empty_ctx_tasks))
	{
		if(unlocked)
			STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->empty_ctx_mutex);
		struct starpu_task *old_task = starpu_task_list_pop_back(&sched_ctx->empty_ctx_tasks);
		unlocked = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->empty_ctx_mutex);

		if(old_task == &stop_submission_task)
			break;

		int ret =  _starpu_push_task_to_workers(old_task);
		/* if we should stop poping from empty ctx tasks */
		if(ret == -EAGAIN) break;
	}
	if(!unlocked)
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->empty_ctx_mutex);


	/* leave the mutex as it was to avoid pbs in the caller function */
	STARPU_PTHREAD_RWLOCK_RDLOCK(&changing_ctx_mutex[sched_ctx->id]);
	return;

}
void starpu_sched_ctx_add_workers(int *workers_to_add, int nworkers_to_add, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int added_workers[nworkers_to_add];
	int n_added_workers = 0;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx_id]);

	STARPU_ASSERT(workers_to_add != NULL && nworkers_to_add > 0);
	_starpu_check_workers(workers_to_add, nworkers_to_add);

	/* if the context has not already been deleted */
	if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
	{
		_starpu_add_workers_to_sched_ctx(sched_ctx, workers_to_add, nworkers_to_add, added_workers, &n_added_workers);
		
		if(n_added_workers > 0)
		{
			_starpu_update_workers_with_ctx(added_workers, n_added_workers, sched_ctx->id);
		}

	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);

	if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
	{
		STARPU_PTHREAD_RWLOCK_RDLOCK(&changing_ctx_mutex[sched_ctx_id]);
		_starpu_fetch_tasks_from_empty_ctx_list(sched_ctx);
		STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);
	}

	return;
}

void starpu_sched_ctx_remove_workers(int *workers_to_remove, int nworkers_to_remove, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int removed_workers[sched_ctx->workers->nworkers];
	int n_removed_workers = 0;

	_starpu_check_workers(workers_to_remove, nworkers_to_remove);

	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx_id]);
	/* if the context has not already been deleted */
	if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
	{
		_starpu_remove_workers_from_sched_ctx(sched_ctx, workers_to_remove, nworkers_to_remove, removed_workers, &n_removed_workers);

		if(n_removed_workers > 0)
			_starpu_update_workers_without_ctx(removed_workers, n_removed_workers, sched_ctx->id, 0);

	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);
	return;
}

int _starpu_nworkers_able_to_execute_task(struct starpu_task *task, struct _starpu_sched_ctx *sched_ctx)
{
	unsigned worker = 0, nworkers = 0;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx->id]);
	struct starpu_worker_collection *workers = sched_ctx->workers;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		STARPU_ASSERT_MSG(worker < STARPU_NMAXWORKERS, "worker id %d", worker);
		if (starpu_worker_can_execute_task(worker, task, 0))
			nworkers++;
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx->id]);

	return nworkers;
}

/* unused sched_ctx have the id STARPU_NMAX_SCHED_CTXS */
void _starpu_init_all_sched_ctxs(struct _starpu_machine_config *config)
{
	STARPU_PTHREAD_KEY_CREATE(&sched_ctx_key, NULL);

	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		config->sched_ctxs[i].id = STARPU_NMAX_SCHED_CTXS;

	return;
}

/* sched_ctx aren't necessarly one next to another */
/* for eg when we remove one its place is free */
/* when we add  new one we reuse its place */
static unsigned _starpu_get_first_free_sched_ctx(struct _starpu_machine_config *config)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(config->sched_ctxs[i].id == STARPU_NMAX_SCHED_CTXS)
			return i;

	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

int _starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait_for_all must not be called from a task or callback");

	return _starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->tasks_barrier);
}

int _starpu_wait_for_n_submitted_tasks_of_sched_ctx(unsigned sched_ctx_id, unsigned n)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait_for_n_submitted_tasks must not be called from a task or callback");

	return _starpu_barrier_counter_wait_until_counter_reaches_down_to_n(&sched_ctx->tasks_barrier, n);
}

void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
#ifndef STARPU_SANITIZE_THREAD
	if (!config->watchdog_ok)
		config->watchdog_ok = 1;
#endif

	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int reached = _starpu_barrier_counter_get_reached_start(&sched_ctx->tasks_barrier);
	int finished = reached == 1;
        /* when finished decrementing the tasks if the user signaled he will not submit tasks anymore
           we can move all its workers to the inheritor context */
	if(finished && sched_ctx->inheritor != STARPU_NMAX_SCHED_CTXS)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&finished_submit_mutex);
		if(sched_ctx->finished_submit)
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&finished_submit_mutex);

			if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
			{
				if(sched_ctx->close_callback)
					sched_ctx->close_callback(sched_ctx->id, sched_ctx->close_args);

				int *workerids = NULL;
				unsigned nworkers = starpu_sched_ctx_get_workers_list(sched_ctx->id, &workerids);
				
				if(nworkers > 0)
				{
					starpu_sched_ctx_add_workers(workerids, nworkers, sched_ctx->inheritor);
					free(workerids);
				}
			}
			_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->tasks_barrier, 0.0);
			return;
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&finished_submit_mutex);
	}

	/* We also need to check for config->submitting = 0 (i.e. the
	 * user calle starpu_drivers_request_termination()), in which
	 * case we need to set config->running to 0 and wake workers,
	 * so they can terminate, just like
	 * starpu_drivers_request_termination() does.
	 */

	STARPU_PTHREAD_MUTEX_LOCK(&config->submitted_mutex);
	if(config->submitting == 0)
	{
		if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
		{
			if(sched_ctx->close_callback)
				sched_ctx->close_callback(sched_ctx->id, sched_ctx->close_args);
		}

		ANNOTATE_HAPPENS_AFTER(&config->running);
		config->running = 0;
		ANNOTATE_HAPPENS_BEFORE(&config->running);
		int s;
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
			{
				_starpu_check_nsubmitted_tasks_of_sched_ctx(config->sched_ctxs[s].id);
			}
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&config->submitted_mutex);

	_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->tasks_barrier, 0.0);

	return;
}

void _starpu_increment_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_increment(&sched_ctx->tasks_barrier, 0.0);
}

int _starpu_get_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return _starpu_barrier_counter_get_reached_start(&sched_ctx->tasks_barrier);
}

int _starpu_check_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return _starpu_barrier_counter_check(&sched_ctx->tasks_barrier);
}

void _starpu_increment_nready_tasks_of_sched_ctx(unsigned sched_ctx_id, double ready_flops)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_increment(&sched_ctx->ready_tasks_barrier, ready_flops);
}

void _starpu_decrement_nready_tasks_of_sched_ctx(unsigned sched_ctx_id, double ready_flops)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->ready_tasks_barrier, ready_flops);
}

int starpu_sched_ctx_get_nready_tasks(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return _starpu_barrier_counter_get_reached_start(&sched_ctx->ready_tasks_barrier);
}

double starpu_sched_ctx_get_nready_flops(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return _starpu_barrier_counter_get_reached_flops(&sched_ctx->ready_tasks_barrier);
}

int _starpu_wait_for_no_ready_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return _starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->ready_tasks_barrier);
}

void starpu_sched_ctx_set_context(unsigned *sched_ctx)
{
	starpu_pthread_setspecific(sched_ctx_key, (void*)sched_ctx);
}

unsigned starpu_sched_ctx_get_context()
{
	unsigned *sched_ctx = (unsigned*)starpu_pthread_getspecific(sched_ctx_key);
	if(sched_ctx == NULL)
		return STARPU_NMAX_SCHED_CTXS;
	STARPU_ASSERT(*sched_ctx < STARPU_NMAX_SCHED_CTXS);
	return *sched_ctx;
}

unsigned _starpu_sched_ctx_get_current_context()
{
	unsigned sched_ctx = starpu_sched_ctx_get_context();
	if (sched_ctx == STARPU_NMAX_SCHED_CTXS)
		return _starpu_get_initial_sched_ctx()->id;
	else
		return sched_ctx;
}

void starpu_sched_ctx_notify_hypervisor_exists()
{
	with_hypervisor = 1;
	int i, j;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hyp_start_sample[i] = starpu_timing_now();
		hyp_start_allow_sample[i] = 0.0;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			flops[i][j] = 0.0;
			data_size[i][j] = 0;
		}
	}
}

unsigned starpu_sched_ctx_check_if_hypervisor_exists()
{
	return with_hypervisor;
}

unsigned _starpu_sched_ctx_allow_hypervisor(unsigned sched_ctx_id)
{
	return 1;
	double now = starpu_timing_now();
	if(hyp_start_allow_sample[sched_ctx_id] > 0.0)
	{
		double allow_sample = (now - hyp_start_allow_sample[sched_ctx_id]) / 1000000.0;
		if(allow_sample < 0.001)
			return 1;
		else
		{
			hyp_start_allow_sample[sched_ctx_id] = 0.0;
			hyp_start_sample[sched_ctx_id] = starpu_timing_now();
			return 0;
		}
	}
	double forbid_sample = (now - hyp_start_sample[sched_ctx_id]) / 1000000.0;
	if(forbid_sample > 0.01)
	{
//		hyp_start_sample[sched_ctx_id] = starpu_timing_now();
		hyp_start_allow_sample[sched_ctx_id] = starpu_timing_now();
		return 1;
	}
	return 0;
}

void starpu_sched_ctx_set_policy_data(unsigned sched_ctx_id, void* policy_data)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->policy_data = policy_data;
}

void* starpu_sched_ctx_get_policy_data(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->policy_data;
}

struct starpu_worker_collection* starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, enum starpu_worker_collection_type  worker_collection_type)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->workers = (struct starpu_worker_collection*)malloc(sizeof(struct starpu_worker_collection));

	switch(worker_collection_type)
	{
	case STARPU_WORKER_LIST:
		sched_ctx->workers->has_next = worker_list.has_next;
		sched_ctx->workers->get_next = worker_list.get_next;
		sched_ctx->workers->add = worker_list.add;
		sched_ctx->workers->remove = worker_list.remove;
		sched_ctx->workers->init = worker_list.init;
		sched_ctx->workers->deinit = worker_list.deinit;
		sched_ctx->workers->init_iterator = worker_list.init_iterator;
		sched_ctx->workers->type = STARPU_WORKER_LIST;
		break;
	}

	return sched_ctx->workers;
}

unsigned starpu_sched_ctx_get_workers_list(unsigned sched_ctx_id, int **workerids)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_worker_collection *workers = sched_ctx->workers;
	*workerids = (int*)malloc(workers->nworkers*sizeof(int));
	int worker;
	unsigned nworkers = 0;
	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		(*workerids)[nworkers++] = worker;
	}
	return nworkers;
}

void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->workers->deinit(sched_ctx->workers);

	free(sched_ctx->workers);
}

struct starpu_worker_collection* starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->workers;
}

int starpu_get_workers_of_sched_ctx(unsigned sched_ctx_id, int *pus, enum starpu_worker_archtype arch)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	struct starpu_worker_collection *workers = sched_ctx->workers;
	int worker;

	int npus = 0;
	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
		if(curr_arch == arch)
			pus[npus++] = worker;
	}

	return npus;
}

starpu_pthread_rwlock_t* _starpu_sched_ctx_get_changing_ctx_mutex(unsigned sched_ctx_id)
{
	return &changing_ctx_mutex[sched_ctx_id];
}

unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if(sched_ctx != NULL)
		return sched_ctx->workers->nworkers;
	else
		return 0;

}

unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2)
{
        struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
        struct _starpu_sched_ctx *sched_ctx2 = _starpu_get_sched_ctx_struct(sched_ctx_id2);

        struct starpu_worker_collection *workers = sched_ctx->workers;
        struct starpu_worker_collection *workers2 = sched_ctx2->workers;
        int worker, worker2;
        int shared_workers = 0;

	struct starpu_sched_ctx_iterator it1, it2;
        if(workers->init_iterator)
                workers->init_iterator(workers, &it1);

        if(workers2->init_iterator)
                workers2->init_iterator(workers2, &it2);

        while(workers->has_next(workers, &it1))
        {
                worker = workers->get_next(workers, &it1);
                while(workers2->has_next(workers2, &it2))
		{
                        worker2 = workers2->get_next(workers2, &it2);
                        if(worker == worker2)
				shared_workers++;
                }
        }

	return shared_workers;
}

unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id)
{
        struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

        struct starpu_worker_collection *workers = sched_ctx->workers;
        int worker;

	struct starpu_sched_ctx_iterator it;
        if(workers->init_iterator)
                workers->init_iterator(workers, &it);


        while(workers->has_next(workers, &it))
        {
                worker = workers->get_next(workers, &it);
		if(worker == workerid)
			return 1;
        }


	return 0;
}

unsigned starpu_sched_ctx_contains_type_of_worker(enum starpu_worker_archtype arch, unsigned sched_ctx_id)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
		if(curr_arch == arch)
			return 1;
	}
	return 0;

}

unsigned _starpu_worker_belongs_to_a_sched_ctx(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	int i;
	struct _starpu_sched_ctx *sched_ctx = NULL;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		 sched_ctx = &config->sched_ctxs[i];
		 if(sched_ctx && sched_ctx->id != STARPU_NMAX_SCHED_CTXS && sched_ctx->id != sched_ctx_id)
			 if(starpu_sched_ctx_contains_worker(workerid, sched_ctx->id))
				 return 1;
	}
	return 0;
}
unsigned starpu_sched_ctx_worker_get_id(unsigned sched_ctx_id)
{
	int workerid = starpu_worker_get_id();
	if(workerid != -1)
		if(starpu_sched_ctx_contains_worker(workerid, sched_ctx_id))
			return workerid;
	return -1;
}
		 
unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return worker->nsched_ctxs > 1;
}

void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor)
{
	STARPU_ASSERT(inheritor < STARPU_NMAX_SCHED_CTXS);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->inheritor = inheritor;
	return;
}

void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_PTHREAD_MUTEX_LOCK(&finished_submit_mutex);
	sched_ctx->finished_submit = 1;
	STARPU_PTHREAD_MUTEX_UNLOCK(&finished_submit_mutex);
	return;
}

#ifdef STARPU_USE_SC_HYPERVISOR

void _starpu_sched_ctx_post_exec_task_cb(int workerid, struct starpu_task *task, size_t data_size2, uint32_t footprint)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);
	if(sched_ctx != NULL && task->sched_ctx != _starpu_get_initial_sched_ctx()->id && 
	   task->sched_ctx != STARPU_NMAX_SCHED_CTXS  && sched_ctx->perf_counters != NULL)
	{
		flops[task->sched_ctx][workerid] += task->flops;
		data_size[task->sched_ctx][workerid] += data_size2;

		if(_starpu_sched_ctx_allow_hypervisor(sched_ctx->id) || task->hypervisor_tag > 0)
		{
			_STARPU_TRACE_HYPERVISOR_BEGIN();
			sched_ctx->perf_counters->notify_post_exec_task(task, data_size[task->sched_ctx][workerid], footprint,
									task->hypervisor_tag, flops[task->sched_ctx][workerid]);
			_STARPU_TRACE_HYPERVISOR_END();
			flops[task->sched_ctx][workerid] = 0.0;
			data_size[task->sched_ctx][workerid] = 0;
		}
	}
}

void starpu_sched_ctx_call_pushed_task_cb(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	if(sched_ctx != NULL && sched_ctx_id != _starpu_get_initial_sched_ctx()->id && sched_ctx_id != STARPU_NMAX_SCHED_CTXS
	   && sched_ctx->perf_counters != NULL && _starpu_sched_ctx_allow_hypervisor(sched_ctx_id))
	{
		_STARPU_TRACE_HYPERVISOR_BEGIN();
		sched_ctx->perf_counters->notify_pushed_task(sched_ctx_id, workerid);
		_STARPU_TRACE_HYPERVISOR_END();
	}
}
#endif //STARPU_USE_SC_HYPERVISOR

int starpu_sched_get_min_priority(void)
{
	return starpu_sched_ctx_get_min_priority(_starpu_sched_ctx_get_current_context());
}

int starpu_sched_get_max_priority(void)
{
	return starpu_sched_ctx_get_max_priority(_starpu_sched_ctx_get_current_context());
}

int starpu_sched_set_min_priority(int min_prio)
{
	return starpu_sched_ctx_set_min_priority(_starpu_sched_ctx_get_current_context(), min_prio);
}

int starpu_sched_set_max_priority(int max_prio)
{
	return starpu_sched_ctx_set_max_priority(_starpu_sched_ctx_get_current_context(), max_prio);
}

int starpu_sched_ctx_get_min_priority(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->min_priority;
}

int starpu_sched_ctx_get_max_priority(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->max_priority;
}

int starpu_sched_ctx_set_min_priority(unsigned sched_ctx_id, int min_prio)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->min_priority = min_prio;
	return 0;
}

int starpu_sched_ctx_set_max_priority(unsigned sched_ctx_id, int max_prio)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->max_priority = max_prio;
	return 0;
}

int starpu_sched_ctx_min_priority_is_set(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->min_priority_is_set;
}

int starpu_sched_ctx_max_priority_is_set(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->max_priority_is_set;
}

unsigned _starpu_sched_ctx_last_worker_awake(struct _starpu_worker *worker)
{
	struct _starpu_sched_ctx_list *l = NULL;
        for (l = worker->sched_ctx_list; l; l = l->next)
        {
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(l->sched_ctx);
		
		unsigned last_worker_awake = 1;
		struct starpu_worker_collection *workers = sched_ctx->workers;
		struct starpu_sched_ctx_iterator it;

		int workerid;
		if(workers->init_iterator)
			workers->init_iterator(workers, &it);
		
		while(workers->has_next(workers, &it))
		{
			workerid = workers->get_next(workers, &it);
			if(workerid != worker->workerid && _starpu_worker_get_status(workerid) != STATUS_SLEEPING)
			{
				last_worker_awake = 0;
				break;
			}
		}
		if(last_worker_awake)
			return 1;
	}
	return 0;
}

