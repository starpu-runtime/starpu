/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2013  INRIA
 * Copyright (C) 2016  Uppsala University
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
#include <core/task.h>

starpu_pthread_rwlock_t changing_ctx_mutex[STARPU_NMAX_SCHED_CTXS];

static starpu_pthread_mutex_t sched_ctx_manag = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_mutex_t finished_submit_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static struct starpu_task stop_submission_task = STARPU_TASK_INITIALIZER;
starpu_pthread_key_t sched_ctx_key;
static unsigned with_hypervisor = 0;
static double hyp_start_sample[STARPU_NMAX_SCHED_CTXS];
static double hyp_start_allow_sample[STARPU_NMAX_SCHED_CTXS];
static double flops[STARPU_NMAX_SCHED_CTXS][STARPU_NMAXWORKERS];
static size_t data_size[STARPU_NMAX_SCHED_CTXS][STARPU_NMAXWORKERS];
static double hyp_actual_start_sample[STARPU_NMAX_SCHED_CTXS];
static double window_size;
static int nobind;
static int occupied_sms = 0;

static unsigned _starpu_get_first_free_sched_ctx(struct _starpu_machine_config *config);

static void _starpu_sched_ctx_put_new_master(unsigned sched_ctx_id);
static void _starpu_sched_ctx_put_workers_to_sleep(unsigned sched_ctx_id, unsigned all);
static void _starpu_sched_ctx_wake_up_workers(unsigned sched_ctx_id, unsigned all);
static void _starpu_sched_ctx_update_parallel_workers_with(unsigned sched_ctx_id);
static void _starpu_sched_ctx_update_parallel_workers_without(unsigned sched_ctx_id);

static void _starpu_worker_gets_into_ctx(unsigned sched_ctx_id, struct _starpu_worker *worker)
{
	unsigned ret_sched_ctx = _starpu_sched_ctx_elt_exists(worker->sched_ctx_list, sched_ctx_id);
	/* the worker was planning to go away in another ctx but finally he changed his mind &
	   he's staying */
	if (!ret_sched_ctx)
	{
		/* add context to worker */
		_starpu_sched_ctx_list_add(&worker->sched_ctx_list, sched_ctx_id);
		worker->nsched_ctxs++;
	}
	worker->removed_from_ctx[sched_ctx_id] = 0;
	if(worker->tmp_sched_ctx == (int) sched_ctx_id)
		worker->tmp_sched_ctx = -1;
	return;
}

void _starpu_worker_gets_out_of_ctx(unsigned sched_ctx_id, struct _starpu_worker *worker)
{
	unsigned ret_sched_ctx = _starpu_sched_ctx_elt_exists(worker->sched_ctx_list, sched_ctx_id);
	/* remove context from worker */
	if(ret_sched_ctx)
	{
		/* don't remove scheduling data here, there might be tasks running and when post_exec
		   executes scheduling data is not there any more, do it when deleting context, then
		   we really won't need it anymore */
		/* struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id); */
		/* if(sched_ctx && sched_ctx->sched_policy && sched_ctx->sched_policy->remove_workers) */
		/* { */
		/* 	_STARPU_TRACE_WORKER_SCHEDULING_PUSH; */
		/* 	sched_ctx->sched_policy->remove_workers(sched_ctx_id, &worker->workerid, 1); */
		/* 	_STARPU_TRACE_WORKER_SCHEDULING_POP; */
		/* } */
		if (!_starpu_sched_ctx_list_remove(&worker->sched_ctx_list, sched_ctx_id))
			worker->nsched_ctxs--;
	}
	return;
}

static void _starpu_update_workers_with_ctx(int *workerids, int nworkers, int sched_ctx_id)
{
	int i;
	struct _starpu_worker *worker = NULL;

	for(i = 0; i < nworkers; i++)
	{
		worker = _starpu_get_worker_struct(workerids[i]);

		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
		_starpu_worker_gets_into_ctx(sched_ctx_id, worker);
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	}

	return;
}

static void _starpu_update_workers_without_ctx(int *workerids, int nworkers, int sched_ctx_id, unsigned now)
{
	int i;
	struct _starpu_worker *worker = NULL;

	for(i = 0; i < nworkers; i++)
	{
		worker = _starpu_get_worker_struct(workerids[i]);
		if(now)
		{
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
			_starpu_worker_gets_out_of_ctx(sched_ctx_id, worker);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
		else
		{
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
			worker->removed_from_ctx[sched_ctx_id] = 1;
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
	}
	return;
}

void starpu_sched_ctx_stop_task_submission()
{
	_starpu_exclude_task_from_dag(&stop_submission_task);
	int ret = _starpu_task_submit_internally(&stop_submission_task);
	STARPU_ASSERT(!ret);
}

void starpu_sched_ctx_worker_shares_tasks_lists(int workerid, int sched_ctx_id)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int curr_workerid = starpu_worker_get_id();
	/* if is the initial sched_ctx no point in taking the mutex, the workers are
	   not launched yet, or if the current worker is calling this */
	if(!sched_ctx->is_initial_sched && workerid != curr_workerid)
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);

	worker->shares_tasks_lists[sched_ctx_id] = 1;

	if(!sched_ctx->is_initial_sched && workerid != curr_workerid)
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
}

static void _starpu_add_workers_to_sched_ctx(struct _starpu_sched_ctx *sched_ctx, int *workerids, int nworkers,
					     int *added_workers, int *n_added_workers)
{
	struct starpu_worker_collection *workers = sched_ctx->workers;
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	int nworkers_to_add = nworkers == -1 ? (int)config->topology.nworkers : nworkers;
	if (!nworkers_to_add)
		return;
	int workers_to_add[nworkers_to_add];

	struct starpu_perfmodel_device devices[nworkers_to_add];
	int ndevices = 0;
	struct _starpu_worker *str_worker = NULL;
	int worker;
	int i = 0;
	for(i = 0; i < nworkers_to_add; i++)
	{
		/* added_workers is NULL for the call of this func at the creation of the context*/
		/* if the function is called at the creation of the context it's no need to do this verif */
		if(added_workers)
		{
			worker = workers->add(workers, (workerids == NULL ? i : workerids[i]));
			if(worker >= 0)
				added_workers[(*n_added_workers)++] = worker;
			else
			{
				int curr_workerid = starpu_worker_get_id();
				struct _starpu_worker *worker_str = _starpu_get_worker_struct(workerids[i]);
				if(curr_workerid != workerids[i])
					STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker_str->sched_mutex);

				worker_str->removed_from_ctx[sched_ctx->id] = 0;

				if(curr_workerid != workerids[i])
					STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker_str->sched_mutex);
			}
		}
		else
		{
			worker = (workerids == NULL ? i : workerids[i]);
			workers->add(workers, worker);
			workers_to_add[i] = worker;
			str_worker = _starpu_get_worker_struct(worker);
			str_worker->tmp_sched_ctx = (int)sched_ctx->id;
		}
	}

	int *wa;
	int na;
	if(added_workers)
	{
		na = *n_added_workers;
		wa = added_workers;
	}
	else
	{
		na = nworkers_to_add;
		wa = workers_to_add;
	}

	for(i = 0; i < na; i++)
	{
		worker = wa[i];
		str_worker = _starpu_get_worker_struct(worker);
		int dev1, dev2;
		unsigned found = 0;
		for(dev1 = 0; dev1 < str_worker->perf_arch.ndevices; dev1++)
		{
			for(dev2 = 0; dev2 < ndevices; dev2++)
			{
				if(devices[dev2].type == str_worker->perf_arch.devices[dev1].type &&
				   devices[dev2].devid == str_worker->perf_arch.devices[dev1].devid)
				{
					devices[dev2].ncores += str_worker->perf_arch.devices[dev1].ncores;
					found = 1;
					break;
				}
			}
			if(!found)
			{
				devices[ndevices].type = str_worker->perf_arch.devices[dev1].type;
				devices[ndevices].devid = str_worker->perf_arch.devices[dev1].devid;
				devices[ndevices].ncores = str_worker->perf_arch.devices[dev1].ncores;
				ndevices++;
			}
			else
				found = 0;
		}
	}

	if(ndevices > 0)
	{

		if(sched_ctx->perf_arch.devices == NULL)
		{
			_STARPU_MALLOC(sched_ctx->perf_arch.devices, ndevices*sizeof(struct starpu_perfmodel_device));
		}
		else
		{
			int nfinal_devices = 0;
			int dev1, dev2;
			unsigned found = 0;
			for(dev1 = 0; dev1 < ndevices; dev1++)
			{
				for(dev2 = 0; dev2 < sched_ctx->perf_arch.ndevices; dev2++)
				{
					if(sched_ctx->perf_arch.devices[dev2].type == devices[dev1].type && sched_ctx->perf_arch.devices[dev2].devid == devices[dev1].devid)
						found = 1;
				}

				if(!found)
				{
					nfinal_devices++;
				}
				else
					found = 0;

			}


			int nsize =  (sched_ctx->perf_arch.ndevices+nfinal_devices);
			_STARPU_REALLOC(sched_ctx->perf_arch.devices, nsize*sizeof(struct starpu_perfmodel_device));

		}

		int dev1, dev2;
		unsigned found = 0;
		for(dev1 = 0; dev1 < ndevices; dev1++)
		{
			for(dev2 = 0; dev2 < sched_ctx->perf_arch.ndevices; dev2++)
			{
				if(sched_ctx->perf_arch.devices[dev2].type == devices[dev1].type && sched_ctx->perf_arch.devices[dev2].devid == devices[dev1].devid)
				{
					if(sched_ctx->perf_arch.devices[dev2].type == STARPU_CPU_WORKER)
						sched_ctx->perf_arch.devices[dev2].ncores += devices[dev1].ncores;

					found = 1;
				}
			}

			if(!found)
			{
				sched_ctx->perf_arch.devices[sched_ctx->perf_arch.ndevices].type = devices[dev1].type;
				sched_ctx->perf_arch.devices[sched_ctx->perf_arch.ndevices].devid = devices[dev1].devid;
				if (sched_ctx->stream_worker != -1)
					sched_ctx->perf_arch.devices[sched_ctx->perf_arch.ndevices].ncores = sched_ctx->nsms;
				else
					sched_ctx->perf_arch.devices[sched_ctx->perf_arch.ndevices].ncores = devices[dev1].ncores;
				sched_ctx->perf_arch.ndevices++;
			}
			else
				found = 0;

		}
	}


	_starpu_sched_ctx_update_parallel_workers_with(sched_ctx->id);

	if(sched_ctx->sched_policy && sched_ctx->sched_policy->add_workers)
	{
		_STARPU_TRACE_WORKER_SCHEDULING_PUSH;
		if(added_workers)
		{
			if(*n_added_workers > 0)
				sched_ctx->sched_policy->add_workers(sched_ctx->id, added_workers, *n_added_workers);
		}
		else
		{
			sched_ctx->sched_policy->add_workers(sched_ctx->id, workers_to_add, nworkers_to_add);
		}
		_STARPU_TRACE_WORKER_SCHEDULING_POP;
	}
	return;
}

static void _starpu_remove_workers_from_sched_ctx(struct _starpu_sched_ctx *sched_ctx, int *workerids,
						  int nworkers, int *removed_workers, int *n_removed_workers)
{
	struct starpu_worker_collection *workers = sched_ctx->workers;

	struct starpu_perfmodel_device devices[workers->nworkers];
	int ndevices = 0;

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

	unsigned found = 0;
	int dev;
	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		int worker = workers->get_next(workers, &it);
		struct _starpu_worker *str_worker = _starpu_get_worker_struct(worker);
		for(dev = 0; dev < str_worker->perf_arch.ndevices; dev++)
		{
			int dev2;
			for(dev2 = 0; dev2 < ndevices; dev2++)
			{
				if(devices[dev2].type == str_worker->perf_arch.devices[dev].type &&
				   devices[dev2].devid == str_worker->perf_arch.devices[dev].devid)
				{
					if(devices[dev2].type == STARPU_CPU_WORKER)
						devices[dev2].ncores += str_worker->perf_arch.devices[dev].ncores;
				}

					found = 1;
			}
			if(!found)
			{
				devices[ndevices].type = str_worker->perf_arch.devices[dev].type;
				devices[ndevices].devid = str_worker->perf_arch.devices[dev].devid;
				devices[ndevices].ncores = str_worker->perf_arch.devices[dev].ncores;
				ndevices++;
			}
			else
				found = 0;
		}
		found = 0;

	}
	sched_ctx->perf_arch.ndevices = ndevices;
	for(dev = 0; dev < ndevices; dev++)
	{
		sched_ctx->perf_arch.devices[dev].type = devices[dev].type;
		sched_ctx->perf_arch.devices[dev].devid = devices[dev].devid;
		sched_ctx->perf_arch.devices[dev].ncores = devices[dev].ncores;
	}

	_starpu_sched_ctx_update_parallel_workers_without(sched_ctx->id);

	return;
}

static void _starpu_sched_ctx_free_scheduling_data(struct _starpu_sched_ctx *sched_ctx)
{
	if(sched_ctx->sched_policy && sched_ctx->sched_policy->remove_workers)
	{
		int *workerids = NULL;

		unsigned nworkers_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &workerids);

		if(nworkers_ctx > 0)
		{
			_STARPU_TRACE_WORKER_SCHEDULING_PUSH;
			sched_ctx->sched_policy->remove_workers(sched_ctx->id, workerids, nworkers_ctx);
			_STARPU_TRACE_WORKER_SCHEDULING_POP;
		}

		free(workerids);
	}
	return;

}

#ifdef STARPU_HAVE_HWLOC
static void _starpu_sched_ctx_create_hwloc_tree(struct _starpu_sched_ctx *sched_ctx)
{
	sched_ctx->hwloc_workers_set = hwloc_bitmap_alloc();

	struct starpu_worker_collection *workers = sched_ctx->workers;
	struct _starpu_worker *worker;
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		unsigned workerid = workers->get_next(workers, &it);
		if(!starpu_worker_is_combined_worker(workerid))
		{
			worker = _starpu_get_worker_struct(workerid);
			hwloc_bitmap_or(sched_ctx->hwloc_workers_set,
					sched_ctx->hwloc_workers_set,
					worker->hwloc_cpu_set);
		}

	}
	return;
}
#endif

struct _starpu_sched_ctx* _starpu_create_sched_ctx(struct starpu_sched_policy *policy, int *workerids,
						   int nworkers_ctx, unsigned is_initial_sched,
						   const char *sched_ctx_name,
						   int min_prio_set, int min_prio,
						   int max_prio_set, int max_prio,
						   unsigned awake_workers,
						   void (*sched_policy_init)(unsigned),
						   void * user_data,
						   int nsub_ctxs, int *sub_ctxs, int nsms)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

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

	STARPU_PTHREAD_MUTEX_INIT(&sched_ctx->waiting_tasks_mutex, NULL);
	starpu_task_list_init(&sched_ctx->waiting_tasks);

	STARPU_PTHREAD_MUTEX_INIT(&sched_ctx->sched_ctx_list_mutex, NULL);

	if (policy)
	{
		_STARPU_MALLOC(sched_ctx->sched_policy, sizeof(struct starpu_sched_policy));
	}
	else
	{
		sched_ctx->sched_policy = NULL;
	}
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
	sched_ctx->iteration = 0;
	sched_ctx->subiteration = 0;
	sched_ctx->main_master = -1;
	sched_ctx->perf_arch.devices = NULL;
	sched_ctx->perf_arch.ndevices = 0;
	sched_ctx->init_sched = sched_policy_init;
	sched_ctx->user_data = user_data;
	sched_ctx->sms_start_idx = 0;
	sched_ctx->sms_end_idx = STARPU_NMAXSMS;
	sched_ctx->nsms = nsms;
	sched_ctx->stream_worker = -1;
	if(nsms > 0)
	{
		STARPU_ASSERT_MSG(workerids, "workerids is needed when setting nsms");
		sched_ctx->sms_start_idx = occupied_sms;
		sched_ctx->sms_end_idx = occupied_sms+nsms;
		occupied_sms += nsms;
		_STARPU_DEBUG("ctx %d: stream worker %d nsms %d ocupied sms %d\n", sched_ctx->id, workerids[0], nsms, occupied_sms);
		STARPU_ASSERT_MSG(occupied_sms <= STARPU_NMAXSMS , "STARPU:requested more sms than available");
		_starpu_worker_set_stream_ctx(workerids[0], sched_ctx);
		sched_ctx->stream_worker = workerids[0];
	}

	sched_ctx->nsub_ctxs = 0;

	int w;
	for(w = 0; w < nworkers; w++)
	{
		sem_init(&sched_ctx->fall_asleep_sem[w], 0, 0);
		sem_init(&sched_ctx->wake_up_sem[w], 0, 0);

		STARPU_PTHREAD_COND_INIT(&sched_ctx->parallel_sect_cond[w], NULL);
		STARPU_PTHREAD_MUTEX_INIT(&sched_ctx->parallel_sect_mutex[w], NULL);
		STARPU_PTHREAD_COND_INIT(&sched_ctx->parallel_sect_cond_busy[w], NULL);
		sched_ctx->busy[w] = 0;

		sched_ctx->parallel_sect[w] = 0;
		sched_ctx->sleeping[w] = 0;
	}

	sched_ctx->parallel_view = 0;

  /*init the strategy structs and the worker_collection of the ressources of the context */
	if(policy)
	{
		_starpu_init_sched_policy(config, sched_ctx, policy);
		sched_ctx->awake_workers = 1;
	}
	else
	{
		sched_ctx->awake_workers = awake_workers;
		starpu_sched_ctx_create_worker_collection(sched_ctx->id, STARPU_WORKER_LIST);
	}

	if(is_initial_sched)
	{
		int i;
		/*initialize the mutexes for all contexts */
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		  {
			STARPU_PTHREAD_RWLOCK_INIT(&changing_ctx_mutex[i], NULL);
		  }
	}

        /*add sub_ctxs before add workers, in order to be able to associate them if necessary */
	if(nsub_ctxs != 0)
	{
		int i;
		for(i = 0; i < nsub_ctxs; i++)
			sched_ctx->sub_ctxs[i] = sub_ctxs[i];
		sched_ctx->nsub_ctxs = nsub_ctxs;
	}
	
	/* after having an worker_collection on the ressources add them */
	_starpu_add_workers_to_sched_ctx(sched_ctx, workerids, nworkers_ctx, NULL, NULL);

#ifdef STARPU_HAVE_HWLOC
	/* build hwloc tree of the context */
	_starpu_sched_ctx_create_hwloc_tree(sched_ctx);
#endif //STARPU_HAVE_HWLOC

	/* if we create the initial big sched ctx we can update workers' status here
	   because they haven't been launched yet */
	if(is_initial_sched)
	{
		int i;
		for(i = 0; i < nworkers; i++)
		{
			struct _starpu_worker *worker = _starpu_get_worker_struct(i);
			if(!_starpu_sched_ctx_list_add(&worker->sched_ctx_list, sched_ctx->id))
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

	struct _starpu_machine_config *config = _starpu_get_machine_config();
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
					_npus = _starpu_get_workers_of_sched_ctx(config->sched_ctxs[s].id, _pus, arch);
					int ctx_min = arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;
					if(_npus > ctx_min)
					{
						int n=0;
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

					_npus = _starpu_get_workers_of_sched_ctx(config->sched_ctxs[s].id, _pus, arch);
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
	struct _starpu_machine_config *config = _starpu_get_machine_config();
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
	sched_ctx = _starpu_create_sched_ctx(selected_policy, workers, nw, 0, sched_ctx_name, 0, 0, 0, 0, 1, NULL, NULL,0, NULL, 0);
	sched_ctx->min_ncpus = min_ncpus;
	sched_ctx->max_ncpus = max_ncpus;
	sched_ctx->min_ngpus = min_ngpus;
	sched_ctx->max_ngpus = max_ngpus;
	_starpu_unlock_mutex_if_prev_locked();
	int *added_workerids;
	unsigned nw_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &added_workerids);
	_starpu_update_workers_without_ctx(added_workerids, nw_ctx, sched_ctx->id, 0);
	free(added_workerids);
	_starpu_relock_mutex_if_prev_locked();
#ifdef STARPU_USE_SC_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return sched_ctx->id;

}

int starpu_sched_ctx_get_nsms(unsigned sched_ctx)
{
	struct _starpu_sched_ctx *sc = _starpu_get_sched_ctx_struct(sched_ctx);
	return sc->nsms;
}

void starpu_sched_ctx_get_sms_interval(int stream_workerid, int *start, int *end)
{
	struct _starpu_sched_ctx *sc = _starpu_worker_get_ctx_stream(stream_workerid);
	*start = sc->sms_start_idx;
	*end = sc->sms_end_idx;
}

int starpu_sched_ctx_get_sub_ctxs(unsigned sched_ctx, int *ctxs)
{
	struct _starpu_sched_ctx *sc = _starpu_get_sched_ctx_struct(sched_ctx);
	int i;
	for(i = 0; i < sc->nsub_ctxs; i++)
		    ctxs[i] = sc->sub_ctxs[i];
	return sc->nsub_ctxs;
}

int starpu_sched_ctx_get_stream_worker(unsigned sub_ctx)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sub_ctx);
	struct starpu_worker_collection *workers = sched_ctx->workers;

	struct starpu_sched_ctx_iterator it;
	int worker = -1;
	
	workers->init_iterator(workers, &it);
	if(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
	}

	return worker;
}

unsigned starpu_sched_ctx_create(int *workerids, int nworkers, const char *sched_ctx_name, ...)
{
	va_list varg_list;
	int arg_type;
	int min_prio_set = 0;
	int max_prio_set = 0;
	int min_prio = 0;
	int max_prio = 0;
	int nsms = 0;
        int *sub_ctxs = NULL;
        int nsub_ctxs = 0;
	void *user_data = NULL;
	struct starpu_sched_policy *sched_policy = NULL;
	unsigned hierarchy_level = 0;
	unsigned nesting_sched_ctx = STARPU_NMAX_SCHED_CTXS;
	unsigned awake_workers = 0;
	void (*init_sched)(unsigned) = NULL;

	va_start(varg_list, sched_ctx_name);
	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type == STARPU_SCHED_CTX_POLICY_NAME)
		{
			char *policy_name = va_arg(varg_list, char *);
			struct _starpu_machine_config *config = _starpu_get_machine_config();
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
		else if (arg_type == STARPU_SCHED_CTX_HIERARCHY_LEVEL)
		{
			hierarchy_level = va_arg(varg_list, unsigned);
		}
		else if (arg_type == STARPU_SCHED_CTX_NESTED)
		{
			nesting_sched_ctx = va_arg(varg_list, unsigned);
		}
		else if (arg_type == STARPU_SCHED_CTX_AWAKE_WORKERS)
		{
			awake_workers = 1;
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_INIT)
		{
			init_sched = va_arg(varg_list, void(*)(unsigned));
		}
		else if (arg_type == STARPU_SCHED_CTX_USER_DATA)
		{
			user_data = va_arg(varg_list, void *);
		}
		else if (arg_type == STARPU_SCHED_CTX_SUB_CTXS)
		{
			sub_ctxs = va_arg(varg_list, int*);
			nsub_ctxs = va_arg(varg_list, int);
		}
		else if (arg_type == STARPU_SCHED_CTX_CUDA_NSMS)
		{
			nsms = va_arg(varg_list, int);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}

	}
	va_end(varg_list);

	if (workerids && nworkers != -1)
	{
		/* Make sure the user doesn't use invalid worker IDs. */
		int num_workers = starpu_worker_get_count();
		int i;
		for (i = 0; i < nworkers; i++)
		{
			if (workerids[i] < 0 || workerids[i] >= num_workers)
			{
				_STARPU_ERROR("Invalid worker ID (%d) specified!\n", workerids[i]);
				return STARPU_NMAX_SCHED_CTXS;
			}
		}
	}

	struct _starpu_sched_ctx *sched_ctx = NULL;
	sched_ctx = _starpu_create_sched_ctx(sched_policy, workerids, nworkers, 0, sched_ctx_name, min_prio_set, min_prio, max_prio_set, max_prio, awake_workers, init_sched, user_data, nsub_ctxs, sub_ctxs, nsms);
	sched_ctx->hierarchy_level = hierarchy_level;
	sched_ctx->nesting_sched_ctx = nesting_sched_ctx;

	_starpu_unlock_mutex_if_prev_locked();
	int *added_workerids;
	unsigned nw_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &added_workerids);
	_starpu_update_workers_with_ctx(added_workerids, nw_ctx, sched_ctx->id);
	free(added_workerids);
	_starpu_relock_mutex_if_prev_locked();
#ifdef STARPU_USE_SC_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return sched_ctx->id;
}

int fstarpu_sched_ctx_create(int *workerids, int nworkers, const char *sched_ctx_name, void ***_arglist)
{
	void **arglist = *_arglist;
	int arg_i = 0;
	int min_prio_set = 0;
	int max_prio_set = 0;
	int min_prio = 0;
	int max_prio = 0;
	int nsms = 0;
        int *sub_ctxs = NULL;
        int nsub_ctxs = 0;
	void *user_data = NULL;
	struct starpu_sched_policy *sched_policy = NULL;
	unsigned hierarchy_level = 0;
	unsigned nesting_sched_ctx = STARPU_NMAX_SCHED_CTXS;
	unsigned awake_workers = 0;
	void (*init_sched)(unsigned) = NULL;

	while (arglist[arg_i] != NULL)
	{
		const int arg_type = (int)(intptr_t)arglist[arg_i];
		if (arg_type == STARPU_SCHED_CTX_POLICY_NAME)
		{
			arg_i++;
			char *policy_name = arglist[arg_i];
			struct _starpu_machine_config *config = _starpu_get_machine_config();
			sched_policy = _starpu_select_sched_policy(config, policy_name);
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_STRUCT)
		{
			arg_i++;
			sched_policy = arglist[arg_i];
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_MIN_PRIO)
		{
			arg_i++;
			min_prio = *(int *)arglist[arg_i];
			min_prio_set = 1;
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_MAX_PRIO)
		{
			arg_i++;
			max_prio = *(int *)arglist[arg_i];
			max_prio_set = 1;
		}
		else if (arg_type == STARPU_SCHED_CTX_HIERARCHY_LEVEL)
		{
			arg_i++;
			int val = *(int *)arglist[arg_i];
			STARPU_ASSERT(val >= 0);
			hierarchy_level = (unsigned)val;
		}
		else if (arg_type == STARPU_SCHED_CTX_NESTED)
		{
			arg_i++;
			int val = *(int *)arglist[arg_i];
			STARPU_ASSERT(val >= 0);
			nesting_sched_ctx = (unsigned)val;
		}
		else if (arg_type == STARPU_SCHED_CTX_AWAKE_WORKERS)
		{
			awake_workers = 1;
		}
		else if (arg_type == STARPU_SCHED_CTX_POLICY_INIT)
		{
			arg_i++;
			init_sched = arglist[arg_i];
		}
		else if (arg_type == STARPU_SCHED_CTX_USER_DATA)
		{
			arg_i++;
			user_data = arglist[arg_i];
		}
		else if (arg_type == STARPU_SCHED_CTX_SUB_CTXS)
		{
			arg_i++;
			sub_ctxs = (int*)arglist[arg_i]; 
			arg_i++;
			nsub_ctxs = *(int*)arglist[arg_i]; 
		}
		else if (arg_type == STARPU_SCHED_CTX_CUDA_NSMS)
		{
			arg_i++;
			nsms = *(int*)arglist[arg_i]; 
		}

		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
		arg_i++;
	}

	if (workerids && nworkers != -1)
	{
		/* Make sure the user doesn't use invalid worker IDs. */
		int num_workers = starpu_worker_get_count();
		int i;
		for (i = 0; i < nworkers; i++)
		{
			if (workerids[i] < 0 || workerids[i] >= num_workers)
			{
				_STARPU_ERROR("Invalid worker ID (%d) specified!\n", workerids[i]);
				return STARPU_NMAX_SCHED_CTXS;
			}
		}
	}

	struct _starpu_sched_ctx *sched_ctx = NULL;
	sched_ctx = _starpu_create_sched_ctx(sched_policy, workerids, nworkers, 0, sched_ctx_name, min_prio_set, min_prio, max_prio_set, max_prio, awake_workers, init_sched, user_data, nsub_ctxs, sub_ctxs, nsms);
	sched_ctx->hierarchy_level = hierarchy_level;
	sched_ctx->nesting_sched_ctx = nesting_sched_ctx;

	_starpu_unlock_mutex_if_prev_locked();
	int *added_workerids;
	unsigned nw_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &added_workerids);
	_starpu_update_workers_with_ctx(added_workerids, nw_ctx, sched_ctx->id);
	free(added_workerids);
	_starpu_relock_mutex_if_prev_locked();
#ifdef STARPU_USE_SC_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return (int)sched_ctx->id;
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
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	int nworkers = config->topology.nworkers;
	int w;
	for(w = 0; w < nworkers; w++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->parallel_sect_mutex[w]);
		while (sched_ctx->busy[w]) {
			STARPU_PTHREAD_COND_WAIT(&sched_ctx->parallel_sect_cond_busy[w], &sched_ctx->parallel_sect_mutex[w]);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->parallel_sect_mutex[w]);
	}
	if(sched_ctx->sched_policy)
	{
		_starpu_deinit_sched_policy(sched_ctx);
		free(sched_ctx->sched_policy);
		sched_ctx->sched_policy = NULL;
	}
	else
	{
		starpu_sched_ctx_delete_worker_collection(sched_ctx->id);
	}

	if (sched_ctx->perf_arch.devices)
	{
		free(sched_ctx->perf_arch.devices);
		sched_ctx->perf_arch.devices = NULL;
	}

	STARPU_PTHREAD_MUTEX_DESTROY(&sched_ctx->empty_ctx_mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&sched_ctx->waiting_tasks_mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&sched_ctx->sched_ctx_list_mutex);
	sched_ctx->id = STARPU_NMAX_SCHED_CTXS;
#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_free(sched_ctx->hwloc_workers_set);
#endif //STARPU_HAVE_HWLOC

	STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	config->topology.nsched_ctxs--;
	STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);
}

void starpu_sched_ctx_delete(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_ASSERT(sched_ctx);

#ifdef STARPU_USE_SC_HYPERVISOR
	if (sched_ctx_id != 0 && sched_ctx_id != STARPU_NMAX_SCHED_CTXS && sched_ctx->perf_counters != NULL)
	{
		_STARPU_TRACE_HYPERVISOR_BEGIN();
		sched_ctx->perf_counters->notify_delete_context(sched_ctx_id);
		_STARPU_TRACE_HYPERVISOR_END();
	}
#endif //STARPU_USE_SC_HYPERVISOR

	unsigned inheritor_sched_ctx_id = sched_ctx->inheritor;
	struct _starpu_sched_ctx *inheritor_sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx->inheritor);

	_starpu_unlock_mutex_if_prev_locked();
	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx_id]);
	STARPU_ASSERT(sched_ctx->id != STARPU_NMAX_SCHED_CTXS);

	int *workerids;
	unsigned nworkers_ctx = starpu_sched_ctx_get_workers_list(sched_ctx->id, &workerids);

	/*if both of them have all the ressources is pointless*/
	/*trying to transfer ressources from one ctx to the other*/
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned nworkers = config->topology.nworkers;

	if(nworkers_ctx > 0 && inheritor_sched_ctx && inheritor_sched_ctx->id != STARPU_NMAX_SCHED_CTXS &&
	   !(nworkers_ctx == nworkers && nworkers_ctx == inheritor_sched_ctx->workers->nworkers))
	{
		starpu_sched_ctx_add_workers(workerids, nworkers_ctx, inheritor_sched_ctx_id);
		starpu_sched_ctx_set_priority(workerids, nworkers_ctx, inheritor_sched_ctx_id, 1);
		starpu_sched_ctx_set_priority_on_level(workerids, nworkers_ctx, inheritor_sched_ctx_id, 1);
	}

	if(!_starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx_id))
	{
		if(!sched_ctx->sched_policy)
			_starpu_sched_ctx_wake_up_workers(sched_ctx_id, 0);
		/*if btw the mutex release & the mutex lock the context has changed take care to free all
		  scheduling data before deleting the context */
		_starpu_update_workers_without_ctx(workerids, nworkers_ctx, sched_ctx_id, 1);
		_starpu_sched_ctx_free_scheduling_data(sched_ctx);
		_starpu_delete_sched_ctx(sched_ctx);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);
	/* workerids is malloc-ed in starpu_sched_ctx_get_workers_list, don't forget to free it when
	   you don't use it anymore */
	free(workerids);
	_starpu_relock_mutex_if_prev_locked();
	occupied_sms -= sched_ctx->nsms;
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
	struct _starpu_machine_config *config = _starpu_get_machine_config();
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
unsigned _starpu_can_push_task(struct _starpu_sched_ctx *sched_ctx, struct starpu_task *task)
{
	if(sched_ctx->sched_policy && sched_ctx->sched_policy->simulate_push_task)
	{
		if (window_size == 0.0) return 1;

		STARPU_PTHREAD_RWLOCK_RDLOCK(&changing_ctx_mutex[sched_ctx->id]);
		double expected_end = sched_ctx->sched_policy->simulate_push_task(task);
		STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx->id]);

		double expected_len = 0.0;
		if(hyp_actual_start_sample[sched_ctx->id] != 0.0)
			expected_len = expected_end - hyp_actual_start_sample[sched_ctx->id] ;
		else
		{
			printf("%d: sc start is 0.0\n", sched_ctx->id);
			expected_len = expected_end - starpu_timing_now();
		}
		if(expected_len < 0.0)
			printf("exp len negative %lf \n", expected_len);
		expected_len /= 1000000.0;
//		printf("exp_end %lf start %lf expected_len %lf \n", expected_end, hyp_actual_start_sample[sched_ctx->id], expected_len);
		if(expected_len > (window_size + 0.2*window_size))
			return 0;
	}
	return 1;
}

void _starpu_fetch_task_from_waiting_list(struct _starpu_sched_ctx *sched_ctx)
{
	if(starpu_task_list_empty(&sched_ctx->waiting_tasks))
		return;
	struct starpu_task *old_task = starpu_task_list_back(&sched_ctx->waiting_tasks);
	if(_starpu_can_push_task(sched_ctx, old_task))
	{
		old_task = starpu_task_list_pop_back(&sched_ctx->waiting_tasks);
		_starpu_push_task_to_workers(old_task);
	}
	return;
}

void _starpu_push_task_to_waiting_list(struct _starpu_sched_ctx *sched_ctx, struct starpu_task *task)
{
	starpu_task_list_push_front(&sched_ctx->waiting_tasks, task);
	return;
}

void starpu_sched_ctx_set_priority_on_level(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx, unsigned priority)
{
	(void) workers_to_add;
	(void) nworkers_to_add;
	(void) sched_ctx;
	(void) priority;
/* 	int w; */
/* 	struct _starpu_worker *worker = NULL; */
/* 	for(w = 0; w < nworkers_to_add; w++) */
/* 	{ */
/* 		worker = _starpu_get_worker_struct(workers_to_add[w]); */
/* 		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex); */
/* 		struct _starpu_sched_ctx_list *l = NULL; */
/* 		for (l = worker->sched_ctx_list; l; l = l->next) */
/* 		{ */
/* 			if(l->sched_ctx != STARPU_NMAX_SCHED_CTXS && l->sched_ctx != sched_ctx && */
/* 			   starpu_sched_ctx_get_hierarchy_level(l->sched_ctx) == starpu_sched_ctx_get_hierarchy_level(sched_ctx)) */
/* 			{ */
/* 				/\* the lock is taken inside the func *\/ */
/* 				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex); */
/* 				starpu_sched_ctx_set_priority(&workers_to_add[w], 1, l->sched_ctx, priority); */
/* 				STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex); */
/* 			} */
/* 		} */
/* 		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex); */
/* 	} */
/* 	return; */

}
static void _set_priority_hierarchically(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx, unsigned priority)
{
	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
	{
		unsigned father = starpu_sched_ctx_get_inheritor(sched_ctx);
		starpu_sched_ctx_set_priority(workers_to_add, nworkers_to_add, father, priority);
		starpu_sched_ctx_set_priority_on_level(workers_to_add, nworkers_to_add, father, priority);
		_set_priority_hierarchically(workers_to_add, nworkers_to_add, father, priority);
	}
	return;
}

void starpu_sched_ctx_add_workers(int *workers_to_add, int nworkers_to_add, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	_starpu_unlock_mutex_if_prev_locked();

	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx_id]);

	STARPU_ASSERT(workers_to_add != NULL && nworkers_to_add > 0);
	_starpu_check_workers(workers_to_add, nworkers_to_add);

	/* if the context has not already been deleted */
	if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
	{
		int added_workers[nworkers_to_add];
		int n_added_workers = 0;

		_starpu_add_workers_to_sched_ctx(sched_ctx, workers_to_add, nworkers_to_add, added_workers, &n_added_workers);

		if(n_added_workers > 0)
		{
			_starpu_update_workers_with_ctx(added_workers, n_added_workers, sched_ctx->id);
		}
		starpu_sched_ctx_set_priority(workers_to_add, nworkers_to_add, sched_ctx_id, 1);
		_set_priority_hierarchically(workers_to_add, nworkers_to_add, sched_ctx_id, 0);

	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);

	_starpu_relock_mutex_if_prev_locked();

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

	_starpu_check_workers(workers_to_remove, nworkers_to_remove);

	_starpu_unlock_mutex_if_prev_locked();

	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx_id]);
	/* if the context has not already been deleted */
	if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
	{
		int removed_workers[sched_ctx->workers->nworkers];
		int n_removed_workers = 0;

		_starpu_remove_workers_from_sched_ctx(sched_ctx, workers_to_remove, nworkers_to_remove, removed_workers, &n_removed_workers);

		if(n_removed_workers > 0)
		{
			_starpu_update_workers_without_ctx(removed_workers, n_removed_workers, sched_ctx_id, 0);
			starpu_sched_ctx_set_priority(removed_workers, n_removed_workers, sched_ctx_id, 1);
		}

	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx_id]);

	_starpu_relock_mutex_if_prev_locked();

	return;
}

int _starpu_nworkers_able_to_execute_task(struct starpu_task *task, struct _starpu_sched_ctx *sched_ctx)
{
	unsigned nworkers = 0;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&changing_ctx_mutex[sched_ctx->id]);
	struct starpu_worker_collection *workers = sched_ctx->workers;

	struct starpu_sched_ctx_iterator it;

	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		STARPU_ASSERT_MSG(worker < STARPU_NMAXWORKERS, "worker id %d", worker);
		if (starpu_worker_can_execute_task_first_impl(worker, task, NULL))
			nworkers++;
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&changing_ctx_mutex[sched_ctx->id]);

	return nworkers;
}

/* unused sched_ctx have the id STARPU_NMAX_SCHED_CTXS */
void _starpu_init_all_sched_ctxs(struct _starpu_machine_config *config)
{
	STARPU_PTHREAD_KEY_CREATE(&sched_ctx_key, NULL);
	window_size = starpu_get_env_float_default("STARPU_WINDOW_TIME_SIZE", 0.0);
	nobind = starpu_get_env_number("STARPU_WORKERS_NOBIND");

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

	_starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->tasks_barrier);
	return 0;
}

int _starpu_wait_for_n_submitted_tasks_of_sched_ctx(unsigned sched_ctx_id, unsigned n)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait_for_n_submitted_tasks must not be called from a task or callback");

	return _starpu_barrier_counter_wait_until_counter_reaches_down_to_n(&sched_ctx->tasks_barrier, n);
}

void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
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

unsigned _starpu_increment_nready_tasks_of_sched_ctx(unsigned sched_ctx_id, double ready_flops, struct starpu_task *task)
{
	unsigned ret = 1;
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	if(!sched_ctx->is_initial_sched)
		STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->waiting_tasks_mutex);

	_starpu_barrier_counter_increment(&sched_ctx->ready_tasks_barrier, ready_flops);


	if(!sched_ctx->is_initial_sched)
	{
		if(!_starpu_can_push_task(sched_ctx, task))
		{
			_starpu_push_task_to_waiting_list(sched_ctx, task);
			ret = 0;
		}

		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->waiting_tasks_mutex);
	}
	return ret;
}

void _starpu_decrement_nready_tasks_of_sched_ctx(unsigned sched_ctx_id, double ready_flops)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	if(!sched_ctx->is_initial_sched)
		STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->waiting_tasks_mutex);

	_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->ready_tasks_barrier, ready_flops);


	if(!sched_ctx->is_initial_sched)
	{
		_starpu_fetch_task_from_waiting_list(sched_ctx);
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->waiting_tasks_mutex);
	}

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
	_starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->ready_tasks_barrier);
	return 0;
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
		hyp_actual_start_sample[i] = 0.0;
	}
}

unsigned starpu_sched_ctx_check_if_hypervisor_exists()
{
	return with_hypervisor;
}

void starpu_sched_ctx_update_start_resizing_sample(unsigned sched_ctx_id, double start_sample)
{
	hyp_actual_start_sample[sched_ctx_id] = start_sample;
}

unsigned _starpu_sched_ctx_allow_hypervisor(unsigned sched_ctx_id)
{
	(void) sched_ctx_id;
	return 1;
#if 0
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
#endif
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

struct starpu_sched_policy *starpu_sched_ctx_get_sched_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->sched_policy;
}

struct starpu_worker_collection* starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, enum starpu_worker_collection_type  worker_collection_type)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_STARPU_MALLOC(sched_ctx->workers, sizeof(struct starpu_worker_collection));

	switch(worker_collection_type)
	{
#ifdef STARPU_HAVE_HWLOC
	case STARPU_WORKER_TREE:
		sched_ctx->workers->has_next = worker_tree.has_next;
		sched_ctx->workers->get_next = worker_tree.get_next;
		sched_ctx->workers->add = worker_tree.add;
		sched_ctx->workers->remove = worker_tree.remove;
		sched_ctx->workers->init = worker_tree.init;
		sched_ctx->workers->deinit = worker_tree.deinit;
		sched_ctx->workers->init_iterator = worker_tree.init_iterator;
		sched_ctx->workers->init_iterator_for_parallel_tasks = worker_tree.init_iterator_for_parallel_tasks;
		sched_ctx->workers->type = STARPU_WORKER_TREE;
		break;
#endif
//	case STARPU_WORKER_LIST:
	default:
		sched_ctx->workers->has_next = worker_list.has_next;
		sched_ctx->workers->get_next = worker_list.get_next;
		sched_ctx->workers->add = worker_list.add;
		sched_ctx->workers->remove = worker_list.remove;
		sched_ctx->workers->init = worker_list.init;
		sched_ctx->workers->deinit = worker_list.deinit;
		sched_ctx->workers->init_iterator = worker_list.init_iterator;
		sched_ctx->workers->init_iterator_for_parallel_tasks = worker_list.init_iterator_for_parallel_tasks;
		sched_ctx->workers->type = STARPU_WORKER_LIST;
		break;

	}

        /* construct the collection of workers(list/tree/etc.) */
	sched_ctx->workers->init(sched_ctx->workers);

	return sched_ctx->workers;
}

void starpu_sched_ctx_display_workers(unsigned sched_ctx_id, FILE *f)
{
	int *workerids = NULL;
	unsigned nworkers;
	unsigned i;

	nworkers = starpu_sched_ctx_get_workers_list(sched_ctx_id, &workerids);
	fprintf(f, "[sched_ctx %u]: %u worker%s\n", sched_ctx_id, nworkers, nworkers>1?"s":"");
	for (i = 0; i < nworkers; i++)
	{
		char name[256];
		starpu_worker_get_name(workerids[i], name, 256);
		fprintf(f, "\t\t%s\n", name);
	}
	free(workerids);
}

unsigned starpu_sched_ctx_get_workers_list_raw(unsigned sched_ctx_id, int **workerids)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	*workerids = sched_ctx->workers->workerids;
	return sched_ctx->workers->nworkers;
}

unsigned starpu_sched_ctx_get_workers_list(unsigned sched_ctx_id, int **workerids)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_worker_collection *workers = sched_ctx->workers;
	unsigned nworkers = 0;
	struct starpu_sched_ctx_iterator it;

	if(!workers) return 0;
	_STARPU_MALLOC(*workerids, workers->nworkers*sizeof(int));

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		int worker = workers->get_next(workers, &it);
		(*workerids)[nworkers++] = worker;
	}
	return nworkers;
}

void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->workers->deinit(sched_ctx->workers);

	free(sched_ctx->workers);
	sched_ctx->workers = NULL;
}

struct starpu_worker_collection* starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->workers;
}

int _starpu_get_workers_of_sched_ctx(unsigned sched_ctx_id, int *pus, enum starpu_worker_archtype arch)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	struct starpu_worker_collection *workers = sched_ctx->workers;

	int npus = 0;
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		int worker = workers->get_next(workers, &it);
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
		if(curr_arch == arch || arch == STARPU_ANY_WORKER)
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
        int shared_workers = 0;
	struct starpu_sched_ctx_iterator it1, it2;

	workers->init_iterator(workers, &it1);
	workers2->init_iterator(workers2, &it2);
        while(workers->has_next(workers, &it1))
        {
                int worker = workers->get_next(workers, &it1);
                while(workers2->has_next(workers2, &it2))
		{
                        int worker2 = workers2->get_next(workers2, &it2);
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
	if(workers)
	{
		unsigned i;

		for (i = 0; i < workers->nworkers; i++)
			if (workerid == workers->workerids[i])
				return 1;
	}
	return 0;
}

unsigned starpu_sched_ctx_contains_type_of_worker(enum starpu_worker_archtype arch, unsigned sched_ctx_id)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	unsigned i;

	for (i = 0; i < workers->nworkers; i++)
	{
		int worker = workers->workerids[i];
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
		if(curr_arch == arch)
			return 1;
	}
	return 0;

}

unsigned _starpu_worker_belongs_to_a_sched_ctx(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		struct _starpu_sched_ctx *sched_ctx = &config->sched_ctxs[i];
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

unsigned starpu_sched_ctx_get_ctx_for_task(struct starpu_task *task)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);
	unsigned ret_sched_ctx = task->sched_ctx;
	if (task->possibly_parallel && !sched_ctx->sched_policy
	    && sched_ctx->nesting_sched_ctx != STARPU_NMAX_SCHED_CTXS)
		 ret_sched_ctx = sched_ctx->nesting_sched_ctx;
	return ret_sched_ctx;
}

unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return worker->nsched_ctxs > 1;
}

void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(inheritor < STARPU_NMAX_SCHED_CTXS);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->inheritor = inheritor;
	return;
}

unsigned starpu_sched_ctx_get_inheritor(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	return	sched_ctx->inheritor;
}

unsigned starpu_sched_ctx_get_hierarchy_level(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	return	sched_ctx->hierarchy_level;
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
	if (workerid < 0)
		return;
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

void starpu_sched_ctx_set_priority(int *workers, int nworkers, unsigned sched_ctx_id, unsigned priority)
{
	if(nworkers != -1)
	{
		int w;
		struct _starpu_worker *worker = NULL;
		for(w = 0; w < nworkers; w++)
		{
			worker = _starpu_get_worker_struct(workers[w]);
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
			_starpu_sched_ctx_list_move(&worker->sched_ctx_list, sched_ctx_id, priority);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
	}
	return;
}

unsigned starpu_sched_ctx_get_priority(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return _starpu_sched_ctx_elt_get_priority(worker->sched_ctx_list, sched_ctx_id);
}

unsigned _starpu_sched_ctx_last_worker_awake(struct _starpu_worker *worker)
{
	struct _starpu_sched_ctx_list_iterator list_it;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		struct _starpu_sched_ctx_elt *e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);

		unsigned last_worker_awake = 1;
		struct starpu_worker_collection *workers = sched_ctx->workers;
		struct starpu_sched_ctx_iterator it;

		workers->init_iterator(workers, &it);
		while(workers->has_next(workers, &it))
		{
			int workerid = workers->get_next(workers, &it);
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

void starpu_sched_ctx_bind_current_thread_to_cpuid(unsigned cpuid)
{
	_starpu_bind_thread_on_cpu(_starpu_get_machine_config(), cpuid, STARPU_NOWORKERID);
}

unsigned starpu_sched_ctx_worker_is_master_for_child_ctx(int workerid, unsigned sched_ctx_id)
{
	if (_starpu_get_nsched_ctxs() <= 1)
		return STARPU_NMAX_SCHED_CTXS;

	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct _starpu_sched_ctx_list_iterator list_it;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		struct _starpu_sched_ctx_elt *e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);
		if(sched_ctx-> main_master == workerid && sched_ctx->nesting_sched_ctx == sched_ctx_id)
			return sched_ctx->id;
	}
	return STARPU_NMAX_SCHED_CTXS;
}

unsigned starpu_sched_ctx_master_get_context(int masterid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(masterid);
	struct _starpu_sched_ctx_list_iterator list_it;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		struct _starpu_sched_ctx_elt *e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);
		if(sched_ctx->main_master == masterid)
			return sched_ctx->id;
	}
	return STARPU_NMAX_SCHED_CTXS;
}

struct _starpu_sched_ctx *__starpu_sched_ctx_get_sched_ctx_for_worker_and_job(struct _starpu_worker *worker, struct _starpu_job *j)
{
	struct _starpu_sched_ctx_list_iterator list_it;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		struct _starpu_sched_ctx_elt *e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);
		if (j->task->sched_ctx == sched_ctx->id)
			return sched_ctx;
	}
	return NULL;
}

void starpu_sched_ctx_revert_task_counters(unsigned sched_ctx_id, double ready_flops)
{
        _starpu_decrement_nsubmitted_tasks_of_sched_ctx(sched_ctx_id);
        _starpu_decrement_nready_tasks_of_sched_ctx(sched_ctx_id, ready_flops);
}

void starpu_sched_ctx_move_task_to_ctx(struct starpu_task *task, unsigned sched_ctx, unsigned manage_mutex, 
				       unsigned with_repush)
{
	/* TODO: make something cleaner which differentiates between calls
	   from push or pop (have mutex or not) and from another worker or not */
	int workerid = starpu_worker_get_id();
	struct _starpu_worker *worker  = NULL;
	if(workerid != -1 && manage_mutex)
	{
		worker = _starpu_get_worker_struct(workerid);
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	}


	task->sched_ctx = sched_ctx;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	_starpu_increment_nsubmitted_tasks_of_sched_ctx(j->task->sched_ctx);

	if(with_repush)
		_starpu_repush_task(j);
	else
		_starpu_increment_nready_tasks_of_sched_ctx(j->task->sched_ctx, j->task->flops, j->task);

	if(workerid != -1 && manage_mutex)
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
}

void starpu_sched_ctx_list_task_counters_increment(unsigned sched_ctx_id, int workerid)
{
	/* Note : often we don't have any sched_mutex taken here but we
	    should, so take it */
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	if (worker->nsched_ctxs > 1)
	{
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
		_starpu_sched_ctx_list_push_event(worker->sched_ctx_list, sched_ctx_id);
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	}
}

void starpu_sched_ctx_list_task_counters_decrement(unsigned sched_ctx_id, int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	if (worker->nsched_ctxs > 1)
		_starpu_sched_ctx_list_pop_event(worker->sched_ctx_list, sched_ctx_id);
}

void starpu_sched_ctx_list_task_counters_reset(unsigned sched_ctx_id, int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	if (worker->nsched_ctxs > 1)
		_starpu_sched_ctx_list_pop_all_event(worker->sched_ctx_list, sched_ctx_id);
}

void starpu_sched_ctx_list_task_counters_increment_all(struct starpu_task *task, unsigned sched_ctx_id)
{
	/* Note that with 1 ctx we will default to the global context,
	   hence our counters are useless */
	if (_starpu_get_nsched_ctxs() > 1)
	{
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
		struct starpu_sched_ctx_iterator it;

		workers->init_iterator_for_parallel_tasks(workers, &it, task);
		STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->sched_ctx_list_mutex);
		while(workers->has_next(workers, &it))
		{
			int worker = workers->get_next(workers, &it);
			starpu_sched_ctx_list_task_counters_increment(sched_ctx_id, worker);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->sched_ctx_list_mutex);
	}
}

void starpu_sched_ctx_list_task_counters_decrement_all(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (_starpu_get_nsched_ctxs() > 1)
	{
		int curr_workerid = starpu_worker_get_id();
		struct _starpu_worker *curr_worker_str = NULL, *worker_str;
		if(curr_workerid != -1)
		{
			curr_worker_str = _starpu_get_worker_struct(curr_workerid);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&curr_worker_str->sched_mutex);
		}

		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
		struct starpu_sched_ctx_iterator it;
		workers->init_iterator_for_parallel_tasks(workers, &it, task);
		STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->sched_ctx_list_mutex);
		while(workers->has_next(workers, &it))
		{
			int worker = workers->get_next(workers, &it);

			worker_str = _starpu_get_worker_struct(worker);
			if (worker_str->nsched_ctxs > 1)
			{
				STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker_str->sched_mutex);
				starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, worker);
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker_str->sched_mutex);
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->sched_ctx_list_mutex);

		if(curr_workerid != -1)
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&curr_worker_str->sched_mutex);
	}
}

void starpu_sched_ctx_list_task_counters_reset_all(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (_starpu_get_nsched_ctxs() > 1)
	{
		int curr_workerid = starpu_worker_get_id();
		struct _starpu_worker *curr_worker_str = NULL, *worker_str;
		if(curr_workerid != -1)
		{
			curr_worker_str = _starpu_get_worker_struct(curr_workerid);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&curr_worker_str->sched_mutex);
		}

		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
		struct starpu_sched_ctx_iterator it;
		workers->init_iterator_for_parallel_tasks(workers, &it, task);
		STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->sched_ctx_list_mutex);
		while(workers->has_next(workers, &it))
		{
			int worker = workers->get_next(workers, &it);
			worker_str = _starpu_get_worker_struct(worker);
			if (worker_str->nsched_ctxs > 1)
			{
				STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker_str->sched_mutex);
				starpu_sched_ctx_list_task_counters_reset(sched_ctx_id, worker);
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker_str->sched_mutex);
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->sched_ctx_list_mutex);

		if(curr_workerid != -1)
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&curr_worker_str->sched_mutex);
	}
}

static unsigned _worker_sleeping_in_other_ctx(unsigned sched_ctx_id, int workerid)
{
	int s;
	for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
	{
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(s);
		if(sched_ctx && sched_ctx->id > 0 && sched_ctx->id < STARPU_NMAX_SCHED_CTXS && sched_ctx->id != sched_ctx_id)
		{
			if(sched_ctx->parallel_sect[workerid])
				return 1;
		}
	}
	return 0;

}

void _starpu_sched_ctx_signal_worker_blocked(unsigned sched_ctx_id, int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	worker->blocked = 1;
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->sleeping[workerid] = 1;
	sem_post(&sched_ctx->fall_asleep_sem[sched_ctx->main_master]);

	return;
}

void _starpu_sched_ctx_signal_worker_woke_up(unsigned sched_ctx_id, int workerid)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sem_post(&sched_ctx->wake_up_sem[sched_ctx->main_master]);
	sched_ctx->sleeping[workerid] = 0;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	worker->blocked = 0;

	return;
}

static void _starpu_sched_ctx_put_workers_to_sleep(unsigned sched_ctx_id, unsigned all)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int current_worker_id = starpu_worker_get_id();
	int master, temp_master = 0;
	struct starpu_worker_collection *workers = sched_ctx->workers;
	struct starpu_sched_ctx_iterator it;
	unsigned sleeping[workers->nworkers];
	int workers_count = 0;

	/* temporarily put a master if needed */
	if (sched_ctx->main_master == -1)
	{
		_starpu_sched_ctx_put_new_master(sched_ctx_id);
		temp_master = 1;
	}
	master = sched_ctx->main_master;

    workers->init_iterator(workers, &it);
    while(workers->has_next(workers, &it))
    {
			int workerid = workers->get_next(workers, &it);
			sleeping[workers_count] = _worker_sleeping_in_other_ctx(sched_ctx_id, workerid);

			if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER
				 && !sched_ctx->parallel_sect[workerid] && (workerid != master || all))
       {
            if (current_worker_id == -1 || workerid != current_worker_id)
            {
                STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->parallel_sect_mutex[workerid]);
                sched_ctx->parallel_sect[workerid] = 1;
                STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->parallel_sect_mutex[workerid]);
            }
        }
				workers_count++;
    }

		workers_count = 0;
    workers->init_iterator(workers, &it);
    while(workers->has_next(workers, &it))
    {
            int workerid = workers->get_next(workers, &it);
            if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER
							 && (workerid != master || all)
               && (current_worker_id == -1 || workerid != current_worker_id)
               && !sleeping[workers_count])
            {
                    sem_wait(&sched_ctx->fall_asleep_sem[master]);
            }
						workers_count++;
    }

		if (temp_master)
			sched_ctx->main_master = -1;

    return;
}

static void _starpu_sched_ctx_wake_up_workers(unsigned sched_ctx_id, unsigned all)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int current_worker_id = starpu_worker_get_id();
	int master, temp_master = 0;
	struct starpu_worker_collection *workers = sched_ctx->workers;
	struct starpu_sched_ctx_iterator it;

	/* temporarily put a master if needed */
	if (sched_ctx->main_master == -1)
	{
		_starpu_sched_ctx_put_new_master(sched_ctx_id);
		temp_master = 1;
	}
	master = sched_ctx->main_master;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		int workerid = workers->get_next(workers, &it);
		if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER
			 && sched_ctx->parallel_sect[workerid] && (workerid != master || all))
		{
			if((current_worker_id == -1 || workerid != current_worker_id) && sched_ctx->sleeping[workerid])
			{
				STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->parallel_sect_mutex[workerid]);
				STARPU_PTHREAD_COND_SIGNAL(&sched_ctx->parallel_sect_cond[workerid]);
				STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->parallel_sect_mutex[workerid]);
				sem_wait(&sched_ctx->wake_up_sem[master]);
			}
			else
				sched_ctx->parallel_sect[workerid] = 0;
		}
	}

	if (temp_master)
		sched_ctx->main_master = -1;

	return;
}

void* starpu_sched_ctx_exec_parallel_code(void* (*func)(void*), void* param, unsigned sched_ctx_id)
{
	_starpu_sched_ctx_put_workers_to_sleep(sched_ctx_id, 1);

	/* execute parallel code */
	void* ret = func(param);

	/* wake up starpu workers */
	_starpu_sched_ctx_wake_up_workers(sched_ctx_id, 1);
	return ret;
}

static void _starpu_sched_ctx_update_parallel_workers_with(unsigned sched_ctx_id)
{
    struct _starpu_sched_ctx * sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	if(sched_ctx->sched_policy)
		return;


	_starpu_sched_ctx_put_new_master(sched_ctx_id);

	if(!sched_ctx->awake_workers)
	{
		_starpu_sched_ctx_put_workers_to_sleep(sched_ctx_id, 0);
	}
}

static void _starpu_sched_ctx_update_parallel_workers_without(unsigned sched_ctx_id)
{
    struct _starpu_sched_ctx * sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	if(sched_ctx->sched_policy)
		return;


	_starpu_sched_ctx_put_new_master(sched_ctx_id);

	if(!sched_ctx->awake_workers)
	{
		_starpu_sched_ctx_wake_up_workers(sched_ctx_id, 0);
	}
}

void starpu_sched_ctx_get_available_cpuids(unsigned sched_ctx_id, int **cpuids, int *ncpuids)
{
	int current_worker_id = starpu_worker_get_id();
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_worker_collection *workers = sched_ctx->workers;
	_STARPU_MALLOC((*cpuids), workers->nworkers*sizeof(int));
	int w = 0;

	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		int workerid = workers->get_next(workers, &it);
		int master = sched_ctx->main_master;
		if(master == current_worker_id || workerid == current_worker_id || current_worker_id == -1)
		{
			(*cpuids)[w++] = starpu_worker_get_bindid(workerid);
		}
	}
	*ncpuids = w;
	return;
}

static void _starpu_sched_ctx_put_new_master(unsigned sched_ctx_id)
{
	int *workerids;
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	unsigned nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);
	unsigned i;

	for (i=0; i<nworkers; i++)
	{
		if (starpu_worker_get_type(workerids[i]) == STARPU_CPU_WORKER)
		{
			sched_ctx->main_master = workerids[i];
			break;
		}
	}
}

struct starpu_perfmodel_arch * _starpu_sched_ctx_get_perf_archtype(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return &sched_ctx->perf_arch;
}

int starpu_sched_ctx_get_worker_rank(unsigned sched_ctx_id)
{
	int idx = 0;
	int curr_workerid = starpu_worker_get_id();
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if(sched_ctx->sched_policy || !sched_ctx->awake_workers)
		return -1;
	struct starpu_worker_collection *workers = sched_ctx->workers;

	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		int worker = workers->get_next(workers, &it);
		if(worker == curr_workerid)
			return idx;
		idx++;
	}

	return -1;
}

void (*starpu_sched_ctx_get_sched_policy_init(unsigned sched_ctx_id))(unsigned)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->init_sched;
}

unsigned starpu_sched_ctx_has_starpu_scheduler(unsigned sched_ctx_id, unsigned *awake_workers)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	*awake_workers = sched_ctx->awake_workers;
	return sched_ctx->sched_policy != NULL;
}

void *starpu_sched_ctx_get_user_data(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_ASSERT(sched_ctx != NULL);
	return sched_ctx->user_data;
}
