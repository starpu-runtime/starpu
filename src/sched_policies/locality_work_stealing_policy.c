/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2017, 2020                          CNRS
 * Copyright (C) 2011-2012,2016                           Inria
 * Copyright (C) 2010-2018                                Université de Bordeaux
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

/* Work stealing policy */

#include <float.h>

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <core/debug.h>
#include <starpu_bitmap.h>

/*
 * Experimental code for improving data cache locality:
 *
 * USE_LOCALITY:
 * - for each data, we record on which worker it was last accessed with the
 *   locality flag.
 *
 * - when pushing a ready task, we choose the worker which has last accessed the
 *   most data of the task with the locality flag.
 *
 * USE_LOCALITY_TASKS:
 * - for each worker, we record the locality data that the task used last (i.e. a rough
 *   estimation of what is contained in the innermost caches).
 *
 * - for each worker, we have a hash table associating from a data handle to
 *   all the ready tasks pushed to it that will use it with the locality flag.
 *
 * - When fetching a task from a queue, pick a task which has the biggest number
 *   of data estimated to be contained in the cache.
 */

//#define USE_LOCALITY


//#define USE_LOCALITY_TASKS

/* Maximum number of recorded locality data per task */
#define MAX_LOCALITY 8

/* Entry for queued_tasks_per_data: records that a queued task is accessing the data with locality flag */
struct locality_entry {
	UT_hash_handle hh;
	starpu_data_handle_t data;
	struct starpu_task *task;
};

struct _starpu_lws_data_per_worker
{
	struct _starpu_fifo_taskq *queue_array;
	int *proxlist;
	starpu_pthread_mutex_t worker_mutex;
	int busy;

#ifdef USE_LOCALITY_TASKS
	/* This records the same as queue_array, but hashed by data accessed with locality flag.  */
	/* FIXME: we record only one task per data, assuming that the access is
	 * RW, and thus only one task is ready to write to it. Do we really need to handle the R case too? */
	struct locality_entry *queued_tasks_per_data;

	/* This records the last data accessed by the worker */
	starpu_data_handle_t last_locality[MAX_LOCALITY];
	int nlast_locality;
#endif
};

struct _starpu_lws_data
{
	struct _starpu_lws_data_per_worker *per_worker;
	unsigned last_pop_worker;
	unsigned last_push_worker;
};


#ifdef STARPU_HAVE_HWLOC

/* Return a worker to steal a task from. The worker is selected
 * according to the proximity list built using the info on te
 * architecture provided by hwloc */
static int select_victim_neighborhood(unsigned sched_ctx_id, int workerid)
{

	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);

	int i;
	int neighbor;
	for(i=0; i<nworkers; i++)
	{
		neighbor = ws->per_worker[workerid].proxlist[i];
		int ntasks = ws->per_worker[neighbor].queue_array->ntasks;

		if (ntasks && ws->per_worker[neighbor].busy)
			return neighbor;
	}

	return workerid;
}
#else
/* Return a worker to steal a task from. The worker is selected
 * in a round-robin fashion */
static int select_victim_round_robin(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned worker = ws->last_pop_worker;
	unsigned nworkers;
	int *workerids;
	unsigned ntasks;

	nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);

	/* If the worker's queue is empty, let's try
	 * the next ones */
	while (1)
	{
		ntasks = ws->per_worker[workerids[worker]].queue_array->ntasks;
		if (ntasks && ws->per_worker[workerids[worker]].busy)
			break;

		worker = (worker + 1) % nworkers;
		if (worker == ws->last_pop_worker)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			ntasks = 0;
			break;
		}
	}

	ws->last_pop_worker = (worker + 1) % nworkers;

	worker = workerids[worker];

	if (ntasks)
		return worker;
	else
		return -1;
}


#endif


/**
 * Return a worker to whom add a task.
 * Selecting a worker is done in a round-robin fashion.
 */
static unsigned select_worker_round_robin(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned worker;
	unsigned nworkers;
	int *workerids;
	nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);

	worker = ws->last_push_worker;
	do
		worker = (worker + 1) % nworkers;
	while (!starpu_worker_can_execute_task_first_impl(workerids[worker], task, NULL));

	ws->last_push_worker = worker;

	return workerids[worker];
}

#ifdef USE_LOCALITY
/* Select a worker according to the locality of the data of the task to be scheduled */
static unsigned select_worker_locality(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	if (nbuffers == 0)
		return -1;

	unsigned i, n;
	unsigned ndata[STARPU_NMAXWORKERS] = { 0 };
	int best_worker = -1;
	unsigned best_ndata = 0;

	n = 0;
	for (i = 0; i < nbuffers; i++)
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			starpu_data_handle_t data = STARPU_TASK_GET_HANDLE(task, i);
			int locality = data->last_locality;
			if (locality >= 0)
				ndata[locality]++;
			n++;
		}

	if (n)
	{
		/* Some locality buffers, choose worker which has most of them */

		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
		struct starpu_sched_ctx_iterator it;
		workers->init_iterator(workers, &it);

		while(workers->has_next(workers, &it))
		{
			int workerid = workers->get_next(workers, &it);
			if (ndata[workerid] > best_ndata && ws->per_worker[workerid].busy)
			{
				best_worker = workerid;
				best_ndata = ndata[workerid];
			}
		}
	}
	return best_worker;
}

/* Record in the data which worker will handle the task with the locality flag */
static void record_data_locality(struct starpu_task *task, int workerid)
{
	/* Record where in locality data where the task went */
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			STARPU_TASK_GET_HANDLE(task, i)->last_locality = workerid;
		}
}
#else
static void record_data_locality(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED)
{
}
#endif

#ifdef USE_LOCALITY_TASKS
/* Record in the worker which data it used last with the locality flag */
static void record_worker_locality(struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	/* Record where in locality data where the task went */
	unsigned i;
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_lws_data_per_worker *data = &ws->per_worker[workerid];

	data->nlast_locality = 0;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			data->last_locality[data->nlast_locality] = STARPU_TASK_GET_HANDLE(task, i);
			data->nlast_locality++;
			if (data->nlast_locality == MAX_LOCALITY)
				break;
		}
}
/* Called when pushing a task to a queue */
static void locality_pushed_task(struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_lws_data_per_worker *data = &ws->per_worker[workerid];
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			struct locality_entry *entry;
			HASH_FIND_PTR(data->queued_tasks_per_data, &handle, entry);
			if (STARPU_LIKELY(!entry))
			{
				_STARPU_MALLOC(entry, sizeof(*entry));
				entry->data = handle;
				entry->task = task;
				HASH_ADD_PTR(data->queued_tasks_per_data, data, entry);
			}
		}
}

/* Pick a task from workerid's queue, for execution on target */
static struct starpu_task *ws_pick_task(struct _starpu_lws_data *ws, int source, int target)
{
	struct _starpu_lws_data_per_worker *data_source = &ws->per_worker[source];
	struct _starpu_lws_data_per_worker *data_target = &ws->per_worker[target];
	unsigned i, j, n = data_target->nlast_locality;
	struct starpu_task *(tasks[MAX_LOCALITY]) = { NULL }, *best_task;
	int ntasks[MAX_LOCALITY] = { 0 }, best_n; /* Number of locality data for this worker used by this task */
	/* Look at the last data accessed by this worker */
	STARPU_ASSERT(n < MAX_LOCALITY);
	for (i = 0; i < n; i++)
	{
		starpu_data_handle_t handle = data_target->last_locality[i];
		struct locality_entry *entry;
		HASH_FIND_PTR(data_source->queued_tasks_per_data, &handle, entry);
		if (entry)
		{
			/* Record task */
			tasks[i] = entry->task;
			ntasks[i] = 1;

			/* And increment counter of the same task */
			for (j = 0; j < i; j++)
			{
				if (tasks[j] == tasks[i])
				{
					ntasks[j]++;
					break;
				}
			}
		}
	}
	/* Now find the task with most locality data for this worker */
	best_n = 0;
	for (i = 0; i < n; i++)
	{
		if (ntasks[i] > best_n)
		{
			best_task = tasks[i];
			best_n = ntasks[i];
		}
	}

	if (best_n > 0)
	{
		/* found an interesting task, try to pick it! */
		if (_starpu_fifo_pop_this_task(data_source->queue_array, target, best_task))
			return best_task;
	}

	/* Didn't find an interesting task, or couldn't run it :( */
	return _starpu_fifo_pop_task(data_source->queue_array, target);
}

/* Called when popping a task from a queue */
static void locality_popped_task(struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_lws_data_per_worker *data = &ws->per_worker[workerid];
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			struct locality_entry *entry;
			HASH_FIND_PTR(data->queued_tasks_per_data, &handle, entry);
			if (STARPU_LIKELY(entry))
			{
				if (entry->task == task)
				{
					HASH_DEL(data->queued_tasks_per_data, entry);
					free(entry);
				}
			}
		}
}
#else
static void record_worker_locality(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
}
/* Called when pushing a task to a queue */
static void locality_pushed_task(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
}
/* Pick a task from workerid's queue, for execution on target */
static struct starpu_task *ws_pick_task(struct _starpu_lws_data *ws, int source, int target)
{
	return _starpu_fifo_pop_task(ws->per_worker[source].queue_array, target);
}
/* Called when popping a task from a queue */
static void locality_popped_task(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
}
#endif


/**
 * Return a worker from which a task can be stolen.
 */
static inline int select_victim(unsigned sched_ctx_id, int workerid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_HAVE_HWLOC
	return select_victim_neighborhood(sched_ctx_id, workerid);
#else
	return select_victim_round_robin(sched_ctx_id);
#endif
}

/**
 * Return a worker on whose queue a task can be pushed. This is only
 * needed when the push is done by the master
 */
static inline unsigned select_worker(struct starpu_task *task, unsigned sched_ctx_id)
{
	return select_worker_round_robin(task, sched_ctx_id);
}


static struct starpu_task *lws_pop_task(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task = NULL;
	unsigned workerid = starpu_worker_get_id_check();

	ws->per_worker[workerid].busy = 0;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (STARPU_RUNNING_ON_VALGRIND || !_starpu_fifo_empty(ws->per_worker[workerid].queue_array))
#endif
	{
		STARPU_PTHREAD_MUTEX_LOCK(&ws->per_worker[workerid].worker_mutex);
		task = ws_pick_task(ws, workerid, workerid);
		if (task)
			locality_popped_task(task, workerid, sched_ctx_id);
		STARPU_PTHREAD_MUTEX_UNLOCK(&ws->per_worker[workerid].worker_mutex);
	}

	if (task)
	{
		/* there was a local task */
		/* printf("Own    task!%d\n",workerid); */
		ws->per_worker[workerid].busy = 1;
		return task;
	}

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	/* While stealing, relieve mutex used to synchronize with pushers */
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);


	/* we need to steal someone's job */
	int victim = select_victim(sched_ctx_id, workerid);
	if (victim == -1)
	{
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
		return NULL;
	}

	if (STARPU_PTHREAD_MUTEX_TRYLOCK(&ws->per_worker[victim].worker_mutex))
	{
		/* victim is busy, don't bother it, come back later */
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
		return NULL;
	}
	if (ws->per_worker[victim].queue_array != NULL && ws->per_worker[victim].queue_array->ntasks > 0)
	{
		task = ws_pick_task(ws, victim, workerid);
	}

	if (task)
	{
		_STARPU_TRACE_WORK_STEALING(workerid, victim);
		_STARPU_TASK_BREAK_ON(task, sched);
		record_data_locality(task, workerid);
		record_worker_locality(task, workerid, sched_ctx_id);
		locality_popped_task(task, victim, sched_ctx_id);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&ws->per_worker[victim].worker_mutex);

	/* Done with stealing, resynchronize with core */
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);

#ifndef STARPU_NON_BLOCKING_DRIVERS
        /* While stealing, perhaps somebody actually give us a task, don't miss
         * the opportunity to take it before going to sleep. */
	if (!task) {
		STARPU_PTHREAD_MUTEX_LOCK(&ws->per_worker[workerid].worker_mutex);
		task = ws_pick_task(ws, workerid, workerid);
		if (task)
			locality_popped_task(task, workerid, sched_ctx_id);
		STARPU_PTHREAD_MUTEX_UNLOCK(&ws->per_worker[workerid].worker_mutex);
	}
#endif
	ws->per_worker[workerid].busy = !!task;
	return task;
}

static int lws_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int workerid = -1;

#ifdef USE_LOCALITY
	workerid = select_worker_locality(task, sched_ctx_id);
#endif
	if (workerid == -1)
		workerid = starpu_worker_get_id();

	/* If the current thread is not a worker but
	 * the main thread (-1) or the current worker is not in the target
	 * context, we find the better one to put task on its queue */
	if (workerid == -1 || !starpu_sched_ctx_contains_worker(workerid, sched_ctx_id) ||
			!starpu_worker_can_execute_task_first_impl(workerid, task, NULL))
		workerid = select_worker(task, sched_ctx_id);

	record_data_locality(task, workerid);
	/* int workerid = starpu_worker_get_id(); */
	/* print_neighborhood(sched_ctx_id, 0); */

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);

	STARPU_AYU_ADDTOTASKQUEUE(_starpu_get_job_associated_to_task(task)->job_id, workerid);
	STARPU_PTHREAD_MUTEX_LOCK(&ws->per_worker[workerid].worker_mutex);
	_STARPU_TASK_BREAK_ON(task, sched);
	_starpu_fifo_push_task(ws->per_worker[workerid].queue_array, task);
	locality_pushed_task(task, workerid, sched_ctx_id);
	starpu_push_task_end(task);

	STARPU_PTHREAD_MUTEX_UNLOCK(&ws->per_worker[workerid].worker_mutex);

	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	/* TODO: implement fine-grain signaling, similar to what eager does */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
		starpu_wake_worker(workers->get_next(workers, &it));
#endif



	return 0;
}

static void lws_add_workers(unsigned sched_ctx_id, int *workerids,unsigned nworkers)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	int workerid;

	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);
		ws->per_worker[workerid].queue_array = _starpu_create_fifo();

		/* Tell helgrid that we are fine with getting outdated values,
		 * this is just an estimation */
		STARPU_HG_DISABLE_CHECKING(ws->per_worker[workerid].queue_array->ntasks);

		ws->per_worker[workerid].queue_array->nprocessed = 0;
		ws->per_worker[workerid].queue_array->ntasks = 0;
		STARPU_PTHREAD_MUTEX_INIT(&ws->per_worker[workerid].worker_mutex, NULL);
		ws->per_worker[workerid].busy = 0;
		STARPU_HG_DISABLE_CHECKING(ws->per_worker[workerid].busy);
	}


#ifdef STARPU_HAVE_HWLOC
	/* Build a proximity list for every worker. It is cheaper to
	 * build this once and then use it for popping tasks rather
	 * than traversing the hwloc tree every time a task must be
	 * stolen */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_tree *tree = (struct starpu_tree*)workers->collection_private;

	/* get the complete list of workers (not just the added one) and rebuild the proxlists */
	nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);
	for (i = 0; i < nworkers; i++)
	{
		int bindid;
		struct starpu_tree *neighbour = NULL;
		struct starpu_sched_ctx_iterator it;

		workerid = workerids[i];
		if (ws->per_worker[workerid].proxlist == NULL)
			_STARPU_CALLOC(ws->per_worker[workerid].proxlist, STARPU_NMAXWORKERS, sizeof(int));
		workers->init_iterator(workers, &it);

		bindid   = starpu_worker_get_bindid(workerid);
		it.value = starpu_tree_get(tree, bindid);
		int cnt = 0;
		for(;;)
		{
			neighbour = (struct starpu_tree*)it.value;
			int *neigh_workerids;
			int neigh_nworkers = _starpu_bindid_get_workerids(neighbour->id, &neigh_workerids);
			int w;
			for(w = 0; w < neigh_nworkers; w++)
			{
				if(!it.visited[neigh_workerids[w]] && workers->present[neigh_workerids[w]])
				{
					ws->per_worker[workerid].proxlist[cnt++] = neigh_workerids[w];
					it.visited[neigh_workerids[w]] = 1;
				}
			}
			if(!workers->has_next(workers, &it))
				break;
			it.value = it.possible_value;
			it.possible_value = NULL;
		}
	}
#endif
}

static void lws_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	int workerid;

	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		_starpu_destroy_fifo(ws->per_worker[workerid].queue_array);
#ifdef STARPU_HAVE_HWLOC
		free(ws->per_worker[workerid].proxlist);
		ws->per_worker[workerid].proxlist = NULL;
#endif
		STARPU_PTHREAD_MUTEX_DESTROY(&ws->per_worker[workerid].worker_mutex);
	}
}

static void lws_initialize_policy(unsigned sched_ctx_id)
{
#ifdef STARPU_HAVE_HWLOC
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_TREE);
#else
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
#endif

	struct _starpu_lws_data *ws;
	_STARPU_MALLOC(ws, sizeof(struct _starpu_lws_data));
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)ws);

	ws->last_pop_worker = 0;
	ws->last_push_worker = 0;
	STARPU_HG_DISABLE_CHECKING(ws->last_pop_worker);
	STARPU_HG_DISABLE_CHECKING(ws->last_push_worker);

	/* unsigned nw = starpu_sched_ctx_get_nworkers(sched_ctx_id); */
	unsigned nw = starpu_worker_get_count();
	_STARPU_CALLOC(ws->per_worker, nw, sizeof(struct _starpu_lws_data_per_worker));

}

static void lws_deinit_policy(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	free(ws->per_worker);
	free(ws);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_lws_policy =
{
	.init_sched = lws_initialize_policy,
	.deinit_sched = lws_deinit_policy,
	.add_workers = lws_add_workers,
	.remove_workers = lws_remove_workers,
	.push_task = lws_push_task,
	.pop_task = lws_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "lws",
	.policy_description = "locality work stealing"
};
