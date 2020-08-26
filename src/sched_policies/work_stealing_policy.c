/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <limits.h>

#include <core/workers.h>
#include <sched_policies/prio_deque.h>
#include <core/debug.h>
#include <starpu_scheduler.h>
#include <core/sched_policy.h>
#include <core/debug.h>
#include <core/task.h>

/* Experimental (dead) code which needs to be tested, fixed... */
/* #define USE_OVERLOAD */

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
#ifdef USE_LOCALITY_TASKS
struct locality_entry
{
	UT_hash_handle hh;
	starpu_data_handle_t data;
	struct starpu_task *task;
};
#endif

struct _starpu_work_stealing_data_per_worker
{
	char fill1[STARPU_CACHELINE_SIZE];
	/* This is read-mostly, only updated when the queue becomes empty or
	 * becomes non-empty, to make it generally cheap to check */
	unsigned notask;	/* whether the queue is empty */
	char fill2[STARPU_CACHELINE_SIZE];

	struct _starpu_prio_deque queue;
	int running;
	int *proxlist;
	int busy;	/* Whether this worker is working on a task */

	/* keep track of the work performed from the beginning of the algorithm to make
	 * better decisions about which queue to select when deferring work
	 */
	unsigned last_pop_worker;

#ifdef USE_LOCALITY_TASKS
	/* This records the same as queue, but hashed by data accessed with locality flag.  */
	/* FIXME: we record only one task per data, assuming that the access is
	 * RW, and thus only one task is ready to write to it. Do we really need to handle the R case too? */
	struct locality_entry *queued_tasks_per_data;

	/* This records the last data accessed by the worker */
	starpu_data_handle_t last_locality[MAX_LOCALITY];
	int nlast_locality;
#endif
};

struct _starpu_work_stealing_data
{
	int (*select_victim)(struct _starpu_work_stealing_data *, unsigned, int);
	struct _starpu_work_stealing_data_per_worker *per_worker;
	/* keep track of the work performed from the beginning of the algorithm to make
	 * better decisions about which queue to select when deferring work
	 */
	unsigned last_push_worker;
};

#ifdef USE_OVERLOAD

/**
 * Minimum number of task we wait for being processed before we start assuming
 * on which worker the computation would be faster.
 */
static int calibration_value = 0;

#endif /* USE_OVERLOAD */


/**
 * Return a worker from which a task can be stolen.
 * Selecting a worker is done in a round-robin fashion, unless
 * the worker previously selected doesn't own any task,
 * then we return the first non-empty worker.
 */
static int select_victim_round_robin(struct _starpu_work_stealing_data *ws, unsigned sched_ctx_id)
{
	unsigned workerid = starpu_worker_get_id_check();
	unsigned worker = ws->per_worker[workerid].last_pop_worker;
	unsigned nworkers;
	int *workerids = NULL;
	nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);
	unsigned ntasks = 0;

	/* If the worker's queue is empty, let's try
	 * the next ones */
	while (1)
	{
		/* Here helgrind would shout that this is unprotected, but we
		 * are fine with getting outdated values, this is just an
		 * estimation */
		if (!ws->per_worker[workerids[worker]].notask)
		{
			if (ws->per_worker[workerids[worker]].busy
						   || starpu_worker_is_blocked_in_parallel(workerids[worker])) {
				ntasks = 1;
				break;
			}
		}

		worker = (worker + 1) % nworkers;
		if (worker == ws->per_worker[workerid].last_pop_worker)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			ntasks = 0;
			break;
		}
	}

	ws->per_worker[workerid].last_pop_worker = (worker + 1) % nworkers;

	worker = workerids[worker];

	if (ntasks)
		return worker;
	else
		return -1;
}

/**
 * Return a worker to whom add a task.
 * Selecting a worker is done in a round-robin fashion.
 */
static unsigned select_worker_round_robin(struct _starpu_work_stealing_data *ws, struct starpu_task *task, unsigned sched_ctx_id)
{
	unsigned worker;
	unsigned nworkers;
	int *workerids;
	nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);

	worker = ws->last_push_worker;
	do
		worker = (worker + 1) % nworkers;
	while (!ws->per_worker[workerids[worker]].running || !starpu_worker_can_execute_task_first_impl(workerids[worker], task, NULL));

	ws->last_push_worker = worker;

	return workerids[worker];
}

#ifdef USE_LOCALITY
/* Select a worker according to the locality of the data of the task to be scheduled */
static unsigned select_worker_locality(struct _starpu_work_stealing_data *ws, struct starpu_task *task, unsigned sched_ctx_id)
{
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	if (nbuffers == 0)
		return -1;

	unsigned i, n;
	unsigned ndata[STARPU_NMAXWORKERS] = { 0 };
	int best_worker = -1;

	n = 0;
	for (i = 0; i < nbuffers; i++)
	{
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			starpu_data_handle_t data = STARPU_TASK_GET_HANDLE(task, i);
			int locality = data->last_locality;
			if (locality >= 0)
				ndata[locality]++;
			n++;
		}
	}

	if (n)
	{
		/* Some locality buffers, choose worker which has most of them */
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
		struct starpu_sched_ctx_iterator it;
		unsigned best_ndata = 0;

		workers->init_iterator(workers, &it);
		while(workers->has_next(workers, &it))
		{
			int workerid = workers->get_next(workers, &it);
			if (ndata[workerid] > best_ndata && ws->per_worker[workerid].running && ws->per_worker[workerid].busy)
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
static void record_worker_locality(struct _starpu_work_stealing_data *ws, struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	/* Record where in locality data where the task went */
	unsigned i;
	struct _starpu_work_stealing_data_per_worker *data = &ws->per_worker[workerid];

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
static void locality_pushed_task(struct _starpu_work_stealing_data *ws, struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	struct _starpu_work_stealing_data_per_worker *data = &ws->per_worker[workerid];
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
static struct starpu_task *ws_pick_task(struct _starpu_work_stealing_data *ws, int source, int target)
{
	struct _starpu_work_stealing_data_per_worker *data_source = &ws->per_worker[source];
	struct _starpu_work_stealing_data_per_worker *data_target = &ws->per_worker[target];
	unsigned i, j, n = data_target->nlast_locality;
	struct starpu_task *(tasks[MAX_LOCALITY]) = { NULL }, *best_task = NULL;
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
		if (_starpu_prio_deque_pop_this_task(&data_source->queue, target, best_task))
		{
			if (!data_source->queue.ntasks)
			{
				STARPU_ASSERT(ws->per_worker[source].notask == 0);
				ws->per_worker[source].notask = 1;
			}
			return best_task;
		}
	}

	/* Didn't find an interesting task, or couldn't run it :( */
	int skipped;
	struct starpu_task *task;

	if (source != target)
		task = _starpu_prio_deque_deque_task_for_worker(&data_source->queue, target, &skipped);
	else
		task = _starpu_prio_deque_pop_task_for_worker(&data_source->queue, target, &skipped);

	if (task && !data_source->queue.ntasks)
	{
		STARPU_ASSERT(ws->per_worker[source].notask == 0);
		ws->per_worker[source].notask = 1;
	}
	return task;
}

/* Called when popping a task from a queue */
static void locality_popped_task(struct _starpu_work_stealing_data *ws, struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	struct _starpu_work_stealing_data_per_worker *data = &ws->per_worker[workerid];
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
static void record_worker_locality(struct _starpu_work_stealing_data *ws STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
}
/* Called when pushing a task to a queue */
static void locality_pushed_task(struct _starpu_work_stealing_data *ws STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
}
/* Pick a task from workerid's queue, for execution on target */
static struct starpu_task *ws_pick_task(struct _starpu_work_stealing_data *ws, int source, int target)
{
	int skipped;
	struct starpu_task *task;
	if (source != target)
		task = _starpu_prio_deque_deque_task_for_worker(&ws->per_worker[source].queue, target, &skipped);
	else
		task = _starpu_prio_deque_pop_task_for_worker(&ws->per_worker[source].queue, target, &skipped);

	if (task && !ws->per_worker[source].queue.ntasks)
	{
		STARPU_ASSERT(ws->per_worker[source].notask == 0);
		ws->per_worker[source].notask = 1;
	}
	return task;
}
/* Called when popping a task from a queue */
static void locality_popped_task(struct _starpu_work_stealing_data *ws STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
}
#endif

#ifdef USE_OVERLOAD

/**
 * Return a ratio helpful to determine whether a worker is suitable to steal
 * tasks from or to put some tasks in its queue.
 *
 * \return	a ratio with a positive or negative value, describing the current state of the worker :
 * 		a smaller value implies a faster worker with an relatively emptier queue : more suitable to put tasks in
 * 		a bigger value implies a slower worker with an reletively more replete queue : more suitable to steal tasks from
 */
static float overload_metric(struct _starpu_work_stealing_data *ws, unsigned sched_ctx_id, unsigned id)
{
	float execution_ratio = 0.0f;
	float current_ratio = 0.0f;

	int nprocessed = _starpu_get_deque_nprocessed(ws->per_worker[id].queue);
	unsigned njobs = _starpu_get_deque_njobs(ws->per_worker[id].queue);

	/* Did we get enough information ? */
	if (ws->performed_total > 0 && nprocessed > 0)
	{
		/* How fast or slow is the worker compared to the other workers */
		execution_ratio = (float) nprocessed / ws->performed_total;
		/* How replete is its queue */
		current_ratio = (float) njobs / nprocessed;
	}
	else
	{
		return 0.0f;
	}

	return (current_ratio - execution_ratio);
}

/**
 * Return the most suitable worker from which a task can be stolen.
 * The number of previously processed tasks, total and local,
 * and the number of tasks currently awaiting to be processed
 * by the tasks are taken into account to select the most suitable
 * worker to steal task from.
 */
static int select_victim_overload(struct _starpu_work_stealing_data *ws, unsigned sched_ctx_id)
{
	unsigned best_worker = 0;
	float best_ratio = FLT_MIN;

	/* Don't try to play smart until we get
	 * enough informations. */
	if (ws->performed_total < calibration_value)
		return select_victim_round_robin(ws, sched_ctx_id);

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
                unsigned worker = workers->get_next(workers, &it);
		float worker_ratio = overload_metric(ws, sched_ctx_id, worker);

		if (worker_ratio > best_ratio && ws->per_worker[worker].running && ws->per_worker[worker].busy)
		{
			best_worker = worker;
			best_ratio = worker_ratio;
		}
	}

	return best_worker;
}

/**
 * Return the most suitable worker to whom add a task.
 * The number of previously processed tasks, total and local,
 * and the number of tasks currently awaiting to be processed
 * by the tasks are taken into account to select the most suitable
 * worker to add a task to.
 */
static unsigned select_worker_overload(struct _starpu_work_stealing_data *ws, struct starpu_task *task, unsigned sched_ctx_id)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned best_worker = 0;
	float best_ratio = FLT_MAX;

	/* Don't try to play smart until we get
	 * enough informations. */
	if (ws->performed_total < calibration_value)
		return select_worker_round_robin(task, sched_ctx_id);

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		float worker_ratio = overload_metric(ws, sched_ctx_id, worker);

		if (worker_ratio < best_ratio && ws->per_worker[worker].running && starpu_worker_can_execute_task_first_impl(worker, task, NULL))
		{
			best_worker = worker;
			best_ratio = worker_ratio;
		}
	}

	return best_worker;
}

#endif /* USE_OVERLOAD */


/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline int select_victim(struct _starpu_work_stealing_data *ws, unsigned sched_ctx_id,
				     int workerid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef USE_OVERLOAD
	return select_victim_overload(ws, sched_ctx_id);
#else
	return select_victim_round_robin(ws, sched_ctx_id);
#endif /* USE_OVERLOAD */
}

/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline unsigned select_worker(struct _starpu_work_stealing_data *ws, struct starpu_task *task, unsigned sched_ctx_id)
{
#ifdef USE_OVERLOAD
	return select_worker_overload(ws, task, sched_ctx_id);
#else
	return select_worker_round_robin(ws, task, sched_ctx_id);
#endif /* USE_OVERLOAD */
}


/* Note: this is not scalable work stealing,  use lws instead */
static struct starpu_task *ws_pop_task(unsigned sched_ctx_id)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task = NULL;
	unsigned workerid = starpu_worker_get_id_check();

	if (ws->per_worker[workerid].busy)
		ws->per_worker[workerid].busy = 0;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (STARPU_RUNNING_ON_VALGRIND || !_starpu_prio_deque_is_empty(&ws->per_worker[workerid].queue))
#endif
	{
		task = ws_pick_task(ws, workerid, workerid);
		if (task)
			locality_popped_task(ws, task, workerid, sched_ctx_id);
	}

	if(task)
	{
		/* there was a local task */
		ws->per_worker[workerid].busy = 1;
		if (_starpu_get_nsched_ctxs() > 1)
		{
			starpu_worker_relax_on();
			_starpu_sched_ctx_lock_write(sched_ctx_id);
			starpu_worker_relax_off();
			starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, workerid);
			if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, workerid, task))
				task = NULL;
			_starpu_sched_ctx_unlock_write(sched_ctx_id);
		}
		return task;
	}

	/* we need to steal someone's job */
	starpu_worker_relax_on();
	int victim = ws->select_victim(ws, sched_ctx_id, workerid);
	starpu_worker_relax_off();
	if (victim == -1)
	{
		return NULL;
	}

	if (_starpu_worker_trylock(victim))
	{
		/* victim is busy, don't bother it, come back later */
		return NULL;
	}
	if (ws->per_worker[victim].running && ws->per_worker[victim].queue.ntasks > 0)
	{
		task = ws_pick_task(ws, victim, workerid);
	}

	if (task)
	{
		_STARPU_TRACE_WORK_STEALING(workerid, victim);
		starpu_sched_task_break(task);
		starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, victim);
		record_data_locality(task, workerid);
		record_worker_locality(ws, task, workerid, sched_ctx_id);
		locality_popped_task(ws, task, victim, sched_ctx_id);
	}
	starpu_worker_unlock(victim);

#ifndef STARPU_NON_BLOCKING_DRIVERS
        /* While stealing, perhaps somebody actually give us a task, don't miss
         * the opportunity to take it before going to sleep. */
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(starpu_worker_get_id());
		if (!task && worker->state_keep_awake)
		{
			task = ws_pick_task(ws, workerid, workerid);
			if (task)
			{
				/* keep_awake notice taken into account here, clear flag */
				worker->state_keep_awake = 0;
				locality_popped_task(ws, task, workerid, sched_ctx_id);
			}
		}
	}
#endif

	if (task &&_starpu_get_nsched_ctxs() > 1)
	{
		starpu_worker_relax_on();
		_starpu_sched_ctx_lock_write(sched_ctx_id);
		starpu_worker_relax_off();
		if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, workerid, task))
			task = NULL;
		_starpu_sched_ctx_unlock_write(sched_ctx_id);
		if (!task)
			return NULL;
	}
	if (ws->per_worker[workerid].busy != !!task)
		ws->per_worker[workerid].busy = !!task;
	return task;
}

static
int ws_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid;

#ifdef USE_LOCALITY
	workerid = select_worker_locality(ws, task, sched_ctx_id);
#else
	workerid = -1;
#endif
	if (workerid == -1)
		workerid = starpu_worker_get_id();

	/* If the current thread is not a worker but
	 * the main thread (-1) or the current worker is not in the target
	 * context, we find the better one to put task on its queue */
	if (workerid == -1 || !starpu_sched_ctx_contains_worker(workerid, sched_ctx_id) ||
			!starpu_worker_can_execute_task_first_impl(workerid, task, NULL))
		workerid = select_worker(ws, task, sched_ctx_id);
	starpu_worker_lock(workerid);
	STARPU_AYU_ADDTOTASKQUEUE(starpu_task_get_job_id(task), workerid);
	starpu_sched_task_break(task);
	record_data_locality(task, workerid);
	STARPU_ASSERT_MSG(ws->per_worker[workerid].running, "workerid=%d, ws=%p\n", workerid, ws);
	_starpu_prio_deque_push_back_task(&ws->per_worker[workerid].queue, task);
	if (ws->per_worker[workerid].queue.ntasks == 1)
	{
		STARPU_ASSERT(ws->per_worker[workerid].notask == 1);
		ws->per_worker[workerid].notask = 0;
	}
	locality_pushed_task(ws, task, workerid, sched_ctx_id);

	starpu_push_task_end(task);
	starpu_worker_unlock(workerid);
	starpu_sched_ctx_list_task_counters_increment(sched_ctx_id, workerid);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	/* TODO: implement fine-grain signaling, similar to what eager does */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
		starpu_wake_worker_relax_light(workers->get_next(workers, &it));
#endif
	return 0;
}

static void ws_add_workers(unsigned sched_ctx_id, int *workerids,unsigned nworkers)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;

	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);
		_starpu_prio_deque_init(&ws->per_worker[workerid].queue);
		ws->per_worker[workerid].notask = 1;
		ws->per_worker[workerid].running = 1;

		/* Tell helgrind that we are fine with getting outdated values,
		 * this is just an estimation */
		STARPU_HG_DISABLE_CHECKING(ws->per_worker[workerid].notask);
		STARPU_HG_DISABLE_CHECKING(ws->per_worker[workerid].queue.ntasks);
		ws->per_worker[workerid].busy = 0;
		STARPU_HG_DISABLE_CHECKING(ws->per_worker[workerid].busy);
	}
}

static void ws_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;

	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];

		_starpu_prio_deque_destroy(&ws->per_worker[workerid].queue);
		ws->per_worker[workerid].running = 0;
		free(ws->per_worker[workerid].proxlist);
		ws->per_worker[workerid].proxlist = NULL;
	}
}

static void initialize_ws_policy(unsigned sched_ctx_id)
{
	struct _starpu_work_stealing_data *ws;
	_STARPU_MALLOC(ws, sizeof(struct _starpu_work_stealing_data));
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)ws);

	ws->last_push_worker = 0;
	STARPU_HG_DISABLE_CHECKING(ws->last_push_worker);
	ws->select_victim = select_victim;

	unsigned nw = starpu_worker_get_count();
	_STARPU_CALLOC(ws->per_worker, nw, sizeof(struct _starpu_work_stealing_data_per_worker));

	/* The application may use any integer */
	if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
	if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
}

static void deinit_ws_policy(unsigned sched_ctx_id)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	free(ws->per_worker);
	free(ws);
}

struct starpu_sched_policy _starpu_sched_ws_policy =
{
	.init_sched = initialize_ws_policy,
	.deinit_sched = deinit_ws_policy,
	.add_workers = ws_add_workers,
	.remove_workers = ws_remove_workers,
	.push_task = ws_push_task,
	.pop_task = ws_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "ws",
	.policy_description = "work stealing",
	.worker_type = STARPU_WORKER_LIST,
};

/* local work stealing policy */
/* Return a worker to steal a task from. The worker is selected according to
 * the proximity list built using the info on te architecture provided by hwloc
 */
#ifdef STARPU_HAVE_HWLOC
static int lws_select_victim(struct _starpu_work_stealing_data *ws, unsigned sched_ctx_id, int workerid)
{
	int nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
	int i;
	for (i = 0; i < nworkers; i++)
	{
		int neighbor = ws->per_worker[workerid].proxlist[i];
		if (ws->per_worker[neighbor].notask)
			continue;
                /* FIXME: do not keep looking again and again at some worker
                 * which has tasks, but that can't execute on me */
		if (ws->per_worker[neighbor].busy
					   || starpu_worker_is_blocked_in_parallel(neighbor))
			return neighbor;
	}
	return -1;
}
#endif

static void lws_add_workers(unsigned sched_ctx_id, int *workerids,
			    unsigned nworkers)
{
	ws_add_workers(sched_ctx_id, workerids, nworkers);

#ifdef STARPU_HAVE_HWLOC
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	/* Build a proximity list for every worker. It is cheaper to
	 * build this once and then use it for popping tasks rather
	 * than traversing the hwloc tree every time a task must be
	 * stolen */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_tree *tree = (struct starpu_tree*)workers->collection_private;
	unsigned i;

	/* get the complete list of workers (not just the added one) and rebuild the proxlists */
	nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);
	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		if (ws->per_worker[workerid].proxlist == NULL)
			_STARPU_CALLOC(ws->per_worker[workerid].proxlist, STARPU_NMAXWORKERS, sizeof(int));
		int bindid;

		struct starpu_sched_ctx_iterator it;
		workers->init_iterator(workers, &it);

		bindid   = starpu_worker_get_bindid(workerid);
		it.value = starpu_tree_get(tree, bindid);
		int cnt = 0;
		for(;;)
		{
			struct starpu_tree *neighbour = (struct starpu_tree*)it.value;
			int *neigh_workerids;
			int neigh_nworkers = starpu_bindid_get_workerids(neighbour->id, &neigh_workerids);
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

static void initialize_lws_policy(unsigned sched_ctx_id)
{
	/* lws is loosely based on ws, except that it might use hwloc. */
	initialize_ws_policy(sched_ctx_id);

	if (starpu_worker_get_count() != starpu_cpu_worker_get_count()
			|| starpu_memory_nodes_get_numa_count() > 1
		)
	{
		_STARPU_DISP("Warning: you are running the default lws scheduler, which is not a very smart scheduler, while the system has GPUs or several memory nodes. Make sure to read the StarPU documentation about adding performance models in order to be able to use the dmda or dmdas scheduler instead.\n");
	}

#ifdef STARPU_HAVE_HWLOC
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data *)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	ws->select_victim = lws_select_victim;
#endif
}

struct starpu_sched_policy _starpu_sched_lws_policy =
{
	.init_sched = initialize_lws_policy,
	.deinit_sched = deinit_ws_policy,
	.add_workers = lws_add_workers,
	.remove_workers = ws_remove_workers,
	.push_task = ws_push_task,
	.pop_task = ws_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "lws",
	.policy_description = "locality work stealing",
#ifdef STARPU_HAVE_HWLOC
	.worker_type = STARPU_WORKER_TREE,
#else
	.worker_type = STARPU_WORKER_LIST,
#endif
};
