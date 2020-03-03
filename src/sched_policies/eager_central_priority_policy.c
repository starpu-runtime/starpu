/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
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

/*
 *	This is policy where every worker use the same JOB QUEUE, but taking
 *	task priorities into account
 *
 *	TODO: merge with eager, after checking the scalability
 */

#include <starpu.h>
#include <starpu_scheduler.h>
#include <starpu_bitmap.h>
#include "prio_deque.h"
#include <limits.h>

#include <common/fxt.h>
#include <core/workers.h>

struct _starpu_eager_central_prio_data
{
	struct _starpu_prio_deque taskq;
	starpu_pthread_mutex_t policy_mutex;
	struct starpu_bitmap *waiters;
};

/*
 * Centralized queue with priorities
 */

static void initialize_eager_center_priority_policy(unsigned sched_ctx_id)
{
	struct _starpu_eager_central_prio_data *data;
	_STARPU_MALLOC(data, sizeof(struct _starpu_eager_central_prio_data));

	/* only a single queue (even though there are several internaly) */
	_starpu_prio_deque_init(&data->taskq);
	data->waiters = starpu_bitmap_create();

	/* Tell helgrind that it's fine to check for empty fifo in
	 * _starpu_priority_pop_task without actual mutex (it's just an
	 * integer) */
	STARPU_HG_DISABLE_CHECKING(data->taskq.ntasks);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);

	/* The application may use any integer */
	if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
	if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
}

static void deinitialize_eager_center_priority_policy(unsigned sched_ctx_id)
{
	/* TODO check that there is no task left in the queue */
	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	/* deallocate the job queue */
	_starpu_prio_deque_destroy(&data->taskq);
	starpu_bitmap_destroy(data->waiters);

	STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);
	free(data);
}

static int _starpu_priority_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_prio_deque *taskq = &data->taskq;

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_worker_relax_off();
	_starpu_prio_deque_push_back_task(taskq, task);

	if (_starpu_get_nsched_ctxs() > 1)
	{
		starpu_worker_relax_on();
		_starpu_sched_ctx_lock_write(sched_ctx_id);
		starpu_worker_relax_off();
		starpu_sched_ctx_list_task_counters_increment_all_ctx_locked(task, sched_ctx_id);
		_starpu_sched_ctx_unlock_write(sched_ctx_id);
	}

	starpu_push_task_end(task);

	/*if there are no tasks block */
	/* wake people waiting for a task */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
#ifndef STARPU_NON_BLOCKING_DRIVERS
	char dowake[STARPU_NMAXWORKERS] = { 0 };
#endif

	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);

#ifdef STARPU_NON_BLOCKING_DRIVERS
		if (!starpu_bitmap_get(data->waiters, worker))
			/* This worker is not waiting for a task */
			continue;
#endif

		if (starpu_worker_can_execute_task_first_impl(worker, task, NULL))
		{
			/* It can execute this one, tell him! */
#ifdef STARPU_NON_BLOCKING_DRIVERS
			starpu_bitmap_unset(data->waiters, worker);
			/* We really woke at least somebody, no need to wake somebody else */
			break;
#else
			dowake[worker] = 1;
#endif
		}
	}
	/* Let the task free */
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	/* Now that we have a list of potential workers, try to wake one */

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		if (dowake[worker])
			if (starpu_wake_worker_relax_light(worker))
				break; // wake up a single worker
	}
#endif

	return 0;
}

static struct starpu_task *_starpu_priority_pop_task(unsigned sched_ctx_id)
{
	struct starpu_task *chosen_task;
	unsigned workerid = starpu_worker_get_id_check();
	int skipped = 0;

	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct _starpu_prio_deque *taskq = &data->taskq;

	/* Here helgrind would shout that this is unprotected, this is just an
	 * integer access, and we hold the sched mutex, so we can not miss any
	 * wake up. */
	if (!STARPU_RUNNING_ON_VALGRIND && _starpu_prio_deque_is_empty(taskq))
	{
		return NULL;
	}

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (!STARPU_RUNNING_ON_VALGRIND && starpu_bitmap_get(data->waiters, workerid))
		/* Nobody woke us, avoid bothering the mutex */
	{
		return NULL;
	}
#endif

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_worker_relax_off();

	chosen_task = _starpu_prio_deque_pop_task_for_worker(taskq, workerid, &skipped);

	if (!chosen_task && skipped)
	{
		/* Notify another worker to do that task */
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

		struct starpu_sched_ctx_iterator it;
		workers->init_iterator(workers, &it);
		while(workers->has_next(workers, &it))
		{
			unsigned worker = workers->get_next(workers, &it);

			if(worker != workerid)
			{
#ifdef STARPU_NON_BLOCKING_DRIVERS
				starpu_bitmap_unset(data->waiters, worker);
#else
				starpu_wake_worker_relax_light(worker);
#endif
			}
		}

	}

	if (!chosen_task)
		/* Tell pushers that we are waiting for tasks for us */
		starpu_bitmap_set(data->waiters, workerid);

	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	if(chosen_task &&_starpu_get_nsched_ctxs() > 1)
	{
		starpu_worker_relax_on();
		_starpu_sched_ctx_lock_write(sched_ctx_id);
		starpu_worker_relax_off();
		starpu_sched_ctx_list_task_counters_decrement_all_ctx_locked(chosen_task, sched_ctx_id);

		if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, workerid, chosen_task))
			chosen_task = NULL;

		_starpu_sched_ctx_unlock_write(sched_ctx_id);
	}

	return chosen_task;
}

static void eager_center_priority_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	unsigned i;
        for (i = 0; i < nworkers; i++)
        {
		int workerid = workerids[i];
		int curr_workerid = _starpu_worker_get_id();
		if(workerid != curr_workerid)
			starpu_wake_worker_locked(workerid);

                starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);
        }
}

struct starpu_sched_policy _starpu_sched_prio_policy =
{
	.add_workers = eager_center_priority_add_workers,
	.init_sched = initialize_eager_center_priority_policy,
	.deinit_sched = deinitialize_eager_center_priority_policy,
	/* we always use priorities in that policy */
	.push_task = _starpu_priority_push_task,
	.pop_task = _starpu_priority_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "prio",
	.policy_description = "eager (with priorities)",
	.worker_type = STARPU_WORKER_LIST,
};
