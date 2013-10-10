/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  INRIA
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
 */

#include <starpu.h>
#include <starpu_scheduler.h>
#include <starpu_bitmap.h>

#include <common/fxt.h>

#define MIN_LEVEL	(-5)
#define MAX_LEVEL	(+5)

#define NPRIO_LEVELS	(MAX_LEVEL - MIN_LEVEL + 1)

struct _starpu_priority_taskq
{
	/* the actual lists
	 *	taskq[p] is for priority [p - STARPU_MIN_PRIO] */
	struct starpu_task_list taskq[NPRIO_LEVELS];
	unsigned ntasks[NPRIO_LEVELS];

	unsigned total_ntasks;
};

struct _starpu_eager_central_prio_data
{
	struct _starpu_priority_taskq *taskq;
	starpu_pthread_mutex_t policy_mutex;
	struct starpu_bitmap *waiters;
};

/*
 * Centralized queue with priorities
 */

static struct _starpu_priority_taskq *_starpu_create_priority_taskq(void)
{
	struct _starpu_priority_taskq *central_queue;

	central_queue = (struct _starpu_priority_taskq *) malloc(sizeof(struct _starpu_priority_taskq));
	central_queue->total_ntasks = 0;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		starpu_task_list_init(&central_queue->taskq[prio]);
		central_queue->ntasks[prio] = 0;
	}

	return central_queue;
}

static void _starpu_destroy_priority_taskq(struct _starpu_priority_taskq *priority_queue)
{
	free(priority_queue);
}

static void initialize_eager_center_priority_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)malloc(sizeof(struct _starpu_eager_central_prio_data));

	/* In this policy, we support more than two levels of priority. */
	starpu_sched_ctx_set_min_priority(sched_ctx_id, MIN_LEVEL);
	starpu_sched_ctx_set_max_priority(sched_ctx_id, MAX_LEVEL);

	/* only a single queue (even though there are several internaly) */
	data->taskq = _starpu_create_priority_taskq();
	data->waiters = starpu_bitmap_create();

	/* Tell helgrind that it's fine to check for empty fifo in
	 * _starpu_priority_pop_task without actual mutex (it's just an
	 * integer) */
	STARPU_HG_DISABLE_CHECKING(data->taskq->total_ntasks);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);

}

static void deinitialize_eager_center_priority_policy(unsigned sched_ctx_id)
{
	/* TODO check that there is no task left in the queue */
	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	/* deallocate the task queue */
	_starpu_destroy_priority_taskq(data->taskq);
	starpu_bitmap_destroy(data->waiters);

	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
	STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);
	free(data);
}

static int _starpu_priority_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct _starpu_priority_taskq *taskq = data->taskq;
	

	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	unsigned priolevel = task->priority - STARPU_MIN_PRIO;
	
	starpu_task_list_push_back(&taskq->taskq[priolevel], task);
	taskq->ntasks[priolevel]++;
	taskq->total_ntasks++;
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	/*if there are no tasks block */
	/* wake people waiting for a task */
	unsigned worker = 0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	
	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);
	
	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);

#ifdef STARPU_NON_BLOCKING_DRIVERS
		if (starpu_bitmap_get(data->waiters, worker))
		{
			/* This worker is waiting for a task */
			unsigned nimpl;
			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
				if (starpu_worker_can_execute_task(worker, task, nimpl))
				{
					/* It can execute this one, tell him! */
					starpu_bitmap_unset(data->waiters, worker);
					break;
				}
		}
#else
		starpu_pthread_mutex_t *sched_mutex;
		starpu_pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
		STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		STARPU_PTHREAD_COND_SIGNAL(sched_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
#endif
	}

	return 0;
}

static struct starpu_task *_starpu_priority_pop_task(unsigned sched_ctx_id)
{
	struct starpu_task *chosen_task = NULL, *task, *nexttask;
	unsigned workerid = starpu_worker_get_id();
	int skipped = 0;

	struct _starpu_eager_central_prio_data *data = (struct _starpu_eager_central_prio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct _starpu_priority_taskq *taskq = data->taskq;

	/* block until some event happens */
	/* Here helgrind would shout that this is unprotected, this is just an
	 * integer access, and we hold the sched mutex, so we can not miss any
	 * wake up. */
	if (taskq->total_ntasks == 0)
		return NULL;

	if (starpu_bitmap_get(data->waiters, workerid))
		/* Nobody woke us, avoid bothering the mutex */
		return NULL;

	/* release this mutex before trying to wake up other workers */
	starpu_pthread_mutex_t *curr_sched_mutex;
	starpu_pthread_cond_t *curr_sched_cond;
	starpu_worker_get_sched_condition(workerid, &curr_sched_mutex, &curr_sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(curr_sched_mutex);
	
	/* all workers will block on this mutex anyway so 
	   there's no need for their own mutex to be locked */
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	unsigned priolevel = NPRIO_LEVELS - 1;
	do
	{
		if (taskq->ntasks[priolevel] > 0)
		{
			for (task  = starpu_task_list_begin(&taskq->taskq[priolevel]);
			     task != starpu_task_list_end(&taskq->taskq[priolevel]) && !chosen_task;
			     task  = nexttask) 
			{
				unsigned nimpl;
				nexttask = starpu_task_list_next(task);
				for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
				{
					if (starpu_worker_can_execute_task(workerid, task, nimpl))
					{
						/* there is some task that we can grab */
						starpu_task_set_implementation(task, nimpl);
						starpu_task_list_erase(&taskq->taskq[priolevel], task);
						chosen_task = task;
						taskq->ntasks[priolevel]--;
						taskq->total_ntasks--;
						_STARPU_TRACE_JOB_POP(task, 0);
					} else skipped = 1;
				}
			}
		}
	}
	while (!chosen_task && priolevel-- > 0);


	if (!chosen_task && skipped)
	{
		/* Notify another worker to do that task */
		unsigned worker = 0;
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

		struct starpu_sched_ctx_iterator it;
		if(workers->init_iterator)
			workers->init_iterator(workers, &it);
		
		while(workers->has_next(workers, &it))
		{
			worker = workers->get_next(workers, &it);
			if(worker != workerid)
			{
#ifdef STARPU_NON_BLOCKING_DRIVERS
				starpu_bitmap_unset(data->waiters, worker);
#else
				starpu_pthread_mutex_t *sched_mutex;
				starpu_pthread_cond_t *sched_cond;
				starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
				STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
				STARPU_PTHREAD_COND_SIGNAL(sched_cond);
				STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
#endif
			}
		}
	
	}

	if (!chosen_task)
		/* Tell pushers that we are waiting for tasks for us */
		starpu_bitmap_set(data->waiters, workerid);

	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	/* leave the mutex how it was found before this */
	STARPU_PTHREAD_MUTEX_LOCK(curr_sched_mutex);

	return chosen_task;
}

struct starpu_sched_policy _starpu_sched_prio_policy =
{
	.init_sched = initialize_eager_center_priority_policy,
	.deinit_sched = deinitialize_eager_center_priority_policy,
	/* we always use priorities in that policy */
	.push_task = _starpu_priority_push_task,
	.pop_task = _starpu_priority_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "prio",
	.policy_description = "eager (with priorities)"
};
