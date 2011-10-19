/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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
#include <common/config.h>
#include <core/workers.h>
#include <common/utils.h>

#define MIN_LEVEL	(-5)
#define MAX_LEVEL	(+5)

#define NPRIO_LEVELS	(MAX_LEVEL - MIN_LEVEL + 1)

struct starpu_priority_taskq_s {
	/* the actual lists 
	 *	taskq[p] is for priority [p - STARPU_MIN_PRIO] */
	struct starpu_task_list taskq[NPRIO_LEVELS];
	unsigned ntasks[NPRIO_LEVELS];

	unsigned total_ntasks;
};

typedef struct eager_central_prio_data{
	struct starpu_priority_taskq_s *taskq;
	pthread_mutex_t sched_mutex;
	pthread_cond_t sched_cond;
} eager_central_prio_data;

/*
 * Centralized queue with priorities 
 */

static struct starpu_priority_taskq_s *_starpu_create_priority_taskq(void)
{
	struct starpu_priority_taskq_s *central_queue;
	
	central_queue = (struct starpu_priority_taskq_s *) malloc(sizeof(struct starpu_priority_taskq_s));
	central_queue->total_ntasks = 0;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		starpu_task_list_init(&central_queue->taskq[prio]);
		central_queue->ntasks[prio] = 0;
	}

	return central_queue;
}

static void _starpu_destroy_priority_taskq(struct starpu_priority_taskq_s *priority_queue)
{
	free(priority_queue);
}

static void initialize_eager_center_priority_policy_for_workers(unsigned sched_ctx_id, int *workerids, unsigned nnew_workers) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_central_prio_data *data = (struct eager_central_prio_data*)sched_ctx->policy_data;

	unsigned nworkers_ctx = sched_ctx->nworkers;

	unsigned i;
	int workerid;
	for (i = 0; i < nnew_workers; i++)
	{
		workerid = workerids[i];
		sched_ctx->sched_mutex[workerid] = &data->sched_mutex;
		sched_ctx->sched_cond[workerid] = &data->sched_cond;
	}

}

static void initialize_eager_center_priority_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_central_prio_data *data = (struct eager_central_prio_data*)malloc(sizeof(struct eager_central_prio_data));

	/* In this policy, we support more than two levels of priority. */
	starpu_sched_set_min_priority(MIN_LEVEL);
	starpu_sched_set_max_priority(MAX_LEVEL);

	/* only a single queue (even though there are several internaly) */
	data->taskq = _starpu_create_priority_taskq();
	sched_ctx->policy_data = (void*) data;

	PTHREAD_MUTEX_INIT(&data->sched_mutex, NULL);
	PTHREAD_COND_INIT(&data->sched_cond, NULL);

	int nworkers = sched_ctx->nworkers;
	int workerid_ctx;
	int workerid;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		sched_ctx->sched_mutex[workerid] = &data->sched_mutex;
		sched_ctx->sched_cond[workerid] = &data->sched_cond;
	}
}

static void deinitialize_eager_center_priority_policy(unsigned sched_ctx_id) 
{
	/* TODO check that there is no task left in the queue */
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_central_prio_data *data = (struct eager_central_prio_data*)sched_ctx->policy_data;

	/* deallocate the task queue */
	_starpu_destroy_priority_taskq(data->taskq);

	PTHREAD_MUTEX_DESTROY(&data->sched_mutex);
        PTHREAD_COND_DESTROY(&data->sched_cond);

        free(data);

	unsigned nworkers_ctx = sched_ctx->nworkers;
	int workerid;
	unsigned workerid_ctx;
	for (workerid_ctx = 0; workerid_ctx < nworkers_ctx; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		sched_ctx->sched_mutex[workerid] = NULL;
		sched_ctx->sched_cond[workerid] = NULL;
	}
	
}

static int _starpu_priority_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_central_prio_data *data = (struct eager_central_prio_data*)sched_ctx->policy_data;

	struct starpu_priority_taskq_s *taskq = data->taskq;

	/* wake people waiting for a task */
	PTHREAD_MUTEX_LOCK(&data->sched_mutex);

	STARPU_TRACE_JOB_PUSH(task, 1);
	
	unsigned priolevel = task->priority - STARPU_MIN_PRIO;

	starpu_task_list_push_front(&taskq->taskq[priolevel], task);
	taskq->ntasks[priolevel]++;
	taskq->total_ntasks++;

	PTHREAD_COND_SIGNAL(&data->sched_cond);
	PTHREAD_MUTEX_UNLOCK(&data->sched_mutex);

	return 0;
}

static struct starpu_task *_starpu_priority_pop_task(unsigned sched_ctx_id)
{
	struct starpu_task *task = NULL;

	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_central_prio_data *data = (struct eager_central_prio_data*)sched_ctx->policy_data;
	
	struct starpu_priority_taskq_s *taskq = data->taskq;

	/* block until some event happens */
	PTHREAD_MUTEX_LOCK(&data->sched_mutex);

	if ((taskq->total_ntasks == 0) && _starpu_machine_is_running())
	{
#ifdef STARPU_NON_BLOCKING_DRIVERS
		PTHREAD_MUTEX_UNLOCK(&data->sched_mutex);
		return NULL;
#else
		PTHREAD_COND_WAIT(&data->sched_cond, &data->sched_mutex);
#endif
	}

	if (taskq->total_ntasks > 0)
	{
		unsigned priolevel = NPRIO_LEVELS - 1;
		do {
			if (taskq->ntasks[priolevel] > 0) {
				/* there is some task that we can grab */
				task = starpu_task_list_pop_back(&taskq->taskq[priolevel]);
				taskq->ntasks[priolevel]--;
				taskq->total_ntasks--;
				STARPU_TRACE_JOB_POP(task, 0);
			}
		} while (!task && priolevel-- > 0);
	}

	PTHREAD_MUTEX_UNLOCK(&data->sched_mutex);

	return task;
}

struct starpu_sched_policy_s _starpu_sched_prio_policy = {
	.init_sched = initialize_eager_center_priority_policy,
	.init_sched_for_workers = initialize_eager_center_priority_policy_for_workers,
	.deinit_sched = deinitialize_eager_center_priority_policy,
	/* we always use priorities in that policy */
	.push_task = _starpu_priority_push_task,
	.pop_task = _starpu_priority_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "prio",
	.policy_description = "eager (with priorities)"
};
