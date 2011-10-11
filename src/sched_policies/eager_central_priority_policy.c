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

static void initialize_eager_center_priority_policy_for_workers(unsigned sched_ctx_id, unsigned nnew_workers) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	unsigned nworkers_ctx = sched_ctx->nworkers;

	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	unsigned ntotal_workers = config->topology.nworkers;

	unsigned all_workers = nnew_workers == ntotal_workers ? ntotal_workers : nworkers_ctx + nnew_workers;

	unsigned workerid_ctx;
	for (workerid_ctx = nworkers_ctx; workerid_ctx < all_workers; workerid_ctx++)
	{
		sched_ctx->sched_mutex[workerid_ctx] = sched_ctx->sched_mutex[0];
		sched_ctx->sched_cond[workerid_ctx] = sched_ctx->sched_cond[0];
	}

}

static void initialize_eager_center_priority_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	/* In this policy, we support more than two levels of priority. */
	starpu_sched_set_min_priority(MIN_LEVEL);
	starpu_sched_set_max_priority(MAX_LEVEL);

	/* only a single queue (even though there are several internaly) */
	struct starpu_priority_taskq_s *taskq = _starpu_create_priority_taskq();
	sched_ctx->policy_data = (void*) taskq;

	pthread_cond_t *global_sched_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
	pthread_mutex_t *global_sched_mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));

	PTHREAD_MUTEX_INIT(global_sched_mutex, NULL);
	PTHREAD_COND_INIT(global_sched_cond, NULL);

	int nworkers = sched_ctx->nworkers;
	int workerid_ctx;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		sched_ctx->sched_mutex[workerid_ctx] = global_sched_mutex;
		sched_ctx->sched_cond[workerid_ctx] = global_sched_cond;
	}
}

static void deinitialize_eager_center_priority_policy(unsigned sched_ctx_id) 
{
	/* TODO check that there is no task left in the queue */
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_priority_taskq_s *taskq = (struct starpu_priority_taskq_s*)sched_ctx->policy_data;

	/* deallocate the task queue */
	_starpu_destroy_priority_taskq(taskq);

	PTHREAD_MUTEX_DESTROY(sched_ctx->sched_mutex[0]);
        PTHREAD_COND_DESTROY(sched_ctx->sched_cond[0]);
        free(sched_ctx->sched_mutex[0]);
        free(sched_ctx->sched_cond[0]);

        free(taskq);
}

static int _starpu_priority_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_priority_taskq_s *taskq = (struct starpu_priority_taskq_s*)sched_ctx->policy_data;

	/* wake people waiting for a task */
	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[0]);

	STARPU_TRACE_JOB_PUSH(task, 1);
	
	unsigned priolevel = task->priority - STARPU_MIN_PRIO;

	starpu_task_list_push_front(&taskq->taskq[priolevel], task);
	taskq->ntasks[priolevel]++;
	taskq->total_ntasks++;

	PTHREAD_COND_SIGNAL(sched_ctx->sched_cond[0]);
	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[0]);

	return 0;
}

static struct starpu_task *_starpu_priority_pop_task(unsigned sched_ctx_id)
{
	struct starpu_task *task = NULL;

	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_priority_taskq_s *taskq = (struct starpu_priority_taskq_s*)sched_ctx->policy_data;

	/* block until some event happens */
	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[0]);

	if ((taskq->total_ntasks == 0) && _starpu_machine_is_running())
	{
#ifdef STARPU_NON_BLOCKING_DRIVERS
		PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[0]);
		return NULL;
#else
		PTHREAD_COND_WAIT(sched_ctx->sched_cond[0], sched_ctx->sched_mutex[0]);
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

	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[0]);

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
