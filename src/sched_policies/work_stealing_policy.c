/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

/* Work stealing policy */

#include <core/workers.h>
#include <sched_policies/deque_queues.h>

typedef struct work_stealing_data{
	struct starpu_deque_jobq_s **queue_array;
	unsigned rr_worker;
	/* keep track of the work performed from the beginning of the algorithm to make
	 * better decisions about which queue to select when stealing or deferring work
	 */
	unsigned performed_total;
	pthread_mutex_t sched_mutex;
	pthread_cond_t sched_cond;
} work_stealing_data;

#ifdef USE_OVERLOAD
static float overload_metric(struct starpu_deque_jobq_s *dequeue_queue, unsigned *performed_total)
{
	float execution_ratio = 0.0f;
	if (*performed_total > 0) {
		execution_ratio = _starpu_get_deque_nprocessed(dequeue_queue)/ *performed_total;
	}

	unsigned performed_queue;
	performed_queue = _starpu_get_deque_nprocessed(dequeue_queue);

	float current_ratio = 0.0f;
	if (performed_queue > 0) {
		current_ratio = _starpu_get_deque_njobs(dequeue_queue)/performed_queue;
	}
	
	return (current_ratio - execution_ratio);
}

/* who to steal work to ? */
static struct starpu_deque_jobq_s *select_victimq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = ws->rr_worker;
	do {
		if (overload_metric(worker) > 0.0f)
		{
			q = ws->queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = ws->queue_array[ws->rr_worker];
	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}

static struct starpu_deque_jobq_s *select_workerq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = ws->rr_worker;
	do {
		if (overload_metric(worker) < 0.0f)
		{
			q = ws->queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = ws->queue_array[ws->rr_worker];
	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}

#else

/* who to steal work to ? */
static struct starpu_deque_jobq_s *select_victimq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	q = ws->queue_array[ws->rr_worker];

	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}


/* when anonymous threads submit tasks, 
 * we need to select a queue where to dispose them */
static struct starpu_deque_jobq_s *select_workerq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	q = ws->queue_array[ws->rr_worker];

	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}

#endif

#ifdef STARPU_DEVEL
#warning TODO rewrite ... this will not scale at all now
#endif
static struct starpu_task *ws_pop_task(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	struct starpu_task *task;

	int workerid = starpu_worker_get_id();

	struct starpu_deque_jobq_s *q;

	q = ws->queue_array[workerid];

	PTHREAD_MUTEX_LOCK(&ws->sched_mutex);

	task = _starpu_deque_pop_task(q, -1);
	if (task) {
		/* there was a local task */
		ws->performed_total++;
		PTHREAD_MUTEX_UNLOCK(&ws->sched_mutex);
		return task;
	}
	
	/* we need to steal someone's job */
	struct starpu_deque_jobq_s *victimq;
	victimq = select_victimq(ws, sched_ctx->nworkers);

	task = _starpu_deque_pop_task(victimq, workerid);
	if (task) {
		STARPU_TRACE_WORK_STEALING(q, victimq);
		ws->performed_total++;
	}

	PTHREAD_MUTEX_UNLOCK(&ws->sched_mutex);

	return task;
}

int ws_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	starpu_job_t j = _starpu_get_job_associated_to_task(task);

	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	int workerid = starpu_worker_get_id();


        struct starpu_deque_jobq_s *deque_queue;
	deque_queue = ws->queue_array[workerid];

        PTHREAD_MUTEX_LOCK(&ws->sched_mutex);
	// XXX reuse ?
        //total_number_of_jobs++;

        STARPU_TRACE_JOB_PUSH(task, 0);
        starpu_job_list_push_front(deque_queue->jobq, j);
        deque_queue->njobs++;
        deque_queue->nprocessed++;

        PTHREAD_COND_SIGNAL(&ws->sched_cond);
        PTHREAD_MUTEX_UNLOCK(&ws->sched_mutex);

        return 0;
}

static void initialize_ws_policy_for_workers(unsigned sched_ctx_id, int *workerids,unsigned nnew_workers) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	unsigned i;
	int workerid;
	
	for (i = 0; i < nnew_workers; i++)
	{
		workerid = workerids[i];
		ws->queue_array[workerid] = _starpu_create_deque();

		sched_ctx->sched_mutex[workerid] = &ws->sched_mutex;
		sched_ctx->sched_cond[workerid] = &ws->sched_cond;
	}
}

static void initialize_ws_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)malloc(sizeof(work_stealing_data));
	sched_ctx->policy_data = (void*)ws;
	
	unsigned nworkers = sched_ctx->nworkers;
	ws->rr_worker = 0;
	ws->queue_array = (struct starpu_deque_jobq_s**)malloc(STARPU_NMAXWORKERS*sizeof(struct starpu_deque_jobq_s*));

	PTHREAD_MUTEX_INIT(&ws->sched_mutex, NULL);
	PTHREAD_COND_INIT(&ws->sched_cond, NULL);

	unsigned workerid_ctx;
	int workerid;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		ws->queue_array[workerid] = _starpu_create_deque();

		sched_ctx->sched_mutex[workerid] = &ws->sched_mutex;
		sched_ctx->sched_cond[workerid] = &ws->sched_cond;
	}
}

static void deinit_ws_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	unsigned nworkers_ctx = sched_ctx->nworkers;

	struct work_stealing_data *data = (struct work_stealing_data*)malloc(sizeof(work_stealing_data));
	

	pthread_mutex_init(&data->sched_mutex, NULL);
	pthread_cond_init(&data->sched_cond, NULL);

	int workerid;
	unsigned workerid_ctx;
	for (workerid_ctx = 0; workerid_ctx < nworkers_ctx; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		_starpu_destroy_deque(&data->queue_array[workerid]);
		sched_ctx->sched_mutex[workerid] = &data->sched_mutex;
		sched_ctx->sched_cond[workerid] = &data->sched_cond;
	}
}

struct starpu_sched_policy_s _starpu_sched_ws_policy = {
	.init_sched = initialize_ws_policy,
	.deinit_sched = deinit_ws_policy,
	.push_task = ws_push_task,
	.pop_task = ws_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "ws",
	.policy_description = "work stealing",
	.init_sched_for_workers = initialize_ws_policy_for_workers
};
