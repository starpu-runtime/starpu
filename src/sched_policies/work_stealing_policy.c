/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011, 2012  INRIA
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
#include <sched_policies/deque_queues.h>

typedef struct work_stealing_data{
	struct _starpu_deque_jobq **queue_array;
	unsigned rr_worker;
	/* keep track of the work performed from the beginning of the algorithm to make
	 * better decisions about which queue to select when stealing or deferring work
	 */
	unsigned performed_total;
	pthread_mutex_t sched_mutex;
	pthread_cond_t sched_cond;
	unsigned last_pop_worker;
static unsigned last_push_worker;
} work_stealing_data;

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
static unsigned select_victim_round_robin(struct starpu_sched_ctx *sched_ctx)
{
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;
	unsigned worker = ws->last_pop_worker;

	/* If the worker's queue is empty, let's try
	 * the next ones */
	while (!ws->queue_array[worker]->njobs)
	{
		worker = (worker + 1) % sched_ctx->nworkers;
		if (worker == ws->last_pop_worker)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			break;
		}
	}

	ws->last_pop_worker = (worker + 1) % sched_ctx->nworkers;

	return worker;
}

/**
 * Return a worker to whom add a task.
 * Selecting a worker is done in a round-robin fashion.
 */
static unsigned select_worker_round_robin(struct starpu_sched_ctx *sched_ctx)
{
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;
	unsigned worker = ws->last_push_worker;

	last_push_worker = (last_push_worker + 1) % sched_ctx->nworkers;

	return worker;
}

#ifdef USE_OVERLOAD

/**
 * Return a ratio helpful to determine whether a worker is suitable to steal
 * tasks from or to put some tasks in its queue.
 *
 * \return	a ratio with a positive or negative value, describing the current state of the worker :
 * 		a smaller value implies a faster worker with an relatively emptier queue : more suitable to put tasks in
 * 		a bigger value implies a slower worker with an reletively more replete queue : more suitable to steal tasks from
 */
static float overload_metric(struct starpu_sched_ctx *sched_ctx, unsigned id)
{
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;
	float execution_ratio = 0.0f;
	float current_ratio = 0.0f;

	int nprocessed = _starpu_get_deque_nprocessed(ws->queue_array[id]);
	unsigned njobs = _starpu_get_deque_njobs(ws->queue_array[id]);

	/* Did we get enough information ? */
	if (performed_total > 0 && nprocessed > 0)
	{
		/* How fast or slow is the worker compared to the other workers */
		execution_ratio = (float) nprocessed / performed_total;
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
static unsigned select_victim_overload(struct starpu_sched_ctx *sched_ctx)
{
	unsigned worker, worker_ctx;
	float  worker_ratio;
	unsigned best_worker = 0;
	float best_ratio = FLT_MIN;	

	/* Don't try to play smart until we get
	 * enough informations. */
	if (performed_total < calibration_value)
		return select_victim_round_robin(sched_ctx);

	for (worker_ctx = 0; worker_ctx < sched_ctx->nworkers; worker_ctx++)
	{
		worker = sched_ctx->workerid[worker_ctx];
		worker_ratio = overload_metric(worker);

		if (worker_ratio > best_ratio)
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
static unsigned select_worker_overload(struct starpu_sched_ctx *sched_ctx)
{
	unsigned worker, worker_ctx;
	float  worker_ratio;
	unsigned best_worker = 0;
	float best_ratio = FLT_MAX;

	/* Don't try to play smart until we get
	 * enough informations. */
	if (performed_total < calibration_value)
		return select_worker_round_robin(sched_ctx);

	for (worker_ctx = 0; worker_ctx < sched_ctx->nworkers; worker_ctx++)
	{
		worker = sched_ctx->workerid[worker_ctx];
		worker_ratio = overload_metric(sched_ctx,  worker);

		if (worker_ratio < best_ratio)
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
static inline unsigned select_victim(struct starpu_sched_ctx *sched_ctx)
{
#ifdef USE_OVERLOAD
	return select_victim_overload(sched_ctx);
#else
	return select_victim_round_robin(sched_ctx);
#endif /* USE_OVERLOAD */
}

/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline unsigned select_worker(struct starpu_sched_ctx *sched_ctx)
{
#ifdef USE_OVERLOAD
	return select_worker_overload(sched_ctx);
#else
	return select_worker_round_robin(sched_ctx);
#endif /* USE_OVERLOAD */
}


#ifdef STARPU_DEVEL
#warning TODO rewrite ... this will not scale at all now
#endif
static struct starpu_task *ws_pop_task(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	struct starpu_task *task;
	struct _starpu_deque_jobq *q;

	int workerid = starpu_worker_get_id();

	STARPU_ASSERT(workerid != -1);

	q = ws->queue_array[workerid];

	PTHREAD_MUTEX_LOCK(&ws->sched_mutex);

	task = _starpu_deque_pop_task(q, workerid);
	if (task)
	{
		/* there was a local task */
		ws->performed_total++;
		PTHREAD_MUTEX_UNLOCK(&ws->sched_mutex);
		q->nprocessed++;
		q->njobs--;
		return task;
	}

	/* we need to steal someone's job */
	unsigned victim = select_victim(sched_ctx);
	struct _starpu_deque_jobq *victimq = ws->queue_array[victim];

	task = _starpu_deque_pop_task(victimq, workerid);
	if (task)
	{
		_STARPU_TRACE_WORK_STEALING(q, workerid);
		ws->performed_total++;

		/* Beware : we have to increase the number of processed tasks of
		 * the stealer, not the victim ! */
		q->nprocessed++;
		victimq->njobs--;
	}

	return task;
}

int ws_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	struct _starpu_deque_jobq *deque_queue;
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task); 
	int workerid = starpu_worker_get_id();

	_STARPU_PTHREAD_MUTEX_LOCK(&ws->sched_mutex);

	/* If the current thread is not a worker but
	 * the main thread (-1), we find the better one to
	 * put task on its queue */
	if (workerid == -1)
		workerid = select_worker(sched_ctx);

	deque_queue = ws->queue_array[workerid];

	_STARPU_TRACE_JOB_PUSH(task, 0);
	_starpu_job_list_push_back(deque_queue->jobq, j);
	deque_queue->njobs++;

	_STARPU_PTHREAD_COND_SIGNAL(&ws->sched_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&ws->sched_mutex);

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
		/**
		 * The first WS_POP_TASK will increase NPROCESSED though no task was actually performed yet,
		 * we need to initialize it at -1.
		 */
		ws->queue_array[workerid]->nprocessed = -1;
		ws->queue_array[workerid]->njobs = 0;

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
	ws->last_pop_worker = 0;
	ws->last_push_worker = 0;

	/**
	 * The first WS_POP_TASK will increase PERFORMED_TOTAL though no task was actually performed yet,
	 * we need to initialize it at -1.
	 */
	ws->performed_total = -1;

	ws->queue_array = (struct starpu_deque_jobq_s**)malloc(STARPU_NMAXWORKERS*sizeof(struct _starpu_deque_jobq*));

	_STARPU_PTHREAD_MUTEX_INIT(&ws->sched_mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(&ws->sched_cond, NULL);

	unsigned workerid_ctx;
	int workerid;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		ws->queue_array[workerid] = _starpu_create_deque();
		/**
		 * The first WS_POP_TASK will increase NPROCESSED though no task was actually performed yet,
		 * we need to initialize it at -1.
		 */
		ws->queue_array[workerid]->nprocessed = -1;
		ws->queue_array[workerid]->njobs = 0;

		sched_ctx->sched_mutex[workerid] = &ws->sched_mutex;
		sched_ctx->sched_cond[workerid] = &ws->sched_cond;

#ifdef USE_OVERLOAD
		enum starpu_perf_archtype perf_arch;
		perf_arch = starpu_worker_get_perf_archtype(workerid);
		calibration_value += (unsigned int) starpu_worker_get_relative_speedup(perf_arch);
#endif /* USE_OVERLOAD */
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

struct starpu_sched_policy _starpu_sched_ws_policy =
{
	.init_sched = initialize_ws_policy,
	.deinit_sched = deinit_ws_policy,
	.push_task = ws_push_task,
	.pop_task = ws_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "ws",
	.policy_description = "work stealing",
	.init_sched_for_workers = initialize_ws_policy_for_workers
};
