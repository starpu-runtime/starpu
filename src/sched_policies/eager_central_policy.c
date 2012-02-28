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

/*
 *	This is just the trivial policy where every worker use the same
 *	JOB QUEUE.
 */

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>

typedef struct eager_center_policy_data {
	struct _starpu_fifo_taskq *fifo;
	pthread_mutex_t sched_mutex;
	pthread_cond_t sched_cond;
} eager_center_policy_data;

static void initialize_eager_center_policy_for_workers(unsigned sched_ctx_id, int *workerids, unsigned nnew_workers) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_center_policy_data *data = (struct eager_center_policy_data*)sched_ctx->policy_data;

	unsigned i;
	int workerid;
	for (i = 0; i < nnew_workers; i++)
	{
		workerid = workerids[i];
		sched_ctx->sched_mutex[workerid] = &data->sched_mutex;
		sched_ctx->sched_cond[workerid] = &data->sched_cond;
	}
}

static void initialize_eager_center_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	struct eager_center_policy_data *data = (struct eager_center_policy_data*)malloc(sizeof(eager_center_policy_data));

	/* there is only a single queue in that trivial design */
	data->fifo =  _starpu_create_fifo();

	PTHREAD_MUTEX_INIT(&data->sched_mutex, NULL);
	PTHREAD_COND_INIT(&data->sched_cond, NULL);

	sched_ctx->policy_data = (void*)data;

	int workerid;
	unsigned workerid_ctx;
	int nworkers_ctx = sched_ctx->nworkers;
	for (workerid_ctx = 0; workerid_ctx < nworkers_ctx; workerid_ctx++){
		workerid = sched_ctx->workerids[workerid_ctx];
		sched_ctx->sched_mutex[workerid] = &data->sched_mutex;
		sched_ctx->sched_cond[workerid] = &data->sched_cond;
	}
}

static void deinitialize_eager_center_policy(unsigned sched_ctx_id) 
{
	/* TODO check that there is no task left in the queue */

	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_center_policy_data *data = (struct eager_center_policy_data*)sched_ctx->policy_data;


	/* deallocate the job queue */
	_starpu_destroy_fifo(data->fifo);

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

static int push_task_eager_policy(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_center_policy_data *data = (struct eager_center_policy_data*)sched_ctx->policy_data;

	int i;
	int workerid;
	for(i = 0; i < sched_ctx->nworkers; i++){
		workerid = sched_ctx->workerids[i]; 
		_starpu_increment_nsubmitted_tasks_of_worker(workerid);
	}

	struct _starpu_fifo_taskq *fifo = data->fifo;
	return _starpu_fifo_push_task(fifo, &data->sched_mutex, &data->sched_cond, task);
}

static struct starpu_task *pop_every_task_eager_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_center_policy_data *data = (struct eager_center_policy_data*)sched_ctx->policy_data;

	static struct _starpu_fifo_taskq *fifo = data->fifo;
	return _starpu_fifo_pop_every_task(fifo, &data->sched_mutex, starpu_worker_get_id());
}

static struct starpu_task *pop_task_eager_policy(unsigned sched_ctx_id)
{
    unsigned workerid = starpu_worker_get_id();
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct eager_center_policy_data *data = (struct eager_center_policy_data*)sched_ctx->policy_data;

	static struct _starpu_fifo_taskq *fifo = data->fifo;
	struct starpu_task *task =  _starpu_fifo_pop_task(fifo, workerid);

	if(task)
	  {
		int i;
		for(i = 0; i <sched_ctx->nworkers; i++)
		  {
			workerid = sched_ctx->workerids[i]; 
			_starpu_decrement_nsubmitted_tasks_of_worker(workerid);
		  }
	  }

	return task;
}

struct starpu_sched_policy _starpu_sched_eager_policy =
{
	.init_sched = initialize_eager_center_policy,
	.init_sched_for_workers = initialize_eager_center_policy_for_workers,
	.deinit_sched = deinitialize_eager_center_policy,
	.push_task = push_task_eager_policy,
	.pop_task = pop_task_eager_policy,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = pop_every_task_eager_policy,
	.policy_name = "eager",
	.policy_description = "greedy policy"
};
