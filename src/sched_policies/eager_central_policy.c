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

typedef struct {
	struct _starpu_fifo_taskq *fifo;
	pthread_mutex_t sched_mutex;
	pthread_cond_t sched_cond;
} eager_center_policy_data;

static void eager_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers) 
{
	eager_center_policy_data *data = (eager_center_policy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	unsigned i;
	int workerid;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, &data->sched_mutex, &data->sched_cond);
	}
}

static void eager_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	unsigned i;
	int workerid;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, NULL, NULL);
	}
}

static void initialize_eager_center_policy(unsigned sched_ctx_id) 
{
	starpu_create_worker_collection_for_sched_ctx(sched_ctx_id, WORKER_LIST);

	eager_center_policy_data *data = (eager_center_policy_data*)malloc(sizeof(eager_center_policy_data));

	/* there is only a single queue in that trivial design */
	data->fifo =  _starpu_create_fifo();

	_STARPU_PTHREAD_MUTEX_INIT(&data->sched_mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(&data->sched_cond, NULL);

	starpu_set_sched_ctx_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_eager_center_policy(unsigned sched_ctx_id) 
{
	/* TODO check that there is no task left in the queue */

	eager_center_policy_data *data = (eager_center_policy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	/* deallocate the job queue */
	_starpu_destroy_fifo(data->fifo);

	_STARPU_PTHREAD_MUTEX_DESTROY(&data->sched_mutex);
	_STARPU_PTHREAD_COND_DESTROY(&data->sched_cond);
	
	starpu_delete_worker_collection_for_sched_ctx(sched_ctx_id);

	free(data);	
}

static int push_task_eager_policy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	eager_center_policy_data *data = (eager_center_policy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
        pthread_mutex_t *changing_ctx_mutex = starpu_get_changing_ctx_mutex(sched_ctx_id);
        unsigned nworkers;
        int ret_val = -1;

        _STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
        nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
        if(nworkers == 0)
        {
                _STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
                return ret_val;
        }

	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);
	int worker;
	if(workers->init_cursor)
		workers->init_cursor(workers);

        while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
	}

	if(workers->init_cursor)
                workers->deinit_cursor(workers);

        ret_val = _starpu_fifo_push_task(data->fifo, &data->sched_mutex, &data->sched_cond, task);
        _STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
        return ret_val;
}

static struct starpu_task *pop_every_task_eager_policy(unsigned sched_ctx_id)
{
	eager_center_policy_data *data = (eager_center_policy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	return _starpu_fifo_pop_every_task(data->fifo, &data->sched_mutex, starpu_worker_get_id());
}

static struct starpu_task *pop_task_eager_policy(unsigned sched_ctx_id)
{
	unsigned workerid = starpu_worker_get_id();
	eager_center_policy_data *data = (eager_center_policy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	
	struct starpu_task *task =  _starpu_fifo_pop_task(data->fifo, workerid);
	
	if(task)
	{
		struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);
		int worker;
		if(workers->init_cursor)
			workers->init_cursor(workers);
		
		while(workers->has_next(workers))
		{
			worker = workers->get_next(workers);
		}
		
		if(workers->init_cursor)
			workers->deinit_cursor(workers);
	}
	
	return task;
}

struct starpu_sched_policy _starpu_sched_eager_policy =
{
	.init_sched = initialize_eager_center_policy,
	.deinit_sched = deinitialize_eager_center_policy,
	.add_workers = eager_add_workers,
	.remove_workers = eager_remove_workers,
	.push_task = push_task_eager_policy,
	.pop_task = pop_task_eager_policy,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = pop_every_task_eager_policy,
	.policy_name = "eager",
	.policy_description = "greedy policy"
};
