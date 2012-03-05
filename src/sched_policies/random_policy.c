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

/* Policy attributing tasks randomly to workers */

#include <core/workers.h>
#include <core/sched_ctx.h>
#include <sched_policies/fifo_queues.h>

static int _random_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */

	unsigned selected = 0;

	double alpha_sum = 0.0;

	unsigned sched_ctx_id = task->sched_ctx;
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);
        int worker;
        if(workers->init_cursor)
                workers->init_cursor(workers);

        while(workers->has_next(workers))
	{
                worker = workers->get_next(workers);

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		alpha_sum += starpu_worker_get_relative_speedup(perf_arch);
	}

	double random = starpu_drand48()*alpha_sum;
//	_STARPU_DEBUG("my rand is %e\n", random);

	double alpha = 0.0;
	while(workers->has_next(workers))
        {
                worker = workers->get_next(workers);

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		double worker_alpha = starpu_worker_get_relative_speedup(perf_arch);

		if (alpha + worker_alpha > random && starpu_worker_can_execute_task(worker, task, 0))
		{
			/* we found the worker */
			selected = worker;
			break;
		}

		alpha += worker_alpha;
	}

	if(workers->init_cursor)
                workers->deinit_cursor(workers);

	/* we should now have the best worker in variable "selected" */
	_starpu_increment_nsubmitted_tasks_of_worker(selected);
	int n = starpu_push_local_task(selected, task, prio);
	return n;
}


static int random_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
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

        ret_val = _random_push_task(task, !!task->priority);
        _STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
        return ret_val;
}

static void random_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers) 
{
	unsigned i;
	int workerid;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		struct _starpu_worker *workerarg = _starpu_get_worker_struct(workerid);
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, &workerarg->sched_mutex, &workerarg->sched_cond);
	}
}

static void random_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	unsigned i;
	int workerid;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		struct _starpu_worker *workerarg = _starpu_get_worker_struct(workerid);
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, &workerarg->sched_mutex, &workerarg->sched_cond);
	}

}

static void initialize_random_policy(unsigned sched_ctx_id) 
{
	starpu_create_worker_collection_for_sched_ctx(sched_ctx_id, WORKER_LIST);
	starpu_srand48(time(NULL));
}

static void deinitialize_random_policy(unsigned sched_ctx_id) 
{
	starpu_delete_worker_collection_for_sched_ctx(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_random_policy =
{
	.init_sched = initialize_random_policy,
	.add_workers = random_add_workers,
	.remove_workers = random_remove_workers,
	.deinit_sched = deinitialize_random_policy,
	.push_task = random_push_task,
	.pop_task = NULL,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "random",
	.policy_description = "weighted random"
};
