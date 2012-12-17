/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#include <starpu_rand.h>
#include <core/workers.h>
#include <core/sched_ctx.h>
#include <sched_policies/fifo_queues.h>
#ifdef HAVE_AYUDAME_H
#include <Ayudame.h>
#endif

static int _random_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */

	unsigned selected = 0;

	double alpha_sum = 0.0;

	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_ctx_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
        int worker;
	struct starpu_iterator it;
        if(workers->init_iterator)
                workers->init_iterator(workers, &it);

        while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		alpha_sum += starpu_worker_get_relative_speedup(perf_arch);
	}

	double random = starpu_drand48()*alpha_sum;
//	_STARPU_DEBUG("my rand is %e\n", random);

	double alpha = 0.0;
	while(workers->has_next(workers, &it))
        {
                worker = workers->get_next(workers, &it);

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

#ifdef HAVE_AYUDAME_H
	if (AYU_event)
	{
		int id = selected;
		AYU_event(AYU_ADDTASKTOQUEUE, _starpu_get_job_associated_to_task(task)->job_id, &id);
	}
#endif

	/* we should now have the best worker in variable "selected" */
	return starpu_push_local_task(selected, task, prio);
}

static int random_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	_starpu_pthread_mutex_t *changing_ctx_mutex = starpu_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers;
        int ret_val = -1;

        _STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
	nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
        if(nworkers == 0)
        {
		_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
                return ret_val;
        }

        ret_val = _random_push_task(task, !!task->priority);
        _STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
        return ret_val;
}

static void initialize_random_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, WORKER_LIST);
	starpu_srand48(time(NULL));
}

static void deinitialize_random_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_random_policy =
{
	.init_sched = initialize_random_policy,
	.add_workers = NULL,
	.remove_workers = NULL,
	.deinit_sched = deinitialize_random_policy,
	.push_task = random_push_task,
	.pop_task = NULL,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "random",
	.policy_description = "weighted random based on worker overall performance"
};
