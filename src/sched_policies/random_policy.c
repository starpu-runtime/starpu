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

/* Policy attributing tasks randomly to workers */

#include <core/workers.h>
#include <core/sched_ctx.h>
#include <sched_policies/fifo_queues.h>

static int _random_push_task(struct starpu_task *task, unsigned prio, struct starpu_sched_ctx *sched_ctx)
{
	/* find the queue */
        unsigned worker, worker_ctx;

	unsigned selected = 0;

	double alpha_sum = 0.0;

	unsigned nworkers = sched_ctx->nworkers;	

	for (worker_ctx = 0; worker_ctx < nworkers; worker_ctx++)
	{
                worker = sched_ctx->workerids[worker_ctx];

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		alpha_sum += starpu_worker_get_relative_speedup(perf_arch);
	}

	double random = starpu_drand48()*alpha_sum;
//	_STARPU_DEBUG("my rand is %e\n", random);

	double alpha = 0.0;
	for (worker_ctx = 0; worker_ctx < nworkers; worker_ctx++)
	{
		worker = sched_ctx->workerids[worker_ctx];

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		double worker_alpha = starpu_worker_get_relative_speedup(perf_arch);

		if (alpha + worker_alpha > random) {
			/* we found the worker */
			selected = worker;
			break;
		}

		alpha += worker_alpha;
	}

	/* we should now have the best worker in variable "selected" */
	_starpu_increment_nsubmitted_tasks_of_worker(selected);
	int n = starpu_push_local_task(selected, task, prio);
	return n;
}


static int random_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

    return _random_push_task(task, 0, sched_ctx);
}

static void initialize_random_policy_for_workers(unsigned sched_ctx_id, int *workerids, unsigned nnew_workers) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	unsigned i;
	int workerid;
	for (i = 0; i < nnew_workers; i++)
	{
		workerid = sched_ctx->workerids[i];
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		sched_ctx->sched_mutex[workerid] = &workerarg->sched_mutex;
		sched_ctx->sched_cond[workerid] = &workerarg->sched_cond;
	}
}

static void initialize_random_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	starpu_srand48(time(NULL));

	unsigned nworkers = sched_ctx->nworkers;	

	unsigned workerid_ctx;
	int workerid;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		sched_ctx->sched_mutex[workerid] = &workerarg->sched_mutex;
		sched_ctx->sched_cond[workerid] = &workerarg->sched_cond;
	}
}

struct starpu_sched_policy_s _starpu_sched_random_policy = {
	.init_sched = initialize_random_policy,
	.init_sched_for_workers = initialize_random_policy_for_workers,
	.deinit_sched = NULL,
	.push_task = random_push_task,
	.pop_task = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "random",
	.policy_description = "weighted random"
};
