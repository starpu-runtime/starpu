/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

/* Distributed queues using performance modeling to assign tasks */

#include <float.h>
#include <limits.h>
#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <core/perfmodel/perfmodel.h>

static pthread_mutex_t big_lock;

static unsigned nworkers, ncombinedworkers;
static enum starpu_perf_archtype applicable_perf_archtypes[STARPU_NARCH_VARIATIONS];
static unsigned napplicable_perf_archtypes = 0;

static struct starpu_fifo_taskq_s *queue_array[STARPU_NMAXWORKERS];

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static double alpha = 1.0;
static double beta = 1.0;

static struct starpu_task *parallel_heft_pop_task(void)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	struct starpu_fifo_taskq_s *fifo = queue_array[workerid];

	task = _starpu_fifo_pop_task(fifo, -1);
	if (task) {
		double model = task->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}

	return task;
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	/* Is this a basic worker or a combined worker ? */
	int nbasic_workers = (int)starpu_worker_get_count();
	int is_basic_worker = (best_workerid < nbasic_workers);

	unsigned memory_node; 
	memory_node = starpu_worker_get_memory_node(best_workerid);

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_on_node(task, memory_node);

	if (is_basic_worker)
	{
		PTHREAD_MUTEX_LOCK(&big_lock);

		struct starpu_fifo_taskq_s *fifo;
		fifo = queue_array[best_workerid];
	
		fifo->exp_end += predicted;
		fifo->exp_len += predicted;
	
		task->predicted = predicted;
	
		int ret;

		if (prio)
		{
			ret = _starpu_fifo_push_prio_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
		}
		else {
			ret = _starpu_fifo_push_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
		}

		PTHREAD_MUTEX_UNLOCK(&big_lock);

		return ret;
	}
	else {
		/* This is a combined worker so we create task aliases */
		struct starpu_combined_worker_s *combined_worker;
		combined_worker = _starpu_get_combined_worker_struct(best_workerid);
		int worker_size = combined_worker->worker_size;
		int *combined_workerid = combined_worker->combined_workerid;

		int ret = 0;
		int i;
		
		task->predicted = predicted;

		starpu_job_t j = _starpu_get_job_associated_to_task(task);
		j->task_size = worker_size;
		j->combined_workerid = best_workerid;
		j->active_task_alias_count = 0;

		PTHREAD_BARRIER_INIT(&j->before_work_barrier, NULL, worker_size);
		PTHREAD_BARRIER_INIT(&j->after_work_barrier, NULL, worker_size);

		PTHREAD_MUTEX_LOCK(&big_lock);

		for (i = 0; i < worker_size; i++)
		{
			struct starpu_task *alias = _starpu_create_task_alias(task);
			int local_worker = combined_workerid[i];

			struct starpu_fifo_taskq_s *fifo;
			fifo = queue_array[local_worker];
		
			fifo->exp_end += predicted;
			fifo->exp_len += predicted;
		
			alias->predicted = predicted;
		
			if (prio)
			{
				ret |= _starpu_fifo_push_prio_task(queue_array[local_worker],
					&sched_mutex[local_worker], &sched_cond[local_worker], alias);
			}
			else {
				ret |= _starpu_fifo_push_task(queue_array[local_worker],
					&sched_mutex[local_worker], &sched_cond[local_worker], alias);
			}
		}

		PTHREAD_MUTEX_UNLOCK(&big_lock);

		return ret;
	}
}

static double compute_expected_end(int workerid, double length)
{
	if (workerid < (int)nworkers)
	{
		/* This is a basic worker */
		struct starpu_fifo_taskq_s *fifo;
		fifo = queue_array[workerid];
		return (fifo->exp_start + fifo->exp_len + length);
	}
	else {
		/* This is a combined worker, the expected end is the end for the latest worker */
		int worker_size;
		int *combined_workerid;
		starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

		double exp_end = DBL_MIN;

		int i;
		for (i = 0; i < worker_size; i++)
		{
			struct starpu_fifo_taskq_s *fifo;
			fifo = queue_array[combined_workerid[i]];
			double local_exp_end = (fifo->exp_start + fifo->exp_len + length);
			exp_end = STARPU_MAX(exp_end, local_exp_end);
		}

		return exp_end;
	}
}

static int _parallel_heft_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker;
	int best = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;

	double local_task_length[nworkers+ncombinedworkers];
	double local_data_penalty[nworkers+ncombinedworkers];
	double exp_end[nworkers+ncombinedworkers];
	double fitness[nworkers+ncombinedworkers];

	int skip_worker[nworkers+ncombinedworkers];

	double best_exp_end = DBL_MAX;
	double model_best = 0.0;
	double penality_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		fifo = queue_array[worker];
		fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
		fifo->exp_end = STARPU_MAX(fifo->exp_end, starpu_timing_now());
	}

	for (worker = 0; worker < (nworkers+ncombinedworkers); worker++)
	{
		if (!starpu_combined_worker_may_execute_task(worker, task))
		{
			/* no one on that queue may execute this task */
			skip_worker[worker] = 1;
			continue;
		}
		else {
			skip_worker[worker] = 0;
		}

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		local_task_length[worker] = starpu_task_expected_length(task, perf_arch);

		unsigned memory_node = starpu_worker_get_memory_node(worker);
		local_data_penalty[worker] = starpu_data_expected_penalty(memory_node, task);

		if (local_task_length[worker] == -1.0)
		{
			forced_best = worker;
			break;
		}

		exp_end[worker] = compute_expected_end(worker, local_task_length[worker]);

		if (exp_end[worker] < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end[worker];
		}
	}

	double best_fitness = -1;
	
	if (forced_best == -1)
	{
		for (worker = 0; worker < nworkers+ncombinedworkers; worker++)
		{

			if (skip_worker[worker])
			{
				/* no one on that queue may execute this task */
				continue;
			}
	
			fitness[worker] = alpha*(exp_end[worker] - best_exp_end) 
					+ beta*(local_data_penalty[worker]);

			if (best == -1 || fitness[worker] < best_fitness)
			{
				/* we found a better solution */
				best_fitness = fitness[worker];
				best = worker;
			}
		}
	}

	STARPU_ASSERT(forced_best != -1 || best != -1);
	
	if (forced_best != -1)
	{
		/* there is no prediction available for that task
		 * with that arch we want to speed-up calibration time
		 * so we force this measurement */
		best = worker;
		model_best = 0.0;
		penality_best = 0.0;
	}
	else 
	{
		model_best = local_task_length[best];
		penality_best = local_data_penalty[best];
	}

	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, prio);
}

static int parallel_heft_push_prio_task(struct starpu_task *task)
{
	return _parallel_heft_push_task(task, 1);
}

static int parallel_heft_push_task(struct starpu_task *task)
{
	if (task->priority == STARPU_MAX_PRIO)
		return _parallel_heft_push_task(task, 1);

	return _parallel_heft_push_task(task, 0);
}

static void initialize_parallel_heft_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	nworkers = topology->nworkers;

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		beta = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		beta = atof(strval_beta);

	_starpu_sched_find_worker_combinations(topology);

	ncombinedworkers = topology->ncombinedworkers;

	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		queue_array[workerid] = _starpu_create_fifo();
	
		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}

	PTHREAD_MUTEX_INIT(&big_lock, NULL);

	/* We pre-compute an array of all the perfmodel archs that are applicable */
	unsigned total_worker_count = nworkers + ncombinedworkers;

	unsigned used_perf_archtypes[STARPU_NARCH_VARIATIONS];
	memset(used_perf_archtypes, 0, sizeof(used_perf_archtypes));

	for (workerid = 0; workerid < total_worker_count; workerid++)
	{
		enum starpu_perf_archtype perf_archtype = starpu_worker_get_perf_archtype(workerid);
		used_perf_archtypes[perf_archtype] = 1;
	}

	napplicable_perf_archtypes = 0;

	int arch;
	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++)
	{
		if (used_perf_archtypes[arch])
			applicable_perf_archtypes[napplicable_perf_archtypes++] = arch;
	}
}

static void deinitialize_parallel_heft_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	unsigned workerid;
	for (workerid = 0; workerid < topology->nworkers; workerid++)
		_starpu_destroy_fifo(queue_array[workerid]);
}

struct starpu_sched_policy_s _starpu_sched_parallel_heft_policy = {
	.init_sched = initialize_parallel_heft_policy,
	.deinit_sched = deinitialize_parallel_heft_policy,
	.push_task = parallel_heft_push_task, 
	.push_prio_task = parallel_heft_push_prio_task, 
	.pop_task = parallel_heft_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "pheft",
	.policy_description = "parallel HEFT"
};
