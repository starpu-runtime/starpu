/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <core/workers.h>
#include <core/mechanisms/fifo_queues.h>
#include <core/perfmodel/perfmodel.h>

static unsigned nworkers;
static struct starpu_fifo_taskq_s *queue_array[STARPU_NMAXWORKERS];

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static double alpha = 1.0;
static double beta = 1.0;

static struct starpu_task *dmda_pop_task(void)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	struct starpu_fifo_taskq_s *fifo = queue_array[workerid];

	task = _starpu_fifo_pop_task(fifo);
	if (task) {
		double model = task->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = _starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}	

	return task;
}

static struct starpu_task *dmda_pop_every_task(uint32_t where)
{
	struct starpu_task *new_list;

	int workerid = starpu_worker_get_id();

	struct starpu_fifo_taskq_s *fifo = queue_array[workerid];

	new_list = _starpu_fifo_pop_every_task(fifo, &sched_mutex[workerid], where);

	while (new_list)
	{
		double model = new_list->predicted;

		fifo->exp_len -= model;
		fifo->exp_start = _starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	
		new_list = new_list->next;
	}

	return new_list;
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	struct starpu_fifo_taskq_s *fifo;
	fifo = queue_array[best_workerid];

	fifo->exp_end += predicted;
	fifo->exp_len += predicted;

	task->predicted = predicted;

	unsigned memory_node = starpu_worker_get_memory_node(best_workerid);

	if (_starpu_get_prefetch_flag())
		_starpu_prefetch_task_input_on_node(task, memory_node);

	if (prio) {
		return _starpu_fifo_push_prio_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
	} else {
		return _starpu_fifo_push_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
	}
}

static int _dm_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker;
	int best = -1;

	double best_exp_end = 0.0;
	double model_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		double exp_end;
		
		fifo = queue_array[worker];

		fifo->exp_start = STARPU_MAX(fifo->exp_start, _starpu_timing_now());
		fifo->exp_end = STARPU_MAX(fifo->exp_end, _starpu_timing_now());

		if (!_starpu_worker_may_execute_task(worker, task->cl->where))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		double local_length = _starpu_task_expected_length(worker, task, perf_arch);

		if (local_length == -1.0) 
		{
			/* there is no prediction available for that task
			 * with that arch we want to speed-up calibration time 
			 * so we force this measurement */
			/* XXX assert we are benchmarking ! */
			best = worker;
			model_best = 0.0;
			exp_end = fifo->exp_start + fifo->exp_len;
			break;
		}


		exp_end = fifo->exp_start + fifo->exp_len + local_length;

		if (best == -1 || exp_end < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end;
			best = worker;
			model_best = local_length;
		}
	}
	
	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, prio);
}

static int _dmda_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker;
	int best = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;

	double local_task_length[nworkers];
	double local_data_penalty[nworkers];
	double exp_end[nworkers];

	double fitness[nworkers];

	double best_exp_end = 10e240;
	double model_best = 0.0;
	double penality_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		fifo = queue_array[worker];

		fifo->exp_start = STARPU_MAX(fifo->exp_start, _starpu_timing_now());
		fifo->exp_end = STARPU_MAX(fifo->exp_end, _starpu_timing_now());

		if (!_starpu_worker_may_execute_task(worker, task->cl->where))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		local_task_length[worker] = _starpu_task_expected_length(worker, task, perf_arch);

		unsigned memory_node = starpu_worker_get_memory_node(worker);
		local_data_penalty[worker] = _starpu_data_expected_penalty(memory_node, task);

		if (local_task_length[worker] == -1.0)
		{
			forced_best = worker;
			break;
		}

		exp_end[worker] = fifo->exp_start + fifo->exp_len + local_task_length[worker];

		if (exp_end[worker] < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end[worker];
		}
	}

	double best_fitness = -1;
	
	if (forced_best == -1)
	{
		for (worker = 0; worker < nworkers; worker++)
		{
			fifo = queue_array[worker];
	
			if (!_starpu_worker_may_execute_task(worker, task->cl->where))
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

	//			fprintf(stderr, "best fitness (worker %d) %le = alpha*(%le) + beta(%le) \n", worker, best_fitness, exp_end[worker] - best_exp_end, local_data_penalty[worker]);
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

static int dm_push_prio_task(struct starpu_task *task)
{
	return _dm_push_task(task, 1);
}

static int dm_push_task(struct starpu_task *task)
{
	if (task->priority == STARPU_MAX_PRIO)
		return _dm_push_task(task, 1);

	return _dm_push_task(task, 0);
}

static int dmda_push_prio_task(struct starpu_task *task)
{
	return _dmda_push_task(task, 1);
}

static int dmda_push_task(struct starpu_task *task)
{
	if (task->priority == STARPU_MAX_PRIO)
		return _dmda_push_task(task, 1);

	return _dmda_push_task(task, 0);
}

static void initialize_dmda_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	nworkers = topology->nworkers;

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		beta = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		beta = atof(strval_beta);

	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		queue_array[workerid] = _starpu_create_fifo();
	
		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}
}

static void deinitialize_dmda_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	unsigned workerid;
	for (workerid = 0; workerid < topology->nworkers; workerid++)
		_starpu_destroy_fifo(queue_array[workerid]);
}

struct starpu_sched_policy_s _starpu_sched_dm_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dm_push_task, 
	.push_prio_task = dm_push_prio_task,
	.pop_task = dmda_pop_task,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dm",
	.policy_description = "performance model"
};

struct starpu_sched_policy_s _starpu_sched_dmda_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_task, 
	.push_prio_task = dmda_push_prio_task, 
	.pop_task = dmda_pop_task,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmda",
	.policy_description = "data-aware performance model"
};
