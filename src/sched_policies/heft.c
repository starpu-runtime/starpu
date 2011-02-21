/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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

#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>
#include <starpu_parameters.h>

static unsigned nworkers;

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static double alpha = STARPU_DEFAULT_ALPHA;
static double beta = STARPU_DEFAULT_BETA;
static double _gamma = STARPU_DEFAULT_GAMMA;
static double idle_power = 0.0;

static double exp_start[STARPU_NMAXWORKERS];
static double exp_end[STARPU_NMAXWORKERS];
static double exp_len[STARPU_NMAXWORKERS];
static double ntasks[STARPU_NMAXWORKERS];

static void heft_init(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	nworkers = topology->nworkers;

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		_gamma = atof(strval_gamma);

	const char *strval_idle_power = getenv("STARPU_IDLE_POWER");
	if (strval_idle_power)
		idle_power = atof(strval_idle_power);

	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		exp_start[workerid] = starpu_timing_now();
		exp_len[workerid] = 0.0;
		exp_end[workerid] = exp_start[workerid]; 
		ntasks[workerid] = 0;

		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}
}

static void heft_post_exec_hook(struct starpu_task *task)
{
	int workerid = starpu_worker_get_id();
	double model = task->predicted;
	
	/* Once we have executed the task, we can update the predicted amount
	 * of work. */
	PTHREAD_MUTEX_LOCK(&sched_mutex[workerid]);
	exp_len[workerid] -= model;
	exp_start[workerid] = starpu_timing_now() + model;
	exp_end[workerid] = exp_start[workerid] + exp_len[workerid];
	ntasks[workerid]--;
	PTHREAD_MUTEX_UNLOCK(&sched_mutex[workerid]);
}

static void heft_push_task_notify(struct starpu_task *task, int workerid)
{
	/* Compute the expected penality */
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	double predicted = starpu_task_expected_length(task, perf_arch);

	/* Update the predictions */
	PTHREAD_MUTEX_LOCK(&sched_mutex[workerid]);

	/* Sometimes workers didn't take the tasks as early as we expected */
	exp_start[workerid] = STARPU_MAX(exp_start[workerid], starpu_timing_now());
	exp_end[workerid] = STARPU_MAX(exp_start[workerid], starpu_timing_now());

	/* If there is no prediction available, we consider the task has a null length */
	if (predicted != -1.0)
	{
		exp_end[workerid] += predicted;
		exp_len[workerid] += predicted;
	}

	ntasks[workerid]++;

	PTHREAD_MUTEX_UNLOCK(&sched_mutex[workerid]);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	PTHREAD_MUTEX_LOCK(&sched_mutex[best_workerid]);
	exp_end[best_workerid] += predicted;
	exp_len[best_workerid] += predicted;
	ntasks[best_workerid]++;
	PTHREAD_MUTEX_UNLOCK(&sched_mutex[best_workerid]);

	task->predicted = predicted;

	if (starpu_get_prefetch_flag())
	{
		unsigned memory_node = starpu_worker_get_memory_node(best_workerid);
		starpu_prefetch_task_input_on_node(task, memory_node);
	}

	return starpu_push_local_task(best_workerid, task, prio);
}

static int _heft_push_task(struct starpu_task *task, unsigned prio)
{
	unsigned worker;
	int best = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;

	double local_task_length[nworkers];
	double local_data_penalty[nworkers];
	double local_power[nworkers];
	double exp_end[nworkers];
	double max_exp_end = 0.0;

	double fitness[nworkers];

	double best_exp_end = 10e240;
	double model_best = 0.0;
	double penality_best = 0.0;

	int ntasks_best = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;

	/* A priori, we know all estimations */
	int unknown = 0;

	/*
	 *	Compute the expected end of the task on the various workers,
	 *	and detect if there is some calibration that needs to be done.
	 */

	for (worker = 0; worker < nworkers; worker++)
	{
		/* Sometimes workers didn't take the tasks as early as we expected */
		exp_start[worker] = STARPU_MAX(exp_start[worker], starpu_timing_now());
		exp_end[worker] = exp_start[worker] + exp_len[worker];
		if (exp_end[worker] > max_exp_end)
			max_exp_end = exp_end[worker];

		if (!starpu_worker_may_execute_task(worker, task))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		local_task_length[worker] = starpu_task_expected_length(task, perf_arch);

		unsigned memory_node = starpu_worker_get_memory_node(worker);
		local_data_penalty[worker] = starpu_data_expected_penalty(memory_node, task);

		double ntasks_end = ntasks[worker] / starpu_worker_get_relative_speedup(perf_arch);

		if (ntasks_best == -1
				|| (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
				|| (!calibrating && local_task_length[worker] == -1.0) /* Not calibrating but this worker is being calibrated */
				|| (calibrating && local_task_length[worker] == -1.0 && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
				) {
			ntasks_best_end = ntasks_end;
			ntasks_best = worker;
		}

		if (local_task_length[worker] == -1.0)
			/* we are calibrating, we want to speed-up calibration time
			 * so we privilege non-calibrated tasks (but still
			 * greedily distribute them to avoid dumb schedules) */
			calibrating = 1;

		if (local_task_length[worker] <= 0.0)
			/* there is no prediction available for that task
			 * with that arch yet, so switch to a greedy strategy */
			unknown = 1;

		if (unknown)
			continue;

		exp_end[worker] = exp_start[worker] + exp_len[worker] + local_task_length[worker];

		if (exp_end[worker] < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end[worker];
		}

		local_power[worker] = starpu_task_expected_power(task, perf_arch);
		if (local_power[worker] == -1.0)
			local_power[worker] = 0.;
	}

	if (unknown)
		forced_best = ntasks_best;

	double best_fitness = -1;

	/*
	 *	Determine which worker optimizes the fitness metric which is a
	 *	trade-off between load-balacing, data locality, and energy
	 *	consumption.
	 */
	
	if (forced_best == -1)
	{
		for (worker = 0; worker < nworkers; worker++)
		{
			if (!starpu_worker_may_execute_task(worker, task))
			{
				/* no one on that queue may execute this task */
				continue;
			}
	
			fitness[worker] = alpha*(exp_end[worker] - best_exp_end) 
					+ beta*(local_data_penalty[worker])
					+ _gamma*(local_power[worker]);

			if (exp_end[worker] > max_exp_end)
				/* This placement will make the computation
				 * longer, take into account the idle
				 * consumption of other cpus */
				fitness[worker] += _gamma * idle_power * (exp_end[worker] - max_exp_end) / 1000000.0;

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
		best = forced_best;
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

static int heft_push_prio_task(struct starpu_task *task)
{
	return _heft_push_task(task, 1);
}

static int heft_push_task(struct starpu_task *task)
{
	if (task->priority > 0)
		return _heft_push_task(task, 1);

	return _heft_push_task(task, 0);
}

static void heft_deinit(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		PTHREAD_MUTEX_DESTROY(&sched_mutex[workerid]);
		PTHREAD_COND_DESTROY(&sched_cond[workerid]);
	}
}

struct starpu_sched_policy_s heft_policy = {
	.init_sched = heft_init,
	.deinit_sched = heft_deinit,
	.push_task = heft_push_task, 
	.push_prio_task = heft_push_prio_task, 
	.push_task_notify = heft_push_task_notify,
	.pop_task = NULL,
	.pop_every_task = NULL,
	.post_exec_hook = heft_post_exec_hook,
	.policy_name = "heft",
	.policy_description = "Heterogeneous Earliest Finish Task"
};
