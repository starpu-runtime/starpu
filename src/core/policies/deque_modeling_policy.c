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

#include <common/config.h>
#include <core/policies/deque_modeling_policy.h>
#include <core/perfmodel/perfmodel.h>

static unsigned nworkers;
static struct starpu_fifo_jobq_s *queue_array[STARPU_NMAXWORKERS];

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static starpu_job_t dm_pop_task(void)
{
	struct starpu_job_s *j;

	int workerid = starpu_worker_get_id();

	struct starpu_fifo_jobq_s *fifo = queue_array[workerid];

	j = _starpu_fifo_pop_task(fifo);
	if (j) {
		double model = j->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = _starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}	

	return j;
}

static struct starpu_job_list_s *dm_pop_every_task(uint32_t where)
{
	struct starpu_job_list_s *new_list;

	int workerid = starpu_worker_get_id();

	struct starpu_fifo_jobq_s *fifo = queue_array[workerid];

	new_list = _starpu_fifo_pop_every_task(fifo, &sched_mutex[workerid], where);
	if (new_list) {
		starpu_job_itor_t i;
		for(i = starpu_job_list_begin(new_list);
			i != starpu_job_list_end(new_list);
			i = starpu_job_list_next(i))
		{
			double model = i->predicted;
	
			fifo->exp_len -= model;
			fifo->exp_start = _starpu_timing_now() + model;
			fifo->exp_end = fifo->exp_start + fifo->exp_len;
		}
	}

	return new_list;
}



static int _dm_push_task(starpu_job_t j, unsigned prio)
{
	/* find the queue */
	struct starpu_fifo_jobq_s *fifo;
	unsigned worker;
	int best = -1;

	double best_exp_end = 0.0;
	double model_best = 0.0;

	struct starpu_task *task = j->task;

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
		double local_length = _starpu_job_expected_length(worker, j, perf_arch);

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

	
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best != -1);

	/* we should now have the best worker in variable "best" */
	fifo = queue_array[best];

	fifo->exp_end += model_best;
	fifo->exp_len += model_best;

	j->predicted = model_best;

	unsigned memory_node = starpu_worker_get_memory_node(best);

	if (_starpu_get_prefetch_flag())
		_starpu_prefetch_task_input_on_node(task, memory_node);

	if (prio) {
		return _starpu_fifo_push_prio_task(queue_array[best], &sched_mutex[best], &sched_cond[best], j);
	} else {
		return _starpu_fifo_push_task(queue_array[best], &sched_mutex[best], &sched_cond[best], j);
	}
}

static int dm_push_prio_task(starpu_job_t j)
{
	return _dm_push_task(j, 1);
}

static int dm_push_task(starpu_job_t j)
{
	if (j->task->priority == STARPU_MAX_PRIO)
		return _dm_push_task(j, 1);

	return _dm_push_task(j, 0);
}

static void initialize_dm_policy(struct starpu_machine_config_s *config, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	nworkers = config->nworkers;

	unsigned workerid;
	for (workerid = 0; workerid < config->nworkers; workerid++)
	{
		queue_array[workerid] = _starpu_create_fifo();
	
		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}
}

static void deinitialize_dm_policy(struct starpu_machine_config_s *config, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
		_starpu_destroy_fifo(queue_array[worker]);
}

struct starpu_sched_policy_s _starpu_sched_dm_policy = {
	.init_sched = initialize_dm_policy,
	.deinit_sched = deinitialize_dm_policy,
	.push_task = dm_push_task, 
	.push_prio_task = dm_push_prio_task,
	.pop_task = dm_pop_task,
	.pop_every_task = dm_pop_every_task,
	.policy_name = "dm",
	.policy_description = "performance model"
};
