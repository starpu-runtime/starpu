/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Joris Pablo
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
 * Copyright (C) 2020       Télécom-Sud Paris
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

#include <starpu_config.h>
#include <starpu.h>
#include <starpu_scheduler.h>
#include <schedulers/starpu_scheduler_toolbox.h>

#include <limits.h>
#include <math.h> /* for fpclassify() checks on knob values */

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

struct _starpu_dmda_data
{
	double alpha;
	double beta;
	double _gamma;
	double idle_power;

	starpu_st_fifo_taskq_t queue_array[STARPU_NMAXWORKERS];
};

/* The dmda scheduling policy uses
 *
 * alpha * T_computation + beta * T_communication + gamma * Consumption
 *
 * Here are the default values of alpha, beta, gamma
 */

#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0

static void initialize_dmda_policy(unsigned sched_ctx_id)
{
	fprintf(stderr, "HELLO FROM MY_DM\n");

	struct _starpu_dmda_data *dt;
	dt = calloc(1, sizeof(struct _starpu_dmda_data));
	assert(dt);

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)dt);

	dt->alpha = starpu_get_env_float_default("STARPU_SCHED_ALPHA", _STARPU_SCHED_ALPHA_DEFAULT);
	dt->beta = starpu_get_env_float_default("STARPU_SCHED_BETA", _STARPU_SCHED_BETA_DEFAULT);
	/* data->_gamma: cost of one Joule in us. If gamma is set to 10^6, then one Joule cost 1s */
	dt->_gamma = starpu_get_env_float_default("STARPU_SCHED_GAMMA", _STARPU_SCHED_GAMMA_DEFAULT);
	/* data->idle_power: Idle power of the whole machine in Watt */
	dt->idle_power = starpu_get_env_float_default("STARPU_IDLE_POWER", 0.0);
}

static void deinitialize_dmda_policy(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	free(dt);
}

static void dmda_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		/* if the worker has already belonged to this context
		   the queue and the synchronization variables have been already initialized */
		dt->queue_array[workerid] = starpu_st_fifo_taskq_create();
	}
}

static void dmda_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		starpu_st_fifo_taskq_destroy(dt->queue_array[workerid]);
		dt->queue_array[workerid] = NULL;
	}
}

static int dm_push_task(struct starpu_task *task)
{
	/* Julia version should look like this:
	 *
	 * best_worker = -1
	 * best_implem = -1
	 * best_EFT = 0
	 * for worker in workers:
	 *    for implem in implems:
	 *        if !worker_can_execute_task_impl(worker, task, implem)
	 *            continue
	 *        end
	 *        EFT = EFT(task, worker, implem)
	 *        if best_worker == -1 || EFT < best_EFT
	 *            best_worker = worker
	 *            best_implem = implem
	 *            best_EFT = EFT
	 *        end
	 *    end
	 * end
	 * push!(data.queue[worker], task, impl)
	 */

	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int best = -1;
	double best_exp_end_of_task = 0.0;
	unsigned best_impl = 0;
	double predicted = 0.0;
	double predicted_transfer = 0.0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;

	double now = starpu_timing_now();

	// Find couple (worker, implem) that minimizes EFT(task, worker, implem)
	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned nimpl;
		unsigned impl_mask;
		unsigned worker = workers->get_next(workers, &it);
		starpu_st_fifo_taskq_t fifo  = dt->queue_array[worker];
		double exp_start = starpu_st_fifo_exp_start_get(fifo);
		double pipeline_len = starpu_st_fifo_pipeline_len_get(fifo);
		double exp_len = starpu_st_fifo_exp_len_get(fifo);

		/* Sometimes workers didn't take the tasks as early as we expected */
		double new_exp_start = isnan(exp_start) ? now + pipeline_len : STARPU_MAX(exp_start, now);

		if (!starpu_worker_can_execute_task_impl(worker, task, &impl_mask))
			continue;

		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!(impl_mask & (1U << nimpl)))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			// todo: handle case where no calibration or no model
			double local_length = starpu_task_worker_expected_length(task, worker, sched_ctx_id, nimpl);
			double local_penalty = starpu_task_expected_data_transfer_time_for(task, worker);
			double exp_end = new_exp_start + exp_len + local_length;

			if (best == -1 || exp_end < best_exp_end_of_task)
			{
				/* a better solution was found */
				best_exp_end_of_task = exp_end;
				best = worker;
				best_impl = nimpl;
				predicted = local_length;
				predicted_transfer = local_penalty;
			}
		}
	}
	STARPU_ASSERT(best >= 0);

	// Set task implem.
	starpu_task_set_implementation(task, best_impl);

	// Update expected start of the next task in the queue and expected end of the last task in the queue
	// This code should be generated automatically.
	starpu_st_fifo_taskq_t fifo = dt->queue_array[best];
	double exp_start = starpu_st_fifo_exp_start_get(fifo);
	double pipeline_len = starpu_st_fifo_pipeline_len_get(fifo);
	double exp_len = starpu_st_fifo_exp_len_get(fifo);
	now = starpu_timing_now();
	starpu_worker_lock(best);
	double new_exp_start = isnan(exp_start) ? now + pipeline_len : STARPU_MAX(exp_start, now);
	starpu_st_fifo_exp_start_set(fifo, new_exp_start);
	double new_exp_end = new_exp_start + exp_len;
	starpu_st_fifo_exp_end_set(fifo, new_exp_end);
	if ((now + predicted_transfer) < new_exp_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0.0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer = (now + predicted_transfer) - new_exp_end;
	}
	double new_exp_len = exp_len;
	if(!isnan(predicted_transfer))
		new_exp_len += predicted_transfer;
	if(!isnan(predicted))
		new_exp_len += predicted;
	starpu_st_fifo_exp_len_set(fifo, new_exp_len);
	starpu_st_fifo_exp_end_set(fifo, new_exp_start + new_exp_len);
	starpu_worker_unlock(best);

	// Not sure what's the purpose of this.
	task->predicted = predicted;
	task->predicted_transfer = predicted_transfer;

	// Prefetch
	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_for(task, best);

	// Push task to worker queue
	starpu_worker_lock(best);
	starpu_st_fifo_taskq_push_back_task(fifo, task);
	starpu_st_fifo_ntasks_inc(fifo, 1);
	starpu_st_fifo_nprocessed_inc(fifo, 1);
#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	starpu_wake_worker_locked(best);
#endif
	starpu_push_task_end(task);
	starpu_worker_unlock(best);
	starpu_sched_ctx_list_task_counters_increment(sched_ctx_id, best);

	return 0;
}

static struct starpu_task *dmda_pop_task(unsigned sched_ctx_id)
{
	/* Julia version should look like this:
	 *
	 * return pop!(data.queue[worker])
	 */
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task;

	unsigned workerid = starpu_worker_get_id_check();

	starpu_st_fifo_taskq_t fifo = dt->queue_array[workerid];

	/* Take the opportunity to update start time */
	double new_exp_start = STARPU_MAX(starpu_timing_now(), starpu_st_fifo_exp_start_get(fifo));
	double new_exp_end = new_exp_start + starpu_st_fifo_exp_end_get(fifo);
	starpu_st_fifo_exp_start_set(fifo, new_exp_start);
	starpu_st_fifo_exp_end_set(fifo, new_exp_end);

	task = starpu_st_fifo_taskq_pop_local_task(fifo);
	if (task)
	{
		double transfer_model = task->predicted_transfer;
		if (!isnan(transfer_model))
		{
			/* We now start the transfer, move it from predicted to pipelined */
			double new_exp_len = starpu_st_fifo_exp_len_get(fifo);
			new_exp_len -= transfer_model;
			double new_pipeline_len = starpu_st_fifo_pipeline_len_get(fifo);
			new_pipeline_len += transfer_model;

			starpu_st_fifo_exp_len_set(fifo, new_exp_len);
			starpu_st_fifo_pipeline_len_set(fifo, new_pipeline_len);

			new_exp_start = starpu_timing_now() + new_pipeline_len;
			new_exp_end = new_exp_start + new_exp_len;
			starpu_st_fifo_exp_start_set(fifo, new_exp_start);
			starpu_st_fifo_exp_end_set(fifo, new_exp_end);
		}
		starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, workerid);
	}

	return task;
}

// This code should be generated automatically.
static void dmda_pre_exec_hook(struct starpu_task *task, unsigned sched_ctx_id)
{
	unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_st_fifo_taskq_t fifo = dt->queue_array[workerid];
	const double now = starpu_timing_now();

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	starpu_worker_lock_self();

	double model = task->predicted;
	double transfer_model = task->predicted_transfer;
	if(!isnan(transfer_model))
	{
		/* The transfer is over, remove it from pipelined */
		starpu_st_fifo_pipeline_len_inc(fifo, -transfer_model);
	}

	if(!isnan(model))
	{
		/* We now start the computation, move it from predicted to pipelined */
		starpu_st_fifo_exp_len_inc(fifo, -model);
		starpu_st_fifo_pipeline_len_inc(fifo, model);
		starpu_st_fifo_exp_start_set(fifo, starpu_timing_now() + starpu_st_fifo_pipeline_len_get(fifo));
		starpu_st_fifo_exp_end_set(fifo, starpu_st_fifo_exp_start_get(fifo) + starpu_st_fifo_exp_len_get(fifo));
	}

	/* Take the opportunity to update start time */
	starpu_st_fifo_exp_start_set(fifo, STARPU_MAX(now + starpu_st_fifo_pipeline_len_get(fifo), starpu_st_fifo_exp_start_get(fifo)));
	starpu_st_fifo_exp_end_set(fifo, starpu_st_fifo_exp_start_get(fifo) + starpu_st_fifo_exp_len_get(fifo));

	starpu_worker_unlock_self();
}

// This code should be generated automatically.
static void dmda_post_exec_hook(struct starpu_task * task, unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned workerid = starpu_worker_get_id_check();
	starpu_st_fifo_taskq_t fifo = dt->queue_array[workerid];
	starpu_worker_lock_self();
	if(!isnan(task->predicted))
		/* The execution is over, remove it from pipelined */
		starpu_st_fifo_pipeline_len_inc(fifo, -task->predicted);
	starpu_st_fifo_exp_start_set(fifo, STARPU_MAX(starpu_timing_now() + starpu_st_fifo_pipeline_len_get(fifo), starpu_st_fifo_exp_start_get(fifo)));
	starpu_st_fifo_exp_end_set(fifo, starpu_st_fifo_exp_start_get(fifo) + starpu_st_fifo_exp_len_get(fifo));
	starpu_worker_unlock_self();
}

struct starpu_sched_policy my_dm_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers,
	.remove_workers = dmda_remove_workers,
	.push_task = dm_push_task,
	.simulate_push_task = NULL,
	.pop_task = dmda_pop_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "mydm",
	.policy_description = "performance model",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};

struct starpu_sched_policy *predefined_policies[] =
{
	&my_dm_policy
};

struct starpu_sched_policy *starpu_get_sched_lib_policy(const char *name)
{
	if (!strcmp(name, "mydm"))
		return &my_dm_policy;
	return NULL;
}

struct starpu_sched_policy **starpu_get_sched_lib_policies(void)
{
	return predefined_policies;
}
