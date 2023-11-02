/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_scheduler.h>
#include <schedulers/starpu_scheduler_toolbox.h>

#include <common/fxt.h>
#include <core/task.h>
#include <core/workers.h>
#include <core/sched_policy.h>
#include <core/debug.h>
#ifdef BUILDING_STARPU
#include <datawizard/memory_nodes.h>
#endif
#include <sched_policies/fifo_queues.h>

#include <limits.h>
#include <math.h> /* for fpclassify() checks on knob values */


#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

//#define NOTIFY_READY_SOON

struct _starpu_dmda_data
{
	double alpha;
	double beta;
	double _gamma;
	double idle_power;

	struct starpu_st_fifo_taskq queue_array[STARPU_NMAXWORKERS];

	long int total_task_cnt;
	long int ready_task_cnt;
	long int eager_task_cnt; /* number of tasks scheduled without model */
	int num_priorities;
};

/* performance steering knobs */

/* . per-scheduler knobs */
static int __s_alpha_knob;
static int __s_beta_knob;
static int __s_gamma_knob;
static int __s_idle_power_knob;

/* . knob variables */
static double __s_alpha__value = 1.0;
static double __s_beta__value = 1.0;
static double __s_gamma__value = 1.0;
static double __s_idle_power__value = 1.0;

/* . per-scheduler knob group */
static struct starpu_perf_knob_group * __kg_starpu_dmda__per_scheduler;

static void sched_knobs__set(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value)
{
	const char * const sched_policy_name = *(const char **)context;
	(void) sched_policy_name;
	if (knob->id == __s_alpha_knob)
	{
		STARPU_ASSERT(fpclassify(value->val_double) == FP_NORMAL);
		__s_alpha__value = value->val_double;
	}
	else if (knob->id == __s_beta_knob)
	{
		STARPU_ASSERT(fpclassify(value->val_double) == FP_NORMAL);
		__s_beta__value = value->val_double;
	}
	else if (knob->id == __s_gamma_knob)
	{
		STARPU_ASSERT(fpclassify(value->val_double) == FP_NORMAL);
		__s_gamma__value = value->val_double;
	}
	else if (knob->id == __s_idle_power_knob)
	{
		STARPU_ASSERT(fpclassify(value->val_double) == FP_NORMAL);
		__s_idle_power__value = value->val_double;
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}

static void sched_knobs__get(const struct starpu_perf_knob * const knob, void *context,       struct starpu_perf_knob_value * const value)
{
	const char * const sched_policy_name = *(const char **)context;
	(void) sched_policy_name;
	if (knob->id == __s_alpha_knob)
	{
		value->val_double = __s_alpha__value;
	}
	else if (knob->id == __s_beta_knob)
	{
		value->val_double = __s_beta__value;
	}
	else if (knob->id == __s_gamma_knob)
	{
		value->val_double = __s_gamma__value;
	}
	else if (knob->id == __s_idle_power_knob)
	{
		value->val_double = __s_idle_power__value;
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}

void _starpu__dmda_c__register_knobs(void)
{
	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_per_scheduler;
		__kg_starpu_dmda__per_scheduler = _starpu_perf_knob_group_register(scope, sched_knobs__set, sched_knobs__get);

		/* TODO: priority capping knobs actually work globally for now, the sched policy name is ignored */
		__STARPU_PERF_KNOB_REG("starpu.dmda", __kg_starpu_dmda__per_scheduler, s_alpha_knob, double, "alpha constant multiplier");

		__STARPU_PERF_KNOB_REG("starpu.dmda", __kg_starpu_dmda__per_scheduler, s_beta_knob, double, "beta constant multiplier");

		__STARPU_PERF_KNOB_REG("starpu.dmda", __kg_starpu_dmda__per_scheduler, s_gamma_knob, double, "gamma constant multiplier");

		__STARPU_PERF_KNOB_REG("starpu.dmda", __kg_starpu_dmda__per_scheduler, s_idle_power_knob, double, "idle_power constant multiplier");
	}
}

void _starpu__dmda_c__unregister_knobs(void)
{
	_starpu_perf_knob_group_unregister(__kg_starpu_dmda__per_scheduler);
	__kg_starpu_dmda__per_scheduler = NULL;
}

/* The dmda scheduling policy uses
 *
 * alpha * T_computation + beta * T_communication + gamma * Consumption
 *
 * Here are the default values of alpha, beta, gamma
 */

#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0

/* This is called when a transfer request is actually pushed to the worker */
static void _starpu_fifo_task_transfer_started(struct starpu_st_fifo_taskq *fifo, struct starpu_task *task, int num_priorities)
{
	double transfer_model = task->predicted_transfer;
	if (isnan(transfer_model))
		return;

	/* We now start the transfer, move it from predicted to pipelined */
	fifo->exp_len -= transfer_model;
	fifo->pipeline_len += transfer_model;
	fifo->exp_start = starpu_timing_now() + fifo->pipeline_len;
	fifo->exp_end = fifo->exp_start + fifo->exp_len;
	if(num_priorities != -1)
	{
		int i;
		int task_prio = starpu_st_normalize_prio(task->priority, num_priorities, task->sched_ctx);
		for(i = 0; i <= task_prio; i++)
			fifo->exp_len_per_priority[i] -= transfer_model;
	}
}

/* This is called when a task is actually pushed to the worker (i.e. the transfer finished */
static void _starpu_fifo_task_started(struct starpu_st_fifo_taskq *fifo, struct starpu_task *task, int num_priorities)
{
	double model = task->predicted;
	double transfer_model = task->predicted_transfer;
	if(!isnan(transfer_model))
		/* The transfer is over, remove it from pipelined */
		fifo->pipeline_len -= transfer_model;

	if(!isnan(model))
	{
		/* We now start the computation, move it from predicted to pipelined */
		fifo->exp_len -= model;
		fifo->pipeline_len += model;
		fifo->exp_start = starpu_timing_now() + fifo->pipeline_len;
		fifo->exp_end= fifo->exp_start + fifo->exp_len;
		if(num_priorities != -1)
		{
			int i;
			int task_prio = starpu_st_normalize_prio(task->priority, num_priorities, task->sched_ctx);
			for(i = 0; i <= task_prio; i++)
				fifo->exp_len_per_priority[i] -= model;
		}
	}
}

/* This is called when a task is actually finished */
static void _starpu_fifo_task_finished(struct starpu_st_fifo_taskq *fifo, struct starpu_task *task, int num_priorities STARPU_ATTRIBUTE_UNUSED)
{
	if(!isnan(task->predicted))
		/* The execution is over, remove it from pipelined */
		fifo->pipeline_len -= task->predicted;
	if (!fifo->pipeline_ntasks)
		_STARPU_DISP("warning: bogus computation of pipeline_ntasks?\n");
	else
		fifo->pipeline_ntasks--;
	fifo->exp_start = STARPU_MAX(starpu_timing_now() + fifo->pipeline_len, fifo->exp_start);
	fifo->exp_end = fifo->exp_start + fifo->exp_len;
}


static struct starpu_task *_dmda_pop_task(unsigned sched_ctx_id, int ready)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task;

	unsigned workerid = starpu_worker_get_id_check();
	struct starpu_st_fifo_taskq *fifo = &dt->queue_array[workerid];

	/* Take the opportunity to update start time */
	fifo->exp_start = STARPU_MAX(starpu_timing_now(), fifo->exp_start);
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	STARPU_ASSERT_MSG(fifo, "worker %u does not belong to ctx %u anymore.\n", workerid, sched_ctx_id);

	if (ready)
		task = starpu_st_fifo_taskq_pop_first_ready_task(fifo, workerid, dt->num_priorities);
	else
		task = starpu_st_fifo_taskq_pop_local_task(fifo);
	if (task)
	{
		_starpu_fifo_task_transfer_started(fifo, task, dt->num_priorities);

		starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, workerid);

#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = starpu_st_non_ready_buffers_count(task, workerid);
			if (non_ready == 0)
				dt->ready_task_cnt++;
		}

		dt->total_task_cnt++;
#endif
	}

	return task;
}

static struct starpu_task *dmda_pop_ready_task(unsigned sched_ctx_id)
{
	return _dmda_pop_task(sched_ctx_id, 1);
}

static struct starpu_task *dmda_pop_task(unsigned sched_ctx_id)
{
	return _dmda_pop_task(sched_ctx_id, 0);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid,
				    double predicted, double predicted_transfer,
				    int prio, unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	/* make sure someone could execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	if (_starpu_get_nsched_ctxs() > 1)
	{
		starpu_worker_relax_on();
		_starpu_sched_ctx_lock_write(sched_ctx_id);
		starpu_worker_relax_off();
		if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, best_workerid, task))
			task = NULL;
		_starpu_sched_ctx_unlock_write(sched_ctx_id);

		if (!task)
			return 0;
	}

	struct starpu_st_fifo_taskq *fifo = &dt->queue_array[best_workerid];

	double now = starpu_timing_now();

#ifdef STARPU_USE_SC_HYPERVISOR
	starpu_sched_ctx_call_pushed_task_cb(best_workerid, sched_ctx_id);
#endif //STARPU_USE_SC_HYPERVISOR

	starpu_worker_lock(best_workerid);

	fifo->pipeline_ntasks++;

	/* Sometimes workers didn't take the tasks as early as we expected */
	fifo->exp_start = isnan(fifo->exp_start) ? now + fifo->pipeline_len : STARPU_MAX(fifo->exp_start, now);
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	/* FIXME: We don't have overlap when running CPU-CPU transfers */
	if ((now + predicted_transfer) < fifo->exp_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0.0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer = (now + predicted_transfer) - fifo->exp_end;
	}

	if(!isnan(predicted_transfer))
	{
		fifo->exp_len += predicted_transfer;
		if(dt->num_priorities != -1)
		{
			int i;
			int task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, task->sched_ctx);
			for(i = 0; i <= task_prio; i++)
				fifo->exp_len_per_priority[i] += predicted_transfer;
		}

	}

	if(!isnan(predicted))
	{
		fifo->exp_len += predicted;
		if(dt->num_priorities != -1)
		{
			int i;
			int task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, task->sched_ctx);
			for(i = 0; i <= task_prio; i++)
				fifo->exp_len_per_priority[i] += predicted;
		}

	}
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	starpu_worker_unlock(best_workerid);

	task->predicted = predicted;
	task->predicted_transfer = predicted_transfer;

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_for(task, best_workerid);

	STARPU_AYU_ADDTOTASKQUEUE(starpu_task_get_job_id(task), best_workerid);

	if (_starpu_get_nsched_ctxs() > 1)
	{
		unsigned stream_ctx_id = starpu_worker_get_sched_ctx_id_stream(best_workerid);
		if(stream_ctx_id != STARPU_NMAX_SCHED_CTXS)
		{
			starpu_worker_relax_on();
			_starpu_sched_ctx_lock_write(sched_ctx_id);
			starpu_worker_relax_off();
			starpu_sched_ctx_move_task_to_ctx_locked(task, stream_ctx_id, 0);
			starpu_sched_ctx_revert_task_counters_ctx_locked(sched_ctx_id, task->flops);
			_starpu_sched_ctx_unlock_write(sched_ctx_id);
		}
	}

	int ret = 0;
	if (prio)
	{
		starpu_worker_lock(best_workerid);
		ret =starpu_st_fifo_taskq_push_sorted_task(&dt->queue_array[best_workerid], task);
		if(dt->num_priorities != -1)
		{
			int i;
			int task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, task->sched_ctx);
			for(i = 0; i <= task_prio; i++)
				dt->queue_array[best_workerid].ntasks_per_priority[i]++;
		}


#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
		starpu_wake_worker_locked(best_workerid);
#endif
		starpu_push_task_end(task);
		starpu_worker_unlock(best_workerid);
	}
	else
	{
		starpu_worker_lock(best_workerid);
		starpu_task_list_push_back (&dt->queue_array[best_workerid].taskq, task);
		dt->queue_array[best_workerid].ntasks++;
		dt->queue_array[best_workerid].nprocessed++;
#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
		starpu_wake_worker_locked(best_workerid);
#endif
		starpu_push_task_end(task);
		starpu_worker_unlock(best_workerid);
	}

	starpu_sched_ctx_list_task_counters_increment(sched_ctx_id, best_workerid);

	return ret;
}

/* TODO: factorise CPU computations, expensive with a lot of cores */
static void compute_all_performance_predictions(struct starpu_task *task,
						unsigned nworkers,
						double local_task_length[nworkers][STARPU_MAXIMPLEMENTATIONS],
						double exp_end[nworkers][STARPU_MAXIMPLEMENTATIONS],
						double *max_exp_endp_of_workers,
						double *min_exp_endp_of_task,
						double local_data_penalty[nworkers][STARPU_MAXIMPLEMENTATIONS],
						double local_energy[nworkers][STARPU_MAXIMPLEMENTATIONS],
						int *forced_worker, int *forced_impl, unsigned sched_ctx_id, unsigned sorted_decision)
{
	int calibrating = 0;
	double max_exp_end_of_workers = DBL_MIN;
	double best_exp_end_of_task = DBL_MAX;
	int ntasks_best = -1;
	int nimpl_best = 0;
	double ntasks_best_end = 0.0;

	/* A priori, we know all estimations */
	int unknown = 0;
	unsigned worker_current = 0;

	int task_prio = 0;

	starpu_task_bundle_t bundle = task->bundle;
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	if(sorted_decision && dt->num_priorities != -1)
		task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, sched_ctx_id);

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	double now = starpu_timing_now();

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(worker_current<nworkers && workers->has_next(workers, &it))
	{
		unsigned nimpl;
		unsigned impl_mask;
		unsigned workerid = workers->get_next(workers, &it);
		struct starpu_st_fifo_taskq *fifo = &dt->queue_array[workerid];
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerid, sched_ctx_id);
		unsigned memory_node = starpu_worker_get_memory_node(workerid);

		STARPU_ASSERT_MSG(fifo != NULL, "workerid %u ctx %u\n", workerid, sched_ctx_id);

		/* Sometimes workers didn't take the tasks as early as we expected */
		double exp_start = isnan(fifo->exp_start) ? now + fifo->pipeline_len : STARPU_MAX(fifo->exp_start, now);

		if (!starpu_worker_can_execute_task_impl(workerid, task, &impl_mask))
			continue;

		for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!(impl_mask & (1U << nimpl)))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			int fifo_ntasks = fifo->ntasks + fifo->pipeline_ntasks;
			double prev_exp_len = fifo->exp_len;
			/* consider the priority of the task when deciding on which workerid to schedule,
			   compute the expected_end of the task if it is inserted before other tasks already scheduled */
			if(sorted_decision)
			{
				if(dt->num_priorities != -1)
				{
					prev_exp_len = fifo->exp_len_per_priority[task_prio];
					fifo_ntasks = fifo->ntasks_per_priority[task_prio];
				}
				else
				{
					starpu_worker_lock(workerid);
					prev_exp_len = starpu_st_fifo_taskq_get_exp_len_prev_task_list(fifo, task, workerid, nimpl, &fifo_ntasks);
					starpu_worker_unlock(workerid);
				}
			}

			exp_end[worker_current][nimpl] = exp_start + prev_exp_len;
			if (exp_end[worker_current][nimpl] > max_exp_end_of_workers)
				max_exp_end_of_workers = exp_end[worker_current][nimpl];

			//_STARPU_DEBUG("Scheduler dmda: task length (%lf) workerid (%u) kernel (%u) \n", local_task_length[workerid][nimpl],workerid,nimpl);

			if (bundle)
			{
				/* TODO : conversion time */
				local_task_length[worker_current][nimpl] = starpu_task_bundle_expected_length(bundle, perf_arch, nimpl);
				if (local_data_penalty)
					local_data_penalty[worker_current][nimpl] = starpu_task_bundle_expected_data_transfer_time(bundle, memory_node);
				if (local_energy)
					local_energy[worker_current][nimpl] = starpu_task_bundle_expected_energy(bundle, perf_arch,nimpl);

			}
			else
			{
				local_task_length[worker_current][nimpl] = starpu_task_worker_expected_length(task, workerid, sched_ctx_id, nimpl);
				if (local_data_penalty)
					local_data_penalty[worker_current][nimpl] = starpu_task_expected_data_transfer_time_for(task, workerid);
				if (local_energy)
					local_energy[worker_current][nimpl] = starpu_task_worker_expected_energy(task, workerid, sched_ctx_id,nimpl);
				double conversion_time = starpu_task_expected_conversion_time(task, perf_arch, nimpl);
				if (conversion_time > 0.0)
					local_task_length[worker_current][nimpl] += conversion_time;
			}
			double ntasks_end = fifo_ntasks / starpu_worker_get_relative_speedup(perf_arch);

			/*
			 * This implements a default greedy scheduler for the
			 * case of tasks which have no performance model, or
			 * whose performance model is not calibrated yet.
			 *
			 * It simply uses the number of tasks already pushed to
			 * the workers, divided by the relative performance of
			 * a CPU and of a GPU.
			 *
			 * This is always computed, but the ntasks_best
			 * selection is only really used if the task indeed has
			 * no performance model, or is not calibrated yet.
			 */
			if (ntasks_best == -1

			    /* Always compute the greedy decision, at least for
			     * the tasks with no performance model. */
			    || (!calibrating && ntasks_end < ntasks_best_end)

			    /* The performance model of this task is not
			     * calibrated on this workerid, try to run it there
			     * to calibrate it there. */
			    || (!calibrating && isnan(local_task_length[worker_current][nimpl]))

			    /* the performance model of this task is not
			     * calibrated on this workerid either, rather run it
			     * there if this one is low on scheduled tasks. */
			    || (calibrating && isnan(local_task_length[worker_current][nimpl]) && ntasks_end < ntasks_best_end)
				)
			{
				ntasks_best_end = ntasks_end;
				ntasks_best = workerid;
				nimpl_best = nimpl;
			}

			if (isnan(local_task_length[worker_current][nimpl]))
				/* we are calibrating, we want to speed-up calibration time
				 * so we privilege non-calibrated tasks (but still
				 * greedily distribute them to avoid dumb schedules) */
				calibrating = 1;

			if (isnan(local_task_length[worker_current][nimpl])
					|| _STARPU_IS_ZERO(local_task_length[worker_current][nimpl]))
				/* there is no prediction available for that task
				 * with that arch (yet or at all), so switch to a greedy strategy */
				unknown = 1;

			if (unknown)
				continue;

			double task_starting_time = exp_start + prev_exp_len;
			if (local_data_penalty)
				task_starting_time = STARPU_MAX(task_starting_time,
					now + local_data_penalty[worker_current][nimpl]);

			exp_end[worker_current][nimpl] = task_starting_time + local_task_length[worker_current][nimpl];

			if (exp_end[worker_current][nimpl] < best_exp_end_of_task)
			{
				/* a better solution was found */
				best_exp_end_of_task = exp_end[worker_current][nimpl];
				nimpl_best = nimpl;
			}

			if (local_energy)
				if (isnan(local_energy[worker_current][nimpl]))
					local_energy[worker_current][nimpl] = 0.;

		}
		worker_current++;
	}

	*forced_worker = unknown?ntasks_best:-1;
	*forced_impl = unknown?nimpl_best:-1;

#ifdef STARPU_VERBOSE
	if (unknown)
	{
		dt->eager_task_cnt++;
	}
#endif

	*min_exp_endp_of_task = best_exp_end_of_task;
	*max_exp_endp_of_workers = max_exp_end_of_workers;
}

static double _dmda_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id, unsigned da, unsigned simulate, unsigned sorted_decision)
{
	/* find the queue */
	int best = -1, best_in_ctx = -1;
	int selected_impl = 0;
	double model_best = 0.0;
	double transfer_model_best = 0.0;

	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;
	int forced_impl = -1;

	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	unsigned nworkers = workers->nworkers;
	double local_task_length[nworkers][STARPU_MAXIMPLEMENTATIONS];
	double local_data_penalty[nworkers][STARPU_MAXIMPLEMENTATIONS];
	double local_energy[nworkers][STARPU_MAXIMPLEMENTATIONS];

	/* Expected end of this task on the workers */
	double exp_end[nworkers][STARPU_MAXIMPLEMENTATIONS];

	/* This is the minimum among the exp_end[] matrix */
	double min_exp_end_of_task;

	/* This is the maximum termination time of already-scheduled tasks over all workers */
	double max_exp_end_of_workers = 0.0;

	double fitness[nworkers][STARPU_MAXIMPLEMENTATIONS];

	compute_all_performance_predictions(task,
					    nworkers,
					    local_task_length,
					    exp_end,
					    &max_exp_end_of_workers,
					    &min_exp_end_of_task,
					    da ? local_data_penalty : NULL,
					    da ? local_energy : NULL,
					    &forced_best,
					    &forced_impl, sched_ctx_id, sorted_decision);


	if (forced_best == -1)
	{
		double best_fitness = -1;
		unsigned worker_current = 0;
		struct starpu_sched_ctx_iterator it;
		workers->init_iterator_for_parallel_tasks(workers, &it, task);
		while(worker_current < nworkers && workers->has_next(workers, &it))
		{
			unsigned worker = workers->get_next(workers, &it);
			unsigned nimpl;
			unsigned impl_mask;

			if (!starpu_worker_can_execute_task_impl(worker, task, &impl_mask))
				continue;

			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!(impl_mask & (1U << nimpl)))
				{
					/* no one on that queue may execute this task */
					continue;
				}
				if (da)
					fitness[worker_current][nimpl] = dt->alpha * __s_alpha__value *(exp_end[worker_current][nimpl] - min_exp_end_of_task)
						+ dt->beta * __s_beta__value *(local_data_penalty[worker_current][nimpl])
						+ dt->_gamma * __s_gamma__value *(local_energy[worker_current][nimpl]);
				else
					fitness[worker_current][nimpl] = exp_end[worker_current][nimpl] - min_exp_end_of_task;

				if (da && exp_end[worker_current][nimpl] > max_exp_end_of_workers)
				{
					/* This placement will make the computation
					 * longer, take into account the idle
					 * consumption of other cpus */
					fitness[worker_current][nimpl] += dt->_gamma * __s_gamma__value * dt->idle_power * __s_idle_power__value * (exp_end[worker_current][nimpl] - max_exp_end_of_workers) / 1000000.0; /* Since gamma is the cost in us of one Joules,
																									  then  d->idle_power * (exp_end - max_exp_end_of_workers)
																									  must be in Joules, thus the / 1000000.0 */
				}

				if (best == -1 || fitness[worker_current][nimpl] < best_fitness)
				{
					/* we found a better solution */
					best_fitness = fitness[worker_current][nimpl];
					best = worker;
					best_in_ctx = worker_current;
					selected_impl = nimpl;

					//_STARPU_DEBUG("best fitness (worker %d) %e = alpha*(%e) + beta(%e) +gamma(%e)\n", worker, best_fitness, exp_end[worker][nimpl] - min_exp_end_of_task, local_data_penalty[worker][nimpl], local_energy[worker][nimpl]);

				}
			}
			worker_current++;
		}
	}
	STARPU_ASSERT(forced_best != -1 || best != -1);

	if (forced_best != -1)
	{
		/* there is no prediction available for that task
		 * with that arch we want to speed-up calibration time
		 * so we force this measurement */
		best = forced_best;
		selected_impl = forced_impl;
		model_best = 0.0;
		transfer_model_best = 0.0;
	}
	else if (task->bundle)
	{
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(best_in_ctx, sched_ctx_id);
		unsigned memory_node = starpu_worker_get_memory_node(best);
		model_best = starpu_task_expected_length(task, perf_arch, selected_impl);
		if (da)
			transfer_model_best = starpu_task_expected_data_transfer_time(memory_node, task);
	}
	else
	{
		model_best = local_task_length[best_in_ctx][selected_impl];
		if (da)
			transfer_model_best = local_data_penalty[best_in_ctx][selected_impl];
	}

	//_STARPU_DEBUG("Scheduler dmda: kernel (%u)\n", selected_impl);
	starpu_task_set_implementation(task, selected_impl);
	starpu_sched_task_break(task);

	if(!simulate)
	{
		/* we should now have the best worker in variable "best" */
		return push_task_on_best_worker(task, best, model_best, transfer_model_best, prio, sched_ctx_id);
	}
	else
	{
		return exp_end[best_in_ctx][selected_impl] ;
	}
}

static int dmda_push_sorted_decision_task(struct starpu_task *task)
{
	return _dmda_push_task(task, 1, task->sched_ctx, 1, 0, 1);
}

static int dmda_push_sorted_task(struct starpu_task *task)
{
#ifdef STARPU_DEVEL
#warning TODO: after defining a scheduling window, use that instead of empty_ctx_tasks
#endif
	return _dmda_push_task(task, 1, task->sched_ctx, 1, 0, 0);
}

static int dm_push_task(struct starpu_task *task)
{
	return _dmda_push_task(task, 0, task->sched_ctx, 0, 0, 0);
}

static double dm_simulate_push_task(struct starpu_task *task)
{
	return _dmda_push_task(task, 0, task->sched_ctx, 0, 1, 0);
}

static int dmda_push_task(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	return _dmda_push_task(task, 0, task->sched_ctx, 1, 0, 0);
}
static double dmda_simulate_push_task(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	return _dmda_push_task(task, 0, task->sched_ctx, 1, 1, 0);
}

static double dmda_simulate_push_sorted_task(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	return _dmda_push_task(task, 1, task->sched_ctx, 1, 1, 0);
}

static double dmda_simulate_push_sorted_decision_task(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	return _dmda_push_task(task, 1, task->sched_ctx, 1, 1, 1);
}

#ifdef NOTIFY_READY_SOON
static void dmda_notify_ready_soon(void *data STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task, double delay)
{
	if (!task->cl)
		return;
	/* fprintf(stderr, "task %lu %p %p %s %s will be ready within %f\n", starpu_task_get_job_id(task), task, task->cl, task->cl->name, task->cl->model?task->cl->model->symbol : NULL, delay); */
	/* TODO: do something with it */
}
#endif

static void dmda_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		struct starpu_st_fifo_taskq *q;
		int workerid = workerids[i];
		/* if the worker has alreadry belonged to this context
		   the queue and the synchronization variables have been already initialized */
		q = &dt->queue_array[workerid];
		starpu_st_fifo_taskq_init(q);
		/* These are only stats, they can be read with races */
		STARPU_HG_DISABLE_CHECKING(q->exp_start);
		STARPU_HG_DISABLE_CHECKING(q->exp_len);
		STARPU_HG_DISABLE_CHECKING(q->exp_end);

		if(dt->num_priorities != -1)
		{
			_STARPU_MALLOC(q->exp_len_per_priority, dt->num_priorities*sizeof(double));
			_STARPU_MALLOC(q->ntasks_per_priority, dt->num_priorities*sizeof(unsigned));
			int j;
			for(j = 0; j < dt->num_priorities; j++)
			{
				q->exp_len_per_priority[j] = 0.0;
				q->ntasks_per_priority[j] = 0;
			}
		}
	}
}

static void dmda_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		if(dt->num_priorities != -1)
		{
			free(dt->queue_array[workerid].exp_len_per_priority);
			free(dt->queue_array[workerid].ntasks_per_priority);
		}
	}
}

static void initialize_dmda_policy(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt;
	_STARPU_CALLOC(dt, 1, sizeof(struct _starpu_dmda_data));

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)dt);

	dt->alpha = starpu_getenv_float_default("STARPU_SCHED_ALPHA", _STARPU_SCHED_ALPHA_DEFAULT);
	dt->beta = starpu_getenv_float_default("STARPU_SCHED_BETA", _STARPU_SCHED_BETA_DEFAULT);
	/* data->_gamma: cost of one Joule in us. If gamma is set to 10^6, then one Joule cost 1s */
#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_getenv("STARPU_SCHED_GAMMA"))
		_STARPU_DISP("Warning: STARPU_SCHED_GAMMA was used, but --enable-blocking-drivers configuration was not set, CPU cores will not actually be sleeping\n");
#endif
	dt->_gamma = starpu_getenv_float_default("STARPU_SCHED_GAMMA", _STARPU_SCHED_GAMMA_DEFAULT);
	/* data->idle_power: Idle power of the whole machine in Watt */
	dt->idle_power = starpu_getenv_float_default("STARPU_IDLE_POWER", 0.0);

	if(starpu_sched_ctx_min_priority_is_set(sched_ctx_id) != 0 && starpu_sched_ctx_max_priority_is_set(sched_ctx_id) != 0)
		dt->num_priorities = starpu_sched_ctx_get_max_priority(sched_ctx_id) - starpu_sched_ctx_get_min_priority(sched_ctx_id) + 1;
	else
		dt->num_priorities = -1;

#ifdef NOTIFY_READY_SOON
	starpu_task_notify_ready_soon_register(dmda_notify_ready_soon, dt);
#endif
}

static void initialize_dmda_sorted_policy(unsigned sched_ctx_id)
{
	initialize_dmda_policy(sched_ctx_id);

	/* The application may use any integer */
	if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
	if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
}

static void deinitialize_dmda_policy(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
#ifdef STARPU_VERBOSE
	{
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
		long int modelled_task_cnt = dt->total_task_cnt - dt->eager_task_cnt;
		_STARPU_DEBUG("%s sched policy (sched_ctx %u): total_task_cnt %ld ready_task_cnt %ld (%.1f%%), modelled_task_cnt = %ld (%.1f%%)%s\n",
			      sched_ctx->sched_policy?sched_ctx->sched_policy->policy_name:"<none>",
			      sched_ctx_id,
			      dt->total_task_cnt,
			      dt->ready_task_cnt,
			      (100.0f*dt->ready_task_cnt)/dt->total_task_cnt,
			      modelled_task_cnt,
			      (100.0f*modelled_task_cnt)/dt->total_task_cnt,
			      modelled_task_cnt==0?" *** Check if performance models are enabled and converging on a per-codelet basis, or use an non-modeling scheduling policy. ***":"");
	}
#endif

	free(dt);
}

/* dmda_pre_exec_hook is called right after the data transfer is done and right
 * before the computation to begin, it is useful to update more precisely the
 * value of the expected start, end, length, etc... */
static void dmda_pre_exec_hook(struct starpu_task *task, unsigned sched_ctx_id)
{
	unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_st_fifo_taskq *fifo = &dt->queue_array[workerid];
	const double now = starpu_timing_now();

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	starpu_worker_lock_self();

	_starpu_fifo_task_started(fifo, task, dt->num_priorities);

	/* Take the opportunity to update start time */
	fifo->exp_start = STARPU_MAX(now + fifo->pipeline_len, fifo->exp_start);
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	starpu_worker_unlock_self();
}

static void _dm_push_task_notify(struct starpu_task *task, int workerid, int perf_workerid, unsigned sched_ctx_id, int da)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_st_fifo_taskq *fifo = &dt->queue_array[workerid];

	/* Compute the expected penalty */
	double predicted = starpu_task_worker_expected_length(task, perf_workerid, task->sched_ctx, starpu_task_get_implementation(task));
	double predicted_transfer = NAN;

	if (da)
		predicted_transfer = starpu_task_expected_data_transfer_time_for(task, workerid);

	double now = starpu_timing_now();

	/* Update the predictions */
	starpu_worker_lock(workerid);

	fifo->pipeline_ntasks++;

	/* Sometimes workers didn't take the tasks as early as we expected */
	fifo->exp_start = isnan(fifo->exp_start) ? now + fifo->pipeline_len : STARPU_MAX(fifo->exp_start, now);
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	if (da)
	{
		/* If there is no prediction available, we consider the task has a null length */
		if (!isnan(predicted_transfer))
		{
			if (now + predicted_transfer < fifo->exp_end)
			{
				/* We may hope that the transfer will be finished by
				 * the start of the task. */
				predicted_transfer = 0;
			}
			else
			{
				/* The transfer will not be finished by then, take the
				 * remainder into account */
				predicted_transfer = (now + predicted_transfer) - fifo->exp_end;
			}
			task->predicted_transfer = predicted_transfer;
			fifo->exp_end += predicted_transfer;
			fifo->exp_len += predicted_transfer;
			if(dt->num_priorities != -1)
			{
				int i;
				int task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, sched_ctx_id);
				for(i = 0; i <= task_prio; i++)
					fifo->exp_len_per_priority[i] += predicted_transfer;
			}

		}
	}

	/* If there is no prediction available, we consider the task has a null length */
	if (!isnan(predicted))
	{
		task->predicted = predicted;
		fifo->exp_end += predicted;
		fifo->exp_len += predicted;
		if(dt->num_priorities != -1)
		{
			int i;
			int task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, sched_ctx_id);
			for(i = 0; i <= task_prio; i++)
				fifo->exp_len_per_priority[i] += predicted;
		}

	}
	if(dt->num_priorities != -1)
	{
		int i;
		int task_prio = starpu_st_normalize_prio(task->priority, dt->num_priorities, sched_ctx_id);
		for(i = 0; i <= task_prio; i++)
			fifo->ntasks_per_priority[i]++;
	}

	fifo->ntasks++;

	starpu_worker_unlock(workerid);
}

static void dm_push_task_notify(struct starpu_task *task, int workerid, int perf_workerid, unsigned sched_ctx_id)
{
	_dm_push_task_notify(task, workerid, perf_workerid, sched_ctx_id, 0);
}

static void dmda_push_task_notify(struct starpu_task *task, int workerid, int perf_workerid, unsigned sched_ctx_id)
{
	_dm_push_task_notify(task, workerid, perf_workerid, sched_ctx_id, 1);
}

static void dmda_post_exec_hook(struct starpu_task * task, unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned workerid = starpu_worker_get_id_check();
	struct starpu_st_fifo_taskq *fifo = &dt->queue_array[workerid];
	starpu_worker_lock_self();
	_starpu_fifo_task_finished(fifo, task, dt->num_priorities);
	starpu_worker_unlock_self();
}

struct starpu_sched_policy _starpu_sched_dm_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dm_push_task,
	.simulate_push_task = dm_simulate_push_task,
	.push_task_notify = dm_push_task_notify,
	.pop_task = dmda_pop_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.policy_name = "dm",
	.policy_description = "performance model",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};

struct starpu_sched_policy _starpu_sched_dmda_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_task,
	.simulate_push_task = dmda_simulate_push_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.policy_name = "dmda",
	.policy_description = "data-aware performance model",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};

struct starpu_sched_policy _starpu_sched_dmda_prio_policy =
{
	.init_sched = initialize_dmda_sorted_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_sorted_task,
	.simulate_push_task = dmda_simulate_push_sorted_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.policy_name = "dmdap",
	.policy_description = "data-aware performance model (priority)",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};

struct starpu_sched_policy _starpu_sched_dmda_sorted_policy =
{
	.init_sched = initialize_dmda_sorted_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_sorted_task,
	.simulate_push_task = dmda_simulate_push_sorted_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_ready_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.policy_name = "dmdas",
	.policy_description = "data-aware performance model (sorted)",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};

struct starpu_sched_policy _starpu_sched_dmda_sorted_decision_policy =
{
	.init_sched = initialize_dmda_sorted_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_sorted_decision_task,
	.simulate_push_task = dmda_simulate_push_sorted_decision_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_ready_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.policy_name = "dmdasd",
	.policy_description = "data-aware performance model (sorted decision)",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};

struct starpu_sched_policy _starpu_sched_dmda_ready_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_task,
	.simulate_push_task = dmda_simulate_push_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_ready_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = dmda_post_exec_hook,
	.policy_name = "dmdar",
	.policy_description = "data-aware performance model (ready)",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};
